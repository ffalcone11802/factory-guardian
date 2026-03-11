import json
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from factory_guardian.dataset import MVTecDataset
from factory_guardian.evaluation import predict
from factory_guardian.model import LiteVAE, get_init_function, ELBOLoss
from factory_guardian.utils.folder import check_dir, path_joiner
from factory_guardian.utils.plot import plot_train_loss


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    category = args.category
    img_size = 256
    batch_size = args.batch_size
    num_workers = args.num_workers


    # -------- PRE-PROCESSING --------

    # Training set pre-processing
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(args.rotation_range),
        transforms.ToTensor(),
    ])

    # Training dataset
    train_dataset = MVTecDataset(f"data/{category}", phase="train", transform=train_transform)

    # Pick 10% of training set for validation
    train_idx, valid_idx = train_test_split(range(len(train_dataset)), test_size=0.1)

    # Samplers for getting training and validation sets
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(valid_idx)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn
    )

    val_loader = DataLoader(
        train_dataset,
        sampler=val_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn
    )


    # -------- TRAINING --------

    # Model
    model = LiteVAE(latent_dim=args.latent_dim).to(device)

    # Init weights
    init_weights = get_init_function(args.init_type)
    model.apply(init_weights)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Loss
    criterion = ELBOLoss(beta=args.beta)

    num_epochs = args.num_epochs
    train_losses = []

    print("Training begun")

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss, x, outputs = train_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch,
            verbose=args.verbose
        )

        train_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}")

        checkpoints_folder = "checkpoints"
        check_dir(checkpoints_folder)

        # Save model checkpoints
        if (epoch + 1) % args.save_checkpoint_freq == 0:
            torch.save(
                model.state_dict(),
                path_joiner(checkpoints_folder, f"{category}_epoch_{epoch + 1}.pth")
            )
            print("Checkpoint saved")

        # Print some reconstructions
        if (epoch + 1) % args.save_imgs_freq == 0:
            num_images = min(x.shape[0], 8)

            img_train = utils.make_grid(
                torch.cat((x[:num_images], outputs[:num_images]), dim=0),
                nrow=num_images
            )
            utils.save_image(
                img_train,
                path_joiner(checkpoints_folder, f"{category}_epoch_{epoch + 1}_train.png")
            )

            x_val, outputs_val = val_epoch(
                model=model,
                val_loader=val_loader,
                device=device
            )

            img_val = utils.make_grid(
                torch.cat((x_val[:num_images], outputs_val[:num_images]), dim=0),
                nrow=num_images
            )
            utils.save_image(
                img_val,
                path_joiner(checkpoints_folder, f"{category}_epoch_{epoch + 1}_val.png")
            )

        weights_folder = path_joiner("results", "weights")
        check_dir(weights_folder)

        # Save final parameters
        if (epoch + 1) == num_epochs:
            torch.save(
                model.state_dict(),
                path_joiner(weights_folder, f"{category}.pth")
            )
            print("Final parameters saved")

    print("Training ended")
    print("--------------------------------------------------------")

    # Plot training loss
    plot_train_loss(train_losses)


    # -------- THRESHOLD SELECTION --------

    model.eval()

    imgs, pixels, _ = predict(
        model=model,
        dataloader=val_loader
    )

    _, img_scores = zip(*imgs)
    _, pixel_scores = zip(*pixels)

    mu = np.mean(img_scores)
    std = np.std(img_scores)
    img_level_th = mu + std * 2
    print(f"Image-level threshold: {img_level_th:.4f}")

    mu = np.mean(pixel_scores)
    std = np.std(pixel_scores)
    px_level_th = mu + std * 2
    print(f"Pixel-level threshold: {px_level_th:.4f}")

    save_folder = path_joiner("results", "params")
    check_dir(save_folder)

    with open(path_joiner(save_folder, f"{category}.json"), "w") as f:
        config = {
            "px_level_th": float(px_level_th.cpu().numpy()),
            "img_level_th": float(img_level_th.cpu().numpy())
        }
        json.dump(config, f, indent=4)

    print("Threshold values saved")
    print("--------------------------------------------------------")


def train_epoch(model, train_loader, device, optimizer, criterion, epoch, verbose=True):
    model.train()
    running_loss = 0.0

    iter_data = tqdm(
        train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}", disable=not verbose
    )

    for x in iter_data:
        x = x.to(device)

        # Forward pass
        outputs, mu, log_var = model(x)

        # Reconstruction + beta * KL
        loss = criterion(outputs, x, mu, log_var)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss, x, outputs


def val_epoch(model, val_loader, device):
    model.eval()
    x = next(iter(val_loader))
    x = x.to(device)
    outputs, _, _ = model(x)
    return x, outputs
