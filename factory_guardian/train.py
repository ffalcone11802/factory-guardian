from typing import Tuple
from argparse import Namespace
import numpy as np
import json
import torch
from torch import optim, nn, Tensor
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from factory_guardian.dataset import MVTecDataset
from factory_guardian.evaluation import predict
from factory_guardian.model import LiteVAE, get_init_function, ELBOLoss
from factory_guardian.utils.folder import (
    check_folder,
    path_joiner,
    CHECKPOINTS_FOLDER,
    WEIGHTS_FOLDER,
    PARAMS_FOLDER
)
from factory_guardian.utils.plot import plot_train_loss


def train(args: Namespace):
    """
    Main function to train the LiteVAE and choose the thresholds.

    Args:
        args (Namespace): Command-line arguments.
    """
    # Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = 256
    category = args.category
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
    train_dataset = MVTecDataset(f"data/{category}", train=True, transform=train_transform)

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

    # Prepare checkpoints and weights folders
    checkpoints_folder = path_joiner(CHECKPOINTS_FOLDER, category)
    check_folder(checkpoints_folder, replace=True)
    check_folder(WEIGHTS_FOLDER)

    num_epochs = args.num_epochs
    train_losses = []

    print("Training begun")

    # Training loop
    for epoch in range(num_epochs):
        # Train epoch
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

        # Save model checkpoints
        if (epoch + 1) % args.save_checkpoint_freq == 0:
            save_path = path_joiner(checkpoints_folder, f"{category}_epoch_{epoch + 1}.pth")
            torch.save(
                model.state_dict(),
                save_path
            )
            print(f"Checkpoint saved at {str(save_path)}")

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

        # Save final parameters
        if (epoch + 1) == num_epochs:
            save_path = path_joiner(WEIGHTS_FOLDER, f"{category}.pth")
            torch.save(
                model.state_dict(),
                save_path
            )
            print(f"Final parameters saved at {str(save_path)}")

    print("Training ended")

    # Plot training loss
    plot_train_loss(category, train_losses)

    print("--------------------------------------------------------")


    # -------- THRESHOLD SELECTION --------

    model.eval()

    # LiteVAE inference
    img_scores, pixel_scores, _ = predict(
        model=model,
        dataloader=val_loader
    )

    # Compute image-level threshold
    mu = np.mean(img_scores)
    std = np.std(img_scores)
    img_level_th = mu + std * 2
    print(f"Image-level threshold: {img_level_th:.4f}")

    # Compute pixel-level threshold
    mu = np.mean(pixel_scores)
    std = np.std(pixel_scores)
    px_level_th = mu + std * 2
    print(f"Pixel-level threshold: {px_level_th:.4f}")

    check_folder(PARAMS_FOLDER)
    save_path = path_joiner(PARAMS_FOLDER, f"{category}.json")

    # Save threshold values
    with open(save_path, "w") as f:
        config = {
            "px_level_th": float(px_level_th),
            "img_level_th": float(img_level_th)
        }
        json.dump(config, f, indent=4)

    print(f"Threshold values saved at {str(save_path)}")
    print("--------------------------------------------------------")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    verbose: bool = False
) -> Tuple[float, Tensor, Tensor]:
    """
    Execute a single training epoch for the LiteVAE on the provided training data.

    Iterate through the training data loader and updates model parameters using
    the specified optimizer and criterion. Calculate the average loss for the epoch
    and return it along with the final batch input and output tensors.

    Args:
        model (nn.Module): VAE to be trained.
        train_loader (DataLoader): DataLoader providing the training data.
        device (torch.device): Device on which the training will be performed.
        optimizer (optim.Optimizer): Optimizer used for updating model parameters.
        criterion (nn.Module): Loss function used for calculating training loss.
        epoch (int): Current epoch number (0-based indexing).
        verbose (bool): Whether to display a progress bar using tqdm. Defaults to False.

    Returns:
        Tuple[float, Tensor, Tensor]: A tuple containing:

            - The average loss for the epoch as a float.
            - The final batch input tensor.
            - The final batch output tensor.
    """
    model.train()
    running_loss = 0.0

    # Initialize tqdm progress bar
    iter_data = tqdm(
        train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}", disable=not verbose
    )

    for x in iter_data:
        x = x.to(device)

        # LiteVAE inference
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


def val_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Tuple[Tensor, Tensor]:
    """
    Perform a single inference pass for the LiteVAE on the provided validation data.

    Args:
        model (nn.Module): VAE to be performed inference on.
        val_loader (DataLoader): DataLoader providing the validation data.
        device (torch.device): Device on which the inference will be performed.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing the validation inputs and the model outputs.
    """
    model.eval()

    # Get a single batch from the validation dataloader
    x = next(iter(val_loader))
    x = x.to(device)

    # LiteVAE inference
    outputs, _, _ = model(x)

    return x, outputs
