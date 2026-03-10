import json
import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, auc
from sympy.matrices.eigen import eigenvals_error_message
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms
from torchinfo import summary
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.ndimage import label

from dataset import CustomDataset, collate_fn
from loss import SSIMLoss, apply_gaussian_smoothing
from utils import plot_partial_roc
from model_new import LiteVAE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_workers = 0
batch_size = 16
category = "zipper"
img_size = 256

train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(f"data/{category}", transform=train_transform)
test_dataset = CustomDataset(f"data/{category}", phase="test", transform=test_transform)
# test_dataset_defect = CustomDataset("data/", phase="test", transform=transform)

print("Loaded dataset")

train_idx, valid_idx = train_test_split(range(len(train_dataset)), test_size=0.1)

# define samplers for getting training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
val_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(
    train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn
)
val_loader = DataLoader(
    train_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, collate_fn=collate_fn
)

# helper function to un-normalize and display an image
# def imshow(img):
#     # img = img / 2 + 0.5  # unnormalize
#     plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor to Image
#
# # obtain one batch of training images
# dataiter = iter(train_loader)
# images = next(dataiter)
# images = images.numpy() # convert images to numpy for display
#
# # plot the images in the batch, along with the corresponding labels
# fig = plt.figure(figsize=(12, 4))
# # display 20 images
# for idx in np.arange(batch_size):
#     ax = fig.add_subplot(2, int(batch_size/2), idx+1, xticks=[], yticks=[])
#     imshow(images[idx])
#     # ax.set_title(classes[labels[idx]])
# plt.show()
#
#
# rgb_img = np.squeeze(images[2])
# channels = ['red channel', 'green channel', 'blue channel']
# # print(classes[labels[2]])
#
# fig = plt.figure(figsize = (12, 4))
# for idx in np.arange(rgb_img.shape[0]):
#     ax = fig.add_subplot(1, 3, idx + 1)
#     img = rgb_img[idx]
#     ax.imshow(img, cmap='gray')
#     ax.set_title(channels[idx])
#     width, height = img.shape
#     thresh = img.max()/2.5
#     for x in range(width):
#         for y in range(height):
#             val = round(img[x][y],2) if img[x][y] !=0 else 0
#             ax.annotate(str(val), xy=(y,x),
#                     horizontalalignment='center',
#                     verticalalignment='center', size=8,
#                     color='white' if img[x][y]<thresh else 'black')
# plt.show()

model = LiteVAE(z_dim=128).to(device)
summary(model, input_data=torch.zeros(1, 3, img_size, img_size).to(device))

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

model.apply(init_weights)
# nn.init.constant_(model.fc_log_var.bias, -5)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-4)

# Lists to store loss values
train_losses = []

num_epochs = 200
ssim_loss = SSIMLoss().to(device)
# ssim_weight = 0.1
# kl_warmup_epochs = max(1, int(0.4 * num_epochs))


model_name = f"{category}.pth"
model_path = os.path.join("weights", model_name)
state_dict = torch.load(model_path, weights_only=True)
model.load_state_dict(state_dict)
print("Model loaded")
with open(os.path.join("weights", f"{category}.json"), "r") as f:
    config = json.load(f)
threshold = config["px_level_th"]
th = config["img_level_th"]


# Training loop
# print("Training begun")
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0
#     alpha = 0.7
#     beta = 0.01#min(1.0, float(epoch + 1) / float(kl_warmup_epochs))
#     printed = False
#
#     for imgs, _, _ in tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}", disable=False):
#         imgs = imgs.to(device)
#
#         # Forward pass
#         outputs, mu, log_var = model(imgs)
#         # loss, recon_mb, loss_dict_new = model.step(
#         #     imgs
#         # )
#
#         # Reconstruction + structural similarity + KL (with warm-up beta)
#         # recon_l1 = F.mse_loss(outputs, imgs, reduction="sum")
#         mse = F.mse_loss(outputs, imgs, reduction="mean")
#         ssim_term = ssim_loss(outputs, imgs)  # se ssim() restituisce similarità in [0,1]
#         # recon = alpha * ssim_term + (1 - alpha) * mse
#         kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
#         kl_div = kl_div.mean()
#         recon = 0.5 * ssim_term + 0.5 * mse
#         loss = recon + 0.00001 * kl_div
#         # recon_ssim = ssim_loss(outputs, imgs)
#         # loss = recon_ssim + 0.1 * kl_div
#
#         # Backward pass and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#     epoch_loss = running_loss / len(train_loader.dataset)
#     train_losses.append(epoch_loss)
#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}")
#     print(recon.item(), 0.00001 * kl_div.item())
#
#     if not printed and (epoch + 1) % 10 == 0:
#         with torch.no_grad():
#             plt.figure(figsize=(20, 10))
#             num_images = 4
#             imgs, _, _ = next(iter(test_loader))
#             outputs, _, _ = model(imgs)
#
#             # Display original images
#             for i in range(num_images):
#                 ax = plt.subplot(2, num_images, i + 1)
#                 plt.imshow(imgs[i].detach().cpu().numpy().transpose(1, 2, 0))
#                 plt.title(f"Original")
#                 plt.axis('off')
#
#             # Display reconstructed images
#             for i in range(num_images):
#                 ax = plt.subplot(2, num_images, i + num_images + 1)
#                 plt.imshow(outputs[i].detach().cpu().numpy().transpose(1, 2, 0))
#                 plt.title("Reconstructed")
#                 plt.axis('off')
#
#             plt.tight_layout()
#             plt.show()
#             printed = True
#
# print("Training ended")
# torch.save(model.state_dict(), os.path.join("weights", f"{category}.pth"))
#
# plt.figure(figsize=(10, 5))
# plt.plot(train_losses)
# plt.title('Convolutional VAE Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.show()
#
#
model.eval()
val_pixel_scores = []
img_scores = []

with torch.no_grad():
    for x, _, _ in val_loader:   # SOLO immagini normali
        x = x.to(device)
        x_hat, _, _ = model(x)

        error_map = torch.sum((x - x_hat)**2, dim=1, keepdim=True)
        error_map = apply_gaussian_smoothing(error_map, 3, 2.0)

        flat = error_map.view(error_map.size(0), -1)
        k = max(1, int(0.01 * flat.size(1)))
        topk = torch.topk(flat, k, dim=1)[0]
        score = topk.mean(dim=1)

        img_scores.extend(score.cpu().numpy())
        val_pixel_scores.extend(error_map.flatten().cpu().numpy())

# threshold = np.percentile(val_pixel_scores, 99.5)
# th = np.percentile(img_scores, 99)
mu = np.mean(img_scores)
std = np.std(img_scores)
th = mu + std * 2
mu = np.mean(val_pixel_scores)
std = np.std(val_pixel_scores)
threshold = mu + std * 2
print(f"Threshold px: {threshold}")
print(f"Threshold img: {th}")
with open(os.path.join("weights", f"{category}.json"), "w") as f:
    json.dump({"px_level_th": float(threshold), "img_level_th": float(th)}, f, indent=4)
threshold = config["px_level_th"]
th = config["img_level_th"]


# AUROC
y_true = []
y_scores = []
y_th = []
pixel_true = []
pixel_scores = []
pixel_th = []


model.eval()
with torch.no_grad():
    for x, label_, gt in test_loader:
        x = x.to(device)
        x_hat, _, _ = model(x)

        error_map = torch.sum((x - x_hat)**2, dim=1, keepdim=True)
        error_map = apply_gaussian_smoothing(error_map, 5, 1.0)

        pixel_true.extend((gt > 0.5).flatten().cpu().numpy().astype(int))
        pixel_scores.extend(error_map.flatten().cpu().numpy())
        pixel_th.extend((error_map > threshold).flatten().cpu().numpy().astype(int))

        flat = error_map.view(error_map.size(0), -1)
        k = max(1, int(0.01 * flat.size(1)))
        topk = torch.topk(flat, k, dim=1)[0]
        score = topk.mean(dim=1)

        y_true.extend(label_.cpu().numpy())
        y_scores.extend(score.flatten().cpu().numpy())
        y_th.extend((score > th).flatten().cpu().numpy().astype(int))


auroc = roc_auc_score(y_true, y_scores)
print("Image-level AUROC:", auroc)
auroc_px = roc_auc_score(pixel_true, pixel_scores, max_fpr=0.3)
print("Pixel-level AUROC:", auroc_px)

plot_partial_roc(y_true, y_scores, th, category)
plot_partial_roc(pixel_true, pixel_scores, threshold, category, max_fpr=True, pixel_level=True)

cls_y = classification_report(y_true, y_th)
print(cls_y)
cls_px = classification_report(pixel_true, pixel_th)
print(cls_px)

defect_scores = np.array(pixel_scores)[np.array(pixel_true)==1]
normal_scores = np.array(pixel_scores)[np.array(pixel_true)==0]

print(defect_scores.mean(), normal_scores.mean())


# FPS
model.eval()
# total_images = 0
# img_list = []
# for img_path in test_dataset.file_list:
#     img = Image.open(img_path).convert("RGB")
#     x = test_transform(img)
#     x = x.unsqueeze(0).to(device)
#     img_list.append(x)
# random.shuffle(img_list)

N = 1000
x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    for _ in range(100):
        model(x)

    start = time.perf_counter()

    for _ in range(N):
        x_hat, _, _ = model(x)

        error_map = torch.sum((x - x_hat) ** 2, dim=1, keepdim=True)
        error_map = apply_gaussian_smoothing(error_map, 5, 1.0)

        thresholded = (error_map > threshold).cpu().numpy().astype(int)

        flat = error_map.view(error_map.size(0), -1)
        k = max(1, int(0.01 * flat.size(1)))
        topk = torch.topk(flat, k, dim=1)[0]
        score = topk.mean(dim=1)
        pred = (score > th).cpu().numpy().astype(int)

    end = time.perf_counter()

avg_time_ms = (end - start) / N * 1000
fps = 1000 / avg_time_ms
print(f"Average inference time: {avg_time_ms:.4f} ms, FPS: {fps:.4f} fps")


model.eval()
with torch.no_grad():
    images, label_, gt = next(iter(test_loader))

    # Get reconstructed images
    reconstructed, _, _ = model(images)

    error_map = torch.sum((images - reconstructed) ** 2, dim=1, keepdim=True)
    error_map = apply_gaussian_smoothing(error_map, kernel_size=5, sigma=1.0)
    gt = gt.cpu().numpy()

    # Convert to numpy for visualization
    images = images.cpu().numpy()
    thresholded = (error_map.cpu().numpy() > threshold).astype(int)

    # score = error_map.amax(dim=(2, 3))
    flat = error_map.view(error_map.size(0), -1)
    k = max(1, int(0.01 * flat.size(1)))
    topk = torch.topk(flat, k, dim=1)[0]
    score = topk.mean(dim=1)
    score = (score > th).cpu().numpy().astype(int)

    error_map = error_map.cpu().numpy()

    # Plot original and reconstructed images
    plt.figure(figsize=(20, 8))
    num_images = 16

    # Display original images
    for i in range(num_images):
        plt.subplot(4, num_images, i + 1)
        plt.imshow(images[i].transpose(1, 2, 0))
        # plt.title(f"{'Defect' if labels[i] == 1 else 'Normal'}")
        plt.axis('off')

    # Display ground truth masks
    for i in range(num_images):
        plt.subplot( 4, num_images, i + num_images + 1)
        plt.imshow(gt[i].transpose(1, 2, 0), cmap="gray")
        plt.axis('off')

    # Display reconstructed images
    for i in range(num_images):
        plt.subplot(4, num_images, i + num_images * 2 + 1)
        plt.imshow(error_map[i].transpose(1, 2, 0), cmap="jet")
        plt.axis('off')

    # Display error maps
    for i in range(num_images):
        plt.subplot(4, num_images, i + num_images * 3 + 1)
        plt.imshow(thresholded[i].transpose(1, 2, 0), cmap="gray")
        plt.title("Defect" if score[i] else "Normal", color="green" if score[i] == label_[i] else "red")
        plt.axis('off')

    plt.tight_layout()
    plt.show()
