from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from factory_guardian.utils.folder import path_joiner, check_folder, PLOTS_FOLDER

check_folder(PLOTS_FOLDER)


def plot_train_loss(category: str, train_losses: List[float]):
    """
    Plot the training loss over epochs. Save the plot to a file.

    Args:
        category (str): Category name associated with the training run.
        train_losses (List[float]): List of training loss values.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, color="tab:blue")
    plt.title("LiteVAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Save to file
    save_path = path_joiner(PLOTS_FOLDER, f"{category}_train_loss.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Training loss plotted, saved at {str(save_path)}")


def plot_qualitative_results(
    category: str,
    img: np.ndarray,
    label: np.ndarray,
    gt: np.ndarray,
    anom_map: np.ndarray,
    anom_score: np.ndarray,
    px_pred: np.ndarray,
    img_pred: np.ndarray,
    max_images: int = 8
):
    """
    Plot qualitative results for a given set of images, displaying original images, ground truth masks,
    anomaly maps with scores, and pixel-level predictions with image-level classifications.
    Save the plot to a file.

    Args:
        category (str): Category name of the dataset or task.
        img (np.ndarray): Array of input images with dimensions (N, C, H, W), where N is the number of samples.
        label (np.ndarray): Array of ground truth labels for images (0 for normal, 1 for defect).
        gt (np.ndarray): Array of ground truth masks with dimensions (N, H, W).
        anom_map (np.ndarray): Array of computed anomaly maps with dimensions (N, H, W).
        anom_score (np.ndarray): Array of anomaly scores for images.
        px_pred (np.ndarray): Array of pixel-level predictions with dimensions (N, H, W).
        img_pred (np.ndarray): Array of image-level classification predictions with dimensions (N,).
            Binary values (0 or 1).
        max_images (int, optional): Maximum number of images to display in the plot. Defaults to 8.
    """
    # Limit the number of images to keep the grid readable
    num_images = min(img.shape[0], max_images)
    plt.figure(figsize=(2.5 * num_images, 8))

    # Helper function to calculate the subplot index
    def _subplot(row, col):
        return row * num_images + col + 1

    # Row 1: original images
    for i in range(num_images):
        plt.subplot(4, num_images, _subplot(0, i))
        plt.imshow(img[i].transpose(1, 2, 0))
        plt.axis("off")

    # Row 2: ground truth masks
    for i in range(num_images):
        plt.subplot(4, num_images, _subplot(1, i))
        plt.imshow(gt[i].transpose(1, 2, 0), cmap="gray")
        plt.axis("off")

    # Row 3: anomaly maps + scores
    for i in range(num_images):
        plt.subplot(4, num_images, _subplot(2, i))
        plt.imshow(anom_map[i].transpose(1, 2, 0), cmap="jet")
        plt.title(f"{anom_score[i]:.4f}", fontsize=9)
        plt.axis("off")

    # Row 4: pixel predictions + image-level classification
    for i in range(num_images):
        plt.subplot(4, num_images, _subplot(3, i))
        plt.imshow(px_pred[i].transpose(1, 2, 0), cmap="gray")
        # Green if correct, red if wrong
        is_correct = img_pred[i] == label[i]
        color = "green" if is_correct else "red"
        plt.title("Defect" if img_pred[i] else "Normal", color=color, fontsize=9)
        plt.axis("off")

    plt.tight_layout()

    # Save to file
    save_path = path_joiner(PLOTS_FOLDER, f"{category}_qualitative_results.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Qualitative results plotted, saved at {str(save_path)}")


def plot_roc_curve(
    category: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    chosen_threshold: float,
    max_fpr: float = 1.0,
    scope: str = "Pixel"
):
    """
    Plot a Receiver Operating Characteristic (ROC) curve. Save it to a file.

    Compute the ROC curve using the provided true labels and scores, highlight a chosen threshold
    on the curve, and optionally imposes a maximum false positive rate for visualization.

    Args:
        category (str): Category name associated with the ROC plot.
        y_true (np.ndarray): True binary labels.
        scores (np.ndarray): Predicted scores.
        chosen_threshold (float): Threshold value to highlight on the curve.
        max_fpr (float): Maximum false positive rate for visual reference. Defaults to 1.0.
        scope (str): Scope level (e.g., "Pixel"). Used in the plot title and file name. Defaults to "Pixel".
    """
    # Compute ROC curve and AUC for each class
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

    # If max_fpr is less than 1.0, add a vertical line at max_fpr
    if max_fpr < 1.0:
        plt.plot([0.3, 0.3], [0, 1], color='red', lw=1, linestyle='--')

    # Choose the closest threshold to the provided one
    idx = np.argmin(np.abs(thresholds - chosen_threshold))
    plt.scatter(fpr[idx], tpr[idx], color='red', s=100, label=f'Th = {chosen_threshold}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{scope}-level ROC Curve for '{category}'")
    plt.legend(loc='lower right')
    plt.grid(True)

    # Save to file
    save_path = path_joiner(PLOTS_FOLDER, f"{category}_{scope}_level_roc.png")
    plt.savefig(save_path)
    plt.close()

    print(f"{scope}-level ROC curve plotted, saved at {str(save_path)}")
