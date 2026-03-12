import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from factory_guardian.utils.folder import path_joiner, check_dir, PLOTS_FOLDER

check_dir(PLOTS_FOLDER)


def plot_train_loss(category, train_losses):
    """Plot training loss over epochs."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, color="tab:blue")
    plt.title("LiteVAE Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    save_path = path_joiner(PLOTS_FOLDER, f"{category}_train_loss.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Training loss plotted, saved at {str(save_path)}")


def plot_qualitative_results(
    category, img, label, gt, anom_map, anom_score, px_pred, img_pred, max_images=8
):
    """Plot a qualitative grid: input, GT, anomaly map, pixel prediction."""
    # Limit the number of images to keep the grid readable
    num_images = min(img.shape[0], max_images)
    plt.figure(figsize=(2.5 * num_images, 8))

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

    # Row 3: anomaly maps + score
    for i in range(num_images):
        plt.subplot(4, num_images, _subplot(2, i))
        plt.imshow(anom_map[i].transpose(1, 2, 0), cmap="jet")
        plt.title(f"{anom_score[i]:.4f}", fontsize=9)
        plt.axis("off")

    # Row 4: pixel prediction + image-level classification
    for i in range(num_images):
        plt.subplot(4, num_images, _subplot(3, i))
        plt.imshow(px_pred[i].transpose(1, 2, 0), cmap="gray")
        # Green if correct, red if wrong
        is_correct = img_pred[i] == label[i]
        color = "green" if is_correct else "red"
        plt.title("Defect" if img_pred[i] else "Normal", color=color, fontsize=9)
        plt.axis("off")

    plt.tight_layout()

    save_path = path_joiner(PLOTS_FOLDER, f"{category}_qualitative_results.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Qualitative results plotted, saved at {str(save_path)}")


def plot_partial_roc(
    category, y_true, scores, chosen_threshold, max_fpr=1.0, scope="pixel"
):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # linea diagonale

    if max_fpr:
        plt.plot([0.3, 0.3], [0, 1], color='red', lw=1, linestyle='--')

    # Trova il punto più vicino sulla curva
    idx = np.argmin(np.abs(thresholds - chosen_threshold))
    plt.scatter(fpr[idx], tpr[idx], color='red', s=100, label=f'Th = {chosen_threshold}')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"{scope}-level ROC Curve for '{category}'")
    plt.legend(loc='lower right')
    plt.grid(True)

    save_path = path_joiner(PLOTS_FOLDER, f"{category}_{scope}_level_roc.png")
    plt.savefig(save_path)
    plt.close()

    print(f"{scope}-level ROC curve plotted, saved at {str(save_path)}")
