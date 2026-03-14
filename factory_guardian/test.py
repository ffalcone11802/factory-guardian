from typing import Tuple, Any, Union
from argparse import Namespace
import numpy as np
import json
import time
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, classification_report

from factory_guardian.dataset import MVTecDataset
from factory_guardian.evaluation import predict, predict_single
from factory_guardian.model import LiteVAE
from factory_guardian.utils.folder import path_joiner, WEIGHTS_FOLDER, PARAMS_FOLDER
from factory_guardian.utils.plot import plot_qualitative_results, plot_roc_curve
from factory_guardian.utils.seed import set_seed


def test(args: Namespace):
    """
    Main function to test the LiteVAE and evaluate its performance.

    Args:
        args (Namespace): Command-line arguments.
    """
    # Select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set seed
    set_seed(args.seed)

    img_size = 256
    category = args.category
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Test pre-processing
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Test dataset
    test_dataset = MVTecDataset(f"data/{category}", train=False, transform=test_transform)

    # Test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=test_dataset.collate_fn
    )

    # Model
    model = LiteVAE(latent_dim=args.latent_dim).to(device)

    # Load weights
    model_path = path_joiner(WEIGHTS_FOLDER, f"{category}.pth")
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Model parameters loaded from {str(model_path)}")
    print("--------------------------------------------------------")

    # Load threshold values
    th_path = path_joiner(PARAMS_FOLDER, f"{category}.json")
    with open(th_path, "r") as f:
        config = json.load(f)
    px_level_th = config["px_level_th"]
    img_level_th = config["img_level_th"]


    # -------- TESTING --------

    model.eval()

    # Complete inference
    imgs, pixels, last_batch = predict(
        model=model,
        dataloader=test_loader,
        test=True,
        img_level_th=img_level_th,
        px_level_th=px_level_th
    )

    img_true, img_scores, img_pred = zip(*imgs)
    px_true, px_scores, px_pred = zip(*pixels)

    # AUROC
    img_level_auroc = roc_auc_score(img_true, img_scores)
    px_level_auroc = roc_auc_score(px_true, px_scores, max_fpr=0.3)
    print(f"Image-level AUROC: {img_level_auroc:.4f}")
    print(f"Pixel-level AUROC: {px_level_auroc:.4f}")
    print("--------------------------------------------------------")

    # Classification report
    img_cls = classification_report(img_true, img_pred)
    px_cls = classification_report(px_true, px_pred)
    print("Image-level classification report\n")
    print(img_cls)
    print("Pixel-level classification report\n")
    print(px_cls)
    print("--------------------------------------------------------")

    # ROC curves
    plot_roc_curve(category, img_true, img_scores, img_level_th, scope="Image")
    plot_roc_curve(category, px_true, px_scores, px_level_th, max_fpr=0.3, scope="Pixel")

    # Plot last batch
    if last_batch is not None:
        plot_qualitative_results(category, *last_batch)

    print("--------------------------------------------------------")

    # Inference time and FPS
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    ms, fps = inference_time(
        model=model,
        x=dummy_input,
        warm_up_epochs=args.num_warm_up_epochs,
        N=args.num_inference_epochs
    )
    print(f"PyTorch - Average inference time: {ms:.4f} ms, FPS: {fps:.4f} fps")


def inference_time(
    model: Any,
    x: Union[Tensor, np.ndarray],
    warm_up_epochs: int = 100,
    N: int = 1000
) -> Tuple[float, float]:
    """
    Evaluate the average inference time and frames per second (FPS)
    for the LiteVAE using the provided input data.

    Perform a warmup phase followed by the actual timed inference loop to
    compute the performance metrics.

    Args:
        model (Any): VAE used to generate reconstructions.
        x (Union[Tensor, np.ndarray]): Input data for inference.
        warm_up_epochs (int): Number of warm-up iterations to execute before
            measuring the inference time. Defaults to 100.
        N (int): Number of inference iterations used to compute average time and FPS.
            Defaults to 1000.

    Returns:
        Tuple[float, float]: A tuple containing the average inference time in milliseconds
            and the frames per second (FPS).
    """
    with torch.inference_mode():
        # Warmup
        for _ in range(warm_up_epochs):
            model(x)

        start = time.perf_counter()

        # Complete inference
        for _ in range(N):
            predict_single(model, x)

        end = time.perf_counter()

    # Compute average time and FPS
    avg_time_ms = (end - start) / N * 1000
    fps = 1000 / avg_time_ms

    return avg_time_ms, fps
