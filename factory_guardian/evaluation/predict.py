from typing import List, Tuple, Any, Optional, Union
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader

from factory_guardian.evaluation.postprocess import post_process


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    test: bool = False,
    img_level_th: Optional[float] = None,
    px_level_th: Optional[float] = None,
) -> Tuple[
        List[Union[Tuple[int, float, int], float]],
        List[Union[Tuple[int, float, int], float]],
        Optional[tuple]
    ]:
    """
    Perform LiteVAE inference for anomaly detection and optionally generate predictions.

    Support both training and testing modes. During testing mode, compute predictions
    and collect data for evaluation metrics like image-level AUROC and pixel-level AUROC.

    Args:
        model (nn.Module): VAE used to generate reconstructions.
        dataloader (DataLoader): DataLoader providing input data for prediction.
        test (bool): Whether the model is in testing mode. Defaults to False.
        img_level_th (float, optional): Threshold for classifying image-level anomalies.
            Only applicable when `test` is True. Defaults to None.
        px_level_th (float, optional): Threshold for classifying pixel-level anomalies.
            Only applicable when `test` is True. Defaults to None.

    Returns:
        Tuple[List[Union[Tuple[int, float, int], float]], List[Union[Tuple[int, float, int], float]], Optional[tuple]]:
            A tuple containing:

            - List of image-level results. If `test` is True, each entry is a tuple
              (true_label, anomaly_score, predicted_label). Otherwise, a list of anomaly scores.
            - List of pixel-level results. If `test` is True, each entry is a tuple
              (true_pixel_label, pixel_anomaly_score, predicted_pixel_label). Otherwise,
              a list of anomaly scores.
            - Last batch results for visualization in testing mode. If `test` is False,
              this is None.
    """
    img_true, img_scores, img_pred = [], [], []
    px_true, px_scores, px_pred = [], [], []

    last_batch = None

    with torch.inference_mode():
        for batch in dataloader:
            x = batch[0] if isinstance(batch, tuple) else batch

            # LiteVAE inference
            outputs, _, _ = model(x)

            # Compute anomaly maps and anomaly scores
            anom_map, anom_score = post_process(
                inputs=x,
                outputs=outputs,
                kernel_size=5,
                sigma=1.0,
            )

            # Save results for image-level AUROC and pixel-level AUROC
            img_scores.extend(anom_score.cpu().numpy())
            px_scores.extend(anom_map.cpu().numpy().ravel())

            if test:
                label, gt = batch[1], batch[2]

                # Save labels for image-level AUROC
                img_true.extend(label.cpu().numpy())

                # Save ground-truth masks for pixel-level AUROC
                gt_int = gt.to(torch.int32)
                px_true.extend(gt_int.cpu().numpy().ravel())

                # Compute predictions based on thresholds
                pred_px = (anom_map > px_level_th).to(torch.int32)
                pred_img = (anom_score > img_level_th).to(torch.int32)

                img_pred.extend(pred_img.cpu().numpy())
                px_pred.extend(pred_px.cpu().numpy().ravel())

                # Save the last batch for visualization
                last_batch = (
                    x.cpu().numpy(),
                    label.cpu().numpy(),
                    gt_int.cpu().numpy(),
                    anom_map.cpu().numpy(),
                    anom_score.cpu().numpy(),
                    pred_px.cpu().numpy(),
                    pred_img.cpu().numpy(),
                )

    # Adjust the output format based on the test argument
    img_items = zip(*(img_true, img_scores, img_pred)) if test else img_scores
    px_items = zip(*(px_true, px_scores, px_pred)) if test else px_scores

    return (
        list(img_items),
        list(px_items),
        last_batch
    )


def predict_single(
    model: Any,
    x: Union[Tensor, np.ndarray]
) -> Tuple[Tensor, Tensor]:
    """
    Perform LiteVAE inference for anomaly detection and generate predictions
    on a single input using fixed thresholds (used only for inference time evaluation).

    Args:
        model (Any): VAE used to generate reconstructions.
        x (Union[Tensor, np.ndarray]): Input data for inference. If not a Tensor,
            convert it to a Tensor.

    Returns:
        Tuple[Tensor, Tensor]: A tuple containing binary image-level anomaly predictions
            and binary pixel-level anomaly predictions.
    """
    # LiteVAE inference
    outputs, _, _ = model(x)

    # Convert inputs and outputs to Tensors if they are not already
    if not isinstance(x, Tensor):
        x = torch.tensor(x)
    if not isinstance(outputs, Tensor):
        outputs = torch.tensor(outputs)

    # Compute anomaly maps and anomaly scores
    anom_map, anom_score = post_process(
        inputs=x,
        outputs=outputs
    )

    # Compute predictions based on fixed thresholds
    pred_px = (anom_map > 0.5).to(torch.int32)
    pred_img = (anom_score > 0.5).to(torch.int32)

    return pred_img, pred_px
