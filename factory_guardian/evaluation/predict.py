import torch
from torch import Tensor

from factory_guardian.evaluation.postprocess import post_process


def predict(
    model,
    dataloader,
    img_level_th = None,
    px_level_th = None,
):
    img_true, img_scores, img_pred = [], [], []
    px_true, px_scores, px_pred = [], [], []

    last_batch = None

    with torch.inference_mode():
        for x, label, gt in dataloader:
            # LiteVAE inference
            outputs, _, _ = model(x)

            # Compute anomaly maps and anomaly scores
            anom_map, anom_score = post_process(
                inputs=x,
                outputs=outputs,
                kernel_size=5,
                sigma=1.0,
            )

            # Save results for image-level AUROC
            img_true.extend(label.cpu().numpy())
            img_scores.extend(anom_score.cpu().numpy())

            # Save results for pixel-level AUROC
            gt_int = gt.to(torch.int32)
            px_true.extend(gt_int.cpu().numpy().ravel())
            px_scores.extend(anom_map.cpu().numpy().ravel())

            if img_level_th is not None and px_level_th is not None:
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

    img_items = (img_true, img_scores) + ((img_pred,) if img_pred else ())
    px_items = (px_true, px_scores) + ((px_pred,) if px_pred else ())

    return (
        list(zip(*img_items)),
        list(zip(*px_items)),
        last_batch
    )


def predict_dummy(model, x):
    # LiteVAE inference
    outputs, _, _ = model(x)

    if not isinstance(outputs, Tensor):
        outputs = torch.tensor(outputs)

    # Compute anomaly maps and anomaly scores
    anom_map, anom_score = post_process(
        inputs=x,
        outputs=outputs
    )

    # Compute predictions based on dummy thresholds
    pred_px = (anom_map > 0.5).to(torch.int32)
    pred_img = (anom_score > 0.5).to(torch.int32)

    return pred_img, pred_px
