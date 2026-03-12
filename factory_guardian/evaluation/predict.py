import torch
from torch import Tensor

from factory_guardian.evaluation.postprocess import post_process


def predict(
    model,
    dataloader,
    test = False,
    img_level_th = None,
    px_level_th = None,
):
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

    img_items = zip(*(img_true, img_scores, img_pred)) if test else img_scores
    px_items = zip(*(px_true, px_scores, px_pred)) if test else px_scores

    return (
        list(img_items),
        list(px_items),
        last_batch
    )


def predict_dummy(model, x):
    # LiteVAE inference
    outputs, _, _ = model(x)

    if not isinstance(x, Tensor):
        x = torch.tensor(x)

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
