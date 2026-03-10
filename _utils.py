import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_partial_roc(y_true, scores, chosen_threshold, category, max_fpr=False, pixel_level=False):
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
    plt.title(f"{'Pixel' if pixel_level else 'Image'}-level ROC Curve for '{category}'")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


# map_ = {
#     0: 0, 1: 3, 2: 6, 3: 9, 4: 12, 5: 15
# }
#
# mapping = {
#     "dec.0.weight": "dec.0.conv.weight",
#     "dec.0.bias": "dec.0.conv.bias",
#     "dec.1.weight": "dec.0.bn.weight",
#     "dec.1.bias": "dec.0.bn.bias",
#     "dec.1.running_mean": "dec.0.bn.running_mean",
#     "dec.1.running_var": "dec.0.bn.running_var",
#     "dec.1.num_batches_tracked": "dec.0.bn.num_batches_tracked",
# }
#
# def build_mapping():
#     new_dict = {}
#     for new, old in map_.items():
#         for k, v in mapping.items():
#             if k.count("0") > 0:
#                 key = k.replace("0", str(old))
#             elif k.count("1") > 0:
#                 key = k.replace("1", str(old+1))
#             value = v.replace("0", str(new))
#             new_dict[key] = value
#     new_dict["dec.18.weight"] = "dec.6.conv.weight"
#     new_dict["dec.18.bias"] = "dec.6.conv.bias"
#     return new_dict
#
# def rename_dict(old_dict):
#     mapping_ = build_mapping()
#     new_dict = {}
#     for k, v in old_dict.items():
#         if k in mapping_:
#             key = mapping_[k]
#         else:
#             key = k
#         if key.startswith("dec") or key.startswith("enc"):
#             key = key[:3] + "oder" + key[3:]
#         new_dict[key] = v
#     return OrderedDict(new_dict)
