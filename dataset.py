import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, folder, phase="train", transform=None):
        path = os.path.join(str(Path(folder)), phase)

        file_list = []
        label_list = []
        mask_list = []
        for goodness in os.listdir(path):
            if os.path.isdir(os.path.join(path, goodness)):
                l = 0 if goodness == "good" else 1
                path_ = os.path.join(path, goodness)
                p = [os.path.join(path, goodness, f) for f in os.listdir(path_) if f.lower().endswith(".png")]
                l = [l] * len(p)
                gt_path = os.path.join(str(Path(folder)), "ground_truth", goodness)
                if os.path.isdir(gt_path):
                    g = [os.path.join(gt_path, f) for f in os.listdir(gt_path) if f.lower().endswith(".png")]
                else:
                    g = [None] * len(p)
                file_list.extend(p)
                label_list.extend(l)
                mask_list.extend(g)

        self.files = list(zip(file_list, label_list, mask_list))

        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_, label, mask = self.files[idx]
        img = Image.open(file_).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if mask is not None:
            mask = Image.open(mask)
            if self.transform is not None:
                mask = self.transform(mask)
        return img, label, mask


def collate_fn(batch):
    img, label, mask = zip(*batch)
    label = tuple(
        torch.tensor(l) for l in label
    )
    mask = tuple(
        torch.zeros(1, i.size(1), i.size(2), device=i.device, dtype=i.dtype)
        if m is None else m
        for i, m in zip(img, mask)
    )
    return (
        torch.stack(img),
        torch.stack(label),
        torch.stack(mask)
    )
