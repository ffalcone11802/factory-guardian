from typing import Callable, Optional, Union, List, Tuple
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

from dataset.sample import Sample, IMG_EXTS
from utils.folder import path_joiner, list_dir


class MVTecDataset(Dataset):
    """
    The MVTecDataset class handles image datasets from MVTec anomaly detection datasets
    for training and testing.

    This class provides functionality to load and preprocess image data, including
    masks when available, depending on the specified pipeline phase ('train' or 'test').

    Args:
        folder (Union[str, Path]): Path to the folder containing the dataset category to read.
        phase (str): Pipeline phase ('train' or 'test'). Defaults to 'train'.
        transform (Callable, optional): Transform function or pipeline for preprocessing images.
            Defaults to None.
    """

    def __init__(
        self,
        folder: Union[str, Path],
        phase: str = "train",
        transform: Optional[Callable] = None
    ):
        self.phase = phase
        self.transform = transform

        # Base path to the folder containing the data to be read
        base = path_joiner(folder, phase)

        # If the phase is "train"...
        if phase == "train":
            self.samples = []

            # ...read the images from the "good" folder
            good_dir = path_joiner(base, "good")
            if good_dir.is_dir():
                imgs = self._list_images(good_dir)
                self.samples = [Sample(img=i, label=0, mask=None) for i in imgs]

        # If the phase is "test"...
        elif phase == "test":
            self.samples = []

            # ...read images and masks from the "good" and "defective" folders
            for goodness in list_dir(base):

                # Set the label to 0 for "good" images and 1 for "defective" images
                label = 0 if goodness.name == "good" else 1

                # Read images from the corresponding folder
                imgs = self._list_images(goodness)

                # Read masks from the corresponding "ground_truth" folder
                gt_dir = path_joiner(folder, "ground_truth", goodness.name)
                if gt_dir.is_dir():
                    masks = self._list_images(gt_dir)
                else:
                    # If the "ground_truth" folder doesn't exist, assume no masks
                    masks = [None] * len(imgs)

                # If the number of images and masks don't match, raise an error
                if masks and len(masks) != len(imgs):
                    raise ValueError(
                        f"Mask count mismatch for '{goodness.name}': "
                        f"{len(masks)} masks != {len(imgs)} images"
                    )

                self.samples.extend(
                    Sample(img=img, label=label, mask=mask)
                    for img, mask in zip(imgs, masks)
                )

        # If the phase is neither "train" nor "test", raise an error
        else:
            raise ValueError("Phase must be either 'train' or 'test'")

    def __len__(self) -> int:
        """
        Get the dataset length.

        Returns:
            int: Length of the dataset, i.e., the number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, int, Optional[Tensor]]]:
        """
        Retrieve an item from the dataset at the specified index.

        If the phase is "train", only the transformed image is returned.
        Otherwise, the transformed image, label, and optionally the transformed mask
        (if it exists) are returned.

        Args:
            idx (int): Index of the dataset item to retrieve.

        Returns:
            Union[Tensor, Tuple[Tensor, int, Optional[Tensor]]]: Transformed image
                and optionally the label and mask, depending on the phase.
                Returns only the transformed image during the "train" phase.
        """
        sample = self.samples[idx]

        # Read and transform image
        img = Image.open(sample.img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Return only the transformed image during the "train" phase
        if self.phase == "train":
            return img

        # Read and transform mask (if available)
        mask = None
        if sample.mask is not None:
            mask = Image.open(sample.mask)
            if self.transform is not None:
                mask = self.transform(mask)

        return img, sample.label, mask

    @staticmethod
    def _list_images(dir_path: Path) -> List[Path]:
        """
        List image files in the specified directory.

        Args:
            dir_path (Path): Path to the folder containing image files.

        Returns:
            List[Path]: List of paths to valid image files (PNG, JPG, JPEG).
        """
        imgs = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        return imgs

    def collate_fn(
        self,
        batch: List[Union[Tensor, Tuple[Tensor, int, Optional[Tensor]]]]
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        Process a batch of data by stacking its elements based on the
        current phase (train or test).

        If the phase is 'train', stack the elements in the batch and return them
        as a single tensor. Otherwise, separate the batch into images, labels, and
        masks, and process each respectively. If no mask is available (no defects),
        create an all-zero tensor.

        Args:
            batch (List[Union[Tensor, Tuple[Tensor, int, Optional[Tensor]]]]):
                List of data elements.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor, Tensor]]: If the phase is set to 'train',
                returns a tensor by stacking the batch. Otherwise, returns a tuple containing
                stacked tensors for images, labels, and masks.
        """
        # If the phase is 'train', stack the elements in the batch (only images)
        # and return them as a single tensor
        if self.phase == "train":
            return torch.stack(batch)

        # Separate the batch into images, labels, and masks
        img, label, mask = zip(*batch)

        # Convert labels to tensors
        label = tuple(
            torch.tensor(l) for l in label
        )

        # Create all-zero tensors if no mask is available
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
