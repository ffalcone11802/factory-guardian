from typing import Callable, Optional, Union, List, Tuple, Any
import torch
from torch import Tensor
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

from factory_guardian.dataset.sample import Sample, IMG_EXTS
from factory_guardian.utils.folder import path_joiner, list_folders


class MVTecDataset(Dataset):
    """
    The MVTecDataset class handles image datasets from MVTec anomaly detection dataset
    for training and testing.

    This class provides functionality to load and preprocess image data, including
    masks when available, depending on the specified phase (train or test).

    Args:
        folder (Union[str, Path]): Path to the folder containing the dataset category to read.
        train (bool): Whether to load the training or testing data. Defaults to True.
        transform (Callable, optional): Transform function or pipeline for preprocessing images.
            Defaults to None.
    """

    def __init__(
        self,
        folder: Union[str, Path],
        train: bool = True,
        transform: Optional[Callable] = None
    ):
        self.train = train
        self.transform = transform

        # Base path to the folder containing the data to be read
        phase = "train" if train else "test"
        base = path_joiner(folder, phase)

        # If the phase is "train"...
        if train:
            self.samples = []

            # ...read the images from the "good" folder
            good_dir = path_joiner(base, "good")
            if good_dir.is_dir():
                imgs = self._list_images(good_dir)
                self.samples = [Sample(img=i, label=0, mask=None) for i in imgs]

        # If the phase is "test"...
        else:
            self.samples = []

            # ...read images and masks from the "good" and "defective" folders
            for goodness in list_folders(base):

                # Set the label to 0 for "good" images and 1 for "defective" images
                label = 0 if goodness.name == "good" else 1

                # Read images from the corresponding folder
                imgs = self._list_images(goodness)

                # Read masks from the corresponding "ground_truth" folder
                gt_dir = path_joiner(folder, "ground_truth", goodness.name)
                if gt_dir.is_dir():
                    masks = self._list_images(gt_dir)
                else:
                    # If the "ground_truth" folder does not exist, assume no masks
                    masks = [None] * len(imgs)

                # If the number of images and masks do not match, raise an error
                if masks and len(masks) != len(imgs):
                    raise ValueError(
                        f"Mask count mismatch for '{goodness.name}': "
                        f"{len(masks)} masks != {len(imgs)} images"
                    )

                self.samples.extend(
                    Sample(img=img, label=label, mask=mask)
                    for img, mask in zip(imgs, masks)
                )

    def __len__(self) -> int:
        """
        Get the dataset length.

        Returns:
            int: Length of the dataset, i.e., the number of samples.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Union[Any, Tuple[Any, int, Optional[Any]]]:
        """
        Retrieve and preprocess a sample from the dataset based on its index.

        Open and transform the image at the provided index. If in the train phase,
        only the transformed image is returned. If in the test phase, retrieve
        and transform the corresponding mask (if available) and return it alongside the image
        and its associated label.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Union[Any, Tuple[Any, int, Optional[Any]]]: The transformed image if in the train phase;
                a tuple containing the transformed image, the label, and optionally
                the transformed mask (if available) if in the test phase.
        """
        sample = self.samples[idx]

        # Read and transform image
        img = Image.open(sample.img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Return only the transformed image during the "train" phase
        if self.train:
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
        List image files in the specified folder.

        Args:
            dir_path (Path): Path to the folder containing image files.

        Returns:
            List[Path]: List of paths to valid image files (PNG, JPG, JPEG).
        """
        imgs = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        return imgs

    def collate_fn(
        self,
        batch: List[Union[Any, Tuple[Any, int, Optional[Any]]]]
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        Utility function to use in the corresponding DataLoader to process a batch of data
        and stack its elements based on the current phase (train or test).

        If in the train phase, stack the elements in the batch and return them
        as a single tensor. If in the test phase, separate the batch into images, labels,
        and masks, and process each respectively. If no mask is available (no defects),
        create an all-zero tensor.

        Args:
            batch (List[Union[Any, Tuple[Any, int, Optional[Any]]]]): List of data elements.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor, Tensor]]: If in the train phase,
                returns a tensor by stacking the batch. If in the test phase,
                returns a tuple containing stacked tensors for images, labels, and masks.
        """
        # If in the train phase, stack the elements in the batch (only images)
        # and return them as a single tensor
        if self.train:
            return torch.stack(batch)

        # Separate the batch into images, labels, and masks
        img, label, mask = zip(*batch)

        # Convert labels to tensors
        label = tuple(
            torch.tensor(l) for l in label
        )

        mask = tuple(
            # If no mask is available, create all-zero tensors...
            torch.zeros(1, i.size(1), i.size(2), device=i.device, dtype=i.dtype) if m is None
            # ...otherwise, ensure that the mask is a binary tensor (0 or 1)
            else m > 0.5
            for i, m in zip(img, mask)
        )

        return (
            torch.stack(img),
            torch.stack(label),
            torch.stack(mask)
        )
