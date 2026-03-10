from typing import Optional
from dataclasses import dataclass
from pathlib import Path

# Allowed image extensions
IMG_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class Sample:
    """
    The Sample class represents a sample data structure with an image, label,
    and optional mask.

    This class provides a convenient way to store and pass around a single data sample,
    consisting of an image file path, a label, and optionally a mask file path.

    Attributes:
        img (Path): Path to the image file.
        label (int): Label corresponding to the image.
        mask (Path, optional): Path to the mask file. Defaults to None.
    """

    img: Path
    label: int
    mask: Optional[Path] = None
