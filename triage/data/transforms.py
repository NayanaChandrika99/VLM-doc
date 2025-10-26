"""
Image transformation utilities shared across RVL-CDIP and DocLayNet adapters.

These helpers centralise torchvision augmentation pipelines so datasets and
training scripts remain consistent with configuration defaults.
"""

from __future__ import annotations

from typing import Literal

from PIL import Image

try:
    import torchvision.transforms as T
except ImportError as exc:  # pragma: no cover - exercised via unit tests
    raise ImportError(
        "torchvision is required for triage.data.transforms. "
        "Install torchvision >=0.14 to continue."
    ) from exc


NORMALIZATION_MAP = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "none": None,
}


def get_default_image_transform(
    image_size: int = 768,
    normalization: Literal["imagenet", "none"] = "imagenet",
    to_grayscale: bool = False,
) -> T.Compose:
    """
    Build the canonical Compose transform for document pages.

    Args:
        image_size: Target square size applied via resize + center crop.
        normalization: Which mean/std pair to apply; set to ``none`` to skip.
        to_grayscale: Convert incoming images to grayscale before tensorisation.

    Returns:
        torchvision Compose object ready to map PIL images to torch tensors.
    """

    base_transform = _to_grayscale if to_grayscale else _to_rgb
    pipeline = [
        T.Lambda(base_transform),
        T.Resize(image_size, interpolation=Image.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ]
    if normalization not in NORMALIZATION_MAP:
        valid = ", ".join(sorted(NORMALIZATION_MAP))
        raise ValueError(f"Unknown normalization '{normalization}'. Expected one of: {valid}.")

    if normalization != "none":
        mean, std = NORMALIZATION_MAP[normalization]
        pipeline.append(T.Normalize(mean=mean, std=std))

    return T.Compose([step for step in pipeline if step is not None])


def _to_rgb(image: Image.Image) -> Image.Image:
    """Ensure the incoming PIL image is RGB."""

    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _to_grayscale(image: Image.Image) -> Image.Image:
    """Convert the PIL image to grayscale while keeping three channels."""

    return image.convert("L").convert("RGB")
