from __future__ import annotations

from typing import Literal

import numpy as np

from synth.placer import Placement


BlendMethod = Literal["none", "gaussian", "alpha"]


def composite_cutout(
    background: np.ndarray,
    donor_rgb: np.ndarray,
    donor_mask: np.ndarray,
    placement: Placement,
    blend_method: BlendMethod,
    gaussian_radius: float = 1.5,
) -> np.ndarray:
    try:
        from PIL import Image, ImageFilter
    except ImportError as exc:
        raise RuntimeError("Pillow is required: pip install -r requirements.txt") from exc

    result = Image.fromarray(np.asarray(background, dtype=np.uint8))
    donor_image = Image.fromarray(np.asarray(donor_rgb, dtype=np.uint8)).resize(
        (placement.width, placement.height), Image.Resampling.LANCZOS
    )
    binary_mask = Image.fromarray((np.asarray(donor_mask) != 0).astype(np.uint8) * 255)
    mask_resample = Image.Resampling.NEAREST if blend_method == "none" else Image.Resampling.LANCZOS
    alpha = binary_mask.resize((placement.width, placement.height), mask_resample)
    if blend_method == "gaussian":
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=gaussian_radius))
    elif blend_method not in {"none", "alpha"}:
        raise ValueError(f"unsupported blend method: {blend_method}")
    if placement.flip_horizontal:
        donor_image = donor_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        alpha = alpha.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    result.paste(donor_image, (placement.x, placement.y), alpha)
    return np.asarray(result).copy()
