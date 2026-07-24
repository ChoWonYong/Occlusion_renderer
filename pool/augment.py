from __future__ import annotations

import random

import numpy as np


def augment_identity(
    rgba: np.ndarray,
    rng: random.Random,
    horizontal_flip: bool = True,
    brightness: tuple[float, float] = (0.9, 1.1),
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Apply one augmentation once; reuse the returned patch for the whole sequence."""
    patch = np.asarray(rgba, dtype=np.uint8).copy()
    flipped = bool(horizontal_flip and rng.random() < 0.5)
    if flipped:
        patch = np.fliplr(patch).copy()
    factor = rng.uniform(*brightness)
    patch[..., :3] = np.clip(patch[..., :3].astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return patch, {"horizontal_flip": flipped, "brightness": factor}

