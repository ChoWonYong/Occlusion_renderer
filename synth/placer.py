from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Placement:
    x: int
    y: int
    width: int
    height: int
    flip_horizontal: bool
    rho_target: float
    rho_actual: float
    score: float


def resize_mask_nearest(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    source = np.asarray(mask, dtype=np.uint8)
    if source.ndim != 2 or source.shape[0] == 0 or source.shape[1] == 0:
        raise ValueError("donor mask must be a non-empty 2-D array")
    if width < 1 or height < 1:
        raise ValueError("resized mask dimensions must be positive")
    y_indices = np.minimum((np.arange(height) * source.shape[0] / height).astype(int), source.shape[0] - 1)
    x_indices = np.minimum((np.arange(width) * source.shape[1] / width).astype(int), source.shape[1] - 1)
    return source[y_indices[:, None], x_indices[None, :]]


def _bbox_pixels(bbox: Sequence[float], image_width: int, image_height: int) -> tuple[int, int, int, int]:
    x, y, width, height = (float(value) for value in bbox)
    x1 = max(0, min(image_width, int(np.floor(x))))
    y1 = max(0, min(image_height, int(np.floor(y))))
    x2 = max(0, min(image_width, int(np.ceil(x + width))))
    y2 = max(0, min(image_height, int(np.ceil(y + height))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("target bbox is empty after clipping")
    return x1, y1, x2, y2


def _overlap_ratio(
    donor_mask: np.ndarray,
    x: int,
    y: int,
    target_xyxy: tuple[int, int, int, int],
) -> float:
    target_x1, target_y1, target_x2, target_y2 = target_xyxy
    donor_height, donor_width = donor_mask.shape
    overlap_x1, overlap_y1 = max(x, target_x1), max(y, target_y1)
    overlap_x2, overlap_y2 = min(x + donor_width, target_x2), min(y + donor_height, target_y2)
    target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
    if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
        return 0.0
    crop = donor_mask[
        overlap_y1 - y : overlap_y2 - y,
        overlap_x1 - x : overlap_x2 - x,
    ]
    return float(np.count_nonzero(crop) / target_area)


def find_placement(
    donor_mask: np.ndarray,
    target_bbox: Sequence[float],
    image_shape: tuple[int, int],
    rho_target: float,
    rng: np.random.Generator,
    scale_range: tuple[float, float] = (0.6, 1.4),
    trials: int = 160,
    tolerance: float = 0.06,
) -> Placement | None:
    """Search road-like, bottom-aligned placements using bbox area as amodal proxy."""
    if not 0.0 < rho_target < 1.0:
        raise ValueError("rho_target must be in (0, 1)")
    image_height, image_width = image_shape
    target_x1, target_y1, target_x2, target_y2 = _bbox_pixels(target_bbox, image_width, image_height)
    target_width, target_height = target_x2 - target_x1, target_y2 - target_y1
    donor_height, donor_width = donor_mask.shape
    if donor_height < 1 or donor_width < 1:
        return None

    min_scale, max_scale = scale_range
    candidates = np.linspace(min_scale, max_scale, num=max(8, min(32, trials // 4)))
    candidates = np.concatenate([candidates, rng.uniform(min_scale, max_scale, size=max(1, trials // 4))])
    best: Placement | None = None
    for height_ratio in candidates:
        resized_height = max(2, int(round(target_height * float(height_ratio))))
        scale = resized_height / donor_height
        resized_width = max(2, int(round(donor_width * scale)))
        if resized_height > image_height or resized_width > image_width:
            fit = min(image_height / resized_height, image_width / resized_width)
            resized_height = max(2, int(resized_height * fit))
            resized_width = max(2, int(resized_width * fit))
        resized_mask = resize_mask_nearest(donor_mask, resized_width, resized_height)

        positions_per_scale = max(4, trials // len(candidates))
        for _ in range(positions_per_scale):
            center_x = int(round(rng.uniform(target_x1, target_x2)))
            x = center_x - resized_width // 2
            bottom_jitter = int(round(rng.uniform(-0.12, 0.12) * target_height))
            y = target_y2 - resized_height + bottom_jitter
            x = max(0, min(image_width - resized_width, x))
            y = max(0, min(image_height - resized_height, y))
            ratio = _overlap_ratio(resized_mask, x, y, (target_x1, target_y1, target_x2, target_y2))
            score = abs(ratio - rho_target)
            placement = Placement(
                x=x,
                y=y,
                width=resized_width,
                height=resized_height,
                flip_horizontal=bool(rng.integers(0, 2)),
                rho_target=float(rho_target),
                rho_actual=ratio,
                score=score,
            )
            if best is None or placement.score < best.score:
                best = placement
    if best is None or best.score > tolerance:
        return None
    return best


def place_mask(mask: np.ndarray, placement: Placement, image_shape: tuple[int, int]) -> np.ndarray:
    resized = resize_mask_nearest(mask, placement.width, placement.height)
    if placement.flip_horizontal:
        resized = np.fliplr(resized)
    canvas = np.zeros(image_shape, dtype=np.uint8)
    canvas[
        placement.y : placement.y + placement.height,
        placement.x : placement.x + placement.width,
    ] = resized
    return canvas

