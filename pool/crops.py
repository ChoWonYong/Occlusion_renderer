"""Crop and instance-matching helpers shared by the SAM3 tracklet builders.

Both the production pool (`pool/build_detector_tracklet_pool.py`) and the crop
policy ablation (`eval/compare_sam3_crop_policies.py`) segment a detector box by
handing SAM3 a padded *context* crop and then keeping only the mask that falls
inside the detector box itself, so the two share this geometry verbatim.
"""

from __future__ import annotations

import numpy as np

from segment.base import Instance


def _clip_bbox(
    bbox: tuple[float, float, float, float], image_width: int, image_height: int
) -> tuple[int, int, int, int] | None:
    x, y, width, height = bbox
    x1 = max(0, min(image_width, int(np.floor(x))))
    y1 = max(0, min(image_height, int(np.floor(y))))
    x2 = max(0, min(image_width, int(np.ceil(x + width))))
    y2 = max(0, min(image_height, int(np.ceil(y + height))))
    return (x1, y1, x2, y2) if x2 - x1 > 1 and y2 - y1 > 1 else None


def crop_with_context(
    image: np.ndarray,
    bbox_xywh: tuple[float, float, float, float],
    padding_ratio: float,
) -> tuple[np.ndarray, np.ndarray, list[float], list[float], list[float]]:
    """Return source bbox crop, padded SAM3 input, and target box in context coordinates."""
    clipped = _clip_bbox(bbox_xywh, image.shape[1], image.shape[0])
    if clipped is None:
        raise ValueError("bbox is empty after clipping")
    x1, y1, x2, y2 = clipped
    pad_x = int(round((x2 - x1) * max(0.0, padding_ratio)))
    pad_y = int(round((y2 - y1) * max(0.0, padding_ratio)))
    context_x1 = max(0, x1 - pad_x)
    context_y1 = max(0, y1 - pad_y)
    context_x2 = min(image.shape[1], x2 + pad_x)
    context_y2 = min(image.shape[0], y2 + pad_y)
    source_crop = image[y1:y2, x1:x2]
    context_crop = image[context_y1:context_y2, context_x1:context_x2]
    target_in_context = [
        float(x1 - context_x1),
        float(y1 - context_y1),
        float(x2 - x1),
        float(y2 - y1),
    ]
    return (
        source_crop,
        context_crop,
        target_in_context,
        [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
        [
            float(context_x1),
            float(context_y1),
            float(context_x2 - context_x1),
            float(context_y2 - context_y1),
        ],
    )


def bbox_iou(first: list[float], second: list[float]) -> float:
    ax, ay, aw, ah = (float(value) for value in first)
    bx, by, bw, bh = (float(value) for value in second)
    intersection_width = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    union = max(0.0, aw * ah) + max(0.0, bw * bh) - intersection
    return intersection / union if union > 0 else 0.0


def select_matching_instance(
    instances: list[Instance], target_bbox: list[float], min_iou: float
) -> tuple[Instance, float] | None:
    if not instances:
        return None
    ranked = sorted(
        ((instance, bbox_iou(instance.bbox, target_bbox)) for instance in instances),
        key=lambda item: (item[1], item[0].score),
        reverse=True,
    )
    return ranked[0] if ranked[0][1] >= min_iou else None
