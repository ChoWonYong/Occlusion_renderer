from __future__ import annotations

from typing import Literal, Mapping

import numpy as np

from common.schema import bbox_to_mask
from depth.order import DepthOrder
from synth.trajectory import FramePlacement


def composite_frame(
    background: np.ndarray,
    background_tracks: list[Mapping[str, object]],
    occluder_rgba: np.ndarray,
    placement: FramePlacement,
    depth_order: DepthOrder,
    blend_method: Literal["none", "gaussian", "alpha"] = "alpha",
    gaussian_radius: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from PIL import Image, ImageFilter
    except ImportError as exc:
        raise RuntimeError("Pillow is required: pip install -r requirements.txt") from exc

    frame = np.asarray(background, dtype=np.uint8)
    image_height, image_width = frame.shape[:2]
    patch = np.asarray(occluder_rgba, dtype=np.uint8)
    if patch.ndim != 3 or patch.shape[2] != 4:
        raise ValueError("occluder patch must be RGBA")
    x, y, width, height = placement.xywh(patch.shape[1], patch.shape[0])
    if width < 2 or height < 2:
        return frame.copy(), np.zeros((image_height, image_width), dtype=np.uint8)
    rgb_image = Image.fromarray(patch[..., :3]).resize((width, height), Image.Resampling.LANCZOS)
    alpha_image = Image.fromarray(patch[..., 3]).resize(
        (width, height), Image.Resampling.NEAREST if blend_method == "none" else Image.Resampling.LANCZOS
    )
    if blend_method == "gaussian":
        alpha_image = alpha_image.filter(ImageFilter.GaussianBlur(gaussian_radius))
    elif blend_method not in {"none", "alpha"}:
        raise ValueError(f"unsupported blend method: {blend_method}")

    rgb = np.asarray(rgb_image).copy()
    alpha = np.asarray(alpha_image).copy()
    source_x1, source_y1 = max(0, -x), max(0, -y)
    source_x2, source_y2 = min(width, image_width - x), min(height, image_height - y)
    if source_x2 <= source_x1 or source_y2 <= source_y1:
        return frame.copy(), np.zeros((image_height, image_width), dtype=np.uint8)
    dest_x1, dest_y1 = max(0, x), max(0, y)
    dest_x2 = dest_x1 + source_x2 - source_x1
    dest_y2 = dest_y1 + source_y2 - source_y1
    rgb = rgb[source_y1:source_y2, source_x1:source_x2]
    alpha = alpha[source_y1:source_y2, source_x1:source_x2]

    full_alpha = np.zeros((image_height, image_width), dtype=np.uint8)
    full_alpha[dest_y1:dest_y2, dest_x1:dest_x2] = alpha
    # A closer real object cuts a hole in the synthetic occluder.
    for track in background_tracks:
        track_id = int(track["track_id"])
        if not depth_order.occluder_is_front(track_id):
            closer_mask = bbox_to_mask(track["bbox"], image_height, image_width)
            full_alpha[closer_mask != 0] = 0
    alpha = full_alpha[dest_y1:dest_y2, dest_x1:dest_x2]
    result = frame.copy()
    roi = result[dest_y1:dest_y2, dest_x1:dest_x2].astype(np.float32)
    weights = alpha.astype(np.float32)[..., None] / 255.0
    result[dest_y1:dest_y2, dest_x1:dest_x2] = np.clip(
        rgb.astype(np.float32) * weights + roi * (1.0 - weights), 0, 255
    ).astype(np.uint8)
    return result, (full_alpha > 0).astype(np.uint8)
