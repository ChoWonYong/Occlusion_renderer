from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class TrackletLayer:
    track_id: int
    rgba: np.ndarray
    bbox_xywh: tuple[float, float, float, float]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedLayer:
    track_id: int
    amodal_mask: np.ndarray
    visible_mask: np.ndarray
    occluder_ids: tuple[int, ...]
    paste_order: int
    provenance: dict[str, Any]


@dataclass(frozen=True)
class _RasterizedLayer:
    layer: TrackletLayer
    rgb: np.ndarray
    alpha: np.ndarray
    bounds: tuple[int, int, int, int]
    amodal_mask: np.ndarray


def _rasterize(layer: TrackletLayer, image_shape: tuple[int, int]) -> _RasterizedLayer | None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required: pip install -r requirements.txt") from exc

    image_height, image_width = image_shape
    patch = np.asarray(layer.rgba, dtype=np.uint8)
    if patch.ndim != 3 or patch.shape[2] != 4:
        raise ValueError("tracklet patch must be RGBA")
    x, y, width, height = layer.bbox_xywh
    target_x = int(round(x))
    target_y = int(round(y))
    target_width = max(1, int(round(width)))
    target_height = max(1, int(round(height)))
    rgb = np.asarray(
        Image.fromarray(patch[..., :3]).resize((target_width, target_height), Image.Resampling.LANCZOS)
    ).copy()
    alpha = np.asarray(
        Image.fromarray(patch[..., 3]).resize((target_width, target_height), Image.Resampling.NEAREST)
    ).copy()
    source_x1, source_y1 = max(0, -target_x), max(0, -target_y)
    source_x2 = min(target_width, image_width - target_x)
    source_y2 = min(target_height, image_height - target_y)
    if source_x2 <= source_x1 or source_y2 <= source_y1:
        return None
    dest_x1, dest_y1 = max(0, target_x), max(0, target_y)
    dest_x2 = dest_x1 + source_x2 - source_x1
    dest_y2 = dest_y1 + source_y2 - source_y1
    rgb = rgb[source_y1:source_y2, source_x1:source_x2]
    alpha = alpha[source_y1:source_y2, source_x1:source_x2]
    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    full_mask[dest_y1:dest_y2, dest_x1:dest_x2] = alpha > 0
    if not full_mask.any():
        return None
    return _RasterizedLayer(
        layer=layer,
        rgb=rgb,
        alpha=alpha,
        bounds=(dest_x1, dest_y1, dest_x2, dest_y2),
        amodal_mask=full_mask,
    )


def composite_tracklet_layers(
    background: np.ndarray,
    layers: list[TrackletLayer],
    *,
    blend_method: Literal["none", "alpha"] = "none",
) -> tuple[np.ndarray, list[RenderedLayer]]:
    """Paste larger masks first and return layer-aware visible/amodal masks."""
    if blend_method not in {"none", "alpha"}:
        raise ValueError("tracklet blend_method must be 'none' or 'alpha'")
    result = np.asarray(background, dtype=np.uint8).copy()
    rasters = [item for layer in layers if (item := _rasterize(layer, result.shape[:2])) is not None]
    rasters.sort(key=lambda item: (-int(item.amodal_mask.sum()), item.layer.track_id))

    for item in rasters:
        x1, y1, x2, y2 = item.bounds
        roi = result[y1:y2, x1:x2]
        if blend_method == "none":
            select = item.alpha > 0
            roi[select] = item.rgb[select]
        else:
            weights = item.alpha.astype(np.float32)[..., None] / 255.0
            roi[:] = np.clip(item.rgb.astype(np.float32) * weights + roi * (1.0 - weights), 0, 255).astype(
                np.uint8
            )

    rendered: list[RenderedLayer] = []
    later_union = np.zeros(result.shape[:2], dtype=bool)
    later_items: list[_RasterizedLayer] = []
    reversed_results: list[RenderedLayer] = []
    for paste_order, item in reversed(list(enumerate(rasters))):
        amodal = item.amodal_mask.astype(bool)
        visible = np.logical_and(amodal, np.logical_not(later_union)).astype(np.uint8)
        covering_ids = tuple(
            later.layer.track_id
            for later in later_items
            if np.logical_and(amodal, later.amodal_mask != 0).any()
        )
        reversed_results.append(
            RenderedLayer(
                track_id=item.layer.track_id,
                amodal_mask=amodal.astype(np.uint8),
                visible_mask=visible,
                occluder_ids=covering_ids,
                paste_order=paste_order,
                provenance=item.layer.provenance,
            )
        )
        later_union |= amodal
        later_items.append(item)
    rendered.extend(reversed(reversed_results))
    return result, rendered
