"""Geometry that turns a scheduled event into a concrete placement (Step 7).

Bridges the scheduler (which victim + which occluder tracklet) and the placement
primitives (target peak, acceptance) with the real box math: it maps the
occluder's source crop into the target image with victim-relative sizing, applies
one constant translation that aligns the occluder to the victim at the peak frame
(so the relative trajectory is preserved), and searches scale multipliers /
peak frames for the placement whose bbox-proxy rho forms an acceptable event.

The occluder box preserves the source object's own growth/shrink (relative motion)
and is anchored bottom-centre so it stays grounded.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from synth.placement import (
    accept_event,
    bbox_cover_ratio,
    candidate_peak_error,
    class_relative_target_height,
    rho_series,
)

BBox = Sequence[float]


def map_occluder_box(
    frame_record: Mapping[str, Any],
    target_size: tuple[int, int],
    *,
    reference_height: float,
    source_reference_height: float,
    translation: tuple[float, float] = (0.0, 0.0),
) -> tuple[float, float, float, float]:
    """Map one occluder source crop into the target image.

    ``reference_height`` is the desired occluder pixel height at the reference
    (peak) frame; ``source_reference_height`` is the source crop height there. The
    per-frame height scales with the source crop's own height (relative motion),
    and the box is anchored bottom-centre in normalised source coordinates, then
    shifted by ``translation``.
    """
    x, y, w, h = (float(value) for value in frame_record["crop_bbox_xywh"])
    source_width, source_height = (float(value) for value in frame_record["source_image_size"])
    target_width, target_height = target_size
    if source_reference_height <= 0 or h <= 0:
        raise ValueError("source heights must be positive")
    relative_motion = h / source_reference_height
    box_height = reference_height * relative_motion
    scale = box_height / h
    new_width, new_height = w * scale, h * scale
    center_x_fraction = (x + w / 2.0) / source_width
    bottom_fraction = (y + h) / source_height
    new_x = center_x_fraction * target_width - new_width / 2.0 + translation[0]
    new_y = bottom_fraction * target_height - new_height + translation[1]
    return (new_x, new_y, new_width, new_height)


def alignment_translation(occluder_box: BBox, victim_box: BBox) -> tuple[float, float]:
    """Constant shift that puts the occluder's bottom-centre on the victim's."""
    ox, oy, ow, oh = (float(value) for value in occluder_box)
    vx, vy, vw, vh = (float(value) for value in victim_box)
    return (vx + vw / 2.0 - (ox + ow / 2.0), vy + vh - (oy + oh))


def search_event_placement(
    occluder_frames: Sequence[Mapping[str, Any]],
    victim_by_position: Mapping[int, BBox],
    *,
    occluder_class: str,
    victim_class: str,
    class_height_factor: Mapping[str, float],
    target_peak: float,
    band: tuple[float, float],
    multipliers: Sequence[float],
    target_size: tuple[int, int],
    sequence_length: int,
    accept_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Search (peak frame x scale multiplier) for an acceptable target-rho event.

    ``occluder_frames`` are the first L frames actually pasted (exposure length L).
    ``victim_by_position`` maps output frame positions to the victim's bbox. The
    peak is aligned to the middle of the exposure; ``start = peak_frame - L//2``.
    Returns the best-scoring accepted placement (with its ``rho_series`` and
    params) or ``None``.
    """
    exposure = len(occluder_frames)
    if exposure == 0:
        return None
    peak_offset = exposure // 2
    source_reference_height = float(occluder_frames[peak_offset]["crop_bbox_xywh"][3])
    accept = dict(accept_kwargs or {})
    best: dict[str, Any] | None = None

    for peak_frame in sorted(victim_by_position):
        start = peak_frame - peak_offset
        if start < 0 or start + exposure > sequence_length:
            continue
        victim_peak = victim_by_position.get(peak_frame)
        if victim_peak is None:
            continue
        victim_height = float(victim_peak[3])
        base_height = class_relative_target_height(
            victim_height, occluder_class, victim_class, class_height_factor
        )
        for multiplier in multipliers:
            reference_height = base_height * float(multiplier)
            occluder_at_peak = map_occluder_box(
                occluder_frames[peak_offset],
                target_size,
                reference_height=reference_height,
                source_reference_height=source_reference_height,
            )
            translation = alignment_translation(occluder_at_peak, victim_peak)
            occluder_boxes = [
                map_occluder_box(
                    occluder_frames[offset],
                    target_size,
                    reference_height=reference_height,
                    source_reference_height=source_reference_height,
                    translation=translation,
                )
                for offset in range(exposure)
            ]
            victim_boxes = [victim_by_position.get(start + offset) for offset in range(exposure)]
            series = rho_series(occluder_boxes, victim_boxes)
            accepted, _ = accept_event(series, band, **accept)
            if not accepted:
                continue
            error = candidate_peak_error(series, target_peak)
            if best is None or error < best["peak_error"]:
                best = {
                    "start_position": start,
                    "peak_frame": peak_frame,
                    "scale_multiplier": float(multiplier),
                    "translation": translation,
                    "reference_height": reference_height,
                    "source_reference_height": source_reference_height,
                    "rho_series": series,
                    "achieved_peak": max(series),
                    "peak_error": error,
                }
    return best


def bbox_cover(occluder_box: BBox, victim_box: BBox) -> float:
    """Convenience re-export for callers wiring the render loop."""
    return bbox_cover_ratio(occluder_box, victim_box)
