"""Geometry that turns a scheduled event into a concrete placement (Step 7).

Bridges the scheduler (which victim + which occluder tracklet) and the placement
primitives with the real box math. The occluder crop is mapped into the target
image at the size a real object of its class would have at the victim's distance,
then shifted by one constant translation: bottom-centre aligned to the victim at
the peak frame, plus a sampled lateral offset. The offset is what sets the
occlusion strength — sliding sideways keeps the occluder at the victim's depth,
so unlike rescaling it costs nothing in physical plausibility.

The occluder box preserves the source object's own growth/shrink (relative
motion) and is anchored bottom-centre so it stays grounded.

Rho is evaluated on the occluder's real mask through a cached integral image per
source crop, so the value the search accepts is the value the renderer produces.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from synth.paste_jitter import IDENTITY, FrameJitter, jitter_alpha, jitter_box
from synth.placement import (
    AlphaIntegral,
    accept_event,
    bbox_cover_ratio,
    class_relative_target_height,
    classify_band,
    offset_span,
    rho_series,
    sample_lateral_offset,
)

BBox = Sequence[float]

# Alpha masks larger than this longest side are downsampled before the
# summed-area table is built. Measured against full-resolution tables on the
# KITTI pool under realistic occluder/victim geometry: mean rho error 0.0003,
# p99 0.006 — versus mean 0.187 for the bounding-box proxy this replaces. Most
# crops are well under the cap (median 52x21), so the memory cost is minor.
INTEGRAL_MAX_SIDE = 128


class AlphaIntegralCache:
    """Lazily-built, path-keyed integral images for occluder crops.

    The tables do not depend on where a crop is pasted, so one pass over the pool
    serves every placement attempt and every retry.

    Per-frame paste jitter breaks that sharing — a flipped or rotated mask is a
    different mask — so a jittered request builds its table on the spot. The
    decoded full-resolution alpha is still cached by path, which is where the cost
    actually sits, so the extra work is the rotate plus the summed-area table.
    """

    def __init__(self, max_side: int = INTEGRAL_MAX_SIDE) -> None:
        self._cache: dict[str, AlphaIntegral] = {}
        self._raw: dict[str, np.ndarray] = {}
        self._max_side = int(max_side)

    def __len__(self) -> int:
        return len(self._cache)

    def get(self, rgba_path: str | Path) -> AlphaIntegral:
        key = str(rgba_path)
        cached = self._cache.get(key)
        if cached is None:
            cached = AlphaIntegral(self._downsample(self._raw_alpha(key)))
            self._cache[key] = cached
        return cached

    def get_jittered(self, rgba_path: str | Path, jitter: FrameJitter) -> AlphaIntegral:
        if not (jitter.flip or jitter.rotation % 360 != 0):
            # Scale rides on the box, and the photometric factors never touch
            # alpha, so this crop's table is the shared unjittered one.
            return self.get(rgba_path)
        alpha = jitter_alpha(self._raw_alpha(str(rgba_path)), jitter)
        return AlphaIntegral(self._downsample(alpha))

    def for_frames(
        self,
        frames: Sequence[Mapping[str, Any]],
        jitters: Sequence[FrameJitter] | None = None,
    ) -> list[AlphaIntegral]:
        if jitters is None:
            return [self.get(frame["rgba_path"]) for frame in frames]
        return [
            self.get_jittered(frame["rgba_path"], jitter)
            for frame, jitter in zip(frames, jitters)
        ]

    def _raw_alpha(self, path: str) -> np.ndarray:
        cached = self._raw.get(path)
        if cached is None:
            from PIL import Image

            with Image.open(path) as image:
                cached = np.asarray(image.convert("RGBA").getchannel("A")).copy()
            self._raw[path] = cached
        return cached

    def _downsample(self, alpha: np.ndarray) -> np.ndarray:
        height, width = alpha.shape[:2]
        longest = max(width, height)
        if longest <= self._max_side:
            return alpha
        from PIL import Image

        scale = self._max_side / longest
        resized = Image.fromarray(np.asarray(alpha, dtype=np.uint8)).resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            Image.Resampling.NEAREST,
        )
        return np.asarray(resized)


def map_occluder_box(
    frame_record: Mapping[str, Any],
    target_size: tuple[int, int],
    *,
    reference_height: float,
    source_reference_height: float,
    translation: tuple[float, float] = (0.0, 0.0),
    jitter: FrameJitter = IDENTITY,
) -> tuple[float, float, float, float]:
    """Map one occluder source crop into the target image.

    ``reference_height`` is the desired occluder pixel height at the reference
    (peak) frame; ``source_reference_height`` is the source crop height there. The
    per-frame height scales with the source crop's own height (relative motion),
    and the box is anchored bottom-centre in normalised source coordinates, then
    shifted by ``translation``.

    ``jitter`` grows the box by that frame's rotation expansion and scale, still
    bottom-centre anchored. Callers must pass the same jitter here as they hand
    the renderer, or the search would score a box the renderer never draws.
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
    box = (new_x, new_y, new_width, new_height)
    if jitter is IDENTITY or not jitter.changes_geometry:
        return box
    return jitter_box(box, (w, h), jitter)


def alignment_translation(occluder_box: BBox, victim_box: BBox) -> tuple[float, float]:
    """Constant shift that puts the occluder's bottom-centre on the victim's."""
    ox, oy, ow, oh = (float(value) for value in occluder_box)
    vx, vy, vw, vh = (float(value) for value in victim_box)
    return (vx + vw / 2.0 - (ox + ow / 2.0), vy + vh - (oy + oh))


def sample_event_placement(
    occluder_frames: Sequence[Mapping[str, Any]],
    victim_by_position: Mapping[int, BBox],
    *,
    factor_occluder: float,
    factor_victim: float,
    integrals: Sequence[AlphaIntegral],
    target_size: tuple[int, int],
    sequence_length: int,
    max_lateral_fraction: float,
    rng: random.Random,
    attempts: int = 24,
    accept_kwargs: Mapping[str, Any] | None = None,
    scene_check: Any = None,
    jitters: Sequence[FrameJitter] | None = None,
) -> dict[str, Any] | None:
    """Draw physical parameters until one yields an acceptable occlusion event.

    ``occluder_frames`` are the first L frames actually pasted (exposure length L)
    and ``integrals`` are their alpha summed-area tables. ``victim_by_position``
    maps output frame positions to the victim's bbox. The peak is aligned to the
    middle of the exposure, so ``start = peak_frame - L//2``.

    Each attempt samples a peak frame and a lateral offset (in units of the span
    at which the two boxes stop overlapping) and keeps the first placement whose
    rho series forms a complete single-peaked event.

    ``scene_check(start, boxes, integrals) -> bool`` optionally vets the candidate
    against everything *else* in the frame. The rho gate above only ever looks at
    the occluder's own victim, so without this a placement can pass while burying
    an unrelated bystander — measured at 118 of 174 over-cap frames in a full run,
    none of them the occluder's intended victim.

    ``jitters`` is the per-frame paste jitter, one entry per exposure frame, and
    must be the sequence ``integrals`` were built from — the boxes below carry its
    rotation expansion and scale, so search and render agree frame by frame.

    Returns ``None`` if no attempt passes.
    """
    exposure = len(occluder_frames)
    if exposure == 0:
        return None
    if jitters is None:
        jitters = [IDENTITY] * exposure
    elif len(jitters) != exposure:
        raise ValueError(f"jitters has {len(jitters)} entries for {exposure} exposure frames")
    peak_offset = exposure // 2
    source_reference_height = float(occluder_frames[peak_offset]["crop_bbox_xywh"][3])
    accept = dict(accept_kwargs or {})

    candidate_peaks = [
        position
        for position in sorted(victim_by_position)
        if position - peak_offset >= 0 and position - peak_offset + exposure <= sequence_length
    ]
    if not candidate_peaks:
        return None

    for _ in range(int(attempts)):
        peak_frame = rng.choice(candidate_peaks)
        start = peak_frame - peak_offset
        victim_peak = victim_by_position[peak_frame]
        reference_height = class_relative_target_height(
            float(victim_peak[3]), factor_occluder, factor_victim
        )
        unshifted = map_occluder_box(
            occluder_frames[peak_offset],
            target_size,
            reference_height=reference_height,
            source_reference_height=source_reference_height,
            jitter=jitters[peak_offset],
        )
        base_translation = alignment_translation(unshifted, victim_peak)
        span = offset_span(unshifted[2], float(victim_peak[2]))
        lateral = sample_lateral_offset(max_lateral_fraction, rng)
        translation = (base_translation[0] + lateral * span, base_translation[1])

        occluder_boxes = [
            map_occluder_box(
                occluder_frames[offset],
                target_size,
                reference_height=reference_height,
                source_reference_height=source_reference_height,
                translation=translation,
                jitter=jitters[offset],
            )
            for offset in range(exposure)
        ]
        victim_boxes = [victim_by_position.get(start + offset) for offset in range(exposure)]
        series = rho_series(occluder_boxes, victim_boxes, integrals)
        accepted, _ = accept_event(series, None, **accept)
        if not accepted:
            continue
        if scene_check is not None and not scene_check(start, occluder_boxes, integrals):
            continue
        peak = max(series)
        return {
            "start_position": start,
            "peak_frame": peak_frame,
            "occluder_boxes": occluder_boxes,
            "translation": translation,
            "lateral_offset_fraction": lateral,
            "offset_span": span,
            "reference_height": reference_height,
            "source_reference_height": source_reference_height,
            "height_factor": float(factor_occluder),
            "rho_series": series,
            "achieved_peak": peak,
            "band": classify_band(peak),
        }
    return None


def bbox_cover(occluder_box: BBox, victim_box: BBox) -> float:
    """Convenience re-export for callers wiring the render loop."""
    return bbox_cover_ratio(occluder_box, victim_box)
