"""Generative occlusion-event placement (Step 5).

Confirmed 2026-07-27 policy. Difficulty is a *derived* quantity, not a target:
physical parameters are drawn from priors and the occlusion bands act only as an
acceptance gate.

- Occluder height factor ~ ``U(class_height_range[class])`` — the real
  intra-class size spread (a hatchback vs a van, a 1.55 m vs a 1.90 m adult), so
  a pasted object is always at a physically plausible size. The previous design
  searched a ``[0.80 .. 1.30]`` multiplier to hit a target rho, which rendered
  69% of occluders at an implausible size (61% of cars at 1.20 m, 36% of people
  at 2.21 m) because the multiplier was silently absorbing the proxy/mask gap
  described below.
- Lateral offset ~ ``U(-a, a) x (w_occluder + w_victim)/2`` — that span is the
  centre separation at which two boxes just stop overlapping, so one prior
  sweeps every class pair from full cover to no cover. Scaling the offset by the
  victim width alone truncates the range for wide occluders (a 150 px car never
  clears a 40 px pedestrian), which is what previously made partial car-on-
  pedestrian occlusion unreachable.
- Rho is measured on the *rendered* occluder mask through a per-frame integral
  image. The victim's amodal mask is a filled rectangle (``bbox_to_mask``), so a
  rectangle query is exact and costs the same O(1) as the old bounding-box
  proxy. Measured against that proxy, the mask-based rho is only 0.40-0.72x the
  proxy value (KITTI car 0.72, MOT17 person 0.59, KITTI person 0.40): the proxy
  over-stated occlusion, by a factor that itself varied with occluder class,
  which made the configured band distribution unachievable in rendered terms.

Class-pair asymmetries in the resulting difficulty (a pedestrian cannot heavily
occlude a car at the same depth) are physical and are deliberately kept.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

import numpy as np

BBox = Sequence[float]

# Occlusion-strength bands. These classify an achieved peak; they are no longer
# sampled as targets. The band table spans up to 1.0, but how far events are
# actually allowed to go is set by ``peak_rho_max`` in config — 0.90, because the
# synthetic data trains the detector rather than the tracker and a frame with no
# visible pixels is an unlearnable target, not a hard one.
PEAK_BANDS: dict[str, tuple[float, float]] = {
    "mild": (0.20, 0.35),
    "moderate": (0.35, 0.65),
    "heavy": (0.65, 1.00),
}

# An event must reach at least the mild floor to be worth rendering.
GATE_FLOOR = 0.20


def classify_band(peak: float) -> str | None:
    """Name the band a peak rho falls in, or ``None`` below the gate floor."""
    for name, (low, high) in PEAK_BANDS.items():
        if low <= peak <= high:
            return name
    return None


def sample_height_factor(
    class_height_range: Mapping[str, Sequence[float]], category: str, rng: random.Random
) -> float:
    """Draw a physically plausible canonical height factor for ``category``."""
    low, high = (float(value) for value in class_height_range[category])
    if low <= 0 or high < low:
        raise ValueError(f"invalid class_height_range for {category!r}: {(low, high)}")
    return rng.uniform(low, high)


def mid_height_factor(
    class_height_range: Mapping[str, Sequence[float]], category: str
) -> float:
    """Range midpoint, used for the victim (its observed size is the depth cue)."""
    low, high = (float(value) for value in class_height_range[category])
    return (low + high) / 2.0


def class_relative_target_height(
    victim_height: float,
    factor_occluder: float,
    factor_victim: float,
) -> float:
    """Occluder pixel height of an object standing at the victim's distance.

    Under perspective projection two objects at the same depth have image heights
    in the ratio of their real heights, so the victim's observed height carries
    all the depth information needed.
    """
    if factor_victim <= 0 or factor_occluder <= 0:
        raise ValueError("class height factors must be positive")
    if victim_height <= 0:
        raise ValueError("victim_height must be positive")
    return victim_height * factor_occluder / factor_victim


def occluder_scale(target_height: float, source_reference_height: float) -> float:
    """Uniform scale that maps the source crop height to the target height."""
    if source_reference_height <= 0:
        raise ValueError("source_reference_height must be positive")
    return float(target_height) / float(source_reference_height)


class AlphaIntegral:
    """Summed-area table over one occluder crop's alpha mask.

    Lets :func:`mask_cover_ratio` answer "how much of this axis-aligned rectangle
    does the occluder's real mask cover" in four lookups, at any paste scale.
    """

    __slots__ = ("table", "height", "width")

    def __init__(self, alpha: np.ndarray) -> None:
        binary = (np.asarray(alpha) > 0).astype(np.int64)
        if binary.ndim != 2:
            raise ValueError("alpha must be a 2-D mask")
        self.height, self.width = binary.shape
        self.table = np.zeros((self.height + 1, self.width + 1), dtype=np.int64)
        self.table[1:, 1:] = binary.cumsum(axis=0).cumsum(axis=1)

    @property
    def fill_ratio(self) -> float:
        area = self.height * self.width
        return float(self.table[-1, -1]) / area if area else 0.0

    def count(self, x1: float, y1: float, x2: float, y2: float) -> int:
        """Set pixels inside the source-crop rectangle ``[x1, x2) x [y1, y2)``."""
        left = min(max(int(round(x1)), 0), self.width)
        right = min(max(int(round(x2)), 0), self.width)
        top = min(max(int(round(y1)), 0), self.height)
        bottom = min(max(int(round(y2)), 0), self.height)
        if right <= left or bottom <= top:
            return 0
        table = self.table
        return int(
            table[bottom, right] - table[top, right] - table[bottom, left] + table[top, left]
        )


def mask_cover_ratio(integral: AlphaIntegral, occluder: BBox, victim: BBox) -> float:
    """Fraction of the victim rectangle covered by the occluder's real mask."""
    ox, oy, ow, oh = (float(value) for value in occluder)
    vx, vy, vw, vh = (float(value) for value in victim)
    if ow <= 0 or oh <= 0 or vw <= 0 or vh <= 0:
        return 0.0
    # Map the victim rectangle into the occluder crop's own pixel grid.
    scale_x = integral.width / ow
    scale_y = integral.height / oh
    covered = integral.count(
        (vx - ox) * scale_x,
        (vy - oy) * scale_y,
        (vx + vw - ox) * scale_x,
        (vy + vh - oy) * scale_y,
    )
    if covered == 0:
        return 0.0
    # One source pixel paints this much target area once the crop is scaled.
    pixel_area = (ow / integral.width) * (oh / integral.height)
    return min(1.0, covered * pixel_area / (vw * vh))


def bbox_cover_ratio(occluder: BBox, victim: BBox) -> float:
    """Bounding-box overlap fraction. Kept for cheap pre-filters and tests."""
    ox, oy, ow, oh = (float(value) for value in occluder)
    vx, vy, vw, vh = (float(value) for value in victim)
    intersection_width = max(0.0, min(ox + ow, vx + vw) - max(ox, vx))
    intersection_height = max(0.0, min(oy + oh, vy + vh) - max(oy, vy))
    victim_area = vw * vh
    return intersection_width * intersection_height / victim_area if victim_area > 0 else 0.0


def rho_series(
    occluder_boxes: Sequence[BBox | None],
    victim_boxes: Sequence[BBox | None],
    integrals: Sequence[AlphaIntegral | None] | None = None,
) -> list[float]:
    """Per-frame rho; 0 where either box is absent.

    With ``integrals`` the ratio uses the occluder's real mask — what the
    renderer will actually produce; without them it falls back to the bounding
    box.
    """
    series: list[float] = []
    for index, (occluder, victim) in enumerate(zip(occluder_boxes, victim_boxes)):
        if occluder is None or victim is None:
            series.append(0.0)
            continue
        integral = integrals[index] if integrals is not None else None
        if integral is None:
            series.append(bbox_cover_ratio(occluder, victim))
        else:
            series.append(mask_cover_ratio(integral, occluder, victim))
    return series


def event_shape(series: Sequence[float], floor: float = 0.1) -> dict[str, Any]:
    """Summarise a rho series: peak, longest >=floor run, number of runs, ends."""
    if not series:
        return {"peak": 0.0, "peak_index": -1, "effective_len": 0, "num_runs": 0, "start": 0.0, "end": 0.0}
    peak = max(series)
    runs: list[int] = []
    current = 0
    for value in series:
        if value >= floor:
            current += 1
        elif current > 0:
            runs.append(current)
            current = 0
    if current > 0:
        runs.append(current)
    return {
        "peak": float(peak),
        "peak_index": int(series.index(peak)),
        "effective_len": max(runs) if runs else 0,
        "num_runs": len(runs),
        "start": float(series[0]),
        "end": float(series[-1]),
    }


def accept_event(
    series: Sequence[float],
    band: tuple[float, float] | None = None,
    *,
    effective_range: tuple[int, int] = (8, 20),
    peak_max: float = 1.00,
    end_max: float = 0.05,
    floor: float = 0.1,
    gate_floor: float = GATE_FLOOR,
) -> tuple[bool, str]:
    """Gate a complete, single-peaked event.

    ``band`` restricts the peak to one band; leaving it ``None`` is the
    generative mode, where any peak from ``gate_floor`` to ``peak_max`` passes.
    """
    metrics = event_shape(series, floor)
    peak = metrics["peak"]
    if peak > peak_max:
        return False, f"peak {peak:.3f} > peak_max {peak_max}"
    if band is None:
        if peak < gate_floor:
            return False, f"peak {peak:.3f} < gate_floor {gate_floor}"
    else:
        low, high = band
        if not (low <= peak <= high):
            return False, f"peak {peak:.3f} outside band [{low}, {high}]"
    if metrics["num_runs"] != 1:
        return False, f"num_runs {metrics['num_runs']} != 1 (not a single event)"
    effective_low, effective_high = effective_range
    if not (effective_low <= metrics["effective_len"] <= effective_high):
        return False, f"effective_len {metrics['effective_len']} outside {effective_range}"
    if metrics["start"] > end_max:
        return False, f"start rho {metrics['start']:.3f} > end_max {end_max}"
    if metrics["end"] > end_max:
        return False, f"end rho {metrics['end']:.3f} > end_max {end_max}"
    return True, "ok"


def offset_span(occluder_width: float, victim_width: float) -> float:
    """Centre separation at which the two boxes stop overlapping.

    Normalising the lateral offset by this makes ``0`` mean "fully aligned" and
    ``1`` mean "just clear" for every class pair, however wide the occluder is.
    """
    return (float(occluder_width) + float(victim_width)) / 2.0


def sample_lateral_offset(max_fraction: float, rng: random.Random) -> float:
    """Draw a lateral offset in units of :func:`offset_span`."""
    return rng.uniform(-float(max_fraction), float(max_fraction))
