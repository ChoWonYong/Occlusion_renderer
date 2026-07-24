"""Target-rho placement and victim-relative scaling (Step 5).

Given a victim already chosen by the scheduler, we size the occluder relative to
that victim's observed size and search over translation/scale so the resulting
per-frame occlusion ratio (rho) forms a *complete* event: 0 -> rise -> single
peak in the target band -> fall -> 0, with the rho >= 0.1 stretch lasting
8-20 frames and peak <= 0.80.

Scaling is victim-relative (confirmed 2026-07-23): the occluder's target height
is the victim's height times the ratio of per-class canonical heights, so a
pasted object matches how big a real object of its class would look next to that
victim. Because the victim's observed size already encodes its depth, this needs
no separate depth proxy.

These are geometry-agnostic primitives. The pipeline maps boxes and builds the
per-frame rho series (bbox proxy for the victim); this module samples the target
peak, sizes the occluder, scores/accepts candidate placements, and picks the best.
The final rendered rho uses the real occluder mask, but the event *shape* is
dominated by geometry, so the bbox proxy is an adequate search signal.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Sequence

BBox = Sequence[float]

# Peak-rho difficulty bands (aligned with occlusion_level bands).
PEAK_BANDS: dict[str, tuple[float, float]] = {
    "mild": (0.20, 0.35),
    "moderate": (0.35, 0.65),
    "heavy": (0.65, 0.80),
}


def sample_target_peak(
    distribution: Mapping[str, float], rng: random.Random
) -> tuple[float, str, tuple[float, float]]:
    """Pick a difficulty band by weight, then a uniform peak within that band."""
    names = list(distribution)
    weights = [float(distribution[name]) for name in names]
    if sum(weights) <= 0:
        raise ValueError("peak_rho_distribution must have a positive total weight")
    band_name = rng.choices(names, weights=weights, k=1)[0]
    low, high = PEAK_BANDS[band_name]
    return rng.uniform(low, high), band_name, (low, high)


def class_relative_target_height(
    victim_height: float,
    occluder_class: str,
    victim_class: str,
    class_height_factor: Mapping[str, float],
) -> float:
    """occluder target height = victim_height * factor[occluder] / factor[victim]."""
    factor_occluder = float(class_height_factor[occluder_class])
    factor_victim = float(class_height_factor[victim_class])
    if factor_victim <= 0 or factor_occluder <= 0:
        raise ValueError("class_height_factor values must be positive")
    if victim_height <= 0:
        raise ValueError("victim_height must be positive")
    return victim_height * factor_occluder / factor_victim


def occluder_scale(target_height: float, source_reference_height: float) -> float:
    """Uniform scale that maps the source crop height to the target height."""
    if source_reference_height <= 0:
        raise ValueError("source_reference_height must be positive")
    return float(target_height) / float(source_reference_height)


def bbox_cover_ratio(occluder: BBox, victim: BBox) -> float:
    """Fraction of the victim bbox area covered by the occluder bbox (rho proxy)."""
    ox, oy, ow, oh = (float(value) for value in occluder)
    vx, vy, vw, vh = (float(value) for value in victim)
    intersection_width = max(0.0, min(ox + ow, vx + vw) - max(ox, vx))
    intersection_height = max(0.0, min(oy + oh, vy + vh) - max(oy, vy))
    victim_area = vw * vh
    return intersection_width * intersection_height / victim_area if victim_area > 0 else 0.0


def rho_series(
    occluder_boxes: Sequence[BBox | None], victim_boxes: Sequence[BBox | None]
) -> list[float]:
    """Per-frame bbox-proxy rho; 0 where either box is absent."""
    series: list[float] = []
    for occluder, victim in zip(occluder_boxes, victim_boxes):
        if occluder is None or victim is None:
            series.append(0.0)
        else:
            series.append(bbox_cover_ratio(occluder, victim))
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
    band: tuple[float, float],
    *,
    effective_range: tuple[int, int] = (8, 20),
    peak_max: float = 0.80,
    end_max: float = 0.05,
    floor: float = 0.1,
) -> tuple[bool, str]:
    """Accept a complete, single-peaked event whose peak lands in ``band``."""
    metrics = event_shape(series, floor)
    low, high = band
    peak = metrics["peak"]
    if peak > peak_max:
        return False, f"peak {peak:.3f} > peak_max {peak_max}"
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


def candidate_peak_error(series: Sequence[float], target_peak: float) -> float:
    peak = max(series) if series else 0.0
    return abs(peak - float(target_peak))


def select_best_placement(
    candidates: Sequence[Mapping[str, Any]],
    target_peak: float,
    band: tuple[float, float],
    **accept_kwargs: Any,
) -> dict[str, Any] | None:
    """Return the accepted candidate whose peak is closest to ``target_peak``.

    Each candidate is a mapping with a ``rho_series`` and arbitrary ``params``
    (start position, translation, scale multiplier). Returns ``None`` if no
    candidate forms an acceptable event, and annotates the winner with its
    achieved peak and peak error.
    """
    best: tuple[float, dict[str, Any]] | None = None
    for candidate in candidates:
        series = candidate["rho_series"]
        accepted, _ = accept_event(series, band, **accept_kwargs)
        if not accepted:
            continue
        error = candidate_peak_error(series, target_peak)
        if best is None or error < best[0]:
            enriched = dict(candidate)
            enriched["achieved_peak"] = max(series) if series else 0.0
            enriched["peak_error"] = error
            best = (error, enriched)
    return best[1] if best is not None else None
