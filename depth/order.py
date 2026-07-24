from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class DepthOrder:
    occluder_depth: float
    track_depths: dict[int, float]

    def occluder_is_front(self, track_id: int) -> bool:
        return self.occluder_depth < self.track_depths[int(track_id)]


def front_only_order(track_ids: list[int]) -> DepthOrder:
    """Phase-1 fallback: injected occluder is in front of every background track."""
    return DepthOrder(occluder_depth=0.0, track_depths={int(track_id): 1.0 for track_id in track_ids})


def median_depth(mask: np.ndarray, depth_map: np.ndarray) -> float:
    values = np.asarray(depth_map)[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        raise ValueError("cannot estimate depth from an empty mask")
    return float(np.median(values))


def order_from_depth_map(
    occluder_mask: np.ndarray,
    track_masks: Mapping[int, np.ndarray],
    depth_map: np.ndarray,
) -> DepthOrder:
    return DepthOrder(
        occluder_depth=median_depth(occluder_mask, depth_map),
        track_depths={int(track_id): median_depth(mask, depth_map) for track_id, mask in track_masks.items()},
    )

