from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Instance:
    bbox: list[float]
    mask: np.ndarray
    category: str
    score: float
    track_id: int | None = None


class SegBackend(ABC):
    @abstractmethod
    def detect_and_mask(self, image: np.ndarray, prompts: list[str]) -> list[Instance]:
        """Return visible instance masks in input-image coordinates."""


class ExternalSegBackend(SegBackend):
    """Contract marker for SAM3/GDINO-SAM2/YOLO-World adapters selected later."""

    def detect_and_mask(self, image: np.ndarray, prompts: list[str]) -> list[Instance]:
        raise RuntimeError(
            "No segmentation backend is configured. Phase-0/1 can use COCO GT polygons; "
            "configure a SAM3 adapter before extracting KITTI crops."
        )

