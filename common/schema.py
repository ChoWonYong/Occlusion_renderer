from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def encode_binary_mask(mask: np.ndarray) -> dict[str, Any]:
    """Encode a 2-D binary mask as an uncompressed COCO RLE."""
    binary = np.asarray(mask, dtype=np.uint8)
    if binary.ndim != 2:
        raise ValueError("mask must be a 2-D array")

    pixels = binary.reshape(-1, order="F")
    counts: list[int] = []
    previous = 0
    run_length = 0
    for pixel in pixels:
        value = int(pixel != 0)
        if value == previous:
            run_length += 1
        else:
            counts.append(run_length)
            run_length = 1
            previous = value
    counts.append(run_length)
    return {"size": [int(binary.shape[0]), int(binary.shape[1])], "counts": counts}


def decode_uncompressed_rle(rle: Mapping[str, Any]) -> np.ndarray:
    size = rle.get("size")
    counts = rle.get("counts")
    if not isinstance(size, Sequence) or len(size) != 2:
        raise ValueError("RLE size must be [height, width]")
    if not isinstance(counts, list):
        raise ValueError("only uncompressed RLE counts are supported")

    total = int(size[0]) * int(size[1])
    flat = np.zeros(total, dtype=np.uint8)
    cursor = 0
    value = 0
    for raw_count in counts:
        count = int(raw_count)
        if count < 0 or cursor + count > total:
            raise ValueError("invalid RLE run length")
        if value:
            flat[cursor : cursor + count] = 1
        cursor += count
        value = 1 - value
    if cursor != total:
        raise ValueError("RLE counts do not match mask size")
    return flat.reshape((int(size[0]), int(size[1])), order="F")


def bbox_to_mask(bbox: Sequence[float], height: int, width: int) -> np.ndarray:
    if len(bbox) != 4:
        raise ValueError("bbox must be [x, y, width, height]")
    x, y, box_width, box_height = (float(value) for value in bbox)
    x1 = max(0, min(width, int(np.floor(x))))
    y1 = max(0, min(height, int(np.floor(y))))
    x2 = max(0, min(width, int(np.ceil(x + box_width))))
    y2 = max(0, min(height, int(np.ceil(y + box_height))))
    mask = np.zeros((height, width), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


def mask_to_bbox(mask: np.ndarray) -> list[float]:
    binary = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def compute_ratio(visible_mask: np.ndarray, amodal_mask: np.ndarray) -> float:
    visible = np.asarray(visible_mask, dtype=bool)
    amodal = np.asarray(amodal_mask, dtype=bool)
    if visible.shape != amodal.shape:
        raise ValueError("visible and amodal masks must have the same shape")
    amodal_area = int(amodal.sum())
    if amodal_area == 0:
        raise ValueError("amodal mask must not be empty")
    visible_area = int(np.logical_and(visible, amodal).sum())
    return float(np.clip(1.0 - visible_area / amodal_area, 0.0, 1.0))


def occlusion_level(ratio: float) -> int:
    # 2026-07-23: bands aligned with the confirmed peak-rho distribution
    # (mild 0.20-0.35, moderate 0.35-0.65, heavy >=0.65). Level 0 is below the
    # mild floor. Note the "effective event" duration still uses a separate
    # rho >= 0.1 threshold, which is intentionally lower than the mild floor.
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("occlusion ratio must be in [0, 1]")
    if ratio < 0.20:
        return 0
    if ratio < 0.35:
        return 1
    if ratio < 0.65:
        return 2
    return 3


def validate_extended_annotation(annotation: Mapping[str, Any]) -> None:
    required = {
        "visible_bbox",
        "amodal_bbox",
        "amodal_segmentation",
        "occlusion_ratio",
        "occlusion_level",
        "occluder_ids",
        "synthetic",
        "provenance",
    }
    missing = required.difference(annotation)
    if missing:
        raise ValueError(f"extended annotation is missing: {sorted(missing)}")
    ratio = float(annotation["occlusion_ratio"])
    if int(annotation["occlusion_level"]) != occlusion_level(ratio):
        raise ValueError("occlusion_level is inconsistent with occlusion_ratio")
    policy = annotation.get("detector_bbox_policy", "visible")
    if policy == "amodal_original":
        if annotation.get("bbox") != annotation.get("amodal_bbox"):
            raise ValueError("COCO bbox must equal amodal_bbox under amodal_original policy")
    elif annotation.get("bbox") != annotation.get("visible_bbox"):
        raise ValueError("COCO bbox must equal visible_bbox")


@dataclass(frozen=True)
class Motion:
    model: str
    p0: tuple[float, float]
    v0: tuple[float, float]
    acceleration: tuple[float, float] = (0.0, 0.0)
    scale0: float = 1.0
    scale_rate: float = 0.0

    def position(self, time: float) -> tuple[float, float]:
        return (
            self.p0[0] + self.v0[0] * time + 0.5 * self.acceleration[0] * time * time,
            self.p0[1] + self.v0[1] * time + 0.5 * self.acceleration[1] * time * time,
        )

    def scale(self, time: float) -> float:
        return max(0.01, self.scale0 + self.scale_rate * time)


@dataclass(frozen=True)
class OccluderTrack:
    track_id: int
    video_id: int
    category: str
    source: dict[str, Any]
    motion: Motion
    frames: list[dict[str, Any]] = field(default_factory=list)
    depth_plane: float | None = None
    synthetic: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OcclusionEvent:
    event_id: int
    video_id: int
    victim_track: int
    occluder_track: int
    frame_start: int
    frame_peak: int
    frame_end: int
    peak_ratio: float
    duration: int
    entry_speed: float
    fully_occluded: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_video_dataset(dataset: Mapping[str, Any]) -> None:
    video_ids = {int(video["id"]) for video in dataset.get("videos", [])}
    image_ids: set[int] = set()
    frame_keys: set[tuple[int, int]] = set()
    for image in dataset.get("images", []):
        image_id = int(image["id"])
        video_id = int(image["video_id"])
        frame_index = int(image["frame_index"])
        if image_id in image_ids:
            raise ValueError(f"duplicate image id: {image_id}")
        if video_id not in video_ids:
            raise ValueError(f"image references unknown video: {video_id}")
        if (video_id, frame_index) in frame_keys:
            raise ValueError(f"duplicate frame index {frame_index} in video {video_id}")
        image_ids.add(image_id)
        frame_keys.add((video_id, frame_index))
    annotation_ids: set[int] = set()
    for annotation in dataset.get("annotations", []):
        annotation_id = int(annotation["id"])
        if annotation_id in annotation_ids:
            raise ValueError(f"duplicate annotation id: {annotation_id}")
        if int(annotation["image_id"]) not in image_ids:
            raise ValueError(f"annotation references unknown image: {annotation['image_id']}")
        if "video_id" not in annotation or "frame_index" not in annotation or "track_id" not in annotation:
            raise ValueError("video annotation requires video_id, frame_index, and track_id")
        annotation_ids.add(annotation_id)
