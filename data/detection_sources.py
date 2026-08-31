from __future__ import annotations

import configparser
from bisect import bisect_left
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from common.config import config_path, resolve_path
from common.io import load_json


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _image_paths(directory: Path) -> list[Path]:
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no source frames found: {directory}")
    return paths


def sample_frame_paths(
    frame_paths: Iterable[Path], source_fps: float, target_fps: float
) -> list[Path]:
    """Sample a frame sequence without consulting annotations.

    Sampling is anchored at the first available source frame and uses round-half-up,
    matching the existing MOT17 resampling convention. Upsampling is intentionally
    disabled: a source frame must never be duplicated into a detector tracklet.
    """
    paths = sorted(frame_paths, key=lambda path: int(path.stem))
    if not paths:
        return []
    if source_fps <= 0.0 or target_fps <= 0.0:
        raise ValueError("fps values must be positive")
    if target_fps >= source_fps:
        return paths

    by_index = {int(path.stem): path for path in paths}
    indices = sorted(by_index)
    first, last = indices[0], indices[-1]
    step = source_fps / target_fps
    target_count = int(math.floor((last - first) / step + 1e-9)) + 1
    sampled: list[Path] = []
    previous: int | None = None
    for offset in range(target_count):
        center = int(math.floor(first + offset * step + 0.5))
        # MOT17 image sequences are normally dense. Searching the nearest actual
        # path keeps this GT-free path robust to an accidentally missing image.
        position = bisect_left(indices, center)
        nearby = indices[max(0, position - 1) : min(len(indices), position + 1)]
        selected = min(nearby, key=lambda index: (abs(index - center), index))
        if previous is not None and selected <= previous:
            continue
        sampled.append(by_index[selected])
        previous = selected
    return sampled


def _frame_records(paths: list[Path]) -> list[dict[str, Any]]:
    from PIL import Image

    records: list[dict[str, Any]] = []
    for sample_index, image_path in enumerate(paths):
        with Image.open(image_path) as image:
            width, height = image.size
        records.append(
            {
                "sample_index": sample_index,
                "frame_index": int(image_path.stem),
                "source_image": str(image_path.resolve()),
                "source_image_size": [width, height],
            }
        )
    return records


def _validate_crop_sequences(split: Mapping[str, Any]) -> list[str]:
    train = {str(item) for item in split.get("train_sequences", [])}
    evaluation = {str(item) for item in split.get("eval_sequences", [])}
    allowed = {str(item) for item in split.get("crop_allowed_sequences", [])}
    if not allowed:
        raise ValueError("split has no crop_allowed_sequences")
    if not allowed <= train:
        raise ValueError(f"crop sequences outside train split: {sorted(allowed - train)}")
    if allowed & evaluation:
        raise ValueError(f"crop/eval leakage detected: {sorted(allowed & evaluation)}")
    return sorted(allowed)


def collect_kitti_sources(
    config: Mapping[str, Any], config_file: Path
) -> list[dict[str, Any]]:
    """Collect train-only KITTI images using only the fixed split, never GT labels."""
    split = load_json(resolve_path(config_file.parent, config["split"]["output"]))
    allowed = _validate_crop_sequences(split)
    root = config_path(config, config_file, "paths", "kitti_tracking")
    image_root = root / "training" / "image_02"
    sequences: list[dict[str, Any]] = []
    for sequence in allowed:
        frames = _frame_records(_image_paths(image_root / sequence))
        sequences.append(
            {
                "source_dataset": "KITTI",
                "sequence": sequence,
                "source_fps": 10.0,
                "target_fps": 10.0,
                "frames": frames,
            }
        )
    return sequences


def _mot_sequence_info(sequence_dir: Path) -> tuple[float, str]:
    parser = configparser.ConfigParser()
    if not parser.read(sequence_dir / "seqinfo.ini"):
        raise FileNotFoundError(f"MOT17 seqinfo.ini not found: {sequence_dir}")
    section = parser["Sequence"]
    source_fps = float(section.get("frameRate", 30.0))
    extension = str(section.get("imExt", ".jpg"))
    if source_fps <= 0.0:
        raise ValueError(f"invalid MOT17 frameRate {source_fps}: {sequence_dir}")
    return source_fps, extension


def collect_mot17_sources(
    config: Mapping[str, Any], config_file: Path
) -> list[dict[str, Any]]:
    """Collect one MOT17 detector view and resample images without reading GT."""
    settings = config["detector_tracklet_pool"]
    detector_view = str(settings.get("mot17_detector_view", "FRCNN")).upper()
    target_fps = float(settings.get("target_fps", 10.0))
    root = config_path(config, config_file, "paths", "mot17") / "train"
    sequence_dirs = sorted(root.glob(f"MOT17-*-{detector_view}"))
    if not sequence_dirs:
        raise FileNotFoundError(f"no MOT17 {detector_view} sequences found under {root}")

    sequences: list[dict[str, Any]] = []
    for sequence_dir in sequence_dirs:
        source_fps, extension = _mot_sequence_info(sequence_dir)
        paths = [
            path
            for path in _image_paths(sequence_dir / "img1")
            if path.suffix.lower() == extension.lower()
        ]
        sampled = sample_frame_paths(paths, source_fps, target_fps)
        sequences.append(
            {
                "source_dataset": "MOT17",
                "sequence": sequence_dir.name,
                "source_fps": source_fps,
                "target_fps": target_fps,
                "frames": _frame_records(sampled),
            }
        )
    return sequences


def collect_detection_sources(
    config: Mapping[str, Any], config_file: Path, dataset: str
) -> list[dict[str, Any]]:
    normalized = dataset.lower()
    if normalized == "kitti":
        return collect_kitti_sources(config, config_file)
    if normalized == "mot17":
        return collect_mot17_sources(config, config_file)
    raise ValueError(f"unsupported detector source dataset: {dataset}")
