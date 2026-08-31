from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from common.config import config_path, load_config, resolve_path
from common.gpu_budget import enforce_account_gpu_budget
from common.io import save_json
from data.detection_sources import collect_detection_sources


# MMDetection 2.x uses this canonical category order for COCO checkpoints.
COCO_CLASS_NAMES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)


class DetectorBackend(Protocol):
    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Return bbox_xyxy, score, and coco_class_name for one RGB image."""


def aspect_fallback_category(bbox_xyxy: list[float]) -> str:
    """Class fallback used only when a detector label is unavailable.

    Upright boxes are treated as people and wide boxes as vehicles. This corrects
    the accidentally reversed heuristic in the initial proposal.
    """
    x1, y1, x2, y2 = (float(value) for value in bbox_xyxy)
    return "person" if y2 - y1 > x2 - x1 else "car"


def map_detector_category(
    coco_class_name: str | None,
    bbox_xyxy: list[float],
    class_map: Mapping[str, str],
    *,
    use_aspect_fallback: bool,
    detector_confidence: float | None = None,
) -> str | None:
    """Use the detector label for class; confidence never triggers aspect relabeling.

    Co-DETR has already assigned each returned query its highest-confidence
    class. The aspect rule is deliberately restricted to the exceptional case
    where that label is absent, not merely low-scoring.
    """
    # detector_confidence is accepted to make the policy testable and explicit,
    # but intentionally has no bearing on the assigned class.
    _ = detector_confidence
    if coco_class_name is not None and coco_class_name in class_map:
        return str(class_map[coco_class_name])
    if coco_class_name is None and use_aspect_fallback:
        return aspect_fallback_category(bbox_xyxy)
    return None


class CoDetrBackend:
    """Thin MMDetection-2.x adapter kept inside the dedicated Co-DETR process."""

    def __init__(
        self,
        repository: Path,
        model_config: Path,
        checkpoint: Path,
        *,
        device: str,
    ) -> None:
        if not repository.is_dir():
            raise FileNotFoundError(
                f"Co-DETR repository not found: {repository}. See docs/CODETR_WORKFLOW.md"
            )
        if not model_config.is_file():
            raise FileNotFoundError(f"Co-DETR model config not found: {model_config}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Co-DETR checkpoint not found: {checkpoint}")
        sys.path.insert(0, str(repository))
        try:
            from mmdet.apis import inference_detector, init_detector
        except ImportError as exc:
            raise RuntimeError(
                "Co-DETR requires its dedicated MMDetection 2.25.3/MMCV 1.5.0 environment"
            ) from exc
        self._inference_detector = inference_detector
        self._model = init_detector(str(model_config), str(checkpoint), device=device)

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        result = self._inference_detector(self._model, image)
        bbox_result = result[0] if isinstance(result, tuple) else result
        if not isinstance(bbox_result, (list, tuple)):
            raise TypeError(f"unexpected Co-DETR result type: {type(bbox_result).__name__}")
        detections: list[dict[str, Any]] = []
        for class_index, boxes in enumerate(bbox_result):
            if class_index >= len(COCO_CLASS_NAMES):
                break
            for row in np.asarray(boxes).reshape(-1, 5):
                detections.append(
                    {
                        "bbox_xyxy": [float(value) for value in row[:4]],
                        "score": float(row[4]),
                        "coco_class_id": class_index,
                        "coco_class_name": COCO_CLASS_NAMES[class_index],
                    }
                )
        return detections


def _load_bgr(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
    # MMDetection 2.x ndarray inference follows OpenCV's BGR convention.
    return rgb[:, :, ::-1].copy()


def export_detections(
    config_file: str | Path,
    dataset: str,
    *,
    backend: DetectorBackend | None = None,
    max_sequences: int | None = None,
    max_frames_per_sequence: int | None = None,
) -> dict[str, Any]:
    """Run Co-DETR and export a versioned, environment-neutral JSON bridge."""
    config, path = load_config(config_file)
    settings = config["codetr_detector"]
    sources = collect_detection_sources(config, path, dataset)
    if max_sequences is not None:
        sources = sources[:max_sequences]
    planned_frames = sum(
        min(len(source["frames"]), max_frames_per_sequence)
        if max_frames_per_sequence is not None
        else len(source["frames"])
        for source in sources
    )
    print(
        {
            "stage": "source_plan",
            "dataset": dataset.upper() if dataset.lower() == "kitti" else "MOT17",
            "sequences": [source["sequence"] for source in sources],
            "frames": planned_frames,
            "gt_used": False,
        },
        file=sys.stderr,
        flush=True,
    )
    if backend is None:
        enforce_account_gpu_budget(config.get("resources", {}))
        backend = CoDetrBackend(
            config_path(config, path, "paths", "codetr_repo"),
            config_path(config, path, "paths", "codetr_config"),
            config_path(config, path, "paths", "codetr_checkpoint"),
            device=str(settings.get("device", "cuda:0")),
        )

    score_threshold = float(settings.get("score_threshold", 0.01))
    max_detections = int(settings.get("max_detections_per_frame", 300))
    class_map = {
        str(key): str(value)
        for key, value in settings.get(
            "coco_class_map",
            {"person": "person", "car": "car", "bus": "car", "truck": "car"},
        ).items()
    }
    use_aspect_fallback = bool(settings.get("aspect_ratio_fallback", True))
    kept_sequences: list[dict[str, Any]] = []
    total_detections = 0
    total_frames = 0
    for sequence in sources:
        frames = list(sequence["frames"])
        if max_frames_per_sequence is not None:
            frames = frames[:max_frames_per_sequence]
        exported_frames: list[dict[str, Any]] = []
        for frame in frames:
            raw = backend.detect(_load_bgr(Path(frame["source_image"])))
            selected: list[dict[str, Any]] = []
            for detection in raw:
                bbox = [float(value) for value in detection["bbox_xyxy"]]
                score = float(detection["score"])
                if score < score_threshold:
                    continue
                category = map_detector_category(
                    detection.get("coco_class_name"),
                    bbox,
                    class_map,
                    use_aspect_fallback=use_aspect_fallback,
                    detector_confidence=score,
                )
                if category is None:
                    continue
                x1, y1, x2, y2 = bbox
                if x2 - x1 <= 1.0 or y2 - y1 <= 1.0:
                    continue
                selected.append(
                    {
                        **dict(detection),
                        "bbox_xyxy": bbox,
                        "score": score,
                        "category": category,
                    }
                )
            selected.sort(key=lambda item: float(item["score"]), reverse=True)
            selected = selected[:max_detections]
            exported_frames.append({**dict(frame), "detections": selected})
            total_detections += len(selected)
            total_frames += 1
            if total_frames % 100 == 0 or total_frames == planned_frames:
                print(
                    {
                        "stage": "detect_progress",
                        "frames_done": total_frames,
                        "frames_total": planned_frames,
                    },
                    file=sys.stderr,
                    flush=True,
                )
        kept_sequences.append({**dict(sequence), "frames": exported_frames})

    output_dir = resolve_path(path.parent, settings["output_dir"])
    output_path = output_dir / f"{dataset.lower()}_detections.json"
    payload = {
        "schema_version": 1,
        "description": "GT-free Co-DETR detections for ByteTrack/SAM3 bridge",
        "detector": {
            "name": "Co-DINO ViT-Large COCO",
            "model_id": str(settings.get("model_id", "zongzhuofan/co-detr-vit-large-coco")),
            "score_threshold": score_threshold,
            "max_detections_per_frame": max_detections,
            "coco_class_map": class_map,
            "aspect_ratio_fallback": use_aspect_fallback,
            "class_assignment": {
                "policy": "highest-confidence Co-DETR label",
                "confidence_threshold_for_class": None,
                "confidence_role": "quality filtering only",
                "aspect_fallback": "missing detector label only",
            },
            "gt_used": False,
        },
        "source_dataset": dataset.upper() if dataset.lower() == "kitti" else "MOT17",
        "sequences": kept_sequences,
    }
    save_json(output_path, payload)
    summary = {
        "dataset": payload["source_dataset"],
        "sequences": len(kept_sequences),
        "frames": total_frames,
        "detections": total_detections,
        "output": str(output_path),
        "gt_used": False,
    }
    save_json(output_dir / f"{dataset.lower()}_summary.json", summary)
    return summary


def _relaunch(args: argparse.Namespace) -> int:
    config, path = load_config(args.config)
    python_path = config_path(config, path, "paths", "codetr_python")
    if not python_path.is_file():
        raise FileNotFoundError(
            f"Co-DETR Python not found: {python_path}. See docs/CODETR_WORKFLOW.md"
        )
    command = [
        str(python_path), "-m", "mining.codetr", "--config", str(path),
        "--dataset", args.dataset, "--codetr-child",
    ]
    if args.max_sequences is not None:
        command.extend(["--max-sequences", str(args.max_sequences)])
    if args.max_frames_per_sequence is not None:
        command.extend(["--max-frames-per-sequence", str(args.max_frames_per_sequence)])
    return subprocess.run(
        command, cwd=Path(__file__).resolve().parents[1], check=False
    ).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GT-free Co-DETR detections")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--dataset", required=True, choices=("kitti", "mot17"))
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--codetr-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.codetr_child:
        raise SystemExit(_relaunch(args))
    print(
        export_detections(
            args.config,
            args.dataset,
            max_sequences=args.max_sequences,
            max_frames_per_sequence=args.max_frames_per_sequence,
        )
    )


if __name__ == "__main__":
    main()
