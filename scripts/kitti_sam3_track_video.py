from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from common.config import config_path, load_config, resolve_path
from common.io import load_json, save_json
from pool.build_kitti_sam3_pool import segment_candidate, validate_crop_split


def select_consecutive_window(
    candidates: list[dict[str, Any]],
    *,
    sequence: str,
    track_id: int,
    length: int,
    start_frame: int | None = None,
) -> list[dict[str, Any]]:
    by_frame = {
        int(item["frame_index"]): item
        for item in candidates
        if str(item["sequence"]) == sequence and int(item["track_id"]) == track_id
    }
    if start_frame is not None:
        frames = list(range(start_frame, start_frame + length))
        missing = [frame for frame in frames if frame not in by_frame]
        if missing:
            raise ValueError(
                f"KITTI track {sequence}/{track_id} is missing requested frames: {missing}"
            )
        return [by_frame[frame] for frame in frames]
    sorted_frames = sorted(by_frame)
    for first in sorted_frames:
        frames = list(range(first, first + length))
        if all(frame in by_frame for frame in frames):
            return [by_frame[frame] for frame in frames]
    raise ValueError(f"no {length}-frame clean run for KITTI track {sequence}/{track_id}")


def _checkerboard(height: int, width: int, tile: int = 12) -> np.ndarray:
    yy, xx = np.indices((height, width))
    cells = (xx // tile + yy // tile) % 2
    return np.repeat(np.where(cells[..., None] == 0, 225, 180), 3, axis=2).astype(np.uint8)


def _fit_panel(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    panel = np.full((height, width, 3), 235, dtype=np.uint8)
    x = (width - resized_width) // 2
    y = (height - resized_height) // 2
    panel[y : y + resized_height, x : x + resized_width] = resized
    return panel


def _comparison_frame(
    candidate: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    panel_width: int = 480,
    panel_height: int = 280,
) -> np.ndarray:
    crop = np.asarray(result["crop"], dtype=np.uint8)
    binary = np.asarray(result["alpha"]) != 0
    overlay = crop.astype(np.float32)
    overlay[binary] = overlay[binary] * 0.45 + np.asarray([255, 40, 40]) * 0.55
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    checker = _checkerboard(crop.shape[0], crop.shape[1])
    checker[binary] = crop[binary]
    panels = [_fit_panel(item, panel_width, panel_height) for item in (crop, overlay, checker)]
    canvas = cv2.hconcat([cv2.cvtColor(panel, cv2.COLOR_RGB2BGR) for panel in panels])
    canvas = cv2.copyMakeBorder(canvas, 38, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
    labels = [
        f"KITTI crop | frame {int(candidate['frame_index']):06d}",
        f"SAM3 '{candidate['sam3_prompt']}' | IoU {float(result['match_iou']):.3f}",
        "RGBA on checker",
    ]
    for index, label in enumerate(labels):
        cv2.putText(
            canvas,
            label,
            (index * panel_width + 10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
    return canvas


def run(
    config_file: str | Path,
    *,
    sequence: str,
    track_id: int,
    start_frame: int | None = None,
    length: int = 30,
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    from PIL import Image
    from segment.sam3 import Sam3Backend

    config, path = load_config(config_file)
    settings = config["kitti_sam3_pool"]
    pool_dir = resolve_path(path.parent, settings["output_dir"])
    metadata = load_json(pool_dir / "candidates.json")
    allowed = validate_crop_split(metadata["split_policy"])
    if sequence not in allowed:
        raise ValueError(f"sequence {sequence} is not train/crop allowed")
    window = select_consecutive_window(
        list(metadata["candidates"]),
        sequence=sequence,
        track_id=track_id,
        length=length,
        start_frame=start_frame,
    )
    prompt = str(window[0]["sam3_prompt"])
    if any(str(item["sam3_prompt"]) != prompt for item in window):
        raise ValueError("one KITTI track window must use one SAM3 prompt")

    segmenter = Sam3Backend(
        config_path(config, path, "paths", "sam3_checkpoint"),
        bpe_path=config_path(config, path, "paths", "sam3_bpe"),
        device=str(settings.get("device", "cuda")),
        confidence_threshold=float(settings.get("confidence_threshold", 0.5)),
        precision=str(settings.get("precision", "auto")),
    )
    first_frame = int(window[0]["frame_index"])
    last_frame = int(window[-1]["frame_index"])
    name = f"kitti_{sequence}_track{track_id}_frames{first_frame}_{last_frame}"
    artifact_dir = resolve_path(path.parent, "../artifacts/phase1/kitti_sam3_track_videos") / name
    rgba_dir = artifact_dir / "rgba"
    frame_dir = artifact_dir / "comparison_frames"
    rgba_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        Path(output_override).expanduser().resolve()
        if output_override is not None
        else resolve_path(path.parent, "../visualizations/kitti_sam3_track_video") / f"{name}.mp4"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (480 * 3, 280 + 38),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open KITTI SAM3 track MP4 writer")

    frame_records: list[dict[str, Any]] = []
    try:
        for offset, candidate in enumerate(window):
            result = segment_candidate(candidate, segmenter, settings)
            if result is None:
                raise RuntimeError(
                    f"SAM3 failed strict frame {candidate['frame_index']} for {sequence}/{track_id}"
                )
            rgba_path = rgba_dir / f"{int(candidate['frame_index']):06d}.png"
            Image.fromarray(result["rgba"]).save(rgba_path)
            comparison = _comparison_frame(candidate, result)
            comparison_path = frame_dir / f"{int(candidate['frame_index']):06d}.jpg"
            cv2.imwrite(str(comparison_path), comparison)
            writer.write(comparison)
            frame_records.append(
                {
                    "offset": offset,
                    "candidate_id": int(candidate["candidate_id"]),
                    "frame_index": int(candidate["frame_index"]),
                    "source_image": candidate["source_image"],
                    "source_bbox_xywh": candidate["source_bbox_xywh"],
                    "rgba": str(rgba_path),
                    "comparison_frame": str(comparison_path),
                    "sam3_score": float(result["selected"].score),
                    "sam3_gt_match_iou": float(result["match_iou"]),
                    "mask_area": int(result["mask_area"]),
                }
            )
    finally:
        writer.release()

    report = {
        "ok": len(frame_records) == length,
        "source_dataset": "KITTI Tracking training",
        "split_role": "train_crop_allowed",
        "eval_leakage": sequence in metadata["split_policy"]["eval_sequences"],
        "sequence": sequence,
        "track_id": track_id,
        "native_category": window[0]["native_category"],
        "category": window[0]["category"],
        "sam3_prompt": prompt,
        "sam3_precision": segmenter.precision,
        "start_frame": first_frame,
        "end_frame": last_frame,
        "length": length,
        "video": str(output_path),
        "artifact_dir": str(artifact_dir),
        "frames": frame_records,
    }
    save_json(artifact_dir / "report.json", report)
    return {
        "ok": report["ok"],
        "video": str(output_path),
        "artifact_dir": str(artifact_dir),
        "frames": len(frame_records),
        "score_min": min(item["sam3_score"] for item in frame_records),
        "iou_min": min(item["sam3_gt_match_iou"] for item in frame_records),
        "eval_leakage": report["eval_leakage"],
    }


def _relaunch(args: argparse.Namespace) -> int:
    config, path = load_config(args.config)
    python_path = config_path(config, path, "paths", "sam3_python")
    command = [
        str(python_path),
        "-m",
        "scripts.kitti_sam3_track_video",
        "--config",
        str(path),
        "--sequence",
        args.sequence,
        "--track-id",
        str(args.track_id),
        "--length",
        str(args.length),
    ]
    if args.start_frame is not None:
        command.extend(["--start-frame", str(args.start_frame)])
    if args.output is not None:
        command.extend(["--output", str(args.output)])
    return subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Render one train-only KITTI SAM3 crop track video")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--track-id", required=True, type=int)
    parser.add_argument("--start-frame", type=int, default=None)
    parser.add_argument("--length", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if importlib.util.find_spec("sam3") is None:
        raise SystemExit(_relaunch(args))
    print(
        run(
            args.config,
            sequence=args.sequence,
            track_id=args.track_id,
            start_frame=args.start_frame,
            length=args.length,
            output_override=args.output,
        )
    )


if __name__ == "__main__":
    main()
