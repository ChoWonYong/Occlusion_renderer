from __future__ import annotations

import argparse
import importlib.util
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from common.config import config_path, load_config, resolve_path
from common.io import load_json, save_json
from pool.build_kitti_sam3_pool import inventory, segment_candidate


def _checkerboard(height: int, width: int, tile: int = 12) -> np.ndarray:
    yy, xx = np.indices((height, width))
    cells = (xx // tile + yy // tile) % 2
    return np.repeat(np.where(cells[..., None] == 0, 225, 180), 3, axis=2).astype(np.uint8)


def _save_visuals(
    output_dir: Path,
    prefix: str,
    candidate: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, str]:
    from PIL import Image, ImageDraw

    output_dir.mkdir(parents=True, exist_ok=True)
    context_image = Image.fromarray(result["context"])
    draw = ImageDraw.Draw(context_image)
    tx, ty, tw, th = result["target_box"]
    draw.rectangle((tx, ty, tx + tw, ty + th), outline=(0, 255, 255), width=3)
    for instance in result["instances"]:
        x, y, width, height = instance.bbox
        color = (0, 255, 0) if instance is result["selected"] else (255, 190, 0)
        draw.rectangle((x, y, x + width, y + height), outline=color, width=2)
        draw.text((x + 2, max(0, y - 12)), f"{instance.score:.2f}", fill=color)

    crop = np.asarray(result["crop"], dtype=np.uint8)
    alpha = np.asarray(result["alpha"], dtype=np.uint8)
    binary = alpha != 0
    overlay = crop.astype(np.float32)
    overlay[binary] = overlay[binary] * 0.45 + np.asarray([255, 40, 40]) * 0.55
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    checker = _checkerboard(crop.shape[0], crop.shape[1])
    checker[binary] = crop[binary]

    context_path = output_dir / f"{prefix}_context_detections.png"
    crop_path = output_dir / f"{prefix}_crop.png"
    overlay_path = output_dir / f"{prefix}_overlay.png"
    rgba_path = output_dir / f"{prefix}_rgba.png"
    comparison_path = output_dir / f"{prefix}_comparison.png"
    context_image.save(context_path)
    Image.fromarray(crop).save(crop_path)
    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(result["rgba"]).save(rgba_path)

    panel_width = crop.shape[1]
    label_height = 30
    sheet = Image.new("RGB", (panel_width * 3, crop.shape[0] + label_height), "white")
    labels = [
        (f"KITTI {candidate['native_category']} bbox", crop),
        (f"SAM3 '{candidate['sam3_prompt']}' mask", overlay),
        ("RGBA on checker", checker),
    ]
    sheet_draw = ImageDraw.Draw(sheet)
    for index, (label, panel) in enumerate(labels):
        sheet.paste(Image.fromarray(panel), (index * panel_width, label_height))
        sheet_draw.text((index * panel_width + 6, 8), label, fill=(20, 20, 20))
    sheet.save(comparison_path)
    return {
        "context_detections": str(context_path),
        "crop": str(crop_path),
        "overlay": str(overlay_path),
        "rgba": str(rgba_path),
        "comparison": str(comparison_path),
    }


def run(
    config_file: str | Path,
    *,
    candidate_id: int | None = None,
    category: str = "car",
    output_override: str | Path | None = None,
) -> dict[str, Any]:
    from segment.sam3 import Sam3Backend

    config, path = load_config(config_file)
    settings = config["kitti_sam3_pool"]
    pool_dir = resolve_path(path.parent, settings["output_dir"])
    candidates_path = pool_dir / "candidates.json"
    if not candidates_path.is_file():
        inventory(path)
    metadata = load_json(candidates_path)
    candidates = list(metadata["candidates"])
    if candidate_id is not None:
        selected_candidates = [
            item for item in candidates if int(item["candidate_id"]) == candidate_id
        ]
        if not selected_candidates:
            raise ValueError(f"KITTI SAM3 candidate_id not found: {candidate_id}")
        candidate = selected_candidates[0]
    else:
        selected_candidates = [item for item in candidates if item["category"] == category]
        if not selected_candidates:
            raise ValueError(f"no KITTI SAM3 candidates for category={category}")
        candidate = max(selected_candidates, key=lambda item: float(item["bbox_area"]))

    segmenter = Sam3Backend(
        config_path(config, path, "paths", "sam3_checkpoint"),
        bpe_path=config_path(config, path, "paths", "sam3_bpe"),
        device=str(settings.get("device", "cuda")),
        confidence_threshold=float(settings.get("confidence_threshold", 0.5)),
        precision=str(settings.get("precision", "auto")),
    )
    result = segment_candidate(candidate, segmenter, settings)
    if result is None:
        raise RuntimeError(
            f"SAM3 prompt '{candidate['sam3_prompt']}' did not produce a valid GT-matched mask "
            f"for candidate {candidate['candidate_id']}"
        )
    mask_area = int(result["mask_area"])
    crop_area = int(result["alpha"].size)
    coverage = mask_area / crop_area if crop_area else 0.0
    output_dir = (
        Path(output_override).expanduser().resolve()
        if output_override is not None
        else resolve_path(path.parent, "../visualizations/kitti_sam3_smoke")
    )
    prefix = (
        f"candidate_{int(candidate['candidate_id']):06d}_"
        f"{candidate['sequence']}_{int(candidate['frame_index']):06d}_"
        f"{candidate['native_category'].lower()}"
    )
    paths = _save_visuals(output_dir, prefix, candidate, result)
    report = {
        "ok": True,
        "backend": "sam3",
        "precision": segmenter.precision,
        "candidate_id": int(candidate["candidate_id"]),
        "split_role": candidate["split_role"],
        "sequence": candidate["sequence"],
        "eval_leakage": candidate["sequence"] in metadata["split_policy"]["eval_sequences"],
        "frame_index": int(candidate["frame_index"]),
        "track_id": int(candidate["track_id"]),
        "native_category": candidate["native_category"],
        "category": candidate["category"],
        "text_prompt": candidate["sam3_prompt"],
        "source_image": candidate["source_image"],
        "source_bbox_xywh": candidate["source_bbox_xywh"],
        "crop_bbox_xywh": result["clipped_box"],
        "sam3_context_bbox_xywh": result["context_box"],
        "detections": len(result["instances"]),
        "selected_score": float(result["selected"].score),
        "selected_gt_iou": float(result["match_iou"]),
        "mask_area": mask_area,
        "crop_area": crop_area,
        "mask_coverage": coverage,
        "outputs": paths,
    }
    report_path = output_dir / f"{prefix}_report.json"
    save_json(report_path, report)
    report["report"] = str(report_path)
    return report


def _relaunch(
    config_file: Path,
    candidate_id: int | None,
    category: str,
    output: Path | None,
) -> int:
    config, path = load_config(config_file)
    python_path = config_path(config, path, "paths", "sam3_python")
    command = [
        str(python_path),
        "-m",
        "scripts.kitti_sam3_smoke",
        "--config",
        str(path),
        "--category",
        category,
    ]
    if candidate_id is not None:
        command.extend(["--candidate-id", str(candidate_id)])
    if output is not None:
        command.extend(["--output", str(output)])
    return subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM3 on one train-only KITTI GT crop")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    parser.add_argument("--candidate-id", type=int, default=None)
    parser.add_argument("--category", default="car")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if importlib.util.find_spec("sam3") is None:
        raise SystemExit(_relaunch(args.config, args.candidate_id, args.category, args.output))
    result = run(
        args.config,
        candidate_id=args.candidate_id,
        category=args.category,
        output_override=args.output,
    )
    print(result)


if __name__ == "__main__":
    main()
