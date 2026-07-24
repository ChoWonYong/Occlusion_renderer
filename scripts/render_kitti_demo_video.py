from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from common.config import config_path, load_config, resolve_path


def _first_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    raise ValueError(f"manifest is empty: {path}")


def run(
    config_file: str | Path,
    output: str | Path,
    *,
    padding: int = 5,
    panel_width: int = 960,
    layout: str = "comparison",
    full_sequence: bool = False,
) -> dict[str, object]:
    if layout not in {"comparison", "synthetic"}:
        raise ValueError("layout must be 'comparison' or 'synthetic'")
    config, path = load_config(config_file)
    synthesis_dir = resolve_path(path.parent, config["tracklet_synthesis"]["output_dir"])
    manifest = _first_manifest(synthesis_dir / "manifest.jsonl")
    source_video = str(manifest["source_video"])
    output_video = str(manifest["output_video"])
    source_dir = config_path(config, path, "paths", "kitti_tracking") / "training" / "image_02" / source_video
    synthetic_dir = synthesis_dir / "frames" / output_video
    synthetic_frames = sorted(synthetic_dir.glob("*.jpg"))
    if not synthetic_frames:
        raise FileNotFoundError(f"no synthetic frames found: {synthetic_dir}")

    starts = [int(item["start_position"]) for item in manifest["tracklets"]]
    ends = [int(item["end_position"]) for item in manifest["tracklets"]]
    first_position = 0 if full_sequence else max(0, min(starts) - padding)
    last_position = (
        len(synthetic_frames) - 1
        if full_sequence
        else min(len(synthetic_frames) - 1, max(ends) + padding)
    )
    selected = synthetic_frames[first_position : last_position + 1]

    sample = cv2.imread(str(selected[0]), cv2.IMREAD_COLOR)
    if sample is None:
        raise RuntimeError(f"failed to read {selected[0]}")
    panel_height = int(round(sample.shape[0] * panel_width / sample.shape[1]))
    header = 34
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(destination),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (panel_width * (2 if layout == "comparison" else 1), panel_height + header),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 writer with codec mp4v")

    written = 0
    try:
        for synthetic_path in selected:
            frame_name = synthetic_path.stem
            source_path = source_dir / f"{frame_name}.png"
            synthetic = cv2.imread(str(synthetic_path), cv2.IMREAD_COLOR)
            original = cv2.imread(str(source_path), cv2.IMREAD_COLOR) if layout == "comparison" else None
            if synthetic is None or (layout == "comparison" and original is None):
                raise RuntimeError(f"failed to read comparison frame {frame_name}")
            synthetic = cv2.resize(synthetic, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
            if layout == "comparison":
                assert original is not None
                original = cv2.resize(original, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
                canvas = cv2.hconcat([original, synthetic])
            else:
                canvas = synthetic
            canvas = cv2.copyMakeBorder(canvas, header, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
            if layout == "comparison":
                cv2.putText(
                    canvas, "Original KITTI", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
            cv2.putText(
                canvas,
                f"Synthetic | frame {frame_name}",
                (panel_width + 15 if layout == "comparison" else 15, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            writer.write(canvas)
            written += 1
    finally:
        writer.release()
    return {
        "output": str(destination),
        "frames": written,
        "fps": 10,
        "source_video": source_video,
        "synthetic_video": output_video,
        "positions": [first_position, last_position],
        "layout": layout,
        "full_sequence": full_sequence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Render original-vs-synthetic KITTI MP4")
    parser.add_argument("--config", default="configs/demo_kitti_scaled_tracklet.yaml", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--padding", type=int, default=5)
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--layout", choices=["comparison", "synthetic"], default="comparison")
    parser.add_argument("--full-sequence", action="store_true")
    args = parser.parse_args()
    print(
        run(
            args.config,
            args.output,
            padding=args.padding,
            panel_width=args.panel_width,
            layout=args.layout,
            full_sequence=args.full_sequence,
        )
    )


if __name__ == "__main__":
    main()
