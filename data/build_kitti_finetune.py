from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.config import config_path, load_config, resolve_path
from common.io import load_json, save_json
from data.build_phase1 import _materialize_images
from data.kitti_tracking import convert_tracking_to_video_coco


def run(config_file: str | Path) -> dict[str, Any]:
    """Build original-only KITTI train/eval COCO files from the fixed sequence split."""
    config, path = load_config(config_file)
    split = load_json(resolve_path(path.parent, config["split"]["output"]))
    kitti_root = config_path(config, path, "paths", "kitti_tracking")
    class_map = config["classes"]["kitti_map"]
    output_root = resolve_path(path.parent, config["dataset"]["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    train_source = convert_tracking_to_video_coco(
        kitti_root, split["train_sequences"], class_map
    )
    eval_source = convert_tracking_to_video_coco(
        kitti_root, split["eval_sequences"], class_map
    )
    train_local = _materialize_images(train_source, output_root, "original_train")
    eval_local = _materialize_images(eval_source, output_root, "original_eval")

    train_path = output_root / config["dataset"]["finetune_train_json"]
    eval_path = output_root / config["dataset"]["eval_json"]
    save_json(train_path, train_local)
    save_json(eval_path, eval_local)
    summary = {
        "train_sequences": list(split["train_sequences"]),
        "eval_sequences": list(split["eval_sequences"]),
        "train_images": len(train_local["images"]),
        "eval_images": len(eval_local["images"]),
        "train_annotations": len(train_local["annotations"]),
        "eval_annotations": len(eval_local["annotations"]),
        "train_json": str(train_path),
        "eval_json": str(eval_path),
    }
    save_json(output_root / "kitti_split_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build original-only KITTI train/eval data for YOLOX fine-tuning"
    )
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
