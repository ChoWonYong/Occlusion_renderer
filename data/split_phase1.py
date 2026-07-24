from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common.config import config_path, load_config, resolve_path
from common.io import save_json
from data.kitti_tracking import discover_sequences


def make_split(
    sequence_ids: list[str], train_sequences: list[str], eval_sequences: list[str]
) -> dict[str, Any]:
    available = set(sequence_ids)
    train = list(train_sequences)
    evaluation = list(eval_sequences)
    if len(train) != len(set(train)):
        raise ValueError("train_sequences contains duplicate sequence ids")
    if len(evaluation) != len(set(evaluation)):
        raise ValueError("eval_sequences contains duplicate sequence ids")
    overlap = set(train) & set(evaluation)
    if overlap:
        raise ValueError(f"train/eval sequences overlap: {sorted(overlap)}")
    assigned = set(train) | set(evaluation)
    if assigned != available:
        missing = sorted(available - assigned)
        unknown = sorted(assigned - available)
        raise ValueError(f"split does not match discovered sequences: missing={missing}, unknown={unknown}")
    return {
        "dataset": "KITTI Tracking training",
        "unit": "sequence",
        "train_sequences": train,
        "eval_sequences": evaluation,
        "crop_allowed_sequences": train,
        "leakage_guard": "Occluder crops must only originate from crop_allowed_sequences or COCO.",
    }


def run(config_file: str | Path) -> dict[str, Any]:
    config, path = load_config(config_file)
    root = config_path(config, path, "paths", "kitti_tracking")
    sequences = discover_sequences(root)
    split = make_split(
        sequences,
        list(config["split"]["train_sequences"]),
        list(config["split"]["eval_sequences"]),
    )
    output = resolve_path(path.parent, config["split"]["output"])
    save_json(output, split)
    return {"output": str(output), **split}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a leakage-safe KITTI Tracking Phase-1 split")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    args = parser.parse_args()
    result = run(args.config)
    print(f"split saved to {result['output']}")
    print(f"train={result['train_sequences']}")
    print(f"eval={result['eval_sequences']}")


if __name__ == "__main__":
    main()
