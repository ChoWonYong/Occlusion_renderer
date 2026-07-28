from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any

from common.config import config_path, load_config, resolve_path


AUG_LEVELS = {
    # aug level -> (online_jitter, mosaic_mixup)
    "none": ("0", "0"),      # resize/normalize only (+ HSV; flip off)
    "jitter": ("1", "0"),    # online flip + HSV, no mosaic/mixup/affine
    "full": ("1", "1"),      # flip + HSV + mosaic + mixup + affine (last-10ep off)
}


def _experiment_name(
    condition: str, seed: int, aug: str = "full", paste_mode: str | None = None
) -> str:
    label = "kitti_finetuned" if condition == "kitti" else condition
    suffix = f"_{paste_mode}" if condition == "treatment" and paste_mode else ""
    return f"phase1_{label}{suffix}_{aug}_seed{seed}"


def build_command(
    config_file: str | Path,
    condition: str,
    seed: int,
    aug: str = "full",
    max_epoch: int | None = None,
    paste_mode: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    if condition not in {"kitti", "baseline", "treatment"}:
        raise ValueError("condition must be kitti, baseline, or treatment")
    if aug not in AUG_LEVELS:
        raise ValueError(f"aug must be one of {sorted(AUG_LEVELS)}")
    config, path = load_config(config_file)
    bytetrack = config_path(config, path, "paths", "bytetrack_repo")
    checkpoint = config_path(config, path, "paths", "coco_pretrained_yolox_x")
    dataset_root = resolve_path(path.parent, config["dataset"]["output_dir"])
    exp_file = resolve_path(path.parent, config["train"]["exp_file"])
    if condition == "kitti":
        train_key = "finetune_train_json"
    elif condition == "treatment":
        # The two paste modes produce different training sets; picking the wrong
        # file would silently compare the wrong pair.
        mode = paste_mode or str(config["dataset"].get("paste_mode", "replace"))
        train_key = f"treatment_{mode}_train_json"
        if train_key not in config["dataset"]:
            train_key = "treatment_train_json"
    else:
        train_key = "baseline_train_json"
    train_ann = Path(config["dataset"][train_key]).name
    experiment_name = _experiment_name(condition, seed, aug, paste_mode)
    online_jitter, mosaic_mixup = AUG_LEVELS[aug]
    epochs = int(max_epoch if max_epoch is not None else config["train"]["epochs"])
    command = [
        sys.executable,
        str(bytetrack / "tools" / "train.py"),
        "-f",
        str(exp_file),
        "-d",
        str(config["train"]["devices"]),
        "-b",
        str(config["train"]["batch_size"]),
        "-expn",
        experiment_name,
        "-c",
        str(checkpoint),
    ]
    if bool(config["train"].get("fp16", True)):
        command.append("--fp16")
    if bool(config["train"].get("occupy_gpu", False)):
        command.append("-o")
    env = os.environ.copy()
    env.update(
        {
            "KDS_NUM_CLASSES": str(len(config["classes"]["names"])),
            "KDS_YOLOX_DATA_DIR": str(dataset_root),
            "KDS_YOLOX_TRAIN_ANN": train_ann,
            # The held-out validation split is evaluated only after training. The
            # trainer still constructs an evaluator, so point it at train data.
            "KDS_YOLOX_VAL_ANN": train_ann,
            "KDS_INPUT_SIZE": ",".join(str(value) for value in config["train"]["input_size"]),
            "KDS_MAX_EPOCH": str(epochs),
            "KDS_ONLINE_JITTER": online_jitter,
            "KDS_MOSAIC_MIXUP": mosaic_mixup,
            "KDS_SEED": str(seed),
            "KDS_EXPERIMENT_NAME": experiment_name,
            "KDS_YOLOX_OUTPUT_DIR": str(resolve_path(path.parent, config["train"]["output_dir"])),
        }
    )
    return command, env


def run(
    config_file: str | Path,
    condition: str,
    seed: int,
    aug: str = "full",
    max_epoch: int | None = None,
    dry_run: bool = False,
    paste_mode: str | None = None,
) -> int:
    command, env = build_command(config_file, condition, seed, aug, max_epoch, paste_mode)
    print(" ".join(shlex.quote(part) for part in command))
    print(f"# aug={aug} online_jitter={env['KDS_ONLINE_JITTER']} mosaic_mixup={env['KDS_MOSAIC_MIXUP']} epochs={env['KDS_MAX_EPOCH']} train_ann={env['KDS_YOLOX_TRAIN_ANN']}")
    if dry_run:
        return 0
    checkpoint = Path(command[command.index("-c") + 1])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"COCO-pretrained YOLOX-X checkpoint not found: {checkpoint}")
    return subprocess.run(command, env=env, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune COCO-pretrained YOLOX-X on KITTI")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    parser.add_argument(
        "--condition", choices=["kitti", "baseline", "treatment"], default="kitti"
    )
    parser.add_argument("--aug", choices=sorted(AUG_LEVELS), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epoch", type=int, default=None, help="override epoch count (smoke)")
    parser.add_argument(
        "--paste-mode",
        choices=["append", "replace"],
        default=None,
        help="which treatment set to train on; must match how data/build_phase1.py was run",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run(
            args.config, args.condition, args.seed, args.aug,
            args.max_epoch, args.dry_run, args.paste_mode,
        )
    )


if __name__ == "__main__":
    main()
