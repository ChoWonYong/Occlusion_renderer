from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from common.config import config_path, load_config, resolve_path
from common.gpu_budget import enforce_account_gpu_budget


# The four photometric steps ByteTrack's _distort bundles, in the order it applies
# them. Selectable individually so an arm can drop one without losing the rest.
PHOTOMETRIC_STEPS = ("brightness", "contrast", "hue", "saturation")
_ALL_PHOTOMETRIC = ",".join(PHOTOMETRIC_STEPS)

AUG_LEVELS = {
    # aug level -> (flip, photometric, mosaic_mixup)
    # Multi-scale resize (exp.random_size) is driven by the trainer and stays on
    # at every level; only letterbox+normalize is left when everything is off.
    "none": ("0", "", "0"),
    "nofliphue": ("0", "brightness,contrast,saturation", "0"),
    "jitter": ("1", _ALL_PHOTOMETRIC, "0"),
    "full": ("1", _ALL_PHOTOMETRIC, "1"),  # + mosaic + mixup + affine (last-10ep off)
}


def _experiment_name(
    condition: str,
    seed: int,
    aug: str = "full",
    paste_mode: str | None = None,
    tag: str | None = None,
) -> str:
    """``tag`` separates arms that differ only in the dataset they were built
    from — same condition, aug and paste mode, different config."""
    suffix = f"_{paste_mode}" if condition == "treatment" and paste_mode else ""
    tag_part = f"_{tag}" if tag else ""
    return f"phase1_{condition}{suffix}{tag_part}_{aug}_seed{seed}"


def build_command(
    config_file: str | Path,
    condition: str,
    seed: int,
    aug: str = "full",
    max_epoch: int | None = None,
    paste_mode: str | None = None,
    resume: bool = False,
    tag: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    if condition not in {"baseline", "treatment"}:
        raise ValueError("condition must be baseline or treatment")
    if aug not in AUG_LEVELS:
        raise ValueError(f"aug must be one of {sorted(AUG_LEVELS)}")
    config, path = load_config(config_file)
    bytetrack = config_path(config, path, "paths", "bytetrack_repo")
    checkpoint = config_path(config, path, "paths", "coco_pretrained_yolox_x")
    dataset_root = resolve_path(path.parent, config["dataset"]["output_dir"])
    exp_file = resolve_path(path.parent, config["train"]["exp_file"])
    if condition == "treatment":
        # The two paste modes produce different training sets; picking the wrong
        # file would silently compare the wrong pair.
        mode = paste_mode or str(config["dataset"].get("paste_mode", "replace"))
        train_key = f"treatment_{mode}_train_json"
        if train_key not in config["dataset"]:
            train_key = "treatment_train_json"
    else:
        train_key = "baseline_train_json"
    train_ann = Path(config["dataset"][train_key]).name
    experiment_name = _experiment_name(condition, seed, aug, paste_mode, tag)
    output_dir = resolve_path(path.parent, config["train"]["output_dir"])
    flip, photometric, mosaic_mixup = AUG_LEVELS[aug]
    epochs = int(max_epoch if max_epoch is not None else config["train"]["epochs"])
    # ByteTrack's resume_train reads its checkpoint from -c when that is set, and
    # takes start_epoch out of the file, so resuming means pointing -c at the run's
    # own latest_ckpt. Leaving it on the COCO weights would silently restart from
    # epoch 0 with pretrained weights.
    if resume:
        checkpoint = output_dir / experiment_name / "latest_ckpt.pth.tar"
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
    if resume:
        command.append("--resume")
    env = os.environ.copy()
    # ByteTrack's multi-GPU launcher spawns workers via the bare command
    # ``python3``. Directly invoking this environment's interpreter does not
    # activate Conda or prepend its bin directory to PATH, so without this the
    # workers can silently fall back to /usr/bin/python3 and miss YOLOX deps.
    python_bin = str(Path(sys.executable).parent)
    env["PATH"] = python_bin + os.pathsep + env.get("PATH", "")
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
            "KDS_FLIP": flip,
            "KDS_PHOTOMETRIC": photometric,
            "KDS_MOSAIC_MIXUP": mosaic_mixup,
            "KDS_SEED": str(seed),
            "KDS_EXPERIMENT_NAME": experiment_name,
            "KDS_YOLOX_OUTPUT_DIR": str(output_dir),
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
    resume: bool = False,
    tag: str | None = None,
) -> int:
    command, env = build_command(
        config_file, condition, seed, aug, max_epoch, paste_mode, resume, tag
    )
    print(" ".join(shlex.quote(part) for part in command))
    print(
        f"# aug={aug} flip={env['KDS_FLIP']} photometric={env['KDS_PHOTOMETRIC'] or '(none)'} "
        f"mosaic_mixup={env['KDS_MOSAIC_MIXUP']} epochs={env['KDS_MAX_EPOCH']} "
        f"train_ann={env['KDS_YOLOX_TRAIN_ANN']}"
    )
    if resume:
        print("# resume=on: continuing from the run's own latest_ckpt (epoch read from it)")
    if dry_run:
        return 0
    config, _ = load_config(config_file)
    budget = enforce_account_gpu_budget(
        config.get("resources", {}), project_limit_key="phase1_train_max_gpus"
    )
    devices = int(config["train"]["devices"])
    if budget["requested"] != devices:
        raise RuntimeError(
            f"training config requests {devices} GPUs, but CUDA_VISIBLE_DEVICES "
            f"exposes {budget['requested']}"
        )
    checkpoint = Path(command[command.index("-c") + 1])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"resume checkpoint not found: {checkpoint}"
            if resume
            else f"COCO-pretrained YOLOX-X checkpoint not found: {checkpoint}"
        )
    return subprocess.run(command, env=env, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune COCO-pretrained YOLOX-X on KITTI")
    parser.add_argument("--config", default="configs/default.yaml", type=Path)
    parser.add_argument("--condition", choices=["baseline", "treatment"], required=True)
    parser.add_argument("--aug", choices=sorted(AUG_LEVELS), default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-epoch", type=int, default=None, help="override epoch count (smoke)")
    parser.add_argument(
        "--paste-mode",
        choices=["append", "replace"],
        default=None,
        help="which treatment set to train on; must match how data/build_phase1.py was run",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help=(
            "extra label in the experiment name, for arms that differ only by the "
            "dataset they were built from (e.g. --tag pastejit_mid)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "continue an interrupted run from <output_dir>/<experiment_name>/latest_ckpt.pth.tar; "
            "the epoch to restart at is read from the checkpoint"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    raise SystemExit(
        run(
            args.config, args.condition, args.seed, args.aug,
            args.max_epoch, args.dry_run, args.paste_mode, args.resume, args.tag,
        )
    )


if __name__ == "__main__":
    main()
