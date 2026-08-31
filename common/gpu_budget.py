from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping


def _visible_device_count() -> int:
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is None:
        return 1
    devices = [
        item.strip()
        for item in value.split(",")
        if item.strip() and item.strip() != "-1"
    ]
    if not devices:
        raise RuntimeError("CUDA_VISIBLE_DEVICES exposes no GPU")
    return len(set(devices))


def parse_account_gpu_usage(
    output: str,
    *,
    current_uid: int,
    uid_for_pid: Callable[[int], int | None],
) -> set[str]:
    """Return distinct GPU UUIDs used by processes owned by ``current_uid``."""
    devices: set[str] = set()
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) < 2:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        if uid_for_pid(pid) == current_uid:
            devices.add(fields[1])
    return devices


def _uid_for_pid(pid: int) -> int | None:
    try:
        return Path(f"/proc/{pid}").stat().st_uid
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None


def enforce_account_gpu_budget(
    resources: Mapping[str, Any],
    *,
    project_limit_key: str = "phase1_detector_max_gpus",
) -> dict[str, int]:
    """Fail closed if this launch could exceed the user's account-wide ceiling."""
    account_limit = int(resources.get("account_gpu_limit", 6))
    project_limit = int(resources.get(project_limit_key, 1))
    requested = _visible_device_count()
    if requested > project_limit:
        raise RuntimeError(
            f"this Phase-1 command exposes {requested} GPUs, "
            f"but its per-command limit is {project_limit}"
        )
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown nvidia-smi error"
        raise RuntimeError(f"cannot verify account-wide GPU budget: {detail}")
    in_use = parse_account_gpu_usage(
        result.stdout,
        current_uid=os.getuid(),
        uid_for_pid=_uid_for_pid,
    )
    # Conservatively count this as an additional GPU even if a selected UUID may
    # overlap. This also discourages sharing a device with another account job.
    if len(in_use) + requested > account_limit:
        raise RuntimeError(
            f"GPU account limit exceeded: {len(in_use)} already in use + "
            f"{requested} requested > {account_limit}"
        )
    return {
        "account_limit": account_limit,
        "already_in_use": len(in_use),
        "requested": requested,
        "remaining_after_launch": account_limit - len(in_use) - requested,
    }
