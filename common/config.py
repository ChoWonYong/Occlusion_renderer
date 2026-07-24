from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required: pip install -r requirements.txt") from exc

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping")
    base_ref = raw.pop("base", None)
    if base_ref:
        base_path = resolve_path(config_path.parent, base_ref)
        base, _ = load_config(base_path)
        raw = _merge(base, raw)
    return raw, config_path


def resolve_path(config_dir: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def config_path(config: Mapping[str, Any], config_file: Path, *keys: str) -> Path:
    value: Any = config
    for key in keys:
        value = value[key]
    return resolve_path(config_file.parent, value)

