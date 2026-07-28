"""Aggregate per-class TrackEval outputs into the fin_tuned_baseline.md table.

For each run, TrackEval is executed once per class (KDS_CAR/TRUCK/PERSON/BICYCLE).
This script reads those per-class outputs and builds the combined ("전체") row the
same way fin_tuned_baseline.md does:

- HOTA/DetA/AssA: detection-weighted across classes at the per-alpha level. For
  each localization threshold alpha, DetA is computed from the summed TP/FN/FP
  across classes and AssA is the TP-weighted mean of the per-class AssA; HOTA is
  sqrt(DetA*AssA), then averaged over the 19 alphas.
- MOTA / IDF1: recomputed from the summed raw counts.
- IDSW / FP / FN: per-class count sums.

Per-class rows are read straight from pedestrian_summary.txt.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path
from typing import Any

from common.config import load_config, resolve_path

ALPHAS = list(range(5, 100, 5))  # 0.05 .. 0.95
CLASS_BENCHMARK = {"car": "KDS_CAR", "truck": "KDS_TRUCK", "person": "KDS_PERSON", "bicycle": "KDS_BICYCLE"}


def _class_dir(tracking_root: Path, run_name: str, benchmark: str) -> Path:
    return tracking_root / run_name / "trackers" / f"{benchmark}-eval" / run_name


def _parse_summary(path: Path) -> dict[str, float]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    header = lines[0].split()
    values = [float(value) for value in lines[1].split()]
    return dict(zip(header, values))


def _parse_detailed_combined(path: Path) -> dict[str, list[float]]:
    rows = list(csv.reader(path.open(encoding="utf-8")))
    header = rows[0]
    combined = next(row for row in rows[1:] if row[0] == "COMBINED")
    record = dict(zip(header, combined))
    return {
        "tp": [float(record[f"HOTA_TP___{a}"]) for a in ALPHAS],
        "fn": [float(record[f"HOTA_FN___{a}"]) for a in ALPHAS],
        "fp": [float(record[f"HOTA_FP___{a}"]) for a in ALPHAS],
        "assa": [float(record[f"AssA___{a}"]) for a in ALPHAS],
    }


def _combined_hota(per_class_detailed: list[dict[str, list[float]]]) -> dict[str, float]:
    hota_alpha, deta_alpha, assa_alpha = [], [], []
    for index in range(len(ALPHAS)):
        total_tp = sum(c["tp"][index] for c in per_class_detailed)
        total_fn = sum(c["fn"][index] for c in per_class_detailed)
        total_fp = sum(c["fp"][index] for c in per_class_detailed)
        denom = total_tp + total_fn + total_fp
        det = total_tp / denom if denom > 0 else 0.0
        ass = (
            sum(c["assa"][index] * c["tp"][index] for c in per_class_detailed) / total_tp
            if total_tp > 0
            else 0.0
        )
        deta_alpha.append(det)
        assa_alpha.append(ass)
        hota_alpha.append(math.sqrt(det * ass))
    count = len(ALPHAS)
    return {
        "HOTA": 100.0 * sum(hota_alpha) / count,
        "DetA": 100.0 * sum(deta_alpha) / count,
        "AssA": 100.0 * sum(assa_alpha) / count,
    }


def aggregate_run(tracking_root: Path, run_name: str, class_names: list[str]) -> dict[str, Any]:
    per_class: dict[str, dict[str, float]] = {}
    detailed: list[dict[str, list[float]]] = []
    total = {"IDSW": 0.0, "FP": 0.0, "FN": 0.0, "GT": 0.0, "IDTP": 0.0, "IDFN": 0.0, "IDFP": 0.0}
    for name in class_names:
        directory = _class_dir(tracking_root, run_name, CLASS_BENCHMARK[name])
        summary = _parse_summary(directory / "pedestrian_summary.txt")
        detailed.append(_parse_detailed_combined(directory / "pedestrian_detailed.csv"))
        per_class[name] = {
            "HOTA": summary["HOTA"], "DetA": summary["DetA"], "AssA": summary["AssA"],
            "MOTA": summary["MOTA"], "IDF1": summary["IDF1"],
            "IDSW": summary["IDSW"], "FP": summary["CLR_FP"], "FN": summary["CLR_FN"],
        }
        total["IDSW"] += summary["IDSW"]
        total["FP"] += summary["CLR_FP"]
        total["FN"] += summary["CLR_FN"]
        total["GT"] += summary["GT_Dets"]
        total["IDTP"] += summary["IDTP"]
        total["IDFN"] += summary["IDFN"]
        total["IDFP"] += summary["IDFP"]

    combined = _combined_hota(detailed)
    combined["MOTA"] = 100.0 * (1.0 - (total["FN"] + total["FP"] + total["IDSW"]) / total["GT"])
    combined["IDF1"] = 100.0 * (2 * total["IDTP"]) / (2 * total["IDTP"] + total["IDFN"] + total["IDFP"])
    combined["IDSW"] = total["IDSW"]
    combined["FP"] = total["FP"]
    combined["FN"] = total["FN"]
    return {"run": run_name, "combined": combined, "per_class": per_class}


COLUMNS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW", "FP", "FN"]


def _row(label: str, metrics: dict[str, float]) -> str:
    cells = []
    for column in COLUMNS:
        value = metrics.get(column)
        if value is None:
            cells.append("-")
        elif column in {"IDSW", "FP", "FN"}:
            cells.append(f"{int(round(value))}")
        else:
            cells.append(f"{value:.2f}")
    return "| " + label + " | " + " | ".join(cells) + " |"


_SEED_SUFFIX = re.compile(r"_seed\d+(?=_|$)")


def arm_of(run_name: str) -> str:
    """Run name with the seed stripped, so seeds of one condition group together."""
    return _SEED_SUFFIX.sub("", run_name)


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and *sample* standard deviation (ddof=1), matching fine_tuned_baseline.md."""
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def _summary_row(label: str, metrics_list: list[dict[str, float]], count: int) -> str:
    cells = []
    for column in COLUMNS:
        values = [m[column] for m in metrics_list if m.get(column) is not None]
        if not values:
            cells.append("-")
            continue
        mean, std = _mean_std(values)
        fmt = "{:.0f}" if column in {"IDSW", "FP", "FN"} else "{:.2f}"
        cells.append(f"**{fmt.format(mean)} ± {fmt.format(std)}**" if count > 1 else fmt.format(mean))
    return f"| **{label}** (n={count}) | " + " | ".join(cells) + " |"


def _section(
    head: str, results: list[dict[str, Any]], key: str, class_name: str | None = None
) -> list[str]:
    def metrics(result: dict[str, Any]) -> dict[str, float]:
        return result[key] if class_name is None else result[key][class_name]

    lines = [head]
    for result in results:
        lines.append(_row(result["run"], metrics(result)))
    grouped: dict[str, list[dict[str, float]]] = {}
    for result in results:
        grouped.setdefault(arm_of(result["run"]), []).append(metrics(result))
    if any(len(items) > 1 for items in grouped.values()):
        lines.append("")
        for arm, items in grouped.items():
            lines.append(_summary_row(arm, items, len(items)))
    return lines


def to_markdown(results: list[dict[str, Any]], class_names: list[str]) -> str:
    head = "| Run | " + " | ".join(COLUMNS) + " |\n|" + "---|" * (len(COLUMNS) + 1)
    lines = ["## 전체 (combined, detection-weighted)"]
    lines += _section(head, results, "combined")
    for name in class_names:
        lines.append(f"\n### {name}")
        lines += _section(head, results, "per_class", name)
    lines.append("")
    lines.append("표준편차는 seed 간 sample standard deviation (ddof=1).")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate per-class TrackEval outputs into a comparison table")
    parser.add_argument("--config", default="configs/phase1_kitti.yaml", type=Path)
    parser.add_argument("--runs", nargs="+", required=True, help="TrackEval run_name directories under tracker.output_dir")
    parser.add_argument("--out", type=Path, default=None, help="write markdown table here")
    args = parser.parse_args()
    config, path = load_config(args.config)
    tracking_root = resolve_path(path.parent, config["tracker"]["output_dir"])
    class_names = list(config["classes"]["names"])
    results = [aggregate_run(tracking_root, run, class_names) for run in args.runs]
    table = to_markdown(results, class_names)
    print(table)
    if args.out:
        args.out.write_text(table + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
