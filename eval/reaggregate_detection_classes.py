"""Reaggregate saved COCO detector metrics over a selected class subset.

COCO bbox AP/AR averages category-level precision/recall entries uniformly.
Therefore a class-subset result can be reproduced exactly from the saved
per-class arrays without running detector inference again.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from common.io import load_json, save_json
from eval.run_detection_metrics import _write_comparison_report


METRICS = ("AP", "AP50", "AP75", "AR100", "Recall50")


def _mean_finite(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def _subset_group(group: Mapping[str, Any], class_names: Sequence[str]) -> dict[str, Any]:
    available = group["per_class"]
    missing = [name for name in class_names if name not in available]
    if missing:
        raise ValueError(f"classes missing from saved detector metrics: {missing}")
    per_class = {name: copy.deepcopy(available[name]) for name in class_names}
    combined = {
        metric: _mean_finite([per_class[name][metric] for name in class_names])
        for metric in METRICS
    }
    ground_truth_by_class = group["ground_truth"]["per_class"]
    ground_truth = {
        "total": sum(int(ground_truth_by_class[name]) for name in class_names),
        "per_class": {
            name: int(ground_truth_by_class[name]) for name in class_names
        },
    }
    return {
        "combined": combined,
        "per_class": per_class,
        "ground_truth": ground_truth,
    }


def reaggregate(summary: Mapping[str, Any], class_names: Sequence[str]) -> dict[str, Any]:
    names = [str(name) for name in class_names]
    if not names or len(names) != len(set(names)):
        raise ValueError("class_names must be a non-empty list of unique names")
    result = copy.deepcopy(dict(summary))
    for run in result["runs"].values():
        run["metrics"] = {
            group_name: _subset_group(group, names)
            for group_name, group in run["metrics"].items()
        }

    comparison = result.get("comparison")
    if comparison is None:
        if {"baseline", "confidence_filtered"}.issubset(result["runs"]):
            comparison = {
                "reference_run": "baseline",
                "target_run": "confidence_filtered",
            }
            result["comparison"] = comparison
        else:
            raise ValueError("saved detector metrics do not define a comparison pair")
    reference = result["runs"][comparison["reference_run"]]["metrics"]
    target = result["runs"][comparison["target_run"]]["metrics"]
    comparison["delta_target_minus_reference"] = {
        group_name: {
            "combined": {
                metric: float(
                    target[group_name]["combined"][metric]
                    - reference[group_name]["combined"][metric]
                )
                for metric in METRICS
            },
            "per_class": {
                name: {
                    metric: float(
                        target[group_name]["per_class"][name][metric]
                        - reference[group_name]["per_class"][name][metric]
                    )
                    for metric in METRICS
                }
                for name in names
            },
        }
        for group_name in target
    }
    if (
        comparison["reference_run"] == "baseline"
        and comparison["target_run"] == "confidence_filtered"
    ):
        result["delta_confidence_filtered_minus_baseline"] = comparison[
            "delta_target_minus_reference"
        ]
    result["protocol"]["evaluation_classes"] = names
    result["protocol"]["class_subset_aggregation"] = (
        "exact COCO category-axis mean over saved per-class metrics"
    )
    result["annotations"] = int(
        next(iter(result["runs"].values()))["metrics"]["all"]["ground_truth"]["total"]
    )
    result["inference_reused"] = True
    return result


def run(summary_file: str | Path, class_names: Sequence[str], output_dir: str | Path) -> dict[str, Any]:
    source = Path(summary_file).resolve()
    destination = Path(output_dir).resolve()
    result = reaggregate(load_json(source), class_names)
    result["source_summary"] = str(source)
    result["output_dir"] = str(destination)
    save_json(destination / "summary.json", result)
    _write_comparison_report(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reaggregate saved detector metrics over selected classes"
    )
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--classes", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.summary, args.classes, args.output_dir)
    print(result["output_dir"])


if __name__ == "__main__":
    main()
