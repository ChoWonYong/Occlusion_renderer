from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from common.io import load_json, save_json
from pool.filter_detector_tracklet_pool import filter_tracklet


DEFAULT_CASES = {
    "padding_00_context": (
        Path("artifacts/official_split/sam3_context_ablation_00pct")
        / "variant_b_context_preserved"
        / "tracklets.json",
        "b_context",
        "0% padding (context preserved)",
    ),
    "padding_10_context": (
        Path("artifacts/official_split/sam3_context_ablation_10pct")
        / "variant_b_context_preserved"
        / "tracklets.json",
        "b_context",
        "10% padding (context preserved)",
    ),
    "padding_20_context": (
        Path("artifacts/official_split/sam3_context_ablation_20pct")
        / "variant_b_context_preserved"
        / "tracklets.json",
        "b_context",
        "20% padding (context preserved)",
    ),
    "padding_20_recrop": (
        Path("artifacts/official_split/sam3_context_ablation_20pct")
        / "variant_a_bbox_clipped"
        / "tracklets.json",
        "a_bbox",
        "20% padding -> detector-bbox re-crop",
    ),
}


FrameKey = tuple[int, int]
Metric = Mapping[str, Any]


def summarize_metrics(metrics: Sequence[Metric]) -> dict[str, Any]:
    if not metrics:
        return {
            "frames": 0,
            "macro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0},
            "micro": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0},
            "complete_frames": 0,
            "complete_rate": 0.0,
            "pixels": {"tp": 0, "fp": 0, "fn": 0},
        }

    macro_precision = fmean(float(metric["precision"]) for metric in metrics)
    macro_recall = fmean(float(metric["recall"]) for metric in metrics)
    tp = sum(int(metric["tp"]) for metric in metrics)
    fp = sum(int(metric["fp"]) for metric in metrics)
    fn = sum(int(metric["fn"]) for metric in metrics)
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    complete_frames = sum(bool(metric["complete"]) for metric in metrics)
    return {
        "frames": len(metrics),
        "macro": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": (
                2 * macro_precision * macro_recall / (macro_precision + macro_recall)
                if macro_precision + macro_recall
                else 0.0
            ),
            "iou": fmean(float(metric["iou"]) for metric in metrics),
        },
        "micro": {
            "precision": micro_precision,
            "recall": micro_recall,
            "f1": (
                2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if micro_precision + micro_recall
                else 0.0
            ),
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        },
        "complete_frames": complete_frames,
        "complete_rate": complete_frames / len(metrics),
        "pixels": {"tp": tp, "fp": fp, "fn": fn},
    }


def load_selected_case(manifest: Path, policy: str, label: str) -> dict[str, Any]:
    payload = load_json(manifest)
    tracklets = payload["tracklets"]
    if len(tracklets) != 200:
        raise ValueError(f"{manifest} contains {len(tracklets)} tracklets, expected 200")

    frame_rows: dict[FrameKey, dict[str, Any]] = {}
    mots_statuses: Counter[str] = Counter()
    category_tracklets: Counter[str] = Counter()
    source_tracklets: Counter[str] = Counter()
    filter_scenarios: Counter[str] = Counter()
    input_frames = sum(len(tracklet["frames"]) for tracklet in tracklets)
    for tracklet in tracklets:
        decision, filtered = filter_tracklet(tracklet, threshold=0.6, min_frames=30)
        filter_scenarios[str(decision["scenario"])] += 1
        if filtered is None:
            continue
        candidate_id = int(filtered["source_candidate_id"])
        category = str(filtered["category"])
        category_tracklets[category] += 1
        source_tracklets[str(filtered["source_dataset"])] += 1
        for frame in filtered["frames"]:
            key = (candidate_id, int(frame["frame_index"]))
            if key in frame_rows:
                raise ValueError(f"duplicate selected frame {key} in {manifest}")
            if float(frame["detector_confidence"]) < 0.6:
                raise ValueError(f"retained frame {key} is below confidence 0.6")
            gt = frame.get("gt", {})
            mots_statuses[str(gt.get("status", "not_run"))] += 1
            metric = None
            if gt.get("matched"):
                metric = gt["metrics"][policy]
            frame_rows[key] = {
                "category": category,
                "source_dataset": str(tracklet["source_dataset"]),
                "metric": metric,
            }

    matched_rows = [row for row in frame_rows.values() if row["metric"] is not None]
    per_class = {
        category: summarize_metrics(
            [row["metric"] for row in matched_rows if row["category"] == category]
        )
        for category in ("car", "person")
    }
    return {
        "label": label,
        "manifest": str(manifest.resolve()),
        "policy": policy,
        "input_tracklets": len(tracklets),
        "input_frames": input_frames,
        "retained_tracklets": sum(category_tracklets.values()),
        "retained_frames": len(frame_rows),
        "filter_scenario_counts": dict(sorted(filter_scenarios.items())),
        "tracklets_by_class": dict(sorted(category_tracklets.items())),
        "tracklets_by_source": dict(sorted(source_tracklets.items())),
        "mots_status_counts": dict(sorted(mots_statuses.items())),
        "mots_matched_frames": len(matched_rows),
        "overall": summarize_metrics([row["metric"] for row in matched_rows]),
        "per_class": per_class,
        "_frame_rows": frame_rows,
    }


def _manifest_frame_keys(path: Path) -> set[FrameKey]:
    tracklets = load_json(path)["tracklets"]
    return {
        (int(tracklet["source_candidate_id"]), int(frame["frame_index"]))
        for tracklet in tracklets
        for frame in tracklet["frames"]
    }


def build_summary(root: Path) -> dict[str, Any]:
    cases: dict[str, Any] = {}
    for name, (relative_manifest, policy, label) in DEFAULT_CASES.items():
        cases[name] = load_selected_case(root / relative_manifest, policy, label)

    common_keys = set.intersection(
        *(
            {key for key, row in case["_frame_rows"].items() if row["metric"] is not None}
            for case in cases.values()
        )
    )
    common_frame_metrics = {
        name: summarize_metrics(
            [case["_frame_rows"][key]["metric"] for key in sorted(common_keys)]
        )
        for name, case in cases.items()
    }

    legacy_manifest = (
        root / "artifacts/official_split/codetr_sam3_tracklets_conf60/tracklets.json"
    )
    legacy_keys = _manifest_frame_keys(legacy_manifest)
    recrop_keys = set(cases["padding_20_recrop"]["_frame_rows"])

    for case in cases.values():
        del case["_frame_rows"]

    return {
        "mode": "selected_200_then_conf60_tracklet_mots_mask_quality",
        "protocol": {
            "selection": "the final 200 tracklets stored by each padding run, followed by the production confidence filter",
            "selection_used_gt": False,
            "detector_confidence_min": 0.6,
            "minimum_consecutive_frames": 30,
            "run_policy": "retain the first longest consecutive confidence-passing run",
            "evaluation_subset": "retained paste-input frames with a matched KITTI MOTS ground-truth instance",
            "macro_definition": "mean of per-frame/object foreground-pixel metrics",
            "pixel_definition": {
                "precision": "TP / (TP + FP)",
                "recall": "TP / (TP + FN)",
                "iou": "TP / (TP + FP + FN)",
                "complete": "recall >= 0.90",
            },
        },
        "case_order": list(DEFAULT_CASES),
        "cases": cases,
        "common_matched_frame_count": len(common_keys),
        "common_matched_frame_metrics": common_frame_metrics,
        "legacy_recrop_manifest_equivalence": {
            "legacy_manifest": str(legacy_manifest.resolve()),
            "reproduced_source_manifest": cases["padding_20_recrop"]["manifest"],
            "legacy_retained_frames": len(legacy_keys),
            "reproduced_retained_frames": len(recrop_keys),
            "same_retained_frame_keys": legacy_keys == recrop_keys,
        },
    }


def _pct(value: float) -> str:
    return f"{100 * value:.2f}"


def write_report(output_dir: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Final selected-200 -> confidence-0.6 tracklet MOTS mask quality",
        "",
        "Each row starts from that run's final capped 200-tracklet manifest, applies detector confidence >= 0.6, "
        "and retains the first longest passing run when it has at least 30 consecutive frames. "
        "Only retained frames matched to a KITTI MOTS instance enter the pixel metrics; MOTS GT was not used for selection or filtering.",
        "The primary values are macro averages over matched frame/object masks.",
        "",
        "## Each case's actual paste-input pool",
        "",
        "| Case | Input tracks | Retained tracks | Retained frames | MOTS matched | Precision (%) | Recall (%) | F1 (%) | IoU (%) | Complete R>=90% (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in summary["case_order"]:
        case = summary["cases"][name]
        metric = case["overall"]
        macro = metric["macro"]
        lines.append(
            f"| {case['label']} | {case['input_tracklets']} | {case['retained_tracklets']} | "
            f"{case['retained_frames']} | {case['mots_matched_frames']} | {_pct(macro['precision'])} | "
            f"{_pct(macro['recall'])} | {_pct(macro['f1'])} | {_pct(macro['iou'])} | "
            f"{_pct(metric['complete_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-class macro metrics",
            "",
            "| Case | Class | MOTS matched | Precision (%) | Recall (%) | F1 (%) | IoU (%) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in summary["case_order"]:
        case = summary["cases"][name]
        for category in ("car", "person"):
            metric = case["per_class"][category]
            macro = metric["macro"]
            lines.append(
                f"| {case['label']} | {category} | {metric['frames']} | "
                f"{_pct(macro['precision'])} | {_pct(macro['recall'])} | "
                f"{_pct(macro['f1'])} | {_pct(macro['iou'])} |"
            )

    lines.extend(
        [
            "",
            f"## Same-frame comparison ({summary['common_matched_frame_count']} common MOTS frames)",
            "",
            "| Case | Precision (%) | Recall (%) | F1 (%) | IoU (%) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name in summary["case_order"]:
        case = summary["cases"][name]
        macro = summary["common_matched_frame_metrics"][name]["macro"]
        lines.append(
            f"| {case['label']} | {_pct(macro['precision'])} | {_pct(macro['recall'])} | "
            f"{_pct(macro['f1'])} | {_pct(macro['iou'])} |"
        )

    equivalence = summary["legacy_recrop_manifest_equivalence"]
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"The reproduced 20% re-crop confidence-filtered pool has exactly the same retained frame keys as the legacy production manifest: "
            f"**{equivalence['same_retained_frame_keys']}** "
            f"({equivalence['legacy_retained_frames']} vs {equivalence['reproduced_retained_frames']} frames).",
            "",
            "Precision = TP/(TP+FP), recall = TP/(TP+FN), and IoU = TP/(TP+FP+FN), using foreground mask pixels.",
            "All metrics above are after the detector-confidence >= 0.6 filter and therefore describe the tracklets actually eligible for paste.",
            "",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate MOTS mask precision/recall for each final capped 200-tracklet pool"
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/official_split/sam3_padding_selected_200_conf60_mots"),
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    summary = build_summary(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "summary.json", summary)
    write_report(output_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
