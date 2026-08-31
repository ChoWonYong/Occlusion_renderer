from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

from common.config import config_path, load_config, resolve_path
from common.io import load_json, save_json
from data.kitti_tracking import load_sequence_labels
from data.mot17 import parse_gt
from mining.detected_tracklets import bbox_iou_xyxy


def _xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    x, y, width, height = (float(value) for value in box)
    return [x, y, x + width, y + height]


def _kitti_category(native: str) -> str | None:
    if native in {"Car", "Van", "Truck", "Tram"}:
        return "car"
    if native in {"Pedestrian", "Person_sitting"}:
        return "person"
    return None


class GroundTruthIndex:
    """Lazy GT reader used only by the post-extraction audit."""

    def __init__(self, config: Mapping[str, Any], config_file: Path) -> None:
        self.kitti_root = config_path(config, config_file, "paths", "kitti_tracking")
        self.mot_root = config_path(config, config_file, "paths", "mot17")
        self._cache: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = {}

    def sequence(self, dataset: str, sequence: str) -> dict[int, list[dict[str, Any]]]:
        key = (dataset.upper(), sequence)
        if key in self._cache:
            return self._cache[key]
        by_frame: dict[int, list[dict[str, Any]]] = {}
        if key[0] == "KITTI":
            labels = load_sequence_labels(
                self.kitti_root / "training" / "label_02" / f"{sequence}.txt"
            )
            for frame_index, objects in labels.items():
                for obj in objects:
                    category = _kitti_category(obj.category)
                    if obj.track_id < 0 or category is None:
                        continue
                    by_frame.setdefault(frame_index, []).append(
                        {
                            "track_id": obj.track_id,
                            "category": category,
                            "bbox_xyxy": list(obj.bbox_xyxy),
                            "clean_reference": obj.occluded == 0 and obj.truncated <= 0.2,
                        }
                    )
        elif key[0] == "MOT17":
            rows = parse_gt(self.mot_root / "train" / sequence / "gt" / "gt.txt")
            for obj in rows:
                if obj.mark != 1 or obj.category_id != 1:
                    continue
                by_frame.setdefault(obj.frame_index, []).append(
                    {
                        "track_id": obj.track_id,
                        "category": "person",
                        "bbox_xyxy": _xywh_to_xyxy(obj.bbox),
                        "clean_reference": obj.visibility >= 0.8,
                    }
                )
        else:
            raise ValueError(f"unsupported source dataset in detector pool: {dataset}")
        self._cache[key] = by_frame
        return by_frame


def _best_gt(
    frame: Mapping[str, Any], objects: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any] | None, float]:
    category = str(frame["category"])
    detector_box = _xywh_to_xyxy(frame["source_bbox_xywh"])
    eligible = [item for item in objects if str(item["category"]) == category]
    if not eligible:
        return None, 0.0
    best = max(eligible, key=lambda item: bbox_iou_xyxy(detector_box, item["bbox_xyxy"]))
    return best, bbox_iou_xyxy(detector_box, best["bbox_xyxy"])


def audit_detector_pool(
    metadata: Mapping[str, Any], gt: GroundTruthIndex, *, iou_threshold: float = 0.5
) -> dict[str, Any]:
    all_ious: list[float] = []
    matched = 0
    clean_matched = 0
    total = 0
    purity_numerator = 0
    purity_denominator = 0
    per_category: dict[str, dict[str, int]] = {}
    for tracklet in metadata.get("tracklets", []):
        identities: Counter[int] = Counter()
        for frame in tracklet.get("frames", []):
            total += 1
            category = str(frame["category"])
            bucket = per_category.setdefault(category, {"frames": 0, "matched_iou50": 0})
            bucket["frames"] += 1
            objects = gt.sequence(str(frame["source_dataset"]), str(frame["sequence"])).get(
                int(frame["frame_index"]), []
            )
            best, iou = _best_gt(frame, objects)
            all_ious.append(iou)
            if best is not None and iou >= iou_threshold:
                matched += 1
                bucket["matched_iou50"] += 1
                identities[int(best["track_id"])] += 1
                if bool(best.get("clean_reference", False)):
                    clean_matched += 1
                    bucket["clean_matched_iou50"] = bucket.get("clean_matched_iou50", 0) + 1
        if identities:
            purity_numerator += max(identities.values())
            purity_denominator += sum(identities.values())
    for bucket in per_category.values():
        bucket["matched_rate_iou50"] = (
            bucket["matched_iou50"] / bucket["frames"] if bucket["frames"] else 0.0
        )
        bucket["clean_frame_precision"] = (
            bucket.get("clean_matched_iou50", 0) / bucket["frames"]
            if bucket["frames"] else 0.0
        )
    return {
        "tracklets": len(metadata.get("tracklets", [])),
        "frames": total,
        "matched_frames_iou50": matched,
        "matched_rate_iou50": matched / total if total else 0.0,
        "clean_matched_frames_iou50": clean_matched,
        "clean_frame_precision": clean_matched / total if total else 0.0,
        "clean_rate_among_matched_frames": clean_matched / matched if matched else 0.0,
        "mean_best_iou": sum(all_ious) / len(all_ious) if all_ious else 0.0,
        "median_best_iou": median(all_ious) if all_ious else 0.0,
        "identity_purity_diagnostic": (
            purity_numerator / purity_denominator if purity_denominator else 0.0
        ),
        "per_category": dict(sorted(per_category.items())),
        "note": "GT is used only for this post-hoc audit, never for candidate selection or SAM3 input",
    }


def _pool_stats(metadatas: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records = [record for metadata in metadatas for record in metadata.get("tracklets", [])]
    # The accepted occluder protocol is two-class even if an old artifact still
    # contains one inert bicycle tracklet.
    records = [record for record in records if str(record.get("category")) in {"car", "person"}]
    frames = sum(int(record.get("length", len(record.get("frames", [])))) for record in records)
    return {
        "tracklets": len(records),
        "frames": frames,
        "category_counts": dict(sorted(Counter(str(record["category"]) for record in records).items())),
    }


def compare(config_file: str | Path) -> dict[str, Any]:
    config, path = load_config(config_file)
    settings = config["detector_tracklet_pool"]
    candidate_file = resolve_path(path.parent, settings["candidate_output_dir"]) / "tracklets.json"
    detector_file = resolve_path(path.parent, settings["output_dir"]) / "tracklets.json"
    for required in (candidate_file, detector_file):
        if not required.is_file():
            raise FileNotFoundError(f"comparison input not found: {required}")
    candidates = load_json(candidate_file)
    detector_pool = load_json(detector_file)
    old_pool_files = [
        resolve_path(path.parent, config["tracklet_pool"]["output_dir"]) / "tracklets.json",
        resolve_path(path.parent, config["kitti_sam3_pool"]["tracklet_output_dir"]) / "tracklets.json",
    ]
    old_pools = [load_json(pool_file) for pool_file in old_pool_files]
    gt = GroundTruthIndex(config, path)
    candidate_audit = audit_detector_pool(candidates, gt)
    final_audit = audit_detector_pool(detector_pool, gt)
    old_stats = _pool_stats(old_pools)
    detector_stats = _pool_stats([detector_pool])
    report = {
        "protocol": {
            "selection_gt_used": False,
            "audit_gt_used": True,
            "iou_match_threshold": 0.5,
            "quality_filter": "deferred_to_phase2",
        },
        "detector_candidates_before_sam3": candidate_audit,
        "detector_pool_after_sam3": final_audit,
        "gt_derived_pool_reference": old_stats,
        "detector_pool": detector_stats,
        "relative_to_gt_pool": {
            "tracklet_count_ratio": (
                detector_stats["tracklets"] / old_stats["tracklets"]
                if old_stats["tracklets"] else 0.0
            ),
            "frame_count_ratio": (
                detector_stats["frames"] / old_stats["frames"] if old_stats["frames"] else 0.0
            ),
            # Primary Phase-1 quality criterion: selected frames should match a
            # clean GT object (MOT visibility>=0.8; KITTI occluded=0 and
            # truncation<=0.2). Identity purity remains diagnostic only because
            # the downstream detector is trained frame by frame.
            "clean_frame_drop_from_gt_selected_ideal": 1.0
            - final_audit["clean_frame_precision"],
        },
    }
    output_dir = resolve_path(
        path.parent,
        config.get("tracklet_comparison", {}).get(
            "output_dir", "../artifacts/phase1/tracklet_comparison"
        ),
    )
    output_path = output_dir / "report.json"
    save_json(output_path, report)
    report["output"] = str(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare GT-free and GT-derived tracklet pools")
    parser.add_argument("--config", default="configs/phase1_detector.yaml", type=Path)
    args = parser.parse_args()
    print(compare(args.config))


if __name__ == "__main__":
    main()
