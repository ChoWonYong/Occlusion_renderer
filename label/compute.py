from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from common.schema import bbox_to_mask, compute_ratio, mask_to_bbox, occlusion_level


VALID_DETECTOR_BBOX_POLICIES = ("visible", "amodal_original")

# 2026-07-28: an ignore rule (iscrowd=1 above an occlusion ratio / below a visible
# area) was implemented and then removed. It does not do what "ignore region"
# implies here. ByteTrack's ``MOTDataset.load_anno_from_ids`` calls
# ``getAnnIds(..., iscrowd=False)``, so a flagged annotation is dropped from the
# target list, and ``yolo_head`` has no crowd handling — SimOTA then assigns
# objectness 0 to every anchor there. The location becomes an explicit *negative*,
# training the detector to report nothing where an occluded object is. That is the
# opposite of ``amodal_original``, whose whole purpose is to teach full-extent
# localisation from partial evidence, and it deleted labels the baseline trains on:
# 3.0% of real annotations in a smoke sequence, of which 15 were only lightly
# occluded (rho < 0.5) and just 6 came from the heavy-occlusion branch.
# Supporting real ignore regions would mean masking those locations out of the
# loss inside YOLOX, which is not something the label layer can express.


def label_frame(
    background_annotation: Mapping[str, Any],
    occluder_mask: np.ndarray,
    image_shape: tuple[int, int],
    occluder_track_ids: Sequence[int],
    mask_source: str = "kitti_bbox_proxy",
    detector_bbox_policy: str = "visible",
) -> dict[str, Any]:
    """Extend a real victim's GT after paste.

    ``detector_bbox_policy`` selects which box the detector-facing ``bbox`` holds:
    - ``"amodal_original"`` (Phase-1 confirmed): keep the original/amodal KITTI box,
      so the detector learns to localise the full extent of an occluded object,
      matching the original-bbox convention used at eval;
    - ``"visible"`` (legacy): the post-paste visible box.
    ``visible_bbox`` and ``amodal_bbox`` are always stored regardless of policy.

    Every victim the baseline trains on stays a positive here, however heavily it
    is covered — see the note at the top of this module.
    """
    if detector_bbox_policy not in VALID_DETECTOR_BBOX_POLICIES:
        raise ValueError(f"unknown detector_bbox_policy: {detector_bbox_policy}")
    height, width = image_shape
    amodal = bbox_to_mask(background_annotation["bbox"], height, width)
    overlap = np.logical_and(amodal != 0, np.asarray(occluder_mask) != 0)
    visible = np.logical_and(amodal != 0, np.logical_not(overlap)).astype(np.uint8)
    ratio = compute_ratio(visible, amodal)
    visible_bbox = mask_to_bbox(visible)
    amodal_bbox = [float(value) for value in background_annotation["bbox"]]
    detector_bbox = amodal_bbox if detector_bbox_policy == "amodal_original" else visible_bbox
    return {
        "bbox": detector_bbox,
        "visible_bbox": visible_bbox,
        # The detector target is the amodal box, so ``area`` must describe that box
        # too: MOTDataset drops annotations with ``area == 0``, and keying it to the
        # visible pixels would silently delete fully covered victims — the same
        # "make it a negative" failure the removed ignore rule had.
        "area": int(amodal.sum()) if detector_bbox_policy == "amodal_original" else int(visible.sum()),
        "visible_area": int(visible.sum()),
        "amodal_bbox": amodal_bbox,
        "occlusion_ratio": ratio,
        "occlusion_level": occlusion_level(ratio),
        "occluder_ids": [int(track_id) for track_id in occluder_track_ids] if ratio > 0 else [],
        "synthetic_occluder": False,
        "detector_bbox_policy": detector_bbox_policy,
        "iscrowd": 0,
        "provenance": {
            "occluded_by_synth": ratio > 0,
            "amodal_mask_source": mask_source,
        },
    }


def label_frame_multi(
    background_annotation: Mapping[str, Any],
    occluder_masks: Mapping[int, np.ndarray],
    image_shape: tuple[int, int],
    mask_source: str = "kitti_bbox_proxy",
    detector_bbox_policy: str = "visible",
) -> dict[str, Any]:
    """Label a real object and retain only synthetic IDs that actually overlap it.

    The overlap scan is restricted to the victim's own bounding box, which is
    exactly the region ``bbox_to_mask`` would fill, so it is equivalent to the
    full-image intersection it replaces.
    """
    height, width = image_shape
    x, y, w, h = (float(value) for value in background_annotation["bbox"])
    left, top = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
    right, bottom = min(width, int(np.ceil(x + w))), min(height, int(np.ceil(y + h)))
    overlapping_ids: list[int] = []
    if right > left and bottom > top:
        for track_id, mask in occluder_masks.items():
            if np.asarray(mask)[top:bottom, left:right].any():
                overlapping_ids.append(int(track_id))
    union = np.zeros((height, width), dtype=np.uint8)
    for track_id in overlapping_ids:
        union |= np.asarray(occluder_masks[track_id], dtype=np.uint8)
    return label_frame(
        background_annotation,
        union,
        image_shape,
        overlapping_ids,
        mask_source,
        detector_bbox_policy,
    )


def label_synthetic_occluder(
    *,
    visible_mask: np.ndarray,
    amodal_mask: np.ndarray,
    occluder_ids: Sequence[int],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build extended GT for a pasted object, including synthetic-on-synthetic occlusion."""
    visible = np.asarray(visible_mask, dtype=np.uint8)
    amodal = np.asarray(amodal_mask, dtype=np.uint8)
    ratio = compute_ratio(visible, amodal)
    visible_bbox = mask_to_bbox(visible)
    return {
        "bbox": visible_bbox,
        "visible_bbox": visible_bbox,
        "area": int(visible.sum()),
        "amodal_bbox": mask_to_bbox(amodal),
        "occlusion_ratio": ratio,
        "occlusion_level": occlusion_level(ratio),
        "occluder_ids": [int(track_id) for track_id in occluder_ids] if ratio > 0 else [],
        "synthetic": True,
        "synthetic_occluder": True,
        "provenance": dict(provenance),
    }
