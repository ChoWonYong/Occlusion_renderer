import unittest

import numpy as np

from common.schema import validate_extended_annotation
from label.compute import label_frame


def _base_annotation() -> dict:
    return {"bbox": [0.0, 0.0, 10.0, 10.0], "track_id": 1, "category_id": 1}


def _left_half_mask() -> np.ndarray:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, :5] = 1  # occluder covers the left half of the victim
    return mask


class DetectorBboxPolicyTest(unittest.TestCase):
    def test_amodal_original_keeps_full_box(self) -> None:
        label = label_frame(
            _base_annotation(), _left_half_mask(), (10, 10), [7],
            detector_bbox_policy="amodal_original",
        )
        self.assertEqual(label["bbox"], [0.0, 0.0, 10.0, 10.0])
        self.assertEqual(label["amodal_bbox"], [0.0, 0.0, 10.0, 10.0])
        self.assertEqual(label["visible_bbox"], [5.0, 0.0, 5.0, 10.0])  # stored separately
        self.assertAlmostEqual(label["occlusion_ratio"], 0.5)
        self.assertEqual(label["occlusion_level"], 2)
        self.assertEqual(label["detector_bbox_policy"], "amodal_original")

    def test_visible_policy_is_the_legacy_default(self) -> None:
        label = label_frame(_base_annotation(), _left_half_mask(), (10, 10), [7])
        self.assertEqual(label["bbox"], [5.0, 0.0, 5.0, 10.0])
        self.assertEqual(label["bbox"], label["visible_bbox"])
        self.assertEqual(label["detector_bbox_policy"], "visible")

    def test_unknown_policy_raises(self) -> None:
        with self.assertRaises(ValueError):
            label_frame(_base_annotation(), _left_half_mask(), (10, 10), [7],
                        detector_bbox_policy="nonsense")


class ValidateExtendedAnnotationTest(unittest.TestCase):
    def _extended(self, policy: str) -> dict:
        visible = [5.0, 0.0, 5.0, 10.0]
        amodal = [0.0, 0.0, 10.0, 10.0]
        return {
            "bbox": amodal if policy == "amodal_original" else visible,
            "visible_bbox": visible,
            "amodal_bbox": amodal,
            "occlusion_ratio": 0.5,
            "occlusion_level": 2,
            "occluder_ids": [7],
            "synthetic": False,
            "detector_bbox_policy": policy,
            "provenance": {},
        }

    def test_accepts_amodal_and_visible_policies(self) -> None:
        validate_extended_annotation(self._extended("amodal_original"))
        validate_extended_annotation(self._extended("visible"))

    def test_rejects_bbox_not_matching_policy(self) -> None:
        broken = self._extended("amodal_original")
        broken["bbox"] = broken["visible_bbox"]  # wrong box for the policy
        with self.assertRaisesRegex(ValueError, "amodal_bbox"):
            validate_extended_annotation(broken)


if __name__ == "__main__":
    unittest.main()
