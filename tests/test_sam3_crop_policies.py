import unittest

import numpy as np

from eval.compare_sam3_crop_policies import (
    evaluate_mask,
    longest_passing_positions,
    make_mask_variants,
    match_mots_instance,
)


class MaskVariantTest(unittest.TestCase):
    def test_zero_context_makes_bbox_and_context_policies_identical(self) -> None:
        context = np.zeros((5, 7, 3), dtype=np.uint8)
        mask = np.zeros(context.shape[:2], dtype=np.uint8)
        mask[1:4, 2:6] = 1
        variants = make_mask_variants(
            context,
            context,
            mask,
            [0, 0, 7, 5],
            [11, 13, 7, 5],
            [11, 13, 7, 5],
        )
        self.assertTrue(
            np.array_equal(variants["a_bbox"]["rgba"], variants["b_context"]["rgba"])
        )
        self.assertEqual(
            variants["a_bbox"]["crop_bbox_xywh"],
            variants["b_context"]["crop_bbox_xywh"],
        )
        self.assertEqual(variants["a_bbox"]["mask_area"], variants["b_context"]["mask_area"])

    def test_context_policy_preserves_mask_outside_detector_bbox(self) -> None:
        context = np.zeros((8, 10, 3), dtype=np.uint8)
        crop = context[2:6, 3:7]
        mask = np.zeros(context.shape[:2], dtype=np.uint8)
        mask[2:6, 3:7] = 1
        mask[1, 4] = 1
        variants = make_mask_variants(
            crop,
            context,
            mask,
            [3, 2, 4, 4],
            [13, 22, 4, 4],
            [10, 20, 10, 8],
        )
        self.assertEqual(variants["a_bbox"]["rgba"].shape, (4, 4, 4))
        self.assertEqual(variants["b_context"]["rgba"].shape, (8, 10, 4))
        self.assertEqual(variants["a_bbox"]["mask_area"], 16)
        self.assertEqual(variants["b_context"]["mask_area"], 17)
        self.assertEqual(variants["a_bbox"]["crop_bbox_xywh"], [13.0, 22.0, 4.0, 4.0])
        self.assertEqual(variants["b_context"]["crop_bbox_xywh"], [10.0, 20.0, 10.0, 8.0])

    def test_full_gt_outside_crop_counts_as_false_negative(self) -> None:
        target = np.zeros((6, 6), dtype=bool)
        target[1:5, 1:5] = True
        alpha_a = np.full((2, 4), 255, dtype=np.uint8)
        alpha_b = np.full((4, 4), 255, dtype=np.uint8)
        a = evaluate_mask(alpha_a, [1, 1, 4, 2], target)
        b = evaluate_mask(alpha_b, [1, 1, 4, 4], target)
        self.assertEqual((a["tp"], a["fp"], a["fn"]), (8, 0, 8))
        self.assertAlmostEqual(a["recall"], 0.5)
        self.assertAlmostEqual(a["precision"], 1.0)
        self.assertEqual((b["tp"], b["fp"], b["fn"]), (16, 0, 0))
        self.assertAlmostEqual(b["recall"], 1.0)
        self.assertTrue(b["complete"])


class MotsAssociationTest(unittest.TestCase):
    def test_detector_bbox_selects_same_class_instance_only(self) -> None:
        labels = np.zeros((10, 12), dtype=np.uint16)
        labels[1:5, 1:4] = 1001
        labels[2:7, 7:11] = 1002
        labels[0:2, 8:10] = 2001
        matched = match_mots_instance(labels, "car", [7, 2, 4, 5])
        self.assertIsNotNone(matched)
        assert matched is not None
        mask, instance_id, overlap, bbox = matched
        self.assertEqual(instance_id, 1002)
        self.assertEqual(bbox, [7.0, 2.0, 4.0, 5.0])
        self.assertAlmostEqual(overlap, 1.0)
        self.assertEqual(int(mask.sum()), 20)


class LongestRunTest(unittest.TestCase):
    def test_first_longest_run_is_retained(self) -> None:
        self.assertEqual(
            longest_passing_positions([False, True, True, False, True, True]),
            [1, 2],
        )


if __name__ == "__main__":
    unittest.main()
