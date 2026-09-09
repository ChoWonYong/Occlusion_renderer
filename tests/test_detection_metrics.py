import unittest

from eval.run_detection_metrics import (
    _group_selector,
    _merge_shards,
    _selected_run_specs,
    evaluate_predictions,
)


class DetectionMetricTest(unittest.TestCase):
    DATASET = {
        "info": {},
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "x.png"}],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [0, 0, 20, 20],
                "area": 400,
                "iscrowd": 0,
                "kitti": {"occluded": 0},
            },
            {
                "id": 2,
                "image_id": 1,
                "category_id": 1,
                "bbox": [50, 50, 20, 20],
                "area": 400,
                "iscrowd": 0,
                "kitti": {"occluded": 1},
            },
        ],
    }
    PREDICTIONS = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 20, 20], "score": 0.9},
        {"image_id": 1, "category_id": 1, "bbox": [50, 50, 20, 20], "score": 0.8},
    ]

    def test_group_selectors_exclude_unknown(self) -> None:
        annotation = lambda value: {"kitti": {"occluded": value}}
        self.assertTrue(_group_selector("non_occluded")(annotation(0)))
        self.assertTrue(_group_selector("occluded")(annotation(1)))
        self.assertTrue(_group_selector("occluded")(annotation(2)))
        self.assertFalse(_group_selector("occluded")(annotation(3)))
        self.assertFalse(_group_selector("non_occluded")(annotation(3)))

    def test_excluded_group_gt_absorbs_its_detection_as_ignore(self) -> None:
        for group in ("all", "occluded", "non_occluded"):
            result = evaluate_predictions(self.DATASET, self.PREDICTIONS, group)
            self.assertAlmostEqual(result["combined"]["AP"], 100.0)
            self.assertAlmostEqual(result["combined"]["Recall50"], 100.0)
            expected = 2 if group == "all" else 1
            self.assertEqual(result["ground_truth"]["total"], expected)

    def test_merge_shards_rejects_duplicate_video(self) -> None:
        payloads = [
            {
                "shard_index": index,
                "num_shards": 2,
                "video_ids": [1],
                "frames": 1,
                "predictions": [],
            }
            for index in range(2)
        ]
        with self.assertRaisesRegex(ValueError, "more than one"):
            _merge_shards(payloads)

    def test_enabled_runs_excludes_inherited_run_specs(self) -> None:
        selected = _selected_run_specs(
            {
                "runs": {"old": {"label": "old"}, "a": {"label": "a"}},
                "enabled_runs": ["a"],
            }
        )
        self.assertEqual(selected, {"a": {"label": "a"}})


if __name__ == "__main__":
    unittest.main()
