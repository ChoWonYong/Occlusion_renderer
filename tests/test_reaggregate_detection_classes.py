import unittest

from eval.reaggregate_detection_classes import METRICS, reaggregate


class ReaggregateDetectionClassesTest(unittest.TestCase):
    def test_selected_classes_are_averaged_and_delta_is_recomputed(self) -> None:
        def group(offset: float):
            per_class = {
                "car": {metric: 10.0 + offset for metric in METRICS},
                "truck": {metric: 90.0 + offset for metric in METRICS},
                "person": {metric: 30.0 + offset for metric in METRICS},
            }
            return {
                "combined": {metric: 0.0 for metric in METRICS},
                "per_class": per_class,
                "ground_truth": {
                    "total": 106,
                    "per_class": {"car": 100, "truck": 1, "person": 5},
                },
            }

        summary = {
            "protocol": {},
            "annotations": 106,
            "runs": {
                "zero": {"metrics": {"all": group(0.0)}},
                "twenty": {"metrics": {"all": group(2.0)}},
            },
            "comparison": {
                "reference_run": "zero",
                "target_run": "twenty",
                "delta_target_minus_reference": {},
            },
        }
        result = reaggregate(summary, ["car", "person"])
        metrics = result["runs"]["zero"]["metrics"]["all"]
        self.assertEqual(list(metrics["per_class"]), ["car", "person"])
        self.assertEqual(metrics["combined"]["AP"], 20.0)
        self.assertEqual(metrics["ground_truth"]["total"], 105)
        self.assertEqual(result["annotations"], 105)
        self.assertEqual(
            result["comparison"]["delta_target_minus_reference"]["all"]["combined"]["AP"],
            2.0,
        )

    def test_missing_class_is_rejected(self) -> None:
        summary = {
            "protocol": {},
            "annotations": 0,
            "runs": {
                "zero": {
                    "metrics": {
                        "all": {
                            "combined": {},
                            "per_class": {},
                            "ground_truth": {"total": 0, "per_class": {}},
                        }
                    }
                }
            },
            "comparison": {
                "reference_run": "zero",
                "target_run": "zero",
                "delta_target_minus_reference": {},
            },
        }
        with self.assertRaisesRegex(ValueError, "missing"):
            reaggregate(summary, ["car"])

    def test_legacy_baseline_summary_gets_comparison_pair(self) -> None:
        per_class = {
            name: {metric: value for metric in METRICS}
            for name, value in (("car", 10.0), ("person", 20.0))
        }
        group = {
            "combined": {},
            "per_class": per_class,
            "ground_truth": {
                "total": 2,
                "per_class": {"car": 1, "person": 1},
            },
        }
        summary = {
            "protocol": {},
            "annotations": 2,
            "runs": {
                "baseline": {"metrics": {"all": group}},
                "confidence_filtered": {"metrics": {"all": group}},
            },
            "delta_confidence_filtered_minus_baseline": {},
        }
        result = reaggregate(summary, ["car", "person"])
        self.assertEqual(result["comparison"]["reference_run"], "baseline")
        self.assertEqual(
            result["delta_confidence_filtered_minus_baseline"],
            result["comparison"]["delta_target_minus_reference"],
        )


if __name__ == "__main__":
    unittest.main()
