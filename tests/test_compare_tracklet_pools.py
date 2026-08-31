import unittest

from eval.compare_tracklet_pools import audit_detector_pool


class _FakeGt:
    def sequence(self, dataset, sequence):
        return {
            0: [{"track_id": 7, "category": "car", "bbox_xyxy": [0, 0, 10, 10], "clean_reference": True}],
            1: [{"track_id": 7, "category": "car", "bbox_xyxy": [1, 0, 11, 10], "clean_reference": False}],
        }


class CompareTrackletPoolsTest(unittest.TestCase):
    def test_audit_reports_iou_and_identity_without_filtering(self) -> None:
        metadata = {
            "tracklets": [
                {
                    "frames": [
                        {"source_dataset": "KITTI", "sequence": "0000", "frame_index": 0, "category": "car", "source_bbox_xywh": [0, 0, 10, 10]},
                        {"source_dataset": "KITTI", "sequence": "0000", "frame_index": 1, "category": "car", "source_bbox_xywh": [1, 0, 10, 10]},
                    ]
                }
            ]
        }
        report = audit_detector_pool(metadata, _FakeGt())
        self.assertEqual(report["matched_rate_iou50"], 1.0)
        self.assertEqual(report["clean_frame_precision"], 0.5)
        self.assertEqual(report["identity_purity_diagnostic"], 1.0)
        self.assertIn("post-hoc", report["note"])


if __name__ == "__main__":
    unittest.main()
