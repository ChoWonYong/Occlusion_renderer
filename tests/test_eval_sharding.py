import unittest

from eval.run_boxmot import _balanced_video_shards, _merge_prediction_shards


class EvalShardingTest(unittest.TestCase):
    def test_balances_whole_videos_by_frame_count(self) -> None:
        frames = {
            1: [{}] * 10,
            2: [{}] * 8,
            3: [{}] * 7,
            4: [{}] * 3,
        }
        shards = _balanced_video_shards(frames, 2)
        self.assertEqual(sorted(video for shard in shards for video in shard), [1, 2, 3, 4])
        self.assertEqual([sum(len(frames[video]) for video in shard) for shard in shards], [13, 15])

    def test_merges_rows_and_restores_video_order(self) -> None:
        payloads = [
            {
                "shard_index": 1,
                "num_shards": 2,
                "videos": [{"id": 2, "name": "b"}],
                "gt_rows": {"car": {"b": ["gt-b\n"]}},
                "pred_rows": {"car": {"b": ["pred-b\n"]}},
            },
            {
                "shard_index": 0,
                "num_shards": 2,
                "videos": [{"id": 1, "name": "a"}],
                "gt_rows": {"car": {"a": ["gt-a\n"]}},
                "pred_rows": {"car": {"a": ["pred-a\n"]}},
            },
        ]
        videos, gt_rows, pred_rows = _merge_prediction_shards(payloads, ["car", "person"])
        self.assertEqual([video["id"] for video in videos], [1, 2])
        self.assertEqual(gt_rows["car"]["a"], ["gt-a\n"])
        self.assertEqual(pred_rows["car"]["b"], ["pred-b\n"])
        self.assertEqual(dict(gt_rows["person"]), {})

    def test_rejects_duplicate_video_assignment(self) -> None:
        payloads = [
            {
                "shard_index": index,
                "num_shards": 2,
                "videos": [{"id": 1, "name": "a"}],
                "gt_rows": {},
                "pred_rows": {},
            }
            for index in range(2)
        ]
        with self.assertRaisesRegex(ValueError, "more than one"):
            _merge_prediction_shards(payloads, ["car"])


if __name__ == "__main__":
    unittest.main()
