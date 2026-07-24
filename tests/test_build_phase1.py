import unittest

from data.build_phase1 import _filter_paste_frames


class FilterPasteFramesTest(unittest.TestCase):
    def _dataset(self) -> dict:
        return {
            "categories": [{"id": 1, "name": "car"}],
            "videos": [{"id": 1, "name": "v1"}, {"id": 2, "name": "v2"}],
            "images": [
                {"id": 10, "video_id": 1, "file_name": "frames/v1/000000.jpg"},  # paste
                {"id": 11, "video_id": 1, "file_name": "frames/v1/000001.jpg"},  # clean passthrough
                {"id": 12, "video_id": 2, "file_name": "frames/v2/000000.jpg"},  # clean passthrough
            ],
            "annotations": [
                {"id": 1, "image_id": 10, "track_id": 5, "synthetic_occluder": False},   # victim
                {"id": 2, "image_id": 10, "track_id": 900001, "synthetic_occluder": True},  # pasted occluder
                {"id": 3, "image_id": 11, "track_id": 5, "synthetic_occluder": False},   # clean victim
                {"id": 4, "image_id": 12, "track_id": 6, "synthetic_occluder": False},
            ],
        }

    def test_keeps_only_frames_with_a_pasted_occluder(self) -> None:
        filtered = _filter_paste_frames(self._dataset())
        self.assertEqual([image["id"] for image in filtered["images"]], [10])
        self.assertEqual({ann["image_id"] for ann in filtered["annotations"]}, {10})
        # both the victim and the occluder annotation on the paste frame are kept
        self.assertEqual(len(filtered["annotations"]), 2)
        # only the video that still has a kept frame remains
        self.assertEqual([video["id"] for video in filtered["videos"]], [1])

    def test_does_not_mutate_input(self) -> None:
        dataset = self._dataset()
        _filter_paste_frames(dataset)
        self.assertEqual(len(dataset["images"]), 3)


if __name__ == "__main__":
    unittest.main()
