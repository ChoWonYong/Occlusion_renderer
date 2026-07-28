import unittest

from data.build_phase1 import _filter_paste_frames, _replace_with_paste_frames, paste_components


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


def _occluder(annotation_id: int, image_id: int, track_id: int) -> dict:
    return {
        "id": annotation_id,
        "image_id": image_id,
        "video_id": 1,
        "track_id": track_id,
        "synthetic_occluder": True,
    }


class PasteComponentsTest(unittest.TestCase):
    def test_tracks_sharing_a_frame_merge_into_one_component(self) -> None:
        synthetic = {
            "annotations": [
                _occluder(1, 100, 900001),
                _occluder(2, 101, 900001),
                _occluder(3, 101, 900002),  # overlaps track 900001 on frame 101
                _occluder(4, 102, 900002),
                _occluder(5, 200, 900003),  # disjoint in time
            ]
        }
        components = paste_components(synthetic)
        self.assertEqual(len(components), 2)
        merged = next(c for c in components if len(c["tracks"]) == 2)
        self.assertEqual(merged["image_ids"], [100, 101, 102])
        lone = next(c for c in components if len(c["tracks"]) == 1)
        self.assertEqual(lone["image_ids"], [200])

    def test_every_frame_belongs_to_exactly_one_component(self) -> None:
        synthetic = {
            "annotations": [
                _occluder(1, 100, 900001),
                _occluder(2, 100, 900002),
                _occluder(3, 101, 900002),
                _occluder(4, 300, 900003),
            ]
        }
        components = paste_components(synthetic)
        seen: list[int] = []
        for component in components:
            seen.extend(component["image_ids"])
        self.assertEqual(len(seen), len(set(seen)))

    def test_no_pastes_means_no_components(self) -> None:
        self.assertEqual(paste_components({"annotations": []}), [])


class ReplaceWithPasteFramesTest(unittest.TestCase):
    def _train(self) -> dict:
        return {
            "categories": [{"id": 1, "name": "car"}],
            "videos": [{"id": 1, "name": "v1"}],
            "images": [
                {"id": 1, "video_id": 1, "file_name": "images/original_train/0.jpg"},
                {"id": 2, "video_id": 1, "file_name": "images/original_train/1.jpg"},
                {"id": 3, "video_id": 1, "file_name": "images/original_train/2.jpg"},
            ],
            "annotations": [
                {"id": 1, "image_id": 1, "video_id": 1, "track_id": 5},
                {"id": 2, "image_id": 2, "video_id": 1, "track_id": 5},
                {"id": 3, "image_id": 3, "video_id": 1, "track_id": 5},
            ],
        }

    def _synthetic(self) -> dict:
        return {
            "categories": [{"id": 1, "name": "car"}],
            "videos": [{"id": 1, "name": "v1_event_synth"}],
            "images": [
                {"id": 50, "video_id": 1, "file_name": "images/synthetic/v1/0.jpg",
                 "source": {"image_id": 1}},
                {"id": 51, "video_id": 1, "file_name": "images/synthetic/v1/1.jpg",
                 "source": {"image_id": 2}},
            ],
            "annotations": [
                {"id": 10, "image_id": 50, "video_id": 1, "track_id": 5},
                _occluder(11, 50, 900001),
                {"id": 12, "image_id": 51, "video_id": 1, "track_id": 5},
                _occluder(13, 51, 900001),
            ],
        }

    def test_image_count_is_preserved(self) -> None:
        result, stats = _replace_with_paste_frames(self._train(), self._synthetic(), 1.0, 0)
        self.assertEqual(len(result["images"]), 3)
        self.assertEqual(stats["frames_replaced"], 2)
        self.assertEqual(stats["components"], 1)
        self.assertEqual(stats["components_selected"], 1)

    def test_replaced_frames_use_the_rendered_file_and_its_annotations(self) -> None:
        result, _ = _replace_with_paste_frames(self._train(), self._synthetic(), 1.0, 0)
        by_id = {image["id"]: image for image in result["images"]}
        self.assertEqual(by_id[1]["file_name"], "images/synthetic/v1/0.jpg")
        self.assertTrue(by_id[1]["pasted"])
        self.assertEqual(by_id[3]["file_name"], "images/original_train/2.jpg")
        self.assertNotIn("pasted", by_id[3])
        # the occluder's own annotation comes along with the frame
        on_first = [a for a in result["annotations"] if a["image_id"] == 1]
        self.assertEqual(len(on_first), 2)
        self.assertTrue(any(a.get("synthetic_occluder") for a in on_first))

    def test_probability_zero_keeps_every_original(self) -> None:
        result, stats = _replace_with_paste_frames(self._train(), self._synthetic(), 0.0, 0)
        self.assertEqual(stats["frames_replaced"], 0)
        self.assertEqual(len(result["images"]), 3)
        self.assertTrue(all("pasted" not in image for image in result["images"]))
        self.assertEqual(len(result["annotations"]), 3)

    def test_annotation_ids_are_unique(self) -> None:
        result, _ = _replace_with_paste_frames(self._train(), self._synthetic(), 1.0, 0)
        ids = [a["id"] for a in result["annotations"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_missing_source_link_is_an_error(self) -> None:
        synthetic = self._synthetic()
        del synthetic["images"][0]["source"]
        with self.assertRaises(KeyError):
            _replace_with_paste_frames(self._train(), synthetic, 1.0, 0)


if __name__ == "__main__":
    unittest.main()
