import unittest

import numpy as np

from label.compute import label_synthetic_occluder
from synth.tracklet_compositor import TrackletLayer, composite_tracklet_layers
from synth.tracklet_pipeline import map_source_bbox


class TrackletCompositorTest(unittest.TestCase):
    @staticmethod
    def _patch(size: int, color: tuple[int, int, int]) -> np.ndarray:
        patch = np.zeros((size, size, 4), dtype=np.uint8)
        patch[..., :3] = color
        patch[..., 3] = 255
        return patch

    def test_large_first_and_occluder_gt(self) -> None:
        background = np.zeros((40, 40, 3), dtype=np.uint8)
        layers = [
            TrackletLayer(10, self._patch(20, (255, 0, 0)), (5, 5, 20, 20)),
            TrackletLayer(11, self._patch(10, (0, 255, 0)), (10, 10, 10, 10)),
        ]
        _, rendered = composite_tracklet_layers(background, layers, blend_method="none")
        self.assertEqual([item.track_id for item in rendered], [10, 11])
        self.assertEqual(rendered[0].occluder_ids, (11,))
        label = label_synthetic_occluder(
            visible_mask=rendered[0].visible_mask,
            amodal_mask=rendered[0].amodal_mask,
            occluder_ids=rendered[0].occluder_ids,
            provenance={},
        )
        self.assertAlmostEqual(label["occlusion_ratio"], 0.25)
        self.assertEqual(label["occlusion_level"], 1)

    def test_source_coordinates_are_normalized(self) -> None:
        frame = {"crop_bbox_xywh": [960, 540, 192, 108], "source_image_size": [1920, 1080]}
        self.assertEqual(map_source_bbox(frame, (1240, 360)), (620.0, 180.0, 64.0, 36.0))

    def test_kitti_scale_preserves_source_bottom_center(self) -> None:
        frame = {"crop_bbox_xywh": [960, 540, 192, 108], "source_image_size": [1920, 1080]}
        mapped = map_source_bbox(
            frame,
            (1240, 360),
            target_height_fraction=0.25,
            source_reference_height=108.0,
            height_fraction_range=(0.1, 0.4),
        )
        self.assertEqual(mapped, (602.0, 126.0, 160.0, 90.0))


if __name__ == "__main__":
    unittest.main()
