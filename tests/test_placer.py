import unittest

import numpy as np

from synth.placer import find_placement, place_mask


class PlacerTest(unittest.TestCase):
    def test_finds_requested_bbox_proxy_overlap(self) -> None:
        donor_mask = np.ones((30, 30), dtype=np.uint8)
        placement = find_placement(
            donor_mask,
            target_bbox=[60, 40, 80, 80],
            image_shape=(180, 240),
            rho_target=0.3,
            rng=np.random.default_rng(7),
            trials=600,
            tolerance=0.08,
        )
        self.assertIsNotNone(placement)
        assert placement is not None
        self.assertLessEqual(abs(placement.rho_actual - 0.3), 0.08)
        full_mask = place_mask(donor_mask, placement, (180, 240))
        self.assertEqual(full_mask.shape, (180, 240))
        self.assertGreater(int(full_mask.sum()), 0)


if __name__ == "__main__":
    unittest.main()

