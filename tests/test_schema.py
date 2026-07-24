import unittest

import numpy as np

from common.schema import (
    compute_ratio,
    decode_uncompressed_rle,
    encode_binary_mask,
    occlusion_level,
)


class SchemaTest(unittest.TestCase):
    def test_rle_round_trip(self) -> None:
        mask = np.zeros((7, 9), dtype=np.uint8)
        mask[1:6, 2:8] = 1
        mask[3, 4:6] = 0
        decoded = decode_uncompressed_rle(encode_binary_mask(mask))
        np.testing.assert_array_equal(decoded, mask)

    def test_ratio_and_level(self) -> None:
        amodal = np.ones((10, 10), dtype=np.uint8)
        visible = amodal.copy()
        visible[:4] = 0
        self.assertAlmostEqual(compute_ratio(visible, amodal), 0.4)
        self.assertEqual(occlusion_level(0.4), 2)

    def test_level_bands_aligned_with_peak_bands(self) -> None:
        # level 0 below the mild floor, then mild / moderate / heavy.
        self.assertEqual(occlusion_level(0.0), 0)
        self.assertEqual(occlusion_level(0.19), 0)
        self.assertEqual(occlusion_level(0.20), 1)
        self.assertEqual(occlusion_level(0.34), 1)
        self.assertEqual(occlusion_level(0.35), 2)
        self.assertEqual(occlusion_level(0.64), 2)
        self.assertEqual(occlusion_level(0.65), 3)
        self.assertEqual(occlusion_level(0.80), 3)


if __name__ == "__main__":
    unittest.main()

