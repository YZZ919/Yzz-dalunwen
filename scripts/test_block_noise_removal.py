#!/usr/bin/env python3
import unittest
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from block_noise_removal import block_noise_ratios, estimate_noise_level, select_features


class BlockNoiseRemovalTests(unittest.TestCase):
    def test_constant_image_has_only_boundary_response(self):
        image = np.full((80, 94), 100, dtype=np.uint8)
        # The cited MATLAB implementation uses zero-padded conv2, so even a
        # constant image retains a small boundary response after cropping.
        self.assertLess(estimate_noise_level(image), 0.1)

    def test_added_noise_increases_estimate(self):
        rng = np.random.RandomState(7)
        smooth = np.full((80, 94), 100, dtype=np.uint8)
        noisy = np.clip(smooth.astype(np.int16) + rng.normal(0, 12, smooth.shape), 0, 255).astype(np.uint8)
        self.assertGreater(estimate_noise_level(noisy), estimate_noise_level(smooth))

    def test_constraint_protection_keeps_minimum(self):
        ratios = np.ones((2, 2), dtype=np.float64)
        ratios[0, 0] = 20.0
        points = np.array([[5, 5], [6, 6], [7, 7], [8, 8], [15, 15]], dtype=np.float64)
        keep, removed = select_features(points, ratios, (20, 20), threshold=10.8, min_keep=3)
        self.assertEqual(int(keep.sum()), 3)
        self.assertEqual(len(removed), 2)
        self.assertTrue(keep[-1])

    def test_noisy_block_is_identified(self):
        rng = np.random.RandomState(11)
        raw = np.full((80, 80), 100, dtype=np.uint8)
        enhanced = raw.copy()
        enhanced[:40, :40] = np.clip(
            100 + rng.normal(0, 25, (40, 40)), 0, 255
        ).astype(np.uint8)
        ratios, _, _ = block_noise_ratios(raw, enhanced, rows=2, cols=2)
        self.assertGreater(ratios[0, 0], 10.8)
        self.assertEqual(float(ratios[1, 1]), 1.0)


if __name__ == "__main__":
    unittest.main()
