from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ezw_compression.metrics import compression_ratio, mean_squared_error, peak_signal_to_noise_ratio


class MetricsTestCase(unittest.TestCase):
    def test_mse_is_zero_for_identical_images(self) -> None:
        image = np.array([[0, 64], [128, 255]], dtype=np.float64)
        self.assertEqual(mean_squared_error(image, image.copy()), 0.0)

    def test_psnr_is_infinite_for_identical_images(self) -> None:
        image = np.ones((4, 4), dtype=np.float64) * 120.0
        self.assertEqual(peak_signal_to_noise_ratio(image, image.copy()), float("inf"))

    def test_compression_ratio(self) -> None:
        self.assertAlmostEqual(compression_ratio(800.0, 200.0), 4.0)


if __name__ == "__main__":
    unittest.main()
