import unittest

from helmetai_fcw.stereo import StereoCalibration


class StereoCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.calibration = StereoCalibration(focal_length_px=700.0, baseline_m=0.12)

    def test_depth_from_disparity(self) -> None:
        depth_m, sigma_m = self.calibration.depth_from_disparity(14.0, disparity_sigma_px=0.5)
        self.assertAlmostEqual(depth_m, 6.0, places=6)
        self.assertGreater(sigma_m, 0.0)

    def test_depth_uncertainty_grows_at_long_range(self) -> None:
        _, near_sigma = self.calibration.depth_from_disparity(16.8, disparity_sigma_px=0.5)
        _, far_sigma = self.calibration.depth_from_disparity(4.2, disparity_sigma_px=0.5)
        self.assertGreater(far_sigma, near_sigma)

    def test_robust_depth_rejects_invalid_samples(self) -> None:
        depth_m, sigma_m = self.calibration.robust_depth([14.0, 14.2, 0.1, 13.8])
        self.assertAlmostEqual(depth_m, 6.0, delta=0.15)
        self.assertGreater(sigma_m, 0.0)


if __name__ == "__main__":
    unittest.main()

