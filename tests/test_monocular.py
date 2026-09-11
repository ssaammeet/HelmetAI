import unittest

from helmetai_fcw.detection import Detection
from helmetai_fcw.monocular import MonocularCalibration, MonocularRangeEstimator, focal_length_from_reference


class MonocularRangeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.estimator = MonocularRangeEstimator(
            MonocularCalibration(
                focal_length_px=800.0,
                principal_point_x_px=400.0,
                image_width_px=800,
                image_height_px=480,
                calibrated=True,
            )
        )

    def test_car_range_comes_from_known_width(self) -> None:
        observation = self.estimator.estimate(
            1.0,
            Detection("car-1", "car", 0.9, 328.0, 120.0, 144.0, 80.0),
        )
        self.assertAlmostEqual(observation.longitudinal_m, 10.0, places=5)
        self.assertAlmostEqual(observation.lateral_m, 0.0, places=5)
        self.assertGreater(observation.range_sigma_m, 0.0)

    def test_reference_formula(self) -> None:
        self.assertAlmostEqual(focal_length_from_reference(10.0, 1.8, 144.0), 800.0)

    def test_person_range_is_supported_with_conservative_uncertainty(self) -> None:
        observation = self.estimator.estimate(1.0, Detection("person-1", "person", 0.9, 382, 0, 36, 100))
        self.assertAlmostEqual(observation.longitudinal_m, 10.0, places=5)
        self.assertGreater(observation.range_sigma_m, 2.0)

    def test_scene_class_without_geometry_model_fails_explicitly(self) -> None:
        with self.assertRaises(ValueError):
            self.estimator.estimate(1.0, Detection("sign-1", "stop_sign", 0.9, 0, 0, 30, 80))

    def test_scene_class_without_geometry_model_is_not_metric_capable(self) -> None:
        self.assertFalse(self.estimator.supports_range(Detection("sign-1", "stop_sign", 0.9, 0, 0, 30, 80)))

    def test_zero_width_box_is_not_metric_capable(self) -> None:
        self.assertFalse(self.estimator.supports_range(Detection("car-1", "car", 0.9, 0, 0, 0, 80)))

    def test_lateral_sign_maps_a_rear_camera_image_to_rider_coordinates(self) -> None:
        calibration = MonocularCalibration(
            focal_length_px=800.0,
            principal_point_x_px=400.0,
            image_width_px=800,
            image_height_px=480,
            lateral_sign=-1.0,
            calibrated=True,
        )
        observation = MonocularRangeEstimator(calibration).estimate(
            1.0,
            Detection("rear-car", "car", 0.9, 480.0, 120.0, 144.0, 80.0),
        )
        self.assertLess(observation.lateral_m, 0.0)


if __name__ == "__main__":
    unittest.main()
