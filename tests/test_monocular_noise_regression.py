"""Regression coverage for monocular-box jitter that mimics closing motion."""

import unittest
from math import pi, sin

from helmetai_fcw.models import AlertLevel, ObservedTarget
from helmetai_fcw.pipeline import ForwardCollisionPipeline


class MonocularNoiseRegressionTests(unittest.TestCase):
    def test_stationary_far_car_with_box_width_jitter_does_not_alert(self) -> None:
        """A periodic +/- 2 px box change at 40 m is not a collision."""

        focal_px, car_width_m, true_range_m = 580.0, 1.8, 40.0
        nominal_width_px = focal_px * car_width_m / true_range_m
        pipeline = ForwardCollisionPipeline()
        assessments = []
        for frame in range(180):
            measured_width_px = nominal_width_px + 2.0 * sin(2.0 * pi * frame / 9.0)
            apparent_range_m = focal_px * car_width_m / measured_width_px
            assessments.append(
                pipeline.process(
                    ObservedTarget(
                        timestamp_s=frame / 15.0,
                        object_id="stationary-car",
                        object_class="car",
                        longitudinal_m=apparent_range_m,
                        lateral_m=0.0,
                        range_sigma_m=0.8,
                        confidence=0.9,
                        calibration_valid=True,
                    )
                )
            )
        self.assertTrue(all(item.level == AlertLevel.NONE for item in assessments))


if __name__ == "__main__":
    unittest.main()
