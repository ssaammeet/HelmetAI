import unittest

from helmetai_fcw.models import AlertLevel, RiskDirection, SystemStatus, TrackEstimate
from helmetai_fcw.rear_risk_engine import RearCollisionRiskEngine


def rear_track(**overrides) -> TrackEstimate:
    values = dict(
        timestamp_s=1.0,
        object_id="rear-car",
        object_class="car",
        longitudinal_m=20.0,
        lateral_m=-0.10,
        longitudinal_velocity_mps=-8.0,
        lateral_velocity_mps=0.0,
        range_sigma_m=0.25,
        velocity_sigma_mps=0.30,
        confidence=0.90,
        calibration_valid=True,
        age_frames=6,
        closing_evidence_frames=3,
        closing_evidence_duration_s=0.40,
    )
    values.update(overrides)
    return TrackEstimate(**values)


class RearCollisionRiskEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RearCollisionRiskEngine()

    def test_closing_vehicle_behind_warns(self) -> None:
        assessment = self.engine.assess(rear_track())
        self.assertEqual(assessment.system_status, SystemStatus.READY)
        self.assertEqual(assessment.direction, RiskDirection.REAR)
        self.assertTrue(assessment.in_path)
        self.assertGreaterEqual(assessment.level, AlertLevel.WARNING)
        self.assertIn("rear_positive_closing_speed", assessment.reasons)

    def test_static_rear_vehicle_does_not_become_a_threat(self) -> None:
        assessment = self.engine.assess(rear_track(longitudinal_velocity_mps=0.0, velocity_sigma_mps=2.0))
        self.assertEqual(assessment.level, AlertLevel.NONE)
        self.assertIn("rear_no_material_closing_speed", assessment.reasons)

    def test_vehicle_outside_rear_corridor_does_not_warn(self) -> None:
        assessment = self.engine.assess(rear_track(lateral_m=5.0))
        self.assertEqual(assessment.level, AlertLevel.NONE)
        self.assertIn("rear_target_outside_corridor", assessment.reasons)

    def test_invalid_rear_calibration_degrades_system(self) -> None:
        assessment = self.engine.assess(rear_track(calibration_valid=False))
        self.assertEqual(assessment.system_status, SystemStatus.DEGRADED)
        self.assertEqual(assessment.level, AlertLevel.NONE)

    def test_rear_closing_trend_requires_elapsed_time(self) -> None:
        assessment = self.engine.assess(
            rear_track(closing_evidence_frames=3, closing_evidence_duration_s=0.20)
        )
        self.assertEqual(assessment.level, AlertLevel.NONE)
        self.assertIn("rear_closing_trend_duration_not_confirmed", assessment.reasons)

    def test_missing_rear_temporal_evidence_fails_closed(self) -> None:
        assessment = self.engine.assess(
            rear_track(closing_evidence_frames=None, closing_evidence_duration_s=None)
        )
        self.assertEqual(assessment.system_status, SystemStatus.DEGRADED)
        self.assertEqual(assessment.level, AlertLevel.NONE)
        self.assertIn("rear_closing_trend_evidence_unavailable", assessment.reasons)


if __name__ == "__main__":
    unittest.main()
