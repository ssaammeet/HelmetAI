import unittest

from helmetai_fcw.models import AlertLevel, SystemStatus, TrackEstimate
from helmetai_fcw.risk_engine import ForwardCollisionRiskEngine


def track(**overrides) -> TrackEstimate:
    values = dict(
        timestamp_s=1.0,
        object_id="lead",
        object_class="car",
        longitudinal_m=25.0,
        lateral_m=0.10,
        longitudinal_velocity_mps=-10.0,
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


class ForwardCollisionRiskEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ForwardCollisionRiskEngine()

    def test_closing_vehicle_in_path_warns(self) -> None:
        assessment = self.engine.assess(track())
        self.assertEqual(assessment.system_status, SystemStatus.READY)
        self.assertTrue(assessment.in_path)
        self.assertGreaterEqual(assessment.level, AlertLevel.WARNING)
        self.assertLess(assessment.conservative_ttc_s, 3.0)

    def test_static_distance_does_not_become_a_threat_from_uncertainty(self) -> None:
        assessment = self.engine.assess(track(longitudinal_velocity_mps=0.0, velocity_sigma_mps=2.0))
        self.assertEqual(assessment.level, AlertLevel.NONE)
        self.assertIn("no_material_closing_speed", assessment.reasons)

    def test_vehicle_outside_forward_corridor_does_not_warn(self) -> None:
        assessment = self.engine.assess(track(lateral_m=5.0))
        self.assertEqual(assessment.level, AlertLevel.NONE)
        self.assertIn("target_outside_forward_corridor", assessment.reasons)

    def test_invalid_calibration_degrades_system(self) -> None:
        assessment = self.engine.assess(track(calibration_valid=False))
        self.assertEqual(assessment.system_status, SystemStatus.DEGRADED)
        self.assertEqual(assessment.level, AlertLevel.NONE)

    def test_very_short_ttc_is_critical(self) -> None:
        assessment = self.engine.assess(track(longitudinal_m=7.0, longitudinal_velocity_mps=-12.0))
        self.assertEqual(assessment.level, AlertLevel.CRITICAL)

    def test_high_uncertainty_pedestrian_does_not_create_a_false_ttc_alarm(self) -> None:
        assessment = self.engine.assess(
            track(
                object_class="person",
                longitudinal_m=19.0,
                longitudinal_velocity_mps=-2.0,
                range_sigma_m=8.7,
                velocity_sigma_mps=0.87,
            )
        )
        self.assertEqual(assessment.system_status, SystemStatus.DEGRADED)
        self.assertEqual(assessment.level, AlertLevel.NONE)
        self.assertIn("range_uncertainty_exceeds_policy", assessment.reasons)

    def test_closing_trend_requires_elapsed_time_not_only_fast_frames(self) -> None:
        too_early = self.engine.assess(
            track(closing_evidence_frames=3, closing_evidence_duration_s=0.20)
        )
        confirmed = self.engine.assess(
            track(closing_evidence_frames=3, closing_evidence_duration_s=0.40)
        )
        self.assertEqual(too_early.level, AlertLevel.NONE)
        self.assertIn("closing_trend_duration_not_confirmed", too_early.reasons)
        self.assertGreaterEqual(confirmed.level, AlertLevel.WARNING)

    def test_missing_temporal_evidence_fails_closed(self) -> None:
        assessment = self.engine.assess(
            track(closing_evidence_frames=None, closing_evidence_duration_s=None)
        )
        self.assertEqual(assessment.system_status, SystemStatus.DEGRADED)
        self.assertEqual(assessment.level, AlertLevel.NONE)
        self.assertIn("closing_trend_evidence_unavailable", assessment.reasons)


if __name__ == "__main__":
    unittest.main()
