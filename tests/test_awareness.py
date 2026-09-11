import unittest

from helmetai_fcw.awareness import AwarenessZone, BlindSpotMonitor, CoverageAwareRiskMap, TurnIntent
from helmetai_fcw.models import AlertLevel, ObservedTarget, RiskAssessment, RiskDirection, SystemStatus


def _target(object_id: str, lateral_m: float, distance_m: float = 5.0) -> ObservedTarget:
    return ObservedTarget(
        timestamp_s=1.0,
        object_id=object_id,
        object_class="car",
        longitudinal_m=distance_m,
        lateral_m=lateral_m,
        range_sigma_m=0.5,
        confidence=0.9,
    )


def _assessment(object_id: str, level: AlertLevel, direction: RiskDirection) -> RiskAssessment:
    return RiskAssessment(
        timestamp_s=1.0,
        object_id=object_id,
        level=level,
        system_status=SystemStatus.READY,
        direction=direction,
    )


class AwarenessTests(unittest.TestCase):
    def test_map_keeps_side_camera_gaps_explicit(self) -> None:
        front = _target("front-1", -1.0)
        rear = _target("rear-1", 1.0)
        snapshot = CoverageAwareRiskMap().build(
            [(front, _assessment("front-1", AlertLevel.WARNING, RiskDirection.FRONT))],
            [(rear, _assessment("rear-1", AlertLevel.ADVISORY, RiskDirection.REAR))],
        )
        self.assertEqual(snapshot.items[0].zone, AwarenessZone.FRONT_LEFT)
        self.assertEqual(snapshot.items[1].zone, AwarenessZone.REAR_RIGHT)
        self.assertIn(AwarenessZone.LEFT_SIDE_UNOBSERVED, snapshot.unobserved_zones)
        self.assertEqual(snapshot.highest_level_by_zone()["front_left"], "WARNING")

    def test_blind_spot_only_activates_for_matching_turn_intent(self) -> None:
        target = _target("rear-1", -1.0)
        monitor = BlindSpotMonitor()
        inactive = monitor.assess(target, TurnIntent.RIGHT)
        active = monitor.assess(target, TurnIntent.LEFT)
        self.assertTrue(inactive.nearby_lateral_target)
        self.assertFalse(inactive.warning_active)
        self.assertTrue(active.warning_active)

    def test_blind_spot_rejects_centreline_target(self) -> None:
        result = BlindSpotMonitor().assess(_target("rear-1", 0.1), TurnIntent.RIGHT)
        self.assertFalse(result.nearby_lateral_target)
        self.assertIn("target_not_in_lateral_band", result.reasons)


if __name__ == "__main__":
    unittest.main()
