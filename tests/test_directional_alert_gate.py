"""Pure tests for Pi GPIO input selection across skipped camera captures."""

from __future__ import annotations

import unittest

from helmetai_fcw.models import AlertLevel, RiskAssessment, RiskDirection, SystemStatus
from helmetai_fcw.pi_runtime import DirectionalAssessmentCache


def assessment(
    direction: RiskDirection,
    level: AlertLevel,
    timestamp_s: float,
    *,
    status: SystemStatus = SystemStatus.READY,
) -> RiskAssessment:
    return RiskAssessment(
        timestamp_s=timestamp_s,
        object_id=f"{direction.value}-target",
        level=level,
        system_status=status,
        direction=direction,
    )


class DirectionalAssessmentCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cache = DirectionalAssessmentCache(freshness_timeout_s=0.45)
        self.cadence = {
            RiskDirection.FRONT: 1,
            RiskDirection.REAR: 3,
        }

    def test_fresh_rear_warning_bridges_a_scheduled_capture_gap(self) -> None:
        rear = assessment(RiskDirection.REAR, AlertLevel.CRITICAL, timestamp_s=10.0)
        self.cache.update([rear])

        inputs = self.cache.evaluate(
            now_s=10.20,
            effective_fps=15.0,
            minimum_risk_fps=5.0,
            cadence_by_direction=self.cadence,
        )

        self.assertEqual(len(inputs), 1)
        self.assertTrue(inputs[0].eligible)
        self.assertEqual(inputs[0].assessment.level, AlertLevel.CRITICAL)
        self.assertEqual(inputs[0].estimated_risk_fps, 5.0)

    def test_rear_warning_is_blocked_when_global_fps_cannot_support_its_cadence(self) -> None:
        front = assessment(RiskDirection.FRONT, AlertLevel.WARNING, timestamp_s=10.0)
        rear = assessment(RiskDirection.REAR, AlertLevel.CRITICAL, timestamp_s=10.0)
        self.cache.update([front, rear])

        inputs = self.cache.evaluate(
            now_s=10.20,
            effective_fps=5.0,
            minimum_risk_fps=5.0,
            cadence_by_direction=self.cadence,
        )
        by_direction = {item.assessment.direction: item for item in inputs}

        self.assertTrue(by_direction[RiskDirection.FRONT].eligible)
        self.assertFalse(by_direction[RiskDirection.REAR].eligible)
        self.assertEqual(by_direction[RiskDirection.REAR].reason, "directional_risk_fps_below_minimum")
        self.assertAlmostEqual(by_direction[RiskDirection.REAR].estimated_risk_fps, 5.0 / 3.0)

    def test_stale_rear_warning_is_removed_and_cannot_reappear(self) -> None:
        self.cache.update([assessment(RiskDirection.REAR, AlertLevel.CRITICAL, timestamp_s=10.0)])

        expired = self.cache.evaluate(
            now_s=10.46,
            effective_fps=15.0,
            minimum_risk_fps=5.0,
            cadence_by_direction=self.cadence,
        )
        self.assertEqual(len(expired), 1)
        self.assertFalse(expired[0].eligible)
        self.assertEqual(expired[0].reason, "assessment_stale")

        self.assertEqual(
            self.cache.evaluate(
                now_s=10.47,
                effective_fps=15.0,
                minimum_risk_fps=5.0,
                cadence_by_direction=self.cadence,
            ),
            (),
        )

    def test_newer_lower_assessment_replaces_a_cached_high_warning(self) -> None:
        self.cache.update([assessment(RiskDirection.REAR, AlertLevel.CRITICAL, timestamp_s=10.0)])
        self.cache.update([assessment(RiskDirection.REAR, AlertLevel.NONE, timestamp_s=10.20)])

        inputs = self.cache.evaluate(
            now_s=10.25,
            effective_fps=15.0,
            minimum_risk_fps=5.0,
            cadence_by_direction=self.cadence,
        )

        self.assertEqual(len(inputs), 1)
        self.assertTrue(inputs[0].eligible)
        self.assertEqual(inputs[0].assessment.level, AlertLevel.NONE)

    def test_non_ready_assessment_never_reaches_the_gpio_selector(self) -> None:
        self.cache.update(
            [
                assessment(
                    RiskDirection.REAR,
                    AlertLevel.CRITICAL,
                    timestamp_s=10.0,
                    status=SystemStatus.DEGRADED,
                )
            ]
        )

        inputs = self.cache.evaluate(
            now_s=10.10,
            effective_fps=15.0,
            minimum_risk_fps=5.0,
            cadence_by_direction=self.cadence,
        )

        self.assertFalse(inputs[0].eligible)
        self.assertEqual(inputs[0].reason, "assessment_system_not_ready")

    def test_measured_direction_rate_overrides_optimistic_scheduled_rate(self):
        self.cache.update([assessment(RiskDirection.REAR, AlertLevel.CRITICAL, 10.0)])
        inputs = self.cache.evaluate(
            now_s=10.1, effective_fps=100, minimum_risk_fps=5,
            cadence_by_direction=self.cadence,
            measured_fps_by_direction={RiskDirection.REAR: 2.0},
        )
        self.assertFalse(inputs[0].eligible)
        self.assertEqual(inputs[0].estimated_risk_fps, 2.0)

    def test_missing_and_nonfinite_measured_rate_fail_closed(self):
        for rates in ({}, {RiskDirection.FRONT: float("nan")}, {RiskDirection.FRONT: -1.0}):
            with self.subTest(rates=rates):
                self.cache.update([assessment(RiskDirection.FRONT, AlertLevel.CRITICAL, 10.0)])
                inputs = self.cache.evaluate(
                    now_s=10.1, effective_fps=100, minimum_risk_fps=5,
                    cadence_by_direction=self.cadence, measured_fps_by_direction=rates,
                )
                self.assertFalse(inputs[0].eligible)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
