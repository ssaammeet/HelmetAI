import unittest

from helmetai_fcw.models import AlertLevel
from helmetai_fcw.pipeline import ForwardCollisionPipeline
from helmetai_fcw.simulator import ClosingScenario, closing_vehicle_observations, run_closing_demo


class ForwardCollisionPipelineTests(unittest.TestCase):
    def test_filter_learns_a_closing_speed(self) -> None:
        pipeline = ForwardCollisionPipeline()
        assessments = [pipeline.process(item) for item in closing_vehicle_observations(ClosingScenario(duration_s=2.0))]
        final = assessments[-1]
        self.assertIsNotNone(final.closing_speed_mps)
        self.assertGreater(final.closing_speed_mps, 3.0)

    def test_closing_demo_escalates_to_critical(self) -> None:
        assessments = run_closing_demo(ClosingScenario(duration_s=5.0))
        levels = {item.level for item in assessments}
        # The production filter requires five consistent closing updates.
        # This fast synthetic approach can therefore enter at WARNING rather
        # than emitting an early advisory from a noisy initial trend.
        self.assertIn(AlertLevel.WARNING, levels)
        self.assertIn(AlertLevel.CRITICAL, levels)

    def test_pipeline_holds_a_higher_alert_briefly(self) -> None:
        pipeline = ForwardCollisionPipeline(alert_hold_s=1.0)
        observations = closing_vehicle_observations(ClosingScenario(duration_s=0.8))
        assessments = [pipeline.process(item) for item in observations]
        # The synthetic stream momentarily dips after a warning; anti-flicker
        # logic must retain that stronger alert for its hold interval.
        self.assertEqual(assessments[5].level, AlertLevel.WARNING)
        self.assertEqual(assessments[6].level, AlertLevel.WARNING)


if __name__ == "__main__":
    unittest.main()
