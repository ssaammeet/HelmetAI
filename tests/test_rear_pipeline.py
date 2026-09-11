import unittest

from helmetai_fcw.models import AlertLevel, RiskDirection
from helmetai_fcw.simulator import RearClosingScenario, rear_closing_vehicle_observations, run_rear_closing_demo


class RearCollisionPipelineTests(unittest.TestCase):
    def test_rear_closing_demo_escalates_to_critical(self) -> None:
        assessments = run_rear_closing_demo(RearClosingScenario(duration_s=4.5))
        levels = {item.level for item in assessments}
        self.assertIn(AlertLevel.ADVISORY, levels)
        self.assertIn(AlertLevel.CRITICAL, levels)
        self.assertTrue(all(item.direction == RiskDirection.REAR for item in assessments))

    def test_rear_observations_reduce_distance(self) -> None:
        observations = rear_closing_vehicle_observations(RearClosingScenario(duration_s=1.0))
        self.assertGreater(observations[0].longitudinal_m, observations[-1].longitudinal_m)


if __name__ == "__main__":
    unittest.main()
