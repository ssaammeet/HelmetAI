import unittest

from helmetai_fcw.models import AlertLevel, RiskDirection
from helmetai_fcw.demo_dashboard import build_demo_timelines


class DemoDashboardTimelineTests(unittest.TestCase):
    def test_dashboard_uses_both_real_risk_pipelines(self) -> None:
        front, rear = build_demo_timelines()
        self.assertGreater(len(front), 10)
        self.assertGreater(len(rear), 10)
        self.assertTrue(any(item.assessment.level == AlertLevel.CRITICAL for item in front))
        self.assertTrue(any(item.assessment.level == AlertLevel.CRITICAL for item in rear))
        self.assertTrue(all(item.assessment.direction == RiskDirection.FRONT for item in front))
        self.assertTrue(all(item.assessment.direction == RiskDirection.REAR for item in rear))


if __name__ == "__main__":
    unittest.main()
