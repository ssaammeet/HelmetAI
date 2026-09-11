import unittest

from helmetai_fcw.performance import DirectionalPerformanceMonitor, EdgePerformanceMonitor


class EdgePerformanceMonitorTests(unittest.TestCase):
    def test_reports_fps_and_budget_status(self) -> None:
        monitor = EdgePerformanceMonitor(window_frames=5, latency_budget_ms=30.0)
        snapshot = monitor.record(5.0, 15.0, 5.0)
        self.assertAlmostEqual(snapshot.end_to_end_ms, 25.0)
        self.assertGreater(snapshot.effective_fps, 39.0)
        self.assertTrue(snapshot.meets_latency_budget)
        self.assertFalse(snapshot.warmup_complete)
        self.assertFalse(snapshot.risk_alerts_permitted)

    def test_reports_when_rolling_latency_exceeds_budget(self) -> None:
        monitor = EdgePerformanceMonitor(window_frames=5, latency_budget_ms=30.0)
        for _ in range(5):
            snapshot = monitor.record(5.0, 30.0, 5.0)
        self.assertFalse(snapshot.meets_latency_budget)
        self.assertEqual(snapshot.p95_end_to_end_ms, 40.0)
        self.assertTrue(snapshot.warmup_complete)
        self.assertFalse(snapshot.risk_alerts_permitted)

    def test_permits_risk_alerts_only_after_warmup_and_healthy_performance(self) -> None:
        monitor = EdgePerformanceMonitor(
            window_frames=5,
            latency_budget_ms=30.0,
            minimum_risk_fps=20.0,
            warmup_frames=3,
        )
        for _ in range(2):
            snapshot = monitor.record(5.0, 15.0, 5.0)
        self.assertFalse(snapshot.warmup_complete)
        self.assertTrue(snapshot.meets_latency_budget)
        self.assertTrue(snapshot.meets_minimum_risk_fps)
        self.assertFalse(snapshot.risk_alerts_permitted)

        snapshot = monitor.record(5.0, 15.0, 5.0)
        self.assertTrue(snapshot.warmup_complete)
        self.assertTrue(snapshot.risk_alerts_permitted)
        self.assertEqual(snapshot.as_dict()["risk_alerts_permitted"], True)

    def test_blocks_risk_alerts_when_effective_fps_is_too_low(self) -> None:
        monitor = EdgePerformanceMonitor(
            window_frames=5,
            latency_budget_ms=300.0,
            minimum_risk_fps=5.0,
            warmup_frames=5,
        )
        for _ in range(5):
            snapshot = monitor.record(20.0, 200.0, 30.0)

        self.assertTrue(snapshot.warmup_complete)
        self.assertTrue(snapshot.meets_latency_budget)
        self.assertLess(snapshot.effective_fps, 5.0)
        self.assertFalse(snapshot.meets_minimum_risk_fps)
        self.assertFalse(snapshot.risk_alerts_permitted)

    def test_rejects_invalid_health_gate_configuration(self) -> None:
        with self.assertRaises(ValueError):
            EdgePerformanceMonitor(minimum_risk_fps=0.0)
        with self.assertRaises(ValueError):
            EdgePerformanceMonitor(warmup_frames=0)

    def test_wall_intervals_include_output_stalls_without_inflating_work(self) -> None:
        monitor = EdgePerformanceMonitor(window_frames=5, minimum_risk_fps=5, warmup_frames=1)
        for _ in range(5):
            snapshot = monitor.record(5, 40, 5, wall_interval_ms=300)
        self.assertEqual(snapshot.end_to_end_ms, 50)
        self.assertEqual(snapshot.average_cycle_interval_ms, 300)
        self.assertAlmostEqual(snapshot.effective_fps, 1000 / 300)
        self.assertTrue(snapshot.meets_latency_budget)
        self.assertFalse(snapshot.risk_alerts_permitted)

    def test_nonfinite_or_impossible_timing_is_rejected(self) -> None:
        monitor = EdgePerformanceMonitor()
        for value in (float("nan"), float("inf"), -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                monitor.record(1, value, 1)
            with self.subTest(wall=value), self.assertRaises(ValueError):
                monitor.record(1, 1, 1, wall_interval_ms=value)
        with self.assertRaises(ValueError):
            monitor.record(5, 40, 5, wall_interval_ms=10)


class DirectionalPerformanceTests(unittest.TestCase):
    def record(self, monitor, timestamp):
        monitor.record(timestamp, capture_ms=5, inference_ms=30, processing_ms=36, detections=0)

    def test_empty_successes_count_but_first_cannot_establish_fps(self):
        monitor = DirectionalPerformanceMonitor()
        self.assertIsNone(monitor.snapshot(0)["last_update_age_s"])
        self.record(monitor, 1)
        self.assertEqual(monitor.snapshot(1)["processed_fps"], 0)
        self.record(monitor, 1.2)
        self.assertAlmostEqual(monitor.snapshot(1.2)["processed_fps"], 5)
        self.assertEqual(monitor.snapshot(1.2)["detections"], 0)
        self.assertEqual(monitor.snapshot(1.2)["processed_frames"], 2)

    def test_idle_time_decays_directional_fps(self):
        monitor = DirectionalPerformanceMonitor()
        self.record(monitor, 1)
        self.record(monitor, 1.1)
        self.assertAlmostEqual(monitor.snapshot(2)["processed_fps"], 1)
        self.assertAlmostEqual(monitor.snapshot(2)["last_update_age_s"], 0.9)

    def test_rolling_window_and_bad_clock(self):
        monitor = DirectionalPerformanceMonitor(window_frames=2)
        for timestamp in (1, 3, 3.1):
            self.record(monitor, timestamp)
        self.assertAlmostEqual(monitor.snapshot(3.1)["processed_fps"], 10)
        self.assertEqual(monitor.snapshot(2)["processed_fps"], 0)
        with self.assertRaises(ValueError):
            self.record(monitor, 2)
        with self.assertRaises(ValueError):
            self.record(monitor, float("nan"))

    def test_disabled_camera_never_has_processing_success(self):
        monitor = DirectionalPerformanceMonitor(enabled=False)
        self.assertFalse(monitor.snapshot(1)["enabled"])
        self.assertEqual(monitor.snapshot(1)["processed_frames"], 0)
        with self.assertRaises(ValueError):
            self.record(monitor, 1)


if __name__ == "__main__":
    unittest.main()
