"""Runtime timing regressions without cameras, GPIO, OpenCV, or real sleeps."""

import json
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from helmetai_fcw import pi_runtime
from helmetai_fcw.alert_actuator import AlertOutputState
from helmetai_fcw.models import AlertLevel


class _FakeClock:
    def __init__(self):
        self.seconds = 100.0

    def now(self):
        return self.seconds

    def advance_ms(self, milliseconds):
        self.seconds += milliseconds / 1000.0


class _FakeRig:
    def __init__(self, clock, *, rear_enabled=True, fail_capture=None, rear_returns_none=False,
                 fail_close=False, close_delay_ms=0.0):
        self.clock = clock
        self.rear = object() if rear_enabled else None
        self.fail_capture = fail_capture
        self.rear_returns_none = rear_returns_none
        self.fail_close = fail_close
        self.close_delay_ms = close_delay_ms
        self.calls = {"front": 0, "rear": 0}
        self.close_calls = 0

    def _capture(self, direction, duration_ms):
        self.calls[direction] += 1
        self.clock.advance_ms(duration_ms)
        if self.fail_capture == (direction, self.calls[direction]):
            raise RuntimeError(f"injected {direction} capture failure")
        return (direction, self.calls[direction])

    def capture_front(self):
        return self._capture("front", 5.0)

    def capture_rear(self):
        frame = self._capture("rear", 10.0)
        return None if self.rear_returns_none else frame

    def close(self):
        self.close_calls += 1
        self.clock.advance_ms(self.close_delay_ms)
        self.rear = None
        if self.fail_close:
            raise RuntimeError("injected rig close failure")


class _FakeDetector:
    def __init__(self, clock, fail_inference=None):
        self.clock = clock
        self.fail_inference = fail_inference
        self.calls = {"front": 0, "rear": 0}
        self.completed_at = {"front": [], "rear": []}

    def detect(self, frame):
        direction, _ = frame
        self.calls[direction] += 1
        self.clock.advance_ms(40.0 if direction == "front" else 50.0)
        if self.fail_inference == (direction, self.calls[direction]):
            raise RuntimeError(f"injected {direction} inference failure")
        self.completed_at[direction].append(self.clock.now())
        return []


class _FakeActuator:
    def __init__(self, fail_apply_call=None):
        # A latched test state makes forgetting close observable even if the
        # first capture throws before the runtime can call apply.
        self.outputs_on = True
        self.close_calls = 0
        self.applied = []
        self.apply_calls = 0
        self.fail_apply_call = fail_apply_call

    def apply(self, requested_level, *, alerts_permitted, now_s=None):
        self.apply_calls += 1
        if self.apply_calls == self.fail_apply_call:
            self.outputs_on = True
            raise RuntimeError("injected actuator apply failure")
        applied_level = requested_level if alerts_permitted else AlertLevel.NONE
        self.outputs_on = applied_level is not AlertLevel.NONE
        result = AlertOutputState(
            requested_level=requested_level,
            applied_level=applied_level,
            alerts_permitted=alerts_permitted,
            buzzer_on=self.outputs_on,
            vibration_on=self.outputs_on,
            led_on=self.outputs_on,
        )
        self.applied.append(result)
        return result

    def close(self):
        self.close_calls += 1
        self.outputs_on = False


class _RuntimeHarness:
    def __init__(self, *, log_delay_ms=0.0, fail_capture=None, fail_inference=None,
                 rear_returns_none=False, fail_rig_close=False, close_delay_ms=0.0,
                 fail_apply_call=None, **config_overrides):
        root = Path(__file__).resolve().parents[1]
        config = pi_runtime.PiRuntimeConfig.from_json(root / "config" / "pi5_dual_camera.example.json")
        self.config = replace(config, **{"rear_capture_every_n_frames": 1, **config_overrides})
        self.clock = _FakeClock()
        self.rig = _FakeRig(
            self.clock,
            rear_enabled=self.config.rear_camera is not None,
            fail_capture=fail_capture,
            rear_returns_none=rear_returns_none,
            fail_close=fail_rig_close,
            close_delay_ms=close_delay_ms,
        )
        self.detector = _FakeDetector(self.clock, fail_inference=fail_inference)
        self.actuator = _FakeActuator(fail_apply_call=fail_apply_call)
        self.log_delay_ms = log_delay_ms
        self.logs = []
        self.log_times = []

    def _print(self, serialized, *args, **kwargs):
        self.log_times.append(self.clock.now())
        self.logs.append(json.loads(serialized))
        self.clock.advance_ms(self.log_delay_ms)

    def run(self, frames):
        fake_time = SimpleNamespace(perf_counter=self.clock.now, monotonic=self.clock.now)
        with ExitStack() as patches:
            patches.enter_context(patch.object(pi_runtime, "time", fake_time))
            patches.enter_context(patch.object(pi_runtime, "OpenCvYoloOnnxDetector", return_value=self.detector))
            patches.enter_context(patch.object(pi_runtime, "PiDualCameraRig", return_value=self.rig))
            patches.enter_context(patch.object(pi_runtime, "create_alert_actuator", return_value=self.actuator))
            patches.enter_context(patch("builtins.print", side_effect=self._print))
            pi_runtime.PiCollisionRiskRuntime(self.config).run(max_frames=frames)

    @property
    def heartbeats(self):
        return [item for item in self.logs if item.get("event") == "runtime_status"]

    @property
    def final(self):
        return self.logs[-1]


class PiRuntimeTimingTests(unittest.TestCase):
    def assert_clean_shutdown(self, harness):
        self.assertEqual(harness.rig.close_calls, 1)
        self.assertEqual(harness.actuator.close_calls, 1)
        self.assertFalse(harness.actuator.outputs_on)
        self.assertTrue(harness.final["outputs_closed"])
        final_output = harness.final["alert_output"]
        self.assertEqual(final_output["applied_level"], "NONE")
        self.assertFalse(final_output["alerts_permitted"])
        for output in ("buzzer_on", "vibration_on", "led_on"):
            self.assertFalse(final_output[output])

    def assert_measured_rate(self, directional, completed_at, snapshot_time):
        # Include age since the latest completed directional update: a
        # skipped or stalled stream must not keep advertising its old rate.
        expected = 0.0 if len(completed_at) < 2 else (len(completed_at) - 1) / (snapshot_time - completed_at[0])
        self.assertAlmostEqual(directional["processed_fps"], expected, delta=0.011)

    def test_rear_work_is_not_double_counted_as_postprocessing(self):
        harness = _RuntimeHarness()
        harness.run(1)

        measured = harness.final["performance_final"]
        # The old nested postprocess timer reports 165 ms for these 105 ms
        # of actual work, incorrectly failing the unchanged 150 ms budget.
        self.assertAlmostEqual(measured["end_to_end_ms"], 105.0)
        self.assertAlmostEqual(measured["capture_ms"], 15.0)
        self.assertAlmostEqual(measured["inference_ms"], 90.0)
        self.assertAlmostEqual(measured["postprocess_ms"], 0.0)
        self.assertTrue(measured["meets_latency_budget"])
        self.assertFalse(measured["risk_alerts_permitted"])
        directional = harness.final["directional_performance"]
        for direction, capture_ms, inference_ms in (("front", 5.0, 40.0), ("rear", 10.0, 50.0)):
            with self.subTest(direction=direction):
                metrics = directional[direction]
                self.assertTrue(metrics["enabled"])
                self.assertEqual(metrics["processed_frames"], 1)
                self.assertAlmostEqual(metrics["capture_ms"], capture_ms)
                self.assertAlmostEqual(metrics["inference_ms"], inference_ms)
                self.assertAlmostEqual(metrics["processing_ms"], capture_ms + inference_ms)
                self.assertEqual(metrics["processed_fps"], 0.0)
        self.assert_clean_shutdown(harness)

    def test_empty_scene_emits_a_heartbeat_for_every_completed_loop(self):
        harness = _RuntimeHarness()
        harness.run(3)

        self.assertEqual(len(harness.heartbeats), 3)
        self.assertEqual([item["frame_count"] for item in harness.heartbeats], [1, 2, 3])
        self.assertEqual(harness.final["event"], "runtime_final")
        self.assertTrue(harness.final["completed"])
        self.assertIsNone(harness.final["error"])
        self.assertEqual(harness.final["processed_frames"], 3)
        self.assertEqual(harness.final["performance_final"]["frame_count"], 3)
        for heartbeat in harness.heartbeats:
            self.assertEqual(heartbeat["directional_performance"]["front"]["detections"], 0)
            self.assertEqual(heartbeat["directional_performance"]["rear"]["detections"], 0)
            self.assertFalse(heartbeat["alert_output"]["buzzer_on"])
        self.assert_clean_shutdown(harness)

    def test_directional_counts_and_rates_follow_work_actually_processed(self):
        harness = _RuntimeHarness(front_inference_every_n_frames=2, rear_capture_every_n_frames=3)
        harness.run(6)

        self.assertEqual(harness.rig.calls, {"front": 6, "rear": 2})
        self.assertEqual(harness.detector.calls, {"front": 3, "rear": 2})
        directional = harness.final["directional_performance"]
        for direction, count in (("front", 3), ("rear", 2)):
            with self.subTest(direction=direction):
                self.assertEqual(directional[direction]["processed_frames"], count)
                self.assert_measured_rate(directional[direction], harness.detector.completed_at[direction], harness.final["performance_timestamp_s"])
        self.assert_clean_shutdown(harness)

    def test_logging_time_reduces_actual_cadence_but_not_compute_latency(self):
        fast = _RuntimeHarness()
        slow = _RuntimeHarness(log_delay_ms=100.0)
        fast.run(3)
        slow.run(3)

        self.assertEqual(len(slow.heartbeats), 3)
        self.assertAlmostEqual(slow.final["performance_final"]["end_to_end_ms"], 105.0)
        # One initial 105 ms cycle, followed by two intervals including the
        # preceding 100 ms heartbeat write: (105 + 205 + 205) / 3.
        self.assertAlmostEqual(slow.final["performance_final"]["effective_fps"], 3000.0 / 515.0, delta=0.011)
        self.assertLess(
            slow.final["performance_final"]["effective_fps"],
            fast.final["performance_final"]["effective_fps"],
        )
        for direction in ("front", "rear"):
            with self.subTest(direction=direction):
                slow_direction = slow.final["directional_performance"][direction]
                self.assert_measured_rate(slow_direction, slow.detector.completed_at[direction], slow.final["performance_timestamp_s"])
                self.assertLess(slow_direction["processed_fps"], fast.final["directional_performance"][direction]["processed_fps"])
        self.assert_clean_shutdown(fast)
        self.assert_clean_shutdown(slow)

    def test_final_metrics_exclude_shutdown_and_final_heartbeat_write_delays(self):
        harness = _RuntimeHarness(log_delay_ms=100.0, close_delay_ms=2000.0)
        harness.run(3)

        last_heartbeat = harness.heartbeats[-1]
        self.assertEqual(harness.final["performance_final"], last_heartbeat["performance"])
        self.assertEqual(harness.final["directional_performance"], last_heartbeat["directional_performance"])
        self.assertEqual(harness.final["performance_timestamp_s"], last_heartbeat["timestamp_s"])
        self.assertGreater(harness.log_times[-1] - harness.final["performance_timestamp_s"], 2.09)
        self.assert_clean_shutdown(harness)

    def test_disabled_rear_camera_has_no_processed_frames(self):
        harness = _RuntimeHarness(rear_camera=None, rear_calibration=None)
        harness.run(3)

        self.assertEqual(harness.rig.calls["rear"], 0)
        self.assertEqual(harness.detector.calls["rear"], 0)
        rear = harness.final["directional_performance"]["rear"]
        self.assertFalse(rear["enabled"])
        self.assertEqual(rear["processed_frames"], 0)
        self.assertEqual(rear["processed_fps"], 0.0)
        self.assertIsNone(rear["last_update_age_s"])
        self.assert_clean_shutdown(harness)

    def test_missing_rear_frame_is_not_counted_as_successful_processing(self):
        harness = _RuntimeHarness(rear_returns_none=True)
        harness.run(3)

        rear = harness.final["directional_performance"]["rear"]
        self.assertEqual(harness.rig.calls["rear"], 3)
        self.assertEqual(harness.detector.calls["rear"], 0)
        self.assertEqual(rear["processed_frames"], 0)
        self.assertEqual(rear["processed_fps"], 0.0)
        self.assertIsNone(rear["last_update_age_s"])
        self.assert_clean_shutdown(harness)

    def test_capture_and_inference_errors_never_report_completed_run(self):
        failures = (
            ({"fail_capture": ("front", 1)}, "front capture failure", 0),
            ({"fail_capture": ("rear", 1)}, "rear capture failure", 0),
            ({"fail_inference": ("front", 2)}, "front inference failure", 1),
            ({"fail_inference": ("rear", 2)}, "rear inference failure", 1),
        )
        for options, error, completed in failures:
            with self.subTest(error=error):
                harness = _RuntimeHarness(**options)
                with self.assertRaisesRegex(RuntimeError, error):
                    harness.run(3)
                self.assertEqual(harness.final["event"], "runtime_final")
                self.assertFalse(harness.final["completed"])
                self.assertIn(error, harness.final["error"])
                self.assertEqual(harness.final["processed_frames"], completed)
                self.assertEqual(len(harness.heartbeats), completed)
                for direction in ("front", "rear"):
                    successful = len(harness.detector.completed_at[direction])
                    reported = harness.final["directional_performance"][direction]["processed_frames"]
                    self.assertEqual(reported, successful)
                self.assert_clean_shutdown(harness)

    def test_camera_cleanup_error_still_closes_actuator_and_reports_unknown_state(self):
        harness = _RuntimeHarness(fail_rig_close=True)
        with self.assertRaisesRegex(RuntimeError, "rig close failure"):
            harness.run(1)

        self.assertEqual(harness.rig.close_calls, 1)
        self.assertEqual(harness.actuator.close_calls, 1)
        self.assertFalse(harness.actuator.outputs_on)
        self.assertFalse(harness.final["completed"])
        self.assertFalse(harness.final["outputs_closed"])
        self.assertIn("rig close failure", harness.final["error"])
        self.assertEqual(harness.final["alert_output"]["applied_level"], "UNKNOWN")
        self.assertIsNone(harness.final["alert_output"]["buzzer_on"])

    def test_failed_actuator_apply_does_not_commit_a_completed_performance_cycle(self):
        for failed_cycle in (1, 2):
            with self.subTest(failed_cycle=failed_cycle):
                harness = _RuntimeHarness(fail_apply_call=failed_cycle)
                with self.assertRaisesRegex(RuntimeError, "actuator apply failure"):
                    harness.run(3)

                completed_cycles = failed_cycle - 1
                self.assertFalse(harness.final["completed"])
                self.assertIn("actuator apply failure", harness.final["error"])
                self.assertEqual(harness.final["processed_frames"], completed_cycles)
                self.assertEqual(len(harness.heartbeats), completed_cycles)
                if completed_cycles == 0:
                    self.assertIsNone(harness.final["performance_final"])
                else:
                    self.assertEqual(harness.final["performance_final"]["frame_count"], completed_cycles)
                    self.assertEqual(harness.final["performance_final"], harness.heartbeats[-1]["performance"])
                # Perception really succeeded in the failed output cycle;
                # preserve that diagnostic evidence without calling the
                # incomplete aggregate cycle a successful runtime sample.
                for direction in ("front", "rear"):
                    self.assertEqual(harness.final["directional_performance"][direction]["processed_frames"], failed_cycle)
                self.assert_clean_shutdown(harness)


if __name__ == "__main__":
    unittest.main()
