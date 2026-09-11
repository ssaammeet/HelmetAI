import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from helmetai_fcw.pi_runtime import PiCollisionRiskRuntime, PiRuntimeConfig


class PiRuntimeConfigTests(unittest.TestCase):
    def test_example_profile_exposes_edge_performance_controls(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = PiRuntimeConfig.from_json(root / "config" / "pi5_dual_camera.example.json")
        # The shipped YOLO11n ONNX model has a fixed 640 x 640 input.  A
        # smaller blob cannot be reshaped by OpenCV-DNN at inference time.
        self.assertEqual(config.detector_input_size_px, 640)
        self.assertEqual(config.detector_cpu_threads, 3)
        self.assertEqual(config.front_inference_every_n_frames, 1)
        self.assertEqual(config.tracking_max_gap_s, 2.0)
        self.assertEqual(config.minimum_risk_fps, 5.0)
        self.assertEqual(config.performance_warmup_frames, 60)
        self.assertEqual(config.risk_assessment_freshness_s, 0.45)
        self.assertFalse(config.alert_actuator.enabled)
        self.assertIsNone(config.alert_actuator.gpio_chip)
        self.assertEqual(config.latency_budget_ms, 150.0)

    def test_uncalibrated_profile_still_allows_perception_bring_up(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = PiRuntimeConfig.from_json(root / "config" / "pi5_dual_camera.example.json")
        PiCollisionRiskRuntime(config)

    def test_profile_uses_explicit_rear_lateral_orientation(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = PiRuntimeConfig.from_json(root / "config" / "pi5_dual_camera.example.json")
        self.assertEqual(config.front_calibration.lateral_sign, 1.0)
        self.assertIsNotNone(config.rear_calibration)
        self.assertEqual(config.rear_calibration.lateral_sign, -1.0)

    def test_duplicate_front_and_rear_camera_indexes_are_rejected(self) -> None:
        root = Path(__file__).resolve().parents[1]
        content = json.loads((root / "config" / "pi5_dual_camera.example.json").read_text(encoding="utf-8"))
        content["rear_camera"]["camera_index"] = content["front_camera"]["camera_index"]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-camera-profile.json"
            path.write_text(json.dumps(content), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different camera_index"):
                PiRuntimeConfig.from_json(path)

    def test_enabled_alert_actuator_requires_explicit_wiring(self) -> None:
        root = Path(__file__).resolve().parents[1]
        content = json.loads((root / "config" / "pi5_dual_camera.example.json").read_text(encoding="utf-8"))
        content["alert_actuator"]["enabled"] = True
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-alert-profile.json"
            path.write_text(json.dumps(content), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least one configured BCM pin"):
                PiRuntimeConfig.from_json(path)

    def test_assessment_freshness_timeout_must_be_positive(self) -> None:
        root = Path(__file__).resolve().parents[1]
        content = json.loads((root / "config" / "pi5_dual_camera.example.json").read_text(encoding="utf-8"))
        content["risk_assessment_freshness_s"] = 0.0
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-freshness-profile.json"
            path.write_text(json.dumps(content), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "risk_assessment_freshness_s must be positive"):
                PiRuntimeConfig.from_json(path)

    def test_omitted_warmup_uses_a_full_performance_window(self) -> None:
        root = Path(__file__).resolve().parents[1]
        content = json.loads((root / "config" / "pi5_dual_camera.example.json").read_text(encoding="utf-8"))
        content.pop("performance_warmup_frames")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "default-warmup-profile.json"
            path.write_text(json.dumps(content), encoding="utf-8")
            config = PiRuntimeConfig.from_json(path)
        self.assertEqual(config.performance_warmup_frames, config.performance_window_frames)

    def test_calibration_and_rear_enabled_flags_require_json_booleans(self) -> None:
        root = Path(__file__).resolve().parents[1]
        original = json.loads((root / "config" / "pi5_dual_camera.example.json").read_text(encoding="utf-8"))
        invalid_values = (
            ("rear_camera.enabled", lambda data: data["rear_camera"].__setitem__("enabled", "false")),
            ("front_calibration.calibrated", lambda data: data["front_calibration"].__setitem__("calibrated", "false")),
            ("rear_calibration.calibrated", lambda data: data["rear_calibration"].__setitem__("calibrated", 0)),
        )
        with TemporaryDirectory() as directory:
            for index, (expected_message, mutate) in enumerate(invalid_values):
                content = json.loads(json.dumps(original))
                mutate(content)
                path = Path(directory) / f"invalid-boolean-{index}.json"
                path.write_text(json.dumps(content), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected_message):
                    PiRuntimeConfig.from_json(path)

    def test_enabled_rear_camera_requires_a_rear_calibration_profile(self) -> None:
        root = Path(__file__).resolve().parents[1]
        content = json.loads((root / "config" / "pi5_dual_camera.example.json").read_text(encoding="utf-8"))
        del content["rear_calibration"]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "missing-rear-calibration.json"
            path.write_text(json.dumps(content), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "rear_calibration is missing"):
                PiRuntimeConfig.from_json(path)

    def test_invalid_performance_limits_are_rejected_not_silently_clamped(self) -> None:
        root = Path(__file__).resolve().parents[1]
        content = json.loads((root / "config" / "pi5_dual_camera.example.json").read_text(encoding="utf-8"))
        content["minimum_risk_fps"] = 0
        with TemporaryDirectory() as directory:
            path = Path(directory) / "invalid-performance-profile.json"
            path.write_text(json.dumps(content), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "minimum_risk_fps"):
                PiRuntimeConfig.from_json(path)


if __name__ == "__main__":
    unittest.main()
