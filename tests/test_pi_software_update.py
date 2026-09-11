"""Portable V6 config/benchmark tests. Never execute the system updater."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("update_config", ROOT / "deploy/prepare_camera_only_config.py")
assert SPEC is not None and SPEC.loader is not None
UPDATE_CONFIG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UPDATE_CONFIG)


class CameraOnlyUpdateConfigTests(unittest.TestCase):
    def test_payload_changes_only_gpio_enabled(self) -> None:
        profile = json.loads((ROOT / "config/pi5_dual_camera.example.json").read_text(encoding="utf-8"))
        profile["front_calibration"]["calibrated"] = True
        profile["rear_calibration"]["calibrated"] = False
        profile["front_camera"]["camera_index"] = 2
        profile["detector"]["model_path"] = "../custom-models/my-model.onnx"
        profile["alert_actuator"]["enabled"] = True
        profile["local_notes"] = {"camera": "ön", "keep": [1, 2, 3]}
        expected = copy.deepcopy(profile)
        expected["alert_actuator"]["enabled"] = False
        self.assertEqual(UPDATE_CONFIG.camera_only_payload(json.dumps(profile)), expected)

    def test_missing_or_null_actuator_becomes_explicitly_disabled(self) -> None:
        for raw in ('{"custom": 42}', '{"custom": 42, "alert_actuator": null}'):
            self.assertEqual(UPDATE_CONFIG.camera_only_payload(raw), {"custom": 42, "alert_actuator": {"enabled": False}})

    def test_malformed_or_ambiguous_json_is_rejected(self) -> None:
        for raw in ('[]', '{', '{"a": NaN}', '{"a": 1, "a": 2}', '{"alert_actuator": []}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                UPDATE_CONFIG.camera_only_payload(raw)

    def test_prepare_validates_original_relative_model_and_preserves_original(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "models").mkdir()
            (directory / "staged").mkdir()
            model = directory / "models/detector.onnx"
            model.write_bytes(b"model left untouched; no inference is performed")
            config = directory / "profile.json"
            output = directory / "staged/profile.json"
            profile = json.loads((ROOT / "config/pi5_dual_camera.example.json").read_text(encoding="utf-8"))
            profile["detector"]["model_path"] = "models/detector.onnx"
            profile["alert_actuator"]["enabled"] = True
            profile["alert_actuator"]["buzzer_bcm_pin"] = 18
            original = json.dumps(profile, indent=2)
            config.write_text(original, encoding="utf-8")
            UPDATE_CONFIG.prepare_config(config, output)
            self.assertEqual(config.read_text(encoding="utf-8"), original)
            staged = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(staged["detector"]["model_path"], "models/detector.onnx")
            self.assertFalse(staged["alert_actuator"]["enabled"])
            self.assertEqual(staged["front_calibration"], profile["front_calibration"])
            self.assertEqual(model.read_bytes(), b"model left untouched; no inference is performed")
            with self.assertRaises(ValueError):
                UPDATE_CONFIG.prepare_config(config, config)
            model.unlink()
            other_output = directory / "missing-model-output.json"
            with self.assertRaises(ValueError):
                UPDATE_CONFIG.prepare_config(config, other_output)
            self.assertFalse(other_output.exists())


class PiBenchmarkResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = (ROOT / "scripts/benchmark_pi.sh").read_text(encoding="utf-8")
        blocks = re.findall(r"<<'PY'\n(.*?)\nPY(?:\n|$)", source, flags=re.DOTALL)
        if len(blocks) != 1:
            raise AssertionError("Expected one embedded benchmark result validator")
        cls.validator = blocks[0]

    def payload(self) -> dict[str, object]:
        return {
            "completed": True,
            "error": None,
            "processed_frames": 300,
            "performance_final": {
                "p95_end_to_end_ms": 80.0,
                "effective_fps": 12.0,
                "minimum_risk_fps": 5.0,
                "warmup_complete": True,
                "meets_latency_budget": True,
                "meets_minimum_risk_fps": True,
                "risk_alerts_permitted": True,
            },
            "directional_performance": {
                "front": {"enabled": True, "processed_frames": 300, "processed_fps": 12.0, "last_update_age_s": 0.01},
                "rear": {"enabled": True, "processed_frames": 100, "processed_fps": 6.0, "last_update_age_s": 0.1},
            },
        }

    def run_validator(self, payload: dict[str, object] | None, *, rear_enabled: bool = True) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            profile = directory / "config.json"
            profile.write_text(json.dumps({
                "rear_camera": {"enabled": rear_enabled},
                "rear_capture_every_n_frames": 9,
                "minimum_risk_fps": 5.0,
                "latency_budget_ms": 150.0,
                "risk_assessment_freshness_s": 0.45,
            }), encoding="utf-8")
            log = directory / "run.jsonl"
            lines = ["not JSON", "[]", '{"event": "started"}']
            if payload is not None:
                lines.append(json.dumps(payload))
            log.write_text("\n".join(lines), encoding="utf-8")
            return subprocess.run(
                [sys.executable, "-c", self.validator, str(log), str(profile), "300"],
                capture_output=True, text=True, check=False, timeout=15,
            )

    def test_uses_measured_direction_rates_not_loop_divided_by_cadence(self) -> None:
        result = self.run_validator(self.payload())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("benchmark_rear_risk_fps=6.00", result.stdout)

    def test_disabled_rear_is_not_required(self) -> None:
        payload = self.payload()
        del payload["directional_performance"]["rear"]
        result = self.run_validator(payload, rear_enabled=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_final_record_fails(self) -> None:
        self.assertEqual(self.run_validator(None).returncode, 4)

    def test_incomplete_failed_or_short_run_fails(self) -> None:
        for field, value in (("completed", False), ("error", "camera failure"), ("processed_frames", 299)):
            payload = self.payload()
            payload[field] = value
            with self.subTest(field=field):
                self.assertEqual(self.run_validator(payload).returncode, 3)

    def test_each_health_gate_fails_closed(self) -> None:
        for field in ("warmup_complete", "meets_latency_budget", "meets_minimum_risk_fps", "risk_alerts_permitted"):
            payload = self.payload()
            payload["performance_final"][field] = False
            with self.subTest(field=field):
                self.assertEqual(self.run_validator(payload).returncode, 3)

    def test_missing_sparse_slow_or_stale_rear_fails(self) -> None:
        cases = (("enabled", False), ("processed_frames", 1), ("processed_fps", 4.9), ("last_update_age_s", 0.451), ("last_update_age_s", -1.0))
        for field, value in cases:
            payload = self.payload()
            payload["directional_performance"]["rear"][field] = value
            with self.subTest(field=field, value=value):
                self.assertEqual(self.run_validator(payload).returncode, 3)
        payload = self.payload()
        del payload["directional_performance"]["rear"]
        self.assertEqual(self.run_validator(payload).returncode, 3)

    def test_nonfinite_or_nonnumeric_direction_measurements_fail(self) -> None:
        for field in ("processed_fps", "last_update_age_s"):
            for value in (float("nan"), float("inf"), None, "6.0", True):
                payload = self.payload()
                payload["directional_performance"]["rear"][field] = value
                with self.subTest(field=field, value=value):
                    self.assertEqual(self.run_validator(payload).returncode, 3)

    def test_checks_actual_values_even_when_health_flags_are_true(self) -> None:
        for field, value in (("p95_end_to_end_ms", 151), ("effective_fps", 4), ("minimum_risk_fps", 4), ("p95_end_to_end_ms", float("nan"))):
            payload = self.payload()
            payload["performance_final"][field] = value
            with self.subTest(field=field, value=value):
                self.assertEqual(self.run_validator(payload).returncode, 3)


class PiUpdateShellSyntaxTests(unittest.TestCase):
    def test_shell_assets_have_lf_and_valid_bash_syntax(self) -> None:
        candidates = [Path("C:/Program Files/Git/usr/bin/bash.exe")]
        discovered = shutil.which("bash")
        if discovered:
            candidates.append(Path(discovered))
        shell = next((candidate for candidate in candidates if candidate.is_file()), None)
        for relative in ("deploy/update_pi.sh", "deploy/install_pi.sh", "scripts/benchmark_pi.sh", "scripts/preflight_pi.sh", "scripts/profile_detector_pi.sh", "scripts/run_pi_diagnostic.sh"):
            path = ROOT / relative
            payload = path.read_bytes()
            self.assertTrue(payload.startswith(b"#!/usr/bin/env bash\n"))
            self.assertNotIn(b"\r", payload)
            if shell is not None:
                # -n parses only: never invoke update_pi.sh on the host.
                result = subprocess.run([str(shell), "-n", path.as_posix()], capture_output=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))


if __name__ == "__main__":
    unittest.main()
