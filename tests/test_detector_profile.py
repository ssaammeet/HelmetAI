"""Image-only profiler bounds, isolation and host smoke checks."""

import contextlib
import io
import json
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from helmetai_fcw import detector_profile


class DetectorProfileTests(unittest.TestCase):
    def test_defaults_are_bounded_and_all_pi_thread_counts_are_tried(self):
        args = detector_profile._parser().parse_args(["--image", "capture.jpg", "--model", "model.onnx"])
        self.assertEqual((args.warmup, args.runs, args.threads), (3, 10, (1, 2, 3, 4)))

    def test_invalid_bounds_and_duplicate_threads_are_rejected(self):
        for flag, value in (("--runs", "0"), ("--runs", "101"), ("--warmup", "0"), ("--warmup", "21"),
                            ("--threads", "1,1"), ("--threads", "8"), ("--threads", "0")):
            with self.subTest(flag=flag, value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    detector_profile._parser().parse_args(["--image", "capture.jpg", flag, value])

    def test_nearest_rank_p95_and_median(self):
        summary = detector_profile._summary(list(range(1, 21)))
        self.assertEqual(summary["median_ms"], 10.5)
        self.assertEqual(summary["p95_ms"], 19)

    def test_module_import_has_no_camera_runtime_or_gpio_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import helmetai_fcw.detector_profile; "
             "assert not any(name in sys.modules for name in "
             "('helmetai_fcw.pi_runtime', 'helmetai_fcw.rpi_camera', 'helmetai_fcw.alert_actuator', 'gpiozero', 'picamera2'))"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_config_model_is_relative_to_config_and_override_is_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            model.touch()
            override = root / "override.onnx"
            override.touch()
            config = root / "config.json"
            config.write_text(json.dumps({"detector": {"model_path": "model.onnx", "confidence": 0.61}}))
            self.assertEqual(detector_profile._settings(config, None), (model.resolve(), 640, 0.61))
            self.assertEqual(detector_profile._settings(config, override), (override.resolve(), 640, 0.61))

    def test_smaller_input_and_nonfinite_threshold_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            model.touch()
            config = root / "config.json"
            for settings in ({"input_size_px": 320}, {"confidence": float("nan")}):
                config.write_text(json.dumps({"detector": {"model_path": "model.onnx", **settings}}))
                with self.assertRaises(ValueError):
                    detector_profile._settings(config, None)

    def test_cli_will_not_overwrite_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            model.touch()
            report = root / "report.json"
            report.write_text("preserve me")
            with contextlib.redirect_stderr(io.StringIO()):
                status = detector_profile.main(["--image", str(root / "missing.jpg"), "--model", str(model), "--output", str(report)])
            self.assertEqual(status, 2)
            self.assertEqual(report.read_text(), "preserve me")

    def test_profile_calls_detector_only_excludes_warmup_and_restores_threads(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and NumPy are optional detector dependencies.")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.onnx"
            model.touch()
            capture = root / "capture.jpg"
            success, jpeg = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
            self.assertTrue(success)
            capture.write_bytes(jpeg.tobytes())
            config = root / "config.json"
            source_config = json.dumps({"detector": {"model_path": "model.onnx", "confidence": 0.61}, "alert_actuator": {"enabled": True}})
            config.write_text(source_config)
            args = detector_profile._parser().parse_args(["--image", str(capture), "--config", str(config), "--runs", "2"])
            original_threads = cv2.getNumThreads()
            with patch.object(detector_profile, "OpenCvYoloOnnxDetector") as constructor, contextlib.redirect_stderr(io.StringIO()):
                constructor.return_value.detect.return_value = []
                report = detector_profile.profile(args)
            self.assertEqual(constructor.call_count, 4)
            self.assertEqual(constructor.return_value.detect.call_count, 4 * (3 + 2))
            self.assertEqual(report["profile_kind"], "captured_jpeg_detector_only")
            self.assertFalse(report["safety_benchmark"])
            self.assertFalse(report["configuration_modified"])
            self.assertEqual(config.read_text(), source_config)
            self.assertEqual(len(report["results"]), 4)
            self.assertTrue(all(len(result["samples_ms"]) == 2 for result in report["results"]))
            self.assertEqual(cv2.getNumThreads(), original_threads)
            self.assertEqual(report["image"]["width_px"], 64)
            self.assertEqual(report["detector"]["confidence_threshold"], 0.61)
            self.assertEqual(len(report["model"]["sha256"]), 64)

    def test_shipped_model_profiles_a_synthetic_jpeg_without_claiming_pi_performance(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV and NumPy are optional detector dependencies.")
        model = Path(__file__).resolve().parents[1] / "models" / "detector.onnx"
        if not model.is_file():
            self.skipTest("The shipped detector model is unavailable.")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            capture = root / "synthetic_test_fixture.jpg"
            success, jpeg = cv2.imencode(".jpg", np.zeros((480, 640, 3), dtype=np.uint8))
            self.assertTrue(success)
            capture.write_bytes(jpeg.tobytes())
            output = root / "host_smoke_report.json"
            with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                status = detector_profile.main([
                    "--image", str(capture), "--model", str(model), "--output", str(output),
                    "--warmup", "1", "--runs", "2",
                ])
            self.assertEqual(status, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(report["safety_benchmark"])
            self.assertEqual(report["environment"]["opencv_version"], cv2.__version__)
            self.assertEqual([r["effective_cpu_threads"] for r in report["results"]], [1, 2, 3, 4])
            self.assertTrue(all(r["median_ms"] > 0 for r in report["results"]))
            self.assertIn("Host-only measurement", report["warning"])


if __name__ == "__main__":
    unittest.main()
