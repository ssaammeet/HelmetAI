import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from helmetai_fcw.camera_check import _write_labelled_frame
from helmetai_fcw.cli import build_parser
from helmetai_fcw.webcam_runtime import WebcamRuntimeConfig, _webcam_detection_label


class DesktopSafetyV7Tests(unittest.TestCase):
    def test_webcam_does_not_print_unverified_metres_or_green_all_clear(self):
        detection = SimpleNamespace(label="truck", confidence=0.7)
        observation = SimpleNamespace(longitudinal_m=12.3)
        for calibrated, demo, status in ((False, False, "degraded"), (True, False, "degraded"), (True, True, "ready")):
            config = SimpleNamespace(calibration=SimpleNamespace(calibrated=calibrated), demonstration_only=demo)
            assessment = SimpleNamespace(direction=SimpleNamespace(value="front"), level=SimpleNamespace(name="NONE"),
                                         system_status=SimpleNamespace(value=status), conservative_ttc_s=2.2)
            text, colour = _webcam_detection_label(config, detection, observation, assessment)
            self.assertNotIn("12.3m", text)
            self.assertNotIn("2.2s", text)
            self.assertEqual(colour, (190, 190, 190))
            self.assertIn("truck", text)

    def test_camera_snapshot_preserves_bgr_and_original_array(self):
        # Pixel below the label area. BGR red must remain [0,0,255].
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[60, 60] = (0, 0, 255)
        original = image.copy()
        fake_cv = Mock()
        fake_cv.FONT_HERSHEY_SIMPLEX = 0
        fake_cv.imwrite.return_value = True
        with tempfile.TemporaryDirectory() as directory:
            _write_labelled_frame(fake_cv, image, Path(directory) / "front.jpg", "FRONT")
        written = fake_cv.imwrite.call_args.args[1]
        np.testing.assert_array_equal(written[60, 60], (0, 0, 255))
        np.testing.assert_array_equal(image, original)
        self.assertIsNot(written, image)
        fake_cv.cvtColor.assert_not_called()

    def test_snapshot_write_failure_is_reported(self):
        fake_cv = Mock()
        fake_cv.FONT_HERSHEY_SIMPLEX = 0
        fake_cv.imwrite.return_value = False
        with self.assertRaises(RuntimeError):
            _write_labelled_frame(fake_cv, np.zeros((80, 80, 3)), Path("unused.jpg"), "FRONT")

    def test_desktop_calibration_and_demo_require_real_json_booleans(self):
        for key in ("calibrated", "demonstration_only"):
            for bad in ("false", "true", 0, 1, None, [], {}):
                with self.subTest(key=key, value=bad), tempfile.TemporaryDirectory() as directory:
                    data = {"webcam": {}, "front_calibration": {"focal_length_px": 580},
                            "detector": {"model_path": "detector.onnx"}}
                    (data["front_calibration"] if key == "calibrated" else data)[key] = bad
                    config = Path(directory) / "config.json"
                    config.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        WebcamRuntimeConfig.from_json(config)

    def test_new_cli_options_remain_explicit(self):
        parser = build_parser()
        video = parser.parse_args(["video-run", "--video", "test.mp4", "--report-dir", "new",
                                   "--max-frames", "20", "--start-seconds", "1.5", "--no-preview"])
        self.assertEqual(video.max_frames, 20)
        self.assertEqual(video.start_seconds, 1.5)
        self.assertEqual(video.report_dir, "new")
        self.assertIsNone(video.source_fps)
        image = parser.parse_args(["image-check", "--image", "photo.jpg", "--model", "model.onnx",
                                   "--output-dir", "new", "--direction", "rear"])
        self.assertEqual(image.direction, "rear")
        self.assertEqual(image.rotate, 0)
        report = parser.parse_args(["log-report", "--runtime", "runtime.log", "--output-dir", "new"])
        self.assertIsNone(report.before)


if __name__ == "__main__":
    unittest.main()
