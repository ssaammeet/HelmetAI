import json
import tempfile
import unittest
from pathlib import Path

from helmetai_fcw.webcam_runtime import WebcamRuntimeConfig


class WebcamConfigTests(unittest.TestCase):
    def test_loads_relative_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = root / "webcam.json"
            config_path.write_text(
                json.dumps(
                    {
                        "webcam": {"camera_index": 1, "width_px": 800, "height_px": 600, "fps": 15},
                        "front_calibration": {"focal_length_px": 700, "calibrated": True},
                        "detector": {"model_path": "../models/detector.onnx"},
                    }
                ),
                encoding="utf-8",
            )
            config = WebcamRuntimeConfig.from_json(config_path)
        self.assertEqual(config.camera_index, 1)
        self.assertEqual(config.width_px, 800)
        self.assertEqual(config.camera_role, "front")
        self.assertTrue(config.calibration.calibrated)
        self.assertTrue(config.model_path.endswith("models" + chr(92) + "detector.onnx") or config.model_path.endswith("models/detector.onnx"))

    def test_rejects_unknown_camera_role(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "webcam.json"
            config_path.write_text(
                json.dumps(
                    {
                        "camera_role": "side",
                        "webcam": {"camera_index": 0},
                        "front_calibration": {"focal_length_px": 700},
                        "detector": {"model_path": "detector.onnx"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                WebcamRuntimeConfig.from_json(config_path)

    def test_loads_explicit_demo_only_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "webcam.json"
            config_path.write_text(
                json.dumps(
                    {
                        "demonstration_only": True,
                        "webcam": {"camera_index": 0},
                        "front_calibration": {"focal_length_px": 700, "calibrated": True},
                        "detector": {"model_path": "detector.onnx"},
                    }
                ),
                encoding="utf-8",
            )
            config = WebcamRuntimeConfig.from_json(config_path)
        self.assertTrue(config.demonstration_only)


if __name__ == "__main__":
    unittest.main()
