"""Integration test for the model and the Pi example profile shipped together."""

import unittest
from pathlib import Path


class DetectorModelIntegrationTests(unittest.TestCase):
    def test_shipped_model_accepts_the_example_profile_input_size(self) -> None:
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency-specific
            self.skipTest(f"NumPy unavailable: {exc}")
        from helmetai_fcw.detection import OpenCvYoloOnnxDetector
        from helmetai_fcw.pi_runtime import PiRuntimeConfig

        root = Path(__file__).resolve().parents[1]
        model = root / "models" / "detector.onnx"
        if not model.is_file():  # pragma: no cover - source-only package variant
            self.skipTest("The shipped detector model is unavailable.")
        config = PiRuntimeConfig.from_json(root / "config" / "pi5_dual_camera.example.json")
        detector = OpenCvYoloOnnxDetector(model, input_size_px=config.detector_input_size_px)
        detections = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertIsInstance(detections, list)


if __name__ == "__main__":
    unittest.main()
