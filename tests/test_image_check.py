"""Offline evidence tests use fixed pixels and fake detections, never a camera."""

import contextlib
import hashlib
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from helmetai_fcw.detection import Detection
from helmetai_fcw.image_check import MAX_IMAGE_PIXELS, _image_dimensions, main, run_image_check

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None


@unittest.skipIf(cv2 is None or np is None, "OpenCV/NumPy optional image dependencies are unavailable.")
class ImageCheckTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.png"
        self.model = self.root / "detector.onnx"
        self.output = self.root / "output"
        self.pixels = np.zeros((60, 80, 3), dtype=np.uint8)
        self.pixels[2:8, 3:10] = [25, 50, 230]  # BGR; asymmetric rotation evidence
        self.assertTrue(cv2.imwrite(str(self.source), self.pixels))
        self.model.write_bytes(b"fake-model-for-unit-test")
        self.detector_patch = patch("helmetai_fcw.image_check.OpenCvYoloOnnxDetector")
        self.detector = self.detector_patch.start()
        self.addCleanup(self.detector_patch.stop)
        self.detector.return_value.detect.return_value = []

    def run_check(self, **kwargs):
        arguments = dict(direction="rear")
        arguments.update(kwargs)
        return run_image_check(self.source, self.model, self.output, **arguments)

    def test_empty_report_is_explicit_and_metrics_are_unknown(self):
        report = self.run_check()
        self.assertEqual(report["detection_count"], 0)
        self.assertTrue(report["empty_detections"])
        self.assertTrue(all(value is None for value in report["metrics"].values()))
        self.assertIsNone(report["physical_direction_verified"])
        self.assertEqual(set(path.name for path in self.output.iterdir()), {"annotated.jpg", "detections.json", "report.md"})
        saved = json.loads((self.output / "detections.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, report)
        markdown = (self.output / "report.md").read_text(encoding="utf-8")
        self.assertIn("No detections", markdown)
        self.assertIn("not measured accuracy", markdown)
        self.assertIn("DETECTION ONLY", markdown)
        self.assertIn("does not establish absence", markdown)
        self.assertEqual(report["model"]["sha256"], hashlib.sha256(self.model.read_bytes()).hexdigest())
        self.assertEqual(report["image"]["sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertEqual(report["artifacts"]["annotated.jpg"]["sha256"], hashlib.sha256((self.output / "annotated.jpg").read_bytes()).hexdigest())

    def test_same_detector_and_bgr_pixels_used_once_with_fixed_preprocessing(self):
        self.run_check(confidence=0.55, cpu_threads=2)
        self.detector.assert_called_once_with(self.model.resolve(), input_size_px=640, confidence_threshold=0.55, cpu_threads=2)
        self.detector.return_value.detect.assert_called_once()
        np.testing.assert_array_equal(self.detector.return_value.detect.call_args.args[0], self.pixels)

    def test_near_identical_cross_label_boxes_remain_as_ambiguity_evidence(self):
        self.detector.return_value.detect.return_value = [
            Detection("", "truck", 0.77, 10, 10, 30, 30),
            Detection("", "parking_meter", 0.71, 10, 10, 30, 30),
            Detection("", "truck", 0.66, 10, 10, 30, 30),
            Detection("", "person", 0.62, 55, 40, 10, 15),
        ]
        report = self.run_check()
        self.assertEqual(report["detection_count"], 4)
        self.assertFalse(report["ambiguities_auto_suppressed"])
        self.assertEqual([pair["detection_ids"] for pair in report["ambiguities"]], [[1, 2], [2, 3]])
        self.assertTrue(all(pair["iou"] == 1 for pair in report["ambiguities"]))
        self.assertEqual([d["label"] for d in report["detections"]], ["truck", "parking_meter", "truck", "person"])
        self.assertIsNone(report["detections"][0]["distance_m"])

    def test_edge_geometry_is_retained_but_flagged_and_not_given_metrics(self):
        self.detector.return_value.detect.return_value = [
            Detection("", "car", 0.9, 0, 10, 20, 20),
            Detection("", "truck", 0.8, 70, 20, 40, 50),
        ]
        report = self.run_check()
        self.assertEqual(report["detections"][1]["box_xywh_px"], [70, 20, 40, 50])
        self.assertFalse(report["detections"][0]["box_extends_outside_image"])
        self.assertTrue(report["detections"][1]["box_extends_outside_image"])
        self.assertTrue(all(d["box_may_be_truncated_at_image_edge"] for d in report["detections"]))

    def test_rotations_are_explicit_clockwise_and_recorded(self):
        for rotate, turns in ((0, 0), (90, -1), (180, 2), (270, 1)):
            with self.subTest(rotate=rotate):
                self.output = self.root / f"rotation_{rotate}"
                report = self.run_check(rotate=rotate)
                np.testing.assert_array_equal(self.detector.return_value.detect.call_args.args[0], np.rot90(self.pixels, turns))
                self.assertEqual(report["transform"]["rotate_clockwise_degrees"], rotate)
                self.assertFalse(report["transform"]["exif_auto_orientation"])
                self.assertFalse(report["transform"]["calibration_applied"])
                self.assertEqual(report["transform"]["width_px"], 60 if rotate in (90, 270) else 80)

    def test_invalid_bounded_arguments_fail_before_model_or_output(self):
        for arguments in ({"direction": "side"}, {"confidence": 0}, {"confidence": -1}, {"confidence": 1.1},
                          {"confidence": float("nan")}, {"confidence": float("inf")}, {"confidence": True},
                          {"cpu_threads": 0}, {"cpu_threads": 17}, {"cpu_threads": 1.5}, {"cpu_threads": True},
                          {"rotate": 45}, {"rotate": True}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.run_check(**arguments)
        self.detector.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_output_directory_and_input_paths_are_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "report.md"
        sentinel.write_text("existing evidence", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_check()
        for target in (self.source, self.model):
            original = target.read_bytes()
            with self.assertRaises(FileExistsError):
                run_image_check(self.source, self.model, target, direction="front")
            self.assertEqual(target.read_bytes(), original)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing evidence")
        self.detector.assert_not_called()

    def test_input_model_hardlink_alias_is_refused(self):
        linked = self.root / "alias.onnx"
        try:
            os.link(self.source, linked)
        except OSError as exc:
            self.skipTest(f"Hardlinks unavailable: {exc}")
        with self.assertRaisesRegex(ValueError, "alias"):
            run_image_check(self.source, linked, self.output, direction="front")
        self.detector.assert_not_called()

    def test_missing_parent_is_not_created(self):
        self.output = self.root / "missing" / "output"
        with self.assertRaisesRegex(ValueError, "parent"):
            self.run_check()
        self.assertFalse(self.output.parent.exists())

    def test_non_image_content_and_wrong_extensions_are_refused(self):
        for name, data in (("fake.png", b"not an image"), ("fake.jpg", b"\xff\xd8\xff"), ("fake.gif", self.source.read_bytes())):
            candidate = self.root / name
            candidate.write_bytes(data)
            with self.subTest(name=name), self.assertRaises(ValueError):
                run_image_check(candidate, self.model, self.output, direction="front")
        self.detector.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_header_limits_are_checked_before_opencv_decode(self):
        self.source.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + struct.pack(">II", 6000, 5000))
        self.assertGreater(6000 * 5000, MAX_IMAGE_PIXELS)
        with patch("cv2.imdecode") as decoder, self.assertRaisesRegex(ValueError, "pixel"):
            self.run_check()
        decoder.assert_not_called()

    def test_input_byte_limits_are_enforced_before_detector_loading(self):
        for limit in ("MAX_IMAGE_BYTES", "MAX_MODEL_BYTES"):
            with self.subTest(limit=limit), patch(f"helmetai_fcw.image_check.{limit}", 1), self.assertRaises(ValueError):
                self.run_check()
        self.detector.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_jpeg_and_png_headers_match_decoded_shape(self):
        self.assertEqual(_image_dimensions(self.source.read_bytes()), (80, 60))
        jpeg = self.root / "image.jpeg"
        self.assertTrue(cv2.imwrite(str(jpeg), self.pixels))
        self.assertEqual(_image_dimensions(jpeg.read_bytes()), (80, 60))
        report = run_image_check(jpeg, self.model, self.output, direction="front")
        self.assertEqual(report["image"]["width_px"], 80)

    def test_bad_detector_geometry_and_scores_produce_no_outputs(self):
        for detection in (Detection("", "car", float("nan"), 0, 0, 20, 20),
                          Detection("", "car", 0.8, 0, 0, 0, 20),
                          Detection("", "car", 2, 0, 0, 20, 20),
                          Detection("", "bad\nlabel", 0.8, 0, 0, 20, 20)):
            with self.subTest(detection=detection):
                self.detector.return_value.detect.return_value = [detection]
                with self.assertRaises(ValueError):
                    self.run_check()
        self.assertFalse(self.output.exists())

    def test_changed_model_during_inference_invalidates_report(self):
        def changed(frame):
            self.model.write_bytes(b"changed")
            return []
        self.detector.return_value.detect.side_effect = changed
        with self.assertRaisesRegex(RuntimeError, "changed"):
            self.run_check()
        self.assertFalse(self.output.exists())

    def test_annotation_remains_readable_for_tiny_input_and_has_detection_banner(self):
        self.assertTrue(cv2.imwrite(str(self.source), np.zeros((1, 1, 3), dtype=np.uint8)))
        with patch("cv2.putText", wraps=cv2.putText) as text_writer:
            report = self.run_check()
        labels = [call.args[1] for call in text_writer.call_args_list]
        self.assertTrue(any(label.startswith("DETECTION ONLY") for label in labels))
        self.assertTrue(any("NO DETECTIONS" in label for label in labels))
        self.assertGreaterEqual(report["rendering"]["width_px"], 1000)
        self.assertTrue((cv2.imread(str(self.output / "annotated.jpg"))).size > 0)

    def test_cli_forwards_arguments_and_returns_success(self):
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            result = main(["--image", str(self.source), "--model", str(self.model), "--output-dir", str(self.output),
                           "--direction", "front", "--rotate", "90", "--cpu-threads", "2"])
        self.assertEqual(result, 0)
        self.assertIn("DETECTION ONLY", stdout.getvalue())
        report = json.loads((self.output / "detections.json").read_text(encoding="utf-8"))
        self.assertEqual(report["direction"], "front")
        self.assertEqual(report["transform"]["rotate_clockwise_degrees"], 90)

    def test_cli_validation_error_has_nonzero_exit_and_no_output(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
            main(["--image", str(self.source), "--model", str(self.model), "--output-dir", str(self.output),
                  "--direction", "front", "--confidence", "nan"])
        self.assertEqual(failure.exception.code, 2)
        self.assertFalse(self.output.exists())

    def test_invalid_onnx_is_a_controlled_error_and_creates_no_outputs(self):
        from helmetai_fcw.detection import OpenCvYoloOnnxDetector as actual_detector
        with patch("helmetai_fcw.image_check.OpenCvYoloOnnxDetector", actual_detector):
            with self.assertRaisesRegex(RuntimeError, "model loading/inference"):
                self.run_check()
        self.assertFalse(self.output.exists())

    def test_shipped_model_blank_image_is_software_smoke_only(self):
        from helmetai_fcw.detection import OpenCvYoloOnnxDetector as actual_detector
        model = Path(__file__).resolve().parents[1] / "models" / "detector.onnx"
        if not model.is_file():
            self.skipTest("Shipped model unavailable in source-only package.")
        self.assertTrue(cv2.imwrite(str(self.source), np.zeros((480, 640, 3), dtype=np.uint8)))
        with patch("helmetai_fcw.image_check.OpenCvYoloOnnxDetector", actual_detector):
            report = run_image_check(self.source, model, self.output, direction="front")
        self.assertEqual(report["mode"], "offline_image_detection_only")
        self.assertTrue(all(value is None for value in report["metrics"].values()))
        self.assertTrue((self.output / "annotated.jpg").is_file())


if __name__ == "__main__":
    unittest.main()
