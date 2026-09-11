import hashlib
import json
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from helmetai_fcw.detection import Detection
from helmetai_fcw.monocular import MonocularCalibration
from helmetai_fcw.video_runtime import VideoCollisionRiskRuntime
from helmetai_fcw.webcam_runtime import WebcamRuntimeConfig


class FakeCapture:
    def __init__(self, count=3, fps=10.0, opened=True, reported_count=None):
        self.count = count
        self.fps = fps
        self.opened = opened
        self.reported_count = count if reported_count is None else reported_count
        self.position = 0
        self.released = False
        self.seek_ok = True
        self.seek_confirmed = True
        self.width = 640
        self.height = 480

    def isOpened(self):
        return self.opened

    def get(self, prop):
        return {1: self.fps, 2: self.width, 3: self.height,
                4: self.reported_count, 5: self.position}[prop]

    def set(self, prop, value):
        if self.seek_ok and self.seek_confirmed:
            self.position = int(value)
        return self.seek_ok

    def read(self):
        if self.position >= self.count:
            return False, None
        self.position += 1
        return True, SimpleNamespace(shape=(self.height, self.width, 3))

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self, opened=True):
        self.opened = opened
        self.released = False
        self.frames = []
        self.fail_write = False
        self.fail_release = False

    def isOpened(self):
        return self.opened

    def write(self, frame):
        if self.fail_write:
            return False
        self.frames.append(frame)

    def release(self):
        self.released = True
        if self.fail_release:
            raise RuntimeError("encoder release failed")


class FakeCv2:
    CAP_PROP_FPS = 1
    CAP_PROP_FRAME_WIDTH = 2
    CAP_PROP_FRAME_HEIGHT = 3
    CAP_PROP_FRAME_COUNT = 4
    CAP_PROP_POS_FRAMES = 5
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self, capture=None, writer=None, key=-1):
        self.capture = capture or FakeCapture()
        self.writer = writer or FakeWriter()
        self.key = key
        self.texts = []
        self.rectangles = []
        self.waits = []
        self.destroyed = False

    def VideoCapture(self, path):
        return self.capture

    def VideoWriter(self, *args):
        return self.writer

    def VideoWriter_fourcc(self, *args):
        return 0

    def getNumThreads(self):
        return 2

    def rectangle(self, frame, start, end, colour, thickness):
        self.rectangles.append((colour, thickness))

    def putText(self, frame, text, *args):
        self.texts.append(text)

    def imshow(self, *args):
        pass

    def waitKey(self, ms):
        self.waits.append(ms)
        return self.key

    def destroyAllWindows(self):
        self.destroyed = True


class VideoRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.input = self.directory / "input.mp4"
        self.input.write_bytes(b"fake input remains untouched")
        self.report = self.directory / "evidence"
        self.output = self.directory / "annotated.mp4"
        self.config = WebcamRuntimeConfig(0, 640, 480, 20, "front",
            MonocularCalibration(600.0, 320.0, 640, 480, calibrated=False), "unused.onnx")
        self.car = Detection("", "car", 0.95, 250.0, 150.0, 120.0, 100.0)
        self.scene = Detection("", "traffic_light", 0.9, 100.0, 50.0, 30.0, 60.0)
        self.detector = Mock()
        self.detector.detect.return_value = [self.car, self.scene]

    def runtime(self, **kwargs):
        return VideoCollisionRiskRuntime(self.config, self.input, report_dir=self.report, **kwargs)

    def run_fake(self, runtime, cv2=None, preview=False, failure=None):
        cv2 = cv2 or FakeCv2()
        with patch.dict("sys.modules", {"cv2": cv2}), patch(
            "helmetai_fcw.video_runtime.OpenCvYoloOnnxDetector", return_value=self.detector,
            side_effect=failure,
        ) as constructor:
            result = runtime.run(preview=preview)
        return result, cv2, constructor

    def records(self):
        return [json.loads(line) for line in (self.report / "frames.jsonl").read_text(encoding="utf-8").splitlines()]

    def summary(self):
        return json.loads((self.report / "summary.json").read_text(encoding="utf-8"))

    def test_eof_report_counts_every_object_and_omits_uncalibrated_metrics(self):
        runtime = self.runtime(output_path=self.output)
        result, cv2, _ = self.run_fake(runtime)
        self.assertEqual(result, 3)
        self.assertTrue(cv2.capture.released)
        self.assertTrue(cv2.writer.released)
        summary = self.summary()
        self.assertEqual(summary["status"], "eof")
        self.assertTrue(summary["completed"])
        self.assertEqual(summary["frame_count"], 3)
        self.assertEqual(summary["detection_count"], 6)
        self.assertEqual(summary["unique_track_count"], 2)
        self.assertEqual(summary["label_counts"], {"car": 3, "traffic_light": 3})
        self.assertEqual(summary["system_status_counts"], {"degraded": 3, "not_evaluated": 3})
        self.assertEqual(summary["source_fps"], 10.0)
        self.assertGreater(summary["processing_fps"], 0)
        self.assertFalse(summary["metric_verified"])
        self.assertFalse(summary["encoded_output_verified"])
        self.assertEqual(summary["annotated_frames_submitted"], 3)
        self.assertEqual(summary["provenance"]["input_video"]["sha256"],
                         hashlib.sha256(self.input.read_bytes()).hexdigest())
        self.assertIsNone(summary["provenance"]["detector_model"]["sha256"])
        self.assertIn("FileNotFoundError", summary["provenance"]["detector_model"]["unavailable_reason"])
        records = self.records()
        self.assertEqual([r["timestamp_s"] for r in records], [0, 0.1, 0.2])
        for record in records:
            self.assertEqual(record["detection_count"], 2)
            for obj in record["objects"]:
                self.assertIsNone(obj["physical_estimates"])
                self.assertFalse(obj["metric_verified"])
                self.assertTrue(obj["pipeline_assessment"]["reasons"])
        self.assertIn("range_calibration_or_frame_invalid", records[0]["objects"][0]["pipeline_assessment"]["reasons"])
        self.assertIn("not an all-clear", (self.report / "summary.md").read_text())
        self.assertFalse(any("TTC" in text for text in cv2.texts))
        self.assertNotIn((80, 180, 80), [colour for colour, _ in cv2.rectangles])
        self.assertEqual(self.input.read_bytes(), b"fake input remains untouched")

    def test_preview_stop_counts_first_frame_and_uses_one_ms_wait(self):
        result, cv2, _ = self.run_fake(self.runtime(), FakeCv2(key=ord("q")), preview=True)
        self.assertEqual(result, 1)
        self.assertEqual(cv2.waits, [1])
        self.assertTrue(cv2.destroyed)
        self.assertEqual(self.summary()["status"], "user_stop")
        self.assertFalse(self.summary()["completed"])

    def test_start_offset_and_frame_limit_use_source_timeline(self):
        result, _, constructor = self.run_fake(self.runtime(start_seconds=0.15, max_frames=2, cpu_threads=2),
                                              FakeCv2(capture=FakeCapture(count=8)))
        self.assertEqual(result, 2)
        self.assertEqual([r["source_frame_index"] for r in self.records()], [2, 3])
        self.assertEqual([r["timestamp_s"] for r in self.records()], [0.2, 0.3])
        self.assertEqual(self.summary()["status"], "max_frames")
        self.assertFalse(self.summary()["completed"])
        self.assertEqual(constructor.call_args.kwargs["cpu_threads"], 2)

    def test_max_limit_exact_end_is_not_claimed_eof_without_read(self):
        self.run_fake(self.runtime(max_frames=3))
        self.assertEqual(self.summary()["status"], "max_frames")

    def test_empty_detections_still_have_frame_evidence(self):
        self.detector.detect.return_value = []
        self.run_fake(self.runtime())
        self.assertEqual(len(self.records()), 3)
        self.assertEqual(self.summary()["detection_count"], 0)
        self.assertEqual(self.summary()["frames_with_detections"], 0)

    def test_rear_pipeline_status_and_reason_are_retained(self):
        self.config = replace(self.config, camera_role="rear")
        self.run_fake(self.runtime())
        assessment = self.records()[0]["objects"][0]["pipeline_assessment"]
        self.assertEqual(assessment["direction"], "rear")
        self.assertIn("rear_calibration_or_frame_invalid", assessment["reasons"])
        self.assertTrue(self.records()[0]["objects"][0]["object_id"].startswith("rear-"))

    def test_demo_with_declared_calibration_still_has_no_physical_metrics(self):
        self.config = replace(self.config, demonstration_only=True,
                              calibration=replace(self.config.calibration, calibrated=True))
        _, cv2, _ = self.run_fake(self.runtime())
        self.assertEqual(self.summary()["mode"], "illustrative_demo")
        for record in self.records():
            for obj in record["objects"]:
                self.assertIsNone(obj["physical_estimates"])
                self.assertIn("illustrative", obj["pipeline_assessment"]["interpretation"])
        self.assertTrue(any("ILLUSTRATIVE DEMO" in text for text in cv2.texts))
        self.assertFalse(any("TTC" in text for text in cv2.texts))
        self.assertNotIn((80, 180, 80), [colour for colour, _ in cv2.rectangles])

    def test_calibrated_estimates_are_unverified_and_nonfinite_ttc_is_null(self):
        self.config = replace(self.config, calibration=replace(self.config.calibration, calibrated=True))
        self.run_fake(self.runtime())
        estimate = self.records()[2]["objects"][0]["physical_estimates"]
        self.assertGreater(estimate["longitudinal_m"], 0)
        self.assertIsNone(estimate["raw_ttc_s"])
        self.assertFalse(estimate["metric_verified"])
        self.assertNotIn("Infinity", (self.report / "frames.jsonl").read_text())

    def test_capture_open_failure_releases_and_finalizes_failed(self):
        cv2 = FakeCv2(capture=FakeCapture(opened=False))
        with self.assertRaisesRegex(RuntimeError, "could not be opened"):
            self.run_fake(self.runtime(), cv2)
        self.assertTrue(cv2.capture.released)
        self.assertEqual(self.summary()["status"], "failed")
        self.assertFalse(self.summary()["completed"])

    def test_invalid_fps_is_not_silently_replaced(self):
        for fps in (0, -2, float("nan"), float("inf")):
            with self.subTest(fps=fps):
                directory = self.directory / f"fps{str(fps)}"
                runtime = VideoCollisionRiskRuntime(self.config, self.input, report_dir=directory)
                cv2 = FakeCv2(capture=FakeCapture(fps=fps))
                with self.assertRaisesRegex(ValueError, "Source FPS"):
                    self.run_fake(runtime, cv2)
                self.assertTrue(cv2.capture.released)
                report = json.loads((directory / "summary.json").read_text())
                self.assertEqual(report["status"], "failed")

    def test_explicit_fps_override_is_recorded(self):
        self.run_fake(self.runtime(source_fps_override=5), FakeCv2(capture=FakeCapture(fps=float("nan"))))
        self.assertIsNone(self.summary()["source_fps"])
        self.assertEqual(self.summary()["effective_source_fps"], 5)
        self.assertEqual(self.records()[1]["timestamp_s"], 0.2)

    def test_detector_init_failure_releases_capture(self):
        cv2 = FakeCv2()
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            self.run_fake(self.runtime(), cv2, failure=RuntimeError("model failed"))
        self.assertTrue(cv2.capture.released)
        self.assertEqual(self.summary()["status"], "failed")

    def test_writer_open_failure_releases_both_resources(self):
        cv2 = FakeCv2(writer=FakeWriter(opened=False))
        with self.assertRaisesRegex(RuntimeError, "could not be created"):
            self.run_fake(self.runtime(output_path=self.output), cv2)
        self.assertTrue(cv2.capture.released)
        self.assertTrue(cv2.writer.released)
        self.assertEqual(self.summary()["status"], "failed")
        self.assertTrue(self.summary()["output_may_be_partial"])

    def test_write_failure_is_failed_not_eof(self):
        cv2 = FakeCv2()
        cv2.writer.fail_write = True
        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            self.run_fake(self.runtime(output_path=self.output), cv2)
        self.assertTrue(cv2.writer.released)
        self.assertEqual(self.summary()["frame_count"], 0)
        self.assertEqual(self.summary()["status"], "failed")

    def test_partial_inference_failure_preserves_prior_frame_evidence(self):
        self.detector.detect.side_effect = [[self.car], RuntimeError("inference failed")]
        cv2 = FakeCv2()
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            self.run_fake(self.runtime(output_path=self.output), cv2)
        self.assertTrue(cv2.capture.released)
        self.assertTrue(cv2.writer.released)
        self.assertEqual(len(self.records()), 1)
        self.assertEqual(self.summary()["frame_count"], 1)
        self.assertEqual(self.summary()["status"], "failed")
        self.assertFalse(self.summary()["completed"])

    def test_premature_decoder_end_is_failure(self):
        cv2 = FakeCv2(capture=FakeCapture(count=1, reported_count=3))
        with self.assertRaisesRegex(RuntimeError, "before the reported end"):
            self.run_fake(self.runtime(), cv2)
        self.assertEqual(self.summary()["status"], "failed")
        self.assertEqual(self.summary()["frame_count"], 1)

    def test_unknown_frame_count_read_stop_does_not_certify_eof(self):
        self.run_fake(self.runtime(), FakeCv2(capture=FakeCapture(count=1, reported_count=0)))
        self.assertEqual(self.summary()["status"], "read_stop_unconfirmed")
        self.assertFalse(self.summary()["completed"])
        self.assertIsNone(self.summary()["reported_source_frame_count"])

    def test_aggregate_cap_is_explicit_without_dropping_frame_evidence(self):
        with patch("helmetai_fcw.video_report.VideoEvidenceReport.TRACK_SUMMARY_LIMIT", 1):
            self.run_fake(self.runtime())
        summary = self.summary()
        self.assertTrue(summary["track_summary_truncated"])
        self.assertIsNone(summary["unique_track_count"])
        self.assertEqual(summary["reported_track_count"], 1)
        self.assertEqual(summary["omitted_track_observation_count"], 3)
        self.assertEqual(summary["detection_count"], 6)
        self.assertEqual(sum(len(record["objects"]) for record in self.records()), 6)

    def test_summary_write_error_invalidates_earlier_success_claim(self):
        original_write = Path.write_text
        failed = []

        def fail_markdown_once(path, data, *args, **kwargs):
            if path.name == "summary.md" and not failed:
                failed.append(True)
                raise OSError("test disk error")
            return original_write(path, data, *args, **kwargs)

        with patch.object(Path, "write_text", fail_markdown_once):
            with self.assertRaisesRegex(OSError, "test disk error"):
                self.run_fake(self.runtime())
        self.assertEqual(self.summary()["status"], "failed")
        self.assertFalse(self.summary()["completed"])
        self.assertIn("Report finalization failed", self.summary()["error"])

    def test_release_failure_cannot_claim_completed(self):
        cv2 = FakeCv2()
        cv2.writer.fail_release = True
        with self.assertRaisesRegex(RuntimeError, "encoder release failed"):
            self.run_fake(self.runtime(output_path=self.output), cv2)
        self.assertTrue(cv2.capture.released)
        self.assertEqual(self.summary()["status"], "failed")
        self.assertFalse(self.summary()["completed"])

    def test_existing_outputs_and_report_are_not_overwritten(self):
        self.output.write_bytes(b"existing annotated output")
        with self.assertRaises(FileExistsError):
            self.run_fake(self.runtime(output_path=self.output))
        self.assertEqual(self.output.read_bytes(), b"existing annotated output")
        with self.assertRaisesRegex(ValueError, "alias"):
            self.run_fake(self.runtime(output_path=self.input))
        self.report.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_fake(self.runtime())

    def test_annotated_output_cannot_collide_with_report_artifacts(self):
        for target in (self.report, self.report / "frames.jsonl", self.report / "summary.json", self.report / "summary.md"):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "conflicts"):
                self.run_fake(self.runtime(output_path=target))
        self.assertFalse(self.report.exists())

    def test_calibrated_resolution_mismatch_fails_without_metrics(self):
        self.config = replace(self.config, calibration=replace(self.config.calibration, calibrated=True, image_width_px=1280))
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            self.run_fake(self.runtime())
        self.assertEqual(self.summary()["status"], "failed")
        self.assertEqual(self.records(), [])

    def test_seek_must_be_confirmed(self):
        cv2 = FakeCv2()
        cv2.capture.seek_confirmed = False
        with self.assertRaisesRegex(RuntimeError, "confirm"):
            self.run_fake(self.runtime(start_seconds=0.1), cv2)
        self.assertTrue(cv2.capture.released)
        self.assertEqual(self.summary()["status"], "failed")

    def test_start_at_end_and_empty_video_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "at or beyond"):
            self.run_fake(self.runtime(start_seconds=0.3))
        other_report = self.directory / "empty"
        runtime = VideoCollisionRiskRuntime(self.config, self.input, report_dir=other_report)
        with self.assertRaisesRegex(RuntimeError, "No source frame"):
            self.run_fake(runtime, FakeCv2(capture=FakeCapture(count=0)))

    def test_option_validation(self):
        bad = [{"max_frames": 0}, {"max_frames": 100001}, {"max_frames": True}, {"max_frames": 1.5},
               {"start_seconds": -1}, {"start_seconds": float("nan")}, {"start_seconds": float("inf")},
               {"cpu_threads": 0}, {"cpu_threads": 257}, {"cpu_threads": True},
               {"source_fps_override": 0}, {"source_fps_override": float("nan")}]
        for kwargs in bad:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.runtime(**kwargs)


if __name__ == "__main__":
    unittest.main()
