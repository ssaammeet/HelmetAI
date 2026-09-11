"""Replay recorded video and retain explainable, unverified offline evidence."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import sys
import time
from typing import Any

from .detection import OpenCvYoloOnnxDetector
from .monocular import MonocularCalibration, MonocularRangeEstimator
from .pipeline import ForwardCollisionPipeline
from .rear_pipeline import RearCollisionPipeline
from .tracking import IoUTracker
from .video_report import VideoEvidenceReport
from .webcam_runtime import WebcamRuntimeConfig


class VideoCollisionRiskRuntime:
    """Recorded-video evaluation; no hardware or physical accuracy guarantee."""

    def __init__(
        self, config: WebcamRuntimeConfig, video_path: str | Path,
        output_path: str | Path | None = None, *, report_dir: str | Path | None = None,
        max_frames: int | None = None, start_seconds: float = 0.0,
        cpu_threads: int | None = None, source_fps_override: float | None = None,
    ) -> None:
        if max_frames is not None and (type(max_frames) is not int or not 1 <= max_frames <= 100_000):
            raise ValueError("max_frames must be an integer from 1 to 100000.")
        if not math.isfinite(start_seconds) or start_seconds < 0:
            raise ValueError("start_seconds must be finite and nonnegative.")
        if cpu_threads is not None and (type(cpu_threads) is not int or not 1 <= cpu_threads <= 256):
            raise ValueError("cpu_threads must be an integer from 1 to 256.")
        if source_fps_override is not None and (not math.isfinite(source_fps_override) or source_fps_override <= 0):
            raise ValueError("source_fps_override must be finite and positive.")
        self.config = config
        self.video_path = Path(video_path)
        self.output_path = Path(output_path) if output_path else None
        self.report_dir = Path(report_dir) if report_dir else None
        self.max_frames = max_frames
        self.start_seconds = start_seconds
        self.cpu_threads = cpu_threads
        self.source_fps_override = source_fps_override
        self.last_summary: dict[str, Any] | None = None

    def run(self, preview: bool = True) -> int:
        if not self.video_path.is_file():
            raise FileNotFoundError(f"Video file was not found: {self.video_path}")
        if self.output_path:
            if self.output_path.resolve() == self.video_path.resolve():
                raise ValueError("Annotated output must not alias the input video.")
            if self.output_path.exists():
                raise FileExistsError(f"Refusing to overwrite annotated output: {self.output_path}")
        if self.report_dir and self.report_dir.exists():
            raise FileExistsError(f"Report directory must be new: {self.report_dir}")
        if self.report_dir and self.output_path:
            report_path, output_path = self.report_dir.resolve(), self.output_path.resolve()
            reserved = {report_path / name for name in ("frames.jsonl", "summary.json", "summary.md")}
            if output_path == report_path or output_path in report_path.parents or output_path in reserved:
                raise ValueError("Annotated output conflicts with the report directory or evidence files.")

        report = VideoEvidenceReport(self.report_dir)
        capture = writer = cv2 = None
        failure: BaseException | None = None
        frame_count = 0
        annotated_frames_submitted = 0
        loop_started = None
        loop_elapsed = 0.0
        start_frame = 0
        source_fps = fps = None
        frame_width = frame_height = None
        actual_threads = None
        total_frames = None
        input_fingerprint = model_fingerprint = None
        opencv_version = None
        status = "failed"
        try:
            try:
                import cv2  # type: ignore
            except ImportError as exc:
                raise RuntimeError("Recorded video requires OpenCV. Run scripts/setup_webcam.ps1 first.") from exc
            opencv_version = getattr(cv2, "__version__", None)
            input_fingerprint = self._fingerprint(self.video_path)
            model_fingerprint = self._fingerprint(Path(self.config.model_path))
            capture = cv2.VideoCapture(str(self.video_path))
            if not capture.isOpened():
                raise RuntimeError(f"Video could not be opened: {self.video_path}")
            raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
            source_fps = raw_fps if math.isfinite(raw_fps) else None
            fps = self.source_fps_override if self.source_fps_override is not None else raw_fps
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError("Source FPS is missing or invalid; provide an explicit positive source_fps_override.")
            frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if frame_width <= 0 or frame_height <= 0:
                raise ValueError("Video reports invalid frame dimensions.")
            calibration = self._calibration_for_video(frame_width, frame_height)
            if calibration.calibrated and not self.config.demonstration_only and (
                calibration.image_width_px != frame_width or calibration.image_height_px != frame_height
            ):
                raise ValueError("Video dimensions differ from the calibration; use a matching calibration.")
            total_raw = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = int(total_raw) if math.isfinite(total_raw) and total_raw > 0 else None
            start_frame = math.ceil(self.start_seconds * fps)
            if total_frames is not None and start_frame >= total_frames:
                raise ValueError("start_seconds is at or beyond the source video end.")
            if start_frame:
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame):
                    raise RuntimeError("Video backend refused the requested start-frame seek.")
                actual_position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
                if not math.isfinite(actual_position) or abs(actual_position - start_frame) > 0.5:
                    raise RuntimeError("Video backend did not confirm the requested start-frame seek.")
            detector = OpenCvYoloOnnxDetector(
                self.config.model_path, input_size_px=self.config.detector_input_size_px,
                confidence_threshold=self.config.detector_confidence, cpu_threads=self.cpu_threads,
            )
            actual_threads = int(cv2.getNumThreads()) if hasattr(cv2, "getNumThreads") else None
            estimator = MonocularRangeEstimator(calibration)
            tracker = IoUTracker(id_prefix=self.config.camera_role)
            pipeline = RearCollisionPipeline() if self.config.camera_role == "rear" else ForwardCollisionPipeline()
            if self.output_path:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive reservation closes the common accidental-overwrite race.
                with self.output_path.open("xb"):
                    pass
                writer = cv2.VideoWriter(str(self.output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
                if not writer.isOpened():
                    raise RuntimeError(f"Annotated video could not be created: {self.output_path}")
            window_name = f"HelmetAI {self.config.camera_role.upper()} Offline - Q: Exit"
            loop_started = time.perf_counter()
            while True:
                if self.max_frames is not None and frame_count >= self.max_frames:
                    status = "max_frames"
                    break
                ok, frame = capture.read()
                if not ok or frame is None:
                    if frame_count == 0:
                        raise RuntimeError("No source frame could be decoded at the requested start.")
                    if total_frames is not None and start_frame + frame_count < total_frames:
                        raise RuntimeError("Video decoding stopped before the reported end of the source.")
                    # OpenCV returns the same false result for EOF and a decode
                    # failure. Without frame-count metadata, do not certify EOF.
                    status = "eof" if total_frames is not None else "read_stop_unconfirmed"
                    break
                if tuple(frame.shape[:2]) != (frame_height, frame_width):
                    raise RuntimeError("Decoded frame dimensions changed during the run.")
                source_index = start_frame + frame_count
                timestamp_s = source_index / fps
                detections = tracker.update(timestamp_s, detector.detect(frame))
                objects = []
                for detection in detections:
                    observation = assessment = None
                    if estimator.supports_range(detection):
                        observation = estimator.estimate(timestamp_s, detection)
                        assessment = pipeline.process(observation)
                        self._draw_detection(cv2, frame, detection, observation.longitudinal_m, assessment)
                    else:
                        self._draw_scene_detection(cv2, frame, detection)
                    objects.append(self._object_record(detection, observation, assessment))
                self._draw_banner(cv2, frame)
                if writer is not None:
                    if writer.write(frame) is False or not writer.isOpened():
                        raise RuntimeError("Annotated video writer failed during frame output.")
                    annotated_frames_submitted += 1
                report.add_frame({
                    "schema_version": 1, "frame_index": frame_count, "source_frame_index": source_index,
                    "timestamp_s": timestamp_s, "timestamp_basis": "source_frame_index/effective_source_fps",
                    "camera_role": self.config.camera_role, "mode": self._mode(), "metric_verified": False,
                    "detection_count": len(objects), "objects": objects,
                })
                frame_count += 1  # Count the rendered/written frame even when Q stops preview.
                if preview:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        status = "user_stop"
                        break
        except BaseException as exc:
            failure = exc
        finally:
            if loop_started is not None:
                loop_elapsed = max(0.0, time.perf_counter() - loop_started)
            for resource in (capture, writer):
                if resource is not None:
                    try:
                        resource.release()
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
            if preview and cv2 is not None:
                try:
                    cv2.destroyAllWindows()
                except BaseException as exc:
                    if failure is None:
                        failure = exc
            if failure is not None:
                status = "failed"
            run = {
                "status": status, "completed": status == "eof", "mode": self._mode(),
                "camera_role": self.config.camera_role,
                "input_video": str(self.video_path.resolve()),
                "annotated_video": str(self.output_path.resolve()) if self.output_path else None,
                "output_may_be_partial": bool(self.output_path and status == "failed"),
                "annotated_frames_submitted": annotated_frames_submitted,
                "annotated_video_verified": False,
                "encoded_output_verified": False,
                "annotation_verification": "not_redecoded" if self.output_path else "not_requested",
                "calibration_declared_valid": self.config.calibration.calibrated,
                "demonstration_only": self.config.demonstration_only,
                "source_fps": source_fps, "effective_source_fps": fps if fps and math.isfinite(fps) else None,
                "source_fps_override": self.source_fps_override,
                "reported_source_frame_count": total_frames,
                "frame_width_px": frame_width, "frame_height_px": frame_height,
                "start_seconds_requested": self.start_seconds, "start_source_frame_index": start_frame,
                "max_frames": self.max_frames, "cpu_threads_requested": self.cpu_threads,
                "opencv_cpu_threads": actual_threads,
                "processing_elapsed_s": loop_elapsed,
                "processing_fps": frame_count / loop_elapsed if frame_count and loop_elapsed > 0 else None,
                "timestamp_basis": "source_frame_index/effective_source_fps; constant-FPS assumption",
                "error": f"{type(failure).__name__}: {failure}" if failure else None,
                "provenance": {
                    "input_video": input_fingerprint, "detector_model": model_fingerprint,
                    "opencv_version": opencv_version, "python_version": sys.version.split()[0],
                    "hash_timing": "before_capture_and_inference",
                    "detector_input_size_px": self.config.detector_input_size_px,
                    "detector_confidence_threshold": self.config.detector_confidence,
                },
            }
            try:
                self.last_summary = report.finalize(run)
            except BaseException as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure
        return frame_count

    def _mode(self) -> str:
        if self.config.demonstration_only:
            return "illustrative_demo"
        return "calibrated_estimate_unverified" if self.config.calibration.calibrated else "uncalibrated"

    @staticmethod
    def _fingerprint(path: Path) -> dict[str, Any]:
        """Hash existing local inputs in chunks without loading the video into RAM."""
        result: dict[str, Any] = {"path": str(path.resolve()), "sha256": None, "size_bytes": None, "unavailable_reason": None}
        try:
            result["size_bytes"] = path.stat().st_size
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            result["sha256"] = digest.hexdigest()
        except OSError as exc:
            result["unavailable_reason"] = f"{type(exc).__name__}: {exc}"
        return result

    def _object_record(self, detection: Any, observation: Any, assessment: Any) -> dict[str, Any]:
        physical = None
        if observation is not None and self.config.calibration.calibrated and not self.config.demonstration_only:
            names = ("raw_ttc_s", "conservative_ttc_s", "required_relative_decel_mps2", "effective_distance_m", "closing_speed_mps")
            physical = {name: self._finite(getattr(assessment, name)) for name in names}
            physical.update({"longitudinal_m": self._finite(observation.longitudinal_m),
                             "lateral_m": self._finite(observation.lateral_m),
                             "range_sigma_m": self._finite(observation.range_sigma_m),
                             "metric_verified": False, "provenance": "calibrated_monocular_width_model_estimate"})
        pipeline_record = {
            "evaluated": assessment is not None,
            "direction": assessment.direction.value if assessment is not None else self.config.camera_role,
            "system_status": assessment.system_status.value if assessment is not None else "not_evaluated",
            "level": assessment.level.name if assessment is not None else None,
            "reasons": list(assessment.reasons) if assessment is not None else ["no_supported_physical_width_model_or_invalid_box"],
            "in_path": assessment.in_path if assessment is not None else None,
            "interpretation": "illustrative_only_not_physical_risk" if self.config.demonstration_only else "pipeline_output_not_safety_validation",
        }
        return {
            "object_id": detection.object_id, "label": detection.label, "confidence": self._finite(detection.confidence),
            "bbox_px": {"x": self._finite(detection.x_px), "y": self._finite(detection.y_px),
                        "width": self._finite(detection.width_px), "height": self._finite(detection.height_px)},
            "range_supported": observation is not None, "mode": self._mode(), "metric_verified": False,
            "pipeline_assessment": pipeline_record, "physical_estimates": physical,
        }

    @staticmethod
    def _finite(value: float | None) -> float | None:
        return value if value is not None and math.isfinite(value) else None

    def _calibration_for_video(self, frame_width: int, frame_height: int) -> MonocularCalibration:
        """Only the explicitly illustrative profile may be automatically scaled."""
        calibration = self.config.calibration
        if not self.config.demonstration_only or frame_width <= 0 or frame_height <= 0:
            return calibration
        scale_x = frame_width / calibration.image_width_px
        return replace(calibration, focal_length_px=calibration.focal_length_px * scale_x,
                       principal_point_x_px=calibration.principal_point_x_px * scale_x,
                       image_width_px=frame_width, image_height_px=frame_height)

    def _draw_detection(self, cv2: Any, frame: Any, detection: Any, distance_m: float, assessment: Any) -> None:
        colour = (190, 190, 190)
        physical_allowed = self.config.calibration.calibrated and not self.config.demonstration_only
        if physical_allowed and assessment.system_status.value == "ready":
            colour = {"NONE": colour, "ADVISORY": (0, 210, 255), "WARNING": (0, 140, 255), "CRITICAL": (0, 0, 255)}[assessment.level.name]
        label = f"{detection.object_id} {detection.label} | {assessment.system_status.value.upper()}"
        if self.config.demonstration_only:
            label += f" | DEMO {assessment.level.name}"
        elif not physical_allowed:
            label += " | UNCALIBRATED"
        elif assessment.system_status.value == "ready":
            label += f" | {assessment.level.name} | est {distance_m:.1f}m"
            if self._finite(assessment.conservative_ttc_s) is not None:
                label += f" TTC {assessment.conservative_ttc_s:.1f}s"
        self._draw_box(cv2, frame, detection, label, colour)

    @staticmethod
    def _draw_box(cv2: Any, frame: Any, detection: Any, label: str, colour: tuple[int, int, int]) -> None:
        x, y = int(detection.x_px), int(detection.y_px)
        cv2.rectangle(frame, (x, y), (x + int(detection.width_px), y + int(detection.height_px)), colour, 2)
        cv2.putText(frame, label, (max(0, x), max(52, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.43, colour, 1)

    @staticmethod
    def _draw_scene_detection(cv2: Any, frame: Any, detection: Any) -> None:
        VideoCollisionRiskRuntime._draw_box(cv2, frame, detection,
            f"{detection.object_id} {detection.label} | SCENE ONLY", (190, 190, 190))

    def _draw_banner(self, cv2: Any, frame: Any) -> None:
        if self.config.demonstration_only:
            label = "ILLUSTRATIVE DEMO - NOT PHYSICAL RISK"
        elif self.config.calibration.calibrated:
            label = "CALIBRATED ESTIMATES - NOT VERIFIED"
        else:
            label = "UNCALIBRATED - NO METRIC RISK"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (25, 25, 25), -1)
        cv2.putText(frame, f"HELMETAI {self.config.camera_role.upper()} | {label}", (8, 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
