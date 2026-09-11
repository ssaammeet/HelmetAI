"""Desktop webcam runtime for testing the FCW pipeline before Raspberry Pi use."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .detection import OpenCvYoloOnnxDetector
from .monocular import MonocularCalibration, MonocularRangeEstimator
from .pipeline import ForwardCollisionPipeline
from .rear_pipeline import RearCollisionPipeline
from .tracking import IoUTracker


def _json_boolean(data: dict[str, Any], key: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a JSON boolean (true or false), not a string/number.")
    return value


def _webcam_detection_label(config: "WebcamRuntimeConfig", detection: Any, observation: Any, assessment: Any) -> tuple[str, tuple[int, int, int]]:
    """Keep unknown/degraded range visibly separate from a risk decision."""
    neutral = (190, 190, 190)
    prefix = f"{assessment.direction.value.upper()} | {detection.label} {detection.confidence:.2f}"
    if config.demonstration_only:
        return f"{prefix} | DEMO {assessment.level.name} | NO VERIFIED RANGE", neutral
    if not config.calibration.calibrated or assessment.system_status.value == "degraded":
        return f"{prefix} | UNVERIFIED / {assessment.system_status.value.upper()}", neutral
    colour = {"NONE": neutral, "ADVISORY": (0, 210, 255), "WARNING": (0, 140, 255), "CRITICAL": (0, 0, 255)}[assessment.level.name]
    text = f"{prefix} | {assessment.level.name} | est {observation.longitudinal_m:.1f}m"
    if assessment.conservative_ttc_s is not None:
        text += f" | est TTC {assessment.conservative_ttc_s:.1f}s"
    return text, colour


@dataclass(frozen=True)
class WebcamRuntimeConfig:
    camera_index: int
    width_px: int
    height_px: int
    fps: int
    camera_role: str
    calibration: MonocularCalibration
    model_path: str
    detector_input_size_px: int = 640
    detector_confidence: float = 0.45
    demonstration_only: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "WebcamRuntimeConfig":
        config_path = Path(path)
        data: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        camera_data = data["webcam"]
        width_px = int(camera_data.get("width_px", 640))
        height_px = int(camera_data.get("height_px", 480))
        calibration_data = data["front_calibration"]
        camera_role = str(data.get("camera_role", "front")).strip().lower()
        if camera_role not in {"front", "rear"}:
            raise ValueError("camera_role must be either 'front' or 'rear'.")
        calibration = MonocularCalibration(
            focal_length_px=float(calibration_data["focal_length_px"]),
            principal_point_x_px=float(calibration_data.get("principal_point_x_px", width_px / 2)),
            image_width_px=width_px,
            image_height_px=height_px,
            pixel_width_sigma_px=float(calibration_data.get("pixel_width_sigma_px", 2.0)),
            lateral_sign=float(calibration_data.get("lateral_sign", 1.0 if camera_role == "front" else -1.0)),
            calibrated=_json_boolean(calibration_data, "calibrated"),
        )
        model_path = Path(str(data["detector"]["model_path"]))
        if not model_path.is_absolute():
            model_path = (config_path.parent / model_path).resolve()
        return cls(
            camera_index=int(camera_data.get("camera_index", 0)),
            width_px=width_px,
            height_px=height_px,
            fps=int(camera_data.get("fps", 20)),
            camera_role=camera_role,
            calibration=calibration,
            model_path=str(model_path),
            detector_input_size_px=int(data["detector"].get("input_size_px", 640)),
            detector_confidence=float(data["detector"].get("confidence", 0.45)),
            demonstration_only=_json_boolean(data, "demonstration_only"),
        )


class WebcamForwardCollisionRuntime:
    """Visual bench-test runner using a laptop internal or USB webcam."""

    def __init__(self, config: WebcamRuntimeConfig) -> None:
        self.config = config

    def run(self, max_frames: int | None = None) -> None:  # pragma: no cover - requires live camera/model
        try:
            import cv2  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Webcam testi için OpenCV gerekli. scripts/setup_webcam.ps1 çalıştırın.") from exc
        detector = OpenCvYoloOnnxDetector(
            self.config.model_path,
            input_size_px=self.config.detector_input_size_px,
            confidence_threshold=self.config.detector_confidence,
        )
        estimator = MonocularRangeEstimator(self.config.calibration)
        tracker = IoUTracker()
        pipeline = RearCollisionPipeline() if self.config.camera_role == "rear" else ForwardCollisionPipeline()
        capture = cv2.VideoCapture(self.config.camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width_px)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height_px)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if not capture.isOpened():
            raise RuntimeError(f"Webcam açılamadı: camera_index={self.config.camera_index}")
        frame_count = 0
        window_name = f"HelmetAI {self.config.camera_role.upper()} Risk Webcam Test - Q: Cikis"
        try:
            while max_frames is None or frame_count < max_frames:
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Webcam karesi okunamadı.")
                timestamp_s = time.monotonic()
                detections = tracker.update(timestamp_s, detector.detect(frame))
                assessments = []
                scene_detections = []
                for detection in detections:
                    if not estimator.supports_range(detection):
                        scene_detections.append(detection)
                        continue
                    observation = estimator.estimate(timestamp_s, detection)
                    assessment = pipeline.process(observation)
                    assessments.append((detection, observation, assessment))
                for detection, observation, assessment in assessments:
                    risk_text, colour = _webcam_detection_label(self.config, detection, observation, assessment)
                    x, y = int(detection.x_px), int(detection.y_px)
                    width, height = int(detection.width_px), int(detection.height_px)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), colour, 2)
                    cv2.putText(frame, risk_text, (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
                for detection in scene_detections:
                    x, y = int(detection.x_px), int(detection.y_px)
                    width, height = int(detection.width_px), int(detection.height_px)
                    colour = (190, 190, 190)
                    cv2.rectangle(frame, (x, y), (x + width, y + height), colour, 2)
                    cv2.putText(
                        frame,
                        f"SCENE | {detection.label} | metric range not calibrated",
                        (x, max(22, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.48,
                        colour,
                        2,
                    )
                if self.config.demonstration_only:
                    status = "DEMO ONLY - NOT ROAD VALIDATED"
                elif not self.config.calibration.calibrated:
                    status = "UNCALIBRATED - NO VALIDATED RISK ALERT"
                else:
                    status = "BENCH TEST - NOT ROAD VALIDATED"
                cv2.putText(frame, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                frame_count += 1
        finally:
            capture.release()
            cv2.destroyAllWindows()
