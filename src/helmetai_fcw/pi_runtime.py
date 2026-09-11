"""Raspberry Pi front and rear collision-risk runtime and configuration loader."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Iterable

from .awareness import BlindSpotMonitor, CoverageAwareRiskMap, TurnIntent
from .alert_actuator import AlertActuatorConfig, create_alert_actuator
from .detection import OpenCvYoloOnnxDetector
from .filtering import AlphaBetaTrackFilter
from .monocular import MonocularCalibration, MonocularRangeEstimator
from .models import AlertLevel, RiskAssessment, RiskDirection, SystemStatus
from .performance import DirectionalPerformanceMonitor, EdgePerformanceMonitor
from .pipeline import ForwardCollisionPipeline
from .rear_pipeline import RearCollisionPipeline
from .rpi_camera import PiCameraConfig, PiDualCameraRig
from .tracking import IoUTracker

SOFTWARE_RELEASE = "pi-software-fix-v6"


@dataclass(frozen=True)
class PiRuntimeConfig:
    front_camera: PiCameraConfig
    rear_camera: PiCameraConfig | None
    front_calibration: MonocularCalibration
    rear_calibration: MonocularCalibration | None
    model_path: str
    detector_input_size_px: int = 640
    detector_confidence: float = 0.45
    detector_cpu_threads: int | None = None
    front_inference_every_n_frames: int = 1
    rear_capture_every_n_frames: int = 3
    tracking_max_gap_s: float = 2.0
    performance_window_frames: int = 60
    latency_budget_ms: float = 150.0
    minimum_risk_fps: float = 5.0
    performance_warmup_frames: int = 60
    # A cached result can bridge an intentionally skipped camera capture, but
    # it must expire quickly rather than becoming a stale rider warning.
    risk_assessment_freshness_s: float = 0.45
    alert_actuator: AlertActuatorConfig = field(default_factory=AlertActuatorConfig)

    def __post_init__(self) -> None:
        """Reject unsafe/ambiguous profiles before any camera or GPIO opens."""

        if self.rear_camera is not None and self.front_camera.camera_index == self.rear_camera.camera_index:
            raise ValueError("front_camera and rear_camera must use different camera_index values.")
        if self.rear_camera is not None and self.rear_calibration is None:
            raise ValueError("rear_camera is enabled but rear_calibration is missing.")
        if self.detector_input_size_px <= 0:
            raise ValueError("detector.input_size_px must be positive.")
        if not 0.0 <= self.detector_confidence <= 1.0:
            raise ValueError("detector.confidence must be between 0 and 1.")
        if self.detector_cpu_threads is not None and self.detector_cpu_threads < 1:
            raise ValueError("detector.cpu_threads must be at least 1 when configured.")
        if self.front_inference_every_n_frames < 1:
            raise ValueError("front_inference_every_n_frames must be at least 1.")
        if self.rear_capture_every_n_frames < 1:
            raise ValueError("rear_capture_every_n_frames must be at least 1.")
        if self.tracking_max_gap_s <= 0:
            raise ValueError("tracking_max_gap_s must be positive.")
        if self.performance_window_frames < 5:
            raise ValueError("performance_window_frames must be at least 5.")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive.")
        if self.minimum_risk_fps <= 0:
            raise ValueError("minimum_risk_fps must be positive.")
        if self.performance_warmup_frames < 1:
            raise ValueError("performance_warmup_frames must be at least 1.")
        if self.risk_assessment_freshness_s <= 0:
            raise ValueError("risk_assessment_freshness_s must be positive.")

    @staticmethod
    def _camera(value: dict[str, Any]) -> PiCameraConfig:
        return PiCameraConfig(
            camera_index=int(value["camera_index"]),
            width_px=int(value.get("width_px", 640)),
            height_px=int(value.get("height_px", 480)),
            fps=int(value.get("fps", 15)),
        )

    @staticmethod
    def _calibration(
        value: dict[str, Any],
        camera: PiCameraConfig,
        default_lateral_sign: float,
        role: str,
    ) -> MonocularCalibration:
        calibrated = value.get("calibrated", False)
        if type(calibrated) is not bool:
            raise ValueError(f"{role}_calibration.calibrated must be a JSON boolean.")
        return MonocularCalibration(
            focal_length_px=float(value["focal_length_px"]),
            principal_point_x_px=float(value.get("principal_point_x_px", camera.width_px / 2)),
            image_width_px=camera.width_px,
            image_height_px=camera.height_px,
            pixel_width_sigma_px=float(value.get("pixel_width_sigma_px", 2.0)),
            lateral_sign=float(value.get("lateral_sign", default_lateral_sign)),
            calibrated=calibrated,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PiRuntimeConfig":
        config_path = Path(path)
        data = json.loads(config_path.read_text(encoding="utf-8"))
        front = cls._camera(data["front_camera"])
        rear_data = data.get("rear_camera")
        rear = None
        if rear_data is not None:
            if not isinstance(rear_data, dict):
                raise ValueError("rear_camera must be a JSON object when configured.")
            rear_enabled = rear_data.get("enabled", True)
            if type(rear_enabled) is not bool:
                raise ValueError("rear_camera.enabled must be a JSON boolean.")
            if rear_enabled:
                rear = cls._camera(rear_data)
        calibration = cls._calibration(
            data["front_calibration"],
            front,
            default_lateral_sign=1.0,
            role="front",
        )
        rear_calibration_data = data.get("rear_calibration")
        if rear is not None and not rear_calibration_data:
            raise ValueError("rear_camera is enabled but rear_calibration is missing.")
        rear_calibration = (
            cls._calibration(
                rear_calibration_data,
                rear,
                default_lateral_sign=-1.0,
                role="rear",
            )
            if rear and rear_calibration_data
            else None
        )
        model_value = str(data["detector"]["model_path"])
        model_path = Path(model_value)
        if not model_path.is_absolute():
            model_path = (config_path.parent / model_path).resolve()
        return cls(
            front_camera=front,
            rear_camera=rear,
            front_calibration=calibration,
            rear_calibration=rear_calibration,
            model_path=str(model_path),
            detector_input_size_px=int(data["detector"].get("input_size_px", 640)),
            detector_confidence=float(data["detector"].get("confidence", 0.45)),
            detector_cpu_threads=(
                None
                if data["detector"].get("cpu_threads") is None
                else int(data["detector"]["cpu_threads"])
            ),
            front_inference_every_n_frames=int(data.get("front_inference_every_n_frames", 1)),
            rear_capture_every_n_frames=int(data.get("rear_capture_every_n_frames", 3)),
            tracking_max_gap_s=float(data.get("tracking_max_gap_s", 2.0)),
            performance_window_frames=int(data.get("performance_window_frames", 60)),
            latency_budget_ms=float(data.get("latency_budget_ms", 150.0)),
            minimum_risk_fps=float(data.get("minimum_risk_fps", 5.0)),
            performance_warmup_frames=int(data.get("performance_warmup_frames", 60)),
            risk_assessment_freshness_s=float(data.get("risk_assessment_freshness_s", 0.45)),
            alert_actuator=AlertActuatorConfig.from_mapping(data.get("alert_actuator")),
        )


@dataclass(frozen=True)
class DirectionalAlertInput:
    """One cached risk input and its direction-specific alert eligibility."""

    assessment: RiskAssessment
    age_s: float
    estimated_risk_fps: float
    minimum_risk_fps: float
    fresh: bool
    meets_minimum_risk_fps: bool
    eligible: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "direction": self.assessment.direction.value,
            "target": self.assessment.object_id,
            "level": self.assessment.level.name,
            "assessment_age_s": round(self.age_s, 3),
            "estimated_risk_fps": round(self.estimated_risk_fps, 2),
            "minimum_risk_fps": self.minimum_risk_fps,
            "fresh": self.fresh,
            "meets_minimum_risk_fps": self.meets_minimum_risk_fps,
            "eligible": self.eligible,
            "reason": self.reason,
        }


@dataclass
class DirectionalAssessmentCache:
    """Keep one latest assessment per direction, subject to strict freshness.

    The rear camera may intentionally be captured less often than the front
    camera. This bridges scheduled capture gaps without allowing an old risk
    result to become a stale GPIO warning.
    """

    freshness_timeout_s: float
    _latest_by_direction: dict[RiskDirection, RiskAssessment] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.freshness_timeout_s <= 0:
            raise ValueError("freshness_timeout_s must be positive.")

    def update(self, assessments: Iterable[RiskAssessment]) -> None:
        """Replace each direction with this cycle's highest assessment.

        A new lower assessment deliberately replaces a held higher level. The
        collision pipeline owns its brief anti-flicker policy; this runtime
        cache must not turn old danger into a stale alert.
        """

        current: dict[RiskDirection, RiskAssessment] = {}
        for assessment in assessments:
            previous = current.get(assessment.direction)
            if previous is None or assessment.level >= previous.level:
                current[assessment.direction] = assessment
        self._latest_by_direction.update(current)

    def evaluate(
        self,
        *,
        now_s: float,
        effective_fps: float,
        minimum_risk_fps: float,
        cadence_by_direction: dict[RiskDirection, int],
        measured_fps_by_direction: dict[RiskDirection, float] | None = None,
    ) -> tuple[DirectionalAlertInput, ...]:
        """Return cached inputs that are fresh and sampled fast enough.

        The Pi runtime always supplies measured completion rates. Cadence-based
        estimation is retained only for older standalone callers. A supplied
        mapping with a missing/invalid direction fails closed at zero FPS.
        """

        if minimum_risk_fps <= 0:
            raise ValueError("minimum_risk_fps must be positive.")
        entries: list[DirectionalAlertInput] = []
        for direction, assessment in tuple(self._latest_by_direction.items()):
            age_s = now_s - assessment.timestamp_s
            fresh = 0.0 <= age_s <= self.freshness_timeout_s
            cadence = max(1, int(cadence_by_direction.get(direction, 1)))
            if measured_fps_by_direction is None:
                estimated_risk_fps = max(0.0, effective_fps) / cadence
            else:
                measured = measured_fps_by_direction.get(direction, 0.0)
                estimated_risk_fps = max(0.0, measured) if isfinite(measured) else 0.0
            meets_rate = estimated_risk_fps >= minimum_risk_fps
            if not fresh:
                # Discard first; a stale result must never reappear in a later
                # cycle if a clock or caller is sampled again.
                del self._latest_by_direction[direction]
                reason = "assessment_stale"
            elif assessment.system_status is not SystemStatus.READY:
                reason = "assessment_system_not_ready"
            elif not meets_rate:
                reason = "directional_risk_fps_below_minimum"
            else:
                reason = None
            eligible = fresh and assessment.system_status is SystemStatus.READY and meets_rate
            entries.append(
                DirectionalAlertInput(
                    assessment=assessment,
                    age_s=age_s,
                    estimated_risk_fps=estimated_risk_fps,
                    minimum_risk_fps=minimum_risk_fps,
                    fresh=fresh,
                    meets_minimum_risk_fps=meets_rate,
                    eligible=eligible,
                    reason=reason,
                )
            )
        return tuple(entries)


class PiCollisionRiskRuntime:
    """Runs calibrated front and rear collision-risk inference on Pi cameras."""

    def __init__(self, config: PiRuntimeConfig) -> None:
        # Perception remains available during camera bring-up.  The risk
        # engines see calibration_valid=False and fail closed, so an unverified
        # focal value can never create an active warning.
        self.config = config

    def run(
        self,
        max_frames: int | None = None,
        turn_intent: TurnIntent = TurnIntent.NONE,
    ) -> None:
        if max_frames is not None and (type(max_frames) is not int or max_frames < 1):
            raise ValueError("max_frames must be a positive integer when supplied.")
        detector = OpenCvYoloOnnxDetector(
            self.config.model_path,
            input_size_px=self.config.detector_input_size_px,
            confidence_threshold=self.config.detector_confidence,
            cpu_threads=self.config.detector_cpu_threads,
        )
        estimator = MonocularRangeEstimator(self.config.front_calibration)
        tracker = IoUTracker(id_prefix="front", max_age_s=self.config.tracking_max_gap_s)
        pipeline = ForwardCollisionPipeline(filter_=AlphaBetaTrackFilter(max_gap_s=self.config.tracking_max_gap_s))
        rear_estimator = MonocularRangeEstimator(self.config.rear_calibration) if self.config.rear_calibration else None
        rear_tracker = IoUTracker(id_prefix="rear", max_age_s=self.config.tracking_max_gap_s) if self.config.rear_camera else None
        rear_pipeline = (
            RearCollisionPipeline(filter_=AlphaBetaTrackFilter(max_gap_s=self.config.tracking_max_gap_s))
            if rear_estimator
            else None
        )
        awareness_map = CoverageAwareRiskMap()
        blind_spot_monitor = BlindSpotMonitor()
        performance = EdgePerformanceMonitor(
            window_frames=self.config.performance_window_frames,
            latency_budget_ms=self.config.latency_budget_ms,
            minimum_risk_fps=self.config.minimum_risk_fps,
            warmup_frames=self.config.performance_warmup_frames,
        )
        direction_monitors = {
            "front": DirectionalPerformanceMonitor(window_frames=self.config.performance_window_frames),
            "rear": DirectionalPerformanceMonitor(
                enabled=self.config.rear_camera is not None,
                window_frames=self.config.performance_window_frames,
            ),
        }
        assessment_cache = DirectionalAssessmentCache(self.config.risk_assessment_freshness_s)
        cadence_by_direction = {
            RiskDirection.FRONT: self.config.front_inference_every_n_frames,
            RiskDirection.REAR: self.config.rear_capture_every_n_frames,
        }
        actuator = create_alert_actuator(self.config.alert_actuator)
        rig = None
        frame_count = 0
        latest_performance = None
        latest_alert_output = None
        latest_alert_inputs: tuple[DirectionalAlertInput, ...] = ()
        previous_decision_s = None
        completed = False
        failure = None
        last_direction_snapshot = None
        last_decision_timestamp_s = None
        try:
            rig = PiDualCameraRig(self.config.front_camera, self.config.rear_camera)
            print(json.dumps({
                "event": "runtime_start", "release": SOFTWARE_RELEASE,
                "timing_schema": 2,
                "front_calibrated": self.config.front_calibration.calibrated,
                "rear_calibrated": None if self.config.rear_calibration is None else self.config.rear_calibration.calibrated,
                "alert_actuator_enabled": self.config.alert_actuator.enabled,
                "note": "Processing FPS is not sensor FPS. Uncalibrated NONE is not an all-clear.",
            }), flush=True)
            while max_frames is None or frame_count < max_frames:
                frame_started_s = time.perf_counter()
                timestamp_s = time.monotonic()
                capture_started_s = time.perf_counter()
                front_frame = rig.capture_front()
                front_capture_ms = (time.perf_counter() - capture_started_s) * 1000.0
                capture_ms = front_capture_ms
                inference_started_s = time.perf_counter()
                if frame_count % self.config.front_inference_every_n_frames == 0:
                    detections = tracker.update(timestamp_s, detector.detect(front_frame))
                else:
                    detections = []
                front_inference_ms = (time.perf_counter() - inference_started_s) * 1000.0
                inference_ms = front_inference_ms

                assessments = []
                front_targets = []
                rear_targets = []
                scene_objects: list[dict[str, object]] = []
                detection_records = []
                for detection in detections:
                    detection_records.append(self._detection_record(detection, "front"))
                    if not estimator.supports_range(detection):
                        scene_objects.append(
                            {
                                "target": detection.object_id,
                                "class": detection.label,
                                "confidence": round(detection.confidence, 3),
                                "range_status": "not_calibrated_for_metric_range",
                            }
                        )
                        continue
                    observation = estimator.estimate(timestamp_s, detection)
                    assessment = pipeline.process(observation)
                    assessments.append(assessment)
                    front_targets.append((observation, assessment))
                if frame_count % self.config.front_inference_every_n_frames == 0:
                    direction_monitors["front"].record(
                        time.monotonic(), capture_ms=front_capture_ms,
                        inference_ms=front_inference_ms,
                        processing_ms=(time.perf_counter() - capture_started_s) * 1000.0,
                        detections=len(detections),
                    )
                if frame_count % self.config.rear_capture_every_n_frames == 0 and rig.rear is not None:
                    rear_capture_started_s = time.perf_counter()
                    rear_frame = rig.capture_rear()
                    rear_capture_ms = (time.perf_counter() - rear_capture_started_s) * 1000.0
                    capture_ms += rear_capture_ms
                    if rear_frame is not None and rear_tracker:
                        rear_timestamp_s = time.monotonic()
                        rear_inference_started_s = time.perf_counter()
                        rear_detections = rear_tracker.update(rear_timestamp_s, detector.detect(rear_frame))
                        rear_inference_ms = (time.perf_counter() - rear_inference_started_s) * 1000.0
                        inference_ms += rear_inference_ms
                        for detection in rear_detections:
                            detection_records.append(self._detection_record(detection, "rear"))
                            if rear_estimator is None or not rear_estimator.supports_range(detection):
                                scene_objects.append(
                                    {
                                        "target": detection.object_id,
                                        "class": detection.label,
                                        "confidence": round(detection.confidence, 3),
                                        "range_status": "not_calibrated_for_metric_range",
                                    }
                                )
                                continue
                            observation = rear_estimator.estimate(rear_timestamp_s, detection)
                            assessment = rear_pipeline.process(observation) if rear_pipeline else None
                            if assessment is None:
                                continue
                            assessments.append(assessment)
                            rear_targets.append((observation, assessment))
                        direction_monitors["rear"].record(
                            time.monotonic(), capture_ms=rear_capture_ms,
                            inference_ms=rear_inference_ms,
                            processing_ms=(time.perf_counter() - rear_capture_started_s) * 1000.0,
                            detections=len(rear_detections),
                        )

                awareness = awareness_map.build(front_targets, rear_targets)
                blind_spot = [
                    result.as_dict()
                    for observation, _ in rear_targets
                    for result in [blind_spot_monitor.assess(observation, turn_intent)]
                    if result.nearby_lateral_target
                ]
                # Capture/inference timers are disjoint. Derive the remaining
                # work from ONE wall interval: rear capture/inference must not
                # also be counted as front postprocessing (v5 regression).
                decision_s = time.perf_counter()
                work_ms = (decision_s - frame_started_s) * 1000.0
                postprocess_ms = max(0.0, work_ms - capture_ms - inference_ms)
                wall_interval_ms = work_ms if previous_decision_s is None else (decision_s - previous_decision_s) * 1000.0
                previous_decision_s = decision_s
                current_performance = performance.record(
                    capture_ms, inference_ms, postprocess_ms,
                    wall_interval_ms=max(work_ms, wall_interval_ms),
                )
                assessment_cache.update(assessments)
                alert_timestamp_s = time.monotonic()
                direction_snapshot = {
                    role: monitor.snapshot(alert_timestamp_s)
                    for role, monitor in direction_monitors.items()
                }
                last_direction_snapshot = direction_snapshot
                last_decision_timestamp_s = alert_timestamp_s
                latest_alert_inputs = assessment_cache.evaluate(
                    now_s=alert_timestamp_s,
                    effective_fps=current_performance.effective_fps,
                    minimum_risk_fps=self.config.minimum_risk_fps,
                    cadence_by_direction=cadence_by_direction,
                    measured_fps_by_direction={
                        RiskDirection(role): float(snapshot["processed_fps"])
                        for role, snapshot in direction_snapshot.items()
                    },
                )
                eligible_assessments = [item.assessment for item in latest_alert_inputs if item.eligible]
                highest_overall = max(eligible_assessments, key=lambda item: item.level, default=None)
                requested_level = highest_overall.level if highest_overall is not None else AlertLevel.NONE
                # A GPIO output is fail-closed: it stays off until the rolling
                # Pi health gate has warmed up/passed and the selected
                # direction itself has fresh, adequate-rate evidence.
                latest_alert_output = actuator.apply(
                    requested_level,
                    alerts_permitted=current_performance.risk_alerts_permitted,
                    now_s=alert_timestamp_s,
                )
                latest_performance = current_performance
                frame_count += 1
                # A heartbeat is emitted even with no objects, so a running
                # empty rear view is distinguishable from an unobserved feed.
                print(json.dumps({
                    "event": "runtime_status", "release": SOFTWARE_RELEASE,
                    "frame_count": frame_count,
                    "timestamp_s": round(alert_timestamp_s, 6),
                    "performance": latest_performance.as_dict(),
                    "directional_performance": direction_snapshot,
                    "detections": detection_records,
                    "alert_inputs": [item.as_dict() for item in latest_alert_inputs],
                    "alert_output": latest_alert_output.as_dict(),
                }, ensure_ascii=False), flush=True)
                for direction in ("front", "rear"):
                    directional = [item for item in assessments if item.direction.value == direction]
                    highest = max(directional, key=lambda item: item.level, default=None)
                    if highest is None:
                        continue
                    print(
                        json.dumps(
                            {
                                "timestamp_s": round(highest.timestamp_s, 3),
                                "target": highest.object_id,
                                "direction": highest.direction.value,
                                "level": highest.level.name,
                                "status": highest.system_status.value,
                                "ttc_s": None if highest.conservative_ttc_s is None else round(highest.conservative_ttc_s, 2),
                                "closing_speed_mps": None if highest.closing_speed_mps is None else round(highest.closing_speed_mps, 2),
                                "reasons": highest.reasons,
                                "scene_objects": scene_objects,
                                "blind_spot": blind_spot,
                                "awareness_map": awareness.as_dict(),
                                "performance": latest_performance.as_dict(),
                                "directional_performance": direction_snapshot,
                                "alert_inputs": [item.as_dict() for item in latest_alert_inputs],
                                "alert_output": latest_alert_output.as_dict(),
                            },
                            ensure_ascii=False,
                        )
                    )
            completed = True
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            # Keep any successful direction in a failed partial loop visible.
            # The aggregate performance_final still describes the last FULL
            # loop; completed=False prevents acceptance of this partial run.
            last_decision_timestamp_s = time.monotonic()
            last_direction_snapshot = {
                role: monitor.snapshot(last_decision_timestamp_s)
                for role, monitor in direction_monitors.items()
            }
            raise
        finally:
            cleanup_error = None
            for resource in (rig, actuator):
                if resource is not None:
                    try:
                        resource.close()
                    except Exception as exc:
                        cleanup_error = cleanup_error or exc
            # This is a diagnostic record, never proof of a passed benchmark.
            # Explicit completion prevents an interrupted/failed partial run
            # from being mistaken for a successful performance result.
            final_output: dict[str, object] = {
                "event": "runtime_final", "release": SOFTWARE_RELEASE,
                "completed": completed and cleanup_error is None,
                "error": failure or (None if cleanup_error is None else str(cleanup_error)),
                "processed_frames": frame_count,
                "performance_final": None if latest_performance is None else latest_performance.as_dict(),
                "performance_timestamp_s": last_decision_timestamp_s,
                # Match performance_final's checkpoint. Camera stop/close and
                # final JSON printing are not another active processing cycle.
                "directional_performance": last_direction_snapshot if last_direction_snapshot is not None else {
                    role: monitor.snapshot(time.monotonic())
                    for role, monitor in direction_monitors.items()
                },
                "alert_inputs": [item.as_dict() for item in latest_alert_inputs],
                "outputs_closed": cleanup_error is None,
            }
            # Do not echo the previous ON state as if it were still applied
            # after close(). Hardware state on a cleanup error is unknown.
            final_output["alert_output"] = {
                "requested_level": "NONE", "applied_level": "NONE" if cleanup_error is None else "UNKNOWN",
                "alerts_permitted": False,
                "buzzer_on": False if cleanup_error is None else None,
                "vibration_on": False if cleanup_error is None else None,
                "led_on": False if cleanup_error is None else None,
            }
            print(json.dumps(final_output, ensure_ascii=False), flush=True)
            if cleanup_error is not None and failure is None:
                raise cleanup_error

    @staticmethod
    def _detection_record(detection: Any, direction: str) -> dict[str, object]:
        return {
            "direction": direction, "target": detection.object_id,
            "class": detection.label, "confidence": round(detection.confidence, 4),
            "bbox_xywh_px": [round(v, 2) for v in (
                detection.x_px, detection.y_px, detection.width_px, detection.height_px
            )],
        }


# Kept as an import-compatible alias for callers using the original MVP name.
PiForwardCollisionRuntime = PiCollisionRiskRuntime
