"""Small, dependency-free latency telemetry for edge deployment validation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import ceil, isfinite
from statistics import fmean


@dataclass(frozen=True)
class PerformanceSnapshot:
    frame_count: int
    capture_ms: float
    inference_ms: float
    postprocess_ms: float
    end_to_end_ms: float
    average_end_to_end_ms: float
    p95_end_to_end_ms: float
    effective_fps: float
    latency_budget_ms: float
    meets_latency_budget: bool
    # These defaults preserve construction compatibility for callers that made
    # snapshots directly before performance health gating was introduced.
    minimum_risk_fps: float = 0.0
    warmup_frames: int = 0
    warmup_complete: bool = True
    meets_minimum_risk_fps: bool = True
    risk_alerts_permitted: bool = True
    average_cycle_interval_ms: float | None = None

    def as_dict(self) -> dict[str, float | bool | int]:
        return {
            "frame_count": self.frame_count,
            "capture_ms": round(self.capture_ms, 2),
            "inference_ms": round(self.inference_ms, 2),
            "postprocess_ms": round(self.postprocess_ms, 2),
            "end_to_end_ms": round(self.end_to_end_ms, 2),
            "average_end_to_end_ms": round(self.average_end_to_end_ms, 2),
            "p95_end_to_end_ms": round(self.p95_end_to_end_ms, 2),
            "effective_fps": round(self.effective_fps, 2),
            "latency_budget_ms": self.latency_budget_ms,
            "meets_latency_budget": self.meets_latency_budget,
            "minimum_risk_fps": self.minimum_risk_fps,
            "warmup_frames": self.warmup_frames,
            "warmup_complete": self.warmup_complete,
            "meets_minimum_risk_fps": self.meets_minimum_risk_fps,
            "risk_alerts_permitted": self.risk_alerts_permitted,
            "average_cycle_interval_ms": round(
                self.average_end_to_end_ms if self.average_cycle_interval_ms is None
                else self.average_cycle_interval_ms, 2
            ),
        }


class EdgePerformanceMonitor:
    """Collect rolling timings and expose a fail-closed risk-alert health gate.

    The gate remains closed until enough samples have been observed and both
    the rolling p95 latency and rolling effective FPS meet their configured
    limits.  It deliberately reports health only; the runtime decides whether
    to suppress, degrade, or otherwise present a risk assessment.
    """

    def __init__(
        self,
        window_frames: int = 60,
        latency_budget_ms: float = 150.0,
        minimum_risk_fps: float = 5.0,
        warmup_frames: int | None = None,
    ) -> None:
        if window_frames < 5:
            raise ValueError("window_frames must be at least 5.")
        if latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive.")
        if minimum_risk_fps <= 0:
            raise ValueError("minimum_risk_fps must be positive.")
        if warmup_frames is not None and warmup_frames < 1:
            raise ValueError("warmup_frames must be at least 1 when provided.")
        self.latency_budget_ms = latency_budget_ms
        self.minimum_risk_fps = minimum_risk_fps
        # A full rolling window is the safe default: before that point p95 and
        # FPS can be overly optimistic because they represent too few frames.
        self.warmup_frames = window_frames if warmup_frames is None else warmup_frames
        self._samples: deque[float] = deque(maxlen=window_frames)
        self._cycle_samples: deque[float] = deque(maxlen=window_frames)
        self._frame_count = 0

    def record(
        self, capture_ms: float, inference_ms: float, postprocess_ms: float,
        *, wall_interval_ms: float | None = None,
    ) -> PerformanceSnapshot:
        """Record disjoint work and optional measured decision-to-decision time.

        Latency is current-cycle work. FPS includes the interval since the
        previous decision (including its JSON output/other loop overhead).
        Legacy callers without a wall interval retain component-sum behavior.
        """
        if not all(isfinite(v) and v >= 0 for v in (capture_ms, inference_ms, postprocess_ms)):
            raise ValueError("Latency measurements cannot be negative.")
        total = capture_ms + inference_ms + postprocess_ms
        if wall_interval_ms is not None and (not isfinite(wall_interval_ms) or wall_interval_ms < total - 1e-6):
            raise ValueError("wall_interval_ms must be finite and at least the measured work duration.")
        self._samples.append(total)
        self._cycle_samples.append(total if wall_interval_ms is None else max(total, wall_interval_ms))
        self._frame_count += 1
        ordered = sorted(self._samples)
        p95_index = max(0, ceil(len(ordered) * 0.95) - 1)
        average = fmean(self._samples)
        p95 = ordered[p95_index]
        average_cycle = fmean(self._cycle_samples)
        effective_fps = 1000.0 / average_cycle if average_cycle > 0 else 0.0
        meets_latency_budget = p95 <= self.latency_budget_ms
        meets_minimum_risk_fps = effective_fps >= self.minimum_risk_fps
        warmup_complete = self._frame_count >= self.warmup_frames
        return PerformanceSnapshot(
            frame_count=self._frame_count,
            capture_ms=capture_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
            end_to_end_ms=total,
            average_end_to_end_ms=average,
            p95_end_to_end_ms=p95,
            effective_fps=effective_fps,
            latency_budget_ms=self.latency_budget_ms,
            meets_latency_budget=meets_latency_budget,
            minimum_risk_fps=self.minimum_risk_fps,
            warmup_frames=self.warmup_frames,
            warmup_complete=warmup_complete,
            meets_minimum_risk_fps=meets_minimum_risk_fps,
            risk_alerts_permitted=(
                warmup_complete and meets_latency_budget and meets_minimum_risk_fps
            ),
            average_cycle_interval_ms=average_cycle,
        )


class DirectionalPerformanceMonitor:
    """Actual completed perception updates, not configured camera sensor FPS.

    No target is required: a successful empty detector pass is still an update.
    At least two updates are required to report a rate. Idle time since the
    latest completion is included so a stopped/skipped direction cannot retain
    a permanently healthy historical FPS.
    """

    def __init__(self, *, enabled: bool = True, window_frames: int = 60) -> None:
        if window_frames < 2:
            raise ValueError("Directional window must contain at least two updates.")
        self.enabled = enabled
        self._timestamps: deque[float] = deque(maxlen=window_frames)
        self.processed_frames = 0
        self.capture_ms = self.inference_ms = self.processing_ms = 0.0
        self.detections = 0

    def record(
        self, completed_at_s: float, *, capture_ms: float, inference_ms: float,
        processing_ms: float, detections: int,
    ) -> None:
        if not self.enabled:
            raise ValueError("Cannot record a disabled camera direction.")
        if not all(isfinite(v) and v >= 0 for v in (completed_at_s, capture_ms, inference_ms, processing_ms)):
            raise ValueError("Directional timings must be finite and non-negative.")
        if self._timestamps and completed_at_s < self._timestamps[-1]:
            raise ValueError("Completion timestamps must be monotonic.")
        if type(detections) is not int or detections < 0:
            raise ValueError("Detection count must be a non-negative integer.")
        self._timestamps.append(completed_at_s)
        self.processed_frames += 1
        self.capture_ms = capture_ms
        self.inference_ms = inference_ms
        self.processing_ms = processing_ms
        self.detections = detections

    def snapshot(self, now_s: float) -> dict[str, object]:
        if not isfinite(now_s):
            raise ValueError("Snapshot timestamp must be finite.")
        age = None if not self._timestamps else now_s - self._timestamps[-1]
        elapsed = 0.0 if not self._timestamps else now_s - self._timestamps[0]
        fps = (
            (len(self._timestamps) - 1) / elapsed
            if len(self._timestamps) >= 2 and elapsed > 0 and age is not None and age >= 0
            else 0.0
        )
        return {
            "enabled": self.enabled,
            "processed_frames": self.processed_frames,
            "processed_fps": fps,
            "capture_ms": round(self.capture_ms, 3),
            "inference_ms": round(self.inference_ms, 3),
            "processing_ms": round(self.processing_ms, 3),
            "last_update_age_s": None if age is None else round(age, 4),
            "detections": self.detections,
        }
