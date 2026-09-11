"""Typed data contracts between perception, tracking and collision-risk layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum


class AlertLevel(IntEnum):
    """Increasing warning urgency. NONE is deliberately zero for comparisons."""

    NONE = 0
    ADVISORY = 1
    WARNING = 2
    CRITICAL = 3


class SystemStatus(str, Enum):
    READY = "ready"
    DEGRADED = "degraded"
    INVALID = "invalid"


class RiskDirection(str, Enum):
    """Camera/risk-zone that produced a collision assessment."""

    FRONT = "front"
    REAR = "rear"


@dataclass(frozen=True)
class ObservedTarget:
    """A depth-enabled perception observation in the motorcycle-forward frame.

    Coordinates are metres. ``longitudinal_m`` grows in the forward direction
    and ``lateral_m`` is positive to the rider's right. The observation must
    come from a calibrated range estimator (stereo, radar or a calibrated
    monocular model) and carries its own uncertainty.
    """

    timestamp_s: float
    object_id: str
    object_class: str
    longitudinal_m: float
    lateral_m: float
    range_sigma_m: float
    confidence: float
    calibration_valid: bool = True
    frame_valid: bool = True


@dataclass(frozen=True)
class TrackEstimate:
    """Filtered target state; negative longitudinal velocity means closing."""

    timestamp_s: float
    object_id: str
    object_class: str
    longitudinal_m: float
    lateral_m: float
    longitudinal_velocity_mps: float
    lateral_velocity_mps: float
    range_sigma_m: float
    velocity_sigma_mps: float
    confidence: float
    calibration_valid: bool
    age_frames: int
    # The built-in filter always emits a concrete count.  ``None`` means an
    # external tracker omitted this required evidence, and the risk engines
    # deliberately fail closed rather than inferring a TTC alert from it.
    closing_evidence_frames: int | None = None
    # The elapsed duration of the consecutive closing trend.  A frame count
    # alone is unsafe because it changes meaning when edge-device FPS drops.
    # ``None`` is likewise treated as unavailable evidence by the engines.
    closing_evidence_duration_s: float | None = None


@dataclass(frozen=True)
class RiskAssessment:
    """Explainable collision-risk result for one tracked target."""

    timestamp_s: float
    object_id: str
    level: AlertLevel
    system_status: SystemStatus
    reasons: tuple[str, ...] = field(default_factory=tuple)
    in_path: bool = False
    raw_ttc_s: float | None = None
    conservative_ttc_s: float | None = None
    required_relative_decel_mps2: float | None = None
    effective_distance_m: float | None = None
    closing_speed_mps: float | None = None
    direction: RiskDirection = RiskDirection.FRONT
