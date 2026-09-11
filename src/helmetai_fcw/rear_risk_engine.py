"""Explainable rear-end collision-risk decision logic.

The rear camera observes vehicles behind the rider.  As in the forward path,
a negative range rate means that the separation to a detected target is
shrinking.  The output is deliberately fail-closed when the rear camera has
not been calibrated or a track is not stable enough.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from .models import AlertLevel, RiskAssessment, RiskDirection, SystemStatus, TrackEstimate


@dataclass(frozen=True)
class RearCollisionConfig:
    """Tunable rear-collision policy values for controlled-course evaluation.

    These are PoC defaults, not rider-facing safety thresholds.  They must be
    calibrated with measured rear-camera data before an on-road deployment.
    """

    supported_classes: frozenset[str] = frozenset({"car", "truck", "bus", "motorcycle", "vehicle"})
    minimum_confidence: float = 0.55
    minimum_track_frames: int = 3
    minimum_closing_speed_mps: float = 0.50
    minimum_closing_evidence_frames: int = 3
    minimum_closing_evidence_s: float = 0.35
    maximum_relative_range_sigma: float = 0.30
    geometry_margin_m: float = 1.20
    distance_uncertainty_sigmas: float = 2.0
    velocity_uncertainty_sigmas: float = 1.0
    corridor_half_width_m: float = 1.45
    corridor_growth_per_m: float = 0.01
    max_corridor_half_width_m: float = 2.40
    advisory_ttc_s: float = 4.5
    warning_ttc_s: float = 2.5
    critical_ttc_s: float = 1.4
    advisory_required_decel_mps2: float = 1.2
    warning_required_decel_mps2: float = 2.8
    critical_required_decel_mps2: float = 4.5


class RearCollisionRiskEngine:
    """Assess a rear-camera track for an approaching vehicle behind the rider."""

    def __init__(self, config: RearCollisionConfig | None = None) -> None:
        self.config = config or RearCollisionConfig()

    def _path_half_width(self, longitudinal_m: float) -> float:
        width = self.config.corridor_half_width_m + self.config.corridor_growth_per_m * max(longitudinal_m, 0)
        return min(width, self.config.max_corridor_half_width_m)

    def assess(self, track: TrackEstimate) -> RiskAssessment:
        if not track.calibration_valid:
            return self._degraded(track, "rear_calibration_or_frame_invalid")
        if track.confidence < self.config.minimum_confidence:
            return self._degraded(track, "rear_perception_confidence_below_minimum")
        if track.age_frames < self.config.minimum_track_frames:
            return self._degraded(track, "rear_track_not_stable_yet")
        if track.object_class.lower() not in self.config.supported_classes:
            return self._none(track, ("rear_unsupported_target_class",))
        if track.longitudinal_m <= 0:
            return self._none(track, ("rear_target_range_invalid",))
        if track.range_sigma_m / track.longitudinal_m > self.config.maximum_relative_range_sigma:
            return self._degraded(track, "rear_range_uncertainty_exceeds_policy")

        in_path = abs(track.lateral_m) <= self._path_half_width(track.longitudinal_m)
        if not in_path:
            return self._none(track, ("rear_target_outside_corridor",), in_path=False)

        mean_closing_speed = max(0.0, -track.longitudinal_velocity_mps)
        if mean_closing_speed < self.config.minimum_closing_speed_mps:
            return self._none(
                track,
                ("rear_no_material_closing_speed",),
                in_path=True,
                closing_speed_mps=mean_closing_speed,
            )
        if track.closing_evidence_frames is None:
            return self._degraded(track, "rear_closing_trend_evidence_unavailable")
        if track.closing_evidence_frames < self.config.minimum_closing_evidence_frames:
            return self._none(
                track,
                ("rear_closing_trend_not_confirmed",),
                in_path=True,
                closing_speed_mps=mean_closing_speed,
            )
        if track.closing_evidence_duration_s is None:
            return self._degraded(track, "rear_closing_trend_duration_unavailable")
        if track.closing_evidence_duration_s < self.config.minimum_closing_evidence_s:
            return self._none(
                track,
                ("rear_closing_trend_duration_not_confirmed",),
                in_path=True,
                closing_speed_mps=mean_closing_speed,
            )

        closing_speed = mean_closing_speed + self.config.velocity_uncertainty_sigmas * track.velocity_sigma_mps
        uncertainty_margin = self.config.distance_uncertainty_sigmas * track.range_sigma_m
        effective_distance = max(0.05, track.longitudinal_m - self.config.geometry_margin_m - uncertainty_margin)
        raw_ttc = track.longitudinal_m / closing_speed
        conservative_ttc = effective_distance / closing_speed
        required_decel = (closing_speed * closing_speed) / (2.0 * effective_distance)
        level = self._level(conservative_ttc, required_decel)
        reasons = ["rear_target_in_corridor", "rear_positive_closing_speed", "rear_uncertainty_margin_applied"]
        if conservative_ttc <= self.config.critical_ttc_s:
            reasons.append("rear_critical_ttc")
        elif conservative_ttc <= self.config.warning_ttc_s:
            reasons.append("rear_warning_ttc")
        elif conservative_ttc <= self.config.advisory_ttc_s:
            reasons.append("rear_advisory_ttc")
        if required_decel >= self.config.critical_required_decel_mps2:
            reasons.append("rear_critical_braking_demand")
        elif required_decel >= self.config.warning_required_decel_mps2:
            reasons.append("rear_warning_braking_demand")
        elif required_decel >= self.config.advisory_required_decel_mps2:
            reasons.append("rear_advisory_braking_demand")
        return RiskAssessment(
            timestamp_s=track.timestamp_s,
            object_id=track.object_id,
            level=level,
            system_status=SystemStatus.READY,
            reasons=tuple(reasons),
            in_path=True,
            raw_ttc_s=raw_ttc,
            conservative_ttc_s=conservative_ttc,
            required_relative_decel_mps2=required_decel,
            effective_distance_m=effective_distance,
            closing_speed_mps=closing_speed,
            direction=RiskDirection.REAR,
        )

    def _level(self, ttc_s: float, required_decel_mps2: float) -> AlertLevel:
        if ttc_s <= self.config.critical_ttc_s or required_decel_mps2 >= self.config.critical_required_decel_mps2:
            return AlertLevel.CRITICAL
        if ttc_s <= self.config.warning_ttc_s or required_decel_mps2 >= self.config.warning_required_decel_mps2:
            return AlertLevel.WARNING
        if ttc_s <= self.config.advisory_ttc_s or required_decel_mps2 >= self.config.advisory_required_decel_mps2:
            return AlertLevel.ADVISORY
        return AlertLevel.NONE

    @staticmethod
    def _none(
        track: TrackEstimate,
        reasons: tuple[str, ...],
        in_path: bool = False,
        closing_speed_mps: float | None = None,
    ) -> RiskAssessment:
        return RiskAssessment(
            timestamp_s=track.timestamp_s,
            object_id=track.object_id,
            level=AlertLevel.NONE,
            system_status=SystemStatus.READY,
            reasons=reasons,
            in_path=in_path,
            raw_ttc_s=inf if closing_speed_mps == 0 else None,
            closing_speed_mps=closing_speed_mps,
            direction=RiskDirection.REAR,
        )

    @staticmethod
    def _degraded(track: TrackEstimate, reason: str) -> RiskAssessment:
        return RiskAssessment(
            timestamp_s=track.timestamp_s,
            object_id=track.object_id,
            level=AlertLevel.NONE,
            system_status=SystemStatus.DEGRADED,
            reasons=(reason,),
            direction=RiskDirection.REAR,
        )
