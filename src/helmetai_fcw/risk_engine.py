"""Explainable forward-collision-risk decision logic."""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

from .models import AlertLevel, RiskAssessment, RiskDirection, SystemStatus, TrackEstimate


@dataclass(frozen=True)
class ForwardCollisionConfig:
    """Tunable FCW policy values.

    Values are PoC defaults only. They must be calibrated on controlled-course
    data before any rider-facing use and are not production safety thresholds.
    """

    supported_classes: frozenset[str] = frozenset({"person", "bicycle", "car", "truck", "bus", "motorcycle", "vehicle"})
    minimum_confidence: float = 0.55
    minimum_track_frames: int = 3
    minimum_closing_speed_mps: float = 0.50
    minimum_closing_evidence_frames: int = 3
    minimum_closing_evidence_s: float = 0.35
    maximum_relative_range_sigma: float = 0.30
    geometry_margin_m: float = 1.50
    distance_uncertainty_sigmas: float = 2.0
    velocity_uncertainty_sigmas: float = 1.0
    corridor_half_width_m: float = 1.35
    corridor_growth_per_m: float = 0.01
    max_corridor_half_width_m: float = 2.20
    advisory_ttc_s: float = 5.0
    warning_ttc_s: float = 3.0
    critical_ttc_s: float = 1.7
    advisory_required_decel_mps2: float = 1.2
    warning_required_decel_mps2: float = 2.8
    critical_required_decel_mps2: float = 4.5


class ForwardCollisionRiskEngine:
    """Assess a filtered object track for a same-path forward collision threat."""

    def __init__(self, config: ForwardCollisionConfig | None = None) -> None:
        self.config = config or ForwardCollisionConfig()

    def _path_half_width(self, longitudinal_m: float) -> float:
        width = self.config.corridor_half_width_m + self.config.corridor_growth_per_m * max(longitudinal_m, 0)
        return min(width, self.config.max_corridor_half_width_m)

    def assess(self, track: TrackEstimate) -> RiskAssessment:
        reasons: list[str] = []
        if not track.calibration_valid:
            return self._degraded(track, "range_calibration_or_frame_invalid")
        if track.confidence < self.config.minimum_confidence:
            return self._degraded(track, "perception_confidence_below_minimum")
        if track.age_frames < self.config.minimum_track_frames:
            return self._degraded(track, "track_not_stable_yet")
        if track.object_class.lower() not in self.config.supported_classes:
            return self._none(track, ("unsupported_target_class",))
        if track.longitudinal_m <= 0:
            return self._none(track, ("target_not_in_front",))
        if track.range_sigma_m / track.longitudinal_m > self.config.maximum_relative_range_sigma:
            return self._degraded(track, "range_uncertainty_exceeds_policy")

        in_path = abs(track.lateral_m) <= self._path_half_width(track.longitudinal_m)
        if not in_path:
            return self._none(track, ("target_outside_forward_corridor",), in_path=False)

        mean_closing_speed = max(0.0, -track.longitudinal_velocity_mps)
        # Uncertainty makes an already-closing target more conservative; it must
        # not manufacture a collision threat for a stationary range estimate.
        if mean_closing_speed < self.config.minimum_closing_speed_mps:
            return self._none(
                track,
                ("no_material_closing_speed",),
                in_path=True,
                closing_speed_mps=mean_closing_speed,
            )
        if track.closing_evidence_frames is None:
            return self._degraded(track, "closing_trend_evidence_unavailable")
        if track.closing_evidence_frames < self.config.minimum_closing_evidence_frames:
            return self._none(
                track,
                ("closing_trend_not_confirmed",),
                in_path=True,
                closing_speed_mps=mean_closing_speed,
            )
        if track.closing_evidence_duration_s is None:
            return self._degraded(track, "closing_trend_duration_unavailable")
        if track.closing_evidence_duration_s < self.config.minimum_closing_evidence_s:
            return self._none(
                track,
                ("closing_trend_duration_not_confirmed",),
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
        reasons.extend(["target_in_forward_corridor", "positive_closing_speed", "uncertainty_margin_applied"])
        if conservative_ttc <= self.config.critical_ttc_s:
            reasons.append("critical_ttc")
        elif conservative_ttc <= self.config.warning_ttc_s:
            reasons.append("warning_ttc")
        elif conservative_ttc <= self.config.advisory_ttc_s:
            reasons.append("advisory_ttc")
        if required_decel >= self.config.critical_required_decel_mps2:
            reasons.append("critical_braking_demand")
        elif required_decel >= self.config.warning_required_decel_mps2:
            reasons.append("warning_braking_demand")
        elif required_decel >= self.config.advisory_required_decel_mps2:
            reasons.append("advisory_braking_demand")

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
            direction=RiskDirection.FRONT,
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
            direction=RiskDirection.FRONT,
        )

    @staticmethod
    def _degraded(track: TrackEstimate, reason: str) -> RiskAssessment:
        return RiskAssessment(
            timestamp_s=track.timestamp_s,
            object_id=track.object_id,
            level=AlertLevel.NONE,
            system_status=SystemStatus.DEGRADED,
            reasons=(reason,),
            direction=RiskDirection.FRONT,
        )
