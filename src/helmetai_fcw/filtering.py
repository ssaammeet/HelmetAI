"""Small alpha-beta tracker suitable for depth observations at video rate."""

from __future__ import annotations

from dataclasses import dataclass

from .models import ObservedTarget, TrackEstimate


@dataclass
class _FilterState:
    timestamp_s: float
    longitudinal_m: float
    lateral_m: float
    longitudinal_velocity_mps: float
    lateral_velocity_mps: float
    range_sigma_m: float
    confidence: float
    calibration_valid: bool
    age_frames: int
    object_class: str
    closing_evidence_frames: int
    closing_evidence_started_s: float | None


class AlphaBetaTrackFilter:
    """Filters noisy stereo range/lateral observations and estimates range rate.

    This intentionally stays lightweight and dependency-free for an edge MVP.
    It is not a substitute for a calibrated EKF/UKF in a production safety case.
    """

    def __init__(
        self,
        alpha: float = 0.58,
        beta: float = 0.28,
        max_gap_s: float = 0.75,
        evidence_speed_mps: float = 0.50,
    ) -> None:
        if not 0 < alpha <= 1 or not 0 < beta <= 1:
            raise ValueError("alpha and beta must be in (0, 1].")
        self.alpha = alpha
        self.beta = beta
        self.max_gap_s = max_gap_s
        self.evidence_speed_mps = evidence_speed_mps
        self._states: dict[str, _FilterState] = {}

    def update(self, observation: ObservedTarget) -> TrackEstimate:
        state = self._states.get(observation.object_id)
        if state is None or observation.timestamp_s <= state.timestamp_s:
            state = _FilterState(
                timestamp_s=observation.timestamp_s,
                longitudinal_m=observation.longitudinal_m,
                lateral_m=observation.lateral_m,
                longitudinal_velocity_mps=0.0,
                lateral_velocity_mps=0.0,
                range_sigma_m=max(observation.range_sigma_m, 0.02),
                confidence=observation.confidence,
                calibration_valid=observation.calibration_valid and observation.frame_valid,
                age_frames=1,
                object_class=observation.object_class,
                closing_evidence_frames=0,
                closing_evidence_started_s=None,
            )
        else:
            dt = observation.timestamp_s - state.timestamp_s
            if dt > self.max_gap_s:
                state = _FilterState(
                    timestamp_s=observation.timestamp_s,
                    longitudinal_m=observation.longitudinal_m,
                    lateral_m=observation.lateral_m,
                    longitudinal_velocity_mps=0.0,
                    lateral_velocity_mps=0.0,
                    range_sigma_m=max(observation.range_sigma_m, 0.02),
                    confidence=observation.confidence,
                    calibration_valid=observation.calibration_valid and observation.frame_valid,
                    age_frames=1,
                    object_class=observation.object_class,
                    closing_evidence_frames=0,
                    closing_evidence_started_s=None,
                )
            else:
                predicted_longitudinal = state.longitudinal_m + state.longitudinal_velocity_mps * dt
                predicted_lateral = state.lateral_m + state.lateral_velocity_mps * dt
                longitudinal_residual = observation.longitudinal_m - predicted_longitudinal
                lateral_residual = observation.lateral_m - predicted_lateral
                quality = min(max(observation.confidence, 0.0), 1.0)
                alpha = self.alpha * (0.35 + 0.65 * quality)
                beta = self.beta * (0.35 + 0.65 * quality)
                # The measurement-to-measurement range rate is low-pass filtered
                # separately. This avoids velocity overshoot during tracker warmup.
                measured_longitudinal_velocity = (observation.longitudinal_m - state.longitudinal_m) / dt
                measured_lateral_velocity = (observation.lateral_m - state.lateral_m) / dt
                filtered_longitudinal_velocity = (1.0 - beta) * state.longitudinal_velocity_mps + beta * measured_longitudinal_velocity
                closing = filtered_longitudinal_velocity <= -self.evidence_speed_mps
                evidence_started_s = (
                    state.closing_evidence_started_s
                    if closing and state.closing_evidence_started_s is not None
                    else observation.timestamp_s if closing else None
                )
                state = _FilterState(
                    timestamp_s=observation.timestamp_s,
                    longitudinal_m=predicted_longitudinal + alpha * longitudinal_residual,
                    lateral_m=predicted_lateral + alpha * lateral_residual,
                    longitudinal_velocity_mps=filtered_longitudinal_velocity,
                    lateral_velocity_mps=(1.0 - beta) * state.lateral_velocity_mps + beta * measured_lateral_velocity,
                    range_sigma_m=max(observation.range_sigma_m, 0.02),
                    confidence=0.7 * state.confidence + 0.3 * quality,
                    calibration_valid=state.calibration_valid and observation.calibration_valid and observation.frame_valid,
                    age_frames=state.age_frames + 1,
                    object_class=observation.object_class,
                    # A single changing detector box must not become a TTC
                    # warning.  Risk engines require this trend to persist
                    # across consecutive observations.
                    closing_evidence_frames=(
                        state.closing_evidence_frames + 1
                        if closing
                        else 0
                    ),
                    closing_evidence_started_s=evidence_started_s,
                )
        self._states[observation.object_id] = state
        # Conservative PoC proxy for the *filtered* range-rate error. Raw depth
        # sigma must not be divided by the video period here: that would turn a
        # far-range optical uncertainty into an artificial collision velocity.
        velocity_sigma = max(0.25, 0.10 * state.range_sigma_m)
        return TrackEstimate(
            timestamp_s=state.timestamp_s,
            object_id=observation.object_id,
            object_class=state.object_class,
            longitudinal_m=state.longitudinal_m,
            lateral_m=state.lateral_m,
            longitudinal_velocity_mps=state.longitudinal_velocity_mps,
            lateral_velocity_mps=state.lateral_velocity_mps,
            range_sigma_m=state.range_sigma_m,
            velocity_sigma_mps=velocity_sigma,
            confidence=state.confidence,
            calibration_valid=state.calibration_valid,
            age_frames=state.age_frames,
            closing_evidence_frames=state.closing_evidence_frames,
            closing_evidence_duration_s=(
                None
                if state.closing_evidence_started_s is None
                else max(0.0, state.timestamp_s - state.closing_evidence_started_s)
            ),
        )
