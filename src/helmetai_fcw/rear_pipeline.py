"""Runtime composition for rear-camera collision-risk assessment."""

from __future__ import annotations

from dataclasses import dataclass

from .filtering import AlphaBetaTrackFilter
from .models import AlertLevel, ObservedTarget, RiskAssessment
from .rear_risk_engine import RearCollisionConfig, RearCollisionRiskEngine


@dataclass
class _AlertMemory:
    level: AlertLevel
    until_s: float


class RearCollisionPipeline:
    """One-call interface from a rear-camera observation to a risk assessment."""

    def __init__(
        self,
        config: RearCollisionConfig | None = None,
        filter_: AlphaBetaTrackFilter | None = None,
        alert_hold_s: float = 0.45,
    ) -> None:
        self.engine = RearCollisionRiskEngine(config)
        self.filter = filter_ or AlphaBetaTrackFilter()
        self.alert_hold_s = alert_hold_s
        self._alerts: dict[str, _AlertMemory] = {}

    def process(self, observation: ObservedTarget) -> RiskAssessment:
        track = self.filter.update(observation)
        assessment = self.engine.assess(track)
        memory = self._alerts.get(assessment.object_id)
        if (
            memory
            and assessment.level < memory.level
            and assessment.timestamp_s < memory.until_s
            and assessment.system_status.value == "ready"
        ):
            return RiskAssessment(
                timestamp_s=assessment.timestamp_s,
                object_id=assessment.object_id,
                level=memory.level,
                system_status=assessment.system_status,
                reasons=assessment.reasons + ("rear_alert_hold",),
                in_path=assessment.in_path,
                raw_ttc_s=assessment.raw_ttc_s,
                conservative_ttc_s=assessment.conservative_ttc_s,
                required_relative_decel_mps2=assessment.required_relative_decel_mps2,
                effective_distance_m=assessment.effective_distance_m,
                closing_speed_mps=assessment.closing_speed_mps,
                direction=assessment.direction,
            )
        if assessment.level > AlertLevel.NONE:
            self._alerts[assessment.object_id] = _AlertMemory(
                level=assessment.level,
                until_s=assessment.timestamp_s + self.alert_hold_s,
            )
        return assessment
