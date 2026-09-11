"""Coverage-aware scene mapping and software-only blind-spot decisions.

The current HelmetAI hardware has a front and a rear camera.  It cannot see
the motorcycle's true left/right blind zones without additional side-facing
cameras.  This module makes that limitation explicit: it maps what the two
cameras can observe and always reports both side zones as unobserved.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .models import AlertLevel, ObservedTarget, RiskAssessment


class TurnIntent(str, Enum):
    """Software input for a future turn-signal or head-gesture adapter."""

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"


class AwarenessZone(str, Enum):
    FRONT_LEFT = "front_left"
    FRONT_CENTRE = "front_centre"
    FRONT_RIGHT = "front_right"
    REAR_LEFT = "rear_left"
    REAR_CENTRE = "rear_centre"
    REAR_RIGHT = "rear_right"
    LEFT_SIDE_UNOBSERVED = "left_side_unobserved"
    RIGHT_SIDE_UNOBSERVED = "right_side_unobserved"


@dataclass(frozen=True)
class AwarenessItem:
    object_id: str
    object_class: str
    zone: AwarenessZone
    level: AlertLevel
    distance_m: float
    ttc_s: float | None


@dataclass(frozen=True)
class AwarenessSnapshot:
    """A partial environmental map that does not pretend to be 360 degrees."""

    items: tuple[AwarenessItem, ...]
    unobserved_zones: tuple[AwarenessZone, ...] = (
        AwarenessZone.LEFT_SIDE_UNOBSERVED,
        AwarenessZone.RIGHT_SIDE_UNOBSERVED,
    )

    @property
    def coverage(self) -> str:
        return "front_rear_partial_side_zones_unobserved"

    def highest_level_by_zone(self) -> dict[str, str]:
        result: dict[str, AlertLevel] = {}
        for item in self.items:
            previous = result.get(item.zone.value, AlertLevel.NONE)
            result[item.zone.value] = max(previous, item.level)
        return {zone: level.name for zone, level in result.items()}

    def as_dict(self) -> dict[str, object]:
        return {
            "coverage": self.coverage,
            "highest_level_by_zone": self.highest_level_by_zone(),
            "unobserved_zones": [zone.value for zone in self.unobserved_zones],
        }


@dataclass(frozen=True)
class BlindSpotConfig:
    """Rear-camera lateral proximity policy pending side-camera validation."""

    supported_classes: frozenset[str] = frozenset({"car", "truck", "bus", "motorcycle", "bicycle"})
    minimum_confidence: float = 0.55
    nearest_longitudinal_m: float = 0.5
    farthest_longitudinal_m: float = 10.0
    minimum_lateral_m: float = 0.65
    maximum_lateral_m: float = 3.5


@dataclass(frozen=True)
class BlindSpotAssessment:
    object_id: str
    object_class: str
    side: TurnIntent | None
    nearby_lateral_target: bool
    warning_active: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "target": self.object_id,
            "class": self.object_class,
            "side": None if self.side is None else self.side.value,
            "nearby_lateral_target": self.nearby_lateral_target,
            "warning_active": self.warning_active,
            "reasons": list(self.reasons),
        }


class BlindSpotMonitor:
    """Evaluate rear-visible lateral targets when a software turn intent exists.

    This is intentionally a rear-camera assist, not a claim of complete blind
    spot coverage.  ``TurnIntent`` can later be supplied by a signal wire,
    IMU/head-gesture module, or application command without changing the
    decision policy.
    """

    def __init__(self, config: BlindSpotConfig | None = None) -> None:
        self.config = config or BlindSpotConfig()

    def assess(self, target: ObservedTarget, turn_intent: TurnIntent = TurnIntent.NONE) -> BlindSpotAssessment:
        target_class = target.object_class.lower()
        if not target.calibration_valid:
            return BlindSpotAssessment(target.object_id, target_class, None, False, False, ("rear_calibration_invalid",))
        if target.confidence < self.config.minimum_confidence:
            return BlindSpotAssessment(target.object_id, target_class, None, False, False, ("perception_confidence_below_minimum",))
        if target_class not in self.config.supported_classes:
            return BlindSpotAssessment(target.object_id, target_class, None, False, False, ("unsupported_blind_spot_class",))
        if not self.config.nearest_longitudinal_m <= target.longitudinal_m <= self.config.farthest_longitudinal_m:
            return BlindSpotAssessment(target.object_id, target_class, None, False, False, ("target_outside_blind_spot_range",))
        if not self.config.minimum_lateral_m <= abs(target.lateral_m) <= self.config.maximum_lateral_m:
            return BlindSpotAssessment(target.object_id, target_class, None, False, False, ("target_not_in_lateral_band",))

        side = TurnIntent.RIGHT if target.lateral_m > 0 else TurnIntent.LEFT
        return BlindSpotAssessment(
            object_id=target.object_id,
            object_class=target_class,
            side=side,
            nearby_lateral_target=True,
            warning_active=turn_intent == side,
            reasons=("rear_lateral_target_visible", "turn_intent_matched" if turn_intent == side else "turn_intent_not_matched"),
        )


class CoverageAwareRiskMap:
    """Place calibrated front/rear risk targets in a coverage-aware map."""

    def __init__(self, centre_band_m: float = 0.55) -> None:
        if centre_band_m <= 0:
            raise ValueError("centre_band_m must be positive.")
        self.centre_band_m = centre_band_m

    def build(
        self,
        front_targets: Iterable[tuple[ObservedTarget, RiskAssessment]],
        rear_targets: Iterable[tuple[ObservedTarget, RiskAssessment]],
    ) -> AwarenessSnapshot:
        items = [self._item(target, assessment, front=True) for target, assessment in front_targets]
        items.extend(self._item(target, assessment, front=False) for target, assessment in rear_targets)
        return AwarenessSnapshot(tuple(items))

    def _item(self, target: ObservedTarget, assessment: RiskAssessment, front: bool) -> AwarenessItem:
        zone = self._zone(front, target.lateral_m)
        return AwarenessItem(
            object_id=target.object_id,
            object_class=target.object_class,
            zone=zone,
            level=assessment.level,
            distance_m=target.longitudinal_m,
            ttc_s=assessment.conservative_ttc_s,
        )

    def _zone(self, front: bool, lateral_m: float) -> AwarenessZone:
        if abs(lateral_m) < self.centre_band_m:
            return AwarenessZone.FRONT_CENTRE if front else AwarenessZone.REAR_CENTRE
        if lateral_m < 0:
            return AwarenessZone.FRONT_LEFT if front else AwarenessZone.REAR_LEFT
        return AwarenessZone.FRONT_RIGHT if front else AwarenessZone.REAR_RIGHT
