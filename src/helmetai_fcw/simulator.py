"""Deterministic synthetic scenarios for FCW development and regression tests."""

from __future__ import annotations

from dataclasses import dataclass
from math import sin

from .models import ObservedTarget, RiskAssessment
from .pipeline import ForwardCollisionPipeline
from .rear_pipeline import RearCollisionPipeline
from .stereo import StereoCalibration


@dataclass(frozen=True)
class ClosingScenario:
    """A same-path lead vehicle that is being closed at a constant relative rate."""

    initial_distance_m: float = 48.0
    closing_speed_mps: float = 11.0
    lateral_m: float = 0.15
    duration_s: float = 6.0
    step_s: float = 0.10
    object_id: str = "lead-car-1"


@dataclass(frozen=True)
class RearClosingScenario:
    """A vehicle behind the rider that reduces its separation distance."""

    initial_distance_m: float = 36.0
    closing_speed_mps: float = 9.0
    lateral_m: float = -0.10
    duration_s: float = 5.0
    step_s: float = 0.10
    object_id: str = "rear-car-1"


def closing_vehicle_observations(scenario: ClosingScenario) -> list[ObservedTarget]:
    """Create stereo-derived observations with deterministic sub-pixel noise."""

    calibration = StereoCalibration(focal_length_px=700.0, baseline_m=0.12)
    observations: list[ObservedTarget] = []
    sample_count = int(scenario.duration_s / scenario.step_s) + 1
    for index in range(sample_count):
        timestamp_s = index * scenario.step_s
        true_distance = max(2.0, scenario.initial_distance_m - scenario.closing_speed_mps * timestamp_s)
        ideal_disparity = calibration.focal_length_px * calibration.baseline_m / true_distance
        # A controlled-bench, sub-pixel stereo example. Field validation must
        # replace these values with measured disparity error distributions.
        measured_disparity = ideal_disparity + 0.025 * sin(index * 1.73)
        measured_distance, sigma_m = calibration.depth_from_disparity(measured_disparity, disparity_sigma_px=0.10)
        observations.append(
            ObservedTarget(
                timestamp_s=timestamp_s,
                object_id=scenario.object_id,
                object_class="car",
                longitudinal_m=measured_distance,
                lateral_m=scenario.lateral_m + 0.03 * sin(index * 0.31),
                range_sigma_m=sigma_m,
                confidence=0.93,
                calibration_valid=True,
                frame_valid=True,
            )
        )
    return observations


def run_closing_demo(scenario: ClosingScenario | None = None) -> list[RiskAssessment]:
    pipeline = ForwardCollisionPipeline()
    return [pipeline.process(observation) for observation in closing_vehicle_observations(scenario or ClosingScenario())]


def rear_closing_vehicle_observations(scenario: RearClosingScenario) -> list[ObservedTarget]:
    """Create deterministic rear-camera observations for regression testing."""

    calibration = StereoCalibration(focal_length_px=700.0, baseline_m=0.12)
    observations: list[ObservedTarget] = []
    sample_count = int(scenario.duration_s / scenario.step_s) + 1
    for index in range(sample_count):
        timestamp_s = index * scenario.step_s
        true_distance = max(2.0, scenario.initial_distance_m - scenario.closing_speed_mps * timestamp_s)
        ideal_disparity = calibration.focal_length_px * calibration.baseline_m / true_distance
        measured_disparity = ideal_disparity + 0.025 * sin(index * 1.41)
        measured_distance, sigma_m = calibration.depth_from_disparity(measured_disparity, disparity_sigma_px=0.10)
        observations.append(
            ObservedTarget(
                timestamp_s=timestamp_s,
                object_id=scenario.object_id,
                object_class="car",
                longitudinal_m=measured_distance,
                lateral_m=scenario.lateral_m + 0.03 * sin(index * 0.23),
                range_sigma_m=sigma_m,
                confidence=0.93,
                calibration_valid=True,
                frame_valid=True,
            )
        )
    return observations


def run_rear_closing_demo(scenario: RearClosingScenario | None = None) -> list[RiskAssessment]:
    pipeline = RearCollisionPipeline()
    return [
        pipeline.process(observation)
        for observation in rear_closing_vehicle_observations(scenario or RearClosingScenario())
    ]
