"""Calibrated single-camera range estimation for the Raspberry Pi MVP.

The first HelmetAI hardware version uses one front and one rear camera, not a
stereo pair. A single camera cannot recover metric depth without assumptions.
This module therefore uses a calibrated pinhole model and an expected physical
width per detected class, then exposes the resulting uncertainty to the risk
engine instead of treating the estimate as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from .detection import Detection
from .models import ObservedTarget


@dataclass(frozen=True)
class ObjectWidthModel:
    """Expected class width and 1-sigma class-size variation, in metres."""

    width_m: float
    sigma_m: float


DEFAULT_OBJECT_WIDTHS: dict[str, ObjectWidthModel] = {
    # Pedestrian and bicycle widths are deliberately assigned larger
    # uncertainty than vehicles.  The values allow an early, conservative
    # forward-risk estimate but must be validated with the installed camera.
    "person": ObjectWidthModel(0.45, 0.20),
    "bicycle": ObjectWidthModel(0.70, 0.25),
    "car": ObjectWidthModel(1.80, 0.18),
    "truck": ObjectWidthModel(2.50, 0.25),
    "bus": ObjectWidthModel(2.55, 0.20),
    "motorcycle": ObjectWidthModel(0.85, 0.15),
    "vehicle": ObjectWidthModel(1.80, 0.30),
}


@dataclass(frozen=True)
class MonocularCalibration:
    """Pinhole calibration for one camera in its mounted orientation."""

    focal_length_px: float
    principal_point_x_px: float
    image_width_px: int
    image_height_px: int
    pixel_width_sigma_px: float = 2.0
    # +1 maps image-right to rider-right; -1 maps image-right to rider-left.
    # A rear-facing camera normally needs -1, but this must be confirmed with
    # the camera-check images after the final helmet installation.
    lateral_sign: float = 1.0
    calibrated: bool = False
    object_widths: dict[str, ObjectWidthModel] = field(default_factory=lambda: dict(DEFAULT_OBJECT_WIDTHS))

    def __post_init__(self) -> None:
        if self.focal_length_px <= 0:
            raise ValueError("focal_length_px must be positive.")
        if self.image_width_px <= 0 or self.image_height_px <= 0:
            raise ValueError("image dimensions must be positive.")
        if self.lateral_sign not in {-1.0, 1.0}:
            raise ValueError("lateral_sign must be either -1.0 or 1.0.")


class MonocularRangeEstimator:
    """Estimate target range/lateral position from a calibrated detection box."""

    def __init__(self, calibration: MonocularCalibration) -> None:
        self.calibration = calibration

    def supports_range(self, detection: Detection) -> bool:
        """Whether this class has a calibrated physical-width model.

        Traffic controls and generic obstacles remain visible to perception,
        but they are not turned into a false metric/TTC value unless a
        class-specific physical model has been calibrated.
        """

        return detection.label.lower() in self.calibration.object_widths and detection.width_px > 0 and detection.height_px > 0

    def estimate(self, timestamp_s: float, detection: Detection) -> ObservedTarget:
        label = detection.label.lower()
        width_model = self.calibration.object_widths.get(label)
        if width_model is None:
            raise ValueError(f"Unsupported class for monocular range: {detection.label}")
        if detection.width_px <= 0:
            raise ValueError("Detection width must be positive.")

        # Z = f * W / w. The error combines vehicle-width variation and pixel
        # localisation uncertainty. This is intentionally conservative.
        range_m = self.calibration.focal_length_px * width_model.width_m / detection.width_px
        relative_width_sigma = sqrt(
            (width_model.sigma_m / width_model.width_m) ** 2
            + (self.calibration.pixel_width_sigma_px / detection.width_px) ** 2
        )
        range_sigma_m = range_m * relative_width_sigma
        centre_x = detection.x_px + detection.width_px / 2.0
        lateral_m = (
            self.calibration.lateral_sign
            * (centre_x - self.calibration.principal_point_x_px)
            * range_m
            / self.calibration.focal_length_px
        )
        return ObservedTarget(
            timestamp_s=timestamp_s,
            object_id=detection.object_id,
            object_class=label,
            longitudinal_m=range_m,
            lateral_m=lateral_m,
            range_sigma_m=range_sigma_m,
            confidence=detection.confidence,
            calibration_valid=self.calibration.calibrated,
            frame_valid=True,
        )


def focal_length_from_reference(reference_distance_m: float, object_width_m: float, measured_width_px: float) -> float:
    """Compute focal length from a flat, front-facing reference target."""

    if reference_distance_m <= 0 or object_width_m <= 0 or measured_width_px <= 0:
        raise ValueError("Reference distance, physical width and pixel width must be positive.")
    return measured_width_px * reference_distance_m / object_width_m
