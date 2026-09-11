"""Dependency-free stereo range math and measurement uncertainty propagation."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import median, pstdev
from typing import Iterable


@dataclass(frozen=True)
class StereoCalibration:
    """Minimal calibrated-stereo geometry needed by the runtime risk pipeline."""

    focal_length_px: float
    baseline_m: float
    minimum_disparity_px: float = 0.25

    def __post_init__(self) -> None:
        if self.focal_length_px <= 0 or self.baseline_m <= 0:
            raise ValueError("Focal length and baseline must be positive.")
        if self.minimum_disparity_px <= 0:
            raise ValueError("minimum_disparity_px must be positive.")

    def depth_from_disparity(self, disparity_px: float, disparity_sigma_px: float = 0.5) -> tuple[float, float]:
        """Return depth and propagated 1-sigma depth uncertainty in metres.

        Z = f * B / disparity. The local propagated uncertainty is
        sigma_Z = f * B / disparity^2 * sigma_disparity.
        """

        if disparity_px < self.minimum_disparity_px:
            raise ValueError("Disparity is too small for a reliable range estimate.")
        if disparity_sigma_px < 0:
            raise ValueError("disparity_sigma_px must not be negative.")
        depth_m = self.focal_length_px * self.baseline_m / disparity_px
        sigma_m = (
            self.focal_length_px
            * self.baseline_m
            * max(disparity_sigma_px, 1e-9)
            / (disparity_px * disparity_px)
        )
        return depth_m, sigma_m

    def robust_depth(self, disparities_px: Iterable[float], disparity_sigma_px: float = 0.5) -> tuple[float, float]:
        """Robust object depth from valid object-mask disparity samples.

        A detector bounding box contains road/background pixels. In production,
        feed this function disparities from an object mask or a conservative
        central region, not the whole bounding box.
        """

        valid = [d for d in disparities_px if d >= self.minimum_disparity_px]
        if not valid:
            raise ValueError("No valid disparities supplied.")
        centre = median(valid)
        depth_m, propagated_sigma_m = self.depth_from_disparity(centre, disparity_sigma_px)
        if len(valid) == 1:
            return depth_m, propagated_sigma_m

        depth_samples = [self.depth_from_disparity(d, disparity_sigma_px)[0] for d in valid]
        sample_sigma_m = pstdev(depth_samples)
        # Keep both the optical uncertainty and observed within-object spread.
        return depth_m, sqrt(propagated_sigma_m**2 + sample_sigma_m**2)

