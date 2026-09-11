"""Save labelled front/rear CSI-camera images to verify physical mounting."""

from __future__ import annotations

import time
from pathlib import Path

from .pi_runtime import PiRuntimeConfig
from .rpi_camera import PiDualCameraRig


def save_camera_check_images(config: PiRuntimeConfig, output_directory: str | Path) -> tuple[Path, Path | None]:
    """Capture one labelled image from each configured Pi camera.

    This is a setup diagnostic, not a risk-analysis path. The rider verifies
    camera direction by covering each lens or placing a visible marker in
    front of it, then checks the saved FRONT/REAR files.
    """

    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - Pi integration path
        raise RuntimeError("OpenCV is required for pi-camera-check.") from exc

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    rig = PiDualCameraRig(config.front_camera, config.rear_camera)
    try:
        # Let auto-exposure and camera configuration settle before capture.
        time.sleep(0.35)
        front = rig.capture_front()
        rear = rig.capture_rear()
    finally:
        rig.close()

    front_path = _write_labelled_frame(
        cv2,
        front,
        destination / "front_camera_check.jpg",
        f"FRONT | camera_index={config.front_camera.camera_index} | lateral_sign={config.front_calibration.lateral_sign:+.0f}",
    )
    rear_path = None
    if rear is not None and config.rear_camera is not None and config.rear_calibration is not None:
        rear_path = _write_labelled_frame(
            cv2,
            rear,
            destination / "rear_camera_check.jpg",
            f"REAR | camera_index={config.rear_camera.camera_index} | lateral_sign={config.rear_calibration.lateral_sign:+.0f}",
        )
    return front_path, rear_path


def _write_labelled_frame(cv2: object, frame: object, path: Path, label: str) -> Path:
    """Write Picamera2 RGB888's BGR byte array without swapping channels.

    RGB888 is the libcamera format name, not the NumPy channel order. See
    Picamera2's official FORMAT_TABLE: RGB888 -> BGR. OpenCV imwrite also
    consumes BGR, so an RGB2BGR conversion here would invert red and blue.
    Copy before drawing: diagnostic labels must not mutate a captured frame.
    """

    image = frame.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 38), (25, 25, 25), -1)
    cv2.putText(image, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write camera-check image: {path}")
    return path
