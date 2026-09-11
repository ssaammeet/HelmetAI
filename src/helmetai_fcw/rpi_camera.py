"""Picamera2 camera lifecycle for the front/rear Raspberry Pi camera pair."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PiCameraConfig:
    camera_index: int
    width_px: int = 640
    height_px: int = 480
    fps: int = 15

    def __post_init__(self) -> None:
        if self.camera_index < 0:
            raise ValueError("camera_index must be non-negative.")
        if self.width_px <= 0 or self.height_px <= 0:
            raise ValueError("camera width and height must be positive.")
        if self.fps <= 0:
            raise ValueError("camera fps must be positive.")


class PiDualCameraRig:
    """Controls both Pi CSI cameras from one Python process.

    The Picamera2 import is delayed so the regular desktop demo needs no Pi
    packages. ``camera_index`` values are inspected at startup and must match
    the physical order reported by ``rpicam-hello --list-cameras``.
    """

    def __init__(self, front: PiCameraConfig, rear: PiCameraConfig | None = None) -> None:
        try:
            from picamera2 import Picamera2  # type: ignore
        except ImportError as exc:  # pragma: no cover - only executable on Pi
            raise RuntimeError(
                "Picamera2 bulunamadı. Raspberry Pi OS üzerinde 'sudo apt install python3-picamera2' çalıştırın."
            ) from exc
        self._picamera_class = Picamera2
        available = Picamera2.global_camera_info()
        requested = [front.camera_index] + ([] if rear is None else [rear.camera_index])
        if len(set(requested)) != len(requested):
            raise RuntimeError("Front and rear cameras must use different camera_index values.")
        if len(available) < len(requested) or any(index < 0 or index >= len(available) for index in requested):
            raise RuntimeError(f"Beklenen kameralar bulunamadı. Picamera2 algıladığı kameralar: {available}")
        self.front = None
        self.rear = None
        try:
            self.front = self._open("front", front)
            self.rear = self._open("rear", rear) if rear is not None else None
        except Exception:
            try:
                self.close()
            except Exception:
                # Preserve the original open/configuration error while still
                # attempting to release the first camera.
                pass
            raise

    def _open(self, role: str, config: PiCameraConfig):  # pragma: no cover - only executable on Pi
        try:
            camera = self._picamera_class(config.camera_index)
            frame_duration_us = int(1_000_000 / config.fps)
            video_config = camera.create_video_configuration(
                main={"size": (config.width_px, config.height_px), "format": "RGB888"},
                controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
            )
            camera.configure(video_config)
            camera.start()
            return camera
        except Exception as exc:
            raise RuntimeError(f"{role} camera could not be opened (camera_index={config.camera_index}): {exc}") from exc

    def capture_front(self):  # pragma: no cover - only executable on Pi
        if self.front is None:
            raise RuntimeError("Front camera is not open.")
        frame = self.front.capture_array("main")
        if frame is None:
            raise RuntimeError("Front camera returned an empty frame.")
        return frame

    def capture_rear(self):  # pragma: no cover - only executable on Pi
        if self.rear is None:
            return None
        frame = self.rear.capture_array("main")
        if frame is None:
            raise RuntimeError("Rear camera returned an empty frame.")
        return frame

    def close(self) -> None:  # pragma: no cover - only executable on Pi
        errors: list[str] = []
        for camera in (self.front, self.rear):
            if camera is not None:
                try:
                    camera.stop()
                except Exception as exc:
                    errors.append(f"stop: {exc}")
                try:
                    camera.close()
                except Exception as exc:
                    errors.append(f"close: {exc}")
        self.front = None
        self.rear = None
        if errors:
            raise RuntimeError("One or more Pi cameras did not close cleanly: " + "; ".join(errors))
