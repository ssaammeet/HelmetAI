#!/usr/bin/env bash
# Validate the install, CSI-camera visibility and configuration before a
# bounded smoke test or latency benchmark. This script never enables a service
# and never changes calibration values.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${1:-/etc/helmetai-fcw/pi5_dual_camera.json}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "HelmetAI config was not found: $CONFIG_PATH" >&2
  exit 2
fi
if ! command -v rpicam-hello >/dev/null 2>&1; then
  echo "rpicam-hello is unavailable. Install the Raspberry Pi OS camera packages first." >&2
  exit 2
fi

echo "== Detected CSI cameras =="
rpicam-hello --list-cameras
echo
echo "== HelmetAI configuration check =="
PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path

from helmetai_fcw.alert_actuator import resolve_gpio_chip
from helmetai_fcw.pi_runtime import PiRuntimeConfig
from picamera2 import Picamera2

config = PiRuntimeConfig.from_json(sys.argv[1])
model = Path(config.model_path)
if not model.is_file():
    raise SystemExit(f"Detector model is missing: {model}")
available = Picamera2.global_camera_info()
requested = [config.front_camera.camera_index]
if config.rear_camera is not None:
    requested.append(config.rear_camera.camera_index)
if any(index >= len(available) for index in requested):
    raise SystemExit(f"Configured camera indexes {requested} are unavailable; Picamera2 detected: {available}")
print(f"model={model}")
print(f"front_camera_index={config.front_camera.camera_index}")
print(f"rear_camera_index={None if config.rear_camera is None else config.rear_camera.camera_index}")
for index in requested:
    print(f"camera_info[{index}]={available[index]}")
print(f"detector_input_size_px={config.detector_input_size_px}")
print(f"front_calibrated={config.front_calibration.calibrated}")
print(f"rear_calibrated={None if config.rear_calibration is None else config.rear_calibration.calibrated}")
print(f"minimum_risk_fps={config.minimum_risk_fps}")
print(f"performance_warmup_frames={config.performance_warmup_frames}")
print(f"alert_actuator_enabled={config.alert_actuator.enabled}")
if config.alert_actuator.enabled:
    try:
        import lgpio
    except ImportError as exc:
        raise SystemExit("alert_actuator is enabled but python3-lgpio is unavailable") from exc
    try:
        gpio_chip_number = resolve_gpio_chip(config.alert_actuator.gpio_chip)
        gpio_chip = lgpio.gpiochip_open(gpio_chip_number)
    except Exception as exc:
        raise SystemExit(
            "alert_actuator is enabled but the current user cannot open "
            f"gpiochip{config.alert_actuator.gpio_chip if config.alert_actuator.gpio_chip is not None else 'auto'}: {exc}"
        ) from exc
    else:
        try:
            print(f"gpio_chip_access=ok (gpiochip{gpio_chip_number})")
        finally:
            lgpio.gpiochip_close(gpio_chip)
    print(f"buzzer_bcm_pin={config.alert_actuator.buzzer_bcm_pin}")
    print(f"vibration_bcm_pin={config.alert_actuator.vibration_bcm_pin}")
    print(f"led_bcm_pin={config.alert_actuator.led_bcm_pin}")
if config.detector_input_size_px != 640:
    raise SystemExit("This packaged detector requires detector.input_size_px=640.")
try:
    import cv2
    import numpy as np

    network = cv2.dnn.readNetFromONNX(str(model))
    blank = np.zeros((config.detector_input_size_px, config.detector_input_size_px, 3), dtype=np.uint8)
    network.setInput(cv2.dnn.blobFromImage(blank, scalefactor=1.0 / 255.0, size=(config.detector_input_size_px, config.detector_input_size_px), swapRB=True))
    output = network.forward()
except Exception as exc:
    raise SystemExit(f"Detector OpenCV smoke check failed: {exc}") from exc
print(f"detector_smoke_output_shape={tuple(output.shape)}")
if not config.front_calibration.calibrated or (config.rear_calibration and not config.rear_calibration.calibrated):
    print("WARNING: detection smoke testing is allowed; each uncalibrated direction keeps its TTC/risk alerts disabled.")
PY

echo
echo "Preflight passed. Save labelled verification images next:"
echo "PYTHONPATH=$PROJECT_DIR/src python3 -m helmetai_fcw pi-camera-check --config $CONFIG_PATH --output-dir ~/helmetai-camera-check"
echo "Open front_camera_check.jpg and rear_camera_check.jpg, then cover one physical lens at a time to confirm the labels and lateral_sign values."
echo "Next controlled test: PYTHONPATH=$PROJECT_DIR/src python3 -m helmetai_fcw pi-run --config $CONFIG_PATH --frames 100"
