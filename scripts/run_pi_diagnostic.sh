#!/usr/bin/env bash
# Camera-only, bounded bring-up run. Logs are retained even when the run fails.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${1:-/etc/helmetai-fcw/pi5_dual_camera.json}"
FRAME_COUNT="${2:-100}"
if [[ ! "$FRAME_COUNT" =~ ^[1-9][0-9]{0,3}$ ]] || (( FRAME_COUNT > 1000 )); then
  echo "Frame count must be between 1 and 1000." >&2
  exit 2
fi
if [[ "$EUID" == 0 ]]; then
  echo "Run diagnostics as your normal Pi user, without sudo." >&2
  exit 2
fi
mkdir -p "$HOME/helmetai-diagnostics"
LOG_DIR="$(mktemp -d "$HOME/helmetai-diagnostics/run.XXXXXX")"
echo "Diagnostic directory (keep these files): $LOG_DIR"
hardware_status() {
  date -u
  if [[ -f /proc/device-tree/model ]]; then
    tr -d '\0' < /proc/device-tree/model
    echo
  fi
  if [[ -f /etc/os-release ]]; then
    cat /etc/os-release
  fi
  uname -m
  python3 --version
  python3 -c 'import cv2, numpy; print("opencv=" + cv2.__version__); print("numpy=" + numpy.__version__)' || true
  if command -v vcgencmd >/dev/null 2>&1; then
    vcgencmd measure_temp || true
    vcgencmd get_throttled || true
    vcgencmd measure_clock arm || true
  fi
}
hardware_status > "$LOG_DIR/before.txt" 2>&1
set +e
PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m helmetai_fcw pi-run --config "$CONFIG_PATH" --frames "$FRAME_COUNT" --camera-only \
  2>&1 | tee "$LOG_DIR/runtime.log"
RUN_STATUSES=("${PIPESTATUS[@]}")
set -e
hardware_status > "$LOG_DIR/after.txt" 2>&1
echo "Diagnostic files: $LOG_DIR"
echo "This is not a passed performance/safety benchmark; do not enable alerts."
if [[ ${RUN_STATUSES[0]} -ne 0 ]]; then
  exit "${RUN_STATUSES[0]}"
fi
exit "${RUN_STATUSES[1]}"
