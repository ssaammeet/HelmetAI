#!/usr/bin/env bash
# JPEG-only detector diagnostic: never opens a camera/GPIO or changes config.
# Example: bash scripts/profile_detector_pi.sh --image captured.jpg \
#   --config /etc/helmetai-fcw/pi5_dual_camera.json --output detector-profile.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m helmetai_fcw.detector_profile "$@"
