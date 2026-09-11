#!/usr/bin/env bash
# Bounded timing benchmark. A pass is not calibration, GPIO or road validation.
# Directional update rates are measured by the runtime, never loop FPS/cadence.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_PATH="${1:-/etc/helmetai-fcw/pi5_dual_camera.json}"
FRAME_COUNT="${2:-300}"

if [[ ! "$FRAME_COUNT" =~ ^[1-9][0-9]*$ ]]; then
  echo "Frame count must be a positive integer." >&2
  exit 2
fi

LOG_FILE="$(mktemp)"
trap 'rm -f "$LOG_FILE"' EXIT
set +e
PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m helmetai_fcw pi-run --config "$CONFIG_PATH" --frames "$FRAME_COUNT" --camera-only | tee "$LOG_FILE"
RUNTIME_STATUS=${PIPESTATUS[0]}
set -e
if [[ $RUNTIME_STATUS -ne 0 ]]; then
  exit "$RUNTIME_STATUS"
fi

python3 - "$LOG_FILE" "$CONFIG_PATH" "$FRAME_COUNT" <<'PY'
import json
import math
import sys
from pathlib import Path


def configured_limits(config_path: str) -> tuple[list[str], float, float, float]:
    profile = json.loads(Path(config_path).read_text(encoding="utf-8"))
    directions = ["front"]
    rear = profile.get("rear_camera")
    if rear and rear.get("enabled", True):
        directions.append("rear")
    freshness = finite_number(profile.get("risk_assessment_freshness_s", 0.45), "freshness limit")
    minimum = finite_number(profile.get("minimum_risk_fps", 5.0), "configured minimum FPS")
    budget = finite_number(profile.get("latency_budget_ms", 150.0), "configured latency budget")
    if min(freshness, minimum, budget) <= 0:
        raise ValueError("configured limits must be positive")
    return directions, freshness, minimum, budget


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def validate(payload: dict[str, object]) -> None:
    if payload.get("completed") is not True or payload.get("error") is not None:
        raise ValueError("run did not complete successfully")
    frames = payload.get("processed_frames")
    if type(frames) is not int or frames != int(sys.argv[3]):
        raise ValueError("run did not process the requested number of frames")
    snapshot = payload["performance_final"]
    if not isinstance(snapshot, dict):
        raise ValueError("invalid performance_final")
    print("benchmark_p95_ms=", snapshot["p95_end_to_end_ms"], sep="")
    print("benchmark_fps=", snapshot["effective_fps"], sep="")
    print("benchmark_minimum_risk_fps=", snapshot.get("minimum_risk_fps"), sep="")
    for field, reason in (
        ("warmup_complete", "performance warm-up did not complete"),
        ("meets_latency_budget", "p95 latency exceeds the configured budget"),
        ("meets_minimum_risk_fps", "effective FPS is below the configured minimum"),
        ("risk_alerts_permitted", "runtime timing health gate is closed"),
    ):
        if snapshot.get(field) is not True:
            raise ValueError(reason)
    minimum = finite_number(snapshot["minimum_risk_fps"], "minimum risk FPS")
    if minimum <= 0:
        raise ValueError("minimum risk FPS must be positive")
    p95 = finite_number(snapshot["p95_end_to_end_ms"], "p95 latency")
    fps = finite_number(snapshot["effective_fps"], "effective FPS")
    if p95 < 0 or fps < minimum:
        raise ValueError("invalid latency or insufficient effective FPS")
    directions, freshness_limit, configured_minimum, latency_budget = configured_limits(sys.argv[2])
    if minimum != configured_minimum:
        raise ValueError("runtime minimum FPS does not match the requested profile")
    if p95 > latency_budget:
        raise ValueError("measured p95 exceeds the configured latency budget")
    directional = payload.get("directional_performance")
    if not isinstance(directional, dict):
        raise ValueError("measured directional_performance is missing")
    for direction in directions:
        measurement = directional.get(direction)
        if not isinstance(measurement, dict) or measurement.get("enabled") is not True:
            raise ValueError(f"{direction} measured update data is missing or disabled")
        count = measurement.get("processed_frames")
        if type(count) is not int or count < 2:
            raise ValueError(f"{direction} needs at least two measured updates")
        rate = finite_number(measurement.get("processed_fps"), f"{direction} measured FPS")
        age = finite_number(measurement.get("last_update_age_s"), f"{direction} update age")
        print(f"benchmark_{direction}_risk_fps={rate:.2f}")
        print(f"benchmark_{direction}_last_update_age_s={age:.3f}")
        if rate < minimum:
            raise ValueError(f"{direction} measured risk path is {rate:.2f} FPS; minimum is {minimum:.2f} FPS")
        if age < 0 or age > freshness_limit:
            raise ValueError(f"{direction} last measured update is stale or invalid")


for line in reversed(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()):
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        continue
    if not isinstance(payload, dict) or "performance_final" not in payload:
        continue
    try:
        validate(payload)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"BENCHMARK FAILED: {exc}.", file=sys.stderr)
        raise SystemExit(3)
    print("BENCHMARK PASSED (timing only; calibration and GPIO validation are separate)")
    raise SystemExit(0)
print("BENCHMARK FAILED: runtime did not emit performance_final.", file=sys.stderr)
raise SystemExit(4)
PY
