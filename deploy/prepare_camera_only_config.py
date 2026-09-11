"""Validate an installed profile and stage a camera-only copy, preserving values.

Run by update_pi.sh with the release src on PYTHONPATH. Validation uses the
original profile path so relative model paths retain their existing meaning.
No camera, model inference or GPIO device is opened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def camera_only_payload(raw: str) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value is not permitted: {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate configuration key: {key}")
            result[key] = value
        return result

    payload = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
    if not isinstance(payload, dict):
        raise ValueError("Installed profile must be a JSON object.")
    actuator = payload.get("alert_actuator")
    if actuator is None:
        actuator = {}
        payload["alert_actuator"] = actuator
    if not isinstance(actuator, dict):
        raise ValueError("alert_actuator must be a JSON object or null.")
    actuator["enabled"] = False
    return payload


def prepare_config(config_path: Path, output_path: Path) -> None:
    from helmetai_fcw.pi_runtime import PiRuntimeConfig

    if config_path.resolve() == output_path.resolve():
        raise ValueError("The staging path must not overwrite the installed profile.")
    payload = camera_only_payload(config_path.read_text(encoding="utf-8"))
    # Validate the original before changing anything. An invalid current
    # configuration is surfaced for manual review, not silently repaired.
    runtime = PiRuntimeConfig.from_json(config_path)
    if not Path(runtime.model_path).is_file():
        raise ValueError(f"Existing detector model is missing: {runtime.model_path}")
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: prepare_camera_only_config.py EXISTING_CONFIG STAGED_CONFIG")
    prepare_config(Path(sys.argv[1]), Path(sys.argv[2]))
