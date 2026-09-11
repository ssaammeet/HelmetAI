"""Read-only, dependency-free analysis of saved HelmetAI runtime logs.

Run ``python -m helmetai_fcw.log_report --runtime runtime.log --output-dir NEW_DIR``.
Only complete runtime_status.detections are counted. A pasted log can contain
several runs, duplicated status lines, cached counters and an incomplete tail.
None of these offline observations certifies recognition, power or road safety.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence


FLAG_NAMES = {
    0: "under_voltage", 1: "arm_frequency_capped", 2: "throttled", 3: "soft_temperature_limit",
    16: "under_voltage", 17: "arm_frequency_capped", 18: "throttled", 19: "soft_temperature_limit",
}
KNOWN_FLAG_MASK = sum(1 << bit for bit in FLAG_NAMES)
LIMITATIONS = [
    "Confidence is a model score, not measured accuracy or a probability of correct recognition.",
    "Detection counts are frame-level outputs, not counts of physical people or vehicles.",
    "Only fresh runtime_status.detections are counted; assessment records and cached directional counters are excluded.",
    "Front/rear are configured software roles; physical lens mapping and label correctness need matching images.",
    "NONE is not an all-clear. Missing calibration, insufficient FPS or stale assessments may suppress alerts.",
    "A missing final record or malformed/truncated paste does not establish that the runtime crashed.",
    "Short or pre-warmup timing is not sustained performance, power suitability or safety certification.",
]


def _finite_number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _new_run(index: int, line: int, directory: str | None, reason: str) -> dict[str, Any]:
    return {
        "run_id": f"run-{index}", "diagnostic_directory": directory,
        "boundary_reason": reason, "line_start": line, "line_end": line,
        "runtime_start": None, "runtime_final": None, "statuses": [],
        "duplicate_status_records": 0, "duplicate_final_records": 0,
        "issues": [], "invalid_json_lines": [],
    }


def _has_records(run: dict[str, Any]) -> bool:
    return bool(run["runtime_start"] or run["runtime_final"] or run["statuses"])


def _parse_runtime(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    content = path.read_bytes()
    lines = content.decode("utf-8-sig", errors="replace").splitlines()
    runs: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    quality: dict[str, Any] = {
        "total_lines": len(lines), "complete_json_lines": 0, "ignored_json_records": 0,
        "non_json_lines": 0, "invalid_json_lines": [], "replacement_characters_present": "\ufffd" in content.decode("utf-8-sig", errors="replace"),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    last_nonempty = max((i for i, line in enumerate(lines, 1) if line.strip()), default=0)

    def begin(line: int, directory: str | None, reason: str) -> dict[str, Any]:
        run = _new_run(len(runs) + 1, line, directory, reason)
        runs.append(run)
        return run

    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        marker = re.match(r"^Diagnostic directory \(keep these files\):\s*(.+)$", line)
        if marker:
            directory = marker.group(1)
            if current is None or _has_records(current) or current["diagnostic_directory"] != directory:
                current = begin(line_number, directory, "diagnostic_directory")
            else:
                current["line_end"] = line_number
            continue
        if not line.startswith("{"):
            if line:
                quality["non_json_lines"] += 1
            continue
        try:
            record = json.loads(line, parse_constant=_reject_constant)
        except (ValueError, RecursionError) as exc:
            issue = {
                "line": line_number, "kind": "malformed_or_truncated_json",
                "at_input_tail": line_number == last_nonempty,
                "detail": str(exc)[:240],
            }
            quality["invalid_json_lines"].append(issue)
            if current is not None:
                current["invalid_json_lines"].append(line_number)
                current["line_end"] = line_number
            continue
        quality["complete_json_lines"] += 1
        event = record.get("event") if isinstance(record, dict) else None
        if event not in ("runtime_start", "runtime_status", "runtime_final"):
            quality["ignored_json_records"] += 1
            continue
        if current is None:
            current = begin(line_number, None, "first_runtime_record")
        if event == "runtime_start":
            if _has_records(current):
                current = begin(line_number, None, "runtime_start")
            current["runtime_start"] = record
        elif event == "runtime_status":
            frame = record.get("frame_count")
            if not isinstance(frame, int) or isinstance(frame, bool) or frame < 1:
                current["issues"].append({"line": line_number, "kind": "invalid_frame_count"})
                continue
            prior = next((item for item in current["statuses"] if item[1]["frame_count"] == frame), None)
            if prior is not None and prior[1] == record:
                current["duplicate_status_records"] += 1
                current["line_end"] = line_number
                continue
            last_frame = current["statuses"][-1][1]["frame_count"] if current["statuses"] else 0
            if current["runtime_final"] is not None or frame < last_frame:
                current = begin(line_number, None, "inferred_frame_reset_or_after_final")
                current["issues"].append({"line": line_number, "kind": "run_boundary_inferred_without_start"})
            elif prior is not None:
                current["issues"].append({"line": line_number, "kind": "conflicting_duplicate_frame_ignored", "frame_count": frame})
                continue
            current["statuses"].append((line_number, record))
        else:
            if current["runtime_final"] == record:
                current["duplicate_final_records"] += 1
            elif current["runtime_final"] is not None:
                current = begin(line_number, None, "additional_final_without_start")
                current["runtime_final"] = record
            else:
                current["runtime_final"] = record
        current["line_end"] = line_number
    return runs, quality


def _group_detections(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record[field], []).append(record)
    result = {}
    for label, rows in sorted(groups.items()):
        scores = [row["confidence"] for row in rows if row["confidence"] is not None]
        result[label] = {
            "count": len(rows), "frame_indices": sorted({row["frame_index"] for row in rows}),
            "classes": sorted({row["class"] for row in rows}),
            "directions": sorted({row["direction"] for row in rows}),
            "tracker_ids": sorted({row["target"] for row in rows if row["target"] is not None}),
            "confidence": {"minimum": min(scores) if scores else None, "maximum": max(scores) if scores else None,
                           "mean": round(statistics.mean(scores), 6) if scores else None, "valid_scores": len(scores)},
        }
    return result


def _bbox_iou(first: Any, second: Any) -> float:
    if not all(isinstance(box, list) and len(box) == 4 and all(_finite_number(v) for v in box)
               and box[2] > 0 and box[3] > 0 for box in (first, second)):
        return 0.0
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return overlap / (aw * ah + bw * bh - overlap)


def _summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    statuses = run["statuses"]
    detections = []
    overlaps = []
    for line_number, status in statuses:
        raw_detections = status.get("detections")
        if not isinstance(raw_detections, list):
            run["issues"].append({"line": line_number, "kind": "detections_missing_or_not_list"})
            continue
        frame_records = []
        for detection in raw_detections:
            if not isinstance(detection, dict) or not isinstance(detection.get("class"), str):
                run["issues"].append({"line": line_number, "kind": "invalid_detection_record"})
                continue
            score = detection.get("confidence")
            if not _finite_number(score) or not 0 <= score <= 1:
                score = None
                run["issues"].append({"line": line_number, "kind": "missing_or_invalid_confidence"})
            direction = detection.get("direction")
            target = detection.get("target")
            item = {
                "line": line_number, "frame_index": status["frame_count"], "class": detection["class"],
                "direction": direction if isinstance(direction, str) else "unknown",
                "target": target if isinstance(target, str) else None,
                "confidence": score, "bbox_xywh_px": detection.get("bbox_xywh_px"),
            }
            frame_records.append(item)
        for i, first in enumerate(frame_records):
            for second in frame_records[i + 1:]:
                if first["direction"] != second["direction"] or first["class"] == second["class"]:
                    continue
                iou = _bbox_iou(first["bbox_xywh_px"], second["bbox_xywh_px"])
                if iou >= 0.8:
                    overlaps.append({"frame_index": status["frame_count"], "direction": first["direction"],
                                     "classes": [first["class"], second["class"]], "iou": round(iou, 6),
                                     "interpretation": "Overlapping different-class outputs; correct class is not verified."})
        detections.extend(frame_records)
    final = run["runtime_final"]
    completed = final.get("completed") if final else None
    completed = completed if isinstance(completed, bool) else None
    error = final.get("error") if final else None
    if completed is True and error is None:
        completion = "completed"
    elif error is not None:
        completion = "reported_error"
    elif completed is False:
        completion = "reported_not_completed"
    else:
        completion = "completion_unknown"
    if final and isinstance(final.get("performance_final"), dict):
        performance_record, metrics, source = final, final["performance_final"], "runtime_final"
    elif statuses:
        performance_record = statuses[-1][1]
        metrics = performance_record.get("performance")
        metrics = metrics if isinstance(metrics, dict) else None
        source = "last_runtime_status"
    else:
        performance_record, metrics, source = {}, None, "unknown"
    start = run["runtime_start"] or {}
    return {
        "run_id": run["run_id"], "diagnostic_directory": run["diagnostic_directory"],
        "boundary_reason": run["boundary_reason"], "line_start": run["line_start"], "line_end": run["line_end"],
        "start_present": bool(run["runtime_start"]), "final_present": final is not None,
        "release": start.get("release") or (final or {}).get("release"),
        "completion_state": completion, "reported_completed": completed, "reported_error": error,
        "reported_final_processed_frames": final.get("processed_frames") if final else None,
        "status_record_count": len(statuses), "status_frame_indices": [s[1]["frame_count"] for s in statuses],
        "duplicate_status_records_ignored": run["duplicate_status_records"],
        "duplicate_final_records_ignored": run["duplicate_final_records"],
        "invalid_json_lines": run["invalid_json_lines"], "issues": run["issues"],
        "reported_configuration": {key: start.get(key) for key in (
            "front_calibrated", "rear_calibrated", "alert_actuator_enabled")},
        "performance": {"source": source, "partial": source != "runtime_final" or completion != "completed",
                        "metrics": metrics, "directional": performance_record.get("directional_performance")},
        "detection_summary": {"total_records": len(detections),
                              "frames_with_detections": sorted({d["frame_index"] for d in detections}),
                              "by_class": _group_detections(detections, "class"),
                              "by_direction": _group_detections(detections, "direction")},
        "detection_records": detections, "overlapping_cross_class_boxes": overlaps,
        "recognition_accuracy_validated": False, "physical_camera_mapping_validated": False,
        "risk_readiness_validated": False,
    }


def decode_throttled(value: int) -> dict[str, Any]:
    """Decode firmware current bits 0..3 and historical bits 16..19."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Throttled value must be a non-negative integer.")
    return {
        "value": value, "hex": hex(value),
        "current_flags": [name for bit, name in FLAG_NAMES.items() if bit < 16 and value & (1 << bit)],
        "historical_flags": [name for bit, name in FLAG_NAMES.items() if bit >= 16 and value & (1 << bit)],
        "unknown_bits_hex": hex(value & ~KNOWN_FLAG_MASK),
    }


def _health_snapshot(path: Path | None) -> dict[str, Any]:
    result: dict[str, Any] = {"source_path": str(path.resolve()) if path else None, "supplied": path is not None,
                              "temperature_c": None, "temperature_readings_c": [], "throttled": None,
                              "throttled_readings": [], "issues": []}
    if path is None:
        return result
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        temp = re.search(r"\btemp\s*=\s*(-?\d+(?:\.\d+)?)", line)
        if temp:
            result["temperature_readings_c"].append(float(temp.group(1)))
        match = re.search(r"\bthrottled\s*=\s*(0[xX][0-9a-fA-F]+)\b", line)
        if match:
            result["throttled_readings"].append({"line": line_number, "raw": match.group(1), **decode_throttled(int(match.group(1), 16))})
        elif "throttled=" in line:
            result["issues"].append({"line": line_number, "kind": "unparseable_throttled_value"})
    for readings, single in (("temperature_readings_c", "temperature_c"), ("throttled_readings", "throttled")):
        if len(result[readings]) == 1:
            result[single] = result[readings][0]
        elif len(result[readings]) > 1:
            result["issues"].append({"kind": f"multiple_{readings}_not_automatically_matched"})
    return result


def _power_health(before: Path | None, after: Path | None, runs: list[dict[str, Any]]) -> dict[str, Any]:
    first, last = _health_snapshot(before), _health_snapshot(after)
    comparison: dict[str, Any] = {"available": False, "newly_observed_bits": None, "cleared_bits": None,
                                   "same_boot_verified": False, "interpretation": "Power history is unknown without one valid reading in each supplied snapshot."}
    if first["throttled"] is not None and last["throttled"] is not None:
        before_value, after_value = first["throttled"]["value"], last["throttled"]["value"]
        comparison.update({
            "available": True, "newly_observed_bits": decode_throttled(after_value & ~before_value),
            "cleared_bits": decode_throttled(before_value & ~after_value),
            "interpretation": "Newly observed bits are a snapshot comparison. Historical bits do not mean an active fault at the after probe. "
                              "An event-between-probes inference requires the same boot and no history reset; these are not verified here. "
                              "Zero flags in these samples are not sustained power certification.",
        })
    supplied = before is not None or after is not None
    association = "single_run_supplied_snapshots" if supplied and len(runs) == 1 else "global_unmatched" if supplied else "unknown"
    association_note = (
        "Snapshot files were supplied by the caller; their temporal correspondence is not independently verified. "
        "With multiple runs they remain global/unmatched and are never assigned to an individual run."
        if supplied else
        "No before/after snapshot files supplied; power history and temporal association are unknown."
    )
    return {"association": association, "run_id": runs[0]["run_id"] if supplied and len(runs) == 1 else None,
            "association_verified": False,
            "association_note": association_note,
            "before": first, "after": last, "comparison": comparison, "hardware_certified": False}


def build_report(runtime_path: str | Path, before_path: str | Path | None = None,
                 after_path: str | Path | None = None) -> dict[str, Any]:
    """Read input files and return a serializable report; perform no writes."""
    runtime = Path(runtime_path)
    before, after = (Path(value) if value is not None else None for value in (before_path, after_path))
    parsed, quality = _parse_runtime(runtime)
    runs = [_summarize_run(run) for run in parsed]
    return {
        "schema_version": 1, "report_kind": "offline_runtime_log_analysis", "runtime_source_path": str(runtime.resolve()),
        "input_quality": quality, "run_count": len(runs), "runs": runs,
        "power_health": _power_health(before, after, runs), "limitations": LIMITATIONS.copy(),
        "conclusion_flags": {
            "fresh_detection_records_present": any(run["detection_summary"]["total_records"] for run in runs),
            "missing_completion_evidence": not runs or any(run["completion_state"] == "completion_unknown" for run in runs),
            "input_has_unparsed_json": bool(quality["invalid_json_lines"]),
            "detection_accuracy_validated": False, "physical_camera_mapping_validated": False,
            "power_suitability_validated": False, "risk_readiness_validated": False,
            "hardware_certified": False, "safe_all_clear": False,
        },
        "configuration_modified": False, "camera_opened": False, "gpio_accessed": False,
    }


def _inline(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).replace("`", "'").replace("\n", " ").replace("\r", " ").replace("<", "&lt;").replace(">", "&gt;")


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Offline runtime log report", "", "This report is diagnostic evidence, not hardware or safety certification.", "",
             f"Runs found: {report['run_count']}. Unparsed JSON lines: {len(report['input_quality']['invalid_json_lines'])}.", ""]
    if not report["runs"]:
        lines.extend(["No recognized runtime records were found. Completion, detections and health remain unknown.", ""])
    for run in report["runs"]:
        lines.extend([f"## {_inline(run['run_id'])}", "", f"Directory: `{_inline(run['diagnostic_directory'])}`.", "",
                      f"Completion evidence: `{run['completion_state']}`; {run['status_record_count']} unique status records. "
                      f"Final record present: {run['final_present']}.", ""])
        performance = run["performance"]
        metrics = performance["metrics"] or {}
        lines.extend([f"Timing source: `{performance['source']}`; partial: {performance['partial']}. "
                      f"Average {_inline(metrics.get('average_end_to_end_ms'))} ms; p95 {_inline(metrics.get('p95_end_to_end_ms'))} ms; "
                      f"effective processing FPS {_inline(metrics.get('effective_fps'))}; warmup complete: {_inline(metrics.get('warmup_complete'))}.", ""])
        directional = performance["directional"]
        if isinstance(directional, dict):
            rates = []
            for direction in ("front", "rear"):
                details = directional.get(direction)
                rates.append(f"{direction}: {_inline(details.get('processed_fps')) if isinstance(details, dict) else 'unknown'} FPS")
            lines.extend(["Reported directional processing rates: " + "; ".join(rates) + ".", ""])
        configuration = run["reported_configuration"]
        lines.extend([f"Reported calibration: front {_inline(configuration['front_calibrated'])}, rear {_inline(configuration['rear_calibrated'])}; "
                      f"actuator enabled: {_inline(configuration['alert_actuator_enabled'])}. "
                      f"Reported latency gate: {_inline(metrics.get('meets_latency_budget'))}; "
                      f"minimum FPS gate: {_inline(metrics.get('meets_minimum_risk_fps'))}; "
                      f"risk alerts permitted: {_inline(metrics.get('risk_alerts_permitted'))}.", ""])
        summary = run["detection_summary"]
        lines.extend([f"Fresh detection records: {summary['total_records']} (not physical object counts).", ""])
        for label, group in summary["by_class"].items():
            lines.append(f"- `{_inline(label)}`: {group['count']} outputs; directions `{_inline(', '.join(group['directions']))}`; "
                         f"frames {group['frame_indices']}; confidence range {_inline(group['confidence']['minimum'])}–{_inline(group['confidence']['maximum'])}.")
        if summary["by_class"]:
            lines.append("")
        if run["overlapping_cross_class_boxes"]:
            lines.extend(["Different-class boxes overlap strongly in some frames. This does not verify which class is correct.", ""])
        if run["invalid_json_lines"]:
            lines.extend([f"Malformed or truncated JSON at input lines {run['invalid_json_lines']}; complete records above were preserved. "
                          "This is not evidence of a runtime crash.", ""])
        if run["duplicate_status_records_ignored"]:
            lines.extend([f"Ignored {run['duplicate_status_records_ignored']} repeated identical status records.", ""])
        if run["issues"]:
            issues = sorted({issue["kind"] for issue in run["issues"]})
            lines.extend(["Data-quality issues (see JSON line references): " + ", ".join(f"`{_inline(issue)}`" for issue in issues) + ".", ""])
    power = report["power_health"]
    lines.extend(["## Supplied power and temperature snapshots", "", f"Association: `{power['association']}`. {power['association_note']}", ""])
    for name in ("before", "after"):
        snapshot = power[name]
        flags = snapshot["throttled"]
        lines.append(f"- {name}: temperature {_inline(snapshot['temperature_c'])} °C; flags `{flags['hex'] if flags else 'unknown'}`; "
                     f"current: {_inline(', '.join(flags['current_flags']) or 'none') if flags else 'unknown'}; "
                     f"historical: {_inline(', '.join(flags['historical_flags']) or 'none') if flags else 'unknown'}.")
    lines.extend(["", power["comparison"]["interpretation"], "", "## Interpretation limits", ""])
    if power["comparison"]["available"]:
        # Insert the comparison before the interpretation-limits heading.
        lines[-2:-2] = [f"Newly observed flag bits: `{power['comparison']['newly_observed_bits']['hex']}`; "
                        f"cleared bits: `{power['comparison']['cleared_bits']['hex']}`.", ""]
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    lines.extend(["", "No configuration was modified; no camera or GPIO was accessed.", ""])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: str | Path,
                 input_paths: Sequence[str | Path] = ()) -> tuple[Path, Path]:
    """Create two new artifacts in a new directory; never overwrite inputs or outputs."""
    destination = Path(output_dir)
    resolved = destination.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output directory already exists; choose a NEW directory: {destination}")
    automatic_inputs = [report.get("runtime_source_path")]
    automatic_inputs.extend(report.get("power_health", {}).get(which, {}).get("source_path") for which in ("before", "after"))
    for value in [*input_paths, *automatic_inputs]:
        if value is not None:
            source = Path(value).resolve()
            if resolved == source or resolved in source.parents:
                raise ValueError(f"Output directory aliases or contains an input: {destination}")
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    markdown = render_markdown(report)
    destination.mkdir(parents=False, exist_ok=False)
    json_path, markdown_path = destination / "report.json", destination / "summary.md"
    with json_path.open("x", encoding="utf-8") as target:
        target.write(payload)
    with markdown_path.open("x", encoding="utf-8") as target:
        target.write(markdown)
    return json_path, markdown_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; its parent must already exist.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_report(args.runtime, args.before, args.after)
        paths = write_report(report, args.output_dir, [p for p in (args.runtime, args.before, args.after) if p is not None])
        for path in paths:
            print(path.resolve())
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print(f"Offline log report failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
