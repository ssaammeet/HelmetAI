"""Synthetic log fixtures only; no private runtime logs are packaged."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from helmetai_fcw import log_report


def status(frame, detections=None, **extra):
    return {"event": "runtime_status", "frame_count": frame, "detections": detections or [],
            "performance": {"frame_count": frame, "effective_fps": 3.0, "average_end_to_end_ms": 333.0,
                            "p95_end_to_end_ms": 600.0, "warmup_complete": False}, **extra}


def detection(label="truck", direction="rear", confidence=0.5315, target="rear-1", bbox=None):
    return {"class": label, "direction": direction, "confidence": confidence, "target": target,
            "bbox_xywh_px": bbox or [10, 20, 100, 80]}


def final(frames=20, **extra):
    return {"event": "runtime_final", "processed_frames": frames, "completed": True, "error": None,
            "performance_final": {"effective_fps": 3.04, "average_end_to_end_ms": 329.0,
                                  "p95_end_to_end_ms": 590.9, "warmup_complete": False}, **extra}


class LogReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runtime = self.root / "runtime.log"

    def report(self, records, before=None, after=None):
        self.runtime.write_text("\n".join(record if isinstance(record, str) else json.dumps(record)
                                          for record in records), encoding="utf-8")
        before_path, after_path = None, None
        if before is not None:
            before_path = self.root / "before.txt"
            before_path.write_text(before, encoding="utf-8")
        if after is not None:
            after_path = self.root / "after.txt"
            after_path.write_text(after, encoding="utf-8")
        return log_report.build_report(self.runtime, before_path, after_path)

    def test_rear_truck_records_preserve_classes_confidences_and_frames(self):
        report = self.report([
            "Diagnostic directory (keep these files): /synthetic/run.first",
            {"event": "runtime_start", "release": "synthetic", "front_calibrated": False, "rear_calibrated": False},
            status(1, [detection()]), status(7, [detection(confidence=0.4522)]), final(),
        ])
        run = report["runs"][0]
        self.assertEqual(run["diagnostic_directory"], "/synthetic/run.first")
        self.assertEqual(run["completion_state"], "completed")
        self.assertEqual(run["detection_summary"]["by_class"]["truck"]["frame_indices"], [1, 7])
        self.assertEqual(run["detection_summary"]["by_class"]["truck"]["confidence"]["minimum"], 0.4522)
        self.assertEqual(run["detection_summary"]["by_class"]["truck"]["tracker_ids"], ["rear-1"])
        self.assertEqual(run["detection_summary"]["by_direction"]["rear"]["count"], 2)
        self.assertNotIn("front", run["detection_summary"]["by_direction"])
        self.assertFalse(run["recognition_accuracy_validated"])
        self.assertEqual(run["reported_final_processed_frames"], 20)
        self.assertEqual(run["status_record_count"], 2)

    def test_assessments_cached_counters_and_repeated_status_do_not_double_count(self):
        fresh = status(1, [detection()], directional_performance={"rear": {"detections": 1}})
        run = self.report([
            fresh, fresh,
            {"timestamp_s": 1, "target": "rear-1", "detections": [detection()],
             "directional_performance": {"rear": {"detections": 1}}},
            status(2, [], directional_performance={"rear": {"detections": 1}}),
            final(directional_performance={"rear": {"detections": 1}}),
        ])["runs"][0]
        self.assertEqual(run["detection_summary"]["total_records"], 1)
        self.assertEqual(run["status_record_count"], 2)
        self.assertEqual(run["duplicate_status_records_ignored"], 1)

    def test_overlapping_different_classes_remain_distinct_not_proven_false_positive(self):
        run = self.report([status(7, [
            detection("parking_meter", confidence=0.6724, target="rear-2", bbox=[265, 70, 372, 405]),
            detection(confidence=0.4522, bbox=[272, 72, 367, 393]),
        ])])["runs"][0]
        self.assertEqual(run["detection_summary"]["total_records"], 2)
        overlap = run["overlapping_cross_class_boxes"][0]
        self.assertGreater(overlap["iou"], 0.94)
        self.assertIn("not verified", overlap["interpretation"])

    def test_mixed_complete_and_truncated_runs_preserve_complete_records(self):
        report = self.report([
            "Diagnostic directory (keep these files): /synthetic/run.first",
            {"event": "runtime_start"}, status(1, [detection()]), final(),
            "Diagnostic files: /synthetic/run.first",
            "Diagnostic directory (keep these files): /synthetic/run.second",
            {"event": "runtime_start"}, status(4, [detection("person")]),
            status(5, [detection("person", "front")]), status(7),
            '{"timestamp_s": 99, "performance": {"frame_count": 7,',
        ])
        self.assertEqual(report["run_count"], 2)
        self.assertEqual(report["runs"][0]["completion_state"], "completed")
        partial = report["runs"][1]
        self.assertEqual(partial["completion_state"], "completion_unknown")
        self.assertIsNone(partial["reported_error"])
        self.assertEqual(partial["status_record_count"], 3)
        self.assertTrue(partial["performance"]["partial"])
        self.assertEqual(partial["performance"]["source"], "last_runtime_status")
        self.assertEqual(partial["performance"]["metrics"]["frame_count"], 7)
        self.assertEqual(report["input_quality"]["invalid_json_lines"][0]["line"], 11)
        self.assertTrue(report["input_quality"]["invalid_json_lines"][0]["at_input_tail"])
        self.assertIn("not evidence of a runtime crash", log_report.render_markdown(report))

    def test_malformed_json_in_middle_does_not_discard_later_final(self):
        report = self.report([status(1), '{"event": broken}', status(2), final(2)])
        self.assertEqual(report["runs"][0]["completion_state"], "completed")
        self.assertEqual(report["runs"][0]["status_record_count"], 2)
        self.assertFalse(report["input_quality"]["invalid_json_lines"][0]["at_input_tail"])

    def test_repeated_start_without_directory_separates_runs(self):
        report = self.report([{"event": "runtime_start"}, status(1), final(1),
                              {"event": "runtime_start"}, status(1)])
        self.assertEqual(report["run_count"], 2)
        self.assertEqual(report["runs"][1]["boundary_reason"], "runtime_start")

    def test_frame_reset_without_start_is_explicitly_inferred(self):
        report = self.report([status(9), status(1, [detection()])])
        self.assertEqual(report["run_count"], 2)
        self.assertEqual(report["runs"][1]["boundary_reason"], "inferred_frame_reset_or_after_final")

    def test_conflicting_duplicate_frame_is_not_silently_counted_twice(self):
        run = self.report([status(1), status(1, [detection()])])["runs"][0]
        self.assertEqual(run["status_record_count"], 1)
        self.assertEqual(run["detection_summary"]["total_records"], 0)
        self.assertIn("conflicting_duplicate_frame_ignored", [issue["kind"] for issue in run["issues"]])

    def test_duplicate_final_is_not_an_extra_run(self):
        report = self.report([status(1), final(1), final(1)])
        self.assertEqual(report["run_count"], 1)
        self.assertEqual(report["runs"][0]["duplicate_final_records_ignored"], 1)

    def test_history_only_flags_do_not_mean_current_throttling(self):
        power = self.report([status(1), final(1)], "temp=71.9'C\nthrottled=0x0\n", "temp=79.0'C\nthrottled=0x50000\n")["power_health"]
        self.assertEqual(power["before"]["temperature_c"], 71.9)
        self.assertEqual(power["after"]["temperature_c"], 79.0)
        self.assertEqual(power["after"]["throttled"]["raw"], "0x50000")
        self.assertEqual(power["after"]["throttled"]["current_flags"], [])
        self.assertEqual(power["after"]["throttled"]["historical_flags"], ["under_voltage", "throttled"])
        self.assertEqual(power["comparison"]["newly_observed_bits"]["hex"], "0x50000")
        self.assertFalse(power["comparison"]["same_boot_verified"])
        self.assertEqual(power["association"], "single_run_supplied_snapshots")
        self.assertFalse(power["association_verified"])

    def test_current_and_unknown_bits_are_preserved(self):
        flags = log_report.decode_throttled(0x50005 | 0x100)
        self.assertEqual(flags["current_flags"], ["under_voltage", "throttled"])
        self.assertEqual(flags["historical_flags"], ["under_voltage", "throttled"])
        self.assertEqual(flags["unknown_bits_hex"], "0x100")

    def test_preexisting_history_is_not_new_and_cleared_history_is_visible(self):
        first = self.report([status(1)], "throttled=0x50000", "throttled=0x50000")["power_health"]["comparison"]
        self.assertEqual(first["newly_observed_bits"]["value"], 0)
        cleared = self.report([status(1)], "throttled=0x50000", "throttled=0x0")["power_health"]["comparison"]
        self.assertEqual(cleared["cleared_bits"]["value"], 0x50000)

    def test_absent_health_is_unknown_not_zero_or_healthy(self):
        report = self.report([status(1)])
        power = report["power_health"]
        self.assertEqual(power["association"], "unknown")
        self.assertEqual(power["association_note"], "No before/after snapshot files supplied; power history and temporal association are unknown.")
        self.assertIsNone(power["before"]["temperature_c"])
        self.assertIsNone(power["after"]["throttled"])
        self.assertFalse(power["comparison"]["available"])
        self.assertFalse(report["conclusion_flags"]["power_suitability_validated"])
        self.assertFalse(report["conclusion_flags"]["safe_all_clear"])

    def test_multiple_runs_keep_supplied_health_global_unmatched(self):
        power = self.report([{"event": "runtime_start"}, status(1), final(1),
                             {"event": "runtime_start"}, status(1)], "throttled=0x0", "throttled=0x50000")["power_health"]
        self.assertEqual(power["association"], "global_unmatched")
        self.assertIsNone(power["run_id"])
        self.assertTrue(power["comparison"]["available"])

    def test_multiple_readings_in_one_snapshot_are_not_guessed(self):
        power = self.report([status(1)], "temp=60.0'C\ntemp=70.0'C\nthrottled=0x0\nthrottled=0x1", "throttled=0x0")["power_health"]
        self.assertIsNone(power["before"]["temperature_c"])
        self.assertIsNone(power["before"]["throttled"])
        self.assertEqual(len(power["before"]["throttled_readings"]), 2)
        self.assertFalse(power["comparison"]["available"])

    def test_no_records_is_unknown_and_still_explained(self):
        report = self.report(["INFO camera configured", "no JSON here"])
        self.assertEqual(report["run_count"], 0)
        self.assertTrue(report["conclusion_flags"]["missing_completion_evidence"])
        self.assertIn("No recognized runtime records", log_report.render_markdown(report))

    def test_explicit_runtime_failure_remains_distinct_from_missing_paste(self):
        run = self.report([status(1), final(1, completed=False, error="camera failed")])["runs"][0]
        self.assertEqual(run["completion_state"], "reported_error")
        self.assertEqual(run["reported_error"], "camera failed")
        self.assertTrue(run["performance"]["partial"])

    def test_invalid_fields_are_reported_and_nonfinite_json_rejected(self):
        report = self.report([
            {"event": "runtime_status", "frame_count": True},
            status(1, [{"class": "truck", "confidence": 2.0}]),
            '{"event":"runtime_status","frame_count":2,"detections":[],"performance":{"effective_fps":NaN}}',
        ])
        run = report["runs"][0]
        self.assertEqual(run["status_record_count"], 1)
        self.assertIsNone(run["detection_records"][0]["confidence"])
        self.assertEqual(run["detection_records"][0]["direction"], "unknown")
        self.assertEqual(len(report["input_quality"]["invalid_json_lines"]), 1)
        json.dumps(report, allow_nan=False)

    def test_final_without_metrics_uses_explicit_partial_status_metrics(self):
        run = self.report([status(2), final(2, performance_final=None)])["runs"][0]
        self.assertEqual(run["completion_state"], "completed")
        self.assertEqual(run["performance"]["source"], "last_runtime_status")
        self.assertTrue(run["performance"]["partial"])

    def test_output_new_directory_only_and_inputs_unchanged(self):
        report = self.report([status(1)], "throttled=0x0", "throttled=0x0")
        before_bytes = self.runtime.read_bytes()
        output = self.root / "new-report"
        json_path, markdown_path = log_report.write_report(report, output)
        self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["schema_version"], 1)
        self.assertIn("Confidence is a model score", markdown_path.read_text(encoding="utf-8"))
        self.assertEqual(self.runtime.read_bytes(), before_bytes)
        existing_bytes = json_path.read_bytes()
        with self.assertRaises(FileExistsError):
            log_report.write_report(report, output)
        self.assertEqual(json_path.read_bytes(), existing_bytes)

    def test_output_input_alias_and_existing_empty_directory_rejected(self):
        report = self.report([status(1)])
        with self.assertRaises(FileExistsError):
            log_report.write_report(report, self.runtime)
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaises(FileExistsError):
            log_report.write_report(report, empty)
        with self.assertRaises(ValueError):
            log_report.write_report(report, self.root / "future", [self.root / "future" / "input.log"])

    def test_cli_outputs_two_artifacts_and_reports_errors_without_overwrite(self):
        self.report([status(1), final(1)])
        output = self.root / "cli-report"
        args = ["--runtime", str(self.runtime), "--output-dir", str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(log_report.main(args), 0)
        self.assertEqual(sorted(path.name for path in output.iterdir()), ["report.json", "summary.md"])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(log_report.main(args), 2)
            self.assertEqual(log_report.main(["--runtime", str(self.root / "missing.log"), "--output-dir", str(self.root / "unused")]), 2)
        self.assertFalse((self.root / "unused").exists())

    def test_module_import_does_not_import_camera_model_or_gpio(self):
        process = subprocess.run([sys.executable, "-c", "import sys; import helmetai_fcw.log_report; "
                                  "assert not any(name in sys.modules for name in "
                                  "('helmetai_fcw.pi_runtime', 'helmetai_fcw.rpi_camera', 'helmetai_fcw.alert_actuator', "
                                  "'helmetai_fcw.detection', 'cv2', 'numpy', 'gpiozero', 'picamera2'))"],
                                 capture_output=True, text=True, check=False)
        self.assertEqual(process.returncode, 0, process.stderr)


if __name__ == "__main__":
    unittest.main()
