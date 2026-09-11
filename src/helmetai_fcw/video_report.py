"""Streaming, explicitly unverified evidence for recorded-video evaluation."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any


LIMITATIONS = [
    "No ground-truth comparison: detection accuracy, distance, TTC and safety are not verified.",
    "NONE means no alert emitted; it is not an all-clear or a safe-scene classification.",
    "Detection counts count per-frame observations, not unique physical objects.",
    "Track IDs are local IoU associations and can split, merge or switch identities.",
    "Source timestamps use frame index / source FPS (constant-frame-rate assumption).",
    "Processing FPS describes this offline run, not source FPS or a hardware performance guarantee.",
    "Uncalibrated and demonstration-only outputs omit numerical physical estimates.",
    "If source frame count is unavailable, decoder read-stop is unconfirmed, not certified EOF.",
    "Per-track summary storage is capped; any overflow is explicit and frame JSONL retains all observations.",
    "Annotated frames are submitted to OpenCV; successful encoding is not verified by re-decoding in this runtime.",
]


class VideoEvidenceReport:
    """Keep per-frame evidence on disk, with compact aggregate counts."""

    TRACK_SUMMARY_LIMIT = 10_000

    def __init__(self, directory: Path | None) -> None:
        self.directory = directory
        self.stream = None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=False)
            self.stream = (directory / "frames.jsonl").open("x", encoding="utf-8")
        self.frames = 0
        self.detections = 0
        self.frames_with_detections = 0
        self.labels: Counter[str] = Counter()
        self.statuses: Counter[str] = Counter()
        self.levels: Counter[str] = Counter()
        self.reasons: Counter[str] = Counter()
        self.tracks: dict[str, dict[str, Any]] = {}
        self.omitted_track_observations = 0

    def add_frame(self, record: dict[str, Any]) -> None:
        if self.stream is not None:
            self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self.stream.flush()
        self.frames += 1
        objects = record["objects"]
        self.frames_with_detections += bool(objects)
        self.detections += len(objects)
        for obj in objects:
            assessment = obj["pipeline_assessment"]
            self.labels[obj["label"]] += 1
            self.statuses[assessment["system_status"]] += 1
            if assessment["level"] is not None:
                self.levels[assessment["level"]] += 1
            self.reasons.update(assessment["reasons"])
            key = obj["object_id"]
            if key not in self.tracks:
                if len(self.tracks) >= self.TRACK_SUMMARY_LIMIT:
                    self.omitted_track_observations += 1
                    continue
                self.tracks[key] = {
                    "object_id": key,
                    "label": obj["label"],
                    "first_source_frame_index": record["source_frame_index"],
                    "first_timestamp_s": record["timestamp_s"],
                    "observation_count": 0,
                    "status_counts": Counter(),
                    "level_counts": Counter(),
                    "reason_counts": Counter(),
                }
            track = self.tracks[key]
            track["last_source_frame_index"] = record["source_frame_index"]
            track["last_timestamp_s"] = record["timestamp_s"]
            track["observation_count"] += 1
            track["status_counts"][assessment["system_status"]] += 1
            if assessment["level"] is not None:
                track["level_counts"][assessment["level"]] += 1
            track["reason_counts"].update(assessment["reasons"])

    def finalize(self, run: dict[str, Any]) -> dict[str, Any]:
        if self.stream is not None:
            self.stream.close()
        summary = {
            "schema_version": 1,
            **run,
            "metric_verified": False,
            "physical_risk_validated": False,
            "ground_truth_evaluated": False,
            "frame_count": self.frames,
            "frames_with_detections": self.frames_with_detections,
            "detection_count": self.detections,
            "unique_track_count": len(self.tracks) if not self.omitted_track_observations else None,
            "reported_track_count": len(self.tracks),
            "track_summary_limit": self.TRACK_SUMMARY_LIMIT,
            "track_summary_truncated": bool(self.omitted_track_observations),
            "omitted_track_observation_count": self.omitted_track_observations,
            "label_counts": dict(sorted(self.labels.items())),
            "system_status_counts": dict(sorted(self.statuses.items())),
            "level_counts": dict(sorted(self.levels.items())),
            "reason_counts": dict(sorted(self.reasons.items())),
            "tracks": list(self.tracks.values()),
            "limitations": LIMITATIONS,
        }
        if self.directory is not None:
            try:
                self._write_summaries(summary)
            except BaseException as exc:
                # A later Markdown/disk failure must not leave a JSON success
                # claim. Invalidate every writable summary before propagating.
                summary.update(status="failed", completed=False,
                               error=f"Report finalization failed: {type(exc).__name__}: {exc}")
                for name, content in (("summary.json", json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
                                      ("summary.md", self._markdown(summary))):
                    try:
                        (self.directory / name).write_text(content, encoding="utf-8")
                    except BaseException:
                        pass  # The original filesystem error is raised below.
                raise
        return summary

    def _write_summaries(self, summary: dict[str, Any]) -> None:
        assert self.directory is not None
        (self.directory / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )
        (self.directory / "summary.md").write_text(self._markdown(summary), encoding="utf-8")

    @staticmethod
    def _markdown(summary: dict[str, Any]) -> str:
        fps = summary.get("processing_fps")
        track_count = summary["unique_track_count"]
        if track_count is None:
            track_count = f"at least {summary['reported_track_count']} (summary truncated)"
        lines = [
            "# HelmetAI offline video evidence", "",
            f"Run status: **{summary['status']}**. Completed to EOF: {summary['completed']}.", "",
            "**Unverified prototype evidence, not physical-risk or road-safety validation.**", "",
            f"Mode: {summary['mode']}. Camera role: {summary['camera_role']}.",
            f"Processed frames: {summary['frame_count']}; per-frame detections: {summary['detection_count']}; "
            f"local tracks: {track_count}.", "",
            f"Source FPS: {summary.get('source_fps')}; effective timeline FPS: {summary.get('effective_source_fps')}.",
            f"Processing FPS: {fps:.2f}." if fps is not None else "Processing FPS: unavailable.",
            "Processing FPS includes decoding, inference, drawing, frame evidence and output writes; excludes initialization and release.", "",
            "## Observation counts", "",
            "| Label | Per-frame detections |", "| --- | ---: |",
        ]
        lines.extend(f"| {label} | {count} |" for label, count in summary["label_counts"].items())
        if not summary["label_counts"]:
            lines.append("| No detections emitted | 0 |")
        lines.extend(["", "## Pipeline evidence", "",
                      f"System statuses: `{json.dumps(summary['system_status_counts'])}`.",
                      f"Emitted levels (not safety classifications): `{json.dumps(summary['level_counts'])}`.",
                      "All objects, statuses and reasons are recorded in `frames.jsonl`; per-track counts are in `summary.json`."])
        if summary.get("error"):
            lines.extend(["", "## Failure", "", str(summary["error"])])
        lines.extend(["", "## Limitations", ""])
        lines.extend(f"- {item}" for item in LIMITATIONS)
        return "\n".join(lines) + "\n"
