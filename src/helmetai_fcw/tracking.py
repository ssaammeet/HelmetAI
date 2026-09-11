"""Lightweight IoU tracker that gives detector boxes stable IDs across frames."""

from __future__ import annotations

from dataclasses import dataclass

from .detection import Detection, with_object_id


@dataclass
class _BoxTrack:
    label: str
    detection: Detection
    last_seen_s: float


def _iou(first: Detection, second: Detection) -> float:
    left = max(first.x_px, second.x_px)
    top = max(first.y_px, second.y_px)
    right = min(first.x_px + first.width_px, second.x_px + second.width_px)
    bottom = min(first.y_px + first.height_px, second.y_px + second.height_px)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.area_px2 + second.area_px2 - intersection
    return intersection / union if union > 0 else 0.0


class IoUTracker:
    """Dependency-free tracker appropriate for an initial single-camera MVP."""

    def __init__(self, match_iou: float = 0.25, max_age_s: float = 1.0, id_prefix: str = "front") -> None:
        if not id_prefix or not id_prefix.replace("_", "").isalnum():
            raise ValueError("id_prefix must contain only letters, digits and underscores.")
        self.match_iou = match_iou
        self.max_age_s = max_age_s
        self.id_prefix = id_prefix
        self._tracks: dict[str, _BoxTrack] = {}
        self._next_id = 1

    def update(self, timestamp_s: float, detections: list[Detection]) -> list[Detection]:
        self._tracks = {
            object_id: track
            for object_id, track in self._tracks.items()
            if timestamp_s - track.last_seen_s <= self.max_age_s
        }
        candidates: list[tuple[float, int, str]] = []
        for index, detection in enumerate(detections):
            for object_id, track in self._tracks.items():
                if track.label == detection.label:
                    candidates.append((_iou(track.detection, detection), index, object_id))
        assignments: dict[int, str] = {}
        used_tracks: set[str] = set()
        for overlap, index, object_id in sorted(candidates, reverse=True):
            if overlap < self.match_iou or index in assignments or object_id in used_tracks:
                continue
            assignments[index] = object_id
            used_tracks.add(object_id)

        tracked: list[Detection] = []
        for index, detection in enumerate(detections):
            object_id = assignments.get(index)
            if object_id is None:
                object_id = f"{self.id_prefix}-{self._next_id}"
                self._next_id += 1
            identified = with_object_id(detection, object_id)
            self._tracks[object_id] = _BoxTrack(
                label=identified.label,
                detection=identified,
                last_seen_s=timestamp_s,
            )
            tracked.append(identified)
        return tracked
