"""Detector-neutral bounding boxes and an optional OpenCV YOLO ONNX adapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Detection:
    object_id: str
    label: str
    confidence: float
    x_px: float
    y_px: float
    width_px: float
    height_px: float

    @property
    def area_px2(self) -> float:
        return self.width_px * self.height_px


class FrameDetector(Protocol):
    def detect(self, frame: object) -> list[Detection]:
        """Return relevant road-scene detections in image pixel coordinates."""


class OpenCvYoloOnnxDetector:
    """Small YOLOv8/YOLO11-style ONNX adapter for CPU-based Pi experiments.

    The release packages a COCO-compatible ONNX detector for controlled Pi
    evaluation. Any replacement model must be licensed, verified with this
    adapter and benchmarked on the deployed hardware. The adapter expects an
    ONNX export with output shape [1, 4 + class_count, candidate_count].
    """

    # The deployed ONNX model follows the COCO class order.  Keep this list
    # deliberately scoped to road-scene classes rather than returning all 80
    # COCO classes, which would create irrelevant tracks for a helmet camera.
    # A detected object is not automatically a range/risk target: the range
    # estimator separately accepts only classes with a calibrated width model.
    coco_labels = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        9: "traffic_light",
        10: "fire_hydrant",
        11: "stop_sign",
        12: "parking_meter",
        13: "bench",
        24: "backpack",
        26: "handbag",
        28: "suitcase",
        39: "bottle",
        56: "chair",
        57: "couch",
        58: "potted_plant",
        60: "dining_table",
    }

    def __init__(
        self,
        model_path: str | Path,
        input_size_px: int = 640,
        confidence_threshold: float = 0.45,
        cpu_threads: int | None = None,
    ) -> None:
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - Pi integration path
            raise RuntimeError("OpenCV and NumPy are required for ONNX detection.") from exc
        self.cv2 = cv2
        self.np = np
        if cpu_threads is not None:
            if cpu_threads < 1:
                raise ValueError("cpu_threads must be positive when supplied.")
            cv2.setNumThreads(cpu_threads)
        cv2.setUseOptimized(True)
        self.input_size_px = input_size_px
        self.confidence_threshold = confidence_threshold
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX detector model not found: {path}")
        self.net = cv2.dnn.readNetFromONNX(str(path))

    def detect(self, frame: object) -> list[Detection]:  # pragma: no cover - requires OpenCV/ONNX hardware path
        cv2 = self.cv2
        image = frame
        height, width = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.input_size_px, self.input_size_px), swapRB=True, crop=False)
        self.net.setInput(blob)
        raw = self.net.forward()
        return self._decode_output(raw, width, height)

    def _decode_output(self, raw: object, width: int, height: int) -> list[Detection]:
        """Decode model output without a Python loop over rejected candidates."""

        cv2, np = self.cv2, self.np
        candidates = raw.squeeze(0).T if raw.ndim == 3 else raw
        if candidates.ndim != 2:
            raise ValueError("Expected a 2D candidate matrix or [1, classes + 4, candidates] output.")
        if candidates.shape[1] < 8:
            return []
        # Choose the strongest class across ALL model classes before filtering
        # to supported labels. Restricting argmax to road classes would promote
        # an unsupported winner into a false road-scene detection.
        class_scores = candidates[:, 4:]
        class_indices = np.argmax(class_scores, axis=1)
        # The original loop compared a Python float to the threshold. Keep
        # float64 here so NumPy scalar casting cannot move a float32 boundary.
        candidate_scores = class_scores[np.arange(len(candidates)), class_indices].astype(float)
        relevant = np.isin(class_indices, tuple(self.coco_labels))
        kept_indices = np.flatnonzero(relevant & ~(candidate_scores < self.confidence_threshold))
        x_scale = width / self.input_size_px
        y_scale = height / self.input_size_px
        boxes: list[list[int]] = []
        scores: list[float] = []
        labels: list[str] = []
        for candidate_index in kept_indices:
            row = candidates[candidate_index]
            score = float(candidate_scores[candidate_index])
            label = self.coco_labels[int(class_indices[candidate_index])]
            centre_x, centre_y, box_width, box_height = row[:4]
            box_width *= x_scale
            box_height *= y_scale
            left = int((centre_x * x_scale) - box_width / 2)
            top = int((centre_y * y_scale) - box_height / 2)
            # A box rounded to zero pixels cannot be tracked or used for
            # monocular geometry.  Dropping it here prevents an invalid
            # detector result from aborting the live risk loop later.
            rounded_width = int(box_width)
            rounded_height = int(box_height)
            if rounded_width <= 0 or rounded_height <= 0:
                continue
            boxes.append([left, top, rounded_width, rounded_height])
            scores.append(score)
            labels.append(label)
        detections: list[Detection] = []
        # NMS must be class-aware.  A person inside a car box, for example,
        # must not suppress either road-scene detection.
        for label in sorted(set(labels)):
            label_indices = [index for index, candidate_label in enumerate(labels) if candidate_label == label]
            label_boxes = [boxes[index] for index in label_indices]
            label_scores = [scores[index] for index in label_indices]
            indices = cv2.dnn.NMSBoxes(label_boxes, label_scores, self.confidence_threshold, 0.45)
            # OpenCV versions return either a flat array or an N-by-1 array.
            for index in np.asarray(indices).reshape(-1):
                selected_local = int(index)
                selected = label_indices[selected_local]
                left, top, box_width, box_height = boxes[selected]
                detections.append(
                    Detection(
                        object_id="",
                        label=labels[selected],
                        confidence=scores[selected],
                        x_px=max(0, left),
                        y_px=max(0, top),
                        width_px=box_width,
                        height_px=box_height,
                    )
                )
        return detections


def with_object_id(detection: Detection, object_id: str) -> Detection:
    return replace(detection, object_id=object_id)
