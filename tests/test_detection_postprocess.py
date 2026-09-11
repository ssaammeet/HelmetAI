"""Regression parity for vectorized filtering, with the original scalar path."""

import unittest
from types import SimpleNamespace

from helmetai_fcw.detection import Detection, OpenCvYoloOnnxDetector

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = np = None


def _scalar_decode(detector, raw, width, height):
    """Former decoder retained as an independent test-only reference."""

    candidates = raw.squeeze(0).T if raw.ndim == 3 else raw
    x_scale, y_scale = width / detector.input_size_px, height / detector.input_size_px
    boxes, scores, labels = [], [], []
    for row in candidates:
        if len(row) < 8:
            continue
        class_index = int(np.argmax(row[4:]))
        score = float(row[4 + class_index])
        label = detector.coco_labels.get(class_index)
        if label is None or score < detector.confidence_threshold:
            continue
        centre_x, centre_y, box_width, box_height = row[:4]
        box_width *= x_scale
        box_height *= y_scale
        left = int((centre_x * x_scale) - box_width / 2)
        top = int((centre_y * y_scale) - box_height / 2)
        rounded_width, rounded_height = int(box_width), int(box_height)
        if rounded_width <= 0 or rounded_height <= 0:
            continue
        boxes.append([left, top, rounded_width, rounded_height])
        scores.append(score)
        labels.append(label)
    detections = []
    for label in sorted(set(labels)):
        label_indices = [i for i, candidate_label in enumerate(labels) if candidate_label == label]
        indices = detector.cv2.dnn.NMSBoxes(
            [boxes[i] for i in label_indices],
            [scores[i] for i in label_indices],
            detector.confidence_threshold,
            0.45,
        )
        for index in np.asarray(indices).reshape(-1):
            selected = label_indices[int(index)]
            left, top, box_width, box_height = boxes[selected]
            detections.append(Detection("", labels[selected], scores[selected], max(0, left), max(0, top), box_width, box_height))
    return detections


@unittest.skipIf(np is None or cv2 is None, "OpenCV and NumPy are optional detector dependencies.")
class DetectionPostprocessTests(unittest.TestCase):
    def setUp(self):
        self.detector = OpenCvYoloOnnxDetector.__new__(OpenCvYoloOnnxDetector)
        self.detector.np = np
        self.detector.cv2 = cv2
        self.detector.input_size_px = 640
        self.detector.confidence_threshold = 0.45

    def _rows(self, specs):
        rows = np.zeros((len(specs), 84), dtype=np.float32)
        for i, (geometry, class_index, score) in enumerate(specs):
            rows[i, :4] = geometry
            rows[i, 4 + class_index] = score
        return rows

    def test_randomized_exact_parity_for_both_supported_output_shapes(self):
        rng = np.random.default_rng(7020)
        for dtype in (np.float32, np.float64):
            for width, height in ((640, 480), (1280, 720), (333, 777)):
                rows = rng.uniform(0, 0.4, (8400, 84)).astype(dtype)
                rows[:, :2] = rng.uniform(-20, 660, (8400, 2))
                rows[:, 2:4] = rng.uniform(-1, 100, (8400, 2))
                winners = rng.integers(0, 80, 300)
                rows[np.arange(300), 4 + winners] = rng.uniform(0.4, 1.0, 300)
                for raw in (rows, rows.T[None, :, :]):
                    with self.subTest(dtype=dtype, dimensions=(width, height), raw_shape=raw.shape):
                        self.assertEqual(
                            self.detector._decode_output(raw, width, height),
                            _scalar_decode(self.detector, raw, width, height),
                        )

    def test_unsupported_strongest_class_does_not_promote_car(self):
        rows = self._rows([((100, 100, 40, 40), 4, 0.99)])
        rows[0, 6] = 0.95  # car is strong, but airplane wins argmax
        self.assertEqual(self.detector._decode_output(rows, 640, 480), [])

    def test_overlapping_classes_survive_and_same_class_is_suppressed(self):
        rows = self._rows([
            ((100, 100, 40, 40), 0, 0.9),
            ((100, 100, 40, 40), 2, 0.8),
            ((101, 101, 40, 40), 2, 0.7),
        ])
        detections = self.detector._decode_output(rows, 640, 480)
        self.assertEqual([d.label for d in detections], ["car", "person"])
        self.assertEqual(detections, _scalar_decode(self.detector, rows, 640, 480))

    def test_tied_classes_keep_first_argmax_and_candidate_order(self):
        rows = self._rows([
            ((100, 100, 40, 40), 0, 0.9),
            ((100, 100, 40, 40), 0, 0.9),
        ])
        rows[:, 6] = rows[:, 4]
        detections = self.detector._decode_output(rows, 640, 480)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "person")
        self.assertEqual(detections, _scalar_decode(self.detector, rows, 640, 480))

    def test_float32_threshold_boundary_matches_python_float_comparison(self):
        nms_calls = []

        def keep_all(boxes, scores, confidence, overlap):
            nms_calls.append((scores, confidence, overlap))
            return list(range(len(boxes)))

        self.detector.cv2 = SimpleNamespace(dnn=SimpleNamespace(NMSBoxes=keep_all))
        threshold32 = np.float32(0.45)
        rows = self._rows([
            ((50, 50, 10, 10), 2, threshold32),
            ((150, 150, 10, 10), 2, np.nextafter(threshold32, np.float32(1.0))),
        ])
        detections = self.detector._decode_output(rows, 640, 480)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections, _scalar_decode(self.detector, rows, 640, 480))
        self.assertTrue(all(call[1:] == (0.45, 0.45) for call in nms_calls))

    def test_zero_pixel_boxes_are_dropped_and_geometry_is_unchanged(self):
        rows = self._rows([
            ((-1.5, -1.5, 10.9, 20.9), 2, 0.9),
            ((1.5, 1.5, 0.99, 20.9), 2, 0.9),
            ((1.5, 1.5, 20.9, 0.99), 2, 0.9),
            ((1.5, 1.5, -2, 20.9), 2, 0.9),
        ])
        detections = self.detector._decode_output(rows, 640, 480)
        self.assertEqual(len(detections), 1)
        self.assertEqual((detections[0].x_px, detections[0].y_px), (0, 0))
        self.assertEqual((detections[0].width_px, detections[0].height_px), (10, 15))
        self.assertEqual(detections, _scalar_decode(self.detector, rows, 640, 480))

    def test_all_supported_classes_remain_supported(self):
        rows = self._rows([((i * 10, 100, 5, 5), class_index, 0.9) for i, class_index in enumerate(self.detector.coco_labels)])
        self.assertEqual(
            [d.label for d in self.detector._decode_output(rows, 640, 480)],
            sorted(self.detector.coco_labels.values()),
        )

    def test_empty_and_too_short_outputs_remain_empty(self):
        for rows in (np.empty((0, 84)), np.empty((3, 7)), np.empty((1, 84, 0))):
            self.assertEqual(self.detector._decode_output(rows, 640, 480), [])

    def test_nms_flat_nested_and_column_array_results(self):
        rows = self._rows([((100, 100, 40, 40), 2, 0.9)])
        for indices in ([0], [[0]], np.array([0]), np.array([[0]])):
            self.detector.cv2 = SimpleNamespace(dnn=SimpleNamespace(NMSBoxes=lambda *args: indices))
            self.assertEqual(len(self.detector._decode_output(rows, 640, 480)), 1)


if __name__ == "__main__":
    unittest.main()
