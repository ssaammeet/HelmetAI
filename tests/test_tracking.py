import unittest

from helmetai_fcw.detection import Detection
from helmetai_fcw.tracking import IoUTracker


class IoUTrackerTests(unittest.TestCase):
    def test_keeps_id_for_same_car_in_next_frame(self) -> None:
        tracker = IoUTracker()
        first = tracker.update(0.0, [Detection("", "car", 0.9, 100, 100, 100, 70)])[0]
        second = tracker.update(0.1, [Detection("", "car", 0.9, 106, 101, 100, 70)])[0]
        self.assertEqual(first.object_id, second.object_id)

    def test_assigns_distinct_ids(self) -> None:
        tracker = IoUTracker()
        detections = tracker.update(
            0.0,
            [
                Detection("", "car", 0.9, 10, 10, 100, 70),
                Detection("", "truck", 0.9, 250, 10, 120, 80),
            ],
        )
        self.assertNotEqual(detections[0].object_id, detections[1].object_id)

    def test_uses_the_configured_camera_prefix(self) -> None:
        tracker = IoUTracker(id_prefix="rear")
        detection = tracker.update(0.0, [Detection("", "car", 0.9, 10, 10, 100, 70)])[0]
        self.assertTrue(detection.object_id.startswith("rear-"))


if __name__ == "__main__":
    unittest.main()
