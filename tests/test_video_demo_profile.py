import unittest

from helmetai_fcw.video_runtime import VideoCollisionRiskRuntime
from helmetai_fcw.webcam_runtime import WebcamRuntimeConfig


class VideoDemoProfileTests(unittest.TestCase):
    def test_demo_profile_scales_to_1080p_source(self) -> None:
        config = WebcamRuntimeConfig.from_json("config/webcam_video_demo_unverified.json")
        runtime = VideoCollisionRiskRuntime(config, "unused.mp4")

        calibration = runtime._calibration_for_video(1920, 1080)

        self.assertEqual(calibration.image_width_px, 1920)
        self.assertEqual(calibration.image_height_px, 1080)
        self.assertEqual(calibration.principal_point_x_px, 960.0)
        self.assertEqual(calibration.focal_length_px, 1740.0)


if __name__ == "__main__":
    unittest.main()
