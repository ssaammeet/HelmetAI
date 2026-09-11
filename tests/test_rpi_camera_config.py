import unittest

from helmetai_fcw.rpi_camera import PiCameraConfig


class PiCameraConfigTests(unittest.TestCase):
    def test_rejects_invalid_camera_parameters_before_hardware_open(self) -> None:
        with self.assertRaises(ValueError):
            PiCameraConfig(camera_index=-1)
        with self.assertRaises(ValueError):
            PiCameraConfig(camera_index=0, width_px=0)
        with self.assertRaises(ValueError):
            PiCameraConfig(camera_index=0, fps=0)


if __name__ == "__main__":
    unittest.main()
