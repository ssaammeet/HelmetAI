import unittest
from contextlib import redirect_stdout
from io import StringIO
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from helmetai_fcw.cli import build_parser
from helmetai_fcw.alert_actuator import AlertActuatorConfig
from helmetai_fcw.pi_runtime import PiRuntimeConfig


class CliCommandTests(unittest.TestCase):
    def test_visual_dashboard_command_is_available(self) -> None:
        arguments = build_parser().parse_args(["demo-dashboard"])
        self.assertEqual(arguments.command, "demo-dashboard")

    def test_real_video_command_accepts_front_or_rear_config(self) -> None:
        arguments = build_parser().parse_args(
            ["video-run", "--config", "config/webcam_rear.example.json", "--video", "rear.mp4", "--no-preview"]
        )
        self.assertEqual(arguments.command, "video-run")
        self.assertEqual(arguments.video, "rear.mp4")
        self.assertTrue(arguments.no_preview)

    def test_pi_run_accepts_software_blind_spot_turn_intent(self) -> None:
        arguments = build_parser().parse_args(["pi-run", "--turn-intent", "left", "--frames", "1"])
        self.assertEqual(arguments.turn_intent, "left")

    def test_pi_camera_check_command_is_available(self) -> None:
        arguments = build_parser().parse_args(["pi-camera-check", "--output-dir", "/tmp/check"])
        self.assertEqual(arguments.command, "pi-camera-check")
        self.assertEqual(arguments.output_dir, "/tmp/check")

    def test_camera_only_flag_disables_outputs_without_changing_saved_config(self):
        root = Path(__file__).resolve().parents[1]
        config = PiRuntimeConfig.from_json(root / "config/pi5_dual_camera.example.json")
        config = replace(config, alert_actuator=AlertActuatorConfig(enabled=True, led_bcm_pin=18))
        arguments = build_parser().parse_args(["pi-run", "--camera-only", "--frames", "2"])
        with patch("helmetai_fcw.cli.PiRuntimeConfig.from_json", return_value=config), patch("helmetai_fcw.cli.PiForwardCollisionRuntime") as runtime:
            self.assertEqual(arguments.handler(arguments), 0)
        used_config = runtime.call_args.args[0]
        self.assertFalse(used_config.alert_actuator.enabled)
        self.assertTrue(config.alert_actuator.enabled)
        self.assertEqual(used_config.front_calibration, config.front_calibration)
        self.assertEqual(used_config.rear_camera, config.rear_camera)

    def test_rear_calibration_instruction_names_the_rear_profile_field(self) -> None:
        arguments = build_parser().parse_args(
            [
                "calibrate-monocular",
                "--camera-role",
                "rear",
                "--reference-distance",
                "10",
                "--object-width",
                "1.8",
                "--pixel-width",
                "104",
            ]
        )
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(arguments.handler(arguments), 0)
        self.assertIn("rear_calibration.focal_length_px", output.getvalue())
        self.assertNotIn("front_calibration.focal_length_px alanına", output.getvalue())


if __name__ == "__main__":
    unittest.main()
