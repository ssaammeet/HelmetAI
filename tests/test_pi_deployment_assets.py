"""Static, cross-platform checks for the Raspberry Pi deployment assets.

These assertions deliberately do not execute the installer: it uses apt, sudo,
systemd and CSI-camera utilities that are unavailable on a Windows development
machine.  They protect the install-time contracts that can be checked before
copying a release to a Pi.
"""

from __future__ import annotations

import unittest
from pathlib import Path


class PiDeploymentAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]
        cls.install_script = (cls.root / "deploy" / "install_pi.sh").read_text(encoding="utf-8")
        cls.service_template = (cls.root / "deploy" / "helmetai-fcw.service").read_text(encoding="utf-8")
        cls.benchmark_script = (cls.root / "scripts" / "benchmark_pi.sh").read_text(encoding="utf-8")
        cls.preflight_script = (cls.root / "scripts" / "preflight_pi.sh").read_text(encoding="utf-8")

    def test_deploy_assets_are_unix_text_files(self) -> None:
        """A CRLF shell script can fail when it is executed directly on Pi OS."""
        for relative_path in (
            "deploy/install_pi.sh",
            "deploy/helmetai-fcw.service",
            "scripts/benchmark_pi.sh",
            "scripts/preflight_pi.sh",
        ):
            payload = (self.root / relative_path).read_bytes()
            self.assertNotIn(b"\r\n", payload, relative_path)

    def test_installer_copies_all_runtime_assets_used_after_install(self) -> None:
        """README commands must still exist after installing into /opt."""
        self.assertIn("/opt/helmetai-fcw/scripts", self.install_script)
        self.assertIn("scripts/benchmark_pi.sh", self.install_script)
        self.assertIn("/opt/helmetai-fcw/scripts/benchmark_pi.sh", self.install_script)
        self.assertIn("scripts/preflight_pi.sh", self.install_script)
        self.assertIn("/opt/helmetai-fcw/scripts/preflight_pi.sh", self.install_script)
        self.assertIn('cp -r "${PROJECT_DIR}/src" /opt/helmetai-fcw/', self.install_script)
        self.assertIn('cp -r "${PROJECT_DIR}/models/." /opt/helmetai-fcw/models/', self.install_script)
        self.assertIn("rpicam-apps gpiod python3-picamera2 python3-opencv python3-numpy", self.install_script)
        self.assertIn('getent group gpio', self.install_script)

    def test_service_template_is_rendered_for_the_installing_user(self) -> None:
        """Do not silently assume Raspberry Pi OS has a user called ``pi``."""
        self.assertIn("User=__HELMETAI_USER__", self.service_template)
        self.assertNotIn("User=pi\n", self.service_template)
        self.assertIn('TARGET_USER="${HELMETAI_USER:-${SUDO_USER:-$(id -un)}}"', self.install_script)
        self.assertIn('if [[ "$TARGET_USER" == "root" ]]', self.install_script)
        self.assertIn('id "$TARGET_USER" >/dev/null 2>&1', self.install_script)
        self.assertIn('s|__HELMETAI_USER__|${TARGET_USER}|g', self.install_script)
        self.assertIn("/etc/systemd/system/helmetai-fcw.service", self.install_script)

        rendered = self.service_template.replace("__HELMETAI_USER__", "helmetai")
        self.assertNotIn("__HELMETAI_USER__", rendered)
        self.assertIn("User=helmetai", rendered)
        self.assertIn("WorkingDirectory=/opt/helmetai-fcw", rendered)
        self.assertIn("Environment=PYTHONPATH=/opt/helmetai-fcw/src", rendered)
        self.assertIn(
            "ExecStart=/usr/bin/python3 -m helmetai_fcw pi-run --config "
            "/etc/helmetai-fcw/pi5_dual_camera.json",
            rendered,
        )

    def test_benchmark_is_location_independent_and_fails_closed(self) -> None:
        """The installed benchmark cannot depend on the caller's current directory."""
        self.assertIn('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"', self.benchmark_script)
        self.assertIn('PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"', self.benchmark_script)
        self.assertIn('PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"', self.benchmark_script)
        self.assertIn('python3 -m helmetai_fcw pi-run --config "$CONFIG_PATH" --frames "$FRAME_COUNT"', self.benchmark_script)
        self.assertIn("PIPESTATUS[0]", self.benchmark_script)
        # Success/failure semantics and measured per-direction rates are
        # exercised by test_pi_software_update, not implementation strings.

    def test_preflight_checks_camera_visibility_and_profile_before_service(self) -> None:
        self.assertIn("rpicam-hello --list-cameras", self.preflight_script)
        self.assertIn("PiRuntimeConfig.from_json", self.preflight_script)
        self.assertIn("detector.input_size_px=640", self.preflight_script)
        self.assertIn("resolve_gpio_chip", self.preflight_script)
        self.assertIn("gpio_chip_access=ok", self.preflight_script)
        self.assertIn("Detector OpenCV smoke check failed", self.preflight_script)
        self.assertIn("WARNING: detection smoke testing is allowed", self.preflight_script)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
