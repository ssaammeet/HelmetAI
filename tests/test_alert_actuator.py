import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from helmetai_fcw.alert_actuator import (
    AlertActuatorConfig,
    LgpioAlertActuator,
    NullAlertActuator,
    create_alert_actuator,
    resolve_gpio_chip,
)
from helmetai_fcw.models import AlertLevel


class _FakeLgpio:
    def __init__(self) -> None:
        self.claims: list[tuple[int, int, int]] = []
        self.writes: list[tuple[int, int, int]] = []
        self.closed: list[int] = []

    def gpiochip_open(self, chip: int) -> int:
        return chip + 100

    def gpio_claim_output(self, chip: int, pin: int, value: int) -> None:
        self.claims.append((chip, pin, value))

    def gpio_write(self, chip: int, pin: int, value: int) -> None:
        self.writes.append((chip, pin, value))

    def gpiochip_close(self, chip: int) -> None:
        self.closed.append(chip)


class AlertActuatorTests(unittest.TestCase):
    def test_disabled_config_is_a_no_op_without_gpio_import(self) -> None:
        actuator = create_alert_actuator(AlertActuatorConfig())
        self.assertIsInstance(actuator, NullAlertActuator)
        state = actuator.apply(AlertLevel.CRITICAL, alerts_permitted=True, now_s=1.0)
        self.assertEqual(state.applied_level, AlertLevel.NONE)
        self.assertFalse(state.buzzer_on)

    def test_enabled_config_requires_unique_safe_pin_mapping(self) -> None:
        with self.assertRaises(ValueError):
            AlertActuatorConfig(enabled=True)
        with self.assertRaises(ValueError):
            AlertActuatorConfig(enabled=True, buzzer_bcm_pin=18, led_bcm_pin=18)
        config = AlertActuatorConfig.from_mapping({"enabled": True, "buzzer_bcm_pin": 18})
        self.assertTrue(config.enabled)
        self.assertEqual(config.buzzer_bcm_pin, 18)

    def test_profile_rejects_coercible_but_unsafe_wiring_values(self) -> None:
        # JSON strings must never turn an output on through Python truthiness
        # or implicit integer conversion.
        with self.assertRaisesRegex(ValueError, "enabled"):
            AlertActuatorConfig.from_mapping({"enabled": "false"})
        with self.assertRaisesRegex(ValueError, "buzzer_bcm_pin"):
            AlertActuatorConfig.from_mapping({"buzzer_bcm_pin": "18"})
        with self.assertRaisesRegex(ValueError, "warning_pulse_period_s"):
            AlertActuatorConfig.from_mapping({"warning_pulse_period_s": True})

    def test_patterns_and_close_drive_outputs_to_safe_off_state(self) -> None:
        fake = _FakeLgpio()
        config = AlertActuatorConfig(
            enabled=True,
            gpio_chip=0,
            buzzer_bcm_pin=18,
            vibration_bcm_pin=23,
            led_bcm_pin=24,
            warning_pulse_period_s=0.5,
        )
        with patch.dict(sys.modules, {"lgpio": fake}):
            actuator = LgpioAlertActuator(config)
            advisory = actuator.apply(AlertLevel.ADVISORY, alerts_permitted=True, now_s=0.0)
            warning = actuator.apply(AlertLevel.WARNING, alerts_permitted=True, now_s=0.0)
            critical = actuator.apply(AlertLevel.CRITICAL, alerts_permitted=True, now_s=1.0)
            blocked = actuator.apply(AlertLevel.CRITICAL, alerts_permitted=False, now_s=1.0)
            actuator.close()
        self.assertEqual(len(fake.claims), 3)
        self.assertTrue(advisory.led_on)
        self.assertTrue(warning.buzzer_on)
        self.assertTrue(critical.vibration_on)
        self.assertEqual(blocked.applied_level, AlertLevel.NONE)
        self.assertEqual(fake.closed, [100])
        # The final writes for every claimed output must be electrical-low.
        self.assertEqual({pin: value for _, pin, value in fake.writes if pin in {18, 23, 24}}, {18: 0, 23: 0, 24: 0})

    def test_active_low_output_and_missing_devices_are_reported_truthfully(self) -> None:
        fake = _FakeLgpio()
        config = AlertActuatorConfig(enabled=True, gpio_chip=0, led_bcm_pin=24, active_high=False)
        with patch.dict(sys.modules, {"lgpio": fake}):
            actuator = LgpioAlertActuator(config)
            critical = actuator.apply(AlertLevel.CRITICAL, alerts_permitted=True, now_s=1.0)
            actuator.close()

        # A critical pattern asks for every channel, but only the configured
        # LED exists.  Telemetry must not claim a missing buzzer/vibrator ran.
        self.assertFalse(critical.buzzer_on)
        self.assertFalse(critical.vibration_on)
        self.assertTrue(critical.led_on)
        # Active-low hardware is de-energised by electrical high at close.
        self.assertEqual(fake.claims, [(100, 24, 1)])
        self.assertEqual(fake.writes[-1], (100, 24, 1))

    def test_auto_detects_the_physical_header_chip_by_label(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gpiochip0").mkdir()
            (root / "gpiochip0" / "label").write_text("gpio-brcmstb@107d508500\n", encoding="utf-8")
            (root / "gpiochip4").mkdir()
            (root / "gpiochip4" / "label").write_text("pinctrl-rp1\n", encoding="utf-8")
            self.assertEqual(resolve_gpio_chip(None, gpio_class_root=root), 4)
            self.assertEqual(resolve_gpio_chip(0, gpio_class_root=root), 0)

    def test_ambiguous_or_missing_auto_detection_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "Could not auto-detect"):
                resolve_gpio_chip(None, gpio_class_root=root)
            for chip, label in (("gpiochip0", "pinctrl-rp1"), ("gpiochip4", "pinctrl-bcm")):
                (root / chip).mkdir()
                (root / chip / "label").write_text(label, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "More than one"):
                resolve_gpio_chip(None, gpio_class_root=root)


if __name__ == "__main__":
    unittest.main()
