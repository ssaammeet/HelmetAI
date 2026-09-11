"""Optional, fail-safe rider-alert outputs for Raspberry Pi deployments.

The risk engine never assumes that a GPIO device exists.  GPIO control is
enabled only by an explicit configuration and uses BCM pin numbers.  A buzzer,
LED or vibration motor must be connected through electronics appropriate for
its current/voltage requirements; in particular, a vibration motor must not
be connected directly to a Raspberry Pi GPIO pin.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import monotonic
from typing import Any, Protocol

from .models import AlertLevel


@dataclass(frozen=True)
class AlertActuatorConfig:
    """Explicit GPIO wiring and pattern settings for optional rider alerts."""

    enabled: bool = False
    # ``None`` means discover the user-facing GPIO controller from Linux's
    # gpiochip labels.  Pi 5 used gpiochip4 on older Bookworm images and
    # gpiochip0 after the kernel's 2024 reordering, so hard-coding either is
    # unsafe for a released helmet profile.
    gpio_chip: int | None = None
    buzzer_bcm_pin: int | None = None
    vibration_bcm_pin: int | None = None
    led_bcm_pin: int | None = None
    active_high: bool = True
    warning_pulse_period_s: float = 0.50

    def __post_init__(self) -> None:
        # The profile is user-edited JSON.  Do not silently coerce strings such
        # as "false" to True or "18" to a GPIO pin: either can unexpectedly
        # energise a real output after deployment.
        if type(self.enabled) is not bool:
            raise ValueError("alert_actuator.enabled must be a JSON boolean.")
        if type(self.active_high) is not bool:
            raise ValueError("alert_actuator.active_high must be a JSON boolean.")
        if self.gpio_chip is not None and (type(self.gpio_chip) is not int or self.gpio_chip < 0):
            raise ValueError("gpio_chip must be a non-negative integer or null for auto-detection.")
        if (
            isinstance(self.warning_pulse_period_s, bool)
            or not isinstance(self.warning_pulse_period_s, (int, float))
            or not isfinite(float(self.warning_pulse_period_s))
            or self.warning_pulse_period_s <= 0
        ):
            raise ValueError("warning_pulse_period_s must be a finite positive number.")
        pins = tuple(pin for pin in self.pins if pin is not None)
        if any(type(pin) is not int or pin < 0 for pin in pins):
            raise ValueError("GPIO BCM pins must be non-negative integers.")
        if len(set(pins)) != len(pins):
            raise ValueError("Each buzzer, vibration and LED output needs a distinct BCM pin.")
        if self.enabled and not pins:
            raise ValueError("An enabled alert actuator needs at least one configured BCM pin.")

    @property
    def pins(self) -> tuple[int | None, int | None, int | None]:
        return (self.buzzer_bcm_pin, self.vibration_bcm_pin, self.led_bcm_pin)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "AlertActuatorConfig":
        if value is None:
            data: Mapping[str, Any] = {}
        elif isinstance(value, Mapping):
            data = value
        else:
            raise ValueError("alert_actuator must be a JSON object.")

        def json_bool(key: str, default: bool) -> bool:
            raw = data.get(key, default)
            if type(raw) is not bool:
                raise ValueError(f"alert_actuator.{key} must be a JSON boolean.")
            return raw

        def optional_chip(key: str) -> int | None:
            raw = data.get(key)
            if raw is None:
                return None
            if type(raw) is not int:
                raise ValueError(f"alert_actuator.{key} must be a JSON integer or null.")
            return raw

        def json_number(key: str, default: float) -> float:
            raw = data.get(key, default)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise ValueError(f"alert_actuator.{key} must be a JSON number.")
            return float(raw)

        def optional_pin(key: str) -> int | None:
            raw = data.get(key)
            if raw is None:
                return None
            if type(raw) is not int:
                raise ValueError(f"alert_actuator.{key} must be a JSON integer or null.")
            return raw

        return cls(
            enabled=json_bool("enabled", False),
            gpio_chip=optional_chip("gpio_chip"),
            buzzer_bcm_pin=optional_pin("buzzer_bcm_pin"),
            vibration_bcm_pin=optional_pin("vibration_bcm_pin"),
            led_bcm_pin=optional_pin("led_bcm_pin"),
            active_high=json_bool("active_high", True),
            warning_pulse_period_s=json_number("warning_pulse_period_s", 0.50),
        )


@dataclass(frozen=True)
class AlertOutputState:
    """The applied output state, included in runtime telemetry."""

    requested_level: AlertLevel
    applied_level: AlertLevel
    alerts_permitted: bool
    buzzer_on: bool
    vibration_on: bool
    led_on: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_level": self.requested_level.name,
            "applied_level": self.applied_level.name,
            "alerts_permitted": self.alerts_permitted,
            "buzzer_on": self.buzzer_on,
            "vibration_on": self.vibration_on,
            "led_on": self.led_on,
        }


class AlertActuator(Protocol):
    def apply(
        self,
        requested_level: AlertLevel,
        *,
        alerts_permitted: bool,
        now_s: float | None = None,
    ) -> AlertOutputState: ...

    def close(self) -> None: ...


def resolve_gpio_chip(
    configured_chip: int | None,
    *,
    gpio_class_root: Path = Path("/sys/class/gpio"),
) -> int:
    """Resolve the Linux gpiochip that exposes the physical 40-pin header.

    Raspberry Pi 5 GPIO numbering changed during the Bookworm lifecycle:
    older kernels exposed the RP1 header controller as gpiochip4, while newer
    kernels deliberately reorder it to gpiochip0.  The controller label,
    rather than a fixed number, is the stable safety contract.  If it cannot
    be determined unambiguously, do not guess which chip drives the helmet's
    physical output; require an explicit, verified ``gpio_chip`` instead.
    """

    if configured_chip is not None:
        return configured_chip
    candidates: list[int] = []
    try:
        entries = tuple(gpio_class_root.glob("gpiochip*"))
    except OSError as exc:  # pragma: no cover - only relevant on a Pi
        raise RuntimeError(f"Cannot inspect Linux GPIO controllers: {exc}") from exc
    for entry in entries:
        suffix = entry.name.removeprefix("gpiochip")
        if not suffix.isdigit():
            continue
        try:
            label = (entry / "label").read_text(encoding="utf-8").strip().lower()
        except OSError:
            continue
        if label.startswith("pinctrl"):
            candidates.append(int(suffix))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError(
            "Could not auto-detect the user-facing GPIO controller. "
            "Run 'gpiodetect', identify the gpiochip labelled pinctrl*, and set alert_actuator.gpio_chip explicitly."
        )
    raise RuntimeError(
        "More than one gpiochip is labelled pinctrl; set alert_actuator.gpio_chip explicitly after checking 'gpiodetect'."
    )


class NullAlertActuator:
    """No-op output used when no physical alert wiring was explicitly enabled."""

    def apply(
        self,
        requested_level: AlertLevel,
        *,
        alerts_permitted: bool,
        now_s: float | None = None,
    ) -> AlertOutputState:
        return AlertOutputState(
            requested_level=requested_level,
            applied_level=AlertLevel.NONE,
            alerts_permitted=alerts_permitted,
            buzzer_on=False,
            vibration_on=False,
            led_on=False,
        )

    def close(self) -> None:
        return None


class LgpioAlertActuator:
    """Drive configured active-high/low GPIO outputs through ``python3-lgpio``."""

    def __init__(self, config: AlertActuatorConfig) -> None:
        if not config.enabled:
            raise ValueError("LgpioAlertActuator requires enabled=True.")
        self.config = config
        try:
            import lgpio  # type: ignore
        except ImportError as exc:  # pragma: no cover - Pi-only dependency
            raise RuntimeError("GPIO alerts require python3-lgpio on Raspberry Pi OS.") from exc
        self._lgpio = lgpio
        self._chip: int | None = None
        self.gpio_chip = resolve_gpio_chip(config.gpio_chip)
        self._written: dict[int, bool] = {}
        try:
            self._chip = lgpio.gpiochip_open(self.gpio_chip)
            for pin in self._configured_pins:
                lgpio.gpio_claim_output(self._chip, pin, self._electrical_value(False))
                self._written[pin] = False
        except Exception as exc:  # pragma: no cover - Pi-only dependency
            try:
                self.close()
            except Exception:
                pass
            raise RuntimeError(f"Could not initialise Raspberry Pi GPIO alert output: {exc}") from exc

    @property
    def _configured_pins(self) -> tuple[int, ...]:
        return tuple(pin for pin in self.config.pins if pin is not None)

    def _electrical_value(self, logical_on: bool) -> int:
        return int(logical_on if self.config.active_high else not logical_on)

    def _set(self, pin: int | None, enabled: bool) -> None:
        if pin is None or self._chip is None or self._written.get(pin) == enabled:
            return
        self._lgpio.gpio_write(self._chip, pin, self._electrical_value(enabled))
        self._written[pin] = enabled

    def apply(
        self,
        requested_level: AlertLevel,
        *,
        alerts_permitted: bool,
        now_s: float | None = None,
    ) -> AlertOutputState:
        applied_level = requested_level if alerts_permitted else AlertLevel.NONE
        clock_s = monotonic() if now_s is None else now_s
        if applied_level == AlertLevel.CRITICAL:
            requested_buzzer_on = requested_vibration_on = requested_led_on = True
        elif applied_level == AlertLevel.WARNING:
            pulse_on = int(clock_s / self.config.warning_pulse_period_s) % 2 == 0
            requested_buzzer_on = requested_vibration_on = requested_led_on = pulse_on
        elif applied_level == AlertLevel.ADVISORY:
            requested_buzzer_on = requested_vibration_on = False
            requested_led_on = True
        else:
            requested_buzzer_on = requested_vibration_on = requested_led_on = False
        # Telemetry describes the output that actually exists, not a desired
        # pattern for a device omitted from the explicit wiring configuration.
        buzzer_on = requested_buzzer_on and self.config.buzzer_bcm_pin is not None
        vibration_on = requested_vibration_on and self.config.vibration_bcm_pin is not None
        led_on = requested_led_on and self.config.led_bcm_pin is not None
        self._set(self.config.buzzer_bcm_pin, buzzer_on)
        self._set(self.config.vibration_bcm_pin, vibration_on)
        self._set(self.config.led_bcm_pin, led_on)
        return AlertOutputState(
            requested_level=requested_level,
            applied_level=applied_level,
            alerts_permitted=alerts_permitted,
            buzzer_on=buzzer_on,
            vibration_on=vibration_on,
            led_on=led_on,
        )

    def close(self) -> None:  # pragma: no cover - Pi-only dependency
        errors: list[str] = []
        for pin in self._configured_pins:
            try:
                self._set(pin, False)
            except Exception as exc:
                errors.append(str(exc))
        if self._chip is not None:
            try:
                self._lgpio.gpiochip_close(self._chip)
            except Exception as exc:
                errors.append(str(exc))
            finally:
                self._chip = None
        if errors:
            raise RuntimeError("Could not safely close GPIO alert output: " + "; ".join(errors))


def create_alert_actuator(config: AlertActuatorConfig) -> AlertActuator:
    """Create a no-op actuator unless the user has explicitly enabled GPIO."""

    return LgpioAlertActuator(config) if config.enabled else NullAlertActuator()
