"""Operator-facing visual demo powered by the real HelmetAI risk pipelines.

This module is intentionally a controlled simulation.  It makes the front and
rear risk-engine decisions visible for a stakeholder demonstration without
claiming that a road or hardware validation has already occurred.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Callable

from .models import AlertLevel, RiskAssessment
from .pipeline import ForwardCollisionPipeline
from .rear_pipeline import RearCollisionPipeline
from .simulator import (
    ClosingScenario,
    RearClosingScenario,
    closing_vehicle_observations,
    rear_closing_vehicle_observations,
)


@dataclass(frozen=True)
class DemoSample:
    """One displayed sample from an actual risk-pipeline assessment."""

    distance_m: float
    assessment: RiskAssessment


def build_demo_timelines() -> tuple[list[DemoSample], list[DemoSample]]:
    """Create deterministic front/rear timelines using production pipelines."""

    front_pipeline = ForwardCollisionPipeline()
    rear_pipeline = RearCollisionPipeline()
    front = [
        DemoSample(observation.longitudinal_m, front_pipeline.process(observation))
        for observation in closing_vehicle_observations(ClosingScenario(duration_s=5.2))
    ]
    rear = [
        DemoSample(observation.longitudinal_m, rear_pipeline.process(observation))
        for observation in rear_closing_vehicle_observations(RearClosingScenario(duration_s=5.2))
    ]
    return front, rear


class HelmetAIDemoDashboard:
    """Small Tk dashboard for stakeholder demonstrations on a Windows laptop."""

    tick_ms = 130
    level_colours = {
        AlertLevel.NONE: "#2e7d32",
        AlertLevel.ADVISORY: "#f9a825",
        AlertLevel.WARNING: "#ef6c00",
        AlertLevel.CRITICAL: "#c62828",
    }

    def __init__(self, root: object, front: list[DemoSample], rear: list[DemoSample]) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.front = front
        self.rear = rear
        self.index = 0
        self.running = False
        self.after_id: str | None = None

        root.title("HelmetAI - Front and Rear Collision Risk Demo")
        root.minsize(1040, 670)
        root.configure(padx=18, pady=16)

        title = ttk.Label(root, text="HelmetAI Collision-Risk Demonstration", font=("Segoe UI", 21, "bold"))
        title.pack(anchor="w")
        subtitle = ttk.Label(
            root,
            text="Risk-engine demonstration: distance, closing speed, TTC, braking demand and safety gates drive every decision.",
            font=("Segoe UI", 10),
        )
        subtitle.pack(anchor="w", pady=(2, 14))

        visual_row = ttk.Frame(root)
        visual_row.pack(fill="both", expand=True)
        self.front_panel = self._create_panel(visual_row, "FRONT CAMERA", "Vehicle ahead is closing")
        self.front_panel["frame"].pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.rear_panel = self._create_panel(visual_row, "REAR CAMERA", "Vehicle behind is closing")
        self.rear_panel["frame"].pack(side="left", fill="both", expand=True, padx=(8, 0))

        controls = ttk.Frame(root)
        controls.pack(fill="x", pady=(14, 0))
        ttk.Button(controls, text="Start", command=self.start).pack(side="left")
        ttk.Button(controls, text="Pause", command=self.pause).pack(side="left", padx=8)
        ttk.Button(controls, text="Reset", command=self.reset).pack(side="left")
        self.progress_var = tk.StringVar(value="Frame 0 / 53")
        ttk.Label(controls, textvariable=self.progress_var).pack(side="right")
        rule_text = (
            "Decision safeguards: calibrated camera + stable track + target in corridor + meaningful closing speed. "
            "Otherwise no collision alert is produced."
        )
        ttk.Label(root, text=rule_text, font=("Segoe UI", 9)).pack(anchor="w", pady=(10, 0))
        self._render()

    def _create_panel(self, parent: object, heading: str, caption: str) -> dict[str, object]:
        frame = self.ttk.LabelFrame(parent, text=heading, padding=12)
        self.ttk.Label(frame, text=caption, font=("Segoe UI", 10)).pack(anchor="w")
        canvas = self.tk.Canvas(frame, height=300, highlightthickness=1, highlightbackground="#b0b0b0", background="#eef3f5")
        canvas.pack(fill="both", expand=True, pady=(8, 10))
        level_var = self.tk.StringVar(value="NONE")
        details_var = self.tk.StringVar(value="Waiting for demo")
        explanation_var = self.tk.StringVar(value="Decision explanation will appear here.")
        self.ttk.Label(frame, textvariable=level_var, font=("Segoe UI", 17, "bold")).pack(anchor="w")
        self.ttk.Label(frame, textvariable=details_var, font=("Segoe UI", 10)).pack(anchor="w", pady=(2, 0))
        self.ttk.Label(frame, textvariable=explanation_var, font=("Segoe UI", 9), wraplength=470).pack(
            anchor="w", pady=(4, 0)
        )
        return {
            "frame": frame,
            "canvas": canvas,
            "level": level_var,
            "details": details_var,
            "explanation": explanation_var,
        }

    @staticmethod
    def _display_ttc(value: float | None) -> str:
        if value is None or not isfinite(value):
            return "—"
        return f"{value:.1f} s"

    def _draw_scene(self, panel: dict[str, object], sample: DemoSample, *, rear: bool) -> None:
        canvas = panel["canvas"]
        canvas.delete("all")
        width = max(int(canvas.winfo_width()), 420)
        height = max(int(canvas.winfo_height()), 300)
        centre_x = width // 2
        bike_y = 80 if rear else height - 80
        max_distance = 40.0 if rear else 52.0
        closeness = 1.0 - min(sample.distance_m, max_distance) / max_distance
        car_y = int((height - 105) - closeness * (height - 190)) if rear else int(70 + closeness * (height - 190))
        car_colour = self.level_colours[sample.assessment.level]

        for y in range(12, height, 34):
            canvas.create_line(centre_x - 2, y, centre_x + 2, y + 14, fill="#aab7b8", width=2)
        canvas.create_polygon(
            centre_x,
            bike_y - 26,
            centre_x - 22,
            bike_y + 26,
            centre_x + 22,
            bike_y + 26,
            fill="#1565c0",
            outline="#0d47a1",
        )
        canvas.create_text(centre_x, bike_y + (45 if rear else -45), text="MOTORCYCLE", font=("Segoe UI", 9, "bold"), fill="#0d47a1")
        canvas.create_rectangle(centre_x - 52, car_y - 25, centre_x + 52, car_y + 25, fill=car_colour, outline="#263238", width=2)
        canvas.create_text(centre_x, car_y, text="APPROACHING VEHICLE", font=("Segoe UI", 9, "bold"), fill="white")
        canvas.create_text(
            12,
            14,
            anchor="nw",
            text=f"Estimated distance: {sample.distance_m:.1f} m",
            font=("Segoe UI", 10, "bold"),
            fill="#263238",
        )

    def _render_panel(self, panel: dict[str, object], sample: DemoSample, *, rear: bool) -> None:
        self._draw_scene(panel, sample, rear=rear)
        assessment = sample.assessment
        panel["level"].set(f"{assessment.level.name}  |  {assessment.direction.value.upper()}")
        braking = "—" if assessment.required_relative_decel_mps2 is None else f"{assessment.required_relative_decel_mps2:.1f} m/s²"
        panel["details"].set(
            f"TTC: {self._display_ttc(assessment.conservative_ttc_s)}   •   "
            f"Closing speed: {assessment.closing_speed_mps or 0.0:.1f} m/s   •   Required braking: {braking}"
        )
        panel["explanation"].set(self._explanation(assessment))

    @staticmethod
    def _explanation(assessment: RiskAssessment) -> str:
        if assessment.system_status.value != "ready":
            return "Decision: risk alert disabled because calibration or track-quality safety checks are not valid yet."
        if not assessment.in_path:
            return "Decision: no alert because the detected vehicle is outside the motorcycle travel corridor."
        if assessment.level == AlertLevel.NONE:
            return "Decision: no alert because the vehicle is not closing at a meaningful relative speed."
        reasons = " • ".join(reason.replace("_", " ") for reason in assessment.reasons[:3])
        return f"Decision factors: {reasons}. Risk is elevated from TTC and required braking demand."

    def _render(self) -> None:
        front_sample = self.front[min(self.index, len(self.front) - 1)]
        rear_sample = self.rear[min(self.index, len(self.rear) - 1)]
        self._render_panel(self.front_panel, front_sample, rear=False)
        self._render_panel(self.rear_panel, rear_sample, rear=True)
        frame_total = min(len(self.front), len(self.rear))
        self.progress_var.set(f"Frame {min(self.index + 1, frame_total)} / {frame_total}")

    def _tick(self) -> None:
        if not self.running:
            return
        frame_total = min(len(self.front), len(self.rear))
        if self.index >= frame_total - 1:
            self.pause()
            return
        self.index += 1
        self._render()
        self.after_id = self.root.after(self.tick_ms, self._tick)

    def start(self) -> None:
        if not self.running:
            self.running = True
            self.after_id = self.root.after(self.tick_ms, self._tick)

    def pause(self) -> None:
        self.running = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def reset(self) -> None:
        self.pause()
        self.index = 0
        self._render()


def main() -> int:
    try:
        import tkinter as tk
    except ImportError as exc:  # pragma: no cover - Windows CPython normally includes Tk
        raise RuntimeError("The Windows Python Tk GUI component is required for the demo dashboard.") from exc
    front, rear = build_demo_timelines()
    root = tk.Tk()
    HelmetAIDemoDashboard(root, front, rear)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
