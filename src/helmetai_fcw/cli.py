"""Command-line demo for the FCW MVP."""

from __future__ import annotations

import argparse
from dataclasses import replace
from math import isfinite

from .awareness import TurnIntent
from .alert_actuator import AlertActuatorConfig
from .camera_check import save_camera_check_images
from .models import AlertLevel
from .demo_dashboard import main as demo_dashboard_main
from .monocular import focal_length_from_reference
from .pi_runtime import PiForwardCollisionRuntime, PiRuntimeConfig
from .webcam_runtime import WebcamForwardCollisionRuntime, WebcamRuntimeConfig
from .video_runtime import VideoCollisionRiskRuntime
from .simulator import ClosingScenario, RearClosingScenario, run_closing_demo, run_rear_closing_demo
from .stereo import StereoCalibration


def _demo(args: argparse.Namespace) -> int:
    scenario = ClosingScenario(
        initial_distance_m=args.initial_distance,
        closing_speed_mps=args.closing_speed,
        duration_s=args.duration,
    )
    assessments = run_closing_demo(scenario)
    last_level = AlertLevel.NONE
    print("direction,time_s,level,system,ttc_s,required_decel_mps2,reasons")
    for assessment in assessments:
        if assessment.level != last_level or assessment.level == AlertLevel.CRITICAL:
            ttc = "" if assessment.conservative_ttc_s is None or not isfinite(assessment.conservative_ttc_s) else f"{assessment.conservative_ttc_s:.2f}"
            decel = "" if assessment.required_relative_decel_mps2 is None else f"{assessment.required_relative_decel_mps2:.2f}"
            print(
                f"{assessment.direction.value},{assessment.timestamp_s:.2f},{assessment.level.name},{assessment.system_status.value},"
                f"{ttc},{decel},{'|'.join(assessment.reasons)}"
            )
        last_level = assessment.level
    return 0


def _rear_demo(args: argparse.Namespace) -> int:
    scenario = RearClosingScenario(
        initial_distance_m=args.initial_distance,
        closing_speed_mps=args.closing_speed,
        duration_s=args.duration,
    )
    assessments = run_rear_closing_demo(scenario)
    last_level = AlertLevel.NONE
    print("direction,time_s,level,system,ttc_s,required_decel_mps2,reasons")
    for assessment in assessments:
        if assessment.level != last_level or assessment.level == AlertLevel.CRITICAL:
            ttc = "" if assessment.conservative_ttc_s is None or not isfinite(assessment.conservative_ttc_s) else f"{assessment.conservative_ttc_s:.2f}"
            decel = "" if assessment.required_relative_decel_mps2 is None else f"{assessment.required_relative_decel_mps2:.2f}"
            print(
                f"{assessment.direction.value},{assessment.timestamp_s:.2f},{assessment.level.name},{assessment.system_status.value},"
                f"{ttc},{decel},{'|'.join(assessment.reasons)}"
            )
        last_level = assessment.level
    return 0


def _demo_dashboard(args: argparse.Namespace) -> int:
    return demo_dashboard_main()


def _stereo(args: argparse.Namespace) -> int:
    calibration = StereoCalibration(args.focal_px, args.baseline_m)
    depth_m, sigma_m = calibration.depth_from_disparity(args.disparity_px, args.disparity_sigma_px)
    print(f"depth_m={depth_m:.3f}")
    print(f"one_sigma_m={sigma_m:.3f}")
    return 0


def _calibrate_monocular(args: argparse.Namespace) -> int:
    focal_length_px = focal_length_from_reference(args.reference_distance, args.object_width, args.pixel_width)
    print(f"focal_length_px={focal_length_px:.3f}")
    print(f"Apply this value to {args.camera_role}_calibration.focal_length_px after controlled validation.")
    print(
        "Bu değeri config/pi5_dual_camera.example.json içindeki "
        f"{args.camera_role}_calibration.focal_length_px alanına yazın."
    )
    return 0


def _pi_run(args: argparse.Namespace) -> int:
    config = PiRuntimeConfig.from_json(args.config)
    if args.camera_only:
        config = replace(config, alert_actuator=AlertActuatorConfig())
    PiForwardCollisionRuntime(config).run(max_frames=args.frames, turn_intent=TurnIntent(args.turn_intent))
    return 0


def _pi_camera_check(args: argparse.Namespace) -> int:
    config = PiRuntimeConfig.from_json(args.config)
    front, rear = save_camera_check_images(config, args.output_dir)
    print(f"front_camera_check={front}")
    if rear is not None:
        print(f"rear_camera_check={rear}")
    print("Confirm FRONT/REAR labels by covering each physical lens before running risk analysis.")
    return 0


def _webcam_run(args: argparse.Namespace) -> int:
    config = WebcamRuntimeConfig.from_json(args.config)
    WebcamForwardCollisionRuntime(config).run(max_frames=args.frames)
    return 0


def _video_run(args: argparse.Namespace) -> int:
    config = WebcamRuntimeConfig.from_json(args.config)
    runtime = VideoCollisionRiskRuntime(
        config, args.video, args.output, report_dir=args.report_dir,
        max_frames=None if args.max_frames == 0 else args.max_frames, start_seconds=args.start_seconds,
        cpu_threads=args.cpu_threads, source_fps_override=args.source_fps,
    )
    processed_frames = runtime.run(preview=not args.no_preview)
    print(f"processed_frames={processed_frames}")
    if args.output:
        print(f"annotated_video={args.output}")
    if args.report_dir:
        print(f"report_directory={args.report_dir}")
    return 0


def _image_check(args: argparse.Namespace) -> int:
    from .image_check import main as image_check_main
    return image_check_main([
        "--image", args.image, "--model", args.model,
        "--output-dir", args.output_dir, "--direction", args.direction,
        "--confidence", str(args.confidence), "--cpu-threads", str(args.cpu_threads),
        "--rotate", str(args.rotate),
    ])


def _log_report(args: argparse.Namespace) -> int:
    from .log_report import main as log_report_main
    options = ["--runtime", args.runtime, "--output-dir", args.output_dir]
    if args.before:
        options.extend(["--before", args.before])
    if args.after:
        options.extend(["--after", args.after])
    return log_report_main(options)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HelmetAI FCW MVP")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="run a synthetic closing-vehicle scenario")
    demo.add_argument("--initial-distance", type=float, default=48.0)
    demo.add_argument("--closing-speed", type=float, default=11.0, help="metres per second")
    demo.add_argument("--duration", type=float, default=6.0)
    demo.set_defaults(handler=_demo)
    rear_demo = commands.add_parser("rear-demo", help="run a synthetic rear-approach scenario")
    rear_demo.add_argument("--initial-distance", type=float, default=36.0)
    rear_demo.add_argument("--closing-speed", type=float, default=9.0, help="metres per second")
    rear_demo.add_argument("--duration", type=float, default=5.0)
    rear_demo.set_defaults(handler=_rear_demo)
    dashboard = commands.add_parser("demo-dashboard", help="open the visual front/rear stakeholder demo")
    dashboard.set_defaults(handler=_demo_dashboard)
    stereo = commands.add_parser("stereo", help="calculate depth from a calibrated disparity")
    stereo.add_argument("--focal-px", type=float, default=700.0)
    stereo.add_argument("--baseline-m", type=float, default=0.12)
    stereo.add_argument("--disparity-px", type=float, required=True)
    stereo.add_argument("--disparity-sigma-px", type=float, default=0.5)
    stereo.set_defaults(handler=_stereo)
    monocular = commands.add_parser("calibrate-monocular", help="calculate focal length from a manual reference measurement")
    monocular.add_argument("--reference-distance", type=float, required=True, help="metres")
    monocular.add_argument("--object-width", type=float, required=True, help="metres")
    monocular.add_argument("--pixel-width", type=float, required=True, help="pixels")
    monocular.add_argument("--camera-role", choices=("front", "rear"), default="front")
    monocular.set_defaults(handler=_calibrate_monocular)
    pi_run = commands.add_parser("pi-run", help="run front/rear risk analysis with Raspberry Pi cameras")
    pi_run.add_argument("--config", default="config/pi5_dual_camera.json")
    pi_run.add_argument("--frames", type=int, default=None, help="optional limit for bench testing")
    pi_run.add_argument("--camera-only", action="store_true", help="force GPIO outputs off in memory without changing saved calibration/configuration")
    pi_run.add_argument(
        "--turn-intent",
        choices=tuple(intent.value for intent in TurnIntent),
        default=TurnIntent.NONE.value,
        help="software-only blind-spot test input; use an adapter for a physical signal later",
    )
    pi_run.set_defaults(handler=_pi_run)
    pi_check = commands.add_parser("pi-camera-check", help="save labelled Pi CSI camera images to verify front/rear mapping")
    pi_check.add_argument("--config", default="config/pi5_dual_camera.json")
    pi_check.add_argument("--output-dir", default="/tmp/helmetai-camera-check")
    pi_check.set_defaults(handler=_pi_camera_check)
    webcam_run = commands.add_parser("webcam-run", help="test FCW with the computer's internal or USB webcam")
    webcam_run.add_argument("--config", default="config/webcam.example.json")
    webcam_run.add_argument("--frames", type=int, default=None)
    webcam_run.set_defaults(handler=_webcam_run)
    video_run = commands.add_parser("video-run", help="annotate a controlled real-video recording with HelmetAI risk output")
    video_run.add_argument("--config", default="config/webcam.example.json")
    video_run.add_argument("--video", required=True, help="input video file, such as a controlled MP4 recording")
    video_run.add_argument("--output", default=None, help="optional annotated MP4 output path")
    video_run.add_argument("--no-preview", action="store_true", help="process without opening a playback window")
    video_run.add_argument("--report-dir", default=None, help="new directory for per-frame evidence and summaries")
    video_run.add_argument("--max-frames", type=int, default=300, help="processing limit (default 300); 0 explicitly requests the whole file")
    video_run.add_argument("--start-seconds", type=float, default=0.0)
    video_run.add_argument("--cpu-threads", type=int, default=None)
    video_run.add_argument("--source-fps", type=float, default=None, help="explicit assumed source FPS if metadata is invalid")
    video_run.set_defaults(handler=_video_run)
    image_check = commands.add_parser("image-check", help="save boxes/labels for one image; no metric risk or camera access")
    image_check.add_argument("--image", required=True)
    image_check.add_argument("--model", required=True)
    image_check.add_argument("--output-dir", required=True)
    image_check.add_argument("--direction", choices=("front", "rear"), required=True)
    image_check.add_argument("--confidence", type=float, default=0.45)
    image_check.add_argument("--cpu-threads", type=int, default=1)
    image_check.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0)
    image_check.set_defaults(handler=_image_check)
    log_report = commands.add_parser("log-report", help="summarize completed and partial Pi logs without safety approval")
    log_report.add_argument("--runtime", required=True)
    log_report.add_argument("--before", default=None)
    log_report.add_argument("--after", default=None)
    log_report.add_argument("--output-dir", required=True)
    log_report.set_defaults(handler=_log_report)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)
