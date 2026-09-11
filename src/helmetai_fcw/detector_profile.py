"""Bounded, image-only OpenCV CPU thread comparison; never a safety benchmark.

Run ``python -m helmetai_fcw.detector_profile --image captured.jpg --config
/etc/helmetai-fcw/pi5_dual_camera.json``. No camera, runtime or GPIO modules are
imported. The report cannot authorize alerts and never changes configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .detection import OpenCvYoloOnnxDetector


def _bounded_integer(value: str, *, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected an integer.") from exc
    if not minimum <= number <= maximum:
        raise argparse.ArgumentTypeError(f"Expected a value between {minimum} and {maximum}.")
    return number


def _thread_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Threads must be comma-separated values from 1,2,3,4.") from exc
    if not counts or len(set(counts)) != len(counts) or any(count not in (1, 2, 3, 4) for count in counts):
        raise argparse.ArgumentTypeError("Choose distinct thread counts from 1,2,3,4.")
    return counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Existing captured JPEG; no camera is opened.")
    parser.add_argument("--config", type=Path, help="Read detector settings only from this JSON configuration.")
    parser.add_argument("--model", type=Path, help="Explicit local ONNX model (overrides config model_path).")
    parser.add_argument("--threads", type=_thread_counts, default=(1, 2, 3, 4))
    parser.add_argument("--warmup", type=lambda v: _bounded_integer(v, minimum=1, maximum=20), default=3)
    parser.add_argument("--runs", type=lambda v: _bounded_integer(v, minimum=1, maximum=100), default=10)
    parser.add_argument("--output", type=Path, help="Optional NEW report file; existing files are never overwritten.")
    return parser


def _settings(config_path: Path | None, model_override: Path | None) -> tuple[Path, int, float]:
    """Load only detector fields, without importing camera or alert code."""

    detector = {}
    if config_path is not None:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        detector = config["detector"]
        if not isinstance(detector, dict):
            raise ValueError("config detector must be a JSON object.")
    if model_override is not None:
        model_path = model_override.resolve()
    elif detector.get("model_path"):
        model_path = Path(detector["model_path"])
        if not model_path.is_absolute():
            model_path = config_path.parent / model_path
        model_path = model_path.resolve()
    else:
        raise ValueError("Supply --config with detector.model_path or --model.")
    input_size = int(detector.get("input_size_px", 640))
    if input_size != 640:
        raise ValueError("The packaged detector profile requires input_size_px=640; it is not resized for speed.")
    confidence = float(detector.get("confidence", 0.45))
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError("detector.confidence must be finite and between 0 and 1.")
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX detector model not found: {model_path}")
    return model_path, input_size, confidence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    return {
        "median_ms": round(statistics.median(samples), 3),
        "p95_ms": round(ordered[math.ceil(0.95 * len(ordered)) - 1], 3),
        "minimum_ms": round(ordered[0], 3),
        "maximum_ms": round(ordered[-1], 3),
        "samples_ms": [round(value, 3) for value in samples],
    }


def profile(args: argparse.Namespace) -> dict[str, object]:
    """Measure synchronous detector-adapter calls on a preloaded JPEG."""

    model_path, input_size, confidence = _settings(args.config, args.model)
    if args.output is not None and args.output.exists():
        raise FileExistsError(f"Report already exists; choose a new --output path: {args.output}")
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy must already be installed; this tool does not download packages.") from exc
    image_path = args.image.resolve()
    image_bytes = image_path.read_bytes()
    if not image_bytes.startswith(b"\xff\xd8"):
        raise ValueError("--image must be an existing JPEG capture.")
    try:
        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise RuntimeError(f"OpenCV could not decode JPEG: {exc}") from exc
    if frame is None:
        raise ValueError(f"Could not decode JPEG: {image_path}")
    original_threads = cv2.getNumThreads()
    original_optimized = cv2.useOptimized()
    results = []
    try:
        for threads in args.threads:
            detector = OpenCvYoloOnnxDetector(
                model_path, input_size_px=input_size, confidence_threshold=confidence, cpu_threads=threads
            )
            for _ in range(args.warmup):
                detector.detect(frame)
            samples = []
            detection_counts = []
            for _ in range(args.runs):
                start_ns = time.perf_counter_ns()
                detections = detector.detect(frame)
                samples.append((time.perf_counter_ns() - start_ns) / 1_000_000.0)
                detection_counts.append(len(detections))
            result = {
                "requested_cpu_threads": threads,
                "effective_cpu_threads": cv2.getNumThreads(),
                "warmup_calls_excluded": args.warmup,
                "measured_calls": args.runs,
                "detection_counts": detection_counts,
                **_summary(samples),
            }
            results.append(result)
            print(
                f"threads={threads} effective={result['effective_cpu_threads']} "
                f"detector_median_ms={result['median_ms']} detector_p95_ms={result['p95_ms']}",
                file=sys.stderr,
                flush=True,
            )
            del detector
    except cv2.error as exc:
        raise RuntimeError(f"OpenCV detector profiling failed: {exc}") from exc
    finally:
        # OpenCV threading is process-global. Restore it for imported callers.
        cv2.setNumThreads(original_threads)
        cv2.setUseOptimized(original_optimized)
    return {
        "schema_version": 1,
        "profile_kind": "captured_jpeg_detector_only",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "measurement_scope": "OpenCV blob preprocessing + synchronous net.forward + decode/class-aware NMS",
        "excluded": ["JPEG file IO/decode", "model loading", "camera capture", "tracking", "risk logic", "GPIO/alerts"],
        "safety_benchmark": False,
        "configuration_modified": False,
        "warning": "Host-only measurement. Not a camera-to-alert benchmark or permission for physical alerts. "
        "Thermals, CPU load and run order can bias results; repeat on the target Pi before changing thread settings.",
        "percentile_method": "nearest_rank",
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "opencv_version": cv2.__version__,
            "numpy_version": np.__version__,
            "logical_cpu_count": os.cpu_count(),
            "backend": "OpenCV DNN default CPU (no accelerator selected)",
        },
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "image": {
            "path": str(image_path),
            "sha256": hashlib.sha256(image_bytes).hexdigest(),
            "width_px": int(frame.shape[1]),
            "height_px": int(frame.shape[0]),
        },
        "detector": {"input_size_px": input_size, "confidence_threshold": confidence, "nms_threshold": 0.45},
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = profile(args)
        payload = json.dumps(report, indent=2, allow_nan=False)
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as destination:
                destination.write(payload + "\n")
        print(payload)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"Detector profile failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
