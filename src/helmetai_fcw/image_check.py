"""One-image, offline detection evidence; no range, risk, camera or actuator path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

from .detection import Detection, OpenCvYoloOnnxDetector

MAX_IMAGE_BYTES = 32 * 1024 * 1024
MAX_MODEL_BYTES = 256 * 1024 * 1024
MAX_IMAGE_PIXELS = 24_000_000
MAX_IMAGE_SIDE = 8192
MAX_DETECTIONS = 1000
AMBIGUITY_IOU = 0.85
DISCLAIMER = (
    "DETECTION ONLY. Confidence is the model's label score, not measured accuracy "
    "or the probability that a detection is correct. No labelled ground truth was "
    "provided. No distance, TTC, risk, safety or road-readiness conclusion is available."
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_dimensions(data: bytes) -> tuple[int, int]:
    """Check dimensions before the OpenCV decoder can allocate an enormous image."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        if len(data) < 24 or data[8:16] != b"\x00\x00\x00\rIHDR":
            raise ValueError("Invalid PNG header.")
        width, height = struct.unpack(">II", data[16:24])
    elif data.startswith(b"\xff\xd8"):
        offset = 2
        width = height = 0
        # SOF markers carrying width/height; DHT, JPG and DAC are not SOF.
        sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
        while offset < len(data):
            if data[offset] != 0xFF:
                raise ValueError("Invalid JPEG marker stream.")
            while offset < len(data) and data[offset] == 0xFF:
                offset += 1
            if offset >= len(data):
                break
            marker = data[offset]
            offset += 1
            if marker in (0xD9, 0xDA):
                break
            if marker in {0x01, *range(0xD0, 0xD9)}:
                continue
            if offset + 2 > len(data):
                break
            size = int.from_bytes(data[offset:offset + 2], "big")
            if size < 2 or offset + size > len(data):
                raise ValueError("Truncated JPEG segment.")
            if marker in sof:
                if size < 8:
                    raise ValueError("Invalid JPEG dimensions segment.")
                height, width = struct.unpack(">HH", data[offset + 3:offset + 7])
                break
            offset += size
    else:
        raise ValueError("Only actual JPEG or PNG image content is supported.")
    if not (0 < width <= MAX_IMAGE_SIDE and 0 < height <= MAX_IMAGE_SIDE):
        raise ValueError(f"Image dimensions must be 1..{MAX_IMAGE_SIDE} pixels per side.")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(f"Image exceeds the {MAX_IMAGE_PIXELS}-pixel limit.")
    return width, height


def _validated_paths(image_path: str | Path, model_path: str | Path, output_directory: str | Path) -> tuple[Path, Path, Path]:
    image = Path(image_path).expanduser().resolve(strict=True)
    model = Path(model_path).expanduser().resolve(strict=True)
    requested_output = Path(output_directory).expanduser().absolute()
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError("Output directory must be new; existing paths are never overwritten.")
    output = requested_output.resolve()
    if not output.parent.is_dir():
        raise ValueError("Output parent directory must already exist.")
    if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Image must be a local JPEG or PNG file.")
    if not model.is_file() or model.suffix.lower() != ".onnx":
        raise ValueError("Model must be a local .onnx file.")
    if image.samefile(model) or output in (image, model):
        raise ValueError("Image, model and output must not alias each other.")
    if not 0 < image.stat().st_size <= MAX_IMAGE_BYTES:
        raise ValueError(f"Image must be nonempty and at most {MAX_IMAGE_BYTES} bytes.")
    if not 0 < model.stat().st_size <= MAX_MODEL_BYTES:
        raise ValueError(f"Model must be nonempty and at most {MAX_MODEL_BYTES} bytes.")
    return image, model, output


def _iou(a: Detection, b: Detection) -> float:
    width = max(0.0, min(a.x_px + a.width_px, b.x_px + b.width_px) - max(a.x_px, b.x_px))
    height = max(0.0, min(a.y_px + a.height_px, b.y_px + b.height_px) - max(a.y_px, b.y_px))
    intersection = width * height
    return intersection / (a.area_px2 + b.area_px2 - intersection)


def _ambiguities(detections: list[Detection]) -> list[dict]:
    pairs = []
    for index, a in enumerate(detections):
        for second_index in range(index + 1, len(detections)):
            b = detections[second_index]
            if a.label != b.label:
                overlap = _iou(a, b)
                if overlap >= AMBIGUITY_IOU:
                    pairs.append({"detection_ids": [index + 1, second_index + 1], "labels": [a.label, b.label], "iou": overlap})
    return pairs


def _validate_detections(detections: list[Detection]) -> None:
    if len(detections) > MAX_DETECTIONS:
        raise ValueError(f"Detector returned more than {MAX_DETECTIONS} detections; no report written.")
    for detection in detections:
        values = (detection.confidence, detection.x_px, detection.y_px, detection.width_px, detection.height_px)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Detector returned non-finite confidence or geometry.")
        if not 0 <= detection.confidence <= 1 or detection.width_px <= 0 or detection.height_px <= 0:
            raise ValueError("Detector returned invalid confidence or geometry.")
        if not detection.label or len(detection.label) > 64 or not all(char.isascii() and (char.isalnum() or char == "_") for char in detection.label):
            raise ValueError("Detector returned an invalid label.")


def _annotated_image(cv2, np, frame, detections: list[Detection], direction: str, ambiguity_count: int):
    """Keep a readable evidence legend separate from the photograph, even at 1x1."""
    height, width = frame.shape[:2]
    scale = min(1.0, 1600 / width, 1200 / height)
    display_width, display_height = max(1, round(width * scale)), max(1, round(height * scale))
    photo = cv2.resize(frame, (display_width, display_height))
    photo_column = max(640, display_width)
    shown = detections[:100]
    banner_height = 96
    canvas = np.full((max(display_height, 150 + len(shown) * 26) + banner_height, photo_column + 510, 3), 24, dtype=np.uint8)
    canvas[banner_height:banner_height + display_height, :display_width] = photo
    colour = (255, 190, 70)  # blue/orange; never a green 'safe' result
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, f"DETECTION ONLY | {direction.upper()} (USER LABEL)", (16, 29), font, 0.78, (255, 255, 255), 2)
    cv2.putText(canvas, "No range / TTC / risk / safety conclusion. Confidence is not accuracy.", (16, 57), font, 0.60, (220, 220, 220), 1)
    cv2.putText(canvas, f"Detections: {len(detections)} | Near-identical cross-label box pairs: {ambiguity_count}", (16, 81), font, 0.56, colour, 1)
    legend_x = photo_column + 16
    cv2.putText(canvas, "ID | MODEL LABEL | CONFIDENCE", (legend_x, 126), font, 0.58, (255, 255, 255), 1)
    anchor_uses: dict[tuple[int, int], int] = {}
    for index, detection in enumerate(shown, 1):
        left = min(display_width - 1, max(0, round(detection.x_px * scale)))
        top = min(display_height - 1, max(0, round(detection.y_px * scale)))
        right = min(display_width - 1, max(0, round((detection.x_px + detection.width_px) * scale)))
        bottom = min(display_height - 1, max(0, round((detection.y_px + detection.height_px) * scale)))
        if right > left and bottom > top:
            cv2.rectangle(canvas, (left, top + banner_height), (right, bottom + banner_height), colour, 2)
            anchor = (left // 20, top // 20)
            used = anchor_uses.get(anchor, 0)
            anchor_uses[anchor] = used + 1
            cv2.putText(canvas, str(index), (left, min(canvas.shape[0] - 5, top + banner_height + 18 + used * 22)), font, 0.6, colour, 2)
        cv2.putText(canvas, f"{index:03d} | {detection.label} | {detection.confidence:.3f}", (legend_x, 158 + (index - 1) * 26), font, 0.56, colour, 1)
    if not detections:
        cv2.putText(canvas, "NO DETECTIONS at this threshold.", (legend_x, 160), font, 0.58, colour, 1)
        cv2.putText(canvas, "This does NOT mean the scene is safe.", (legend_x, 189), font, 0.54, (220, 220, 220), 1)
    elif len(detections) > len(shown):
        cv2.putText(canvas, "First 100 shown; all retained in JSON.", (legend_x, canvas.shape[0] - 18), font, 0.54, (220, 220, 220), 1)
    return canvas, {"scale": scale, "image_origin_xy_px": [0, banner_height], "labels_shown": len(shown), "width_px": canvas.shape[1], "height_px": canvas.shape[0]}


def _markdown(report: dict) -> str:
    lines = ["# Offline image check - DETECTION ONLY", "", DISCLAIMER, "",
             f"Direction: **{report['direction']}**, supplied by user; physical camera direction is unverified.", "",
             "## Provenance", "", f"- Image: `{report['image']['path']}`", f"- Image SHA-256: `{report['image']['sha256']}`",
             f"- Model: `{report['model']['path']}`", f"- Model SHA-256: `{report['model']['sha256']}`",
             f"- Annotation SHA-256: `{report['artifacts']['annotated.jpg']['sha256']}`",
             f"- Source size: {report['image']['width_px']} x {report['image']['height_px']} pixels.",
             f"- Explicit clockwise rotation: {report['transform']['rotate_clockwise_degrees']} degrees. No EXIF auto-rotation, mirroring or calibration.",
             "- Coordinates: rotated image pixels; the annotation is separately scaled/padded for readability.",
             "- Preprocessing: existing OpenCvYoloOnnxDetector; 640 x 640 stretch resize, 1/255 scaling, BGR to RGB (swapRB=True), crop=False; no letterbox.",
             f"- Threshold: {report['model']['confidence_threshold']}; OpenCV CPU threads: {report['model']['cpu_threads']}.",
             "- Adapter retains its supported COCO-label subset and class-aware NMS; this is not an all-class/raw-output dump.", "",
             "## Detections", ""]
    if not report["detections"]:
        lines.append("**No detections at the selected threshold. This does not establish absence of objects or a safe scene.**")
    else:
        lines.extend(["| ID | Model label | Confidence (not accuracy) | x, y, width, height (pixels) |", "| --- | --- | --- | --- |"])
        for detection in report["detections"]:
            box = detection["box_xywh_px"]
            lines.append(f"| {detection['id']} | {detection['label']} | {detection['confidence']:.6f} | {', '.join(f'{v:g}' for v in box)} |")
    lines.extend(["", "## Cross-label ambiguity evidence", "",
                  f"Criterion: different labels with box IoU >= {AMBIGUITY_IOU}. These pairs are retained, not automatically suppressed. "
                  "Overlap is evidence for human review, not proof of a false label or of two distinct objects.", ""])
    if report["ambiguities"]:
        for pair in report["ambiguities"]:
            lines.append(f"- Detections {pair['detection_ids'][0]} / {pair['detection_ids'][1]}: {' / '.join(pair['labels'])}; IoU={pair['iou']:.6f}.")
    else:
        lines.append("No pair met this criterion; that does not validate the remaining labels.")
    lines.extend(["", "## Limits", "",
                  "Accuracy, precision, recall, distance, TTC, risk, Pi latency and physical camera direction are unverified (JSON null). "
                  "No camera, GPIO, alerts, live configuration or network was used. A single image cannot validate tracking or motion. "
                  "Compare the visible boxes with the photograph manually; record ground-truth labels separately before computing accuracy.", "",
                  "Boxes are preserved as returned by the shared adapter; out-of-frame geometry and edge-touching boxes are flagged in JSON. "
                  "Edge-touching/extended boxes may be truncated; metric validity is not implied. "
                  "Only the rendered box edges are clipped. This diagnostic does not repair detector geometry or calibrate the camera.", ""])
    return "\n".join(lines)


def run_image_check(image_path: str | Path, model_path: str | Path, output_directory: str | Path, *, direction: str,
                    confidence: float = 0.45, cpu_threads: int = 1, rotate: int = 0) -> dict:
    """Run exactly one inference; only create files inside a new output directory."""
    if direction not in {"front", "rear"}:
        raise ValueError("direction must be front or rear.")
    if isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 < confidence <= 1:
        raise ValueError("confidence must be finite and in (0, 1].")
    if type(cpu_threads) is not int or not 1 <= cpu_threads <= 16:
        raise ValueError("cpu_threads must be an integer in 1..16.")
    if type(rotate) is not int or rotate not in {0, 90, 180, 270}:
        raise ValueError("rotate must be 0, 90, 180 or 270 clockwise degrees.")
    image_path, model_path, output = _validated_paths(image_path, model_path, output_directory)
    import cv2
    import numpy as np

    with image_path.open("rb") as handle:
        image_data = handle.read(MAX_IMAGE_BYTES + 1)
    if len(image_data) > MAX_IMAGE_BYTES:
        raise ValueError("Image grew beyond the input size limit while reading.")
    width, height = _image_dimensions(image_data)
    try:
        frame = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    except cv2.error as exc:
        raise ValueError("OpenCV could not decode the input image.") from exc
    if frame is None or frame.shape[:2] != (height, width):
        raise ValueError("OpenCV could not decode a JPEG/PNG with the declared dimensions.")
    if rotate:
        code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotate]
        frame = cv2.rotate(frame, code)
    model_hash = _sha256_file(model_path)
    try:
        detector = OpenCvYoloOnnxDetector(model_path, input_size_px=640, confidence_threshold=confidence, cpu_threads=cpu_threads)
        detections = detector.detect(frame)
    except cv2.error as exc:
        raise RuntimeError(f"OpenCV model loading/inference failed: {exc}") from exc
    _validate_detections(detections)
    if _sha256_file(model_path) != model_hash:
        raise RuntimeError("Model changed during inference; no report written.")
    ambiguities = _ambiguities(detections)
    annotated, rendering = _annotated_image(cv2, np, frame, detections, direction, len(ambiguities))
    ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("OpenCV could not encode the annotation.")
    annotation_bytes = encoded.tobytes()
    rotated_height, rotated_width = frame.shape[:2]
    report = {
        "schema_version": 1, "mode": "offline_image_detection_only", "disclaimer": DISCLAIMER,
        "direction": direction, "physical_direction_verified": None,
        "image": {"path": str(image_path), "sha256": hashlib.sha256(image_data).hexdigest(), "bytes": len(image_data), "width_px": width, "height_px": height},
        "model": {"path": str(model_path), "sha256": model_hash, "confidence_threshold": confidence, "cpu_threads": cpu_threads,
                  "adapter": "OpenCvYoloOnnxDetector", "input_size_px": 640,
                  "preprocessing": {"resize": "stretch", "scale": 1 / 255, "swapRB": True, "crop": False, "source_channel_order": "BGR"}},
        "transform": {"rotate_clockwise_degrees": rotate, "exif_auto_orientation": False, "mirrored": False,
                      "calibration_applied": False, "width_px": rotated_width, "height_px": rotated_height},
        "rendering": rendering,
        "detection_count": len(detections), "empty_detections": not detections,
        "detections": [{"id": index, "label": detection.label, "confidence": detection.confidence,
                        "box_xywh_px": [detection.x_px, detection.y_px, detection.width_px, detection.height_px],
                        "box_extends_outside_image": (detection.x_px < 0 or detection.y_px < 0 or detection.x_px + detection.width_px > rotated_width or detection.y_px + detection.height_px > rotated_height),
                        "box_may_be_truncated_at_image_edge": (detection.x_px <= 0 or detection.y_px <= 0 or detection.x_px + detection.width_px >= rotated_width or detection.y_px + detection.height_px >= rotated_height),
                        "distance_m": None, "ttc_s": None, "risk": None} for index, detection in enumerate(detections, 1)],
        "ambiguity_iou_threshold": AMBIGUITY_IOU, "ambiguities": ambiguities, "ambiguities_auto_suppressed": False,
        "metrics": {"accuracy": None, "precision": None, "recall": None, "distance_m": None, "ttc_s": None, "risk": None, "pi_latency_ms": None},
        "artifacts": {"annotated.jpg": {"sha256": hashlib.sha256(annotation_bytes).hexdigest()}},
        "opencv_version": cv2.__version__,
    }
    json_bytes = (json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    markdown_bytes = _markdown(report).encode("utf-8")
    output.mkdir(exist_ok=False)  # Atomic reservation; never merge into existing output.
    # Exclusive creates defend each artifact against accidental concurrent writes.
    # On an I/O failure, partial outputs are deliberately retained for diagnosis.
    for name, contents in (("annotated.jpg", annotation_bytes), ("detections.json", json_bytes), ("report.md", markdown_bytes)):
        with (output / name).open("xb") as handle:
            handle.write(contents)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline JPEG/PNG model evidence only: no distance, TTC, risk or camera access.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory; its parent must already exist.")
    parser.add_argument("--direction", required=True, choices=("front", "rear"))
    parser.add_argument("--confidence", type=float, default=0.45)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--rotate", type=int, choices=(0, 90, 180, 270), default=0, help="Explicit clockwise diagnostic rotation; not calibration.")
    args = parser.parse_args(argv)
    try:
        report = run_image_check(args.image, args.model, args.output_dir, direction=args.direction,
                                 confidence=args.confidence, cpu_threads=args.cpu_threads, rotate=args.rotate)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        parser.exit(2, f"image-check: {exc}\n")
    print(f"DETECTION ONLY: {report['detection_count']} detections; {len(report['ambiguities'])} ambiguity pairs. Outputs: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
