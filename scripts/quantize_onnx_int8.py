"""Create an optional INT8 ONNX copy for edge-device benchmarking.

This tool is intentionally not part of the Pi runtime.  A quantized detector
must be validated for accuracy and OpenCV-DNN compatibility on the exact
target device before it replaces the default FP model in a configuration.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Create an INT8 ONNX model for HelmetAI benchmark use.")
    parser.add_argument("--input", required=True, help="source FP ONNX path")
    parser.add_argument("--output", required=True, help="target INT8 ONNX path")
    parser.add_argument(
        "--validate-opencv",
        action="store_true",
        help="run one blank-frame OpenCV-DNN inference on the generated model",
    )
    parser.add_argument("--input-size", type=int, default=640, help="square inference input for OpenCV validation")
    args = parser.parse_args()

    source = Path(args.input)
    target = Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(f"Input ONNX model was not found: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required. Install requirements-optimization.txt on a development computer first."
        ) from exc
    quantize_dynamic(str(source), str(target), weight_type=QuantType.QInt8)
    print(f"quantized_model={target}")
    if args.validate_opencv:
        if args.input_size <= 0:
            raise ValueError("--input-size must be positive.")
        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore

            network = cv2.dnn.readNetFromONNX(str(target))
            blank = np.zeros((args.input_size, args.input_size, 3), dtype=np.uint8)
            network.setInput(cv2.dnn.blobFromImage(blank, 1 / 255.0, (args.input_size, args.input_size), swapRB=True))
            network.forward()
        except Exception as exc:
            print(f"opencv_validation=FAILED: {type(exc).__name__}: {exc}")
            print("Do not select this model in the OpenCV live runtime.")
            return 2
        print("opencv_validation=PASSED")
    print("Benchmark this model on the target Pi and validate detections before using it in a live configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
