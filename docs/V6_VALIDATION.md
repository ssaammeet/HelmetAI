# V6 software correction validation — 2026-09-08

This records development-host checks, not Raspberry Pi acceptance.

## Environment and tests

- Host: Windows AMD64, Python 3.10.11, OpenCV 5.0.0, NumPy 2.2.6.
- Baseline: 80 unittest tests passed before this update.
- Updated suite: 130 tests passed, no skips in the development environment.
- Real packaged ONNX forward pass and JPEG-only profile smoke executed on the
  development host; the deployed model is unchanged, fixed 640 input.
- Bash syntax checked using Git Bash for installer, updater, preflight,
  benchmark, thread profiler and diagnostic runner. The updater was NOT run
  against a real Linux/systemd installation here.
- Runtime timing tests also run with Python `-S`: fake clocks, detector,
  camera rig and actuator; no camera, GPIO or OpenCV dependency in these tests.

## Regression evidence

Disjoint simulated work: front capture 5 ms + front inference 40 ms + rear
capture 10 ms + rear inference 50 ms = **105 ms**. The old nested timer produced
**165 ms** by counting the rear work again. The V6 regression asserts 105 ms,
15 ms capture, 90 ms inference, and 0 ms simulated postprocessing.

Additional tests cover:

- Logging stalls lower measured throughput without inflating compute latency.
- Empty scenes still emit one runtime heartbeat per completed loop.
- Actual front/rear update counts follow executed passes, not configured FPS.
- Teardown does not change the last completed decision's timing summary.
- Capture, detector, actuator and cleanup failures cannot report a completed run.
- Partial failures preserve partial directional evidence without committing an
  unsuccessful loop as a complete aggregate performance sample.
- Real measured directional rates override optimistic cadence estimates.
- All supported COCO labels, original argmax rules, box geometry, confidence
  boundaries and class-aware NMS are preserved by vectorized candidate filtering.
- JPEG profiling does not import camera/GPIO runtime, alter configuration or
  overwrite an existing report. Thread settings are restored after profiling.
- The updater preserves existing model, camera and calibration settings, stages
  only GPIO-disabled configuration, and provides a recoverable code/config backup.
- The benchmark rejects interrupted/short runs, missing/invalid direction
  metrics, stale/slow rear updates, non-finite values and failed health gates.

## Not established

The user previously ran the older package on Pi 5/aarch64/Trixie. That log
showed invalid/unverified calibration and insufficient processing rate. Its
aggregate timing contained the old double-counting defect. This is not a
post-fix Pi result.

V6 has not been physically run on that Pi, and no target FPS improvement,
metric distance accuracy, false-alarm rate or road safety result is promised.
Cooling/power investigation is deferred by the user, not solved by this patch.
Configuration thresholds have not been relaxed to manufacture a pass.

Independent GPIO stall watchdog and physical alert integration remain outside
this camera-only update. Keep outputs disabled and the service stopped until
the relevant controlled validation is complete. Successful unit tests or a
timing-only benchmark do not certify a road-ready system.
