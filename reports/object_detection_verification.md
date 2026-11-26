# Object Detection and Tracking — Verification Report

## Scope
Verification of `CodeAlpha_ObjectDetectionTracking` (CodeAlpha AI Internship — Task 4).

## Hardware & Runtime
- **CPU:** 11th Gen Intel(R) Core(TM) i3-1115G4 @ 3.00GHz (4 cores)
- **RAM:** 7.5 GiB total (~3.7 GiB available)
- **GPU/CUDA:** None (`CPUExecutionProvider` via `onnxruntime 1.32.0`)
- **Model:** Pretrained `yolov8n.onnx` (`12.24 MiB`, input `(1, 3, 640, 640)`, output `(1, 84, 8400)`)
- **Tracker:** SORT (`filterpy` Kalman filter + IoU association)

## Automated Unit Tests (Mocked)
Command:
```bash
./venv/bin/python -m pytest -q tests
```
Output:
```
......                                                                   [100%]
6 passed in 1.28s
```

## Real Inference & Multi-Frame Tracking Verification
- **Single-frame real inference (`bus.jpg`):**
  - Detections: `[('person', 0.89), ('person', 0.88), ('person', 0.88), ('bus', 0.84), ('person', 0.44)]`
- **Multi-frame tracking (`10-frame` panning video `bus_pan.mp4`):**
  - `Frame 0: 4 tracks, IDs=[1, 2, 3, 4], fps=7.14`
  - `Frame 4: 4 tracks, IDs=[1, 2, 3, 4], fps=6.72`
  - `Frame 9: 4 tracks, IDs=[1, 2, 3, 4], fps=9.23`
  - **Mean CPU FPS:** `7.59` (measured on Intel Core i3-1115G4, CPU-only)

## Lint & Formatting
Commands:
```bash
./venv/bin/ruff check .
./venv/bin/ruff format --check .
```
Outputs:
- `All checks passed!`
- `6 files already formatted`

## Application Startup Verification
- Headless Streamlit server (`streamlit run src/app.py --server.headless true --server.port 8503`) returned `HTTP 200`.
- `streamlit.testing.v1.AppTest` verified UI initialization with zero exceptions and no automatic webcam activation on startup.
