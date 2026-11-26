# Object Detection & Tracking — Accuracy Verification Report

## 1. Environment
- **CPU:** 11th Gen Intel(R) Core(TM) i3-1115G4 @ 3.00GHz (2 cores, 4 threads)
- **RAM:** 7.5 GiB total
- **GPU/CUDA:** None (`CPUExecutionProvider` via `onnxruntime 1.32.0`)
- **Python:** 3.14
- **Libraries:** `opencv-python 5.1.0`, `onnxruntime 1.32.0`, `numpy 2.5.5`, `filterpy`, `scipy 1.18.1`, `streamlit 1.67.0`

## 2. Models Evaluated

| Model | File | Input Shape | Output Shape | Size |
|:---|:---|:---|:---|:---|
| YOLOv8n (baseline) | `models/yolov8n.onnx` | `[1, 3, 640, 640]` (static) | `[1, 84, 8400]` | 12.24 MiB |
| YOLO26n (default) | `models/yolo26n.onnx` | `[batch, 3, height, width]` (dynamic) | `[batch, 84, anchors]` | 10.57 MiB |
| YOLOv8s | `models/yolov8s.onnx` | `[1, 3, 640, 640]` (static) | `[1, 84, 8400]` | 42.79 MiB |

## 3. Evaluation Dataset
- **Source:** COCO128 (128 images from COCO val2017 with YOLO-format ground-truth annotations)
- **Val split:** 64 images (even-indexed, used for threshold calibration only)
- **Test split:** 64 images (odd-indexed, used for final model comparison)
- Splits are deterministic and reproducible.

## 4. Unit Tests
```
$ ./venv/bin/python -m pytest -q
............................                                             [100%]
28 passed in 1.38s
```

## 5. Lint & Formatting
```
$ ./venv/bin/ruff check .
All checks passed!

$ ./venv/bin/ruff format --check .
9 files already formatted
```

## 6. Streamlit Startup
```
$ ./venv/bin/streamlit run src/app.py --server.headless true --server.port 8505
# HTTP 200 confirmed
```

## 7. Tracking Metrics Status
**NOT VERIFIED.** No ground-truth track-ID annotations are available. The SORT improvements (Hungarian assignment, class-aware gating, max_age=15) are structural algorithm improvements whose quantitative tracking impact (HOTA, IDF1, MOTA, ID switches) cannot be measured without labelled tracking sequences.
