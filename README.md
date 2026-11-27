# Object Detection and Tracking

A CPU-compatible computer vision application that performs object detection with pretrained YOLO ONNX models and multi-frame object tracking using SORT.

## Problem Statement

Object detection identifies objects independently in every frame without knowing whether an object in frame *t* is the same one from frame *t−1*. This project combines pretrained YOLO detection with a SORT Kalman-filter tracker to assign persistent tracking IDs across video frames.

## Features Implemented

- **Multi-model detection:** YOLO26n (default), YOLOv8n (legacy), YOLOv8s (high accuracy) via `onnxruntime` (`CPUExecutionProvider`)
- **Correct preprocessing:** Letterbox resizing with integer pad offsets, BGR→RGB conversion, [0,1] normalization
- **Class-aware NMS:** Prevents overlapping objects of different classes from suppressing each other
- **Automatic output layout dispatch:** Supports both YOLOv8 `(1, 84, N)` and YOLO26 end-to-end `(1, N, 6)` ONNX output formats
- **SORT multi-object tracking:** `filterpy` Kalman filters, **Hungarian (optimal) IoU assignment** via `scipy.optimize.linear_sum_assignment`, **class-aware matching** preventing cross-class ID mutations
- **Detection mode presets:** Balanced (~12 FPS), Accuracy (~9 FPS), High Accuracy (~3.5 FPS), Legacy
- **Annotated output:** Bounding boxes, class labels, confidence scores, tracking IDs, FPS/object-count overlay
- **Streamlit UI** (`src/app.py`) with model selector, threshold sliders, input-size control, and detection mode presets
- **OpenCV CLI** (`src/pipeline.py`) with configurable `--model`, `--conf`, `--iou`, `--input-size`
- Webcam capture is never auto-started; it requires explicit user action

## Model Selection and Evaluation

| Model | mAP@50 | FPS (i3-1115G4) | Size | Decision |
|:---|:---:|:---:|:---:|:---|
| YOLOv8n (baseline) | 0.404 | 8.9 | 12.24 MiB | Retained as legacy mode |
| **YOLO26n** (default) | **0.490** | **11.7** | **10.18 MiB** | **Selected: +21% mAP, +31% faster, smaller** |
| YOLOv8s | 0.664 | 3.6 | 42.79 MiB | High-accuracy mode |

All metrics were measured on the COCO128 test split (64 images, 506 annotated objects) using identical evaluation code. See `reports/accuracy_comparison.md` for threshold calibration, resolution experiments, and detailed results.

> **Note:** Confidence scores displayed in the application are the model's raw output scores and are **not** calibrated probabilities that a detection is correct.

## Technologies Used

- Python 3.14
- OpenCV (`opencv-python`)
- ONNX Runtime (`onnxruntime`)
- `filterpy`, NumPy, SciPy
- Streamlit
- pytest & Ruff

## Project Structure

```text
ObjectDetectionTracking/
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── models/                  # Gitignored; place ONNX models here
│   ├── yolo26n.onnx         # Default model (10.18 MiB)
│   ├── yolov8n.onnx         # Legacy baseline (12.24 MiB)
│   └── yolov8s.onnx         # High-accuracy mode (42.79 MiB)
├── reports/
│   ├── accuracy_baseline.md
│   ├── accuracy_comparison.md
│   ├── accuracy_verification.md
│   └── object_detection_verification.md
├── src/
│   ├── app.py               # Streamlit UI
│   ├── coco_labels.py       # 80 COCO class names
│   ├── detector.py          # YOLO ONNX inference + preprocessing
│   ├── pipeline.py          # Video processing + CLI
│   └── tracker.py           # SORT tracker (Hungarian + class-aware)
└── tests/
    └── test_detection_tracking.py
```

## Installation & Model Download

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir -p models

# Default model (YOLO26n — recommended)
pip install ultralytics --quiet
yolo export model=yolo26n.pt format=onnx dynamic=True simplify=True
mv yolo26n.onnx models/

# Legacy baseline (YOLOv8n)
curl -L -o models/yolov8n.onnx https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx

# Optional high-accuracy model (YOLOv8s)
yolo export model=yolov8s.pt format=onnx simplify=True
mv yolov8s.onnx models/
```

## Running the Application

**Streamlit UI:**

```bash
./venv/bin/streamlit run src/app.py
```

**OpenCV CLI:**

```bash
# Default (YOLO26n)
./venv/bin/python -m src.pipeline --source /path/to/video.mp4

# High-accuracy mode
./venv/bin/python -m src.pipeline --source /path/to/video.mp4 --model models/yolov8s.onnx --conf 0.25 --iou 0.50

# Webcam (starts only when explicitly invoked)
./venv/bin/python -m src.pipeline --webcam
```

## Running Tests & Lint Checks

```bash
./venv/bin/python -m pytest -q tests
./venv/bin/ruff check .
./venv/bin/ruff format --check .
```

## Algorithm Overview

### Object Detection (YOLO → ONNX Runtime)

1. **Letterbox:** Resize input while preserving aspect ratio, then pad to model input size with (114, 114, 114).
2. **Preprocess:** BGR→RGB, /255.0 normalization, NCHW transpose.
3. **Inference:** ONNX Runtime `CPUExecutionProvider` with `ORT_ENABLE_ALL` graph optimization.
4. **Decode:** Automatically dispatch between YOLOv8 `(1, 84, N)` and YOLO26 `(1, N, 6)` output formats.
5. **Post-process:** Class-aware NMS, unscale and clip boxes to original image coordinates.

### Object Tracking (SORT + Hungarian)

1. **Predict:** Kalman filter predicts each track's bounding box for the current frame.
2. **Associate:** Compute pairwise IoU between predictions and new detections, with class-aware gating (different-class pairs get IoU=0). Solve optimal assignment with `scipy.optimize.linear_sum_assignment`.
3. **Update/Create/Delete:** Matched tracks are updated; unmatched detections create new tracks; tracks unseen for `max_age` frames are deleted.

## Hardware & Limitations

- Verified on Intel Core i3-1115G4 (CPU-only, 7.5 GiB RAM).
- **Balanced mode:** ~11.7 FPS, mAP@50=0.490 (YOLO26n 640px, 2 threads).
- **Accuracy mode:** ~8.6 FPS, mAP@50=0.501 (YOLO26n 768px).
- **High accuracy mode:** ~3.6 FPS, mAP@50=0.664 (YOLOv8s 640px).
- SORT relies on motion + IoU (no appearance embeddings), so ID switches can occur during prolonged occlusion.
- Tracking metrics (HOTA, IDF1, MOTA) are not evaluated due to the absence of ground-truth track-ID annotations.
- Peak RSS is bounded at ~359 MiB across all configurations.

## Future Improvements

- Add appearance embeddings (Deep SORT / BoT-SORT) for better re-identification after occlusion.
- Evaluate on a larger labelled dataset with ground-truth track IDs.
- Support saving annotated output videos directly from the Streamlit UI.

## Third-Party Attribution & License

- YOLO26n and YOLOv8 architectures by Ultralytics (exported to ONNX), AGPL-3.0 licensed.
- SORT tracking concept based on Bewley et al. (2016), using `filterpy`.
- Released under the MIT License.
