# CodeAlpha Object Detection and Tracking

**CodeAlpha AI Internship — Task 4**

A CPU-compatible computer vision application that performs object detection with a pretrained YOLOv8n ONNX model and multi-frame object tracking using SORT.

## Problem Statement

Object detection alone identifies objects independently on every frame without knowing whether an object in frame *t* is the same one from frame *t-1*. This project combines pretrained YOLOv8n detection with a SORT Kalman-filter tracker to assign persistent tracking IDs across video frames.

## Features Implemented

- Pretrained YOLOv8n inference via `onnxruntime` (`CPUExecutionProvider`)
- Letterbox resizing, confidence filtering, and OpenCV NMS post-processing
- SORT multi-object tracking using `filterpy` Kalman filters and IoU association
- Annotated video output with bounding boxes, class labels, confidence scores, tracking IDs, and FPS/object-count overlay
- Support for local video files and explicit user-triggered webcam capture (never auto-started on launch)
- Streamlit UI (`src/app.py`) and OpenCV CLI (`src/pipeline.py`)

## Technologies Used

- Python 3.14
- OpenCV (`opencv-python`)
- ONNX Runtime (`onnxruntime`)
- `filterpy` & NumPy
- Streamlit
- pytest & Ruff

## Project Structure

```text
CodeAlpha_ObjectDetectionTracking/
├── .gitignore
├── pyproject.toml
├── requirements.txt
├── models/                  # Gitignored; place yolov8n.onnx here
├── reports/
│   └── object_detection_verification.md
├── src/
│   ├── app.py
│   ├── coco_labels.py
│   ├── detector.py
│   ├── pipeline.py
│   └── tracker.py
└── tests/
    └── test_detection_tracking.py
```

## Installation & Model Download

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

mkdir -p models
curl -L -o models/yolov8n.onnx https://huggingface.co/Kalray/yolov8/resolve/main/yolov8n.onnx
```

## Running the Application

**Streamlit UI:**

```bash
./venv/bin/streamlit run src/app.py
```

**OpenCV CLI:**

```bash
./venv/bin/python -m src.pipeline --source /path/to/video.mp4
# Or for webcam (starts only when invoked):
./venv/bin/python -m src.pipeline --webcam
```

## Running Tests & Lint Checks

```bash
./venv/bin/python -m pytest -q tests
./venv/bin/ruff check .
./venv/bin/ruff format --check .
```

## Algorithm Overview (Detection vs. Tracking)

- **Object Detection (YOLOv8n):** Processes each frame independently to predict bounding boxes, COCO class probabilities, and confidence scores. The pretrained weights are used directly without retraining.
- **Object Tracking (SORT):** Uses a constant-velocity Kalman filter to predict each existing track's next bounding box, computes pairwise IoU between predictions and new detections, and assigns persistent `track_id` values across frames. Tracks that disappear for more than `max_age` frames are removed.

## Hardware & Limitations

- Verified on an Intel Core i3-1115G4 (CPU-only, ~7.6 FPS on `640x480`-scale frames during a 10-frame panning test).
- SORT relies purely on motion and box overlap (IoU), so ID switches can occur during prolonged occlusion or rapid camera movement.

## Future Improvements

- Add appearance embeddings (Deep SORT) for better re-identification after occlusion.
- Support saving annotated output videos directly from the Streamlit UI.

## Third-Party Attribution & License

- Pretrained YOLOv8n architecture by Ultralytics (exported to ONNX).
- SORT tracking concept based on Bewley et al. (2016), using `filterpy`.
- Released under the MIT License.
