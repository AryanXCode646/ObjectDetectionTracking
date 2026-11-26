# Object Detection & Tracking — Baseline Audit (`reports/accuracy_baseline.md`)

## 1. Hardware & Environment
- **CPU:** 11th Gen Intel(R) Core(TM) i3-1115G4 @ 3.00GHz (2 physical cores, 4 threads)
- **RAM:** 7.5 GiB total (~3.0 GiB available)
- **GPU/CUDA:** None (CPU-only)
- **Runtime:** Python 3.14, `onnxruntime 1.32.0` (`CPUExecutionProvider`), `opencv-python 5.1.0`, `numpy 2.5.5`, `filterpy`, `streamlit 1.67.0`

## 2. Baseline Model Inspection (`models/yolov8n.onnx`)
- **File size:** `12.24 MiB`
- **Input tensor:** `images`, shape `[1, 3, 640, 640]` (static shape — non-640 resolutions fail unless re-exported with dynamic shapes or matching size), `tensor(float)`
- **Output tensor:** `predictions`, shape `[1, 84, 8400]`, `tensor(float)`
- **Baseline tests:** `pytest -q` passes `6 passed in 1.43s`; `ruff check .` passes.

## 3. Code Audit Findings

### Detection Pipeline (`src/detector.py`)
1. **Static Input Shape (`640x640`):** `yolov8n.onnx` only accepts `[1, 3, 640, 640]`; passing `768` or `832` raises an ONNX Runtime shape mismatch.
2. **Letterbox Offset Precision (`letterbox_image`):** When `target_size - new_size` is odd (e.g., `1 px`), `left = round(pad_w - 0.1) = 0` and `right = 1`, yet `letterbox_image` returns `pad_w = 0.5` instead of `float(left) = 0.0`, introducing a 0.5 px coordinate shift.
3. **Class-Agnostic NMS (`parse_yolov8_output`):** `cv2.dnn.NMSBoxes` is called without class offsets, so overlapping objects of **different** classes (e.g., `person` on `bicycle`, `person` carrying `backpack`, `person` in front of `bus`) can suppress each other, causing false negatives.
4. **Single Output Layout:** `parse_yolov8_output` only supports `(1, 84, N)` and cannot decode end-to-end `(1, N, 6)` ONNX models such as `YOLO26n` (`[x1, y1, x2, y2, conf, class_id]`).
5. **Default Confidence Threshold (`0.35`):** Set without labelled validation evidence.

### Tracking Pipeline (`src/tracker.py`)
1. **No Class-Aware Association:** `SortTracker.update()` computes IoU across all classes indiscriminately, allowing a `person` track to match an overlapping `car` or `bicycle` detection and mutate its `class_id`/`label`.
2. **Greedy Association:** Uses greedy highest-IoU matching rather than optimal bipartite matching (`scipy.optimize.linear_sum_assignment`).
3. **`min_hits=1`:** Every single-frame false positive immediately emits a new `track_id`, increasing ID churn.
4. **Motion-Only Association:** SORT uses constant-velocity Kalman filtering + IoU without appearance embeddings, so rapid camera motion or prolonged occlusion can cause ID switches.

### Application & Pipeline (`src/app.py`, `src/pipeline.py`)
- No model selector, input-size selector, NMS IoU threshold slider, or balanced/accuracy mode controls.
- Webcam capture is safely user-triggered (`Start` button) and OpenCV `VideoCapture` is released in `finally`.
