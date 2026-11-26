import sys
import tempfile
from pathlib import Path

import cv2
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import YOLOv8OnnxDetector
from src.pipeline import iter_tracked_frames
from src.tracker import SortTracker

MODELS_DIR = PROJECT_ROOT / "models"

# Detection mode presets: (model_file, input_size, conf, iou, threads, description)
DETECTION_MODES = {
    "Balanced (YOLO26n 640)": {
        "model": "yolo26n.onnx",
        "input_size": 640,
        "conf": 0.25,
        "iou": 0.50,
        "threads": 2,
        "desc": "Best speed/accuracy trade-off (~11 FPS, mAP50≈0.49).",
    },
    "Accuracy (YOLO26n 768)": {
        "model": "yolo26n.onnx",
        "input_size": 768,
        "conf": 0.25,
        "iou": 0.50,
        "threads": None,
        "desc": "Higher resolution for small objects (~8 FPS, mAP50≈0.50).",
    },
    "High Accuracy (YOLOv8s 640)": {
        "model": "yolov8s.onnx",
        "input_size": 640,
        "conf": 0.25,
        "iou": 0.50,
        "threads": None,
        "desc": "Larger model, highest detection accuracy (~3.5 FPS, mAP50≈0.66).",
    },
    "Legacy (YOLOv8n 640)": {
        "model": "yolov8n.onnx",
        "input_size": 640,
        "conf": 0.25,
        "iou": 0.50,
        "threads": None,
        "desc": "Original baseline model (~8 FPS, mAP50≈0.46).",
    },
}


def _resolve_model_path(model_filename: str) -> Path | None:
    path = MODELS_DIR / model_filename
    return path if path.exists() else None


def main() -> None:
    st.set_page_config(page_title="Object Detection & Tracking", layout="wide")
    st.title("Object Detection & Tracking")
    st.caption(
        "CPU-compatible YOLO (ONNX) object detection paired with SORT multi-frame tracking. "
        "Confidence scores indicate the model's output probability for each detected class "
        "and are **not** calibrated probabilities that a detection is correct."
    )

    with st.sidebar:
        st.subheader("Detection Mode")
        mode_names = list(DETECTION_MODES.keys())
        selected_mode = st.selectbox("Preset", mode_names, index=0)
        preset = DETECTION_MODES[selected_mode]
        st.caption(preset["desc"])

        st.divider()
        st.subheader("Advanced Controls")
        conf_threshold = st.slider("Confidence threshold", 0.05, 0.90, float(preset["conf"]), 0.05)
        iou_threshold = st.slider("NMS IoU threshold", 0.20, 0.80, float(preset["iou"]), 0.05)
        input_size = st.select_slider(
            "Input size", options=[416, 480, 544, 640, 768, 832], value=preset["input_size"]
        )

        st.divider()
        st.subheader("Video Source")
        source_mode = st.radio("Source", ["Upload video", "Local file path", "Webcam (index 0)"])
        max_frames = st.number_input("Max frames (0 = full video)", min_value=0, value=120, step=10)

    model_path = _resolve_model_path(preset["model"])
    if model_path is None:
        st.error(
            f"Model weights not found at `{MODELS_DIR / preset['model']}`. "
            "See README.md for model download instructions."
        )
        return

    video_source: int | Path | None = None
    temp_path: Path | None = None

    if source_mode == "Upload video":
        uploaded = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uploaded.name).suffix
            ) as tmp:
                tmp.write(uploaded.read())
                temp_path = Path(tmp.name)
            video_source = temp_path
    elif source_mode == "Local file path":
        raw_path = st.text_input("Video file path", placeholder="/path/to/video.mp4")
        if raw_path.strip():
            video_source = Path(raw_path.strip())
    else:
        st.info("Webcam capture only starts when you click Start.")
        video_source = 0

    col_start, col_stop = st.columns([1, 5])
    with col_start:
        start_clicked = st.button("Start", type="primary")
    with col_stop:
        stop_clicked = st.button("Stop")

    if stop_clicked:
        st.session_state.running = False

    if not start_clicked:
        if video_source is None:
            st.info("Select a video source and click Start to run detection and tracking.")
        return

    if video_source is None:
        st.warning("Please provide a valid video file first.")
        return

    frame_slot = st.empty()
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    fps_slot = metrics_col1.empty()
    count_slot = metrics_col2.empty()
    mode_slot = metrics_col3.empty()
    mode_slot.metric("Mode", selected_mode.split("(")[0].strip())

    detector = YOLOv8OnnxDetector(
        model_path,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        input_size=(input_size, input_size),
        class_aware_nms=True,
        intra_op_num_threads=preset.get("threads"),
    )
    tracker = SortTracker(max_age=15, min_hits=1, iou_threshold=0.3, class_aware=True)
    limit = int(max_frames) if max_frames > 0 else None

    try:
        for annotated, tracks, fps in iter_tracked_frames(
            video_source, detector, tracker, max_frames=limit
        ):
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_slot.image(rgb_frame, channels="RGB", use_container_width=True)
            fps_slot.metric("Processing FPS", f"{fps:.1f}")
            count_slot.metric("Tracked objects", len(tracks))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        st.error(f"Video processing error: {exc}")
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
