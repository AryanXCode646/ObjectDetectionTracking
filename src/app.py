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

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.onnx"


def main() -> None:
    st.set_page_config(page_title="Object Detection & Tracking", layout="wide")
    st.title("Object Detection & Tracking")
    st.caption(
        "CPU-compatible YOLOv8n (ONNX) object detection paired with SORT multi-frame tracking."
    )

    with st.sidebar:
        st.subheader("Controls")
        source_mode = st.radio(
            "Video source", ["Upload video", "Local file path", "Webcam (index 0)"]
        )
        conf_threshold = st.slider("Confidence threshold", 0.10, 0.90, 0.35, 0.05)
        max_frames = st.number_input("Max frames (0 = full video)", min_value=0, value=120, step=10)

    if not DEFAULT_MODEL_PATH.exists():
        st.error(
            f"Model weights not found at `{DEFAULT_MODEL_PATH}`. "
            "Follow README.md to download `yolov8n.onnx`."
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
    metrics_col1, metrics_col2 = st.columns(2)
    fps_slot = metrics_col1.empty()
    count_slot = metrics_col2.empty()

    detector = YOLOv8OnnxDetector(DEFAULT_MODEL_PATH, conf_threshold=conf_threshold)
    tracker = SortTracker()
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
