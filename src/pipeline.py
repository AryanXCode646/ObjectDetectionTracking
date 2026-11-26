import argparse
import time
from collections.abc import Generator
from pathlib import Path

import cv2
import numpy as np

from src.detector import YOLOv8OnnxDetector
from src.tracker import SortTracker, TrackedObject


def draw_tracks(
    frame: np.ndarray,
    tracks: list[TrackedObject],
    fps: float | None = None,
) -> np.ndarray:
    """Draw bounding boxes, class labels, confidence scores, and tracking IDs."""
    annotated = frame.copy()
    for obj in tracks:
        x1, y1, x2, y2 = [round(v) for v in obj.bbox]
        color = ((obj.track_id * 67) % 200 + 55, (obj.track_id * 131) % 200 + 55, 80)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f"ID {obj.track_id}: {obj.label} ({obj.confidence:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_y = max(y1 - 8, th + 6)
        cv2.rectangle(annotated, (x1, text_y - th - 6), (x1 + tw + 6, text_y + 2), color, -1)
        cv2.putText(
            annotated,
            text,
            (x1 + 3, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )

    hud = f"Objects: {len(tracks)}"
    if fps is not None:
        hud = f"FPS: {fps:.1f} | {hud}"
    cv2.putText(
        annotated,
        hud,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 128),
        2,
        cv2.LINE_AA,
    )
    return annotated


def iter_tracked_frames(
    source: int | str | Path,
    detector: YOLOv8OnnxDetector,
    tracker: SortTracker,
    max_frames: int | None = None,
) -> Generator[tuple[np.ndarray, list[TrackedObject], float], None, None]:
    """Stream frames from a video file or camera index with detection and tracking."""
    if isinstance(source, Path):
        if not source.exists():
            raise FileNotFoundError(f"Video file not found: {source}")
        cap_source: int | str = str(source)
    elif isinstance(source, str):
        if not Path(source).exists():
            raise FileNotFoundError(f"Video file not found: {source}")
        cap_source = source
    else:
        cap_source = source

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video source: {source}")

    frame_count = 0
    try:
        while True:
            if max_frames is not None and frame_count >= max_frames:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            start = time.perf_counter()
            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            elapsed = max(time.perf_counter() - start, 1e-6)
            fps = 1.0 / elapsed

            annotated = draw_tracks(frame, tracks, fps=fps)
            frame_count += 1
            yield annotated, tracks, fps
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO ONNX + SORT Object Detection and Tracking")
    parser.add_argument("--source", type=str, default=None, help="Path to local video file")
    parser.add_argument("--webcam", action="store_true", help="Use default webcam (index 0)")
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo26n.onnx",
        help="Path to YOLO ONNX model",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    parser.add_argument(
        "--input-size", type=int, default=640, help="Input resolution (single side)"
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit")
    args = parser.parse_args()

    if not args.webcam and not args.source:
        parser.error("Specify --source <video_path> or --webcam.")

    source: int | str = 0 if args.webcam else str(args.source)
    detector = YOLOv8OnnxDetector(
        args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        input_size=(args.input_size, args.input_size),
        class_aware_nms=True,
    )
    tracker = SortTracker(max_age=15, min_hits=1, iou_threshold=0.3, class_aware=True)

    try:
        for annotated, _tracks, _fps in iter_tracked_frames(
            source, detector, tracker, max_frames=args.max_frames
        ):
            cv2.imshow("CodeAlpha Object Detection & Tracking", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
