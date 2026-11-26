import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.detector import Detection, YOLOv8OnnxDetector, letterbox_image, parse_yolov8_output
from src.pipeline import draw_tracks, iter_tracked_frames
from src.tracker import SortTracker


class TestDetectorAndTracker(unittest.TestCase):
    def test_missing_model_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            YOLOv8OnnxDetector(Path("/nonexistent/yolov8n.onnx"))

    def test_empty_frame_raises(self) -> None:
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        detector = YOLOv8OnnxDetector("unused.onnx", session=mock_session)
        with self.assertRaises(ValueError):
            detector.detect(np.array([], dtype=np.uint8))

    def test_letterbox_and_parse_yolov8_output(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, scale, pad = letterbox_image(frame, (640, 640))
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 1.0)

        # Build synthetic (1, 84, 2) output with 1 confident 'person' detection at center
        raw = np.zeros((1, 84, 2), dtype=np.float32)
        raw[0, 0, 0] = 320.0  # cx
        raw[0, 1, 0] = 320.0  # cy
        raw[0, 2, 0] = 100.0  # w
        raw[0, 3, 0] = 200.0  # h
        raw[0, 4, 0] = 0.90  # class 0 ('person')

        detections = parse_yolov8_output(raw, (480, 640), scale, pad, conf_threshold=0.5)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "person")
        self.assertAlmostEqual(detections[0].confidence, 0.90, places=2)

    def test_sort_tracker_persists_ids_and_handles_empty_detections(self) -> None:
        tracker = SortTracker(max_age=2, min_hits=1, iou_threshold=0.3)

        det1 = [
            Detection(bbox=(100.0, 100.0, 200.0, 250.0), confidence=0.9, class_id=0, label="person")
        ]
        tracks1 = tracker.update(det1)
        self.assertEqual(len(tracks1), 1)
        first_id = tracks1[0].track_id

        # Slightly shifted box in next frame should keep same ID
        det2 = [
            Detection(
                bbox=(106.0, 102.0, 206.0, 252.0), confidence=0.88, class_id=0, label="person"
            )
        ]
        tracks2 = tracker.update(det2)
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].track_id, first_id)

        # Empty detections for > max_age frames should drop the track
        for _ in range(4):
            tracks_empty = tracker.update([])
            self.assertEqual(tracks_empty, [])

        # New detection after expiry receives a new ID
        tracks_new = tracker.update(det1)
        self.assertEqual(len(tracks_new), 1)
        self.assertNotEqual(tracks_new[0].track_id, first_id)

    def test_draw_tracks_output_shape(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        tracker = SortTracker()
        tracks = tracker.update(
            [Detection(bbox=(20.0, 20.0, 80.0, 100.0), confidence=0.85, class_id=2, label="car")]
        )
        annotated = draw_tracks(frame, tracks, fps=12.5)
        self.assertEqual(annotated.shape, frame.shape)

    def test_invalid_video_path_and_resource_cleanup(self) -> None:
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        detector = YOLOv8OnnxDetector("unused.onnx", session=mock_session)
        tracker = SortTracker()

        with self.assertRaises(FileNotFoundError):
            list(iter_tracked_frames(Path("/nonexistent/video.mp4"), detector, tracker))

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        with patch("src.pipeline.cv2.VideoCapture", return_value=mock_cap):
            frames = list(iter_tracked_frames(0, detector, tracker))
        self.assertEqual(frames, [])
        mock_cap.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
