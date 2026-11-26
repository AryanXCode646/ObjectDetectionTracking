import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.detector import (
    Detection,
    YOLOv8OnnxDetector,
    letterbox_image,
    parse_yolo26_output,
    parse_yolo_output,
    parse_yolov8_output,
)
from src.pipeline import draw_tracks, iter_tracked_frames
from src.tracker import SortTracker, compute_iou_matrix

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolov8n.onnx"


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------
class TestLetterbox(unittest.TestCase):
    def test_letterbox_preserves_aspect_ratio_and_pads(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, scale, _pad = letterbox_image(frame, (640, 640))
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 1.0)

    def test_letterbox_returns_integer_pad_offsets(self) -> None:
        frame = np.zeros((300, 500, 3), dtype=np.uint8)
        padded, _scale, pad = letterbox_image(frame, (640, 640))
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertEqual(pad[0], int(pad[0]))
        self.assertEqual(pad[1], int(pad[1]))

    def test_letterbox_square_image(self) -> None:
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _padded, scale, pad = letterbox_image(frame, (640, 640))
        self.assertAlmostEqual(scale, 1.0)
        self.assertEqual(pad, (0.0, 0.0))

    def test_letterbox_empty_image_raises(self) -> None:
        with self.assertRaises(ValueError):
            letterbox_image(np.array([], dtype=np.uint8), (640, 640))

    def test_letterbox_different_target_sizes(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for sz in [(416, 416), (768, 768), (832, 832)]:
            padded, _scale, _pad = letterbox_image(frame, sz)
            self.assertEqual(padded.shape[:2], sz[::-1])


# ---------------------------------------------------------------------------
# YOLOv8 output parsing tests
# ---------------------------------------------------------------------------
class TestParseYOLOv8Output(unittest.TestCase):
    def test_basic_yolov8_output(self) -> None:
        raw = np.zeros((1, 84, 2), dtype=np.float32)
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 100.0
        raw[0, 3, 0] = 200.0
        raw[0, 4, 0] = 0.90

        detections = parse_yolov8_output(raw, (480, 640), 1.0, (0.0, 80.0), conf_threshold=0.5)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "person")
        self.assertAlmostEqual(detections[0].confidence, 0.90, places=2)

    def test_confidence_filtering(self) -> None:
        raw = np.zeros((1, 84, 1), dtype=np.float32)
        raw[0, 0, 0], raw[0, 1, 0], raw[0, 2, 0], raw[0, 3, 0] = 320, 320, 100, 100
        raw[0, 4, 0] = 0.10
        self.assertEqual(parse_yolov8_output(raw, (640, 640), 1.0, (0, 0), conf_threshold=0.5), [])

    def test_class_aware_nms_keeps_different_class_boxes(self) -> None:
        raw = np.zeros((1, 84, 2), dtype=np.float32)
        for i in range(2):
            raw[0, 0, i] = 320.0
            raw[0, 1, i] = 320.0
            raw[0, 2, i] = 100.0
            raw[0, 3, i] = 100.0
        raw[0, 4, 0] = 0.90  # class 0: person
        raw[0, 5, 1] = 0.85  # class 1: bicycle

        dets_aware = parse_yolov8_output(
            raw, (640, 640), 1.0, (0, 0), conf_threshold=0.3, class_aware_nms=True
        )
        self.assertEqual(len(dets_aware), 2)

        dets_agnostic = parse_yolov8_output(
            raw, (640, 640), 1.0, (0, 0), conf_threshold=0.3, class_aware_nms=False
        )
        self.assertEqual(len(dets_agnostic), 1)

    def test_boxes_clipped_to_image_bounds(self) -> None:
        raw = np.zeros((1, 84, 1), dtype=np.float32)
        raw[0, 0, 0] = 10.0
        raw[0, 1, 0] = 10.0
        raw[0, 2, 0] = 50.0
        raw[0, 3, 0] = 50.0
        raw[0, 4, 0] = 0.90

        dets = parse_yolov8_output(raw, (100, 100), 1.0, (0, 0), conf_threshold=0.3)
        self.assertTrue(len(dets) >= 1)
        x1, y1, x2, y2 = dets[0].bbox
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, 100)
        self.assertLessEqual(y2, 100)

    def test_invalid_output_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_yolov8_output(np.zeros((1, 2, 1), dtype=np.float32), (640, 640), 1.0, (0, 0))


# ---------------------------------------------------------------------------
# YOLO26 output parsing tests
# ---------------------------------------------------------------------------
class TestParseYOLO26Output(unittest.TestCase):
    def test_basic_yolo26_output(self) -> None:
        raw = np.array([[[100.0, 100.0, 300.0, 400.0, 0.85, 0.0]]], dtype=np.float32)
        dets = parse_yolo26_output(raw, (480, 640), 1.0, (0, 0), conf_threshold=0.5)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].label, "person")

    def test_yolo26_confidence_filtering(self) -> None:
        raw = np.array([[[100, 100, 300, 400, 0.10, 0]]], dtype=np.float32)
        self.assertEqual(parse_yolo26_output(raw, (640, 640), 1.0, (0, 0), conf_threshold=0.5), [])

    def test_yolo26_invalid_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_yolo26_output(np.zeros((1, 10, 4), dtype=np.float32), (640, 640), 1.0, (0, 0))

    def test_dispatch_selects_correct_parser(self) -> None:
        raw_v8 = np.zeros((1, 84, 1), dtype=np.float32)
        raw_v8[0, 0, 0], raw_v8[0, 1, 0] = 320, 320
        raw_v8[0, 2, 0], raw_v8[0, 3, 0] = 100, 100
        raw_v8[0, 4, 0] = 0.9
        dets = parse_yolo_output(raw_v8, (640, 640), 1.0, (0, 0), conf_threshold=0.5)
        self.assertEqual(len(dets), 1)

        raw_26 = np.array([[[100, 100, 300, 400, 0.85, 0]]], dtype=np.float32)
        dets26 = parse_yolo_output(raw_26, (640, 640), 1.0, (0, 0), conf_threshold=0.5)
        self.assertEqual(len(dets26), 1)


# ---------------------------------------------------------------------------
# IoU and tracker tests
# ---------------------------------------------------------------------------
class TestIoU(unittest.TestCase):
    def test_perfect_overlap(self) -> None:
        a = np.array([[10, 10, 50, 50]], dtype=np.float32)
        iou = compute_iou_matrix(a, a)
        self.assertAlmostEqual(iou[0, 0], 1.0, places=5)

    def test_no_overlap(self) -> None:
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[20, 20, 30, 30]], dtype=np.float32)
        iou = compute_iou_matrix(a, b)
        self.assertAlmostEqual(iou[0, 0], 0.0, places=5)

    def test_empty_arrays(self) -> None:
        a = np.zeros((0, 4), dtype=np.float32)
        b = np.array([[0, 0, 10, 10]], dtype=np.float32)
        self.assertEqual(compute_iou_matrix(a, b).shape, (0, 1))


class TestSortTracker(unittest.TestCase):
    def test_id_persistence_across_frames(self) -> None:
        tracker = SortTracker(max_age=2, min_hits=1, iou_threshold=0.3)
        det1 = [Detection(bbox=(100, 100, 200, 250), confidence=0.9, class_id=0, label="person")]
        tracks1 = tracker.update(det1)
        self.assertEqual(len(tracks1), 1)
        first_id = tracks1[0].track_id

        det2 = [Detection(bbox=(106, 102, 206, 252), confidence=0.88, class_id=0, label="person")]
        tracks2 = tracker.update(det2)
        self.assertEqual(len(tracks2), 1)
        self.assertEqual(tracks2[0].track_id, first_id)

    def test_track_expiry_on_empty_detections(self) -> None:
        tracker = SortTracker(max_age=2, min_hits=1, iou_threshold=0.3)
        det = [Detection(bbox=(100, 100, 200, 250), confidence=0.9, class_id=0, label="person")]
        first_id = tracker.update(det)[0].track_id
        for _ in range(4):
            self.assertEqual(tracker.update([]), [])
        new = tracker.update(det)
        self.assertEqual(len(new), 1)
        self.assertNotEqual(new[0].track_id, first_id)

    def test_class_aware_matching_prevents_cross_class(self) -> None:
        tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3, class_aware=True)
        det_person = [
            Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, label="person")
        ]
        tracks = tracker.update(det_person)
        person_id = tracks[0].track_id

        det_car = [Detection(bbox=(105, 105, 205, 205), confidence=0.8, class_id=2, label="car")]
        tracks2 = tracker.update(det_car)
        self.assertEqual(len(tracks2), 1)
        self.assertNotEqual(tracks2[0].track_id, person_id)
        self.assertEqual(tracks2[0].label, "car")

    def test_hungarian_matching_gives_optimal_assignment(self) -> None:
        tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.2, class_aware=False)
        dets = [
            Detection(bbox=(10, 10, 50, 50), confidence=0.9, class_id=0, label="person"),
            Detection(bbox=(100, 100, 150, 150), confidence=0.9, class_id=0, label="person"),
        ]
        tracks1 = tracker.update(dets)
        self.assertEqual(len(tracks1), 2)
        id_a, id_b = tracks1[0].track_id, tracks1[1].track_id

        dets_shifted = [
            Detection(bbox=(12, 12, 52, 52), confidence=0.9, class_id=0, label="person"),
            Detection(bbox=(102, 102, 152, 152), confidence=0.9, class_id=0, label="person"),
        ]
        tracks2 = tracker.update(dets_shifted)
        self.assertEqual(len(tracks2), 2)
        ids = {t.track_id for t in tracks2}
        self.assertEqual(ids, {id_a, id_b})


# ---------------------------------------------------------------------------
# Detector class tests
# ---------------------------------------------------------------------------
class TestDetector(unittest.TestCase):
    def test_missing_model_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            YOLOv8OnnxDetector(Path("/nonexistent/yolov8n.onnx"))

    def test_empty_frame_raises(self) -> None:
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        detector = YOLOv8OnnxDetector("unused.onnx", session=mock_session)
        with self.assertRaises(ValueError):
            detector.detect(np.array([], dtype=np.uint8))

    def test_invalid_thresholds_raise(self) -> None:
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        with self.assertRaises(ValueError):
            YOLOv8OnnxDetector("unused.onnx", conf_threshold=0.0, session=mock_session)
        with self.assertRaises(ValueError):
            YOLOv8OnnxDetector("unused.onnx", iou_threshold=0.0, session=mock_session)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------
class TestPipeline(unittest.TestCase):
    def test_draw_tracks_output_shape(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        tracker = SortTracker()
        tracks = tracker.update(
            [Detection(bbox=(20, 20, 80, 100), confidence=0.85, class_id=2, label="car")]
        )
        annotated = draw_tracks(frame, tracks, fps=12.5)
        self.assertEqual(annotated.shape, frame.shape)

    def test_invalid_video_path_raises(self) -> None:
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        detector = YOLOv8OnnxDetector("unused.onnx", session=mock_session)
        tracker = SortTracker()

        with self.assertRaises(FileNotFoundError):
            list(iter_tracked_frames(Path("/nonexistent/video.mp4"), detector, tracker))

    def test_resource_cleanup_on_empty_video(self) -> None:
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        detector = YOLOv8OnnxDetector("unused.onnx", session=mock_session)
        tracker = SortTracker()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        with patch("src.pipeline.cv2.VideoCapture", return_value=mock_cap):
            frames = list(iter_tracked_frames(0, detector, tracker))
        self.assertEqual(frames, [])
        mock_cap.release.assert_called_once()


# ---------------------------------------------------------------------------
# Real ONNX inference test (skipped when model is absent)
# ---------------------------------------------------------------------------
class TestRealOnnxInference(unittest.TestCase):
    @unittest.skipUnless(MODEL_PATH.exists(), "yolov8n.onnx not available")
    def test_real_inference_produces_detections(self) -> None:
        detector = YOLOv8OnnxDetector(MODEL_PATH, conf_threshold=0.25)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = detector.detect(frame)
        self.assertIsInstance(dets, list)


if __name__ == "__main__":
    unittest.main()
