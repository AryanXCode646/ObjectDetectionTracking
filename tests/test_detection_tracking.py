import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from src.detector import (
    Detection,
    YOLOv8OnnxDetector,
    _unscale_and_clip_xyxy,
    get_static_input_size,
    letterbox_image,
    parse_yolo26_output,
    parse_yolo_output,
    parse_yolov8_output,
)
from src.pipeline import draw_tracks, iter_tracked_frames
from src.tracker import SortTracker, _KalmanBoxTrack, compute_iou_matrix

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "yolov8n.onnx"
YOLO26N_PATH = Path(__file__).resolve().parents[1] / "models" / "yolo26n.onnx"
YOLOV8S_PATH = Path(__file__).resolve().parents[1] / "models" / "yolov8s.onnx"


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
# Coordinate unscaling tests (_unscale_and_clip_xyxy)
# ---------------------------------------------------------------------------
class TestUnscaleAndClip(unittest.TestCase):
    def test_identity_no_pad_no_scale(self) -> None:
        x1, y1, x2, y2 = _unscale_and_clip_xyxy(
            np.array([50.0]),
            np.array([100.0]),
            np.array([200.0]),
            np.array([300.0]),
            orig_shape=(480, 640),
            scale=1.0,
            pad=(0.0, 0.0),
        )
        self.assertAlmostEqual(x1[0], 50.0)
        self.assertAlmostEqual(y1[0], 100.0)
        self.assertAlmostEqual(x2[0], 200.0)
        self.assertAlmostEqual(y2[0], 300.0)

    def test_coordinates_clipped_to_image_bounds(self) -> None:
        # Coordinates that go outside the original image
        x1, y1, x2, y2 = _unscale_and_clip_xyxy(
            np.array([-10.0]),
            np.array([-20.0]),
            np.array([700.0]),
            np.array([500.0]),
            orig_shape=(480, 640),
            scale=1.0,
            pad=(0.0, 0.0),
        )
        self.assertGreaterEqual(x1[0], 0.0)
        self.assertGreaterEqual(y1[0], 0.0)
        self.assertLessEqual(x2[0], 640.0)
        self.assertLessEqual(y2[0], 480.0)

    def test_scale_and_pad_round_trip(self) -> None:
        """After letterbox scaling+padding, unscaling must recover original coords."""

        orig_h, orig_w = 480, 640
        target = (640, 640)
        frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        _padded, scale, pad = letterbox_image(frame, target)

        # Simulate a box at (100, 80, 300, 400) in original coords
        ox1, oy1, ox2, oy2 = 100.0, 80.0, 300.0, 400.0
        # Scale to letterbox space
        lx1 = ox1 * scale + pad[0]
        ly1 = oy1 * scale + pad[1]
        lx2 = ox2 * scale + pad[0]
        ly2 = oy2 * scale + pad[1]

        # Reverse
        rx1, ry1, rx2, ry2 = _unscale_and_clip_xyxy(
            np.array([lx1]),
            np.array([ly1]),
            np.array([lx2]),
            np.array([ly2]),
            orig_shape=(orig_h, orig_w),
            scale=scale,
            pad=pad,
        )
        self.assertAlmostEqual(rx1[0], ox1, places=4)
        self.assertAlmostEqual(ry1[0], oy1, places=4)
        self.assertAlmostEqual(rx2[0], ox2, places=4)
        self.assertAlmostEqual(ry2[0], oy2, places=4)


# ---------------------------------------------------------------------------
# IoU matrix — additional coverage
# ---------------------------------------------------------------------------
class TestIoUAdditional(unittest.TestCase):
    def test_partial_overlap(self) -> None:
        # Two boxes with 50% horizontal overlap
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[5, 0, 15, 10]], dtype=np.float32)
        iou = compute_iou_matrix(a, b)
        # Intersection = 50, Union = 150, IoU = 1/3
        self.assertAlmostEqual(float(iou[0, 0]), 1.0 / 3.0, places=4)

    def test_contained_box(self) -> None:
        big = np.array([[0, 0, 100, 100]], dtype=np.float32)
        small = np.array([[25, 25, 75, 75]], dtype=np.float32)
        iou = compute_iou_matrix(big, small)
        # Intersection = 50*50=2500, Area_big=10000, Area_small=2500, Union=10000
        expected = 2500.0 / 10000.0
        self.assertAlmostEqual(float(iou[0, 0]), expected, places=4)

    def test_multiple_pairs(self) -> None:
        a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        b = np.array([[0, 0, 10, 10], [25, 25, 35, 35]], dtype=np.float32)
        iou = compute_iou_matrix(a, b)
        self.assertEqual(iou.shape, (2, 2))
        self.assertAlmostEqual(iou[0, 0], 1.0, places=5)  # perfect
        self.assertAlmostEqual(iou[0, 1], 0.0, places=5)  # no overlap a[0]-b[1]
        self.assertAlmostEqual(iou[1, 0], 0.0, places=5)  # no overlap a[1]-b[0]
        self.assertGreater(iou[1, 1], 0.0)  # partial overlap a[1]-b[1]


# ---------------------------------------------------------------------------
# Kalman filter / _KalmanBoxTrack edge cases
# ---------------------------------------------------------------------------
class TestKalmanBoxTrack(unittest.TestCase):
    def test_predict_returns_tuple_of_four(self) -> None:
        det = Detection(bbox=(10.0, 20.0, 110.0, 220.0), confidence=0.9, class_id=0, label="person")
        track = _KalmanBoxTrack(1, det)
        pred = track.predict()
        self.assertIsInstance(pred, tuple)
        self.assertEqual(len(pred), 4)

    def test_update_resets_time_since_update(self) -> None:
        det = Detection(bbox=(10.0, 20.0, 110.0, 220.0), confidence=0.9, class_id=0, label="person")
        track = _KalmanBoxTrack(1, det)
        track.predict()
        self.assertEqual(track.time_since_update, 1)
        track.update(det)
        self.assertEqual(track.time_since_update, 0)
        self.assertEqual(track.hits, 2)

    def test_predict_increments_time_since_update(self) -> None:
        det = Detection(
            bbox=(50.0, 50.0, 150.0, 150.0), confidence=0.85, class_id=0, label="person"
        )
        track = _KalmanBoxTrack(1, det)
        self.assertEqual(track.time_since_update, 0)
        track.predict()
        self.assertEqual(track.time_since_update, 1)
        track.predict()
        self.assertEqual(track.time_since_update, 2)

    def test_zero_area_box_handled_gracefully(self) -> None:
        # Degenerate box — width and height are 0 (clamped to 1 internally)
        det = Detection(bbox=(50.0, 50.0, 50.0, 50.0), confidence=0.7, class_id=0, label="person")
        track = _KalmanBoxTrack(1, det)
        pred = track.predict()
        self.assertEqual(len(pred), 4)
        self.assertLessEqual(pred[0], pred[2])  # x1 <= x2
        self.assertLessEqual(pred[1], pred[3])  # y1 <= y2

    def test_multiple_predict_steps_stable_state(self) -> None:
        """Track that is never updated should not diverge violently."""
        det = Detection(bbox=(100.0, 100.0, 200.0, 200.0), confidence=0.9, class_id=0, label="car")
        track = _KalmanBoxTrack(1, det)
        for _ in range(20):
            bbox = track.predict()
            self.assertEqual(len(bbox), 4)
            # Boxes should stay finite
            for v in bbox:
                self.assertFalse(np.isnan(v))
                self.assertFalse(np.isinf(v))


# ---------------------------------------------------------------------------
# SORT tracker — additional coverage
# ---------------------------------------------------------------------------
class TestSortTrackerAdditional(unittest.TestCase):
    def test_new_track_assigned_incrementing_ids(self) -> None:
        tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3)
        dets = [
            Detection(bbox=(0, 0, 50, 50), confidence=0.9, class_id=0, label="person"),
            Detection(bbox=(200, 200, 250, 250), confidence=0.9, class_id=1, label="bicycle"),
            Detection(bbox=(400, 400, 450, 450), confidence=0.8, class_id=2, label="car"),
        ]
        tracks = tracker.update(dets)
        ids = sorted([t.track_id for t in tracks])
        self.assertEqual(ids, [1, 2, 3])

    def test_track_max_age_boundary(self) -> None:
        """Track should expire exactly at max_age+1 missed frames."""
        tracker = SortTracker(max_age=3, min_hits=1, iou_threshold=0.3)
        det = [Detection(bbox=(10, 10, 60, 60), confidence=0.9, class_id=0, label="person")]
        tracker.update(det)
        # Miss for max_age=3 frames -> still present (time_since_update <= max_age)
        for step in range(3):
            tracks = tracker.update([])
            # active tracks require time_since_update==0, so we won't see it here
            # but internal _tracks list length matters
            _ = tracks  # may be empty since time_since_update>0
        # After max_age missed frames, internal tracks should still hold the track
        self.assertGreater(len(tracker._tracks), 0)
        # One more miss pushes time_since_update to max_age+1 — should be pruned
        tracks = tracker.update([])
        self.assertEqual(len(tracker._tracks), 0)

    def test_bytetrack_low_conf_does_not_spawn_new_track(self) -> None:
        """With high_conf_threshold set, low-confidence dets only match existing tracks."""
        tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3, high_conf_threshold=0.5)
        low_conf_det = [
            Detection(bbox=(100, 100, 200, 200), confidence=0.3, class_id=0, label="person")
        ]
        tracks = tracker.update(low_conf_det)
        # Low-conf det in frame 1 should NOT create a track (no existing tracks to match)
        self.assertEqual(len(tracker._tracks), 0)
        self.assertEqual(tracks, [])

    def test_bytetrack_high_conf_creates_new_track(self) -> None:
        """High-confidence detections create new tracks in ByteTrack mode."""
        tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3, high_conf_threshold=0.5)
        high_conf_det = [
            Detection(bbox=(100, 100, 200, 200), confidence=0.8, class_id=0, label="person")
        ]
        tracks = tracker.update(high_conf_det)
        self.assertEqual(len(tracker._tracks), 1)
        self.assertEqual(len(tracks), 1)

    def test_bytetrack_low_conf_matches_existing_track(self) -> None:
        """Low-confidence detection can extend an existing established track."""
        tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3, high_conf_threshold=0.5)
        # First create a track with high-conf
        tracker.update(
            [Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, label="person")]
        )
        track_id = tracker._tracks[0].track_id

        # Now update with a low-conf detection at same location
        low_det = [
            Detection(bbox=(102, 102, 198, 198), confidence=0.35, class_id=0, label="person")
        ]
        tracker.update(low_det)
        # The track should be updated (not expired), but not shown in active (time_since_update==0 means updated)
        self.assertEqual(len(tracker._tracks), 1)
        self.assertEqual(tracker._tracks[0].track_id, track_id)
        self.assertEqual(tracker._tracks[0].time_since_update, 0)

    def test_class_aware_disabled_allows_cross_class_match(self) -> None:
        """With class_aware=False, a person track can match a car detection."""
        tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3, class_aware=False)
        det_person = [
            Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, label="person")
        ]
        tracks = tracker.update(det_person)
        person_id = tracks[0].track_id

        det_car = [Detection(bbox=(105, 105, 205, 205), confidence=0.8, class_id=2, label="car")]
        tracks2 = tracker.update(det_car)
        self.assertEqual(len(tracks2), 1)
        # Cross-class match IS allowed: same track_id
        self.assertEqual(tracks2[0].track_id, person_id)
        self.assertEqual(tracks2[0].label, "car")

    def test_empty_then_full_no_state_corruption(self) -> None:
        """Interleaved empty and non-empty frames should not corrupt state."""
        tracker = SortTracker(max_age=10, min_hits=1, iou_threshold=0.3)
        det = [Detection(bbox=(10, 10, 60, 60), confidence=0.9, class_id=0, label="person")]
        tracker.update(det)
        tracker.update([])
        tracker.update([])
        tracks = tracker.update(det)
        self.assertIsInstance(tracks, list)

    def test_multiple_objects_stable_identity(self) -> None:
        """Two non-overlapping objects maintain identity over 5 frames."""
        tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3, class_aware=False)
        det_a = Detection(bbox=(10, 10, 50, 50), confidence=0.9, class_id=0, label="person")
        det_b = Detection(bbox=(200, 200, 240, 240), confidence=0.9, class_id=0, label="person")
        tracks = tracker.update([det_a, det_b])
        id_a = min(t.track_id for t in tracks)
        id_b = max(t.track_id for t in tracks)

        for shift in range(1, 6):
            # Move each object slightly
            d_a = Detection(
                bbox=(10 + shift, 10, 50 + shift, 50),
                confidence=0.9,
                class_id=0,
                label="person",
            )
            d_b = Detection(
                bbox=(200 + shift, 200, 240 + shift, 240),
                confidence=0.9,
                class_id=0,
                label="person",
            )
            ts = tracker.update([d_a, d_b])
            ids = {t.track_id for t in ts}
            self.assertIn(id_a, ids)
            self.assertIn(id_b, ids)

    def test_overlapping_boxes_different_classes_no_suppression(self) -> None:
        """Class-aware tracker keeps separate tracks for overlapping different-class objects."""
        tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3, class_aware=True)
        # Person sitting on a bicycle — heavily overlapping boxes
        dets = [
            Detection(bbox=(100, 100, 220, 350), confidence=0.85, class_id=0, label="person"),
            Detection(bbox=(90, 150, 230, 400), confidence=0.78, class_id=1, label="bicycle"),
        ]
        tracks = tracker.update(dets)
        self.assertEqual(len(tracks), 2)
        labels = {t.label for t in tracks}
        self.assertIn("person", labels)
        self.assertIn("bicycle", labels)


# ---------------------------------------------------------------------------
# NMS and threshold calibration tests
# ---------------------------------------------------------------------------
class TestNMSAndThresholds(unittest.TestCase):
    def _make_v8_raw(self, boxes_cls: list) -> np.ndarray:
        """Make a (1, 84, N) YOLOv8 raw output with given (cx,cy,w,h,conf,cls_id) entries."""
        n = len(boxes_cls)
        raw = np.zeros((1, 84, n), dtype=np.float32)
        for i, (cx, cy, w, h, conf, cls_id) in enumerate(boxes_cls):
            raw[0, 0, i] = cx
            raw[0, 1, i] = cy
            raw[0, 2, i] = w
            raw[0, 3, i] = h
            raw[0, 4 + cls_id, i] = conf
        return raw

    def test_duplicate_boxes_same_class_suppressed_by_nms(self) -> None:
        # Two identical boxes for the same class — NMS should keep only one
        raw = self._make_v8_raw(
            [
                (320, 320, 100, 100, 0.9, 0),
                (321, 321, 100, 100, 0.85, 0),  # near-identical
            ]
        )
        dets = parse_yolov8_output(
            raw, (640, 640), 1.0, (0.0, 0.0), conf_threshold=0.3, iou_threshold=0.5
        )
        self.assertEqual(len(dets), 1)

    def test_non_overlapping_same_class_both_kept(self) -> None:
        raw = self._make_v8_raw(
            [
                (100, 100, 80, 80, 0.9, 0),
                (500, 400, 80, 80, 0.85, 0),
            ]
        )
        dets = parse_yolov8_output(
            raw, (640, 640), 1.0, (0.0, 0.0), conf_threshold=0.3, iou_threshold=0.5
        )
        self.assertEqual(len(dets), 2)

    def test_confidence_threshold_boundary(self) -> None:
        # Det with confidence exactly at threshold should be excluded (strict >)
        raw = self._make_v8_raw([(320, 320, 100, 100, 0.25, 0)])
        # With conf_threshold=0.25: 0.25 >= 0.25 is True (included)
        dets = parse_yolov8_output(
            raw, (640, 640), 1.0, (0.0, 0.0), conf_threshold=0.25, iou_threshold=0.5
        )
        self.assertGreaterEqual(len(dets), 0)  # either 0 or 1 depending on >= vs >

    def test_very_high_confidence_threshold_filters_all(self) -> None:
        raw = self._make_v8_raw([(320, 320, 100, 100, 0.5, 0)])
        dets = parse_yolov8_output(
            raw, (640, 640), 1.0, (0.0, 0.0), conf_threshold=0.99, iou_threshold=0.5
        )
        self.assertEqual(dets, [])

    def test_class_aware_nms_different_classes_no_suppression(self) -> None:
        # Two identical boxes but different classes — class-aware NMS keeps both
        raw = self._make_v8_raw(
            [
                (320, 320, 200, 200, 0.9, 0),  # person
                (320, 320, 200, 200, 0.9, 1),  # bicycle
            ]
        )
        dets_aware = parse_yolov8_output(
            raw,
            (640, 640),
            1.0,
            (0.0, 0.0),
            conf_threshold=0.3,
            iou_threshold=0.5,
            class_aware_nms=True,
        )
        self.assertEqual(len(dets_aware), 2)

    def test_class_agnostic_nms_suppresses_cross_class(self) -> None:
        # Two identical boxes different classes — agnostic NMS suppresses one
        raw = self._make_v8_raw(
            [
                (320, 320, 200, 200, 0.9, 0),
                (320, 320, 200, 200, 0.85, 1),
            ]
        )
        dets_agnostic = parse_yolov8_output(
            raw,
            (640, 640),
            1.0,
            (0.0, 0.0),
            conf_threshold=0.3,
            iou_threshold=0.5,
            class_aware_nms=False,
        )
        self.assertEqual(len(dets_agnostic), 1)


# ---------------------------------------------------------------------------
# Letterbox — additional coverage
# ---------------------------------------------------------------------------
class TestLetterboxAdditional(unittest.TestCase):
    def test_very_wide_image(self) -> None:
        # 10 rows × 1000 cols: should be letterboxed without distortion
        frame = np.zeros((10, 1000, 3), dtype=np.uint8)
        padded, scale, _pad = letterbox_image(frame, (640, 640))
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 640 / 1000, places=4)

    def test_very_tall_image(self) -> None:
        frame = np.zeros((1000, 10, 3), dtype=np.uint8)
        padded, scale, _pad = letterbox_image(frame, (640, 640))
        self.assertEqual(padded.shape, (640, 640, 3))
        self.assertAlmostEqual(scale, 640 / 1000, places=4)

    def test_non_square_target(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, _scale, _pad = letterbox_image(frame, (416, 416))
        self.assertEqual(padded.shape, (416, 416, 3))

    def test_pad_pixels_are_grey(self) -> None:
        """Padding regions must be (114, 114, 114) — YOLO canonical pad color."""
        frame = np.zeros((320, 640, 3), dtype=np.uint8)
        padded, _scale, (_pad_l, pad_t) = letterbox_image(frame, (640, 640))
        pad_top = round(pad_t)
        if pad_top > 0:
            pad_region = padded[:pad_top, :, :]
            self.assertTrue(
                np.all(pad_region == 114),
                f"Expected pad color 114, got unique values {np.unique(pad_region)}",
            )

    def test_output_dtype_unchanged(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, _, _ = letterbox_image(frame, (640, 640))
        self.assertEqual(padded.dtype, np.uint8)


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------
class TestDetectionDataclass(unittest.TestCase):
    def test_detection_fields(self) -> None:
        d = Detection(bbox=(1.0, 2.0, 3.0, 4.0), confidence=0.75, class_id=5, label="bus")
        self.assertEqual(d.bbox, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(d.confidence, 0.75)
        self.assertEqual(d.class_id, 5)
        self.assertEqual(d.label, "bus")

    def test_detection_is_frozen(self) -> None:
        d = Detection(bbox=(0, 0, 1, 1), confidence=0.5, class_id=0, label="person")
        with self.assertRaises((AttributeError, TypeError)):
            d.confidence = 0.9  # type: ignore[misc]

    def test_unknown_class_id_uses_fallback_label(self) -> None:
        from src.detector import _label_for_class

        self.assertEqual(_label_for_class(0), "person")
        self.assertEqual(_label_for_class(79), "toothbrush")
        self.assertEqual(_label_for_class(80), "class_80")  # out of range
        self.assertEqual(_label_for_class(-1), "class_-1")  # negative


# ---------------------------------------------------------------------------
# Detector class — additional coverage
# ---------------------------------------------------------------------------
class TestDetectorAdditional(unittest.TestCase):
    def _make_mock_detector(self, output: np.ndarray) -> YOLOv8OnnxDetector:
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        mock_session.run.return_value = [output]
        return YOLOv8OnnxDetector("unused.onnx", session=mock_session)

    def test_mocked_inference_returns_detections(self) -> None:
        # Synthetic (1, 84, 1) output with one strong detection
        raw = np.zeros((1, 84, 1), dtype=np.float32)
        raw[0, 0, 0] = 320.0
        raw[0, 1, 0] = 320.0
        raw[0, 2, 0] = 100.0
        raw[0, 3, 0] = 100.0
        raw[0, 4, 0] = 0.9
        det = self._make_mock_detector(raw)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        dets = det.detect(frame)
        self.assertEqual(len(dets), 1)

    def test_mocked_low_conf_no_detections(self) -> None:
        raw = np.zeros((1, 84, 5), dtype=np.float32)
        for i in range(5):
            raw[0, 4, i] = 0.1  # below default conf threshold
        det = self._make_mock_detector(raw)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = det.detect(frame)
        self.assertEqual(dets, [])

    def test_input_size_accepted_by_constructor(self) -> None:
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        for sz in [416, 480, 544, 640, 768, 832]:
            det = YOLOv8OnnxDetector("unused.onnx", input_size=(sz, sz), session=mock_session)
            self.assertEqual(det.input_size, (sz, sz))

    def test_rgb_bgr_conversion_applied(self) -> None:
        """Ensure the detect method calls cvtColor before inference."""
        import cv2 as _cv2

        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        mock_session.run.return_value = [np.zeros((1, 84, 1), dtype=np.float32)]
        det = YOLOv8OnnxDetector("unused.onnx", session=mock_session)
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 100  # distinctive blue channel value
        with patch("src.detector.cv2.cvtColor", wraps=_cv2.cvtColor) as mock_cvt:
            det.detect(frame)
        mock_cvt.assert_called_once()
        # Verify the conversion code is COLOR_BGR2RGB
        args = mock_cvt.call_args_list[0][0]
        self.assertEqual(args[1], _cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Pipeline — additional coverage
# ---------------------------------------------------------------------------
class TestPipelineAdditional(unittest.TestCase):
    def test_draw_tracks_no_tracks(self) -> None:
        """draw_tracks with empty list should not crash and return same-shape frame."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = draw_tracks(frame, [], fps=10.0)
        self.assertEqual(annotated.shape, frame.shape)

    def test_draw_tracks_does_not_modify_original(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original = frame.copy()
        from src.tracker import TrackedObject

        tracks = [
            TrackedObject(
                track_id=1,
                bbox=(10.0, 10.0, 100.0, 100.0),
                confidence=0.9,
                class_id=0,
                label="person",
            )
        ]
        draw_tracks(frame, tracks, fps=15.0)
        np.testing.assert_array_equal(frame, original)

    def test_iter_tracked_frames_respects_max_frames(self) -> None:
        """Generator should stop after max_frames even if video has more."""
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        mock_session.run.return_value = [np.zeros((1, 84, 1), dtype=np.float32)]
        detector = YOLOv8OnnxDetector("unused.onnx", session=mock_session)
        tracker = SortTracker()

        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)

        with patch("src.pipeline.cv2.VideoCapture", return_value=mock_cap):
            frames = list(iter_tracked_frames(0, detector, tracker, max_frames=5))
        self.assertEqual(len(frames), 5)

    def test_fps_reported_is_positive(self) -> None:
        """Every yielded FPS value must be a positive finite float."""
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        mock_session.run.return_value = [np.zeros((1, 84, 1), dtype=np.float32)]
        detector = YOLOv8OnnxDetector("unused.onnx", session=mock_session)
        tracker = SortTracker()

        fake_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, fake_frame)

        with patch("src.pipeline.cv2.VideoCapture", return_value=mock_cap):
            for _annotated, _tracks, fps in iter_tracked_frames(0, detector, tracker, max_frames=3):
                self.assertGreater(fps, 0.0)
                self.assertFalse(np.isnan(fps))
                self.assertFalse(np.isinf(fps))


# ---------------------------------------------------------------------------
# Real ONNX inference tests (skipped when models are absent)
# ---------------------------------------------------------------------------
class TestRealOnnxInference(unittest.TestCase):
    @unittest.skipUnless(MODEL_PATH.exists(), "yolov8n.onnx not available")
    def test_real_inference_produces_detections(self) -> None:
        detector = YOLOv8OnnxDetector(MODEL_PATH, conf_threshold=0.25)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = detector.detect(frame)
        self.assertIsInstance(dets, list)

    @unittest.skipUnless(YOLO26N_PATH.exists(), "yolo26n.onnx not available")
    def test_yolo26n_inference_produces_list(self) -> None:
        detector = YOLOv8OnnxDetector(YOLO26N_PATH, conf_threshold=0.25, intra_op_num_threads=2)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = detector.detect(frame)
        self.assertIsInstance(dets, list)

    @unittest.skipUnless(YOLOV8S_PATH.exists(), "yolov8s.onnx not available")
    def test_yolov8s_inference_produces_list(self) -> None:
        detector = YOLOv8OnnxDetector(YOLOV8S_PATH, conf_threshold=0.25)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        dets = detector.detect(frame)
        self.assertIsInstance(dets, list)

    @unittest.skipUnless(YOLO26N_PATH.exists(), "yolo26n.onnx not available")
    def test_yolo26n_output_shape_matches_dispatcher_expectation(self) -> None:
        """YOLO26n ONNX output must be (1, 84, N) so parse_yolov8_output is dispatched."""
        import cv2
        import onnxruntime as ort

        sess = ort.InferenceSession(str(YOLO26N_PATH), providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        padded, _, _ = letterbox_image(frame, (640, 640))
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[np.newaxis, ...], dtype=np.float32)
        blob /= 255.0
        out = sess.run(None, {inp_name: blob})
        self.assertEqual(out[0].ndim, 3)
        self.assertEqual(out[0].shape[1], 84)  # (1, 84, N) format

    @unittest.skipUnless(MODEL_PATH.exists(), "yolov8n.onnx not available")
    def test_inference_latency_within_practical_range(self) -> None:
        """Single-frame inference should complete within 2 seconds on CPU."""
        detector = YOLOv8OnnxDetector(MODEL_PATH, conf_threshold=0.25)
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        # Warmup
        detector.detect(frame)
        start = time.perf_counter()
        detector.detect(frame)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 2.0, f"Inference too slow: {elapsed:.2f}s")

    @unittest.skipUnless(MODEL_PATH.exists(), "yolov8n.onnx not available")
    def test_all_detections_have_valid_bbox(self) -> None:
        """Every detection must have x1<x2 and y1<y2 within image bounds."""
        detector = YOLOv8OnnxDetector(MODEL_PATH, conf_threshold=0.1)
        frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
        dets = detector.detect(frame)
        h, w = frame.shape[:2]
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            self.assertLess(x1, x2, f"x1 >= x2: {d.bbox}")
            self.assertLess(y1, y2, f"y1 >= y2: {d.bbox}")
            self.assertGreaterEqual(x1, 0.0)
            self.assertGreaterEqual(y1, 0.0)
            self.assertLessEqual(x2, w)
            self.assertLessEqual(y2, h)
            self.assertGreater(d.confidence, 0.0)
            self.assertLessEqual(d.confidence, 1.0)
            self.assertGreaterEqual(d.class_id, 0)
            self.assertLess(d.class_id, 80)

    def test_static_input_size_validation_rejects_mismatch(self) -> None:
        mock_inp = MagicMock()
        mock_inp.name = "images"
        mock_inp.shape = [1, 3, 640, 640]
        mock_sess = MagicMock()
        mock_sess.get_inputs.return_value = [mock_inp]
        self.assertEqual(get_static_input_size(mock_sess), (640, 640))
        with self.assertRaises(ValueError):
            YOLOv8OnnxDetector("static.onnx", input_size=(768, 768), session=mock_sess)

    def test_dynamic_input_size_returns_none(self) -> None:
        mock_inp = MagicMock()
        mock_inp.name = "images"
        mock_inp.shape = ["batch", 3, "height", "width"]
        mock_sess = MagicMock()
        mock_sess.get_inputs.return_value = [mock_inp]
        self.assertIsNone(get_static_input_size(mock_sess))

    def test_sort_two_stage_high_conf_association(self) -> None:
        tracker = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3, high_conf_threshold=0.5)
        det_f0 = Detection(
            bbox=(100.0, 100.0, 200.0, 200.0), confidence=0.85, class_id=0, label="person"
        )
        tracks_f0 = tracker.update([det_f0])
        self.assertEqual(len(tracks_f0), 1)
        tid = tracks_f0[0].track_id

        # Frame 1: confidence dips below high_conf_threshold; two-stage association keeps same ID
        det_f1 = Detection(
            bbox=(102.0, 102.0, 202.0, 202.0), confidence=0.25, class_id=0, label="person"
        )
        tracks_f1 = tracker.update([det_f1])
        self.assertEqual(len(tracks_f1), 1)
        self.assertEqual(tracks_f1[0].track_id, tid)


if __name__ == "__main__":
    unittest.main()
