from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter

from src.detector import Detection


@dataclass(frozen=True)
class TrackedObject:
    """Tracked object with persistent ID across frames."""

    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two (N, 4) and (M, 4) xyxy box arrays."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    ax1, ay1, ax2, ay2 = np.split(boxes_a, 4, axis=1)
    bx1, by1, bx2, by2 = np.split(boxes_b, 4, axis=1)

    inter_x1 = np.maximum(ax1, bx1.T)
    inter_y1 = np.maximum(ay1, by1.T)
    inter_x2 = np.minimum(ax2, bx2.T)
    inter_y2 = np.minimum(ay2, by2.T)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(0.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = np.maximum(0.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b.T - inter_area

    return np.where(union > 0.0, inter_area / union, 0.0)


def _bbox_to_z(bbox: tuple[float, float, float, float]) -> np.ndarray:
    w = max(1.0, bbox[2] - bbox[0])
    h = max(1.0, bbox[3] - bbox[1])
    cx = bbox[0] + w / 2.0
    cy = bbox[1] + h / 2.0
    return np.array([[cx], [cy], [w * h], [w / h]], dtype=np.float64)


def _x_to_bbox(state: np.ndarray) -> tuple[float, float, float, float]:
    cx, cy, area, ratio = (
        float(state[0, 0]),
        float(state[1, 0]),
        float(state[2, 0]),
        float(state[3, 0]),
    )
    area = max(1.0, area)
    ratio = max(0.01, ratio)
    w = float(np.sqrt(area * ratio))
    h = float(area / w)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class _KalmanBoxTrack:
    """Single object track backed by a constant-velocity Kalman filter."""

    def __init__(self, track_id: int, detection: Detection) -> None:
        self.track_id = track_id
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.label = detection.label
        self.hits = 1
        self.time_since_update = 0

        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float64,
        )
        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4] = _bbox_to_z(detection.bbox)

    def predict(self) -> tuple[float, float, float, float]:
        if (self.kf.x[6, 0] + self.kf.x[2, 0]) <= 0:
            self.kf.x[6, 0] = 0.0
        self.kf.predict()
        self.time_since_update += 1
        return _x_to_bbox(self.kf.x)

    def update(self, detection: Detection) -> None:
        self.time_since_update = 0
        self.hits += 1
        self.confidence = detection.confidence
        self.class_id = detection.class_id
        self.label = detection.label
        self.kf.update(_bbox_to_z(detection.bbox))

    def current_bbox(self) -> tuple[float, float, float, float]:
        return _x_to_bbox(self.kf.x)


class SortTracker:
    """Lightweight SORT tracker using Kalman filtering and IoU association."""

    def __init__(
        self,
        max_age: int = 10,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._tracks: list[_KalmanBoxTrack] = []

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """Update tracker state with current frame detections and return active tracks."""
        predicted_boxes: list[tuple[float, float, float, float]] = []
        for track in self._tracks:
            predicted_boxes.append(track.predict())

        matched_track_indices: set[int] = set()
        matched_det_indices: set[int] = set()

        if self._tracks and detections:
            trk_arr = np.array(predicted_boxes, dtype=np.float32)
            det_arr = np.array([d.bbox for d in detections], dtype=np.float32)
            iou_matrix = compute_iou_matrix(det_arr, trk_arr)

            # Greedy highest-IoU assignment
            candidates = np.argwhere(iou_matrix >= self.iou_threshold)
            if len(candidates) > 0:
                scores = iou_matrix[candidates[:, 0], candidates[:, 1]]
                order = np.argsort(-scores)
                for idx in order:
                    det_idx, trk_idx = int(candidates[idx, 0]), int(candidates[idx, 1])
                    if det_idx not in matched_det_indices and trk_idx not in matched_track_indices:
                        self._tracks[trk_idx].update(detections[det_idx])
                        matched_det_indices.add(det_idx)
                        matched_track_indices.add(trk_idx)

        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices:
                self._tracks.append(_KalmanBoxTrack(self._next_id, det))
                self._next_id += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.max_age]

        active: list[TrackedObject] = []
        for track in self._tracks:
            if track.time_since_update == 0 and track.hits >= self.min_hits:
                active.append(
                    TrackedObject(
                        track_id=track.track_id,
                        bbox=track.current_bbox(),
                        confidence=track.confidence,
                        class_id=track.class_id,
                        label=track.label,
                    )
                )
        return active
