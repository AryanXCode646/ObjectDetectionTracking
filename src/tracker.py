from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

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
    """SORT tracker using Kalman filtering, Hungarian IoU assignment, and class-aware matching."""

    def __init__(
        self,
        max_age: int = 15,
        min_hits: int = 1,
        iou_threshold: float = 0.3,
        *,
        class_aware: bool = True,
        high_conf_threshold: float | None = None,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.class_aware = class_aware
        self.high_conf_threshold = high_conf_threshold
        self._next_id = 1
        self._tracks: list[_KalmanBoxTrack] = []

    def _associate(
        self,
        det_indices: list[int],
        detections: list[Detection],
        trk_indices: list[int],
        predicted_boxes: list[tuple[float, float, float, float]],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not det_indices or not trk_indices:
            return [], list(det_indices), list(trk_indices)

        det_arr = np.array([detections[i].bbox for i in det_indices], dtype=np.float32)
        trk_arr = np.array([predicted_boxes[j] for j in trk_indices], dtype=np.float32)
        iou_matrix = compute_iou_matrix(det_arr, trk_arr)

        if self.class_aware:
            det_cls = np.array([detections[i].class_id for i in det_indices])[:, None]
            trk_cls = np.array([self._tracks[j].class_id for j in trk_indices])[None, :]
            iou_matrix = np.where(det_cls == trk_cls, iou_matrix, 0.0)

        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matches: list[tuple[int, int]] = []
        matched_dets: set[int] = set()
        matched_trks: set[int] = set()

        for r, c in zip(row_ind, col_ind, strict=True):
            if iou_matrix[r, c] >= self.iou_threshold:
                d_idx = det_indices[int(r)]
                t_idx = trk_indices[int(c)]
                matches.append((d_idx, t_idx))
                matched_dets.add(d_idx)
                matched_trks.add(t_idx)

        unmatched_dets = [i for i in det_indices if i not in matched_dets]
        unmatched_trks = [j for j in trk_indices if j not in matched_trks]
        return matches, unmatched_dets, unmatched_trks

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """Update tracker state with current frame detections and return active tracks."""
        predicted_boxes = [track.predict() for track in self._tracks]
        all_trk_indices = list(range(len(self._tracks)))

        if self.high_conf_threshold is not None:
            # Two-stage ByteTrack-style association: high-confidence first, then low-confidence
            # detections matched only against remaining unmatched tracks.
            high_dets = [
                i for i, d in enumerate(detections) if d.confidence >= self.high_conf_threshold
            ]
            low_dets = [
                i for i, d in enumerate(detections) if d.confidence < self.high_conf_threshold
            ]
            matches_hi, unmatched_hi, unmatched_trks = self._associate(
                high_dets, detections, all_trk_indices, predicted_boxes
            )
            matches_lo, _, _ = self._associate(
                low_dets, detections, unmatched_trks, predicted_boxes
            )
            matches = matches_hi + matches_lo
            unmatched_new_dets = unmatched_hi
        else:
            all_det_indices = list(range(len(detections)))
            matches, unmatched_new_dets, _ = self._associate(
                all_det_indices, detections, all_trk_indices, predicted_boxes
            )

        for det_idx, trk_idx in matches:
            self._tracks[trk_idx].update(detections[det_idx])

        for det_idx in unmatched_new_dets:
            self._tracks.append(_KalmanBoxTrack(self._next_id, detections[det_idx]))
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
