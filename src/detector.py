from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from src.coco_labels import COCO_CLASSES


@dataclass(frozen=True)
class Detection:
    """Single object detection in original image coordinates."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    label: str


def letterbox_image(
    image: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize image with unchanged aspect ratio using padding.

    Returns ``(padded_image, scale, (left_pad, top_pad))`` where ``left_pad`` and
    ``top_pad`` match the exact integer border offsets applied by ``copyMakeBorder``.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be empty.")

    orig_h, orig_w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = round(orig_w * scale), round(orig_h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pad_w = (target_w - new_w) / 2.0
    pad_h = (target_h - new_h) / 2.0

    top, bottom = round(pad_h - 0.1), round(pad_h + 0.1)
    left, right = round(pad_w - 0.1), round(pad_w + 0.1)
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, scale, (float(left), float(top))


def _unscale_and_clip_xyxy(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    orig_shape: tuple[int, int],
    scale: float,
    pad: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map letterbox xyxy coordinates back to clipped original image coordinates."""
    pad_w, pad_h = pad
    orig_h, orig_w = orig_shape
    ox1 = np.clip((x1 - pad_w) / scale, 0.0, float(orig_w))
    oy1 = np.clip((y1 - pad_h) / scale, 0.0, float(orig_h))
    ox2 = np.clip((x2 - pad_w) / scale, 0.0, float(orig_w))
    oy2 = np.clip((y2 - pad_h) / scale, 0.0, float(orig_h))
    return ox1, oy1, ox2, oy2


def _label_for_class(class_id: int) -> str:
    return COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else f"class_{class_id}"


def get_static_input_size(session: Any) -> tuple[int, int] | None:
    """Return ``(width, height)`` if the ONNX model has a static 4D input shape, else ``None``."""
    inputs = session.get_inputs()
    if not inputs:
        return None
    shape = getattr(inputs[0], "shape", None)
    if isinstance(shape, (list, tuple)) and len(shape) == 4:
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return (w, h)
    return None


def parse_yolov8_output(
    raw_output: np.ndarray,
    orig_shape: tuple[int, int],
    scale: float,
    pad: tuple[float, float],
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    *,
    class_aware_nms: bool = True,
) -> list[Detection]:
    """Decode YOLOv8 ONNX output ``(1, 84, N)`` into filtered ``Detection`` objects."""
    if raw_output.ndim != 3 or raw_output.shape[1] < 5:
        raise ValueError(f"Unexpected model output shape: {raw_output.shape}")

    preds = raw_output[0].T  # (N, 4 + num_classes)
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = class_scores[np.arange(len(preds)), class_ids]

    keep = confidences >= conf_threshold
    if not np.any(keep):
        return []

    boxes_xywh = boxes_xywh[keep]
    confidences = confidences[keep]
    class_ids = class_ids[keep]

    half_w = boxes_xywh[:, 2] / 2.0
    half_h = boxes_xywh[:, 3] / 2.0
    x1, y1, x2, y2 = _unscale_and_clip_xyxy(
        boxes_xywh[:, 0] - half_w,
        boxes_xywh[:, 1] - half_h,
        boxes_xywh[:, 0] + half_w,
        boxes_xywh[:, 1] + half_h,
        orig_shape=orig_shape,
        scale=scale,
        pad=pad,
    )

    w = x2 - x1
    h = y2 - y1
    valid = (w > 0.0) & (h > 0.0)
    if not np.any(valid):
        return []

    x1, y1, w, h = x1[valid], y1[valid], w[valid], h[valid]
    confidences = confidences[valid]
    class_ids = class_ids[valid]

    # Class-aware NMS offsets each class into a disjoint coordinate region.
    if class_aware_nms:
        max_wh = 7680.0
        offsets = class_ids.astype(np.float64) * max_wh
        nms_boxes = np.column_stack([x1 + offsets, y1 + offsets, w, h]).tolist()
    else:
        nms_boxes = np.column_stack([x1, y1, w, h]).tolist()

    indices = cv2.dnn.NMSBoxes(
        nms_boxes,
        confidences.tolist(),
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
    )
    if len(indices) == 0:
        return []

    detections: list[Detection] = []
    for idx in np.asarray(indices).flatten():
        bx, by, bw, bh = float(x1[idx]), float(y1[idx]), float(w[idx]), float(h[idx])
        cls_id = int(class_ids[idx])
        detections.append(
            Detection(
                bbox=(bx, by, bx + bw, by + bh),
                confidence=float(confidences[idx]),
                class_id=cls_id,
                label=_label_for_class(cls_id),
            )
        )
    return detections


def parse_yolo26_output(
    raw_output: np.ndarray,
    orig_shape: tuple[int, int],
    scale: float,
    pad: tuple[float, float],
    conf_threshold: float = 0.25,
) -> list[Detection]:
    """Decode end-to-end NMS-free YOLO26 ONNX output ``(1, N, 6)``."""
    if raw_output.ndim != 3 or raw_output.shape[2] != 6:
        raise ValueError(f"Unexpected YOLO26 output shape: {raw_output.shape}")

    preds = raw_output[0]
    confidences = preds[:, 4]
    keep = confidences >= conf_threshold
    if not np.any(keep):
        return []

    preds = preds[keep]
    x1, y1, x2, y2 = _unscale_and_clip_xyxy(
        preds[:, 0],
        preds[:, 1],
        preds[:, 2],
        preds[:, 3],
        orig_shape=orig_shape,
        scale=scale,
        pad=pad,
    )

    detections: list[Detection] = []
    for i in range(len(preds)):
        bx1, by1, bx2, by2 = float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])
        if bx2 <= bx1 or by2 <= by1:
            continue
        cls_id = round(float(preds[i, 5]))
        detections.append(
            Detection(
                bbox=(bx1, by1, bx2, by2),
                confidence=float(preds[i, 4]),
                class_id=cls_id,
                label=_label_for_class(cls_id),
            )
        )
    return detections


def parse_yolo_output(
    raw_output: np.ndarray,
    orig_shape: tuple[int, int],
    scale: float,
    pad: tuple[float, float],
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
    *,
    class_aware_nms: bool = True,
) -> list[Detection]:
    """Dispatch ONNX output decoding based on tensor layout."""
    if raw_output.ndim == 3 and raw_output.shape[2] == 6:
        return parse_yolo26_output(
            raw_output,
            orig_shape=orig_shape,
            scale=scale,
            pad=pad,
            conf_threshold=conf_threshold,
        )
    return parse_yolov8_output(
        raw_output,
        orig_shape=orig_shape,
        scale=scale,
        pad=pad,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        class_aware_nms=class_aware_nms,
    )


class YOLOv8OnnxDetector:
    """CPU-compatible YOLO ONNX object detector (supports YOLOv8 and YOLO26 layouts)."""

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
        session: Any = None,
        *,
        class_aware_nms: bool = True,
        intra_op_num_threads: int | None = None,
    ) -> None:
        if not 0.0 < conf_threshold <= 1.0:
            raise ValueError("Confidence threshold must be in (0.0, 1.0].")
        if not 0.0 < iou_threshold <= 1.0:
            raise ValueError("IoU threshold must be in (0.0, 1.0].")
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.class_aware_nms = class_aware_nms

        if session is not None:
            self._session = session
        else:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found at {self.model_path}. "
                    "See README.md for model download instructions."
                )
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if intra_op_num_threads is not None:
                sess_options.intra_op_num_threads = intra_op_num_threads
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        self._input_name = self._session.get_inputs()[0].name
        self.static_input_size = get_static_input_size(self._session)
        if self.static_input_size is not None and self.input_size != self.static_input_size:
            raise ValueError(
                f"Model {self.model_path.name} requires static input size "
                f"{self.static_input_size}, got {self.input_size}."
            )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run object detection on a single BGR frame."""
        if frame is None or frame.size == 0:
            raise ValueError("Input frame cannot be empty.")

        padded, scale, pad = letterbox_image(frame, self.input_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[np.newaxis, ...], dtype=np.float32)
        blob /= 255.0

        outputs = self._session.run(None, {self._input_name: blob})
        return parse_yolo_output(
            outputs[0],
            orig_shape=frame.shape[:2],
            scale=scale,
            pad=pad,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            class_aware_nms=self.class_aware_nms,
        )
