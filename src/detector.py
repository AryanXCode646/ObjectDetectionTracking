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
    """Resize image with unchanged aspect ratio using padding."""
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
    return padded, scale, (pad_w, pad_h)


def parse_yolov8_output(
    raw_output: np.ndarray,
    orig_shape: tuple[int, int],
    scale: float,
    pad: tuple[float, float],
    conf_threshold: float = 0.35,
    iou_threshold: float = 0.45,
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

    pad_w, pad_h = pad
    orig_h, orig_w = orig_shape

    # Convert center (cx, cy, w, h) in letterbox space to (x, y, w, h) in original image space
    x1 = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0 - pad_w) / scale
    y1 = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0 - pad_h) / scale
    w = boxes_xywh[:, 2] / scale
    h = boxes_xywh[:, 3] / scale

    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    w = np.clip(w, 0, orig_w - x1)
    h = np.clip(h, 0, orig_h - y1)

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
        bx, by, bw, bh = nms_boxes[idx]
        cls_id = int(class_ids[idx])
        label = COCO_CLASSES[cls_id] if 0 <= cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
        detections.append(
            Detection(
                bbox=(float(bx), float(by), float(bx + bw), float(by + bh)),
                confidence=float(confidences[idx]),
                class_id=cls_id,
                label=label,
            )
        )
    return detections


class YOLOv8OnnxDetector:
    """CPU-compatible YOLOv8 ONNX object detector."""

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        input_size: tuple[int, int] = (640, 640),
        session: Any = None,
    ) -> None:
        if not 0.0 < conf_threshold <= 1.0:
            raise ValueError("Confidence threshold must be in (0.0, 1.0].")
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size

        if session is not None:
            self._session = session
        else:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"ONNX model not found at {self.model_path}. "
                    "See README.md for model download instructions."
                )
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
        self._input_name = self._session.get_inputs()[0].name

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run object detection on a single BGR frame."""
        if frame is None or frame.size == 0:
            raise ValueError("Input frame cannot be empty.")

        padded, scale, pad = letterbox_image(frame, self.input_size)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        outputs = self._session.run(None, {self._input_name: blob})
        return parse_yolov8_output(
            outputs[0],
            orig_shape=frame.shape[:2],
            scale=scale,
            pad=pad,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
        )
