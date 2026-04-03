"""
modules/object_detector.py
--------------------------
YOLOv8-based object detector using the Ultralytics library.
Detects people, vehicles, and potentially dangerous objects.
Provides descriptive labels for alert generation.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from utils.logger import get_logger

logger = get_logger("ObjectDetector")


@dataclass
class Detection:
    """Single object detection result.

    Attributes:
        bbox:        (x1, y1, x2, y2) bounding-box in pixel coordinates.
        class_id:    COCO class integer id.
        class_name:  Human-readable class label (e.g. 'person').
        confidence:  Detection confidence in [0, 1].
        is_threat:   Whether this detection is a threat class (weapon etc).
    """
    bbox:       tuple        # (x1, y1, x2, y2)
    class_id:   int
    class_name: str
    confidence: float
    is_threat:  bool = False

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self):
        return self.width * self.height

    def describe(self) -> str:
        """Generate a human-readable description of this detection."""
        threat_str = " [THREAT]" if self.is_threat else ""
        return f"{self.class_name}{threat_str} (conf: {self.confidence:.0%})"


class ObjectDetector:
    """YOLOv8-based object detector.

    Args:
        model_path:      Path to YOLOv8 weights file.
        conf:            Minimum confidence threshold.
        iou:             NMS IoU threshold.
        allowed_classes: List of COCO class IDs to keep.
        threat_classes:  List of COCO class IDs considered threats.
        device:          Inference device.
    """

    _DEFAULT_CLASSES = [0, 1, 2, 3, 5, 7, 24, 25, 26, 43, 76]
    _DEFAULT_THREATS = [43, 76]

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf: float = 0.40,
        iou: float = 0.45,
        allowed_classes: Optional[List[int]] = None,
        threat_classes: Optional[List[int]] = None,
        device: str = "cpu",
    ):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is not installed. Run: pip install ultralytics"
            ) from exc

        self.conf           = conf
        self.iou            = iou
        self.allowed_classes = allowed_classes or self._DEFAULT_CLASSES
        self.threat_classes  = threat_classes or self._DEFAULT_THREATS
        self.device         = device

        logger.info("Loading YOLO model: %s on device=%s", model_path, device)
        self._model = YOLO(model_path)
        self._names = self._model.names   # {class_id: name}
        logger.info("YOLO model loaded. %d classes available.", len(self._names))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame.

        Returns:
            List of Detection objects filtered to allowed classes.
        """
        results = self._model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.allowed_classes:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                name = self._names.get(cls_id, str(cls_id))
                is_threat = cls_id in self.threat_classes

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=cls_id,
                        class_name=name,
                        confidence=conf,
                        is_threat=is_threat,
                    )
                )

        return detections

    def get_scene_description(self, detections: List[Detection]) -> str:
        """Generate a human-readable description of all detections."""
        if not detections:
            return "No objects detected"

        counts = {}
        threats = []
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
            if det.is_threat:
                threats.append(det.class_name)

        parts = [f"{count} {name}{'s' if count > 1 else ''}"
                 for name, count in counts.items()]
        desc = "Detected: " + ", ".join(parts)

        if threats:
            desc += f" | ⚠ THREAT: {', '.join(set(threats))} detected!"

        return desc
