"""
utils/visualizer.py
--------------------
Low-level frame annotation helpers used by the pipeline.
Draws bounding boxes, labels, alert banners, trajectory trails,
FPS counters, and anomaly score overlays on video frames.
"""

import cv2
import numpy as np
from typing import Tuple


def draw_bbox(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw a rectangle on *frame* in-place and return it."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    return frame


def draw_label(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int],
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    font_scale: float = 0.55,
    thickness: int = 1,
) -> np.ndarray:
    """Draw a text label with a filled background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(position[0]), int(position[1])
    cv2.rectangle(frame, (x, y - th - baseline), (x + tw, y + baseline), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame


def draw_alert_banner(
    frame: np.ndarray,
    text: str = "ANOMALY DETECTED",
    alert_type: str = "danger",
    alpha: float = 0.55,
) -> np.ndarray:
    """Overlay a semi-transparent banner at the top of the frame.

    Args:
        alert_type: 'danger' (red), 'warning' (orange), or 'info' (blue)
    """
    h, w = frame.shape[:2]
    banner_h = 48
    overlay = frame.copy()

    colors = {
        "danger":  (0, 0, 200),
        "warning": (0, 140, 255),
        "info":    (200, 100, 0),
    }
    banner_color = colors.get(alert_type, (0, 0, 200))

    cv2.rectangle(overlay, (0, 0), (w, banner_h), banner_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.85
    thickness = 2
    (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
    tx = (w - tw) // 2
    ty = banner_h - 12
    cv2.putText(frame, text, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def draw_trajectory(
    frame: np.ndarray,
    history: list,
    color: Tuple[int, int, int],
    max_points: int = 30,
) -> np.ndarray:
    """Draw the trajectory trail of a tracked object."""
    points = history[-max_points:]
    if len(points) >= 2:
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], False, color, 2, cv2.LINE_AA)
    return frame


def put_fps_counter(
    frame: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
) -> np.ndarray:
    """Render an FPS counter in the top-left corner."""
    text = f"FPS: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, position, font, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def put_anomaly_score(
    frame: np.ndarray,
    score: float,
    position: Tuple[int, int] = (10, 60),
) -> np.ndarray:
    """Render the anomaly probability score on the frame."""
    text = f"Anomaly Score: {score:.3f}"
    color = (0, 0, 255) if score > 0.5 else (0, 220, 0)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame


def put_object_count(
    frame: np.ndarray,
    count: int,
    position: Tuple[int, int] = (10, 90),
) -> np.ndarray:
    """Render the detected object count."""
    text = f"Objects: {count}"
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2, cv2.LINE_AA)
    return frame


def put_status_text(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 120),
    color: Tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Render status text on the frame."""
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame
