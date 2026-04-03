"""
modules/object_tracker.py
--------------------------
IoU-based centroid/bounding-box tracker that assigns persistent IDs to detections.
Pure NumPy implementation — no scipy or external tracker library needed.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from modules.object_detector import Detection
from utils.logger import get_logger

logger = get_logger("ObjectTracker")


@dataclass
class Track:
    """A single tracked object.

    Attributes:
        track_id:   Unique integer ID for this track.
        bbox:       Current (x1, y1, x2, y2) bounding box.
        class_id:   COCO class of the associated detection.
        class_name: Human-readable class name.
        confidence: Latest detection confidence.
        is_threat:  Whether this track is a threat class.
        age:        Total number of frames this track has existed.
        hits:       Number of consecutive frames with a matched detection.
        misses:     Number of consecutive frames without a matched detection.
        history:    List of (cx, cy) centers across all matched frames.
    """
    track_id:   int
    bbox:       Tuple[float, float, float, float]
    class_id:   int
    class_name: str
    confidence: float
    is_threat:  bool = False
    age:        int = 0
    hits:       int = 0
    misses:     int = 0
    history:    List[Tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def wh(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1, y2 - y1)

    def update(self, det: Detection):
        """Update this track with a new matched detection."""
        self.bbox       = det.bbox
        self.class_id   = det.class_id
        self.class_name = det.class_name
        self.confidence = det.confidence
        self.is_threat  = det.is_threat
        self.hits  += 1
        self.misses = 0
        self.age   += 1
        cx, cy = self.center
        self.history.append((cx, cy))

    def miss(self):
        """Mark a frame where this track had no matching detection."""
        self.misses += 1
        self.age    += 1

    @property
    def speed(self) -> float:
        """Compute instantaneous speed from last two history points."""
        if len(self.history) < 2:
            return 0.0
        p1 = self.history[-2]
        p2 = self.history[-1]
        return float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def _iou(boxA: tuple, boxB: tuple) -> float:
    """Compute intersection-over-union for two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter   = inter_w * inter_h

    if inter == 0:
        return 0.0

    areaA = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    areaB = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union  = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


class ObjectTracker:
    """Greedy IoU-based multi-object tracker.

    Args:
        max_age:    Frames a track survives without a detection match.
        min_hits:   Minimum consecutive matches before a track is returned.
        iou_thresh: Minimum IoU to associate a detection with an existing track.
    """

    def __init__(
        self,
        max_age: int   = 30,
        min_hits: int  = 3,
        iou_thresh: float = 0.30,
    ):
        self.max_age    = max_age
        self.min_hits   = min_hits
        self.iou_thresh = iou_thresh

        self._tracks:   Dict[int, Track] = {}
        self._next_id:  int = 1

    def update(self, detections: List[Detection]) -> List[Track]:
        """Process detections for one frame and update all tracks.

        Returns:
            List of confirmed Track objects.
        """
        track_ids  = list(self._tracks.keys())
        track_list = [self._tracks[tid] for tid in track_ids]

        matched_track_ids: set = set()
        matched_det_indices: set = set()

        # ── Greedy matching by IoU ────────────────────────────────────────────
        if track_list and detections:
            iou_matrix = np.zeros((len(track_list), len(detections)), dtype=np.float32)
            for ti, trk in enumerate(track_list):
                for di, det in enumerate(detections):
                    iou_matrix[ti, di] = _iou(trk.bbox, det.bbox)

            flat = iou_matrix.flatten()
            order = np.argsort(-flat)
            for idx in order:
                ti, di = divmod(int(idx), len(detections))
                if iou_matrix[ti, di] < self.iou_thresh:
                    break
                t_id = track_ids[ti]
                if t_id in matched_track_ids or di in matched_det_indices:
                    continue
                self._tracks[t_id].update(detections[di])
                matched_track_ids.add(t_id)
                matched_det_indices.add(di)

        # ── Missed tracks ─────────────────────────────────────────────────────
        for tid in track_ids:
            if tid not in matched_track_ids:
                self._tracks[tid].miss()

        # ── New tracks for unmatched detections ───────────────────────────────
        for di, det in enumerate(detections):
            if di not in matched_det_indices:
                new_track = Track(
                    track_id=self._next_id,
                    bbox=det.bbox,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    confidence=det.confidence,
                    is_threat=det.is_threat,
                )
                new_track.hits = 1
                new_track.history.append(new_track.center)
                self._tracks[self._next_id] = new_track
                self._next_id += 1

        # ── Remove stale tracks ───────────────────────────────────────────────
        stale = [tid for tid, trk in self._tracks.items() if trk.misses > self.max_age]
        for tid in stale:
            del self._tracks[tid]

        # ── Return confirmed tracks ───────────────────────────────────────────
        confirmed = [
            trk for trk in self._tracks.values()
            if trk.hits >= self.min_hits
        ]
        return confirmed

    def reset(self):
        """Clear all tracks (useful between clips)."""
        self._tracks.clear()
        self._next_id = 1

    @property
    def active_tracks(self) -> List[Track]:
        return list(self._tracks.values())
