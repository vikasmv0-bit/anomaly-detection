"""
modules/feature_extractor.py
-----------------------------
Combines MobileNetV2 spatial features with motion features to produce
a fixed-size scene feature vector for BiLSTM anomaly detection.

Feature layout (INPUT_SIZE = SPATIAL_FEATURE_DIM + MOTION_FEATURE_DIM = 162):
  - MobileNet spatial features (mean-pooled): 128 dims
  - Per-track motion features (padded/truncated to N_MAX_TRACKS):
    [cx_norm, cy_norm, w_norm, h_norm, speed, angle] × N_MAX_TRACKS = 30
  - Scene-level features: [num_objects, avg_speed, max_speed, density] = 4
"""

from __future__ import annotations

import math
import cv2
import numpy as np
import torch
from typing import List

from modules.object_tracker import Track
from utils.logger import get_logger

logger = get_logger("FeatureExtractor")


class FeatureExtractor:
    """Compute combined spatial + motion feature vectors from tracked objects.

    Args:
        config:       Project Config object.
        frame_width:  Width of the source video frame in pixels.
        frame_height: Height of the source video frame in pixels.
        device:       Torch device for MobileNet inference.
    """

    FEATURES_PER_TRACK = 6

    def __init__(self, config, frame_width: int = 640, frame_height: int = 480, device: str = "cpu"):
        self.cfg     = config
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.device  = torch.device(device)
        self._prev_centers: dict = {}   # track_id → prev (cx, cy)

        # Initialize MobileNet extractor
        self._mobilenet = None
        self._mobilenet_ready = False
        try:
            from models.mobilenet_extractor import MobileNetFeatureExtractor
            self._mobilenet = MobileNetFeatureExtractor(
                output_dim=config.SPATIAL_FEATURE_DIM,
                pretrained=True,
                freeze_backbone=True,
                proj_path=config.MOBILENET_PROJ_PATH,
            ).to(self.device)
            self._mobilenet.eval()
            self._mobilenet_ready = True
            logger.info("MobileNet feature extractor initialized.")
        except Exception as e:
            logger.warning("MobileNet init failed: %s. Using zero spatial features.", e)

    def extract(self, tracks: List[Track], frame: np.ndarray = None) -> np.ndarray:
        """Build a combined feature vector for the current frame.

        Args:
            tracks: Confirmed tracks from ObjectTracker.
            frame:  BGR frame for MobileNet crop extraction (optional).

        Returns:
            np.ndarray of shape (INPUT_SIZE,) with float32 values.
        """
        # ── Spatial features from MobileNet ──────────────────────────────────
        spatial_features = np.zeros(self.cfg.SPATIAL_FEATURE_DIM, dtype=np.float32)

        if self._mobilenet_ready and frame is not None and len(tracks) > 0:
            try:
                crops = self._extract_crops(frame, tracks)
                if crops:
                    with torch.no_grad():
                        mean_feat = self._mobilenet.extract_mean_feature(crops, self.device)
                    spatial_features = mean_feat.cpu().numpy()
            except Exception as e:
                logger.debug("MobileNet extraction failed: %s", e)

        # ── Motion features ──────────────────────────────────────────────────
        motion_features = self._compute_motion_features(tracks)

        # ── Concatenate ──────────────────────────────────────────────────────
        combined = np.concatenate([spatial_features, motion_features])

        # Update previous-center cache
        self._prev_centers = {
            trk.track_id: trk.center for trk in tracks
        }

        return combined

    def _extract_crops(self, frame: np.ndarray, tracks: List[Track]) -> List[torch.Tensor]:
        """Extract and preprocess image crops for each tracked object."""
        crops = []
        h, w = frame.shape[:2]

        for trk in tracks[:self.cfg.N_MAX_TRACKS]:
            x1, y1, x2, y2 = [int(v) for v in trk.bbox]
            # Clamp to frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 < 10 or y2 - y1 < 10:
                continue

            crop = frame[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, (224, 224))
            # Convert BGR → RGB and HWC → CHW
            crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
            crop_tensor = torch.from_numpy(crop_rgb).permute(2, 0, 1).float()
            crops.append(crop_tensor)

        return crops

    def _compute_motion_features(self, tracks: List[Track]) -> np.ndarray:
        """Compute motion feature vector from tracks."""
        n_max = self.cfg.N_MAX_TRACKS
        motion = np.zeros(self.cfg.MOTION_FEATURE_DIM, dtype=np.float32)

        speeds = []
        track_slice = tracks[:n_max]

        for i, trk in enumerate(track_slice):
            cx, cy = trk.center
            w, h = trk.wh

            # Normalise positions and sizes
            cx_n = cx / max(self.frame_w, 1)
            cy_n = cy / max(self.frame_h, 1)
            w_n  = w  / max(self.frame_w, 1)
            h_n  = h  / max(self.frame_h, 1)

            # Speed & angle from previous center
            speed, angle = self._compute_motion_vector(trk.track_id, cx, cy)
            speeds.append(speed)

            offset = i * self.FEATURES_PER_TRACK
            motion[offset + 0] = cx_n
            motion[offset + 1] = cy_n
            motion[offset + 2] = w_n
            motion[offset + 3] = h_n
            motion[offset + 4] = speed
            motion[offset + 5] = angle

        # ── Scene-level features (last 4 elements) ────────────────────────────
        scene_start = n_max * self.FEATURES_PER_TRACK
        num_obj     = len(tracks)
        avg_speed   = float(np.mean(speeds)) if speeds else 0.0
        max_speed   = float(np.max(speeds))  if speeds else 0.0

        area_covered = sum((t.wh[0] * t.wh[1]) for t in tracks)
        frame_area  = max(self.frame_w * self.frame_h, 1)
        density     = min(area_covered / frame_area, 1.0)

        motion[scene_start + 0] = num_obj / max(n_max, 1)
        motion[scene_start + 1] = avg_speed
        motion[scene_start + 2] = max_speed
        motion[scene_start + 3] = float(density)

        return motion

    def _compute_motion_vector(self, track_id: int, cx: float, cy: float):
        """Return (speed, angle) for a track."""
        prev = self._prev_centers.get(track_id)
        if prev is None:
            return 0.0, 0.0
        dx = cx - prev[0]
        dy = cy - prev[1]
        speed = math.hypot(dx, dy)
        angle = math.atan2(dy, dx) % (2 * math.pi)
        return speed, angle
