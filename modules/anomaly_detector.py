"""
modules/anomaly_detector.py
----------------------------
Sliding-window BiLSTM inference engine with descriptive alert generation.

Each call to predict() appends a new feature vector to an internal ring buffer.
Once the buffer contains SEQUENCE_LENGTH vectors, the BiLSTM runs a forward pass
and returns the anomaly probability along with a description of what's happening.
"""

from __future__ import annotations

import collections
import time
import numpy as np

from utils.logger import get_logger

logger = get_logger("AnomalyDetector")


class AnomalyDetector:
    """BiLSTM-based anomaly detector with sliding-window buffer and alert system.

    Args:
        weights_path: Path to a saved .pth checkpoint, or "dummy" for random weights.
        config:       Project Config object.
        device:       Torch device string.
    """

    def __init__(self, weights_path: str, config, device: str = "cpu"):
        import torch
        from models.bilstm_model import BiLSTMClassifier

        self.cfg     = config
        self.device  = torch.device(device)
        self._buffer = collections.deque(maxlen=config.SEQUENCE_LENGTH)
        self._score_history = collections.deque(maxlen=15) # Smooth over 15 frames for stability
        self._score  = 0.0
        self._dummy  = (weights_path == "dummy")

        # Alert history
        self._alerts = []
        self._frame_scores = []
        self._frame_count = 0

        if self._dummy:
            logger.warning(
                "AnomalyDetector running in DUMMY mode. "
                "Predictions are from randomly initialised weights."
            )
            self.model = BiLSTMClassifier(
                input_size=config.INPUT_SIZE,
                hidden_size=config.HIDDEN_SIZE,
                num_layers=config.NUM_LAYERS,
                dropout=config.DROPOUT,
            ).to(self.device)
        else:
            self.model = BiLSTMClassifier.load(weights_path, config).to(self.device)

        self.model.eval()
        logger.info(
            "AnomalyDetector ready — seq_len=%d, threshold=%.2f, mode=%s",
            config.SEQUENCE_LENGTH, config.ANOMALY_THRESHOLD,
            "dummy" if self._dummy else "trained",
        )

    def predict(self, feature_vector: np.ndarray, tracks=None, detections=None):
        """Append feature vector and predict anomaly probability.

        Args:
            feature_vector: 1-D numpy array of shape (INPUT_SIZE,).
            tracks: Optional list of Track objects for description.
            detections: Optional list of Detection objects for description.

        Returns:
            Tuple (is_anomaly: bool, score: float, description: str).
        """
        import torch

        self._buffer.append(feature_vector.astype(np.float32))
        self._frame_count += 1

        if len(self._buffer) < self.cfg.SEQUENCE_LENGTH:
            return False, 0.0, "Warming up... ({}/{} frames)".format(
                len(self._buffer), self.cfg.SEQUENCE_LENGTH
            )

        # Build (1, seq_len, input_size) tensor
        seq = np.stack(list(self._buffer), axis=0)
        seq_t = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            raw_prob = self.model(seq_t).item()

        # Smooth score using moving average to prevent 1-frame false positive spikes
        self._score_history.append(raw_prob)
        prob = sum(self._score_history) / len(self._score_history)

        self._score = prob
        self._frame_scores.append({"frame": self._frame_count, "score": prob})

        is_anomaly = prob >= self.cfg.ANOMALY_THRESHOLD
        
        # Generate description
        description = self._generate_description(is_anomaly, prob, tracks, detections)

        # Record alert if anomaly
        if is_anomaly:
            alert = {
                "frame": self._frame_count,
                "score": round(prob, 4),
                "description": description,
                "timestamp": time.strftime("%H:%M:%S"),
                "type": "danger" if prob > 0.80 else "warning",
            }
            self._alerts.append(alert)
            logger.warning("[!] ANOMALY frame=%d score=%.3f: %s",
                         self._frame_count, prob, description)

        return is_anomaly, prob, description

    def _generate_description(self, is_anomaly, score, tracks=None, detections=None):
        """Generate a human-readable description with heuristics for Accident, Fighting, and Theft."""
        if not is_anomaly:
            return "Normal activity"

        parts = []
        is_accident = False
        is_fighting = False
        is_theft = False

        vehicles = {"car", "truck", "bus", "motorcycle", "bicycle"}
        accessories = {"handbag", "backpack", "suitcase", "umbrella"}

        # Heuristics using YOLO-detected and tracked objects
        if tracks and len(tracks) >= 1:
            num_persons = 0
            num_vehicles = 0
            num_accessories = 0
            fast_persons = 0
            fast_vehicles = 0

            for t in tracks:
                if t.class_name == "person":
                    num_persons += 1
                    if t.speed > 5: fast_persons += 1
                elif t.class_name in vehicles:
                    num_vehicles += 1
                    if t.speed > 8: fast_vehicles += 1
                elif t.class_name in accessories:
                    num_accessories += 1

            overlapping_persons = False
            overlapping_vehicle_person = False
            overlapping_vehicle_vehicle = False
            overlapping_person_bag = False

            for i, t1 in enumerate(tracks):
                for j in range(i + 1, len(tracks)):
                    t2 = tracks[j]
                    x_left  = max(t1.bbox[0], t2.bbox[0])
                    y_top   = max(t1.bbox[1], t2.bbox[1])
                    x_right = min(t1.bbox[2], t2.bbox[2])
                    y_bottom = min(t1.bbox[3], t2.bbox[3])
                    if x_right > x_left and y_bottom > y_top:
                        classes = {t1.class_name, t2.class_name}
                        if classes == {"person"}:
                            overlapping_persons = True
                        elif "person" in classes and len(classes.intersection(vehicles)) > 0:
                            overlapping_vehicle_person = True
                        elif len(classes.intersection(vehicles)) == 2:
                            overlapping_vehicle_vehicle = True
                        elif "person" in classes and len(classes.intersection(accessories)) > 0:
                            overlapping_person_bag = True

            # Accident: vehicles involved in collision or fast vehicle anomaly
            if (num_vehicles > 0 and (overlapping_vehicle_vehicle or overlapping_vehicle_person)) or \
               (num_vehicles > 0 and score > 0.60):
                is_accident = True

            # Fighting: persons present + high anomaly probability (speed can be unreliable at different resolutions)
            if num_persons >= 2 and score > 0.65:
                is_fighting = True
            elif num_persons >= 1 and score > 0.75:
                is_fighting = True

            # Theft: person + accessory + anomaly
            if num_persons >= 1 and num_accessories >= 1 and score > 0.60:
                is_theft = True

        # Check for YOLO threats (Weapons/Knives/etc)
        if detections:
            threats = [d for d in detections if d.is_threat]
            if threats:
                threat_names = set(d.class_name for d in threats)
                parts.append(f"⚠ THREAT: {', '.join(threat_names)} detected")

        # Prioritize the heuristic tags
        if is_accident:
            parts.append("ACCIDENT: Traffic incident or collision detected")
        elif is_fighting:
            parts.append("FIGHTING: Altercation or erratic crowd movement")
        elif is_theft:
            parts.append("THEFT/SUSPICIOUS: Rapid movement near personal belongings")
        
        # Generic fallbacks
        if not is_accident and not is_fighting and not is_theft:
            if tracks and any(t.speed > 15 for t in tracks):
                parts.append(f"{sum(1 for t in tracks if t.speed > 15)} object(s) moving rapidly")
            if tracks and len(tracks) > 5:
                parts.append(f"Crowded scene ({len(tracks)} objects)")

        if not parts:
            if score > 0.95:
                parts.append("High anomaly score — critical event detected")
            elif score > 0.90:
                parts.append("Unusual behavior pattern detected")
            else:
                return "Normal activity" # Suppress generic alerts if < 0.90 but still > threshold

        return " | ".join(parts)

    @property
    def current_score(self) -> float:
        return self._score

    @property
    def alerts(self) -> list:
        return self._alerts.copy()

    @property
    def frame_scores(self) -> list:
        return self._frame_scores.copy()

    def get_recent_alerts(self, n: int = 20) -> list:
        return self._alerts[-n:]

    def get_stats(self) -> dict:
        """Get summary statistics."""
        scores = [s["score"] for s in self._frame_scores]
        anomaly_frames = len(self._alerts)
        total_frames = self._frame_count

        return {
            "total_frames": total_frames,
            "anomaly_frames": anomaly_frames,
            "anomaly_percentage": round(100 * anomaly_frames / max(total_frames, 1), 1),
            "peak_score": round(max(scores) if scores else 0, 4),
            "avg_score": round(float(np.mean(scores)) if scores else 0, 4),
            "total_alerts": len(self._alerts),
        }

    def reset(self):
        """Clear all buffers and history."""
        self._buffer.clear()
        self._score = 0.0
        self._alerts.clear()
        self._frame_scores.clear()
        self._frame_count = 0
