"""
app.py
-------
Flask web application for the Real-Time Surveillance Anomaly Detection System.
Supports multiple concurrent streams (local webcam, IP cameras, uploaded videos).
"""
from __future__ import annotations

import os
import sys
import time
import json
import uuid
import threading
import cv2
import numpy as np

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, Response, request, jsonify, send_file
from config.config import Config
from modules.object_tracker import ObjectTracker
from modules.feature_extractor import FeatureExtractor
from modules.anomaly_detector import AnomalyDetector
from modules.video_stream import VideoStream, FileVideoProcessor
from utils.logger import get_logger
from utils import visualizer as viz

logger = get_logger("FlaskApp", log_dir=os.path.join(os.path.dirname(__file__), "logs"))

# ── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
cfg = Config()

app.config["SECRET_KEY"] = cfg.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = cfg.MAX_CONTENT_LENGTH

os.makedirs(cfg.UPLOAD_DIR, exist_ok=True)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(cfg.LOG_DIR, exist_ok=True)

# ── Global State ─────────────────────────────────────────────────────────────
detector = None

def _init_detector():
    """Initialize the YOLOv8 detector globally (lazy load)."""
    global detector
    if detector is not None:
        return detector
    try:
        from modules.object_detector import ObjectDetector
        detector = ObjectDetector(
            model_path=cfg.YOLO_MODEL,
            conf=cfg.DETECTION_CONF,
            iou=cfg.DETECTION_IOU,
            allowed_classes=cfg.ALLOWED_CLASSES,
            threat_classes=cfg.THREAT_CLASSES,
        )
        logger.info("YOLOv8 detector initialized.")
    except Exception as e:
        logger.error("Failed to init YOLOv8: %s. Using stub detector.", e)
        class _StubDetector:
            def detect(self, frame): return []
            def get_scene_description(self, dets): return "Detector not available"
        detector = _StubDetector()
    return detector

def _annotate_frame(frame, tracks, is_anomaly, score, description, det, detections):
    """Draw annotations on an un-resized frame."""
    for trk in tracks:
        color = cfg.THREAT_COLOR if trk.is_threat else (
            cfg.ANOMALY_COLOR if is_anomaly else cfg.NORMAL_COLOR
        )
        viz.draw_bbox(frame, trk.bbox, color, cfg.LINE_THICKNESS)
        label = f"ID:{trk.track_id} {trk.class_name} {trk.confidence:.0%}"
        if trk.is_threat:
            label = f"⚠ {label}"
        x1, y1 = int(trk.bbox[0]), int(trk.bbox[1])
        viz.draw_label(frame, label, (x1, max(0, y1 - 5)),
                      cfg.TEXT_COLOR, color, cfg.FONT_SCALE)
        viz.draw_trajectory(frame, trk.history, color)

    viz.put_anomaly_score(frame, score)
    viz.put_object_count(frame, len(tracks))

    if is_anomaly:
        alert_type = "danger" if score > 0.8 else "warning"
        viz.draw_alert_banner(frame, f"⚠ {description}", alert_type)

    scene_desc = det.get_scene_description(detections) if hasattr(det, 'get_scene_description') else ""
    viz.put_status_text(frame, scene_desc, (10, frame.shape[0] - 15), (200, 200, 200))
    return frame

class StreamSession:
    """Manages an independent video processing pipeline."""
    def __init__(self, stream_id, source, is_upload=False, filename=None):
        self.stream_id = stream_id
        self.source = source
        self.is_upload = is_upload
        self.filename = filename

        self.tracker = ObjectTracker(max_age=cfg.MAX_TRACK_AGE, min_hits=cfg.MIN_TRACK_HITS, iou_thresh=cfg.IOU_THRESHOLD)
        self.extractor = None  # Lazy initialized with actual frame dimensions
        self.anomaly_det = AnomalyDetector(weights_path=cfg.WEIGHTS_PATH, config=cfg)
        
        self.video_stream = None
        self.active = False
        self.thread = None
        
        self.latest_frame = None
        self.upload_state = {
            "processing": False,
            "progress": 0,
            "total_frames": 0,
            "current_frame": 0,
            "results": None,
            "output_file": None,
            "filename": filename,
        }

    def start(self):
        self.active = True
        if self.is_upload:
            self.thread = threading.Thread(target=self._process_uploaded_video, daemon=True)
        else:
            self.thread = threading.Thread(target=self._process_live_feed, daemon=True)
        self.thread.start()

    def stop(self):
        self.active = False
        if self.video_stream:
            try:
                self.video_stream.stop()
            except AttributeError:
                self.video_stream.release()

    def _process_live_feed(self):
        _init_detector()
        try:
            # Cast to int if it's a single digit (e.g. "0" for webcam)
            src = int(self.source) if str(self.source).isdigit() else self.source
            self.video_stream = VideoStream(source=src, width=cfg.FRAME_WIDTH, height=cfg.FRAME_HEIGHT)
            self.video_stream.start()

            # Initialize Extractor with fixed width and height since VideoStream resizes
            self.extractor = FeatureExtractor(config=cfg, frame_width=640, frame_height=480)

            while self.active:
                ret, frame = self.video_stream.read()
                if not ret or frame is None:
                    time.sleep(0.05)
                    continue

                try:
                    detections = detector.detect(frame)
                    tracks = self.tracker.update(detections)
                    feat_vec = self.extractor.extract(tracks, frame)
                    is_anomaly, score, desc = self.anomaly_det.predict(feat_vec, tracks, detections)

                    annotated = _annotate_frame(frame.copy(), tracks, is_anomaly, score, desc, detector, detections)
                    self.latest_frame = annotated

                except Exception as e:
                    logger.error("Live processing error: %s", e)
                    self.latest_frame = frame

                time.sleep(0.01)

        except Exception as e:
            logger.error("Stream %s failed: %s", self.stream_id, e)
            self.active = False

    def _process_uploaded_video(self):
        _init_detector()
        self.upload_state["processing"] = True
        
        try:
            self.video_stream = FileVideoProcessor(self.source)
            self.upload_state["total_frames"] = self.video_stream.total_frames
            
            # Initialize Extractor with original video dimensions
            self.extractor = FeatureExtractor(config=cfg, frame_width=self.video_stream.width, frame_height=self.video_stream.height)

            out_filename = f"annotated_{self.filename}"
            if out_filename.endswith(".avi") == False:
                out_filename = out_filename.rsplit(".", 1)[0] + ".avi"
            output_path = os.path.join(cfg.OUTPUT_DIR, out_filename)

            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out_fps = self.video_stream.fps / max(cfg.FRAME_SKIP, 1)
            writer = cv2.VideoWriter(output_path, fourcc, out_fps, (self.video_stream.width, self.video_stream.height))

            frame_idx = 0
            start_time = time.time()
            sleep_time = 1.0 / out_fps if out_fps > 0 else 0.03

            while self.active:
                ret, frame = self.video_stream.read()
                if not ret:
                    break

                if frame_idx % max(cfg.FRAME_SKIP, 1) != 0:
                    frame_idx += 1
                    continue

                try:
                    detections = detector.detect(frame)
                    tracks = self.tracker.update(detections)
                    feat_vec = self.extractor.extract(tracks, frame)
                    is_anomaly, score, desc = self.anomaly_det.predict(feat_vec, tracks, detections)

                    annotated = _annotate_frame(frame.copy(), tracks, is_anomaly, score, desc, detector, detections)
                    self.latest_frame = annotated
                    writer.write(annotated)
                except Exception as e:
                    logger.error("Upload processing error: %s", e)

                frame_idx += 1
                self.upload_state["current_frame"] = frame_idx
                self.upload_state["progress"] = int(100 * frame_idx / max(self.video_stream.total_frames, 1))

                # Sleep to simulate real-time playback for the live mjPEG feed
                time_to_process = time.time() - start_time
                expected_time = (frame_idx / max(cfg.FRAME_SKIP, 1)) * sleep_time
                if expected_time > time_to_process:
                    time.sleep(expected_time - time_to_process)

            elapsed = time.time() - start_time
            writer.release()
            self.video_stream.release()

            stats = self.anomaly_det.get_stats()
            stats["processing_fps"] = round(frame_idx / max(elapsed, 0.01), 1)
            stats["elapsed_seconds"] = round(elapsed, 1)

            self.upload_state["processing"] = False
            self.upload_state["progress"] = 100
            self.upload_state["results"] = {
                "stats": stats,
                "alerts": self.anomaly_det.get_recent_alerts(50),
                "scores": self.anomaly_det.frame_scores[-200:],
            }
            self.upload_state["output_file"] = out_filename
            logger.info("Upload processing complete: %d frames in %.1fs", frame_idx, elapsed)

        except Exception as e:
            logger.error("Upload process error: %s", e)
            self.upload_state["processing"] = False
            self.upload_state["results"] = {"error": str(e)}

        self.active = False

active_streams = {}

# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/start_camera", methods=["POST"])
def start_camera():
    source = request.json.get("source", "0") if request.json else "0"
    stream_id = f"cam_{uuid.uuid4().hex[:8]}"
    
    session = StreamSession(stream_id, source, is_upload=False)
    session.start()
    active_streams[stream_id] = session
    
    return jsonify({"status": "started", "stream_id": stream_id})

@app.route("/stop_camera/<stream_id>")
def stop_camera(stream_id):
    if stream_id in active_streams:
        active_streams[stream_id].stop()
        del active_streams[stream_id]
        return jsonify({"status": "stopped"})
    return jsonify({"status": "error", "message": "Stream not found"}), 404

def _generate_frames(stream_id):
    while True:
        session = active_streams.get(stream_id)
        if not session or not session.active and session.latest_frame is None:
            # End stream if session is invalid or stopped and no final frame
            break
            
        frame = session.latest_frame
        if frame is None:
            time.sleep(0.1)
            continue

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        time.sleep(0.03)
        
        if not session.active:
            # Yield the final frame a few times then break
            break

@app.route("/video_feed/<stream_id>")
def video_feed(stream_id):
    if stream_id not in active_streams:
        return "Stream not found", 404
    return Response(_generate_frames(stream_id), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}:
        return jsonify({"error": "Unsupported format."}), 400

    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(cfg.UPLOAD_DIR, filename)
    file.save(filepath)
    
    stream_id = f"upload_{uuid.uuid4().hex[:8]}"
    session = StreamSession(stream_id, filepath, is_upload=True, filename=file.filename)
    session.start()
    active_streams[stream_id] = session

    return jsonify({"status": "processing", "stream_id": stream_id, "filename": file.filename})

@app.route("/upload_progress/<stream_id>")
def upload_progress(stream_id):
    session = active_streams.get(stream_id)
    if not session or not session.is_upload:
        return jsonify({"error": "Upload stream not found"}), 404
    return jsonify(session.upload_state)

@app.route("/alerts")
def get_alerts():
    all_alerts = {}
    
    for sid, session in list(active_streams.items()):
        if session.anomaly_det:
            all_alerts[sid] = {
                "alerts": session.anomaly_det.get_recent_alerts(30),
                "stats": session.anomaly_det.get_stats(),
                "scores": session.anomaly_det.frame_scores[-100:],
                "source": session.filename if session.is_upload else f"Camera ({session.source})",
                "is_upload": session.is_upload,
                "upload_state": session.upload_state if session.is_upload else None
            }
            
    return jsonify({"streams": all_alerts})

@app.route("/download/<filename>")
def download_file(filename):
    filepath = os.path.join(cfg.OUTPUT_DIR, filename)
    if os.path.isfile(filepath):
        return send_file(filepath, as_attachment=True)
    return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    logger.info("Starting Multi-Stream Anomaly Detection System...")
    app.run(host=cfg.FLASK_HOST, port=cfg.FLASK_PORT, debug=cfg.FLASK_DEBUG, threaded=True, use_reloader=False)
