"""
config/config.py
----------------
Central configuration for the Real-Time Surveillance Anomaly Detection System.
All hyper-parameters, paths, thresholds, and model settings are managed here.
"""

import os


class Config:
    # -----------------------------------------------------------------------
    # Video Input
    # -----------------------------------------------------------------------
    VIDEO_SOURCE = 0          # 0 = webcam, or path to a video file
    FRAME_SKIP   = 1          # Process every Nth frame (1 = no skip)
    MAX_FRAMES   = None       # None = process entire video
    FRAME_WIDTH  = 640        # Resize frame width for processing
    FRAME_HEIGHT = 480        # Resize frame height for processing

    # -----------------------------------------------------------------------
    # YOLOv8 Object Detection
    # -----------------------------------------------------------------------
    YOLO_MODEL          = "yolov8s.pt"   # Upgraded to Small model for better bounding boxes
    DETECTION_CONF      = 0.20           # Lower for grainy/low-res surveillance cameras
    DETECTION_IOU       = 0.45           # NMS IoU threshold
    # COCO class IDs to keep:
    #   0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck,
    #   24=backpack, 25=umbrella, 26=handbag, 43=knife, 76=scissors
    ALLOWED_CLASSES     = [0, 1, 2, 3, 5, 7, 24, 25, 26, 43, 76]

    # Threat class IDs (weapons / suspicious items)
    THREAT_CLASSES      = [43, 76]  # knife, scissors
    THREAT_CLASS_NAMES  = {43: "knife", 76: "scissors"}

    # Human-readable labels for alert descriptions
    CLASS_LABELS = {
        0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
        5: "bus", 7: "truck", 24: "backpack", 25: "umbrella",
        26: "handbag", 43: "knife", 76: "scissors",
    }

    # -----------------------------------------------------------------------
    # Object Tracker
    # -----------------------------------------------------------------------
    MAX_TRACK_AGE   = 30   # Frames before a track is dropped
    MIN_TRACK_HITS  = 1    # Frames a track must be confirmed before output (1 = instant)
    IOU_THRESHOLD   = 0.30 # IoU for track-to-detection assignment

    # -----------------------------------------------------------------------
    # MobileNetV2 Feature Extraction
    # -----------------------------------------------------------------------
    MOBILENET_FEATURE_DIM = 1280   # MobileNetV2 penultimate layer output
    SPATIAL_FEATURE_DIM   = 128    # Projected spatial feature size
    CROP_SIZE             = 224    # Input size for MobileNet

    # -----------------------------------------------------------------------
    # Combined Feature Extraction
    # -----------------------------------------------------------------------
    N_MAX_TRACKS      = 5
    FEATURE_PER_TRACK = 6          # cx, cy, w, h, speed, angle
    SCENE_FEATURES    = 4          # num_objects, avg_speed, max_speed, density
    MOTION_FEATURE_DIM = N_MAX_TRACKS * FEATURE_PER_TRACK + SCENE_FEATURES  # 34
    # Total input = spatial (128) + motion (34) = 162
    # But MobileNet features are per-track, so we use mean-pooled (128) + motion (34)
    INPUT_SIZE = SPATIAL_FEATURE_DIM + MOTION_FEATURE_DIM  # 162

    # -----------------------------------------------------------------------
    # BiLSTM Anomaly Detector
    # -----------------------------------------------------------------------
    SEQUENCE_LENGTH = 30   # Sliding window length (frames)
    HIDDEN_SIZE     = 128  # BiLSTM hidden units per direction
    NUM_LAYERS      = 2    # Stacked BiLSTM layers
    DROPOUT         = 0.5  # High Dropout to prevent overfitting

    ANOMALY_THRESHOLD = 0.85  # Increased so indoor webcams don't trigger it constantly

    # -----------------------------------------------------------------------
    # Model / Weight Paths
    # -----------------------------------------------------------------------
    BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_DIR       = os.path.join(BASE_DIR, "models")
    WEIGHTS_PATH    = os.path.join(MODEL_DIR, "bilstm_weights.pth")
    MOBILENET_PROJ_PATH = os.path.join(MODEL_DIR, "mobilenet_projection.pth")

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    TRAIN_EPOCHS    = 100
    EARLY_STOP_PATIENCE = 15  # Stop training if val score plateaues
    BATCH_SIZE      = 32
    LEARNING_RATE   = 5e-4    # 0.0005 – more stable on larger dataset
    WEIGHT_DECAY    = 1e-3    # Strong regularisation against False Positives
    TRAIN_SPLIT     = 0.80    # 80 % train, 20 % validation
    SEED            = 42

    # -----------------------------------------------------------------------
    # Dataset Paths  (update these before training)
    # -----------------------------------------------------------------------
    DATA_DIR        = os.path.join(BASE_DIR, "data")
    UCSD_ROOT       = os.path.join(DATA_DIR, "UCSD_Anomaly_Dataset", "UCSD_Anomaly_Dataset.v1p2")
    UCF_ROOT        = os.path.join(DATA_DIR, "ucf crime data")
    AVENUE_ROOT     = os.path.join(DATA_DIR, "Avenue")
    CACHE_DIR       = os.path.join(DATA_DIR, "feature_cache")

    # -----------------------------------------------------------------------
    # Output / Logging
    # -----------------------------------------------------------------------
    LOG_DIR         = os.path.join(BASE_DIR, "logs")
    UPLOAD_DIR      = os.path.join(BASE_DIR, "uploads")
    OUTPUT_DIR      = os.path.join(BASE_DIR, "output")
    OUTPUT_VIDEO    = os.path.join(BASE_DIR, "output", "annotated.avi")
    SAVE_OUTPUT     = False
    DISPLAY         = False     # No OpenCV window in Flask mode

    # -----------------------------------------------------------------------
    # Flask
    # -----------------------------------------------------------------------
    FLASK_HOST      = "0.0.0.0"
    FLASK_PORT      = 5000
    FLASK_DEBUG     = True
    SECRET_KEY      = "surveillance-anomaly-detection-2024"
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500 MB max upload

    # -----------------------------------------------------------------------
    # Visualization
    # -----------------------------------------------------------------------
    NORMAL_COLOR    = (0, 255, 0)     # BGR green for normal tracks
    ANOMALY_COLOR   = (0, 0, 255)     # BGR red for anomaly state
    THREAT_COLOR    = (0, 0, 255)     # BGR red for threats
    WARNING_COLOR   = (0, 165, 255)   # BGR orange for warnings
    TEXT_COLOR      = (255, 255, 255) # White text
    FONT_SCALE      = 0.55
    LINE_THICKNESS  = 2
