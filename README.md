# 🎯 Real-Time Surveillance Anomaly Detection System

A deep learning-powered system for detecting abnormal activities in CCTV footage and live camera feeds using **YOLOv8**, **MobileNetV2**, and **BiLSTM**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Live Camera Detection** | Real-time object detection and anomaly analysis from webcam/CCTV |
| **Video Upload Analysis** | Upload recorded surveillance videos for batch processing |
| **YOLOv8 Object Detection** | Detects people, vehicles, weapons, and suspicious objects |
| **MobileNetV2 Features** | Deep spatial features extracted from detected objects |
| **BiLSTM Temporal Analysis** | Sequence-based anomaly classification using behavioral patterns |
| **Object Tracking** | Persistent ID tracking across frames with trajectory visualization |
| **Web Dashboard** | Dark-themed real-time dashboard with alerts, charts, and stats |
| **Alert System** | Descriptive alerts explaining exactly what anomalies are occurring |

## 🏗 Architecture

```
Video Input → YOLOv8 Detection → Object Tracking → Feature Extraction → BiLSTM → Anomaly Alert
                                                    ↑
                                              MobileNetV2 (spatial)
                                              + Motion features (speed, direction)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd "d:\survillence camera"
pip install -r requirements.txt
```

### 2. Run the Web Dashboard

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

### 3. Use the System

- **Live Camera**: Click "Start Camera" on the dashboard
- **Upload Video**: Switch to "Upload Video" mode and drag-drop a video file

## 📊 Datasets for Training

| Dataset | Size | Description | Download |
|---------|------|-------------|----------|
| **UCF-Crime** | ~130GB | 1900 surveillance videos, 13 anomaly types | [UCF](https://www.crcv.ucf.edu/projects/real-world/) |
| **UCSD Pedestrian** | ~1GB | Campus walkway anomaly clips | [UCSD](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm) |
| **Avenue Dataset** | ~2GB | CUHK Avenue anomaly clips | [Avenue](http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html) |

Store datasets in:
```
d:\survillence camera\data\
  ├── UCF_Crimes\
  ├── UCSD_Anomaly_Dataset\
  └── Avenue\
```

## 🧠 Training Pipeline

### Step 1: Extract Features from Dataset

```python
from config.config import Config
from data.dataset_loader import UCSDPedestrianLoader
from data.preprocess import build_cache

cfg = Config()
loader = UCSDPedestrianLoader(cfg.UCSD_ROOT, subset="UCSDped2")

# Build video list
videos = []
for clip in loader.get_train_clips():
    videos.append({"path": clip, "label": 0})  # Normal

build_cache(videos, cfg, output_dir="data/feature_cache/ucsd")
```

### Step 2: Train BiLSTM

```bash
python models/train.py --cache data/feature_cache/ucsd --epochs 50
```

### Step 3: Evaluate

```bash
python models/evaluate.py --cache data/feature_cache/ucsd --weights models/bilstm_weights.pth
```

### Step 4: Use Trained Weights

Update `config/config.py` → set `WEIGHTS_PATH` to your trained checkpoint, then restart the Flask app.

## 📁 Project Structure

```
survillence camera/
├── app.py                      # Flask web server
├── requirements.txt            # Python dependencies
├── config/
│   └── config.py               # Central configuration
├── modules/
│   ├── object_detector.py      # YOLOv8 detection
│   ├── object_tracker.py       # IoU-based tracking
│   ├── feature_extractor.py    # MobileNet + motion features
│   ├── anomaly_detector.py     # BiLSTM inference
│   └── video_stream.py         # Camera/video handling
├── models/
│   ├── bilstm_model.py         # BiLSTM architecture
│   ├── mobilenet_extractor.py  # MobileNetV2 features
│   ├── train.py                # Training script
│   └── evaluate.py             # Evaluation script
├── data/
│   ├── dataset_loader.py       # UCF/UCSD/Avenue loaders
│   └── preprocess.py           # Feature caching
├── utils/
│   ├── logger.py               # Logging
│   ├── visualizer.py           # Frame annotations
│   └── metrics.py              # AUC, F1, etc.
├── templates/
│   └── index.html              # Dashboard HTML
└── static/
    ├── css/style.css           # Dark theme styles
    └── js/app.js               # Dashboard logic
```

## 🔧 Configuration

Edit `config/config.py` to customize:

- **Detection confidence** (`DETECTION_CONF`)
- **Anomaly threshold** (`ANOMALY_THRESHOLD`)
- **Tracked classes** (`ALLOWED_CLASSES`)
- **BiLSTM architecture** (`HIDDEN_SIZE`, `NUM_LAYERS`, `SEQUENCE_LENGTH`)
- **Training hyperparameters** (`LEARNING_RATE`, `BATCH_SIZE`, `TRAIN_EPOCHS`)
