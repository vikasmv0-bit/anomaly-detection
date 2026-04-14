"""
extract_features.py
--------------------
Extract feature sequences from dataset videos and save to cache.
Run this BEFORE training the BiLSTM model.

Usage:
    # UCSD Pedestrian dataset
    python extract_features.py --dataset ucsd

    # Avenue dataset
    python extract_features.py --dataset avenue

    # UCF-Crime dataset (takes longer due to dataset size)
    python extract_features.py --dataset ucf

    # Custom video folder
    python extract_features.py --folder path/to/videos --label 0
"""

from __future__ import annotations

import argparse
import os
import sys
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from data.dataset_loader import (
    UCSDPedestrianLoader,
    AvenueLoader,
    UCFCrimeLoader,
    DCSASSLoader,
    iterate_video_frames,
)
from data.preprocess import build_cache, extract_features_from_video, create_sliding_windows
from utils.logger import get_logger
import numpy as np

logger = get_logger("ExtractFeatures")
cfg = Config()


def extract_ucsd(subset="UCSDped2"):
    """Extract features from UCSD Pedestrian dataset."""
    loader = UCSDPedestrianLoader(cfg.UCSD_ROOT, subset=subset)

    train_clips = loader.get_train_clips()
    test_clips  = loader.get_test_clips()

    if not train_clips and not test_clips:
        logger.error(
            "No UCSD clips found at: %s\n"
            "Download from: http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm\n"
            "Extract to: %s", cfg.UCSD_ROOT, cfg.UCSD_ROOT
        )
        return

    # For UCSD, training clips are normal, test clips contain anomalies
    # We'll process frame-by-frame from each clip directory
    videos = []

    for clip_dir in train_clips:
        # Find video files or process frame images
        video_files = glob.glob(os.path.join(clip_dir, "*.avi")) + \
                      glob.glob(os.path.join(clip_dir, "*.mp4"))
        if video_files:
            for vf in video_files:
                videos.append({"path": vf, "label": 0})
        else:
            # UCSD uses frame images — create a temporary video from them
            logger.info("Converting frames in %s to features...", clip_dir)
            frames = loader.load_clip_frames(clip_dir)
            if frames:
                # Save as temp video for processing
                temp_path = os.path.join(cfg.CACHE_DIR, "temp_clip.avi")
                os.makedirs(cfg.CACHE_DIR, exist_ok=True)
                import cv2
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"XVID"), 30, (w, h))
                for f in frames:
                    writer.write(f)
                writer.release()
                videos.append({"path": temp_path, "label": 0})

    for clip_dir in test_clips:
        video_files = glob.glob(os.path.join(clip_dir, "*.avi")) + \
                      glob.glob(os.path.join(clip_dir, "*.mp4"))
        if video_files:
            for vf in video_files:
                videos.append({"path": vf, "label": 1})
        else:
            frames = loader.load_clip_frames(clip_dir)
            if frames:
                temp_path = os.path.join(cfg.CACHE_DIR, "temp_clip_test.avi")
                os.makedirs(cfg.CACHE_DIR, exist_ok=True)
                import cv2
                h, w = frames[0].shape[:2]
                writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*"XVID"), 30, (w, h))
                for f in frames:
                    writer.write(f)
                writer.release()
                videos.append({"path": temp_path, "label": 1})

    output_dir = os.path.join(cfg.CACHE_DIR, "ucsd")
    logger.info("Processing %d video sources for UCSD...", len(videos))
    build_cache(videos, cfg, output_dir, seq_len=cfg.SEQUENCE_LENGTH, stride=10)


def extract_avenue():
    """Extract features from Avenue dataset."""
    loader = AvenueLoader(cfg.AVENUE_ROOT)

    train_vids = loader.get_training_videos()
    test_vids  = loader.get_testing_videos()

    if not train_vids and not test_vids:
        logger.error(
            "No Avenue videos found at: %s\n"
            "Download from: http://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html\n"
            "Extract to: %s", cfg.AVENUE_ROOT, cfg.AVENUE_ROOT
        )
        return

    videos = []
    for vf in train_vids:
        videos.append({"path": vf, "label": 0})
    for vf in test_vids:
        videos.append({"path": vf, "label": 1})

    output_dir = os.path.join(cfg.CACHE_DIR, "avenue")
    logger.info("Processing %d videos for Avenue...", len(videos))
    build_cache(videos, cfg, output_dir, seq_len=cfg.SEQUENCE_LENGTH, stride=10)


def extract_ucf():
    """Extract features from UCF-Crime dataset."""
    loader = UCFCrimeLoader(cfg.UCF_ROOT)

    anomaly_vids = loader.get_anomaly_videos()
    normal_vids  = loader.get_normal_videos()

    if not anomaly_vids and not normal_vids:
        logger.error(
            "No UCF-Crime videos found at: %s\n"
            "Request access from: https://www.crcv.ucf.edu/projects/real-world/\n"
            "Extract to: %s", cfg.UCF_ROOT, cfg.UCF_ROOT
        )
        return

    videos = []
    for v in normal_vids:
        videos.append({"path": v["path"], "label": 0})
    for v in anomaly_vids:
        videos.append({"path": v["path"], "label": 1})

    output_dir = os.path.join(cfg.CACHE_DIR, "ucf")
    logger.info("Processing %d videos for UCF-Crime...", len(videos))
    build_cache(videos, cfg, output_dir, seq_len=cfg.SEQUENCE_LENGTH, stride=15)


def extract_dcass():
    """Extract features from DCSASS indoor dataset."""
    # Build path to the extracted DCSASS folder
    root = os.path.join(cfg.DATA_DIR, "dcass dataset", "DCSASS Dataset")
    loader = DCSASSLoader(root)

    vids = loader.get_all_videos()

    if not vids:
        logger.error("No DCSASS videos found at: %s", root)
        return

    videos = []
    for v in vids:
        videos.append({"path": v["path"], "label": v["label"]})

    output_dir = os.path.join(cfg.CACHE_DIR, "dcass")
    logger.info("Processing %d videos for DCSASS...", len(videos))
    # Stride 10 for better temporal resolution in indoor scenes
    build_cache(videos, cfg, output_dir, seq_len=cfg.SEQUENCE_LENGTH, stride=10)


def extract_from_folder(folder: str, label: int = 0, name: str = "custom"):
    """Extract features from a folder of videos."""
    video_files = sorted(
        glob.glob(os.path.join(folder, "*.mp4")) +
        glob.glob(os.path.join(folder, "*.avi")) +
        glob.glob(os.path.join(folder, "*.mov")) +
        glob.glob(os.path.join(folder, "*.mkv"))
    )

    if not video_files:
        logger.error("No video files found in: %s", folder)
        return

    videos = [{"path": vf, "label": label} for vf in video_files]
    output_dir = os.path.join(cfg.CACHE_DIR, name)
    logger.info("Processing %d videos from folder...", len(videos))
    build_cache(videos, cfg, output_dir, seq_len=cfg.SEQUENCE_LENGTH, stride=10)


def generate_dummy_cache(output_dir: str = None):
    """Generate dummy/synthetic feature cache for testing the training pipeline.

    This creates fake data so you can test that training works even
    without a real dataset downloaded.
    """
    if output_dir is None:
        output_dir = os.path.join(cfg.CACHE_DIR, "ucsd")

    os.makedirs(output_dir, exist_ok=True)

    n_samples = 500
    seq_len = cfg.SEQUENCE_LENGTH
    feat_dim = cfg.INPUT_SIZE

    logger.info("Generating %d dummy sequences (seq_len=%d, feat_dim=%d)...",
                n_samples, seq_len, feat_dim)

    rng = np.random.default_rng(42)

    sequences = rng.standard_normal((n_samples, seq_len, feat_dim)).astype(np.float32)
    labels = np.zeros(n_samples, dtype=np.float32)

    # Make ~30% anomalous with different distribution
    n_anomaly = int(n_samples * 0.3)
    anomaly_idx = rng.choice(n_samples, n_anomaly, replace=False)
    labels[anomaly_idx] = 1.0
    # Anomalous sequences have higher variance and offset
    sequences[anomaly_idx] = (sequences[anomaly_idx] * 2.0 + 0.5).astype(np.float32)

    np.save(os.path.join(output_dir, "sequences.npy"), sequences)
    np.save(os.path.join(output_dir, "labels.npy"), labels)

    logger.info(
        "Dummy cache saved to %s — %d normal, %d anomaly",
        output_dir, n_samples - n_anomaly, n_anomaly,
    )
    print(f"\n[OK] Dummy cache created at: {output_dir}")
    print(f"     Now run: python models/train.py --cache {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract features from surveillance datasets for BiLSTM training."
    )
    parser.add_argument(
        "--dataset", choices=["ucsd", "avenue", "ucf", "dcass", "dummy"],
        help="Which dataset to process. Use 'dummy' to generate synthetic test data."
    )
    parser.add_argument("--folder", help="Path to a folder of video files.")
    parser.add_argument("--label", type=int, default=0, help="Label for --folder videos (0=normal, 1=anomaly).")
    parser.add_argument("--name", default="custom", help="Cache name for --folder mode.")
    args = parser.parse_args()

    if args.dataset == "ucsd":
        extract_ucsd()
    elif args.dataset == "avenue":
        extract_avenue()
    elif args.dataset == "ucf":
        extract_ucf()
    elif args.dataset == "dcass":
        extract_dcass()
    elif args.dataset == "dummy":
        generate_dummy_cache()
    elif args.folder:
        extract_from_folder(args.folder, args.label, args.name)
    else:
        print("Please specify --dataset or --folder. Use --dataset dummy to test the pipeline.")
        parser.print_help()


if __name__ == "__main__":
    main()
