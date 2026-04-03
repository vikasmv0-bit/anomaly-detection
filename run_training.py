"""
run_training.py
----------------
Full training pipeline:
  Step 1 – Extract features from UCSD Ped1 + Ped2 (train + test clips)
  Step 2 – Extract features from Avenue Dataset (train + test videos)
  Step 3 – Merge all caches
  Step 4 – Train BiLSTM with proper class-weighted loss & more epochs

Datasets used:
  UCSDped1 : 34 train (normal) + 36 test (anomaly) clips
  UCSDped2 : 16 train (normal) + 12 test (anomaly) clips
  Avenue   : 16 train (normal) + 21 test (anomaly) videos

Run:
    python run_training.py
"""

from __future__ import annotations
import os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from config.config import Config
from data.dataset_loader import UCSDPedestrianLoader, AvenueLoader
from data.preprocess import build_cache
from utils.logger import get_logger

logger = get_logger("RunTraining")
cfg    = Config()

# ── Paths ──────────────────────────────────────────────────────────────────
UCSD_ROOT   = os.path.join(cfg.DATA_DIR, "UCSD_Anomaly_Dataset",
                           "UCSD_Anomaly_Dataset.v1p2")
AVENUE_ROOT = os.path.join(cfg.DATA_DIR, "Avenue", "Avenue Dataset")

CACHE_UCSD_PED1 = os.path.join(cfg.CACHE_DIR, "ucsd_ped1")
CACHE_UCSD_PED2 = os.path.join(cfg.CACHE_DIR, "ucsd_ped2")
CACHE_AVENUE    = os.path.join(cfg.CACHE_DIR, "avenue")
CACHE_MERGED    = os.path.join(cfg.CACHE_DIR, "merged_full")

# Known anomaly test clips for UCSDped1 (frames contain bikes/carts/wheelchairs)
# All test clips contain at least some anomaly frames – label = 1
# Train clips are anomaly-free – label = 0
UCSD_TEST_LABEL  = 1
UCSD_TRAIN_LABEL = 0


import cv2

def frames_to_temp_video(frames, out_path):
    """Write a list of (H,W,3) frames → temporary .avi file."""
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path, fourcc, 10, (w, h))
    for f in frames:
        if len(f.shape) == 2:          # grayscale → BGR
            f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
        writer.write(f)
    writer.release()
    return out_path


# -- Step 1: UCSD ped1 ------------------------------------------------------
def prepare_ucsd_subset(subset_name: str, cache_dir: str) -> str:
    print(f"\n  >> Processing {subset_name} ...")
    loader = UCSDPedestrianLoader(UCSD_ROOT, subset=subset_name)

    train_clips = loader.get_train_clips()
    test_clips  = loader.get_test_clips()

    tmp_dir = os.path.join(cfg.CACHE_DIR, f"_tmp_{subset_name.lower()}")
    os.makedirs(tmp_dir, exist_ok=True)

    videos = []

    for i, clip_dir in enumerate(train_clips):
        frames = loader.load_clip_frames(clip_dir)
        if frames:
            tmp = os.path.join(tmp_dir, f"train_{i:03d}.avi")
            if not os.path.isfile(tmp):          # skip if already converted
                frames_to_temp_video(frames, tmp)
            videos.append({"path": tmp, "label": UCSD_TRAIN_LABEL})

    for i, clip_dir in enumerate(test_clips):
        frames = loader.load_clip_frames(clip_dir)
        if frames:
            tmp = os.path.join(tmp_dir, f"test_{i:03d}.avi")
            if not os.path.isfile(tmp):
                frames_to_temp_video(frames, tmp)
            videos.append({"path": tmp, "label": UCSD_TEST_LABEL})

    n_norm   = sum(1 for v in videos if v["label"] == 0)
    n_anom   = sum(1 for v in videos if v["label"] == 1)
    print(f"    {subset_name}: {len(videos)} clips  "
          f"({n_norm} normal train | {n_anom} anomaly test)")

    build_cache(videos, cfg, cache_dir,
                seq_len=cfg.SEQUENCE_LENGTH, stride=5)      # stride=5 → more sequences
    return cache_dir


# -- Step 2: Avenue ---------------------------------------------------------
def prepare_avenue() -> str:
    print("\n  >> Processing Avenue Dataset ...")
    loader = AvenueLoader(AVENUE_ROOT)

    train_vids = loader.get_training_videos()
    test_vids  = loader.get_testing_videos()

    videos = []
    for vf in train_vids:
        videos.append({"path": vf, "label": 0})   # training = normal
    for vf in test_vids:
        videos.append({"path": vf, "label": 1})   # testing  = anomaly

    print(f"    Avenue: {len(train_vids)} normal training | "
          f"{len(test_vids)} anomaly testing")

    build_cache(videos, cfg, CACHE_AVENUE,
                seq_len=cfg.SEQUENCE_LENGTH, stride=5)
    return CACHE_AVENUE


# -- Step 3: Merge ----------------------------------------------------------
def merge_caches(*cache_dirs: str, output_dir: str) -> str:
    print(f"\n  >> Merging {len(cache_dirs)} caches ...")
    all_seqs, all_labels = [], []

    for d in cache_dirs:
        sq = os.path.join(d, "sequences.npy")
        lb = os.path.join(d, "labels.npy")
        if os.path.isfile(sq) and os.path.isfile(lb):
            s = np.load(sq)
            l = np.load(lb)
            all_seqs.append(s)
            all_labels.append(l)
            n0 = int((l == 0).sum())
            n1 = int((l == 1).sum())
            print(f"    {os.path.basename(d):20s}: {len(s):5d} seqs "
                  f"({n0} normal | {n1} anomaly)")
        else:
            print(f"    WARNING: no cache at {d} – skipping")

    if not all_seqs:
        raise RuntimeError("No cache data found to merge!")

    merged_seqs   = np.concatenate(all_seqs,   axis=0)
    merged_labels = np.concatenate(all_labels, axis=0)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "sequences.npy"), merged_seqs)
    np.save(os.path.join(output_dir, "labels.npy"),    merged_labels)

    n0 = int((merged_labels == 0).sum())
    n1 = int((merged_labels == 1).sum())
    print(f"\n    MERGED TOTAL : {len(merged_seqs)} sequences")
    print(f"    Normal       : {n0}  ({100*n0/len(merged_seqs):.1f}%)")
    print(f"    Anomaly      : {n1}  ({100*n1/len(merged_seqs):.1f}%)")
    print(f"    Shape        : {merged_seqs.shape}")
    print(f"    Saved to     : {output_dir}")
    return output_dir


# -- Step 4: Train ----------------------------------------------------------
def launch_training(cache_dir: str):
    print("\n  >> Launching BiLSTM training ...")
    cmd = (
        f'"{sys.executable}" models/train.py '
        f'--cache "{cache_dir}" '
        f'--epochs 100 '          # more epochs for thorough training
        f'--lr 0.0005 '           # slightly lower LR for stability
        f'--batch-size 32'
    )
    print(f"    CMD: {cmd}\n")
    ret = os.system(cmd)
    sys.exit(ret)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  FULL BiLSTM Training Pipeline")
    print("  Datasets: UCSDped1 + UCSDped2 + Avenue")
    print("=" * 65)

    # ── 1. UCSD ──
    print("\n[STEP 1/4]  UCSD Pedestrian Dataset")
    ped1_cache = prepare_ucsd_subset("UCSDped1", CACHE_UCSD_PED1)
    ped2_cache = prepare_ucsd_subset("UCSDped2", CACHE_UCSD_PED2)

    # ── 2. Avenue ──
    print("\n[STEP 2/4]  Avenue Dataset")
    avenue_cache = prepare_avenue()

    # ── 3. Merge ──
    print("\n[STEP 3/4]  Merging all caches")
    merged = merge_caches(
        ped1_cache, ped2_cache, avenue_cache,
        output_dir=CACHE_MERGED,
    )

    # ── 4. Train ──
    print("\n[STEP 4/4]  BiLSTM Training")
    launch_training(merged)


if __name__ == "__main__":
    main()
