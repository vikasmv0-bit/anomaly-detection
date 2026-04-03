"""
models/train.py
----------------
Training script for the BiLSTM anomaly detection model.

Features:
  - Class-weighted BCE loss  (handles normal/anomaly imbalance)
  - ReduceLROnPlateau scheduler
  - Early stopping (patience=15)
  - Saves best checkpoint by val AUC

Usage:
    python models/train.py --cache data/feature_cache/merged_full
    python models/train.py --cache data/feature_cache/merged_full --epochs 100 --lr 0.0005
"""

from __future__ import annotations

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.preprocess import CachedSequenceDataset
from models.bilstm_model import BiLSTMClassifier
from utils.logger import get_logger
from utils.metrics import compute_auc

logger = get_logger("Train")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_pos_weight(labels_array: np.ndarray) -> float:
    """Compute BCE pos_weight = n_negative / n_positive for class balancing."""
    n_pos = max(float(labels_array.sum()), 1.0)
    n_neg = max(float((labels_array == 0).sum()), 1.0)
    return n_neg / n_pos


def train(args):
    cfg = Config()
    set_seed(cfg.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    # ── Dataset ────────────────────────────────────────────────────────────
    full_ds = CachedSequenceDataset(cache_dir=args.cache)
    n_total = len(full_ds)
    n_train = int(n_total * cfg.TRAIN_SPLIT)
    n_val   = n_total - n_train

    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.SEED),
    )
    logger.info("Split: %d train / %d val  (total=%d)", n_train, n_val, n_total)

    # Class balance stats
    all_labels = full_ds.labels
    n_normal  = int((all_labels == 0).sum())
    n_anomaly = int((all_labels == 1).sum())
    pos_weight_val = compute_pos_weight(all_labels)

    # Cap pos_weight to avoid extreme bias towards anomalies (False Positive prevention)
    pos_weight_val = min(float(pos_weight_val), 1.5)

    logger.info(
        "Class distribution – normal: %d | anomaly: %d | pos_weight (capped): %.2f",
        n_normal, n_anomaly, pos_weight_val,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = BiLSTMClassifier(
        input_size=cfg.INPUT_SIZE,
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
    ).to(device)
    logger.info("Trainable params: %d", model.parameter_count())

    # Weighted BCE loss – penalises missing anomalies more
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Rebuild last layer without sigmoid (BCEWithLogitsLoss does it internally)
    # We'll wrap with a manual sigmoid only during inference
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=8, factor=0.5, min_lr=1e-6
    )

    os.makedirs(cfg.MODEL_DIR, exist_ok=True)
    best_auc      = 0.0
    no_improve    = 0
    EARLY_STOP    = getattr(cfg, "EARLY_STOP_PATIENCE", 15)  # Enable early stopping

    logger.info("Starting training for %d epochs ...", args.epochs)
    logger.info("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0

        for seqs, labels in train_loader:
            seqs   = seqs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            # forward – get raw logits (model's sigmoid is bypassed for loss)
            logits = model.forward_logits(seqs)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            train_loss += loss.item() * seqs.size(0)

        train_loss /= max(n_train, 1)

        # ── Validation ──
        model.eval()
        all_scores = []
        all_labels_val = []

        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs  = seqs.to(device)
                preds = model(seqs).cpu().numpy().flatten()   # sigmoid output
                lbls  = labels.cpu().numpy().flatten()
                all_scores.extend(preds.tolist())
                all_labels_val.extend(lbls.tolist())

        unique_labels = set(all_labels_val)
        if len(unique_labels) > 1:
            val_auc = compute_auc(np.array(all_labels_val), np.array(all_scores))
        else:
            val_auc = float("nan")

        auc_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "n/a (one class in val)"
        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_auc=%s  lr=%.2e",
            epoch, args.epochs, train_loss, auc_str,
            optimizer.param_groups[0]["lr"],
        )

        scheduler.step(val_auc if not np.isnan(val_auc) else 0.0)

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc    = val_auc
            no_improve  = 0
            model.save(cfg.WEIGHTS_PATH)
            logger.info("** New best AUC=%.4f -- checkpoint saved", best_auc)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP:
                logger.info(
                    "Early stopping triggered after %d epochs with no improvement.",
                    EARLY_STOP,
                )
                break

    logger.info("=" * 60)
    logger.info("Training complete.  Best val AUC : %.4f", best_auc)
    logger.info("Weights saved to   : %s", cfg.WEIGHTS_PATH)


def main():
    parser = argparse.ArgumentParser(description="Train BiLSTM anomaly detector.")
    parser.add_argument("--cache",      required=True, help="Path to cached .npy feature directory.")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    args = parser.parse_args()

    cfg = Config()
    args.epochs     = args.epochs     or cfg.TRAIN_EPOCHS
    args.lr         = args.lr         or cfg.LEARNING_RATE
    args.batch_size = args.batch_size or cfg.BATCH_SIZE

    train(args)


if __name__ == "__main__":
    main()
