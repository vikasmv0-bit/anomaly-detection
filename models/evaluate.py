"""
models/evaluate.py
-------------------
Evaluation script for the BiLSTM anomaly detection model.

Computes AUC-ROC, accuracy, precision, recall, F1, and confusion matrix.

Usage:
    python models/evaluate.py --cache data/feature_cache/ucsd --weights models/bilstm_weights.pth
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import Config
from data.preprocess import CachedSequenceDataset
from models.bilstm_model import BiLSTMClassifier
from utils.logger import get_logger
from utils.metrics import compute_auc, compute_metrics

logger = get_logger("Evaluate")


def evaluate(args):
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = BiLSTMClassifier(
        input_size=cfg.INPUT_SIZE,
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        dropout=cfg.DROPOUT,
    ).to(device)

    state = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info("Model loaded from %s", args.weights)

    # Load dataset
    dataset = CachedSequenceDataset(cache_dir=args.cache)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Run inference
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for seqs, labels in loader:
            seqs  = seqs.to(device)
            preds = model(seqs).cpu().numpy().flatten()
            lbls  = labels.cpu().numpy().flatten()
            all_scores.extend(preds.tolist())
            all_labels.extend(lbls.tolist())

    y_true   = np.array(all_labels)
    y_scores = np.array(all_scores)
    y_pred   = (y_scores >= args.threshold).astype(int)

    # Compute metrics
    auc = compute_auc(y_true, y_scores)
    metrics = compute_metrics(y_true.astype(int), y_pred)

    # Print results
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Samples:    {len(y_true)}")
    print(f"  Threshold:  {args.threshold:.2f}")
    print(f"  AUC-ROC:    {auc:.4f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1']:.4f}")
    print()
    print("  Confusion Matrix:")
    print(f"    TN={metrics['tn']}  FP={metrics['fp']}")
    print(f"    FN={metrics['fn']}  TP={metrics['tp']}")
    print("=" * 50)

    # Save results
    results_path = os.path.join(cfg.MODEL_DIR, "eval_results.txt")
    with open(results_path, "w") as f:
        f.write(f"AUC-ROC: {auc:.4f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1: {metrics['f1']:.4f}\n")
    logger.info("Results saved to %s", results_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BiLSTM anomaly detector.")
    parser.add_argument("--cache",      required=True, help="Path to cached .npy feature directory.")
    parser.add_argument("--weights",    required=True, help="Path to trained .pth checkpoint.")
    parser.add_argument("--threshold",  type=float, default=0.55, help="Anomaly threshold.")
    parser.add_argument("--batch-size", type=int,   default=32)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
