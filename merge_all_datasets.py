"""
merge_all_datasets.py
---------------------
Combines cached feature sequences from UCF, DCSASS, UCSD, and Avenue
into a single dataset for master training.
"""

import os
import numpy as np
from config.config import Config

def merge_caches():
    cfg = Config()
    output_dir = os.path.join(cfg.CACHE_DIR, "merged_full")
    os.makedirs(output_dir, exist_ok=True)

    dataset_dirs = [
        os.path.join(cfg.CACHE_DIR, "ucf"),
        os.path.join(cfg.CACHE_DIR, "dcass"),
        os.path.join(cfg.CACHE_DIR, "avenue"),
        os.path.join(cfg.CACHE_DIR, "ucsd_ped1"),
        os.path.join(cfg.CACHE_DIR, "ucsd"), # This is likely UCSDped2
    ]

    all_seqs = []
    all_labs = []

    print("[*] Merging datasets for master BiLSTM training...")

    for d in dataset_dirs:
        seq_path = os.path.join(d, "sequences.npy")
        lab_path = os.path.join(d, "labels.npy")

        if os.path.isfile(seq_path) and os.path.isfile(lab_path):
            print(f"  + Loading: {os.path.basename(d)}")
            s = np.load(seq_path)
            l = np.load(lab_path)
            
            # Simple downsampling for extremely large datasets if needed (disabled for now)
            # if len(s) > 50000:
            #     idx = np.random.choice(len(s), 50000, replace=False)
            #     s, l = s[idx], l[idx]
            
            all_seqs.append(s)
            all_labs.append(l)
            print(f"    - Sequences: {len(s)} | Anomaly rate: {100 * l.sum() / len(l):.1f}%")
        else:
            print(f"  x Skipping: {os.path.basename(d)} (Cache not found)")

    if not all_seqs:
        print("[!] No caches found to merge. Run extraction first!")
        return

    merged_seqs = np.concatenate(all_seqs, axis=0)
    merged_labs = np.concatenate(all_labs, axis=0)

    print(f"\n[*] Merged Total: {len(merged_seqs)} sequences")
    print(f"    - Shape: {merged_seqs.shape}")
    print(f"    - Overall Anomaly Rate: {100 * merged_labs.sum() / len(merged_labs):.1f}%")

    np.save(os.path.join(output_dir, "sequences.npy"), merged_seqs)
    np.save(os.path.join(output_dir, "labels.npy"), merged_labs)
    print(f"\n[OK] Master cache saved to: {output_dir}")

if __name__ == "__main__":
    merge_caches()
