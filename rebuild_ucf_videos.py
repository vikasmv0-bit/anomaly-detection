import os
import glob
import cv2
import re
from config.config import Config

def main():
    cfg = Config()
    data_dir = cfg.UCF_ROOT
    
    print(f"Scanning for extracted .png frames in {data_dir}...")
    
    png_files = glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
    
    groups = {}
    for p in png_files:
        cat_dir = os.path.dirname(p)
        fname = os.path.basename(p)
        # Assuming format like "Abuse001_x264_10.png"
        m = re.match(r"(.*)_(\d+)\.png$", fname)
        if m:
            prefix = m.group(1)
            frame_idx = int(m.group(2))
            key = (cat_dir, prefix)
            if key not in groups:
                groups[key] = []
            groups[key].append((frame_idx, p))
            
    print(f"Found {len(groups)} unique video sequences from {len(png_files)} images.")
    
    processed = 0
    for (cat_dir, prefix), frames in groups.items():
        out_path = os.path.join(cat_dir, f"{prefix}.mp4")
        if os.path.exists(out_path):
            continue
            
        frames.sort(key=lambda x: x[0])
        
        img0 = cv2.imread(frames[0][1])
        if img0 is None:
            continue
            
        h, w = img0.shape[:2]
        
        # Depending on how the frames were sampled (e.g., 1 frame every 10 frames of a 30fps video), 
        # saving at 30 fps will make the video fast, but the anomaly model relies on relative 
        # sequences rather than absolute time, so this is fine.
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
        
        for idx, path in frames:
            img = cv2.imread(path)
            if img is not None:
                writer.write(img)
                
        writer.release()
        processed += 1
        if processed % 10 == 0:
            print(f"Rebuilt {processed}/{len(groups)} videos...")
            
    print(f"Finished rebuilding {processed} videos. You can now extract features!")

if __name__ == '__main__':
    main()
