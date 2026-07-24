
from pathlib import Path
import csv
import sys
 
import cv2
 
from camera import get_camera_matrix
import config
 
IMAGE_DIR = config.IMAGE_DIR
LABEL_DIR = config.LABEL_DIR
DEPTH_DIR = config.DEPTH_DIR
 
OUTPUT_CSV = "panel_coordinates.csv"
 
 
def main():
    if not LABEL_DIR.exists():
        print(f"[compute_3d_panels] LABEL_DIR not found: {LABEL_DIR}")
        print(
            "This step is optional -- the repo already ships with a precomputed "
            "panel_coordinates.csv. To regenerate it from your own dataset, set "
            "DATASET_DIR (or IMAGE_DIR/LABEL_DIR/DEPTH_DIR) in config.py first."
        )
        sys.exit(0)
 
    rows = []
 
    for label_file in LABEL_DIR.glob("*.txt"):
        image_name = label_file.stem + ".jpg"
 
        image_path = IMAGE_DIR / image_name
        depth_path = DEPTH_DIR / (label_file.stem + ".png")
 
        if not image_path.exists():
            print("Missing image:", image_name)
            continue
 
        if not depth_path.exists():
            print("Missing depth:", depth_path.name)
            continue
 
        image = cv2.imread(str(image_path))
        depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
 
        h, w = image.shape[:2]
 
        fx, fy, cx0, cy0 = get_camera_matrix(w, h)
 
        with open(label_file) as f:
            panel_id = 1
 
            for line in f:
                if not line.strip():
                    continue
 
                cls, x, y, bw, bh = map(float, line.split())
 
                px = int(x * w)
                py = int(y * h)
 
                z = float(depth[py, px]) / 255.0
 
                X = (px - cx0) * z / fx
                Y = (py - cy0) * z / fy
                Z = z
 
                rows.append([image_name, panel_id, round(X, 4), round(Y, 4), round(Z, 4)])
 
                panel_id += 1
 
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "panel_id", "x", "y", "z"])
        writer.writerows(rows)
 
    print("Done.")
    print("Panels:", len(rows))
 
 
if __name__ == "__main__":
    main()
 
