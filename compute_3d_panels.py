from pathlib import Path
import cv2
import csv

from camera import get_camera_matrix

# -----------------------------
# CHANGE THESE PATHS
# -----------------------------

IMAGE_DIR = Path(r"C:\Users\visha\Downloads\Solar Panel.v1i.yolov5pytorch\train\images")

LABEL_DIR = Path(r"C:\Users\visha\Downloads\Solar Panel.v1i.yolov5pytorch\train\labels")
DEPTH_DIR = Path(r"C:\Users\visha\Downloads\Solar Panel.v1i.yolov5pytorch\train\depth")

OUTPUT_CSV = "panel_coordinates.csv"

# -----------------------------

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

            cls, x, y, bw, bh = map(float, line.split())

            px = int(x * w)
            py = int(y * h)

            z = float(depth[py, px]) / 255.0

            X = (px - cx0) * z / fx
            Y = (py - cy0) * z / fy
            Z = z

            rows.append([
                image_name,
                panel_id,
                round(X,4),
                round(Y,4),
                round(Z,4)
            ])

            panel_id += 1

with open(OUTPUT_CSV,"w",newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "image",
        "panel_id",
        "x",
        "y",
        "z"
    ])

    writer.writerows(rows)

print("Done.")
print("Panels:",len(rows))