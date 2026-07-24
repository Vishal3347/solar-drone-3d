"""
Loads solar panel pixel locations for the RL pipeline (env.py) and for
inference reporting.

IMPORTANT: panel_coordinates_3d_exact_detailed.csv contains detections
from 379 SEPARATE drone photos. Each photo is its own independent view
of a different physical section of the farm, so panel pixel locations
must stay grouped PER IMAGE -- merging all 379 images' panels onto one
shared grid would incorrectly treat unrelated photos as one physical
scene. load_grouped_by_image() below preserves that separation and is
what env.py uses.

Two loading modes are supported:

1. DEFAULT (works out of the box): loads panel locations from the
   precomputed CSV shipped with this repo
   (config.PANEL_COORDS_CSV = panel_coordinates_3d_exact_detailed.csv).

2. LIVE YOLO INFERENCE (optional): if you have a trained YOLOv8 model
   and want to run detection on a folder of your own YOLO-format
   label .txt files (e.g. Ultralytics' `runs/detect/predict/labels`),
   call load_from_yolo_labels() with that folder instead.
"""

from pathlib import Path
import pandas as pd

import config


def load_grouped_by_image(csv_path=config.PANEL_COORDS_CSV):
    """
    Load panel pixel locations grouped per source image.
    Returns: {image_name: [(px, py), (px, py), ...], ...}
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"[read_yolo] CSV not found: {csv_path} -- returning empty data")
        return {}

    df = pd.read_csv(csv_path)

    if config.PIXEL_X_COL not in df.columns or config.PIXEL_Y_COL not in df.columns:
        print(
            f"[read_yolo] CSV '{csv_path}' has no "
            f"'{config.PIXEL_X_COL}'/'{config.PIXEL_Y_COL}' columns -- returning empty data"
        )
        return {}

    grouped = {}
    for image_name, group in df.groupby("image"):
        grouped[image_name] = list(
            zip(group[config.PIXEL_X_COL].astype(int), group[config.PIXEL_Y_COL].astype(int))
        )

    total_panels = sum(len(v) for v in grouped.values())
    print(f"[read_yolo] Loaded {total_panels} panels across {len(grouped)} images from {csv_path}")
    return grouped


def load_from_csv(csv_path=config.PANEL_COORDS_CSV):
    """Load a flat (px, py) list across ALL images (kept for backward compatibility /
    reporting only -- do NOT use this for the RL env, since it merges unrelated photos
    onto one grid. Use load_grouped_by_image() instead)."""
    grouped = load_grouped_by_image(csv_path)
    flat = [loc for locs in grouped.values() for loc in locs]
    return flat


def load_from_yolo_labels(labels_folder, image_width=config.ENV_SIZE, image_height=config.ENV_SIZE):
    """Load (px, py) pixel locations from a single folder of raw YOLO label .txt files
    (one image's worth of detections)."""
    labels_folder = Path(labels_folder)
    locations = []

    if not labels_folder.exists():
        print(f"[read_yolo] Labels folder not found: {labels_folder} -- returning empty list")
        return locations

    for txt_file in labels_folder.glob("*.txt"):
        with open(txt_file) as f:
            for line in f:
                if not line.strip():
                    continue
                cls, x, y, w, h = map(float, line.split())
                px = int(x * image_width)
                py = int(y * image_height)
                locations.append((px, py))

    print(f"[read_yolo] Loaded {len(locations)} panel locations from {labels_folder}")
    return locations


# Populate at import time so env.py / reporting scripts can use immediately.
panel_locations_by_image = load_grouped_by_image()
panel_locations = [loc for locs in panel_locations_by_image.values() for loc in locs]

if __name__ == "__main__":
    print(f"Images: {len(panel_locations_by_image)}")
    print(f"Total panels: {len(panel_locations)}")
    first_image = next(iter(panel_locations_by_image))
    print(f"Example -- '{first_image}': {panel_locations_by_image[first_image]}")