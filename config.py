
from pathlib import Path

# -----------------------------------------------------------------
# Model paths
# -----------------------------------------------------------------
YOLO_MODEL_PATH = "models/yolov8.pt"          # only needed for real inference
CALIBRATION_FILE = "calibration/camera_calib.npz"  # optional, camera.py works without it

# -----------------------------------------------------------------
# Detection parameters
# -----------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# -----------------------------------------------------------------
# Panel coordinate data
# -----------------------------------------------------------------
# This CSV already ships with the repo and contains real,
# precomputed 3D panel detections (pixel + world coordinates).
PANEL_COORDS_CSV = "panel_coordinates_3d_exact_detailed.csv"

# Columns in PANEL_COORDS_CSV used as the drone-frame pixel location
# of each detected panel (used by read_yolo.py / env.py).
PIXEL_X_COL = "pixel_x"
PIXEL_Y_COL = "pixel_y"

# -----------------------------------------------------------------
# Raw dataset (only needed if you want to regenerate the CSV above
# from scratch via compute_3d_panels.py). Point these at your own
# Roboflow-style export: images/, labels/ (YOLO txt), depth/ (PNG).
# -----------------------------------------------------------------
DATASET_DIR = Path("dataset")               # base folder, override as needed
IMAGE_DIR = DATASET_DIR / "images"
LABEL_DIR = DATASET_DIR / "labels"
DEPTH_DIR = DATASET_DIR / "depth"

# -----------------------------------------------------------------
# RL training
# -----------------------------------------------------------------
RL_EPISODES = 5000
RL_LEARNING_RATE = 0.001
RL_BATCH_SIZE = 32
RL_TOTAL_TIMESTEPS = 100_000

# Size of the square drone-navigation grid used by env.py (pixels).
# Matches the frame size the pixel_x/pixel_y coordinates were computed at.
ENV_SIZE = 640

# -----------------------------------------------------------------
# Output
# -----------------------------------------------------------------
OUTPUT_DIR = Path("results/")
SAVE_VISUALIZATIONS = True