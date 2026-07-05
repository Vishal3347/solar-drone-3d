import numpy as np

# Approximate camera intrinsics
# Replace these with your drone camera values if known.
def get_camera_matrix(width, height):
    fx = width
    fy = width

    cx = width / 2
    cy = height / 2

    return fx, fy, cx, cy