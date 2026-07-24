

"""
Renders the 3D solar panel layout from panel_coordinates.csv.
 
Saves a PNG to config.OUTPUT_DIR and also opens an interactive window
when a display is available (safe to run headlessly / over SSH too).
"""
 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
 
import config
 
df = pd.read_csv("panel_coordinates.csv")
 
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")
 
sc = ax.scatter(df["x"], df["y"], df["z"], c=df["z"], cmap="viridis", s=5)
 
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"3D Solar Panel Positions ({len(df)} panels, {df['image'].nunique()} images)")
fig.colorbar(sc, ax=ax, shrink=0.6, label="Z depth")
 
if config.SAVE_VISUALIZATIONS:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.OUTPUT_DIR / "panel_coordinates_3d.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")
 
try:
    plt.show()
except Exception as e:
    print(f"(No display available to show the plot interactively: {e})")
 
