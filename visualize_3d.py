import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("panel_coordinates.csv")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(df["x"], df["y"], df["z"], s=5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("3D Solar Panel Positions")

plt.show()