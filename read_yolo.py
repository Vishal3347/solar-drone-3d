from pathlib import Path

panel_locations = []

labels_folder = Path(
    r"C:\Users\visha\runs\detect\predict\labels"
)

for txt_file in labels_folder.glob("*.txt"):

    with open(txt_file) as f:

        for line in f:

            cls, x, y, w, h = map(
                float,
                line.split()
            )

            px = int(x * 640)
            py = int(y * 512)

            panel_locations.append(
                (px, py)
            )

print(panel_locations)