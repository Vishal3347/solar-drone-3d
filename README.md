# Solar Drone 3D - AI-Powered Solar Panel Detection & Visualization

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A cutting-edge AI system that detects solar panels from drone imagery, extracts precise 3D coordinates, and provides real-time visualization for solar farm analysis and monitoring.

![Solar Panel Detection](https://img.shields.io/badge/AI-YOLOv8-brightblue)
![3D Visualization](https://img.shields.io/badge/Visualization-3D%20Graphics-orange)
![RL Integration](https://img.shields.io/badge/RL-Drone%20Control-red)

---

## 🎯 Overview

Solar Drone 3D is an intelligent system that:

- **Detects** solar panels in drone images using YOLOv8
- **Calculates** precise 3D coordinates of panel locations
- **Visualizes** panel layouts in 3D space
- **Trains** RL agents for drone navigation
- **Monitors** solar farm productivity and health

### Key Features

✨ **AI-Powered Detection**
- YOLOv8 neural network for accurate panel detection
- Real-time inference on drone feeds
- High accuracy with optimized model weights

🌐 **3D Coordinate Extraction**
- Converts 2D image coordinates to 3D world coordinates
- Camera calibration and transformation matrices
- Precise geolocation data

📊 **Advanced Visualization**
- Interactive 3D panel layout visualization
- Real-time drone flight path simulation
- Multi-view rendering

🤖 **Reinforcement Learning Integration**
- RL agents for optimal drone path planning
- Custom training environment
- Autonomous navigation

📈 **Comprehensive Analysis**
- Panel health monitoring
- Productivity metrics
- Performance tracking

---

## 🛠️ Technology Stack

| Technology | Purpose | Version |
|-----------|---------|---------|
| **Python** | Core language | 3.8+ |
| **YOLOv8** | Object detection | Latest |
| **OpenCV** | Image processing | 4.5+ |
| **NumPy** | Numerical computing | 1.20+ |
| **Matplotlib** | Visualization | 3.5+ |
| **PyTorch** | Deep learning | 2.0+ |
| **Gymnasium** | RL environment | 0.27+ |

---

## 📁 Project Structure

```
solar-drone-3d/
├── camera.py                          # Camera calibration & transformation
├── compute_3d_panels.py               # 3D coordinate calculation
├── env.py                             # RL environment
├── read_yolo.py                       # YOLOv8 inference
├── train.py                           # RL agent training
├── test_agents.py                     # Agent testing
├── visual_test.py                     # Testing module
├── visualize_3d.py                    # 3D visualization
├── panel_coordinates.py               # Panel data processing
├── config.py                          # Configuration settings
├── requirements.txt                   # Dependencies
├── .env.example                       # Environment variables template
├── panel_coordinates.csv              # Sample panel data
├── panel_coordinates_3d_exact_detailed.csv  # Detailed 3D coordinates
└── README.md                          # This file
```

### File Descriptions

| File | Purpose |
|------|---------|
| `camera.py` | Camera calibration, distortion correction, coordinate transformation |
| `compute_3d_panels.py` | Calculates 3D coordinates from 2D image points |
| `env.py` | Custom Gymnasium environment for drone control |
| `read_yolo.py` | Loads and runs YOLOv8 for panel detection |
| `train.py` | Trains RL agent for drone navigation |
| `test_agents.py` | Tests trained agents on sample data |
| `visualize_3d.py` | Renders 3D visualization of panels and drone path |
| `panel_coordinates.py` | Processes and manages panel coordinate data |
| `config.py` | Configuration for models, paths, and parameters |

---

## 📥 Download

The source code can be cloned via Git (see below). The panel detections in this repo were generated from the following dataset:

- **Dataset:** [Solar Panel Dataset (Roboflow)](https://universe.roboflow.com/tensrai/solar-panel-zitzr/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)

> The precomputed CSVs (`panel_coordinates.csv`, `panel_coordinates_3d_exact_detailed.csv`) already ship with this repo, so you don't need the raw dataset above to run the pipeline — it's only needed if you want to regenerate 3D coordinates from your own images via `compute_3d_panels.py`.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU acceleration, optional)
- 4GB+ RAM
- 5GB+ disk space

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/Vishal3347/solar-drone-3d.git
cd solar-drone-3d
```

#### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv solardrone_env
solardrone_env\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv solardrone_env
source solardrone_env/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your settings (YOLO model path, output directory, etc.)
```

#### 5. Load Panel Data & Train the RL Agent

```bash
python read_yolo.py    # loads panel positions from panel_coordinates_3d_exact_detailed.csv
python train.py        # trains PPO agent for 100,000 timesteps (~90s on CPU)
python test_agents.py  # evaluates the trained agent across 10 missions
```

#### 6. Visualize the 3D Panel Positions

```bash
python visualize_3d.py
```

Saves a plot to `results/panel_coordinates_3d.png` (see [Results & Metrics](#-results--metrics) below).

---

## 📖 Usage Guide

### 1. Load Panel Locations

`panel_coordinates_3d_exact_detailed.csv` ships with the repo (3,607 real panel detections across 379 drone images). Load it grouped by source image:

```python
from read_yolo import panel_locations_by_image

for image_name, panels in panel_locations_by_image.items():
    print(image_name, "->", panels)  # [(px, py), (px, py), ...]
```

### 2. 3D Coordinates

The CSV already contains computed 3D coordinates (`x`, `y`, `z` columns, in meters, relative to each image's camera center). To regenerate them from your own raw dataset (images + YOLO labels + depth maps), configure the paths in `config.py` and run:

```bash
python compute_3d_panels.py
```

### 3. Visualize Panels in 3D

```bash
python visualize_3d.py
```

### 4. Train RL Agent

Each episode is one drone "mission" inspecting all panels in one real image:

```bash
python train.py
```

### 5. Test the Trained Agent

```bash
python test_agents.py
```

Prints per-mission results, e.g.:
```
Episode 0: image=DJI_..._0194...jpg | panels visited=10/10 (100%) | steps=150 | battery left=85.0
...
Fully completed missions: 10/10
Average completion rate: 100.0%
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

```python
# Model paths
YOLO_MODEL_PATH = "models/yolov8.pt"
CALIBRATION_FILE = "calibration/camera_calib.npz"

# Detection parameters
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# RL training
RL_EPISODES = 5000
RL_LEARNING_RATE = 0.001
RL_BATCH_SIZE = 32

# Output
OUTPUT_DIR = "results/"
SAVE_VISUALIZATIONS = True
```

---

## 🧠 How It Works

### Solar Panel Detection Pipeline

```
Drone Image
    ↓
[YOLOv8 Model]  ← Detects solar panels
    ↓
Bounding Boxes (x, y, w, h)
    ↓
[Camera Calibration]  ← Applies camera matrix
    ↓
3D Coordinates (X, Y, Z)
    ↓
[3D Visualization]  ← Renders 3D layout
```

### RL Agent Training Flow

```
[Drone Simulation Environment]
         ↓
[Agent Observation] ← Panel locations, drone position
         ↓
[RL Agent] ← Policy network decides action
         ↓
[Execute Action] ← Move drone, collect panels
         ↓
[Reward Signal] ← Optimization metric
         ↓
[Update Policy] ← Gradient descent
```

---

## 📊 Results & Metrics

### 3D Panel Coordinate Data

From `panel_coordinates_3d_exact_detailed.csv`:

- **3,607 panel detections** across **379 drone images** (avg. ~9.5 panels/image, range 2-15)
- X range: -0.30 to +0.28 m | Y range: -0.29 to +0.27 m | Z (depth): 0 to 0.78

![3D Solar Panel Positions](results/panel_coordinates_3d.png)

*3D scatter of all 3,607 detected panel positions, colored by depth (Z). Generated by `visualize_3d.py`.*

### RL Agent Performance

Trained with `train.py` (PPO, 100,000 timesteps, ~90s on CPU) and evaluated with `test_agents.py` over 10 missions, each inspecting one real image's panels:

| Metric | Result |
|---|---|
| Missions fully completed | **10/10** |
| Average completion rate | **100%** |
| Avg. steps to complete a mission | ~150-250 |

---

## 🔧 Troubleshooting

### Issue: "YOLO model not found"

**Solution:**
```bash
# Download YOLOv8 model
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

### Issue: "Camera calibration file missing"

**Solution:**
```bash
# Create calibration
python camera.py --calibrate
# Provide checkerboard images in 'calibration_images/' folder
```

### Issue: "Out of memory during training"

**Solution:**
```python
# In config.py, reduce:
RL_BATCH_SIZE = 16  # from 32
```

### Issue: "3D visualization not rendering"

**Solution:**
```bash
# Install Matplotlib backend
pip install PyOpenGL
```

---

## 📈 Performance Optimization

### Speed Up Detection

If you add real YOLOv8 inference (not required for the default pipeline in this repo), use the lighter `yolov8n.pt` model instead of `yolov8m`/`yolov8l` for faster CPU inference.

### Enable GPU

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Batch Processing

```python
# Process multiple images efficiently
results = detector.detect_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
```

---

## 🔮 Future Enhancements

- [ ] Multi-drone coordination
- [ ] Real-time streaming from drone feed
- [ ] Advanced 3D reconstruction
- [ ] Cloud integration for remote monitoring
- [ ] Mobile app for field deployment
- [ ] Weather-adaptive algorithms
- [ ] Thermal imaging support
- [ ] Integration with solar management systems

---

## 📚 Resources

### Documentation

- [YOLOv8 Documentation](https://docs.ultralytics.com)
- [OpenCV Camera Calibration](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html)
- [Gymnasium RL Documentation](https://gymnasium.farama.org/)

### Research Papers

- YOLOv8: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- 3D Reconstruction: Computer Vision techniques
- RL: Gymnasium and PyTorch documentation

---


---

## 🤝 Contributing

Contributions welcome! To contribute:

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the MIT License. See [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Vishal Saha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files...
```

---

## 👤 Author

**Vishal Saha**

- GitHub: [@Vishal3347](https://github.com/Vishal3347)
- LinkedIn: www.linkedin.com/in/vishal-saha-4ba36428a

---