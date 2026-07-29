# 🚁 AeroTrack: Dynamic Early-Exit Architecture for UAV Tracking

AeroTrack is an object tracking architecture optimized for Unmanned Aerial Vehicles (UAVs). Built on the solid foundations of the **[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)** detection framework and the **[ByteTrack](https://github.com/ifzhang/ByteTrack)** tracking engine, our approach introduces an innovative dynamic routing mechanism (**Early Exit**) driven by a **Decision Gate**.

This architecture makes it possible to bypass the network's deep layers when detection confidence is high enough, delivering a drastic reduction in computational cost (GFLOPs) without sacrificing tracking accuracy, thanks to the IoU and NWD distance metrics.

---

## 🎥 Video Demonstration

AeroTrack architecture (Early Exit + NWD) in action:

https://github.com/user-attachments/assets/ca8e1d86-d2df-4852-928f-89b6337f349b

---

## ⚙️ Installation and Setup

To run AeroTrack, you need to set up the Python environment and install the customized architecture.

### 1. Environment Prerequisites

It is recommended to use a virtual environment (Conda or venv) with **Python 3.8+** and a version of **PyTorch** compatible with your CUDA version.

```bash
# Example with venv
python3 -m venv venv
source venv/bin/activate
```

### 2. Fetching Submodules

AeroTrack uses the original ByteTrack repository as a submodule for baseline evaluation. Run the following to fetch it:

```bash
git submodule update --init --recursive
```

### 3. Installing Dependencies and AeroTrack

AeroTrack relies on the YOLOX engine. Run the following commands from the project root to install the required dependencies and link the project.

```bash
# Install base dependencies
pip install -r requirements.txt

# Install tracking-specific dependencies (MOT)
pip install cython
pip install cython_bbox
pip install motmetrics

# Install AeroTrack in development mode (Compiles C++ and Cython NWD extensions)
pip install -v -e .
```

### 4. Preparing the Weights

Make sure to place your trained weights file (`early_exit_weights.pth`) in the `weights/` folder at the project root.

---

## 🚀 Evaluation and Inference

We've set up a robust automation script to test the architecture across all its configurations smoothly.

### Automated Run (Recommended)

The `run_evaluations.sh` script automatically runs 4 experiments, combining the distance metrics (IoU / NWD) with the activation of dynamic routing (Baseline / Early Exit).

To launch the full evaluation:

```bash
# 1. Grant execution rights to the script
chmod +x run_evaluations.sh

# 2. Run the evaluation
./run_evaluations.sh
```

Under the hood, the script runs the following command for each mode:

```bash
python3 tools/track.py --fp16 --fuse -d 1 -b 1 -f exps/aerotrack_proposed.py -c weights/early_exit_weights.pth --distance <metric> [--early_exit] --save_vis
```

### Results Analysis

For each experiment, AeroTrack will generate a dedicated results folder containing:

- **`mot_evaluation_metrics.csv`**: Detailed tracking results (MOTA, IDF1, FPS, etc.).
- **`early_exit_stats.csv`**: The exact ratio of frames that took the short path and the effective GFLOPs saved.
- **`track_vis/`**: A folder containing frame-by-frame visualizations of the UAV tracking, indicating which path was taken (Early Exit or Full).

---

## 📝 Project Structure

- **`tools/track.py`**: Main script for launching inference and MOT tracking.
- **`exps/aerotrack_proposed.py`**: Definition file for our unified architecture.
- **`run_evaluations.sh`**: Bash script for automating comparative evaluations.
- **`aerotrack/`**: Model source code containing the `DecisionGate` and `EarlyHead` logic, alongside custom NWD distance implementations.
- **`third_party/ByteTrack/`**: The pristine, original ByteTrack repository included as a submodule for baseline reference and evaluation.