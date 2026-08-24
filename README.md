# AeroTrack

AeroTrack is an object tracking architecture optimized for Unmanned Aerial Vehicles (UAVs). Built on the solid foundations of the YOLOX detection framework and the ByteTrack tracking engine, our approach introduces an innovative dynamic routing mechanism (Early Exit) driven by a Decision Gate.

---
# ⚙️ Installation and Setup

To run AeroTrack, you need to set up the Python environment and install the customized architecture.

> **Note:** Edge deployment requires an NVIDIA Jetson platform with JetPack (TensorRT) installed.

## 1. Environment Prerequisites

It is recommended to use a virtual environment with Python 3.8+ and a version of PyTorch compatible with your CUDA version.

```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Fetching Submodules

AeroTrack uses the original ByteTrack repository as a submodule for baseline evaluation. Run the following to fetch it:

```bash
git submodule update --init --recursive
```

## 3. Installing Dependencies

Install the required dependencies and link the project from the root directory. Ensure `tensorrt` is available in your environment (typically pre-installed on NVIDIA Jetson devices via JetPack).

```bash
# Install base dependencies
pip install -r requirements.txt
pip install cython cython_bbox motmetrics

# Install AeroTrack in development mode (Compiles C++ and Cython NWD extensions)
pip install -v -e . --no-build-isolation
```

---

# 🛠️ Hardware Deployment: Dual-Engine Compilation

Because standard TensorRT compiles models into rigid, static graphs, deploying the Early Exit routing requires compiling two separate engines.

## 1. Exporting ONNX Graphs

Place your trained weights file (`early_exit_weights.pth`) in the `weights/` folder. Export the shallow and deep stages into isolated `.onnx` files.

```bash
python tools/export_onnx.py -f exps/aerotrack_proposed.py -c weights/early_exit_weights.pth
```

## 2. Compiling TensorRT Engines

Compile the exported ONNX graphs into optimized FP16 TensorRT `.trt` engines targeted specifically for your local hardware.

```bash
python tools/build_dual_trt.py
```

---

# 🚀 Evaluation and Inference

The evaluation pipeline seamlessly integrates the compiled Dual-Engine architecture using a native Python drop-in wrapper. To ensure a strictly fair hardware comparison, all configurations (including the Baseline) are executed using the TensorRT engines.

## Automated Run (Recommended)

The `run_evaluations.sh` script automatically runs the 4 core experiments, combining the distance metrics (IoU / NWD) with the activation of dynamic routing (Baseline / Early Exit).

> **Important:** To accurately measure physical power draw on an NVIDIA Jetson device via hardware sensors, the script must be run with `sudo`.

```bash
# 1. Grant execution rights to the script
chmod +x run_evaluations.sh

# 2. Run the evaluation suite (requires absolute python path inside the script if using venv)
sudo ./run_evaluations.sh
```

Under the hood, the script executes the following four configurations on the Jetson GPU:

| Configuration | Flags | Behavior |
|---|---|---|
| **Baseline + IoU** | `--trt` | Forces the Deep Stage on 100% of frames |
| **Baseline + NWD** | `--trt --distance nwd` | Forces the Deep Stage on 100% of frames |
| **Early Exit + IoU** | `--trt --early_exit` | Activates dynamic routing |
| **AeroTrack (Ours)** | `--trt --early_exit --distance nwd` | Activates dynamic routing |

## Manual Edge Tracking with Energy Monitoring

If you want to run a specific configuration manually:

```bash
sudo /absolute/path/to/venv/bin/python tools/track.py \
    --trt \
    --fp16 \
    --fuse \
    -d 1 -b 1 \
    -f exps/aerotrack_proposed.py \
    --distance nwd \
    --early_exit \
    --monitor_energy \
    --save_vis
```

### CLI Flags Breakdown

| Flag | Description |
|---|---|
| `--trt` | Mandatory for edge benchmarking. Bypasses PyTorch and loads the Dual TensorRT engines (`early_stage_fp16.trt` & `deep_stage_fp16.trt`). |
| `--early_exit` | Activates the dynamic Decision Gate. If omitted, the pipeline acts as a Baseline, unconditionally routing every frame through the full deep network. |
| `--monitor_energy` | Spawns a background thread querying Jetson hardware power states (`tegrastats`). |
| `--distance` | Selects the tracking association metric (`iou` or `nwd`). |
| `--save_vis` | Generates frame-by-frame tracking visualizations. |

---

# 📊 Results Analysis

For each experiment, AeroTrack generates a dedicated results folder containing:

- **`mot_evaluation_metrics.csv`**: Detailed tracking accuracy (MOTA, IDF1), hardware speed (Clean Pipeline FPS, Best/Worst Frame Latency in ms), and physical energy efficiency (Energy_Total_J, Avg_Power_W, Energy_Per_Frame_mJ).
- **`early_exit_stats.csv`**: The exact ratio of frames that routed to the early exit versus the deep network, including theoretical GFLOPs saved.
- **`track_vis/`**: A folder containing visualizations of the UAV tracking, indicating which branch (Early or Full) was executed per frame.

---

# 📝 Project Structure

- **`tools/track.py`**: Main script for launching inference, MOT evaluation, and latency/energy benchmarking.
- **`tools/export_onnx.py`**: Extracts and splits the PyTorch model into independent Early and Deep ONNX graphs.
- **`tools/build_dual_trt.py`**: Compiles the ONNX graphs into target-specific FP16 TensorRT engines.
- **`tools/dual_model_loader.py`**: The Python inference wrapper managing asynchronous execution, zero-copy memory pointers, and dynamic routing between the C++ TensorRT engines.
- **`exps/aerotrack_proposed.py`**: Definition file for the unified architecture thresholds and parameters.
- **`aerotrack/`**: Core source code containing the PyTorch DecisionGate, EarlyHead logic, and custom NWD distance implementations.
- **`third_party/ByteTrack/`**: The pristine, original ByteTrack repository included as a submodule for baseline reference and evaluation.
