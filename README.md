# AeroTrack

AeroTrack is a UAV multi-object tracking framework built on top of YOLOX + ByteTrack, with dynamic early-exit routing for compute-efficient inference in easy frames while keeping full-capacity processing for ambiguous scenes.

> **Paper status:** pre-publication.
> **Paper link (to be updated):** `TBD`

## Highlights

- Dynamic routing with an early-exit branch and decision gate.
- UAV-focused tracking pipeline with IoU and NWD association options.
- Built-in evaluation outputs for MOT metrics, latency, FPS, energy, and routing statistics.
- TensorRT dual-engine path for edge deployment.

## Architecture

![AeroTrack Architecture](docs/assets/frame_routing_stacked.png)

## Ablation Snapshot

The ablation study reports a best **IDF1 = 85.1** for the full AeroTrack configuration.

| Configuration | IDF1 |
|---|---:|
| AeroTrack (best ablation setting) | **85.1** |

---

## Repository Layout

```text
/home/runner/work/AeroTrack/AeroTrack
├── aerotrack/              # Core library (models, tracker, evaluators, utils)
├── exps/                   # Experiment definitions
├── tools/                  # Train/eval/demo/data conversion scripts
├── third_party/ByteTrack/  # Upstream submodule
├── docs/
│   ├── assets/             # Figures
│   └── *.mp4               # Demo videos
├── run_evaluation.sh       # 4-config evaluation runner
├── requirements.txt
└── setup.py
```

---

## Setup

### 1) Environment

```bash
cd /home/runner/work/AeroTrack/AeroTrack
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -v -e . --no-build-isolation
```

### 2) Submodules

```bash
git submodule update --init --recursive
```

---

## Dataset Format

The default experiment expects:

```text
/home/runner/work/AeroTrack/AeroTrack/dataset/UAVSwarm/
├── train/
│   └── <sequence>/img1/*.jpg
├── test/
│   └── <sequence>/img1/*.jpg
└── annotations/
    ├── train.json
    └── test.json
```

To convert MOT-style folders to COCO-like JSON used by this repo:

```bash
python tools/convert_mot_to_coco.py
```

---

## Training

```bash
python tools/train.py \
  -f exps/aerotrack_proposed.py \
  -d 1 \
  -b 16
```

To resume from checkpoint:

```bash
python tools/train.py -f exps/aerotrack_proposed.py -d 1 -b 16 --resume -c /absolute/path/to/checkpoint.pth.tar
```

---

## Evaluation (PyTorch checkpoint)

```bash
python tools/track.py \
  -f exps/aerotrack_proposed.py \
  -c /absolute/path/to/early_exit_weights.pth \
  -d 1 -b 1 \
  --fp16 --fuse \
  --distance nwd \
  --early_exit \
  --save_vis
```

### Baseline vs AeroTrack comparisons

Run all 4 combinations (IoU/NWD × baseline/early-exit):

```bash
chmod +x run_evaluation.sh
./run_evaluation.sh
```

---

## TensorRT Edge Path (Optional)

1) Export dual ONNX graphs:

```bash
python aerotrack/models/export_dual.py
```

2) Build TensorRT engines:

```bash
python tools/build_dual_trt.py
```

3) Evaluate with TensorRT:

```bash
python tools/track.py \
  -f exps/aerotrack_proposed.py \
  -d 1 -b 1 \
  --trt --fp16 --fuse \
  --distance nwd --early_exit
```

---

## Outputs

Each run writes outputs under:

```text
YOLOX_outputs/<experiment_name>_<mode>_<distance>/
```

Key files:

- `mot_evaluation_metrics.csv` (MOTA/IDF1 + latency/FPS + optional energy stats)
- `early_exit_stats.csv` (early-exit usage and routing ratios)
- `track_results/*.txt` (per-sequence tracking results)
- `track_vis/` (optional visualizations)

---

## Model Weights Policy (Recommended)

**Do not commit large trained weights directly to git history.**

For open-source usability, publish at least one reproducible inference checkpoint via **GitHub Releases** (or a stable external host) and add the download link here. This gives users immediate reproducibility without bloating the repository.

Suggested release items:

- `aerotrack_best.pth` (PyTorch checkpoint)
- Optional TensorRT engines for a specific hardware target (clearly labeled)
- SHA256 checksum + short note of the exact experiment file/commit used

---

## Citation

```bibtex
@misc{aerotrack2026,
  title={AeroTrack: Dynamic Routing for UAV Multi-Object Tracking},
  author={Ayoub Bnina},
  year={2026},
  note={Preprint forthcoming. Link will be added upon publication.}
}
```

---

## Acknowledgements

- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
