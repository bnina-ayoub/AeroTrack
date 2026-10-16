# Later-run evaluation archive

This archive is represented by the `v1.0-later-run` Git tag. It includes:

- the complete `YOLOX_outputs/` directory;
- raw per-frame latency logs and power/energy telemetry emitted by the
  evaluation runs;
- MOT tracking outputs under each run's `track_results/` directory;
- the complete `jetson_ablation/` directory;
- `deep_stage_fp16.trt` and `early_stage_fp16.trt`;
- SHA-256 entries for the available model and engine artifacts in
  `results/2026-08-25/SHA256SUMS`.

The Jetson Nano environment is JetPack 4.6.6, based on L4T 32.7.6. No ONNX
files were present in this checkout, so none are included or hashed.
