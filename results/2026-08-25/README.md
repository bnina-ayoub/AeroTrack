# 2026-08-25 reproducibility archive

This directory is reserved for the raw artifacts used for the paper run:

- raw latency logs;
- raw power logs;
- MOT tracking outputs;
- the execution environment;
- SHA-256 hashes for weights, ONNX graphs, and TensorRT engines.

Datasets and generated outputs are intentionally not copied into this
repository. The raw latency/power logs and MOT outputs referenced by the paper
are in Overleaf and should be added here manually from that source.

The repository has no commit dated 2026-08-25 and no local copy of the
original Overleaf artifacts. The `v1.0-paper` tag therefore points to the
nearest preceding paper-era commit, `649a9d6e5871e383a26258e4818ba5d826438c9b`
(2026-08-24). The current `YOLOX_outputs/` directory and the `jetson_ablation/` results are
also included in the refreshed `v1.0-paper` tag. They remain available under
the separate `v1.0-later-run` tag as the later-run archive.

No ONNX files were available for the later run. The two available TensorRT
engines are included at repository root and covered by `SHA256SUMS`.

## Verification

From the repository root:

```bash
sha256sum -c results/2026-08-25/SHA256SUMS
```

The manifest currently covers the model artifacts present in this checkout.
It does not claim that those files are the exact August 25 inputs until the
archived Overleaf logs and original environment are supplied.
