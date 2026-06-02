# ANYFLOW Free Colab / Kaggle Smoke Training

This path is intentionally tiny. It verifies the Mamba-only video, TTS, and music flow paths without a dataset, transformer module, or latent-space VAE.

## Target

- GPU: NVIDIA T4 / sm_75 preferred
- Python: 3.10+
- PyTorch target: 2.10.0 CUDA 12.8 build
- TileLang target: 0.1.10

The code also runs on CPU for shape tests, but the intended inference profile is sm_75.
For the strict production target, use `anyflow/STRICT_SM75_SETUP.md`.

## Install

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install pytest
```

If TileLang is not preinstalled in the notebook:

```bash
cd /content
git clone https://github.com/tile-ai/tilelang.git
cd tilelang
git checkout v0.1.10
pip install . --no-build-isolation
```

Then place or clone this workspace and run from the `Asymmetric Flow` root.

## Quick Smoke Train

```bash
python -B -m anyflow.colab
```

Expected result: a dictionary with `device`, `steps`, `last_loss`, `last_modality`, optional `cuda_allocated`, and `cuda_name`.

## Test

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest tests/test_anyflow_smoke.py -q
```

Expected result:

```text
25 passed
```

## Readiness Report

```bash
python -B -m anyflow.verify --device cuda --dim 16 --vocab-size 32
```

The readiness report includes deterministic runtime checks, transformer-free assembly checks, modality loss/generation smoke checks, quantization tensor/state-dict shape checks, and kernel parity checks.

## Dynamic Compile Check

Fast config and eager-backend probe:

```bash
python -B -m anyflow.compile_validation --device cuda --dim 8 --state-dim 2
python -B -m anyflow.compile_validation --device cuda --dim 8 --state-dim 2 --run-compile --backend eager
```

Strict Inductor dynamic-shape probe:

```bash
python -B -m anyflow.compile_validation --device cuda --dim 8 --state-dim 2 --run-compile --require-sm75
```

## Tiny Inference Benchmark

```bash
python -B -m anyflow.benchmark --device cuda --dim 16 --vocab-size 32 --steps 1 --warmup 1 --repeats 3
```

This reports per-modality generation latency, output shapes, and CUDA peak memory fields.

For a strict T4-class production gate, add:

```bash
python -B -m anyflow.verify --device cuda --strict-versions --require-sm75 --require-tilelang
```

## Minimal Notebook Cell

```python
from anyflow import QuickTrainConfig, run_quick_train

result = run_quick_train(QuickTrainConfig(steps=3, dim=32, batch_size=1))
print(result)
```

## Strict sm_75 Check

```python
from anyflow import sm75_inference_config, initialize_anyflow_runtime

cfg = sm75_inference_config(require_sm75=True)
print(initialize_anyflow_runtime(cfg))
```

This raises if the notebook GPU is not compute capability 7.5.
