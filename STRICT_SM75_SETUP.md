# ANYFLOW Strict sm_75 Environment

This is the strict production gate for the current ANYFLOW target:

- PyTorch `2.10.0` CUDA `12.8`
- TileLang `0.1.10`
- NVIDIA T4-class compute capability `sm_75`
- Transformer-free and VAE-free ANYFLOW executable code

The local development machine used for the current smoke run reports PyTorch
`2.11.0+cu128` and TileLang metadata as `unknown`, so this strict gate must be
run in a clean Colab/Kaggle or equivalent T4 environment before claiming full
target-platform completion.

## Colab/Kaggle Bootstrap

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install pytest
```

Install TileLang `0.1.10`:

```bash
cd /content
git clone https://github.com/tile-ai/tilelang.git
cd tilelang
git checkout v0.1.10
python -m pip install . --no-build-isolation
```

Return to the ANYFLOW workspace root:

```bash
cd "/content/Asymmetric Flow"
```

## Required Gates

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest tests/test_anyflow_smoke.py -q
python -B -m anyflow.verify --device cuda --strict-versions --require-sm75 --require-tilelang
python -B -m anyflow.compile_validation --device cuda --dim 8 --state-dim 2 --run-compile --require-sm75
python -B -m anyflow.colab
python -B -m anyflow.benchmark --device cuda --dim 16 --vocab-size 32 --steps 1 --warmup 1 --repeats 3
```

The strict readiness command must report:

- `passed_required: true`
- `torch_version` beginning with `2.10.0`
- `tilelang_version` beginning with `0.1.10`
- `cuda_sm: 75`
- successful transformer-free, quantization, kernel parity, and modality smoke checks
- successful dynamic-shape `torch.compile` parity in `anyflow.compile_validation`

## Optional Compile Gate

Compilation is intentionally separate from the fast smoke path because first-run
Inductor and TileLang compilation can take longer on free notebooks.

```bash
python -B -m anyflow.benchmark --device cuda --dim 16 --vocab-size 32 --steps 1 --warmup 1 --repeats 3 --compile-model --require-sm75
```
