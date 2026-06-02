# ANYFLOW Implementation Report

## Current Scope

ANYFLOW is now a transformer-free, VAE-free, pixel/waveform-space framework for quick multimodal flow-matching experiments:

- Video: spatial/video latent tensors are treated directly as data tensors, not VAE latents.
- TTS / voice cloning: waveform or acoustic-channel tensors are conditioned by Mamba-only text and speaker paths.
- Music: multi-track audio tensors use the same asymmetric flow engine with track flattening and restoration.
- Inference target: sm_75 / NVIDIA T4-class compatibility is the default runtime profile.

## Local Documentation Applied

- `SKILL-Mamba-3-Asymmetric Flow.mdV2.md`: guided the production constraints: deterministic setup, Mamba-3-style SSM backbone, asymmetric flow, low-level TileLang hooks, and operational smoke testing.
- `PyTorch_2.10.0_Complete_Documentation.txt`: implementation uses `torch.compile(..., backend="inductor", dynamic=True)`, deterministic algorithms, inductor deterministic configuration, combo-kernel configuration when exposed, and native float8 dtype checks.
- `TileLang_v0.1.10_Complete_Documentation.txt`: custom kernels follow the `@tilelang.jit` / `@T.prim_func` / `T.Kernel` pattern, use explicit dtype/shape assumptions, and keep PyTorch fallbacks because low-precision formats depend on target architecture.
- `Attention Residuals.md`: implemented as `GroupedDepthResidualRouter` and `AttentionResidualMambaStack`, preserving depth-wise residual routing while replacing transformer sublayers with Mamba blocks.

## Reference Repository Findings

- `attn_res`: useful for depth-wise Attention Residuals and GQA grouping. The transformer layer itself was not imported; only the residual routing concept was ported.
- `delta-attention-residuals-code`: confirms the Attention Residuals pattern around learned pseudo-queries over prior layer outputs. Qwen/transformer code was excluded.
- `Gated-Sparse-Attention` and `sparse attention`: useful as inspiration for gating, sparse routing, and stability. Token sparse attention kernels were excluded from the default path to keep ANYFLOW Mamba-only.
- `Spatial-Mamba`: main visual reference for structure-aware scan/fusion. ANYFLOW video uses multi-direction spatial and temporal Mamba scanning.
- `VMamba`: useful for SS2D, cross-scan, selective-scan fallback design, and vision SSM conventions.
- `BVI-Mamba`: confirms video restoration/enhancement workflows and multi-frame low-light video patterns.
- `MVC`: strongest reference for SSM-only TTS conditioning. ANYFLOW mirrors the spirit with Mamba text conditioning, style/speaker modulation, and no attention duration path.
- `rectified-flow-pytorch`: objective reference for rectified flow matching, direct velocity learning, and Euler sampling. Transformer/Unet dependencies were not imported.
- `LakonLab-Asymmetric Flow`: reference for production flow training and asymmetric-flow scheduling; ANYFLOW includes a thin registry bridge but keeps the standalone package independent.
- `Mixture of Experts (MoE)/mammothmoda`: useful for fine-grained routed expert scaling, no-token-dropping inference, shared experts, and Top-K activation. Qwen/DiT/VAE paths were not imported.
- `Mixture of Experts (MoE)/TAG-MoE`: useful for task-aware routing and semantic-intent biasing so unified generation/editing tasks do not collapse into one dense parameter path.
- `tilelang/examples/fusedmoe`: useful for grouped-by-expert dispatch layout, fused gate/up projection, SiLU expert gating, and grouped down projection. ANYFLOW keeps a PyTorch dispatch-plan fallback and exposes TileLang-friendly grouping metadata.

## PDF Findings Applied

- `Asymmetric Flow.pdf`: ANYFLOW keeps direct data-space flow matching and asymmetric time warping. It does not require a latent-space VAE and keeps sampling as a direct Euler velocity integration.
- `Attention_Residuals.pdf`: depth-wise residual aggregation is implemented with learned pseudo-queries and block-level state routing. The transformer self-attention layer from the paper is replaced by Mamba blocks.
- `Delta Attention Residuals.pdf`: reinforced the need to route informative deltas/previous states rather than blindly summing residuals. ANYFLOW uses block partial states to reduce uniform residual accumulation.
- `BVI-Mamba ...pdf`: informed the video path: multi-frame video processing should avoid expensive explicit optical flow and use VSS/Mamba-style temporal-spatial state updates.
- `MambaVF ...pdf`: reinforced temporal fusion without explicit motion estimation; ANYFLOW video scans the temporal axis plus spatial directions.
- `MAMBAVOICECLONING ...pdf`: informed the TTS path: SSM-only conditioning, style/speaker modulation, and no attention module at inference.
- `VMamba Visual State Space Model.pdf`: informed multi-direction visual scans and SS2D-style state-space design.
- `Gated Sparse Attention ...pdf`: used only as non-token-attention inspiration for gating and bounded routing stability; sparse token attention kernels were not imported.
- `PiD ...pdf`: reviewed but not adopted because the current requirement explicitly avoids latent-space VAE workflows. A future optional pixel decoder can be considered only if it remains outside the core VAE-free path.
- `Mamoda2.5 ...DiT-MoE.pdf`: informed sparse fine-grained multimodal routing: large total expert capacity with small active Top-K compute. ANYFLOW ports the routed-expert principle into Mamba state experts instead of DiT blocks.
- `MiM-DiT MoE in MoE ...pdf`: informed coarse/fine routing. ANYFLOW exposes task/modality bias and optional region bias so routing can separate video, TTS, music, restoration, and edit-like subregions without token attention.
- `TAG-MoE ...pdf`: informed task-aware gating and routing regularization. ANYFLOW includes task IDs, modality-to-task mapping, load-balancing loss, router z-loss, entropy telemetry, and per-expert usage counts.
- `Nemotron 3 Super ...pdf`: informed hybrid Mamba-MoE scaling and low-precision deployment priorities. ANYFLOW keeps the MoE experts as Mamba-only SSM bodies and leaves quantization compatibility through existing FP8/MXFP4 utilities.
- `CoInteract ...pdf`: informed region-specialized routing and auxiliary structure supervision. ANYFLOW adds optional region IDs for spatial/video token routing while keeping the core video engine VAE-free.
- `SANA-Streaming ...pdf`: informed real-time system co-design, fused kernels, flow matching, and mixed precision. ANYFLOW keeps sm_75 runtime checks and a grouped MoE dispatch plan suitable for future TileLang kernels.
- `SP-MoMamba ...pdf`: strongest MoE reference for this project: mixture of state-space experts, multi-scale specialization, and local spatial modulation. ANYFLOW implements Mamba state experts plus a shared local modulation expert.

## Implemented Advanced Features

- Deterministic runtime initialization:
  - `torch.use_deterministic_algorithms(True)`
  - `torch._inductor.config.combo_kernels = True` when available
  - `torch._inductor.config.deterministic = True` when available
  - CUDA/cuDNN deterministic flags
- sm_75 inference profile:
  - `sm75_inference_config()`
  - FP16 autocast preference for T4-class GPUs
  - optional strict sm_75 runtime check
  - strict Colab/Kaggle target setup documented in `anyflow/STRICT_SM75_SETUP.md`
- Mamba-3-style block:
  - fused input projection for state, gate, dt, B, and C paths
  - diagonal log-domain SSM transition
  - depthwise local mixing
  - optional TileLang scan backend with PyTorch fallback
- Attention Residuals without transformers:
  - learned pseudo-query depth routing
  - grouped query-style heads over prior depth states with shared grouped key/value states
  - block residual accumulation
  - Mamba block transforms only
- Mixture of Experts without transformers:
  - `AnyFlowMoEConfig`, `TopKTaskRouter`, `MambaMoELayer`, and `AnyFlowMoEAdapter`
  - Top-K task/modality-aware routing with optional spatial/region bias
  - routed experts are Mamba-3 state-space blocks, not dense token-attention layers
  - shared local modulation expert for SP-MoMamba-style edge/local detail preservation
  - no-token-dropping dispatch metadata through `build_expert_dispatch()`
  - training-ready auxiliary router loss: load balancing, router z-loss, entropy telemetry, and per-expert counts
  - modality presets through `make_moe_config_for_modality()`
- Packed/ragged sequence handling:
  - text/audio conditioning uses sequence lengths and packed per-sample processing
  - `Mamba3Block` and `Mamba3Stack` skip padded tails by executing only valid ragged segments when `lengths` are supplied
  - padding is masked from pooled conditioning
  - legacy token-token attention fallback was removed from `anyflow`; ragged paths use packed segment processing plus Mamba-local mixing
- Quantization utilities:
  - native FP8 tile quantization when the local PyTorch exposes float8 dtypes
  - MXFP4 packed storage fallback
  - dequantization and shape restoration helpers
  - quantized state-dict export/import for low-memory checkpoints
  - activation quantization context with FP8 request and MXFP4 fallback
- Auxiliary modality losses:
  - multi-resolution STFT loss for TTS/music velocity fields
  - audio temporal-difference loss
  - video temporal-consistency loss over frame differences
  - optional engine integration through `AuxiliaryLossConfig`
- Low-VRAM audio generation:
  - chunked TTS/music generation through `DeepThinkingAudioFlowEngine.generate_chunked()`
  - unified wrapper `AnyFlowEngine.generate_audio_chunked()`
  - overlap-average stitching for long waveform or multi-track outputs
- Quick Colab/Kaggle test path:
  - `build_colab_smoke_engine()`
  - `run_quick_train()`
  - reusable `SyntheticAnyFlowDataConfig` and `iter_synthetic_anyflow_batches()` for video/TTS/music smoke batches
  - one-call synthetic video/TTS/music smoke training
  - unified `AnyFlowTrainer` path with AdamW, AMP/scaler support, grad clipping, metrics, full checkpoint save/load, and quantized checkpoint save/load
  - notebook/runbook commands in `anyflow/COLAB_KAGGLE_QUICKSTART.md`
- Readiness verification:
  - `run_anyflow_verification()`
  - `python -B -m anyflow.verify`
  - optional strict checks for PyTorch `2.10.0`, TileLang `0.1.10`, and sm_75
  - required smoke checks for deterministic runtime, transformer-free assembly, all three modality losses, generation shapes, MXFP4 tensor/state-dict shape restoration, and kernel parity
- Dynamic compile validation:
  - `run_compile_validation()`
  - `python -B -m anyflow.compile_validation`
  - validates `torch.compile` availability, dynamic-shape runtime flags, combo-kernel requests, deterministic Inductor requests, and optional compiled dynamic-shape parity
- sm_75 inference benchmarking:
  - `benchmark_sm75_inference()`
  - `python -B -m anyflow.benchmark`
  - per-modality generation latency, output shape validation, optional compile flag, and CUDA peak memory fields
- Kernel validation:
  - `run_kernel_validation()`
  - Mamba scan parity against the PyTorch reference path
  - asymmetric flow-step parity against the reference update
  - direct TileLang kernel execution is attempted when CUDA and TileLang are available

## Verification

Validated with:

```bash
python -B -m pytest tests/test_anyflow_smoke.py -q
```

Result:

```text
25 passed
```

Manual quick train also ran through the Colab/Kaggle entrypoint on the local CUDA device and completed 3 optimizer steps:

```bash
python -B -m anyflow.colab
```

Result:

```text
{'device': 'cuda', 'steps': 3, 'last_modality': 'music', 'cuda_allocated': ..., 'cuda_name': 'NVIDIA GeForce GTX 1650 SUPER', ...}
```

Additional invariants:

- No `scaled_dot_product_attention`, `varlen_attn`, `torch.nn.attention`, `diffusers`, `AutoModel`, `x_transformers`, or VAE imports are present in executable ANYFLOW Python files.
- `assert_transformer_free()` verifies that no `nn.Transformer`, `nn.TransformerEncoder`, `nn.TransformerDecoder`, or `nn.MultiheadAttention` modules are attached to the assembled engine.
- `python -B -m anyflow.verify --device cpu --dim 16 --vocab-size 32` passes all required checks on the local machine, including quantized state-dict and kernel parity checks; it reports local CUDA as sm_75, local PyTorch as `2.11.0+cu128`, and TileLang metadata as `unknown`, so strict PyTorch `2.10.0` / TileLang `0.1.10` production validation remains a separate environment gate.
- CPU determinism smoke coverage confirms repeated seeded video-flow loss values match exactly.

## Future Additions

- Add a true TileLang sm_75 selective-scan kernel benchmark with numerical parity tests against the PyTorch scan.
- Extend waveform losses with log-mel, loudness, and phase-aware terms.
- Extend video-specific losses with optical-flow consistency and patch-level perceptual metrics without VAE encoding.
- Extend chunked TTS/music inference with true finite look-ahead Mamba state carry.
- Add dataset adapters for tiny Colab/Kaggle smoke sets: Moving-MNIST-style video, LJ Speech clips, and small multi-track stems.
- Add export path for sm_75 inference: `torch.export` / AOTInductor package with fallback eager mode.
- Extend telemetry with peak CUDA allocated/reserved, tokens or samples per second, and deterministic run hash.
- Add quantized optimizer-state sharding and per-module precision policies for larger checkpoints.
