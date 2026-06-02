from __future__ import annotations

from dataclasses import dataclass

import torch

from .attention_residual import assert_transformer_free
from .compiler import initialize_anyflow_runtime, sm75_inference_config
from .data import SyntheticAnyFlowDataConfig, iter_synthetic_anyflow_batches
from .engine import AnyFlowEngine, build_anyflow_small
from .training import AnyFlowTrainConfig, AnyFlowTrainer


@dataclass(slots=True)
class QuickTrainConfig:
    steps: int = 3
    dim: int = 32
    batch_size: int = 1
    video_channels: int = 2
    audio_channels: int = 1
    vocab_size: int = 64
    lr: float = 2e-4
    compile_model: bool = False
    device: str | None = None
    seed: int = 1337


def build_colab_smoke_engine(config: QuickTrainConfig | None = None) -> AnyFlowEngine:
    cfg = config or QuickTrainConfig()
    runtime = sm75_inference_config(require_sm75=False)
    initialize_anyflow_runtime(runtime)
    engine = build_anyflow_small(
        video_channels=cfg.video_channels,
        audio_channels=cfg.audio_channels,
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        runtime=runtime,
    )
    assert_transformer_free(engine)
    if cfg.compile_model:
        engine.compile()
    return engine


def run_quick_train(config: QuickTrainConfig | None = None) -> dict[str, float | int | str | None]:
    cfg = config or QuickTrainConfig()
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    engine = build_colab_smoke_engine(cfg).to(device)
    trainer = AnyFlowTrainer(
        engine,
        AnyFlowTrainConfig(lr=cfg.lr, grad_clip=0.5, compile_model=False, use_amp=True),
        runtime=engine.runtime,
    )

    last_metrics = None
    data_cfg = SyntheticAnyFlowDataConfig(
        batch_size=cfg.batch_size,
        video_channels=cfg.video_channels,
        audio_channels=cfg.audio_channels,
        vocab_size=cfg.vocab_size,
        seed=cfg.seed,
    )
    for batch in iter_synthetic_anyflow_batches(cfg.steps, data_cfg, device=device):
        last_metrics = trainer.train_step(batch)

    assert last_metrics is not None

    return {
        "device": device,
        "steps": cfg.steps,
        "last_loss": last_metrics.loss,
        "last_modality": last_metrics.modality,
        "cuda_allocated": last_metrics.cuda_allocated,
        "cuda_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }


if __name__ == "__main__":
    print(run_quick_train())
