from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .attention_residual import assert_transformer_free
from .compiler import AnyFlowRuntimeConfig, initialize_anyflow_runtime, precision_context, sm75_inference_config
from .engine import AnyFlowBatch, AnyFlowEngine, build_anyflow_small
from .quantization import dequantize_state_dict, quantize_state_dict, quantized_state_dict_stats


@dataclass(slots=True)
class AnyFlowTrainConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float | None = 1.0
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.float16
    compile_model: bool = False
    require_transformer_free: bool = True


@dataclass(slots=True)
class TrainStepMetrics:
    step: int
    modality: str
    loss: float
    grad_norm: float | None
    seconds: float
    cuda_allocated: int | None
    cuda_reserved: int | None


def _make_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(Path(path), map_location=map_location)


def _move_optional_tensor(x: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    return None if x is None else x.to(device)


def move_batch_to_device(batch: AnyFlowBatch, device: torch.device | str) -> AnyFlowBatch:
    target = torch.device(device)
    return AnyFlowBatch(
        modality=batch.modality,
        clean=batch.clean.to(target),
        source=_move_optional_tensor(batch.source, target),
        text_tokens=_move_optional_tensor(batch.text_tokens, target),
        text_lengths=_move_optional_tensor(batch.text_lengths, target),
        speaker_embedding=_move_optional_tensor(batch.speaker_embedding, target),
        metadata=dict(batch.metadata),
    )


class AnyFlowTrainer:
    def __init__(
        self,
        engine: AnyFlowEngine,
        train_config: AnyFlowTrainConfig | None = None,
        runtime: AnyFlowRuntimeConfig | None = None,
    ) -> None:
        self.engine = engine
        self.train_config = train_config or AnyFlowTrainConfig()
        self.runtime = runtime or engine.runtime
        initialize_anyflow_runtime(self.runtime)
        if self.train_config.require_transformer_free:
            assert_transformer_free(self.engine)
        if self.train_config.compile_model:
            self.engine.compile()
        self.optimizer = torch.optim.AdamW(
            self.engine.parameters(),
            lr=self.train_config.lr,
            betas=(self.train_config.beta1, self.train_config.beta2),
            weight_decay=self.train_config.weight_decay,
        )
        self.scaler = _make_grad_scaler(self.train_config.use_amp and torch.cuda.is_available())
        self.global_step = 0

    @property
    def device(self) -> torch.device:
        return next(self.engine.parameters()).device

    def to(self, device: torch.device | str) -> "AnyFlowTrainer":
        self.engine.to(device)
        return self

    def train_step(self, batch: AnyFlowBatch) -> TrainStepMetrics:
        batch = move_batch_to_device(batch, self.device)
        self.engine.train()
        self.optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        use_amp = self.train_config.use_amp and self.device.type == "cuda"
        with precision_context(enabled=use_amp, device_type=self.device.type, dtype=self.train_config.amp_dtype):
            out = self.engine.training_loss(batch)
            loss = out.loss

        grad_norm: float | None = None
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            if self.train_config.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                norm = torch.nn.utils.clip_grad_norm_(self.engine.parameters(), self.train_config.grad_clip)
                grad_norm = float(norm.detach().cpu())
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.train_config.grad_clip is not None:
                norm = torch.nn.utils.clip_grad_norm_(self.engine.parameters(), self.train_config.grad_clip)
                grad_norm = float(norm.detach().cpu())
            self.optimizer.step()

        elapsed = time.perf_counter() - start
        self.global_step += 1
        if torch.cuda.is_available() and self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
        else:
            allocated = None
            reserved = None
        return TrainStepMetrics(
            step=self.global_step,
            modality=batch.modality,
            loss=float(loss.detach().cpu()),
            grad_norm=grad_norm,
            seconds=elapsed,
            cuda_allocated=allocated,
            cuda_reserved=reserved,
        )

    def save_checkpoint(self, path: str | Path, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "engine": self.engine.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.global_step,
            "extra": extra or {},
        }
        torch.save(payload, Path(path))

    def load_checkpoint(self, path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
        payload = _torch_load(path, map_location=map_location)
        self.engine.load_state_dict(payload["engine"])
        self.optimizer.load_state_dict(payload["optimizer"])
        if payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])
        self.global_step = int(payload.get("global_step", 0))
        return dict(payload.get("extra", {}))

    def save_quantized_checkpoint(
        self,
        path: str | Path,
        *,
        format: str = "mxfp4",
        tile_size: int | None = None,
        min_numel: int = 1,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        q_state = quantize_state_dict(
            self.engine.state_dict(),
            format=format,
            tile_size=tile_size,
            min_numel=min_numel,
        )
        payload = {
            "engine_quantized": q_state,
            "global_step": self.global_step,
            "extra": extra or {},
        }
        torch.save(payload, Path(path))
        return quantized_state_dict_stats(q_state)

    def load_quantized_checkpoint(
        self,
        path: str | Path,
        map_location: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> dict[str, Any]:
        payload = _torch_load(path, map_location=map_location)
        device = self.device if map_location == "cpu" else map_location
        state = dequantize_state_dict(payload["engine_quantized"], device=device, dtype=dtype)
        self.engine.load_state_dict(state)
        self.global_step = int(payload.get("global_step", 0))
        return dict(payload.get("extra", {}))


def build_sm75_trainer(
    video_channels: int = 2,
    audio_channels: int = 1,
    vocab_size: int = 64,
    dim: int = 32,
    train_config: AnyFlowTrainConfig | None = None,
    require_sm75: bool = False,
) -> AnyFlowTrainer:
    runtime = sm75_inference_config(require_sm75=require_sm75)
    engine = build_anyflow_small(
        video_channels=video_channels,
        audio_channels=audio_channels,
        vocab_size=vocab_size,
        dim=dim,
        runtime=runtime,
    )
    return AnyFlowTrainer(engine=engine, train_config=train_config, runtime=runtime)
