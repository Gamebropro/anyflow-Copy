from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn

from .audio import AudioVelocityNet, DeepThinkingAudioFlowEngine
from .compiler import AnyFlowRuntimeConfig, compile_anyflow, initialize_anyflow_runtime
from .video import DeepThinkingVideoFlowEngine, VideoVelocityNet


Modality = Literal["video", "tts", "music"]


@dataclass(slots=True)
class AnyFlowBatch:
    modality: Modality
    clean: torch.Tensor
    source: torch.Tensor | None = None
    text_tokens: torch.Tensor | None = None
    text_lengths: torch.Tensor | None = None
    speaker_embedding: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AnyFlowEngine(nn.Module):
    def __init__(
        self,
        video: DeepThinkingVideoFlowEngine,
        tts: DeepThinkingAudioFlowEngine,
        music: DeepThinkingAudioFlowEngine,
        runtime: AnyFlowRuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        self.video = video
        self.tts = tts
        self.music = music
        self.runtime = runtime or AnyFlowRuntimeConfig()

    def initialize(self) -> dict[str, Any]:
        return initialize_anyflow_runtime(self.runtime)

    def compile(self) -> "AnyFlowEngine":
        self.video.model = compile_anyflow(self.video.model, self.runtime)
        self.tts.model = compile_anyflow(self.tts.model, self.runtime)
        self.music.model = compile_anyflow(self.music.model, self.runtime)
        return self

    def _select(self, modality: Modality) -> DeepThinkingVideoFlowEngine | DeepThinkingAudioFlowEngine:
        if modality == "video":
            return self.video
        if modality == "tts":
            return self.tts
        if modality == "music":
            return self.music
        raise ValueError(f"unknown modality: {modality}")

    def training_loss(self, batch: AnyFlowBatch):
        engine = self._select(batch.modality)
        kwargs = {
            "text_tokens": batch.text_tokens,
            "text_lengths": batch.text_lengths,
            "speaker_embedding": batch.speaker_embedding,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if batch.modality == "video":
            return engine.training_loss(batch.clean, source=batch.source)
        return engine.training_loss(batch.clean, source=batch.source, **kwargs)

    def optimizer_step(
        self,
        batch: AnyFlowBatch,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None = None,
        grad_clip: float | None = 1.0,
    ) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        out = self.training_loss(batch)
        loss = out.loss
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip)
            optimizer.step()
        return loss.detach()

    @torch.no_grad()
    def generate(self, modality: Modality, shape: tuple[int, ...], device: torch.device | str, steps: int | None = None, **kwargs):
        return self._select(modality).generate(shape, device=device, steps=steps, **kwargs)

    @torch.no_grad()
    def generate_audio_chunked(
        self,
        modality: Literal["tts", "music"],
        shape: tuple[int, ...],
        device: torch.device | str,
        steps: int | None = None,
        chunk_length: int = 2048,
        overlap: int = 128,
        **kwargs,
    ) -> torch.Tensor:
        if modality not in {"tts", "music"}:
            raise ValueError("chunked audio generation supports only 'tts' or 'music'")
        engine = self._select(modality)
        return engine.generate_chunked(
            shape,
            device=device,
            steps=steps,
            chunk_length=chunk_length,
            overlap=overlap,
            **kwargs,
        )


def build_anyflow_small(
    video_channels: int = 4,
    audio_channels: int = 1,
    vocab_size: int = 512,
    dim: int = 128,
    runtime: AnyFlowRuntimeConfig | None = None,
    use_moe: bool = False,
) -> AnyFlowEngine:
    video = DeepThinkingVideoFlowEngine(
        VideoVelocityNet(
            in_channels=video_channels,
            dim=dim,
            depth=3,
            patch_size=(1, 2, 2),
            use_moe=use_moe,
            moe_num_experts=2,
            moe_top_k=1,
        )
    )
    tts = DeepThinkingAudioFlowEngine(
        AudioVelocityNet(
            channels=audio_channels,
            dim=dim,
            depth=3,
            vocab_size=vocab_size,
            tracks=1,
            use_moe=use_moe,
            moe_modality="tts",
            moe_num_experts=2,
            moe_top_k=1,
        ),
        mode="tts",
    )
    music = DeepThinkingAudioFlowEngine(
        AudioVelocityNet(
            channels=audio_channels,
            dim=dim,
            depth=4,
            vocab_size=vocab_size,
            tracks=4,
            use_moe=use_moe,
            moe_modality="music",
            moe_num_experts=2,
            moe_top_k=1,
        ),
        mode="music",
    )
    return AnyFlowEngine(video=video, tts=tts, music=music, runtime=runtime)
