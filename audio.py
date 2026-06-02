from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .flow import AsymmetricFlowMatcher, AsymmetricFlowSchedule, FlowSampler
from .losses import AuxiliaryLossConfig, auxiliary_flow_loss
from .mamba3 import Mamba3Config, Mamba3Stack, RMSNorm, VarLenResidualMixer
from .moe import AnyFlowMoEAdapter, make_moe_config_for_modality


class AudioStem(nn.Module):
    def __init__(self, channels: int, dim: int, stride: int = 2) -> None:
        super().__init__()
        self.stride = stride
        self.proj = nn.Conv1d(channels, dim, kernel_size=7, stride=stride, padding=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).transpose(1, 2).contiguous()


class AudioHead(nn.Module):
    def __init__(self, dim: int, channels: int, stride: int = 2) -> None:
        super().__init__()
        self.proj = nn.ConvTranspose1d(dim, channels, kernel_size=2 * stride, stride=stride, padding=stride // 2)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        y = self.proj(x.transpose(1, 2).contiguous())
        return y[..., :target_len]


class TextConditioner(nn.Module):
    def __init__(self, vocab_size: int, dim: int, depth: int = 4, heads: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        cfg = Mamba3Config(
            dim=dim,
            state_dim=16,
            expansion=2,
            use_varlen_mixer=True,
            mixer_heads=heads,
            attention_residual=True,
            residual_heads=heads,
            residual_groups=max(1, heads // 2),
        )
        self.encoder = Mamba3Stack(cfg, depth=depth)
        self.varlen = VarLenResidualMixer(dim, heads=heads)
        self.norm = RMSNorm(dim)

    def forward(self, tokens: torch.Tensor | None, lengths: torch.Tensor | None) -> torch.Tensor | None:
        if tokens is None:
            return None
        if lengths is None:
            lengths = torch.full((tokens.shape[0],), tokens.shape[1], device=tokens.device, dtype=torch.long)
        x = self.embedding(tokens)
        x = self.encoder(x, lengths=lengths)
        x = x + self.varlen(self.norm(x), lengths=lengths)
        mask = (torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]).unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / lengths.clamp_min(1).to(x.dtype).unsqueeze(-1)
        return pooled


class AudioVelocityNet(nn.Module):
    def __init__(
        self,
        channels: int = 1,
        dim: int = 256,
        depth: int = 8,
        vocab_size: int = 512,
        tracks: int = 1,
        stride: int = 2,
        use_moe: bool = False,
        moe_modality: str | None = None,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_aux_loss_weight: float = 1.0,
        moe_prefer_tilelang: bool = True,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.tracks = tracks
        self.flat_channels = channels * tracks
        self.moe_modality = moe_modality or ("music" if tracks > 1 else "tts")
        self.moe_aux_loss_weight = float(moe_aux_loss_weight)
        self.last_moe_aux_loss: torch.Tensor | None = None
        self.stem = AudioStem(self.flat_channels, dim, stride=stride)
        self.time_mlp = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.text = TextConditioner(vocab_size, dim, depth=max(2, depth // 2))
        self.speaker = nn.Linear(dim, dim, bias=False)
        cfg = Mamba3Config(
            dim=dim,
            state_dim=16,
            expansion=2,
            use_varlen_mixer=True,
            attention_residual=True,
            residual_heads=4 if dim % 4 == 0 else 1,
            residual_groups=1,
        )
        self.backbone = Mamba3Stack(cfg, depth=depth)
        self.moe: AnyFlowMoEAdapter | None = None
        if use_moe:
            moe_cfg = make_moe_config_for_modality(
                self.moe_modality,
                dim,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                prefer_tilelang=moe_prefer_tilelang,
            )
            moe_cfg.expert_depth = 1
            self.moe = AnyFlowMoEAdapter(moe_cfg)
        self.film = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 2))
        self.head = AudioHead(dim, self.flat_channels, stride=stride)

    def _flatten_tracks(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        original = tuple(x.shape)
        if x.ndim == 4:
            b, tracks, channels, length = x.shape
            return x.reshape(b, tracks * channels, length), original
        if x.ndim == 3:
            return x, original
        raise ValueError("audio input must be [B,C,L] or [B,Tracks,C,L]")

    def _restore_tracks(self, x: torch.Tensor, original: tuple[int, ...]) -> torch.Tensor:
        if len(original) == 4:
            b, tracks, channels, length = original
            return x.reshape(b, tracks, channels, length)
        return x

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        text_tokens: torch.Tensor | None = None,
        text_lengths: torch.Tensor | None = None,
        speaker_embedding: torch.Tensor | None = None,
        moe_task_ids: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        self.last_moe_aux_loss = None
        flat, original = self._flatten_tracks(x)
        y = self.stem(flat)
        cond = self.time_mlp(time.to(y.dtype).unsqueeze(-1))
        text_cond = self.text(text_tokens, text_lengths)
        if text_cond is not None:
            cond = cond + text_cond
        if speaker_embedding is not None:
            cond = cond + self.speaker(speaker_embedding.to(cond.dtype))
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        y = y * (1 + gamma[:, None]) + beta[:, None]
        y = self.backbone(y)
        if self.moe is not None:
            token_lengths = torch.full((y.shape[0],), y.shape[1], device=y.device, dtype=torch.long)
            moe_out = self.moe(
                y,
                lengths=token_lengths,
                modality=self.moe_modality,
                task_ids=moe_task_ids,
                return_router=True,
            )
            y = moe_out.hidden_states
            self.last_moe_aux_loss = moe_out.aux_loss * self.moe_aux_loss_weight
        flat_out = self.head(y, flat.shape[-1])
        return self._restore_tracks(flat_out, original)


@dataclass(slots=True)
class AudioTrainOutput:
    loss: torch.Tensor
    prediction: torch.Tensor
    noised: torch.Tensor
    time: torch.Tensor
    auxiliary_loss: torch.Tensor | None = None
    moe_auxiliary_loss: torch.Tensor | None = None


class DeepThinkingAudioFlowEngine(nn.Module):
    def __init__(
        self,
        model: AudioVelocityNet,
        mode: str,
        matcher: AsymmetricFlowMatcher | None = None,
        sampler: FlowSampler | None = None,
        auxiliary_loss_config: AuxiliaryLossConfig | None = None,
    ) -> None:
        super().__init__()
        if mode not in {"tts", "music"}:
            raise ValueError("mode must be 'tts' or 'music'")
        self.mode = mode
        self.model = model
        schedule = AsymmetricFlowSchedule(skew=1.25 if mode == "tts" else 1.55)
        self.matcher = matcher or AsymmetricFlowMatcher(schedule=schedule)
        self.sampler = sampler or FlowSampler(schedule=schedule, steps=48 if mode == "tts" else 64)
        self.auxiliary_loss_config = auxiliary_loss_config

    def forward(self, x_t: torch.Tensor, time: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x_t, time, **kwargs)

    def training_loss(self, clean_audio: torch.Tensor, source: torch.Tensor | None = None, **kwargs) -> AudioTrainOutput:
        loss, sample, pred = self.matcher(self.model, clean_audio, source=source, **kwargs)
        aux = auxiliary_flow_loss(pred, sample.target_velocity, self.mode, self.auxiliary_loss_config)
        moe_aux = getattr(self.model, "last_moe_aux_loss", None)
        total_aux = aux if moe_aux is None else aux + moe_aux
        return AudioTrainOutput(
            loss=loss + total_aux,
            prediction=pred,
            noised=sample.x_t,
            time=sample.time,
            auxiliary_loss=total_aux,
            moe_auxiliary_loss=moe_aux,
        )

    @torch.no_grad()
    def generate(self, shape: tuple[int, ...], device: torch.device | str, steps: int | None = None, **kwargs) -> torch.Tensor:
        source = torch.randn(shape, device=device)
        return self.sampler.euler(self.model, source, steps=steps, **kwargs)

    @torch.no_grad()
    def generate_chunked(
        self,
        shape: tuple[int, ...],
        device: torch.device | str,
        steps: int | None = None,
        chunk_length: int = 2048,
        overlap: int = 128,
        **kwargs,
    ) -> torch.Tensor:
        if len(shape) not in {3, 4}:
            raise ValueError("chunked audio shape must be [B,C,L] or [B,Tracks,C,L]")
        if chunk_length <= 0:
            raise ValueError("chunk_length must be positive")
        if overlap < 0 or overlap >= chunk_length:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_length")

        target = torch.device(device)
        total_length = int(shape[-1])
        if total_length <= chunk_length:
            return self.generate(shape, device=target, steps=steps, **kwargs)

        leading = shape[:-1]
        out = torch.zeros(shape, device=target)
        weights = torch.zeros(total_length, device=target, dtype=out.dtype)
        stride = chunk_length - overlap
        start = 0
        while start < total_length:
            end = min(start + chunk_length, total_length)
            chunk = self.generate((*leading, end - start), device=target, steps=steps, **kwargs)
            out[..., start:end] = out[..., start:end] + chunk
            weights[start:end] = weights[start:end] + 1
            if end == total_length:
                break
            start += stride

        view_shape = (1,) * (len(shape) - 1) + (total_length,)
        return out / weights.clamp_min(1).view(view_shape)
