from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .engine import AnyFlowBatch, Modality


@dataclass(slots=True)
class SyntheticAnyFlowDataConfig:
    batch_size: int = 1
    video_channels: int = 2
    video_frames: int = 1
    video_height: int = 8
    video_width: int = 8
    audio_channels: int = 1
    audio_length: int = 256
    music_tracks: int = 4
    vocab_size: int = 64
    tts_text_max_len: int = 16
    music_text_max_len: int = 12
    text_min_len: int = 3
    ragged_text: bool = True
    speaker_dim: int | None = None
    seed: int = 1337


def _generator(device: torch.device | str, seed: int) -> torch.Generator:
    target = torch.device(device)
    gen = torch.Generator(device=target if target.type == "cuda" else "cpu")
    gen.manual_seed(seed)
    return gen


def _randn(shape: tuple[int, ...], device: torch.device, generator: torch.Generator) -> torch.Tensor:
    kwargs = {"device": device}
    if generator.device.type == device.type:
        kwargs["generator"] = generator
    return torch.randn(shape, **kwargs)


def _randint(
    low: int,
    high: int,
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    kwargs = {"device": device}
    if generator.device.type == device.type:
        kwargs["generator"] = generator
    return torch.randint(low, high, shape, **kwargs)


def _text_condition(
    cfg: SyntheticAnyFlowDataConfig,
    max_len: int,
    device: torch.device,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = _randint(0, cfg.vocab_size, (cfg.batch_size, max_len), device, generator)
    if cfg.ragged_text:
        min_len = max(1, min(cfg.text_min_len, max_len))
        lengths = _randint(min_len, max_len + 1, (cfg.batch_size,), device, generator)
    else:
        lengths = torch.full((cfg.batch_size,), max_len, device=device, dtype=torch.long)
    mask = torch.arange(max_len, device=device).unsqueeze(0) >= lengths.unsqueeze(1)
    tokens = tokens.masked_fill(mask, 0)
    return tokens, lengths


def make_synthetic_anyflow_batch(
    modality: Modality,
    cfg: SyntheticAnyFlowDataConfig | None = None,
    *,
    device: torch.device | str = "cpu",
    step: int = 0,
    generator: torch.Generator | None = None,
) -> AnyFlowBatch:
    config = cfg or SyntheticAnyFlowDataConfig()
    target = torch.device(device)
    gen = generator or _generator(target, config.seed + step)

    if modality == "video":
        clean = _randn(
            (
                config.batch_size,
                config.video_channels,
                config.video_frames,
                config.video_height,
                config.video_width,
            ),
            target,
            gen,
        )
        return AnyFlowBatch(modality="video", clean=clean, metadata={"synthetic_step": step})

    if modality == "tts":
        clean = _randn((config.batch_size, config.audio_channels, config.audio_length), target, gen)
        tokens, lengths = _text_condition(config, config.tts_text_max_len, target, gen)
        speaker = (
            _randn((config.batch_size, config.speaker_dim), target, gen)
            if config.speaker_dim is not None
            else None
        )
        return AnyFlowBatch(
            modality="tts",
            clean=clean,
            text_tokens=tokens,
            text_lengths=lengths,
            speaker_embedding=speaker,
            metadata={"synthetic_step": step},
        )

    if modality == "music":
        clean = _randn(
            (config.batch_size, config.music_tracks, config.audio_channels, config.audio_length),
            target,
            gen,
        )
        tokens, lengths = _text_condition(config, config.music_text_max_len, target, gen)
        return AnyFlowBatch(
            modality="music",
            clean=clean,
            text_tokens=tokens,
            text_lengths=lengths,
            metadata={"synthetic_step": step},
        )

    raise ValueError(f"unknown modality: {modality}")


def iter_synthetic_anyflow_batches(
    steps: int,
    cfg: SyntheticAnyFlowDataConfig | None = None,
    *,
    device: torch.device | str = "cpu",
    modalities: Sequence[Modality] = ("video", "tts", "music"),
) -> Iterable[AnyFlowBatch]:
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if not modalities:
        raise ValueError("modalities cannot be empty")
    config = cfg or SyntheticAnyFlowDataConfig()
    target = torch.device(device)
    gen = _generator(target, config.seed)
    for step in range(steps):
        modality = modalities[step % len(modalities)]
        yield make_synthetic_anyflow_batch(modality, config, device=target, step=step, generator=gen)
