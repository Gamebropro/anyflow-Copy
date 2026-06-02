from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class AuxiliaryLossConfig:
    audio_stft_weight: float = 0.0
    audio_temporal_weight: float = 0.0
    video_temporal_weight: float = 0.0
    stft_fft_sizes: tuple[int, ...] = (64, 128, 256)
    eps: float = 1e-5

    @property
    def enabled(self) -> bool:
        return any(
            weight > 0.0
            for weight in (self.audio_stft_weight, self.audio_temporal_weight, self.video_temporal_weight)
        )


def _zero_like_loss(x: torch.Tensor) -> torch.Tensor:
    return x.sum() * 0.0


def _flatten_audio(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        b, c, length = x.shape
        return x.reshape(b * c, length)
    if x.ndim == 4:
        b, tracks, channels, length = x.shape
        return x.reshape(b * tracks * channels, length)
    raise ValueError("audio tensors must be [B,L], [B,C,L], or [B,Tracks,C,L]")


def temporal_difference_loss(pred: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    dim = dim if dim >= 0 else pred.ndim + dim
    if pred.shape[dim] < 2:
        return _zero_like_loss(pred)
    pred_diff = pred.diff(dim=dim)
    target_diff = target.diff(dim=dim)
    return F.l1_loss(pred_diff, target_diff)


def multi_resolution_stft_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    fft_sizes: tuple[int, ...] = (64, 128, 256),
    eps: float = 1e-5,
) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    pred_flat = _flatten_audio(pred.float())
    target_flat = _flatten_audio(target.float())
    length = pred_flat.shape[-1]
    if length < 2:
        return _zero_like_loss(pred)

    total = pred_flat.new_tensor(0.0)
    count = 0
    seen: set[int] = set()
    for requested_fft in fft_sizes:
        n_fft = max(2, min(int(requested_fft), length))
        if n_fft in seen:
            continue
        seen.add(n_fft)
        hop = max(1, n_fft // 4)
        window = torch.hann_window(n_fft, device=pred_flat.device, dtype=pred_flat.dtype)
        pred_spec = torch.stft(
            pred_flat,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        target_spec = torch.stft(
            target_flat,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )
        pred_mag = pred_spec.abs()
        target_mag = target_spec.abs()
        total = total + F.l1_loss(pred_mag, target_mag)
        total = total + F.l1_loss((pred_mag + eps).log(), (target_mag + eps).log())
        count += 2
    return total / max(count, 1)


def video_temporal_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    if pred.ndim != 5:
        raise ValueError("video tensors must be [B,C,T,H,W]")
    return temporal_difference_loss(pred, target, dim=2)


def auxiliary_flow_loss(
    pred_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    modality: str,
    config: AuxiliaryLossConfig | None,
) -> torch.Tensor:
    if config is None or not config.enabled:
        return _zero_like_loss(pred_velocity)

    total = _zero_like_loss(pred_velocity)
    if modality in {"tts", "music"}:
        if config.audio_stft_weight > 0.0:
            total = total + config.audio_stft_weight * multi_resolution_stft_loss(
                pred_velocity,
                target_velocity,
                fft_sizes=config.stft_fft_sizes,
                eps=config.eps,
            )
        if config.audio_temporal_weight > 0.0:
            total = total + config.audio_temporal_weight * temporal_difference_loss(
                pred_velocity,
                target_velocity,
                dim=-1,
            )
        return total

    if modality == "video":
        if config.video_temporal_weight > 0.0:
            total = total + config.video_temporal_weight * video_temporal_consistency_loss(pred_velocity, target_velocity)
        return total

    raise ValueError(f"unknown modality: {modality}")
