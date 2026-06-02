from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .packed import pack_padded, valid_token_mask
from .tilelang_kernels import mamba_scan_tilelang_or_ref


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep).div_(keep)


@dataclass(slots=True)
class Mamba3Config:
    dim: int
    state_dim: int = 16
    expansion: int = 2
    dt_rank: int | str = "auto"
    conv_kernel: int = 3
    dropout: float = 0.0
    drop_path: float = 0.0
    use_varlen_mixer: bool = False
    mixer_heads: int = 4
    prefer_tilelang: bool = True
    attention_residual: bool = True
    residual_block_size: int = 4
    residual_heads: int = 4
    residual_groups: int = 1

    @property
    def inner_dim(self) -> int:
        return self.dim * self.expansion

    @property
    def resolved_dt_rank(self) -> int:
        return math.ceil(self.dim / 16) if self.dt_rank == "auto" else int(self.dt_rank)


class VarLenResidualMixer(nn.Module):
    """Packed local SSM mixer kept for API compatibility.

    This class intentionally does not use token-token attention. It handles
    ragged batches through packed lengths and applies a depthwise local SSM-like
    convolution plus projection to remove padding work in text/audio paths.
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.in_proj = nn.Linear(dim, dim * 2, bias=False)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        batch, seqlen, dim = x.shape
        if lengths is None:
            u, gate = self.in_proj(x).chunk(2, dim=-1)
            y = self.depthwise(u.transpose(1, 2)).transpose(1, 2)
            y = y * F.silu(gate)
            return self.out(F.dropout(y, p=self.dropout, training=self.training))

        packed = pack_padded(x, lengths)
        out = x.new_zeros(batch, seqlen, dim)
        start = 0
        for i, length in enumerate(lengths.tolist()):
            end = start + int(length)
            seq = packed.values[start:end].unsqueeze(0)
            u, gate = self.in_proj(seq).chunk(2, dim=-1)
            y = self.depthwise(u.transpose(1, 2)).transpose(1, 2)[:, : int(length)]
            y = self.out(F.dropout(y * F.silu(gate[:, : int(length)]), p=self.dropout, training=self.training))
            out[i, : int(length)] = y.squeeze(0)
            start = end
        return out


class DiagonalStateSpace(nn.Module):
    def __init__(self, channels: int, state_dim: int, prefer_tilelang: bool = True) -> None:
        super().__init__()
        a = torch.arange(1, state_dim + 1, dtype=torch.float32).repeat(channels, 1)
        self.a_log = nn.Parameter(torch.log(a))
        self.a_log._no_weight_decay = True
        self.skip = nn.Parameter(torch.ones(channels))
        self.skip._no_weight_decay = True
        self.dt_bias = nn.Parameter(torch.zeros(channels))
        self.prefer_tilelang = prefer_tilelang

    def forward(self, u: torch.Tensor, dt: torch.Tensor, b_vec: torch.Tensor, c_vec: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return mamba_scan_tilelang_or_ref(
            u,
            dt,
            self.a_log,
            b_vec,
            c_vec,
            self.skip,
            z=z,
            dt_bias=self.dt_bias,
            prefer_tilelang=self.prefer_tilelang,
        )


class Mamba3Block(nn.Module):
    def __init__(self, config: Mamba3Config) -> None:
        super().__init__()
        self.config = config
        inner = config.inner_dim
        dt_rank = config.resolved_dt_rank
        fused_dim = inner * 2 + dt_rank + config.state_dim * 2
        self.norm = RMSNorm(config.dim)
        self.in_proj = nn.Linear(config.dim, fused_dim, bias=False)
        self.depthwise = nn.Conv1d(
            inner,
            inner,
            kernel_size=config.conv_kernel,
            padding=config.conv_kernel - 1,
            groups=inner,
            bias=True,
        )
        self.dt_proj = nn.Linear(dt_rank, inner, bias=True)
        self.ssm = DiagonalStateSpace(inner, config.state_dim, prefer_tilelang=config.prefer_tilelang)
        self.out_norm = RMSNorm(inner)
        self.out_proj = nn.Linear(inner, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.drop_path = DropPath(config.drop_path)
        self.varlen_mixer = (
            VarLenResidualMixer(config.dim, heads=config.mixer_heads, dropout=config.dropout)
            if config.use_varlen_mixer
            else None
        )

        nn.init.normal_(self.dt_proj.weight, std=dt_rank**-0.5)
        nn.init.constant_(self.dt_proj.bias, -3.0)

    def _forward_ragged(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        out = x.new_zeros(x.shape)
        for i, length in enumerate(lengths.tolist()):
            length_i = int(length)
            if length_i <= 0:
                continue
            out[i : i + 1, :length_i] = self._forward_dense(x[i : i + 1, :length_i], lengths=None)
        return out

    def _forward_dense(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        fused = self.in_proj(h)
        inner = self.config.inner_dim
        dt_rank = self.config.resolved_dt_rank
        u, z, dt_low, b_vec, c_vec = torch.split(
            fused,
            [inner, inner, dt_rank, self.config.state_dim, self.config.state_dim],
            dim=-1,
        )
        u = self.depthwise(u.transpose(1, 2))[..., : x.shape[1]].transpose(1, 2)
        u = F.silu(u)
        dt = self.dt_proj(dt_low)
        y = self.ssm(u, dt, b_vec, c_vec, z)
        y = self.out_proj(self.out_norm(y))
        out = residual + self.drop_path(self.dropout(y))

        if self.varlen_mixer is not None:
            mixed = self.varlen_mixer(self.norm(out), lengths=lengths)
            if lengths is not None:
                mask = valid_token_mask(lengths, out.shape[1]).unsqueeze(-1).to(out.dtype)
                mixed = mixed * mask
            out = out + self.drop_path(self.dropout(mixed))

        return out

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is not None:
            if lengths.numel() != x.shape[0]:
                raise ValueError("lengths must have one entry per batch element")
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp(0, x.shape[1])
            if bool((lengths < x.shape[1]).any().item()):
                return self._forward_ragged(x, lengths)
        return self._forward_dense(x, lengths=lengths)


class Mamba3Stack(nn.Module):
    def __init__(self, config: Mamba3Config, depth: int) -> None:
        super().__init__()
        self.config = config
        self.blocks = nn.ModuleList([Mamba3Block(config) for _ in range(depth)])
        self.norm = RMSNorm(config.dim)
        self.attention_residual_stack = None
        if config.attention_residual:
            from .attention_residual import AttentionResidualMambaStack

            self.attention_residual_stack = AttentionResidualMambaStack(
                self.blocks,
                dim=config.dim,
                heads=config.residual_heads,
                groups=config.residual_groups,
                block_size=config.residual_block_size,
                output_norm=self.norm,
            )

    def _forward_dense(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if self.attention_residual_stack is not None:
            return self.attention_residual_stack(x, lengths=lengths)
        for block in self.blocks:
            x = block(x, lengths=lengths)
        return self.norm(x)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if lengths is not None:
            if lengths.numel() != x.shape[0]:
                raise ValueError("lengths must have one entry per batch element")
            lengths = lengths.to(device=x.device, dtype=torch.long).clamp(0, x.shape[1])
            if bool((lengths < x.shape[1]).any().item()):
                out = x.new_zeros(x.shape)
                for i, length in enumerate(lengths.tolist()):
                    length_i = int(length)
                    if length_i <= 0:
                        continue
                    out[i : i + 1, :length_i] = self._forward_dense(x[i : i + 1, :length_i], lengths=None)
                return out
        return self._forward_dense(x, lengths=lengths)
