from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
import torch.nn.functional as F


try:
    import tilelang
    import tilelang.language as T
except Exception:
    tilelang = None
    T = None


def tilelang_available() -> bool:
    return tilelang is not None and T is not None and hasattr(tilelang, "jit")


def mamba_scan_ref(
    u: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b_vec: torch.Tensor,
    c_vec: torch.Tensor,
    skip: torch.Tensor,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if u.ndim != 3:
        raise ValueError("u must be [batch, seqlen, channels]")
    if dt.shape != u.shape:
        raise ValueError("dt must match u shape")
    if a.shape[0] != u.shape[-1]:
        raise ValueError("a must be [channels, d_state]")

    batch, seqlen, channels = u.shape
    d_state = a.shape[-1]
    state = torch.zeros(batch, channels, d_state, device=u.device, dtype=torch.float32)
    outputs: list[torch.Tensor] = []

    a32 = -torch.exp(a.float())
    skip32 = skip.float()
    dt32 = dt.float()
    if dt_bias is not None:
        dt32 = dt32 + dt_bias.float().view(1, 1, channels)
    dt32 = F.softplus(dt32)

    b32 = b_vec.float()
    c32 = c_vec.float()
    u32 = u.float()

    for step in range(seqlen):
        delta = dt32[:, step]
        decay = torch.exp(delta.unsqueeze(-1) * a32.unsqueeze(0))
        drive = delta.unsqueeze(-1) * b32[:, step].unsqueeze(1) * u32[:, step].unsqueeze(-1)
        state = decay * state + drive
        y = (state * c32[:, step].unsqueeze(1)).sum(dim=-1) + skip32.view(1, channels) * u32[:, step]
        outputs.append(y.to(u.dtype))

    out = torch.stack(outputs, dim=1)
    if z is not None:
        out = out * F.silu(z)
    return out


def asymmetric_flow_step_ref(x: torch.Tensor, velocity: torch.Tensor, dt: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(dt):
        dt = torch.tensor(dt, device=x.device, dtype=x.dtype)
    while dt.ndim < x.ndim:
        dt = dt.view(*dt.shape, 1)
    return x + velocity * dt.to(dtype=x.dtype)


@lru_cache(maxsize=32)
def build_tilelang_asym_flow_step(numel: int, block: int = 256) -> Any | None:
    if not tilelang_available():
        return None

    @tilelang.jit(out_idx=[3])
    def _kernel(n: int, block_size: int):
        dtype = T.float32

        @T.prim_func
        def main(
            X: T.Tensor((n,), dtype),
            V: T.Tensor((n,), dtype),
            Dt: T.Tensor((1,), dtype),
            Out: T.Tensor((n,), dtype),
        ):
            with T.Kernel(T.ceildiv(n, block_size), threads=block_size) as bx:
                for i in T.Parallel(block_size):
                    idx = bx * block_size + i
                    if idx < n:
                        Out[idx] = X[idx] + V[idx] * Dt[0]

        return main

    return _kernel(numel, block)


@lru_cache(maxsize=16)
def build_tilelang_mamba_scan(batch: int, seqlen: int, channels: int, d_state: int, block_d: int = 64) -> Any | None:
    if not tilelang_available():
        return None

    @tilelang.jit(out_idx=[6])
    def _kernel(bsz: int, length: int, dim: int, state_dim: int, block_dim: int):
        dtype = T.float32

        @T.prim_func
        def main(
            U: T.Tensor((bsz, length, dim), dtype),
            Dt: T.Tensor((bsz, length, dim), dtype),
            A: T.Tensor((dim, state_dim), dtype),
            Bv: T.Tensor((bsz, length, state_dim), dtype),
            Cv: T.Tensor((bsz, length, state_dim), dtype),
            Skip: T.Tensor((dim,), dtype),
            Out: T.Tensor((bsz, length, dim), dtype),
        ):
            with T.Kernel(T.ceildiv(dim, block_dim), bsz, threads=128) as (bd, bb):
                state = T.alloc_fragment((block_dim, state_dim), dtype)
                for i, j in T.Parallel(block_dim, state_dim):
                    state[i, j] = 0.0
                for t in T.serial(length):
                    for i, j in T.Parallel(block_dim, state_dim):
                        c = bd * block_dim + i
                        if c < dim:
                            delta = T.log(1.0 + T.exp(Dt[bb, t, c]))
                            state[i, j] = T.exp(delta * A[c, j]) * state[i, j] + delta * Bv[bb, t, j] * U[bb, t, c]
                    for i in T.Parallel(block_dim):
                        c = bd * block_dim + i
                        acc = T.alloc_fragment((1,), dtype)
                        acc[0] = 0.0
                        if c < dim:
                            for j in T.serial(state_dim):
                                acc[0] += state[i, j] * Cv[bb, t, j]
                            Out[bb, t, c] = acc[0] + Skip[c] * U[bb, t, c]

        return main

    return _kernel(batch, seqlen, channels, d_state, block_d)


def mamba_scan_tilelang_or_ref(
    u: torch.Tensor,
    dt: torch.Tensor,
    a: torch.Tensor,
    b_vec: torch.Tensor,
    c_vec: torch.Tensor,
    skip: torch.Tensor,
    z: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    prefer_tilelang: bool = True,
) -> torch.Tensor:
    if prefer_tilelang and u.is_cuda and u.dtype == torch.float32:
        kernel = build_tilelang_mamba_scan(u.shape[0], u.shape[1], u.shape[2], a.shape[1])
        if kernel is not None:
            try:
                dt_eff = dt
                if dt_bias is not None:
                    dt_eff = dt_eff + dt_bias.view(1, 1, -1)
                a_eff = -torch.exp(a.float())
                out = kernel(
                    u.contiguous(),
                    dt_eff.float().contiguous(),
                    a_eff.contiguous(),
                    b_vec.float().contiguous(),
                    c_vec.float().contiguous(),
                    skip.float().contiguous(),
                )
                if z is not None:
                    out = out * F.silu(z.float())
                return out.to(dtype=u.dtype)
            except Exception:
                pass
    return mamba_scan_ref(u, dt, a, b_vec, c_vec, skip, z=z, dt_bias=dt_bias)
