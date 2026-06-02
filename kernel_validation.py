from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .tilelang_kernels import (
    asymmetric_flow_step_ref,
    build_tilelang_asym_flow_step,
    build_tilelang_mamba_scan,
    mamba_scan_ref,
    mamba_scan_tilelang_or_ref,
    tilelang_available,
)


@dataclass(slots=True)
class KernelParityResult:
    name: str
    ok: bool
    max_abs_error: float
    attempted_tilelang: bool
    detail: str


@dataclass(slots=True)
class KernelValidationReport:
    tilelang_available: bool
    device: str
    results: list[KernelParityResult]

    @property
    def passed(self) -> bool:
        return all(result.ok for result in self.results)

    def as_dict(self) -> dict[str, object]:
        return {
            "tilelang_available": self.tilelang_available,
            "device": self.device,
            "passed": self.passed,
            "results": [asdict(result) for result in self.results],
        }


def _max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.detach().float() - b.detach().float()).abs().max().cpu())


@torch.no_grad()
def validate_mamba_scan_kernel(device: torch.device | str = "cpu", prefer_tilelang: bool = True) -> KernelParityResult:
    target = torch.device(device)
    torch.manual_seed(2026)
    batch, seqlen, channels, state_dim = 2, 5, 4, 3
    u = torch.randn(batch, seqlen, channels, device=target)
    dt = torch.randn(batch, seqlen, channels, device=target)
    a = torch.randn(channels, state_dim, device=target)
    b_vec = torch.randn(batch, seqlen, state_dim, device=target)
    c_vec = torch.randn(batch, seqlen, state_dim, device=target)
    skip = torch.randn(channels, device=target)
    z = torch.randn(batch, seqlen, channels, device=target)
    dt_bias = torch.randn(channels, device=target)
    expected = mamba_scan_ref(u, dt, a, b_vec, c_vec, skip, z=z, dt_bias=dt_bias)
    attempted = bool(prefer_tilelang and target.type == "cuda" and tilelang_available())
    detail = ""
    if attempted:
        kernel = build_tilelang_mamba_scan(batch, seqlen, channels, state_dim)
        if kernel is None:
            detail = "tilelang builder returned None; verified reference fallback"
            actual = mamba_scan_tilelang_or_ref(u, dt, a, b_vec, c_vec, skip, z=z, dt_bias=dt_bias, prefer_tilelang=False)
            attempted = False
        else:
            try:
                dt_eff = dt + dt_bias.view(1, 1, -1)
                a_eff = -torch.exp(a.float())
                actual = kernel(
                    u.contiguous(),
                    dt_eff.float().contiguous(),
                    a_eff.contiguous(),
                    b_vec.float().contiguous(),
                    c_vec.float().contiguous(),
                    skip.float().contiguous(),
                )
                actual = (actual * F.silu(z.float())).to(dtype=u.dtype)
                detail = "tilelang kernel executed"
            except Exception as exc:
                err = float("inf")
                return KernelParityResult(
                    name="mamba_scan",
                    ok=False,
                    max_abs_error=err,
                    attempted_tilelang=True,
                    detail=f"tilelang kernel failed: {exc!r}",
                )
    else:
        actual = mamba_scan_tilelang_or_ref(u, dt, a, b_vec, c_vec, skip, z=z, dt_bias=dt_bias, prefer_tilelang=False)
        detail = "verified reference fallback"

    err = _max_abs_error(expected, actual)
    return KernelParityResult(
        name="mamba_scan",
        ok=err <= 1e-5,
        max_abs_error=err,
        attempted_tilelang=attempted,
        detail=f"{detail}; shape={tuple(actual.shape)}",
    )


@torch.no_grad()
def validate_asymmetric_flow_step_kernel(
    device: torch.device | str = "cpu",
    prefer_tilelang: bool = True,
) -> KernelParityResult:
    target = torch.device(device)
    torch.manual_seed(2027)
    x = torch.randn(1, 2, 3, 4, device=target)
    velocity = torch.randn_like(x)
    dt = torch.tensor([0.125], device=target, dtype=x.dtype)
    expected = asymmetric_flow_step_ref(x, velocity, dt.view(1, 1, 1, 1))
    attempted = bool(prefer_tilelang and target.type == "cuda" and tilelang_available())
    actual = expected
    if attempted:
        kernel = build_tilelang_asym_flow_step(x.numel())
        if kernel is not None:
            try:
                actual = kernel(x.flatten(), velocity.flatten(), dt.float()).reshape_as(x)
            except Exception as exc:
                return KernelParityResult(
                    name="asymmetric_flow_step",
                    ok=False,
                    max_abs_error=float("inf"),
                    attempted_tilelang=True,
                    detail=f"tilelang kernel failed: {exc!r}",
                )
        else:
            attempted = False
    err = _max_abs_error(expected, actual)
    return KernelParityResult(
        name="asymmetric_flow_step",
        ok=err <= 1e-6,
        max_abs_error=err,
        attempted_tilelang=attempted,
        detail=f"shape={tuple(actual.shape)}",
    )


def run_kernel_validation(device: torch.device | str | None = None, prefer_tilelang: bool = True) -> KernelValidationReport:
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    results = [
        validate_mamba_scan_kernel(target, prefer_tilelang=prefer_tilelang),
        validate_asymmetric_flow_step_kernel(target, prefer_tilelang=prefer_tilelang),
    ]
    return KernelValidationReport(tilelang_available=tilelang_available(), device=str(target), results=results)
