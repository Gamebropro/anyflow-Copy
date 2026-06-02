from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from .tilelang_kernels import asymmetric_flow_step_ref, build_tilelang_asym_flow_step


def append_dims(t: torch.Tensor, target_ndim: int) -> torch.Tensor:
    while t.ndim < target_ndim:
        t = t.unsqueeze(-1)
    return t


@dataclass(slots=True)
class FlowSample:
    x_t: torch.Tensor
    target_velocity: torch.Tensor
    time: torch.Tensor
    clean: torch.Tensor
    source: torch.Tensor


@dataclass(slots=True)
class AsymmetricFlowSchedule:
    skew: float = 1.35
    eps: float = 1e-4
    min_time: float = 1e-3
    max_time: float = 1.0 - 1e-3

    def sample_time(self, batch: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        t = torch.rand(batch, device=device, dtype=dtype)
        t = t.clamp(self.min_time, self.max_time)
        return t

    def warp(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma = torch.as_tensor(self.skew, device=t.device, dtype=t.dtype)
        t = t.clamp(self.eps, 1.0 - self.eps)
        a = t.pow(gamma)
        b = (1.0 - t).pow(gamma)
        denom = (a + b).clamp_min(self.eps)
        s = a / denom
        da = gamma * t.pow(gamma - 1.0)
        db = -gamma * (1.0 - t).pow(gamma - 1.0)
        ds_dt = (da * denom - a * (da + db)) / denom.pow(2)
        return s, ds_dt


class AsymmetricFlowMatcher(nn.Module):
    def __init__(self, schedule: AsymmetricFlowSchedule | None = None, loss: str = "pseudo_huber") -> None:
        super().__init__()
        self.schedule = schedule or AsymmetricFlowSchedule()
        if loss not in {"mse", "l1", "pseudo_huber"}:
            raise ValueError("loss must be mse, l1, or pseudo_huber")
        self.loss = loss

    def make_sample(self, clean: torch.Tensor, source: torch.Tensor | None = None) -> FlowSample:
        if source is None:
            source = torch.randn_like(clean)
        time = self.schedule.sample_time(clean.shape[0], clean.device, torch.float32)
        s, ds_dt = self.schedule.warp(time)
        s_view = append_dims(s.to(clean.dtype), clean.ndim)
        ds_view = append_dims(ds_dt.to(clean.dtype), clean.ndim)
        x_t = source.lerp(clean, s_view)
        target = (clean - source) * ds_view
        return FlowSample(x_t=x_t, target_velocity=target, time=time, clean=clean, source=source)

    def loss_value(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss == "mse":
            return F.mse_loss(pred, target)
        if self.loss == "l1":
            return F.l1_loss(pred, target)
        c = 5.4e-4 * pred[0].numel() ** 0.5
        return (F.mse_loss(pred, target, reduction="none") + c * c).sqrt().sub(c).mean()

    def forward(
        self,
        model: nn.Module,
        clean: torch.Tensor,
        source: torch.Tensor | None = None,
        **model_kwargs,
    ) -> tuple[torch.Tensor, FlowSample, torch.Tensor]:
        sample = self.make_sample(clean, source=source)
        pred = model(sample.x_t, sample.time, **model_kwargs)
        return self.loss_value(pred, sample.target_velocity), sample, pred


class FlowSampler:
    def __init__(self, schedule: AsymmetricFlowSchedule | None = None, steps: int = 32, prefer_tilelang: bool = True) -> None:
        self.schedule = schedule or AsymmetricFlowSchedule()
        self.steps = steps
        self.prefer_tilelang = prefer_tilelang

    @torch.no_grad()
    def euler(
        self,
        model: nn.Module | Callable[..., torch.Tensor],
        source: torch.Tensor,
        steps: int | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        total_steps = int(steps or self.steps)
        x = source
        times = torch.linspace(0.0, 1.0, total_steps + 1, device=x.device, dtype=torch.float32)
        for i in range(total_steps):
            t0 = times[i].expand(x.shape[0]).clamp(self.schedule.min_time, self.schedule.max_time)
            t1 = times[i + 1].expand(x.shape[0]).clamp(self.schedule.min_time, self.schedule.max_time)
            dt = append_dims((t1 - t0).to(x.dtype), x.ndim)
            pred = model(x, t0, **model_kwargs)
            if self.prefer_tilelang and x.is_cuda and x.dtype == torch.float32 and dt.numel() == x.shape[0]:
                kernel = build_tilelang_asym_flow_step(x.numel())
                if kernel is not None and x.shape[0] == 1:
                    flat_dt = dt.reshape(1).float()
                    try:
                        x = kernel(x.flatten(), pred.flatten(), flat_dt).reshape_as(x)
                        continue
                    except Exception:
                        pass
            x = asymmetric_flow_step_ref(x, pred, dt)
        return x
