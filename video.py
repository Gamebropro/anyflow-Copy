from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .flow import AsymmetricFlowMatcher, AsymmetricFlowSchedule, FlowSampler, append_dims
from .losses import AuxiliaryLossConfig, auxiliary_flow_loss
from .mamba3 import Mamba3Block, Mamba3Config, RMSNorm
from .moe import AnyFlowMoEAdapter, make_moe_config_for_modality


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(torch.linspace(0, math.log(10000), half, device=time.device, dtype=time.dtype) * -1)
        args = time[:, None] * freq[None]
        emb = torch.cat((args.sin(), args.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb


class VideoPatchEmbed(nn.Module):
    def __init__(self, in_channels: int, dim: int, patch_size: tuple[int, int, int]) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        pt, ph, pw = self.patch_size
        pad_t = (pt - x.shape[2] % pt) % pt
        pad_h = (ph - x.shape[3] % ph) % ph
        pad_w = (pw - x.shape[4] % pw) % pw
        if pad_t or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_t))
        y = self.proj(x)
        b, c, t, h, w = y.shape
        return y.permute(0, 2, 3, 4, 1).contiguous(), (t, h, w)


class SpatialMambaVideoBlock(nn.Module):
    def __init__(self, dim: int, state_dim: int = 16, expansion: int = 2, drop_path: float = 0.0) -> None:
        super().__init__()
        heads = 4 if dim % 4 == 0 else 1
        cfg = Mamba3Config(
            dim=dim,
            state_dim=state_dim,
            expansion=expansion,
            drop_path=drop_path,
            attention_residual=True,
            residual_heads=heads,
            residual_groups=1,
        )
        self.hw = Mamba3Block(cfg)
        self.hw_rev = Mamba3Block(cfg)
        self.wh = Mamba3Block(cfg)
        self.wh_rev = Mamba3Block(cfg)
        self.time = Mamba3Block(cfg)
        self.norm = RMSNorm(dim)
        self.fuse = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = x.shape
        hw = x.reshape(b * t, h * w, c)
        y_hw = self.hw(hw).reshape(b, t, h, w, c)
        y_hw_rev = torch.flip(self.hw_rev(torch.flip(hw, dims=(1,))), dims=(1,)).reshape(b, t, h, w, c)

        wh = x.transpose(2, 3).reshape(b * t, w * h, c)
        y_wh = self.wh(wh).reshape(b, t, w, h, c).transpose(2, 3)
        y_wh_rev = torch.flip(self.wh_rev(torch.flip(wh, dims=(1,))), dims=(1,)).reshape(b, t, w, h, c).transpose(2, 3)

        ts = x.permute(0, 2, 3, 1, 4).reshape(b * h * w, t, c)
        y_t = self.time(ts).reshape(b, h, w, t, c).permute(0, 3, 1, 2, 4)

        y = (y_hw + y_hw_rev + y_wh + y_wh_rev + y_t) / 5.0
        return x + self.fuse(self.norm(y))


class VideoVelocityNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        dim: int = 256,
        depth: int = 8,
        state_dim: int = 16,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        use_moe: bool = False,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_aux_loss_weight: float = 1.0,
        moe_prefer_tilelang: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.moe_aux_loss_weight = float(moe_aux_loss_weight)
        self.last_moe_aux_loss: torch.Tensor | None = None
        self.patch = VideoPatchEmbed(in_channels, dim, patch_size)
        self.time_embed = nn.Sequential(SinusoidalTimeEmbedding(dim), nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))
        self.blocks = nn.ModuleList(
            [SpatialMambaVideoBlock(dim=dim, state_dim=state_dim, drop_path=i / max(depth - 1, 1) * 0.05) for i in range(depth)]
        )
        self.moe: AnyFlowMoEAdapter | None = None
        if use_moe:
            moe_cfg = make_moe_config_for_modality(
                "video",
                dim,
                num_experts=moe_num_experts,
                top_k=moe_top_k,
                prefer_tilelang=moe_prefer_tilelang,
            )
            moe_cfg.state_dim = state_dim
            moe_cfg.expert_depth = 1
            self.moe = AnyFlowMoEAdapter(moe_cfg)
        pt, ph, pw = patch_size
        self.out = nn.Sequential(RMSNorm(dim), nn.Linear(dim, in_channels * pt * ph * pw))

    def _region_ids(self, batch: int, grid: tuple[int, int, int], device: torch.device) -> torch.Tensor | None:
        if self.moe is None:
            return None
        region_count = self.moe.layer.config.region_count
        if region_count <= 0:
            return None
        gt, gh, gw = grid
        spatial = torch.arange(gh * gw, device=device, dtype=torch.long).view(1, gh, gw)
        temporal = torch.arange(gt, device=device, dtype=torch.long).view(gt, 1, 1)
        ids = (spatial + temporal) % region_count
        return ids.reshape(1, gt * gh * gw).expand(batch, -1)

    def _unpatchify(self, tokens: torch.Tensor, grid: tuple[int, int, int], out_shape: tuple[int, ...]) -> torch.Tensor:
        b, gt, gh, gw, _ = tokens.shape
        pt, ph, pw = self.patch_size
        c = self.in_channels
        x = self.out(tokens).view(b, gt, gh, gw, c, pt, ph, pw)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        x = x.view(b, c, gt * pt, gh * ph, gw * pw)
        return x[..., : out_shape[2], : out_shape[3], : out_shape[4]]

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        moe_task_ids: torch.Tensor | None = None,
        moe_region_ids: torch.Tensor | None = None,
        **_: torch.Tensor,
    ) -> torch.Tensor:
        self.last_moe_aux_loss = None
        tokens, grid = self.patch(x)
        temb = self.time_embed(time.to(tokens.dtype)).view(tokens.shape[0], 1, 1, 1, -1)
        tokens = tokens + temb
        for block in self.blocks:
            tokens = block(tokens)
        if self.moe is not None:
            flat_tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
            region_ids = moe_region_ids
            if region_ids is None:
                region_ids = self._region_ids(tokens.shape[0], grid, tokens.device)
            moe_out = self.moe(
                flat_tokens,
                modality="video",
                task_ids=moe_task_ids,
                region_ids=region_ids,
                return_router=True,
            )
            flat_tokens = moe_out.hidden_states
            self.last_moe_aux_loss = moe_out.aux_loss * self.moe_aux_loss_weight
            tokens = flat_tokens.view_as(tokens)
        return self._unpatchify(tokens, grid, tuple(x.shape))


@dataclass(slots=True)
class VideoTrainOutput:
    loss: torch.Tensor
    prediction: torch.Tensor
    noised: torch.Tensor
    time: torch.Tensor
    auxiliary_loss: torch.Tensor | None = None
    moe_auxiliary_loss: torch.Tensor | None = None


class DeepThinkingVideoFlowEngine(nn.Module):
    def __init__(
        self,
        model: VideoVelocityNet,
        matcher: AsymmetricFlowMatcher | None = None,
        sampler: FlowSampler | None = None,
        auxiliary_loss_config: AuxiliaryLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        schedule = AsymmetricFlowSchedule(skew=1.45)
        self.matcher = matcher or AsymmetricFlowMatcher(schedule=schedule)
        self.sampler = sampler or FlowSampler(schedule=schedule, steps=32)
        self.auxiliary_loss_config = auxiliary_loss_config

    def forward(self, x_t: torch.Tensor, time: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x_t, time, **kwargs)

    def training_loss(self, clean_video: torch.Tensor, source: torch.Tensor | None = None) -> VideoTrainOutput:
        loss, sample, pred = self.matcher(self.model, clean_video, source=source)
        aux = auxiliary_flow_loss(pred, sample.target_velocity, "video", self.auxiliary_loss_config)
        moe_aux = getattr(self.model, "last_moe_aux_loss", None)
        total_aux = aux if moe_aux is None else aux + moe_aux
        return VideoTrainOutput(
            loss=loss + total_aux,
            prediction=pred,
            noised=sample.x_t,
            time=sample.time,
            auxiliary_loss=total_aux,
            moe_auxiliary_loss=moe_aux,
        )

    @torch.no_grad()
    def generate(self, shape: tuple[int, int, int, int, int], device: torch.device | str, steps: int = 32) -> torch.Tensor:
        source = torch.randn(shape, device=device)
        return self.sampler.euler(self.model, source, steps=steps)
