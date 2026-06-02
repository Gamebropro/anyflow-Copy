from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from .mamba3 import DropPath, Mamba3Block, Mamba3Config, RMSNorm
from .packed import valid_token_mask


RouterScore = Literal["softmax", "sigmoid"]


@dataclass(slots=True)
class AnyFlowMoEConfig:
    dim: int
    num_experts: int = 4
    top_k: int = 2
    state_dim: int = 8
    expansion: int = 2
    expert_depth: int = 1
    num_shared_experts: int = 1
    task_count: int = 8
    region_count: int = 0
    router_score: RouterScore = "sigmoid"
    router_jitter: float = 0.0
    router_z_loss_weight: float = 1.0e-3
    load_balance_weight: float = 1.0e-2
    entropy_weight: float = 0.0
    dropout: float = 0.0
    drop_path: float = 0.0
    prefer_tilelang: bool = True

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        if self.expert_depth <= 0:
            raise ValueError("expert_depth must be positive")
        if self.num_shared_experts < 0:
            raise ValueError("num_shared_experts must be non-negative")
        if self.router_score not in {"softmax", "sigmoid"}:
            raise ValueError("router_score must be 'softmax' or 'sigmoid'")


@dataclass(slots=True)
class MoERouterStats:
    aux_loss: torch.Tensor
    load_balance_loss: torch.Tensor
    router_z_loss: torch.Tensor
    entropy: torch.Tensor
    tokens_per_expert: torch.Tensor
    importance: torch.Tensor
    selected_fraction: torch.Tensor


@dataclass(slots=True)
class MoERouterOutput:
    indices: torch.Tensor
    weights: torch.Tensor
    logits: torch.Tensor
    valid_mask: torch.Tensor
    stats: MoERouterStats


@dataclass(slots=True)
class MambaMoEOutput:
    hidden_states: torch.Tensor
    router: MoERouterOutput

    @property
    def aux_loss(self) -> torch.Tensor:
        return self.router.stats.aux_loss


@dataclass(slots=True)
class MoEDispatchPlan:
    flat_token_indices: torch.Tensor
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    tokens_per_expert: torch.Tensor
    group_offsets: torch.Tensor


MODALITY_TASK_IDS = {
    "video": 0,
    "tts": 1,
    "voice": 1,
    "music": 2,
    "image_edit": 3,
    "video_edit": 4,
    "restoration": 5,
}


def modality_to_task_ids(modality: str, batch_size: int, device: torch.device | str) -> torch.Tensor:
    task_id = MODALITY_TASK_IDS.get(modality, 0)
    return torch.full((batch_size,), task_id, device=device, dtype=torch.long)


def build_expert_dispatch(
    indices: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor | None = None,
    num_experts: int | None = None,
) -> MoEDispatchPlan:
    if indices.ndim != 3:
        raise ValueError("indices must have shape [batch, sequence, top_k]")
    if weights.shape != indices.shape:
        raise ValueError("weights must match indices shape")
    batch, seqlen, top_k = indices.shape
    device = indices.device
    flat_positions = torch.arange(batch * seqlen, device=device, dtype=torch.long).view(batch, seqlen)
    flat_positions = flat_positions.unsqueeze(-1).expand(batch, seqlen, top_k)
    if lengths is None:
        valid = torch.ones((batch, seqlen), device=device, dtype=torch.bool)
    else:
        valid = valid_token_mask(lengths.to(device=device, dtype=torch.long).clamp(0, seqlen), seqlen)
    valid = valid.unsqueeze(-1).expand_as(indices)
    selected_experts = indices[valid].to(torch.long)
    selected_tokens = flat_positions[valid]
    selected_weights = weights[valid]
    expert_count = int(num_experts) if num_experts is not None else int(indices.max().detach().item()) + 1
    if selected_experts.numel() == 0:
        empty_long = torch.empty(0, device=device, dtype=torch.long)
        empty_weight = torch.empty(0, device=device, dtype=weights.dtype)
        counts = torch.zeros(expert_count, device=device, dtype=torch.long)
        offsets = F.pad(torch.cumsum(counts, dim=0), (1, 0))
        return MoEDispatchPlan(empty_long, empty_long, empty_weight, counts, offsets)

    order = torch.argsort(selected_experts, stable=True)
    selected_experts = selected_experts[order]
    selected_tokens = selected_tokens[order]
    selected_weights = selected_weights[order]
    counts = torch.bincount(selected_experts, minlength=expert_count)
    offsets = F.pad(torch.cumsum(counts, dim=0), (1, 0))
    return MoEDispatchPlan(selected_tokens, selected_experts, selected_weights, counts, offsets)


class TopKTaskRouter(nn.Module):
    def __init__(self, config: AnyFlowMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.norm = RMSNorm(config.dim)
        self.gate = nn.Linear(config.dim, config.num_experts, bias=False)
        self.task_bias = nn.Embedding(config.task_count, config.num_experts) if config.task_count > 0 else None
        self.region_bias = nn.Embedding(config.region_count, config.num_experts) if config.region_count > 0 else None
        self.register_buffer("expert_bias", torch.zeros(config.num_experts, dtype=torch.float32), persistent=True)
        nn.init.normal_(self.gate.weight, std=config.dim**-0.5)
        if self.task_bias is not None:
            nn.init.zeros_(self.task_bias.weight)
        if self.region_bias is not None:
            nn.init.zeros_(self.region_bias.weight)

    def _add_task_bias(self, logits: torch.Tensor, task_ids: torch.Tensor | None) -> torch.Tensor:
        if self.task_bias is None or task_ids is None:
            return logits
        task_ids = task_ids.to(device=logits.device, dtype=torch.long).clamp(0, self.config.task_count - 1)
        if task_ids.ndim == 1:
            return logits + self.task_bias(task_ids).unsqueeze(1)
        if task_ids.ndim == 2:
            return logits + self.task_bias(task_ids)
        raise ValueError("task_ids must have shape [batch] or [batch, sequence]")

    def _add_region_bias(self, logits: torch.Tensor, region_ids: torch.Tensor | None) -> torch.Tensor:
        if self.region_bias is None or region_ids is None:
            return logits
        region_ids = region_ids.to(device=logits.device, dtype=torch.long).clamp(0, self.config.region_count - 1)
        if region_ids.ndim != 2:
            raise ValueError("region_ids must have shape [batch, sequence]")
        return logits + self.region_bias(region_ids)

    def _scores(self, logits: torch.Tensor) -> torch.Tensor:
        if self.config.router_score == "softmax":
            return torch.softmax(logits.float(), dim=-1).type_as(logits)
        return torch.sigmoid(logits.float()).type_as(logits)

    def _topk_weights(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self._scores(logits)
        routing_scores = scores + self.expert_bias.to(device=scores.device, dtype=scores.dtype)
        _, indices = torch.topk(routing_scores, k=self.config.top_k, dim=-1, sorted=False)
        selected = torch.gather(scores, dim=-1, index=indices)
        weights = selected / selected.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(selected.dtype).eps)
        return indices, weights

    def _stats(self, logits: torch.Tensor, indices: torch.Tensor, valid: torch.Tensor) -> MoERouterStats:
        probs = torch.softmax(logits.float(), dim=-1)
        valid_f = valid.to(dtype=probs.dtype).unsqueeze(-1)
        denom = valid_f.sum().clamp_min(1.0)
        importance = (probs * valid_f).sum(dim=(0, 1)) / denom
        selected = F.one_hot(indices, num_classes=self.config.num_experts).to(dtype=probs.dtype).sum(dim=2)
        selected = selected * valid_f
        selected_fraction = selected.sum(dim=(0, 1)) / (denom * float(self.config.top_k))
        load_balance = self.config.num_experts * torch.sum(importance * selected_fraction)
        log_z = torch.logsumexp(logits.float(), dim=-1)
        z_loss = ((log_z.square() * valid.to(dtype=log_z.dtype)).sum() / valid.to(dtype=log_z.dtype).sum().clamp_min(1.0))
        entropy = (-(probs * probs.clamp_min(1.0e-9).log()).sum(dim=-1) * valid.to(dtype=probs.dtype)).sum()
        entropy = entropy / valid.to(dtype=probs.dtype).sum().clamp_min(1.0)
        token_counts = selected.sum(dim=(0, 1)).round().to(dtype=torch.long)
        aux = (
            self.config.load_balance_weight * load_balance
            + self.config.router_z_loss_weight * z_loss
            - self.config.entropy_weight * entropy
        )
        return MoERouterStats(
            aux_loss=aux,
            load_balance_loss=load_balance,
            router_z_loss=z_loss,
            entropy=entropy,
            tokens_per_expert=token_counts,
            importance=importance,
            selected_fraction=selected_fraction,
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
        region_ids: torch.Tensor | None = None,
    ) -> MoERouterOutput:
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, sequence, dim]")
        batch, seqlen, _ = x.shape
        h = self.norm(x).float()
        logits = self.gate(h).float()
        logits = self._add_task_bias(logits, task_ids)
        logits = self._add_region_bias(logits, region_ids)
        if self.training and self.config.router_jitter > 0.0:
            jitter = torch.empty_like(logits).uniform_(-self.config.router_jitter, self.config.router_jitter)
            logits = logits + jitter
        indices, weights = self._topk_weights(logits)
        if lengths is None:
            valid = torch.ones((batch, seqlen), device=x.device, dtype=torch.bool)
        else:
            valid = valid_token_mask(lengths.to(device=x.device, dtype=torch.long).clamp(0, seqlen), seqlen)
            weights = weights * valid.unsqueeze(-1).to(dtype=weights.dtype)
        stats = self._stats(logits, indices, valid)
        return MoERouterOutput(indices=indices, weights=weights, logits=logits, valid_mask=valid, stats=stats)


class _MambaExpert(nn.Module):
    def __init__(self, config: AnyFlowMoEConfig) -> None:
        super().__init__()
        mamba_cfg = Mamba3Config(
            dim=config.dim,
            state_dim=config.state_dim,
            expansion=config.expansion,
            dropout=config.dropout,
            drop_path=config.drop_path,
            use_varlen_mixer=True,
            prefer_tilelang=config.prefer_tilelang,
            attention_residual=False,
        )
        self.blocks = nn.ModuleList([Mamba3Block(mamba_cfg) for _ in range(config.expert_depth)])
        self.norm = RMSNorm(config.dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, lengths=lengths)
        return self.norm(x)


class _LocalSharedExpert(nn.Module):
    def __init__(self, dim: int, multiplier: int, dropout: float) -> None:
        super().__init__()
        hidden = dim * max(1, multiplier)
        self.norm = RMSNorm(dim)
        self.up = nn.Linear(dim, hidden * 2, bias=False)
        self.depthwise = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.down = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        h, gate = self.up(self.norm(x)).chunk(2, dim=-1)
        h = self.depthwise(h.transpose(1, 2)).transpose(1, 2)
        y = self.down(self.dropout(h * F.silu(gate)))
        if lengths is not None:
            mask = valid_token_mask(lengths.to(device=x.device, dtype=torch.long).clamp(0, x.shape[1]), x.shape[1])
            y = y * mask.unsqueeze(-1).to(dtype=y.dtype)
        return y


class MambaMoELayer(nn.Module):
    def __init__(self, config: AnyFlowMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.router = TopKTaskRouter(config)
        self.experts = nn.ModuleList([_MambaExpert(config) for _ in range(config.num_experts)])
        self.shared_expert = (
            _LocalSharedExpert(config.dim, config.num_shared_experts, config.dropout)
            if config.num_shared_experts > 0
            else None
        )
        self.out_norm = RMSNorm(config.dim)
        self.out = nn.Linear(config.dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.drop_path = DropPath(config.drop_path)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
        region_ids: torch.Tensor | None = None,
        return_router: bool = False,
    ) -> torch.Tensor | MambaMoEOutput:
        route = self.router(x, lengths=lengths, task_ids=task_ids, region_ids=region_ids)
        expert_outputs = torch.stack([expert(x, lengths=lengths) for expert in self.experts], dim=2)
        gather_index = route.indices.unsqueeze(-1).expand(*route.indices.shape, x.shape[-1])
        selected = torch.gather(expert_outputs, dim=2, index=gather_index)
        mixed = (selected * route.weights.unsqueeze(-1)).sum(dim=2)
        if self.shared_expert is not None:
            mixed = mixed + self.shared_expert(x, lengths=lengths)
        y = self.out(self.out_norm(mixed))
        out = x + self.drop_path(self.dropout(y))
        if lengths is not None:
            mask = valid_token_mask(lengths.to(device=x.device, dtype=torch.long).clamp(0, x.shape[1]), x.shape[1])
            out = out * mask.unsqueeze(-1).to(dtype=out.dtype)
        if return_router:
            return MambaMoEOutput(hidden_states=out, router=route)
        return out


class AnyFlowMoEAdapter(nn.Module):
    def __init__(self, config: AnyFlowMoEConfig) -> None:
        super().__init__()
        self.layer = MambaMoELayer(config)

    def forward(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor | None = None,
        modality: str | None = None,
        task_ids: torch.Tensor | None = None,
        region_ids: torch.Tensor | None = None,
        return_router: bool = False,
    ) -> torch.Tensor | MambaMoEOutput:
        if task_ids is None and modality is not None:
            task_ids = modality_to_task_ids(modality, x.shape[0], x.device)
        return self.layer(
            x,
            lengths=lengths,
            task_ids=task_ids,
            region_ids=region_ids,
            return_router=return_router,
        )


def make_moe_config_for_modality(
    modality: str,
    dim: int,
    *,
    num_experts: int = 4,
    top_k: int = 2,
    prefer_tilelang: bool = True,
) -> AnyFlowMoEConfig:
    if modality == "video":
        return AnyFlowMoEConfig(dim=dim, num_experts=num_experts, top_k=top_k, state_dim=8, region_count=16, prefer_tilelang=prefer_tilelang)
    if modality in {"tts", "voice"}:
        return AnyFlowMoEConfig(dim=dim, num_experts=num_experts, top_k=top_k, state_dim=6, num_shared_experts=2, prefer_tilelang=prefer_tilelang)
    if modality == "music":
        return AnyFlowMoEConfig(dim=dim, num_experts=num_experts, top_k=top_k, state_dim=10, num_shared_experts=2, prefer_tilelang=prefer_tilelang)
    return AnyFlowMoEConfig(dim=dim, num_experts=num_experts, top_k=top_k, prefer_tilelang=prefer_tilelang)
