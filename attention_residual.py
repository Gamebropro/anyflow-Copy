from __future__ import annotations

import math

import torch
from torch import nn

from .mamba3 import RMSNorm


class GroupedDepthResidualRouter(nn.Module):
    """Attention-Residuals depth router without token-token attention.

    The router follows the Attention Residuals idea from the local
    specification: every layer receives a learned pseudo-query and mixes prior
    layer states. The grouping mirrors GQA, but it is applied only across the
    finite depth-state axis, never across sequence tokens.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int = 4,
        groups: int = 1,
        block_size: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        if heads % groups != 0:
            raise ValueError("heads must be divisible by groups")
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.groups = groups
        self.block_size = max(1, block_size)
        self.head_dim = dim // heads
        self.group_repeats = heads // groups
        self.query = nn.Parameter(torch.empty(depth, heads, self.head_dim))
        self.key_norm = RMSNorm(dim, eps=eps)
        self.out = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.query, std=self.head_dim**-0.5)

    def _shape_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(*x.shape[:-1], self.heads, self.head_dim)

    def _shape_grouped_heads(self, x: torch.Tensor) -> torch.Tensor:
        heads = self._shape_heads(x)
        return heads.view(*heads.shape[:-2], self.groups, self.group_repeats, self.head_dim)

    def _depth_mix(self, layer: int, states: list[torch.Tensor], partial: torch.Tensor | None) -> torch.Tensor:
        values = states if partial is None else [*states, partial]
        if not values:
            raise ValueError("GroupedDepthResidualRouter requires at least one state")
        stacked = torch.stack(values, dim=2)
        keys = self._shape_grouped_heads(self.key_norm(stacked)).mean(dim=-2)
        vals = self._shape_grouped_heads(stacked).mean(dim=-2)
        q = self.query[layer].view(self.groups, self.group_repeats, self.head_dim)
        q = q.view(1, 1, 1, self.groups, self.group_repeats, self.head_dim)
        logits = (keys.unsqueeze(-2) * q).sum(dim=-1) / math.sqrt(self.head_dim)
        weights = logits.softmax(dim=2)
        mixed = (weights.unsqueeze(-1) * vals.unsqueeze(-2)).sum(dim=2)
        mixed = mixed.reshape(stacked.shape[0], stacked.shape[1], self.dim)
        return self.out(mixed)

    def forward_layer_input(self, layer: int, states: list[torch.Tensor], partial: torch.Tensor | None) -> torch.Tensor:
        return self._depth_mix(layer, states, partial)

    def should_close_block(self, layer: int) -> bool:
        return (layer + 1) % self.block_size == 0


class AttentionResidualMambaStack(nn.Module):
    """Block Attention Residuals wrapper for Mamba blocks.

    This implements the spec's block residual algorithm while keeping each
    layer transform as a Mamba block. The only softmax is over previous depth
    states, so compute remains linear in sequence length.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        dim: int,
        heads: int = 4,
        groups: int = 1,
        block_size: int = 4,
        output_norm: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.router = GroupedDepthResidualRouter(
            dim=dim,
            depth=len(blocks),
            heads=heads,
            groups=groups,
            block_size=block_size,
        )
        self.output_norm = output_norm if output_norm is not None else RMSNorm(dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        closed_blocks: list[torch.Tensor] = [x]
        partial: torch.Tensor | None = torch.zeros_like(x)
        out = x
        for layer, block in enumerate(self.blocks):
            routed = self.router.forward_layer_input(layer, closed_blocks, partial)
            out = block(routed, lengths=lengths)
            partial = out if partial is None else partial + out
            if self.router.should_close_block(layer):
                closed_blocks.append(partial)
                partial = torch.zeros_like(out)
        return self.output_norm(out)


def assert_transformer_free(module: nn.Module) -> None:
    forbidden = (
        nn.Transformer,
        nn.TransformerEncoder,
        nn.TransformerDecoder,
        nn.MultiheadAttention,
    )
    for name, child in module.named_modules():
        if isinstance(child, forbidden):
            raise AssertionError(f"Transformer-style module is not allowed in ANYFLOW: {name}={type(child).__name__}")
