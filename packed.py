from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(slots=True)
class PackedSequence:
    values: torch.Tensor
    lengths: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    batch_size: int

    def to(self, device: torch.device | str) -> "PackedSequence":
        return PackedSequence(
            values=self.values.to(device),
            lengths=self.lengths.to(device),
            cu_seqlens=self.cu_seqlens.to(device),
            max_seqlen=self.max_seqlen,
            batch_size=self.batch_size,
        )


def lengths_to_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    lengths_i32 = lengths.to(torch.int32)
    zero = torch.zeros(1, dtype=torch.int32, device=lengths.device)
    return torch.cat((zero, torch.cumsum(lengths_i32, dim=0)), dim=0)


def pack_sequence_list(sequences: Sequence[torch.Tensor]) -> PackedSequence:
    if not sequences:
        raise ValueError("sequences cannot be empty")
    feature_shape = sequences[0].shape[1:]
    for seq in sequences:
        if seq.shape[1:] != feature_shape:
            raise ValueError("all packed sequences must share feature dimensions")
    lengths = torch.tensor([seq.shape[0] for seq in sequences], device=sequences[0].device, dtype=torch.long)
    values = torch.cat(tuple(sequences), dim=0)
    return PackedSequence(
        values=values,
        lengths=lengths,
        cu_seqlens=lengths_to_cu_seqlens(lengths),
        max_seqlen=int(lengths.max().item()),
        batch_size=len(sequences),
    )


def pack_padded(x: torch.Tensor, lengths: torch.Tensor, batch_first: bool = True) -> PackedSequence:
    if not batch_first:
        x = x.transpose(0, 1)
    if x.shape[0] != lengths.numel():
        raise ValueError("batch dimension must match lengths")
    pieces = [x[i, : int(lengths[i].item())] for i in range(x.shape[0])]
    return pack_sequence_list(pieces)


def packed_to_padded(packed: PackedSequence, fill_value: float = 0.0) -> torch.Tensor:
    out_shape = (packed.batch_size, packed.max_seqlen, *packed.values.shape[1:])
    out = packed.values.new_full(out_shape, fill_value)
    start = 0
    for i, length in enumerate(packed.lengths.tolist()):
        end = start + int(length)
        out[i, : int(length)] = packed.values[start:end]
        start = end
    return out


def valid_token_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max().item())
    idx = torch.arange(max_len, device=lengths.device)
    return idx.unsqueeze(0) < lengths.unsqueeze(1)


def apply_packed_segments(
    x: torch.Tensor,
    lengths: torch.Tensor,
    fn,
) -> torch.Tensor:
    packed = pack_padded(x, lengths)
    pieces: list[torch.Tensor] = []
    start = 0
    for length in lengths.tolist():
        end = start + int(length)
        pieces.append(fn(packed.values[start:end]))
        start = end
    return packed_to_padded(
        PackedSequence(
            values=torch.cat(pieces, dim=0),
            lengths=packed.lengths,
            cu_seqlens=packed.cu_seqlens,
            max_seqlen=packed.max_seqlen,
            batch_size=packed.batch_size,
        )
    )
