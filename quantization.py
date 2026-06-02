from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .compiler import precision_context


@dataclass(slots=True)
class TileQuantizedTensor:
    values: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    tile_size: int
    format: str
    layout: str = "last_dim_tiles"

    def dequantize(self) -> torch.Tensor:
        if self.format.startswith("fp8"):
            return dequantize_fp8_tile(self)
        if self.format.startswith("mxfp4"):
            return dequantize_mxfp4(self)
        raise ValueError(f"unsupported quantized format: {self.format}")


_QTENSOR_MARKER = "__anyflow_tile_quantized_tensor__"
_QSTATE_MARKER = "__anyflow_quantized_state_dict__"


def _fp8_dtype(format: str = "e4m3") -> torch.dtype:
    candidates = {
        "e4m3": ("float8_e4m3fn", "float8_e4m3fnuz"),
        "e5m2": ("float8_e5m2", "float8_e5m2fnuz"),
    }
    for name in candidates.get(format, candidates["e4m3"]):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            return dtype
    raise RuntimeError(f"torch build does not expose native FP8 dtype for {format}")


def _pad_last_dim(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    pad = (multiple - x.shape[-1] % multiple) % multiple
    if pad == 0:
        return x.contiguous(), 0
    return torch.nn.functional.pad(x, (0, pad)).contiguous(), pad


def _restore_shape(x: torch.Tensor, original_shape: tuple[int, ...]) -> torch.Tensor:
    return x.reshape(*x.shape[:-1], -1)[..., : original_shape[-1]].reshape(original_shape)


def quantize_fp8_tile(
    x: torch.Tensor,
    tile_size: int = 128,
    format: str = "e4m3",
    eps: float = 1e-6,
) -> TileQuantizedTensor:
    if x.numel() == 0:
        raise ValueError("cannot quantize an empty tensor")
    dtype = _fp8_dtype(format)
    fp8_max = 448.0 if format == "e4m3" else 57344.0
    x_float = x.detach() if not x.is_floating_point() else x
    padded, _ = _pad_last_dim(x_float, tile_size)
    tiled = padded.reshape(-1, padded.shape[-1] // tile_size, tile_size)
    amax = tiled.abs().float().amax(dim=-1).clamp_min(eps)
    scales = (amax / fp8_max).to(torch.float32)
    q = (tiled.float() / scales.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(dtype)
    q = q.reshape(padded.shape)
    return TileQuantizedTensor(
        values=q,
        scales=scales.reshape(*padded.shape[:-1], padded.shape[-1] // tile_size),
        original_shape=tuple(x.shape),
        padded_shape=tuple(padded.shape),
        tile_size=tile_size,
        format=f"fp8_{format}",
    )


def dequantize_fp8_tile(q: TileQuantizedTensor) -> torch.Tensor:
    tiled = q.values.float().reshape(-1, q.padded_shape[-1] // q.tile_size, q.tile_size)
    scales = q.scales.float().reshape(-1, q.padded_shape[-1] // q.tile_size)
    x = tiled * scales.unsqueeze(-1)
    x = x.reshape(q.padded_shape)
    return _restore_shape(x, q.original_shape)


def _mxfp4_codebook(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    return torch.tensor(values, device=device, dtype=dtype)


def quantize_mxfp4(x: torch.Tensor, tile_size: int = 32, eps: float = 1e-6) -> TileQuantizedTensor:
    if tile_size % 2 != 0:
        raise ValueError("MXFP4 tile_size must be even so nibbles can be packed")
    padded, _ = _pad_last_dim(x, tile_size)
    tiled = padded.float().reshape(-1, padded.shape[-1] // tile_size, tile_size)
    amax = tiled.abs().amax(dim=-1).clamp_min(eps)
    scales = amax / 6.0
    normalized = tiled / scales.unsqueeze(-1)
    codebook = _mxfp4_codebook(x.device, normalized.dtype)
    distances = (normalized.unsqueeze(-1) - codebook).abs()
    codes = distances.argmin(dim=-1).to(torch.uint8)
    low = codes[..., 0::2]
    high = codes[..., 1::2] << 4
    packed = (low | high).reshape(*padded.shape[:-1], padded.shape[-1] // 2).contiguous()
    return TileQuantizedTensor(
        values=packed,
        scales=scales.reshape(*padded.shape[:-1], padded.shape[-1] // tile_size).to(torch.float32),
        original_shape=tuple(x.shape),
        padded_shape=tuple(padded.shape),
        tile_size=tile_size,
        format="mxfp4_e2m1",
    )


def dequantize_mxfp4(q: TileQuantizedTensor) -> torch.Tensor:
    packed = q.values.reshape(*q.padded_shape[:-1], q.padded_shape[-1] // 2)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(*q.padded_shape)
    codebook = _mxfp4_codebook(q.values.device, q.scales.dtype)
    values = codebook[codes.long()]
    tiled = values.reshape(-1, q.padded_shape[-1] // q.tile_size, q.tile_size)
    scales = q.scales.reshape(-1, q.padded_shape[-1] // q.tile_size).to(values.dtype)
    x = (tiled * scales.unsqueeze(-1)).reshape(q.padded_shape)
    return _restore_shape(x, q.original_shape)


def serialize_quantized_tensor(q: TileQuantizedTensor) -> dict[str, Any]:
    return {
        _QTENSOR_MARKER: True,
        "values": q.values.detach().cpu(),
        "scales": q.scales.detach().cpu(),
        "original_shape": tuple(q.original_shape),
        "padded_shape": tuple(q.padded_shape),
        "tile_size": int(q.tile_size),
        "format": q.format,
        "layout": q.layout,
    }


def deserialize_quantized_tensor(payload: Mapping[str, Any], device: torch.device | str | None = None) -> TileQuantizedTensor:
    if not payload.get(_QTENSOR_MARKER):
        raise ValueError("payload is not an ANYFLOW quantized tensor")
    values = payload["values"]
    scales = payload["scales"]
    if device is not None:
        values = values.to(device)
        scales = scales.to(device)
    return TileQuantizedTensor(
        values=values,
        scales=scales,
        original_shape=tuple(payload["original_shape"]),
        padded_shape=tuple(payload["padded_shape"]),
        tile_size=int(payload["tile_size"]),
        format=str(payload["format"]),
        layout=str(payload.get("layout", "last_dim_tiles")),
    )


def _quantize_tensor_by_format(x: torch.Tensor, format: str, tile_size: int) -> TileQuantizedTensor:
    normalized = format.lower()
    if normalized in {"mxfp4", "mxfp4_e2m1", "mx_fp4"}:
        return quantize_mxfp4(x, tile_size=tile_size)
    if normalized in {"fp8", "fp8_e4m3", "e4m3"}:
        return quantize_fp8_tile(x, tile_size=tile_size, format="e4m3")
    if normalized in {"fp8_e5m2", "e5m2"}:
        return quantize_fp8_tile(x, tile_size=tile_size, format="e5m2")
    raise ValueError(f"unsupported quantized state_dict format: {format}")


def tensor_nbytes(x: torch.Tensor) -> int:
    return x.numel() * x.element_size()


def quantize_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    format: str = "mxfp4",
    tile_size: int | None = None,
    min_numel: int = 1,
) -> dict[str, Any]:
    q_tile = tile_size if tile_size is not None else (32 if format.lower().startswith("mx") else 128)
    tensors: dict[str, Any] = {}
    original_bytes = 0
    stored_bytes = 0
    quantized_tensors = 0
    for name, tensor in state_dict.items():
        cpu_tensor = tensor.detach().cpu()
        original_bytes += tensor_nbytes(cpu_tensor)
        if cpu_tensor.is_floating_point() and cpu_tensor.numel() >= min_numel:
            q = _quantize_tensor_by_format(cpu_tensor, format=format, tile_size=q_tile)
            tensors[name] = serialize_quantized_tensor(q)
            stored_bytes += estimate_quantized_bytes(q)
            quantized_tensors += 1
        else:
            tensors[name] = cpu_tensor.clone()
            stored_bytes += tensor_nbytes(cpu_tensor)
    return {
        _QSTATE_MARKER: True,
        "format": format,
        "tile_size": q_tile,
        "tensors": tensors,
        "stats": {
            "original_bytes": original_bytes,
            "stored_bytes": stored_bytes,
            "quantized_tensors": quantized_tensors,
            "total_tensors": len(tensors),
            "compression_ratio": math.inf if stored_bytes == 0 else original_bytes / stored_bytes,
        },
    }


def dequantize_state_dict(
    payload: Mapping[str, Any],
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor]:
    if not payload.get(_QSTATE_MARKER):
        raise ValueError("payload is not an ANYFLOW quantized state_dict")
    out: dict[str, torch.Tensor] = {}
    for name, entry in payload["tensors"].items():
        if isinstance(entry, Mapping) and entry.get(_QTENSOR_MARKER):
            tensor = deserialize_quantized_tensor(entry, device=device).dequantize()
        elif torch.is_tensor(entry):
            tensor = entry.to(device) if device is not None else entry.clone()
        else:
            raise TypeError(f"unsupported state_dict entry for {name}: {type(entry).__name__}")
        if dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(dtype=dtype)
        out[name] = tensor
    return out


def quantized_state_dict_stats(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not payload.get(_QSTATE_MARKER):
        raise ValueError("payload is not an ANYFLOW quantized state_dict")
    return dict(payload["stats"])


class AnyFlowPrecisionContext(contextlib.AbstractContextManager["AnyFlowPrecisionContext"]):
    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        quantize_activations: bool = False,
        tile_size: int = 128,
        activation_format: str = "fp8_e4m3",
        fallback_format: str | None = "mxfp4",
    ) -> None:
        self.enabled = enabled
        self.dtype = dtype
        self.quantize_activations = quantize_activations
        self.tile_size = tile_size
        self.activation_format = activation_format
        self.fallback_format = fallback_format
        self._ctx: contextlib.AbstractContextManager[None] | None = None

    def __enter__(self) -> "AnyFlowPrecisionContext":
        self._ctx = precision_context(self.enabled, dtype=self.dtype)
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool | None:
        assert self._ctx is not None
        return self._ctx.__exit__(exc_type, exc, tb)

    def activation(self, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize_activations or not x.is_floating_point():
            return x
        try:
            return _quantize_tensor_by_format(x, self.activation_format, self.tile_size).dequantize().to(x.dtype)
        except RuntimeError:
            if self.fallback_format is None:
                raise
            tile_size = 32 if self.fallback_format.lower().startswith("mx") else self.tile_size
            return _quantize_tensor_by_format(x, self.fallback_format, tile_size).dequantize().to(x.dtype)


def estimate_quantized_bytes(q: TileQuantizedTensor) -> int:
    return q.values.numel() * q.values.element_size() + q.scales.numel() * q.scales.element_size()


def compression_ratio(x: torch.Tensor, q: TileQuantizedTensor) -> float:
    original = x.numel() * x.element_size()
    quantized = estimate_quantized_bytes(q)
    return math.inf if quantized == 0 else original / quantized
