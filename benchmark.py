from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .attention_residual import assert_transformer_free
from .compiler import initialize_anyflow_runtime, sm75_inference_config
from .engine import Modality, build_anyflow_small
from .verification import _tilelang_version


@dataclass(slots=True)
class Sm75BenchmarkConfig:
    device: str | None = None
    dim: int = 16
    vocab_size: int = 32
    video_channels: int = 2
    audio_channels: int = 1
    steps: int = 1
    warmup: int = 1
    repeats: int = 3
    compile_model: bool = False
    require_sm75: bool = False


@dataclass(slots=True)
class InferenceMetric:
    modality: str
    shape: tuple[int, ...]
    mean_seconds: float
    min_seconds: float
    max_seconds: float
    cuda_peak_allocated: int | None
    cuda_peak_reserved: int | None


@dataclass(slots=True)
class Sm75BenchmarkReport:
    torch_version: str
    tilelang_version: str | None
    device: str
    cuda_name: str | None
    cuda_sm: int | None
    compile_model: bool
    metrics: list[InferenceMetric]

    def as_dict(self) -> dict[str, Any]:
        return {
            "torch_version": self.torch_version,
            "tilelang_version": self.tilelang_version,
            "device": self.device,
            "cuda_name": self.cuda_name,
            "cuda_sm": self.cuda_sm,
            "compile_model": self.compile_model,
            "metrics": [asdict(metric) for metric in self.metrics],
        }


def _cuda_sm() -> tuple[str | None, int | None]:
    if not torch.cuda.is_available():
        return None, None
    major, minor = torch.cuda.get_device_capability()
    return torch.cuda.get_device_name(), major * 10 + minor


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _shape_for(modality: Modality, cfg: Sm75BenchmarkConfig) -> tuple[int, ...]:
    if modality == "video":
        return (1, cfg.video_channels, 1, 4, 4)
    if modality == "tts":
        return (1, cfg.audio_channels, 16)
    if modality == "music":
        return (1, 4, cfg.audio_channels, 16)
    raise ValueError(f"unknown modality: {modality}")


@torch.no_grad()
def benchmark_sm75_inference(config: Sm75BenchmarkConfig | None = None) -> Sm75BenchmarkReport:
    cfg = config or Sm75BenchmarkConfig()
    runtime = sm75_inference_config(require_sm75=cfg.require_sm75)
    initialize_anyflow_runtime(runtime)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    engine = build_anyflow_small(
        video_channels=cfg.video_channels,
        audio_channels=cfg.audio_channels,
        vocab_size=cfg.vocab_size,
        dim=cfg.dim,
        runtime=runtime,
    ).to(device)
    assert_transformer_free(engine)
    engine.eval()
    if cfg.compile_model:
        engine.compile()

    metrics: list[InferenceMetric] = []
    for modality in ("video", "tts", "music"):
        shape = _shape_for(modality, cfg)
        for _ in range(cfg.warmup):
            engine.generate(modality, shape, device=device, steps=cfg.steps)
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times: list[float] = []
        for _ in range(cfg.repeats):
            start = time.perf_counter()
            out = engine.generate(modality, shape, device=device, steps=cfg.steps)
            _sync(device)
            elapsed = time.perf_counter() - start
            if tuple(out.shape) != shape:
                raise RuntimeError(f"{modality} generated {tuple(out.shape)}, expected {shape}")
            times.append(elapsed)
        if device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
        else:
            peak_allocated = None
            peak_reserved = None
        metrics.append(
            InferenceMetric(
                modality=modality,
                shape=shape,
                mean_seconds=sum(times) / len(times),
                min_seconds=min(times),
                max_seconds=max(times),
                cuda_peak_allocated=peak_allocated,
                cuda_peak_reserved=peak_reserved,
            )
        )

    cuda_name, cuda_sm = _cuda_sm()
    return Sm75BenchmarkReport(
        torch_version=torch.__version__,
        tilelang_version=_tilelang_version(),
        device=str(device),
        cuda_name=cuda_name,
        cuda_sm=cuda_sm,
        compile_model=cfg.compile_model,
        metrics=metrics,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a tiny ANYFLOW sm_75 inference benchmark.")
    parser.add_argument("--device", default=None, help="Device, for example cpu or cuda.")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1, help="Euler sampling steps per generated sample.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--require-sm75", action="store_true")
    args = parser.parse_args()
    report = benchmark_sm75_inference(
        Sm75BenchmarkConfig(
            device=args.device,
            dim=args.dim,
            vocab_size=args.vocab_size,
            steps=args.steps,
            warmup=args.warmup,
            repeats=args.repeats,
            compile_model=args.compile_model,
            require_sm75=args.require_sm75,
        )
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
