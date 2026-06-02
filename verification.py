from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from .attention_residual import assert_transformer_free
from .compiler import AnyFlowRuntimeConfig, initialize_anyflow_runtime, sm75_inference_config
from .engine import AnyFlowBatch, AnyFlowEngine, build_anyflow_small
from .kernel_validation import run_kernel_validation
from .quantization import dequantize_state_dict, quantize_mxfp4, quantize_state_dict, quantized_state_dict_stats


@dataclass(slots=True)
class VerificationCheck:
    name: str
    ok: bool
    required: bool
    detail: str


@dataclass(slots=True)
class AnyFlowVerificationReport:
    torch_version: str
    tilelang_version: str | None
    cuda_name: str | None
    cuda_sm: int | None
    checks: list[VerificationCheck]

    @property
    def passed_required(self) -> bool:
        return all(check.ok or not check.required for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "torch_version": self.torch_version,
            "tilelang_version": self.tilelang_version,
            "cuda_name": self.cuda_name,
            "cuda_sm": self.cuda_sm,
            "passed_required": self.passed_required,
            "checks": [asdict(check) for check in self.checks],
        }

    def raise_for_required_failures(self) -> None:
        failed = [check for check in self.checks if check.required and not check.ok]
        if failed:
            details = "; ".join(f"{check.name}: {check.detail}" for check in failed)
            raise RuntimeError(f"ANYFLOW verification failed required checks: {details}")


def _tilelang_version() -> str | None:
    try:
        import tilelang
    except Exception:
        return None
    return str(getattr(tilelang, "__version__", "unknown"))


def _cuda_info() -> tuple[str | None, int | None]:
    if not torch.cuda.is_available():
        return None, None
    major, minor = torch.cuda.get_device_capability()
    return torch.cuda.get_device_name(), major * 10 + minor


def _check_version(actual: str | None, expected_prefix: str, required: bool, name: str) -> VerificationCheck:
    ok = actual is not None and actual.startswith(expected_prefix)
    detail = f"actual={actual!r}, expected_prefix={expected_prefix!r}"
    return VerificationCheck(name=name, ok=ok, required=required, detail=detail)


def _check_runtime_config(runtime: AnyFlowRuntimeConfig) -> list[VerificationCheck]:
    applied = initialize_anyflow_runtime(runtime)
    checks = [
        VerificationCheck(
            name="deterministic_algorithms",
            ok=bool(torch.are_deterministic_algorithms_enabled()),
            required=True,
            detail=f"applied={applied.get('deterministic_algorithms')}",
        ),
        VerificationCheck(
            name="runtime_target_sm75",
            ok=runtime.target_sm == 75,
            required=True,
            detail=f"target_sm={runtime.target_sm}",
        ),
        VerificationCheck(
            name="runtime_combo_kernels_requested",
            ok=runtime.combo_kernels,
            required=True,
            detail=f"combo_kernels={runtime.combo_kernels}",
        ),
        VerificationCheck(
            name="runtime_dynamic_compile_requested",
            ok=runtime.compile_dynamic and runtime.dynamic_shapes,
            required=True,
            detail=f"compile_dynamic={runtime.compile_dynamic}, dynamic_shapes={runtime.dynamic_shapes}",
        ),
    ]
    combo_applied = applied.get("inductor.combo_kernels")
    if combo_applied is not None:
        checks.append(
            VerificationCheck(
                name="inductor_combo_kernels_applied_when_available",
                ok=bool(combo_applied),
                required=False,
                detail=f"applied={combo_applied}",
            )
        )
    det_applied = applied.get("inductor.deterministic")
    if det_applied is not None:
        checks.append(
            VerificationCheck(
                name="inductor_deterministic_applied_when_available",
                ok=bool(det_applied),
                required=False,
                detail=f"applied={det_applied}",
            )
        )
    return checks


@torch.no_grad()
def _check_generation(engine: AnyFlowEngine, device: torch.device) -> list[VerificationCheck]:
    video = engine.generate("video", (1, 2, 1, 4, 4), device=device, steps=1)
    tts = engine.generate("tts", (1, 1, 16), device=device, steps=1)
    music = engine.generate("music", (1, 4, 1, 16), device=device, steps=1)
    return [
        VerificationCheck("video_generate_shape", tuple(video.shape) == (1, 2, 1, 4, 4), True, f"shape={tuple(video.shape)}"),
        VerificationCheck("tts_generate_shape", tuple(tts.shape) == (1, 1, 16), True, f"shape={tuple(tts.shape)}"),
        VerificationCheck("music_generate_shape", tuple(music.shape) == (1, 4, 1, 16), True, f"shape={tuple(music.shape)}"),
    ]


def _check_training_losses(engine: AnyFlowEngine, device: torch.device, vocab_size: int) -> list[VerificationCheck]:
    video = torch.randn(1, 2, 1, 4, 4, device=device)
    audio = torch.randn(1, 1, 16, device=device)
    music = torch.randn(1, 4, 1, 16, device=device)
    tokens = torch.randint(0, vocab_size, (1, 5), device=device)
    lengths = torch.tensor([5], device=device)
    outs = [
        ("video_loss_finite", engine.training_loss(AnyFlowBatch("video", video)).loss),
        ("tts_loss_finite", engine.training_loss(AnyFlowBatch("tts", audio, text_tokens=tokens, text_lengths=lengths)).loss),
        ("music_loss_finite", engine.training_loss(AnyFlowBatch("music", music, text_tokens=tokens, text_lengths=lengths)).loss),
    ]
    return [
        VerificationCheck(name, bool(torch.isfinite(loss).item()), True, f"loss={float(loss.detach().cpu())}")
        for name, loss in outs
    ]


def run_anyflow_verification(
    *,
    strict_versions: bool = False,
    require_sm75: bool = False,
    require_tilelang: bool = False,
    device: str | torch.device | None = None,
    dim: int = 16,
    vocab_size: int = 32,
) -> AnyFlowVerificationReport:
    runtime = sm75_inference_config(require_sm75=require_sm75)
    checks = _check_runtime_config(runtime)
    tile_ver = _tilelang_version()
    cuda_name, cuda_sm = _cuda_info()
    checks.append(_check_version(torch.__version__, "2.10.0", strict_versions, "pytorch_2_10_0"))
    checks.append(_check_version(tile_ver, "0.1.10", require_tilelang, "tilelang_0_1_10"))
    checks.append(
        VerificationCheck(
            name="cuda_sm75",
            ok=cuda_sm == 75,
            required=require_sm75,
            detail=f"cuda_sm={cuda_sm}, cuda_name={cuda_name}",
        )
    )

    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    engine = build_anyflow_small(video_channels=2, audio_channels=1, vocab_size=vocab_size, dim=dim, runtime=runtime).to(target)
    try:
        assert_transformer_free(engine)
        checks.append(VerificationCheck("transformer_free_engine", True, True, "no forbidden module instances"))
    except AssertionError as exc:
        checks.append(VerificationCheck("transformer_free_engine", False, True, str(exc)))

    checks.extend(_check_training_losses(engine, target, vocab_size))
    checks.extend(_check_generation(engine, target))
    x = torch.randn(2, 17, device=target)
    q = quantize_mxfp4(x)
    checks.append(
        VerificationCheck(
            name="mxfp4_roundtrip_shape",
            ok=q.dequantize().shape == x.shape,
            required=True,
            detail=f"shape={tuple(q.dequantize().shape)}",
        )
    )
    q_state = quantize_state_dict(engine.state_dict(), format="mxfp4", tile_size=32)
    restored = dequantize_state_dict(q_state, device=target)
    shapes_ok = all(tuple(restored[name].shape) == tuple(tensor.shape) for name, tensor in engine.state_dict().items())
    stats = quantized_state_dict_stats(q_state)
    checks.append(
        VerificationCheck(
            name="mxfp4_state_dict_roundtrip_shapes",
            ok=shapes_ok and stats["quantized_tensors"] > 0,
            required=True,
            detail=f"quantized_tensors={stats['quantized_tensors']}, compression_ratio={stats['compression_ratio']:.3f}",
        )
    )
    kernel_report = run_kernel_validation(device=target)
    for result in kernel_report.results:
        checks.append(
            VerificationCheck(
                name=f"kernel_parity_{result.name}",
                ok=result.ok,
                required=True,
                detail=f"max_abs_error={result.max_abs_error}, attempted_tilelang={result.attempted_tilelang}, {result.detail}",
            )
        )
    return AnyFlowVerificationReport(
        torch_version=torch.__version__,
        tilelang_version=tile_ver,
        cuda_name=cuda_name,
        cuda_sm=cuda_sm,
        checks=checks,
    )


def assert_anyflow_ready(**kwargs: Any) -> AnyFlowVerificationReport:
    report = run_anyflow_verification(**kwargs)
    report.raise_for_required_failures()
    return report
