from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import torch

from .compiler import AnyFlowRuntimeConfig, compile_anyflow, initialize_anyflow_runtime, sm75_inference_config
from .mamba3 import Mamba3Block, Mamba3Config


@dataclass(slots=True)
class CompileValidationConfig:
    device: str | None = None
    dim: int = 8
    state_dim: int = 2
    seqlen_a: int = 3
    seqlen_b: int = 5
    run_compile: bool = False
    backend: str | None = None
    mode: str | None = None
    fullgraph: bool = False
    dynamic: bool = True
    require_sm75: bool = False


@dataclass(slots=True)
class CompileValidationCheck:
    name: str
    ok: bool
    required: bool
    detail: str


@dataclass(slots=True)
class CompileValidationReport:
    torch_version: str
    device: str
    run_compile: bool
    backend: str
    mode: str | None
    dynamic: bool
    checks: list[CompileValidationCheck]

    @property
    def passed_required(self) -> bool:
        return all(check.ok or not check.required for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "torch_version": self.torch_version,
            "device": self.device,
            "run_compile": self.run_compile,
            "backend": self.backend,
            "mode": self.mode,
            "dynamic": self.dynamic,
            "passed_required": self.passed_required,
            "checks": [asdict(check) for check in self.checks],
        }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _make_probe(dim: int, state_dim: int) -> Mamba3Block:
    cfg = Mamba3Config(
        dim=dim,
        state_dim=state_dim,
        expansion=1,
        use_varlen_mixer=False,
        prefer_tilelang=False,
        attention_residual=False,
    )
    return Mamba3Block(cfg).eval()


@torch.no_grad()
def run_compile_validation(config: CompileValidationConfig | None = None) -> CompileValidationReport:
    cfg = config or CompileValidationConfig()
    runtime = sm75_inference_config(require_sm75=cfg.require_sm75)
    runtime.compile_dynamic = cfg.dynamic
    if cfg.backend is not None:
        runtime.compile_backend = cfg.backend
    if cfg.mode is not None:
        runtime.compile_mode = cfg.mode
    elif runtime.compile_backend != "inductor":
        runtime.compile_mode = None
    runtime.compile_fullgraph = cfg.fullgraph
    applied = initialize_anyflow_runtime(runtime)
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checks = [
        CompileValidationCheck(
            "torch_compile_available",
            hasattr(torch, "compile"),
            True,
            f"torch_compile={hasattr(torch, 'compile')}",
        ),
        CompileValidationCheck(
            "runtime_dynamic_shapes_requested",
            bool(runtime.dynamic_shapes and runtime.compile_dynamic),
            True,
            f"dynamic_shapes={runtime.dynamic_shapes}, compile_dynamic={runtime.compile_dynamic}",
        ),
        CompileValidationCheck(
            "runtime_combo_kernels_requested",
            bool(runtime.combo_kernels),
            True,
            f"combo_kernels={runtime.combo_kernels}, applied={applied.get('inductor.combo_kernels')}",
        ),
        CompileValidationCheck(
            "runtime_inductor_deterministic_requested",
            bool(runtime.inductor_deterministic),
            True,
            f"inductor_deterministic={runtime.inductor_deterministic}, applied={applied.get('inductor.deterministic')}",
        ),
    ]

    torch.manual_seed(runtime.seed)
    probe = _make_probe(cfg.dim, cfg.state_dim).to(device)
    x_a = torch.randn(1, cfg.seqlen_a, cfg.dim, device=device)
    x_b = torch.randn(1, cfg.seqlen_b, cfg.dim, device=device)
    eager_a = probe(x_a)
    eager_b = probe(x_b)
    checks.append(
        CompileValidationCheck(
            "eager_dynamic_shape_probe",
            tuple(eager_a.shape) == tuple(x_a.shape) and tuple(eager_b.shape) == tuple(x_b.shape),
            True,
            f"shape_a={tuple(eager_a.shape)}, shape_b={tuple(eager_b.shape)}",
        )
    )

    if cfg.run_compile:
        try:
            compiled = compile_anyflow(
                probe,
                runtime,
                backend=runtime.compile_backend,
                mode=runtime.compile_mode,
                fullgraph=runtime.compile_fullgraph,
                dynamic=runtime.compile_dynamic,
            )
            compiled_a = compiled(x_a)
            compiled_b = compiled(x_b)
            _sync(device)
            err_a = float((compiled_a.float() - eager_a.float()).abs().max().cpu())
            err_b = float((compiled_b.float() - eager_b.float()).abs().max().cpu())
            checks.append(
                CompileValidationCheck(
                    "compiled_dynamic_shape_parity",
                    err_a <= 1e-5 and err_b <= 1e-5,
                    True,
                    f"max_abs_error_a={err_a}, max_abs_error_b={err_b}",
                )
            )
        except Exception as exc:
            checks.append(
                CompileValidationCheck(
                    "compiled_dynamic_shape_parity",
                    False,
                    True,
                    f"compile failed: {exc!r}",
                )
            )
    else:
        checks.append(
            CompileValidationCheck(
                "compiled_dynamic_shape_parity",
                True,
                False,
                "skipped; pass --run-compile to execute torch.compile",
            )
        )

    return CompileValidationReport(
        torch_version=torch.__version__,
        device=str(device),
        run_compile=cfg.run_compile,
        backend=runtime.compile_backend,
        mode=runtime.compile_mode,
        dynamic=runtime.compile_dynamic,
        checks=checks,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate ANYFLOW dynamic torch.compile configuration.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--state-dim", type=int, default=2)
    parser.add_argument("--seqlen-a", type=int, default=3)
    parser.add_argument("--seqlen-b", type=int, default=5)
    parser.add_argument("--run-compile", action="store_true")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--static", action="store_true", help="Set dynamic=False for a negative/control probe.")
    parser.add_argument("--require-sm75", action="store_true")
    args = parser.parse_args()
    report = run_compile_validation(
        CompileValidationConfig(
            device=args.device,
            dim=args.dim,
            state_dim=args.state_dim,
            seqlen_a=args.seqlen_a,
            seqlen_b=args.seqlen_b,
            run_compile=args.run_compile,
            backend=args.backend,
            mode=args.mode,
            fullgraph=args.fullgraph,
            dynamic=not args.static,
            require_sm75=args.require_sm75,
        )
    )
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.passed_required else 1


if __name__ == "__main__":
    raise SystemExit(main())
