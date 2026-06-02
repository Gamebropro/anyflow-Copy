from __future__ import annotations

import contextlib
import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass(slots=True)
class AnyFlowRuntimeConfig:
    seed: int = 1337
    deterministic: bool = True
    combo_kernels: bool = True
    inductor_deterministic: bool = True
    dynamic_shapes: bool = True
    compile_backend: str = "inductor"
    compile_mode: str = "max-autotune-no-cudagraphs"
    compile_fullgraph: bool = False
    compile_dynamic: bool = True
    matmul_precision: str = "high"
    allow_tf32: bool = False
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = True
    cuda_allocator_conf: str = "expandable_segments:True"
    autocast_dtype: torch.dtype = torch.bfloat16
    target_sm: int | None = 75
    require_target_sm: bool = False
    compile_options: dict[str, Any] = field(default_factory=dict)


def _set_config_attr(obj: Any, name: str, value: Any) -> bool:
    try:
        if hasattr(obj, name):
            setattr(obj, name, value)
            return True
    except Exception:
        return False
    return False


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_anyflow_runtime(config: AnyFlowRuntimeConfig | None = None) -> dict[str, Any]:
    cfg = config or AnyFlowRuntimeConfig()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if cfg.cuda_allocator_conf and os.name != "nt":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", cfg.cuda_allocator_conf)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _seed_everything(cfg.seed)
    torch.use_deterministic_algorithms(cfg.deterministic)
    torch.set_float32_matmul_precision(cfg.matmul_precision)

    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf32
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = cfg.cudnn_deterministic

    applied: dict[str, Any] = {
        "torch_version": torch.__version__,
        "deterministic_algorithms": cfg.deterministic,
        "matmul_precision": cfg.matmul_precision,
    }

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        current_sm = major * 10 + minor
        applied["cuda.sm"] = current_sm
        applied["cuda.name"] = torch.cuda.get_device_name()
        if cfg.target_sm is not None:
            applied["target_sm"] = cfg.target_sm
            if cfg.require_target_sm and current_sm != cfg.target_sm:
                raise RuntimeError(f"ANYFLOW is configured for sm_{cfg.target_sm}, but current GPU is sm_{current_sm}")
    else:
        applied["cuda.sm"] = None

    try:
        import torch._dynamo.config as dynamo_config

        applied["dynamo.dynamic_shapes"] = _set_config_attr(dynamo_config, "dynamic_shapes", cfg.dynamic_shapes)
        applied["dynamo.capture_scalar_outputs"] = _set_config_attr(dynamo_config, "capture_scalar_outputs", True)
    except Exception as exc:
        applied["dynamo_error"] = repr(exc)

    try:
        import torch._inductor.config as inductor_config

        applied["inductor.combo_kernels"] = _set_config_attr(inductor_config, "combo_kernels", cfg.combo_kernels)
        applied["inductor.deterministic"] = _set_config_attr(
            inductor_config, "deterministic", cfg.inductor_deterministic
        )
        _set_config_attr(inductor_config, "coordinate_descent_tuning", True)
        _set_config_attr(inductor_config, "triton.cudagraphs", False)
    except Exception as exc:
        applied["inductor_error"] = repr(exc)

    return applied


def sm75_inference_config(seed: int = 1337, require_sm75: bool = False) -> AnyFlowRuntimeConfig:
    return AnyFlowRuntimeConfig(
        seed=seed,
        deterministic=True,
        combo_kernels=True,
        inductor_deterministic=True,
        dynamic_shapes=True,
        compile_mode="reduce-overhead",
        compile_fullgraph=False,
        compile_dynamic=True,
        matmul_precision="highest",
        allow_tf32=False,
        cudnn_benchmark=False,
        cudnn_deterministic=True,
        autocast_dtype=torch.float16,
        target_sm=75,
        require_target_sm=require_sm75,
        compile_options={"triton.cudagraphs": False},
    )


def compile_anyflow(
    module_or_fn: torch.nn.Module | Callable[..., Any],
    config: AnyFlowRuntimeConfig | None = None,
    **overrides: Any,
) -> torch.nn.Module | Callable[..., Any]:
    cfg = config or AnyFlowRuntimeConfig()
    initialize_anyflow_runtime(cfg)
    if not hasattr(torch, "compile"):
        return module_or_fn

    options = dict(cfg.compile_options)
    options.update(overrides.pop("options", {}))
    options.setdefault("triton.cudagraphs", False)
    backend = overrides.pop("backend", cfg.compile_backend)
    mode = overrides.pop("mode", cfg.compile_mode)
    fullgraph = overrides.pop("fullgraph", cfg.compile_fullgraph)
    dynamic = overrides.pop("dynamic", cfg.compile_dynamic)

    compile_kwargs = {
        "backend": backend,
        "mode": mode,
        "fullgraph": fullgraph,
        "dynamic": dynamic,
        **overrides,
    }
    if backend == "inductor":
        compile_kwargs["options"] = options

    return torch.compile(module_or_fn, **compile_kwargs)


@contextlib.contextmanager
def precision_context(
    enabled: bool = True,
    device_type: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
):
    if not enabled:
        yield
        return

    if device_type is None:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"

    with torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled):
        yield
