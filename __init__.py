from importlib import import_module

from .audio import AudioVelocityNet, DeepThinkingAudioFlowEngine
from .attention_residual import AttentionResidualMambaStack, GroupedDepthResidualRouter, assert_transformer_free
from .compiler import AnyFlowRuntimeConfig, compile_anyflow, initialize_anyflow_runtime
from .compiler import sm75_inference_config
from .data import SyntheticAnyFlowDataConfig, iter_synthetic_anyflow_batches, make_synthetic_anyflow_batch
from .engine import AnyFlowBatch, AnyFlowEngine, build_anyflow_small
from .flow import AsymmetricFlowMatcher, AsymmetricFlowSchedule, FlowSample, FlowSampler
from .kernel_validation import KernelParityResult, KernelValidationReport, run_kernel_validation
from .losses import (
    AuxiliaryLossConfig,
    auxiliary_flow_loss,
    multi_resolution_stft_loss,
    temporal_difference_loss,
    video_temporal_consistency_loss,
)
from .mamba3 import Mamba3Block, Mamba3Config, Mamba3Stack, VarLenResidualMixer
from .moe import (
    AnyFlowMoEAdapter,
    AnyFlowMoEConfig,
    MambaMoEOutput,
    MambaMoELayer,
    MoEDispatchPlan,
    MoERouterOutput,
    MoERouterStats,
    TopKTaskRouter,
    build_expert_dispatch,
    make_moe_config_for_modality,
    modality_to_task_ids,
)
from .quantization import (
    AnyFlowPrecisionContext,
    TileQuantizedTensor,
    dequantize_state_dict,
    dequantize_fp8_tile,
    dequantize_mxfp4,
    quantize_state_dict,
    quantize_fp8_tile,
    quantize_mxfp4,
    quantized_state_dict_stats,
)
from .training import AnyFlowTrainConfig, AnyFlowTrainer, TrainStepMetrics, build_sm75_trainer, move_batch_to_device
from .verification import AnyFlowVerificationReport, VerificationCheck, assert_anyflow_ready, run_anyflow_verification
from .video import DeepThinkingVideoFlowEngine, SpatialMambaVideoBlock, VideoVelocityNet

__all__ = [
    "AnyFlowBatch",
    "AnyFlowEngine",
    "AnyFlowMoEAdapter",
    "AnyFlowMoEConfig",
    "AnyFlowPrecisionContext",
    "AnyFlowRuntimeConfig",
    "AnyFlowTrainConfig",
    "AnyFlowTrainer",
    "AnyFlowVerificationReport",
    "AsymmetricFlowMatcher",
    "AsymmetricFlowSchedule",
    "AttentionResidualMambaStack",
    "AudioVelocityNet",
    "AuxiliaryLossConfig",
    "CompileValidationCheck",
    "CompileValidationConfig",
    "CompileValidationReport",
    "DeepThinkingAudioFlowEngine",
    "DeepThinkingVideoFlowEngine",
    "FlowSample",
    "FlowSampler",
    "GroupedDepthResidualRouter",
    "InferenceMetric",
    "KernelParityResult",
    "KernelValidationReport",
    "Mamba3Block",
    "Mamba3Config",
    "Mamba3Stack",
    "MambaMoEOutput",
    "MambaMoELayer",
    "MoEDispatchPlan",
    "MoERouterOutput",
    "MoERouterStats",
    "QuickTrainConfig",
    "SpatialMambaVideoBlock",
    "Sm75BenchmarkConfig",
    "Sm75BenchmarkReport",
    "SyntheticAnyFlowDataConfig",
    "TileQuantizedTensor",
    "TopKTaskRouter",
    "TrainStepMetrics",
    "VarLenResidualMixer",
    "VideoVelocityNet",
    "VerificationCheck",
    "assert_anyflow_ready",
    "auxiliary_flow_loss",
    "benchmark_sm75_inference",
    "build_anyflow_small",
    "build_colab_smoke_engine",
    "build_sm75_trainer",
    "build_expert_dispatch",
    "assert_transformer_free",
    "compile_anyflow",
    "dequantize_fp8_tile",
    "dequantize_mxfp4",
    "dequantize_state_dict",
    "initialize_anyflow_runtime",
    "iter_synthetic_anyflow_batches",
    "make_synthetic_anyflow_batch",
    "make_moe_config_for_modality",
    "modality_to_task_ids",
    "move_batch_to_device",
    "multi_resolution_stft_loss",
    "quantize_fp8_tile",
    "quantize_mxfp4",
    "quantize_state_dict",
    "quantized_state_dict_stats",
    "run_quick_train",
    "run_anyflow_verification",
    "run_compile_validation",
    "run_kernel_validation",
    "sm75_inference_config",
    "temporal_difference_loss",
    "video_temporal_consistency_loss",
]


_LAZY_EXPORTS = {
    "InferenceMetric": ("anyflow.benchmark", "InferenceMetric"),
    "Sm75BenchmarkConfig": ("anyflow.benchmark", "Sm75BenchmarkConfig"),
    "Sm75BenchmarkReport": ("anyflow.benchmark", "Sm75BenchmarkReport"),
    "benchmark_sm75_inference": ("anyflow.benchmark", "benchmark_sm75_inference"),
    "CompileValidationCheck": ("anyflow.compile_validation", "CompileValidationCheck"),
    "CompileValidationConfig": ("anyflow.compile_validation", "CompileValidationConfig"),
    "CompileValidationReport": ("anyflow.compile_validation", "CompileValidationReport"),
    "run_compile_validation": ("anyflow.compile_validation", "run_compile_validation"),
    "QuickTrainConfig": ("anyflow.colab", "QuickTrainConfig"),
    "build_colab_smoke_engine": ("anyflow.colab", "build_colab_smoke_engine"),
    "run_quick_train": ("anyflow.colab", "run_quick_train"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        value = getattr(import_module(module_name), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'anyflow' has no attribute {name!r}")
