"""PyTorch Lightning models organized by modality type.

This package contains PyTorch Lightning model implementations organized into
subdirectories by modality type:
- text: Text-only models (BERT, LSTM, CNN)
- multimodal: 2-modality fusion models
- trimodal: 3-modality fusion models  
- tabular: Tabular-only models
- utils: Shared utilities and training infrastructure

For backward compatibility, all models are also exported at the top level.
"""

# Text models
from .text import (
    TextBertBase,
    TextBertBaseConfig,
    TextBertClassification,
    TextBertClassificationConfig,
    TextLSTM,
    TextCNN,
)

# Multimodal models
from .multimodal import (
    MultimodalBert,
    MultimodalCNN,
    MultimodalBertCrossAttn,
    CrossAttentionFusion,
    MultimodalBertGateFusion,
    GateFusion,
    MultimodalBertMoE,
    MixtureOfExperts,
)

# Trimodal models
from .trimodal import (
    TrimodalBert,
    TrimodalCrossAttentionBert,
    BidirectionalCrossAttention,
    TrimodalGateFusionBert,
    TrimodalGateFusion,
)

# Tabular models
from .tabular import (
    TabAE,
    TabularEmbeddingConfig,
    TabularEmbeddingModule,
)

# Utilities
from .utils import (
    # dist_utils
    get_world_size,
    get_rank,
    is_main_process,
    synchronize,
    get_local_process_group,
    get_local_rank,
    get_local_size,
    create_local_process_group,
    all_gather,
    gather,
    shared_random_seed,
    reduce_dict,
    print_gpu_memory_usage,
    print_gpu_memory_stats,
    # pl_model_plots
    compute_metrics,
    plot_to_tensorboard,
    roc_metric_plot,
    pr_metric_plot,
    # pl_train
    setup_logger,
    my_auto_wrap_policy,
    is_fsdp_available,
    model_train,
    extract_preds_and_labels,
    model_inference,
    model_online_inference,
    predict_stack_transform,
    unwrap_fsdp_model,
    save_prediction,
    save_model,
    save_artifacts,
    load_artifacts,
    load_model,
    load_checkpoint,
    load_onnx_model,
)

__all__ = [
    # Text models
    "TextBertBase",
    "TextBertBaseConfig",
    "TextBertClassification",
    "TextBertClassificationConfig",
    "TextLSTM",
    "TextCNN",
    # Multimodal models
    "MultimodalBert",
    "MultimodalCNN",
    "MultimodalBertCrossAttn",
    "CrossAttentionFusion",
    "MultimodalBertGateFusion",
    "GateFusion",
    "MultimodalBertMoE",
    "MixtureOfExperts",
    # Trimodal models
    "TrimodalBert",
    "TrimodalCrossAttentionBert",
    "BidirectionalCrossAttention",
    "TrimodalGateFusionBert",
    "TrimodalGateFusion",
    # Tabular models
    "TabAE",
    "TabularEmbeddingConfig",
    "TabularEmbeddingModule",
    # Utilities - dist_utils
    "get_world_size",
    "get_rank",
    "is_main_process",
    "synchronize",
    "get_local_process_group",
    "get_local_rank",
    "get_local_size",
    "create_local_process_group",
    "all_gather",
    "gather",
    "shared_random_seed",
    "reduce_dict",
    "print_gpu_memory_usage",
    "print_gpu_memory_stats",
    # Utilities - pl_model_plots
    "compute_metrics",
    "plot_to_tensorboard",
    "roc_metric_plot",
    "pr_metric_plot",
    # Utilities - pl_train
    "setup_logger",
    "my_auto_wrap_policy",
    "is_fsdp_available",
    "model_train",
    "extract_preds_and_labels",
    "model_inference",
    "model_online_inference",
    "predict_stack_transform",
    "unwrap_fsdp_model",
    "save_prediction",
    "save_model",
    "save_artifacts",
    "load_artifacts",
    "load_model",
    "load_checkpoint",
    "load_onnx_model",
]
