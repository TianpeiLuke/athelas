"""
Shared configuration constants for PyTorch Lightning models.

This module defines which hyperparameters should be logged to TensorBoard.
"""

# Minimal whitelist of hyperparameters to log to TensorBoard
# Excludes runtime artifacts (risk_tables, imputation_dict, label_to_id, id_to_label, etc.)
# that are saved separately in model artifacts
TENSORBOARD_FIELDS = {
    # === Training hyperparameters ===
    "lr",
    "batch_size",
    "max_epochs",
    "weight_decay",
    "optimizer",
    "warmup_steps",
    "gradient_clip_val",
    "early_stop_patience",
    "adam_epsilon",
    "run_scheduler",
    "val_check_interval",
    "fp16",
    # === Model architecture ===
    "model_class",
    "hidden_common_dim",
    "num_classes",
    "is_binary",
    "dropout_keep",
    "embed_size",  # Added at runtime, but small (768)
    # === Fusion-specific (trimodal/bimodal) ===
    "fusion_hidden_dim",
    "fusion_dropout",
    # === CNN-specific ===
    "kernel_size",
    "num_layers",
    "num_channels",
    # === Text configuration ===
    "tokenizer",
    "primary_tokenizer",
    "secondary_tokenizer",
    "fixed_tokenizer_length",
    "is_embeddings_trainable",
    "reinit_pooler",
    "reinit_layers",
    "primary_reinit_pooler",
    "secondary_reinit_pooler",
    "primary_reinit_layers",
    "secondary_reinit_layers",
    "max_sen_len",
    "chunk_trancate",
    "max_total_chunks",
    # === Data field names (for reference) ===
    "primary_text_name",
    "secondary_text_name",
    "text_name",
    "text_input_ids_key",
    "text_attention_mask_key",
    # === Classification ===
    "class_weights",
    "multiclass_categories",
    # === Evaluation ===
    "metric_choices",
    "early_stop_metric",
    # === Preprocessing ===
    "smooth_factor",
    "count_threshold",
}


def filter_config_for_tensorboard(config: dict) -> dict:
    """
    Filter configuration dictionary to include only TensorBoard-relevant fields.

    This removes large runtime artifacts like:
    - risk_tables (large nested dictionaries)
    - imputation_dict (preprocessing artifacts)
    - label_to_id, id_to_label (can be reconstructed)
    - Any torch.Tensor objects

    Args:
        config: Full configuration dictionary

    Returns:
        Filtered dictionary with only essential hyperparameters
    """
    import torch

    return {
        k: v
        for k, v in config.items()
        if k in TENSORBOARD_FIELDS and not isinstance(v, torch.Tensor)
    }
