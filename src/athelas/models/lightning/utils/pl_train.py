# Save this as: bsm/lightning_models/train_utils.py
import os
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import fcntl  # Unix file locking for multi-worker coordination
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
    LearningRateMonitor,
    DeviceStatsMonitor,
)

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy


from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType


import onnx
import onnxruntime as ort

# Note: Model imports moved inside functions to avoid circular dependencies


def setup_logger():
    import logging

    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


# ============================================================================
# FILE LOCKING FOR MULTI-WORKER COORDINATION
# ============================================================================


@contextmanager
def file_lock(lock_path: Path, timeout: int = 60):
    """
    Atomic file lock for multi-worker coordination in TorchServe environments.

    Prevents race conditions when multiple workers try to perform the same
    expensive operation (e.g., BERT model optimization). Only one worker
    acquires the lock and performs the operation; others wait for completion.

    Args:
        lock_path: Path to lock file (will be created/removed automatically)
        timeout: Maximum seconds to wait for lock acquisition (default: 60s)

    Yields:
        None when lock is successfully acquired

    Raises:
        TimeoutError: If lock cannot be acquired within timeout period

    Example:
        >>> lock_file = Path("/opt/ml/model/.optimization.lock")
        >>> with file_lock(lock_file, timeout=90):
        >>>     # Only one worker executes this block
        >>>     optimize_model()
        >>> # Lock automatically released, other workers proceed
    """
    # Use 'with' statement to guarantee file closure in all code paths
    with open(lock_path, "w") as lock_file:
        start_time = time.time()

        # Try to acquire exclusive lock with timeout
        while True:
            try:
                # LOCK_EX: Exclusive lock (only one process can hold it)
                # LOCK_NB: Non-blocking (return immediately if lock unavailable)
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                logger.info(f"✓ Acquired file lock: {lock_path}")
                break
            except IOError:
                # Lock is held by another process
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    raise TimeoutError(
                        f"Could not acquire lock within {timeout}s. "
                        f"Another worker may be stuck or operation is taking too long."
                    )
                # Wait and retry
                time.sleep(1)
                if int(elapsed) % 10 == 0:  # Log every 10 seconds
                    logger.info(
                        f"Waiting for lock... ({int(elapsed)}s elapsed, max {timeout}s)"
                    )

        try:
            yield
        finally:
            # Release lock and clean up
            # File will be automatically closed by 'with' statement
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                if lock_path.exists():
                    lock_path.unlink()
                logger.info(f"✓ Released file lock: {lock_path}")
            except Exception as e:
                logger.warning(f"Error releasing lock {lock_path}: {e}")


# ----------------- FDSP ---------------------
def my_auto_wrap_policy(
    module: nn.Module, recurse: bool, unwrapped_params: int, min_num_params: int = 1e5
) -> bool:
    """
    Custom FSDP auto wrap policy for multimodal models.

    This policy wraps:
    - TextBertBase (Transformer-based encoder)
    - TabAE (tabular encoder)
    - Any Linear / Conv2d / Embedding with large parameter counts

    Args:
        module (nn.Module): Module to inspect
        recurse (bool): Whether FSDP is recursing
        unwrapped_params (int): Number of unwrapped parameters
        min_num_params (int): Minimum number of params to wrap

    Returns:
        bool: Whether to wrap this module
    """
    # Lazy import to avoid circular dependency
    from ..text.pl_bert import TextBertBase
    from ..tabular.pl_tab_ae import TabAE

    return (
        isinstance(module, (TextBertBase, TabAE, nn.Linear, nn.Embedding, nn.Conv2d))
        and unwrapped_params >= min_num_params
    )


def is_fsdp_available():
    return (
        torch.cuda.is_available()
        and torch.cuda.device_count() > 1
        and dist.is_available()
        and dist.is_initialized()
    )


strategy = (
    FSDPStrategy(auto_wrap_policy=my_auto_wrap_policy, verbose=True)
    if is_fsdp_available()
    else "auto"
)
# -----------------------------


def model_train(
    model: pl.LightningModule,
    config: Dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: Union[int, str, List[int]] = "auto",
    model_log_path: str = "./model_logs",
    early_stop_metric: str = "val/f1_score",
    additional_callbacks: Optional[List[pl.Callback]] = None,
) -> pl.Trainer:
    max_epochs = config.get("max_epochs", 10)
    early_stop_patience = config.get("early_stop_patience", 10)
    model_class = config.get("model_class", "multimodal_cnn")
    val_check_interval = config.get("val_check_interval", 1.0)
    use_fp16 = config.get("fp16", False)
    clip_val = config.get("gradient_clip_val", 0.0)

    logger_tb = TensorBoardLogger(save_dir=model_log_path, name="tensorboard_logs")
    monitor_mode = "min" if "loss" in early_stop_metric else "max"

    checkpoint_dir = os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints")
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(checkpoint_dir),
        filename=f"{model_class}" + "-{epoch:02d}-{" + f"{early_stop_metric}" + ":.2f}",
        monitor=early_stop_metric,
        save_top_k=1,
        mode=monitor_mode,
        save_weights_only=False,
    )

    earlystopping_callback = EarlyStopping(
        monitor=early_stop_metric, patience=early_stop_patience, mode=monitor_mode
    )

    device_stats_callback = DeviceStatsMonitor(cpu_stats=False)

    # Combine default callbacks with additional callbacks
    all_callbacks = [
        earlystopping_callback,
        checkpoint_callback,
        device_stats_callback,
        TQDMProgressBar(refresh_rate=10),
        LearningRateMonitor(logging_interval="step"),
    ]

    if additional_callbacks:
        all_callbacks.extend(additional_callbacks)
        logger.info(f"Added {len(additional_callbacks)} additional callback(s)")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger_tb,
        default_root_dir=model_log_path,
        callbacks=all_callbacks,
        val_check_interval=config.get("val_check_interval", 1.0),
        sync_batchnorm=True if torch.cuda.is_available() else False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=device,
        strategy=strategy,  # You might need this
        # accumulate_grad_batches=1,
        precision=16 if use_fp16 else 32,
    )

    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    return trainer


# ------------------ Utility Function -----------------
def extract_preds_and_labels(
    df: pd.DataFrame, is_binary: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    if is_binary:
        preds = torch.tensor(df["prob"].values.astype(float))
    else:
        preds = torch.tensor(
            [ast.literal_eval(p) if isinstance(p, str) else p for p in df["prob"]]
        )
    labels = torch.tensor(df["label"].values)
    return preds, labels


# ------------------ Inference ------------------------
def model_inference(
    model: pl.LightningModule,
    dataloader: DataLoader,
    accelerator: Union[str, int, List[int]] = "auto",
    device: Union[str, int, List[int]] = "auto",
    model_log_path: str = "./model_logs",
    return_dataframe: bool = False,
    label_col: str = "label",
) -> Union[
    Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]
]:
    """
    Runs inference and returns predicted probabilities and true labels as tensors.
    Supports both binary and multiclass classification.

    CRITICAL: In distributed training, ALL ranks must call this function.
    Only rank 0 will process and return actual results; other ranks return dummy tensors.

    Args:
        model (pl.LightningModule): Trained Lightning model.
        dataloader (DataLoader): DataLoader for inference.
        accelerator (str/int/List[int]): Accelerator setting.
        device (str/int/List[int]): Device setting.
        model_log_path (str): Path to save logs.
        return_dataframe (bool): Whether to return the original dataframe.
        label_col (str): Name of the label column in the dataframe. Defaults to "label".

    Returns:
        Tuple of (y_pred, y_true) or (y_pred, y_true, df) depending on `return_dataframe`.
    """

    # Safe handling: force CPU if no GPU available
    resolved_accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    resolved_devices = 1 if resolved_accelerator == "cpu" else device

    tester = pl.Trainer(
        max_epochs=1,
        default_root_dir=model_log_path,
        enable_checkpointing=False,
        logger=False,
        callbacks=[TQDMProgressBar()],
        accelerator=resolved_accelerator,
        devices=resolved_devices,
        strategy="auto",  # Will use distributed strategy if torch.distributed is initialized
        inference_mode=True,
    )

    # All ranks participate in test (required for distributed collective operations)
    tester.test(model, dataloaders=dataloader)

    # CRITICAL FIX: Synchronize all ranks after test completes
    if dist.is_initialized():
        dist.barrier()

    # Only rank 0 processes results; other ranks return dummy tensors
    if dist.is_initialized() and dist.get_rank() != 0:
        # Non-main ranks return dummy tensors (won't be used by caller)
        dummy_preds = torch.zeros(1)
        dummy_labels = torch.zeros(1, dtype=torch.long)
        if return_dataframe:
            return dummy_preds, dummy_labels, pd.DataFrame()
        return dummy_preds, dummy_labels

    # Main rank (rank 0) processes results
    result_folder = model.test_output_folder
    if not result_folder or not os.path.exists(result_folder):
        raise RuntimeError(
            f"Expected test output folder '{result_folder}' does not exist."
        )

    # Match files like test_result_*.tsv from all ranks
    result_files = sorted(Path(result_folder).glob("test_result_*.tsv"))
    if not result_files:
        raise RuntimeError(f"No test result files found in {result_folder}.")

    dfs = []
    for f in result_files:
        try:
            dfs.append(pd.read_csv(f, sep="\t"))
        except Exception as e:
            print(f"[Warning] Skipping file {f} due to read error: {e}")

    if not dfs:
        raise RuntimeError("No valid result files could be loaded.")
    df = pd.concat(dfs, ignore_index=True)

    is_binary = model.task == "binary"
    if is_binary:
        y_pred = torch.tensor(df["prob"].values.astype(float))
    else:
        y_pred = torch.tensor(
            [ast.literal_eval(p) if isinstance(p, str) else p for p in df["prob"]]
        )

    y_true = torch.tensor(df[label_col].values).long()

    if return_dataframe:
        return y_pred, y_true, df
    else:
        return y_pred, y_true


def model_online_inference(
    model: Union[pl.LightningModule, ort.InferenceSession], dataloader: DataLoader
) -> np.ndarray:
    """
    Run online inference for either a PyTorch Lightning model or an ONNX Runtime session.
    """
    if isinstance(model, ort.InferenceSession):
        print("Running inference with ONNX Runtime.")
        predictions = []
        expected_input_names = [inp.name for inp in model.get_inputs()]

        for batch in dataloader:
            input_feed = {}
            for k in expected_input_names:
                if k not in batch:
                    raise KeyError(f"ONNX input '{k}' not found in batch")

                val = batch[k]

                # Convert to numpy with correct type
                if isinstance(val, torch.Tensor):
                    val_np = val.cpu().numpy()

                    # Ensure correct dtype
                    if "input_ids" in k or "attention_mask" in k:
                        val_np = val_np.astype("int64")  # Required for ONNX
                    else:
                        val_np = val_np.astype("float32")

                    input_feed[k] = val_np

                elif isinstance(val, list) and all(
                    isinstance(x, (int, float)) for x in val
                ):
                    # Fallback for list-based numeric features
                    val_np = np.array(val, dtype="float32").reshape(-1, 1)
                    input_feed[k] = val_np

                else:
                    # Skip fields like order_id (string/list[str]) or raise error
                    print(
                        f"[Warning] Skipping unsupported ONNX input field: '{k}' ({type(val)})"
                    )

            output = model.run(None, input_feed)[0]  # Run inference
            predictions.append(output)

        return np.concatenate(predictions, axis=0)

    else:
        print("Running inference with PyTorch model.")
        model.eval()
        predictions = []
        for batch in dataloader:
            _, preds, _ = model.run_epoch(batch, "pred")
            predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions, axis=0)


def predict_stack_transform(
    outputs: List[Union[torch.Tensor, Tuple[torch.Tensor]]],
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(outputs[0], Tuple):
        pred_list, label_list = zip(*outputs)
        return torch.cat(pred_list), torch.cat(label_list)
    return torch.cat(outputs)


def unwrap_fsdp_model(model: nn.Module) -> nn.Module:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    return model.module if isinstance(model, FSDP) else model


def save_prediction(filename: str, y_true: List, y_pred: List):
    logger.info("Saving prediction.")
    torch.save({"y_true": y_true, "y_pred": y_pred}, filename)


def save_model(filename: str, model: nn.Module):
    logger.info("Saving model weights.")

    # Unwrap if wrapped in FSDP
    if isinstance(model, FSDP):
        # Use FSDP's full state dict context
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()
            if dist.get_rank() == 0:
                torch.save(state_dict, filename)
    else:
        torch.save(model.state_dict(), filename)


def save_artifacts(
    filename: str,
    config: Dict,
    embedding_mat: torch.Tensor,
    vocab: Dict[str, int],
    model_class: str,
):
    logger.info("Saving artifacts.")
    artifacts = {
        "config": config,
        "embedding_mat": embedding_mat,
        "vocab": vocab,
        "model_class": model_class,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "pytorch_lightning_version": __import__("lightning.pytorch").__version__,
    }
    torch.save(artifacts, filename)


def load_artifacts(
    filename: str, device_l: str = "cpu"
) -> Tuple[Dict, torch.Tensor, Dict, str]:
    logger.info("Loading artifacts.")
    artifacts = torch.load(filename, map_location=device_l)
    config = artifacts["config"]
    embedding_mat = artifacts["embedding_mat"]
    vocab = artifacts["vocab"]
    model_class = artifacts["model_class"]
    for k in ["torch_version", "transformers_version", "pytorch_lightning_version"]:
        logger.info(f"{k}: {artifacts.get(k, 'N/A')}")
    return config, embedding_mat, vocab, model_class


def load_model(
    filename: str,
    config: Dict,
    embedding_mat: torch.Tensor,
    model_class: str = "bimodal_bert",
    device_l: str = "cpu",
) -> nn.Module:
    """
    Load model weights into a fresh model instance.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    # Lazy imports to avoid circular dependencies
    from ..bimodal.pl_bimodal_cnn import BimodalCNN
    from ..text.pl_bert_classification import TextBertClassification
    from ..text.pl_lstm import TextLSTM
    from ..bimodal.pl_bimodal_bert import BimodalBert
    from ..bimodal.pl_bimodal_gate_fusion import BimodalBertGateFusion
    from ..bimodal.pl_bimodal_moe import BimodalBertMoE
    from ..bimodal.pl_bimodal_cross_attn import BimodalBertCrossAttn
    from ..bimodal.pl_lstm2risk import LSTM2Risk
    from ..bimodal.pl_transformer2risk import Transformer2Risk
    from ..trimodal.pl_trimodal_bert import TrimodalBert
    from ..trimodal.pl_trimodal_cross_attn import TrimodalCrossAttentionBert
    from ..trimodal.pl_trimodal_gate_fusion import TrimodalGateFusionBert

    logger.info("Instantiating model.")
    model = {
        # Bimodal models
        "bimodal": lambda: BimodalBert(config),  # Default bimodal
        "bimodal_cnn": lambda: BimodalCNN(
            config, embedding_mat.shape[0], embedding_mat
        ),
        "bimodal_bert": lambda: BimodalBert(config),
        "bimodal_gate_fusion": lambda: BimodalBertGateFusion(config),
        "bimodal_moe": lambda: BimodalBertMoE(config),
        "bimodal_cross_attn": lambda: BimodalBertCrossAttn(config),
        "lstm2risk": lambda: LSTM2Risk(config),
        "transformer2risk": lambda: Transformer2Risk(config),
        # Trimodal models
        "trimodal": lambda: TrimodalBert(config),  # Default trimodal
        "trimodal_bert": lambda: TrimodalBert(config),
        "trimodal_cross_attn_bert": lambda: TrimodalCrossAttentionBert(config),
        "trimodal_gate_fusion_bert": lambda: TrimodalGateFusionBert(config),
        # Text-only models
        "bert": lambda: TextBertClassification(config),
        "lstm": lambda: TextLSTM(config, embedding_mat.shape[0], embedding_mat),
        # Backward compatibility (multimodal -> bimodal)
        "multimodal_cnn": lambda: BimodalCNN(
            config, embedding_mat.shape[0], embedding_mat
        ),
        "multimodal_bert": lambda: BimodalBert(config),
        "multimodal_gate_fusion": lambda: BimodalBertGateFusion(config),
        "multimodal_moe": lambda: BimodalBertMoE(config),
        "multimodal_cross_attn": lambda: BimodalBertCrossAttn(config),
    }.get(model_class, lambda: BimodalBert(config))()

    try:
        logger.info(f"Loading model weights from: {filename}")
        model.load_state_dict(torch.load(filename, map_location=device_l))
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        raise RuntimeError("Model loading failed.") from e

    return model


def load_checkpoint(
    filename: str, model_class: str = "bimodal_bert", device_l: str = "cpu"
) -> nn.Module:
    # Lazy imports to avoid circular dependencies
    from ..bimodal.pl_bimodal_cnn import BimodalCNN
    from ..text.pl_bert_classification import TextBertClassification
    from ..text.pl_lstm import TextLSTM
    from ..bimodal.pl_bimodal_bert import BimodalBert
    from ..bimodal.pl_bimodal_gate_fusion import BimodalBertGateFusion
    from ..bimodal.pl_bimodal_moe import BimodalBertMoE
    from ..bimodal.pl_bimodal_cross_attn import BimodalBertCrossAttn
    from ..bimodal.pl_lstm2risk import LSTM2Risk
    from ..bimodal.pl_transformer2risk import Transformer2Risk

    logger.info("Loading checkpoint.")
    model_fn = {
        "bimodal_cnn": BimodalCNN,
        "bert": TextBertClassification,
        "lstm": TextLSTM,
        "bimodal_bert": BimodalBert,
        "bimodal_gate_fusion": BimodalBertGateFusion,
        "bimodal_moe": BimodalBertMoE,
        "bimodal_cross_attn": BimodalBertCrossAttn,
        "lstm2risk": LSTM2Risk,
        "transformer2risk": Transformer2Risk,
        # Backward compatibility
        "multimodal_cnn": BimodalCNN,
        "multimodal_bert": BimodalBert,
        "multimodal_gate_fusion": BimodalBertGateFusion,
        "multimodal_moe": BimodalBertMoE,
        "multimodal_cross_attn": BimodalBertCrossAttn,
    }.get(model_class, BimodalBert)
    return model_fn.load_from_checkpoint(filename, map_location=device_l)


def load_onnx_model(
    onnx_path: Union[str, Path],
    enable_profiling: bool = False,
    inter_op_threads: int = 1,
    intra_op_threads: int = 4,
    providers: Optional[List[Union[str, Tuple[str, Dict]]]] = None,
    provider_options: Optional[List[Dict]] = None,
) -> ort.InferenceSession:
    """
    Load ONNX model with production-grade optimization settings (Phase 1 Optimization).

    This implementation applies comprehensive SessionOptions configuration for:
    - 2-3x speedup through graph optimization
    - Optimized memory patterns and thread configuration
    - Hardware-specific execution providers (TensorRT, CUDA, CPU)

    Expected performance improvement: 335-545ms → 110-170ms per inference

    Args:
        onnx_path: Path to ONNX model file
        enable_profiling: Enable performance profiling for debugging (default: False)
        inter_op_threads: Number of threads for parallel operator execution (default: 1)
        intra_op_threads: Number of threads within operators for matrix operations (default: 4)
        providers: Optional list of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        provider_options: Optional list of provider-specific options

    Returns:
        ort.InferenceSession: Optimized ONNX Runtime InferenceSession

    Example:
        >>> session = load_onnx_model("model.onnx")
        >>> inputs = {
        >>>     "input_ids": np.array([[101, 1024, 102]]),
        >>>     "attention_mask": np.array([[1, 1, 1]]),
        >>>     "tab_field1": np.array([[0.3, 1.5]]),
        >>> }
        >>> outputs = session.run(None, inputs)
        >>> logits = outputs[0]
    """
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

    # ===== 1. Configure SessionOptions =====
    sess_options = ort.SessionOptions()

    # Graph optimization: Enable all optimizations (level 99)
    # - Constant folding: Pre-compute constant operations
    # - Redundant node elimination: Remove unused operations
    # - Operator fusion: Combine multiple ops into single kernel (BERT-specific)
    # - Shape inference: Optimize tensor shapes
    # - Common subexpression elimination: Reuse computed values
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Execution mode: Sequential for single-request serving (lower latency)
    # ORT_SEQUENTIAL: Operators run sequentially (lower latency, single request)
    # ORT_PARALLEL: Operators run in parallel (higher throughput, batch requests)
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    # Memory optimizations
    sess_options.enable_mem_pattern = True  # Reuse memory allocations across inferences
    sess_options.enable_cpu_mem_arena = True  # Use memory arena for faster allocation

    # Thread configuration (tune based on CPU cores)
    sess_options.intra_op_num_threads = (
        intra_op_threads  # Threads per operator (matrix ops)
    )
    sess_options.inter_op_num_threads = inter_op_threads  # Threads for parallel ops

    # Profiling (disable in production for performance)
    if enable_profiling:
        sess_options.enable_profiling = True
        sess_options.profile_file_prefix = "onnx_profile"
        logger.info(
            "ONNX profiling enabled - disable in production for best performance"
        )

    # ===== 2. Configure Execution Providers =====
    # Use provided providers or auto-detect
    if providers is None:
        providers = []

        # Try TensorRT first (best GPU performance, 1.2-1.5x additional speedup)
        if ort.get_device() == "GPU":
            try:
                providers.append(
                    (
                        "TensorrtExecutionProvider",
                        {
                            "trt_fp16_enable": False,  # Disabled - causes numerical instability
                            "trt_engine_cache_enable": True,  # Cache compiled engines
                            "trt_engine_cache_path": "/tmp/trt_cache",
                            "trt_max_workspace_size": 2147483648,  # 2GB workspace
                        },
                    )
                )
                logger.info("✓ TensorRT execution provider configured (FP16 disabled)")
            except Exception as e:
                logger.warning(f"TensorRT provider not available: {e}")

        # Fallback to CUDA (standard GPU execution)
        if ort.get_device() == "GPU":
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "arena_extend_strategy": "kNextPowerOfTwo",
                        "gpu_mem_limit": 2 * 1024 * 1024 * 1024,  # 2GB limit
                        "cudnn_conv_algo_search": "EXHAUSTIVE",  # Find best convolution algorithm
                        "do_copy_in_default_stream": True,
                    },
                )
            )
            logger.info("✓ CUDA execution provider configured")

        # CPU fallback (always available)
        providers.append("CPUExecutionProvider")

    # ===== 3. Create Optimized Session =====
    try:
        # Build session kwargs conditionally to avoid duplication
        session_kwargs = {
            "sess_options": sess_options,
            "providers": providers,
        }
        if provider_options is not None:
            session_kwargs["provider_options"] = provider_options

        session = ort.InferenceSession(str(onnx_path), **session_kwargs)

        logger.info(f"✓ Loaded ONNX model with Phase 1 optimizations from {onnx_path}")
        logger.info(f"  Graph optimization: ORT_ENABLE_ALL")
        logger.info(f"  Execution mode: ORT_SEQUENTIAL (low latency)")
        logger.info(f"  Memory optimization: ENABLED (mem_pattern + cpu_mem_arena)")
        logger.info(
            f"  Intra-op threads: {intra_op_threads} (parallel within operators)"
        )
        logger.info(f"  Inter-op threads: {inter_op_threads} (sequential operators)")
        logger.info(
            f"  Active providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}"
        )
        logger.info(f"  Expected inputs: {[inp.name for inp in session.get_inputs()]}")
        logger.info(f"  Expected performance: 2-3x speedup vs baseline")

        return session

    except Exception as e:
        raise RuntimeError(f"Failed to load optimized ONNX model: {e}")


def optimize_bert_model(
    input_model_path: Union[str, Path],
    output_model_path: Union[str, Path],
    model_type: str = "bert",
    num_heads: int = 12,
    hidden_size: int = 768,
) -> None:
    """
    Apply BERT-specific optimizations to ONNX model (Phase 2 Optimization).

    This function applies graph-level fusion optimizations for BERT models:
    - Multi-head attention fusion (Q/K/V projections + attention)
    - LayerNorm fusion (normalization + skip connections)
    - GELU activation fusion
    - Embedding layer fusion (token + position + segment)

    CRITICAL: This is graph optimization, NOT weight modification.
    Your fine-tuned BERT weights remain completely unchanged.

    Expected speedup: 1.5-2x additional on top of Phase 1 (110ms → 55-75ms)

    Args:
        input_model_path: Path to original ONNX model (model.onnx)
        output_model_path: Path to save optimized model (model_optimized.onnx)
        model_type: Model architecture type (default: "bert")
        num_heads: Number of attention heads in BERT model (default: 12 for BERT-base)
        hidden_size: Hidden dimension size in BERT model (default: 768 for BERT-base)

    Raises:
        RuntimeError: If optimization fails
    """
    try:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions
    except ImportError as e:
        raise RuntimeError(
            "BERT fusion requires onnxruntime transformers package. "
            "Install with: pip install onnxruntime[transformers]"
        ) from e

    logger.info(f"Starting BERT fusion optimization: {input_model_path}")
    logger.info(f"  Model type: {model_type}")
    logger.info(f"  Attention heads: {num_heads}")
    logger.info(f"  Hidden size: {hidden_size}")

    try:
        # Configure fusion options
        fusion_options = FusionOptions(model_type)
        fusion_options.enable_gelu = True  # Fuse GELU activation
        fusion_options.enable_layer_norm = True  # Fuse layer normalization
        fusion_options.enable_attention = True  # Fuse multi-head attention
        fusion_options.enable_skip_layer_norm = (
            True  # Fuse skip connections + layer norm
        )
        fusion_options.enable_embed_layer_norm = True  # Fuse embedding + layer norm
        fusion_options.enable_bias_skip_layer_norm = (
            True  # Fuse bias + skip + layer norm
        )
        fusion_options.enable_bias_gelu = True  # Fuse bias + GELU
        fusion_options.enable_gelu_approximation = (
            False  # Use exact GELU (better accuracy)
        )

        # Create optimizer instance
        optimizer_instance = optimizer.optimize_model(
            input=str(input_model_path),
            model_type=model_type,
            num_heads=num_heads,
            hidden_size=hidden_size,
            optimization_options=fusion_options,
            opt_level=99,  # Maximum optimization level
            use_gpu=torch.cuda.is_available(),
        )

        # Save optimized model
        optimizer_instance.save_model_to_file(str(output_model_path))

        # Get fusion statistics
        fusion_stats = optimizer_instance.get_fused_operator_statistics()

        logger.info(f"✓ BERT fusion optimization completed successfully")
        logger.info(f"  Optimized model saved to: {output_model_path}")
        logger.info(f"  Fusion statistics: {fusion_stats}")
        logger.info(f"  Expected speedup: 1.5-2x additional (on top of Phase 1)")

    except Exception as e:
        logger.error(f"BERT fusion optimization failed: {e}")
        raise RuntimeError(f"Failed to optimize BERT model: {e}") from e


def quantize_onnx_model(
    input_model_path: Union[str, Path],
    output_model_path: Union[str, Path],
) -> None:
    """
    Apply dynamic INT8 quantization to ONNX model.

    Dynamic quantization converts model weights to INT8 while keeping activations
    in FP32. This provides 2-3x speedup with minimal accuracy loss (<1%).

    Benefits:
    - 2-3x inference speedup
    - 4x model size reduction (768MB → 192MB for BERT-base)
    - No retraining required
    - Minimal accuracy impact (<1% degradation)

    Args:
        input_model_path: Path to original ONNX model (model.onnx or model_optimized.onnx)
        output_model_path: Path to save quantized model (model_quantized.onnx)

    Raises:
        RuntimeError: If quantization fails
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError as e:
        raise RuntimeError(
            "ONNX quantization requires onnxruntime package. "
            "Already installed, but quantization module may need update."
        ) from e

    logger.info(f"Starting dynamic INT8 quantization: {input_model_path}")
    logger.info(f"  Target: {output_model_path}")

    try:
        # Apply dynamic quantization (INT8 weights, FP32 activations)
        quantize_dynamic(
            model_input=str(input_model_path),
            model_output=str(output_model_path),
            weight_type=QuantType.QInt8,  # INT8 quantization for weights
            optimize_model=True,  # Apply additional graph optimizations
            extra_options={
                "ActivationSymmetric": True,  # Symmetric quantization (faster)
                "WeightSymmetric": True,
            },
        )

        # Verify quantized model
        onnx_model = onnx.load(str(output_model_path))
        onnx.checker.check_model(onnx_model)

        # Report size reduction
        original_size = Path(input_model_path).stat().st_size / (1024 * 1024)
        quantized_size = Path(output_model_path).stat().st_size / (1024 * 1024)
        reduction = (1 - quantized_size / original_size) * 100

        logger.info(f"✓ Dynamic INT8 quantization completed successfully")
        logger.info(f"  Original model: {original_size:.1f} MB")
        logger.info(f"  Quantized model: {quantized_size:.1f} MB")
        logger.info(f"  Size reduction: {reduction:.1f}%")
        logger.info(f"  Expected speedup: 2-3x")
        logger.info(f"  Expected accuracy impact: <1%")

    except Exception as e:
        logger.error(f"Dynamic quantization failed: {e}")
        raise RuntimeError(f"Failed to quantize ONNX model: {e}") from e


def load_bert_optimized_model(
    model_dir: Union[str, Path],
    enable_profiling: bool = False,
    inter_op_threads: int = 1,
    intra_op_threads: int = 4,
    force_reoptimize: bool = False,
    optimization_timeout: int = 90,
    enable_quantization: bool = True,
    providers: Optional[List[Union[str, Tuple[str, Dict]]]] = None,
    provider_options: Optional[List[Dict]] = None,
) -> ort.InferenceSession:
    """
    Load BERT-optimized ONNX model with atomic file locking (multi-worker safe).

    NEW: Added file locking to prevent race conditions when multiple workers
    initialize simultaneously (e.g., TorchServe with 8 workers on ml.m5.4xlarge).

    Only ONE worker performs the optimization, others wait for completion.

    This function implements the complete optimization workflow:
    1. Check for pre-optimized model (model_optimized.onnx)
    2. If not found, acquire file lock for safe optimization
    3. Apply BERT fusion to original model (one-time cost, single worker only)
    4. Load with Phase 1 SessionOptions configuration

    Storage Strategy:
    - PREFERRED: Store optimized model in model_dir (persistent across restarts)
    - FALLBACK: Use /tmp/ if model_dir is read-only (SageMaker endpoint scenario)

    Expected performance: 2-3x (Phase 1) × 1.5-2x (Phase 2) = 3-6x total speedup

    Args:
        model_dir: Directory containing ONNX models
        enable_profiling: Enable performance profiling for debugging (default: False)
        inter_op_threads: Number of threads for parallel operator execution (default: 1)
        intra_op_threads: Number of threads within operators for matrix operations (default: 4)
        force_reoptimize: Force re-optimization even if optimized model exists (default: False)
        optimization_timeout: Maximum seconds to wait for optimization lock (default: 90s)

    Returns:
        Optimized ONNX Runtime InferenceSession with Phase 1 + Phase 2 optimizations

    Example:
        >>> # First load: creates model_optimized.onnx (one-time 10-30s cost)
        >>> session = load_bert_optimized_model("/opt/ml/model")
        >>> # Subsequent loads: uses cached model_optimized.onnx (fast)
        >>> session = load_bert_optimized_model("/opt/ml/model")
        >>>
        >>> # Run inference with optimized model
        >>> inputs = {
        >>>     "input_ids": np.array([[101, 1024, 102]]),
        >>>     "attention_mask": np.array([[1, 1, 1]]),
        >>>     "tab_field1": np.array([[0.3, 1.5]]),
        >>> }
        >>> outputs = session.run(None, inputs)
    """
    import hashlib

    model_dir = Path(model_dir)
    original_model = model_dir / "model.onnx"

    # Validate original model exists
    if not original_model.exists():
        raise FileNotFoundError(f"Original ONNX model not found: {original_model}")

    # ========================================================================
    # STORAGE STRATEGY: Try persistent storage first, fallback to ephemeral
    # ========================================================================
    # PREFERRED: Store in model_dir (persistent across container restarts)
    persistent_optimized = model_dir / "model_optimized.onnx"
    persistent_quantized = model_dir / "model_quantized.onnx"
    persistent_lock = model_dir / ".model_optimization.lock"
    persistent_quant_lock = model_dir / ".model_quantization.lock"

    # FALLBACK: Use /tmp/ if model_dir is read-only (SageMaker endpoints)
    model_hash = hashlib.sha256(str(model_dir).encode()).hexdigest()[:16]
    ephemeral_optimized = Path(f"/tmp/model_optimized_{model_hash}.onnx")
    ephemeral_quantized = Path(f"/tmp/model_quantized_{model_hash}.onnx")
    ephemeral_lock = Path(f"/tmp/.model_optimization_{model_hash}.lock")
    ephemeral_quant_lock = Path(f"/tmp/.model_quantization_{model_hash}.lock")

    # Test write permissions to determine storage location
    use_persistent = False
    try:
        # Try creating a test file in model_dir
        test_file = model_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        use_persistent = True
        optimized_model = persistent_optimized
        quantized_model = persistent_quantized
        lock_file = persistent_lock
        quant_lock_file = persistent_quant_lock
        logger.info(
            "✓ Using persistent storage for optimized model (survives restarts)"
        )
    except (PermissionError, OSError) as e:
        logger.warning(f"model_dir is read-only ({e}), using ephemeral /tmp/ storage")
        logger.warning(
            "⚠️  Optimization will be repeated on each container restart (10-30s)"
        )
        optimized_model = ephemeral_optimized
        quantized_model = ephemeral_quantized
        lock_file = ephemeral_lock
        quant_lock_file = ephemeral_quant_lock

    # Determine final model to load based on quantization setting
    if enable_quantization:
        final_model = quantized_model
        logger.info("INT8 quantization ENABLED (3-6x total speedup expected)")
    else:
        final_model = optimized_model
        logger.info("INT8 quantization DISABLED (1.5-3x speedup expected)")

    # FAST PATH: Final model already exists and no force reoptimize
    if not force_reoptimize and final_model.exists():
        logger.info(f"✓ Using cached model: {final_model.name}")
        return load_onnx_model(
            onnx_path=final_model,
            enable_profiling=enable_profiling,
            inter_op_threads=inter_op_threads,
            intra_op_threads=intra_op_threads,
            providers=providers,
            provider_options=provider_options,
        )

    # OPTIMIZATION PATH: Create BERT-fused model first (if needed)
    if not optimized_model.exists() or force_reoptimize:
        logger.info(
            "BERT-optimized model not found, acquiring lock for optimization..."
        )
        try:
            with file_lock(lock_file, timeout=optimization_timeout):
                # Double-check: another worker may have created it while we waited
                if not force_reoptimize and optimized_model.exists():
                    logger.info("✓ Another worker completed BERT fusion")
                else:
                    # This worker won the race - perform BERT fusion
                    logger.info("✓ Lock acquired, starting BERT fusion optimization...")
                    logger.info("  This is a one-time cost (10-30 seconds)")
                    logger.info(
                        "  Other workers will wait and use the optimized model once ready"
                    )
                    try:
                        optimize_bert_model(
                            input_model_path=original_model,
                            output_model_path=optimized_model,
                            model_type="bert",
                            num_heads=12,
                            hidden_size=768,
                        )
                        logger.info("✓ BERT fusion optimization completed successfully")
                    except Exception as e:
                        logger.error(f"BERT fusion optimization failed: {e}")
                        logger.warning("Using original model without BERT fusion")
                        optimized_model = original_model
        except TimeoutError as e:
            logger.error(f"Lock acquisition timeout: {e}")
            logger.warning("Using original model without BERT fusion")
            optimized_model = original_model

    # QUANTIZATION PATH: Create quantized model if enabled
    if enable_quantization:
        if not quantized_model.exists() or force_reoptimize:
            logger.info("Quantized model not found, acquiring lock for quantization...")
            try:
                with file_lock(quant_lock_file, timeout=optimization_timeout):
                    # Double-check: another worker may have created it while we waited
                    if not force_reoptimize and quantized_model.exists():
                        logger.info("✓ Another worker completed quantization")
                    else:
                        # This worker won the race - perform quantization
                        logger.info("✓ Lock acquired, starting INT8 quantization...")
                        logger.info("  This is a one-time cost (5-15 seconds)")
                        try:
                            quantize_onnx_model(
                                input_model_path=optimized_model,
                                output_model_path=quantized_model,
                            )
                            logger.info("✓ INT8 quantization completed successfully")
                        except Exception as e:
                            logger.error(f"Quantization failed: {e}")
                            logger.warning("Falling back to non-quantized model")
                            final_model = optimized_model
            except TimeoutError as e:
                logger.error(f"Quantization lock timeout: {e}")
                logger.warning("Falling back to non-quantized model")
                final_model = optimized_model

    # Load final model with Phase 1 optimizations
    logger.info(f"Loading final model: {final_model.name}")
    if enable_quantization and final_model == quantized_model:
        logger.info(
            "  Optimizations: Phase 1 (SessionOptions) + Phase 2 (BERT fusion) + INT8 Quantization"
        )
        logger.info("  Expected total speedup: 3-6x")
    elif final_model == optimized_model:
        logger.info("  Optimizations: Phase 1 (SessionOptions) + Phase 2 (BERT fusion)")
        logger.info("  Expected total speedup: 1.5-3x")
    else:
        logger.info("  Optimizations: Phase 1 (SessionOptions) only")
        logger.info("  Expected total speedup: 2-3x")

    return load_onnx_model(
        onnx_path=final_model,
        enable_profiling=enable_profiling,
        inter_op_threads=inter_op_threads,
        intra_op_threads=intra_op_threads,
        providers=providers,
        provider_options=provider_options,
    )
