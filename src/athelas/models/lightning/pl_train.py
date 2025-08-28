# Save this as: bsm/lightning_models/train_utils.py
import os
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping, ModelCheckpoint, TQDMProgressBar, LearningRateMonitor, DeviceStatsMonitor
)

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy


from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType


import onnx
import onnxruntime as ort


from .pl_multimodal_cnn import MultimodalCNN
from .pl_bert_classification import TextBertClassification
from .pl_bert import TextBertBase
from .pl_tab_ae import TabAE
from .pl_lstm import TextLSTM
from .pl_multimodal_bert import MultimodalBert
from .pl_multimodal_gate_fusion import MultimodalBertGateFusion
from .pl_multimodal_moe import MultimodalBertMoE
from .pl_multimodal_cross_attn import MultimodalBertCrossAttn


def setup_logger():
    import logging
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


logger = setup_logger()


#----------------- FDSP ---------------------
def my_auto_wrap_policy(module: nn.Module, 
                        recurse: bool, 
                        unwrapped_params: int, 
                        min_num_params: int = 1e5) -> bool:
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
    return (
        isinstance(module, (TextBertBase, TabAE, nn.Linear, nn.Embedding, nn.Conv2d))
        and unwrapped_params >= min_num_params
    )

def is_fsdp_available():
    return torch.cuda.is_available() and torch.cuda.device_count() > 1 and dist.is_available() and dist.is_initialized()

strategy = FSDPStrategy(auto_wrap_policy=my_auto_wrap_policy, verbose=True) if is_fsdp_available() else "auto"
#-----------------------------

    
def model_train(
    model: pl.LightningModule,
    config: Dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: Union[int, str, List[int]] = "auto",
    model_log_path: str = './model_logs',
    early_stop_metric: str = 'val/f1_score'
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
        filename=f'{model_class}' + '-{epoch:02d}-{' + f'{early_stop_metric}' + ':.2f}',
        monitor=early_stop_metric,
        save_top_k=1,
        mode=monitor_mode,
        save_weights_only=False
    )

    earlystopping_callback = EarlyStopping(
        monitor=early_stop_metric,
        patience=early_stop_patience,
        mode=monitor_mode
    )
    
    device_stats_callback = DeviceStatsMonitor(cpu_stats=False)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger_tb,
        default_root_dir=model_log_path,
        callbacks=[
            earlystopping_callback,
            checkpoint_callback,
            device_stats_callback,
            TQDMProgressBar(refresh_rate=10),
            LearningRateMonitor(logging_interval='step')
        ],
        val_check_interval=config.get("val_check_interval", 1.0),
        sync_batchnorm=True if torch.cuda.is_available() else False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=device,
        strategy=strategy, # You might need this
        #accumulate_grad_batches=1,
        precision=16 if use_fp16 else 32
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    return trainer


#------------------ Utility Function -----------------
def extract_preds_and_labels(df: pd.DataFrame, is_binary: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    if is_binary:
        preds = torch.tensor(df["prob"].values.astype(float))
    else:
        preds = torch.tensor([ast.literal_eval(p) if isinstance(p, str) else p for p in df["prob"]])
    labels = torch.tensor(df["label"].values)
    return preds, labels


#------------------ Inference ------------------------
def model_inference(
    model: pl.LightningModule,
    dataloader: DataLoader,
    accelerator: Union[str, int, List[int]] = "auto",
    device: Union[str, int, List[int]] = "auto",
    model_log_path: str = './model_logs',
    return_dataframe: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]]:
    """
    Runs inference and returns predicted probabilities and true labels as tensors.
    Supports both binary and multiclass classification.
    
    Args:
        model (pl.LightningModule): Trained Lightning model.
        dataloader (DataLoader): DataLoader for inference.
        accelerator (str/int/List[int]): Accelerator setting.
        device (str/int/List[int]): Device setting.
        model_log_path (str): Path to save logs.
        return_dataframe (bool): Whether to return the original dataframe.
    
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
        strategy="auto",
        inference_mode=True
    )

    tester.test(model, dataloaders=dataloader)
    result_folder = model.test_output_folder
    if not result_folder or not os.path.exists(result_folder):
        raise RuntimeError(f"Expected test output folder '{result_folder}' does not exist.")

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
        y_pred = torch.tensor([ast.literal_eval(p) if isinstance(p, str) else p for p in df["prob"]])

    y_true = torch.tensor(df["label"].values).long()

    if return_dataframe:
        return y_pred, y_true, df
    else:
        return y_pred, y_true

    
def model_online_inference(model: Union[pl.LightningModule, ort.InferenceSession], dataloader: DataLoader) -> np.ndarray:
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
                    
                elif isinstance(val, list) and all(isinstance(x, (int, float)) for x in val):
                    # Fallback for list-based numeric features
                    val_np = np.array(val, dtype="float32").reshape(-1, 1)
                    input_feed[k] = val_np

                else:
                    # Skip fields like order_id (string/list[str]) or raise error
                    print(f"[Warning] Skipping unsupported ONNX input field: '{k}' ({type(val)})")

            output = model.run(None, input_feed)[0]  # Run inference
            predictions.append(output)

        return np.concatenate(predictions, axis=0)
    
    else:
        print("Running inference with PyTorch model.")
        model.eval()
        predictions = []
        for batch in dataloader:
            _, preds, _ = model.run_epoch(batch, 'pred')
            predictions.append(preds.detach().cpu().numpy())
        return np.concatenate(predictions, axis=0)



def predict_stack_transform(outputs: List[Union[torch.Tensor, Tuple[torch.Tensor]]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if isinstance(outputs[0], Tuple):
        pred_list, label_list = zip(*outputs)
        return torch.cat(pred_list), torch.cat(label_list)
    return torch.cat(outputs)


def unwrap_fsdp_model(model: nn.Module) -> nn.Module:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    return model.module if isinstance(model, FSDP) else model


def save_prediction(filename: str, y_true: List, y_pred: List):
    logger.info("Saving prediction.")
    torch.save({'y_true': y_true, 'y_pred': y_pred}, filename)


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


def save_artifacts(filename: str, config: Dict, embedding_mat: torch.Tensor, vocab: Dict[str, int], model_class: str):
    logger.info("Saving artifacts.")
    artifacts = {
        'config': config,
        'embedding_mat': embedding_mat,
        'vocab': vocab,
        'model_class': model_class,
        'torch_version': torch.__version__,
        'transformers_version': __import__('transformers').__version__,
        'pytorch_lightning_version': __import__('lightning.pytorch').__version__,
    }
    torch.save(artifacts, filename)


def load_artifacts(filename: str, device_l: str = 'cpu') -> Tuple[Dict, torch.Tensor, Dict, str]:
    logger.info("Loading artifacts.")
    artifacts = torch.load(filename, map_location=device_l)
    config = artifacts['config']
    embedding_mat = artifacts['embedding_mat']
    vocab = artifacts['vocab']
    model_class = artifacts['model_class']
    for k in ['torch_version', 'transformers_version', 'pytorch_lightning_version']:
        logger.info(f"{k}: {artifacts.get(k, 'N/A')}")
    return config, embedding_mat, vocab, model_class



def load_model(filename: str, config: Dict, embedding_mat: torch.Tensor, model_class: str = 'multimodal_bert', device_l: str = 'cpu') -> nn.Module:
    """
    Load model weights into a fresh model instance.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    logger.info("Instantiating model.")
    model = {
        'multimodal_cnn': lambda: MultimodalCNN(config, embedding_mat.shape[0], embedding_mat),
        'bert': lambda: TextBertClassification(config),
        'lstm': lambda: TextLSTM(config, embedding_mat.shape[0], embedding_mat),
        'multimodal_bert': lambda: MultimodalBert(config),
        'multimodal_gate_fusion': lambda: MultimodalBertGateFusion(config),
        'multimodal_moe': lambda: MultimodalBertMoE(config),
        'multimodal_cross_attn': lambda: MultimodalBertCrossAttn(config)
    }.get(model_class, lambda: MultimodalBert(config))()

    try:
        logger.info(f"Loading model weights from: {filename}")
        model.load_state_dict(torch.load(filename, map_location=device_l))
        logger.info("Model weights loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model weights: {e}")
        raise RuntimeError("Model loading failed.") from e

    return model


def load_checkpoint(filename: str, model_class: str = 'multimodal_bert', device_l: str = 'cpu') -> nn.Module:
    logger.info("Loading checkpoint.")
    model_fn = {
        'multimodal_cnn': MultimodalCNN,
        'bert': TextBertClassification,
        'lstm': TextLSTM,
        'multimodal_bert': MultimodalBert,
        'multimodal_gate_fusion': MultimodalBertGateFusion,
        'multimodal_moe': MultimodalBertMoE,
        'multimodal_cross_attn': MultimodalBertCrossAttn
    }.get(model_class, MultimodalBert)
    return model_fn.load_from_checkpoint(filename, map_location=device_l)



def load_onnx_model(onnx_path: Union[str, Path]) -> ort.InferenceSession:
    """
    Load an ONNX model exported by MultimodalBert.export_to_onnx and return an ONNX Runtime InferenceSession.

    Args:
        onnx_path (str or Path): Path to the ONNX model file.

    Returns:
        ort.InferenceSession: A session object that can be used to run inference.
    
    Example:
        >>> session = load_onnx_model("model.onnx")
        >>> inputs = {
        >>>     "input_ids": np.array([[101, 1024, 102]]),
        >>>     "attention_mask": np.array([[1, 1, 1]]),
        >>>     "tab_field1": np.array([[0.3, 1.5]]),
        >>>     ...
        >>> }
        >>> outputs = session.run(None, inputs)
        >>> logits = outputs[0]
    """
    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else ['CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
        logger.info(f"Successfully loaded ONNX model from {onnx_path}")
        logger.info(f"Expected ONNX model inputs: {[i.name for i in session.get_inputs()]}")
        return session
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")
