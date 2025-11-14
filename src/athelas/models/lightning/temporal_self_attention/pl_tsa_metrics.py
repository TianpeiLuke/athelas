#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.functional import (
    f1_score,
    precision,
    recall,
    auroc,
    roc,
    average_precision,
    specificity,
    accuracy,
    precision_recall_curve,
)

# TSA-specific metrics for fraud detection
SUPPORTED_TSA_METRICS = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1_score,
    "auroc": auroc,
    "pr_auc": average_precision,  # Precision-Recall AUC
    "specificity": specificity,
}


def compute_tsa_metrics(
    preds: Tensor,
    targets: Tensor,
    metric_choices: List[str],
    task: str = "binary",
    stage: str = None,
) -> Dict[str, Tensor]:
    """
    Compute TSA-specific metrics for fraud detection.
    
    Args:
        preds: Model predictions [B, n_classes] or [B] for binary
        targets: Ground truth labels [B]
        metric_choices: List of metrics to compute
        task: "binary" or "multiclass"
        stage: Stage prefix ("train", "val", "test")
        
    Returns:
        Dictionary of computed metrics
    """
    prefix = f"{stage}/" if stage else ""
    metrics = {}
    
    # Convert predictions to probabilities if needed
    if task == "binary":
        if preds.dim() > 1:
            # If logits provided, convert to probabilities
            preds = torch.softmax(preds, dim=-1)[:, 1]  # Positive class probability
    else:
        if preds.dim() > 1:
            preds = torch.softmax(preds, dim=-1)
    
    # Compute each requested metric
    for metric_name in metric_choices:
        if metric_name not in SUPPORTED_TSA_METRICS:
            print(f"Warning: Metric '{metric_name}' not supported. Skipping.")
            continue
            
        metric_fn = SUPPORTED_TSA_METRICS[metric_name]
        
        try:
            if metric_name == "pr_auc":
                # Use average_precision for PR-AUC
                value = metric_fn(preds, targets, task=task)
            elif metric_name in ["accuracy", "precision", "recall", "f1_score", "auroc", "specificity"]:
                value = metric_fn(preds, targets, task=task)
            else:
                value = metric_fn(preds, targets)
            
            metrics[f"{prefix}{metric_name}"] = value
            
        except Exception as e:
            print(f"Warning: Failed to compute {metric_name}: {e}")
            metrics[f"{prefix}{metric_name}"] = torch.tensor(0.0)
    
    return metrics


def generate_tsa_score_files(
    preds: Tensor,
    targets: Tensor,
    output_dir: str,
    stage: str = "test",
    ids: Optional[List] = None
):
    """
    Generate TSA-compatible score files for deployment.
    
    Creates:
    - score_file.csv: Predictions with percentile mapping
    - tag_file.csv: Ground truth labels
    - score_distribution.png: Score distribution analysis
    
    Args:
        preds: Model predictions [B]
        targets: Ground truth labels [B]
        output_dir: Output directory path
        stage: Stage name ("test", "val", etc.)
        ids: Optional list of sample IDs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to numpy for processing
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # Generate percentiles for score mapping
    percentiles = np.percentile(preds_np, np.arange(0, 101, 1))
    
    # Create score file
    score_data = {
        'score': preds_np,
        'percentile': [
            np.searchsorted(percentiles, score, side='right') - 1
            for score in preds_np
        ]
    }
    
    if ids is not None:
        score_data['id'] = ids
    
    score_df = pd.DataFrame(score_data)
    score_file_path = os.path.join(output_dir, f"{stage}_score_file.csv")
    score_df.to_csv(score_file_path, index=False)
    
    # Create tag file
    tag_data = {'label': targets_np}
    if ids is not None:
        tag_data['id'] = ids
        
    tag_df = pd.DataFrame(tag_data)
    tag_file_path = os.path.join(output_dir, f"{stage}_tag_file.csv")
    tag_df.to_csv(tag_file_path, index=False)
    
    # Generate score distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(preds_np[targets_np == 0], bins=50, alpha=0.7, label='Non-Fraud', density=True)
    plt.hist(preds_np[targets_np == 1], bins=50, alpha=0.7, label='Fraud', density=True)
    plt.xlabel('Fraud Score')
    plt.ylabel('Density')
    plt.title('TSA Score Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    dist_plot_path = os.path.join(output_dir, f"{stage}_score_distribution.png")
    plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"TSA score files generated:")
    print(f"  - Score file: {score_file_path}")
    print(f"  - Tag file: {tag_file_path}")
    print(f"  - Distribution plot: {dist_plot_path}")


def plot_tsa_performance_curves(
    y_pred: Tensor,
    y_true: Tensor,
    y_val_pred: Tensor,
    y_val_true: Tensor,
    output_path: str,
    writer: Optional[SummaryWriter] = None,
    global_step: int = 0
):
    """
    Generate TSA-specific ROC and PR curves for fraud detection analysis.
    
    Features:
    - Binary classification ROC/PR curves
    - Test vs validation comparison
    - TensorBoard integration
    - Score distribution analysis
    - Fraud detection specific visualizations
    
    Args:
        y_pred: Test predictions [B]
        y_true: Test labels [B]
        y_val_pred: Validation predictions [B]
        y_val_true: Validation labels [B]
        output_path: Output directory path
        writer: Optional TensorBoard writer
        global_step: Global training step
    """
    
    # Ensure tensors are on CPU
    y_pred = y_pred.detach().cpu()
    y_true = y_true.detach().cpu()
    y_val_pred = y_val_pred.detach().cpu()
    y_val_true = y_val_true.detach().cpu()
    
    # ROC Curve
    fpr_test, tpr_test, _ = roc(y_pred, y_true, task="binary")
    fpr_val, tpr_val, _ = roc(y_val_pred, y_val_true, task="binary")
    auc_test = auroc(y_pred, y_true, task="binary")
    auc_val = auroc(y_val_pred, y_val_true, task="binary")
    
    plt.figure(figsize=(15, 5))
    
    # ROC subplot
    plt.subplot(1, 3, 1)
    plt.plot(fpr_test, tpr_test, color='red', lw=2, label=f'Test AUC = {auc_test:.3f}')
    plt.plot(fpr_val, tpr_val, color='blue', lw=2, label=f'Val AUC = {auc_val:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('TSA ROC Curve (Fraud Detection)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # PR Curve
    precision_test, recall_test, _ = precision_recall_curve(y_pred, y_true, task="binary")
    precision_val, recall_val, _ = precision_recall_curve(y_val_pred, y_val_true, task="binary")
    ap_test = average_precision(y_pred, y_true, task="binary")
    ap_val = average_precision(y_val_pred, y_val_true, task="binary")
    
    plt.subplot(1, 3, 2)
    plt.plot(recall_test, precision_test, color='red', lw=2, label=f'Test AP = {ap_test:.3f}')
    plt.plot(recall_val, precision_val, color='blue', lw=2, label=f'Val AP = {ap_val:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('TSA Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Score Distribution
    plt.subplot(1, 3, 3)
    plt.hist(y_pred[y_true == 0], bins=30, alpha=0.7, label='Non-Fraud', density=True)
    plt.hist(y_pred[y_true == 1], bins=30, alpha=0.7, label='Fraud', density=True)
    plt.xlabel('Fraud Score')
    plt.ylabel('Density')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_path, exist_ok=True)
    plot_path = os.path.join(output_path, "tsa_performance_curves.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # Log to TensorBoard if available
    if writer:
        writer.add_figure("TSA/Performance_Curves", plt.gcf(), global_step=global_step)
    
    plt.close()
    
    print(f"TSA performance curves saved: {plot_path}")
    
    # Print summary metrics
    print(f"\nTSA Performance Summary:")
    print(f"Test AUC: {auc_test:.4f}, Val AUC: {auc_val:.4f}")
    print(f"Test AP: {ap_test:.4f}, Val AP: {ap_val:.4f}")


def analyze_gate_scores(
    gate_scores: Tensor,
    output_path: str,
    stage: str = "test"
):
    """
    Analyze gate function scores for dual-sequence TSA models.
    
    Args:
        gate_scores: Gate scores [B, 2] (CID, CCID importance)
        output_path: Output directory path
        stage: Stage name
    """
    os.makedirs(output_path, exist_ok=True)
    
    gate_scores_np = gate_scores.detach().cpu().numpy()
    
    plt.figure(figsize=(12, 4))
    
    # Gate score distribution
    plt.subplot(1, 3, 1)
    plt.hist(gate_scores_np[:, 0], bins=30, alpha=0.7, label='CID Importance')
    plt.hist(gate_scores_np[:, 1], bins=30, alpha=0.7, label='CCID Importance')
    plt.xlabel('Gate Score')
    plt.ylabel('Frequency')
    plt.title('Gate Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gate score scatter
    plt.subplot(1, 3, 2)
    plt.scatter(gate_scores_np[:, 0], gate_scores_np[:, 1], alpha=0.5)
    plt.xlabel('CID Gate Score')
    plt.ylabel('CCID Gate Score')
    plt.title('CID vs CCID Gate Scores')
    plt.grid(True, alpha=0.3)
    
    # Gate dominance
    plt.subplot(1, 3, 3)
    dominance = np.argmax(gate_scores_np, axis=1)
    unique, counts = np.unique(dominance, return_counts=True)
    labels = ['CID Dominant', 'CCID Dominant']
    plt.pie(counts, labels=[labels[i] for i in unique], autopct='%1.1f%%')
    plt.title('Sequence Dominance')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_path, f"{stage}_gate_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save gate statistics
    stats = {
        'cid_mean': float(gate_scores_np[:, 0].mean()),
        'cid_std': float(gate_scores_np[:, 0].std()),
        'ccid_mean': float(gate_scores_np[:, 1].mean()),
        'ccid_std': float(gate_scores_np[:, 1].std()),
        'cid_dominant_pct': float((dominance == 0).mean() * 100),
        'ccid_dominant_pct': float((dominance == 1).mean() * 100),
    }
    
    stats_path = os.path.join(output_path, f"{stage}_gate_stats.json")
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Gate analysis saved:")
    print(f"  - Plot: {plot_path}")
    print(f"  - Stats: {stats_path}")
    print(f"  - CID dominant: {stats['cid_dominant_pct']:.1f}%")
    print(f"  - CCID dominant: {stats['ccid_dominant_pct']:.1f}%")
