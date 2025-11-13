import time
import math
import torch
import numpy as np
from torch import nn
from focalloss import *
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60

    return "%dm %ds" % (m, s)


def plot_PR_curve(model_names, scores, y, fig_name):
    plt.figure(dpi=400)
    for model_name in model_names:
        preds = scores[model_name]
        precision, recall, thresholds = metrics.precision_recall_curve(
            y, preds, pos_label=1
        )
        pr_auc = metrics.auc(recall, precision)

        plt.title("PR-AUC")
        plt.plot(
            recall,
            precision,
            linewidth=0.5,
            label="{} {}".format(model_name, np.round(pr_auc, 4)),
        )
        plt.legend(fontsize=12)
        plt.plot([0, 1], [1, 0], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.savefig(fig_name + ".png")


def plot_PR_curve_percentage(model_names, scores, y, fig_name, percentile):
    plt.figure(dpi=400)
    tmp_y = y
    for model_name in model_names:
        cutoff = np.percentile(scores[model_name], percentile)
        tmp_score = scores[model_name]
        preds = tmp_score[tmp_score > cutoff]
        y = tmp_y[tmp_score > cutoff]
        precision, recall, thresholds = metrics.precision_recall_curve(
            y, preds, pos_label=1
        )
        pr_auc = metrics.auc(recall, precision)

        plt.title("PR-AUC - Operating Percentage " + str(100 - percentile) + "%")
        plt.plot(
            recall,
            precision,
            linewidth=0.5,
            label="{} {}".format(model_name, np.round(pr_auc, 4)),
        )
        plt.legend(fontsize=12)
        plt.plot([0, 1], [1, 0], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.savefig(fig_name + "Operating" + str(percentile) + ".png")


def plot_PR_curve_sub(model_names, scores, y, fig_name, precision_level):
    plt.figure(dpi=400)
    for model_name in model_names:
        preds = scores[model_name]
        precision, recall, thresholds = metrics.precision_recall_curve(
            y, preds, pos_label=1
        )
        pr_auc = metrics.auc(recall, precision)
        precision_ops = precision[precision > precision_level]
        recall_ops = recall[-len(precision_ops) :]
        pr_auc_ops = metrics.auc(recall_ops, precision_ops)

        plt.title("PR-AUC for Precision > " + str(precision_level))
        plt.plot(
            recall,
            precision,
            linewidth=0.5,
            label="{} {}".format(model_name, np.round(pr_auc_ops, 4)),
        )
        plt.legend(fontsize=12)
        plt.plot([0, 1], [1, 0], "r--")
        plt.xlim([0, 1])
        plt.ylim([precision_level, 1])
        plt.ylabel("Precision")
        plt.xlabel("Recall")
        plt.savefig(fig_name + "precision" + str(precision_level) + ".png")


def plot_ROC_curve(model_names, scores, y, fig_name):
    plt.figure(dpi=400)
    for model_name in model_names:
        preds = scores[model_name]
        fpr, tpr, thresholds = metrics.roc_curve(y, preds, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)

        plt.title("ROC-AUC")
        plt.plot(
            fpr,
            tpr,
            linewidth=0.5,
            label="{} {}".format(model_name, np.round(roc_auc, 4)),
        )
        plt.legend(fontsize=12)
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.savefig(fig_name + ".png")


def plot_ROC_curve_sub(model_names, scores, y, fig_name, ops_level):
    plt.figure(dpi=400)
    for model_name in model_names:
        preds = scores[model_name]
        fpr, tpr, thresholds = metrics.roc_curve(y, preds, pos_label=1)
        fpr_ops = fpr[fpr < ops_level]
        tpr_ops = tpr[: len(fpr_ops)]

        roc_auc = metrics.auc(fpr_ops, tpr_ops)

        plt.title("ROC-AUC for FPR < " + str(ops_level))
        plt.plot(
            fpr,
            tpr,
            linewidth=0.5,
            label="{} {}".format(model_name, np.round(roc_auc, 4)),
        )
        plt.legend(fontsize=12)
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, ops_level])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        #         print(fpr_ops[-1],tpr_ops[-1])
        plt.savefig(fig_name + "FPR" + str(ops_level) + ".png")


def plot_ROC_curve_percentile(model_names, scores, y, fig_name, percentile):
    plt.figure(dpi=400)
    tmp_y = y
    for model_name in model_names:
        cutoff = np.percentile(scores[model_name], percentile)
        tmp_score = scores[model_name]
        preds = tmp_score[tmp_score > cutoff]
        y = tmp_y[tmp_score > cutoff]
        fpr, tpr, thresholds = metrics.roc_curve(y, preds, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)

        plt.title("ROC AUC - Operating Percentage " + str(100 - percentile) + "%")
        plt.plot(
            fpr,
            tpr,
            linewidth=0.5,
            label="{} {}".format(model_name, np.round(roc_auc, 4)),
        )
        plt.legend(fontsize=12)
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.savefig(fig_name + "Operating" + str(percentile) + ".png")


def get_operating_points(preds, y, percentile):
    operating_points = {}
    cutoff = np.percentile(preds, percentile)
    fpr, tpr, thresholds = metrics.roc_curve(y, preds, pos_label=1)
    operating_points["fpr"] = fpr[thresholds > cutoff][-1]
    operating_points["tpr"] = tpr[thresholds > cutoff][-1]

    precision = metrics.precision_score(y, preds > cutoff)
    recall = metrics.recall_score(y, preds > cutoff)
    operating_points["precision"] = precision
    operating_points["recall"] = recall

    return operating_points


def get_operating_points_scorevalue(preds, y, cutoff):
    operating_points = {}
    #     cutoff=np.percentile(preds,percentile)
    fpr, tpr, thresholds = metrics.roc_curve(y, preds, pos_label=1)
    operating_points["fpr"] = fpr[thresholds > cutoff][-1]
    operating_points["tpr"] = tpr[thresholds > cutoff][-1]

    precision = metrics.precision_score(y, preds > cutoff)
    recall = metrics.recall_score(y, preds > cutoff)
    operating_points["precision"] = precision
    operating_points["recall"] = recall

    return operating_points
