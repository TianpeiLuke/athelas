from scipy.special import expit
import pandas as pd
import numpy as np
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    accuracy_score,
    f1_score,
    roc_curve,
    precision_score,
    recall_score,
)


class base_loss(object):
    """
    This is the default loss object with fixed weight provided by MTGBM

    Parameters
    ----------
        val_sublabel_idx: validation set indices dictionary for each task

        num_label: total number of tasks

        eval_mat: evaluation results to be stored during training

        w: user-chosen fixed weight vector
    """

    def __init__(self, val_sublabel_idx, num_label):
        self.val_label_idx = val_sublabel_idx  # type: dict
        self.eval_mat = []
        beta = 0.2
        self.w = np.array(
            [1, 0.1 * beta, 0.1 * beta, 0.1 * beta, 0.1 * beta, 0.1 * beta]
        )
        self.num_label = num_label

    def base_eval(self, preds, train_data):
        """
        Evaluation function to be passed to the MTGBM package

        Parameters
        ----------
            preds: prediction score

            train_data: training labels
        """
        labels_mat = train_data.get_label().reshape((self.num_label, -1)).transpose()
        preds_mat = preds.reshape((self.num_label, -1)).transpose()
        preds_mat = expit(preds_mat)
        preds_mat = np.clip(preds_mat, 1e-15, 1 - 1e-15)

        curr_score = []
        for j in range(self.num_label):
            s = roc_auc_score(
                labels_mat[self.val_label_idx[j], j],
                preds_mat[self.val_label_idx[j], j],
            )
            curr_score.append(s)
        self.eval_mat.append(curr_score)
        weighted_score_vec = curr_score * self.w
        # flip the positivity of wavg_auc to make sure early_stopping will work
        wavg_auc = 0 - np.sum(weighted_score_vec) / np.sum(self.w)
        return "base_metric", wavg_auc, False

    def base_obj(self, preds, train_data, ep=None):
        """
        Objective function to be passed to the MTGBM package

        Parameters
        ----------
            preds: prediction score

            train_data: training labels

            ep: adjusted esplon, default is None
        """
        labels_mat = train_data.get_label().reshape((self.num_label, -1)).transpose()
        preds_mat = expit(preds.reshape((self.num_label, -1)).transpose())
        preds_mat = np.clip(preds_mat, 1e-15, 1 - 1e-15)
        # gradients: G_i
        grad_i = preds_mat - labels_mat
        # hessians: H_i
        hess_i = preds_mat * (1.0 - preds_mat)
        # ensemble G and H
        grad_n = grad_i * np.array(self.w)
        grad = np.sum(grad_n, axis=1)  # G_e in Algorithm 2
        hess = np.sum((hess_i) * np.array(self.w), axis=1)  # H_e in Algorithm 2
        return grad, hess, grad_i, hess_i
