from scipy.special import expit, rel_entr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import jaccard_score
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


class custom_loss_KDswap(object):
    """
    This is the customized loss object with our proposed weight but NO KD loss implementation

    Parameters
    ----------
        num_label: total number of tasks

        trn_sublabel_idx: training set indices dictionary for each task

        val_sublabel_idx: validation set indices dictionay for each task

        w_trn_mat: keep tract of weights changing during training process

        eval_mat: evaluation results to be stored during training

        pat: patience parameter in KD loss for detecting best performance

        curr_obj_round: current round when package calling objective function, start with 0

        curr_eval_round: current round when package calling evaluation function, start with 0

        max_score: the best model performance during evaluation

        counter: counting how many times the performance are decreasing consecutively

        best_pred: predicted score at the best model performance

        pre_pred: predicted score from previous round

        weight_method: four different methods of weights changing: "None", "tenIters", "sqrt", "delta"
    """

    def __init__(
        self,
        num_label,
        val_sublabel_idx,
        trn_sublabel_idx,
        patience,
        weight_method=None,
    ):
        self.num_col = num_label
        self.val_label_idx = val_sublabel_idx
        self.trn_sublabel_idx = trn_sublabel_idx
        self.eval_mat = []
        self.w_trn_mat = []
        self.pat = patience
        self.curr_obj_round = 0
        self.curr_eval_round = 0
        self.max_score = {}
        self.counter = np.zeros(num_label, dtype=int)
        self.replaced = np.repeat(False, num_label)
        self.best_pred = {}
        self.pre_pred = {}
        self.weight_method = weight_method

    def self_obj(self, preds, train_data, ep):
        """
        Objective function to be passed to the MTGBM package

        Parameters
        ----------
            preds: prediction score

            train_data: training labels

            ep: adjusted esplon, default is None
        """
        self.curr_obj_round += 1
        labels_mat = train_data.get_label().reshape((self.num_col, -1)).transpose()
        preds_mat = expit(preds.reshape((self.num_col, -1)).transpose())
        preds_mat = np.clip(preds_mat, 1e-15, 1 - 1e-15)

        for j in range(self.num_col):
            if j in self.max_score:
                best_round = self.max_score[j][0]  # max evaluation round
                if self.curr_obj_round == best_round + 1:
                    self.best_pred[j] = self.pre_pred[j]

            if self.counter[j] == self.pat and self.replaced[j] == False:
                labels_mat[:, j] = self.best_pred[j]
                self.replaced[j] = True
                print(
                    "!TASK ",
                    j,
                    " replaced,",
                    "curr_round: ",
                    self.curr_obj_round,
                    " check counter: ",
                    self.counter[j],
                )
                self.counter[j] = 0
            self.pre_pred[j] = preds_mat[:, j]

        # gradients: G_i
        grad_i = self.grad(labels_mat, preds_mat)
        # hessians: H_i
        hess_i = self.hess(preds_mat)
        if self.weight_method == "tenIters":
            # update weight every 10 iters
            i = self.curr_obj_round - 1
            if i % 50 == 0:
                self.similar = self.similarity_vec(
                    labels_mat[:, 0],
                    preds_mat,
                    self.num_col,
                    self.trn_sublabel_idx,
                    0.1,
                )
            w = self.similar
            self.w_trn_mat.append(w)
        elif self.weight_method == "sqrt":
            # sqrt weight
            w = self.similarity_vec(
                labels_mat[:, 0], preds_mat, self.num_col, self.trn_sublabel_idx, 0.5
            )
            w = np.sqrt(w)
            self.w_trn_mat.append(w)
        elif self.weight_method == "delta":
            # delta weight
            simi = self.similarity_vec(
                labels_mat[:, 0], preds_mat, self.num_col, self.trn_sublabel_idx, 0.1
            )
            self.similar.append(simi)
            if self.curr_obj_round == 1:
                w = self.similar[0]
            else:
                i = self.curr_obj_round - 1
                diff = self.similar[i] - self.similar[i - 1]
                w = self.w_trn_mat[i - 1] + diff * 0.01
            self.w_trn_mat.append(w)
        else:
            # only adj weight
            w = self.similarity_vec(
                labels_mat[:, 0], preds_mat, self.num_col, self.trn_sublabel_idx, 0.1
            )
            self.w_trn_mat.append(w)

        # ensemble G and H
        grad_n = self.normalize(grad_i)
        grad = np.sum(grad_n * np.array(w), axis=1)  # G_e in Algorithm 2
        hess = np.sum(hess_i * np.array(w), axis=1)  # H_e in Algorithm 2

        return grad, hess, grad_i, hess_i

    def self_eval(self, preds, train_data):
        """
        Evaluation function to be passed to the MTGBM package

        Parameters
        ----------
            preds: prediction score

            train_data: training labels
        """
        self.curr_eval_round += 1
        labels_mat = train_data.get_label().reshape((self.num_col, -1)).transpose()
        preds_mat = preds.reshape((self.num_col, -1)).transpose()
        preds_mat = expit(preds_mat)
        preds_mat = np.clip(preds_mat, 1e-15, 1 - 1e-15)

        w = self.w_trn_mat[self.curr_eval_round - 1]

        curr_score = []
        for j in range(self.num_col):
            s = roc_auc_score(
                labels_mat[self.val_label_idx[j], j],
                preds_mat[self.val_label_idx[j], j],
            )
            curr_score.append(s)

        self.eval_mat.append(curr_score)
        print("--- task eval score: ", np.round(curr_score, 4))

        for j in range(self.num_col):
            if not self.replaced[j]:
                if self.curr_eval_round == 1:
                    self.max_score[j] = [self.curr_eval_round, curr_score[j]]
                else:
                    if curr_score[j] >= self.max_score[j][1]:
                        self.max_score[j] = [self.curr_eval_round, curr_score[j]]
                        self.counter[j] = 0
                    else:
                        self.counter[j] += 1

        weighted_score_vec = curr_score * w
        # flip the positivity of wavg_auc to make sure early_stopping will work
        wavg_auc = 0 - np.sum(weighted_score_vec) / np.sum(w)
        print("--- self_eval score: ", np.round(wavg_auc, 4))
        return "self_eval", wavg_auc, False

    def similarity_vec(self, main_label, sub_predmat, num_col, ind_dic, lr):
        """
        Calculate similarity between subtask and main task by inverse JS divergence

        Parameters
        ----------
            main_label: true main task label

            sub_predmat: subtasks prediction matrix

            num_col: total number of labels

            ind_dic: subtask indices dictionary

            lr: learning rate for scaling down weights
        """
        dis = []
        for j in range(1, num_col):
            dis.append(
                jensenshannon(main_label[ind_dic[j]], sub_predmat[ind_dic[j], j])
            )
        dis_norm = self.unit_scale(np.reciprocal(dis)) * lr
        w = np.insert(dis_norm, 0, 1)
        return w

    def grad(self, y_true, y_pred):
        """
        Calculate gradients
        """
        grad = y_pred - y_true
        return grad

    def hess(self, y_pred):
        """
        Calculate hessian values
        """
        hess = y_pred * (1.0 - y_pred)
        return hess

    def normalize(self, vec):
        """
        Standard normalize
        """
        norm_vec = (vec - np.mean(vec, axis=0)) / np.std(vec, axis=0)
        return norm_vec

    def unit_scale(self, vec):
        """
        l2 standardizing into a scale of (0,1)
        """
        return vec / np.linalg.norm(vec)
