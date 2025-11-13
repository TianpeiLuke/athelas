import pandas as pd
import numpy as np
import glob
import os
import time

import matplotlib.pyplot as plt
import math
import lightgbm as lgb
import random
import seaborn as sns
import pickle

from math import log2, log10
from sklearn import preprocessing
from sklearn.metrics import (
    log_loss,
    roc_auc_score,
    accuracy_score,
    f1_score,
    roc_curve,
    precision_score,
    recall_score,
)
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    auc,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split

seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
from collections import Counter
from util import *


class MultiShareTrees:
    """
    A shared multi-tree chain structure

    Parameters
    ----------
    train, test: train data, test data

    train_labels, test_labels: all tasks labels under train and test respectively

    config: hyper parameter setting

    idx_trn_dic: a dict of indices for each payment method in training data

    idx_test_dic: a dict of indices for each payment method in test data

    X_trn, X_ts: training, test set in numpy array

    y_trn, y_ts: training, test labels in numpy array

    num_label: total number of tasks
    """

    def __init__(self, train, train_labels, test, test_labels, config):
        self.train = train
        self.train_labels = train_labels
        self.test = test
        self.test_labels = test_labels
        self.config = from_json(config)

        self.idx_trn_dic = filter_idx(self.train_labels)
        self.idx_test_dic = filter_idx(self.test_labels)
        self.X_trn = None
        self.y_trn = None
        self.X_ts = None
        self.y_ts = None
        self.params = None
        self.num_label = None

        self.build()

    def build(self):
        """
        Initialzation
        """
        targets = ["isFraud", "isCCfrd", "isDDfrd", "isGCfrd", "isLOCfrd", "isCimfrd"]
        y_train = self.train_labels[targets].copy().reset_index(drop=True)
        y_test = self.test_labels[targets].copy().reset_index(drop=True)
        print("train_label shape: ", y_train.shape, "test_label shape: ", y_test.shape)

        X_train = (
            self.train.drop(["paymeth", "isFraud"], axis=1)
            .copy()
            .reset_index(drop=True)
        )
        X_test = (
            self.test.drop(["paymeth", "isFraud"], axis=1).copy().reset_index(drop=True)
        )
        print(
            "original train_data shape: ",
            X_train.shape,
            "original test_data shape: ",
            X_test.shape,
        )

        self.X_trn = np.array(X_train.copy())
        self.y_trn = np.array(y_train.copy())
        self.X_ts = np.array(X_test.copy())
        self.y_ts = np.array(y_test.copy())

        self.params = {
            "metric": ["binary_logloss", "auc"],
            "max_depth": self.config.max_depth,
            "learning_rate": self.config.learning_rate,
            "bagging_fraction": self.config.bagging_fraction,
            "feature_fraction": self.config.feature_fraction,
            "verbose": self.config.verbose,
            "lambda_l1": self.config.lambda_l1,
            "lambda_l2": self.config.lambda_l2,
            "num_leaves": self.config.num_leaves,
            "min_child_weight": self.config.min_child_weight,
            "min_data_in_leaf": self.config.min_data_in_leaf,
            "num_threads": self.config.num_threads,
            "metric_freq": self.config.metric_freq,
            "data_random_seed": self.config.data_random_seed,
            "objective": "binary",
            "early_stopping_round": self.config.early_stopping_round,
            "is_provide_training_metric": True,
        }
        self.num_label = self.y_trn.shape[1]

    def data_initializer(self, X, y, init_type=None, base_preds=None):
        """
        A function for creating extra initializers for all tasks

        Parameters
        ----------
        X: original features

        y: data labels

        init_type: different ways of initializing:
                   None: all cols are initialized as random uniform(0,1) values
                   1: all cols are initalized using baseline prediction
                   2: the first task is from unif(0,1), the rest are from baseline

        base_preds: prediction from baseline results
        """
        num_label = y.shape[1]
        cols = X.shape[1]
        rows = X.shape[0]
        base_preds = np.array(base_preds)

        if init_type == 1:
            X_trn_new = np.concatenate((X, base_preds), axis=1)
        elif init_type == 2:
            init_main = np.random.uniform(0, 1, rows).reshape((rows, 1))
            sub_pred = np.delete(base_preds, 0, axis=1)
            pred_init_mat = np.concatenate((init_main, sub_pred), axis=1)
            X_trn_new = np.concatenate((X, pred_init_mat), axis=1)
        else:
            pred_init_mat = np.random.uniform(0, 1, rows * num_label).reshape(
                (rows, num_label)
            )
            X_trn_new = np.concatenate((X, pred_init_mat), axis=1)
        return X_trn_new

    def multi_shared_trees(
        self, epochs, num_boost, path, init_type=None, base_preds=None
    ):
        """
        Scenario 1: all tasks share information

        Parameters
        ----------
        epochs: number of epochs

        num_boost: total boosting round / epochs

        path: a file path where the serial model can be saved

        init_type: choosing initializers

        base_preds: baseline prediction if possible
        """
        X_trn_new = self.data_initializer(self.X_trn, self.y_trn, init_type, base_preds)
        cols = self.X_trn.shape[1]

        auc_mat = np.zeros((epochs, self.y_trn.shape[1]))
        logloss = np.zeros((epochs, self.y_trn.shape[1]))
        mod_pre = {}
        # dtrain = [0]*num_label
        ypred = np.zeros((X_trn_new.shape[0], self.num_label))
        # ypred_ts = np.zeros((X_ts_new.shape[0], num_label))
        evals = {}
        dpred_trn = {}
        # dpred_ts_1000 = {}

        for i in range(epochs):
            start_time = time.time()

            for j in range(self.num_label):
                trn = np.delete(X_trn_new[self.idx_trn_dic[j]], cols + j, 1)
                trn_label = self.y_trn[self.idx_trn_dic[j], j]
                X_tr, X_vl, y_tr, y_vl = train_test_split(
                    trn, trn_label, test_size=0.3, random_state=seed
                )
                d_train = lgb.Dataset(X_tr, label=y_tr)
                d_valid = lgb.Dataset(X_vl, label=y_vl)

                evals[i, j] = {}
                callback = [lgb.log_evaluation(50), lgb.record_evaluation(evals[i, j])]
                if i == 0:
                    mod = lgb.train(
                        self.params,
                        num_boost_round=num_boost,
                        train_set=d_train,
                        valid_sets=[d_valid],
                        keep_training_booster=True,
                        callbacks=callback,
                    )
                else:
                    mod = lgb.train(
                        self.params,
                        num_boost_round=num_boost,
                        train_set=d_train,
                        valid_sets=[d_valid],
                        keep_training_booster=True,
                        callbacks=callback,
                        init_model=mod_pre[j],
                    )
                mod_pre[j] = mod
                mod.save_model(path + "/model" + str(i) + "_" + str(j) + ".txt")
                trn_all = np.delete(X_trn_new, cols + j, 1)
                pred = mod.predict(trn_all)  # _Booster__inner_predict
                auc = roc_auc_score(trn_label, pred[self.idx_trn_dic[j]])
                loss = log_loss(trn_label, pred[self.idx_trn_dic[j]])
                ypred[:, j] = pred
                auc_mat[i, j] = auc
                logloss[i, j] = loss

            print("------- iter:", i)
            print("auc", auc_mat[i, :])
            #         print("test auc", auc_mat_ts[i,:])
            print("logloss", logloss[i, :])
            #         print("test logloss", logloss_ts[i,:])
            print(
                "--- iter ",
                i,
                " took %.2f mins ---" % ((time.time() - start_time) / 60),
            )
            X_trn_new = np.concatenate((self.X_trn, ypred), axis=1)
            dpred_trn[i] = ypred
        #         X_ts_new = np.concatenate((X_ts, ypred_ts), axis = 1)
        #         dpred_ts_1000[i] = ypred_ts
        return evals, dpred_trn, auc_mat, logloss, mod_pre

    def alltrees_test(self, path, epochs, init_type=None, base_preds=None):
        """
        Test steps for Scenario 1:

        Parameters
        ----------
        path: same file path where a serial model from training is saved

        epochs: number of epochs, should be same as training epochs

        init_type: choosing initializers, should be same as training method

        base_preds: baseline prediction for test data if possible
        """
        cols = self.X_ts.shape[1]
        X_test_new = self.data_initializer(self.X_ts, self.y_ts, init_type, base_preds)
        final = np.zeros((self.X_ts.shape[0], self.num_label))
        for i in range(epochs):
            for j in range(self.num_label):
                dat = np.delete(X_test_new, cols + j, 1)
                file = path + "model" + str(i) + "_" + str(j) + ".txt"
                clf = lgb.Booster(model_file=file)
                final[:, j] = clf.predict(dat)
            X_test_new = np.concatenate((self.X_ts, final), axis=1)

        auc = []
        for j in range(self.num_label):
            auc.append(
                roc_auc_score(
                    self.y_ts[self.idx_test_dic[j], j], final[self.idx_test_dic[j], j]
                )
            )
        print("test_AUC_taskwise: ", np.round(auc, 4))

        return final

    def multi_shared_subtrees(
        self, epochs, num_boost, path, init_type=None, base_preds=None
    ):
        """
        Scenario 2: only subtasks share information

        Parameters
        ----------
        epochs: number of epochs

        num_boost: total boosting round / epochs

        path: a file path where the serial model can be saved

        init_type: choosing initializers

        base_preds: baseline prediction if possible
        """
        X_trn_new = self.data_initializer(self.X_trn, self.y_trn, init_type, base_preds)
        cols = self.X_trn.shape[1]

        auc_mat = np.zeros((epochs, self.y_trn.shape[1]))
        logloss = np.zeros((epochs, self.y_trn.shape[1]))
        ypred = np.zeros((X_trn_new.shape[0], self.num_label))
        evals = {}
        mod_pre = {}
        pred_list = {}

        for i in range(epochs):
            start_time = time.time()

            for j in range(self.num_label):
                if j == 0:
                    trn = self.X_trn
                    trn_label = self.y_trn[self.idx_trn_dic[j], j]
                    trn_all = self.X_trn
                else:
                    trn = np.delete(X_trn_new[self.idx_trn_dic[j]], [cols, cols + j], 1)
                    trn_label = self.y_trn[self.idx_trn_dic[j], j]
                    trn_all = np.delete(X_trn_new, [cols, cols + j], 1)

                X_tr, X_vl, y_tr, y_vl = train_test_split(
                    trn, trn_label, test_size=0.3, random_state=seed
                )
                d_train = lgb.Dataset(X_tr, label=y_tr)
                d_valid = lgb.Dataset(X_vl, label=y_vl)

                evals[i, j] = {}
                callback = [lgb.log_evaluation(50), lgb.record_evaluation(evals[i, j])]
                if i == 0:
                    mod = lgb.train(
                        self.params,
                        num_boost_round=num_boost,
                        train_set=d_train,
                        valid_sets=[d_valid],
                        keep_training_booster=True,
                        callbacks=callback,
                    )
                else:
                    mod = lgb.train(
                        self.params,
                        num_boost_round=num_boost,
                        train_set=d_train,
                        valid_sets=[d_valid],
                        keep_training_booster=True,
                        callbacks=callback,
                        init_model=mod_pre[j],
                    )
                mod_pre[j] = mod
                mod.save_model(path + "/model" + str(i) + "_" + str(j) + ".txt")
                pred = mod.predict(trn_all)  # _Booster__inner_predict
                auc = roc_auc_score(trn_label, pred[self.idx_trn_dic[j]])
                loss = log_loss(trn_label, pred[self.idx_trn_dic[j]])
                ypred[:, j] = pred
                auc_mat[i, j] = auc
                logloss[i, j] = loss
            print("------- iter:", i)
            print("auc", auc_mat[i, :])
            #         print("test auc", auc_mat_ts[i,:])
            print("logloss", logloss[i, :])
            #         print("test logloss", logloss_ts[i,:])
            print(
                "--- iter ",
                i,
                " took %.2f mins ---" % ((time.time() - start_time) / 60),
            )
            X_trn_new = np.concatenate((self.X_trn, ypred), axis=1)
            pred_list[i] = ypred

        return evals, pred_list, auc_mat, logloss, mod_pre

    def subtrees_test(self, path, epochs, init_type=None, base_preds=None):
        """
        Test steps for Scenario 2:

        Parameters
        ----------
        path: same file path where a serial model from training is saved

        epochs: number of epochs, should be same as training epochs

        init_type: choosing initializers, should be same as training method

        base_preds: baseline prediction for test data if possible
        """
        cols = self.X_ts.shape[1]
        X_test_new = self.data_initializer(self.X_ts, self.y_ts, init_type, base_preds)
        # initial final_pred matrix
        final = np.zeros((self.X_ts.shape[0], self.num_label))
        # test prediction
        for i in range(epochs):
            for j in range(self.num_label):
                if j == 0:
                    dat = self.X_ts
                else:
                    dat = np.delete(X_test_new, [cols, cols + j], 1)
                file = path + "model" + str(i) + "_" + str(j) + ".txt"
                clf = lgb.Booster(model_file=file)
                final[:, j] = clf.predict(dat)
            X_test_new = np.concatenate((self.X_ts, final), axis=1)

        auc = []
        for j in range(self.num_label):
            auc.append(
                roc_auc_score(
                    self.y_ts[self.idx_test_dic[j], j], final[self.idx_test_dic[j], j]
                )
            )
        print("test_AUC_taskwise: ", np.round(auc, 4))
        return final
