import pandas as pd
import numpy as np
import glob
import os
import time
import pickle

import matplotlib.pyplot as plt
import math
import lightgbm as lgb

# import lightgbmmt as lgbm
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


class LgbBaseline:
    """
    Train and test seperate single lgb as baseline models

    Parameters
    ----------
    X_train: training dataset

    Y_train: training label matrix with all tasks

    X_test: test dataset

    Y_test: test label matrix with all tasks

    idx_trn_dic: a dict of training data indices for each task

    idx_test_dic: a dict of test data indices for each task

    config: a configuration file containing self-defined parameters

    params: hyper parameters used for lgb modeling, all values are obtained from config file

    y_pred: a dict for storing predicted scores under each lgb model

    eval_result: a dict for storing evaluation results during training for each baseline lgb
    """

    def __init__(self, X_train, Y_train, X_test, Y_test, config=None):
        self.config = from_json(config)
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.idx_trn_dic = filter_idx(Y_train)
        self.idx_test_dic = filter_idx(Y_test)
        self.params = {
            "metric": "auc",
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
        }
        self.y_pred = {}
        self.eval_result = {}

    def train(self, params, X_train, y_train, X_test, y_test, idx_trn, idx_test, col):
        """
        A function for both training and test for baseline model

        Parameters
        ----------
        X_train: training dataset

        Y_train: training label matrix with all tasks

        X_test: test dataset

        Y_test: test label matrix with all tasks

        idx_trn_dic: a dict of training data indices for each task

        idx_test_dic: a dict of test data indices for each task

        col: key index for each task
        """
        X_trn, y_trn = X_train.iloc[idx_trn[col]], y_train.iloc[idx_trn[col]]
        X_ts, y_ts = X_test.iloc[idx_test[col]], y_test.iloc[idx_test[col]]
        # initializing empty vecs
        y_lgb = np.zeros(X_ts.shape[0])
        # train validation split
        X_tr, X_vl, y_tr, y_vl = train_test_split(
            X_trn, y_trn, test_size=0.3, random_state=seed
        )
        d_train = lgb.Dataset(X_tr, label=y_tr)
        d_valid = lgb.Dataset(X_vl, label=y_vl)

        evals = {}
        callback = [lgb.log_evaluation(50), lgb.record_evaluation(evals)]
        model = lgb.train(
            params,
            num_boost_round=1000,
            train_set=d_train,
            valid_sets=[d_valid],
            keep_training_booster=True,
            callbacks=callback,
        )

        y_lgb = model.predict(X_ts)
        print(
            "test metrics:",
            roc_auc_score(y_ts, y_lgb),
            log_loss(y_ts, y_lgb),
            f1_score(y_ts, y_lgb.round(0)),
        )
        return y_lgb, evals, model

    def baseline_model(self):
        """
        Construct six separate single lgb baselines for each task
        """
        print("Overall lgb result: ")
        y_overall_hat, all_evals, modall = self.train(
            self.params,
            self.X_train,
            self.Y_train.isFraud,
            self.X_test,
            self.Y_test.isFraud,
            self.idx_trn_dic,
            self.idx_test_dic,
            0,
        )
        self.y_pred[0] = y_overall_hat
        self.eval_result[0] = all_evals
        modall.save_model("baseline_overall.txt")

        print("CC single lgb result: ")
        yhat_cc, eval_cc, modcc = self.train(
            self.params,
            self.X_train,
            self.Y_train.isCCfrd,
            self.X_test,
            self.Y_test.isCCfrd,
            self.idx_trn_dic,
            self.idx_test_dic,
            1,
        )
        self.y_pred[1] = yhat_cc
        self.eval_result[1] = eval_cc
        modcc.save_model("baseline_cc.txt")

        print("DD single lgb result: ")
        yhat_dd, eval_dd, moddd = self.train(
            self.params,
            self.X_train,
            self.Y_train.isDDfrd,
            self.X_test,
            self.Y_test.isDDfrd,
            self.idx_trn_dic,
            self.idx_test_dic,
            2,
        )
        self.y_pred[2] = yhat_dd
        self.eval_result[2] = eval_dd
        moddd.save_model("baseline_dd.txt")

        print("GC single lgb result: ")
        yhat_gc, eval_gc, modgc = self.train(
            self.params,
            self.X_train,
            self.Y_train.isGCfrd,
            self.X_test,
            self.Y_test.isGCfrd,
            self.idx_trn_dic,
            self.idx_test_dic,
            3,
        )
        self.y_pred[3] = yhat_gc
        self.eval_result[3] = eval_gc
        modgc.save_model("baseline_gc.txt")

        print("LOC single lgb result: ")
        yhat_loc, eval_loc, modloc = self.train(
            self.params,
            self.X_train,
            self.Y_train.isLOCfrd,
            self.X_test,
            self.Y_test.isLOCfrd,
            self.idx_trn_dic,
            self.idx_test_dic,
            4,
        )
        self.y_pred[4] = yhat_loc
        self.eval_result[4] = eval_loc
        modloc.save_model("baseline_loc.txt")

        print("Cim single lgb result: ")
        yhat_cim, eval_cim, modcim = self.train(
            self.params,
            self.X_train,
            self.Y_train.isCimfrd,
            self.X_test,
            self.Y_test.isCimfrd,
            self.idx_trn_dic,
            self.idx_test_dic,
            5,
        )
        self.y_pred[5] = yhat_cim
        self.eval_result[5] = eval_cim
        modcim.save_model("baseline_cim.txt")
