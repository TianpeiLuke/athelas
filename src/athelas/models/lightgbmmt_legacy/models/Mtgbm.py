import pandas as pd
import numpy as np
import glob
import os
import time

import matplotlib.pyplot as plt
import math
import lightgbm as lgb
import lightgbmmt as lgbm
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
from scipy.special import expit, rel_entr

from customLossKDswap import custom_loss_KDswap
from baseLoss import base_loss
from customLossNoKD import custom_loss_noKD

seed = 10
np.random.seed(seed)
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
from collections import Counter
from util import *


class MtGbm:
    """
    MTGBM with our proposed weighting method and KD loss implementation

    Parameters
    ----------
    config: a configuration file containing self-defined parameters

    X_train: training dataset

    train_label: training label matrix with all tasks

    sub_tasks_list: subtasks name list

    main_target: name of the main target column (default: 'is_abusive')

    loss_type: choose different loss types.
               Please choose "auto_weight" for our proposed weight,
               or "auto_weight_KD" with KD loss included

    Attributes
    ----------
    y_train, y_train_s: main task, sub tasks in training set

    model: the MTGBM training model

    Methods
    -------
    train(): Train the MTGBM model

    predict(X_test, test_label=None): Make predictions on test data

    evaluate(X_test, test_label, df_pred, idx_test_dic=None): Evaluate model performance
    """

    def __init__(
        self,
        config,
        X_train,
        train_label,
        sub_tasks_list,
        main_target="is_abusive",
        loss_type=None,
    ):
        # Handle both config file path and direct parameter dictionary
        if isinstance(config, dict):
            self.params = config
        else:
            self.params = from_json(config)
        self.loss_type = loss_type
        self.train_label = train_label
        self.X_train = X_train
        self.main_target = main_target
        self.y_train = train_label[main_target].copy().reset_index(drop=True)
        self.targets = sub_tasks_list

        # Debug: Check what we're working with
        print(f"Main target: {main_target}")
        print(f"Sub tasks: {sub_tasks_list}")
        print(f"Available columns in train_label: {list(train_label.columns)}")

        # Ensure y_train_s is always a DataFrame by using double bracket notation
        # This forces pandas to return a DataFrame even for single columns
        if isinstance(self.targets, list):
            if len(self.targets) == 1:
                # Single column - use double brackets to force DataFrame
                self.y_train_s = train_label[self.targets].copy().reset_index(drop=True)
                # If it's still a Series, convert it
                if isinstance(self.y_train_s, pd.Series):
                    self.y_train_s = pd.DataFrame(self.y_train_s)
            else:
                # Multiple columns
                self.y_train_s = train_label[self.targets].copy().reset_index(drop=True)
        else:
            # Single string target
            self.y_train_s = train_label[[self.targets]].copy().reset_index(drop=True)

        self.model = None

    def train(self):
        """
        Model training and validation process
        """
        print("Training set size: ", self.X_train.shape)
        print(
            "Training main task shape: ",
            len(self.y_train),
            " Training sub tasks shape: ",
            self.y_train_s.shape,
        )

        # --- train validation split
        X_tr, X_vl, Y_tr, Y_vl = train_test_split(
            self.X_train, self.y_train, test_size=0.1, random_state=seed
        )
        tr_idx, val_idx = Y_tr.index, Y_vl.index
        # sub tasks split
        # Get number of subtask columns safely
        if isinstance(self.y_train_s, pd.DataFrame):
            num_subtask_cols = self.y_train_s.shape[1]
        else:
            num_subtask_cols = 1
            # Convert Series to DataFrame if needed
            self.y_train_s = pd.DataFrame(self.y_train_s)

        arr = np.array(range(num_subtask_cols))
        Y_tr2, Y_vl2 = (
            self.y_train_s.iloc[tr_idx, arr],
            self.y_train_s.iloc[val_idx, arr],
        )

        trn_labels = self.train_label.iloc[tr_idx].reset_index()
        val_labels = self.train_label.iloc[val_idx].reset_index()

        # Create dynamic dictionaries based on number of tasks (main + subtasks)
        # The loss function expects indices starting from 1 for subtasks
        num_tasks = 1 + len(self.targets)  # main task + subtasks
        idx_trn_dic = {}
        idx_val_dic = {}

        # Index 0 is for main task, indices 1+ are for subtasks
        for i in range(num_tasks):
            idx_trn_dic[i] = trn_labels.index
            idx_val_dic[i] = val_labels.index

        cate_feature = []
        debug = True
        num_label = 1 + len(self.targets)

        # Handle both dict and object parameter access
        def get_param(key, default=None):
            if isinstance(self.params, dict):
                return self.params.get(key, default)
            else:
                return getattr(self.params, key, default)

        mt_params = {
            "objective": "custom",
            "num_labels": num_label,
            "tree_learner": "serial2",
            "boosting": "gbdt",
            "max_depth": get_param("max_depth", 16),
            "learning_rate": get_param("learning_rate", 0.05),
            "bagging_fraction": get_param("bagging_fraction", 0.9),
            "feature_fraction": get_param("feature_fraction", 0.9),
            "verbosity": get_param("verbosity", 1),
            "lambda_l1": get_param("lambda_l1", 0.5),
            "lambda_l2": get_param("lambda_l2", 0.05),
            "num_leaves": get_param("num_leaves", 750),
            "min_child_weight": get_param("min_child_weight", 0.1),
            "min_data_in_leaf": get_param("min_data_in_leaf", 100),
            "num_threads": get_param("num_threads", 80),
            "metric_freq": get_param("metric_freq", 10),
            "data_random_seed": get_param("data_random_seed", 17),
        }
        verbose_eval = get_param("verbose_eval", 50)
        num_rounds = get_param("num_rounds", 100)
        #        early_stopping_rounds = self.params.early_stopping_rounds

        d_train = lgbm.Dataset(
            X_tr,
            label=np.concatenate([Y_tr.values.reshape((-1, 1)), Y_tr2.values], axis=1),
        )
        d_valid = lgbm.Dataset(
            X_vl,
            label=np.concatenate([Y_vl.values.reshape((-1, 1)), Y_vl2.values], axis=1),
        )

        start_time = time.time()
        #         trn_label_mat = np.array(pd.concat([Y_tr, Y_tr2], axis=1))
        #         val_label_mat = np.array(pd.concat([Y_vl, Y_vl2], axis=1))

        if self.loss_type == "auto_weight":
            cl = custom_loss_noKD(num_label, idx_val_dic, idx_trn_dic)
            self.model = lgbm.train(
                mt_params,
                train_set=d_train,
                num_boost_round=num_rounds,
                valid_sets=d_valid,
                #                                  early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
                fobj=cl.self_obj,
                feval=cl.self_eval,
            )

        elif self.loss_type == "auto_weight_KD":
            cl = custom_loss_KDswap(num_label, idx_val_dic, idx_trn_dic, 100)
            self.model = lgbm.train(
                mt_params,
                train_set=d_train,
                num_boost_round=num_rounds,
                valid_sets=d_valid,
                #                                    early_stopping_rounds=early_stopping_rounds,
                verbose_eval=verbose_eval,
                fobj=cl.self_obj,
                feval=cl.self_eval,
            )

        else:
            # default setting from original MTGBM implementation with fixed weight vector
            cl = base_loss(idx_val_dic)
            self.model = lgbm.train(
                mt_params,
                train_set=d_train,
                num_boost_round=num_rounds,
                valid_sets=d_valid,
                verbose_eval=verbose_eval,
                #                                  early_stopping_rounds=early_stopping_rounds,
                fobj=cl.base_obj,
                feval=cl.base_eval,
            )
        self.model.set_num_labels(num_label)
        self.model.save_model("model.txt")
        print("--- training time: %.2f mins ---" % ((time.time() - start_time) / 60))

        # --- Check evaluation results from model training
        plt.style.use("ggplot")
        # -- plot evaluation curve
        eval_score = np.array(cl.eval_mat)
        subtask_name = self.targets
        task_name = np.insert(subtask_name, 0, "main")
        for j in range(eval_score.shape[1]):
            plt.plot(eval_score[:, j], label=task_name[j])
        plt.legend(ncol=2)
        plt.ylim(0.94, 1)
        plt.title("Evaluation Results")
        plt.savefig("mtg.png")
        plt.show()
        # -- plot subtask weight changing
        weight = np.array(cl.w_trn_mat)
        for j in range(1, weight.shape[1]):
            plt.plot(weight[:, j], label=task_name[j])
        plt.legend(ncol=2)
        plt.title("Weights Changing Trend")
        plt.savefig("weight_change.png")
        plt.show()

    def predict(self, X_test, test_label=None):
        """
        Model prediction

        Parameters
        ----------
        X_test: test dataset features
        test_label: optional test labels for evaluation metrics

        Returns
        -------
        df_pred: DataFrame with predictions for main task and subtasks
        """
        print("Test set size: ", X_test.shape)

        temp = self.model.predict(X_test)
        y_lgbmt = expit(temp[:, 0])
        y_lgbmtsub = expit(temp[:, 1:])

        # If test labels are provided, calculate and print metrics
        if test_label is not None:
            y_test = test_label[self.main_target].copy().reset_index(drop=True)
            # Ensure y_test_s is always a DataFrame
            if len(self.targets) == 1:
                y_test_s = (
                    test_label[self.targets].to_frame().copy().reset_index(drop=True)
                )
            else:
                y_test_s = test_label[self.targets].copy().reset_index(drop=True)
            print(
                "Test main task shape: ",
                len(y_test),
                " Test sub tasks shape: ",
                y_test_s.shape,
            )
            print(
                "main task test metrics:",
                " AUC ",
                roc_auc_score(y_test, y_lgbmt),
                " logloss ",
                log_loss(y_test, y_lgbmt),
                " f1 score ",
                f1_score(y_test, y_lgbmt.round(0)),
            )

        # Create prediction DataFrame with flexible column names
        pred_dict = {self.main_target: y_lgbmt}
        for i, target in enumerate(self.targets):
            pred_dict[target] = y_lgbmtsub[:, i]

        df_pred = pd.DataFrame(pred_dict)
        return df_pred

    def evaluate(self, X_test, test_label, df_pred, idx_test_dic=None):
        """
        Evaluate final results

        Parameters
        ----------
        X_test: test dataset features
        test_label: test labels
        df_pred: prediction DataFrame from predict() method
        idx_test_dic: optional dictionary of indices for each task
        """
        y_test = test_label[self.main_target].copy().reset_index(drop=True)
        # Ensure y_test_s is always a DataFrame
        if len(self.targets) == 1:
            y_test_s = test_label[self.targets].to_frame().copy().reset_index(drop=True)
        else:
            y_test_s = test_label[self.targets].copy().reset_index(drop=True)
        y_lgbmt = df_pred[self.main_target]

        # If idx_test_dic is not provided, use all indices for each task
        if idx_test_dic is None:
            idx_test_dic = {i: y_test_s.index for i in range(len(self.targets))}

        # Extract true and predicted values for each subtask
        subtask_metrics = []
        for i, target in enumerate(self.targets):
            if i in idx_test_dic:
                true_vals = y_test_s.iloc[idx_test_dic[i]][target]
                pred_vals = df_pred.iloc[idx_test_dic[i]][target]
                subtask_metrics.append((true_vals, pred_vals, target))

        # --- plot feature importance
        train_columns = self.X_train.columns
        feature_importances = (
            self.model.feature_importance() / sum(self.model.feature_importance())
        ) * 100
        results = pd.DataFrame(
            {"Features": train_columns, "Importances": feature_importances}
        )

        sns.set(font_scale=0.75)
        sns.barplot(
            x="Importances",
            y="Features",
            data=results.sort_values(by="Importances", ascending=False)[0:20],
        )
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.show()

        # --- plot ROC curves
        plt.style.use("ggplot")
        fpr, tpr, thres = roc_curve(y_test, y_lgbmt)

        plt.plot([0, 1], [0, 1], "k--")
        plt.plot(
            fpr,
            tpr,
            label=f"main task ({self.main_target}), AUC=%0.4f"
            % roc_auc_score(y_test, y_lgbmt),
        )

        # Plot ROC curves for each subtask
        for true_vals, pred_vals, target_name in subtask_metrics:
            if len(true_vals) > 0:  # Only plot if we have data
                fpr_sub, tpr_sub, _ = roc_curve(true_vals, pred_vals)
                plt.plot(
                    fpr_sub,
                    tpr_sub,
                    label=f"{target_name}, AUC = %0.4f"
                    % roc_auc_score(true_vals, pred_vals),
                )

        plt.legend()
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("MTGBM ROC")
        # plt.savefig('ROC_mt.png')
        plt.show()

        # --- plot PRAUC curves
        pr, rec, pr_thres = precision_recall_curve(y_test, y_lgbmt)

        plt.plot(
            rec, pr, label=f"main task ({self.main_target}), prAUC=%0.4f" % auc(rec, pr)
        )

        # Plot PR curves for each subtask
        for true_vals, pred_vals, target_name in subtask_metrics:
            if len(true_vals) > 0:  # Only plot if we have data
                p_sub, r_sub, _ = precision_recall_curve(true_vals, pred_vals)
                plt.plot(
                    r_sub,
                    p_sub,
                    label=f"{target_name}, AUC = %0.4f" % auc(r_sub, p_sub),
                )

        plt.legend()
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.title("MTGBM PRcurves")
        # plt.savefig('PR_mt.png')
        plt.show()
