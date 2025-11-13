import pandas as pd
import numpy as np
import glob
import os
import time
import pickle
import json


def data_loader(path):
    """
    Loader and concatenate raw data from DAWS
    """
    data_paths = [os.path.join(path, f) for f in os.listdir(path)]
    data_paths = [i for i in data_paths if os.path.isfile(i)]
    l = []
    data_paths.sort()
    for filename in data_paths:
        # print(filename)
        df = pd.read_csv(filename, delimiter=",", header=None)
        l.append(df)
    df = pd.concat(l, axis=0, ignore_index=True)
    return df


def get_lgb_varimp(model, train_columns):
    """
    Create feature importance table
    """
    feature_importances = (
        model.feature_importance() / sum(model.feature_importance())
    ) * 100
    results = pd.DataFrame(
        {"Features": train_columns, "Importances": feature_importances}
    )
    results.sort_values(by="Importances", inplace=True, ascending=False)
    return results


def from_json(cfg):
    """
    Creates config from json
    """
    params = json.loads(json.dumps(cfg), object_hook=HelperObject)
    return params


def reduce_mem_usage(df):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    return df


def cramers_corrected_stat(x, y):
    """
    Corrected Cramer correlation calculation
    """
    result = -1
    if len(np.unique(x)) == 1:
        print("First variable is constant")
    elif len(np.unique(y)) == 1:
        print("Second variable is constant")
    else:
        conf_matrix = pd.crosstab(x, y)

        if conf_matrix.shape[0] == 2:
            correct = False
        else:
            correct = True

        chi2 = ss.chi2_contingency(conf_matrix, correction=correct)[0]
        n = sum(conf_matrix.sum())
        phi2 = chi2 / n
        r, k = conf_matrix.shape
        phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        rcorr = r - ((r - 1) ** 2) / (n - 1)
        kcorr = k - ((k - 1) ** 2) / (n - 1)
        result = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
        return result


def cramer_corr_mat(labels, num_col):
    """
    Basic Cramer correlation calculation
    """
    rows = []
    for i in range(num_col):
        col = []
        for j in range(num_col):
            cramers = cramers_corrected_stat(
                labels[:, i], labels[:, j]
            )  # Cramer's V test
            col.append(cramers)
        rows.append(col)


def condition_paymeth(labels):
    """
    Major payment types condition in filter data
    """
    cc = labels.paymeth == "CC"
    dd = labels.paymeth == "DD"
    gc = labels.paymeth == "GC"
    loc = labels.paymeth == "LineOfCredit"
    cim = labels.paymeth == "Cimarron"
    return cc, dd, gc, loc, cim


def filter_idx(labels):
    """
    Filter payment type indices
    """
    cc, dd, gc, loc, cim = condition_paymeth(labels)
    idx = {
        0: labels.index,
        1: labels[cc].index,
        2: labels[dd].index,
        3: labels[gc].index,
        4: labels[loc].index,
        5: labels[cim].index,
    }
    return idx


def create_paymeth_label(df):
    """
    Create task labels for each payment type
    """
    if df.paymeth == "CC" and df.isFraud == 0:
        df.isCCfrd = 0
    if df.paymeth == "DD" and df.isFraud == 0:
        df.isDDfrd = 0
    if df.paymeth == "GC" and df.isFraud == 0:
        df.isGCfrd = 0
    if df.paymeth == "LineOfCredit" and df.isFraud == 0:
        df.isLOCfrd = 0
    if df.paymeth == "Cimarron" and df.isFraud == 0:
        df.isCimfrd = 0
    if (
        df.paymeth == "CC,GC"
        or df.paymeth == "DD,GC"
        or df.paymeth == "GC,LineOfCredit"
    ):
        df.paymeth = "GC"
        if df.isFraud == 0:
            df.isGCfrd = 0
    return df


class HelperObject(object):
    """
    Helper class to convert json into Python object
    """

    def __init__(self, dict_):
        self.__dict__.update(dict_)
