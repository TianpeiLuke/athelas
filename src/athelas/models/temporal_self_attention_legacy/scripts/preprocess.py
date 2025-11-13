import glob
import os
import pickle as pkl
import time
import typing
from datetime import datetime

import numpy as np
import pandas as pd

from typing import Dict, Any


# Preprocessing sequence OTFs to tensors
import json
import sys
import pickle

seq_len = 51
SEP = ";SEP;"

mtx_from_dict_fill_default = (
    lambda input_data, var_list_otf, var_list, map_dict: np.array(
        [
            [
                map_dict[var_list[i]] if a in ["", "My Text String"] else a
                for a in input_data[var_list_otf[i]].split(SEP)
            ]
            for i in range(len(var_list_otf))
        ]
    ).transpose()
)
arr_from_dict_fill_default = lambda input_data, var_list, map_dict: np.expand_dims(
    np.array(
        [
            map_dict[var_list[i]]
            if input_data[var_list[i]] in ["", "My Text String"]
            else input_data[var_list[i]]
            for i in range(len(var_list))
        ]
    ),
    axis=0,
)
arr_from_dict = lambda x, y: np.expand_dims(np.array([x[i] for i in y]), axis=0)


def data_parsing(input_data):
    """
    Data parsing and sanity check
    :param input_data: input data
    :return: pass_check, terminal_vec_lst, neighbor_vec_lst, neighbor_y_lst
    """
    global seq_num_scale_, seq_num_min_, num_static_scale_, num_static_min_, categorical_map, default_value_dict

    #     print('Check input. '
    #                   'Source: {}'.format(input_data))

    if not isinstance(input_data, dict):
        #         print('Sanity check failed. '
        #               'Input data is not a dict obj. '
        #               'Source: {}'.format(input_data))
        return False, None, None, None

    input_data["objectId"] = "CURRENT"

    for VAR in input_data_seq_cat_otf_vars:
        if VAR not in input_data:
            #             print('Sanity check failed. '
            #                   'Input data does not contain required key. '
            #                   'Source: {}'.format(input_data))
            return False, None, None, None

    for VAR in input_data_seq_cat_vars:
        if VAR not in input_data:
            #             print('Sanity check failed. '
            #                   'Input data does not contain required key. '
            #                   'Source: {}'.format(input_data))
            return False, None, None, None

    for VAR in input_data_seq_num_otf_vars:
        if VAR not in input_data:
            #             print('Sanity check failed. '
            #                   'Input data does not contain required key. '
            #                   'Source: {}'.format(input_data))
            return False, None, None, None

    for VAR in input_data_seq_num_vars:
        if VAR not in input_data:
            #             print('Sanity check failed. '
            #                   'Input data does not contain required key. '
            #                   'Source: {}'.format(input_data))
            return False, None, None, None

    for VAR in input_data_dense_num_vars:
        if VAR not in input_data:
            #             print('Sanity check failed. '
            #                   'Input data does not contain required key. '
            #                   'Source: {}'.format(input_data))
            return False, None, None, None

    no_history_flag = input_data[
        "payment_risk.bfs_order_cat_seq_by_cid.c_billingaddrlatlongconfidence_seq"
    ] in ["", "My Text String"]

    if len(
        input_data["payment_risk.bfs_order_cat_seq_by_cid.c_declr_seq"].split(SEP)
    ) != len(
        input_data[
            "payment_risk.bfs_order_cat_seq_by_cid.c_billingaddrlatlongconfidence_seq"
        ].split(SEP)
    ):
        #         print('Sanity check warning. '
        #               'Input data OTF not matured. '
        #               'Use only currently order for evaluation. '
        #               'Source: {}'.format(input_data))
        no_history_flag = True
    #         return False, None, None, None

    for i in numerical_cat_vars_indices:
        #         if cur_var in ['','My Text String']:
        #             print('Sanity check failed. '
        #                   'Input data numeric categorical variable value wrong. '
        #                   'Source: {}'.format(input_data))
        #             return False, None, None, None

        cur_var = input_data[input_data_seq_cat_vars[i]]
        if cur_var not in ["", "My Text String"]:
            cur_var = str(int(float(cur_var)))
            input_data[input_data_seq_cat_vars[i]] = cur_var

        if not no_history_flag:
            var_seq_list = [
                str(int(float(var_))) if var_ != "" else var_
                for var_ in input_data[input_data_seq_cat_otf_vars[i]].split(SEP)
            ]
            input_data[input_data_seq_cat_otf_vars[i]] = SEP.join(var_seq_list)

    # parse string to lst
    if not no_history_flag:
        seq_cat_vars_mtx = mtx_from_dict_fill_default(
            input_data,
            input_data_seq_cat_otf_vars,
            input_data_seq_cat_vars,
            default_value_dict,
        )
        seq_num_vars_mtx = mtx_from_dict_fill_default(
            input_data,
            input_data_seq_num_otf_vars,
            input_data_seq_num_vars,
            default_value_dict,
        )
    seq_cat_vars_lst = arr_from_dict_fill_default(
        input_data, input_data_seq_cat_vars, default_value_dict
    )
    seq_num_vars_lst = arr_from_dict_fill_default(
        input_data, input_data_seq_num_vars, default_value_dict
    )
    dense_num_vars_lst = arr_from_dict_fill_default(
        input_data, input_data_dense_num_vars, default_value_dict
    )

    if not no_history_flag:
        if len(seq_cat_vars_mtx) == len(seq_num_vars_mtx):
            if sum(
                seq_cat_vars_mtx[:, -1].argsort() == seq_cat_vars_mtx[:, -1].argsort()
            ) != len(seq_cat_vars_mtx):
                #                 print('Sanity check warning. '
                #                       'Input data OTFs have same length but mismatch lines. '
                #                       'Use only currently order for evaluation. '
                #                       'Source: {} {}'.format(seq_cat_vars_mtx, seq_num_vars_mtx))
                #                 return False, None, None, None
                no_history_flag = True
        else:
            #             print('Sanity check warning. '
            #                   'Input data OTFs have mismatch length. '
            #                   'Use only currently order for evaluation. '
            #                   'Source: {} {}'.format(seq_cat_vars_mtx, seq_num_vars_mtx))
            no_history_flag = True
    #             return False, None, None, None

    if not no_history_flag:
        seq_cat_mtx = np.concatenate(
            [seq_cat_vars_mtx[:, :-2], seq_cat_vars_lst[:, :-2]]
        )
    else:
        seq_cat_mtx = seq_cat_vars_lst[:, :-2]
    #     seq_cat_mtx[seq_cat_mtx=='']='-1'
    seq_cat_mtx = seq_cat_mtx.astype("str")
    #     print(seq_cat_mtx)
    seq_cat_mtx = transform_object.transform(seq_cat_mtx)
    #     print(seq_cat_mtx)
    seq_cat_mtx[seq_cat_mtx == "None"] = "0"
    seq_cat_mtx = seq_cat_mtx.astype(int)
    if not no_history_flag:
        seq_cat_mtx = np.pad(
            seq_cat_mtx, [(seq_len - 1 - len(seq_cat_vars_mtx), 0), (0, 0)]
        )
    else:
        seq_cat_mtx = np.pad(seq_cat_mtx, [(seq_len - 1, 0), (0, 0)])

    if not no_history_flag:
        seq_num_mtx = np.concatenate(
            [seq_num_vars_mtx[:, :-1], seq_num_vars_lst[:, :-1]]
        )
    else:
        seq_num_mtx = seq_num_vars_lst[:, :-1]
    seq_num_mtx = np.concatenate(
        [seq_num_mtx, np.ones((seq_num_mtx.shape[0], 1))], axis=1
    )
    #     seq_num_mtx[seq_num_mtx=='']='0'
    seq_num_mtx = seq_num_mtx.astype(float)
    seq_num_mtx[:, :-2] = seq_num_mtx[:, :-2] * np.array(seq_num_scale_) + np.array(
        seq_num_min_
    )
    seq_num_mtx[:, -2] = seq_num_mtx[-1, -2] - seq_num_mtx[:, -2]
    if not no_history_flag:
        seq_num_mtx = np.pad(
            seq_num_mtx, [(seq_len - 1 - len(seq_num_vars_mtx), 0), (0, 0)]
        )
    else:
        seq_num_mtx = np.pad(seq_num_mtx, [(seq_len - 1, 0), (0, 0)])

    dense_num_vars_lst = dense_num_vars_lst[:, :-2]
    #     dense_num_vars_lst[dense_num_vars_lst=='']='0'
    dense_num_vars_lst = dense_num_vars_lst.astype(float)
    dense_num_arr = dense_num_vars_lst * np.array(num_static_scale_) + np.array(
        num_static_min_
    )

    #     print('Data sanity check passed and preprocessed.')
    return True, seq_cat_mtx, seq_num_mtx, dense_num_arr


def dataframe_to_tensor(df):
    df["json"] = df.apply(lambda x: x.to_json(), axis=1)

    X_seq_cat_list = []
    X_seq_num_list = []
    X_num_list = []
    Y_list = []
    housekeeping_list = []

    for i in range(df.shape[0]):
        json1_str = df.iloc[i]["json"]
        input_data = json.loads(json1_str)
        ret, seq_cat_mtx, seq_num_mtx, dense_num_arr = data_parsing(input_data)
        if not ret:
            continue
        if np.max(seq_num_mtx[:, -2]) > 10000000:
            seq_num_mtx[:, -2] = 10000000
        X_seq_cat_list.append(seq_cat_mtx)
        X_seq_num_list.append(seq_num_mtx)
        X_num_list.append(dense_num_arr)
        Y_list.append(arr_from_dict(input_data, ["IS_FRD"]))
    #         housekeeping_list.append(arr_from_dict(input_data, housekeeping_vars))

    train_X_seq_cat = np.stack(X_seq_cat_list, axis=0)
    train_X_seq_num = np.stack(X_seq_num_list, axis=0)
    train_X_num = np.concatenate(X_num_list, axis=0)
    train_Y = np.concatenate(Y_list, axis=0)
    #     housekeeping_all=np.concatenate(housekeeping_list,axis=0)

    return train_X_seq_cat, train_X_seq_num, train_X_num, train_Y  # , housekeeping_all


if __name__ == "__main__":
    ### Input ########################################
    training_data_path = "/opt/ml/processing/input/training"
    testing_data_path = "/opt/ml/processing/input/testing"
    calibration_data_path = "/opt/ml/processing/input/calibration"

    config_path = "/opt/ml/processing/input/config"

    input_data_path = "/opt/ml/processing/input/data"
    output_path = "/opt/ml/processing/output"  # add to see if it works

    sys.path.append(config_path + "/scripts/")
    from CategoricalTransformer import *
    from params import *

    print("Reading config data. ")
    preprocessor_file = "preprocessor.pkl"
    preprocessor = pickle.load(open(os.path.join(config_path, preprocessor_file), "rb"))
    percentile_score_map = preprocessor["bin_map"]
    seq_num_scale_ = preprocessor["seq_num_scale_"]
    seq_num_min_ = preprocessor["seq_num_min_"]
    num_static_scale_ = preprocessor["num_static_scale_"]
    num_static_min_ = preprocessor["num_static_min_"]

    # Will be deleted
    num_static_scale_ = np.delete(num_static_scale_, [266, 267])
    num_static_min_ = np.delete(num_static_min_, [266, 267])

    from CategoricalTransformer import *
    from params import *

    cat_to_index_file = "cat_to_index.json"
    with open(os.path.join(config_path, cat_to_index_file), "r") as f:
        categorical_map = json.load(f)

    for key in list(categorical_map):
        categorical_map[key.replace(".", "__DOT__")] = categorical_map[key]

    default_value_dict_file = "default_value_dict.json"
    with open(os.path.join(config_path, default_value_dict_file), "r") as f:
        default_value_dict = json.load(f)

    for key in list(default_value_dict):
        default_value_dict[key.replace(".", "__DOT__")] = default_value_dict[key]

    columns_list = input_data_seq_cat_vars[:-2]
    transform_object = CategoricalTransformer(
        categorical_map=categorical_map, columns_list=columns_list
    )

    original_var_list = list(
        pd.read_csv(config_path + "/model_var_list.csv", header=None).iloc[:, 0]
    )
    var_list = list(
        set(
            [
                "objectId",
                "orderDate",
                "transactionDate",
                "marketplaceCountryCode",
                "marketplaceId",
                "isQueued",
                "paymeth",
                "ictry_cd",
                "bctry_cd",
                "sctry_cd",
                "cctry_cd",
                "isSidelined",
                # "ruleIds",
                "emailorg",
                "creditCardIds",
            ]
            + original_var_list
        )
    )
    config: Dict[str, Any] = {
        "tag": "IS_FRD",
        "target_positive_rate": 0.2,
        "model_var_list": original_var_list,
        "calib_data_required_size": 2000000,
        "var_list": var_list,
    }

    os.mkdir("/opt/ml/processing/config")
    pkl.dump(config, open("/opt/ml/processing/output/config.pkl", "wb"))

    target_positive_rate = config["target_positive_rate"]
    tag = config["tag"]
    model_var_list = config["model_var_list"]

    #     metadata_path = "/opt/ml/processing/input/metadata/metadata.csv"
    #     metadata = pd.read_csv(metadata_path)
    #     print("meta data shape: ", metadata.shape)

    ### Load input data
    print("Reading training data from {}".format(training_data_path))
    train_data = pd.concat(
        [
            pd.read_csv(f, sep="\t", low_memory=False, dtype=str, keep_default_na=False)
            for f in glob.glob(os.path.join(training_data_path, "*.csv.gz").sorted())
        ]
    )

    train_data["orderDate"] = train_data["orderDate"].astype(np.int64)
    train_data.columns = [sub.replace("__DOT__", ".") for sub in train_data.columns]

    #     for i in config["var_list"]:
    #         if i in set(metadata.loc[~(metadata.iscategory.astype(bool)), "varname"]):
    #             train_data[i] = pd.to_numeric(train_data[i])
    print("training data size: ", train_data.shape)

    print("Reading calibration data from {}".format(calibration_data_path))
    calib_data = pd.concat(
        [
            pd.read_csv(f, sep="\t", low_memory=False, dtype=str, keep_default_na=False)
            for f in glob.glob(os.path.join(calibration_data_path, "*.csv.gz").sorted())
        ]
    )
    calib_data["orderDate"] = calib_data["orderDate"].astype(np.int64)
    calib_data.columns = [sub.replace("__DOT__", ".") for sub in calib_data.columns]

    #     for i in config["var_list"]:
    #         if i in set(metadata.loc[~(metadata.iscategory.astype(bool)), "varname"]):
    #             calib_data[i] = pd.to_numeric(calib_data[i])
    print("calibration data size: ", calib_data.shape)

    #     currency_path = "/opt/ml/processing/input/currency_variables/currency_variables.csv"
    #     curr_df = pd.read_csv(currency_path)
    #     print("currency variables shape: ", curr_df.shape)

    ### NA tag ########################################
    # remove NA & -1 tag and convert tag to int
    train_data = train_data[train_data["creditCardIds"] != "9990-0012191-573601"]
    train_data = train_data.loc[
        (~train_data[tag].isnull()) & (train_data[tag] != -1), :
    ]

    train_data[tag] = pd.to_numeric(train_data[tag], downcast="integer")

    if "marketplaceCountryCode" in train_data.columns:
        print(
            "marketplace distribution: ",
            train_data.marketplaceCountryCode.value_counts(),
        )

    ### downsample train data
    ones_cond, zeros_cond = train_data[tag] == 1, train_data[tag] == 0
    positive_count, negative_count = (
        len(train_data[ones_cond]),
        len(train_data[zeros_cond]),
    )
    total_count = positive_count + negative_count
    positive_rate = positive_count / total_count
    print("train data positive_rate:  ", positive_rate)

    if positive_rate < target_positive_rate:
        negative_downsampled_cnt = int(
            positive_count * (1 - target_positive_rate) / target_positive_rate
        )
        positive_downsampled_cnt = positive_count

    print(total_count, negative_count, positive_count, sep=",")

    train_binned_downsampled_data = pd.concat(
        [
            train_data[zeros_cond].sample(negative_downsampled_cnt),
            train_data[ones_cond].sample(positive_downsampled_cnt),
        ],
        ignore_index=True,
    )

    print("Filtering and downsampling finished.")
    print(negative_downsampled_cnt, positive_downsampled_cnt, sep=",")

    ### data shape after processing
    print("Train data shape after preprocessing: {}".format(train_data.shape))
    print(
        "train_binned_downsampled_data data shape after preprocessing: {}".format(
            train_binned_downsampled_data.shape
        )
    )
    # print("Test data shape after preprocessing: {}".format(test_data.shape))
    print("Calib data shape after preprocessing: {}".format(calib_data.shape))

    ####### Convert dataframe to tensor ###############
    train_X_seq_cat, train_X_seq_num, train_X_num, train_Y = dataframe_to_tensor(
        train_binned_downsampled_data
    )

    dataset = "train"
    p = "0"
    out_dir = "/opt/ml/processing/output" + "/" + dataset
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.save(
        file=os.path.join(out_dir, "{}_X_seq_cat_v{}.npy".format(dataset, p)),
        arr=train_X_seq_cat,
    )
    np.save(
        file=os.path.join(out_dir, "{}_X_seq_num_v{}.npy".format(dataset, p)),
        arr=train_X_seq_num,
    )
    np.save(
        file=os.path.join(out_dir, "{}_X_num_v{}.npy".format(dataset, p)),
        arr=train_X_num,
    )
    np.save(file=os.path.join(out_dir, "{}_Y_v{}.npy".format(dataset, p)), arr=train_Y)

    del (
        train_binned_downsampled_data,
        train_X_seq_cat,
        train_X_seq_num,
        train_X_num,
        train_Y,
    )
    #####################
    cali_X_seq_cat, cali_X_seq_num, cali_X_num, cali_Y = dataframe_to_tensor(calib_data)

    dataset = "cali"
    p = "0"
    out_dir = "/opt/ml/processing/output" + "/" + dataset
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.save(
        file=os.path.join(out_dir, "{}_X_seq_cat_v{}.npy".format(dataset, p)),
        arr=cali_X_seq_cat,
    )
    np.save(
        file=os.path.join(out_dir, "{}_X_seq_num_v{}.npy".format(dataset, p)),
        arr=cali_X_seq_num,
    )
    np.save(
        file=os.path.join(out_dir, "{}_X_num_v{}.npy".format(dataset, p)),
        arr=cali_X_num,
    )
    np.save(file=os.path.join(out_dir, "{}_Y_v{}.npy".format(dataset, p)), arr=cali_Y)

    del calib_data, cali_X_seq_cat, cali_X_seq_num, cali_X_num, cali_Y
    #######################

    print("Reading test data from {}".format(testing_data_path))
    test_data = pd.concat(
        [
            pd.read_csv(f, sep="\t", low_memory=False, dtype=str, keep_default_na=False)
            for f in glob.glob(os.path.join(testing_data_path, "*.csv.gz").sorted())
        ]
    )  # [:5] #remove the [:100]
    test_data.columns = [sub.replace("__DOT__", ".") for sub in test_data.columns]
    #     for i in config["var_list"]:
    #         if i in set(metadata.loc[~(metadata.iscategory.astype(bool)), "varname"]):
    #             test_data[i] = pd.to_numeric(test_data[i])
    print("input data size: ", test_data.shape)

    test_data = test_data.loc[(~test_data[tag].isnull()) & (test_data[tag] != "-1"), :]
    # test_data[tag] = test_data[tag].apply(int)
    test_data[tag] = pd.to_numeric(test_data[tag], downcast="integer")

    print("Test data shape after preprocessing: {}".format(test_data.shape))

    #############
    vali_X_seq_cat, vali_X_seq_num, vali_X_num, vali_Y = dataframe_to_tensor(test_data)

    dataset = "vali"
    p = "0"
    out_dir = "/opt/ml/processing/output" + "/" + dataset
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.save(
        file=os.path.join(out_dir, "{}_X_seq_cat_v{}.npy".format(dataset, p)),
        arr=vali_X_seq_cat,
    )
    np.save(
        file=os.path.join(out_dir, "{}_X_seq_num_v{}.npy".format(dataset, p)),
        arr=vali_X_seq_num,
    )
    np.save(
        file=os.path.join(out_dir, "{}_X_num_v{}.npy".format(dataset, p)),
        arr=vali_X_num,
    )
    np.save(file=os.path.join(out_dir, "{}_Y_v{}.npy".format(dataset, p)), arr=vali_Y)

    del test_data, vali_X_seq_cat, vali_X_seq_num, vali_X_num, vali_Y
