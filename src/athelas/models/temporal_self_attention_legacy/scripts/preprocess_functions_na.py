import os
import sys
import gc
import shutil

# output_path = "/opt/ml/processing/output"
config_path = "/opt/ml/processing/input/config"
sys.path.append(config_path + "/scripts/")
from CategoricalTransformer import *
from params_na import *

import glob
import pickle as pkl
import time
import typing
from datetime import datetime

import numpy as np
import pandas as pd
import math

from typing import Dict, Any

# Preprocessing sequence OTFs to tensors
import json
import pickle

with open("/opt/ml/config/resourceconfig.json") as host_config_file:
    host_config = json.load(host_config_file)
current_host = host_config["current_host"]
hosts = host_config["hosts"]
print(current_host)
print(hosts)

config: Dict[str, Any] = {
    "tag": "IS_FRD",
    "target_positive_rate": 0.2,
}

target_positive_rate = config["target_positive_rate"]
tag = config["tag"]

seq_len = 51
SEP = "~"

# print("Reading config data. ")
preprocessor_file = "preprocessor_na.pkl"
preprocessor = pickle.load(open(os.path.join(config_path, preprocessor_file), "rb"))
# percentile_score_map = preprocessor["bin_map"]
seq_num_scale_ = preprocessor["seq_num_scale_"]
seq_num_min_ = preprocessor["seq_num_min_"]
num_static_scale_ = preprocessor["num_static_scale_"]
num_static_min_ = preprocessor["num_static_min_"]

# preprocessor_file = 'preprocessor_scaler_launch_partial.pkl'
# preprocessor_percentile_file = f'percentile_score_{seconds}.pkl'
# preprocessor = pickle.load(open(os.path.join(model_dir, preprocessor_file), 'rb'))
# percentile_score_map = pickle.load(open(os.path.join(model_dir, preprocessor_percentile_file), 'rb'))
#     percentile_score_map = preprocessor['bin_map']
# seq_num_scale_ = preprocessor['seq_num_scale_']
# seq_num_min_ = preprocessor['seq_num_min_']
# num_static_scale_ = preprocessor['num_static_scale_']
# num_static_min_ = preprocessor['num_static_min_']

# Will be deleted when generating new min-max scaler parameters
# num_static_scale_ = np.delete(num_static_scale_, [266, 267])
# num_static_min_ = np.delete(num_static_min_, [266, 267])

cat_to_index_file = "cat_to_index_na.json"
with open(os.path.join(config_path, cat_to_index_file), "r") as f:
    categorical_map = json.load(f)

for key in list(categorical_map):
    categorical_map[key.replace(".", "__DOT__")] = categorical_map[key]

default_value_dict_file = "default_value_dict_na.json"
with open(os.path.join(config_path, default_value_dict_file), "r") as f:
    default_value_dict = json.load(f)

for key in list(default_value_dict):
    default_value_dict[key.replace(".", "__DOT__")] = default_value_dict[key]

columns_list = input_data_seq_cat_vars[:-2]
transform_object = CategoricalTransformer(
    categorical_map=categorical_map, columns_list=columns_list
)

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


def sequence_data_parsing(
    input_data,
    input_data_seq_cat_otf_vars,
    input_data_seq_num_otf_vars,
    objectid_otf_name,
):
    """
    Sequence data parsing and sanity check
    :param input_data: input data
    :return: pass_check, seq_cat_mtx, seq_num_mtx
    """
    global \
        seq_num_scale_, \
        seq_num_min_, \
        num_static_scale_, \
        num_static_min_, \
        categorical_map, \
        default_value_dict

    for VAR in input_data_seq_cat_otf_vars:
        if VAR not in input_data:
            print(
                "Sanity check failed. "
                "Input data does not contain required key in input_data_seq_cat_otf_vars. Example variable in OTF class: {}"
                "Source: {}".format(objectid_otf_name, input_data)
            )
            return False, None, None

    for VAR in input_data_seq_num_otf_vars:
        if VAR not in input_data:
            print(
                "Sanity check failed. "
                "Input data does not contain required key in input_data_seq_num_otf_vars. Example variable in OTF class: {}"
                "Source: {}".format(objectid_otf_name, input_data)
            )
            return False, None, None

    no_history_flag = input_data[objectid_otf_name] in ["", "My Text String"]

    for i in numerical_cat_vars_indices:
        #         if cur_var in ['','My Text String']:
        #             print('Sanity check failed. '
        #                   'Input data numeric categorical variable value wrong. '
        #                   'Source: {}'.format(input_data))
        #             return False, None, None, None

        cur_var = input_data[input_data_seq_cat_vars[i]]
        if cur_var not in ["", "My Text String", "false"]:
            cur_var = str(int(float(cur_var)))
            input_data[input_data_seq_cat_vars[i]] = cur_var

        if not no_history_flag:
            var_seq_list = [
                str(int(float(var_))) if var_ != "" else var_
                for var_ in input_data[input_data_seq_cat_otf_vars[i]].split(SEP)
            ]
            input_data[input_data_seq_cat_otf_vars[i]] = SEP.join(var_seq_list)

    # Hard coded for fingerprintRiskValue, will remove later
    cur_var = input_data[input_data_seq_cat_vars[38]]
    if cur_var not in ["", "My Text String", "false"]:
        if float(cur_var) == 0:
            cur_var = str(int(float(cur_var)))
        else:
            cur_var = str(float(cur_var))
        input_data[input_data_seq_cat_vars[38]] = cur_var

    if not no_history_flag:
        var_seq_list_ = [
            str(float(var_)) if var_ != "" else var_
            for var_ in input_data[input_data_seq_cat_otf_vars[38]].split(SEP)
        ]
        var_seq_list = [
            str(int(float(var_))) if (var_ != "" and float(var_) == 0) else var_
            for var_ in var_seq_list_
        ]
        input_data[input_data_seq_cat_otf_vars[38]] = SEP.join(var_seq_list)
    ##

    columns_list = input_data_seq_cat_vars[:-2]
    transform_object = CategoricalTransformer(
        categorical_map=categorical_map, columns_list=columns_list
    )

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

    if not no_history_flag:
        if len(seq_cat_vars_mtx) == len(seq_num_vars_mtx):
            if sum(
                seq_cat_vars_mtx[:, -1].argsort() == seq_cat_vars_mtx[:, -1].argsort()
            ) != len(seq_cat_vars_mtx):
                print(
                    "Sanity check warning. "
                    "Input data OTFs have same length but mismatch lines. "
                    "Use only currently order for evaluation. "
                    "Source: {} {}".format(seq_cat_vars_mtx, seq_num_vars_mtx)
                )
                #                 return False, None, None, None
                #                 no_history_flag = True
                intersect1d, comm1, comm2 = np.intersect1d(
                    seq_cat_vars_mtx[:, -1],
                    seq_num_vars_mtx[:, -1],
                    return_indices=True,
                )
                seq_cat_vars_mtx = seq_cat_vars_mtx[comm1, :]
                seq_num_vars_mtx = seq_num_vars_mtx[comm2, :]
        else:
            print(
                "Sanity check warning. "
                "Input data OTFs have mismatch length. "
                "Use only currently order for evaluation. "
                "Source: {} {}".format(seq_cat_vars_mtx, seq_num_vars_mtx)
            )
            #             no_history_flag = True
            #             return False, None, None, None
            intersect1d, comm1, comm2 = np.intersect1d(
                seq_cat_vars_mtx[:, -1], seq_num_vars_mtx[:, -1], return_indices=True
            )
            seq_cat_vars_mtx = seq_cat_vars_mtx[comm1, :]
            seq_num_vars_mtx = seq_num_vars_mtx[comm2, :]

    if not no_history_flag:
        seq_cat_mtx = np.concatenate(
            [seq_cat_vars_mtx[:, :-2], seq_cat_vars_lst[:, :-2]]
        )
    else:
        seq_cat_mtx = seq_cat_vars_lst[:, :-2]
    #     seq_cat_mtx[seq_cat_mtx=='']='-1'
    seq_cat_mtx = seq_cat_mtx.astype("str")
    seq_cat_mtx = transform_object.transform(seq_cat_mtx)
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
    #     if np.max(seq_num_mtx[:,-2])>20000000:
    #         print('Sanity check failed. '
    #                   'Sequence OTF recorded wrong orderDate. '
    #                   'Source: {}'.format(input_data))
    #         return False, None, None

    if not no_history_flag:
        seq_num_mtx = np.pad(
            seq_num_mtx, [(seq_len - 1 - len(seq_num_vars_mtx), 0), (0, 0)]
        )
    else:
        seq_num_mtx = np.pad(seq_num_mtx, [(seq_len - 1, 0), (0, 0)])

    # print('Data sanity check passed and preprocessed for sequence.')
    return True, seq_cat_mtx, seq_num_mtx


def data_parsing(input_data):
    """
    Data parsing and sanity check
    :param input_data: input data
    :return: pass_check, seq_cat_mtx_cid, seq_num_mtx_cid, seq_cat_mtx_ccid, seq_num_mtx_ccid, dense_num_arr
    """
    global seq_num_scale_, seq_num_min_, num_static_scale_, num_static_min_, categorical_map, default_value_dict

    #     print('Check input. '
    #                   'Source: {}'.format(input_data))

    if not isinstance(input_data, dict):
        print(
            "Sanity check failed. Input data is not a dict obj. Source: {}".format(
                input_data
            )
        )
        return False, None, None, None, None, None, None

    input_data["objectId"] = "CURRENT"

    for VAR in input_data_dense_num_vars:
        if VAR not in input_data:
            print(
                "Sanity check failed. "
                "Input data does not contain required key in input_data_dense_num_vars. "
                "Source: {} {}".format(VAR, input_data)
            )
            return False, None, None, None, None, None, None

    for VAR in input_data_seq_cat_vars:
        if VAR not in input_data:
            print(
                "Sanity check failed. "
                "Input data does not contain required key in input_data_seq_cat_vars. "
                "Source: {} {}".format(VAR, input_data)
            )
            return False, None, None, None, None, None, None

    for VAR in input_data_seq_num_vars:
        if VAR not in input_data:
            print(
                "Sanity check failed. "
                "Input data does not contain required key in input_data_seq_num_vars. "
                "Source: {} {}".format(VAR, input_data)
            )
            return False, None, None, None, None, None, None

    objectid_otf_name_cid = "payment_risk.retail_order_cat_seq_by_cid.c_objectid_seq"
    ret_cid, seq_cat_mtx_cid, seq_num_mtx_cid = sequence_data_parsing(
        input_data,
        input_data_seq_cat_otf_vars_cid,
        input_data_seq_num_otf_vars_cid,
        objectid_otf_name_cid,
    )

    if ret_cid == False:
        return False, None, None, None, None, None, None

    objectid_otf_name_ccid = "payment_risk.retail_order_cat_seq_by_ccid.c_objectid_seq"
    ret_ccid, seq_cat_mtx_ccid, seq_num_mtx_ccid = sequence_data_parsing(
        input_data,
        input_data_seq_cat_otf_vars_ccid,
        input_data_seq_num_otf_vars_ccid,
        objectid_otf_name_ccid,
    )

    if ret_ccid == False:
        return False, None, None, None, None, None, None

    dense_num_vars_lst = arr_from_dict_fill_default(
        input_data, input_data_dense_num_vars, default_value_dict
    )
    dense_num_vars_lst = dense_num_vars_lst[:, :-2]
    #     dense_num_vars_lst[dense_num_vars_lst=='']='0'
    dense_num_vars_lst = dense_num_vars_lst.astype(float)
    dense_num_arr = dense_num_vars_lst * np.array(num_static_scale_) + np.array(
        num_static_min_
    )

    # print('Data sanity check passed and preprocessed.')
    return (
        True,
        seq_cat_mtx_cid.astype(np.int16),
        seq_num_mtx_cid,
        seq_cat_mtx_ccid.astype(np.int16),
        seq_num_mtx_ccid,
        dense_num_arr,
        arr_from_dict(input_data, ["IS_FRD"]),
    )


def data_parsing_old(input_data):
    """
    Data parsing and sanity check
    :param input_data: input data
    :return: pass_check, terminal_vec_lst, neighbor_vec_lst, neighbor_y_lst
    """
    global \
        seq_num_scale_, \
        seq_num_min_, \
        num_static_scale_, \
        num_static_min_, \
        categorical_map, \
        default_value_dict

    if "orderDate" not in input_data:
        print("orderDate missing. Check input. Source: {}".format(input_data))
        return False, None, None, None, None

    #     print('Check input. '
    #                   'Source: {}'.format(input_data))

    if not isinstance(input_data, dict):
        #         print('Sanity check failed. '
        #               'Input data is not a dict obj. '
        #               'Source: {}'.format(input_data))
        return False, None, None, None, None

    input_data["objectId"] = "CURRENT"

    for VAR in input_data_seq_cat_otf_vars:
        if VAR not in input_data:
            #             print('Sanity check failed. '
            #                   'Input data does not contain required key. '
            #                   'Source: {}'.format(input_data))
            return False, None, None, None, None

    for VAR in input_data_seq_cat_vars:
        if VAR not in input_data:
            #             print('Sanity check failed. '
            #                   'Input data does not contain required key. '
            #                   'Source: {}'.format(input_data))
            return False, None, None, None, None

    for VAR in input_data_seq_num_otf_vars:
        if VAR not in input_data:
            #             print('Sanity check failed. '
            #                   'Input data does not contain required key. '
            #                   'Source: {}'.format(input_data))
            return False, None, None, None, None

    for VAR in input_data_seq_num_vars:
        if VAR not in input_data:
            #             print('Sanity check failed. '
            #                   'Input data does not contain required key. '
            #                   'Source: {}'.format(input_data))
            return False, None, None, None, None

    for VAR in input_data_dense_num_vars:
        if VAR not in input_data:
            #             print('Sanity check failed. '
            #                   'Input data does not contain required key. '
            #                   'Source: {}'.format(input_data))
            return False, None, None, None, None

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
    return (
        True,
        seq_cat_mtx,
        seq_num_mtx,
        dense_num_arr,
        arr_from_dict(input_data, ["IS_FRD"]),
    )


import multiprocessing as mp
from multiprocessing import Pool, cpu_count


def parallel_data_parsing(df, num_workers=80):
    """
    Parallelize the data parsing process.

    Args:
        df: The pandas dataframe containing the "json" column.
        data_parsing: The function to parse the JSON data.
        num_workers: The number of worker processes to use.

    Returns:
        X_seq_cat_list, X_seq_num_list, X_num_list: Lists containing the parsed data.
    """

    df["json_str"] = df.apply(lambda x: x.to_json(), axis=1)

    X_seq_cat_cid_list = []
    X_seq_num_cid_list = []
    X_seq_cat_ccid_list = []
    X_seq_num_ccid_list = []
    X_num_list = []
    Y_list = []
    housekeeping_list = []

    print(f"Starting Pool of {num_workers}")
    pool = mp.Pool(processes=num_workers)
    results = []
    for row in df.itertuples(index=False):
        input_data = json.loads(row.json_str)
        results.append(pool.apply_async(data_parsing, args=(input_data,)))
    pool.close()
    pool.join()
    print(f"Pool closed")

    del df
    gc.collect()

    # X_seq_cat_cid_list = []
    # X_seq_cat_ccid_list = []
    # X_seq_num_list = []
    # X_num_list = []
    for result in results:
        #         seq_cat_mtx_cid, seq_num_mtx_cid, seq_cat_mtx_ccid, seq_num_mtx_ccid, dense_num_arr, arr_from_dict(input_data, ["IS_FRD"])
        (
            ret,
            seq_cat_mtx_cid,
            seq_num_mtx_cid,
            seq_cat_mtx_ccid,
            seq_num_mtx_ccid,
            dense_num_arr,
            y,
        ) = result.get()
        # ret, seq_cat_mtx, seq_num_mtx, dense_num_arr, y = result.get()
        if ret:
            if np.max(seq_num_mtx_cid[:, -2]) > 10000000:
                seq_num_mtx_cid[:, -2] = 10000000
            if np.min(seq_num_mtx_cid[:, -2]) < 0:
                continue
            if np.max(seq_num_mtx_ccid[:, -2]) > 10000000:
                seq_num_mtx_ccid[:, -2] = 10000000
            if np.min(seq_num_mtx_ccid[:, -2]) < 0:
                continue
            X_seq_cat_cid_list.append(seq_cat_mtx_cid)
            X_seq_num_cid_list.append(seq_num_mtx_cid)
            X_seq_cat_ccid_list.append(seq_cat_mtx_ccid)
            X_seq_num_ccid_list.append(seq_num_mtx_ccid)
            X_num_list.append(dense_num_arr)
            Y_list.append(y)

    train_X_seq_cat_cid = np.stack(X_seq_cat_cid_list, axis=0)
    train_X_seq_num_cid = np.stack(X_seq_num_cid_list, axis=0)
    train_X_seq_cat_ccid = np.stack(X_seq_cat_ccid_list, axis=0)
    train_X_seq_num_ccid = np.stack(X_seq_num_ccid_list, axis=0)
    train_X_num = np.concatenate(X_num_list, axis=0)
    train_Y = np.concatenate(Y_list, axis=0)

    #     return train_X_seq_cat, train_X_seq_num, train_X_num, train_Y
    return (
        train_X_seq_cat_cid,
        train_X_seq_num_cid,
        train_X_seq_cat_ccid,
        train_X_seq_num_ccid,
        train_X_num,
        train_Y,
    )


# Used for parallel data loading
from joblib import Parallel, delayed


def read_csv_(filename):
    return pd.read_csv(
        filename, sep="\t", low_memory=False, dtype=str, keep_default_na=False
    )


def processing_training_data_by_chunk(training_files, tag):
    ### Load input data
    print(len(training_files))
    print(training_files)

    if len(training_files) < cpu_count() - 9:
        num_jobs = len(training_files)
    else:
        num_jobs = -10

    df_list = Parallel(n_jobs=num_jobs)(delayed(read_csv_)(f) for f in training_files)
    train_data = pd.concat(df_list, ignore_index=True)

    #     train_data = pd.concat(
    #         [
    #             pd.read_csv(f, sep="\t", low_memory=False, dtype=str, keep_default_na=False)
    #             for f in sorted(glob.glob(os.path.join(training_data_path, "*.csv.gz")))
    #         ]
    #     )

    train_data["orderDate"] = pd.to_numeric(train_data["orderDate"], downcast="integer")
    train_data = train_data[train_data["orderDate"] > time.time() - 240 * 24 * 3600]
    # Holiday filter, will remove
    train_data = train_data[
        (train_data["orderDate"] >= 1701158400) | (train_data["orderDate"] < 1700726400)
    ]
    train_data = train_data[
        (train_data["orderDate"] >= 1697094000) | (train_data["orderDate"] < 1696921200)
    ]
    train_data = train_data[
        (train_data["orderDate"] >= 1689231600) | (train_data["orderDate"] < 1688972400)
    ]

    train_data.columns = [sub.replace("__DOT__", ".") for sub in train_data.columns]

    #     for i in config["var_list"]:
    #         if i in set(metadata.loc[~(metadata.iscategory.astype(bool)), "varname"]):
    #             train_data[i] = pd.to_numeric(train_data[i])
    print("chunk training data size: ", train_data.shape)

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
    print("chunk train data positive_rate:  ", positive_rate)

    if positive_rate < target_positive_rate:
        negative_downsampled_cnt = int(
            positive_count * (1 - target_positive_rate) / target_positive_rate
        )
        positive_downsampled_cnt = positive_count

    print(total_count, negative_count, positive_count, sep=",")

    train_data = pd.concat(
        [
            train_data[zeros_cond].sample(negative_downsampled_cnt),
            train_data[ones_cond].sample(positive_downsampled_cnt),
        ],
        ignore_index=True,
    )

    print("Chunk filtering and downsampling finished.")
    print(negative_downsampled_cnt, positive_downsampled_cnt, sep=",")

    ### data shape after processing
    print("Chunk Train data shape after preprocessing: {}".format(train_data.shape))

    ####### Convert dataframe to tensor ###############
    #     train_X_seq_cat, train_X_seq_num, train_X_num, train_Y = dataframe_to_tensor(
    #         train_data
    #     )
    #     train_X_seq_cat, train_X_seq_num, train_X_num, train_Y = parallel_data_parsing(
    #         train_data
    #     )
    (
        train_X_seq_cat_cid,
        train_X_seq_num_cid,
        train_X_seq_cat_ccid,
        train_X_seq_num_ccid,
        train_X_num,
        train_Y,
    ) = parallel_data_parsing(train_data)

    del train_data
    gc.collect()

    #     return train_X_seq_cat, train_X_seq_num, train_X_num, train_Y
    return (
        train_X_seq_cat_cid,
        train_X_seq_num_cid,
        train_X_seq_cat_ccid,
        train_X_seq_num_ccid,
        train_X_num,
        train_Y,
    )


def processing_calibration_data_by_chunk(calib_files, tag):
    if len(calib_files) < cpu_count() - 9:
        num_jobs = len(calib_files)
    else:
        num_jobs = -10

    df_list = Parallel(n_jobs=num_jobs)(delayed(read_csv_)(f) for f in calib_files)
    calib_data = pd.concat(df_list, ignore_index=True)

    #     calib_data = pd.concat(
    #         [
    #             pd.read_csv(f, sep="\t", low_memory=False, dtype=str, keep_default_na=False)
    #             for f in sorted(glob.glob(os.path.join(calibration_data_path, "*.csv.gz")))
    #         ]
    #     )

    calib_data["orderDate"] = pd.to_numeric(calib_data["orderDate"], downcast="integer")
    calib_data = calib_data[calib_data["orderDate"] > time.time() - 150 * 24 * 3600]
    # Holiday filter, will remove
    calib_data = calib_data[
        (calib_data["orderDate"] >= 1701158400) | (calib_data["orderDate"] < 1700726400)
    ]
    calib_data = calib_data[
        (calib_data["orderDate"] >= 1697094000) | (calib_data["orderDate"] < 1696921200)
    ]
    calib_data = calib_data[
        (calib_data["orderDate"] >= 1689231600) | (calib_data["orderDate"] < 1688972400)
    ]

    calib_data.columns = [sub.replace("__DOT__", ".") for sub in calib_data.columns]

    #     for i in config["var_list"]:
    #         if i in set(metadata.loc[~(metadata.iscategory.astype(bool)), "varname"]):
    #             calib_data[i] = pd.to_numeric(calib_data[i])
    print("chunk calibration data size: ", calib_data.shape)

    #     currency_path = "/opt/ml/processing/input/currency_variables/currency_variables.csv"
    #     curr_df = pd.read_csv(currency_path)
    #     print("currency variables shape: ", curr_df.shape)

    # Avoid NaN and -1 for calib_data tag to avoid error, note they will not be used to compute metric
    calib_data[tag] = pd.to_numeric(calib_data[tag], downcast="integer")
    calib_data.loc[calib_data[tag] == -1, tag] = 0
    calib_data[tag] = calib_data[tag].fillna(0)
    print("Chunk calib data shape after preprocessing: {}".format(calib_data.shape))

    ####### Convert dataframe to tensor ###############
    #     cali_X_seq_cat, cali_X_seq_num, cali_X_num, cali_Y = dataframe_to_tensor(calib_data)
    (
        cali_X_seq_cat_cid,
        cali_X_seq_num_cid,
        cali_X_seq_cat_ccid,
        cali_X_seq_num_ccid,
        cali_X_num,
        cali_Y,
    ) = parallel_data_parsing(calib_data)

    del calib_data
    gc.collect()

    return (
        cali_X_seq_cat_cid,
        cali_X_seq_num_cid,
        cali_X_seq_cat_ccid,
        cali_X_seq_num_ccid,
        cali_X_num,
        cali_Y,
    )


def processing_validation_data_by_chunk(test_files, tag):
    # Parallel dataloading

    if len(test_files) < cpu_count() - 9:
        num_jobs = len(test_files)
    else:
        num_jobs = -10

    df_list = Parallel(n_jobs=num_jobs)(delayed(read_csv_)(f) for f in test_files)
    test_data = pd.concat(df_list, ignore_index=True)

    #     test_data = pd.concat(
    #         [
    #             pd.read_csv(f, sep="\t", low_memory=False, dtype=str, keep_default_na=False)
    #             for f in sorted(glob.glob(os.path.join(testing_data_path, "*.csv.gz")))
    #         ]
    #     )

    test_data["orderDate"] = pd.to_numeric(test_data["orderDate"], downcast="integer")
    test_data = test_data[test_data["orderDate"] > time.time() - 90 * 24 * 3600]
    # Holiday filter, will remove
    test_data = test_data[
        (test_data["orderDate"] >= 1701158400) | (test_data["orderDate"] < 1700726400)
    ]
    test_data = test_data[
        (test_data["orderDate"] >= 1697094000) | (test_data["orderDate"] < 1696921200)
    ]
    test_data = test_data[
        (test_data["orderDate"] >= 1689231600) | (test_data["orderDate"] < 1688972400)
    ]

    test_data.columns = [sub.replace("__DOT__", ".") for sub in test_data.columns]
    #     for i in config["var_list"]:
    #         if i in set(metadata.loc[~(metadata.iscategory.astype(bool)), "varname"]):
    #             test_data[i] = pd.to_numeric(test_data[i])
    print("chunk test input data size: ", test_data.shape)

    test_data = test_data.loc[(~test_data[tag].isnull()) & (test_data[tag] != "-1"), :]
    test_data[tag] = pd.to_numeric(test_data[tag], downcast="integer")

    print("Chunk test data shape after preprocessing: {}".format(test_data.shape))

    ####### Convert dataframe to tensor ###############
    #     vali_X_seq_cat, vali_X_seq_num, vali_X_num, vali_Y = dataframe_to_tensor(test_data)
    (
        vali_X_seq_cat_cid,
        vali_X_seq_num_cid,
        vali_X_seq_cat_ccid,
        vali_X_seq_num_ccid,
        vali_X_num,
        vali_Y,
    ) = parallel_data_parsing(test_data)

    del test_data
    gc.collect()

    return (
        vali_X_seq_cat_cid,
        vali_X_seq_num_cid,
        vali_X_seq_cat_ccid,
        vali_X_seq_num_ccid,
        vali_X_num,
        vali_Y,
    )


def stream_save(A_list, A_name, dataset, out_dir="/opt/ml/processing/output", p="0"):
    # Create an empty array with the same shape and dtype as the final concatenated array
    final_shape = (sum(arr.shape[0] for arr in A_list),) + A_list[0].shape[1:]
    A = np.memmap(
        filename=os.path.join(out_dir, "{}_{}_v{}.raw".format(dataset, A_name, p)),
        dtype=A_list[0].dtype,
        mode="w+",
        shape=final_shape,
    )

    # Assign each chunk to the corresponding position in the memmap array
    current_position = 0
    for i in range(len(A_list)):
        chunk = A_list[i]
        A[current_position : current_position + chunk.shape[0], ...] = chunk
        current_position += chunk.shape[0]

        A.flush()

    for i in range(len(A_list)):
        os.remove(
            os.path.join(out_dir, "tmp/{}_{}_chunk_{}.npy".format(dataset, A_name, i))
        )

    np.save(
        file=os.path.join(out_dir, "{}_{}_v{}.npy".format(dataset, A_name, p)), arr=A
    )
    os.remove(os.path.join(out_dir, "{}_{}_v{}.raw".format(dataset, A_name, p)))
    print(f"{dataset} {A_name} shape {final_shape}")


def chunk_processing(data_path, dataset, num_chunk):
    print("Reading {} data from {}".format(dataset, data_path))

    # Parallel dataloading
    files = sorted(glob.glob(os.path.join(data_path, "*.csv.gz")))

    print(f"totol number of {dataset} files {len(files)}, with files ", files)

    chunk_size = int(len(files) / num_chunk) if len(files) >= num_chunk else 1

    num_chunk = math.ceil(len(files) / chunk_size)
    print(f"num_chunk is {num_chunk}")

    out_dir = "/opt/ml/processing/output" + "/" + "tmp"
    os.mkdir(out_dir)

    for i_chunk in range(num_chunk):
        show_memory(unit="MB", threshold=50)

        print(f"{dataset} chunk {i_chunk} start processing")
        start = i_chunk * chunk_size
        if (1 + i_chunk) * chunk_size > len(files):
            end = len(files)
        else:
            end = (1 + i_chunk) * chunk_size
        print("start, end", start, end)
        files_chunk = files[start:end]
        print("files_chunk", files_chunk)

        if dataset == "train":
            (
                X_seq_cat_cid_chunk,
                X_seq_num_cid_chunk,
                X_seq_cat_ccid_chunk,
                X_seq_num_ccid_chunk,
                X_num_chunk,
                Y_chunk,
            ) = processing_training_data_by_chunk(files_chunk, tag)
        elif dataset == "cali":
            (
                X_seq_cat_cid_chunk,
                X_seq_num_cid_chunk,
                X_seq_cat_ccid_chunk,
                X_seq_num_ccid_chunk,
                X_num_chunk,
                Y_chunk,
            ) = processing_calibration_data_by_chunk(files_chunk, tag)
        elif dataset == "vali":
            (
                X_seq_cat_cid_chunk,
                X_seq_num_cid_chunk,
                X_seq_cat_ccid_chunk,
                X_seq_num_ccid_chunk,
                X_num_chunk,
                Y_chunk,
            ) = processing_validation_data_by_chunk(files_chunk, tag)

        print(
            "X_seq_cat_cid_chunk, X_seq_num_cid_chunk, X_seq_cat_ccid_chunk, X_seq_num_ccid_chunk, X_num_chunk, Y_chunk shapes"
        )
        print(
            X_seq_cat_cid_chunk.shape,
            X_seq_num_cid_chunk.shape,
            X_seq_cat_ccid_chunk.shape,
            X_seq_num_ccid_chunk.shape,
            X_num_chunk.shape,
            Y_chunk.shape,
        )

        print(f"Y_chunk unique {np.unique(Y_chunk)}")
        Y_chunk = Y_chunk.astype(int)
        print(f"Y_chunk unique after astype {np.unique(Y_chunk)}")

        np.save(
            file=os.path.join(
                out_dir, "{}_cid_X_seq_cat_chunk_{}.npy".format(dataset, i_chunk)
            ),
            arr=X_seq_cat_cid_chunk,
        )
        np.save(
            file=os.path.join(
                out_dir, "{}_cid_X_seq_num_chunk_{}.npy".format(dataset, i_chunk)
            ),
            arr=X_seq_num_cid_chunk,
        )
        np.save(
            file=os.path.join(
                out_dir, "{}_ccid_X_seq_cat_chunk_{}.npy".format(dataset, i_chunk)
            ),
            arr=X_seq_cat_ccid_chunk,
        )
        np.save(
            file=os.path.join(
                out_dir, "{}_ccid_X_seq_num_chunk_{}.npy".format(dataset, i_chunk)
            ),
            arr=X_seq_num_ccid_chunk,
        )
        np.save(
            file=os.path.join(
                out_dir, "{}_X_num_chunk_{}.npy".format(dataset, i_chunk)
            ),
            arr=X_num_chunk,
        )
        np.save(
            file=os.path.join(out_dir, "{}_Y_chunk_{}.npy".format(dataset, i_chunk)),
            arr=Y_chunk,
        )
        print(f"{dataset} chunk {i_chunk} finished processing")
        print(f"Y_chunk number of 1: {np.sum(Y_chunk == 1)}")

    X_seq_cat_cid_chunk_list = []
    X_seq_num_cid_chunk_list = []
    X_seq_cat_ccid_chunk_list = []
    X_seq_num_ccid_chunk_list = []
    X_num_chunk_list = []
    Y_chunk_list = []
    for i_chunk in range(num_chunk):
        print("i_chunk", i_chunk)
        X_seq_cat_cid_chunk = np.load(
            os.path.join(
                out_dir, "{}_cid_X_seq_cat_chunk_{}.npy".format(dataset, i_chunk)
            ),
            mmap_mode="r",
        )
        X_seq_num_cid_chunk = np.load(
            os.path.join(
                out_dir, "{}_cid_X_seq_num_chunk_{}.npy".format(dataset, i_chunk)
            ),
            mmap_mode="r",
        )
        X_seq_cat_ccid_chunk = np.load(
            os.path.join(
                out_dir, "{}_ccid_X_seq_cat_chunk_{}.npy".format(dataset, i_chunk)
            ),
            mmap_mode="r",
        )
        X_seq_num_ccid_chunk = np.load(
            os.path.join(
                out_dir, "{}_ccid_X_seq_num_chunk_{}.npy".format(dataset, i_chunk)
            ),
            mmap_mode="r",
        )
        X_num_chunk = np.load(
            os.path.join(out_dir, "{}_X_num_chunk_{}.npy".format(dataset, i_chunk)),
            mmap_mode="r",
        )
        Y_chunk = np.load(
            os.path.join(out_dir, "{}_Y_chunk_{}.npy".format(dataset, i_chunk)),
            mmap_mode="r",
        )

        X_seq_cat_cid_chunk_list.append(X_seq_cat_cid_chunk)
        X_seq_num_cid_chunk_list.append(X_seq_num_cid_chunk)
        X_seq_cat_ccid_chunk_list.append(X_seq_cat_ccid_chunk)
        X_seq_num_ccid_chunk_list.append(X_seq_num_ccid_chunk)
        X_num_chunk_list.append(X_num_chunk)
        Y_chunk_list.append(Y_chunk)

    print("Y_chunk_list", Y_chunk_list)

    if len(hosts) > 1:
        out_dir = "/opt/ml/processing/output"
        stream_save(
            X_seq_cat_cid_chunk_list,
            "cid_X_seq_cat",
            dataset,
            out_dir,
            "0_" + current_host,
        )
        stream_save(
            X_seq_num_cid_chunk_list,
            "cid_X_seq_num",
            dataset,
            out_dir,
            "0_" + current_host,
        )
        stream_save(
            X_seq_cat_ccid_chunk_list,
            "ccid_X_seq_cat",
            dataset,
            out_dir,
            "0_" + current_host,
        )
        stream_save(
            X_seq_num_ccid_chunk_list,
            "ccid_X_seq_num",
            dataset,
            out_dir,
            "0_" + current_host,
        )
        stream_save(X_num_chunk_list, "X_num", dataset, out_dir, "0_" + current_host)
        stream_save(Y_chunk_list, "Y", dataset, out_dir, "0_" + current_host)
    else:
        stream_save(X_seq_cat_cid_chunk_list, "cid_X_seq_cat", dataset)
        stream_save(X_seq_num_cid_chunk_list, "cid_X_seq_num", dataset)
        stream_save(X_seq_cat_ccid_chunk_list, "ccid_X_seq_cat", dataset)
        stream_save(X_seq_num_ccid_chunk_list, "ccid_X_seq_num", dataset)
        stream_save(X_num_chunk_list, "X_num", dataset)
        stream_save(Y_chunk_list, "Y", dataset)

    out_dir = "/opt/ml/processing/output" + "/" + "tmp"
    try:
        shutil.rmtree(out_dir)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    return


def show_memory(unit="GB", threshold=1):
    from sys import getsizeof

    scale = {"B": 1, "KB": 1024, "MB": 1048576, "GB": 1073741824}[unit]
    for i in list(globals().keys()):
        memory = eval("getsizeof({})".format(i)) // scale
        if memory >= threshold:
            print(i, memory)
