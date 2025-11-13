import os
import sys
import gc
import shutil
import glob
import pickle as pkl
import time
import typing
from datetime import datetime
from typing import Dict, Any
import numpy as np
import pandas as pd
import math
import json
import pickle

config_path = "/opt/ml/processing/input/config"
sys.path.append(config_path + "/scripts/")


def get_remaining_space(path="/"):
    statvfs = os.statvfs(path)
    block_size = statvfs.f_frsize  # Fundamental file system block size
    remaining_space = statvfs.f_bavail * block_size
    return remaining_space


def get_files_in_folder(folder_path):
    files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, file))
    ]
    return files


def get_file_size(file_path):
    return os.path.getsize(file_path)


def delete_files_and_release_space(file_paths):
    for file_path in file_paths:
        try:
            with open(file_path, "w"):  # Open the file for writing (truncate it)
                pass
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except OSError as e:
            print(f"Error deleting {file_path}: {e}")


def simple_save(A_list, A_name, dataset, out_dir="/opt/ml/processing/output", p="0"):
    data_dir = "/opt/ml/processing/input/training"

    # Create an empty array with the same shape and dtype as the final concatenated array
    A = np.concatenate(A_list, axis=0)

    remaining_space = get_remaining_space(data_dir)
    print(f"Remaining space on {data_dir}: {remaining_space / 1024 / 1024 / 1024} GBs")

    remaining_space = get_remaining_space(out_dir)
    print(f"Remaining space on {out_dir}: {remaining_space / 1024 / 1024 / 1024} GBs")

    i = 1
    for A_i in A_list:
        print(
            "remove",
            os.path.join(data_dir, "{}_{}_v0_algo-{}.npy".format(dataset, A_name, i)),
        )
        del A_i
        delete_files_and_release_space(
            [os.path.join(data_dir, "{}_{}_v0_algo-{}.npy".format(dataset, A_name, i))]
        )
        i += 1
    #         os.remove(os.path.join(data_dir, "{}_{}_v0_algo-{}.npy".format(dataset, A_name, i)))

    remaining_space = get_remaining_space(data_dir)
    print(f"Remaining space on {data_dir}: {remaining_space / 1024 / 1024 / 1024} GBs")

    remaining_space = get_remaining_space(out_dir)
    print(f"Remaining space on {out_dir}: {remaining_space / 1024 / 1024 / 1024} GBs")

    np.save(
        file=os.path.join(out_dir, "{}_{}_v{}.npy".format(dataset, A_name, p)), arr=A
    )

    print(f"{dataset} {A_name} shape {A.shape}")

    print(f"files in {data_dir}: ")
    print(os.listdir(data_dir))

    files_to_check = get_files_in_folder(data_dir)
    for file_path in files_to_check:
        file_size = get_file_size(file_path)
        print(f"Size of {file_path}: {file_size / 1024 / 1024 / 1024} GBs")

    print(f"files in {out_dir}: ")
    print(os.listdir(out_dir))

    files_to_check = get_files_in_folder(out_dir)
    for file_path in files_to_check:
        file_size = get_file_size(file_path)
        print(f"Size of {file_path}: {file_size / 1024 / 1024 / 1024} GBs")


if __name__ == "__main__":
    ### Input ########################################
    data_dir = "/opt/ml/processing/input/training"

    dataset = "train"

    X_seq_cat_cid_chunk_list = []
    X_seq_num_cid_chunk_list = []
    X_seq_cat_ccid_chunk_list = []
    X_seq_num_ccid_chunk_list = []
    X_num_chunk_list = []
    Y_chunk_list = []
    for i in range(1, 3):
        print("i", i)

        X_seq_cat_cid_chunk = np.load(
            os.path.join(
                data_dir, "{}_cid_X_seq_cat_v0_algo-{}.npy".format(dataset, i)
            ),
            mmap_mode="r",
        )
        X_seq_num_cid_chunk = np.load(
            os.path.join(
                data_dir, "{}_cid_X_seq_num_v0_algo-{}.npy".format(dataset, i)
            ),
            mmap_mode="r",
        )
        X_seq_cat_ccid_chunk = np.load(
            os.path.join(
                data_dir, "{}_ccid_X_seq_cat_v0_algo-{}.npy".format(dataset, i)
            ),
            mmap_mode="r",
        )
        X_seq_num_ccid_chunk = np.load(
            os.path.join(
                data_dir, "{}_ccid_X_seq_num_v0_algo-{}.npy".format(dataset, i)
            ),
            mmap_mode="r",
        )
        X_num_chunk = np.load(
            os.path.join(data_dir, "{}_X_num_v0_algo-{}.npy".format(dataset, i)),
            mmap_mode="r",
        )
        Y_chunk = np.load(
            os.path.join(data_dir, "{}_Y_v0_algo-{}.npy".format(dataset, i)),
            mmap_mode="r",
        )

        X_seq_cat_cid_chunk_list.append(X_seq_cat_cid_chunk)
        X_seq_num_cid_chunk_list.append(X_seq_num_cid_chunk)
        X_seq_cat_ccid_chunk_list.append(X_seq_cat_ccid_chunk)
        X_seq_num_ccid_chunk_list.append(X_seq_num_ccid_chunk)
        X_num_chunk_list.append(X_num_chunk)
        Y_chunk_list.append(Y_chunk)

    simple_save(X_seq_cat_cid_chunk_list, "cid_X_seq_cat", dataset)
    simple_save(X_seq_num_cid_chunk_list, "cid_X_seq_num", dataset)
    simple_save(X_seq_cat_ccid_chunk_list, "ccid_X_seq_cat", dataset)
    simple_save(X_seq_num_ccid_chunk_list, "ccid_X_seq_num", dataset)
    simple_save(X_num_chunk_list, "X_num", dataset)
    simple_save(Y_chunk_list, "Y", dataset)

    print("Finished merging")
