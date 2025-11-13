import os
import random
import torch
import numpy as np
import math
from torch.utils.data import Dataset, IterableDataset


# Dataset
class OrderSeqDataset(Dataset):
    """
    For single sequence
    """

    def __init__(
        self, x_seq_cat_mtx, x_seq_num_mtx, y_array, x_engineered_num_mtx=None
    ):
        self.x_seq_cat_mtx = x_seq_cat_mtx
        self.x_seq_num_mtx = x_seq_num_mtx
        self.y_array = y_array
        self.x_engineered_num_mtx = x_engineered_num_mtx

    def __getitem__(self, idx: int):
        x_seq_cat = torch.Tensor(self.x_seq_cat_mtx[idx])
        x_seq_num = torch.Tensor(self.x_seq_num_mtx[idx])
        label = torch.Tensor(np.array(self.y_array[idx]))

        # not_padded is at the last column and not min/max normalized
        # indicating whether the event is not padded
        not_padded = x_seq_num[:, -1]

        # time_to_last is at the second last column and not min/max normalized
        # Adding clone() because using mmap_mode="r+" when reading a NumPy array,
        # This avoids affecting the original NumPy array in disk.
        time_to_last = x_seq_num[:, -2].clone().unsqueeze(dim=1)

        if self.x_engineered_num_mtx is not None:
            x_engineered_num = self.x_engineered_num_mtx[idx]

            return {
                "x_seq_cat": x_seq_cat,
                "x_seq_num": x_seq_num[:, :-2],
                "y": label,
                "x_engineered_num": x_engineered_num,
                "time_to_last": time_to_last,
                "not_padded": not_padded,
            }
        else:
            return {
                "x_seq_cat": x_seq_cat,
                "x_seq_num": x_seq_num[:, :-2],
                "y": label,
                "time_to_last": time_to_last,
                "not_padded": not_padded,
            }

    def __len__(self):
        return len(self.y_array)


class TwoOrderSeqDataset(Dataset):
    """
    For two sequences, cid is short for customerId and ccid is short for creditCardIds
    """

    def __init__(
        self,
        x_seq_cat_mtx_cid,
        x_seq_num_mtx_cid,
        x_seq_cat_mtx_ccid,
        x_seq_num_mtx_ccid,
        y_array,
        x_engineered_num_mtx=None,
    ):
        self.x_seq_cat_mtx_cid = x_seq_cat_mtx_cid
        self.x_seq_num_mtx_cid = x_seq_num_mtx_cid
        self.x_seq_cat_mtx_ccid = x_seq_cat_mtx_ccid
        self.x_seq_num_mtx_ccid = x_seq_num_mtx_ccid
        self.y_array = y_array
        self.x_engineered_num_mtx = x_engineered_num_mtx

    def __getitem__(self, idx: int):
        x_seq_cat_cid = torch.Tensor(self.x_seq_cat_mtx_cid[idx])
        x_seq_num_cid = torch.Tensor(self.x_seq_num_mtx_cid[idx])
        x_seq_cat_ccid = torch.Tensor(self.x_seq_cat_mtx_ccid[idx])
        x_seq_num_ccid = torch.Tensor(self.x_seq_num_mtx_ccid[idx])
        label = torch.Tensor(np.array(self.y_array[idx]))

        # not_padded is at the last column and not min/max normalized
        # indicating whether the event is not padded
        not_padded_cid = x_seq_num_cid[:, -1]
        not_padded_ccid = x_seq_num_ccid[:, -1]

        # time_to_last is at the second last column and not min/max normalized
        # Adding clone() because using mmap_mode="r+" when reading a NumPy array,
        # This avoids affecting the original NumPy array in disk.
        time_to_last_cid = x_seq_num_cid[:, -2].clone().unsqueeze(dim=1)
        time_to_last_ccid = x_seq_num_ccid[:, -2].clone().unsqueeze(dim=1)

        if self.x_engineered_num_mtx is not None:
            x_engineered_num = self.x_engineered_num_mtx[idx]

            return {
                "x_seq_cat_cid": x_seq_cat_cid,
                "x_seq_num_cid": x_seq_num_cid[:, :-2],
                "time_to_last_cid": time_to_last_cid,
                "not_padded_cid": not_padded_cid,
                "x_seq_cat_ccid": x_seq_cat_ccid,
                "x_seq_num_ccid": x_seq_num_ccid[:, :-2],
                "time_to_last_ccid": time_to_last_ccid,
                "not_padded_ccid": not_padded_ccid,
                "y": label,
                "x_engineered_num": x_engineered_num,
            }
        else:
            return {
                "x_seq_cat_cid": x_seq_cat_cid,
                "x_seq_num_cid": x_seq_num_cid[:, :-2],
                "time_to_last_cid": time_to_last_cid,
                "not_padded_cid": not_padded_cid,
                "x_seq_cat_ccid": x_seq_cat_ccid,
                "x_seq_num_ccid": x_seq_num_ccid[:, :-2],
                "time_to_last_ccid": time_to_last_ccid,
                "not_padded_ccid": not_padded_ccid,
                "y": label,
            }

    def __len__(self):
        return len(self.y_array)


# IterableDataset
class MultiWorkerOrderSeqIterableDataset(IterableDataset):
    """
    For streaming use case, e.g., no need to copy and paste the entire dataset to EBS when launching a sagemaker
    training job, data will be fetched as needed.
    Data in a two level folder structure:
        root_dir
        - event_folders (e.g., customerId that indicating certain customer's history)
          - event_folder (e.g., orderDate which is the epoch timestamp for customer placing the order,
                                indicating different target event)
            - x_seq_cat.npy
            - x_seq_num.npy
            - x_engineered_num.npy
            - y.npy
    """

    def __init__(self, root_dir, transform=None):
        super(MultiWorkerOrderSeqIterableDataset).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.event_folders = self.find_subfolders(self.root_dir)
        random.shuffle(self.event_folders)  # Shuffle the subfolder paths
        self.start = 0
        self.end = len(self.event_folders)

    def find_subfolders(self, root_dir):
        subfolders = []

        # Loop through first level of subfolders
        for entry in os.scandir(root_dir):
            if entry.is_dir():
                subfolder_path = entry.path

                # Loop through second level subfolders
                for subentry in os.scandir(subfolder_path):
                    if subentry.is_dir():
                        subfolders.append(subentry.path)

        return subfolders

    def data_generator(self, start, end):
        for i, event_folder_path in enumerate(self.event_folders):
            if i < start:
                continue
            if i >= end:
                return StopIteration()

            # File names can be changed accordingly
            x_seq_cat_mtx = np.load(os.path.join(event_folder_path, "x_seq_cat.npy"))
            x_seq_num_mtx = np.load(os.path.join(event_folder_path, "x_seq_num.npy"))
            x_engineered_num_mtx = np.load(
                os.path.join(event_folder_path, "x_engineered_num.npy")
            )
            y_array = np.load(os.path.join(event_folder_path, "y.npy"))

            x_seq_cat = torch.Tensor(x_seq_cat_mtx)
            x_seq_num = torch.Tensor(x_seq_num_mtx)
            x_engineered_num = torch.Tensor(x_engineered_num_mtx)
            label = torch.Tensor(y_array)

            # not_padded is at the last column and not min/max normalized
            # indicating whether the event is not padded
            not_padded = x_seq_num[:, -1]

            # time_to_last is at the second last column and not min/max normalized
            # Adding clone() because using mmap_mode="r+" when reading a NumPy array,
            # This avoids affecting the original NumPy array in disk.
            time_to_last = x_seq_num[:, -2].clone().unsqueeze(dim=1)

            data_point = {
                "x_seq_cat": x_seq_cat,
                "x_seq_num": x_seq_num[:, :-2],
                "y": label,
                "x_engineered_num": x_engineered_num,
                "time_to_last": time_to_last,
                "not_padded": not_padded,
            }

            if self.transform:
                data_point = self.transform(data_point)
            yield data_point

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(
                math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

        return self.data_generator(iter_start, iter_end)
