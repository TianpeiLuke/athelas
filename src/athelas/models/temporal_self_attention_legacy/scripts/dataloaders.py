import os
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from datasets import OrderSeqDataset, TwoOrderSeqDataset


def load_data_single_seq(args, batch_size, data_version):
    # Try with a toy example. For training the model, you need to load full size of data
    if args.use_small_batch:
        N = 20000
        # ###### load 2*N data points for each dataset
        x_engineered_train = np.load(
            os.path.join(f"{args.train_data_folder}_X_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        x_seq_num_train = np.load(
            os.path.join(f"{args.train_data_folder}_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        x_seq_cat_train = np.load(
            os.path.join(f"{args.train_data_folder}_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]

        x_engineered_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_X_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        x_seq_num_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        x_seq_cat_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]

        x_engineered_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_X_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        x_seq_num_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        x_seq_cat_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]

        train_Y = np.load(
            os.path.join(f"{args.train_data_folder}_Y_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        vali_Y = np.load(
            os.path.join(f"{args.vali_data_folder}_Y_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        cali_Y = np.load(
            os.path.join(f"{args.cali_data_folder}_Y_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
    else:
        # ###### load full size of data
        x_engineered_train = np.load(
            os.path.join(f"{args.train_data_folder}_X_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        x_seq_num_train = np.load(
            os.path.join(f"{args.train_data_folder}_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        x_seq_cat_train = np.load(
            os.path.join(f"{args.train_data_folder}_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )

        x_engineered_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_X_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        x_seq_num_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        x_seq_cat_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )

        x_engineered_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_X_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        x_seq_num_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        x_seq_cat_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )

        train_Y = np.load(
            os.path.join(f"{args.train_data_folder}_Y_{data_version}.npy"),
            mmap_mode="r+",
        )
        vali_Y = np.load(
            os.path.join(f"{args.vali_data_folder}_Y_{data_version}.npy"),
            mmap_mode="r+",
        )
        cali_Y = np.load(
            os.path.join(f"{args.cali_data_folder}_Y_{data_version}.npy"),
            mmap_mode="r+",
        )
    # Sanity check fo cali data tag, they will not be used for evaluation
    if len(np.unique(vali_Y)) > 2:
        cali_Y = np.nan_to_num(cali_Y)
        cali_Y[cali_Y == "nan"] = -1
        cali_Y[cali_Y == "NaN"] = -1
        cali_Y = cali_Y.astype(int)
        cali_Y[cali_Y != 1] = 0

    if len(train_Y.shape) == 1:
        train_Y = np.expand_dims(train_Y, axis=1)
        vali_Y = np.expand_dims(vali_Y, axis=1)
        cali_Y = np.expand_dims(cali_Y, axis=1)

    train_dataset = OrderSeqDataset(
        x_seq_cat_mtx=x_seq_cat_train,
        x_seq_num_mtx=x_seq_num_train,
        y_array=train_Y,
        x_engineered_num_mtx=x_engineered_train,
    )

    vali_dataset = OrderSeqDataset(
        x_seq_cat_mtx=x_seq_cat_vali,
        x_seq_num_mtx=x_seq_num_vali,
        y_array=vali_Y,
        x_engineered_num_mtx=x_engineered_vali,
    )
    cali_dataset = OrderSeqDataset(
        x_seq_cat_mtx=x_seq_cat_cali,
        x_seq_num_mtx=x_seq_num_cali,
        y_array=cali_Y,
        x_engineered_num_mtx=x_engineered_cali,
    )

    if args.local_rank >= 0:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            pin_memory=True,
        )

        vali_sampler = torch.utils.data.distributed.DistributedSampler(
            vali_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )
        vali_dataloader = DataLoader(
            dataset=vali_dataset,
            batch_size=batch_size,
            shuffle=(vali_sampler is None),
            sampler=vali_sampler,
            pin_memory=True,
        )
        cali_sampler = torch.utils.data.distributed.DistributedSampler(
            cali_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )
        cali_dataloader = DataLoader(
            dataset=cali_dataset,
            batch_size=batch_size,
            shuffle=(cali_sampler is None),
            sampler=cali_sampler,
            pin_memory=True,
        )
    else:
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        vali_dataloader = DataLoader(
            dataset=vali_dataset, batch_size=batch_size, shuffle=False
        )
        cali_dataloader = DataLoader(
            dataset=cali_dataset, batch_size=batch_size, shuffle=False
        )

    if args.local_rank >= 0:
        return (
            train_sampler,
            train_dataloader,
            vali_sampler,
            vali_dataloader,
            cali_sampler,
            cali_dataloader,
        )
    else:
        return train_dataloader, vali_dataloader, cali_dataloader


def load_data_two_seq(args, batch_size, data_version):
    # Try with a toy example. For training the model, you need to load full size of data
    if args.use_small_batch:
        N = 20000
        # ###### load 2*N data points for each dataset
        x_engineered_num_train = np.load(
            os.path.join(f"{args.train_data_folder}_X_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        cid_x_seq_num_train = np.load(
            os.path.join(f"{args.train_data_folder}_cid_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        cid_x_seq_cat_train = np.load(
            os.path.join(f"{args.train_data_folder}_cid_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]

        x_engineered_num_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_X_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        cid_x_seq_num_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_cid_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        cid_x_seq_cat_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_cid_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]

        x_engineered_num_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_X_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        cid_x_seq_num_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_cid_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        cid_x_seq_cat_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_cid_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]

        ccid_x_seq_num_train = np.load(
            os.path.join(f"{args.train_data_folder}_ccid_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        ccid_x_seq_cat_train = np.load(
            os.path.join(f"{args.train_data_folder}_ccid_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]

        ccid_x_seq_num_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_ccid_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        ccid_x_seq_cat_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_ccid_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]

        ccid_x_seq_num_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_ccid_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        ccid_x_seq_cat_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_ccid_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]

        train_Y = np.load(
            os.path.join(f"{args.train_data_folder}_Y_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        vali_Y = np.load(
            os.path.join(f"{args.vali_data_folder}_Y_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
        cali_Y = np.load(
            os.path.join(f"{args.cali_data_folder}_Y_{data_version}.npy"),
            mmap_mode="r+",
        )[: 2 * N]
    else:
        # ###### load full size of data
        x_engineered_num_train = np.load(
            os.path.join(f"{args.train_data_folder}_X_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        cid_x_seq_num_train = np.load(
            os.path.join(f"{args.train_data_folder}_cid_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        cid_x_seq_cat_train = np.load(
            os.path.join(f"{args.train_data_folder}_cid_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )

        x_engineered_num_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_X_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        cid_x_seq_num_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_cid_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        cid_x_seq_cat_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_cid_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )

        x_engineered_num_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_X_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        cid_x_seq_num_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_cid_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        cid_x_seq_cat_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_cid_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )

        ccid_x_seq_num_train = np.load(
            os.path.join(f"{args.train_data_folder}_ccid_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        ccid_x_seq_cat_train = np.load(
            os.path.join(f"{args.train_data_folder}_ccid_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )

        ccid_x_seq_num_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_ccid_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        ccid_x_seq_cat_vali = np.load(
            os.path.join(f"{args.vali_data_folder}_ccid_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )

        ccid_x_seq_num_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_ccid_X_seq_num_{data_version}.npy"),
            mmap_mode="r+",
        )
        ccid_x_seq_cat_cali = np.load(
            os.path.join(f"{args.cali_data_folder}_ccid_X_seq_cat_{data_version}.npy"),
            mmap_mode="r+",
        )

        train_Y = np.load(
            os.path.join(f"{args.train_data_folder}_Y_{data_version}.npy"),
            mmap_mode="r+",
        )
        vali_Y = np.load(
            os.path.join(f"{args.vali_data_folder}_Y_{data_version}.npy"),
            mmap_mode="r+",
        )
        cali_Y = np.load(
            os.path.join(f"{args.cali_data_folder}_Y_{data_version}.npy"),
            mmap_mode="r+",
        )

    # Sanity check fo cali data tag, they will not be used for evaluation
    if len(np.unique(vali_Y)) > 2:
        cali_Y = np.nan_to_num(cali_Y)
        cali_Y[cali_Y == "nan"] = -1
        cali_Y[cali_Y == "NaN"] = -1
        cali_Y = cali_Y.astype(int)
        cali_Y[cali_Y != 1] = 0

    if len(train_Y.shape) == 1:
        train_Y = np.expand_dims(train_Y, axis=1)
        vali_Y = np.expand_dims(vali_Y, axis=1)
        cali_Y = np.expand_dims(cali_Y, axis=1)

    train_dataset = TwoOrderSeqDataset(
        x_seq_cat_mtx_cid=cid_x_seq_cat_train,
        x_seq_num_mtx_cid=cid_x_seq_num_train,
        x_seq_cat_mtx_ccid=ccid_x_seq_cat_train,
        x_seq_num_mtx_ccid=ccid_x_seq_num_train,
        y_array=train_Y,
        x_engineered_num_mtx=x_engineered_num_train,
    )
    vali_dataset = TwoOrderSeqDataset(
        x_seq_cat_mtx_cid=cid_x_seq_cat_vali,
        x_seq_num_mtx_cid=cid_x_seq_num_vali,
        x_seq_cat_mtx_ccid=ccid_x_seq_cat_vali,
        x_seq_num_mtx_ccid=ccid_x_seq_num_vali,
        y_array=vali_Y,
        x_engineered_num_mtx=x_engineered_num_vali,
    )
    cali_dataset = TwoOrderSeqDataset(
        x_seq_cat_mtx_cid=cid_x_seq_cat_cali,
        x_seq_num_mtx_cid=cid_x_seq_num_cali,
        x_seq_cat_mtx_ccid=ccid_x_seq_cat_cali,
        x_seq_num_mtx_ccid=ccid_x_seq_num_cali,
        y_array=cali_Y,
        x_engineered_num_mtx=x_engineered_num_cali,
    )

    if args.local_rank >= 0:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )
        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            pin_memory=True,
        )

        vali_sampler = torch.utils.data.distributed.DistributedSampler(
            vali_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )
        vali_dataloader = DataLoader(
            dataset=vali_dataset,
            batch_size=batch_size,
            shuffle=(vali_sampler is None),
            sampler=vali_sampler,
            pin_memory=True,
        )
        cali_sampler = torch.utils.data.distributed.DistributedSampler(
            cali_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank()
        )
        cali_dataloader = DataLoader(
            dataset=cali_dataset,
            batch_size=batch_size,
            shuffle=(cali_sampler is None),
            sampler=cali_sampler,
            pin_memory=True,
        )
    else:
        train_dataloader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        vali_dataloader = DataLoader(
            dataset=vali_dataset, batch_size=batch_size, shuffle=False
        )
        cali_dataloader = DataLoader(
            dataset=cali_dataset, batch_size=batch_size, shuffle=False
        )

    if args.local_rank >= 0:
        return (
            train_sampler,
            train_dataloader,
            vali_sampler,
            vali_dataloader,
            cali_sampler,
            cali_dataloader,
        )
    else:
        return train_dataloader, vali_dataloader, cali_dataloader
