import argparse
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle as pkl
import time
import torch
from scipy.special import softmax, expit

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# import self written python files
from models import (
    OrderFeatureAttentionClassifier,
    TwoSeqMoEOrderFeatureAttentionClassifier,
)
from pytorchtools import EarlyStopping
from utilities import (
    create_optimizer,
    train,
    evaluation_single_seq,
    print_performance,
    seed_everything,
    timeSince,
)
from dataloaders import load_data_single_seq, load_data_two_seq


def get_parser():
    parser = argparse.ArgumentParser(description="Temporal Self Atttention")

    # MultiHeadAttention
    parser.add_argument(
        "--dim_attn_feedforward",
        type=int,
        default=64,
        help="feedforward dimension for multi-head attention",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=1,
        help="number of heads for multi-head attention",
    )
    parser.add_argument(
        "--n_layers_order",
        type=int,
        default=6,
        help="number of layers for multi-head order attention",
    )
    parser.add_argument(
        "--n_layers_feature",
        type=int,
        default=6,
        help="number of layers for multi-head feature attention",
    )

    # Training
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        metavar="B",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=70,
        metavar="EPOCHS",
        help="number of epochs to train (default: 70)",
    )
    parser.add_argument("--seed", type=int, default=0, metavar="SD", help="random seed")
    parser.add_argument(
        "--optim",
        default="adam",
        type=str,
        help="optimizer",
        choices=["sgd", "rmsprop", "adam"],
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="learning rate (default: 0.00001)",
    )
    parser.add_argument(
        "--scheduler_maxlr",
        type=float,
        default=1e-3,
        metavar="SMLR",
        help="schedule max learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="weight decay for optimizers"
    )
    parser.add_argument(
        "--beta1", default=0.9, type=float, help="Adam coefficients beta_1"
    )
    parser.add_argument(
        "--beta2", default=0.999, type=float, help="Adam coefficients beta_2"
    )

    # Embedding and data
    parser.add_argument(
        "--n_cat_features",
        type=int,
        default=53,
        metavar="NCF",
        help="number of categorical features for sequence",
    )
    parser.add_argument(
        "--n_num_features",
        type=int,
        default=47,
        metavar="NNF",
        help="number of numerical features for sequence",
    )
    parser.add_argument(
        "--n_embedding",
        type=int,
        default=1352,
        metavar="NE",
        help="size of feature embedding lookup table",
    )
    parser.add_argument(
        "--dim_embedding_table",
        type=int,
        default=128,
        help="embedding lookup table dimension",
    )
    parser.add_argument(
        "--seq_len", type=int, default=51, metavar="SL", help="Order sequence length"
    )
    parser.add_argument(
        "--n_classes",
        type=int,
        default=2,
        metavar="NC",
        help="number of traget classes",
    )
    parser.add_argument(
        "--n_engineered_num_features",
        type=int,
        default=0,
        metavar="DS",
        help="number of static features to use",
    )
    parser.add_argument(
        "--emb_tbl_use_bias",
        type=int,
        default=1,
        metavar="UB",
        help="whether to use use_bias for feature embedding",
    )
    parser.add_argument("--dropout", default=0.1, type=float, help="dropout parameter")
    parser.add_argument(
        "--use_small_batch",
        type=int,
        default=0,
        metavar="US",
        help="whether to use small dataset for quick testing",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=0,
        metavar="SE",
        help="unix time when run the exp",
    )
    parser.add_argument(
        "--use_time_seq",
        type=int,
        default=1,
        metavar="UT",
        help="whether to use time sequence",
    )
    parser.add_argument(
        "--use_mlp", type=int, default=1, metavar="UM", help="whether to mlp block"
    )
    parser.add_argument(
        "--modelname",
        default="OrderFeature",
        type=str,
        help="modelname",
        choices=["OrderFeature", "TwoSeqMoEOrderFeature"],
    )
    parser.add_argument(
        "--test_name", default="order-attention", type=str, help="test_name"
    )

    parser.add_argument(
        "--loss",
        default="CrossEntropyLoss",
        type=str,
        help="loss",
    )
    parser.add_argument(
        "--loss_gamma", type=float, default=2, metavar="LG", help="gamma for focal loss"
    )
    parser.add_argument(
        "--loss_alpha",
        type=float,
        default=0.25,
        metavar="LA",
        help="gamma for focal loss",
    )
    parser.add_argument(
        "--cyclical_factor",
        type=float,
        default=2,
        help="1->Modified focal loss, 2->Cyclical focal loss (default=2)",
    )
    parser.add_argument(
        "--gamma_hc",
        type=float,
        default=0,
        help="Cyclical focal loss gamma (default=0)",
    )
    parser.add_argument(
        "--gamma_pos",
        type=float,
        default=0,
        help="Asymetric focal loss positive gamma (default=0)",
    )
    parser.add_argument(
        "--gamma_neg",
        type=float,
        default=4,
        help="Asymetric focal loss negative gamma (default=4)",
    )
    parser.add_argument(
        "--loss_reduction", default="mean", type=str, help="loss_reduction"
    )
    parser.add_argument("--data_version", default="v0", type=str, help="data_cersion")
    parser.add_argument(
        "--load_model",
        type=int,
        default=0,
        metavar="LM",
        help="whether to load pretrained model",
    )
    parser.add_argument(
        "--model_path", default="N/A", type=str, help="path of pretrained model"
    )
    parser.add_argument("--steps_per_epoch", type=int, default=2000, metavar="SPE")
    parser.add_argument(
        "--patience", type=int, default=10, metavar="patience for early stopping"
    )

    parser.add_argument(
        "--use_moe",
        type=int,
        default=1,
        metavar="UMOE",
        help="whether to use mixture of experts inside transformer layer",
    )
    parser.add_argument(
        "--num_experts",
        type=int,
        default=1,
        metavar="NE",
        help="number of experts to use",
    )
    parser.add_argument(
        "--use_amp",
        type=int,
        default=0,
        metavar="UAMP",
        help="whether to use automatic mixed precision",
    )

    parser.add_argument(
        "--num_seq",
        default="1",
        type=str,
        help="number of sequence to use",
        choices=["1", "2"],
    )

    # DDP for sagemaker training job
    parser.add_argument("--local_rank", default=int(os.environ["LOCAL_RANK"]), type=int)
    # Data, model, and output directories
    parser.add_argument(
        "--result_dir", type=str, default=os.getenv("SM_OUTPUT_DATA_DIR")
    )
    parser.add_argument("--model_dir", type=str, default=os.getenv("SM_MODEL_DIR"))
    # Data channels is for SageMaker training job invoked within both MODS pipeline and local notebook
    parser.add_argument(
        "--data_dir", type=str, default=os.getenv("SM_INPUT_DIR") + "/data"
    )
    parser.add_argument("--train", type=str, default=os.getenv("SM_CHANNEL_TRAIN"))
    parser.add_argument("--vali", type=str, default=os.getenv("SM_CHANNEL_VALI"))
    parser.add_argument("--cali", type=str, default=os.getenv("SM_CHANNEL_CALI"))

    return parser


def load_data(args, batch_size, data_version):
    if args.num_seq == "1":
        return load_data_single_seq(args, batch_size, data_version)
    elif args.num_seq == "2":
        return load_data_two_seq(args, batch_size, data_version)


def build_model(args, modelname):
    if modelname == "OrderFeature":
        model = OrderFeatureAttentionClassifier(
            args.n_cat_features,
            args.n_num_features,
            args.n_classes,
            args.n_embedding,
            args.seq_len,
            args.n_engineered_num_features,
            args.dim_embedding_table,
            args.dim_attn_feedforward,
            args.use_mlp,
            args.num_heads,
            args.dropout,
            args.n_layers_order,
            args.n_layers_feature,
            args.emb_tbl_use_bias,
            args.use_moe,
            args.num_experts,
            args.use_time_seq,
        ).to(args.device)
    elif modelname == "TwoSeqMoEOrderFeature":
        # Need to check
        model = TwoSeqMoEOrderFeatureAttentionClassifier(
            args.n_cat_features,
            args.n_num_features,
            args.n_classes,
            args.n_embedding,
            args.seq_len,
            args.n_engineered_num_features,
            args.dim_embedding_table,
            args.dim_attn_feedforward,
            args.num_heads,
            args.dropout,
            args.n_layers_order,
            args.n_layers_feature,
            args.emb_tbl_use_bias,
            args.use_moe,
            args.num_experts,
            args.use_time_seq,
        ).to(args.device)

    if args.local_rank >= 0:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    return model


def main():
    ##########################
    parser = get_parser()

    args = parser.parse_args()

    # Setting up logger
    level = logging.INFO
    handlers = [
        logging.FileHandler(f"{args.result_dir}tsa_train_log.log"),
        logging.StreamHandler(),
    ]

    logging.basicConfig(level=level, handlers=handlers)

    logger = logging.getLogger("tsa log")

    # Constructs scaler once, at the beginning of the convergence run, using default args.
    # If your network fails to converge with default GradScaler args, please file an issue.
    # The same GradScaler instance should be used for the entire convergence run.
    # If you perform multiple convergence runs in the same script, each run should use
    # a dedicated fresh GradScaler instance.  GradScaler instances are lightweight.
    scaler = torch.cuda.amp.GradScaler() if args.use_amp == 1 else None

    # Set up seed
    if args.seed != 0:
        seed_everything(args.seed)

    # Check if sagemaker training job get the correct input files
    all_files = glob.glob(f"{args.data_dir}" + "/**/*.pkl", recursive=True)
    logger.info(all_files)
    all_files = glob.glob(f"{args.data_dir}" + "/**/*.npy", recursive=True)
    logger.info(all_files)
    args.train_data_folder = f"{args.data_dir}/train/train"
    args.vali_data_folder = f"{args.data_dir}/vali/vali"
    args.cali_data_folder = f"{args.data_dir}/cali/cali"
    #     args.cali_data_folder = f"{args.data_dir}/test/test"

    logger.info(args)

    ###########################
    ## Folders
    model_name = "{}/{}.pt".format(args.model_dir, args.test_name)
    logger.info("Model Name - {}".format(model_name))
    checkpoint_name = "{}/models/{}_{}_checkpoint.pt".format(
        args.result_dir, args.test_name, args.seconds
    )
    logger.info("Checkpoint Name - {}".format(checkpoint_name))
    # last_epoch_model_name = "{}/models/{}_{}_last_epoch.pt".format(args.result_dir, args.test_name, args.seconds)

    epoch_model_folder = "{}/models/{}_{}".format(
        args.result_dir, args.seconds, args.test_name
    )
    os.system(f"mkdir -p {epoch_model_folder}")

    model_folder = "{}/models".format(args.result_dir)
    os.system(f"mkdir -p {model_folder}")

    # args.local_rank = -1 means local mode
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        args.device = f"cuda:{args.local_rank}"

        dist.init_process_group(backend="nccl", init_method="env://")

    if args.local_rank >= 0:
        (
            train_sampler,
            train_dataloader,
            vali_sampler,
            vali_dataloader,
            cali_sampler,
            cali_dataloader,
        ) = load_data(args, batch_size=args.batch_size, data_version=args.data_version)
    else:
        train_dataloader, vali_dataloader, cali_dataloader = load_data(
            args, batch_size=args.batch_size, data_version=args.data_version
        )

    model = build_model(args, modelname=args.modelname)
    optimizer, scheduler = create_optimizer(args, model.parameters())

    logger.info(
        f"""model size: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"""
    )

    if args.load_model == 1:
        # Load checkpoint
        if "checkpoint" in args.model_path:
            checkpoint = torch.load(
                args.result_dir + args.model_path, map_location=torch.device("cpu")
            )

            state_dict = checkpoint["model"]
            from collections import OrderedDict

            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                k = "module." + k
                new_state_dict[k] = v

            model.load_state_dict(new_state_dict)

            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            epoch_start = checkpoint["epoch"] + 1
            best_auc = checkpoint["best_auc"]
            del checkpoint, state_dict, new_state_dict
            torch.cuda.empty_cache()
            logger.info("Checkpoint loaded.")
        else:
            # Load previous trained model
            epoch_start = 0

            state_dict = torch.load(
                args.result_dir + args.model_path, map_location=torch.device("cpu")
            )
            from collections import OrderedDict

            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                k = "module." + k
                new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
    else:
        epoch_start = 0

    t0 = time.time()
    best_auc = 0
    train_loss_list = []
    valid_loss_list = []
    train_epochs_loss = []
    valid_epochs_loss = []
    auc_list = []
    measures = ["auc"]  # evaluation measures
    # Other metrics
    # measures = ['loss', 'accuracy', 'auc', 'pr_auc', 'aps', 'precision', 'recall', 'f1'] # evaluation measures

    use_attn_mask = False
    use_key_padding_mask = False if args.modelname == "Feature" else True

    patience = args.patience
    use_time_seq = args.use_time_seq
    use_engineered_features = True if args.n_engineered_num_features > 0 else False
    early_stopping = EarlyStopping(
        patience=patience, verbose=True, trace_func=logger.info
    )

    # Use to exit processes for all GPUs once early stopping happens
    exit_flag = torch.zeros(1).to(args.device)
    for epoch in range(epoch_start, args.max_epoch):
        dist.all_reduce(exit_flag, op=dist.ReduceOp.MAX)  # Synchronize the exit flag

        if exit_flag.item() == 1:
            logger.info("Early stopping, loading final model")
            state_dict = torch.load(model_name, map_location=torch.device("cpu"))
            from collections import OrderedDict

            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if "module" not in k:
                    k = "module." + k
                new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
            logger.info("Final model loaded")

            logger.info("Inferencing validation data")
            perf_valid, scores_valid, labels_valid = evaluation_single_seq(
                args,
                vali_dataloader,
                model,
                epoch,
                vali_sampler,
                measures,
                valid_loss_list,
                valid_epochs_loss,
                args.device,
                logger,
                use_time_seq,
                use_engineered_features,
                use_key_padding_mask,
                use_attn_mask,
            )
            logger.info("Finished inferencing validation data")

            if args.local_rank == 0:
                logger.info("early stopping")

                logger.info("Generating score file")
                score = pd.DataFrame(
                    {
                        "score": np.array(softmax(scores_valid.data.tolist(), axis=1))[
                            :, -1
                        ]
                    }
                )
                score.to_csv(
                    args.model_dir + "/score_file.csv", header=False, index=False
                )

                logger.info("Generating tag file")
                score = pd.DataFrame({"IS_FRD": np.array(labels_valid.data.tolist())})
                score.to_csv(
                    args.model_dir + "/tag_file.csv", header=False, index=False
                )
                dist.barrier()
            else:
                dist.barrier()
            ###################################################################
            # read calibration data, generate percentile score
            cali_loss_list = []
            cali_epochs_loss = []
            measures = []
            logger.info("Inferencing calibration data")
            perf_cali, scores_cali, labels_cali = evaluation_single_seq(
                args,
                cali_dataloader,
                model,
                epoch,
                cali_sampler,
                measures,
                cali_loss_list,
                cali_epochs_loss,
                args.device,
                logger,
                use_time_seq,
                use_engineered_features,
                use_key_padding_mask,
                use_attn_mask,
            )

            logger.info("Finished inferencing calibration data")

            if args.local_rank == 0:
                logger.info("early stopping")
                y_score_calib = np.array(softmax(scores_cali.data.tolist(), axis=1))[
                    :, -1
                ]

                logger.info("Generating score file")
                score = pd.DataFrame({"score": y_score_calib})
                score.to_csv(
                    args.model_dir + "/cali_score_file.csv", header=False, index=False
                )

                logger.info("Generating tag file")
                score = pd.DataFrame({"IS_FRD": np.array(labels_cali.data.tolist())})
                score.to_csv(
                    args.model_dir + "/cali_tag_file.csv", header=False, index=False
                )

                # generate the percentiles
                bins = 1000
                percntls = np.array(range(bins)) / (bins + 0.0)
                percentile_score_sorted_arr = np.percentile(
                    y_score_calib, percntls * 100
                )
                ### Save percentile mapping
                model_location = args.model_dir + "/percentile_score.pkl"
                pkl.dump(
                    np.column_stack((percentile_score_sorted_arr, percntls)).tolist(),
                    open(model_location, "wb"),
                )
                logger.info("saved postprocess!")
                dist.barrier()
            else:
                dist.barrier()
            logger.info(f"Exiting training loop at epoch {epoch}")

            dist.barrier()
            logger.info(f"exit loop for {args.local_rank}")
            break

        train(
            args=args,
            loader=train_dataloader,
            model=model,
            epoch=epoch,
            train_sampler=train_sampler,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss_list=train_loss_list,
            train_epochs_loss=train_epochs_loss,
            device=f"cuda:{args.local_rank}",
            logger=logger,
            scaler=scaler,
            use_time_seq=use_time_seq,
            use_engineered_features=use_engineered_features,
            use_key_padding_mask=use_key_padding_mask,
            use_attn_mask=use_attn_mask,
        )
        # This should be commented out if using scheduler.step() per batch
        scheduler.step()

        perf_valid, _, _ = evaluation_single_seq(
            args,
            vali_dataloader,
            model,
            epoch,
            vali_sampler,
            measures,
            valid_loss_list,
            valid_epochs_loss,
            args.device,
            logger,
            use_time_seq,
            use_engineered_features,
            use_key_padding_mask,
            use_attn_mask,
        )

        # print validation performance
        if args.local_rank == 0:  # epoch % PRINT_EVERY_EPOCH == 0:
            logger.info("Epoch {} TimeSince {}\n".format(epoch, timeSince(t0)))
            logger.info("[VALI] {} ----------------".format(epoch))
            print_performance(perf_valid, logger, measures)  # print only
            valid_loss = -perf_valid["auc"]
            early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                exit_flag.fill_(1)

            # Save best model on validation dataset so far as checkpoint
            auc_list.append(perf_valid["auc"])
            if perf_valid["auc"] > best_auc:
                best_auc = perf_valid["auc"]
                logger.info("Updated model!\n")
                torch.save(model.module.state_dict(), model_name)
                checkpoint = {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_auc": best_auc,
                }
                torch.save(checkpoint, checkpoint_name)
            dist.barrier()
        else:
            dist.barrier()

        # Can choose to save model for each epoch
        # epoch_model_name = "{}/{}_{}_epoch_{}.pt".format(
        #     epoch_model_folder, args.test_name, args.seconds, epoch
        # )
        if args.local_rank == 0:
            # Can choose to save model for each epoch
            # torch.save(model.module.state_dict(), epoch_model_name)

            # Plot training curve
            plt.figure(figsize=(12, 8))
            plt.subplot(221)
            plt.plot(train_epochs_loss[1:], "-o", label="train_loss")
            plt.title("train_loss")
            plt.subplot(222)
            plt.plot(valid_epochs_loss[1:], "-o", label="valid_loss")
            plt.title("epochs_loss")
            plt.legend()
            plt.subplot(223)
            plt.plot(auc_list, "-o", label="valid_auc")
            plt.title("valid_auc")
            plt.legend()
            plt.savefig(model_name + ".png")
            plt.close()
            # torch.save(model.module.state_dict(), last_epoch_model_name)
            dist.barrier()
        else:
            dist.barrier()

    logger.info("exit all")
    exit()


if __name__ == "__main__":
    main()
