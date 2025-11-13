import math
import numpy as np
import os
import random
import sklearn.metrics as metrics
import time
import torch
import torch.distributed as dist
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    OneCycleLR,
)

from asl_focal_loss import *
from focalloss import *


# return a lower triangle True matrix
def get_subsequent_mask(len_s, device):
    """For masking out the subsequent info."""
    subsequent_mask = (
        torch.triu(torch.ones((len_s, len_s), device=device), diagonal=1)
    ).bool()
    return subsequent_mask


def set_loss(args):
    if args.loss == "FocalLoss":
        criterion = FocalLoss(
            gamma=args.loss_gamma, alpha=args.loss_alpha, reduction=args.loss_reduction
        )
    elif args.loss == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(reduction="sum")
    elif args.loss == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss(reduction="sum")
    elif args.loss == "Cyclical_FocalLoss":
        criterion = Cyclical_FocalLoss(
            gamma_pos=args.gamma_pos,
            gamma_neg=args.gamma_neg,
            epochs=args.max_epoch,
            gamma_hc=args.gamma_hc,
            factor=args.cyclical_factor,
            reduction="sum",
        )
    return criterion


def create_optimizer(args, model_params):
    if args.optim == "adam":
        optimizer = Adam(
            params=model_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
    scheduler = OneCycleLR(
        optimizer=optimizer, max_lr=args.scheduler_maxlr, total_steps=args.max_epoch
    )
    # Other scheduler choices
    # scheduler = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=5, T_mult=1, eta_min=1e-5)
    # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=10, eta_min=1e-5
    # scheduler = CyclicLR(optimizer,base_lr=1e-5,max_lr=1e-3,step_size_up=10,mode="triangular2",cycle_momentum=False)

    return optimizer, scheduler


def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


def iteration_single_seq(
    batch,
    model,
    use_time_seq,
    use_engineered_features,
    use_key_padding_mask=True,
    use_attn_mask=False,
    device="cpu",
):
    x_seq_cat, x_seq_num, y = (
        batch["x_seq_cat"].to(device),
        batch["x_seq_num"].to(device),
        batch["y"].long().view(-1).to(device),
    )

    if use_key_padding_mask:
        key_padding_mask = torch.logical_not(
            torch.nn.functional.pad(batch["not_padded"], (0, 1), value=True)
        ).to(device)
    else:
        key_padding_mask = None

    attn_mask = get_subsequent_mask(50, device).to(device) if use_attn_mask else None

    time_seq = batch["time_to_last"].to(device) if use_time_seq else None
    x_engineered = (
        batch["x_engineered_num"].to(device) if use_engineered_features else None
    )

    pred, _ = model(
        x_seq_cat, x_seq_num, x_engineered, time_seq, attn_mask, key_padding_mask
    )

    return pred, y


def iteration_two_seq(
    batch,
    model,
    use_time_seq,
    use_engineered_features,
    use_key_padding_mask=True,
    use_attn_mask=False,
    device="cpu",
):
    x_seq_cat_cid, x_seq_num_cid, x_seq_cat_ccid, x_seq_num_ccid, y = (
        batch["x_seq_cat_cid"].to(device),
        batch["x_seq_num_cid"].to(device),
        batch["x_seq_cat_ccid"].to(device),
        batch["x_seq_num_ccid"].to(device),
        batch["y"].long().view(-1).to(device),
    )

    if use_key_padding_mask:
        key_padding_mask_cid = torch.logical_not(
            torch.nn.functional.pad(batch["not_padded_cid"], (0, 1), value=True)
        ).to(device)
        key_padding_mask_ccid = torch.logical_not(
            torch.nn.functional.pad(batch["not_padded_ccid"], (0, 1), value=True)
        ).to(device)
    else:
        key_padding_mask_cid = None
        key_padding_mask_ccid = None

    attn_mask = get_subsequent_mask(50, device).to(device) if use_attn_mask else None

    time_seq_cid = batch["time_to_last_cid"].to(device) if use_time_seq else None
    time_seq_ccid = batch["time_to_last_ccid"].to(device) if use_time_seq else None
    x_engineered = (
        batch["x_engineered_num"].to(device) if use_engineered_features else None
    )

    pred, _ = model(
        x_seq_cat_cid,
        x_seq_num_cid,
        time_seq_cid,
        x_seq_cat_ccid,
        x_seq_num_ccid,
        time_seq_ccid,
        x_engineered,
        attn_mask,
        key_padding_mask_cid,
        key_padding_mask_ccid,
    )

    return pred, y


def iteration(
    batch,
    model,
    use_time_seq,
    use_engineered_features,
    use_key_padding_mask=True,
    use_attn_mask=False,
    device="cpu",
):
    if len(batch) >= 9:
        return iteration_two_seq(
            batch,
            model,
            use_time_seq,
            use_engineered_features,
            use_key_padding_mask,
            use_attn_mask,
            device,
        )
    else:
        return iteration_single_seq(
            batch,
            model,
            use_time_seq,
            use_engineered_features,
            use_key_padding_mask,
            use_attn_mask,
            device,
        )


def train(
    args,
    loader,
    model,
    epoch,
    train_sampler,
    optimizer,
    scheduler,
    train_loss_list,
    train_epochs_loss,
    device,
    logger,
    scaler=None,
    use_time_seq=True,
    use_engineered_features=True,
    use_key_padding_mask=True,
    use_attn_mask=False,
    n_classes=2,
):
    if args.local_rank >= 0:
        train_sampler.set_epoch(epoch)
    model.train()

    criterion = set_loss(args)

    train_epoch_loss = []

    use_amp = False if scaler is None else True

    for batch in loader:
        with torch.cuda.amp.autocast(enabled=use_amp):
            optimizer.zero_grad()

            pred, y_ = iteration(
                batch,
                model,
                use_time_seq,
                use_engineered_features,
                use_key_padding_mask,
                use_attn_mask,
                device,
            )

            if args.loss == "Cyclical_FocalLoss":
                loss = criterion(pred, y_, epoch)
            else:
                loss = criterion(pred, y_)

            ## (Optional) Flooding loss: https://arxiv.org/abs/2002.08709
            # loss = (loss-b).abs()+b # b is the flooding level.

        if use_amp:
            # Calls backward() on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()
            # scaler.step() first unscales the gradients of the optimizer's assigned params.
            # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
            # otherwise, optimizer.step() is skipped.
            scaler.step(optimizer)

            # Updates the scale for next iteration.
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        ## (Optional) Put scheduler.step() for each batch, need to comment out the one in main() function
        #         scheduler.step()

        loss_cpu = loss.detach().cpu().item() / len(batch["y"])
        train_loss_list.append(loss_cpu)
        train_epoch_loss.append(loss_cpu)

    train_epochs_loss.append(np.average(train_epoch_loss))
    if args.local_rank == 0:
        logger.info("average_training_loss={};".format(np.average(train_epoch_loss)))


def evaluation_single_seq(
    args,
    loader,
    model,
    epoch,
    sampler,
    measures,
    valid_loss_list,
    valid_epochs_loss,
    device,
    logger,
    use_time_seq=True,
    use_engineered_features=True,
    use_key_padding_mask=True,
    use_attn_mask=False,
    n_classes=2,
):
    """
    evaluation model performance with DDP setting
    """

    if args.local_rank >= 0:
        sampler.set_epoch(epoch)

    model.eval()

    criterion = set_loss(args)

    # score, label: record the prediction and label for each GPU
    loss, score, label = 0, [], []
    valid_epoch_loss = []

    with torch.no_grad():
        for batch in loader:
            pred, y_ = iteration(
                batch,
                model,
                use_time_seq,
                use_engineered_features,
                use_key_padding_mask,
                use_attn_mask,
                device,
            )

            loss = criterion(pred, y_)
            loss_cpu = loss.detach().cpu().item() / len(batch["y"])

            valid_epoch_loss.append(loss_cpu)
            if valid_loss_list is not None:
                valid_loss_list.append(loss_cpu)

            if args.local_rank >= 0:
                score.append(pred)
                label.append(y_)
            else:
                score.extend(pred.data.tolist())
                label.extend(y_.data.tolist())

    if args.local_rank >= 0:
        # scores, labels: record the prediction and label for all GPUs
        dist.barrier()
        scores = distributed_concat(torch.concat(score, dim=0), len(sampler.dataset))
        labels = distributed_concat(torch.concat(label, dim=0), len(sampler.dataset))
    else:
        scores = score
        labels = label

    if args.local_rank <= 0:
        if measures == None or len(measures) == 0:
            perf = {}
        else:
            perf = get_performance(scores.data.tolist(), labels.data.tolist(), measures)
        perf["loss"] = np.average(valid_epoch_loss)
        if valid_epochs_loss is not None:
            valid_epochs_loss.append(np.average(valid_epoch_loss))
            logger.info(
                "average_validation_loss={};".format(np.average(valid_epoch_loss))
            )
        if args.local_rank == 0:
            dist.barrier()
        return perf, scores, labels
    else:
        dist.barrier()
        return None, None, None


def get_performance(score, label, measures):
    """
    get classification performance for binary task

    Args:
     score - 2D np.array or list
     label - 1D np.array or list
    """

    ndigits = 4
    performance = {}

    accuracy, auc = None, None
    precision, recall, f1 = None, None, None

    label_pred = np.argmax(score, -1)

    if "accuracy" in measures:
        accuracy = accuracy_score(label, label_pred)
        performance["accuracy"] = round(accuracy, ndigits)

    if "precision" in measures:
        precision = precision_score(label, label_pred, pos_label=1)
        performance["precision"] = round(precision, ndigits)

    if "recall" in measures:
        recall = recall_score(label, label_pred, pos_label=1)
        performance["recall"] = round(recall, ndigits)

    if "f1" in measures:
        f1 = f1_score(label, label_pred, pos_label=1)
        performance["f1"] = round(f1, ndigits)

    if "auc" in measures:
        #         auc = roc_auc_score(np.array(label), np.array(expit(score)))
        auc = roc_auc_score(
            np.array(label), np.array(softmax(score, axis=1))[:, -1]
        )  # if len(np.unique(np.array(label)))!=1 else -1
        performance["auc"] = round(auc, ndigits)

    if "pr_auc" in measures:
        precisions, recalls, thresholds = metrics.precision_recall_curve(
            np.array(label), np.array(softmax(score, axis=1))[:, -1], pos_label=1
        )
        pr_auc = metrics.auc(recalls, precisions)
        performance["pr_auc"] = round(pr_auc, ndigits)

    if "aps" in measures:
        aps = average_precision_score(
            np.array(label), np.array(softmax(score, axis=1))[:, -1], pos_label=1
        )
        performance["aps"] = round(aps, ndigits)

    #     performance = {'accuracy': round(accuracy, ndigits),
    #                    'precision': round(precision, ndigits),
    #                    'recall': round(recall, ndigits),
    #                    'f1': round(f1, ndigits),
    #                    'auc': round(auc, ndigits),
    #                    'pr_auc': round(pr_auc, ndigits),
    #                    'aps': round(aps, ndigits)}

    return performance


def print_performance(
    perf, logger, measures=None, printif=True, writeif=False, **kargs
):
    """
    print classification performance for binary task

    Args:
     per   - dictionary with measure name as keys and performance as values
             or perf = get_performance(score, label)
     kargs - epoch, loss, global_step
             wf = open(outfile_name, 'w')
             tbf = create_tensorboard(tensorboard_name)
    """

    if measures is None:
        measures = sorted(perf.keys())

    if printif:
        maxchrlen = max([len(x) for x in measures])
        for mea in measures:
            logger.info(
                mea + " " * (maxchrlen - len(mea)) + " {:.4f}".format(perf[mea])
            )
        logger.info("")

    if writeif:
        assert "wf" in kargs.keys(), "missing argument: wf"
        kargs["wf"].writerow(perf)
        return kargs["wf"]


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = True


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60

    return "%dm %ds" % (m, s)
