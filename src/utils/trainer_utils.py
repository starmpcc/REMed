import math
import os
import re
from ast import literal_eval
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate.logging import get_logger
from accelerate.utils import is_tpu_available
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

logger = get_logger(__name__, "INFO")


@dataclass
class Task:
    name: str
    num_classes: int
    property: str


def get_task(task_name, src_data):
    if task_name == "meds_single_task":
        # NOTE as of meds 0.3.0, it supports only for binary classification task
        return Task(task_name, 1, "binary")
    elif re.findall("mortality|readmission|los", task_name):
        return Task(task_name, 1, "binary")
    elif re.findall("diagnosis", task_name):
        if src_data == "umcdb":
            return Task(task_name, 13, "multilabel")
        else:
            return Task(task_name, 17, "multilabel")
    elif re.findall("creatinine|platelets", task_name):
        return Task(task_name, 5, "multiclass")
    elif re.findall("wbc|bicarbonate|sodium", task_name):
        return Task(task_name, 3, "multiclass")
    elif re.findall("hb", task_name):
        return Task(task_name, 4, "multiclass")


# To Load & Save n_epoch
class N_Epoch:
    def __init__(self):
        self.epoch = 0

    def __call__(self):
        return self.epoch

    def increment(self):
        self.epoch += 1

    def state_dict(self):
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]


def load_model(path, model):
    logger.info(f"Loading checkpoint from {path}")
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    if "pred_model.model.embed_positions.weight" in state_dict:
        del state_dict["pred_model.model.embed_positions.weight"]
    model.load_state_dict(state_dict, strict=False)
    logger.info("Successfully loaded the checkpoint")
    return model


def get_max_seq_len(args):
    df = pd.read_csv(
        os.path.join(
            args.input_path, f"{args.pred_time}h", f"{args.src_data}_cohort.csv"
        ),
        usecols=["time", "hi_start"],
    )
    if args.time >= 0:
        df["hi_start"] = df["hi_start"].map(literal_eval).map(lambda x: x[args.time])
    else:
        df["hi_start"] = 0
    max_seq_len = df.apply(
        lambda x: x["time"].count(",") + 1 - x["hi_start"], axis=1
    ).max()
    max_seq_len = math.ceil(max_seq_len / 128) * 128

    return max_seq_len


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(
        self, patience=7, verbose=True, delta=0, compare="increase", metric="avg_auroc"
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.target_metric_min = 0
        self.delta = delta
        self.compare_score = self.increase if compare == "increase" else self.decrease
        self.metric = metric

    def __call__(self, target_metric):
        update_token = False
        score = target_metric

        if self.best_score is None:
            self.best_score = score

        if self.compare_score(score):
            self.counter += 1
            logger.info(
                f"EarlyStopping counter: {self.counter} out of {self.patience} ({target_metric:.6f})",
                main_process_only=True,
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.verbose:
                logger.info(
                    f"Validation {self.metric} {self.compare_score.__name__}d {self.target_metric_min:.6f} --> {target_metric:.6f})",
                    main_process_only=True,
                )
            self.target_metric_min = target_metric
            self.counter = 0
            update_token = True

        return update_token

    def increase(self, score):
        if score < self.best_score + self.delta:
            return True
        else:
            return False

    def decrease(self, score):
        if score > self.best_score + self.delta:
            return True
        else:
            return False

    def state_dict(self):
        return {
            "best_score": self.best_score,
            "counter": self.counter,
            "early_stop": self.early_stop,
            "target_metric_min": self.target_metric_min,
        }

    def load_state_dict(self, state_dict):
        self.best_score = state_dict["best_score"]
        self.counter = state_dict["counter"]
        self.early_stop = state_dict["early_stop"]
        self.target_metric_min = state_dict["target_metric_min"]

    def reset(self):
        self.counter = 0
        self.early_stop = False


def log_from_dict(metric_dict, split, n_epoch):
    log_dict = {"epoch": n_epoch, split: metric_dict}
    return log_dict


class PredLoss:
    def __init__(self, args):
        self.args = args
        # How to drop na in binary??
        self.bce = nn.BCELoss(reduction="sum")
        self.ce = nn.NLLLoss(reduction="sum", ignore_index=-1)
        self.sim = nn.CosineSimilarity(dim=-1)

    def __call__(self, output, reprs):
        # NOTE: If null label is too many in binary/multilabel, it will be cause a nan loss.
        losses, preds, truths, masks = {}, {}, {}, {}
        loss_total = 0
        # To suport Rag Retriever
        tasks = [i for i in self.args.tasks if i.name in output["target"].keys()]
        for task in tasks:
            pred = output["pred"][task.name]
            target = output["target"][task.name]
            if task.property == "binary":
                # Calculate mask for -1(NaN)
                mask = (target != -1).bool()
                pred = mask * pred
                target = mask * target
                loss = self.bce(pred, target)
            elif task.property == "multilabel":
                # Calculate mask for -1(NaN)
                mask = (target.sum(axis=-1) > 0).bool().unsqueeze(-1)
                pred = mask * pred
                target = mask * target
                loss = self.bce(pred, target) / task.num_classes
            elif task.property == "multiclass":
                mask = (target.sum(axis=-1) > 0).bool().unsqueeze(-1)
                nl = (pred + 1e-10).log()  # For numerical Stability
                pred = mask * pred
                nl = mask * nl
                target = mask * target
                loss = self.ce(nl, target.argmax(dim=1))
            else:
                raise NotImplementedError()
            losses[task.name] = loss / self.args.local_batch_size
            preds[task.name] = pred
            truths[task.name] = target
            masks[task.name] = mask
            loss_total += loss

        logging_outputs = {
            # SHould detach or not??
            "loss_total": loss_total,
            "preds": preds,
            "truths": truths,
            "losses": losses,
            "masks": masks,
        }

        return loss_total, logging_outputs


class BaseMetric:
    def __init__(self, args, target):
        self.args = args
        self._update_target = target
        self.is_tpu = is_tpu_available()
        self.reset()

    def reset(self):
        raise NotImplementedError()

    def __call__(self, out, accelerator=None):
        raise NotImplementedError()

    def get_metrics(self):
        raise NotImplementedError()

    def gather(self, accelerator, *args):
        if accelerator is not None:
            args = accelerator.gather_for_metrics(args)
        args = [(i if i.shape else i.unsqueeze(0)) for i in args]
        if len(args) == 1:
            return args[0]
        else:
            return args

    @property
    def compare(self):
        return "decrease" if "loss" in self.update_target else "increase"

    @property
    def update_target(self):
        return self._update_target


class PredMetric(BaseMetric):
    def __init__(self, args, target="avg_auroc"):
        self.tasks = args.tasks
        super().__init__(args, target)

    def reset(self):
        self.losses = {k.name: [] for k in self.tasks}
        self.truths = {k.name: [] for k in self.tasks}
        self.preds = {k.name: [] for k in self.tasks}
        self.masks = {k.name: [] for k in self.tasks}

    def __call__(self, out, accelerator=None):
        # NOTE: On train step, only compute metrics for the master process
        tasks = [i for i in self.tasks if i.name in out["preds"].keys()]
        for task in tasks:
            mask = out["masks"][task.name]
            if task.property != "binary":
                mask = mask.squeeze(-1)
            truth = out["truths"][task.name]
            pred = out["preds"][task.name]
            loss = out["losses"][task.name]

            truth, pred, mask, loss = self.gather(accelerator, truth, pred, mask, loss)

            self.truths[task.name].append(truth.detach().cpu().float().numpy())
            self.preds[task.name].append(pred.detach().cpu().float().numpy())
            self.losses[task.name].append(loss.detach().cpu().float().numpy())
            self.masks[task.name].append(mask.detach().cpu().numpy())

    def get_metrics(self):
        # For REMed
        tasks = [i for i in self.tasks if len(self.preds[i.name]) != 0]
        for task in tasks:
            self.losses[task.name] = np.concatenate(self.losses[task.name], 0)
            self.truths[task.name] = np.concatenate(self.truths[task.name], 0)
            self.preds[task.name] = np.concatenate(self.preds[task.name], 0)
            self.masks[task.name] = np.concatenate(self.masks[task.name], 0)
            self.truths[task.name] = self.truths[task.name][self.masks[task.name]]
            self.preds[task.name] = self.preds[task.name][self.masks[task.name]]

        self.epoch_dict = {}
        for task in tasks:
            self.epoch_dict[task.name + "_loss"] = np.mean(self.losses[task.name])
            self.epoch_dict[task.name + "_auprc"] = self.auprc(task)
            self.epoch_dict[task.name + "_auroc"] = self.auroc(task)
            self.epoch_dict[task.name + "_acc"] = self.acc(task)

        self.epoch_dict["avg_loss"] = np.mean(
            [self.epoch_dict[k] for k in self.epoch_dict.keys() if "loss" in k]
        )
        self.epoch_dict["avg_auprc"] = np.mean(
            [self.epoch_dict[k] for k in self.epoch_dict.keys() if "auprc" in k]
        )
        self.epoch_dict["avg_auroc"] = np.mean(
            [self.epoch_dict[k] for k in self.epoch_dict.keys() if "auroc" in k]
        )
        self.epoch_dict["avg_acc"] = np.mean(
            [self.epoch_dict[k] for k in self.epoch_dict.keys() if "acc" in k]
        )
        self.reset()
        return self.epoch_dict

    def auroc(self, task):
        return roc_auc_score(
            self.truths[task.name],
            self.preds[task.name],
            average="micro",
            multi_class="ovr",
        )

    def auprc(self, task):
        return average_precision_score(
            self.truths[task.name],
            self.preds[task.name],
            average="micro",
        )

    def acc(self, task):
        if task.property in ["binary", "multilabel"]:
            return accuracy_score(
                self.truths[task.name].round(), self.preds[task.name].round()
            )
        elif task.property == "multiclass":
            return accuracy_score(
                self.truths[task.name].argmax(axis=1),
                self.preds[task.name].argmax(axis=1),
            )
        else:
            raise NotImplementedError()
