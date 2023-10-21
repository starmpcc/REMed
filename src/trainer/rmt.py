import os
from contextlib import nullcontext
from copy import deepcopy

import torch
from accelerate.logging import get_logger
from tqdm import tqdm

from ..dataset import CachedRMTDataset, RMTDataset
from ..models import CachedRMT, FlattenRMT
from ..utils.trainer_utils import (
    EarlyStopping,
    N_Epoch,
    PredLoss,
    PredMetric,
    get_max_seq_len,
    load_model,
    log_from_dict,
)
from . import register_trainer
from .base import Trainer

logger = get_logger(__name__)


class N_Chunk:
    def __init__(self):
        self.chunk = 1

    def __call__(self):
        return self.chunk

    def increment(self):
        self.chunk += 1

    def state_dict(self):
        return {"chunk": self.chunk}

    def load_state_dict(self, state_dict):
        self.chunk = state_dict["chunk"]


class RMTTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

    def train(self):
        self.early_stopping = EarlyStopping(
            patience=self.args.patience,
            compare=self.metric.compare,
            metric=self.metric.update_target,
        )
        self.n_epoch = N_Epoch()
        self.n_chunk = N_Chunk()
        train_loader = self.dataloader_set("train")
        valid_loader = self.dataloader_set("valid")

        model = self.architecture(self.args)
        if self.args.pretrained:
            pretrained_path = os.path.join(
                self.args.save_dir, self.args.pretrained, "checkpoint_best.pt"
            )
            model = load_model(pretrained_path, model)

        if self.args.enable_fsdp:
            model = self.accelerator.prepare(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)

        if self.args.scheduler:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1 / 100, end_factor=1, total_iters=500
            )
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 1, 1, 1)

        rmt_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 0.5)

        if self.args.enable_fsdp:
            self.model = model
            (
                self.train_loader,
                self.valid_loader,
                self.optimizer,
                self.scheduler,
                self.rmt_scheduler,
            ) = self.accelerator.prepare(
                train_loader, valid_loader, optimizer, scheduler, rmt_scheduler
            )
        else:
            (
                self.train_loader,
                self.valid_loader,
                self.optimizer,
                self.scheduler,
                self.rmt_scheduler,
                self.model,
            ) = self.accelerator.prepare(
                train_loader, valid_loader, optimizer, scheduler, rmt_scheduler, model
            )
        self.accelerator.register_for_checkpointing(self.early_stopping)
        self.accelerator.register_for_checkpointing(self.n_epoch)
        self.accelerator.register_for_checkpointing(self.n_chunk)
        resume_path = os.path.join(self.args.save_dir, self.args.exp_name, "checkpoint")

        if os.path.exists(resume_path):
            self.accelerator.load_state(resume_path)
        else:
            logger.info("No checkpoint to resume", main_process_only=True)
        if self.early_stopping.early_stop:
            # Case that terminated during test
            return
        self.best_state_dict = deepcopy(
            self.accelerator.unwrap_model(self.model).state_dict()
        )

        # Do Curriculum Learning
        while self.n_epoch() < self.args.n_epochs:
            self.train_loader.dataset.set_max_seq_len(self.n_chunk())
            self.epoch("train", self.train_loader, self.n_epoch())

            if self.evaluation(self.n_epoch()):
                logger.info(
                    f"Incrementing rmt_n_chunks.. from {self.n_chunk()} to {self.n_chunk()+1}"
                )
                self.n_chunk.increment()
                self.rmt_scheduler.step()
                self.early_stopping.reset()
                if self.n_chunk() > self.args.rmt_n_chunks:
                    break
                try:
                    self.model.module.load_state_dict(self.best_state_dict)
                except:
                    self.model.load_state_dict(self.best_state_dict)
                else:
                    breakpoint()
            self.n_epoch.increment()

            self.accelerator.save_state(resume_path)
            logger.info("save checkpoint...", main_process_only=True)
            self.accelerator.wait_for_everyone()

    def evaluation(self, n_epoch):
        metric_dict = self.epoch("valid", self.valid_loader, n_epoch)
        best_model_path = os.path.join(
            self.args.save_dir,
            self.args.exp_name,
            "checkpoint_best.pt",
        )
        if self.early_stopping(metric_dict[self.metric.update_target]):
            state_dict = self.accelerator.unwrap_model(self.model).state_dict()
            self.accelerator.save(state_dict, best_model_path)
            logger.info("save best model...", main_process_only=True)
            self.best_state_dict = deepcopy(state_dict)
        return self.early_stopping.early_stop

    def epoch(self, split, data_loader, n_epoch=0):
        if split == "train":
            self.model.train()
            context = nullcontext()
            accelerator = None
        else:
            self.model.eval()
            context = torch.no_grad()
            accelerator = self.accelerator
        with context:
            if self.accelerator.is_main_process:
                t = tqdm(data_loader, desc=f"{split} epoch {n_epoch}")
            else:
                t = data_loader
            for sample in t:
                output, reprs = self.model(self.n_chunk(), **sample)
                loss, logging_outputs = self.criterion(output, reprs)
                if split == "train":
                    self.optimizer.zero_grad(set_to_none=True)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                self.metric(logging_outputs, accelerator)
                if (
                    not self.args.debug
                    and self.accelerator.is_main_process
                    and self.args.log_loss
                ):
                    self.accelerator.log({f"{split}_loss": loss})
        metrics = self.metric.get_metrics()
        log_dict = log_from_dict(metrics, split, n_epoch)
        if self.args.debug:
            print(log_dict)
        else:
            self.accelerator.log(log_dict)
        return metrics


@register_trainer("flatten_rmt")
class FlattenRMTTrainer(RMTTrainer):
    def __init__(self, args):
        args.max_seq_len = get_max_seq_len(args)
        super().__init__(args)

        self.dataset = RMTDataset
        self.architecture = FlattenRMT
        self.criterion = PredLoss(self.args)
        self.metric = PredMetric(self.args)
        self.data_path = os.path.join(
            args.input_path, f"{args.pred_time}h", f"{args.src_data}.h5"
        )

@register_trainer("cached_rmt")
class CachedRMTTrainer(RMTTrainer):
    def __init__(self, args):
        args.max_seq_len = get_max_seq_len(args)
        super().__init__(args)
        self.dataset = CachedRMTDataset
        self.architecture = CachedRMT
        self.criterion = PredLoss(self.args)
        self.metric = PredMetric(self.args)
        if args.encoded_dir:
            self.data_path = os.path.join(
                args.encoded_dir, f"{args.src_data}_encoded.h5"
            )
        else:
            self.data_path = os.path.join(
                args.save_dir, args.pretrained, f"{args.src_data}_encoded.h5"
            )
