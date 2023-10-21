import logging
import os
from contextlib import nullcontext

import torch
from tqdm import tqdm

from ..dataset import ReprDataset
from ..models import REMed
from ..utils.trainer_utils import PredLoss, PredMetric, get_max_seq_len, log_from_dict
from . import register_trainer
from .base import Trainer

logger = logging.getLogger(__name__)


@register_trainer("remed")
class REMedTrainer(Trainer):
    def __init__(self, args):
        args.max_seq_len = get_max_seq_len(args)
        super().__init__(args)

        self.dataset = ReprDataset
        self.architecture = REMed
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

    def epoch(self, split, data_loader, n_epoch=0):
        def step(sample):
            output, reprs = self.model(**sample)
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
                if split == "train":
                    self.model.set_mode("scorer")
                    step(sample)

                self.model.set_mode("predictor")
                step(sample)

        metrics = self.metric.get_metrics()
        log_dict = log_from_dict(metrics, split, n_epoch)
        if self.args.debug:
            print(log_dict)
        else:
            self.accelerator.log(log_dict)
        return metrics
