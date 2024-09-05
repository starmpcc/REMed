import logging
import os
from contextlib import nullcontext

import polars as pl
import torch
from tqdm import tqdm

from ..dataset import ReprDataset, MEDSReprDataset
from ..models import REMed
from ..utils.trainer_utils import PredLoss, PredMetric, get_max_seq_len, log_from_dict
from . import register_trainer
from .base import Trainer

logger = logging.getLogger(__name__)


@register_trainer("remed")
class REMedTrainer(Trainer):
    def __init__(self, args):
        if args.src_data != "meds":
            args.max_seq_len = get_max_seq_len(args)
        super().__init__(args)

        if args.src_data == "meds":
            self.dataset = MEDSReprDataset
            self.data_path = args.input_path
        else:
            self.dataset = ReprDataset
            if args.encoded_dir:
                self.data_path = os.path.join(
                    args.encoded_dir, f"{args.src_data}_encoded.h5"
                )
            else:
                self.data_path = os.path.join(
                    args.save_dir, args.pretrained, f"{args.src_data}_encoded.h5"
                )

        self.architecture = REMed
        self.criterion = PredLoss(self.args)
        self.metric = PredMetric(self.args)

    def epoch(self, split, data_loader, n_epoch=0):
        def step(sample):
            net_output, reprs = self.model(**sample)
            loss, logging_outputs = self.criterion(net_output, reprs)
            if split == self.train_subset:
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
            
            return net_output, logging_outputs

        if split == self.train_subset:
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

            do_output_cohort = False
            if self.args.src_data == "meds" and (
                split == self.test_subset and self.test_cohort is not None
            ):
                if self.accelerator.num_processes == 1:
                    # check if test cohort is valid
                    assert set(data_loader.dataset.manifest) == set(self.test_cohort["subject_id"]), (
                        "a set of patient ids in the test cohort should equal to that in the test dataset"
                    )
                    predicted_cohort = {"subject_id": [], "boolean_prediction": []}
                    do_output_cohort = True
                else:
                    logger.warning(
                        "not yet implemented to output predicted labels and probs with "
                        "--test_cohort in multi-processing environment. please run with "
                        "--num_processes=1 in accelerate launch."
                    )

            for sample in t:
                if split == self.train_subset:
                    self.model.set_mode("scorer")
                    net_output, logging_output = step(sample)

                self.model.set_mode("predictor")
                net_output, logging_output = step(sample)

                # meds -- output
                if do_output_cohort:
                    predicted_cohort["subject_id"].extend(sample["subject_id"].tolist())
                    predicted_cohort["boolean_prediction"].extend(
                        net_output['pred']['meds_single_task'].view(-1).tolist()
                    )

        if do_output_cohort:
            predicted_cohort = pl.DataFrame(predicted_cohort)
            self.test_cohort = self.test_cohort.join(predicted_cohort, on="subject_id", how="left")

        metrics = self.metric.get_metrics()
        log_dict = log_from_dict(metrics, split, n_epoch)
        if self.log is None:
            print(log_dict)
        else:
            self.accelerator.log(log_dict)
        return metrics
