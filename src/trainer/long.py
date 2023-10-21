import os

from ..dataset import ReprDataset
from ..models import UniHPFAgg
from ..utils.trainer_utils import PredLoss, PredMetric
from . import register_trainer
from .base import Trainer


@register_trainer("long")
class LongTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.dataset = ReprDataset
        self.architecture = UniHPFAgg
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
        if self.args.train_cls_header:
            self.model.set_mode("classifier")
        return super().epoch(split, data_loader, n_epoch)
