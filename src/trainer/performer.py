import os

from ..dataset import FlattenDataset, ReprDataset
from ..models import CachedPerformer, FlattenPerformer
from ..utils.trainer_utils import PredLoss, PredMetric
from . import register_trainer
from .base import Trainer


@register_trainer("flatten_performer")
class FlattenPerformerTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.dataset = FlattenDataset
        self.architecture = FlattenPerformer
        self.criterion = PredLoss(self.args)
        self.metric = PredMetric(self.args)
        self.data_path = os.path.join(
            args.input_path, f"{args.pred_time}h", f"{args.src_data}.h5"
        )

@register_trainer("cached_performer")
class CachedPerformerTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.dataset = ReprDataset
        self.architecture = CachedPerformer
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
