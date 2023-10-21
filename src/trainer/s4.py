import os

from accelerate.utils import is_tpu_available

from ..dataset import FlattenDataset, ReprDataset
from ..models import CachedS4, FlattenS4
from ..utils.trainer_utils import PredLoss, PredMetric
from . import register_trainer
from .base import Trainer


@register_trainer("flatten_s4")
class FlattenS4Trainer(Trainer):
    def __init__(self, args):
        assert not is_tpu_available(), "TPU is not supported for this task"

        super().__init__(args)

        self.dataset = FlattenDataset
        self.architecture = FlattenS4
        self.criterion = PredLoss(self.args)
        self.metric = PredMetric(self.args)
        self.data_path = os.path.join(
            args.input_path, f"{args.pred_time}h", f"{args.src_data}.h5"
        )


@register_trainer("cached_s4")
class CachedS4Trainer(Trainer):
    def __init__(self, args):
        assert not is_tpu_available(), "TPU is not supported for this task"

        super().__init__(args)

        self.dataset = ReprDataset
        self.architecture = CachedS4
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
