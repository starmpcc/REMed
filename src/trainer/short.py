import os

from accelerate.logging import get_logger

from ..dataset import EHRDataset
from ..models import UniHPF
from ..utils.trainer_utils import PredLoss, PredMetric
from . import register_trainer
from .base import Trainer

logger = get_logger(__name__, "INFO")


@register_trainer("short")
class ShortTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.dataset = EHRDataset
        self.architecture = UniHPF
        self.criterion = PredLoss(self.args)
        self.metric = PredMetric(self.args)
        self.data_path = os.path.join(
            args.input_path, f"{args.pred_time}h", f"{args.src_data}.h5"
        )
