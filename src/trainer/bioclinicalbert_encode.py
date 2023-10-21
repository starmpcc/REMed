import os

from accelerate.logging import get_logger

from ..models import BioClinicalBERT
from . import register_trainer
from .base import Trainer

logger = get_logger(__name__, "INFO")


@register_trainer("bioclinicalbert_encode")
class BioClinicalBERTEncodeTrainer(Trainer):
    def __init__(self, args):
        args.encode_events = True
        super().__init__(args)
        self.data_path = os.path.join(
            args.input_path, f"{args.pred_time}h", f"{args.src_data}.h5"
        )

    def train(self):
        pass

    def test(self):
        pass

    def encode_events(self):
        self.model = BioClinicalBERT(self.args)
        self.model = self.accelerator.prepare(self.model)
        return super().encode_events()
