import torch.nn as nn
from transformers import MegaConfig
from transformers.models.mega.modeling_mega import *


class MegaLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = MegaConfig(
            hidden_size=args.pred_dim,
            num_hidden_layers=args.n_flatten_layers,
            intermediate_size=args.pred_dim * 2,
            ema_projection_size=16,
            bidirectional=True,
            shared_representation_size=args.pred_dim // 4,
            use_chunking=True,
            chunk_size=args.mega_chunk_size,
            nffn_hidden_size=args.pred_dim * 2,
            max_positions=args.max_seq_len + self.args.mega_chunk_size,
        )
        self.mega_block = MegaBlock(self.config)
        self.apply(self._init_weights)

    def forward(self, x, mask):
        x = x.transpose(0, 1)
        out = self.mega_block(x, mask)[0]
        return out.transpose(0, 1)

    def _init_weights(self, module):
        MegaPreTrainedModel._init_weights(self, module)
