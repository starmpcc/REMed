import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class EventEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_dim = args.pred_dim

    def forward(self, x):
        # Input: (B*S, L, E)
        # Output: (B, S, E)
        return


class TransformerEventEncoder(EventEncoder):
    def __init__(self, args):
        super().__init__(args)

        encoder_layers = TransformerEncoderLayer(
            args.pred_dim,
            args.n_heads,
            args.pred_dim * 4,
            args.dropout,
            batch_first=True,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers, args.n_enc_layers, enable_nested_tensor=False
        )

    def forward(self, all_codes_embs, input_ids, **kwargs):
        # all_codes_embs: (B * S, L, Hidden) -- (16 * 512, 128, 512)
        # input_ids: (B, S, L) -- (16, 512, 128)
        if input_ids.ndim == 2:
            assert input_ids.size(0) == all_codes_embs.size(0)
            input_ids = input_ids.unsqueeze(1) # (B, L) -> (B, 1, L)

        B, S, L = input_ids.shape
        # All-padding col -> cause nan output -> unmask it (and multiply 0 to the results)
        src_pad_mask = (input_ids.reshape(-1, L).eq(0)) ^ (
            input_ids.reshape(-1, L).sum(dim=-1).eq(0).unsqueeze(-1).repeat(1, L)
        )
        encoder_output = self.transformer_encoder(
            all_codes_embs, src_key_padding_mask=src_pad_mask
        )
        if self.args.train_type != "encoder_pt":
            out = self.pooling(encoder_output, src_pad_mask).view(B, S, self.pred_dim)
            out = out * input_ids.sum(dim=-1).ne(0).unsqueeze(-1).repeat(
                1, 1, self.pred_dim
            )
        else:
            out = encoder_output.reshape(*input_ids.shape, -1)

        return out

    def pooling(self, x, src_pad_mask):
        if self.args.encoder_pooling == "cls":
            x = x[:, 0, :]
        elif self.args.encoder_pooling == "mean":
            mask = ~src_pad_mask.unsqueeze(-1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        return x
