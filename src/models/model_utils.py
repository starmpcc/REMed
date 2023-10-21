import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """

        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class FlattenTimeEncoding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = nn.Dropout(p=args.dropout)
        div_term = torch.exp(
            torch.arange(0, args.pred_dim, 2) * (-math.log(10000.0) / args.pred_dim)
        )
        self.register_buffer("div_term", div_term)
        self.args = args

    def forward(self, x, times, **kwargs):
        times = self.args.pred_time * 60 - times  # Arange from zeros
        pe = torch.zeros_like(x)
        pe[:, :, 0::2] = torch.sin(times.unsqueeze(-1) * self.div_term)
        pe[:, :, 1::2] = torch.cos(times.unsqueeze(-1) * self.div_term)
        x = x + pe
        x = self.dropout(x)
        return x


# from https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
def get_slopes(n, alibi_const):
    def _get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - alibi_const)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return _get_slopes_power_of_2(
            n
        )  # In the paper, we only train models that have 2^a heads for some a. This function has
    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n)
        )  # when the number of heads is not a power of 2, we use this workaround.
        return (
            _get_slopes_power_of_2(closest_power_of_2)
            + get_slopes(2 * closest_power_of_2, alibi_const)[0::2][
                : n - closest_power_of_2
            ]
        )


# merge mask, src_pad_mask into one float mask (would be added into attention matrix)
def merge_masks(x, mask, src_pad_mask):
    if mask is None:
        mask = torch.zeros_like(src_pad_mask)
    return mask + src_pad_mask * torch.finfo(x.dtype).min
