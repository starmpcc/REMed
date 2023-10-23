import math

import torch
import torch.nn as nn
from transformers.models.roformer.modeling_roformer import (
    RoFormerConfig,
    RoFormerEncoder,
)


class Retriever(nn.Module):
    def __init__(self, pred_dim):
        super().__init__()
        pred_dim = pred_dim + 1  # To handle time
        self.model = nn.Sequential(
            nn.Linear(pred_dim, pred_dim // 2),
            nn.LayerNorm(pred_dim // 2),
            nn.ReLU(),
            nn.Linear(pred_dim // 2, pred_dim // 4),
            nn.LayerNorm(pred_dim // 4),
            nn.ReLU(),
            nn.Linear(pred_dim // 4, pred_dim // 8),
            nn.LayerNorm(pred_dim // 8),
            nn.ReLU(),
            nn.Linear(pred_dim // 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, reprs, times, **kwargs):
        times = times.unsqueeze(-1).type(reprs.dtype)
        return self.model(torch.cat([reprs, times], dim=-1)).squeeze()


class ReprTimeEnc(nn.Module):
    def __init__(self, pred_dim, dropout, pred_time):
        super().__init__()
        self.pred_time = pred_time
        div_term = torch.exp(
            torch.arange(0, pred_dim, 2) * (-math.log(10000.0) / pred_dim)
        )
        self.register_buffer("div_term", div_term)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, times, **kwargs):
        # Process input mask
        times = self.pred_time * 60 - times
        src_pad_mask = x.eq(0).all(dim=-1)

        pe = torch.zeros_like(x)
        pe[:, :, 0::2] = torch.sin(times.unsqueeze(-1) * self.div_term)
        pe[:, :, 1::2] = torch.cos(times.unsqueeze(-1) * self.div_term)
        x = x + pe

        x = self.dropout(x)

        return x, src_pad_mask


class Predictor(nn.Module):
    def __init__(
        self, pred_dim, dropout, pred_time, n_layers, n_heads, max_retrieve_len
    ):
        super().__init__()
        self.time_enc = ReprTimeEnc(pred_dim, dropout, pred_time)
        config = RoFormerConfig(
            hidden_size=pred_dim,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            intermediate_size=pred_dim * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=max_retrieve_len,
        )
        self.model = RoFormerEncoder(config)

    def forward(self, reprs, times, **kwargs):
        x, src_pad_mask = self.time_enc(reprs, times, **kwargs)
        mask = src_pad_mask * torch.tensor(
            torch.finfo(x.dtype).min, dtype=x.dtype
        )  # Convert to float type mask
        mask = mask.unsqueeze(-1).unsqueeze(1)
        x = self.model(x, attention_mask=mask)["last_hidden_state"]
        return x


# This is simplified for single, binary classification
# For multi-task or multi-class, please refer the original code
class PredOutPutLayer(nn.Module):
    def __init__(self, pred_dim):
        super().__init__()
        self.final_proj = nn.Linear(pred_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        # Mean Pooling
        mask = x.ne(0).any(dim=-1).unsqueeze(-1)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)
        logit = self.final_proj(x)
        pred = self.sigmoid(logit)
        return pred


class REMed(nn.Module):
    def __init__(
        self,
        pred_dim=512,  # Model hidden dimension size
        n_heads=8,  # Number of heads for Transformer Predictor
        n_layers=2,  # Number of layers for Transformer Predictor
        dropout=0.2,  # Dropout rate
        max_retrieve_len=128,  # Maximum number of retrieved events (k of Top-k)
        pred_time=48,  # Prediction time. Set to maximum  of your input time (h)
        **kwargs
    ):
        super().__init__()
        self.pred_dim = pred_dim
        self.max_retrieve_len = max_retrieve_len

        self.predictor = Predictor(
            pred_dim, dropout, pred_time, n_layers, n_heads, max_retrieve_len
        )
        self.emb2out_model = PredOutPutLayer(pred_dim)
        self.retriever = Retriever(pred_dim)

        self.register_buffer("random_token_emb", torch.randn(pred_dim))

        self.set_mode("retriever")

    def set_mode(self, mode):
        self.mode = mode
        if mode == "retriever":
            self.requires_grad_(False)
            self.retriever.requires_grad_(True)
        elif mode == "predictor":
            self.requires_grad_(True)
            self.retriever.requires_grad_(False)

    # Reprs: Batch of list of event vectors (B, L, E)
    # Times: Batch of list of event times (B, L) (unit=Minute)
    def forward(self, reprs, times, **kwargs):
        # Add Padding to use cutoff (if all events is rejected, should retrive paddings all)
        reprs = nn.functional.pad(reprs, (0, 0, 0, self.max_retrieve_len))
        times = nn.functional.pad(times, (0, self.max_retrieve_len))
        # To implement right-side padding
        times = torch.where(reprs.eq(0).all(dim=-1), 1e10, times)
        sim = self.retriever(reprs, times)

        _sim = torch.where(
            reprs.eq(0).all(dim=-1),
            torch.zeros_like(sim),
            sim,
        )
        topk_values, topk_indices = torch.topk(_sim, self.max_retrieve_len, dim=1)

        topk = torch.gather(
            reprs, 1, topk_indices.unsqueeze(-1).repeat(1, 1, self.pred_dim)
        )
        topk_times = torch.gather(times, 1, topk_indices)
        B, K, E = topk.shape

        # Sort for RoFormer (by time!)
        topk_times, topk_indices = topk_times.sort(dim=1)
        topk = topk.gather(1, topk_indices.unsqueeze(-1).repeat(1, 1, E))
        topk_values = topk_values.gather(1, topk_indices)

        def _retriever_path():
            _topk_values = topk_values.reshape(B * K, 1)
            _topk = topk.reshape(B * K, 1, -1)
            _topk_times = topk_times.reshape(B * K, 1)

            zero_idcs = _topk.eq(0).all(dim=-1)
            _topk_times = torch.where(
                zero_idcs,
                torch.zeros_like(_topk_times),
                _topk_times,
            )

            _topk_values = torch.where(
                zero_idcs,
                torch.zeros_like(_topk_values),
                _topk_values,
            )

            # To Prevent NaN
            _topk = torch.where(
                zero_idcs.unsqueeze(-1),
                self.random_token_emb.expand(B * K, 1, E),
                _topk,
            )
            _topk_values += 1e-10  # To Prevent NaN
            _topk_values = (
                _topk_values.reshape(B, K)
                / _topk_values.reshape(B, K).sum(dim=1, keepdim=True)
            ).reshape(B * K)

            res = self.predictor(_topk, times=_topk_times, **kwargs)
            pred = self.emb2out_model(res, **kwargs)

            pred = torch.sum(
                (_topk_values.unsqueeze(-1) * pred).reshape(B, K, -1),
                dim=1,
            )

            return pred

        def _predictor_path():
            topk[:, 0, :] = torch.where(
                topk[:, 0, :].sum(dim=-1, keepdim=True) == 0,
                self.random_token_emb.expand(B, E),
                topk[:, 0, :],
            )  # To prevent NaN
            res = self.predictor(topk, times=topk_times, **kwargs)
            pred = self.emb2out_model(res, **kwargs)

            return pred

        # If training, iterate two paths
        if self.training:
            if self.mode == "retriever":
                pred = _retriever_path()
                self.set_mode("predictor")
            else:
                pred = _predictor_path()
                self.set_mode("retriever")
        # If evaluating, use only predictor path
        else:
            pred = _predictor_path()

        return pred
