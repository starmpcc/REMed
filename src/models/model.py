import re

import torch
import torch.nn as nn
from accelerate.logging import get_logger
from torch.nn.functional import pad
from transformers import AutoModel
from transformers.models.roformer.modeling_roformer import (
    RoFormerConfig,
    RoFormerEncoder,
)

from .eventencoder import *
from .model_utils import *

try:
    from performer_pytorch import Performer
    from performer_pytorch.performer_pytorch import FixedPositionalEmbedding

    from .mega import MegaLayer
    from .rmt import (
        CompatibleRecurrentMemoryTransformer,
        CompatibleRecurrentMemoryTransformerWrapper,
    )
    from .s4 import S4Layer
except:
    pass

logger = get_logger(__name__, "INFO")


class BioClinicalBERT(nn.Module):
    class _Inp2Emb(nn.Module):
        def __init__(self, args):
            super().__init__()
            self.args = args
            self.model = AutoModel.from_pretrained(
                "emilyalsentzer/Bio_ClinicalBERT", max_length=args.max_word_len
            )

        def forward(self, input_ids, **kwargs):
            B, S, W = input_ids.shape
            input_ids = input_ids.view(B * S, W)
            mask = input_ids.ne(0)
            output = self.model(input_ids, attention_mask=mask)[1]
            output = output.view(B, S, -1)
            return output

    class _EventEncoder(nn.Module):
        def __init__(self, args):
            super().__init__()

        def forward(self, x, **kwargs):
            return x

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input2emb_model = self._Inp2Emb(args)
        self.eventencoder_model = self._EventEncoder(args)

    def forward(self):
        raise NotImplementedError


class FlattenRMT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input2emb_model = DescEmb(args)
        model = CompatibleRecurrentMemoryTransformer(
            num_memory_tokens=args.rmt_mem_size,
            dim=args.pred_dim,
            depth=args.n_flatten_layers,
            dim_head=args.pred_dim // args.n_heads,
            heads=args.n_heads,
            seq_len=args.rmt_chunk_size - args.rmt_mem_size * 2,
            causal=False,
        )
        self.model = CompatibleRecurrentMemoryTransformerWrapper(model)
        self.emb2out_model = PredOutPutLayer(args)

    def forward(self, n_chunks, input_ids, **kwargs):
        x = self.input2emb_model(input_ids, **kwargs)
        mask = input_ids.ne(0)
        x = self.model(x, mask=mask, rmt_n_chunks=n_chunks)
        x = self.emb2out_model(x, **kwargs)
        return x, None


class CachedRMT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.time_enc = ReprTimeEnc(args)
        model = CompatibleRecurrentMemoryTransformer(
            num_memory_tokens=args.rmt_mem_size,
            dim=args.pred_dim,
            depth=args.n_agg_layers,
            dim_head=args.pred_dim // args.n_heads,
            heads=args.n_heads,
            seq_len=args.rmt_chunk_size - args.rmt_mem_size * 2,
            causal=False,
        )
        self.model = CompatibleRecurrentMemoryTransformerWrapper(model)
        self.emb2out_model = PredOutPutLayer(args)

    def forward(self, n_chunks, repr, **kwargs):
        x, _, src_pad_mask = self.time_enc(repr, **kwargs)
        x = self.model(repr, mask=~src_pad_mask, rmt_n_chunks=n_chunks)
        x = self.emb2out_model(x, **kwargs)
        return x, None


class FlattenPerformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input2emb_model = DescEmb(args)
        self.model = Performer(
            args.pred_dim,
            args.n_flatten_layers,
            args.n_heads,
            args.pred_dim // args.n_heads,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )
        self.rotary_emb = FixedPositionalEmbedding(
            args.pred_dim // args.n_heads, args.max_seq_len
        )
        self.emb2out_model = PredOutPutLayer(args)

    def forward(self, input_ids, **kwargs):
        x = self.input2emb_model(input_ids, **kwargs)
        # In performer, True is allowed to attend
        mask = input_ids.ne(0)
        pos_emb = self.rotary_emb(x)
        x = self.model(x, mask=mask, pos_emb=pos_emb)
        x = self.emb2out_model(x, **kwargs)
        return x, None


class CachedPerformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.time_enc = ReprTimeEnc(args)
        self.model = Performer(
            args.pred_dim,
            args.n_agg_layers,
            args.n_heads,
            args.pred_dim // args.n_heads,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )
        self.rotary_emb = FixedPositionalEmbedding(
            args.pred_dim // args.n_heads, args.max_seq_len
        )
        self.emb2out_model = PredOutPutLayer(args)

    def forward(self, repr, **kwargs):
        x, _, src_pad_mask = self.time_enc(repr, **kwargs)
        # In performer, True is allowed to attend
        pos_emb = self.rotary_emb(x)
        x = self.model(repr, mask=~src_pad_mask, pos_emb=pos_emb)
        x = self.emb2out_model(x, **kwargs)
        return x, None


class FlattenMega(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input2emb_model = DescEmb(args)
        self.mega = nn.ModuleList(
            [MegaLayer(args) for _ in range(args.n_flatten_layers)]
        )
        self.emb2out_model = PredOutPutLayer(args)

    def forward(self, input_ids, **kwargs):
        x = self.input2emb_model(input_ids, **kwargs)
        # In MEGA, True is allowd to attend
        src_pad_mask = input_ids.ne(0).long()
        for l in self.mega:
            x = l(x, mask=src_pad_mask)
        net_output = self.emb2out_model(x, **kwargs)
        return net_output, None


class CachedMega(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.time_enc = ReprTimeEnc(args)
        self.mega = nn.ModuleList([MegaLayer(args) for _ in range(args.n_agg_layers)])
        self.emb2out_model = PredOutPutLayer(args)

    def forward(self, repr, **kwargs):
        x, _, src_pad_mask = self.time_enc(repr, **kwargs)
        x = repr
        for l in self.mega:
            x = l(x, mask=(~src_pad_mask).long())
        net_output = self.emb2out_model(x, **kwargs)
        return net_output, None


class FlattenS4(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.input2emb_model = DescEmb(args)
        self.s4 = nn.ModuleList([S4Layer(args) for _ in range(args.n_flatten_layers)])
        self.emb2out_model = PredOutPutLayer(args)

    def forward(self, input_ids, **kwargs):
        x = self.input2emb_model(input_ids, **kwargs)
        mask = input_ids.ne(0).unsqueeze(-1)
        # In S4, True is allowd to attend
        for l in self.s4:
            x = l(x.float(), mask)
        net_output = self.emb2out_model(x, **kwargs)
        return net_output, None


class CachedS4(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.time_enc = ReprTimeEnc(args)
        self.s4 = nn.ModuleList([S4Layer(args) for _ in range(args.n_agg_layers)])
        self.emb2out_model = PredOutPutLayer(args)

    def forward(self, repr, **kwargs):
        x, _, src_pad_mask = self.time_enc(repr, **kwargs)
        for l in self.s4:
            x = l(x.float(), ~src_pad_mask.unsqueeze(-1))
        net_output = self.emb2out_model(x, **kwargs)
        return net_output, None


class RetrieverMLP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        pred_dim = args.pred_dim
        if args.scorer_use_time:
            pred_dim += 1
        self.model = nn.Sequential(
            nn.Linear(pred_dim, args.pred_dim // 2),
            nn.LayerNorm(args.pred_dim // 2),
            nn.ReLU(),
            nn.Linear(args.pred_dim // 2, args.pred_dim // 4),
            nn.LayerNorm(args.pred_dim // 4),
            nn.ReLU(),
            nn.Linear(args.pred_dim // 4, args.pred_dim // 8),
            nn.LayerNorm(args.pred_dim // 8),
            nn.ReLU(),
            nn.Linear(args.pred_dim // 8, 1),
            nn.Sigmoid(),
        )

    def forward(self, repr, time, **kwargs):
        if self.args.scorer_use_time:
            time = time.unsqueeze(-1).float()
            res = self.model(torch.cat([repr, time], dim=-1)).squeeze()
        else:
            res = self.model(repr).squeeze()
        return res


class REMed(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.max_retrieve_len = args.max_retrieve_len

        self.pred_model = TransformerPredictor(args)

        # For numerical stability, Adapted from prompt tuning
        if self.args.query_gen:
            self.query_gen = nn.Sequential(
                nn.Linear(args.pred_dim, args.pred_dim // 2),
                nn.Tanh(),
                nn.Linear(args.pred_dim // 2, args.pred_dim),
            )

            self.register_buffer("initial_query", torch.randn(args.pred_dim))
        elif self.args.scorer:
            self.scorer = RetrieverMLP(args)
        else:
            self.query = nn.Parameter(torch.randn(args.pred_dim))
        self.emb2out_model = PredOutPutLayer(args)
        self.sim = nn.CosineSimilarity(dim=-1)
        self.register_buffer("random_token_emb", torch.randn(args.pred_dim))

        self.requires_grad_(False)
        if self.args.query_gen:
            self.query_gen.requires_grad_(True)
        elif self.args.scorer:
            self.scorer.requires_grad_(True)
        else:
            self.query.requires_grad_(True)

    def set_task(self, task):
        self.task_name = task.name

    def set_mode(self, mode):
        self.mode = mode
        if self.args.query_gen:
            scoring_module = self.query_gen
        elif self.args.scorer:
            scoring_module = self.scorer
        else:
            scoring_module = self.query
        if mode == "scorer":
            self.requires_grad_(False)
            scoring_module.requires_grad_(True)
        elif mode == "predictor":
            self.requires_grad_(True)
            scoring_module.requires_grad_(False)

    def forward(self, repr, times, label, **kwargs):
        # Add Padding to use cutoff (if all events is rejected, should retrive paddings all)
        repr = pad(repr, (0, 0, 0, self.max_retrieve_len))
        times = pad(times, (0, self.max_retrieve_len))
        # To implement right-side padding
        times = torch.where(repr.eq(0).all(dim=-1), 1e10, times)
        if self.args.scorer:
            sim = self.scorer(repr, times)
        else:
            if self.args.query_gen:
                query = self.query_gen(self.initial_query)
            else:
                query = self.query
            sim = self.sim(repr, query.unsqueeze(0)).squeeze()

            sim = (sim + 1) / 2
        _sim = torch.where(
            repr.eq(0).all(dim=-1),
            torch.zeros_like(sim) + self.args.rejection_cutoff,
            sim,
        )
        topk_values, topk_indices = torch.topk(_sim, self.max_retrieve_len, dim=1)

        topk = torch.gather(
            repr, 1, topk_indices.unsqueeze(-1).repeat(1, 1, self.args.pred_dim)
        )
        topk_times = torch.gather(times, 1, topk_indices)
        B, K, E = topk.shape

        # Sort for RoFormer (by time!)
        topk_times, topk_indices = topk_times.sort(dim=1)
        topk = topk.gather(1, topk_indices.unsqueeze(-1).repeat(1, 1, E))
        topk_values = topk_values.gather(1, topk_indices)

        if self.training and self.mode == "scorer":
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

            res = self.pred_model(_topk, times=_topk_times, **kwargs)
            res = self.emb2out_model(res, label, **kwargs)

            for k in res["pred"]:
                res["pred"][k] = torch.sum(
                    (_topk_values.unsqueeze(-1) * res["pred"][k]).reshape(B, K, -1),
                    dim=1,
                )

        else:
            topk[:, 0, :] = torch.where(
                topk[:, 0, :].sum(dim=-1, keepdim=True) == 0,
                self.random_token_emb.expand(B, E),
                topk[:, 0, :],
            )  # To prevent NaN
            res = self.pred_model(topk, times=topk_times, **kwargs)
            res = self.emb2out_model(res, label, **kwargs)

        return res, None


class UniHPFAgg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pred_model = TransformerPredictor(args)

        self.emb2out_model = PredOutPutLayer(args)

    def set_mode(self, mode):
        if mode == "classifier":
            self.requires_grad_(False)
            self.emb2out_model.requires_grad_(True)
        elif mode == "all":
            self.requires_grad_(True)

    def forward(self, repr, times, **kwargs):
        x = self.pred_model(repr, times=times, **kwargs)
        net_output = self.emb2out_model(x, **kwargs)

        return net_output, None


class UniHPF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.input2emb_model = DescEmb(args)
        self.eventencoder_model = TransformerEventEncoder(args)

        requires_grad = args.train_type != "long"
        self.input2emb_model.requires_grad_(requires_grad)
        self.eventencoder_model.requires_grad_(requires_grad)

        self.pred_model = TransformerPredictor(args)
        self.emb2out_model = PredOutPutLayer(args)

    def forward(self, **kwargs):
        all_codes_embs = self.input2emb_model(**kwargs)  # (B, S, E)
        events = self.eventencoder_model(all_codes_embs, **kwargs)
        x = self.pred_model(events, **kwargs)
        net_output = self.emb2out_model(x, **kwargs)

        return net_output, events

class DescEmb(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.input_index_size = 28996
        self.type_index_size = 8  # mimic3 + eicu + mimic4
        self.dpe_index_size = 100  # eICU has noisy text-numbers

        self.input_ids_embedding = nn.Embedding(
            self.input_index_size, self.args.pred_dim, padding_idx=0
        )

        self.type_ids_embedding = nn.Embedding(
            self.type_index_size, self.args.pred_dim, padding_idx=0
        )

        self.dpe_ids_embedding = nn.Embedding(
            self.dpe_index_size, self.args.pred_dim, padding_idx=0
        )

        max_len = args.max_word_len

        self.pos_encoder = PositionalEncoding(args.pred_dim, args.dropout, max_len)
        self.time_encoder = FlattenTimeEncoding(args)
        self.layer_norm = nn.LayerNorm(args.pred_dim, eps=1e-12)

    def forward(self, input_ids, type_ids, dpe_ids, times, **kwargs):
        B, S = input_ids.shape[0], input_ids.shape[1]

        x = self.input_ids_embedding(input_ids)
        x += self.type_ids_embedding(type_ids)
        x += self.dpe_ids_embedding(dpe_ids)
        if "flatten" in self.args.train_type:  # (B, S, E) -> (B, S, E)
            x = self.time_encoder(x, times, **kwargs)
        else:  # (B, S, W, E) -> (B*S, W, E)
            x = x.view(B * S, -1, self.args.pred_dim)
            x = self.pos_encoder(x)
        x = self.layer_norm(x)
        return x


class PredOutPutLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.final_projs = nn.ModuleDict()
        for task in args.tasks:
            if task.property == "binary":
                num_header = 1
            else:
                num_header = task.num_classes
            self.final_projs[task.name] = nn.Linear(args.pred_dim, num_header)
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, label, input_ids=None, **kwargs):
        outputs = {"target": label, "pred": {}}

        if self.args.pred_pooling == "cls":
            x = x[:, 0, :]
        elif self.args.pred_pooling == "mean":
            # Note: First tokens of non-padding events are CLS
            if input_ids is not None:
                if len(input_ids.shape) == 3:  # Hi
                    input_ids = input_ids[:, :, 0]
                mask = input_ids.ne(0).unsqueeze(-1)
            else:
                mask = x.ne(0).any(dim=-1).unsqueeze(-1)
            x = (x * mask).sum(dim=1) / mask.sum(dim=1)

        for i, task in enumerate(self.args.tasks):
            logit = self.final_projs[task.name](x)

            if task.property == "multiclass":
                pred = self.softmax(logit)
            else:
                pred = self.sigmoid(logit)

            outputs["pred"][task.name] = pred

        return outputs


class ReprTimeEnc(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        if self.args.pos_enc == "sinusoidal_time":
            div_term = torch.exp(
                torch.arange(0, self.args.pred_dim, 2)
                * (-math.log(10000.0) / self.args.pred_dim)
            )
            self.register_buffer("div_term", div_term)

        elif self.args.pos_enc == "alibi_time_sym":
            assert re.search(r"short|rag|long", args.train_type) is not None
            self.register_buffer(
                "slopes",
                torch.Tensor(get_slopes(args.n_heads, args.alibi_const)),
            )
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, x, times, **kwargs):
        B, S, E = x.shape
        # Process input mask
        times = self.args.pred_time * 60 - times
        src_pad_mask = x.eq(0).all(dim=-1)

        if self.args.pos_enc == "sinusoidal_time":
            pe = torch.zeros_like(x)
            pe[:, :, 0::2] = torch.sin(times.unsqueeze(-1) * self.div_term)
            pe[:, :, 1::2] = torch.cos(times.unsqueeze(-1) * self.div_term)
            x = x + pe
        if self.args.pos_enc == "alibi_time_sym":
            mask = -(times.unsqueeze(1).repeat(1, S, 1) - times.unsqueeze(2)).abs()
            mask = torch.einsum("i, jkl -> jikl", self.slopes, mask).reshape(-1, S, S)
        else:
            mask = None

        x = self.dropout(x)

        return x, mask, src_pad_mask


class TransformerPredictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.max_seq_len = (
            args.max_retrieve_len if args.train_type == "remed" else args.max_seq_len
        )
        self.time_enc = ReprTimeEnc(args)
        config = RoFormerConfig(
            hidden_size=args.pred_dim,
            num_hidden_layers=args.n_agg_layers,
            num_attention_heads=args.n_heads,
            intermediate_size=args.pred_dim * 4,
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
            max_position_embeddings=self.max_seq_len,
        )
        self.model = RoFormerEncoder(config)

    def forward(self, repr, times, **kwargs):
        x, mask, src_pad_mask = self.time_enc(repr, times, **kwargs)
        mask = merge_masks(x, mask, src_pad_mask).unsqueeze(-1).unsqueeze(1)
        x = self.model(x, attention_mask=mask)["last_hidden_state"]
        return x
