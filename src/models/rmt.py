from typing import List, Optional

import torch
import torch.nn.functional as F
from einops import pack, rearrange, repeat, unpack
from recurrent_memory_transformer_pytorch.recurrent_memory_transformer import *
from torch import nn


class CompatibleRecurrentMemoryTransformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        num_memory_tokens,
        seq_len,
        causal=True,
        dim_head=64,
        heads=8,
        ff_mult=4,
        use_flash_attn=False,
        ignore_index=-1,
        rotary_pos_emb=True,
        use_xl_memories=False,
        xl_mem_len=None,
        enhanced_xl_recurrence=False,  # add simple method for enhancing receptive field of xl memories, from ernie-doc paper
        emb_gradient_frac=0.1,  # trick from cogview paper that leads to a bit more stability
        memory_not_causal=True,  # flash attention behaves a bit more optimally if causal mask is not explicitly passed in - but if the memories perform better without a causal mask, it is necessary to have this turned on
    ):
        super().__init__()
        self.causal = causal
        self.seq_len = seq_len

        self.emb_gradient_frac = emb_gradient_frac

        assert num_memory_tokens > 0

        self.rotary_pos_emb = RotaryEmbedding(dim_head) if rotary_pos_emb else None

        # memory related

        self.num_memory_tokens = num_memory_tokens

        self.read_memory_emb = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.read_memory_emb, std=0.02)

        self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))
        nn.init.normal_(self.memory_tokens, std=0.02)

        # xl memories

        xl_mem_len = default(xl_mem_len, seq_len)
        assert xl_mem_len <= seq_len
        self.xl_mem_len = xl_mem_len

        self.use_xl_memories = use_xl_memories
        assert not (
            rotary_pos_emb and use_xl_memories
        ), "rotary not compatible with xl memories yet"

        self.enhanced_xl_recurrence = enhanced_xl_recurrence

        # layers

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            causal=causal,
                            heads=heads,
                            use_flash_attn=use_flash_attn,
                            use_custom_causal_attn_mask=memory_not_causal,
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.to_logits = RMSNorm(dim)

        self.ignore_index = ignore_index

        # whether to use custom attention mask if causal and memory should not be causal

        self.use_custom_causal_attn_mask = causal and memory_not_causal

    def init_memory(self, batch):
        return repeat(self.memory_tokens, "m d -> b m d", b=batch)

    def forward(
        self,
        x,
        read_memories=None,
        *,
        mask=None,
        labels=None,
        xl_memories: Optional[List[torch.Tensor]] = None,
    ):
        b, n, _, device, mem_length = (
            *x.shape,
            x.device,
            self.num_memory_tokens,
        )

        assert n <= self.seq_len

        pos = torch.arange(n, device=device)

        # trick from cogview paper

        x = frac_gradient(x, self.emb_gradient_frac)

        # prepare read and write memories, as in paper

        write_memories = self.init_memory(b)

        if exists(read_memories):
            read_mem_length = mem_length
            read_memories = read_memories + self.read_memory_emb
        else:
            read_mem_length = 0
            read_memories = x[:, 0:0]

        # concat to main sequence using einop's pack

        x, ps = pack([read_memories, x, write_memories], "b * d")

        # take care of mask

        if exists(mask):
            mask = F.pad(mask, (read_mem_length, mem_length), value=True)

        # custom causal mask, if needed

        if self.use_custom_causal_attn_mask:
            causal_mask = torch.ones((n, n), device=device, dtype=torch.bool).tril()

            causal_mask = F.pad(
                causal_mask, (0, mem_length, read_mem_length, 0), value=False
            )
            causal_mask = F.pad(
                causal_mask, (read_mem_length, 0, 0, mem_length), value=True
            )

            assert not exists(mask)
            mask = rearrange(causal_mask, "i j -> 1 1 i j")

        # rotary embedding - offset main positions by 10000, and keep all memories at position 0

        rotary_emb = None

        if exists(self.rotary_pos_emb):
            pos = pos + 10000
            pos = F.pad(pos, (read_mem_length, mem_length), value=0)

            rotary_emb = self.rotary_pos_emb(pos)

        # prepare xl memories

        xl_memories = default(xl_memories, [])
        xl_memories_iter = iter(xl_memories)
        new_xl_memories = []

        if (
            self.enhanced_xl_recurrence and len(xl_memories) > 1
        ):  # simply shift all the xl memories down by one, so lower layer gets access to representations from layer above
            xl_memories = [*xl_memories[1:], xl_memories[0]]

        # attention and feedforward

        for attn, ff in self.layers:
            attn_out, xl_memories = attn(
                x,
                mask=mask,
                xl_memories=next(xl_memories_iter, None),
                rotary_emb=rotary_emb,
            )
            new_xl_memories.append(xl_memories)

            x = x + attn_out

            x = ff(x) + x

        # whether to return xl memories

        next_xl_memories = None

        if self.use_xl_memories:
            next_xl_memories = list(
                map(
                    lambda t: torch.detach(t[..., -self.xl_mem_len :, :]),
                    new_xl_memories,
                )
            )

        # split out memories using unpack

        read_memories, x, write_memories = unpack(x, ps, "b * d")

        # to logits

        logits = self.to_logits(x)

        return logits, write_memories, next_xl_memories


class CompatibleRecurrentMemoryTransformerWrapper(nn.Module):
    def __init__(self, transformer: CompatibleRecurrentMemoryTransformer):
        super().__init__()
        self.transformer = transformer
        self.seq_len = transformer.seq_len

    def forward(
        self,
        x,
        memories=None,
        *,
        mask=None,
        xl_memories: Optional[List[torch.Tensor]] = None,
        labels=None,
        rmt_n_chunks=1,
    ):
        seq_len = self.seq_len

        labels = None

        # segment input

        segments = x.split(seq_len, dim=1)
        total_length = x.shape[1]
        num_segments = len(segments)
        segment_length_frac = tuple(map(lambda t: t.shape[1] / total_length, segments))

        # default values

        label_segments = mask_segments = (None,)

        # take care of labels

        if exists(labels):
            label_segments = labels.split(seq_len, dim=-1)

        # take care of the mask

        if exists(mask):
            mask_segments = mask.split(seq_len, dim=1)

        # keep replay buffer

        replay_buffer = [memories]

        # replay buffer for xl memories

        xl_segments = [xl_memories]

        # forward and get all outputs (can be either loss or logits)

        for i, segment, mask_segment, label_segment, loss_weight in zip_longest(
            list(range(len(segments))),
            segments,
            mask_segments,
            label_segments,
            segment_length_frac,
        ):
            # Should flow grad on last chunk -> leftside padding is required!!
            context = (
                nullcontext if i >= len(segments) - rmt_n_chunks else torch.no_grad
            )
            with context():
                output, memories, xl_memories = self.transformer(
                    segment, memories, mask=mask_segment, labels=label_segment
                )

            replay_buffer.append(memories)

            xl_segments.append(xl_memories)

        return output
