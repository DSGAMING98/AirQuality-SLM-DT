"""Compact decoder-only language model trained from random initialization.

Qwen-inspired *educational* architecture: RoPE + GQA + QK RMSNorm + SwiGLU.
This is NOT official Qwen3 weights or a pretrained Qwen3 model.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MiniQwenConfig:
    vocab_size: int
    d_model: int = 192
    n_heads: int = 6
    n_kv_heads: int = 2
    n_layers: int = 3
    d_ff: int = 512
    max_seq_len: int = 160

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        assert self.n_heads % self.n_kv_heads == 0
        assert (self.d_model // self.n_heads) % 2 == 0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def make_rope_cache(length: int, head_dim: int):
    position = torch.arange(length, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = position[:, None] * inv_freq[None, :]
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # x shape: [batch, heads, sequence, head_dim]
    sequence_length = x.size(-2)
    cos = cos[:sequence_length].to(x.dtype)[None, None, :, :]
    sin = sin[:sequence_length].to(x.dtype)[None, None, :, :]
    even, odd = x[..., ::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: MiniQwenConfig):
        super().__init__()
        self.heads = config.n_heads
        self.kv_heads = config.n_kv_heads
        self.dim_per_head = config.d_model // config.n_heads
        self.q = nn.Linear(config.d_model, config.n_heads * self.dim_per_head, bias=False)
        self.k = nn.Linear(config.d_model, config.n_kv_heads * self.dim_per_head, bias=False)
        self.v = nn.Linear(config.d_model, config.n_kv_heads * self.dim_per_head, bias=False)
        self.o = nn.Linear(config.n_heads * self.dim_per_head, config.d_model, bias=False)
        self.q_norm = RMSNorm(self.dim_per_head)
        self.k_norm = RMSNorm(self.dim_per_head)

    def forward(self, x, cos, sin):
        batch, sequence, _ = x.shape
        q = self.q(x).view(batch, sequence, self.heads, self.dim_per_head).transpose(1, 2)
        k = self.k(x).view(batch, sequence, self.kv_heads, self.dim_per_head).transpose(1, 2)
        v = self.v(x).view(batch, sequence, self.kv_heads, self.dim_per_head).transpose(1, 2)
        q = apply_rope(self.q_norm(q), cos, sin)
        k = apply_rope(self.k_norm(k), cos, sin)
        k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
        v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        # Optimized causal attention avoids explicitly building NxN attention matrices.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
        out = out.transpose(1, 2).contiguous().view(batch, sequence, -1)
        return self.o(out)


class SwiGLU(nn.Module):
    def __init__(self, config: MiniQwenConfig):
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.down = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: MiniQwenConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        return x + self.ffn(self.ffn_norm(x))


class MiniQwenModel(nn.Module):
    def __init__(self, config: MiniQwenConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.output = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output.weight = self.tok_emb.weight  # weight tying, lowers memory use
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        cos, sin = make_rope_cache(config.max_seq_len, config.d_model // config.n_heads)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def forward(self, input_ids, labels=None):
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError('Input sequence exceeds max_seq_len')
        h = self.tok_emb(input_ids)
        for block in self.blocks:
            h = block(h, self.rope_cos, self.rope_sin)
        h = self.final_norm(h)
        if labels is None:
            return self.output(h[:, -1, :])  # only latest token needed to generate
        active = labels != -100
        if not active.any():
            raise ValueError('No answer tokens available for loss calculation')
        # Project only supervised answer positions to huge Qwen vocabulary.
        # This is dramatically more memory-efficient than [batch, seq, vocab].
        supervised_logits = self.output(h[active])
        return F.cross_entropy(supervised_logits.float(), labels[active])

    @torch.no_grad()
    def generate(self, input_ids, eos_token_id: int, max_new_tokens: int = 50):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.max_seq_len:]
            logits = self(context)
            next_id = logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_id), dim=-1)
            if (next_id == eos_token_id).all():
                break
        self.train(was_training)
        return input_ids
