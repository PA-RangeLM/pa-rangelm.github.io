"""  Differential Transformer (ICLR 2025)  。

non-English text removed：
third_party/unilm-diff-transformer/Diff-Transformer/multihead_diffattn.py

non-English text removed，non-English text removed10.26/10.20non-English text removed
Prototype Auxiliary Branch + RangeLM non-English text removed。non-English text removed
non-English text removed，non-English text removed；non-English text removed
non-English text removed，non-English text removed。
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def lambda_init_fn(layer_index):
    """  V1  。"""
    return 0.8 - 0.6 * math.exp(-0.3 * float(layer_index))


class RMSNorm(nn.Module):
    """  RMSNorm。"""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(dtype=x.dtype) * self.weight


class ResidualDifferentialCrossAttention(nn.Module):
    """ 。

    Q1/Q2 non-English text removed，K1/K2 non-English text removed V non-English text removed。
    non-English text removed softmax non-English text removed V，non-English text removed RMSNorm。
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        layer_index,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        if embed_dim % (2 * num_heads) != 0:
            raise ValueError(
                'embed_dim must be divisible by 2*num_heads for differential attention'
            )

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.qk_head_dim = self.embed_dim // (2 * self.num_heads)
        self.value_head_dim = self.embed_dim // self.num_heads
        self.scale = self.qk_head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)

        self.lambda_init = lambda_init_fn(layer_index)
        self.lambda_q1 = nn.Parameter(torch.empty(self.qk_head_dim))
        self.lambda_k1 = nn.Parameter(torch.empty(self.qk_head_dim))
        self.lambda_q2 = nn.Parameter(torch.empty(self.qk_head_dim))
        self.lambda_k2 = nn.Parameter(torch.empty(self.qk_head_dim))
        for parameter in (
            self.lambda_q1,
            self.lambda_k1,
            self.lambda_q2,
            self.lambda_k2,
        ):
            nn.init.normal_(parameter, mean=0.0, std=0.1)

        self.head_norm = RMSNorm(self.value_head_dim, eps=1e-5)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query, memory):
        batch_size, query_count, _ = query.shape
        memory_count = memory.size(1)

        q = self.q_proj(query).view(
            batch_size,
            query_count,
            2 * self.num_heads,
            self.qk_head_dim,
        ).permute(0, 2, 1, 3)
        k = self.k_proj(memory).view(
            batch_size,
            memory_count,
            2 * self.num_heads,
            self.qk_head_dim,
        ).permute(0, 2, 1, 3)
        value = self.v_proj(memory).view(
            batch_size,
            memory_count,
            self.num_heads,
            self.value_head_dim,
        ).permute(0, 2, 1, 3)

        logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attention = F.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        attention = self.attn_drop(attention)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1).float())
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2).float())
        lambda_full = (lambda_1 - lambda_2 + self.lambda_init).to(q.dtype)

        attention = attention.view(
            batch_size,
            self.num_heads,
            2,
            query_count,
            memory_count,
        )
        differential_weights = attention[:, :, 0] - lambda_full * attention[:, :, 1]

        correction = torch.matmul(differential_weights, value)
        correction = self.head_norm(correction)
        correction = correction * (1.0 - self.lambda_init)
        correction = correction.transpose(1, 2).reshape(
            batch_size, query_count, self.embed_dim
        )
        correction = self.out_proj(correction)
        return self.proj_drop(correction)
