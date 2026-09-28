"""Lightweight HiRA integration for the PA-RangeLM Transformer decoder.

non-English text removed：hqsiswiliam/hira（ICLR 2025）。non-English text removed，
non-English text removed A、B；non-English text removed，
non-English text removed。non-English text removed：

    W_eff = W_0 * (1 + scale * (B @ A))

non-English text removed ``nn.Linear``，non-English text removed ``weight``/``bias`` non-English text removed，
non-English text removed Prototype+RangeLM non-English text removed。
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class HiRALinear(nn.Linear):
    """  Hadamard  。"""

    def __init__(self, in_features, out_features, bias=True, rank=8, scale=1.0):
        if rank <= 0:
            raise ValueError('HiRA rank must be positive, got {}'.format(rank))
        super().__init__(in_features, out_features, bias=bias)
        self.hira_rank = int(rank)
        self.hira_scale = float(scale)

        self.hira_A = nn.Parameter(torch.empty(self.hira_rank, in_features))
        self.hira_B = nn.Parameter(torch.empty(out_features, self.hira_rank))
        self.reset_hira_parameters()

        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def reset_hira_parameters(self):
        """ ：A=Kaiming，B=0， 。"""
        nn.init.kaiming_uniform_(self.hira_A, a=math.sqrt(5))
        nn.init.zeros_(self.hira_B)

    @classmethod
    def from_linear(cls, linear, rank=8, scale=1.0):
        """  ``nn.Linear``   HiRA  。"""
        if not isinstance(linear, nn.Linear):
            raise TypeError('HiRA can only wrap nn.Linear, got {}'.format(type(linear)))
        adapted = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            rank=rank,
            scale=scale,
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        with torch.no_grad():
            adapted.weight.copy_(linear.weight)
            if linear.bias is not None:
                adapted.bias.copy_(linear.bias)
        adapted.train(linear.training)
        return adapted

    def hira_update(self):
        """  B@A。"""
        return torch.matmul(self.hira_B, self.hira_A)

    def effective_weight(self):
        modulation = 1.0 + self.hira_scale * self.hira_update()
        return self.weight * modulation

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.bias)


def attach_hira_to_decoder(decoder, rank=8, scale=1.0):
    """Attach HiRA to all PA-RangeLM Transformer decoder blocks.

    non-English text removed ``QKV + FFN`` non-English text removed：
    self_attn.qkv、cross_attn.q_map/k_map/v_map、mlp.fc1/fc2。
    non-English text removed、non-English text removed、non-English text removed。
    """
    blocks = decoder.blocks.blocks
    target_paths = (
        'self_attn.qkv',
        'cross_attn.q_map',
        'cross_attn.k_map',
        'cross_attn.v_map',
        'mlp.fc1',
        'mlp.fc2',
    )
    attached = []
    for block_index, block in enumerate(blocks):
        for path in target_paths:
            parent = block
            path_parts = path.split('.')
            for part in path_parts[:-1]:
                parent = getattr(parent, part, None)
                if parent is None:
                    break
            if parent is None:
                raise RuntimeError(
                    'Decoder block {} has no HiRA target {}'.format(block_index, path)
                )
            child_name = path_parts[-1]
            original = getattr(parent, child_name, None)
            if not isinstance(original, nn.Linear):
                raise RuntimeError(
                    'Decoder block {} target {} is not nn.Linear'.format(block_index, path)
                )
            setattr(parent, child_name, HiRALinear.from_linear(original, rank=rank, scale=scale))
            attached.append('decoder.blocks.blocks.{}.{}'.format(block_index, path))
    return attached


def iter_hira_parameters(module):
    """  A/B； 。"""
    for child in module.modules():
        if isinstance(child, HiRALinear):
            yield child.hira_A
            yield child.hira_B


def count_hira_modules(module):
    return sum(isinstance(child, HiRALinear) for child in module.modules())
