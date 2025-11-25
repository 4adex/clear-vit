import torch
from torch import nn
from torch.nn import functional as F

from .alibi_attention import ALiBiAttention


class ALiBiBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1, causal=True):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = ALiBiAttention(dim, num_heads, dropout, causal=causal)

        hidden = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
