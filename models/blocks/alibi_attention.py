import math
import torch
from torch import nn
from torch.nn import functional as F


def get_relative_positions(seq_len: int) -> torch.Tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y  # shape: (seq_len, seq_len)


def get_alibi_slope(num_heads: int) -> torch.Tensor:
    x = (2 ** 8) ** (1 / num_heads)
    m = torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
    return m.unsqueeze(-1).unsqueeze(-1)  # (num_heads, 1, 1)


class ALiBiAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0, causal=True, max_len=2048):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.causal = causal
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

        self.kqv = nn.Linear(dim, dim * 3, bias=False)
        self.register_buffer("slopes", get_alibi_slope(num_heads))

        if causal:
            self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len)))

    def forward(self, x):
        """
        x: (B, N, D)
        """
        B, N, D = x.shape

        q, k, v = self.kqv(x).chunk(3, dim=-1)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,N,Dh)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B,H,N,Dh)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # attention score
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B,H,N,N)

        # ALiBi bias
        rel = get_relative_positions(N).to(x.device)  # (N,N)
        bias = self.slopes * rel  # (H, N, N)
        attn = attn + bias.unsqueeze(0)  # (B, H, N, N)

        # causal mask
        if self.causal:
            attn = attn.masked_fill(self.mask[:N, :N] == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B,H,N,Dh)
        out = out.transpose(1, 2).reshape(B, N, D)  # (B,N,D)
        return self.dropout(out)
