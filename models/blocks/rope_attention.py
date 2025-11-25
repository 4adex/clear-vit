import torch
import torch.nn as nn
from models.blocks.rope_2d import apply_rotary_emb

class RoPEAttention(nn.Module):
    """
    Multi-Head Attention with 2D Rotary Position Embeddings.
    """

    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must divide num_heads"

        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis):
        """
        Args:
            x: (B, N, D)
            freqs_cis: rotary frequency tensor of shape (heads, N, head_dim/2)
        """

        B, N, D = x.shape

        # qkv shape: (B, N, 3D)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, heads, N, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # --- Apply RoPE on q and k only ---
        q, k = apply_rotary_emb(q, k, freqs_cis)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        return out
