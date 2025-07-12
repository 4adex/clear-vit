import torch.nn as nn
import torch
from typing import Callable, OrderedDict, Optional
from functools import partial
from models.blocks import MLPBlock
import math

def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)        
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    depth = freqs.shape[1]
    # No float 16 for this range
    with torch.amp.autocast("cuda",enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(depth, N, num_heads, -1).permute(0, 2, 1, 3)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
                    
    return freqs_cis

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
        
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)

class RoPEEncoderBlock(nn.Module):
    """Transformer encoder block with RoPE support."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = RoPEMultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor, freqs_cis=None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, freqs_cis=freqs_cis, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
    
class RoPEEncoder(nn.Module):
    """Transformer Model Encoder with RoPE support."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = RoPEEncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.ModuleDict(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor, freqs_cis_list=None):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.dropout(input)
        
        for i, (name, layer) in enumerate(self.layers.items()):
            freqs_cis = freqs_cis_list[i] if freqs_cis_list is not None else None
            x = layer(x, freqs_cis=freqs_cis)
        
        return self.ln(x)


def jax_lecun_normal(layer, fan_in):
    """(re-)initializes layer weight in the same way as jax.nn.initializers.lecun_normal and bias to zero"""

    # constant is stddev of standard normal truncated to (-2, 2)
    std = math.sqrt(1 / fan_in) / .87962566103423978
    nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class RoPEMultiheadAttention(nn.Module):
    """ Custom multihead attention for Rope 2d"""

    def __init__(self, d_model: int, h: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.dropout = dropout
        
        self.d_k = d_model // h
        
        # Separate linear layers for Q, K, V (similar to MultiHeadAttentionBlock)
        self.w_q = nn.Linear(d_model, d_model)  # concatenated queries of all heads
        self.w_k = nn.Linear(d_model, d_model)  # concatenated keys of all heads
        self.w_v = nn.Linear(d_model, d_model)  # concatenated values of all heads
        
        # Output projection matrix
        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout_layer = nn.Dropout(dropout)
        
        # self._reset_parameters()

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = query.shape[-1]
        
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
    
    def forward(self, q, k, v, freqs_cis=None, mask=None, need_weights=False):
        # Linear projections
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        
        # Apply RoPE to q and k (skip CLS token at position 0)
        if freqs_cis is not None:
            q_rope = query[:, :, 1:]  # Skip CLS token
            k_rope = key[:, :, 1:]    # Skip CLS token
            q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, freqs_cis)
            query = torch.cat([query[:, :, :1], q_rope], dim=2)  # Concat back with CLS token
            key = torch.cat([key[:, :, :1], k_rope], dim=2)      # Concat back with CLS token
        
        # Apply attention
        x, self.attention_scores = RoPEMultiheadAttention.attention(
            query, key, value, mask, self.dropout_layer
        )
        
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        output = self.w_o(x)
        
        if need_weights:
            return output, self.attention_scores
        return output, None


class RoPESimpleVisionTransformer(nn.Module):
    """Vision Transformer with RoPE"""
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 100,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        rope_theta: float = 10.0,
        use_traditional_pos_emb: bool = False, # Option to use both RoPE and traditional pos embedding
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.representation_size = representation_size
        self.norm_layer = norm_layer
        self.rope_theta = rope_theta
        self.use_traditional_pos_emb = use_traditional_pos_emb

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        
        h = w = image_size // patch_size 

        seq_length = (image_size // patch_size) ** 2
        if self.use_traditional_pos_emb:
            self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, hidden_dim) * 0.02, requires_grad=True)
        else:
            self.pos_embedding = None
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02, requires_grad=True)

        self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)
        freqs = []
        for i in range(self.num_layers):
            freqs.append(
                init_random_2d_freqs(dim=hidden_dim // num_heads, num_heads=num_heads, theta=rope_theta)
            )
        freqs = torch.stack(freqs, dim=1).view(2, self.num_layers, -1)
        self.freqs = nn.Parameter(freqs.clone(), requires_grad=True)

        t_x, t_y = init_t_xy(end_x=h, end_y=w)
        self.register_buffer('freqs_t_x', t_x)
        self.register_buffer('freqs_t_y', t_y)


        self.encoder = RoPEEncoder(
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)
        
        # Init the patchify stem    
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1] // self.conv_proj.groups
        jax_lecun_normal(self.conv_proj, fan_in)

        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            jax_lecun_normal(self.heads.pre_logits, fan_in)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x
    
    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        n, c, h, w = x.shape
        x = self._process_input(x)
        n = x.shape[0]

        if self.use_traditional_pos_emb:
            x = x + self.pos_embedding

        cls_token = self.cls_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        # Compute RoPE frequencies
        # Handle different input sizes
        if self.freqs_t_x.shape[0] != x.shape[1] - 1:  # -1 for CLS token
            t_x, t_y = init_t_xy(end_x=w // self.patch_size, end_y=h // self.patch_size)
            t_x, t_y = t_x.to(x.device), t_y.to(x.device)
        else:
            t_x, t_y = self.freqs_t_x, self.freqs_t_y
        
        freqs_cis_list = self.compute_cis(self.freqs, t_x, t_y)

        x = self.encoder(x, freqs_cis_list=freqs_cis_list)

        # Use only the CLS token for classification
        x = x[:, 0]

        x = self.heads(x)

        return x