import torch.nn as nn
import torch
from typing import Callable, OrderedDict, Optional
from functools import partial
from models.blocks import MLPBlock
import math

class MultiHeadAttentionBlock(nn.Module):
    """Multihead attention module as in el at Vaswani"""

    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.dropout = dropout

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)  # concatenated queries of all heads
        self.w_k = nn.Linear(d_model, d_model)  # concatenated keys of all heads
        self.w_v = nn.Linear(d_model, d_model)  # concatenated values of all heads

        # the output projection matrix
        self.w_o = nn.Linear(d_model, d_model)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(
        query, key, value, mask, dropout: nn.Dropout
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(
            dim=-1
        )  # (batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq len, d_model) -> (batch, seq_len, h, dk) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout
        )

        # (Batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)

# class SpatialSuppressionAttention(nn.Module):
#     """Multihead attention but with an injected noise cancelling filter per head"""

#     def __init__(self, d_model: int, h: int, dropout: float, kernel_size: int = 3):
#         super().__init__()
#         self.d_model = d_model
#         self.h = h
#         assert d_model % h == 0, "d_model is not divisible by h"

#         self.dropout = dropout

#         self.d_k = d_model // h

#         self.w_q = nn.Linear(d_model, d_model)
#         self.w_k = nn.Linear(d_model, d_model)
#         self.w_v = nn.Linear(d_model, d_model)

#         self.kernel_radius = kernel_size // 2

#         self.inhibitory_kernel = nn.Parameter(
#             torch.randn(2 * self.kernel_radius +1, 2 * self.kernel_radius + 1)
#         )

#         # Output projection matrix
#         self.w_o = nn.Linear(d_model, d_model)
#         self.dropout = nn.Dropout(dropout)

#     def get_spatial_neighbors(self, seq_len: int, spatial_dim: int):
#         """
#         Get spatial neighbours for each position in a 2D grid.
#         Assumes sequence is arranged as a spatial_dim x spatial_dim grid.
#         """

#         neighbours = {}
#         kernel_radius = self.kernel_radius

#         for i in range(seq_len):
#             row = i // spatial_dim
#             column = i % spatial_dim

#             neighbour_list = []
#             for dr in range(-kernel_radius, kernel_radius + 1):
#                 for dc in range(-kernel_radius, kernel_radius + 1):
#                     if dr == 0 and dc == 0: # skip the central position (the patch itself)
#                         continue

#                     new_row = row + dr
#                     new_col = column + dc

#                     # Check bounds
#                     if 0 <= new_row < spatial_dim and 0 <= new_col < spatial_dim :
#                         neighbour_idx = new_row * spatial_dim + new_col
#                         kernel_idx_row = dc + kernel_radius
#                         kernel_idx_col = dr + kernel_radius
#                         neighbour_list.append((neighbour_idx, kernel_idx_row, kernel_idx_col)) # what is point of having kernel idxes here?
                    
#             neighbours[i] = neighbour_list

#         return neighbours
    
#     def apply_suppression(self, attention_scores: torch.Tensor, spatial_dim: int):
#         """
#         Apply spatial surround supression to attention scores.
#         """

#         batch_size, num_heads, seq_len, _ = attention_scores.shape

#         neighbors = self.get_spatial_neighbors(seq_len, spatial_dim)

#         suppressed_scores = attention_scores.clone()

#         #  Apply surround suppression: S̃ij = Sij - Σ(u,v)∈N(i) Ku-i,v-j * Suv
#         for i in range(seq_len):
#             if i in neighbors:
#                 for neighbor_idx, k_row, k_col in neighbors[i]:
#                     kernel_weight = self.inhibitory_kernel[k_row, k_col]

#                     suppressed_scores[:, :, i, :] -= (
#                         kernel_weight * attention_scores[:, :, neighbor_idx, :]
#                     )
        
#         return suppressed_scores
    
#     # we are not keeping the mask method here
#     def attention_with_suppression(self,
#         query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dropout: nn.Dropout, inhibitory_kernel: torch.Tensor, spatial_dim: int
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         d_k = query.shape[-1]

#         attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

#         # Apply spatial surround suppression
#         if spatial_dim is not None:
#             # apply the surround supression
#             attention_scores = self.apply_suppression(attention_scores, spatial_dim)

#         attention_probs = attention_scores.softmax(dim = -1)



#         attention_probs = dropout(attention_probs)

#         return (attention_probs @ value, attention_probs)
    
#     def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, spatial_dim = None):
#         batch_size, seq_len, _ = q.shape

#         if spatial_dim is None:

#             spatial_dim = int(math.sqrt((seq_len-1)))
#             assert spatial_dim * spatial_dim == (seq_len-1), \
#                 "seq_len must be a perfect square if spatial_dim not provided"

#         query = self.w_q(q).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
#         key = self.w_k(k).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
#         value = self.w_v(v).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)


#         x, _ = self.attention_with_suppression(query, key, value, self.dropout, self.inhibitory_kernel, spatial_dim)

#         x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

#         return self.w_o(x)


class SpatialSuppressionAttention(nn.Module):
    """Multihead attention but with an injected noise cancelling filter per head"""

    def __init__(self, d_model: int, h: int, dropout: float, suppression_radius: int = 1):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.dropout = dropout

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.suppression_radius = suppression_radius
        kernel_size = 2 * suppression_radius + 1

        self.kernel_radius = kernel_size // 2

        self.suppression_conv = nn.Conv2d(
            in_channels=h,
            out_channels=h,
            kernel_size=kernel_size,
            padding=suppression_radius,
            groups=h,  # One kernel per head
            bias=False
        )

        # Output projection matrix
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def attention(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # Raw attention scores
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply surround-suppression
        suppressed_scores = attention_scores - self.suppression_conv(attention_scores)
        
        attention_probs = suppressed_scores.softmax(dim=-1)
        
        if dropout is not None:
            attention_probs = dropout(attention_probs)
            
        return (attention_probs @ value), attention_probs

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        batch_size, seq_len, _ = q.shape
        
        query = self.w_q(q).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, seq_len, self.h, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)

        x, self.attention_scores = self.attention(
            query, key, value, self.dropout
        )

        # (Batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)




class EncoderBlock(nn.Module):
    """Transformer encoder block."""

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
        
        # Here using 
        # self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)

        # Here using SSSAttention
        self.self_attention = SpatialSuppressionAttention(hidden_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

        # Fix init for the custom attention weights
        bound = math.sqrt(3 / hidden_dim)
        nn.init.uniform_(self.self_attention.w_q.weight, -bound, bound)
        nn.init.uniform_(self.self_attention.w_k.weight, -bound, bound)
        nn.init.uniform_(self.self_attention.w_v.weight, -bound, bound)
        nn.init.uniform_(self.self_attention.w_o.weight, -bound, bound)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x= self.self_attention(x, x, x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y
    
class Encoder(nn.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

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
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        return self.ln(self.layers(self.dropout(input)))


def jax_lecun_normal(layer, fan_in):
    """(re-)initializes layer weight in the same way as jax.nn.initializers.lecun_normal and bias to zero"""

    # constant is stddev of standard normal truncated to (-2, 2)
    std = math.sqrt(1 / fan_in) / .87962566103423978
    nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)

class SSSVisionTransformer(nn.Module):
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
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv2d(
            in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )
        
        h = w = image_size // patch_size 

        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_length, hidden_dim) * 0.02, requires_grad=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02, requires_grad=True)

        self.encoder = Encoder(
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
        x = self._process_input(x)
        n = x.shape[0]

        x = x + self.pos_embedding

        cls_token = self.cls_token.expand(n, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        x = self.encoder(x)

        # Use only the CLS token for classification
        x = x[:, 0]

        x = self.heads(x)

        return x