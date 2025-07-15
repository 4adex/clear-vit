import torch.nn as nn
import torch
from typing import Callable, OrderedDict, Optional, Dict
from functools import partial
# from models.blocks import MLPBlock
import math

from detectron2.layers import ShapeSpec
from .backbone import Backbone

class MLPBlock(nn.Module):
    def __init__(self, in_dim, mlp_dim, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, mlp_dim)
        self.activation = nn.GELU()
        self.dropout_1 = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(mlp_dim, in_dim)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x



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
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

        # Fix init discrepancy between nn.MultiheadAttention and that of big_vision
        bound = math.sqrt(3 / hidden_dim)
        nn.init.uniform_(self.self_attention.in_proj_weight, -bound, bound)
        nn.init.uniform_(self.self_attention.out_proj.weight, -bound, bound)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
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

class SimpleVisionTransformer(nn.Module):
    """Vision Transformer modified per https://arxiv.org/abs/2205.01580."""
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



class SimpleViT(Backbone):
    def __init__(
        self,
        img_size=1024,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        attention_dropout=0.0,
        use_abs_pos=True,
        use_cls_token=False,
        out_feature="last_feat",
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size for ViT.
            in_chans (int): Number of input channels (usually 3 for RGB).
            embed_dim (int): Hidden dimension.
            depth (int): Number of encoder blocks.
            num_heads (int): Number of self-attention heads.
            mlp_ratio (float): MLP hidden dim as a ratio of embed_dim.
            dropout (float): Dropout rate.
            attention_dropout (float): Attention dropout rate.
            use_abs_pos (bool): Use absolute positional encoding.
            use_cls_token (bool): If True, prepend class token.
            out_feature (str): Output feature name.
        """
        super().__init__()
        self.use_cls_token = use_cls_token
        self.out_feature = out_feature

        self.vit = SimpleVisionTransformer(
            image_size=img_size,
            patch_size=patch_size,
            num_layers=depth,
            num_heads=num_heads,
            hidden_dim=embed_dim,
            mlp_dim=int(embed_dim * mlp_ratio),
            dropout=dropout,
            attention_dropout=attention_dropout,
            num_classes=0,  # classification head is disabled
            representation_size=None,
        )

        self._out_feature_channels = {out_feature: embed_dim}
        self._out_feature_strides = {out_feature: patch_size}
        self._out_features = [out_feature]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        n, c, h, w = x.shape
        patch_size = self.vit.patch_size

        print(f"Input shape: {(h, w)}, Patch size: {patch_size}")
        
        # Remove strict assertion - handle variable sizes like standard ViT
        # assert h % patch_size == 0 and w % patch_size == 0, "Image size must be divisible by patch size"

        x = self.vit._process_input(x)
        
        # Handle positional embedding for variable input sizes
        current_seq_len = x.shape[1]  # Number of patches
        if current_seq_len != self.vit.pos_embedding.shape[1]:
            # Interpolate positional embedding to match current sequence length
            pos_embed = self._interpolate_pos_embed(
                self.vit.pos_embedding, 
                current_seq_len
            )
        else:
            pos_embed = self.vit.pos_embedding
            
        x = x + pos_embed
        
        if self.use_cls_token:
            cls_token = self.vit.cls_token.expand(n, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            x = self.vit.encoder(x)
            x = x[:, 1:]  # drop CLS token
        else:
            x = self.vit.encoder(x)

        # Convert back to feature map shape (N, C, H', W')
        h_patches = h // patch_size
        w_patches = w // patch_size
        x = x.transpose(1, 2).reshape(n, self.vit.hidden_dim, h_patches, w_patches)

        return {self.out_feature: x}

    def _interpolate_pos_embed(self, pos_embed, target_seq_len):
        """Interpolate positional embeddings for variable input sizes."""
        import torch.nn.functional as F
        
        # Assuming square patches
        old_seq_len = pos_embed.shape[1]
        old_size = int(math.sqrt(old_seq_len))
        new_size = int(math.sqrt(target_seq_len))
        
        if old_size != new_size:
            pos_embed = pos_embed.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(
                pos_embed, size=(new_size, new_size), mode='bicubic', align_corners=False
            )
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, target_seq_len, -1)
        
        return pos_embed

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }