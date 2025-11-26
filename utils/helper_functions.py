import torch
import torch.nn as nn
import math
from typing import Tuple, Optional, Union


# ============================================================================
# Model Parameter Utilities
# ============================================================================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    Count the number of parameters in a model.
    
    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters
    
    Returns:
        Total number of parameters
    
    Example:
        >>> model = SimpleVisionTransformer(...)
        >>> total_params = count_parameters(model)
        >>> trainable_params = count_parameters(model, trainable_only=True)
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """
    Print a summary of model parameters.
    
    Args:
        model: PyTorch model
        model_name: Name to display in summary
    
    Example:
        >>> model = RoPESimpleVisionTransformer(...)
        >>> print_model_summary(model, "ViT-RoPE")
    """
    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    non_trainable_params = total_params - trainable_params
    
    print(f"\n{'='*60}")
    print(f"{model_name} Parameter Summary")
    print(f"{'='*60}")
    print(f"Total parameters:        {total_params:,}")
    print(f"Trainable parameters:    {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"{'='*60}\n")


# ============================================================================
# Positional Encoding Utilities
# ============================================================================

def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False) -> torch.Tensor:
    """
    Generate 2D sinusoidal positional embeddings (like MAE/MoCo v3).
    
    Args:
        embed_dim: Embedding dimension
        grid_size: Height/width of the grid (assumes square)
        cls_token: If True, add a zero vector for CLS token
    
    Returns:
        Positional embeddings of shape [grid_size*grid_size (+1 if cls_token), embed_dim]
    
    Example:
        >>> pos_embed = get_2d_sincos_pos_embed(768, 14, cls_token=True)
        >>> print(pos_embed.shape)  # [197, 768] (14*14 + 1 for CLS)
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_w, grid_h, indexing='xy')  # W, H
    grid = torch.stack(grid, dim=0)  # 2, H, W
    
    grid = grid.reshape(2, -1)  # 2, H*W
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token:
        pos_embed = torch.cat([torch.zeros(1, embed_dim), pos_embed], dim=0)
    
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    """
    Generate 2D sinusoidal positional embeddings from a grid.
    
    Args:
        embed_dim: Embedding dimension
        grid: Grid positions of shape [2, H*W]
    
    Returns:
        Positional embeddings of shape [H*W, embed_dim]
    """
    assert embed_dim % 2 == 0
    
    # Use half of dimensions for each axis
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # H*W, D/2
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # H*W, D/2
    
    emb = torch.cat([emb_h, emb_w], dim=1)  # H*W, D
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """
    Generate 1D sinusoidal positional embeddings.
    
    Args:
        embed_dim: Output dimension for each position
        pos: Positions to be encoded, shape [M,]
    
    Returns:
        Positional embeddings of shape [M, embed_dim]
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # D/2
    
    pos = pos.reshape(-1)  # M
    out = torch.einsum('m,d->md', pos, omega)  # M, D/2
    
    emb_sin = torch.sin(out)  # M, D/2
    emb_cos = torch.cos(out)  # M, D/2
    
    emb = torch.cat([emb_sin, emb_cos], dim=1)  # M, D
    return emb


# ============================================================================
# Image and Patch Utilities
# ============================================================================

def get_num_patches(image_size: int, patch_size: int) -> int:
    """
    Calculate number of patches for a square image.
    
    Args:
        image_size: Size of the square image
        patch_size: Size of each square patch
    
    Returns:
        Number of patches (H/P * W/P)
    
    Example:
        >>> num_patches = get_num_patches(224, 16)
        >>> print(num_patches)  # 196 (14 * 14)
    """
    assert image_size % patch_size == 0, f"Image size {image_size} not divisible by patch size {patch_size}"
    return (image_size // patch_size) ** 2


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    Convert images to patches.
    
    Args:
        images: Tensor of shape [B, C, H, W]
        patch_size: Size of each square patch
    
    Returns:
        Patches of shape [B, num_patches, patch_size*patch_size*C]
    
    Example:
        >>> images = torch.randn(8, 3, 224, 224)
        >>> patches = patchify(images, 16)
        >>> print(patches.shape)  # [8, 196, 768]
    """
    B, C, H, W = images.shape
    assert H == W and H % patch_size == 0
    
    num_patches_per_dim = H // patch_size
    patches = images.reshape(B, C, num_patches_per_dim, patch_size, num_patches_per_dim, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(B, num_patches_per_dim**2, -1)
    
    return patches


def unpatchify(patches: torch.Tensor, patch_size: int, channels: int = 3) -> torch.Tensor:
    """
    Convert patches back to images.
    
    Args:
        patches: Tensor of shape [B, num_patches, patch_size*patch_size*C]
        patch_size: Size of each square patch
        channels: Number of image channels
    
    Returns:
        Images of shape [B, C, H, W]
    
    Example:
        >>> patches = torch.randn(8, 196, 768)
        >>> images = unpatchify(patches, 16, 3)
        >>> print(images.shape)  # [8, 3, 224, 224]
    """
    B, num_patches, _ = patches.shape
    num_patches_per_dim = int(num_patches ** 0.5)
    assert num_patches_per_dim ** 2 == num_patches
    
    H = W = num_patches_per_dim * patch_size
    patches = patches.reshape(B, num_patches_per_dim, num_patches_per_dim, channels, patch_size, patch_size)
    images = patches.permute(0, 3, 1, 4, 2, 5).reshape(B, channels, H, W)
    
    return images


# ============================================================================
# Interpolation Utilities (for different input sizes)
# ============================================================================

def interpolate_pos_embed(pos_embed: torch.Tensor, 
                          orig_size: int, 
                          new_size: int,
                          has_cls_token: bool = True) -> torch.Tensor:
    """
    Interpolate positional embeddings for different image sizes.
    Useful when fine-tuning on different resolution than pre-training.
    
    Args:
        pos_embed: Original positional embedding [1, N, D]
        orig_size: Original grid size (e.g., 14 for 224x224 with patch_size=16)
        new_size: New grid size (e.g., 28 for 448x448 with patch_size=16)
        has_cls_token: If True, first token is CLS token (not interpolated)
    
    Returns:
        Interpolated positional embedding
    
    Example:
        >>> # Pre-trained on 224x224, fine-tune on 384x384
        >>> orig_pos_embed = model.pos_embedding  # [1, 197, 768]
        >>> new_pos_embed = interpolate_pos_embed(orig_pos_embed, 14, 24, True)
        >>> print(new_pos_embed.shape)  # [1, 577, 768] (24*24 + 1)
    """
    if orig_size == new_size:
        return pos_embed
    
    if has_cls_token:
        cls_token, pos_tokens = pos_embed[:, :1], pos_embed[:, 1:]
    else:
        pos_tokens = pos_embed
    
    # Reshape to 2D grid
    embed_dim = pos_tokens.shape[-1]
    pos_tokens = pos_tokens.reshape(1, orig_size, orig_size, embed_dim).permute(0, 3, 1, 2)
    
    # Interpolate
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, 
        size=(new_size, new_size), 
        mode='bicubic', 
        align_corners=False
    )
    
    # Reshape back
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_size * new_size, embed_dim)
    
    if has_cls_token:
        pos_embed = torch.cat([cls_token, pos_tokens], dim=1)
    else:
        pos_embed = pos_tokens
    
    return pos_embed


# ============================================================================
# Initialization Utilities
# ============================================================================

def init_weights_vit_timm(module: nn.Module):
    """
    Initialize weights following timm's ViT implementation.
    
    Args:
        module: PyTorch module to initialize
    
    Example:
        >>> model = SimpleVisionTransformer(...)
        >>> model.apply(init_weights_vit_timm)
    """
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_vit_jax(module: nn.Module):
    """
    Initialize weights following JAX's ViT implementation (big_vision).
    
    Args:
        module: PyTorch module to initialize
    
    Example:
        >>> model = SimpleVisionTransformer(...)
        >>> model.apply(init_weights_vit_jax)
    """
    if isinstance(module, nn.Linear):
        if module.weight.shape[0] == module.weight.shape[1]:  # Square matrix
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.normal_(module.bias, std=1e-6)
    elif isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ============================================================================
# Attention Utilities
# ============================================================================

def get_attention_mask(seq_len: int, causal: bool = False, device: str = 'cpu') -> Optional[torch.Tensor]:
    """
    Generate attention mask for transformers.
    
    Args:
        seq_len: Sequence length
        causal: If True, create causal (autoregressive) mask
        device: Device to create tensor on
    
    Returns:
        Attention mask of shape [seq_len, seq_len] or None
    
    Example:
        >>> mask = get_attention_mask(10, causal=True)
        >>> print(mask)  # Lower triangular matrix
    """
    if not causal:
        return None
    
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask


def compute_attention_stats(attn_weights: torch.Tensor) -> dict:
    """
    Compute statistics about attention weights for analysis.
    
    Args:
        attn_weights: Attention weights [B, num_heads, seq_len, seq_len]
    
    Returns:
        Dictionary with attention statistics
    
    Example:
        >>> attn_weights = torch.softmax(torch.randn(8, 12, 197, 197), dim=-1)
        >>> stats = compute_attention_stats(attn_weights)
        >>> print(stats['entropy'])
    """
    B, H, N, _ = attn_weights.shape
    
    # Compute entropy (measure of attention dispersion)
    entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean()
    
    # Compute attention distance (how far tokens attend on average)
    positions = torch.arange(N, device=attn_weights.device).float()
    pos_diff = positions.unsqueeze(0) - positions.unsqueeze(1)
    avg_distance = (attn_weights * pos_diff.abs().unsqueeze(0).unsqueeze(0)).sum(dim=-1).mean()
    
    # Compute attention to CLS token (if exists)
    cls_attention = attn_weights[:, :, :, 0].mean()
    
    return {
        'entropy': entropy.item(),
        'avg_distance': avg_distance.item(),
        'cls_attention': cls_attention.item(),
        'max_attention': attn_weights.max().item(),
        'min_attention': attn_weights.min().item(),
    }


# ============================================================================
# Checkpoint Utilities
# ============================================================================

def load_checkpoint(model: nn.Module, 
                   checkpoint_path: str,
                   strict: bool = True,
                   map_location: str = 'cpu') -> dict:
    """
    Load model checkpoint with error handling.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to checkpoint file
        strict: If True, require exact match of state dict keys
        map_location: Device to load checkpoint on
    
    Returns:
        Dictionary with checkpoint metadata (epoch, best_acc, etc.)
    
    Example:
        >>> model = SimpleVisionTransformer(...)
        >>> metadata = load_checkpoint(model, 'best_model.pth')
        >>> print(f"Loaded checkpoint from epoch {metadata.get('epoch', 'unknown')}")
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel/DistributedDataParallel wrapped models
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    print("✓ Checkpoint loaded successfully")
    
    return {k: v for k, v in checkpoint.items() if k != 'model_state_dict' and k != 'state_dict'}


def save_checkpoint(model: nn.Module,
                   checkpoint_path: str,
                   epoch: int = 0,
                   optimizer: Optional[nn.Module] = None,
                   **kwargs):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        checkpoint_path: Path to save checkpoint
        epoch: Current epoch
        optimizer: Optimizer state (optional)
        **kwargs: Additional metadata to save
    
    Example:
        >>> save_checkpoint(
        ...     model, 
        ...     'checkpoint.pth', 
        ...     epoch=10, 
        ...     optimizer=optimizer,
        ...     best_acc=95.2
        ... )
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    checkpoint.update(kwargs)
    
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved to: {checkpoint_path}")


# ============================================================================
# Visualization Utilities
# ============================================================================

def visualize_attention_map(attn_weights: torch.Tensor,
                           token_idx: int = 0,
                           head_idx: int = 0) -> torch.Tensor:
    """
    Extract attention map for visualization.
    
    Args:
        attn_weights: Attention weights [B, num_heads, seq_len, seq_len]
        token_idx: Which token's attention to visualize (0 for CLS)
        head_idx: Which attention head to visualize
    
    Returns:
        Attention map [seq_len] for the specified token and head
    
    Example:
        >>> attn_map = visualize_attention_map(attn_weights, token_idx=0, head_idx=0)
        >>> # Plot this with matplotlib to see what CLS token attends to
    """
    return attn_weights[0, head_idx, token_idx, :]


# ============================================================================
# Debugging Utilities
# ============================================================================

def check_nan_inf(tensor: torch.Tensor, name: str = "tensor"):
    """
    Check for NaN or Inf values in tensor (useful for debugging).
    
    Args:
        tensor: Tensor to check
        name: Name for logging
    
    Raises:
        ValueError if NaN or Inf found
    
    Example:
        >>> x = model(images)
        >>> check_nan_inf(x, "model_output")
    """
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")


def print_tensor_stats(tensor: torch.Tensor, name: str = "tensor"):
    """
    Print statistics about a tensor (for debugging).
    
    Args:
        tensor: Tensor to analyze
        name: Name for logging
    
    Example:
        >>> print_tensor_stats(attention_weights, "attention")
    """
    print(f"\n{name} Statistics:")
    print(f"  Shape: {tensor.shape}")
    print(f"  dtype: {tensor.dtype}")
    print(f"  device: {tensor.device}")
    print(f"  min: {tensor.min().item():.6f}")
    print(f"  max: {tensor.max().item():.6f}")
    print(f"  mean: {tensor.mean().item():.6f}")
    print(f"  std: {tensor.std().item():.6f}")
    print(f"  Has NaN: {torch.isnan(tensor).any().item()}")
    print(f"  Has Inf: {torch.isinf(tensor).any().item()}")