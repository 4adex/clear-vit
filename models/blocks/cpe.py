import torch
import torch.nn as nn
from typing import Tuple


class ConditionalPositionEncoding(nn.Module):
    """
    Conditional Position Encoding (CPE) for Vision Transformers.
    
    CPE generates positional encodings conditionally based on the input tokens,
    using depth-wise convolutions. This allows the model to learn position information
    that is input-dependent rather than fixed.
    
    Reference: "Conditional Positional Encodings for Vision Transformers"
    https://arxiv.org/abs/2102.10882
    
    Args:
        dim: Hidden dimension of the input tokens
        kernel_size: Size of the convolutional kernel (default: 3)
        padding: Padding for the convolution (default: 1)
        groups: Number of groups for grouped convolution (default: None, uses dim)
    """
    
    def __init__(
        self, 
        dim: int, 
        kernel_size: int = 3,
        padding: int = 1,
        groups: int = None
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.padding = padding
        
        # Use grouped convolution (depth-wise when groups=dim)
        if groups is None:
            groups = dim
            
        self.proj = nn.Conv2d(
            dim, 
            dim, 
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            bias=True
        )
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Apply conditional positional encoding.
        
        Args:
            x: Input tensor of shape (B, N, C) where N = H * W (+ 1 if CLS token exists)
            H: Height of the feature map
            W: Width of the feature map
            
        Returns:
            Output tensor of shape (B, N, C) with added positional information
        """
        B, N, C = x.shape
        
        # Check if CLS token exists (N == H*W + 1)
        has_cls_token = (N == H * W + 1)
        
        if has_cls_token:
            # Separate CLS token and spatial tokens
            cls_token, spatial_tokens = x[:, :1], x[:, 1:]
        else:
            spatial_tokens = x
            
        # Reshape to 2D: (B, N, C) -> (B, C, H, W)
        spatial_tokens = spatial_tokens.transpose(1, 2).reshape(B, C, H, W)
        
        # Apply depth-wise convolution for positional encoding
        spatial_tokens = self.proj(spatial_tokens)
        
        # Reshape back: (B, C, H, W) -> (B, N, C)
        spatial_tokens = spatial_tokens.flatten(2).transpose(1, 2)
        
        # Add back CLS token if it existed
        if has_cls_token:
            x = torch.cat([cls_token, spatial_tokens], dim=1)
        else:
            x = spatial_tokens
            
        return x


class CPE2D(nn.Module):
    """
    2D Conditional Position Encoding with optional zero initialization.
    
    This is a wrapper around ConditionalPositionEncoding that can optionally
    initialize the convolution weights to zero, making CPE a residual connection
    at initialization.
    
    Args:
        dim: Hidden dimension of the input tokens
        kernel_size: Size of the convolutional kernel (default: 3)
        zero_init: Whether to initialize weights to zero (default: False)
    """
    
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        zero_init: bool = False
    ):
        super().__init__()
        padding = kernel_size // 2
        self.cpe = ConditionalPositionEncoding(
            dim=dim,
            kernel_size=kernel_size,
            padding=padding
        )
        
        if zero_init:
            nn.init.zeros_(self.cpe.proj.weight)
            if self.cpe.proj.bias is not None:
                nn.init.zeros_(self.cpe.proj.bias)
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Apply 2D conditional positional encoding.
        
        Args:
            x: Input tensor of shape (B, N, C)
            H: Height of the feature map
            W: Width of the feature map
            
        Returns:
            Output tensor with positional information added
        """
        return x + self.cpe(x, H, W)


class CPEBlock(nn.Module):
    """
    CPE Block that can be inserted into transformer layers.
    
    This block applies CPE and can optionally use it in a residual manner.
    Useful for integrating CPE into existing ViT architectures.
    
    Args:
        dim: Hidden dimension
        kernel_size: Convolution kernel size (default: 3)
        residual: Whether to use residual connection (default: True)
        zero_init: Whether to zero-initialize (default: True when residual=True)
    """
    
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        residual: bool = True,
        zero_init: bool = None
    ):
        super().__init__()
        self.residual = residual
        
        # Default zero_init to True if using residual
        if zero_init is None:
            zero_init = residual
            
        if residual:
            self.cpe = CPE2D(dim=dim, kernel_size=kernel_size, zero_init=zero_init)
        else:
            padding = kernel_size // 2
            self.cpe = ConditionalPositionEncoding(
                dim=dim,
                kernel_size=kernel_size,
                padding=padding
            )
    
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Apply CPE block.
        
        Args:
            x: Input tensor of shape (B, N, C)
            H: Height of the feature map
            W: Width of the feature map
            
        Returns:
            Output tensor with positional encoding
        """
        if self.residual:
            return self.cpe(x, H, W)
        else:
            return self.cpe(x, H, W)