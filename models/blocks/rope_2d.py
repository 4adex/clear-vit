import torch
import torch.nn as nn
import math
from typing import Optional

def init_random_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    """
    Initialize random 2D frequencies for RoPE.

    IMPORTANT: `dim` here is interpreted as the *per-head* dimension (head_dim),
    not the full model dimension. If you have full `model_dim`, call with
        head_dim = model_dim // num_heads
    """
    # per-head dimension
    head_dim = dim
    # half the per-head dimension used by complex pairing (we pack pairs -> half length)
    half = head_dim // 2
    if half <= 0:
        raise ValueError(f"head_dim (dim) must be >= 2. Got dim={dim}.")

    mag = 1 / (theta ** (torch.arange(0, head_dim, 4)[: (head_dim // 4)].float() / head_dim))
    # If head_dim is not a multiple of 4, we keep the same logic but ensure mag length matches intended half.
    # Build per-head frequency vectors of length `half`
    freqs_x = []
    freqs_y = []
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        # mag has length head_dim//4, and each of the two concatenations produces half length
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1)
        # Ensure length is exactly `half` (in case of rounding)
        fx = fx[:half]
        fy = fy[:half]
        freqs_x.append(fx)
        freqs_y.append(fy)

    freqs_x = torch.stack(freqs_x, dim=0)  # (num_heads, half)
    freqs_y = torch.stack(freqs_y, dim=0)  # (num_heads, half)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)  # (2, num_heads, half)
    return freqs


def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    """
    Compute mixed complex exponentials for 2D RoPE.

    Accepts freqs in either shape:
      (2, num_heads, half)                -> depth = 1
      (2, depth, num_heads, half)         -> depth >= 1

    Returns freqs_cis of shape (depth, num_heads, N, half)
    where N = len(t_x) (sequence length), half = head_dim // 2
    """
    N = t_x.shape[0]

    # Detect freqs layout
    if freqs.ndim == 3:
        # (2, num_heads, half)
        depth = 1
        # use einsum to compute (depth=1, num_heads, N, half)
        freqs_x = torch.einsum('n, h k -> d h n k', t_x, freqs[0].unsqueeze(0))  # d=1
        freqs_y = torch.einsum('n, h k -> d h n k', t_y, freqs[1].unsqueeze(0))
    elif freqs.ndim == 4:
        # (2, depth, num_heads, half)
        depth = freqs.shape[1]
        # freqs[0] -> (depth, num_heads, half)
        # Using einsum directly: 'n, d h k -> d h n k'
        freqs_x = torch.einsum('n, d h k -> d h n k', t_x, freqs[0])
        freqs_y = torch.einsum('n, d h k -> d h n k', t_y, freqs[1])
    else:
        raise ValueError(f"Unexpected freqs.ndim={freqs.ndim}. Expected 3 or 4 dims.")

    # No float16 for angle computations
    with torch.amp.autocast("cuda", enabled=False):
        # freqs_x/freqs_y now shaped (depth, num_heads, N, half)
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
    # If freqs_cis is (N, M) or (d, h, N, k) pattern
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.ndim == 4 and freqs_cis.shape[0] == x.shape[-3] and freqs_cis.shape[1] == x.shape[-2] and freqs_cis.shape[2] == x.shape[-1]:
        # This branch is unlikely with complex view shapes but included for completeness
        shape = [d if i >= ndim - 4 else 1 for i, d in enumerate(x.shape)]
    else:
        raise ValueError(
            f"freqs_cis shape {freqs_cis.shape} doesn't match expected patterns for x shape {x.shape}"
        )
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """
    xq/xk: (batch, num_heads, seq_len, head_dim)
    freqs_cis: complex tensor shaped (depth, num_heads, seq_len, half) OR (num_heads, seq_len, half)
               after reshape_for_broadcast it'll broadcast to the complex view of xq/xk.

    Note: we view xq/xk as complex by grouping last dim into pairs (..., half, 2)
    """
    # Convert to complex view: last dim must be even (paired)
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Broadcast freqs to match complex-shaped x (xq_ shape)
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # multiply and convert back to real representation, then flatten the paired dims
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)
