"""Utility functions for the clear-vit project."""

from .helper_functions import (
    # Parameter utilities
    count_parameters,
    print_model_summary,
    
    # Positional encoding utilities
    get_2d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid,
    
    # Patch utilities
    get_num_patches,
    patchify,
    unpatchify,
    
    # Interpolation utilities
    interpolate_pos_embed,
    
    # Initialization utilities
    init_weights_vit_timm,
    init_weights_vit_jax,
    
    # Attention utilities
    get_attention_mask,
    compute_attention_stats,
    visualize_attention_map,
    
    # Checkpoint utilities
    load_checkpoint,
    save_checkpoint,
    
    # Debugging utilities
    check_nan_inf,
    print_tensor_stats,
)

__all__ = [
    'count_parameters',
    'print_model_summary',
    'get_2d_sincos_pos_embed',
    'get_1d_sincos_pos_embed_from_grid',
    'get_num_patches',
    'patchify',
    'unpatchify',
    'interpolate_pos_embed',
    'init_weights_vit_timm',
    'init_weights_vit_jax',
    'get_attention_mask',
    'compute_attention_stats',
    'visualize_attention_map',
    'load_checkpoint',
    'save_checkpoint',
    'check_nan_inf',
    'print_tensor_stats',
]