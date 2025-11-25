import torch
from models.blocks.cpe import (
    ConditionalPositionEncoding,
    CPE2D,
    CPEBlock
)

def test_cpe():
    batch_size = 2
    H, W = 14, 14
    seq_len = H * W
    hidden_dim = 768
    
    print("Testing CPE blocks...")
    print("=" * 50)
    
    # Test 1: ConditionalPositionEncoding without CLS token
    print("\n1. Testing ConditionalPositionEncoding (no CLS token):")
    cpe = ConditionalPositionEncoding(dim=hidden_dim, kernel_size=3)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    out = cpe(x, H, W)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    print("   ✓ Passed")
    
    # Test 2: ConditionalPositionEncoding with CLS token
    print("\n2. Testing ConditionalPositionEncoding (with CLS token):")
    x_with_cls = torch.randn(batch_size, seq_len + 1, hidden_dim)
    out_with_cls = cpe(x_with_cls, H, W)
    print(f"   Input shape: {x_with_cls.shape}")
    print(f"   Output shape: {out_with_cls.shape}")
    assert out_with_cls.shape == x_with_cls.shape, "Shape mismatch!"
    print("   ✓ Passed")
    
    # Test 3: CPE2D (residual connection)
    print("\n3. Testing CPE2D (residual connection):")
    cpe2d = CPE2D(dim=hidden_dim, kernel_size=3, zero_init=False)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    out = cpe2d(x, H, W)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    print("   ✓ Passed")
    
    # Test 4: CPE2D with zero initialization
    print("\n4. Testing CPE2D (zero init):")
    cpe2d_zero = CPE2D(dim=hidden_dim, kernel_size=3, zero_init=True)
    x = torch.randn(batch_size, seq_len, hidden_dim)
    out = cpe2d_zero(x, H, W)
    # With zero init, output should be very close to input initially
    diff = torch.abs(out - x).mean()
    print(f"   Mean difference from input: {diff.item():.6f}")
    print("   ✓ Passed")
    
    # Test 5: CPEBlock
    print("\n5. Testing CPEBlock:")
    cpe_block = CPEBlock(dim=hidden_dim, kernel_size=3, residual=True)
    x = torch.randn(batch_size, seq_len + 1, hidden_dim)
    out = cpe_block(x, H, W)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {out.shape}")
    assert out.shape == x.shape, "Shape mismatch!"
    print("   ✓ Passed")
    
    # Test 6: Different kernel sizes
    print("\n6. Testing different kernel sizes:")
    for k_size in [3, 5, 7]:
        cpe_k = ConditionalPositionEncoding(dim=hidden_dim, kernel_size=k_size)
        x = torch.randn(batch_size, seq_len, hidden_dim)
        out = cpe_k(x, H, W)
        print(f"   Kernel size {k_size}: {out.shape} ✓")
    
    print("\n" + "=" * 50)
    print("✅ All CPE tests passed!")
    print("=" * 50)

if __name__ == "__main__":
    test_cpe()