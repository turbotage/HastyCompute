"""
Minimal reproducible example for PyTorch CUDA precision bug with large filters.

Bug Description:
- conv2d and conv_transpose2d produce incorrect results on CUDA with filters >= 10x10
- CPU results are correct
- 1D and 3D convolutions work correctly on CUDA
- Only 2D convolutions are affected

ROOT CAUSE IDENTIFIED:
- TensorFloat-32 (TF32) is enabled by default in cuDNN for Ampere+ GPUs
- TF32 reduces mantissa precision from 23 bits (FP32) to 10 bits
- This causes ~1e-2 precision errors with large convolution filters

SOLUTION:
- Set torch.backends.cudnn.conv.fp32_precision = 'ieee' (PyTorch 2.9+)
- OR torch.backends.cudnn.allow_tf32 = False (older API)
- OR use torch.float64 (double precision)

Environment where bug was confirmed:
- PyTorch 2.8.0 + CUDA 12.9
- PyTorch 2.9.0 + CUDA 13.0
- cuDNN 9.1.3
- NVIDIA RTX 4090 (Ampere architecture)

Expected behavior:
- CPU and CUDA should produce identical results (within floating-point tolerance ~1e-6)

Actual behavior:
- With TF32: CPU vs CUDA differences are ~1e-2 to 2e-2 for filters >= 10x10
- With IEEE FP32: Correct results (diff ~0)
"""

import torch
import torch.nn.functional as F


def test_conv2d_precision():
    """Test conv2d precision on CPU vs CUDA for various filter sizes."""
    print("="*80)
    print("PyTorch CUDA Bug Report: conv2d precision issue with large filters")
    print("="*80)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*80)
    
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available - cannot demonstrate bug")
        return
    
    torch.manual_seed(42)
    input_tensor = torch.randn(1, 1, 100, 100)
    
    print("\nTest 1: conv2d with different filter sizes")
    print("-" * 80)
    print(f"{'Filter Size':<15} {'Max CPU vs CUDA Diff':<25} {'Status'}")
    print("-" * 80)
    
    for size in [4, 6, 8, 10, 12, 14, 16, 18]:
        filter_tensor = torch.randn(1, 1, size, size)
        
        # CPU computation
        out_cpu = F.conv2d(input_tensor, filter_tensor, padding=size//2)
        
        # CUDA computation
        out_cuda = F.conv2d(input_tensor.cuda(), filter_tensor.cuda(), padding=size//2)
        
        # Compare
        diff = torch.max(torch.abs(out_cpu - out_cuda.cpu())).item()
        status = '✓ OK' if diff < 1e-5 else '✗ FAIL'
        print(f'{size}x{size:<11} {diff:<25.2e} {status}')
    
    print("\n" + "="*80)


def test_conv_transpose2d_precision():
    """Test conv_transpose2d precision on CPU vs CUDA."""
    print("\nTest 2: conv_transpose2d with 12x12 filters")
    print("-" * 80)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available")
        return
    
    torch.manual_seed(42)
    
    # Typical scenario: upsampling in wavelet reconstruction
    input_tensor = torch.randn(1, 4, 37, 37)
    filter_12x12 = torch.randn(4, 1, 12, 12)
    
    # CPU computation
    out_cpu = F.conv_transpose2d(input_tensor, filter_12x12, stride=2)
    
    # CUDA computation
    out_cuda = F.conv_transpose2d(input_tensor.cuda(), filter_12x12.cuda(), stride=2)
    
    # Compare
    diff = torch.max(torch.abs(out_cpu - out_cuda.cpu())).item()
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Filter shape: {filter_12x12.shape}")
    print(f"Output shape: {out_cpu.shape}")
    print(f"Max CPU vs CUDA diff: {diff:.2e}")
    
    if diff > 1e-5:
        print("✗ FAIL - Large precision error on CUDA")
    else:
        print("✓ OK - Results match within tolerance")
    
    print("="*80)


def test_1d_vs_2d_vs_3d():
    """Demonstrate that only 2D convolutions are affected."""
    print("\nTest 3: Comparing 1D, 2D, and 3D convolutions with 12-coefficient filters")
    print("-" * 80)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available")
        return
    
    torch.manual_seed(42)
    
    # Test 1D conv_transpose1d
    input_1d = torch.randn(1, 4, 37)
    filter_1d = torch.randn(4, 1, 12)
    out_1d_cpu = F.conv_transpose1d(input_1d, filter_1d, stride=2)
    out_1d_cuda = F.conv_transpose1d(input_1d.cuda(), filter_1d.cuda(), stride=2)
    diff_1d = torch.max(torch.abs(out_1d_cpu - out_1d_cuda.cpu())).item()
    
    # Test 2D conv_transpose2d
    input_2d = torch.randn(1, 4, 37, 37)
    filter_2d = torch.randn(4, 1, 12, 12)
    out_2d_cpu = F.conv_transpose2d(input_2d, filter_2d, stride=2)
    out_2d_cuda = F.conv_transpose2d(input_2d.cuda(), filter_2d.cuda(), stride=2)
    diff_2d = torch.max(torch.abs(out_2d_cpu - out_2d_cuda.cpu())).item()
    
    # Test 3D conv_transpose3d
    input_3d = torch.randn(1, 4, 16, 16, 16)
    filter_3d = torch.randn(4, 1, 12, 12, 12)
    out_3d_cpu = F.conv_transpose3d(input_3d, filter_3d, stride=2)
    out_3d_cuda = F.conv_transpose3d(input_3d.cuda(), filter_3d.cuda(), stride=2)
    diff_3d = torch.max(torch.abs(out_3d_cpu - out_3d_cuda.cpu())).item()
    
    print(f"{'Dimension':<15} {'Max CPU vs CUDA Diff':<25} {'Status'}")
    print("-" * 80)
    print(f"{'1D (12 coeffs)':<15} {diff_1d:<25.2e} {'✓ OK' if diff_1d < 1e-4 else '✗ FAIL'}")
    print(f"{'2D (12x12)':<15} {diff_2d:<25.2e} {'✓ OK' if diff_2d < 1e-4 else '✗ FAIL'}")
    print(f"{'3D (12x12x12)':<15} {diff_3d:<25.2e} {'✓ OK' if diff_3d < 1e-4 else '✗ FAIL'}")
    
    print("\n" + "="*80)
    print("EXPECTED RESULT: Only 2D should fail, 1D and 3D should pass")
    print("="*80)


def test_float32_vs_float64():
    """Test if double precision avoids the bug."""
    print("\nTest 4: float32 vs float64 precision")
    print("-" * 80)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available")
        return
    
    torch.manual_seed(42)
    input_tensor = torch.randn(1, 1, 100, 100)
    filter_12x12 = torch.randn(1, 1, 12, 12)
    
    print(f"{'Data Type':<15} {'Max CPU vs CUDA Diff':<25} {'Status'}")
    print("-" * 80)
    
    # Test float32
    x32 = input_tensor.to(torch.float32)
    f32 = filter_12x12.to(torch.float32)
    out_cpu_32 = F.conv2d(x32, f32, padding=6)
    out_cuda_32 = F.conv2d(x32.cuda(), f32.cuda(), padding=6)
    diff_32 = torch.max(torch.abs(out_cpu_32 - out_cuda_32.cpu())).item()
    
    # Test float64
    x64 = input_tensor.to(torch.float64)
    f64 = filter_12x12.to(torch.float64)
    out_cpu_64 = F.conv2d(x64, f64, padding=6)
    out_cuda_64 = F.conv2d(x64.cuda(), f64.cuda(), padding=6)
    diff_64 = torch.max(torch.abs(out_cpu_64 - out_cuda_64.cpu())).item()
    
    print(f"{'float32':<15} {diff_32:<25.2e} {'✗ FAIL' if diff_32 > 1e-5 else '✓ OK'}")
    print(f"{'float64':<15} {diff_64:<25.2e} {'✗ FAIL' if diff_64 > 1e-10 else '✓ OK'}")
    
    print("\n" + "="*80)
    print("WORKAROUND: Using float64 (double precision) avoids the bug!")
    print("="*80)


def test_cudnn_settings():
    """Test if cuDNN TF32 settings affect the bug - THIS IS THE ROOT CAUSE!"""
    print("\nTest 5: TensorFloat-32 (TF32) precision testing")
    print("-" * 80)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available")
        return
    
    torch.manual_seed(42)
    input_tensor = torch.randn(1, 1, 100, 100)
    filter_12x12 = torch.randn(1, 1, 12, 12)
    
    # CPU reference
    out_cpu = F.conv2d(input_tensor, filter_12x12, padding=6)
    
    # Save original settings
    original_allow_tf32 = torch.backends.cudnn.allow_tf32
    try:
        original_conv_precision = torch.backends.cudnn.conv.fp32_precision
    except:
        original_conv_precision = None
    
    print(f"{'Precision Mode':<30} {'Max CPU vs CUDA Diff':<20} {'Status'}")
    print("-" * 80)
    
    # Test 1: TF32 enabled (default on Ampere+ GPUs)
    torch.backends.cudnn.allow_tf32 = True
    out_cuda_tf32 = F.conv2d(input_tensor.cuda(), filter_12x12.cuda(), padding=6)
    diff_tf32 = torch.max(torch.abs(out_cpu - out_cuda_tf32.cpu())).item()
    print(f"{'TF32 (default)':<30} {diff_tf32:<20.2e} {'✗ FAIL' if diff_tf32 > 1e-5 else '✓ OK'}")
    
    # Test 2: Disable TF32 - old API
    torch.backends.cudnn.allow_tf32 = False
    out_cuda_fp32_old = F.conv2d(input_tensor.cuda(), filter_12x12.cuda(), padding=6)
    diff_fp32_old = torch.max(torch.abs(out_cpu - out_cuda_fp32_old.cpu())).item()
    print(f"{'IEEE FP32 (old API)':<30} {diff_fp32_old:<20.2e} {'✗ FAIL' if diff_fp32_old > 1e-5 else '✓ OK'}")
    
    # Test 3: New API (PyTorch 2.9+)
    try:
        torch.backends.cudnn.conv.fp32_precision = 'ieee'
        out_cuda_ieee = F.conv2d(input_tensor.cuda(), filter_12x12.cuda(), padding=6)
        diff_ieee = torch.max(torch.abs(out_cpu - out_cuda_ieee.cpu())).item()
        print(f"{'IEEE FP32 (new API)':<30} {diff_ieee:<20.2e} {'✗ FAIL' if diff_ieee > 1e-5 else '✓ OK'}")
    except:
        pass
    
    # Test conv_transpose2d
    print(f"\\nTesting conv_transpose2d:")
    print("-" * 80)
    torch.manual_seed(42)
    input_coeffs = torch.randn(1, 4, 37, 37)
    filter_trans = torch.randn(4, 1, 12, 12)
    out_cpu_trans = F.conv_transpose2d(input_coeffs, filter_trans, stride=2)
    
    torch.backends.cudnn.allow_tf32 = True
    out_trans_tf32 = F.conv_transpose2d(input_coeffs.cuda(), filter_trans.cuda(), stride=2)
    diff_trans_tf32 = torch.max(torch.abs(out_cpu_trans - out_trans_tf32.cpu())).item()
    print(f"{'TF32':<30} {diff_trans_tf32:<20.2e} {'✗ FAIL' if diff_trans_tf32 > 1e-5 else '✓ OK'}")
    
    torch.backends.cudnn.allow_tf32 = False
    out_trans_fp32 = F.conv_transpose2d(input_coeffs.cuda(), filter_trans.cuda(), stride=2)
    diff_trans_fp32 = torch.max(torch.abs(out_cpu_trans - out_trans_fp32.cpu())).item()
    print(f"{'IEEE FP32':<30} {diff_trans_fp32:<20.2e} {'✗ FAIL' if diff_trans_fp32 > 1e-5 else '✓ OK'}")
    
    # Restore settings
    torch.backends.cudnn.allow_tf32 = original_allow_tf32
    if original_conv_precision:
        try:
            torch.backends.cudnn.conv.fp32_precision = original_conv_precision
        except:
            pass
    
    print("\n" + "="*80)
    print("ROOT CAUSE IDENTIFIED: TensorFloat-32 (TF32)")
    print("="*80)
    print("TF32 is a reduced precision mode for Ampere+ GPUs:")
    print("  - Reduces mantissa from 23 bits (FP32) to 10 bits")
    print("  - Causes precision loss with large convolution filters")
    print(f"\\nError with TF32: {diff_tf32:.2e}")
    print(f"Error with IEEE FP32: {diff_fp32_old:.2e}")
    print(f"\\nSolution: Disable TF32 for convolutions")
    print("="*80)


def main():
    """Run all tests to demonstrate the bug."""
    test_conv2d_precision()
    test_conv_transpose2d_precision()
    test_1d_vs_2d_vs_3d()
    test_float32_vs_float64()
    test_cudnn_settings()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("ROOT CAUSE: TensorFloat-32 (TF32) in cuDNN")
    print("  - TF32 is enabled by default on Ampere+ GPUs (RTX 30xx/40xx, A100, etc.)")
    print("  - Reduces mantissa precision from 23 bits (FP32) to 10 bits")
    print("  - Causes ~1e-2 precision errors with large filters (>= 10x10)")
    print("")
    print("This bug affects:")
    print("  - F.conv2d with filters >= 10x10 in FLOAT32 with TF32 enabled")
    print("  - F.conv_transpose2d with filters >= 10x10 in FLOAT32 with TF32 enabled")
    print("  - Only 2D operations on Ampere+ GPUs (1D and 3D less affected)")
    print("")
    print("This bug does NOT affect:")
    print("  - CPU computations (always full FP32)")
    print("  - Filters <= 8x8")
    print("  - FLOAT64/double precision")
    print("  - When TF32 is disabled")
    print("")
    print("SOLUTIONS (in order of preference):")
    print("  1. Disable TF32 for convolutions (BEST - no memory overhead):")
    print("     torch.backends.cudnn.conv.fp32_precision = 'ieee'  # PyTorch 2.9+")
    print("     OR")
    print("     torch.backends.cudnn.allow_tf32 = False  # Older API")
    print("")
    print("  2. Use torch.float64 (double precision):")
    print("     - Perfect precision")
    print("     - 2x memory usage")
    print("     - Potentially slower")
    print("")
    print("  3. Use smaller wavelets with filters <= 8x8")
    print("")
    print("RECOMMENDATION:")
    print("  Add this at the start of your script for precise convolutions:")
    print("    torch.backends.cudnn.allow_tf32 = False")
    print("")
    print("Impact:")
    print("  - Wavelet transforms with long-filter wavelets (db6, db8, sym8, coif2, coif3)")
    print("  - Any application using 2D convolution with filters >= 10x10")
    print("  - Deep learning models with large kernels on Ampere+ GPUs")
    print("="*80)


if __name__ == "__main__":
    main()
