# TorchScript-Compatible Wavelet Transforms for C++/LibTorch

A pure PyTorch implementation of 1D, 2D, and 3D wavelet transforms that can be exported to C++ via TorchScript, eliminating the dependency on pywt in production environments.

## Features

- **Full TorchScript Compatibility**: All modules can be compiled with `torch.jit.script()` for C++ deployment
- **Embedded Filter Banks**: 5 orthogonal wavelet families built-in (no external dependencies)
- **Multi-dimensional Support**: 1D, 2D, and 3D transforms
- **Complex Number Support**: Handles both real (float32) and complex (complex64) inputs
- **GPU Acceleration**: Full CUDA support
- **Multiple Padding Modes**: zero, reflect, replicate, circular
- **Multi-level Decomposition**: Supports arbitrary decomposition levels

## Supported Wavelets

All wavelets have been validated against pywt with machine precision errors (< 1e-6):

| Wavelet | Filter Length | Description |
|---------|---------------|-------------|
| `haar` | 2 | Haar wavelet (simplest) |
| `db2` | 4 | Daubechies 2 |
| `db4` | 8 | Daubechies 4 |
| `sym2` | 4 | Symlet 2 |
| `coif1` | 6 | Coiflet 1 |

**Note**: Biorthogonal wavelets (e.g., bior2.2) are not supported because they have different filter lengths for decomposition vs reconstruction, requiring special handling not implemented in this convolution-based approach.

## Usage

### Python

```python
import torch
from wavelet_tests import WaveDec1D, WaveRec1D, WaveDec2D, WaveRec2D, WaveDec3D, WaveRec3D

# 1D example
data_1d = torch.randn(1, 64)  # (batch, length)
dec_1d = WaveDec1D(wavelet='db4', mode='zero', level=3)
rec_1d = WaveRec1D(wavelet='db4', mode='zero')

coeffs_1d = dec_1d(data_1d)
reconstructed_1d = rec_1d(coeffs_1d)

# 2D example
data_2d = torch.randn(1, 64, 64)  # (batch, height, width)
dec_2d = WaveDec2D(wavelet='sym2', mode='reflect', level=2)
rec_2d = WaveRec2D(wavelet='sym2', mode='reflect')

coeffs_2d = dec_2d(data_2d)
reconstructed_2d = rec_2d(coeffs_2d)

# 3D example
data_3d = torch.randn(1, 32, 32, 32)  # (batch, depth, height, width)
dec_3d = WaveDec3D(wavelet='coif1', mode='circular', level=2)
rec_3d = WaveRec3D(wavelet='coif1', mode='circular')

coeffs_3d = dec_3d(data_3d)
reconstructed_3d = rec_3d(coeffs_3d)

# Complex input support
complex_data = torch.randn(1, 64, dtype=torch.complex64)
dec_complex = WaveDec1D(wavelet='db2', mode='zero', level=2)
coeffs_complex = dec_complex(complex_data)  # Filters automatically cast to complex64
```

### Export to TorchScript

```python
import torch

# Create and trace modules
dec = WaveDec2D(wavelet='db4', mode='zero', level=3)
rec = WaveRec2D(wavelet='db4', mode='zero')

# Script the modules
scripted_dec = torch.jit.script(dec)
scripted_rec = torch.jit.script(rec)

# Save for C++ loading
scripted_dec.save('wavedec2d_db4_zero_level3.pt')
scripted_rec.save('waverec2d_db4_zero.pt')
```

### C++ Loading

```cpp
#include <torch/script.h>

// Load the scripted modules
torch::jit::script::Module dec = torch::jit::load("wavedec2d_db4_zero_level3.pt");
torch::jit::script::Module rec = torch::jit::load("waverec2d_db4_zero.pt");

// Use them
auto input = torch::randn({1, 256, 256});
auto coeffs = dec.forward({input}).toTensorList();
auto output = rec.forward({coeffs}).toTensor();
```

## Test Results

**Total tests: 960** (5 wavelets × 4 modes × 4 levels × 2 devices × 2 dtypes × 3 dims)

**Pass rate: 100.0%** (960/960 passed)

All tests validate reconstruction error < 1e-5 (typically achieving < 1e-6).

### Test Coverage

- **Wavelets**: haar, db2, db4, sym2, coif1
- **Padding modes**: zero, reflect, replicate, circular
- **Decomposition levels**: 1, 2, 3, 4
- **Devices**: CPU, CUDA
- **Data types**: float32, complex64
- **Dimensions**: 1D, 2D, 3D

### Typical Errors

- 1D transforms: 3-8 × 10⁻⁷
- 2D transforms: 9-20 × 10⁻⁷
- 3D transforms: 1-6 × 10⁻⁶

All errors are within machine precision for float32.

## Model Sizes

Each TorchScript module is extremely lightweight:

- **Decomposition modules**: ~12-18 KB
- **Reconstruction modules**: ~12-18 KB
- **Total for one configuration**: ~30 KB

For example, with 5 wavelets × 4 modes × 4 levels = 80 decomposition + 20 reconstruction configs:
- Total storage: ~3 MB (negligible)

## Implementation Details

### Decomposition

- Uses `conv1d`, `conv2d`, `conv3d` with stride=2 for downsampling
- Filters are flipped before convolution (per wavelet transform convention)
- Padding is applied before convolution to maintain proper boundary conditions
- Complex support via automatic filter casting to input dtype

### Reconstruction

- Uses `conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d` with stride=2 for upsampling
- Reconstruction filters are NOT flipped (used as-is)
- Handles size mismatches (0 or 1 sample differences) automatically
- Complex support via automatic filter casting to input dtype

### Filter Banks

All filter coefficients match `pywt.Wavelet()` exactly:
- Decomposition lowpass (dec_lo)
- Decomposition highpass (dec_hi)
- Reconstruction lowpass (rec_lo)
- Reconstruction highpass (rec_hi)

## Development & Testing

Run the comprehensive test suite:

```bash
python wavelet_tests.py
```

This will:
1. Run basic sanity checks (1D, 2D, 3D)
2. Save example TorchScript .pt files
3. Test all 960 configurations
4. Print detailed results and summary

## Known Limitations

1. **Biorthogonal wavelets not supported**: Different filter lengths for decomposition vs reconstruction require special handling
2. **Fixed padding logic**: Unlike pywt's signal extension modes, uses PyTorch's built-in padding
3. **Memory usage**: Each level creates new tensors (no in-place operations for TorchScript compatibility)
4. **Size adjustments**: May differ by 0-1 samples from pywt due to padding/reconstruction differences (automatically handled)

## References

- Based on the pywt library filter coefficients
- Wavelet theory: Mallat, S. "A Wavelet Tour of Signal Processing"
- TorchScript documentation: https://pytorch.org/docs/stable/jit.html
