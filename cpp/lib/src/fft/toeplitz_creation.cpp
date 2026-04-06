module;

#include <finufft.h>
#include <cufinufft.h>
#include <cuda_runtime.h>

module hasty_fft_mod;

namespace hasty {
namespace fft {

// ── internal helpers ─────────────────────────────────────────────────────────

// Circular reversal along dimension d: indices [0, n-1, n-2, ..., 1]
static Tensor circ_reverse(const Tensor& t, i64 d)
{
    i64 n = t.size(d);
    // arange(n) → [0,1,..,n-1]; neg → [0,-1,..]; remainder(n) → [0,n-1,..,1]
    Tensor idx = arange(n, TensorOptions(t.device(), eScalarType::Long));
    return t.index_select(d, (-idx).remainder(n));
}

// Run a single Type-1 (NTU, adjoint) NUFFT on CUDA.
//
// omega      : [ndim, npts] float32 on CUDA — k-space trajectory
// weights    : [npts]       cfloat  on CUDA — density weights
// nmodes_fft : cufinufft convention — nmodes_fft[0] = #x-modes (fastest),
//              nmodes_fft[1] = #y-modes, nmodes_fft[2] = #z-modes
// shape_out  : image-order reshape for the flat output.
//              For im_size = {NZ,NY,NX}: nmodes_fft = {NX,NY,NZ}
//              and cufinufft stores x-fastest → reshape = {NZ,NY,NX} = im_size ✓
// double_prec: run in f64, cast result back to cfloat
// cmcl_mode  : use CMCL mode ordering (DC at index N/2) instead of FFT ordering (DC at 0)
//
// Returns cfloat tensor of shape `shape_out`
static Tensor ntu_nufft(
    const Tensor&               omega,
    const Tensor&               weights,
    ArrayRef<i64>               shape_out,
    bool                        double_prec,
    bool                        cmcl_mode = false)
{
    int ndim  = (int)shape_out.size();
    i64 total = 1;
    for (auto m : shape_out) total *= m;

    Device dev(eDeviceType::CUDA, (DeviceIndex)(int)omega.device().index);

    auto execute = [&]<typename T>() -> Tensor
    {
        constexpr eScalarType cplx_dtype =
            std::is_same_v<T, f32> ? eScalarType::ComplexFloat : eScalarType::ComplexDouble;
        constexpr eScalarType real_dtype =
            std::is_same_v<T, f32> ? eScalarType::Float : eScalarType::Double;

        Tensor coords_w = omega.to(real_dtype).contiguous();
        Tensor wts_w    = weights.to(cplx_dtype).unsqueeze(0).contiguous();
        Tensor output   = zeros(shape_out, TensorOptions(dev, cplx_dtype)).unsqueeze(0);

        NufftOptions<cuda_t, T, NTU> opts;
        opts.mode_order = cmcl_mode ? NufftOptions<cuda_t, T, NTU>::eModeOrder::CMCL
                                    : NufftOptions<cuda_t, T, NTU>::eModeOrder::FFT;

        if (ndim == 1) {
            NufftPlan<cuda_t, T, 1, NTU> plan({shape_out[0]}, opts);
            plan.setpts(coords_w);
            plan.execute(wts_w, output);
        } else if (ndim == 2) {
            NufftPlan<cuda_t, T, 2, NTU> plan({shape_out[0], shape_out[1]}, opts);
            plan.setpts(coords_w);
            plan.execute(wts_w, output);
        }
        else if (ndim == 3) {
            NufftPlan<cuda_t, T, 3, NTU> plan({shape_out[0], shape_out[1], shape_out[2]}, opts);
            plan.setpts(coords_w);
            plan.execute(wts_w, output);
        }
        else {
            throw std::runtime_error("Unsupported ndim in ntu_nufft: " + std::to_string(ndim));
        }

        return output.squeeze(0).contiguous();
    };

    return double_prec ? execute.template operator()<f64>()
                       : execute.template operator()<f32>();
}


// Spatial NUFFT: like ntu_nufft but returns H[0..N-1] in spatial order rather than
// FFT-mode-ordered Fourier coefficients.
//
// cufinufft type-1 with N modes in CMCL ordering gives output[m] = H[m - N/2].
// By pre-modulating the weights with exp(i * sum_d (N_d/2 * omega_d[j])), we shift
// the output so that output[m] = H[m], i.e. the spatial PSF at integer positions 0..N-1.
//
// This matches TorchKbNufft's adjoint output format without increasing the number of
// output modes (N internal grid stays ~2N, same as a standard N-mode NUFFT).
//
// omega    : [ndim, npts] — trajectory; omega[0]=kx (fastest), omega[1]=ky, …
// shape_out: im_size, e.g. {NY, NX}  (slowest first)
// nmodes_cmcl: same as nmodes_fft — reversed im_size, {NX, NY, …}
static Tensor ntu_nufft_spatial(
    const Tensor&               omega,
    const Tensor&               weights,
    ArrayRef<i64>               shape_out,
    bool                        double_prec)
{
    int ndim  = (int)shape_out.size();
    i64 npts  = omega.size(1);
    Device dev(eDeviceType::CUDA, (DeviceIndex)(int)omega.device().index);

    // Phase shift = sum_d (shape_out[ndim-1-d] / 2) * omega[d]
    // For 2D: NX/2 * kx + NY/2 * ky  (shape_out = {NY,NX}, omega[0]=kx, omega[1]=ky)
    eScalarType real_dtype = double_prec ? eScalarType::Double : eScalarType::Float;
    eScalarType cplx_dtype = double_prec ? eScalarType::ComplexDouble : eScalarType::ComplexFloat;
    Tensor omega_r = omega.to(real_dtype);
    Tensor phase = zeros({npts}, TensorOptions(dev, real_dtype));
    for (int d = 0; d < ndim; ++d) {
        double half_n = static_cast<double>(shape_out[ndim - 1 - d]) * 0.5;
        phase = phase.add(omega_r.select(0, d).mul(Scalar{half_n}));
    }

    // w_mod[j] = w[j] * exp(i * phase[j])
    Tensor exp_phase = view_as_complex(
        stack({phase.cos(), phase.sin()}, -1).contiguous()
    );
    Tensor w_mod = weights.to(cplx_dtype).mul(exp_phase);

    return ntu_nufft(omega, w_mod, shape_out, double_prec, /*cmcl_mode=*/true);
}


// Mirrors TorchKbNufft's adjoint_flip_and_concat.
//
// dim: omega-row index being processed (starts at 1, up to ndim-1).
//      In C++ (no batch dims) the corresponding image dim index == dim.
//
// Doubles the kernel along `dim` by concatenating:
//   [ kernel_normal | zero_slice | kernel_flipped.narrow.flip ]
//
// Uses ntu_nufft_spatial (CMCL + weight modulation) so that each NUFFT call
// returns H[0..N-1] in spatial order — matching TorchKbNufft's adjoint output.
static Tensor adjoint_flip_and_concat(
    int                             dim,
    const Tensor&                   omega,      // [ndim, npts] float
    const Tensor&                   weights,    // [npts] cfloat
    int                             ndim,
    ArrayRef<i64>                   shape_out,
    bool                            double_prec)
{
    // Build [ndim, 1] flip coefficient: -1 at the coordinate row corresponding to
    // image axis `dim`.  omega[i] maps to image axis ndim-1-i (kx→x, ky→y, kz→z),
    // so extending along image axis `dim` requires negating coordinate ndim-1-dim.
    auto make_flip = [&]() {
        Tensor fc = ones({ndim, 1}, TensorOptions(omega.device(), eScalarType::Float));
        fc.select(0, ndim - 1 - dim).fill_(Scalar{-1.0f});
        return fc;
    };

    Tensor kernel1 = (dim < ndim - 1)
        ? adjoint_flip_and_concat(dim + 1, omega, weights, ndim, shape_out, double_prec)
        : ntu_nufft_spatial(omega, weights, shape_out, double_prec);
    Tensor kernel2 = (dim < ndim - 1)
        ? adjoint_flip_and_concat(dim + 1, make_flip().mul(omega).contiguous(), weights, ndim, shape_out, double_prec)
        : ntu_nufft_spatial(make_flip().mul(omega).contiguous(), weights, shape_out, double_prec);

    // Zero block: same shape as kernel1 but size 1 along `dim`
    std::vector<i64> zero_shape = kernel1.sizes().vec();
    zero_shape[dim] = 1;
    Tensor zero_block = zeros(ArrayRef<i64>(zero_shape), TensorOptions(kernel1.device(), kernel1.dtype()));

    // Trim kernel2 (drop index 0 along dim), flip, then concat
    i64 len        = kernel2.size(dim) - 1;
    Tensor k2_trim = kernel2.narrow(dim, 1, len).flip({(i64)dim});

    return cat({kernel1, zero_block, k2_trim}, (i64)dim);
}


// Mirrors TorchKbNufft's reflect_conj_concat.
// Reflects and conjugates the kernel starting at `dim`, then concatenates.
static Tensor reflect_conj_concat(const Tensor& kernel, i64 dim)
{
    i64 ndim = kernel.ndimension();

    // tmp = conj(kernel) circularly reversed on every dim in [dim, ndim-1]
    Tensor tmp = kernel.conj().clone();
    for (i64 d = dim; d < ndim; ++d)
        tmp = circ_reverse(tmp, d);

    // Zero block: shape[dim] = 1
    std::vector<i64> zero_shape = kernel.sizes().vec();
    zero_shape[dim] = 1;
    Tensor zero_block = zeros(ArrayRef<i64>(zero_shape), TensorOptions(kernel.device(), kernel.dtype()));

    // Exclude DC repetition, then cat
    i64 len    = kernel.size(dim) - 1;
    Tensor rhs = cat({zero_block, tmp.narrow(dim, 1, len)}, dim);

    return cat({kernel, rhs}, dim);
}


// Mirrors TorchKbNufft's hermitify.
// Enforces Hermitian symmetry: average tensor with its coord-reversed conjugate.
static Tensor hermitify(const Tensor& kernel, i64 dim)
{
    i64 ndim = kernel.ndimension();
    Tensor k = kernel.clone();
    for (i64 d = dim; d < ndim; ++d)
        k = circ_reverse(k, d);
    return kernel.add(k.conj()).mul(Scalar{0.5});
}


// ── public API ────────────────────────────────────────────────────────────────

// Creates the FFT-space Toeplitz kernel approximating A'WA.
//
// coords    : [ndim, npts] float32 on CUDA — k-space trajectory in rad/voxel
//             coords[0] = x (fastest), coords[1] = y, coords[2] = z
// weights   : [npts] cfloat on CUDA — density compensation weights (ones = none)
// im_size   : image size, e.g. {NX} / {NY,NX} / {NZ,NY,NX}
// double_prec: run NUFFTs in f64 for accuracy, return cfloat kernel
//
// Returns a cfloat tensor of shape 2*im_size in FFT/frequency space.
// Pass the result to transform_toeplitz_kernel() before use with
// toeplitz_multiplication().
Tensor create_toeplitz_kernel(
    const Tensor&           coords,
    const Tensor&           weights,
    ArrayRef<i64>           im_size,
    bool                    double_prec)
{
    int ndim = (int)im_size.size();
    if (ndim < 1 || ndim > 3)
        throw std::runtime_error("create_toeplitz_kernel: ndim must be 1, 2, or 3");

    // Ensure weights are complex float
    Tensor wts = weights;
    if (!wts.is_complex())
        wts = view_as_complex(stack({wts, zeros_like(wts)}, -1).contiguous());

    Tensor kernel;
    if (ndim == 1) {
        // adjoint_flip_and_concat covers dims 1..ndim-1 (empty for 1-D);
        // reflect_conj_concat below handles dim 0 in all cases.
        kernel = ntu_nufft_spatial(coords, wts, im_size, double_prec);
    } else {
        kernel = adjoint_flip_and_concat(
            1, coords, wts, ndim, im_size, double_prec);
    }

    // Hermitian-symmetric extension along dim 0
    kernel = reflect_conj_concat(kernel, 0);

    // Enforce exact Hermitian symmetry
    kernel = hermitify(kernel, 0);

    // FFT over all spatial dimensions
    std::vector<i64> fft_dims(ndim);
    for (int i = 0; i < ndim; ++i) fft_dims[i] = i;
    kernel = fftn(kernel, nullopt, ArrayRef<i64>(fft_dims));

    kernel = kernel.contiguous();
    // Scale: 1 / (2 * NX * sqrt(prod(im_size)))  where NX = im_size[ndim-1] (fastest dim).
    // VkFFT performConvolution+performZeropadding normalises only the non-X inverse passes
    // (Y, Z, …), leaving a net gain of 2*NX that must be cancelled here.
    double scale = 0.5;
    for (auto s : im_size) scale /= static_cast<double>(s); //std::sqrt(static_cast<double>(s));
    scale /= static_cast<double>(im_size[ndim - 1]);
    kernel = kernel.mul(Scalar{scale});

    // Cast back to cfloat now that all double-precision work is done
    if (kernel.dtype() != eScalarType::ComplexFloat)
        kernel = kernel.to(eScalarType::ComplexFloat);

    return kernel;
}

Tensor create_toeplitz_kernel_standard(
    const Tensor&           coords,
    const Tensor&           weights,
    ArrayRef<i64>           im_size
)
{
    i64 ndim = (i64)im_size.size();

    // Expect coords tensor of shape [ndim, M]
    if (coords.ndimension() != 2 || coords.size(0) != ndim)
        throw std::runtime_error("create_toeplitz_kernel_standard: coords must be [ndim, M]");

    i64 M = coords.size(1);

    if (weights.ndimension() != 1)
        throw std::runtime_error("create_toeplitz_kernel_standard: weights must be 1-D");

    // Build nmodes (image-sized) and full_shape (2x image-sized) from im_size
    std::vector<i64> full_shape(ndim);
    for (std::size_t i = 0; i < ndim; ++i) {
        full_shape[i] = 2 * im_size[i];
    }

    if (weights.dtype() != eScalarType::ComplexFloat)
        throw std::runtime_error("create_toeplitz_kernel_standard: weights must be complex float");

    // NUFFT onto the 2N grid in FFT mode: DC is at index 0, so output is

    Tensor kernel = ntu_nufft(coords, weights, full_shape, /*double_prec=*/false);

    // Zero the Nyquist slice along each dim independently — h[-N,*] and h[*,-N]
    // are not part of the circulant embedding and must be set to zero.
    // Each dim is zeroed separately (full slab, not just the corner element).
    for (std::size_t i = 0; i < ndim; ++i)
        kernel.narrow((int)i, (i64)im_size[i], 1).zero_();

    // FFT the full kernel and scale by 1 / prod(2*im_size)
    std::vector<i64> fft_dims((int)ndim);
    for (int i = 0; i < (int)ndim; ++i) fft_dims[i] = i;
    kernel = fftn(kernel, nullopt, ArrayRef<i64>(fft_dims));

    kernel = kernel.contiguous();

    double scale = 0.5;
    for (auto s : im_size) scale /= static_cast<double>(s);
    scale /= static_cast<double>(im_size[ndim - 1]);
    kernel = kernel.mul(Scalar{static_cast<float>(scale)});

    //kernel.imag().zero_(); // enforce real-valued kernel (should be exact, but zero out any tiny residual imag part)

    if (kernel.dtype() != eScalarType::ComplexFloat)
        throw std::runtime_error("create_toeplitz_kernel_standard: unexpected kernel dtype after processing");

    return kernel;
}



}// namespace fft
} // namespace hasty
