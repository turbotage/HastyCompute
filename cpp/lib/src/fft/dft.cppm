module;

export module hasty_fft_mod:dft;

import std;
import hasty_util_mod;
import hasty_tensor_mod;

namespace hasty {
namespace fft {

// Precomputed configuration for evaluating
//   F(ξ_j) = Σ_n f[n] × exp(-i × ξ_j · coords[n])
//
// Convention — same as cuFINUFFT type-2 + the CMCL half-pixel correction:
//   ξ_d = 2π k_d / N_d  ∈ [-π, π]   (k_d in cycles/FOV_d, N_d grid size)
//   coords[n, d] = n_d + 0.5 − N_d/2
//
// This is algebraically identical to forward_exact's kernel exp(−2πi k·r)
// with r[n,d] = (n_d + 0.5)/N_d − 0.5, because ξ·coords = 2πk·r.
//
// Usage:
//   auto cfg = make_dft_config({nx, ny, nz}, batch_size, device);
//   auto xi  = k_traj.clone();
//   xi.select(1,0).mul_(Scalar(2πf / nx));  // x
//   xi.select(1,1).mul_(Scalar(2πf / ny));  // y
//   xi.select(1,2).mul_(Scalar(2πf / nz));  // z
//   auto F = dft(cfg, image_flat, xi);      // [M] ComplexFloat

export struct DFTConfig {
    std::vector<i64> sizes;  // spatial grid sizes [d] — informational
    i64 batch_size;          // query-frequency outer batch size
    Tensor coords;           // [N, d] float — coords[n,d] = n_d + 0.5 − sizes[d]*0.5
};

// Build a DFTConfig for a full C-order Cartesian grid.
// sizes = [n0, n1, ..., nd-1]; dimension 0 is the slowest (outermost) axis.
export DFTConfig make_dft_config(
    const std::vector<i64>& sizes,
    i64 batch_size,
    Device device)
{
    const i64 d = (i64)sizes.size();
    i64 N = 1;
    for (auto s : sizes) N *= s;

    const TensorOptions opts_l{device, eScalarType::Long};

    std::vector<i64> strides(d);
    strides[d - 1] = 1;
    for (i64 i = d - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * sizes[i + 1];

    auto n_idx = arange(N, opts_l);
    std::vector<Tensor> cols;
    cols.reserve(d);
    for (i64 dim = 0; dim < d; ++dim) {
        auto nd = n_idx.div(Scalar(strides[dim]))
                       .remainder(Scalar(sizes[dim]))
                       .to(eScalarType::Float);
        // q_d = n_d + 0.5 - N_d/2
        cols.push_back(nd.add(Scalar(0.5f)).sub(Scalar((f32)sizes[dim] * 0.5f)));
    }

    return DFTConfig{sizes, batch_size, stack(cols, 1).contiguous()};
}

// F[j] = Σ_n f[n] × exp(−i × xi[j] · cfg.coords[n])
//
// xi : [M, d] float  — each component in [−π, π]
// f  : [N] ComplexFloat  — N must equal cfg.coords.size(0)
// Returns [M] ComplexFloat.
//
// Memory per inner iteration: O(batch_size × n_chunk) complex values (~8 MB default).
export Tensor dft(const DFTConfig& cfg, const Tensor& f, const Tensor& xi)
{
    const i64    M   = xi.size(0);
    const i64    N   = f.size(0);
    const Device dev = f.device();
    const TensorOptions opts_c{dev, eScalarType::ComplexFloat};
    const TensorOptions opts_f{dev, eScalarType::Float};
    const i64 n_chunk = 1 << 14;  // 16 384 voxels per inner chunk

    if (cfg.coords.size(0) != N)
        throw std::invalid_argument("dft: f.size(0) must match cfg.coords.size(0)");

    auto F = zeros({M}, opts_c);

    for (i64 m0 = 0; m0 < M; m0 += cfg.batch_size) {
        const i64 mB   = std::min(cfg.batch_size, M - m0);
        const Tensor xi_b = xi.narrow(0, m0, mB);   // [mB, d]
        auto F_b = zeros({mB}, opts_c);

        for (i64 n0 = 0; n0 < N; n0 += n_chunk) {
            const i64 nB = std::min(n_chunk, N - n0);
            const Tensor q_b = cfg.coords.narrow(0, n0, nB);  // [nB, d]
            const Tensor f_b = f.narrow(0, n0, nB);            // [nB] complex

            // phase[mB, nB] = xi_b @ q_b.T
            auto phase = mm(xi_b, q_b.transpose(0, 1));  // [mB, nB] float

            // kern = exp(−i × phase)
            auto kern = view_as_complex(
                stack({zeros({mB, nB}, opts_f), phase.neg()}, -1)
                    .contiguous()).exp();  // [mB, nB] complex

            F_b.add_(mv(kern, f_b));      // [mB]
        }

        F.narrow(0, m0, mB).copy_(F_b);
    }

    return F;
}

}
}
