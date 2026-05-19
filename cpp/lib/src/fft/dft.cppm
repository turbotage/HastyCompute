module;

export module hasty_fft_mod:dft;

import std;
import hasty_util_mod;
import hasty_tensor_mod;

namespace hasty {
namespace fft {

// Sign convention: F[j] = Σ_n f[n] · exp(sign·i · ξ[j]·x[n])
// Neg (−1) matches cuFINUFFT / FINUFFT type-2 default.
export enum struct eDFTSign : i32 { Neg = -1, Pos = +1 };

// Config for a C-order Cartesian grid.
// coords: [d, sizes[0], ..., sizes[d-1]] float
//   coords[dim, n0, ..., n_{d-1}] = pixel index along dim (0-based integer).
// ξ[j, dim] must use the same ordering: col 0 = frequency for dim 0.
export struct DFTConfig {
    std::vector<i64> sizes;      // [d] spatial grid sizes in C order
    i64              batch_size; // k-space points processed per pass
    eDFTSign         sign = eDFTSign::Neg;
    Tensor           coords;     // [d, sizes[0], sizes[1], ..., sizes[d-1]] float
};

export DFTConfig make_dft_config(
    const std::vector<i64>& sizes,
    i64      batch_size,
    Device   device,
    eDFTSign sign = eDFTSign::Neg)
{
    const i64 d = (i64)sizes.size();
    const TensorOptions opts_f{device, eScalarType::Float};

    // Build coords [d, sizes[0], ..., sizes[d-1]].
    // coords[dim, ...] = integer pixel index along that dim at every location.
    std::vector<Tensor> slices;
    slices.reserve(d);
    for (i64 dim = 0; dim < d; ++dim) {
        std::vector<i64> shape(d, 1);
        shape[dim] = sizes[dim];
        slices.push_back(
            arange(sizes[dim], opts_f)
                .sub(Scalar((f32)(sizes[dim] / 2)))
                .reshape(shape)
                .expand(std::vector<i64>(sizes.begin(), sizes.end()))
                .contiguous());
    }

    return DFTConfig{sizes, batch_size, sign, stack(slices, 0).contiguous()};
}

// F[j] = Σ_{n0,...,n_{d-1}} f[n0,...,n_{d-1}] · exp(sign·i · Σ_d ξ[j,d]·n_d)
//
// f  : [sizes[0], ..., sizes[d-1]] ComplexFloat — shape must exactly match cfg.sizes.
// xi : [M, d] float — non-uniform frequency points, same dim ordering as cfg.sizes.
// Returns [M] ComplexFloat.
export Tensor dft(const DFTConfig& cfg, const Tensor& f, const Tensor& xi)
{
    const i64 d = (i64)cfg.sizes.size();
    const i64 M = xi.size(0);

    if (f.ndimension() != d)
        throw std::invalid_argument(
            "dft: f has " + std::to_string(f.ndimension()) +
            " dims, expected " + std::to_string(d));
    for (i64 i = 0; i < d; ++i)
        if (f.size(i) != cfg.sizes[i])
            throw std::invalid_argument(
                "dft: f.size(" + std::to_string(i) + ")=" +
                std::to_string(f.size(i)) + " != cfg.sizes[" +
                std::to_string(i) + "]=" + std::to_string(cfg.sizes[i]));

    const Device dev = f.device();
    const TensorOptions opts_f{dev, eScalarType::Float};
    const TensorOptions opts_c{dev, eScalarType::ComplexFloat};
    const float s = (cfg.sign == eDFTSign::Neg) ? -1.0f : +1.0f;

    auto f_c = f.to(eScalarType::ComplexFloat);   // [n0,...,n_{d-1}]
    auto F   = zeros({M}, opts_c);

    for (i64 m0 = 0; m0 < M; m0 += cfg.batch_size) {
        const i64    mB   = std::min(cfg.batch_size, M - m0);
        const Tensor xi_b = xi.narrow(0, m0, mB);   // [mB, d]

        // Reshape xi for broadcasting with coords [d, n0,...,n_{d-1}]:
        //   xi_bc [mB, d, 1,...,1]  ×  coords.unsqueeze(0) [1, d, n0,...,n_{d-1}]
        //   → product [mB, d, n0,...,n_{d-1}]  → .sum(1)  →  phase [mB, n0,...,n_{d-1}]
        std::vector<i64> xi_shape = {mB, d};
        for (i64 i = 0; i < d; ++i) xi_shape.push_back(1);
        auto xi_bc = xi_b.reshape(xi_shape);                            // [mB, d, 1,...,1]

        auto phase = (xi_bc * cfg.coords.unsqueeze(0)).sum(1);          // [mB, n0,...,n_{d-1}]

        // kern = exp(sign·i·phase)  →  [mB, n0,...,n_{d-1}] complex
        auto kern = view_as_complex(
            stack({zeros_like(phase), phase.mul(Scalar(s))}, -1)
                .contiguous()).exp();

        // Multiply and sum all spatial dims to get [mB]
        Tensor result = kern.mul(f_c.unsqueeze(0));                     // [mB, n0,...,n_{d-1}]
        for (i64 dim = d; dim >= 1; --dim)
            result = result.sum(dim);                                   // [mB]

        F.narrow(0, m0, mB).copy_(result);
    }

    return F;
}

}
}
