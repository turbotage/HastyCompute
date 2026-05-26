module;

export module hasty_mri_mod:forward;

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_fft_mod;

namespace hasty {
namespace mri {

// Computes the exact MRI forward signal at M paired (k, t) samples:
//
//   S[c, m] = Σ_n  coil[c,n] · ρ[n] · exp(-z[n]·t[m])
//             · Π_q exp(i · b_q[n] · α_q[m])
//             · exp(-i · k[m] · n)
//
// kspace_trajectory [M, sdim] in same units as make_dft_config coords (i.e. [-π,π]).
// nonlin_gradient_waveforms [Q, M]: accumulated NL gradient encoding per sample.
// Returns [C, M].

export Tensor forward_exact(
    const Tensor& magnetization,         // [n0,...,n_{sdim-1}] Float
    const Tensor& sensitivity_maps,      // [C, n0,...,n_{sdim-1}] ComplexFloat
    const Tensor& rate_map,              // [n0,...,n_{sdim-1}] ComplexFloat
    const Tensor& timestamps,            // [M] Float
    const Tensor& kspace_trajectory,     // [M, sdim] Float — coords in [-π,π]
    const Tensor& nonlin_gradient_waveforms, // [Q, M] Float
    const Tensor& nonlin_gradient_basis, // [Q, n0,...,n_{sdim-1}] Float
    bool apply_ratemap = true,
    bool apply_nonlin  = true
) {
    using namespace std::numbers;

    const i64    sdim   = magnetization.ndimension();
    const i64    C      = sensitivity_maps.size(0);
    const i64    M      = timestamps.size(0);
    const i64    Q      = nonlin_gradient_waveforms.size(0);
    const Device device = magnetization.device();

    if (kspace_trajectory.size(0) != M)
        throw std::invalid_argument("kspace_trajectory and timestamps length M must match");

    auto chk_device = [&](const Tensor& t, std::string_view name) {
        if (t.device() != device)
            throw std::invalid_argument(std::string(name) + " must be on same device as magnetization");
    };
    chk_device(sensitivity_maps,          "sensitivity_maps");
    chk_device(rate_map,                  "rate_map");
    chk_device(timestamps,                "timestamps");
    chk_device(kspace_trajectory,         "kspace_trajectory");
    chk_device(nonlin_gradient_waveforms, "nonlin_gradient_waveforms");
    chk_device(nonlin_gradient_basis,     "nonlin_gradient_basis");

    auto chk_dtype = [&](const Tensor& t, eScalarType expected, std::string_view name) {
        if (t.scalar_type() != expected)
            throw std::invalid_argument(std::string(name) + " has wrong dtype");
    };
    chk_dtype(magnetization,             eScalarType::Float,        "magnetization");
    chk_dtype(sensitivity_maps,          eScalarType::ComplexFloat, "sensitivity_maps");
    chk_dtype(rate_map,                  eScalarType::ComplexFloat, "rate_map");
    chk_dtype(timestamps,                eScalarType::Float,        "timestamps");
    chk_dtype(kspace_trajectory,         eScalarType::Float,        "kspace_trajectory");
    chk_dtype(nonlin_gradient_waveforms, eScalarType::Float,        "nonlin_gradient_waveforms");
    chk_dtype(nonlin_gradient_basis,     eScalarType::Float,        "nonlin_gradient_basis");

    if (sdim < 1)
        throw std::invalid_argument("magnetization must have at least 1 spatial dimension");
    if (sensitivity_maps.ndimension() != sdim + 1)
        throw std::invalid_argument("sensitivity_maps must have ndim == sdim + 1");
    for (i64 i = 0; i < sdim; ++i)
        if (sensitivity_maps.size(i + 1) != magnetization.size(i))
            throw std::invalid_argument("sensitivity_maps spatial dims must match magnetization");
    if (rate_map.ndimension() != sdim)
        throw std::invalid_argument("rate_map must have same ndim as magnetization");
    for (i64 i = 0; i < sdim; ++i)
        if (rate_map.size(i) != magnetization.size(i))
            throw std::invalid_argument("rate_map shape must match magnetization");
    if (timestamps.ndimension() != 1)
        throw std::invalid_argument("timestamps must be 1-D [M]");
    if (kspace_trajectory.ndimension() != 2 || kspace_trajectory.size(1) != sdim)
        throw std::invalid_argument("kspace_trajectory must be [M, sdim]");
    if (nonlin_gradient_waveforms.ndimension() != 2 || nonlin_gradient_waveforms.size(1) != M)
        throw std::invalid_argument("nonlin_gradient_waveforms must be [Q, M]");
    if (nonlin_gradient_basis.ndimension() != sdim + 1)
        throw std::invalid_argument("nonlin_gradient_basis must have ndim == sdim + 1");
    if (nonlin_gradient_basis.size(0) != Q)
        throw std::invalid_argument("nonlin_gradient_basis Q dim must match nonlin_gradient_waveforms");
    for (i64 i = 0; i < sdim; ++i)
        if (nonlin_gradient_basis.size(i + 1) != magnetization.size(i))
            throw std::invalid_argument("nonlin_gradient_basis spatial dims must match magnetization");

    const TensorOptions opts_c = TensorOptions(device, eScalarType::ComplexFloat);
    const Scalar        pos_i  = Scalar(std::complex<f32>(0.0f, 1.0f));

    // DFT config: integer 0-based pixel coords, same C-order as magnetization.
    // kspace_trajectory[:,d] is the frequency for dim d (e.g. col 0 = kx for [nx,ny,nz]).
    std::vector<i64> sp_sizes(sdim);
    for (i64 i = 0; i < sdim; ++i) sp_sizes[i] = magnetization.size(i);

    const auto dft_cfg = fft::make_dft_config(sp_sizes, /*batch_size=*/M, device);

    // mag and rate_map stay as [n0,...,n_{sdim-1}] throughout — no flattening.
    const Tensor mag_c = magnetization.to(eScalarType::ComplexFloat);  // [n0,...,n_{sdim-1}]

    Tensor signal = zeros({C, M}, opts_c);

    for (i64 m = 0; m < M; ++m) {
        const f32    t_val = timestamps.select(0, m).item<f32>();
        const Tensor xi_m  = kspace_trajectory.narrow(0, m, 1);  // [1, sdim]

        // phi: per-voxel non-Fourier modulation for sample m.
        //   exp(-z[n]·t_m) · Π_q exp(i · b_q[n] · α_q[m])
        // Stays as [n0,...,n_{sdim-1}] ComplexFloat throughout.
        Tensor phi;
        if (apply_ratemap) {
            phi = rate_map.mul(Scalar(-t_val)).exp();          // [n0,...] ComplexFloat
        } else {
            phi = ones(std::vector<i64>(sp_sizes.begin(), sp_sizes.end()), opts_c);
        }
        if (apply_nonlin) {
            for (i64 q = 0; q < Q; ++q) {
                const f32 alpha = nonlin_gradient_waveforms
                                      .select(0, q).select(0, m).item<f32>();
                // b_q stays as [n0,...,n_{sdim-1}]
                const Tensor b_q = nonlin_gradient_basis.select(0, q)
                                       .to(eScalarType::ComplexFloat);
                phi = phi.mul((b_q.mul(Scalar(alpha)).mul(pos_i)).exp());
            }
        }

        // base = ρ · φ   [n0,...,n_{sdim-1}] ComplexFloat — coil-independent
        const Tensor base = mag_c.mul(phi);

        // S[c, m] = DFT{ coil[c] · base }(k[m])
        for (i64 c = 0; c < C; ++c) {
            // coil[c]: [n0,...,n_{sdim-1}] ComplexFloat — same shape as base
            const Tensor f_cm = sensitivity_maps.select(0, c).mul(base);
            signal.select(0, c).narrow(0, m, 1).copy_(
                fft::dft(dft_cfg, f_cm, xi_m)   // [1]
            );
        }
    }

    return signal;
}

}
}
