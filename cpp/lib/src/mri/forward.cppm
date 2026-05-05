module;

export module mri_mod:forward;

import std;
import hasty_util_mod;
import hasty_tensor_mod;

namespace hasty {
namespace mri {

export Tensor forward_exact(
    const Tensor& magnetization,
    const Tensor& sensitivity_maps,
    const Tensor& rate_map,
    const Tensor& timestamps,
    const Tensor& kspace_trajectory,
    const Tensor& nonlin_gradient_waveforms,
    const Tensor& nonlin_gradient_basis,
    bool apply_ratemap  = true,
    bool apply_nonlin   = true,
    i64  k_batch_size   = 64,
    i64  t_batch_size   = 0
) {
    using namespace std::numbers;

    const i64    sdim   = magnetization.ndimension();
    const i64    C      = sensitivity_maps.size(0);
    const i64    N      = magnetization.numel();
    const i64    K      = kspace_trajectory.size(0);
    const i64    T      = timestamps.size(0);
    const i64    Q      = nonlin_gradient_waveforms.size(0);
    const Device device = magnetization.device();

    if (t_batch_size <= 0) t_batch_size = T;

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
        throw std::invalid_argument("sensitivity_maps must have ndim == magnetization.ndim + 1");
    for (i64 i = 0; i < sdim; ++i)
        if (sensitivity_maps.size(i + 1) != magnetization.size(i))
            throw std::invalid_argument("sensitivity_maps spatial dims must match magnetization");
    if (rate_map.ndimension() != sdim)
        throw std::invalid_argument("rate_map must have same ndim as magnetization");
    for (i64 i = 0; i < sdim; ++i)
        if (rate_map.size(i) != magnetization.size(i))
            throw std::invalid_argument("rate_map shape must match magnetization");
    if (timestamps.ndimension() != 1)
        throw std::invalid_argument("timestamps must be 1-D");
    if (kspace_trajectory.ndimension() != 2 || kspace_trajectory.size(1) != sdim)
        throw std::invalid_argument("kspace_trajectory must be [K, sdim]");
    if (nonlin_gradient_waveforms.ndimension() != 2 || nonlin_gradient_waveforms.size(1) != T)
        throw std::invalid_argument("nonlin_gradient_waveforms must be [Q, T]");
    if (nonlin_gradient_basis.ndimension() != sdim + 1)
        throw std::invalid_argument("nonlin_gradient_basis must have ndim == magnetization.ndim + 1");
    if (nonlin_gradient_basis.size(0) != Q)
        throw std::invalid_argument("nonlin_gradient_basis Q dim must match nonlin_gradient_waveforms");
    for (i64 i = 0; i < sdim; ++i)
        if (nonlin_gradient_basis.size(i + 1) != magnetization.size(i))
            throw std::invalid_argument("nonlin_gradient_basis spatial dims must match magnetization");

    const TensorOptions opts_f = TensorOptions(device, eScalarType::Float);
    const TensorOptions opts_l = TensorOptions(device, eScalarType::Long);
    const TensorOptions opts_c = TensorOptions(device, eScalarType::ComplexFloat);

    const Scalar neg_2pi_i = Scalar(std::complex<f32>(0.0f, -2.0f * (f32)pi_v<f64>));
    const Scalar pos_i     = Scalar(std::complex<f32>(0.0f,  1.0f));

    std::vector<i64> sp_sizes(sdim);
    for (i64 i = 0; i < sdim; ++i) sp_sizes[i] = magnetization.size(i);

    std::vector<i64> sp_strides(sdim);
    sp_strides[sdim - 1] = 1;
    for (i64 i = sdim - 2; i >= 0; --i)
        sp_strides[i] = sp_strides[i + 1] * sp_sizes[i + 1];

    const Tensor n_idx = arange(N, opts_l);

    std::vector<Tensor> r_cols;
    r_cols.reserve(sdim);
    for (i64 i = 0; i < sdim; ++i) {
        const i64 n = sp_sizes[i];
        Tensor g     = arange(n, opts_f).add(Scalar(0.5f)).div(Scalar((f32)n)).sub(Scalar(0.5f));
        Tensor idx_i = n_idx.div(Scalar(sp_strides[i])).remainder(Scalar(n)).to(eScalarType::Long);
        r_cols.push_back(g.index_select(0, idx_i));
    }
    const Tensor r = stack(r_cols, 1).contiguous();

    const Tensor mag_flat   = magnetization.flatten().to(eScalarType::ComplexFloat);
    const Tensor rate_flat  = rate_map.flatten();
    const Tensor coil_flat  = sensitivity_maps.reshape({C, N});
    const Tensor spatial_wt = coil_flat.mul(mag_flat.unsqueeze(0));
    const Tensor nl_fields  = nonlin_gradient_basis.reshape({Q, N});
    const Tensor nl_alpha   = nonlin_gradient_waveforms;

    Tensor signal = zeros({C, K, T}, opts_c);

    for (i64 t0 = 0; t0 < T; t0 += t_batch_size) {
        const i64    tB  = std::min(t_batch_size, T - t0);
        const Tensor t_b = timestamps.narrow(0, t0, tB);

        Tensor time_phase;
        if (apply_ratemap) {
            // exp(-z(r) * t)  where z is the complex ratemap (T2 decay + B0 dephasing)
            time_phase = rate_flat.unsqueeze(1)
                                  .mul(t_b.to(eScalarType::ComplexFloat).unsqueeze(0))
                                  .mul(Scalar(-1.0f))
                                  .exp();
        } else {
            time_phase = ones({N, tB}, opts_c);
        }

        if (apply_nonlin) {
            for (i64 q = 0; q < Q; ++q) {
                time_phase = time_phase.mul(
                    nl_fields.select(0, q).to(eScalarType::ComplexFloat)
                              .unsqueeze(1)
                              .mul(nl_alpha.select(0, q).narrow(0, t0, tB)
                                          .to(eScalarType::ComplexFloat).unsqueeze(0))
                              .mul(pos_i)
                              .exp()
                );
            }
        }

        for (i64 k0 = 0; k0 < K; k0 += k_batch_size) {
            const i64    kB        = std::min(k_batch_size, K - k0);
            const Tensor k_b       = kspace_trajectory.narrow(0, k0, kB).to(eScalarType::ComplexFloat);
            const Tensor dft_phase = mm(k_b, r.to(eScalarType::ComplexFloat).transpose(0, 1))
                                       .mul(neg_2pi_i)
                                       .exp();

            for (i64 c = 0; c < C; ++c) {
                signal.select(0, c).narrow(0, k0, kB).narrow(1, t0, tB).copy_(
                    mm(dft_phase.mul(spatial_wt.select(0, c).unsqueeze(0)), time_phase)
                );
            }
        }
    }

    return signal;
}

}
}
