module;

export module hasty_mri_mod:off_fourier_interpolators;

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_linalg_mod;

namespace hasty {
namespace mri {

export struct PhiLowrankResult {
    Tensor Omega;    // [K, L]      — U * sqrt(S)
    Tensor S;        // [L]         — singular values (Float)
    Tensor Upsilon;  // [n_hist, L] — V * sqrt(S)
};

// Weighting strategy for the phi low-rank SVD.
// Weights column h of the phi operator, biasing the decomposition towards bins
// that carry more signal energy.
//
//   None      — unweighted Frobenius norm, every bin equally important
//   L1Mass    — w_h = Σ_{n∈h} ρ[n]  (total PD mass, current default)
//   L2Energy  — w_h = Σ_{n∈h} ρ[n]² (Parseval-optimal: minimises signal
//               L2 energy error when cross-bin correlations are negligible)
export enum struct eBinWeighting : i32 {
    None               = 0,
    L1Mass             = 1,
    L2Energy           = 2,
    FreqAwareLowrank   = 3,
};

export struct HistogramResult {
    Tensor mask_idx;        // [N_mask]   long  — flat voxel indices in the image
    Tensor voxel_to_bin;    // [N_mask]   long  — bin index for each masked voxel
    i64    n_hist;
    Tensor z_map_hist;      // [n_hist]   complex — weighted-mean z per bin
    Tensor nl_fields_hist;  // [Q, n_hist] float  — weighted-mean fields per bin
    Tensor bin_weights;     // [n_hist]   float  — Σ ρ   per bin  (L1 mass)
    Tensor bin_l2_energy;   // [n_hist]   float  — Σ ρ²  per bin  (L2 energy)
};


// ---------------------------------------------------------------------------
// make_phi_operator
// ---------------------------------------------------------------------------

export linalg::LinearOperator make_phi_operator(
    const Tensor& z_map,
    const Tensor& nl_fields,
    const Tensor& nl_alpha,
    const Tensor& timestamps,
    i64 chunk_size = 512
) {
    const i64    K      = timestamps.size(0);
    const i64    N      = z_map.size(0);
    const i64    Q      = nl_fields.size(0);
    const Device device = z_map.device();

    auto chk_device = [&](const Tensor& t, std::string_view name) {
        if (t.device() != device)
            throw std::invalid_argument(std::string(name) + " device mismatch");
    };
    auto chk_dtype = [&](const Tensor& t, eScalarType expected, std::string_view name) {
        if (t.scalar_type() != expected)
            throw std::invalid_argument(std::string(name) + " wrong dtype");
    };

    chk_dtype(z_map,      eScalarType::ComplexFloat, "z_map");
    chk_dtype(nl_fields,  eScalarType::Float,        "nl_fields");
    chk_dtype(nl_alpha,   eScalarType::Float,        "nl_alpha");
    chk_dtype(timestamps, eScalarType::Float,        "timestamps");

    if (z_map.ndimension() != 1)
        throw std::invalid_argument("z_map must be 1-D [N]");
    if (nl_fields.ndimension() != 2 || nl_fields.size(1) != N)
        throw std::invalid_argument("nl_fields must be [Q, N]");
    if (nl_alpha.ndimension() != 2 || nl_alpha.size(0) != Q || nl_alpha.size(1) != K)
        throw std::invalid_argument("nl_alpha must be [Q, K]");
    if (timestamps.ndimension() != 1)
        throw std::invalid_argument("timestamps must be 1-D");

    chk_device(nl_fields,  "nl_fields");
    chk_device(nl_alpha,   "nl_alpha");
    chk_device(timestamps, "timestamps");

    // Build phi in double precision so the SVD MatVec has full f64 accuracy.
    // PETSc Lanczos already uses double internally; without this, every
    // matvec application round-trips float32→double→float32, accumulating
    // ~1e-7 relative error per application over K=400k terms.
    const Scalar pos_i_d = Scalar(std::complex<f64>(0.0, 1.0));

    auto build_phi = [=](i64 k0, i64 kB) -> Tensor {
        auto t_c  = timestamps.narrow(0, k0, kB).to(eScalarType::ComplexDouble);
        auto zm_d = z_map.to(eScalarType::ComplexDouble);
        auto phi  = (zm_d.unsqueeze(0).mul(-t_c.unsqueeze(1))).exp();
        for (i64 q = 0; q < Q; ++q) {
            phi = phi.mul(
                nl_fields.select(0, q).to(eScalarType::ComplexDouble).unsqueeze(0)
                          .mul(nl_alpha.select(0, q).narrow(0, k0, kB)
                                       .to(eScalarType::ComplexDouble).unsqueeze(1))
                          .mul(pos_i_d).exp()
            );
        }
        return phi;  // [kB, N] ComplexDouble
    };

    auto mv_fn = [=](const Tensor& x) -> Tensor {
        auto result = zeros({K}, TensorOptions(device, eScalarType::ComplexDouble));
        for (i64 k0 = 0; k0 < K; k0 += chunk_size) {
            const i64 kB = std::min(chunk_size, K - k0);
            result.narrow(0, k0, kB).copy_(mv(build_phi(k0, kB), x));
        }
        return result;
    };

    auto rmv_fn = [=](const Tensor& y) -> Tensor {
        auto result = zeros({N}, TensorOptions(device, eScalarType::ComplexDouble));
        for (i64 k0 = 0; k0 < K; k0 += chunk_size) {
            const i64 kB = std::min(chunk_size, K - k0);
            result += mv(build_phi(k0, kB).conj().transpose(0, 1), y.narrow(0, k0, kB));
        }
        return result;
    };

    return linalg::LinearOperator(K, N,
        std::move(mv_fn), std::move(rmv_fn),
        eScalarType::ComplexDouble, device);
}


// ---------------------------------------------------------------------------
// phi_lowrank_weights
// Computes the weight vector [n_hist] for the phi_lowrank weighted SVD.
// Returns nullopt for eBinWeighting::None (unweighted).
// ---------------------------------------------------------------------------

export Opt<Tensor> phi_lowrank_weights(
    eBinWeighting weighting,
    const HistogramResult& hist,
    const Tensor& timestamps    // [K] float — needed for FreqAwareLowrank
) {
    using namespace std::numbers;
    switch (weighting) {
        case eBinWeighting::None:
            return nullopt;
        case eBinWeighting::L1Mass:
            return hist.bin_weights;
        case eBinWeighting::L2Energy:
            return hist.bin_l2_energy;
        case eBinWeighting::FreqAwareLowrank: {
            float T_readout = (timestamps.max() - timestamps.min()).item<f32>();
            float pi_f = (float)pi_v<f64>;
            auto phase_h    = hist.z_map_hist.imag().abs()
                                  .mul(Scalar(T_readout)).div(Scalar(pi_f));
            auto freq_factor = clamp(phase_h, Scalar(1.0f), Scalar(1e9f));
            return hist.bin_l2_energy.mul(freq_factor);
        }
    }
    return nullopt;
}


// ---------------------------------------------------------------------------
// phi_lowrank
// Weighted SVD of the phi operator. Pass weights from phi_lowrank_weights.
// Weights scale columns of the operator so high-energy bins dominate the SVD.
// ---------------------------------------------------------------------------
export PhiLowrankResult phi_lowrank(
    const linalg::LinearOperator& op,
    i64 L,
    const Opt<Tensor>& weights = nullopt,
    i64 ncv = -1,
    i64 mpd = -1
) {
    const i64    K      = op.m();
    const i64    N      = op.n();
    const Device device = op.device();

    linalg::SVDResult svd;
    Tensor w_sqrt;

    if (weights.has_value()) {
        w_sqrt = pow(*weights, 0.5).to(op.dtype());

        linalg::LinearOperator op_w(K, N,
            [op, w_sqrt](const Tensor& v) { return op.matvec(v.mul(w_sqrt)); },
            [op, w_sqrt](const Tensor& y) { return op.rmatvec(y).mul(w_sqrt); },
            op.dtype(), device
        );
        svd = linalg::operator_svd(op_w, L, ncv, mpd);
    } else {
        svd = linalg::operator_svd(op, L, ncv, mpd);
    }

    Tensor& Omega   = svd.U;
    Tensor& Upsilon = svd.Vh.transpose_(0, 1);

    if (weights.has_value()) {
        auto w_safe = where(w_sqrt.abs().gt(Scalar(1e-15f)),
                            w_sqrt,
                            ones({N}, TensorOptions(device, op.dtype())));
        Upsilon = Upsilon.div(w_safe.unsqueeze(1));
    }

    // SVD computed in double; cast back to float32 for reconstruction use.
    return PhiLowrankResult{
        Omega.to(eScalarType::ComplexFloat),
        svd.S.to(eScalarType::Float),
        Upsilon.to(eScalarType::ComplexFloat)
    };
}


// ---------------------------------------------------------------------------
// extract_histogram
//
// Bins masked voxels by their (z_real, z_imag, f_0, ..., f_{Q-1}) features.
// n_rate bins for each component of z_map, n_nl bins per nonlinear field.
// Weighted means use mag as weight. bin_weights = Σ_{j in bin} mag_j.
// ---------------------------------------------------------------------------

export HistogramResult extract_histogram(
    const Tensor& mag_flat,
    const Tensor& z_map_flat,
    const Tensor& nl_fields,
    i64 n_rate,
    i64 n_nl
) {
    const i64    N      = mag_flat.size(0);
    const i64    Q      = nl_fields.size(0);
    const Device device = mag_flat.device();

    const TensorOptions opts_f = TensorOptions(device, eScalarType::Float);
    const TensorOptions opts_l = TensorOptions(device, eScalarType::Long);

    if (mag_flat.scalar_type() != eScalarType::Float)
        throw std::invalid_argument("mag_flat must be Float");
    if (z_map_flat.scalar_type() != eScalarType::ComplexFloat)
        throw std::invalid_argument("z_map_flat must be ComplexFloat");
    if (nl_fields.scalar_type() != eScalarType::Float)
        throw std::invalid_argument("nl_fields must be Float");
    if (nl_fields.ndimension() != 2 || nl_fields.size(1) != N)
        throw std::invalid_argument("nl_fields must be [Q, N]");

    auto mask     = mag_flat.gt(Scalar(0.0f));
    auto mask_idx = arange(N, opts_l).masked_select(mask);
    const i64 N_mask = mask_idx.size(0);

    auto mag_m = mag_flat.masked_select(mask);

    auto select_masked = [&](const Tensor& x) {
        return x.masked_select(mask);
    };

    auto bin_feat = [&](const Tensor& x, i64 n) -> Tensor {
        auto lo  = x.min();
        auto hi  = x.max();
        auto idx = x.sub(lo).div(hi.sub(lo).add(Scalar(1e-12f)))
                    .mul(Scalar((f32)(n - 1)))
                    .to(eScalarType::Long);
        return clamp(idx, Scalar((i64)0), Scalar((i64)(n - 1)));
    };

    // Strides: z_real | z_imag | nl_0 | ... | nl_{Q-1}  (row-major)
    i64 nl_total = 1;
    for (i64 q = 0; q < Q; ++q) nl_total *= n_nl;

    auto z_real_m = select_masked(z_map_flat.real());
    auto z_imag_m = select_masked(z_map_flat.imag());

    auto bin_flat = bin_feat(z_real_m, n_rate).mul(Scalar(n_rate * nl_total))
                     .add(bin_feat(z_imag_m, n_rate).mul(Scalar(nl_total)));

    i64 nl_stride = nl_total;
    for (i64 q = 0; q < Q; ++q) {
        nl_stride /= n_nl;
        auto nl_m  = select_masked(nl_fields.select(0, q));
        bin_flat   = bin_flat.add(bin_feat(nl_m, n_nl).mul(Scalar(nl_stride)));
    }

    auto [unique_bins, voxel_to_bin] = unique_with_inverse(bin_flat);
    const i64 n_hist = unique_bins.size(0);

    auto scatter_wmean_f = [&](const Tensor& vals) -> Tensor {
        auto wv  = zeros({n_hist}, opts_f);
        auto wc  = zeros({n_hist}, opts_f);
        wv.scatter_add_(0, voxel_to_bin, vals.mul(mag_m));
        wc.scatter_add_(0, voxel_to_bin, mag_m);
        return wv.div(clamp(wc, Scalar(1e-30f), Scalar(1e30f)));
    };

    auto z_real_hist = scatter_wmean_f(z_real_m);
    auto z_imag_hist = scatter_wmean_f(z_imag_m);
    auto z_map_hist  = view_as_complex(stack({z_real_hist, z_imag_hist}, 1).contiguous());

    std::vector<Tensor> nl_cols;
    nl_cols.reserve(Q);
    for (i64 q = 0; q < Q; ++q)
        nl_cols.push_back(scatter_wmean_f(select_masked(nl_fields.select(0, q))).unsqueeze(0));

    auto nl_fields_hist = Q > 0 ? cat(nl_cols, 0)
                                : empty({0, n_hist}, opts_f);

    auto bin_weights = zeros({n_hist}, opts_f);
    bin_weights.scatter_add_(0, voxel_to_bin, mag_m);

    auto bin_l2_energy = zeros({n_hist}, opts_f);
    bin_l2_energy.scatter_add_(0, voxel_to_bin, mag_m.mul(mag_m));

    return HistogramResult{
        mask_idx, voxel_to_bin, n_hist,
        z_map_hist, nl_fields_hist, bin_weights, bin_l2_energy
    };
}


// ---------------------------------------------------------------------------
// time_segmented_phi
//
// Classic time-segmentation approximation (off-resonance only, ignores NL).
// Fixes L spatial maps Υ[h,l] = exp(-z_h·τ_l) at uniformly-spaced segment
// times τ_l ∈ [t_min, t_max], then finds the weighted-LS-optimal temporal
// coefficients Ω[k,l] that minimise
//   Σ_h w_h |Σ_l Ω[k,l]·Υ[h,l] - exp(-z_h·t_k)|²
// for each k independently.
//
// Normal equations:  G·Ω[k,:]ᵀ = R[k,:]ᵀ
//   G[l,m]  = Σ_h w_h·conj(Υ[h,l])·Υ[h,m]         [L×L, precomputed once]
//   R[k,l]  = Σ_h w_h·conj(Υ[h,l])·exp(-z_h·t_k)  [K×L, chunked matmul]
//
// Returns PhiLowrankResult — same struct as SVD → same approx_signal call.
// Computed in f64 for numerical stability, cast to cf32 on return.
// ---------------------------------------------------------------------------

export PhiLowrankResult time_segmented_phi(
    const HistogramResult& hist,
    const Tensor& timestamps,                   // [K] float
    i64 L,
    eBinWeighting weighting = eBinWeighting::L2Energy,
    i64 chunk_size          = 512
) {
    const i64    K      = timestamps.size(0);
    const i64    n_hist = hist.n_hist;
    const Device device = timestamps.device();

    using namespace std::numbers;
    const float pi_f = (float)pi_v<f64>;

    const TensorOptions opts_cd = TensorOptions(device, eScalarType::ComplexDouble);
    const TensorOptions opts_d  = TensorOptions(device, eScalarType::Double);

    // ── Weights ──────────────────────────────────────────────────────────────
    Tensor w;
    switch (weighting) {
        case eBinWeighting::None:
            w = ones({n_hist}, TensorOptions(device, eScalarType::Float)); break;
        case eBinWeighting::L1Mass:
            w = hist.bin_weights; break;
        case eBinWeighting::L2Energy:
            w = hist.bin_l2_energy; break;
        case eBinWeighting::FreqAwareLowrank: {
            float T_readout = (timestamps.max() - timestamps.min()).item<f32>();
            auto phase_h    = hist.z_map_hist.imag().abs()
                                  .mul(Scalar(T_readout)).div(Scalar(pi_f));
            auto freq_factor = clamp(phase_h, Scalar(1.0f), Scalar(1e9f));
            w = hist.bin_l2_energy.mul(freq_factor); break;
        }
    }
    auto w_d = w.to(eScalarType::Double).to(eScalarType::ComplexDouble);  // [n_hist] cd

    // ── Segment times τ_l ∈ [t_min, t_max] ──────────────────────────────────
    const f64 t_min = (f64)timestamps.min().item<f32>();
    const f64 t_max = (f64)timestamps.max().item<f32>();
    const f64 denom = (L > 1) ? (f64)(L - 1) : 1.0;
    auto tau = arange(L, opts_d)
                   .div(Scalar(denom))
                   .mul(Scalar(t_max - t_min))
                   .add(Scalar(t_min));                          // [L] double

    // ── Spatial maps: Υ[n_hist, L] = exp(-z_h·τ_l) ──────────────────────────
    auto z_d      = hist.z_map_hist.to(eScalarType::ComplexDouble);  // [n_hist]
    auto tau_cd   = tau.to(eScalarType::ComplexDouble);               // [L]
    auto Upsilon  = (-z_d.unsqueeze(1).mul(tau_cd.unsqueeze(0))).exp(); // [n_hist, L]

    // ── A[n_hist, L] = w_h · conj(Υ[h,l])  (used in Gram + RHS) ────────────
    auto A = Upsilon.conj().mul(w_d.unsqueeze(1));                    // [n_hist, L]

    // ── Gram matrix G[L, L] = Aᵀ · Υ ────────────────────────────────────────
    auto G = mm(A.transpose(0, 1), Upsilon);                          // [L, L]

    // ── RHS R[K, L] = phi[k,h] · A[h,l]  (chunked) ──────────────────────────
    auto t_cd = timestamps.to(eScalarType::ComplexDouble);            // [K]
    auto R    = zeros({K, L}, opts_cd);

    for (i64 k0 = 0; k0 < K; k0 += chunk_size) {
        const i64 kB = std::min(chunk_size, K - k0);
        // phi_chunk[kB, n_hist] = exp(-z_h · t_k)
        auto phi_chunk = (-z_d.unsqueeze(0)
                          .mul(t_cd.narrow(0, k0, kB).unsqueeze(1))).exp();  // [kB, n_hist]
        R.narrow(0, k0, kB).copy_(mm(phi_chunk, A));                          // [kB, L]
    }

    // ── Ω: solve G @ Ωᵀ = Rᵀ  →  Ω = (G⁻¹ Rᵀ)ᵀ ────────────────────────────
    // G is [L×L], linalg_solve is O(L³) — negligible for L ≤ 16.
    auto Omega = linalg_solve(G, R.transpose(0, 1))
                     .transpose(0, 1).contiguous();                   // [K, L]

    return PhiLowrankResult{
        Omega.to(eScalarType::ComplexFloat),
        Upsilon.to(eScalarType::ComplexFloat)
    };
}


// ---------------------------------------------------------------------------
// (Factored histogram / factored phi removed — use time_segmented_phi or
//  the joint phi_lowrank SVD for off-resonance-only comparison.)
// ---------------------------------------------------------------------------



// ---------------------------------------------------------------------------
// approx_signal
//
// S[c, k] ≈ Σ_l  Omega[k,l] · ( Σ_b Upsilon[b,l] · agg[c,b,k] )
// where agg[b,k] = Σ_{j: bin[j]=b}  mag[j]·coil[c,j]·exp(-2πi·k·r[j])
//
// coords_masked in normalized units [-0.5, 0.5], k_traj in cycles/FOV.
// ---------------------------------------------------------------------------

export Tensor approx_signal(
    const Tensor& Omega,
    const Tensor& Upsilon,
    const Tensor& mag_masked,
    const Tensor& coilmaps_masked,
    const Tensor& coords_masked,
    const Tensor& k_traj,
    const Tensor& voxel_to_bin,
    i64 n_hist,
    i64 k_batch_size = 64
) {
    using namespace std::numbers;

    const i64    K      = Omega.size(0);
    const i64    C      = coilmaps_masked.size(0);
    const i64    N_mask = mag_masked.size(0);
    const Device device = Omega.device();

    const Scalar neg_2pi_i = Scalar(std::complex<f32>(0.0f, -2.0f * (f32)pi_v<f64>));
    const TensorOptions opts_c = TensorOptions(device, eScalarType::ComplexFloat);

    Tensor signal = zeros({C, K}, opts_c);

    for (i64 k0 = 0; k0 < K; k0 += k_batch_size) {
        const i64    kB        = std::min(k_batch_size, K - k0);
        const Tensor k_b       = k_traj.narrow(0, k0, kB).to(eScalarType::ComplexFloat);
        const Tensor dft_phase = mm(k_b,
                                    coords_masked.to(eScalarType::ComplexFloat).transpose(0, 1))
                                   .mul(neg_2pi_i).exp();              // [kB, N_mask]
        const Tensor omega_b   = Omega.narrow(0, k0, kB);             // [kB, L]

        for (i64 c = 0; c < C; ++c) {
            const Tensor image_m = mag_masked.mul(coilmaps_masked.select(0, c));  // [N_mask]

            for (i64 k = 0; k < kB; ++k) {
                auto agg = zeros({n_hist}, opts_c);
                agg.scatter_add_(0, voxel_to_bin, dft_phase.select(0, k).mul(image_m));

                // [L] = mm([1, n_hist], [n_hist, L]).squeeze(0)
                const Tensor basis = mm(agg.unsqueeze(0), Upsilon).squeeze(0);

                signal.select(0, c).select(0, k0 + k).copy_(
                    omega_b.select(0, k).mul(basis).sum()
                );
            }
        }
    }

    return signal;
}

}
}
