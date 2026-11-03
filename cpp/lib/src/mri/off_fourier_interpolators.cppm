module;

export module hasty_mri_mod:off_fourier_interpolators;

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_linalg_mod;

namespace hasty {
namespace mri {

export struct PhiLowrankResult {
    Tensor Omega;    // [K, L]        — U * sqrt(S)
    Tensor S;        // [L]           — singular values (Float)
    Tensor Upsilon;  // [n_voxels, L] — V * sqrt(S)
};

// Weighting strategy for the phi low-rank SVD.
// Weights voxel j's column of the phi operator, biasing the decomposition
// towards voxels that carry more signal energy.
//
//   None      — unweighted Frobenius norm, every voxel equally important
//   L1Mass    — w_j = ρ[j]  (PD magnitude, current default)
//   L2Energy  — w_j = ρ[j]² (Parseval-optimal: minimises signal L2 energy
//               error when cross-voxel correlations are negligible)
//   Count     — w_j = 1 for every masked voxel. Unlike L1Mass/L2Energy,
//               this needs NO PD/magnitude value — only the mask and field
//               maps, both known a priori from scanner geometry/B0 map,
//               never the reconstructed image itself. The other two
//               require ground-truth PD, which doesn't exist before
//               reconstruction in a real pipeline (mag is only available
//               here because this test driver has a synthetic reference).
export enum struct eFeatureWeighting : i32 {
    None               = 0,
    L1Mass             = 1,
    L2Energy           = 2,
    FreqAwareLowrank   = 3,
    Count              = 4,
};

// One term in the concomitant field expansion:
//   B_con(r,t) ≈ Σ_q spatial_map_q(r) · waveform_q(t)
// where waveform_q arises from gradient products G_i(t)·G_j(t), NOT k(t).
// GNL separable terms (proportional to k-axis) are handled by CoordinateWarp.
//
// waveform is spoke-shaped [nspokes, nsamps] (not flat [K]) so make_phi_operator
// can subsample in time *within* a spoke — gradient products vary smoothly along
// a spoke, but jump between spokes, so subsampling must respect that boundary.
export struct ConcomitantBasis {
    Tensor spatial_map;  // [N] float               — field map evaluated at r-space voxels
    Tensor waveform;     // [nspokes, nsamps] float  — accumulated concomitant phase per sample
};

// Per-voxel feature set for the phi low-rank SVD — NOT a histogram. Every
// masked voxel keeps its own exact feature value (no value-based binning,
// no quantization error). voxel_to_feature is currently always the trivial
// identity (arange(N_mask)) since there is no compression step anymore;
// it's kept as an explicit indirection only so downstream consumers
// (make_phi_operator/approx_signal/Toeplitz embedding builders) have one
// stable "compact feature-row index" concept rather than two different ones.
export struct VoxelFeatures {
    Tensor mask_idx;           // [N_mask]      long  — flat voxel indices in image
    Tensor voxel_to_feature;   // [N_mask]      long  — feature-row index for each masked voxel (identity)
    i64    n_voxels;
    Tensor z_map_voxel;        // [n_voxels]    ComplexFloat — exact per-voxel z
    Tensor conc_fields_voxel;  // [Q, n_voxels] float        — exact per-voxel concomitant fields
    Tensor conc_waveforms;     // [Q, nspokes, nsamps] float — stacked waveforms from ConcomitantBasis
    Tensor voxel_weights;      // [n_voxels]    float        — ρ   per voxel (L1 mass)
    Tensor voxel_l2_energy;    // [n_voxels]    float        — ρ²  per voxel (L2 energy)
    Tensor voxel_count;        // [n_voxels]    float        — 1   per voxel (no PD needed)
};


// ---------------------------------------------------------------------------
// extract_voxel_features
//
// Builds the exact per-voxel feature set used by the phi low-rank SVD — z
// (off-resonance/decay) and the concomitant spatial fields, read directly at
// every masked voxel. No binning, no histogram, no quantization: this
// codebase used to bin these by value (n_rate per z-component, n_conc_bins
// per concomitant field), but that bin COUNT is a product over axis
// resolutions, independent of how many voxels actually exist
// (n_rate^2 * n_conc_bins^Q) — once a real (non-degenerate) T2/B0 map is in
// play this routinely exceeds N_mask itself, at which point "histogramming"
// was pure overhead, not compression, plus a source of quantization error
// the SVD step could never see or correct for. Going through every voxel
// exactly is the trivial upper bound on both accuracy (zero quantization
// error) and cost (n_voxels == N_mask, never more).
// concomitant may be empty (Q=0): only z features used.
// ---------------------------------------------------------------------------

export VoxelFeatures extract_voxel_features(
    const Tensor& mag_flat,
    const Tensor& z_map_flat,
    const std::vector<ConcomitantBasis>& concomitant
) {
    const i64    N      = mag_flat.size(0);
    const i64    Q      = (i64)concomitant.size();
    const Device device = mag_flat.device();
    const TensorOptions opts_f = TensorOptions(device, eScalarType::Float);
    const TensorOptions opts_l = TensorOptions(device, eScalarType::Long);

    if (mag_flat.scalar_type() != eScalarType::Float)
        throw std::invalid_argument("mag_flat must be Float");
    if (z_map_flat.scalar_type() != eScalarType::ComplexFloat)
        throw std::invalid_argument("z_map_flat must be ComplexFloat");

    i64 nspokes = 0, nsamps = 0;
    for (i64 q = 0; q < Q; ++q) {
        if (concomitant[q].spatial_map.scalar_type() != eScalarType::Float)
            throw std::invalid_argument("concomitant spatial_map must be Float");
        if (concomitant[q].spatial_map.ndimension() != 1 || concomitant[q].spatial_map.size(0) != N)
            throw std::invalid_argument("concomitant spatial_map must be [N]");
        if (concomitant[q].waveform.scalar_type() != eScalarType::Float)
            throw std::invalid_argument("concomitant waveform must be Float");
        if (concomitant[q].waveform.ndimension() != 2)
            throw std::invalid_argument("concomitant waveform must be [nspokes, nsamps]");
        if (q == 0) { nspokes = concomitant[q].waveform.size(0); nsamps = concomitant[q].waveform.size(1); }
        else if (concomitant[q].waveform.size(0) != nspokes || concomitant[q].waveform.size(1) != nsamps)
            throw std::invalid_argument("concomitant waveform spoke shape mismatch across terms");
    }

    auto mask     = mag_flat.gt(Scalar(0.0f));
    auto mask_idx = arange(N, opts_l).masked_select(mask);
    const i64 n_voxels = mask_idx.size(0);

    auto select_masked = [&](const Tensor& x) { return x.masked_select(mask); };

    auto mag_m         = select_masked(mag_flat);
    auto z_map_voxel   = select_masked(z_map_flat).contiguous();   // [n_voxels] ComplexFloat, exact per-voxel z

    std::vector<Tensor> conc_cols;
    conc_cols.reserve(Q);
    for (i64 q = 0; q < Q; ++q)
        conc_cols.push_back(select_masked(concomitant[q].spatial_map).unsqueeze(0));
    auto conc_fields_voxel = Q > 0 ? cat(conc_cols, 0) : empty({0, n_voxels}, opts_f);

    std::vector<Tensor> wf_rows;
    wf_rows.reserve(Q);
    for (i64 q = 0; q < Q; ++q) wf_rows.push_back(concomitant[q].waveform.unsqueeze(0));
    auto conc_waveforms = Q > 0 ? cat(wf_rows, 0) : empty({0, 0, 0}, opts_f);

    auto voxel_to_feature = arange(n_voxels, opts_l);   // identity
    auto voxel_weights    = mag_m.contiguous();
    auto voxel_l2_energy  = mag_m.mul(mag_m).contiguous();
    auto voxel_count      = ones({n_voxels}, opts_f);

    return VoxelFeatures{
        mask_idx, voxel_to_feature, n_voxels,
        z_map_voxel, conc_fields_voxel, conc_waveforms,
        voxel_weights, voxel_l2_energy, voxel_count
    };
}


// ---------------------------------------------------------------------------
// SpokeSubsampleInfo / PhiOperatorResult
//
// make_phi_operator subsamples time within each spoke (gradient waveforms —
// hence the concomitant phase — vary smoothly along a spoke, but jump between
// spokes, so the stride must respect spoke boundaries). The resulting SVD
// (phi_lowrank) runs on K_sub = nspokes*n_sub rows instead of the full
// K = nspokes*nsamps, then upsample_omega_spokes reconstructs the full-rate
// Omega via cubic interpolation per spoke. subsample=1 -> n_sub == nsamps,
// i.e. no subsampling (identity upsample).
// ---------------------------------------------------------------------------

export struct SpokeSubsampleInfo {
    i64 nspokes;
    i64 nsamps;     // full samples per spoke
    i64 n_sub;      // subsampled samples per spoke
    i64 subsample;  // stride
};

export struct PhiOperatorResult {
    linalg::LinearOperator op;  // [K_sub, n_voxels] — operates on subsampled time samples
    SpokeSubsampleInfo      spoke_info;
};

// ── Timing: how much of an operator_svd call is OUR matvec/rmatvec (the
// build_phi exp() evaluation) vs SLEPc/PETSc's own orthogonalization/restart
// bookkeeping (everything else inside SVDSolve). Same accumulate-into-global
// pattern as coordinate_warp.cppm's WarpTimingStats -- CUDA-synced around
// each timed region, otherwise this would measure host-side dispatch latency
// instead of actual device work.
export struct PhiOperatorTimingStats {
    double matvec_s  = 0.0;
    double rmatvec_s = 0.0;
    i64    matvec_calls  = 0;
    i64    rmatvec_calls = 0;
};

namespace { PhiOperatorTimingStats g_phi_timing; }

export PhiOperatorTimingStats get_phi_operator_timing_stats() { return g_phi_timing; }
export void reset_phi_operator_timing_stats() { g_phi_timing = PhiOperatorTimingStats{}; }


// ---------------------------------------------------------------------------
// make_phi_operator
//
// Builds the [K_sub, n_voxels] phi operator from VoxelFeatures:
//   phi[k,h] = exp(-z_h · t_k) · Π_q exp(i · conc_fields_voxel[q,h] · conc_waveforms[q,k])
// where z_h = B0+off-resonance (complex), conc terms are real accumulated phase.
//
// timestamps is spoke-shaped [nspokes, nsamps]. subsample=1 uses every sample
// (K_sub == nspokes*nsamps); subsample=S keeps every S-th sample within each
// spoke (indices 0, S, 2S, ... clamped to the spoke length) — pair with
// phi_lowrank then upsample_omega_spokes to recover full time resolution.
//
// The exp()/matmul-heavy work (building phi, the Lanczos matvec contraction)
// runs in ComplexFloat — the dominant cost is transcendental evaluation, and a
// one-time ~1e-7 relative error per phi entry doesn't compound across SVD
// iterations. Only the small per-chunk *output* vector is cast up to
// ComplexDouble before accumulating into the result, so the externally
// visible operator dtype (and Lanczos's own iterate) stays double precision —
// avoiding the float32<->float64 round-trip-per-iteration error growth that
// motivated building everything in double originally.
// ---------------------------------------------------------------------------

export PhiOperatorResult make_phi_operator(
    const VoxelFeatures& feat,
    const Tensor& timestamps,   // [nspokes, nsamps] float
    i64 subsample  = 1,
    i64 chunk_size = 512
) {
    if (timestamps.scalar_type() != eScalarType::Float)
        throw std::invalid_argument("timestamps must be Float");
    if (timestamps.ndimension() != 2)
        throw std::invalid_argument("timestamps must be [nspokes, nsamps]");
    if (subsample < 1)
        throw std::invalid_argument("subsample must be >= 1");

    const i64    nspokes = timestamps.size(0);
    const i64    nsamps  = timestamps.size(1);
    const i64    N       = feat.n_voxels;
    const i64    Q       = feat.conc_fields_voxel.size(0);
    const Device device  = feat.z_map_voxel.device();

    if (Q > 0 && (feat.conc_waveforms.size(0) != Q
               || feat.conc_waveforms.size(1) != nspokes
               || feat.conc_waveforms.size(2) != nsamps))
        throw std::invalid_argument("conc_waveforms shape mismatch with timestamps");

    // Uniform stride within each spoke; n_sub may undershoot nsamps-1 by up to
    // (subsample-1) samples when it doesn't divide evenly — upsample_omega_spokes
    // clamps the trailing tail flat (constant extrapolation) in that case.
    const i64 n_sub = (nsamps - 1) / subsample + 1;
    auto sub_idx = arange(n_sub, TensorOptions(device, eScalarType::Long))
                       .mul(Scalar(subsample));

    const i64 K_sub = nspokes * n_sub;

    auto timestamps_sub = timestamps.index_select(1, sub_idx).reshape({K_sub});

    Tensor conc_waveforms_sub = Q > 0
        ? feat.conc_waveforms.index_select(2, sub_idx).reshape({Q, K_sub})
        : empty({0, K_sub}, TensorOptions(device, eScalarType::Float));

    const Scalar pos_i_f = Scalar(std::complex<f32>(0.0f, 1.0f));

    // Hoisted out of build_phi -- none of these depend on the chunk (k0,kB),
    // so computing them inside the per-chunk lambda (as a previous version
    // did) redundantly recast the SAME n_voxels-sized array on every one of
    // K_sub/chunk_size chunks per matvec/rmatvec call (hundreds of times per
    // call at realistic chunk_size). Compute once per operator construction.
    const Tensor zm_f = feat.z_map_voxel.to(eScalarType::ComplexFloat);              // [n_voxels]
    std::vector<Tensor> conc_fields_cf;
    conc_fields_cf.reserve(Q);
    for (i64 q = 0; q < Q; ++q)
        conc_fields_cf.push_back(
            feat.conc_fields_voxel.select(0, q).to(eScalarType::ComplexFloat).unsqueeze(0));  // [1,n_voxels]

    // ComplexFloat — see function doc for why this doesn't reintroduce the
    // round-trip error the original ComplexDouble build avoided.
    //
    // exp(a)*exp(b)*exp(c) == exp(a+b+c) EXACTLY (not an approximation) --
    // sum all Q+1 phase/decay contributions into one exponent first, call
    // exp() ONCE at the end, instead of Q+1 separate exp() calls multiplied
    // together. exp() of a [kB,n_voxels] complex tensor is the dominant cost
    // here (confirmed via HastyPhiSvdTimingTest: ~99.9% of phi_lowrank's
    // wall time is inside this lambda) -- cutting Q+1 transcendental passes
    // down to 1 is a direct, exact reduction of that cost, not a tradeoff.
    auto build_phi = [=](i64 k0, i64 kB) -> Tensor {
        auto t_c = timestamps_sub.narrow(0, k0, kB).to(eScalarType::ComplexFloat);
        auto exponent = zm_f.unsqueeze(0).mul(-t_c.unsqueeze(1));  // [kB, n_voxels]
        for (i64 q = 0; q < Q; ++q) {
            exponent = exponent.add(
                conc_fields_cf[q]
                    .mul(conc_waveforms_sub.select(0, q).narrow(0, k0, kB)
                             .to(eScalarType::ComplexFloat).unsqueeze(1))
                    .mul(pos_i_f)
            );
        }
        return exponent.exp();  // [kB, n_voxels] ComplexFloat
    };

    // Operator's EXTERNALLY-VISIBLE dtype is ComplexFloat (SLEPc's Lanczos
    // basis vectors/orthogonalization now run in float32 -- half the memory
    // per basis vector, faster BLAS -- since build_phi's transcendental
    // evaluation is float32-precision anyway, ~1e-7 rel err/entry; a
    // double-precision iterate was never recovering accuracy float32 had
    // already lost there). Internal ACCUMULATION (the actual reduction-style
    // sums below -- mv_fn's per-chunk dot product, rmv_fn's cross-chunk sum
    // over k) still happens via a ComplexDouble buffer, only cast down to
    // ComplexFloat once at the very end -- these sums are genuinely
    // integral-like (many terms accumulating into one output element), so
    // keeping the ACCUMULATOR in double avoids compounding round-off across
    // chunks even though each chunk's own terms are float32-evaluated.
    auto mv_fn = [=](const Tensor& x) -> Tensor {
        if (device.type == eDeviceType::CUDA) cuda::synchronize(device);
        auto t0 = std::chrono::steady_clock::now();

        auto result = zeros({K_sub}, TensorOptions(device, eScalarType::ComplexDouble));
        auto x_f    = x.to(eScalarType::ComplexFloat);
        for (i64 k0 = 0; k0 < K_sub; k0 += chunk_size) {
            const i64 kB = std::min(chunk_size, K_sub - k0);
            result.narrow(0, k0, kB).copy_(
                mv(build_phi(k0, kB), x_f).to(eScalarType::ComplexDouble));
        }
        auto out = result.to(eScalarType::ComplexFloat);

        if (device.type == eDeviceType::CUDA) cuda::synchronize(device);
        g_phi_timing.matvec_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++g_phi_timing.matvec_calls;
        return out;
    };

    auto rmv_fn = [=](const Tensor& y) -> Tensor {
        if (device.type == eDeviceType::CUDA) cuda::synchronize(device);
        auto t0 = std::chrono::steady_clock::now();

        auto result = zeros({N}, TensorOptions(device, eScalarType::ComplexDouble));
        auto y_f    = y.to(eScalarType::ComplexFloat);
        for (i64 k0 = 0; k0 < K_sub; k0 += chunk_size) {
            const i64 kB = std::min(chunk_size, K_sub - k0);
            auto chunk = mv(build_phi(k0, kB).conj().transpose(0, 1), y_f.narrow(0, k0, kB));
            result += chunk.to(eScalarType::ComplexDouble);
        }
        auto out = result.to(eScalarType::ComplexFloat);

        if (device.type == eDeviceType::CUDA) cuda::synchronize(device);
        g_phi_timing.rmatvec_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++g_phi_timing.rmatvec_calls;
        return out;
    };

    linalg::LinearOperator op(K_sub, N,
        std::move(mv_fn), std::move(rmv_fn),
        eScalarType::ComplexFloat, device);

    return PhiOperatorResult{
        std::move(op),
        SpokeSubsampleInfo{nspokes, nsamps, n_sub, subsample}
    };
}


// ---------------------------------------------------------------------------
// upsample_omega_spokes
//
// Reconstructs full time-resolution Omega [nspokes*nsamps, L] from the
// subsampled-SVD Omega_sub [nspokes*n_sub, L] via Catmull-Rom cubic
// interpolation along the time axis, independently per spoke (Upsilon, the
// spatial/voxel-feature basis, is time-independent and needs no reconstruction).
//
// subsample=1 (n_sub == nsamps) is the identity — returned unchanged.
// ---------------------------------------------------------------------------

export Tensor upsample_omega_spokes(
    const Tensor& Omega_sub,           // [nspokes*n_sub, L] ComplexFloat
    const SpokeSubsampleInfo& info
) {
    if (info.n_sub == info.nsamps)
        return Omega_sub;

    const i64    L      = Omega_sub.size(1);
    const Device device = Omega_sub.device();

    auto src = Omega_sub.reshape({info.nspokes, info.n_sub, L});

    const TensorOptions opts_f = TensorOptions(device, eScalarType::Float);
    const TensorOptions opts_l = TensorOptions(device, eScalarType::Long);

    auto j         = arange(info.nsamps, opts_f);
    auto u         = j.div(Scalar((f32)info.subsample));
    auto u_clamped = clamp(u, Scalar(0.0f), Scalar((f32)(info.n_sub - 1)));
    auto i0_f      = hasty::floor(u_clamped);
    auto t         = u_clamped.sub(i0_f);                 // fractional part, [nsamps]
    auto i0        = i0_f.to(eScalarType::Long);

    auto clamp_idx = [&](const Tensor& idx) {
        return clamp(idx, Scalar((i64)0), Scalar((i64)(info.n_sub - 1)));
    };
    auto i_m1 = clamp_idx(i0.sub(Scalar((i64)1)));
    auto i_p1 = clamp_idx(i0.add(Scalar((i64)1)));
    auto i_p2 = clamp_idx(i0.add(Scalar((i64)2)));
    auto i0c  = clamp_idx(i0);

    auto t2 = t.mul(t);
    auto t3 = t2.mul(t);

    auto w_m1 = t3.mul(Scalar(-0.5f)).add(t2).sub(t.mul(Scalar(0.5f)));
    auto w_0  = t3.mul(Scalar(1.5f)).sub(t2.mul(Scalar(2.5f))).add(Scalar(1.0f));
    auto w_1  = t3.mul(Scalar(-1.5f)).add(t2.mul(Scalar(2.0f))).add(t.mul(Scalar(0.5f)));
    auto w_2  = t3.mul(Scalar(0.5f)).sub(t2.mul(Scalar(0.5f)));

    auto cw = [&](const Tensor& w) {
        return w.to(eScalarType::ComplexFloat).unsqueeze(0).unsqueeze(2);  // [1, nsamps, 1]
    };
    auto tap = [&](const Tensor& idx) {
        return src.index_select(1, idx);  // [nspokes, nsamps, L]
    };

    auto out = tap(i_m1).mul(cw(w_m1))
                  .add(tap(i0c).mul(cw(w_0)))
                  .add(tap(i_p1).mul(cw(w_1)))
                  .add(tap(i_p2).mul(cw(w_2)));

    return out.reshape({info.nspokes * info.nsamps, L});
}


// ---------------------------------------------------------------------------
// phi_lowrank_weights
// Computes the weight vector [n_voxels] for the phi_lowrank weighted SVD.
// Returns nullopt for eFeatureWeighting::None (unweighted).
// ---------------------------------------------------------------------------

export Opt<Tensor> phi_lowrank_weights(
    eFeatureWeighting weighting,
    const VoxelFeatures& feat,
    const Tensor& timestamps    // [K] float — needed for FreqAwareLowrank
) {
    using namespace std::numbers;
    switch (weighting) {
        case eFeatureWeighting::None:
            return nullopt;
        case eFeatureWeighting::L1Mass:
            return feat.voxel_weights;
        case eFeatureWeighting::L2Energy:
            return feat.voxel_l2_energy;
        case eFeatureWeighting::Count:
            return feat.voxel_count;
        case eFeatureWeighting::FreqAwareLowrank: {
            float T_readout = (timestamps.max() - timestamps.min()).item<f32>();
            float pi_f = (float)pi_v<f64>;
            auto phase_h    = feat.z_map_voxel.imag().abs()
                                  .mul(Scalar(T_readout)).div(Scalar(pi_f));
            auto freq_factor = clamp(phase_h, Scalar(1.0f), Scalar(1e9f));
            return feat.voxel_l2_energy.mul(freq_factor);
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
    i64 mpd = -1,
    double tol = 5e-7
) {
    const i64    K      = op.m();
    const i64    N      = op.n();
    const Device device = op.device();

    if (L > std::min(K, N))
        throw std::invalid_argument(
            "phi_lowrank: requested L=" + std::to_string(L) +
            " exceeds operator rank bound min(K,N)=" + std::to_string(std::min(K, N)) +
            " — n_voxels should never be this small (no off-resonance/concomitant "
            "variation in the input?); this is a caller bug, not something to silently clamp.");

    linalg::SVDResult svd;
    Tensor w_sqrt;

    if (weights.has_value()) {
        w_sqrt = pow(*weights, 0.5).to(op.dtype());

        linalg::LinearOperator op_w(K, N,
            [op, w_sqrt](const Tensor& v) { return op.matvec(v.mul(w_sqrt)); },
            [op, w_sqrt](const Tensor& y) { return op.rmatvec(y).mul(w_sqrt); },
            op.dtype(), device
        );
        svd = linalg::operator_svd(op_w, L, ncv, mpd, tol);
    } else {
        svd = linalg::operator_svd(op, L, ncv, mpd, tol);
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
// subsample_voxel_features
//
// Gathers a VoxelFeatures down to a subset of voxels (by index into the
// existing n_voxels-indexed arrays). conc_waveforms is shared across all
// voxels (time/spoke-indexed, not voxel-indexed) and passed through
// unchanged. Used by phi_lowrank_batched to build small per-batch operators.
// ---------------------------------------------------------------------------
export VoxelFeatures subsample_voxel_features(const VoxelFeatures& feat, const Tensor& idx)
{
    return VoxelFeatures{
        feat.mask_idx.index_select(0, idx),
        feat.voxel_to_feature.index_select(0, idx),
        idx.size(0),
        feat.z_map_voxel.index_select(0, idx),
        feat.conc_fields_voxel.index_select(1, idx),
        feat.conc_waveforms,
        feat.voxel_weights.index_select(0, idx),
        feat.voxel_l2_energy.index_select(0, idx),
        feat.voxel_count.index_select(0, idx)
    };
}


// ---------------------------------------------------------------------------
// bin_voxel_features / bin_z_features / bin_conc_features
//
// Value-based binning, REINTRODUCED here scoped to a single feature piece
// (z alone, or concomitant alone) -- the reason it was removed earlier was
// the JOINT z*concomitant feature space exploding combinatorially
// (n_rate^2 * n_conc_bins^Qc), which a per-voxel SVD pays for in voxel
// count rather than bin count regardless. Now that z and concomitant are
// fit SEPARATELY (phi_lowrank_outer_product combines them only AFTER each
// fit), that multiplication never happens -- a z-only 2D histogram or a
// concomitant-only Qc-dim histogram is independently small (thousands, not
// millions), so binning is viable again exactly where it wasn't before.
//
// Per-dimension uniform bins over [min,max], combined into one flat index
// via mixed-radix, then unique_with_inverse finds the actually-OCCUPIED
// bins (real anatomy never fills the full combinatorial grid) and gives the
// voxel->bin mapping directly. Each occupied bin's representative feature
// value is the voxel_weights-weighted mean of the voxels mapping to it (not
// the bin's geometric center -- the weighted mean is what the downstream
// weighted SVD actually wants to fit well).
// ---------------------------------------------------------------------------
struct BinResult {
    Tensor centroid;      // [n_occupied, D] real -- weighted mean feature value per bin
    Tensor voxel_to_bin;  // [n_voxels] long -- which occupied bin each voxel maps to
    Tensor bin_weight;    // [n_occupied] real -- summed voxel_weights per bin
    std::vector<i64> n_occupied_per_dim;  // [D] -- distinct occupied 1D bin values per dimension,
                                            // BEFORE combining into the joint flat index (diagnostic:
                                            // how much of the joint bin count comes from each axis).
};

static BinResult bin_features_nd(const Tensor& features, const Tensor& voxel_weights, ArrayRef<i64> bins_per_dim)
{
    const i64 n_voxels = features.size(0);
    const i64 D = features.size(1);
    const Device dev = features.device();
    const auto opts_f = TensorOptions(dev, eScalarType::Float);
    const auto opts_l = TensorOptions(dev, eScalarType::Long);

    Tensor flat_idx = zeros({n_voxels}, opts_l);
    std::vector<i64> n_occupied_per_dim(D);
    i64 stride = 1;
    for (i64 d = 0; d < D; ++d) {
        auto col = features.select(1, d).to(eScalarType::Float);
        double lo = col.min().item<float>(), hi = col.max().item<float>();
        double range = std::max(hi - lo, 1e-12);
        auto bin_d = col.sub(Scalar((float)lo)).div(Scalar((float)range)).mul(Scalar((float)bins_per_dim[d]))
                         .clamp(Scalar(0.0f), Scalar((float)(bins_per_dim[d] - 1)))
                         .to(eScalarType::Long);
        n_occupied_per_dim[d] = unique_with_inverse(bin_d).first.size(0);
        flat_idx = flat_idx.add(bin_d.mul(Scalar(stride)));
        stride *= bins_per_dim[d];
    }

    auto [occupied, voxel_to_bin] = unique_with_inverse(flat_idx);
    const i64 n_occ = occupied.size(0);

    auto w = voxel_weights.to(eScalarType::Float).contiguous();
    auto bin_weight = zeros({n_occ}, opts_f);
    bin_weight.scatter_add_(0, voxel_to_bin, w);
    auto bin_weight_safe = bin_weight.clamp_min(Scalar(1e-12f));

    auto centroid = zeros({n_occ, D}, opts_f);
    for (i64 d = 0; d < D; ++d) {
        auto weighted_sum = zeros({n_occ}, opts_f);
        weighted_sum.scatter_add_(0, voxel_to_bin, features.select(1, d).to(eScalarType::Float).mul(w));
        centroid.select(1, d).copy_(weighted_sum.div(bin_weight_safe));
    }

    return BinResult{centroid, voxel_to_bin, bin_weight, n_occupied_per_dim};
}

export struct BinnedFeatures {
    VoxelFeatures feat;       // n_voxels == n_occupied bins
    Tensor        voxel_to_bin;  // [n_voxels_original] long, expand a per-bin result back via index_select(0, voxel_to_bin)
    std::vector<i64> n_occupied_per_dim;  // per-dimension distinct-value counts, see BinResult
};

export BinnedFeatures bin_z_features(const VoxelFeatures& z_only_feat, i64 n_r2_bins, i64 n_b0_bins)
{
    const Device dev = z_only_feat.z_map_voxel.device();
    auto z_ri = stack({z_only_feat.z_map_voxel.real(), z_only_feat.z_map_voxel.imag()}, 1);   // [n_voxels, 2]

    auto br = bin_features_nd(z_ri, z_only_feat.voxel_weights, {n_r2_bins, n_b0_bins});
    const i64 n_occ = br.centroid.size(0);

    auto z_map_binned = view_as_complex(br.centroid.contiguous());   // [n_occ] complex

    VoxelFeatures feat{
        zeros({n_occ}, TensorOptions(dev, eScalarType::Long)),   // mask_idx -- meaningless post-binning, unused downstream
        arange(n_occ, TensorOptions(dev, eScalarType::Long)),
        n_occ,
        z_map_binned.to(eScalarType::ComplexFloat),
        empty({0, n_occ}, TensorOptions(dev, eScalarType::Float)),
        empty({0, 0, 0}, TensorOptions(dev, eScalarType::Float)),
        br.bin_weight, br.bin_weight.mul(br.bin_weight), ones({n_occ}, TensorOptions(dev, eScalarType::Float))
    };
    return BinnedFeatures{feat, br.voxel_to_bin, br.n_occupied_per_dim};
}

export BinnedFeatures bin_conc_features(const VoxelFeatures& conc_only_feat, i64 bins_per_q)
{
    const i64 Qc = conc_only_feat.conc_fields_voxel.size(0);
    const Device dev = conc_only_feat.z_map_voxel.device();
    auto features = conc_only_feat.conc_fields_voxel.transpose(0, 1).contiguous();   // [n_voxels, Qc]

    std::vector<i64> bins_per_dim(Qc, bins_per_q);
    auto br = bin_features_nd(features, conc_only_feat.voxel_weights, bins_per_dim);
    const i64 n_occ = br.centroid.size(0);

    VoxelFeatures feat{
        zeros({n_occ}, TensorOptions(dev, eScalarType::Long)),
        arange(n_occ, TensorOptions(dev, eScalarType::Long)),
        n_occ,
        zeros({n_occ}, TensorOptions(dev, eScalarType::ComplexFloat)),
        br.centroid.transpose(0, 1).contiguous(),     // [Qc, n_occ]
        conc_only_feat.conc_waveforms,                // time-domain, not binned -- shared as-is
        br.bin_weight, br.bin_weight.mul(br.bin_weight), ones({n_occ}, TensorOptions(dev, eScalarType::Float))
    };
    return BinnedFeatures{feat, br.voxel_to_bin, br.n_occupied_per_dim};
}


// ---------------------------------------------------------------------------
// split_conc_dc
//
// Exact decomposition Φ_conc(r,t) = Φ_DC(r) + Φ_AC(r,t), Φ_DC(r) = Σ_q
// N_q(r)·w̄_q (w̄_q = time-average of conc_waveforms[q,:,:]). Φ_DC is a fixed,
// closed-form, t-independent per-voxel phase -- zero SVD/rank cost to apply
// (just exp(-i·Φ_DC) multiplied into Upsilon after the conc-only fit, see
// phi_lowrank_outer_product's caller). Φ_AC (the returned feat's
// conc_waveforms) has smaller dynamic range than the original, so the
// downstream per-voxel SVD needs fewer modes to fit it to a given tolerance.
// Exact, not approximate -- pure redistribution of where the DC term is
// handled, no information lost.
// ---------------------------------------------------------------------------
export struct ConcDcSplit {
    VoxelFeatures feat_ac;   // conc_waveforms replaced by the zero-mean residual
    Tensor        phi_dc;    // [n_voxels] real -- Σ_q N_q(r)·w̄_q
};

export ConcDcSplit split_conc_dc(const VoxelFeatures& feat)
{
    const i64 Q = feat.conc_fields_voxel.size(0);
    if (Q == 0)
        return ConcDcSplit{feat, zeros({feat.n_voxels}, TensorOptions(feat.z_map_voxel.device(), eScalarType::Float))};

    auto w_bar = feat.conc_waveforms.mean({(i64)1, (i64)2});                  // [Q]
    auto conc_waveforms_ac = feat.conc_waveforms.sub(w_bar.reshape({Q, 1, 1}));

    auto phi_dc = feat.conc_fields_voxel.mul(w_bar.unsqueeze(1)).sum(0);      // [n_voxels]

    VoxelFeatures feat_ac = feat;
    feat_ac.conc_waveforms = conc_waveforms_ac;
    return ConcDcSplit{feat_ac, phi_dc};
}


// ---------------------------------------------------------------------------
// phi_lowrank_outer_product
//
// Combines two SEPARATE per-voxel SVD fits (e.g. z-only and concomitant-only,
// see notes/MRI_Physics.tex's "Reducing the per-term cost..." discussion and
// the surrounding chat -- jointly fitting z+concomitant in one SVD needs more
// temporal modes than either alone, since concomitant phase adds independent
// oscillatory structure the SVD can't reuse z's modes for; fitting them
// SEPARATELY and combining here avoids that coupling at the SVD stage) into
// one combined PhiLowrankResult, via the exact algebraic identity
// exp(-tz)*exp(iΦ_conc) ≈ [Σ_l U_a[k,l]S_a[l]Υ_a[j,l]]·[Σ_o U_b[k,o]S_b[o]Υ_b[j,o]]
//                       =  Σ_{l,o} (U_a[k,l]U_b[k,o])·(S_a[l]S_b[o])·(Υ_a[j,l]Υ_b[j,o])
// i.e. plain outer product over the (l,o) mode index, flattened. Matches the
// established Omega=U (raw), S separate, Upsilon=conj(V) convention exactly
// -- no extra conjugates needed, an outer product of two factorizations in
// that convention stays in that convention.
// ---------------------------------------------------------------------------
export PhiLowrankResult phi_lowrank_outer_product(
    const PhiLowrankResult& a, const PhiLowrankResult& b)
{
    const i64 K  = a.Omega.size(0);
    const i64 N  = a.Upsilon.size(0);
    const i64 La = a.Omega.size(1), Lb = b.Omega.size(1);

    auto Omega = a.Omega.unsqueeze(2).mul(b.Omega.unsqueeze(1)).reshape({K, La * Lb});
    auto S     = a.S.unsqueeze(1).mul(b.S.unsqueeze(0)).reshape({La * Lb});
    auto Upsilon = a.Upsilon.unsqueeze(2).mul(b.Upsilon.unsqueeze(1)).reshape({N, La * Lb});

    return PhiLowrankResult{Omega, S, Upsilon};
}


// ---------------------------------------------------------------------------
// phi_lowrank_batched
//
// SLEPc's TRLanczos allocates Krylov/Lanczos basis vectors of size
// O(ncv * n_voxels) for the SVD's right-singular-vector side (this is
// intrinsic to Lanczos-type methods, not an orientation bug -- see the
// investigation in the doc above precompute_coil_gram's sibling discussion;
// confirmed no swap of which operator dimension is "m" vs "n" changes this,
// since SOME basis has to span the n_voxels-dimensional space regardless).
// At production scale (n_voxels ~1e7) this is tens of GB, genuinely OOMs
// regardless of caching-allocator hygiene.
//
// Fix: partition voxels into batches small enough to fit (each batch's own
// rank-L SVD is cheap, K_sub x batch_size), then COMBINE the per-batch
// temporal bases (Omega_i, K-indexed, same K for every batch since only the
// voxel/column set differs) via a small merge-SVD -- this is standard
// incremental/block-SVD/distributed-PCA practice (e.g. Brand 2002's
// incremental SVD; the "concatenate U_i*S_i and re-SVD" merge step is
// Eckart-Young-optimal recombination of the per-batch low-rank summaries,
// weighted by each batch's own energy). Unlike single-subsample approaches,
// this sees EVERY voxel (no representativeness assumption needed -- directly
// addresses the concern that random/single-sample subsetting could miss
// rare-but-important feature values, e.g. short-T2 outlier voxels).
//
// Math (see also doc above phi_lowrank for the Omega=U, S, Upsilon=conj(V)
// convention this matches exactly):
//   Per batch i: fit rank-L SVD of A_i (the batch's [K_sub, batch_size_i]
//   operator) -> Omega_i=U_i, S_i (Upsilon_i discarded -- only valid for
//   that batch's own voxels, not needed once we re-derive the full Upsilon
//   below).
//   Merge: M = concat_i(Omega_i * S_i) along columns -> [K_sub, B*L]. SVD of
//   M (cheap dense/economy SVD, B*L is tiny) -> top-L (U_g, S_g) is the
//   combined Omega/S estimate.
//   Extend: for ANY voxel j (not just sampled ones), using SVD orthonormality
//   of U_g:  conj(V_full[j,l]) = (1/S_g[l]) * conj(rmatvec_full(U_g[:,l])[j])
//   i.e. Upsilon_full[:,l] = (1/S_g[l]) * conj(op.rmatvec(U_g[:,l])) -- a
//   SINGLE adjoint matvec per mode, no Lanczos basis storage needed. Done
//   per-batch (reusing each batch's own bounded-size operator) so this
//   extension step ALSO never materializes a full-n_voxels-wide intermediate
//   (make_phi_operator's build_phi chunks over K but not over voxels, so a
//   full-N rmatvec call would hit the exact same memory wall this function
//   exists to avoid).
//
// batch_size: configurable -- larger batches make each batch's own rank-L
// truncation capture more of that batch's local structure (this is an
// approximation, exact only if each batch is itself exactly rank<=L), so
// quality improves monotonically with batch_size up to whatever fits in
// memory. Independent batches (no shared state across the per-batch SVD
// fits) -- straightforward to dispatch across multiple GPUs later via the
// same load-balancer pattern used in normal_toeplitz.cppm's
// RUN_ALL_ON_INPUT_DEVICE vs load-balanced branches; not done here, this
// pass keeps it sequential on one device.
// ---------------------------------------------------------------------------
export PhiLowrankResult phi_lowrank_batched(
    const VoxelFeatures& feat,
    const Tensor& timestamps,        // [nspokes, nsamps] float
    const Tensor& timestamps_flat,   // [K] float -- for phi_lowrank_weights (FreqAwareLowrank)
    i64 subsample,
    eFeatureWeighting weighting,
    i64 L,
    i64 batch_size,
    i64 ncv = -1,
    i64 mpd = -1
) {
    using namespace linalg;

    const i64    N      = feat.n_voxels;
    const Device device = feat.z_map_voxel.device();
    const i64    B       = (N + batch_size - 1) / batch_size;
    const i64    mpd_val = mpd > 0 ? mpd : std::max((i64)8 * L, (i64)60);
    const i64    ncv_val = ncv > 0 ? ncv : (L + mpd_val);

    std::vector<VoxelFeatures> feat_batches;
    feat_batches.reserve(B);

    std::vector<Tensor> cols;   // each [K_sub, L] ComplexDouble = U_i * S_i
    cols.reserve(B);

    for (i64 b = 0; b < B; ++b) {
        const i64 start = b * batch_size;
        const i64 end   = std::min(N, start + batch_size);
        auto idx = arange(start, end, TensorOptions(device, eScalarType::Long));
        auto feat_b = subsample_voxel_features(feat, idx);

        // make_phi_operator's build_phi materializes a [chunk_size, batch_n]
        // ComplexFloat tensor per chunk (chunks over K only, never over
        // voxels) -- its default chunk_size=512 assumes batch_n is modest;
        // at batch_n~1e6 that's already ~3.8GiB per chunk. Scale chunk_size
        // down so each chunk stays near a fixed memory budget regardless of
        // batch_size, instead of inheriting a constant tuned for a much
        // smaller n_voxels.
        const i64 batch_n = end - start;
        const i64 target_bytes = 256 * 1024 * 1024;  // ~256MiB per build_phi chunk
        const i64 chunk_size_b = std::max((i64)1, target_bytes / (batch_n * 8));

        auto phi_res_b = make_phi_operator(feat_b, timestamps, subsample, chunk_size_b);
        auto weights_b = phi_lowrank_weights(weighting, feat_b, timestamps_flat);
        auto plr_b = phi_lowrank(phi_res_b.op, L, weights_b, ncv_val, mpd_val);

        auto cols_b = plr_b.Omega.to(eScalarType::ComplexDouble)
                          .mul(plr_b.S.to(eScalarType::ComplexDouble).unsqueeze(0));  // [K_sub,L]

        feat_batches.push_back(std::move(feat_b));
        cols.push_back(cols_b);

        // Same caching-allocator-retention issue as elsewhere in this
        // codebase -- each batch's SVD/Lanczos run builds and discards its
        // own workspace; clear so the NEXT batch (and the merge/extension
        // steps after) start from a clean slate instead of accumulating
        // idle-but-reserved memory across B batches.
        if (device.type == eDeviceType::CUDA) { cuda::synchronize(device); cuda::empty_cache(); }
    }

    auto M = cat(cols, 1);                       // [K_sub, B*L] ComplexDouble
    auto merge_svd = linalg_svd(M, false);        // descending singular values
    auto U_g = merge_svd.U.narrow(1, 0, L).contiguous();   // [K_sub,L] ComplexDouble
    auto S_g = merge_svd.S.narrow(0, 0, L).contiguous();   // [L] real Double

    auto Upsilon_full = zeros({N, L}, TensorOptions(device, eScalarType::ComplexDouble));
    for (i64 b = 0; b < B; ++b) {
        const i64 batch_n = feat_batches[b].n_voxels;
        const i64 target_bytes = 256 * 1024 * 1024;
        const i64 chunk_size_b = std::max((i64)1, target_bytes / (batch_n * 8));
        auto phi_res_b = make_phi_operator(feat_batches[b], timestamps, subsample, chunk_size_b);
        for (i64 l = 0; l < L; ++l) {
            auto y = phi_res_b.op.rmatvec(U_g.select(1, l).contiguous());   // [batch_n] ComplexDouble
            double S_l = S_g.select(0, l).item<double>();
            auto col = y.conj().div(Scalar(S_l));
            Upsilon_full.narrow(0, b * batch_size, batch_n).select(1, l).copy_(col);
        }
        if (device.type == eDeviceType::CUDA) { cuda::synchronize(device); cuda::empty_cache(); }
    }

    return PhiLowrankResult{
        U_g.to(eScalarType::ComplexFloat),
        S_g.to(eScalarType::Float),
        Upsilon_full.to(eScalarType::ComplexFloat)
    };
}


// ---------------------------------------------------------------------------
// time_segmented_phi
//
// Classic time-segmentation approximation (off-resonance only).
// Fixes L spatial maps Υ[h,l] = exp(-z_h·τ_l) at uniformly-spaced segment
// times τ_l ∈ [t_min, t_max], then finds the weighted-LS-optimal temporal
// coefficients Ω[k,l] that minimise
//   Σ_h w_h |Σ_l Ω[k,l]·Υ[h,l] - exp(-z_h·t_k)|²
// for each k independently.
//
// Concomitant fields are handled by make_phi_operator / phi_lowrank.
// Returns PhiLowrankResult — same struct as SVD → same approx_signal call.
// ---------------------------------------------------------------------------

export PhiLowrankResult time_segmented_phi(
    const VoxelFeatures& feat,
    const Tensor& timestamps,                   // [K] float
    i64 L,
    eFeatureWeighting weighting = eFeatureWeighting::L2Energy,
    i64 chunk_size          = 512
) {
    const i64    K      = timestamps.size(0);
    const i64    n_voxels = feat.n_voxels;
    const Device device = timestamps.device();

    using namespace std::numbers;
    const float pi_f = (float)pi_v<f64>;

    const TensorOptions opts_cd = TensorOptions(device, eScalarType::ComplexDouble);
    const TensorOptions opts_d  = TensorOptions(device, eScalarType::Double);

    // ── Weights ──────────────────────────────────────────────────────────────
    Tensor w;
    switch (weighting) {
        case eFeatureWeighting::None:
            w = ones({n_voxels}, TensorOptions(device, eScalarType::Float)); break;
        case eFeatureWeighting::L1Mass:
            w = feat.voxel_weights; break;
        case eFeatureWeighting::L2Energy:
            w = feat.voxel_l2_energy; break;
        case eFeatureWeighting::Count:
            w = feat.voxel_count; break;
        case eFeatureWeighting::FreqAwareLowrank: {
            float T_readout = (timestamps.max() - timestamps.min()).item<f32>();
            auto phase_h    = feat.z_map_voxel.imag().abs()
                                  .mul(Scalar(T_readout)).div(Scalar(pi_f));
            auto freq_factor = clamp(phase_h, Scalar(1.0f), Scalar(1e9f));
            w = feat.voxel_l2_energy.mul(freq_factor); break;
        }
    }
    auto w_d = w.to(eScalarType::Double).to(eScalarType::ComplexDouble);  // [n_voxels] cd

    // ── Segment times τ_l ∈ [t_min, t_max] ──────────────────────────────────
    const f64 t_min = (f64)timestamps.min().item<f32>();
    const f64 t_max = (f64)timestamps.max().item<f32>();
    const f64 denom = (L > 1) ? (f64)(L - 1) : 1.0;
    auto tau = arange(L, opts_d)
                   .div(Scalar(denom))
                   .mul(Scalar(t_max - t_min))
                   .add(Scalar(t_min));                          // [L] double

    // ── Spatial maps: Υ[n_voxels, L] = exp(-z_h·τ_l) ──────────────────────────
    auto z_d      = feat.z_map_voxel.to(eScalarType::ComplexDouble);  // [n_voxels]
    auto tau_cd   = tau.to(eScalarType::ComplexDouble);               // [L]
    auto Upsilon  = (-z_d.unsqueeze(1).mul(tau_cd.unsqueeze(0))).exp(); // [n_voxels, L]

    // ── A[n_voxels, L] = w_h · conj(Υ[h,l])  (used in Gram + RHS) ────────────
    auto A = Upsilon.conj().mul(w_d.unsqueeze(1));                    // [n_voxels, L]

    // ── Gram matrix G[L, L] = Aᵀ · Υ ────────────────────────────────────────
    auto G = mm(A.transpose(0, 1), Upsilon);                          // [L, L]

    // ── RHS R[K, L] = phi[k,h] · A[h,l]  (chunked) ──────────────────────────
    auto t_cd = timestamps.to(eScalarType::ComplexDouble);            // [K]
    auto R    = zeros({K, L}, opts_cd);

    for (i64 k0 = 0; k0 < K; k0 += chunk_size) {
        const i64 kB = std::min(chunk_size, K - k0);
        // phi_chunk[kB, n_voxels] = exp(-z_h · t_k)
        auto phi_chunk = (-z_d.unsqueeze(0)
                          .mul(t_cd.narrow(0, k0, kB).unsqueeze(1))).exp();  // [kB, n_voxels]
        R.narrow(0, k0, kB).copy_(mm(phi_chunk, A));                          // [kB, L]
    }

    // ── Ω: solve G @ Ωᵀ = Rᵀ  →  Ω = (G⁻¹ Rᵀ)ᵀ ────────────────────────────
    auto Omega = linalg_solve(G, R.transpose(0, 1))
                     .transpose(0, 1).contiguous();                   // [K, L]

    return PhiLowrankResult{
        Omega.to(eScalarType::ComplexFloat),
        Tensor{},  // S not produced by time segmentation
        Upsilon.to(eScalarType::ComplexFloat)
    };
}


// ---------------------------------------------------------------------------
// approx_signal
//
// S[c, k] ≈ Σ_l  Omega[k,l] · ( Σ_b Upsilon[b,l] · agg[c,b,k] )
// where agg[b,k] = Σ_{j: voxel_to_feature[j]=b}  mag[j]·coil[c,j]·exp(-2πi·k·r[j])
//
// coords_masked in normalized units [-0.5, 0.5], k_traj in cycles/FOV.
// For warped operators, pass q-space coords instead of r-space coords.
// ---------------------------------------------------------------------------

export Tensor approx_signal(
    const Tensor& Omega,
    const Tensor& Upsilon,
    const Tensor& mag_masked,
    const Tensor& coilmaps_masked,
    const Tensor& coords_masked,
    const Tensor& k_traj,
    const Tensor& voxel_to_feature,
    i64 n_voxels,
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
                auto agg = zeros({n_voxels}, opts_c);
                agg.scatter_add_(0, voxel_to_feature, dft_phase.select(0, k).mul(image_m));

                // [L] = mm([1, n_voxels], [n_voxels, L]).squeeze(0)
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
