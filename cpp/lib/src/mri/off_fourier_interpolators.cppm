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
//   Count     — w_h = Σ_{n∈h} 1 (bin population count). Unlike L1Mass/
//               L2Energy, this needs NO PD/magnitude value — only the mask
//               and field maps, both known a priori from scanner geometry/
//               B0 map, never the reconstructed image itself. The other two
//               require ground-truth PD, which doesn't exist before
//               reconstruction in a real pipeline (mag is only available
//               here because this test driver has a synthetic reference).
export enum struct eBinWeighting : i32 {
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

export struct HistogramResult {
    Tensor mask_idx;          // [N_mask]    long  — flat voxel indices in image
    Tensor voxel_to_bin;      // [N_mask]    long  — bin index for each masked voxel
    i64    n_hist;
    Tensor z_map_hist;        // [n_hist]    ComplexFloat — weighted-mean z per bin
    Tensor conc_fields_hist;  // [Q, n_hist] float        — weighted-mean concomitant fields per bin
    Tensor conc_waveforms;    // [Q, nspokes, nsamps] float — stacked waveforms from ConcomitantBasis
    Tensor bin_weights;       // [n_hist]    float        — Σ ρ   per bin (L1 mass)
    Tensor bin_l2_energy;     // [n_hist]    float        — Σ ρ²  per bin (L2 energy)
    Tensor bin_count;         // [n_hist]    float        — Σ 1   per bin (population count, no PD needed)
};


// ---------------------------------------------------------------------------
// extract_histogram
//
// Bins masked voxels by their (z_real, z_imag, f_0, ..., f_{Q-1}) features.
// n_rate bins for each component of z_map, n_conc_bins per concomitant field.
// Weighted means use mag as weight. bin_weights = Σ_{j in bin} mag_j.
// concomitant may be empty (Q=0): only z_map features used.
// ---------------------------------------------------------------------------

export HistogramResult extract_histogram(
    const Tensor& mag_flat,
    const Tensor& z_map_flat,
    const std::vector<ConcomitantBasis>& concomitant,
    i64 n_rate,
    i64 n_conc_bins = 1
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
        if (q == 0) {
            nspokes = concomitant[q].waveform.size(0);
            nsamps  = concomitant[q].waveform.size(1);
        } else if (concomitant[q].waveform.size(0) != nspokes || concomitant[q].waveform.size(1) != nsamps) {
            throw std::invalid_argument("concomitant waveform spoke shape mismatch across terms");
        }
    }

    auto mask     = mag_flat.gt(Scalar(0.0f));
    auto mask_idx = arange(N, opts_l).masked_select(mask);
    const i64 N_mask = mask_idx.size(0);

    auto mag_m = mag_flat.masked_select(mask);

    auto select_masked = [&](const Tensor& x) {
        return x.masked_select(mask);
    };

    auto bin_feat = [&](const Tensor& x, i64 n) -> Tensor {
        const float lo_f   = x.min().item<f32>();
        const float hi_f   = x.max().item<f32>();
        const float range  = hi_f - lo_f;
        const float scale  = std::max({std::abs(lo_f), std::abs(hi_f), 1e-30f});

        // Collapse to a single bin when the range is dominated by floating-
        // point noise rather than real signal -- e.g. a channel disabled at
        // the physics level (b0_scale/conc_scale/gnl_scale=0) is supposed to
        // be exactly constant, but still passes through a full complex
        // FFT/NUFFT warp pipeline that's never literally identity at the
        // bit level, leaving ~1e-6 relative round-off. The old fixed 1e-12
        // ABSOLUTE epsilon (in the division denominator) couldn't catch
        // this -- noise sits orders of magnitude above that -- so min-max
        // normalization spread bit-noise across all n bins, multiplying out
        // into a huge, physically meaningless n_hist (seen: 420097 bins for
        // an all-effects-disabled sanity config that should need exactly 1).
        if (range < scale * 1e-4f)
            return zeros({x.size(0)}, TensorOptions(x.device(), eScalarType::Long));

        auto idx = x.sub(Scalar(lo_f)).div(Scalar(range + 1e-12f))
                    .mul(Scalar((f32)(n - 1)))
                    .to(eScalarType::Long);
        return clamp(idx, Scalar((i64)0), Scalar((i64)(n - 1)));
    };

    // Strides: z_real | z_imag | conc_0 | ... | conc_{Q-1}  (row-major)
    i64 conc_total = 1;
    for (i64 q = 0; q < Q; ++q) conc_total *= n_conc_bins;

    auto z_real_m = select_masked(z_map_flat.real());
    auto z_imag_m = select_masked(z_map_flat.imag());

    auto bin_flat = bin_feat(z_real_m, n_rate).mul(Scalar(n_rate * conc_total))
                     .add(bin_feat(z_imag_m, n_rate).mul(Scalar(conc_total)));

    i64 conc_stride = conc_total;
    for (i64 q = 0; q < Q; ++q) {
        conc_stride /= n_conc_bins;
        auto conc_m = select_masked(concomitant[q].spatial_map);
        bin_flat    = bin_flat.add(bin_feat(conc_m, n_conc_bins).mul(Scalar(conc_stride)));
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

    // Stack concomitant field centroids: [Q, n_hist]
    std::vector<Tensor> conc_cols;
    conc_cols.reserve(Q);
    for (i64 q = 0; q < Q; ++q)
        conc_cols.push_back(scatter_wmean_f(select_masked(concomitant[q].spatial_map)).unsqueeze(0));

    auto conc_fields_hist = Q > 0 ? cat(conc_cols, 0)
                                  : empty({0, n_hist}, opts_f);

    // Stack waveforms: [Q, nspokes, nsamps]
    std::vector<Tensor> wf_rows;
    wf_rows.reserve(Q);
    for (i64 q = 0; q < Q; ++q)
        wf_rows.push_back(concomitant[q].waveform.unsqueeze(0));

    auto conc_waveforms = Q > 0 ? cat(wf_rows, 0)
                                : empty({0, 0, 0}, opts_f);

    auto bin_weights = zeros({n_hist}, opts_f);
    bin_weights.scatter_add_(0, voxel_to_bin, mag_m);

    auto bin_l2_energy = zeros({n_hist}, opts_f);
    bin_l2_energy.scatter_add_(0, voxel_to_bin, mag_m.mul(mag_m));

    // Population count per bin — Σ 1, NOT mag-derived. Needs only mask_idx
    // (mask/field-maps are known a priori) and voxel_to_bin (derived from
    // z_map/concomitant fields, also known a priori) — no PD/mag value
    // anywhere in this computation, unlike L1Mass/L2Energy which need the
    // very thing reconstruction is solving for.
    auto bin_count = zeros({n_hist}, opts_f);
    bin_count.scatter_add_(0, voxel_to_bin, ones({N_mask}, opts_f));

    return HistogramResult{
        mask_idx, voxel_to_bin, n_hist,
        z_map_hist, conc_fields_hist, conc_waveforms,
        bin_weights, bin_l2_energy, bin_count
    };
}

// ---------------------------------------------------------------------------
// extract_features_no_histogram
//
// Same output shape as extract_histogram (a HistogramResult), but WITHOUT
// any value-based binning: every masked voxel is its own "bin"
// (n_hist == N_mask, voxel_to_bin == identity). No bin_feat/unique_with_
// inverse/scatter_add_ anywhere — z_map_hist/conc_fields_hist are the exact
// per-voxel feature values, not weighted-mean bin centroids.
//
// This exists because value-based histogramming has an unbounded cost
// blowup: n_hist scales as a PRODUCT of per-axis bin resolutions
// (n_rate^2 * n_conc_bins^Q), independent of how many voxels actually exist,
// and once a real (non-degenerate) T2/B0 map is in play this routinely
// exceeds N_mask itself -- at that point histogramming is pure overhead, not
// compression. Going through every voxel exactly is the trivial upper bound
// on accuracy (zero quantization error) AND on cost (n_hist == N_mask,
// never more) -- downstream (make_phi_operator/phi_lowrank) is unchanged,
// since it only consumes HistogramResult's fields generically.
// ---------------------------------------------------------------------------

export HistogramResult extract_features_no_histogram(
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
        if (q == 0) { nspokes = concomitant[q].waveform.size(0); nsamps = concomitant[q].waveform.size(1); }
        else if (concomitant[q].waveform.size(0) != nspokes || concomitant[q].waveform.size(1) != nsamps)
            throw std::invalid_argument("concomitant waveform spoke shape mismatch across terms");
    }

    auto mask     = mag_flat.gt(Scalar(0.0f));
    auto mask_idx = arange(N, opts_l).masked_select(mask);
    const i64 n_hist = mask_idx.size(0);   // one "bin" per masked voxel, exactly

    auto select_masked = [&](const Tensor& x) { return x.masked_select(mask); };

    auto mag_m       = select_masked(mag_flat);
    auto z_map_hist  = select_masked(z_map_flat).contiguous();   // [n_hist] ComplexFloat, exact per-voxel z

    std::vector<Tensor> conc_cols;
    conc_cols.reserve(Q);
    for (i64 q = 0; q < Q; ++q)
        conc_cols.push_back(select_masked(concomitant[q].spatial_map).unsqueeze(0));
    auto conc_fields_hist = Q > 0 ? cat(conc_cols, 0) : empty({0, n_hist}, opts_f);

    std::vector<Tensor> wf_rows;
    wf_rows.reserve(Q);
    for (i64 q = 0; q < Q; ++q) wf_rows.push_back(concomitant[q].waveform.unsqueeze(0));
    auto conc_waveforms = Q > 0 ? cat(wf_rows, 0) : empty({0, 0, 0}, opts_f);

    auto voxel_to_bin  = arange(n_hist, opts_l);   // identity: bin h == masked voxel h
    auto bin_weights   = mag_m.contiguous();
    auto bin_l2_energy = mag_m.mul(mag_m).contiguous();
    auto bin_count     = ones({n_hist}, opts_f);

    return HistogramResult{
        mask_idx, voxel_to_bin, n_hist,
        z_map_hist, conc_fields_hist, conc_waveforms,
        bin_weights, bin_l2_energy, bin_count
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
    linalg::LinearOperator op;  // [K_sub, n_hist] — operates on subsampled time samples
    SpokeSubsampleInfo      spoke_info;
};


// ---------------------------------------------------------------------------
// make_phi_operator
//
// Builds the [K_sub, n_hist] phi operator from HistogramResult:
//   phi[k,h] = exp(-z_h · t_k) · Π_q exp(i · conc_fields_hist[q,h] · conc_waveforms[q,k])
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
    const HistogramResult& hist,
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
    const i64    N       = hist.n_hist;
    const i64    Q       = hist.conc_fields_hist.size(0);
    const Device device  = hist.z_map_hist.device();

    if (Q > 0 && (hist.conc_waveforms.size(0) != Q
               || hist.conc_waveforms.size(1) != nspokes
               || hist.conc_waveforms.size(2) != nsamps))
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
        ? hist.conc_waveforms.index_select(2, sub_idx).reshape({Q, K_sub})
        : empty({0, K_sub}, TensorOptions(device, eScalarType::Float));

    const Scalar pos_i_f = Scalar(std::complex<f32>(0.0f, 1.0f));

    const Tensor z_map_hist       = hist.z_map_hist;
    const Tensor conc_fields_hist = hist.conc_fields_hist;

    // ComplexFloat — see function doc for why this doesn't reintroduce the
    // round-trip error the original ComplexDouble build avoided.
    auto build_phi = [=](i64 k0, i64 kB) -> Tensor {
        auto t_c  = timestamps_sub.narrow(0, k0, kB).to(eScalarType::ComplexFloat);
        auto zm_f = z_map_hist.to(eScalarType::ComplexFloat);
        auto phi  = (zm_f.unsqueeze(0).mul(-t_c.unsqueeze(1))).exp();  // [kB, n_hist]
        for (i64 q = 0; q < Q; ++q) {
            phi = phi.mul(
                conc_fields_hist.select(0, q).to(eScalarType::ComplexFloat).unsqueeze(0)
                    .mul(conc_waveforms_sub.select(0, q).narrow(0, k0, kB)
                             .to(eScalarType::ComplexFloat).unsqueeze(1))
                    .mul(pos_i_f).exp()
            );
        }
        return phi;  // [kB, n_hist] ComplexFloat
    };

    auto mv_fn = [=](const Tensor& x) -> Tensor {
        auto result = zeros({K_sub}, TensorOptions(device, eScalarType::ComplexDouble));
        auto x_f    = x.to(eScalarType::ComplexFloat);
        for (i64 k0 = 0; k0 < K_sub; k0 += chunk_size) {
            const i64 kB = std::min(chunk_size, K_sub - k0);
            result.narrow(0, k0, kB).copy_(
                mv(build_phi(k0, kB), x_f).to(eScalarType::ComplexDouble));
        }
        return result;
    };

    auto rmv_fn = [=](const Tensor& y) -> Tensor {
        auto result = zeros({N}, TensorOptions(device, eScalarType::ComplexDouble));
        auto y_f    = y.to(eScalarType::ComplexFloat);
        for (i64 k0 = 0; k0 < K_sub; k0 += chunk_size) {
            const i64 kB = std::min(chunk_size, K_sub - k0);
            auto chunk = mv(build_phi(k0, kB).conj().transpose(0, 1), y_f.narrow(0, k0, kB));
            result += chunk.to(eScalarType::ComplexDouble);
        }
        return result;
    };

    linalg::LinearOperator op(K_sub, N,
        std::move(mv_fn), std::move(rmv_fn),
        eScalarType::ComplexDouble, device);

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
// spatial/bin basis, is time-independent and needs no reconstruction).
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
        case eBinWeighting::Count:
            return hist.bin_count;
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

    if (L > std::min(K, N))
        throw std::invalid_argument(
            "phi_lowrank: requested L=" + std::to_string(L) +
            " exceeds operator rank bound min(K,N)=" + std::to_string(std::min(K, N)) +
            " — n_hist should never be this small (no off-resonance/concomitant "
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
        case eBinWeighting::Count:
            w = hist.bin_count; break;
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
// where agg[b,k] = Σ_{j: bin[j]=b}  mag[j]·coil[c,j]·exp(-2πi·k·r[j])
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
