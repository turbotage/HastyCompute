#include <numbers>
#include <cmath>

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_linalg_mod;
import hasty_fft_mod;
import hasty_io_mod;
import hasty_io_mod_nifti;
import hasty_viz_mod;
import hasty_server_mod;
import hasty_python_mod;
import hasty_mri_mod;

// ── Problem ───────────────────────────────────────────────────────────────────
//
// Three physically distinct non-Fourier effects, kept as separate channel
// groups so any combination of them can be toggled independently both in the
// exact reference (forward_exact) and in the approximation pipeline:
//
//   off-resonance — rate_map (z_map), handled by extract_histogram/phi SVD.
//   GNL (separable) — a single axis-aligned term whose waveform is
//     proportional to k_axis(t); absorbed into a CoordinateWarp q = u(r),
//     NOT part of the histogram/SVD.
//   concomitant (non-separable) — terms whose waveform comes from gradient
//     products (NOT proportional to any k_axis(t)); handled by
//     extract_histogram/phi SVD same as off-resonance.
struct Problem {
    hasty::Tensor mag;                 // [nx,ny,nz] float
    hasty::Tensor rate_map;             // [nx,ny,nz] complex — off-resonance
    hasty::Tensor sensitivity_maps;     // [C,nx,ny,nz] complex

    // All per-sample data (trajectory, timestamps, waveforms) is stored
    // spoke-shaped [nspokes, nsamps, ...] — the natural unit gradients are
    // played out in, and the shape make_phi_operator's subsampling needs.
    // Flat [K, ...] views are taken on demand via .reshape() (a no-op view
    // for these contiguous tensors), never stored separately.
    hasty::Tensor k_traj_spoke;         // [nspokes,nsamps,3] float
    hasty::Tensor timestamps_spoke;     // [nspokes,nsamps] float
    hasty::i64    nspokes, nsamps;

    // GNL — Qg channels, one per axis (x²,y²,z²). field_fn (built at warp-
    // construction time in run_all_combos, not stored here) evaluates each
    // channel's value+gradient analytically at arbitrary positions —
    // CoordinateWarp's Newton inversion needs that, not a discrete array.
    hasty::i64    Qg;
    hasty::Tensor gnl_basis_vol;          // [Qg,nx,ny,nz] float — for forward_exact (discrete, fixed r-grid)
    hasty::Tensor gnl_waveform_spoke;     // [Qg,nspokes,nsamps] float — for forward_exact
    std::vector<hasty::i64> gnl_axes;     // per-channel axis (0=x,1=y,2=z)
    std::vector<float>      gnl_c_phys;   // per-channel coupling coefficient
    std::vector<float>      gnl_dx_phys;  // per-channel R-GRID voxel size along that axis [m]

    // Concomitant — Qc non-separable channels.
    hasty::i64    Qc;
    hasty::Tensor conc_basis_vol;        // [Qc,nx,ny,nz] float — for forward_exact
    hasty::Tensor conc_field_flat;       // [Qc,N] float        — spatial map, for extract_histogram
    hasty::Tensor conc_waveform_spoke;   // [Qc,nspokes,nsamps] float — for ConcomitantBasis / forward_exact

    hasty::Tensor mag_flat;              // [N] float — masked PD (0 outside histogram mask)
    hasty::Tensor pd_flat;               // [N] float — raw PD, no morphological mask applied
    hasty::Tensor z_map_flat;            // [N] complex

    hasty::i64 K, N, C;

    hasty::Tensor k_traj_flat()      const { return k_traj_spoke.reshape({K, 3}); }
    hasty::Tensor timestamps_flat()  const { return timestamps_spoke.reshape({K}); }
    hasty::Tensor gnl_waveform_flat()  const { return gnl_waveform_spoke.reshape({Qg, K}); }
    hasty::Tensor conc_waveform_flat() const { return conc_waveform_spoke.reshape({Qc, K}); }
};

// Minimal stopwatch for phase-by-phase progress prints -- always paired with
// std::flush at the call site since stdout fully buffers when redirected to
// a log file (no TTY), so without flushing nothing appears until exit.
struct Stopwatch {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

static bool cuda_available()
{
    try {
        hasty::empty({1}, hasty::TensorOptions{
            hasty::Device{hasty::eDeviceType::CUDA, 0}, hasty::eScalarType::Float});
        return true;
    } catch (...) { return false; }
}

// ── Real-data problem from NIfTI-derived tensors ─────────────────────────────
//
// pd_vol    [nx,ny,nz] float  — proton density magnetization (mask already applied)
// b0_hz_vol [nx,ny,nz] float  — B0 field map in Hz
// mask_vol  [nx,ny,nz] bool   — nonzero = include in histogram
// pixdim_*  mm (from NIfTI header pixdim[1..3])
// GNL (y²) and concomitant (z², x·y, x·z) fields are synthetic, computed from
// physical voxel positions.

static Problem make_problem_real(
    const hasty::Tensor& pd_vol,
    const hasty::Tensor& b0_hz_vol,
    const hasty::Tensor& mask_vol,
    float pixdim_x_mm, float pixdim_y_mm, float pixdim_z_mm,
    hasty::i64 n_spokes, hasty::i64 n_samp,
    hasty::Device dev,
    float b0_scale   = 1.0f,  // scale B0 field
    float te_start_s = 0.0f,  // readout start time (s)
    float conc_scale = 1.0f,  // concomitant phase scale (1.0 → physical magnitude at B0_tesla)
    float gnl_scale  = 1.0f,  // GNL (warp channel) amplitude scale (1.0 → default c_y2)
    float B0_tesla   = 7.0f)  // main field strength — concomitant fields scale as 1/B0
{
    using namespace hasty;
    using namespace std::numbers;

    const float gamma = 2.0f * (float)pi_v<f64> * 42.577e6f;
    const float dt    = 4e-6f;

    const i64 nx = pd_vol.size(0);
    const i64 ny = pd_vol.size(1);
    const i64 nz = pd_vol.size(2);
    const i64 N  = nx * ny * nz;
    const i64 K  = n_spokes * n_samp;
    const i64 Qc = 4;  // field_z2, field_radial, field_xz, field_yz — concomitant (non-separable)
    const i64 C  = 1;

    const float dx_m  = pixdim_x_mm * 1e-3f;
    const float dy_m  = pixdim_y_mm * 1e-3f;
    const float dz_m  = pixdim_z_mm * 1e-3f;
    const float FOV_x = (float)nx * dx_m;
    const float FOV_y = (float)ny * dy_m;
    const float FOV_z = (float)nz * dz_m;
    const float dx_min      = std::min({dx_m, dy_m, dz_m});
    const float k_max_rad_m = (float)pi_v<f64> / dx_min;

    const TensorOptions opts_f = TensorOptions(dev, eScalarType::Float);
    const TensorOptions opts_c = TensorOptions(dev, eScalarType::ComplexFloat);
    const TensorOptions opts_l = TensorOptions(dev, eScalarType::Long);

    // Index decomposition uses float64 to avoid float32 precision loss for N > 2^24.
    auto n_idx   = arange(N, opts_l);
    auto n_idx_d = n_idx.to(eScalarType::Double);
    auto ix    = n_idx_d.div(Scalar((f64)(ny * nz))).to(eScalarType::Long);
    auto iy    = n_idx_d.div(Scalar((f64)nz)).to(eScalarType::Long).remainder(Scalar((i64)ny));
    auto iz    = n_idx.remainder(Scalar((i64)nz));

    auto grid_x = arange(nx, opts_f).add(Scalar(0.5f)).div(Scalar((f32)nx)).sub(Scalar(0.5f));
    auto grid_y = arange(ny, opts_f).add(Scalar(0.5f)).div(Scalar((f32)ny)).sub(Scalar(0.5f));
    auto grid_z = arange(nz, opts_f).add(Scalar(0.5f)).div(Scalar((f32)nz)).sub(Scalar(0.5f));

    auto rx = grid_x.index_select(0, ix);
    auto ry = grid_y.index_select(0, iy);
    auto rz = grid_z.index_select(0, iz);

    auto phys_x = rx.mul(Scalar(FOV_x));
    auto phys_y = ry.mul(Scalar(FOV_y));
    auto phys_z = rz.mul(Scalar(FOV_z));

    auto mask_f   = mask_vol.to(dev).to(eScalarType::Float).flatten();
    auto pd_flat  = pd_vol.to(dev).to(eScalarType::Float).flatten();
    auto mag_flat = pd_flat.mul(mask_f);
    auto mag_3d   = mag_flat.reshape({nx, ny, nz});

    auto mask_bool = mask_vol.to(dev).to(eScalarType::Bool).flatten();
    auto b0_flat   = b0_hz_vol.to(dev).to(eScalarType::Float).flatten()
                        .mul(Scalar(b0_scale));
    auto z_map_flat = view_as_complex(
        stack({zeros({N}, opts_f),
               b0_flat.mul(Scalar(2.0f * (float)pi_v<f64>))}, 1).contiguous());
    auto rate_map   = z_map_flat.reshape({nx, ny, nz});

    auto sensitivity_maps = ones({C, nx, ny, nz}, opts_c);

    // Concomitant fields: standard 2nd-order Maxwell concomitant terms for a
    // z-main-field system (Bernstein/King/Zhou, Handbook of MRI Pulse
    // Sequences §10.2; Norris & Hutchison 1990). For gradients Gx,Gy,Gz:
    //   B_conc = (Gx²+Gy²)/(8B0)·z² + Gz²/(8B0)·(x²+y²)
    //            + Gx·Gz/(2B0)·xz   + Gy·Gz/(2B0)·yz
    // Note there is no x·y term in the standard theory — it only ever
    // appears as Gx² or Gy² individually, never the Gx·Gy cross product.
    auto field_z2     = phys_z.mul(phys_z);
    auto field_radial  = phys_x.mul(phys_x).add(phys_y.mul(phys_y));   // x²+y²
    auto field_xz      = phys_x.mul(phys_z);
    auto field_yz      = phys_y.mul(phys_z);

    // GNL (separable) channels: one per axis (x²,y²,z²), each axis-aligned —
    // u_x = x + c_x·x², u_y = y + c_y·y², u_z = z + c_z·z². Each is
    // independently separable into the warp (its waveform is proportional to
    // that axis's own k(t), built below from k_traj). Coefficients all use
    // the same c_d = gnl_scale·0.03/FOV_d form so every axis gets the same
    // relative Jacobian perturbation bound (~0.03·gnl_scale), regardless of
    // FOV extent along that axis.
    // gnl_basis_vol stays a discrete per-voxel array — forward_exact needs a
    // fixed r-grid array (it's the exact, unwarped reference). CoordinateWarp
    // itself no longer takes these as arrays — its field_fn (built in
    // run_all_combos) evaluates the same x²/y²/z² formulas analytically at
    // arbitrary (off-grid) positions instead, see CoordinateWarp::FieldFn doc.
    auto gnl_field_x2 = phys_x.mul(phys_x);
    auto gnl_field_y2 = phys_y.mul(phys_y);
    auto gnl_field_z2 = phys_z.mul(phys_z);
    auto gnl_field_x2_grad = phys_x.mul(Scalar(2.0f));
    auto gnl_field_y2_grad = phys_y.mul(Scalar(2.0f));
    auto gnl_field_z2_grad = phys_z.mul(Scalar(2.0f));
    const float c_x2 = gnl_scale * 0.03f / FOV_x;
    const float c_y2 = gnl_scale * 0.03f / FOV_y;
    const float c_z2 = gnl_scale * 0.03f / FOV_z;
    const i64   Qg   = 3;

    auto conc_basis_vol = stack({field_z2.reshape({nx, ny, nz}),
                                  field_radial.reshape({nx, ny, nz}),
                                  field_xz.reshape({nx, ny, nz}),
                                  field_yz.reshape({nx, ny, nz})}, 0);
    auto conc_field_flat = stack({field_z2, field_radial, field_xz, field_yz}, 0);

    auto gnl_basis_vol = stack({gnl_field_x2.reshape({nx, ny, nz}),
                                 gnl_field_y2.reshape({nx, ny, nz}),
                                 gnl_field_z2.reshape({nx, ny, nz})}, 0);
    std::vector<i64>   gnl_axes_v    = {0, 1, 2};
    std::vector<float> gnl_c_phys_v  = {c_x2, c_y2, c_z2};
    std::vector<float> gnl_dx_phys_v = {dx_m, dy_m, dz_m};

    // Koosh-ball trajectory
    std::vector<float> dirs(n_spokes * 3);
    for (i64 s = 0; s < n_spokes; ++s) {
        float f  = (float)s;
        float th = std::acos(1.0f - 2.0f * f / (float)n_spokes);
        float ph = f * (float)pi_v<f64> * (3.0f - std::sqrt(5.0f));
        dirs[s*3+0] = std::sin(th) * std::cos(ph);
        dirs[s*3+1] = std::sin(th) * std::sin(ph);
        dirs[s*3+2] = std::cos(th);
    }

    std::vector<float> k_data(K * 3), t_data(K);
    for (i64 s = 0; s < n_spokes; ++s) {
        for (i64 j = 0; j < n_samp; ++j) {
            float k_r = (-k_max_rad_m) + 2.0f * k_max_rad_m * (float)j / (float)(n_samp - 1);
            i64   idx = s * n_samp + j;
            k_data[idx*3+0] = dirs[s*3+0] * k_r * dx_m;
            k_data[idx*3+1] = dirs[s*3+1] * k_r * dy_m;
            k_data[idx*3+2] = dirs[s*3+2] * k_r * dz_m;
            t_data[idx]     = te_start_s + (float)j * dt;
        }
    }
    auto cpu = Device{eDeviceType::CPU};
    auto k_traj_flat_local = Tensor::from_blob(k_data.data(), {K, 3}, eScalarType::Float, cpu).clone().to(dev);
    auto timestamps_flat_local = Tensor::from_blob(t_data.data(), {K}, eScalarType::Float, cpu).clone().to(dev);
    auto k_traj_spoke    = k_traj_flat_local.reshape({n_spokes, n_samp, 3}).contiguous();
    auto timestamps_spoke = timestamps_flat_local.reshape({n_spokes, n_samp}).contiguous();

    // Concomitant waveforms, derived from the ACTUAL readout gradient (not an
    // invented pulse): this Koosh-ball spoke is a single linear k-space ramp,
    // so G(t) = (1/γ)·dk/dt is CONSTANT in magnitude and direction throughout
    // each spoke — Gx[s],Gy[s],Gz[s] below, direction=dirs[s], magnitude
    // shared across all spokes (same k_max_rad_m, same ramp duration).
    //
    // Phase resets at the start of every spoke (t_local = j*dt, not
    // cumulative across spokes) — each readout has its own gradient ramp
    // from the preceding RF excitation, so there's no physical reason for
    // concomitant phase to carry over spoke-to-spoke. Since G is constant
    // per spoke, the within-spoke integral ∫coupling dt is just a linear
    // ramp coupling_q(s)·(j·dt) — no cumsum needed.
    const float G_radial_slope = 2.0f * k_max_rad_m / ((float)(n_samp - 1) * dt * gamma);  // [T/m]

    std::vector<float> alpha_z2_v(K), alpha_radial_v(K), alpha_xz_v(K), alpha_yz_v(K);
    for (i64 s = 0; s < n_spokes; ++s) {
        const float Gx = G_radial_slope * dirs[s*3+0];
        const float Gy = G_radial_slope * dirs[s*3+1];
        const float Gz = G_radial_slope * dirs[s*3+2];

        const float coupling_z2     = (Gx*Gx + Gy*Gy) / (8.0f * B0_tesla);
        const float coupling_radial = (Gz*Gz)         / (8.0f * B0_tesla);
        const float coupling_xz     = (Gx*Gz)         / (2.0f * B0_tesla);
        const float coupling_yz     = (Gy*Gz)         / (2.0f * B0_tesla);

        for (i64 j = 0; j < n_samp; ++j) {
            const float t_local = (float)j * dt;
            const i64   idx     = s * n_samp + j;
            alpha_z2_v[idx]     = -gamma * conc_scale * coupling_z2     * t_local;
            alpha_radial_v[idx] = -gamma * conc_scale * coupling_radial * t_local;
            alpha_xz_v[idx]     = -gamma * conc_scale * coupling_xz     * t_local;
            alpha_yz_v[idx]     = -gamma * conc_scale * coupling_yz     * t_local;
        }
    }
    auto alpha_z2     = Tensor::from_blob(alpha_z2_v.data(),     {K}, eScalarType::Float, cpu).clone().to(dev);
    auto alpha_radial = Tensor::from_blob(alpha_radial_v.data(), {K}, eScalarType::Float, cpu).clone().to(dev);
    auto alpha_xz     = Tensor::from_blob(alpha_xz_v.data(),     {K}, eScalarType::Float, cpu).clone().to(dev);
    auto alpha_yz     = Tensor::from_blob(alpha_yz_v.data(),     {K}, eScalarType::Float, cpu).clone().to(dev);

    auto conc_waveform_spoke = stack({alpha_z2, alpha_radial, alpha_xz, alpha_yz}, 0)  // [4,K]
                                   .reshape({Qc, n_spokes, n_samp}).contiguous();

    // GNL waveforms: alpha_d(t) = -c_d * k_d(t) / dx_d, proportional to k_d(t)
    // by construction — this is what makes each term separable into the warp
    // (one per axis: field_d2(r)*alpha_d(t) == -k_d(t)*(c_d*d²), matching
    // k(t)·(u(r)-r) for u_d = d + c_d·d²).
    auto alpha_x2 = k_traj_flat_local.select(1, 0).div(Scalar(dx_m)).mul(Scalar(-c_x2)).contiguous();
    auto alpha_y2 = k_traj_flat_local.select(1, 1).div(Scalar(dy_m)).mul(Scalar(-c_y2)).contiguous();
    auto alpha_z2_gnl = k_traj_flat_local.select(1, 2).div(Scalar(dz_m)).mul(Scalar(-c_z2)).contiguous();
    auto gnl_waveform_spoke = stack({alpha_x2, alpha_y2, alpha_z2_gnl}, 0)
                                  .reshape({Qg, n_spokes, n_samp}).contiguous();

    // ── Physics diagnostics ──────────────────────────────────────────────────
    {
        const float T_readout = (float)(n_samp - 1) * dt;
        const float pi_f      = (float)pi_v<f64>;

        auto b0_in_mask  = b0_flat.masked_select(mask_bool);
        float b0_maxabs  = std::max(std::abs(b0_in_mask.min().item<float>()),
                                     std::abs(b0_in_mask.max().item<float>()));
        float b0_max_phase = b0_maxabs * 2.0f * pi_f * T_readout;

        float max_phase_z2     = field_z2.abs().max().item<float>()     * alpha_z2.abs().max().item<float>();
        float max_phase_radial = field_radial.abs().max().item<float>() * alpha_radial.abs().max().item<float>();
        float max_phase_xz     = field_xz.abs().max().item<float>()     * alpha_xz.abs().max().item<float>();
        float max_phase_yz     = field_yz.abs().max().item<float>()     * alpha_yz.abs().max().item<float>();
        float max_phase_x2gnl  = gnl_field_x2.abs().max().item<float>() * alpha_x2.abs().max().item<float>();
        float max_phase_y2gnl  = gnl_field_y2.abs().max().item<float>() * alpha_y2.abs().max().item<float>();
        float max_phase_z2gnl  = gnl_field_z2.abs().max().item<float>() * alpha_z2_gnl.abs().max().item<float>();

        auto jac_x_in_mask = gnl_field_x2_grad.masked_select(mask_bool).mul(Scalar(c_x2)).add(Scalar(1.0f));
        auto jac_y_in_mask = gnl_field_y2_grad.masked_select(mask_bool).mul(Scalar(c_y2)).add(Scalar(1.0f));
        auto jac_z_in_mask = gnl_field_z2_grad.masked_select(mask_bool).mul(Scalar(c_z2)).add(Scalar(1.0f));
        float jac_min = std::min({jac_x_in_mask.min().item<float>(),
                                   jac_y_in_mask.min().item<float>(),
                                   jac_z_in_mask.min().item<float>()});
        float jac_max = std::max({jac_x_in_mask.max().item<float>(),
                                   jac_y_in_mask.max().item<float>(),
                                   jac_z_in_mask.max().item<float>()});

        std::cout << "\n  ── problem physics ──────────────────────────────────────\n"
                  << "  volume: " << nx << "×" << ny << "×" << nz << "\n"
                  << "  trajectory: " << n_spokes << " spokes × " << n_samp << " samp  K=" << K << "\n"
                  << "  B0=" << B0_tesla << "T\n"
                  << "  off-res: max phase over readout=" << std::fixed << std::setprecision(2)
                  << b0_max_phase << " rad\n"
                  << "  concomitant: max phase  z²=" << max_phase_z2
                  << " (x²+y²)=" << max_phase_radial
                  << " xz=" << max_phase_xz << " yz=" << max_phase_yz << " rad\n"
                  << "  GNL: max phase  x²=" << std::setprecision(3) << max_phase_x2gnl
                  << " y²=" << max_phase_y2gnl << " z²=" << max_phase_z2gnl
                  << " rad   det(J_u) ∈ [" << jac_min << ", " << jac_max << "]  "
                  << ((jac_min > 0.0f) ? "OK" : "VIOLATED") << "\n"
                  << "  ─────────────────────────────────────────────────────────\n\n";
    }

    return Problem{
        mag_3d, rate_map, sensitivity_maps,
        k_traj_spoke, timestamps_spoke, n_spokes, n_samp,
        Qg, gnl_basis_vol, gnl_waveform_spoke,
        gnl_axes_v, gnl_c_phys_v, gnl_dx_phys_v,
        Qc, conc_basis_vol, conc_field_flat, conc_waveform_spoke,
        mag_flat, pd_flat, z_map_flat,
        K, N, C
    };
}

// ── Exact reference signal, parameterised by which effects are active ───────

static hasty::Tensor exact_signal_combo(
    const Problem& prob,
    const hasty::Tensor& sub_idx,   // [M] long
    bool use_offres, bool use_gnl, bool use_conc)
{
    using namespace hasty;

    auto k_b = prob.k_traj_flat().index_select(0, sub_idx);
    auto t_b = prob.timestamps_flat().index_select(0, sub_idx);

    std::vector<Tensor> wf_parts, basis_parts;
    if (use_gnl) {
        wf_parts.push_back(prob.gnl_waveform_flat().index_select(1, sub_idx));
        basis_parts.push_back(prob.gnl_basis_vol);
    }
    if (use_conc) {
        wf_parts.push_back(prob.conc_waveform_flat().index_select(1, sub_idx));
        basis_parts.push_back(prob.conc_basis_vol);
    }

    const bool any_nl = !wf_parts.empty();
    Tensor wf, basis;
    if (any_nl) {
        wf    = cat(wf_parts, 0);
        basis = cat(basis_parts, 0);
    } else {
        wf    = empty({0, sub_idx.size(0)}, TensorOptions(prob.mag.device(), eScalarType::Float));
        basis = empty({0, prob.mag.size(0), prob.mag.size(1), prob.mag.size(2)},
                       TensorOptions(prob.mag.device(), eScalarType::Float));
    }

    return mri::forward_exact(prob.mag, prob.sensitivity_maps, prob.rate_map,
                               t_b, k_b, wf, basis, use_offres, any_nl);
}

// ── Approximant: histogram + phi SVD ─────────────────────────────────────────
//
// There is exactly ONE approximation procedure per Problem: full off-resonance
// + ALL concomitant channels low-rank SVD, ALWAYS warped to q-space first via
// the GNL CoordinateWarp. No per-combo branching — "off" effects are
// controlled at Problem-construction time (TestConfig's b0_scale/conc_scale/
// gnl_scale, e.g. gnl_scale=0 makes the warp the identity), never by
// selectively zeroing inputs here. The 8 (off-res,GNL,concomitant) combos are
// a property of the EXACT reference only (exact_signal_combo) — they test how
// the single fixed approximation's error behaves as terms are removed from
// the ground truth, not 8 different approximations.
//
// The warp is applied to magnetization/z-map/concomitant maps ONCE, BEFORE
// histogramming — not after. extract_histogram/phi SVD then run natively on
// q-space data, so Upsilon/Omega ARE q-space objects already. Signal
// evaluation (approx_signal_nufft_combo) needs no warp call at all.

struct ApproxResult {
    hasty::Tensor Omega;        // [K, L] raw (not S-scaled)
    hasty::Tensor S;            // [L]
    hasty::Tensor Upsilon;      // [n_hist, L]
    hasty::mri::HistogramResult hist;
    hasty::Tensor mag_flat_eff; // [N_q] — magnetization warped to q-space
    std::vector<hasty::i64> q_shape; // q-grid shape Upsilon/mask_idx/voxel_to_bin/mag_flat_eff live on
    std::vector<float> oversample_factor; // per-axis resolution multiplier — k_traj rescale = 1/this
};

static ApproxResult build_approximant(
    const Problem& prob,
    hasty::i64 L,
    hasty::i64 subsample,
    hasty::mri::eBinWeighting weighting,
    const hasty::mri::CoordinateWarp& warp)
{
    using namespace hasty;
    using namespace hasty::mri;

    // Every field gets pure relabeling — no density correction baked into
    // the warp call itself (see CoordinateWarp class doc). ρ is the one
    // place the density correction (Jacobian + oversample) actually belongs
    // — applied explicitly, exactly once, here — since ρ(r)·φ(r) is a
    // single integrand evaluated at the same r-position, and the
    // substitution applies once to that whole product, not once per field.
    auto to_q_field = [&](const Tensor& flat, double tol) -> Tensor {
        return warp.warp_field_r_to_q(flat.reshape(warp.r_shape()), tol)
                   .reshape({warp.n_q_voxels()});
    };

    // warp_field_r_to_q always works in the complex domain internally (its
    // NUFFT pipeline is complex-only) -- mag is physically real (proton
    // density itself has no phase; complex behavior lives in z_map/rate_map
    // instead, NOT here), so take the real part right after warping, before
    // multiplying by the (real) density correction. extract_histogram below
    // requires mag_flat_eff to be Float, not ComplexFloat.
    Stopwatch sw;
    auto mag_flat_eff = to_q_field(prob.mag_flat, /*tol=*/1e-6).real().contiguous()
                             .mul(warp.density_correction_q().reshape({warp.n_q_voxels()}));
    std::cout << "  [build_approximant] mag warp: " << std::fixed << std::setprecision(3) << sw.elapsed() << "s\n" << std::flush;

    sw = Stopwatch();
    auto z_eff        = to_q_field(prob.z_map_flat, /*tol=*/1e-2);
    std::cout << "  [build_approximant] z warp:   " << sw.elapsed() << "s\n" << std::flush;

    sw = Stopwatch();
    std::vector<ConcomitantBasis> conc_list;
    conc_list.reserve(prob.Qc);
    for (i64 q = 0; q < prob.Qc; ++q)
        conc_list.push_back(ConcomitantBasis{
            to_q_field(prob.conc_field_flat.select(0, q).contiguous(), /*tol=*/1e-2),
            prob.conc_waveform_spoke.select(0, q).contiguous()
        });
    std::cout << "  [build_approximant] conc warp (x" << prob.Qc << "): " << sw.elapsed() << "s\n" << std::flush;

    // Histogramming removed entirely -- n_rate/n_conc_bins binning blows up
    // combinatorially (n_rate^2 * n_conc_bins^Qc, independent of voxel
    // count) and once a real T2/B0 map is in play this can exceed N_mask
    // itself, at which point it's pure overhead, not compression.
    // extract_features_no_histogram treats every masked voxel as its own
    // bin (n_hist == N_mask exactly, zero quantization error) -- downstream
    // (make_phi_operator/phi_lowrank) is unchanged, it only consumes
    // HistogramResult's fields generically.
    sw = Stopwatch();
    auto hist = extract_features_no_histogram(mag_flat_eff, z_eff, conc_list);
    std::cout << "  [build_approximant] extract_features_no_histogram: " << sw.elapsed() << "s  (n_hist=" << hist.n_hist << ")\n" << std::flush;

    // z_eff and conc_list's q-grid-sized spatial maps are consumed by
    // extract_histogram above (it only keeps the small per-bin histogram
    // tensors) — drop them before the SVD step instead of holding them
    // alive until this function returns.
    z_eff = Tensor();
    conc_list.clear();
    conc_list.shrink_to_fit();

    sw = Stopwatch();
    auto phi_res = make_phi_operator(hist, prob.timestamps_spoke, subsample);
    std::cout << "  [build_approximant] make_phi_operator: " << sw.elapsed() << "s\n" << std::flush;

    const i64 mpd_val = std::max((i64)8*L, (i64)60);
    const i64 ncv_val = L + mpd_val;
    sw = Stopwatch();
    auto weights = phi_lowrank_weights(weighting, hist, prob.timestamps_flat());
    auto plr = phi_lowrank(phi_res.op, L, weights, ncv_val, mpd_val);
    std::cout << "  [build_approximant] phi_lowrank (SVD/Lanczos, ncv=" << ncv_val << "): " << sw.elapsed() << "s\n" << std::flush;

    sw = Stopwatch();
    auto Omega_full = upsample_omega_spokes(plr.Omega, phi_res.spoke_info);  // [K, L]
    std::cout << "  [build_approximant] upsample_omega_spokes: " << sw.elapsed() << "s\n" << std::flush;

    auto q_shape_vec = std::vector<i64>(warp.q_shape().begin(), warp.q_shape().end());
    auto oversample_vec = std::vector<float>(warp.oversample_factor().begin(), warp.oversample_factor().end());
    return ApproxResult{Omega_full, plr.S, plr.Upsilon, std::move(hist), mag_flat_eff, q_shape_vec, oversample_vec};
}

// ── Approx signal via L forward NUFFTs ───────────────────────────────────────
//
// img_l[n] = Upsilon[bin(n), l] · ρ_eff[n]   (already in q-space, since the
// warp was applied upstream in build_approximant). The NUFFT plan/grid use
// q_shape (the grid Upsilon/mask_idx/mag_flat_eff actually live on), which
// may differ from the r-grid's resolution.
// S[k] = Σ_l Omega[k,l]·S_phi[l] · NUFFT(img_l, k_traj)[k]
// No CoordinateWarp call here at all — grid-agnostic, identical for every combo.

static hasty::Tensor approx_signal_nufft_combo(
    const Problem& prob,
    const hasty::Tensor& Omega_full,     // [K, L] raw
    const hasty::Tensor& S_phi,          // [L]
    const hasty::Tensor& Upsilon,        // [n_hist, L]
    const hasty::Tensor& voxel_to_bin,   // [N_mask] long
    const hasty::Tensor& mask_idx,       // [N_mask] long
    const hasty::Tensor& mag_flat_eff,   // [N_q]
    const std::vector<hasty::i64>& q_shape,
    const std::vector<float>& oversample_factor)
{
    using namespace hasty;
    using namespace hasty::fft;

    const i64    K   = prob.K;
    const i64    L   = Upsilon.size(1);
    const i64    nx  = q_shape[0], ny = q_shape[1], nz = q_shape[2];
    const i64    N   = nx * ny * nz;
    const Device dev = prob.mag.device();

    const TensorOptions opts_c{dev, eScalarType::ComplexFloat};
    const TensorOptions opts_f{dev, eScalarType::Float};

    auto Omega_s = Omega_full.mul(S_phi.unsqueeze(0));   // [K, L] — absorb singular values
    auto rho_m   = mag_flat_eff.index_select(0, mask_idx).to(eScalarType::ComplexFloat);

    // prob.k_traj is in r-grid pixel-domain convention (coord = k_phys·dx_R).
    // dx_Q = dx_R/oversample_factor (margin/padding alone never changes dx),
    // so coords need only rescale by 1/oversample_factor — pure padding
    // (oversample_factor=1) needs no rescale at all.
    auto ktraj_r = prob.k_traj_flat();
    std::vector<float> k_scale_v = {1.0f/oversample_factor[0], 1.0f/oversample_factor[1], 1.0f/oversample_factor[2]};
    auto k_scale = Tensor::from_blob(k_scale_v.data(), {3}, eScalarType::Float, Device{eDeviceType::CPU})
                       .clone().to(dev);
    auto ktraj  = ktraj_r.mul(k_scale.unsqueeze(0));
    auto coords = zeros({3, K}, opts_f);
    coords.select(0, 0).copy_(ktraj.select(1, 2));  // kz → iz dim
    coords.select(0, 1).copy_(ktraj.select(1, 1));  // ky → iy dim
    coords.select(0, 2).copy_(ktraj.select(1, 0));  // kx → ix dim
    coords = coords.contiguous();
    ktraj_r = Tensor();  // consumed building coords; not needed past this point
    ktraj   = Tensor();
    k_scale = Tensor();

    hasty::cuda::cuda_empty_cache();

    NufftOptions<cuda_t, f32, UTN> nufft_opts;
    nufft_opts.ntransf = 1;
    NufftPlan<cuda_t, f32, 3, UTN> plan({nx, ny, nz}, nufft_opts);
    plan.setpts(coords);
    coords = Tensor();  // setpts copies points into the plan's own buffers

    auto signal = zeros({K}, opts_c);
    auto F_l    = zeros({1, K}, opts_c).contiguous();

    for (i64 l = 0; l < L; ++l) {
        auto ups_vox = Upsilon.select(1, l).index_select(0, voxel_to_bin);
        auto src_l   = ups_vox.mul(rho_m);

        auto img_flat = zeros({N}, opts_c);
        img_flat.scatter_add_(0, mask_idx, src_l);
        auto img_l = img_flat.reshape({nx, ny, nz});

        plan.execute(img_l.contiguous().unsqueeze(0), F_l);
        signal.add_(Omega_s.select(1, l).mul(F_l.select(0, 0)));
    }

    return signal.unsqueeze(0);  // [1, K]
}

// ── Error / correlation metrics ──────────────────────────────────────────────

// Plain relative L2 error ||S_approx - S_ref|| / ||S_ref||. Computes the
// difference BEFORE norming (not via ne²+na²-2|dot|, which subtracts two
// large nearly-equal numbers in float32 and can cancel to a hard zero below
// the metric's own resolution floor — this form doesn't have that problem).
static float rel_err(const hasty::Tensor& S_ref, const hasty::Tensor& S_approx)
{
    float num = S_approx.sub(S_ref).norm().item<float>();
    float den = S_ref.norm().item<float>();
    return num / (den + 1e-30f);
}

static hasty::Tensor to_mag(const hasty::Tensor& S)
{
    return S.select(0, 0).abs().cpu().contiguous();
}

static hasty::Tensor to_phase(const hasty::Tensor& S)
{
    using namespace hasty;
    auto s1 = S.select(0, 0).cpu().contiguous();
    auto re = s1.real().contiguous();
    auto im = s1.imag().contiguous();
    auto rv = re.spanning_view();
    auto iv = im.spanning_view();
    i64  M  = rv.sizes[0];
    std::vector<float> ph(M);
    const float* rp = static_cast<const float*>(rv.data);
    const float* ip = static_cast<const float*>(iv.data);
    for (i64 i = 0; i < M; ++i) ph[i] = std::atan2(ip[i], rp[i]);
    return Tensor::from_blob(ph.data(), {M}, eScalarType::Float, Device{eDeviceType::CPU}).clone();
}

static float pearson(const hasty::Tensor& a, const hasty::Tensor& b)
{
    using namespace hasty;
    auto ad = a.to(eScalarType::Double);
    auto bd = b.to(eScalarType::Double);
    double n   = (double)a.size(0);
    double sa  = ad.sum().item<double>();
    double sb  = bd.sum().item<double>();
    double sab = ad.mul(bd).sum().item<double>();
    double sa2 = ad.mul(ad).sum().item<double>();
    double sb2 = bd.mul(bd).sum().item<double>();
    double num = n * sab - sa * sb;
    double den = std::sqrt(std::max(0.0, (n*sa2 - sa*sa) * (n*sb2 - sb*sb)));
    return (float)(num / (den + 1e-60));
}

static float circ_corr(const hasty::Tensor& phase_a, const hasty::Tensor& phase_b)
{
    using namespace hasty;
    auto diff = phase_a.sub(phase_b).contiguous();
    auto dv = diff.spanning_view();
    i64 M = dv.sizes[0];
    const float* d = static_cast<const float*>(dv.data);
    double sum_cos = 0.0;
    for (i64 i = 0; i < M; ++i) sum_cos += std::cos(d[i]);
    return (float)(sum_cos / M);
}

static float mean_rel_err(const hasty::Tensor& ref, const hasty::Tensor& approx)
{
    using namespace hasty;
    auto err = ref.sub(approx).abs().contiguous();
    auto rab = ref.abs().contiguous();
    const i64 N_m = err.size(0);
    const float* ep = static_cast<const float*>(err.spanning_view().data);
    const float* rp = static_cast<const float*>(rab.spanning_view().data);
    double sum = 0.0; int cnt = 0;
    for (i64 i = 0; i < N_m; ++i) {
        float r = rp[i] > 1e-30f ? ep[i] / rp[i] : 0.0f;
        if (std::isfinite(r)) { sum += r; ++cnt; }
    }
    return cnt ? (float)(sum / cnt) : 0.0f;
}

// ── Core-space CP-rank-P fit diagnostic ────────────────────────────────────────
//
// T[hj,hi,k] = G[hj,k]*conj(G[hi,k]) with G = Upsilon @ diag(S) @ Omega^T.
// Since Upsilon has orthonormal columns, CP-rank-P approximation of T is
// equivalent (same Frobenius error) to CP-rank-P approximation of the tiny
// core tensor M[l,l',k] = W[k,l]*conj(W[k,l']), W = Omega .* S  [K,L].
// This lets us evaluate ALS variants in milliseconds instead of minutes.

static hasty::Tensor cpcore_rsolve_trunc(const hasty::Tensor& B, const hasty::Tensor& G, double rel_eps = 1e-9)
{
    using namespace hasty;

    // Diagonal (Tikhonov) jitter — see nh_rsolve_herm in normal_toeplitz.cppm
    // for why: once this ALS diagnostic has nearly converged, G's eigenvalues
    // cluster near zero/near-duplicate, which can make LAPACK's syevd fail to
    // converge outright. Jitter far below rel_eps*d_max separates clustered
    // eigenvalues without perturbing the truncated-pseudo-inverse result.
    const i64    P   = G.size(0);
    const Device dev = G.device();
    const double g_scale = G.abs().max().item<double>();
    const double jitter  = g_scale * 1e-12 + 1e-300;
    auto diag_idx = arange(P, TensorOptions(dev, eScalarType::Long)).mul(Scalar((i64)(P + 1)));
    auto jitter_t = ones({P}, TensorOptions(dev, G.scalar_type())).mul(Scalar(jitter));
    auto G_reg    = G.reshape({P * P}).clone();
    G_reg.scatter_add_(0, diag_idx, jitter_t);
    G_reg = G_reg.reshape({P, P});

    auto eig = linalg_eigh(G_reg);
    double d_max    = eig.eigenvalues.abs().max().item<double>();
    double d_thresh = d_max * rel_eps + 1e-300;
    auto mask  = eig.eigenvalues.gt(Scalar(d_thresh)).to(eScalarType::Double);
    auto d_inv = (mask / eig.eigenvalues.clamp_min(d_thresh))
                     .to(eScalarType::ComplexDouble)
                     .unsqueeze(0);
    auto BV = mm(B, eig.eigenvectors);
    return mm(BV.mul(d_inv), eig.eigenvectors.conj().transpose(0,1).contiguous());
}

static double cpcore_rel_err(const hasty::Tensor& W,      // [K,L] ComplexDouble
                              const hasty::Tensor& Acore,  // [L,P]
                              const hasty::Tensor& Bcore,  // [L,P]
                              const hasty::Tensor& Lam)    // [K,P]
{
    using namespace hasty;
    auto row_norm2  = W.abs().pow(Scalar(2.0)).sum(1);            // [K] real
    double M_norm_sq = row_norm2.pow(Scalar(2.0)).sum().item<double>();

    auto F = mm(W, Acore.conj().contiguous());                    // [K,P]
    auto C = mm(W.conj().contiguous(), Bcore);                    // [K,P]
    double cross_re = Lam.mul(F.mul(C).conj()).sum().real().item<double>();

    auto gram_A = mm(Acore.conj().transpose(0,1).contiguous(), Acore);  // [P,P]
    auto gram_B = mm(Bcore.conj().transpose(0,1).contiguous(), Bcore);
    auto gram_L = mm(Lam  .conj().transpose(0,1).contiguous(), Lam);
    double Mhat_norm_sq = gram_L.conj().mul(gram_A.conj()).mul(gram_B).sum().real().item<double>();

    double err_sq = M_norm_sq - 2.0 * cross_re + Mhat_norm_sq;
    return std::sqrt(std::max(err_sq, 0.0)) / std::sqrt(std::max(M_norm_sq, 1e-300));
}

static void core_cp_diagnostic(const hasty::Tensor& Omega,   // [K,L]
                                const hasty::Tensor& S_phi,   // [L] Float
                                hasty::i64 p,
                                hasty::i64 n_iter,
                                double rel_eps)
{
    using namespace hasty;
    const i64 L = Omega.size(1);
    const i64 K = Omega.size(0);
    const Device dev = Omega.device();
    const auto cpu = Device{eDeviceType::CPU};

    // pair_ord below has only L*L entries — clamp defensively regardless of
    // what the caller computed p from.
    p = std::min(p, L*L);

    auto Sc = S_phi.to(eScalarType::Double).to(eScalarType::ComplexDouble);  // [L]
    auto W  = Omega.to(eScalarType::ComplexDouble).mul(Sc.unsqueeze(0)).contiguous();  // [K,L]

    auto Sd       = S_phi.to(eScalarType::Double);
    auto pair_ord = argsort(Sd.unsqueeze(1).mul(Sd.unsqueeze(0)).reshape({L*L}), 0, /*descending=*/true);

    std::vector<float> Acore_h(L*p, 0.f), Bcore_h(L*p, 0.f);
    std::vector<i64> l1s(p), l2s(p);
    {
        auto ord_cpu = pair_ord.cpu().contiguous();
        const i64* po = ord_cpu.const_data_ptr<i64>();
        for (i64 pi = 0; pi < p; ++pi) {
            i64 idx = po[pi];
            l1s[pi] = idx / L; l2s[pi] = idx % L;
            Acore_h[l1s[pi]*p + pi] = 1.0f;
            Bcore_h[l2s[pi]*p + pi] = 1.0f;
        }
    }
    auto Acore = Tensor::from_blob(Acore_h.data(), {L,p}, eScalarType::Float, cpu)
                    .clone().to(dev).to(eScalarType::ComplexDouble);
    auto Bcore = Tensor::from_blob(Bcore_h.data(), {L,p}, eScalarType::Float, cpu)
                    .clone().to(dev).to(eScalarType::ComplexDouble);

    auto Lam = zeros({K,p}, TensorOptions(dev, eScalarType::ComplexDouble));
    for (i64 pi = 0; pi < p; ++pi)
        Lam.select(1,pi).copy_(W.select(1,l1s[pi]).mul(W.select(1,l2s[pi]).conj()));

    std::cout << "\n  Core-space CP fit [L=" << L << ", P=" << p << ", rel_eps=" << rel_eps << "]\n"
              << "    init (SVD-pairs, no ALS): rel_err=" << std::scientific << std::setprecision(3)
              << cpcore_rel_err(W, Acore, Bcore, Lam) << "\n";

    for (i64 it = 0; it < n_iter; ++it) {
        auto F = mm(W, Acore.conj().contiguous());   // [K,P]
        auto C = mm(W.conj().contiguous(), Bcore);   // [K,P]

        auto gram_A = mm(Acore.conj().transpose(0,1).contiguous(), Acore);
        auto gram_B = mm(Bcore.conj().transpose(0,1).contiguous(), Bcore);
        Lam = cpcore_rsolve_trunc(F.mul(C), gram_A.conj().mul(gram_B), rel_eps);

        auto gram_Lam = mm(Lam.conj().transpose(0,1).contiguous(), Lam);

        Acore = cpcore_rsolve_trunc(
            mm(W.transpose(0,1).contiguous(), Lam.conj().mul(C)),
            gram_B.mul(gram_Lam.conj()), rel_eps);

        gram_A = mm(Acore.conj().transpose(0,1).contiguous(), Acore);

        auto F_new = mm(W, Acore.conj().contiguous());   // [K,P]
        Bcore = cpcore_rsolve_trunc(
            mm(W.transpose(0,1).contiguous(), Lam.mul(F_new.conj())),
            gram_A.mul(gram_Lam), rel_eps);

        gram_A = mm(Acore.conj().transpose(0,1).contiguous(), Acore);
        gram_B = mm(Bcore.conj().transpose(0,1).contiguous(), Bcore);
        F = mm(W, Acore.conj().contiguous());
        C = mm(W.conj().contiguous(), Bcore);
        auto Lam_eval = cpcore_rsolve_trunc(F.mul(C), gram_A.conj().mul(gram_B), rel_eps);

        std::cout << "    iter " << std::setw(2) << it
                  << ": rel_err=" << std::scientific << std::setprecision(3)
                  << cpcore_rel_err(W, Acore, Bcore, Lam_eval) << "\n";
    }
}

// ── TestConfig ────────────────────────────────────────────────────────────
//
// Bundles everything needed for one full test run: the physical-effect
// scaling factors (used at Problem-construction time, make_problem_real) and
// the approximant/SVD knobs (used by run_all_combos) — so the same driver
// can be invoked repeatedly with different settings.
struct TestConfig {
    std::string label = "default";

    // Problem construction (trajectory + physical-effect amplitude scaling)
    hasty::i64 n_spokes   = 500;
    hasty::i64 n_samp     = 800;
    float      te_start_s = 1e-3f;
    float      b0_scale   = 1.0f;   // off-resonance amplitude scale
    float      conc_scale = 1.0f;   // concomitant amplitude scale
    float      gnl_scale  = 1.0f;   // GNL (warp channel) amplitude scale
    float      B0_tesla   = 7.0f;   // main field strength — concomitant fields scale as 1/B0

    // Approximant / SVD knobs
    hasty::i64 L            = 8;
    hasty::i64 n_rate       = 14000;
    hasty::i64 n_conc_bins  = 2;
    hasty::i64 subsample    = 1;    // time subsampling stride within each spoke (1 = none)
    hasty::i64 P            = 8;
    hasty::i64 n_sub        = 300;  // # k-space points sampled for error metrics
    // q-grid sizing — margin (padding, no resolution change) and
    // oversample_factor (genuine resolution increase, ZIP-style) are
    // independent: q_shape[d] = ceil(oversample_factor·(r_shape[d] + 2·margin_voxels)).
    // margin_voxels: extra array bounds at the SAME voxel size as r-grid, just
    // enough to keep displaced voxels off the boundary (check via
    // CoordinateWarp::max_abs_displacement_pix()).
    hasty::i64 margin_voxels = 4;
    // oversample_factor: per-axis resolution multiplier (nullopt = 1, no
    // resolution change). >1 reduces multilinear scatter's discretization
    // error at the cost of a bigger q-grid and a NUFFT k-space rescale.
    hasty::Opt<float> oversample_factor = hasty::nullopt;
    // Explicit q-grid shape override, e.g. {320,320,320} — if set, used
    // directly instead of deriving from margin_voxels/oversample_factor (the
    // caller is then responsible for making it big enough).
    std::vector<hasty::i64> q_shape = {};
    hasty::mri::eBinWeighting weighting       = hasty::mri::eBinWeighting::L1Mass;
    bool                       run_normal_op_compare = true;

    // Diagnostic: bypass the SVD entirely, replacing Omega/S/Upsilon with the
    // exact trivial rank-1 operator (Omega=1, S=1, Upsilon=1 everywhere) that
    // phi truly equals when off-res/concomitant are negligible (z_h≈0,
    // conc waveform≈0 ⇒ phi[k,h]=1 for every h, independent of bin identity —
    // not just approximately low-rank, but EXACTLY constant). If this removes
    // the residual error, the bug is in the SVD/Lanczos stage, not the warp.
    bool debug_bypass_svd = false;
};

// ── run_all_combos ───────────────────────────────────────────────────────────
//
// Builds the ONE approximant for this Problem (full off-res + all
// concomitant low-rank SVD, always warped to q-space via GNL), then compares
// its signal prediction against the 2³=8 exact-reference combinations
// (off-resonance, GNL, concomitant independently included/excluded in
// forward_exact). This measures how the single fixed approximation's error
// behaves as terms are removed from the ground truth — it is NOT 8 separate
// approximations.
//
// Pass/fail is judged only on the full combo (all three terms present),
// since that's the actual target accuracy; the other 7 rows are diagnostic
// (showing each term's contribution to the signal).
//
// The naive-L² vs non-hermitian-CP-ALS normal-operator comparison is a
// self-consistency check of the CP rank-reduction (independent of any exact
// reference) and is run once here too, when enabled.

static int run_all_combos(const Problem& prob, const TestConfig& cfg)
{
    using namespace hasty;
    using namespace hasty::mri;

    std::cout << "\n===== TestConfig: " << cfg.label
              << "  (L=" << cfg.L << " n_rate=" << cfg.n_rate
              << " n_conc_bins=" << cfg.n_conc_bins << " subsample=" << cfg.subsample
              << " P=" << cfg.P << " b0_scale=" << cfg.b0_scale
              << " conc_scale=" << cfg.conc_scale << " gnl_scale=" << cfg.gnl_scale
              << ") =====\n";

    // ── Approximant (built ONCE, full physics, always warped) ─────────────────
    const std::array<i64,3> r_shape{prob.mag.size(0), prob.mag.size(1), prob.mag.size(2)};
    const float os = cfg.oversample_factor.value_or(1.0f);
    const std::array<float,3> oversample = {os, os, os};

    std::array<i64,3> q_shape;
    if (!cfg.q_shape.empty()) {
        if (cfg.q_shape.size() != 3)
            throw std::invalid_argument("TestConfig::q_shape must have 3 entries (this Problem is 3D)");
        q_shape = {cfg.q_shape[0], cfg.q_shape[1], cfg.q_shape[2]};
    } else {
        for (i64 d = 0; d < 3; ++d)
            q_shape[d] = (i64)std::ceil(oversample[d] * (float)(r_shape[d] + 2*cfg.margin_voxels));
    }

    // dx_axis is always the r-grid's own voxel size now — CoordinateWarp
    // applies oversample_factor internally to both the base position and the
    // displacement together, so callers never convert it themselves.
    //
    // field_fn evaluates value+gradient ANALYTICALLY at arbitrary (off-grid)
    // positions — required by CoordinateWarp's batched Newton inversion,
    // which queries positions that generally aren't on the r-grid at all. A
    // discrete per-r-voxel array (the old API) can't answer that without an
    // extra interpolation (and its own error) on every Newton step. This
    // test's GNL terms (x²,y²,z²) happen to depend on a single axis each, but
    // field_fn's signature doesn't assume that — pos is the full [N,3]
    // position and a general (e.g. spherical-harmonic) field_fn would just
    // read other columns too.
    std::vector<CoordinateWarp::Channel> gnl_channels;
    gnl_channels.reserve(prob.Qg);
    for (i64 q = 0; q < prob.Qg; ++q) {
        const i64  axis    = prob.gnl_axes[q];
        const float dx_axis = prob.gnl_dx_phys[q];
        CoordinateWarp::FieldFn field_fn = [axis, dx_axis](const Tensor& pos) -> std::pair<Tensor, Tensor> {
            auto x_phys = pos.select(1, axis).mul(Scalar(dx_axis));   // pixel -> physical
            auto val    = x_phys.mul(x_phys);                          // x_phys²
            auto grad   = zeros_like(pos);                             // [N, D]
            grad.select(1, axis).copy_(x_phys.mul(Scalar(2.0f * dx_axis)));  // d(x²)/d(pix) = 2x·dx
            return {val, grad};
        };
        gnl_channels.push_back(CoordinateWarp::Channel{
            std::move(field_fn), axis, prob.gnl_c_phys[q], dx_axis});
    }

    reset_warp_timing_stats();
    CoordinateWarp warp(gnl_channels, r_shape, q_shape, prob.mag.device(), oversample);
    {
        auto t = get_warp_timing_stats();
        std::cout << "  [warp ctor timing] newton=" << std::fixed << std::setprecision(3) << t.newton_s
                  << "s (" << t.newton_calls << " calls)\n";
    }

    {
        // Margin check: warped voxels whose displacement pushes them too
        // close to (or past) the q-grid boundary lose accuracy in FINUFFT's
        // spreading kernel (finite support) instead of being correctly
        // gridded — a real source of error if q_shape has no margin beyond
        // oversample_factor·r_shape.
        const auto& max_disp = warp.max_abs_displacement_pix();
        std::cout << "  max |GNL displacement| (q-pixels): [";
        for (size_t d = 0; d < max_disp.size(); ++d)
            std::cout << (d ? ", " : "") << std::fixed << std::setprecision(2) << max_disp[d];
        std::cout << "]  q_shape=[" << q_shape[0] << "," << q_shape[1] << "," << q_shape[2]
                  << "]  r_shape=[" << r_shape[0] << "," << r_shape[1] << "," << r_shape[2]
                  << "]  oversample=" << os << "\n";
        for (i64 d = 0; d < 3; ++d) {
            const float margin = ((float)q_shape[d] - oversample[d]*(float)r_shape[d]) / 2.0f;
            if (max_disp[d] > margin)
                std::cout << "  [WARNING] axis " << d << ": max displacement " << max_disp[d]
                          << " q-px exceeds half the margin (" << margin
                          << " q-px) — boundary clamping likely, increase margin_voxels.\n";
        }
    }

    auto approx = build_approximant(prob, cfg.L, cfg.subsample, cfg.weighting, warp);
    {
        auto t = get_warp_timing_stats();
        std::cout << "  [warp timing, cumulative incl. ctor]\n"
                  << "    newton:      " << std::fixed << std::setprecision(3) << t.newton_s
                  << "s (" << t.newton_calls << " calls)\n"
                  << "    nufft build: " << t.nufft_build_s << "s (" << t.nufft_build_calls << " calls)\n"
                  << "    fft:         " << t.fft_s << "s (" << t.fft_calls << " calls)\n"
                  << "    nufft apply: " << t.nufft_apply_s << "s (" << t.nufft_apply_calls << " calls)\n";
    }
    const i64 N_mask = approx.hist.mask_idx.size(0);
    const i64 N_q    = q_shape[0] * q_shape[1] * q_shape[2];
    std::cout << "  n_hist=" << approx.hist.n_hist << "  N_mask=" << N_mask
              << "  (" << std::fixed << std::setprecision(1) << (100.0f*N_mask/(float)N_q) << "% of q-grid N)\n";

    if (cfg.debug_bypass_svd) {
        std::cout << "  [debug_bypass_svd] replacing Omega/S/Upsilon with exact"
                      " rank-1 identity (Omega=1, S=1, Upsilon=1) — bypasses SVD entirely\n";
        const Device dev = prob.mag.device();
        approx.Omega   = ones({prob.K, (i64)1}, TensorOptions(dev, eScalarType::ComplexFloat));
        approx.S       = ones({(i64)1}, TensorOptions(dev, eScalarType::Float));
        approx.Upsilon = ones({approx.hist.n_hist, (i64)1}, TensorOptions(dev, eScalarType::ComplexFloat));
    }

    std::mt19937 rng(42);
    std::uniform_int_distribution<i64> dist(0, prob.K - 1);
    std::vector<i64> idx_buf(cfg.n_sub);
    for (i64 i = 0; i < cfg.n_sub; ++i) idx_buf[i] = dist(rng);
    auto sub_idx = Tensor::from_blob(idx_buf.data(), {cfg.n_sub}, eScalarType::Long, Device{eDeviceType::CPU})
                       .clone().to(prob.mag.device());

    Stopwatch sw_approx;
    auto S_approx_full = approx_signal_nufft_combo(
        prob, approx.Omega, approx.S, approx.Upsilon,
        approx.hist.voxel_to_bin, approx.hist.mask_idx, approx.mag_flat_eff,
        approx.q_shape, approx.oversample_factor);
    std::cout << "  approx_signal_nufft_combo (L=" << approx.Omega.size(1) << " NUFFTs): "
              << std::fixed << std::setprecision(3) << sw_approx.elapsed() << "s\n" << std::flush;
    auto S_approx = S_approx_full.index_select(1, sub_idx);

    // mag_flat_eff (q-grid sized) isn't used again — only Omega/S/Upsilon/
    // hist.mask_idx/voxel_to_bin are needed by the normal-op block below.
    approx.mag_flat_eff = Tensor();

    // ── Sensitivity table: each partial-physics exact reference vs the FULL
    //    exact (Y,Y,Y) — the true signal. Shows how much each combo's missing
    //    term(s) actually change the signal, independent of the approximant. ──
    struct Combo { bool offres, gnl, conc; const char* label; };
    const std::vector<Combo> combos = {
        {false, false, false, "none"},
        {true,  false, false, "off-res only"},
        {false, true,  false, "GNL only"},
        {false, false, true,  "concomitant only"},
        {true,  true,  false, "off-res + GNL"},
        {true,  false, true,  "off-res + concomitant"},
        {false, true,  true,  "GNL + concomitant"},
    };

    auto S_full = exact_signal_combo(prob, sub_idx, true, true, true);  // (Y,Y,Y) — the true signal

    std::cout << "\n  Sensitivity (each partial combo vs full exact (Y,Y,Y)):\n"
              << "  off-res GNL conc  |S| Pearson  phase circ-r  rel_err\n"
              << "  " << std::string(64, '-') << "\n";

    for (const auto& c : combos) {
        auto S_partial = exact_signal_combo(prob, sub_idx, c.offres, c.gnl, c.conc);

        float err     = rel_err(S_full, S_partial);
        float mag_p   = pearson(to_mag(S_full), to_mag(S_partial));
        float phase_c = circ_corr(to_phase(S_full), to_phase(S_partial));

        std::cout << "  " << (c.offres ? "Y" : "N") << "       "
                  << (c.gnl    ? "Y" : "N") << "   "
                  << (c.conc   ? "Y" : "N") << "    "
                  << std::fixed << std::setprecision(4) << std::setw(10) << mag_p
                  << "  " << std::setw(12) << phase_c
                  << "  " << std::scientific << std::setprecision(3) << err
                  << "   (" << c.label << ")\n";
    }

    // ── The actual accuracy check: approximant vs the FULL exact (Y,Y,Y). ────
    float err     = rel_err(S_full, S_approx);
    float mag_p   = pearson(to_mag(S_full), to_mag(S_approx));
    float phase_c = circ_corr(to_phase(S_full), to_phase(S_approx));
    bool  pass    = err < 0.05f;
    int   failures = !pass;

    std::cout << "\n  Approximant vs full exact (Y,Y,Y):\n"
              << "    |S| Pearson=" << std::fixed << std::setprecision(4) << mag_p
              << "  phase circ-r=" << phase_c
              << "  rel_err=" << std::scientific << std::setprecision(3) << err
              << "  " << (pass ? "PASS" : "FAIL") << "\n";

    // ── Normal operator self-consistency (naive L² vs non-hermitian CP-ALS) ───
    //
    // Operates entirely on Upsilon/Omega/mask_idx/voxel_to_bin — all already
    // q-space objects since the warp happened upstream in build_approximant.
    // The Toeplitz machinery never touches CoordinateWarp directly; it only
    // convolves k-space-derived kernels against basis-weighted grid images,
    // which is identical whether those grid positions mean r or q. Nothing
    // about CP-ALS changes for the q-space case — same code path.
    if (cfg.run_normal_op_compare) {
        // q-grid shape — approx.hist.mask_idx/voxel_to_bin index into this grid,
        // not the r-grid, since the warp ran upstream in build_approximant.
        const i64 nx = approx.q_shape[0], ny = approx.q_shape[1], nz = approx.q_shape[2];
        const Device dev = prob.mag.device();
        const TensorOptions opts_f{dev, eScalarType::Float};

        // Same coord rescaling as approx_signal_nufft_combo — this kernel's
        // im_size is the q-grid, so coords must express k_phys·dx_Q, not
        // k_phys·dx_R; dx_Q = dx_R/oversample_factor (margin alone needs no rescale).
        auto ktraj_r = prob.k_traj_flat();
        std::vector<float> k_scale_v = {1.0f/approx.oversample_factor[0],
                                         1.0f/approx.oversample_factor[1],
                                         1.0f/approx.oversample_factor[2]};
        auto k_scale = Tensor::from_blob(k_scale_v.data(), {3}, eScalarType::Float, Device{eDeviceType::CPU})
                           .clone().to(dev);
        auto ktraj  = ktraj_r.mul(k_scale.unsqueeze(0));
        auto coords = zeros({3, prob.K}, opts_f);
        coords.select(0,0).copy_(ktraj.select(1,2));
        coords.select(0,1).copy_(ktraj.select(1,1));
        coords.select(0,2).copy_(ktraj.select(1,0));
        coords = coords.contiguous();
        ktraj_r = Tensor();  // consumed building coords; coords itself is still
        ktraj   = Tensor();  // needed below (both embedding calls use it)
        k_scale = Tensor();

        PhiLowrankResult plr{approx.Omega, approx.S, approx.Upsilon};
        const i64 L_actual = approx.Omega.size(1);
        const i64 p_use = std::min(cfg.P, L_actual*L_actual);

        {
            Stopwatch sw_cp;
            core_cp_diagnostic(approx.Omega, approx.S, p_use, /*n_iter=*/25, /*rel_eps=*/1e-9);
            std::cout << "  core_cp_diagnostic: " << std::fixed << std::setprecision(3) << sw_cp.elapsed() << "s\n" << std::flush;
        }

        auto rho_r = rand({nx, ny, nz}, opts_f);
        auto rho_i = rand({nx, ny, nz}, opts_f);
        auto rho   = view_as_complex(stack({rho_r, rho_i}, -1).contiguous());
        rho_r = Tensor();  // stack() above copied into rho — these aren't needed anymore
        rho_i = Tensor();
        auto mask_3d = zeros({nx*ny*nz}, TensorOptions(dev, eScalarType::Bool));
        mask_3d.scatter_(0, approx.hist.mask_idx,
            ones({approx.hist.mask_idx.size(0)}, TensorOptions(dev, eScalarType::Bool)));
        rho = rho.reshape({nx*ny*nz}).masked_fill(mask_3d.logical_not(), Scalar(0.f))
                 .reshape({nx, ny, nz});
        mask_3d = Tensor();  // consumed by masked_fill above

        // Toeplitz kernel building does its own NUFFT-style gridding
        // (create_toeplitz_kernel_standard) — same cudaMalloc-vs-caching-
        // allocator concern as the NUFFT plan above.
        hasty::cuda::cuda_empty_cache();

        auto t0n = std::chrono::steady_clock::now();
        auto emb_naive = make_normal_naive_off_fourier_toeplitz_embeddings(
            coords, {nx, ny, nz}, approx.hist.mask_idx, approx.hist.voxel_to_bin, plr,
            eComputeStrategyBuildOffFourierEmbeddings::RUN_ALL_ON_INPUT_DEVICE,
            eStorageStrategyBuildOffFourierEmbeddings::STORE_IN_FILE);
        double t_nb = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0n).count();
        std::cout << "  naive Toeplitz embeddings build: " << std::fixed << std::setprecision(3) << t_nb << "s\n" << std::flush;

        t0n = std::chrono::steady_clock::now();
        auto emb_nh = make_normal_nonhermitian_off_fourier_toeplitz_embeddings(
            coords, {nx, ny, nz}, approx.hist.mask_idx, approx.hist.voxel_to_bin, plr, p_use,
            /*n_als_iter=*/50,
            eComputeStrategyBuildOffFourierEmbeddings::RUN_ALL_ON_INPUT_DEVICE,
            eStorageStrategyBuildOffFourierEmbeddings::STORE_IN_FILE);
        double t_nhb = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0n).count();
        std::cout << "  non-hermitian Toeplitz embeddings build (P=" << p_use << ", n_als_iter=50): "
                  << t_nhb << "s\n" << std::flush;

        Stopwatch sw_apply;
        auto out_naive = apply_normal_toeplitz_off_fourier_operator(emb_naive, rho);
        std::cout << "  apply naive normal op: " << std::fixed << std::setprecision(3) << sw_apply.elapsed() << "s\n" << std::flush;
        sw_apply = Stopwatch();
        auto out_nh    = apply_normal_toeplitz_off_fourier_operator(emb_nh, rho);
        std::cout << "  apply non-hermitian normal op: " << sw_apply.elapsed() << "s\n" << std::flush;

        auto flat_ref = out_naive.reshape({nx*ny*nz}).index_select(0, approx.hist.mask_idx).cpu().contiguous();
        auto flat_nh  = out_nh   .reshape({nx*ny*nz}).index_select(0, approx.hist.mask_idx).cpu().contiguous();

        float mre = mean_rel_err(flat_ref, flat_nh);
        std::cout << "  Normal op (ref=naïve L²):  non-hermitian CP-ALS (P=" << p_use
                  << ")  mean_rel_err=" << std::scientific << std::setprecision(3) << mre
                  << "  (build: naive=" << std::fixed << std::setprecision(2) << t_nb
                  << "s  nh=" << t_nhb << "s)\n";

        hasty::viz::orthoslicer(out_naive.abs(), {cfg.label + " normal_naive L²=" + std::to_string(cfg.L), std::nullopt}, true, false);
        hasty::viz::orthoslicer(out_nh.abs(),    {cfg.label + " normal_NH P=" + std::to_string(p_use) + " L=" + std::to_string(cfg.L), std::nullopt}, true, false);
    }

    return failures;
}

// ── Standalone NUFFT-vs-Newton timing probe ──────────────────────────────────
//
// Run BEFORE anything else (registration, histogram, SVD, ...) so you get an
// answer in minutes, not after an hour-long run with no visibility. Builds a
// real CoordinateWarp (single x² GNL channel, harmless test field) at
// increasing grid sizes and reuses the timing instrumentation already wired
// into coordinate_warp.cppm (get_warp_timing_stats/reset_warp_timing_stats) —
// the ctor pays the Newton cost, the first warp_field_r_to_q call pays the
// NUFFT (build+apply) cost. Interleaved per size (Newton then NUFFT) and
// flushed after every line, so you can Ctrl-C as soon as you have enough
// info and nothing is silently buffered.
static void benchmark_warp_components()
{
    using namespace hasty;
    using namespace hasty::mri;

    const Device dev = cuda_available() ? Device{eDeviceType::CUDA, 0} : Device{eDeviceType::CPU};
    std::cout << "\n=== Warp component timing probe (device=" << dev.str() << ") ===\n" << std::flush;

    // Cube side lengths whose cube lands near the target total voxel counts.
    // NUFFT memory scales as total_voxels * upsampfac^D (8x for default
    // upsampfac=2.0 in 3D), regardless of shape -- 1e8 is noticeably bigger
    // than production's actual logged q_shape (232*328*288 ~= 2.19e7), so
    // cap the top end near production's real scale instead of a round
    // number that was never actually validated against available VRAM.
    const std::vector<std::pair<i64,double>> sizes = {
        {100, 1e6}, {216, 1e7}, {280, 2.19e7}, {320, 3.28e7}
    };

    for (auto& [side, approx_n] : sizes) {
        std::array<i64,3> shape{side, side, side};
        const i64 n_total = side*side*side;
        std::cout << "-- N~" << approx_n << "  (actual " << n_total << " = " << side << "^3) --\n" << std::flush;

        try {
            CoordinateWarp::FieldFn field_fn = [](const Tensor& pos) -> std::pair<Tensor, Tensor> {
                auto x = pos.select(1, 0);
                auto val  = x.mul(x);
                auto grad = zeros_like(pos);
                grad.select(1, 0).copy_(x.mul(Scalar(2.0f)));
                return {val, grad};
            };
            std::vector<CoordinateWarp::Channel> channels = {
                CoordinateWarp::Channel{field_fn, 0, 1e-4f, 1.0f}
            };

            reset_warp_timing_stats();
            auto t0 = std::chrono::steady_clock::now();
            CoordinateWarp warp(channels, ArrayRef<i64>(shape), ArrayRef<i64>(shape), dev);
            double ctor_wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            {
                auto t = get_warp_timing_stats();
                std::cout << "   Newton  (ctor):  wall=" << std::fixed << std::setprecision(3) << ctor_wall
                          << "s  newton_s=" << t.newton_s << "  (" << t.newton_calls << " call)\n" << std::flush;
            }

            auto field_r = rand({n_total}, TensorOptions(dev, eScalarType::Float));
            reset_warp_timing_stats();
            auto t1 = std::chrono::steady_clock::now();
            auto out = warp.warp_field_r_to_q(field_r.reshape(shape), /*tol=*/1e-3);
            double call_wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
            {
                auto t = get_warp_timing_stats();
                std::cout << "   NUFFT (1st call): wall=" << std::fixed << std::setprecision(3) << call_wall
                          << "s  build=" << t.nufft_build_s << "  fft=" << t.fft_s
                          << "  apply=" << t.nufft_apply_s << "\n" << std::flush;
            }
        } catch (const std::exception& e) {
            std::cout << "   [SKIPPED] " << e.what() << "\n" << std::flush;
            hasty::cuda::cuda_empty_cache();
        }
    }
    std::cout << "=== probe done ===\n\n" << std::flush;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int non_fourier_interp_test(const std::vector<TestConfig>& configs)
{
    // Auto-flush every "<<" -- without this, stdout fully buffers when
    // redirected to a log file (no TTY), so none of this file's existing
    // "\n"-terminated (not std::endl) prints actually appear until process
    // exit. That's almost certainly why a 1hr+ run showed nothing in
    // logs/python_script.log-adjacent output.
    std::cout.setf(std::ios::unitbuf);

    benchmark_warp_components();

    bool show_locally = false;

    std::cout << "Off-Fourier interpolation test\n"
              << "============================\n";

    const std::string nifti_dir =
        "/home/turbotage/Documents/GitHub/HastyData/downloads/traveling_heads_7t/"
        "TH2_data_ES_s1/upload_ES/ES_20181008/subject1/";

    hasty::Tensor pd;
    hasty::Tensor b0;
    hasty::Tensor mask;
    std::array<float, 3> pixdim_mm = {1.0f, 1.0f, 1.0f};

    {
        auto b0_img = hasty::io::nifti::read_nifti(nifti_dir + "b0fieldHZ.nii.gz");
        auto pd_img = hasty::io::nifti::read_nifti(nifti_dir + "gre_qsm_mag.nii.gz");

        b0 = hasty::io::nifti::transform_nifti_data(b0_img);

        {
            auto b0_uuid = hasty::python::push_nifti_image(b0_img, "b0");
            auto pd_uuid = hasty::python::push_nifti_image(pd_img, "pd");
            auto result = hasty::python::run_script(
                hasty::python::scripts_dir() + "/register_nifti.py",
                {
                    "--fixed="    + hasty::python::uuid_to_hex(b0_uuid),
                    "--moving="   + hasty::python::uuid_to_hex(pd_uuid),
                    "--transform=ResampleOnly",
                },
                /*num_outputs=*/1);
            if (result.exit_code != 0) {
                std::cerr << "Registration failed (exit " << result.exit_code << ")\n";
                return 1;
            }
            auto reg_uuid_arr = hasty::python::hex_to_uuid_array(result.output_uuids.at(0));
            std::string reg_key(reinterpret_cast<const char*>(reg_uuid_arr.data()), 16);
            b0 = hasty::server::global_generic_value_bank.fetch_value(reg_key).as_tensor();
            hasty::viz::orthoslicer(b0, {"b0_registered_to_pd", std::nullopt}, true, show_locally);
        }


        pd = hasty::io::nifti::transform_nifti_data(pd_img);

        auto b0_mean = b0.mean();
        auto b0_std  = b0.std();
        std::cout << "b0 mean: " << b0_mean.item<hasty::f32>()
                  << "  std: " << b0_std.item<hasty::f32>() << "\n";

        mask = pd[0];

        pixdim_mm = {pd_img.header.pixdim[1],
                     pd_img.header.pixdim[2],
                     pd_img.header.pixdim[3]};
        std::cout << "pixdim (mm): "
                  << pixdim_mm[0] << " x " << pixdim_mm[1] << " x " << pixdim_mm[2] << "\n";

        auto mask_mean = mask.mean();
        auto mask_std  = mask.std();
        std::cout << "mask mean: " << mask_mean.item<hasty::f32>()
                  << "  std: " << mask_std.item<hasty::f32>() << "\n";

        try {
            std::cout << "mask min: " << mask.min().item<hasty::f32>()
                      << "  max: "   << mask.max().item<hasty::f32>()
                      << "  numel: " << mask.numel() << "\n";
        } catch (...) {}

        mask = mask > (mask_mean - 0.5f * mask_std);

        try {
            auto n_true = mask.to(hasty::eScalarType::Long).sum().item<hasty::i64>();
            std::cout << "mask true: " << n_true << " / " << mask.numel()
                      << " (" << (100.0 * n_true / (double)mask.numel()) << "%)\n";
        } catch (...) {}

        hasty::viz::orthoslicer(mask, {"mask", std::nullopt}, false, show_locally);

        mask = mask.to(hasty::Device{hasty::eDeviceType::CUDA, 0});
        {
            auto sph = hasty::ellipsoid_mask(
                {mask.size(0), mask.size(1), mask.size(2)},
                {0.4f*mask.size(0), 0.4f*mask.size(1), 0.4f*mask.size(2)},
                {}, hasty::TensorOptions(mask.device()));
            mask = mask || sph;
            hasty::viz::orthoslicer(std::move(sph), {"spherical_mask", std::nullopt}, false, show_locally);
        }
        mask = hasty::mask_erode(std::move(mask), 1, 2, {});
        mask = hasty::mask_dilate(std::move(mask), 3, 8, {});
        mask = mask.cpu();

        auto not_mask = mask.logical_not();
        pd[hasty::Slice(),not_mask] = 0.0f;

        hasty::viz::orthoslicer(mask, {"mask_dilated", std::nullopt}, false, show_locally);

        hasty::server::global_generic_value_bank.clear_all();
    }

    std::cout << "=====================================================\n"
              << "  off-Fourier interpolator accuracy test\n"
              << "=====================================================\n";

    if (!cuda_available()) {
        std::cout << "[CUDA] not available — cannot run tests.\n";
        return 1;
    }

    hasty::Device cuda0{hasty::eDeviceType::CUDA, 0};
    int failures = 0;

    {
        hasty::Tensor pd_3d   = (pd.ndimension() > 3) ? pd.select(0, 0) : pd;
        hasty::Tensor b0_3d   = (b0.ndimension() > 3) ? b0.select(0, 0) : b0;
        hasty::Tensor mask_3d = mask.to(hasty::eScalarType::Bool);

        std::cout << "\nReal-data shape: "
                  << pd_3d.size(0) << " x " << pd_3d.size(1)
                  << " x " << pd_3d.size(2) << "\n";

        // Each TestConfig gets its own Problem (scaling factors are baked in
        // at construction time) and runs the full 8-combo sweep.
        for (const auto& cfg : configs) {
            auto prob_real = make_problem_real(
                pd_3d, b0_3d, mask_3d,
                pixdim_mm[0], pixdim_mm[1], pixdim_mm[2],
                cfg.n_spokes, cfg.n_samp,
                cuda0,
                cfg.b0_scale, cfg.te_start_s, cfg.conc_scale, cfg.gnl_scale, cfg.B0_tesla);

            failures += run_all_combos(prob_real, cfg);
        }
    }

    std::cout << "\n=====================================================\n"
              << "  " << failures << " failure(s)\n"
              << "=====================================================\n";
    return failures > 0 ? 1 : 0;
}


// ── Bandlimited interp identity check ────────────────────────────────────────
//
// Isolates the core math added to CoordinateWarp's scatter/gather (forward-FFT
// + full type-2 NUFFT, mode_order=DEFAULT/CMCL, sign=POS) from everything else
// (padding, Newton inversion, Jacobian) by duplicating just that pipeline here
// against a small random image, queried at ITS OWN exact integer grid
// positions. Bandlimited reconstruction at a sample point must recover that
// sample exactly (up to `tol`) — a hard identity, not an approximation — so
// this either confirms the core math is right or proves it isn't, before
// spending minutes on the full real-data pipeline.
// Stage 0: verify execute()'s RAW semantics directly — feed ARBITRARY (not
// FFT'd) data as "modes", brute-force compute the expected
// Sum_k c_k*exp(sign*i*k_eff*x) by hand (CMCL: k_eff = array_index - shape/2),
// compare against plan.execute(). Uses a CUBE shape (nx=ny=nz) and a query
// point with equal coords on all 3 axes, so the brute-force formula is
// invariant to axis-reversal ordering (sidesteps that entirely) — isolates
// purely "does execute() compute the formula I think it computes" from any
// fftshift/mode_order assumptions about FFT'd input.
static bool nufft_execute_semantics_check()
{
    using namespace hasty;
    using namespace hasty::fft;

    std::cout << "\n=== Stage 0: raw execute() semantics check ===\n";

    const i64 n = 4, N = n * n * n;
    const Device dev{eDeviceType::CUDA, 0};
    const TensorOptions opts_f(dev, eScalarType::Float);

    auto img = view_as_complex(
        stack({rand({n, n, n}, opts_f), rand({n, n, n}, opts_f)}, -1).contiguous());

    const float p     = 1.3f;                              // same pixel coord on all 3 axes
    const float c_rad = 2.0f * (float)std::numbers::pi_v<double> * p / (float)n;

    // Brute-force expected value (CMCL: k_eff = index - n/2), sign=+1.
    std::complex<double> expected(0.0, 0.0);
    auto img_cpu_r = img.real().to(Device{eDeviceType::CPU});
    auto img_cpu_i = img.imag().to(Device{eDeviceType::CPU});
    for (i64 ix = 0; ix < n; ++ix)
    for (i64 iy = 0; iy < n; ++iy)
    for (i64 iz = 0; iz < n; ++iz) {
        const double k_eff = (double)((ix - n/2) + (iy - n/2) + (iz - n/2));
        const double phase = k_eff * (double)c_rad;
        const double vr = (double)img_cpu_r.select(0, ix).select(0, iy).select(0, iz).item<float>();
        const double vi = (double)img_cpu_i.select(0, ix).select(0, iy).select(0, iz).item<float>();
        std::complex<double> v(vr, vi);
        expected += v * std::complex<double>(std::cos(phase), std::sin(phase));
    }

    auto coords = ones({3, 1}, opts_f).mul(Scalar(c_rad)).contiguous();

    NufftOptions<cuda_t, f32, UTN> opts;
    opts.ntransf = 1;
    opts.sign    = decltype(opts)::eNufftSign::POS;   // matches +i above; mode_order DEFAULT (CMCL)

    const std::array<i64, 3> shape{n, n, n};
    NufftPlan<cuda_t, f32, 3, UTN> plan(shape, opts);
    plan.setpts(coords);

    auto input  = img.reshape({1, n, n, n}).contiguous();
    auto output = zeros({1, 1}, TensorOptions(dev, eScalarType::ComplexFloat)).contiguous();
    plan.execute(input, output);

    auto out_cpu = output.reshape({1}).to(Device{eDeviceType::CPU});
    std::complex<double> got((double)out_cpu.real().item<float>(), (double)out_cpu.imag().item<float>());

    const double err = std::abs(got - expected);
    const double mag = std::abs(expected) + 1e-30;

    std::cout << "  expected = " << expected.real() << " + " << expected.imag() << "i\n"
              << "  got      = " << got.real() << " + " << got.imag() << "i\n"
              << "  rel err  = " << std::scientific << (err / mag) << "\n";

    const bool pass = (err / mag) < 1e-3;
    std::cout << "  " << (pass ? "PASS" : "FAIL") << "\n\n";
    return pass;
}

// Stage 0b: stage 0 used a cubic shape + equal-on-all-axes query coord,
// which HIDES axis correspondence (a reversed vs. direct coords<->axis
// pairing would look identical). Use a non-cubic shape and a query point
// with DIFFERENT coords per axis, test the formula under BOTH possible
// axis-pairing conventions (direct: row d <-> axis d; reversed: row d <->
// axis D-1-d, what make_nufft_coords/coordinate_warp.cppm assumes), and see
// which one execute() actually implements.
static bool nufft_axis_order_check()
{
    using namespace hasty;
    using namespace hasty::fft;

    std::cout << "\n=== Stage 0b: axis-correspondence check (non-cubic) ===\n";

    const i64 nx = 4, ny = 6, nz = 8;
    const Device dev{eDeviceType::CUDA, 0};
    const TensorOptions opts_f(dev, eScalarType::Float);

    auto img = view_as_complex(
        stack({rand({nx, ny, nz}, opts_f), rand({nx, ny, nz}, opts_f)}, -1).contiguous());

    const float px = 1.0f, py = 2.0f, pz = 3.0f;   // distinct pixel coords per axis
    const float cx = 2.0f * (float)std::numbers::pi_v<double> * px / (float)nx;
    const float cy = 2.0f * (float)std::numbers::pi_v<double> * py / (float)ny;
    const float cz = 2.0f * (float)std::numbers::pi_v<double> * pz / (float)nz;

    std::complex<double> expected(0.0, 0.0);
    auto img_cpu_r = img.real().to(Device{eDeviceType::CPU});
    auto img_cpu_i = img.imag().to(Device{eDeviceType::CPU});
    for (i64 ix = 0; ix < nx; ++ix)
    for (i64 iy = 0; iy < ny; ++iy)
    for (i64 iz = 0; iz < nz; ++iz) {
        const double kx = (double)(ix - nx/2), ky = (double)(iy - ny/2), kz = (double)(iz - nz/2);
        const double phase = kx*(double)cx + ky*(double)cy + kz*(double)cz;
        const double vr = (double)img_cpu_r.select(0, ix).select(0, iy).select(0, iz).item<float>();
        const double vi = (double)img_cpu_i.select(0, ix).select(0, iy).select(0, iz).item<float>();
        std::complex<double> v(vr, vi);
        expected += v * std::complex<double>(std::cos(phase), std::sin(phase));
    }

    NufftOptions<cuda_t, f32, UTN> opts;
    opts.ntransf = 1;
    opts.sign    = decltype(opts)::eNufftSign::POS;

    const std::array<i64, 3> shape{nx, ny, nz};
    auto input  = img.reshape({1, nx, ny, nz}).contiguous();

    auto run = [&](const Tensor& coords) -> std::complex<double> {
        NufftPlan<cuda_t, f32, 3, UTN> plan(shape, opts);
        plan.setpts(coords);
        auto output = zeros({1, 1}, TensorOptions(dev, eScalarType::ComplexFloat)).contiguous();
        plan.execute(input, output);
        auto out_cpu = output.reshape({1}).to(Device{eDeviceType::CPU});
        return {(double)out_cpu.real().item<float>(), (double)out_cpu.imag().item<float>()};
    };

    // Direct: row 0=cx(axis0/x), row1=cy(axis1/y), row2=cz(axis2/z).
    std::vector<float> direct_buf{cx, cy, cz};
    auto coords_direct = Tensor::from_blob(direct_buf.data(), {3, 1}, eScalarType::Float, Device{eDeviceType::CPU})
                              .clone().to(dev);
    auto got_direct = run(coords_direct);

    // Reversed: row0=cz(axis2), row1=cy(axis1), row2=cx(axis0) — matches
    // make_nufft_coords' "row d <-> axis D-1-d" convention.
    std::vector<float> reversed_buf{cz, cy, cx};
    auto coords_reversed = Tensor::from_blob(reversed_buf.data(), {3, 1}, eScalarType::Float, Device{eDeviceType::CPU})
                                .clone().to(dev);
    auto got_reversed = run(coords_reversed);

    const double mag = std::abs(expected) + 1e-30;
    const double err_direct   = std::abs(got_direct   - expected) / mag;
    const double err_reversed = std::abs(got_reversed - expected) / mag;

    std::cout << "  expected         = " << expected.real()   << " + " << expected.imag()   << "i\n"
              << "  got (direct)     = " << got_direct.real() << " + " << got_direct.imag() << "i  rel_err=" << std::scientific << err_direct << "\n"
              << "  got (reversed)   = " << got_reversed.real() << " + " << got_reversed.imag() << "i  rel_err=" << err_reversed << "\n";

    const bool direct_ok   = err_direct   < 1e-3;
    const bool reversed_ok = err_reversed < 1e-3;
    std::cout << "  direct convention is " << (direct_ok ? "CORRECT" : "WRONG") << "\n"
              << "  reversed (make_nufft_coords) convention is " << (reversed_ok ? "CORRECT" : "WRONG") << "\n\n";

    return direct_ok || reversed_ok;   // just confirms one of them works; caller should check the printed verdict
}

// Stage 0c: execute() and axis convention are both confirmed correct (0/0b
// passed) — the only remaining untested piece of stage 1's pipeline is
// fftn/fftshift themselves. Isolate them completely from NUFFT: compute
// c_k=fftshift(fftn(x)) via the wrapper, and separately brute-force
// C_true[k_eff] = Sum_n x[n]*exp(-2*pi*i*k_eff*n/N) by hand for one
// CMCL-indexed k_eff, compare against c_k at that same index.
static bool fftshift_cmcl_check()
{
    using namespace hasty;
    using namespace hasty::fft;

    std::cout << "\n=== Stage 0c: fftn+fftshift vs brute-force DFT ===\n";

    const i64 nx = 16, ny = 20, nz = 12;
    const Device dev{eDeviceType::CUDA, 0};
    const TensorOptions opts_f(dev, eScalarType::Float);

    auto img = view_as_complex(
        stack({rand({nx, ny, nz}, opts_f), rand({nx, ny, nz}, opts_f)}, -1).contiguous());

    auto c_k = fftshift(fftn(img));   // should be CMCL-ordered: index j <-> k_eff = j - shape/2

    // Pick one CMCL index per axis, brute-force its DFT coefficient by hand.
    const i64 jx = 3, jy = 5, jz = 2;            // array indices into c_k
    const i64 kx = jx - nx/2, ky = jy - ny/2, kz = jz - nz/2;  // CMCL k_eff

    std::complex<double> expected(0.0, 0.0);
    auto img_cpu_r = img.real().to(Device{eDeviceType::CPU});
    auto img_cpu_i = img.imag().to(Device{eDeviceType::CPU});
    for (i64 ix = 0; ix < nx; ++ix)
    for (i64 iy = 0; iy < ny; ++iy)
    for (i64 iz = 0; iz < nz; ++iz) {
        const double phase = -2.0 * std::numbers::pi_v<double> *
            ((double)kx*ix/(double)nx + (double)ky*iy/(double)ny + (double)kz*iz/(double)nz);
        const double vr = (double)img_cpu_r.select(0, ix).select(0, iy).select(0, iz).item<float>();
        const double vi = (double)img_cpu_i.select(0, ix).select(0, iy).select(0, iz).item<float>();
        std::complex<double> v(vr, vi);
        expected += v * std::complex<double>(std::cos(phase), std::sin(phase));
    }

    auto got_t = c_k.select(0, jx).select(0, jy).select(0, jz).to(Device{eDeviceType::CPU});
    std::complex<double> got((double)got_t.real().item<float>(), (double)got_t.imag().item<float>());

    const double err = std::abs(got - expected);
    const double mag = std::abs(expected) + 1e-30;

    std::cout << "  k_eff=(" << kx << "," << ky << "," << kz << ")\n"
              << "  expected = " << expected.real() << " + " << expected.imag() << "i\n"
              << "  got      = " << got.real()      << " + " << got.imag()      << "i\n"
              << "  rel err  = " << std::scientific << (err / mag) << "\n";

    const bool pass = (err / mag) < 1e-3;
    std::cout << "  " << (pass ? "PASS" : "FAIL") << "\n\n";
    return pass;
}

static bool bandlimited_interp_identity_check()
{
    using namespace hasty;
    using namespace hasty::fft;

    std::cout << "\n=== Stage 1: bandlimited interp identity check (isolated, no warp) ===\n";

    const i64 nx = 16, ny = 20, nz = 12;
    const i64 N  = nx * ny * nz;
    const Device dev{eDeviceType::CUDA, 0};
    const TensorOptions opts_f(dev, eScalarType::Float);

    auto img = view_as_complex(
        stack({rand({nx, ny, nz}, opts_f), rand({nx, ny, nz}, opts_f)}, -1).contiguous());

    // Forward FFT (DC-first), fftshift to DC-centered — matches the CMCL
    // mode_order convention used by coordinate_warp.cppm's executors.
    auto c_k = fftshift(fftn(img)).reshape({N}).contiguous();

    // Query positions: the grid's own exact integer LITERAL (FFT-relative,
    // 0-based) coords — NOT centered. fftn/fftshift's CMCL indexing pairs
    // array index j with k_eff=j-shape/2, but that's relative to the FFT's
    // own array-index origin (literal 0..N-1), not the physical centered
    // convention (i-shape/2) used elsewhere for spatial positions. Feeding
    // centered coords here introduces a constant per-axis N/2 shift, i.e. an
    // alternating (-1)^k_eff phase error across all frequencies.
    std::vector<float> pix_buf((std::size_t)N * 3);
    {
        i64 idx = 0;
        for (i64 ix = 0; ix < nx; ++ix)
        for (i64 iy = 0; iy < ny; ++iy)
        for (i64 iz = 0; iz < nz; ++iz) {
            pix_buf[(std::size_t)idx * 3 + 0] = (float)ix;
            pix_buf[(std::size_t)idx * 3 + 1] = (float)iy;
            pix_buf[(std::size_t)idx * 3 + 2] = (float)iz;
            ++idx;
        }
    }
    auto pix = Tensor::from_blob(pix_buf.data(), {N, 3}, eScalarType::Float, Device{eDeviceType::CPU})
                   .clone().to(dev);

    // Axis-reversed [-pi,pi] coords — same convention as make_nufft_coords.
    const std::array<i64, 3> shape{nx, ny, nz};
    auto coords = zeros({3, N}, opts_f);
    for (i64 d = 0; d < 3; ++d) {
        const float scale = 2.0f * (float)std::numbers::pi_v<double> / (float)shape[3 - 1 - d];
        coords.select(0, d).copy_(pix.select(1, 3 - 1 - d).mul(Scalar(scale)));
    }
    coords = coords.contiguous();

    NufftOptions<cuda_t, f32, UTN> opts;
    opts.ntransf = 1;
    opts.sign    = decltype(opts)::eNufftSign::POS;   // mode_order left DEFAULT (CMCL)

    NufftPlan<cuda_t, f32, 3, UTN> plan(shape, opts);
    plan.setpts(coords);

    auto input  = c_k.reshape({1, nx, ny, nz}).contiguous();
    auto output = zeros({1, N}, TensorOptions(dev, eScalarType::ComplexFloat)).contiguous();
    plan.execute(input, output);

    auto recon = output.reshape({N}).div(Scalar((float)N));
    auto orig  = img.reshape({N});

    auto  err      = recon.sub(orig).abs();
    float max_err  = err.max().item<float>();
    float ref_norm = orig.abs().mean().item<float>();
    float rel_err  = err.mean().item<float>() / (ref_norm + 1e-30f);

    std::cout << "  N=" << N << " (" << nx << "x" << ny << "x" << nz << ")\n"
              << "  max |recon-orig| = " << std::scientific << max_err << "\n"
              << "  rel err          = " << rel_err << "\n";

    const bool pass = rel_err < 1e-3f;
    std::cout << "  " << (pass ? "PASS" : "FAIL") << "\n\n";
    return pass;
}

int main()
{
    hasty::InferenceMode im;

    if (!nufft_execute_semantics_check()) {
        std::cout << "execute()'s raw semantics don't match the assumed "
                     "Sum_k c_k*exp(sign*i*k_eff*x) (CMCL) formula — bug is in "
                     "how sign/mode_order map to the underlying NUFFT call, "
                     "not in the FFT/fftshift wrapping. Skipping later stages.\n";
        return 1;
    }

    nufft_axis_order_check();   // prints which axis convention is correct; informational, doesn't gate

    if (!fftshift_cmcl_check()) {
        std::cout << "fftn+fftshift doesn't match brute-force DFT at a CMCL "
                     "index — bug is in the FFT/fftshift step itself, not in "
                     "NUFFT execute() or axis ordering. Skipping later stages.\n";
        return 1;
    }

    if (!bandlimited_interp_identity_check()) {
        std::cout << "execute() semantics check passed but the full "
                     "FFT+fftshift+reconstruct pipeline still fails — bug is "
                     "in the fftshift/CMCL-alignment step, not in execute() "
                     "itself. Skipping the full (slow) real-data pipeline.\n";
        return 1;
    }


    hasty::io::setup_default_dirs();

    std::vector<TestConfig> configs;
    // {
    //     TestConfig cfg;
    //     cfg.label      = "default";
    //     cfg.b0_scale   = 0.5f;
    //     cfg.conc_scale = 0.5f;
    //     cfg.gnl_scale  = 1.0f;
    //     configs.push_back(cfg);
    // }
    {
        // Same as above, but bypasses the SVD (Omega/S/Upsilon replaced with
        // the exact trivial rank-1 operator) — isolates whether the residual
        // error is from SVD/Lanczos numerics or from the warp/NUFFT pipeline.
        TestConfig cfg;
        cfg.label           = "GNL-only sanity, SVD bypassed";
        cfg.n_rate          = 2000;
        cfg.b0_scale        = 1e-9f;
        cfg.conc_scale      = 1e-9f;
        cfg.gnl_scale       = 1e-9f;
        cfg.oversample_factor = hasty::nullopt;
        cfg.debug_bypass_svd = true;
        configs.push_back(cfg);
    }

    return non_fourier_interp_test(configs);
}
