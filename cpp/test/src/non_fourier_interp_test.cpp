#include <numbers>
#include <cmath>

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_linalg_mod;
import hasty_fft_mod;
import hasty_io_mod;
import hasty_viz_mod;
import hasty_server_mod;
import hasty_python_mod;
import mri_mod;

struct Problem {
    hasty::Tensor mag;
    hasty::Tensor rate_map;
    hasty::Tensor sensitivity_maps;
    hasty::Tensor k_traj;
    hasty::Tensor timestamps;
    hasty::Tensor nl_waveforms;    // [Q, K]
    hasty::Tensor nl_basis;        // [Q, n0, n1, n2]
    hasty::Tensor mag_flat;        // [N] float — masked PD (0 outside histogram mask)
    hasty::Tensor pd_flat;         // [N] float — raw PD, no morphological mask applied
    hasty::Tensor z_map_flat;      // [N]   complex
    hasty::Tensor nl_fields_flat;  // [Q, N] float
    hasty::Tensor coords_flat;     // [N, 3] float  normalized [-0.5, 0.5]
    hasty::i64 n_dim, K, N, C;
    float FOV, dt;
};

static bool cuda_available()
{
    try {
        hasty::empty({1}, hasty::TensorOptions{
            hasty::Device{hasty::eDeviceType::CUDA, 0}, hasty::eScalarType::Float});
        return true;
    } catch (...) { return false; }
}

// ── Synthetic ball phantom problem ──────────────────────────────────────────

static Problem make_problem(hasty::i64 n_dim, hasty::i64 n_spokes, hasty::i64 n_samp,
                             hasty::Device dev)
{
    using namespace hasty;
    using namespace std::numbers;

    const float gamma = 2.0f * (float)pi_v<f64> * 42.577e6f;
    const float dt    = 4e-6f;
    const float FOV   = 0.08f;
    const float dx    = FOV / (float)n_dim;
    const i64   K     = n_spokes * n_samp;
    const i64   N     = n_dim * n_dim * n_dim;
    const i64   Q     = 2;
    const i64   C     = 1;

    const TensorOptions opts_f = TensorOptions(dev, eScalarType::Float);
    const TensorOptions opts_c = TensorOptions(dev, eScalarType::ComplexFloat);
    const TensorOptions opts_l = TensorOptions(dev, eScalarType::Long);

    auto grid_1d = [&](i64 n) {
        return arange(n, opts_f).add(Scalar(0.5f)).div(Scalar((f32)n)).sub(Scalar(0.5f));
    };

    auto n_idx = arange(N, opts_l);
    auto ix    = n_idx.div(Scalar((i64)(n_dim * n_dim))).to(eScalarType::Long);
    auto iy    = n_idx.div(Scalar((i64)n_dim)).remainder(Scalar((i64)n_dim)).to(eScalarType::Long);
    auto iz    = n_idx.remainder(Scalar((i64)n_dim)).to(eScalarType::Long);
    auto rx    = grid_1d(n_dim).index_select(0, ix);
    auto ry    = grid_1d(n_dim).index_select(0, iy);
    auto rz    = grid_1d(n_dim).index_select(0, iz);

    auto r_sq      = rx.mul(rx).add(ry.mul(ry)).add(rz.mul(rz));
    auto mag_flat  = pow(r_sq, 0.5).lt(Scalar(0.4f)).to(eScalarType::Float);
    auto mag       = mag_flat.reshape({n_dim, n_dim, n_dim});

    auto z_imag     = rz.mul(Scalar(1000.0f * 2.0f * (f32)pi_v<f64>));
    auto z_map_flat = view_as_complex(stack({zeros({N}, opts_f), z_imag}, 1).contiguous());
    auto rate_map   = z_map_flat.reshape({n_dim, n_dim, n_dim});

    auto sensitivity_maps = ones({C, n_dim, n_dim, n_dim}, opts_c);

    const float k_max_rad_m = (float)pi_v<f64> / dx;
    const float k_norm_fac  = FOV / (2.0f * (float)pi_v<f64>);

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
            i64 idx   = s * n_samp + j;
            k_data[idx*3+0] = dirs[s*3+0] * k_r * k_norm_fac;
            k_data[idx*3+1] = dirs[s*3+1] * k_r * k_norm_fac;
            k_data[idx*3+2] = dirs[s*3+2] * k_r * k_norm_fac;
            t_data[idx]     = (float)j * dt;  // synth: no TE offset
        }
    }
    auto cpu = Device{eDeviceType::CPU};
    auto k_traj     = Tensor::from_blob(k_data.data(), {K, 3}, eScalarType::Float, cpu).clone().to(dev);
    auto timestamps = Tensor::from_blob(t_data.data(), {K},    eScalarType::Float, cpu).clone().to(dev);

    auto phys_z = rz.mul(Scalar(FOV));
    auto phys_x = rx.mul(Scalar(FOV));
    auto phys_y = ry.mul(Scalar(FOV));
    auto field_z2 = phys_z.mul(phys_z);
    auto field_xy = phys_x.mul(phys_y);

    i64 half = n_samp / 2;
    std::vector<float> g_bip(n_samp);
    for (i64 j = 0; j < n_samp; ++j) g_bip[j] = j < half ? 1.0f : -1.0f;

    float z_sq_max = 0.5f * FOV * 0.5f * FOV;
    float xy_max   = 0.5f * FOV * 0.5f * FOV;
    float G_z2_amp = 0.5f * 2.0f * (float)pi_v<f64> / (gamma * dt * (float)half * z_sq_max);
    float G_xy_amp = 0.5f * (float)pi_v<f64>         / (gamma * dt * (float)half * xy_max * 0.5f);

    std::vector<float> G_z2(K), G_xy(K);
    for (i64 s = 0; s < n_spokes; ++s) {
        float dz2 = dirs[s*3+2] * dirs[s*3+2];
        float dxy = dirs[s*3+0] * dirs[s*3+1];
        for (i64 j = 0; j < n_samp; ++j) {
            G_z2[s*n_samp+j] = G_z2_amp * dz2 * g_bip[j];
            G_xy[s*n_samp+j] = G_xy_amp * dxy * g_bip[j];
        }
    }
    auto G_z2_t = Tensor::from_blob(G_z2.data(), {K}, eScalarType::Float, cpu).clone().to(dev);
    auto G_xy_t = Tensor::from_blob(G_xy.data(), {K}, eScalarType::Float, cpu).clone().to(dev);

    auto alpha_z2 = G_z2_t.to(eScalarType::Double).cumsum(0).to(eScalarType::Float).mul(Scalar(-gamma * dt));
    auto alpha_xy = G_xy_t.to(eScalarType::Double).cumsum(0).to(eScalarType::Float).mul(Scalar(-gamma * dt));

    auto nl_waveforms   = stack({alpha_z2, alpha_xy}, 0);
    auto nl_basis       = stack({field_z2.reshape({n_dim,n_dim,n_dim}),
                                  field_xy.reshape({n_dim,n_dim,n_dim})}, 0);
    auto nl_fields_flat = stack({field_z2, field_xy}, 0);
    auto coords_flat    = stack({rx, ry, rz}, 1);

    // Synthetic: pd_flat = mag_flat (ball phantom already has correct support)
    return Problem{
        mag, rate_map, sensitivity_maps,
        k_traj, timestamps,
        nl_waveforms, nl_basis,
        mag_flat, mag_flat, z_map_flat, nl_fields_flat, coords_flat,
        n_dim, K, N, C, FOV, dt
    };
}

// ── Real-data problem from NIfTI-derived tensors ─────────────────────────────
//
// pd_vol    [nx,ny,nz] float  — proton density magnetization (mask already applied)
// b0_hz_vol [nx,ny,nz] float  — B0 field map in Hz
// mask_vol  [nx,ny,nz] bool   — nonzero = include in histogram
// pixdim_*  mm (from NIfTI header pixdim[1..3])
// Nonlinear fields (z², x·y) are synthetic, computed from physical voxel positions.

static Problem make_problem_real(
    const hasty::Tensor& pd_vol,
    const hasty::Tensor& b0_hz_vol,
    const hasty::Tensor& mask_vol,
    float pixdim_x_mm, float pixdim_y_mm, float pixdim_z_mm,
    hasty::i64 n_spokes, hasty::i64 n_samp,
    hasty::Device dev,
    float b0_scale   = 1.0f,  // scale B0 field
    float te_start_s = 0.0f,  // readout start time (s)
    float nl_scale   = 1.0f)  // NL gradient amplitude scale (1.0 → ±π rad peak for z²)
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
    const i64 Q  = 2;
    const i64 C  = 1;

    const float dx_m  = pixdim_x_mm * 1e-3f;
    const float dy_m  = pixdim_y_mm * 1e-3f;
    const float dz_m  = pixdim_z_mm * 1e-3f;
    const float FOV_x = (float)nx * dx_m;
    const float FOV_y = (float)ny * dy_m;
    const float FOV_z = (float)nz * dz_m;
    const float FOV   = (FOV_x + FOV_y + FOV_z) / 3.0f;  // used for NL field amplitudes only
    const float dx_min      = std::min({dx_m, dy_m, dz_m});
    const float k_max_rad_m = (float)pi_v<f64> / dx_min;
    // Per-dimension k-space normalization: k_traj[k,d] in cycles/FOV_d so that
    // NUFFT coords = 2π·k_d/N_d stay in [-π,π].  A single FOV_avg would cause
    // coords > π for the small-FOV dimensions on non-cubic volumes, silently
    // wrapping k-space points to wrong locations in cuFINUFFT.
    const float k_norm_x = FOV_x / (2.0f * (float)pi_v<f64>);
    const float k_norm_y = FOV_y / (2.0f * (float)pi_v<f64>);
    const float k_norm_z = FOV_z / (2.0f * (float)pi_v<f64>);

    const TensorOptions opts_f = TensorOptions(dev, eScalarType::Float);
    const TensorOptions opts_c = TensorOptions(dev, eScalarType::ComplexFloat);
    const TensorOptions opts_l = TensorOptions(dev, eScalarType::Long);

    // Normalized coords per-dimension [-0.5, 0.5] — same convention as forward_exact
    // Index decomposition uses float64 to avoid float32 precision loss for N > 2^24:
    // Long.div(scalar) promotes through float32; odd integers near N=20M round by 1,
    // making the quotient floor to nx instead of nx-1 → index_select out of bounds.
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

    // Magnetization: raw PD (pd_flat) and morphologically masked PD (mag_flat)
    // pd_flat: full signal support — used for the exact reference signal
    // mag_flat: pd * mask — used for histogram binning and approx interpolator
    auto mask_f   = mask_vol.to(dev).to(eScalarType::Float).flatten();
    auto pd_flat  = pd_vol.to(dev).to(eScalarType::Float).flatten();
    auto mag_flat = pd_flat.mul(mask_f);
    auto mag_3d   = mag_flat.reshape({nx, ny, nz});

    // z_map from real B0 (pure imaginary: no T2 in this test)
    auto mask_bool = mask_vol.to(dev).to(eScalarType::Bool).flatten();
    auto b0_flat   = b0_hz_vol.to(dev).to(eScalarType::Float).flatten()
                        .mul(Scalar(b0_scale));
    auto z_map_flat = view_as_complex(
        stack({zeros({N}, opts_f),
               b0_flat.mul(Scalar(2.0f * (float)pi_v<f64>))}, 1).contiguous());
    auto rate_map   = z_map_flat.reshape({nx, ny, nz});

    auto sensitivity_maps = ones({C, nx, ny, nz}, opts_c);

    // Synthetic nonlinear fields: z² and x·y (in m²)
    auto field_z2       = phys_z.mul(phys_z);
    auto field_xy       = phys_x.mul(phys_y);
    auto nl_basis       = stack({field_z2.reshape({nx, ny, nz}),
                                  field_xy.reshape({nx, ny, nz})}, 0);
    auto nl_fields_flat = stack({field_z2, field_xy}, 0);

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
            k_data[idx*3+0] = dirs[s*3+0] * k_r * k_norm_x;
            k_data[idx*3+1] = dirs[s*3+1] * k_r * k_norm_y;
            k_data[idx*3+2] = dirs[s*3+2] * k_r * k_norm_z;
            t_data[idx]     = te_start_s + (float)j * dt;  // TE offset + within-readout time
        }
    }
    auto cpu = Device{eDeviceType::CPU};
    auto k_traj     = Tensor::from_blob(k_data.data(), {K, 3}, eScalarType::Float, cpu).clone().to(dev);
    auto timestamps = Tensor::from_blob(t_data.data(), {K},    eScalarType::Float, cpu).clone().to(dev);

    // Bipolar nonlinear waveforms (zero net phase per spoke).
    // G_z2_amp / G_xy_amp designed so peak z² phase = ±π·nl_scale rad (polar spoke, edge voxel).
    i64   half     = n_samp / 2;
    float z_sq_max = 0.5f * FOV_z * 0.5f * FOV_z;
    float xy_max   = 0.5f * FOV_x * 0.5f * FOV_y;
    float G_z2_amp = nl_scale * 0.5f * 2.0f * (float)pi_v<f64> / (gamma * dt * (float)half * z_sq_max);
    float G_xy_amp = nl_scale * 0.5f * (float)pi_v<f64>         / (gamma * dt * (float)half * xy_max * 0.5f);

    std::vector<float> g_bip(n_samp);
    for (i64 j = 0; j < n_samp; ++j) g_bip[j] = j < half ? 1.0f : -1.0f;

    std::vector<float> G_z2(K), G_xy(K);
    for (i64 s = 0; s < n_spokes; ++s) {
        float dz2 = dirs[s*3+2] * dirs[s*3+2];
        float dxy = dirs[s*3+0] * dirs[s*3+1];
        for (i64 j = 0; j < n_samp; ++j) {
            G_z2[s*n_samp+j] = G_z2_amp * dz2 * g_bip[j];
            G_xy[s*n_samp+j] = G_xy_amp * dxy * g_bip[j];
        }
    }
    auto G_z2_t = Tensor::from_blob(G_z2.data(), {K}, eScalarType::Float, cpu).clone().to(dev);
    auto G_xy_t = Tensor::from_blob(G_xy.data(), {K}, eScalarType::Float, cpu).clone().to(dev);

    auto alpha_z2 = G_z2_t.to(eScalarType::Double).cumsum(0).to(eScalarType::Float).mul(Scalar(-gamma * dt));
    auto alpha_xy = G_xy_t.to(eScalarType::Double).cumsum(0).to(eScalarType::Float).mul(Scalar(-gamma * dt));

    // ── Physics diagnostics ──────────────────────────────────────────────────
    // Only the WITHIN-READOUT phase variation matters for interpolator complexity.
    // The TE phase exp(-z·TE) is identical for all k-space samples (fixed TE per spoke)
    // → constant column factor in phi, doesn't affect SVD spectrum.
    {
        using namespace std::numbers;
        const float T_readout = (float)(n_samp - 1) * dt;
        const float pi_f      = (float)pi_v<f64>;

        auto b0_in_mask  = b0_flat.masked_select(mask_bool);
        float b0_min_hz  = b0_in_mask.min().item<float>();
        float b0_max_hz  = b0_in_mask.max().item<float>();
        float b0_mean_hz = b0_in_mask.mean().item<float>();
        float b0_std_hz  = b0_in_mask.std().item<float>();
        float b0_maxabs  = std::max(std::abs(b0_min_hz), std::abs(b0_max_hz));
        float b0_max_phase = b0_maxabs * 2.0f * pi_f * T_readout;

        float max_alpha_z2  = alpha_z2.abs().max().item<float>();  // rad/m²
        float max_alpha_xy  = alpha_xy.abs().max().item<float>();  // rad/m²
        float max_field_z2  = field_z2.abs().max().item<float>();  // m²
        float max_field_xy  = field_xy.abs().max().item<float>();  // m²
        float max_phase_z2  = max_field_z2 * max_alpha_z2;        // rad
        float max_phase_xy  = max_field_xy * max_alpha_xy;        // rad

        std::cout << "\n  ── problem physics ──────────────────────────────────────\n"
                  << "  volume: " << nx << "×" << ny << "×" << nz
                  << "  (FOV " << FOV_x*100 << " × " << FOV_y*100 << " × " << FOV_z*100 << " cm)\n"
                  << "  trajectory: " << n_spokes << " spokes × " << n_samp << " samp"
                  << "  K=" << K << "\n"
                  << "  timing: TE=" << te_start_s*1e3f << "ms  readout="
                  << T_readout*1e3f << "ms\n"
                  << "\n"
                  << "  B0 (×" << b0_scale << " scale) inside mask:\n"
                  << "    min=" << std::fixed << std::setprecision(1) << b0_min_hz
                  << " Hz  max=" << b0_max_hz
                  << " Hz  mean=" << b0_mean_hz
                  << " Hz  std=" << b0_std_hz << " Hz\n"
                  << "    max phase over readout: "
                  << std::setprecision(2) << b0_max_phase << " rad"
                  << "  (max|B0|×2π×T_readout)\n"
                  << "\n"
                  << "  NL fields (nl_scale=" << nl_scale << ", bipolar → resets each spoke):\n"
                  << "    z²: max|α_z2|=" << std::scientific << std::setprecision(3)
                  << max_alpha_z2 << " rad/m²"
                  << "  max|b_z2|=" << max_field_z2 << " m²"
                  << "  → max phase=" << std::fixed << std::setprecision(2) << max_phase_z2 << " rad\n"
                  << "    x·y: max|α_xy|=" << std::scientific << std::setprecision(3)
                  << max_alpha_xy << " rad/m²"
                  << "  max|b_xy|=" << max_field_xy << " m²"
                  << "  → max phase=" << std::fixed << std::setprecision(2) << max_phase_xy << " rad\n"
                  << "  ─────────────────────────────────────────────────────────\n\n";
    }

    auto nl_waveforms = stack({alpha_z2, alpha_xy}, 0);
    auto coords_flat  = stack({rx, ry, rz}, 1);

    return Problem{
        mag_3d, rate_map, sensitivity_maps,
        k_traj, timestamps,
        nl_waveforms, nl_basis,
        mag_flat, pd_flat, z_map_flat, nl_fields_flat, coords_flat,
        nx, K, N, C, FOV, dt
    };
}

// ── Exact paired signal via forward_exact ────────────────────────────────────
//
// Evaluates S[c, m] at each paired (k_m, t_m) using mri::forward_exact.

static hasty::Tensor exact_signal_paired(
    const Problem& prob,
    const hasty::Tensor& sub_idx,  // [M] long — k-space sample indices
    bool apply_ratemap = true,
    bool apply_nonlin  = true,
    hasty::i64 batch_size = 8)
{
    using namespace hasty;

    auto k_b  = prob.k_traj.index_select(0, sub_idx);       // [M, 3]
    auto t_b  = prob.timestamps.index_select(0, sub_idx);   // [M]
    auto nl_b = prob.nl_waveforms.index_select(1, sub_idx); // [Q, M]

    return mri::forward_exact(
        prob.mag, prob.sensitivity_maps, prob.rate_map,
        t_b, k_b, nl_b, prob.nl_basis,
        apply_ratemap, apply_nonlin, batch_size);  // [C, M]
}

// ── Approx signal via L forward NUFFTs ───────────────────────────────────────
//
// Computes S_approx for ALL K k-space samples using L Type-2 NUFFTs (UTN):
//   img_l[n] = Υ[bin(n), l] · ρ[n]   (L images on the Cartesian grid)
//   F_l[k]   = NUFFT(img_l, k_traj)   (one forward NUFFT per basis image)
//   S[k]     = Σ_l Ω[k, l] · F_l[k]
//
// Returns [C=1, K] complex signal at ALL k-space points.
// Caller selects the n_sub comparison samples via index_select.
//
// Coordinate convention (FFT mode, sign=-1):
//   NUFFT computes exp(-i · x_j · n) where n ∈ [0, N-1].
//   We set x_j_d = 2π k_d / N_d and apply the half-pixel correction
//   exp(i·π·Σ_d k_d·(N_d-1)/N_d) afterwards to match
//   approx_signal's convention exp(-2πi k · r_norm) with r_norm=(n+0.5)/N-0.5.

static hasty::Tensor approx_signal_nufft(
    const Problem& prob,
    const hasty::Tensor& Omega,        // [K, L]
    const hasty::Tensor& Upsilon,      // [n_hist, L]
    const hasty::Tensor& voxel_to_bin, // [N_mask] long
    const hasty::Tensor& mask_idx)     // [N_mask] long
{
    using namespace hasty;
    using namespace hasty::fft;
    using namespace std::numbers;

    const i64    K      = prob.K;
    const i64    L      = Omega.size(1);
    const i64    N_mask = mask_idx.size(0);
    const i64    nx     = prob.mag.size(0);
    const i64    ny     = prob.mag.size(1);
    const i64    nz     = prob.mag.size(2);
    const i64    N      = prob.N;
    const Device dev    = prob.mag.device();
    const float  pi_f   = (float)pi_v<f64>;

    const TensorOptions opts_c{dev, eScalarType::ComplexFloat};
    const TensorOptions opts_f{dev, eScalarType::Float};

    // Pre-compute masked rho and k-space coords (shared across all l)
    auto rho_m = prob.mag_flat.index_select(0, mask_idx).to(eScalarType::ComplexFloat);

    // NUFFT k-space coords [3, K]: NufftPlan reverses im_size={nx,ny,nz} to nmodes={nz,ny,nx},
    // so NUFFT dim-0 (coords[0]) maps to the last (fastest) C-order dim = our nz.
    auto ktraj  = prob.k_traj;  // [K, 3] float
    auto coords = zeros({3, K}, opts_f);
    coords.select(0, 0).copy_(ktraj.select(1, 2).mul(Scalar(2.0f * pi_f / (float)nz)));
    coords.select(0, 1).copy_(ktraj.select(1, 1).mul(Scalar(2.0f * pi_f / (float)ny)));
    coords.select(0, 2).copy_(ktraj.select(1, 0).mul(Scalar(2.0f * pi_f / (float)nx)));
    coords = coords.contiguous();

    // Half-pixel correction for r_norm = (n+0.5)/N - 0.5 convention.
    //
    // CMCL mode (default) treats index n as mode m = n - N/2, giving:
    //   F[j] = exp(+iπ·Σ k_d) · Σ_n img[n] · exp(-2πi k·n/N)
    //
    // We want exp(-2πi k·r_norm[n]) = exp(-2πi k·n/N) · exp(πi k·(N-1)/N).
    //
    // Correction = desired / CMCL factor = exp(-πi Σ_d k_d/N_d)
    // (small: |phase| ≤ π/2 per dimension at Nyquist)
    auto corr_arg = ktraj.select(1, 0).mul(Scalar(-pi_f / (float)nx))
                       .add(ktraj.select(1, 1).mul(Scalar(-pi_f / (float)ny)))
                       .add(ktraj.select(1, 2).mul(Scalar(-pi_f / (float)nz)));
    auto correction = view_as_complex(
        stack({zeros({K}, opts_f), corr_arg}, 1).contiguous()).exp();  // [K] complex

    // CMCL mode (default): all N indices treated uniformly as m = n - N/2.
    // FFT mode is WRONG here: it wraps indices n > N/2 to negative frequencies,
    // misinterpreting real spatial voxels in the upper half of the image.
    NufftOptions<cuda_t, f32, UTN> nufft_opts;
    nufft_opts.ntransf = 1;
    NufftPlan<cuda_t, f32, 3, UTN> plan({nx, ny, nz}, nufft_opts);
    plan.setpts(coords);

    auto signal = zeros({K}, opts_c);
    auto F_l    = zeros({1, K}, opts_c).contiguous();

    for (i64 l = 0; l < L; ++l) {
        // Build image for basis l via 1D scatter_add_ (reliable for complex tensors)
        auto Upsilon_l   = Upsilon.select(1, l);                           // [n_hist]
        auto Upsilon_vox = Upsilon_l.index_select(0, voxel_to_bin);        // [N_mask]
        auto src_l       = Upsilon_vox.mul(rho_m);                         // [N_mask] complex

        auto img_flat = zeros({N}, opts_c);
        img_flat.scatter_add_(0, mask_idx, src_l);  // 1D scatter — works for complex

        auto img_l = img_flat.reshape({nx, ny, nz}).contiguous().unsqueeze(0); // [1,nx,ny,nz]
        plan.execute(img_l, F_l);

        signal.add_(Omega.select(1, l).mul(correction).mul(F_l.select(0, 0)));
    }

    return signal.unsqueeze(0);  // [1, K]
}

// ── NUFFT-based per-bin weights ───────────────────────────────────────────────
//
// For each histogram bin h, computes the exact k-space energy contribution:
//   w_h = Σ_k |agg(k,h)|²   where  agg(k,h) = NUFFT(m_h, k)
// and m_h is mag_flat restricted to voxels in bin h.
//
// This is the optimal weight for the SVD given the actual image support and
// the k-space trajectory (no density compensation assumption needed).
//
// Runs ceil(n_hist / bin_batch_size) batched NUFFTs.
// Returns [n_hist] float weights on the same device as prob.mag.
//
static hasty::Tensor compute_nufft_bin_weights(
    const Problem& prob,
    const hasty::mri::HistogramResult& hist)
{
    using namespace hasty;
    using namespace hasty::fft;
    using namespace std::numbers;

    const i64   n_hist = hist.n_hist;
    const i64   N_mask = hist.mask_idx.size(0);
    const i64   K      = prob.K;
    const i64   N      = prob.N;
    const i64   nx     = prob.mag.size(0);
    const i64   ny     = prob.mag.size(1);
    const i64   nz     = prob.mag.size(2);
    const Device dev   = prob.mag.device();
    const float pi_f   = (float)pi_v<f64>;

    const TensorOptions opts_c{dev, eScalarType::ComplexFloat};
    const TensorOptions opts_f{dev, eScalarType::Float};
    const TensorOptions opts_l_cpu{Device{eDeviceType::CPU}, eScalarType::Long};

    // mag_masked [N_mask] complex — image values at masked voxels
    auto mag_masked = prob.mag_flat.index_select(0, hist.mask_idx)
                                    .to(eScalarType::ComplexFloat);

    // voxel_to_bin and mask_idx stay on device; scatter done on GPU per batch

    // NUFFT plan — same coord convention as approx_signal_nufft
    auto ktraj  = prob.k_traj;   // [K, 3]
    auto coords = zeros({3, K}, opts_f);
    coords.select(0, 0).copy_(ktraj.select(1, 2).mul(Scalar(2.0f * pi_f / (float)nz)));
    coords.select(0, 1).copy_(ktraj.select(1, 1).mul(Scalar(2.0f * pi_f / (float)ny)));
    coords.select(0, 2).copy_(ktraj.select(1, 0).mul(Scalar(2.0f * pi_f / (float)nx)));
    coords = coords.contiguous();

    NufftOptions<cuda_t, f32, UTN> nufft_opts;
    nufft_opts.ntransf = 1;
    NufftPlan<cuda_t, f32, 3, UTN> plan({nx, ny, nz}, nufft_opts);
    plan.setpts(coords);

    auto weights = zeros({n_hist}, opts_f);
    auto img     = zeros({1, nx, ny, nz}, opts_c).contiguous();
    auto F_out   = zeros({1, K},          opts_c).contiguous();

    for (i64 h = 0; h < n_hist; ++h) {
        img.zero_();
        auto in_bin = hist.voxel_to_bin.eq(Scalar(h)).to(eScalarType::ComplexFloat);
        img.reshape({N}).scatter_add_(0, hist.mask_idx, mag_masked.mul(in_bin));
        plan.execute(img, F_out);
        weights.select(0, h).copy_(F_out.abs().pow(2).sum());
    }

    return weights;
}


// ── DFT + interpolants ───────────────────────────────────────────────────────
//
// S_dft[k] = Σ_l Omega[k,l] · DFT(img_l, xi[k])
//   img_l[n] = Upsilon[bin(n), l] · ρ[n]
//
// Uses the unified hasty::fft::DFTConfig / dft convention:
//   xi[k, d] = 2π k_d / N_d ∈ [−π, π]   →   same kernel as forward_exact.
// Returns [1, n_sub].

static hasty::Tensor approx_signal_dft(
    const hasty::fft::DFTConfig& cfg,    // coords for N_mask masked voxels
    const hasty::Tensor& Omega,           // [K, L]
    const hasty::Tensor& Upsilon,         // [n_hist, L]
    const hasty::Tensor& voxel_to_bin,   // [N_mask] long
    const hasty::Tensor& rho_m,           // [N_mask] ComplexFloat
    const hasty::Tensor& sub_idx,         // [n_sub] long
    const hasty::Tensor& xi_sub)          // [n_sub, d] float ∈ [−π, π]
{
    using namespace hasty;

    const i64    L      = Omega.size(1);
    const i64    N_mask = rho_m.size(0);
    const Device dev    = rho_m.device();
    const TensorOptions opts_c{dev, eScalarType::ComplexFloat};

    auto omega_sub = Omega.index_select(0, sub_idx);   // [n_sub, L]
    auto signal    = zeros({(i64)sub_idx.size(0)}, opts_c);

    for (i64 l = 0; l < L; ++l) {
        // img_l[n] = Upsilon[bin(n), l] × ρ[n]
        auto ups_vox = Upsilon.select(1, l).index_select(0, voxel_to_bin);  // [N_mask]
        auto img_l   = ups_vox.mul(rho_m);                                   // [N_mask]

        auto F_l = fft::dft(cfg, img_l, xi_sub);   // [n_sub]
        signal.add_(omega_sub.select(1, l).mul(F_l));
    }

    return signal.unsqueeze(0);  // [1, n_sub]
}


// ── Exact histogram DFT (no SVD) ─────────────────────────────────────────────
//
// S_hist_exact[k] = Σ_h exp(−z_h · t_k) · A_h(k)
//                 = Σ_n ρ[n] · exp(−z_{bin(n)} · t_k) · exp(−i ξ_k · q_n)
//
// Uses the EXACT bin phi (no SVD decomposition) via the unified DFT convention.
// Comparing this with S_approx_dft isolates the SVD approximation error in
// signal space, bypassing the 80-voxel phi sampling bias:
//   circ_corr(S_both, S_hist_exact) ≈ 1.0 → DFT convention correct
//   circ_corr(S_hist_exact, S_approx_dft) ≈ 0.917 → SVD is the bottleneck
//   circ_corr(S_hist_exact, S_approx_dft) ≈ 1.0 → Omega×Upsilon cancellation
// Returns [1, n_sub].

static hasty::Tensor exact_histogram_signal_dft(
    const hasty::fft::DFTConfig& cfg,         // coords for N_mask masked voxels
    const hasty::mri::HistogramResult& hist,
    const hasty::Tensor& rho_m,               // [N_mask] ComplexFloat
    const hasty::Tensor& t_sub,               // [n_sub] float timestamps
    const hasty::Tensor& xi_sub)              // [n_sub, d] float ∈ [−π, π]
{
    using namespace hasty;

    const i64    n_sub  = xi_sub.size(0);
    const i64    N_mask = rho_m.size(0);
    const Device dev    = rho_m.device();
    const TensorOptions opts_c{dev, eScalarType::ComplexFloat};
    const TensorOptions opts_f{dev, eScalarType::Float};
    const i64 n_chunk = 1 << 14;

    auto z_h  = hist.z_map_hist.to(eScalarType::ComplexFloat);  // [n_hist]
    auto t_cf = t_sub.to(eScalarType::ComplexFloat);              // [n_sub]

    auto signal = zeros({n_sub}, opts_c);

    for (i64 m0 = 0; m0 < n_sub; m0 += cfg.batch_size) {
        const i64 mB    = std::min(cfg.batch_size, n_sub - m0);
        const Tensor xi_b = xi_sub.narrow(0, m0, mB);   // [mB, d]
        const Tensor t_b  = t_cf.narrow(0, m0, mB);     // [mB]
        auto sig_b = zeros({mB}, opts_c);

        for (i64 n0 = 0; n0 < N_mask; n0 += n_chunk) {
            const i64 nB = std::min(n_chunk, N_mask - n0);
            const Tensor q_b    = cfg.coords.narrow(0, n0, nB);          // [nB, d]
            const Tensor rho_b  = rho_m.narrow(0, n0, nB);               // [nB]
            const Tensor bins_b = hist.voxel_to_bin.narrow(0, n0, nB);   // [nB]
            const Tensor z_b    = z_h.index_select(0, bins_b);           // [nB]

            // phi_exact[mB, nB] = exp(−z_b[n] · t_b[m])
            auto phi_b = (-z_b.unsqueeze(0).mul(t_b.unsqueeze(1))).exp();  // [mB, nB]

            // DFT kernel[mB, nB] = exp(−i × xi_b @ q_b.T)
            auto phase = mm(xi_b, q_b.transpose(0, 1));   // [mB, nB] float
            auto kern  = view_as_complex(
                stack({zeros({mB, nB}, opts_f), phase.neg()}, -1)
                    .contiguous()).exp();                   // [mB, nB] complex

            sig_b.add_(mv(phi_b.mul(kern), rho_b));       // [mB]
        }

        signal.narrow(0, m0, mB).copy_(sig_b);
    }

    return signal.unsqueeze(0);  // [1, n_sub]
}


// ── Error metric ─────────────────────────────────────────────────────────────
//
// Phase-corrected relative L2 error. Minimises over a global scalar phase φ:
//   min_φ ‖S_ref − e^{iφ} S_approx‖ / ‖S_ref‖
// A global phase offset on all k-space samples has no effect on image quality.

static float phase_corrected_rel_err(const hasty::Tensor& S_ref,
                                     const hasty::Tensor& S_approx)
{
    float ne      = S_ref.norm().item<float>();
    float na      = S_approx.norm().item<float>();
    float dot_abs = S_ref.conj().mul(S_approx).sum().abs().item<float>();
    float err_sq  = ne*ne + na*na - 2.0f * dot_abs;
    return std::sqrt(std::max(err_sq, 0.0f)) / (ne + 1e-30f);
}

// ── Core test ─────────────────────────────────────────────────────────────────
//
// 1. Extract histogram → low-rank decomposition (Ω, Υ).
// 2. Compute approx signal at n_sub randomly-spaced k-space points.
// 3. Compute exact paired (k_m, t_m) signal at the SAME points (direct NUFFT sum).
// 4. Compare with phase-corrected relative L2 error.

static bool run_test(const Problem& prob,
                     hasty::i64 n_sub,
                     const std::vector<hasty::i64>& L_values,
                     hasty::i64 n_rate, hasty::i64 n_nl,
                     const std::string& label,
                     hasty::mri::eBinWeighting weighting = hasty::mri::eBinWeighting::L1Mass,
                     // Lo > 0: run time-segmented comparison with L=L_values.back().
                     // Lo = 0: skip time-segmented comparison.
                     hasty::i64 Lo  = 7,
                     hasty::i64 Lnl = 4,
                     bool show_fft_error_plots        = false,
                     bool show_phi_error_plots        = false,
                     bool show_phi_error_vs_B0_plots  = false,
                     bool show_signal_err_vs_mag_plot = false)
{
    using namespace hasty;
    std::cout << "\n[test: " << label << "]\n";
    std::cout << "  K=" << prob.K << "  N=" << prob.N << "  n_sub=" << n_sub << "\n";

    auto hist = mri::extract_histogram(
        prob.mag_flat, prob.z_map_flat, prob.nl_fields_flat, n_rate, n_nl);
    const i64 N_mask = hist.mask_idx.size(0);
    const float mask_frac = (float)N_mask / (float)prob.N;
    std::cout << "  n_hist=" << hist.n_hist
              << "  N_mask=" << N_mask
              << "  (" << std::fixed << std::setprecision(1)
              << (100.0f * mask_frac) << "% of N)\n"
              << "  mag_flat sum=" << prob.mag_flat.sum().item<float>()
              << "  pd_flat sum="  << prob.pd_flat.sum().item<float>() << "\n";

    auto op = mri::make_phi_operator(
        hist.z_map_hist, hist.nl_fields_hist,
        prob.nl_waveforms, prob.timestamps);

    // Random subsample of k-space points (seed 42 for reproducibility)
    std::mt19937 rng(42);
    std::uniform_int_distribution<i64> dist(0, prob.K - 1);
    std::vector<i64> idx_buf(n_sub);
    for (i64 i = 0; i < n_sub; ++i) idx_buf[i] = dist(rng);
    auto sub_idx = Tensor::from_blob(idx_buf.data(), {n_sub}, eScalarType::Long,
                                     Device{eDeviceType::CPU})
                       .clone().to(prob.mag.device());

    // 4 exact physics variants (computed once, before the L loop)
    std::cout << "  Computing 4 exact signal variants ...\n";
    auto t0_exact = std::chrono::steady_clock::now();
    auto S_both   = exact_signal_paired(prob, sub_idx, true,  true);
    auto S_offres = exact_signal_paired(prob, sub_idx, true,  false);
    auto S_nonlin = exact_signal_paired(prob, sub_idx, false, true);
    auto S_nufft  = exact_signal_paired(prob, sub_idx, false, false);
    double exact_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0_exact).count();
    std::cout << "  |S_exact (both)| = " << S_both.norm().item<float>()
              << "  (exact variants: " << std::fixed << std::setprecision(2)
              << exact_s << "s)\n";

    // ── Quantization ceiling diagnostic ──────────────────────────────────────
    // Replace each voxel's B0 with its histogram bin-mean B0, then run forward_exact.
    // S_quantized = S_exact_hist (exact at histogram level, no SVD approximation).
    //
    // circ_corr(S_both, S_quantized):
    //   ≈ 0.917 → bin width is the bottleneck (increase n_rate)
    //   ≈ 1.000 → bin width is fine, error comes from SVD rank / Omega-Upsilon bug
    {
        const i64 nx = prob.mag.size(0);
        const i64 ny = prob.mag.size(1);
        const i64 nz = prob.mag.size(2);
        const i64 Nq = prob.N;
        const TensorOptions opts_fq = TensorOptions(prob.mag.device(), eScalarType::Float);

        auto bin_z_r = hist.z_map_hist.real().index_select(0, hist.voxel_to_bin);
        auto bin_z_i = hist.z_map_hist.imag().index_select(0, hist.voxel_to_bin);
        auto z_qr = zeros({Nq}, opts_fq);
        auto z_qi = zeros({Nq}, opts_fq);
        z_qr.scatter_(0, hist.mask_idx, bin_z_r);
        z_qi.scatter_(0, hist.mask_idx, bin_z_i);
        auto z_quant_flat  = view_as_complex(stack({z_qr, z_qi}, 1).contiguous());
        auto rate_map_quant = z_quant_flat.reshape({nx, ny, nz});

        auto k_q  = prob.k_traj.index_select(0, sub_idx);
        auto t_q  = prob.timestamps.index_select(0, sub_idx);
        auto nl_q = prob.nl_waveforms.index_select(1, sub_idx);

        auto S_quant = mri::forward_exact(
            prob.mag, prob.sensitivity_maps, rate_map_quant,
            t_q, k_q, nl_q, prob.nl_basis,
            true, false, 8);  // [C, n_sub]

        auto quant_r_tmp = S_quant.select(0,0).abs().cpu().contiguous();
        auto quant_p_tmp = [&]() {
            auto s1 = S_quant.select(0,0).cpu().contiguous();
            auto re = s1.real().contiguous(); auto im = s1.imag().contiguous();
            auto rv = re.spanning_view(); auto iv = im.spanning_view();
            i64 M = rv.sizes[0];
            std::vector<float> ph(M);
            const float* rp = static_cast<const float*>(rv.data);
            const float* ip = static_cast<const float*>(iv.data);
            for (i64 i = 0; i < M; ++i) ph[i] = std::atan2(ip[i], rp[i]);
            return Tensor::from_blob(ph.data(), {M}, eScalarType::Float,
                                     Device{eDeviceType::CPU}).clone();
        }();

        // inline pearson / circ_corr (lambdas not yet defined)
        auto quant_pear = [](const Tensor& a, const Tensor& b) -> float {
            auto ad = a.to(eScalarType::Double); auto bd = b.to(eScalarType::Double);
            double n=a.size(0), sa=ad.sum().item<double>(), sb=bd.sum().item<double>();
            double sab=ad.mul(bd).sum().item<double>();
            double sa2=ad.mul(ad).sum().item<double>(), sb2=bd.mul(bd).sum().item<double>();
            double num=n*sab-sa*sb;
            double den=std::sqrt(std::max(0.0,(n*sa2-sa*sa)*(n*sb2-sb*sb)));
            return (float)(num/(den+1e-60));
        };
        auto quant_circ = [](const Tensor& pa, const Tensor& pb) -> float {
            auto diff = pa.sub(pb).contiguous();
            auto dv = diff.spanning_view(); i64 M = dv.sizes[0];
            const float* d = static_cast<const float*>(dv.data);
            double s=0.0; for(i64 i=0;i<M;++i) s+=std::cos(d[i]);
            return (float)(s/M);
        };

        auto ref_m_tmp = S_both.select(0,0).abs().cpu().contiguous();
        auto ref_p_tmp = [&]() {
            auto s1 = S_both.select(0,0).cpu().contiguous();
            auto re = s1.real().contiguous(); auto im = s1.imag().contiguous();
            auto rv = re.spanning_view(); auto iv = im.spanning_view();
            i64 M = rv.sizes[0];
            std::vector<float> ph(M);
            const float* rp = static_cast<const float*>(rv.data);
            const float* ip = static_cast<const float*>(iv.data);
            for (i64 i = 0; i < M; ++i) ph[i] = std::atan2(ip[i], rp[i]);
            return Tensor::from_blob(ph.data(), {M}, eScalarType::Float,
                                     Device{eDeviceType::CPU}).clone();
        }();

        float qpear = quant_pear(ref_m_tmp, quant_r_tmp);
        float qcirc = quant_circ(ref_p_tmp, quant_p_tmp);
        std::cout << "\n  ── Quantization ceiling (histogram bin-mean B0, exact forward) ──\n"
                  << "  circ_corr(S_exact, S_quantized) = " << std::fixed << std::setprecision(4)
                  << qcirc << "  |S| Pearson = " << qpear << "\n"
                  << "  → If ≈ 0.917: bin width is the bottleneck (increase n_rate)\n"
                  << "  → If ≈ 1.000: SVD rank is the bottleneck (increase L / check bug)\n\n";
    }

    // [C,M] complex → [M] float magnitude on CPU
    auto to_mag = [](const Tensor& S) {
        return S.select(0, 0).abs().cpu().contiguous();
    };

    // [C,M] complex → [M] float phase atan2(imag,real) on CPU
    auto to_phase = [](const Tensor& S) -> Tensor {
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
        return Tensor::from_blob(ph.data(), {M}, eScalarType::Float,
                                 Device{eDeviceType::CPU}).clone();
    };

    // Pearson r of two [M] float CPU tensors.
    // Cast to double to avoid catastrophic cancellation: for large-valued signals
    // (e.g. |S| ~ 30M), n*Σa² - (Σa)² loses all precision in float32.
    auto pearson = [](const Tensor& a, const Tensor& b) -> float {
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
    };

    // Circular correlation for phase: mean(cos(θ_a - θ_b)).
    // Range [-1, 1]. Properly handles S¹ topology — Pearson on raw phase
    // breaks because samples near ±π appear maximally different even when close.
    auto circ_corr = [](const Tensor& phase_a, const Tensor& phase_b) -> float {
        // cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
        // Compute via complex unit vectors: Re(e^{ia} · conj(e^{ib}))
        // Using atan2 phases: encode as real/imag of the phase angle differences
        auto diff = phase_a.sub(phase_b);  // [M] float, difference mod wrapping
        // cos(diff): compute via stacking real+imag view and using complex exp
        // Simpler: use spanning_view and std::cos element-wise on CPU
        auto dv = diff.contiguous().spanning_view();
        i64 M = dv.sizes[0];
        const float* d = static_cast<const float*>(dv.data);
        double sum_cos = 0.0;
        for (i64 i = 0; i < M; ++i) sum_cos += std::cos(d[i]);
        return (float)(sum_cos / M);
    };

    auto ref_mag   = to_mag(S_both);
    auto ref_phase = to_phase(S_both);

    // ── Unified DFT setup (function-scope variables used in L loop too) ───────
    // xi_sub[j,d] = 2π k_d/N_d ∈ [-π,π] — same convention as cuFINUFFT.
    // dft_cfg_m has coords for the N_mask masked voxels.
    using namespace std::numbers;
    const i64   gx  = prob.mag.size(0);
    const i64   gy  = prob.mag.size(1);
    const i64   gz  = prob.mag.size(2);
    const float pif = (float)pi_v<f64>;
    const TensorOptions opts_f_xi{prob.mag.device(), eScalarType::Float};

    auto dft_cfg_full = fft::make_dft_config({gx, gy, gz}, /*batch_size=*/64,
                                              prob.mag.device());
    auto dft_cfg_m = dft_cfg_full;
    dft_cfg_m.coords = dft_cfg_full.coords.index_select(0, hist.mask_idx);

    auto k_sub_dft = prob.k_traj.index_select(0, sub_idx);  // [n_sub, 3]
    auto xi_sub    = zeros({(i64)sub_idx.size(0), (i64)3}, opts_f_xi);
    xi_sub.select(1,0).copy_(k_sub_dft.select(1,0).mul(Scalar(2.0f*pif/(float)gx)));
    xi_sub.select(1,1).copy_(k_sub_dft.select(1,1).mul(Scalar(2.0f*pif/(float)gy)));
    xi_sub.select(1,2).copy_(k_sub_dft.select(1,2).mul(Scalar(2.0f*pif/(float)gz)));
    xi_sub = xi_sub.contiguous();

    auto rho_m_dft = prob.mag_flat.index_select(0, hist.mask_idx)
                                   .to(eScalarType::ComplexFloat);

    // ── exact_histogram_signal_dft ────────────────────────────────────────────
    // Exact bin-level signal via the unified DFT (no SVD).
    //   circ_corr(S_both, S_hist_exact_dft) ≈ 1.0000 → DFT convention correct
    //   circ_corr(S_both, S_hist_exact_dft) ≈ 0.917  → DFT convention DIFFERS
    //   circ_corr(S_hist_exact_dft, S_approx_dft) ≈ 0.917 → SVD is the error
    //   circ_corr(S_hist_exact_dft, S_approx_dft) ≈ 1.000 → something else
    auto t0_he = std::chrono::steady_clock::now();
    auto S_he_full = exact_histogram_signal_dft(
        dft_cfg_m, hist, rho_m_dft,
        prob.timestamps.index_select(0, sub_idx), xi_sub);
    double he_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0_he).count();

    auto mhe_pre = to_mag(S_he_full);
    auto phe_pre = to_phase(S_he_full);
    std::cout << "\n  ── exact_histogram_signal_dft (exact bin phi, no SVD) ──────────\n"
              << "  circ_corr(S_both, S_hist_exact_dft) = " << std::fixed
              << std::setprecision(4) << circ_corr(ref_phase, phe_pre)
              << "  |S| Pearson = " << pearson(ref_mag, mhe_pre)
              << "  (" << std::setprecision(2) << he_s << "s)\n"
              << "  → 1.0000: DFT convention correct; SVD Omega×Upsilon is the bottleneck\n"
              << "  → 0.9174: DFT convention in approx_signal_dft differs from forward_exact\n\n";

    std::cout << "\n  L   n_hist   phase_corr_err   status\n"
              << "  " << std::string(44, '-') << "\n";

    bool all_pass = true;
    // NUFFT-based bin weights: w_h = Σ_k |NUFFT(m_h, k)|²
    // Computed once before the L loop; expensive but exact signal-contribution weight.
    auto t0_nw = std::chrono::steady_clock::now();
    auto nufft_weights = compute_nufft_bin_weights(prob, hist);
    double nw_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0_nw).count();
    std::cout << "  NUFFT bin weights: " << std::fixed << std::setprecision(2)
              << nw_s << "s  (sum=" << nufft_weights.sum().item<float>() << ")\n";

    for (auto L : L_values) {
        // SLEPc TRLanczos constraint: ncv ≤ nsv + mpd.
        // Use large mpd to retain more vectors per restart → better subspace accuracy.
        // ncv = L + mpd (maximum allowed).
        const i64 mpd_val = std::max((i64)8*L, (i64)60);
        const i64 ncv_val = L + mpd_val;  // satisfies ncv ≤ L + mpd exactly
        // FreqAwareLowrank: w_h = Σρ² × max(1, total_phase_h / π)
        // total_phase_h combines off-res + NL phase variation per bin.
        Opt<Tensor> bin_freq_aware;
        Opt<Tensor> bin_nufft_aware;   // NUFFT weight × freq_factor
        if (weighting == mri::eBinWeighting::FreqAwareLowrank) {
            using namespace std::numbers;
            const float pi_f    = (float)pi_v<f64>;
            float T_readout     = (prob.timestamps.max() - prob.timestamps.min()).item<float>();
            // Off-res contribution: |ω_h| × T_readout / π
            auto phase_h = hist.z_map_hist.imag().abs().mul(Scalar(T_readout)).div(Scalar(pi_f));
            // NL contribution: max|α_q| × |b_q_h| / π per NL field
            for (i64 q = 0; q < hist.nl_fields_hist.size(0); ++q) {
                float max_aq = prob.nl_waveforms.select(0,q).abs().max().item<float>();
                phase_h = phase_h.add(
                    hist.nl_fields_hist.select(0,q).abs().mul(Scalar(max_aq)).div(Scalar(pi_f)));
            }
            auto freq_factor = hasty::clamp(phase_h, Scalar(1.0f), Scalar(1e9f));
            bin_freq_aware  = Opt<Tensor>{hist.bin_l2_energy.mul(freq_factor)};
            bin_nufft_aware = Opt<Tensor>{nufft_weights.mul(freq_factor)};
        }

        auto t0_svd = std::chrono::steady_clock::now();
        auto [Omega, Upsilon] = mri::phi_lowrank(op, L,
                                                  weighting,
                                                  Opt<Tensor>{hist.bin_weights},
                                                  Opt<Tensor>{hist.bin_l2_energy},
                                                  bin_freq_aware,
                                                  ncv_val, mpd_val);
        double svd_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_svd).count();

        // NUFFT-weighted SVD: same L, but weights derived from actual k-space energy
        auto t0_nsvd = std::chrono::steady_clock::now();
        auto [Omega_nw, Upsilon_nw] = mri::phi_lowrank(op, L,
                                                         mri::eBinWeighting::FreqAwareLowrank,
                                                         Opt<Tensor>{hist.bin_weights},
                                                         Opt<Tensor>{hist.bin_l2_energy},
                                                         bin_nufft_aware,
                                                         ncv_val, mpd_val);
        double nsvd_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_nsvd).count();

        // Singular values: S[l] = ||Omega[:,l]||² (Omega = U·√S, U has unit columns)
        auto sv = Omega.abs();  sv = sv.mul(sv);  sv = sv.sum(0).cpu();  // S[l]=||Omega[:,l]||²
        std::cout << "  [sv L=" << L << "]";
        auto sv_view = sv.spanning_view();
        const float* sv_ptr = static_cast<const float*>(sv_view.data);
        float sv_total = 0.0f;
        for (i64 l = 0; l < L; ++l) sv_total += sv_ptr[l];
        for (i64 l = 0; l < L; ++l)
            std::cout << " " << std::scientific << std::setprecision(2) << sv_ptr[l];
        std::cout << "  (sum=" << sv_total << ")\n";

        // Compute approx for ALL K k-space samples via L forward NUFFTs,
        // then select the n_sub comparison samples.
        auto t0_approx = std::chrono::steady_clock::now();
        auto S_approx_full = approx_signal_nufft(
            prob, Omega, Upsilon, hist.voxel_to_bin, hist.mask_idx);   // [1, K]
        double approx_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_approx).count();
        auto S_approx = S_approx_full.index_select(1, sub_idx);        // [1, n_sub]

        float err  = phase_corrected_rel_err(S_both, S_approx);
        bool  pass = err < 0.05f;
        all_pass   = all_pass && pass;

        std::cout << "  " << std::setw(3) << L
                  << "  " << std::setw(7) << hist.n_hist
                  << "  " << std::scientific << std::setprecision(3) << err
                  << "  " << (pass ? "PASS" : "FAIL") << "\n";

        // DFT + interpolants: SVD Omega/Upsilon via unified DFT.
        // circ_corr(S_hist_exact_dft, S_approx_dft) isolates SVD multiplication error.
        auto t0_dft = std::chrono::steady_clock::now();
        auto S_dft  = approx_signal_dft(
            dft_cfg_m, Omega, Upsilon, hist.voxel_to_bin,
            rho_m_dft, sub_idx, xi_sub);  // [1, n_sub]
        double dft_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_dft).count();

        // NUFFT-weighted signal
        auto t0_nw_sig = std::chrono::steady_clock::now();
        auto S_nw_full = approx_signal_nufft(
            prob, Omega_nw, Upsilon_nw, hist.voxel_to_bin, hist.mask_idx);
        double nw_sig_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_nw_sig).count();
        auto S_nw  = S_nw_full.index_select(1, sub_idx);

        // Time-segmented: same L segments, LS-optimal temporal basis, z only (no NL).
        auto t0_ts = std::chrono::steady_clock::now();
        auto ts_phi   = mri::time_segmented_phi(hist, prob.timestamps, L, weighting);
        auto S_ts_full = approx_signal_nufft(
            prob, ts_phi.Omega, ts_phi.Upsilon, hist.voxel_to_bin, hist.mask_idx);
        double ts_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_ts).count();
        auto S_ts = S_ts_full.index_select(1, sub_idx);

        // Per-signal mag and phase
        auto mo  = to_mag(S_offres);  auto po  = to_phase(S_offres);
        auto mn  = to_mag(S_nonlin);  auto pn  = to_phase(S_nonlin);
        auto mu  = to_mag(S_nufft);   auto pu  = to_phase(S_nufft);
        auto ma  = to_mag(S_approx);  auto pa  = to_phase(S_approx);
        auto md  = to_mag(S_dft);     auto pd  = to_phase(S_dft);
        auto mnw = to_mag(S_nw);      auto pnw = to_phase(S_nw);
        auto mts = to_mag(S_ts);      auto pts = to_phase(S_ts);

        const std::string approx_lbl = "approx NUFFT (L=" + std::to_string(L) + ")";
        const std::string dft_lbl    = "approx DFT  (L=" + std::to_string(L) + ")";
        const std::string nw_lbl     = "nufft-w     (L=" + std::to_string(L) + ")";
        const std::string ts_lbl     = "time-seg    (L=" + std::to_string(L) + ")";
        const std::vector<std::string> sig_labels = {
            "exact (off-res+nonlin)", "exact (off-res only)",
            "exact (nonlin only)",    "exact (pure DFT)",
            approx_lbl, dft_lbl, nw_lbl, ts_lbl
        };

        // Sort helpers: one by |S_exact| (magnitude plot), one by arg(S_exact) (phase plot)
        auto make_sort_idx = [&](const Tensor& key) -> Tensor {
            auto rv = key.spanning_view();
            i64 M = rv.sizes[0];
            const float* p = static_cast<const float*>(rv.data);
            std::vector<i64> idx(M);
            std::iota(idx.begin(), idx.end(), 0LL);
            std::sort(idx.begin(), idx.end(), [&](i64 a, i64 b){ return p[a] < p[b]; });
            return Tensor::from_blob(idx.data(), {M}, eScalarType::Long,
                                     Device{eDeviceType::CPU}).clone();
        };

        auto sort_idx_mag   = make_sort_idx(ref_mag);
        auto sort_idx_phase = make_sort_idx(ref_phase);

        auto srt_mag   = [&](const Tensor& t){ return t.index_select(0, sort_idx_mag); };
        auto srt_phase = [&](const Tensor& t){ return t.index_select(0, sort_idx_phase); };

        // Correlation table
        std::cout << "\n  Correlations vs S_exact(off-res+nonlin)  [L=" << L << "]\n"
                  << "  " << std::left  << std::setw(26) << "signal"
                  << " " << std::right << std::setw(10)  << "|S| Pearson"
                  << " " << std::setw(14) << "phase circ-r" << "\n"
                  << "  " << std::string(52, '-') << "\n";
        auto print_row = [&](const std::string& name, const Tensor& m, const Tensor& p) {
            std::cout << "  " << std::left << std::setw(26) << name
                      << " " << std::right << std::fixed << std::setprecision(4)
                      << std::setw(10) << pearson(ref_mag, m)
                      << " " << std::setw(14) << circ_corr(ref_phase, p) << "\n";
        };
        print_row("exact (off-res only)",  mo, po);
        print_row("exact (nonlin only)",   mn, pn);
        print_row("exact (pure DFT)",      mu, pu);
        print_row("hist_exact_dft",        mhe_pre, phe_pre);
        print_row(approx_lbl,              ma, pa);
        print_row(dft_lbl,                 md, pd);
        print_row(nw_lbl,                  mnw, pnw);
        print_row(ts_lbl,                  mts, pts);
        // Key diagnostic: circ_corr(S_hist_exact_dft, S_approx_dft)
        // isolates whether SVD Omega×Upsilon multiplication causes the 0.9174
        std::cout << "  circ_corr(hist_exact_dft, approx_dft) = " << std::fixed
                  << std::setprecision(4) << circ_corr(phe_pre, pd) << "\n";
        std::cout << "  timing: SVD=" << std::fixed << std::setprecision(2) << svd_s
                  << "s  nufft_svd=" << nsvd_s
                  << "s  approx_nufft=" << approx_s
                  << "s  nufft_w_sig=" << nw_sig_s
                  << "s  approx_dft=" << dft_s
                  << "s  time_seg=" << ts_s
                  << "s  (exact variants=" << exact_s << "s)\n\n";

        if (show_phi_error_plots || show_phi_error_vs_B0_plots || show_fft_error_plots) {

        // ── Phi approximation quality scatter plots ───────────────────────────
        // Sample one voxel per unique histogram bin — spans the full B0 range.
        // Uniform random sampling clusters near B0≈0 (most brain voxels live there)
        // → only a few distinct phi_approx rows → "4 values" in scatter.
        if (show_phi_error_plots || show_phi_error_vs_B0_plots) {
            const i64 n_phi_vox_req = 80;  // max bins/voxels to sample
            const i64 n_phi_t       = 150; // k-space time points

            const i64 N_mask_phi = hist.mask_idx.size(0);
            const i64 n_hist_phi = hist.n_hist;
            const Device dev = prob.mag.device();
            const TensorOptions opts_c{dev, eScalarType::ComplexFloat};
            const TensorOptions opts_f{dev, eScalarType::Float};
            const TensorOptions opts_l_cpu{Device{eDeviceType::CPU}, eScalarType::Long};

            // Build bin → voxel-positions-in-mask map (CPU)
            auto vtb_cpu = hist.voxel_to_bin.cpu().contiguous();
            auto vtb_view = vtb_cpu.spanning_view();
            const i64* vtb_ptr = static_cast<const i64*>(vtb_view.data);

            std::vector<std::vector<i64>> bin_to_vox(n_hist_phi);
            for (i64 v = 0; v < N_mask_phi; ++v)
                bin_to_vox[vtb_ptr[v]].push_back(v);

            std::vector<i64> nonempty_bins;
            nonempty_bins.reserve(n_hist_phi);
            for (i64 b = 0; b < n_hist_phi; ++b)
                if (!bin_to_vox[b].empty()) nonempty_bins.push_back(b);

            // Shuffle bins and take up to n_phi_vox_req of them
            std::mt19937 rng_phi(7);
            std::shuffle(nonempty_bins.begin(), nonempty_bins.end(), rng_phi);
            const i64 n_phi_vox = std::min(n_phi_vox_req, (i64)nonempty_bins.size());
            nonempty_bins.resize(n_phi_vox);

            // One random representative voxel per selected bin
            std::vector<i64> vox_buf(n_phi_vox);
            for (i64 i = 0; i < n_phi_vox; ++i) {
                i64 b = nonempty_bins[i];
                std::uniform_int_distribution<i64> dvox_b(0, (i64)bin_to_vox[b].size() - 1);
                vox_buf[i] = bin_to_vox[b][dvox_b(rng_phi)];
            }

            // Sample random time points
            std::vector<i64> t_buf(n_phi_t);
            std::uniform_int_distribution<i64> dt_dist(0, prob.K - 1);
            for (i64 i = 0; i < n_phi_t; ++i) t_buf[i] = dt_dist(rng_phi);

            auto vox_idx = Tensor::from_blob(vox_buf.data(), {n_phi_vox},
                                              eScalarType::Long, Device{eDeviceType::CPU})
                               .clone().to(dev);
            auto t_idx   = Tensor::from_blob(t_buf.data(), {n_phi_t},
                                              eScalarType::Long, Device{eDeviceType::CPU})
                               .clone().to(dev);

            auto flat_idx = hist.mask_idx.index_select(0, vox_idx);  // [n_phi_vox] flat image idx

            // ── phi_exact[n, k]: exp(-z[n]*t[k]) * Π_q exp(i*b_q[n]*alpha_q[k])
            auto z_n      = prob.z_map_flat.index_select(0, flat_idx);           // [n_phi_vox] complex
            auto t_k      = prob.timestamps.index_select(0, t_idx)
                                .to(eScalarType::ComplexFloat);                   // [n_phi_t] complex
            // [n_phi_vox, n_phi_t]: exp(-z[n] * t[k])
            auto phi_exact = (z_n.unsqueeze(1).mul(t_k.unsqueeze(0)).mul(Scalar(-1.0f))).exp();

            const Scalar pos_i{std::complex<float>(0.0f, 1.0f)};
            for (i64 q = 0; q < prob.nl_waveforms.size(0); ++q) {
                auto b_n  = prob.nl_fields_flat.select(0, q)
                                .index_select(0, flat_idx)
                                .to(eScalarType::ComplexFloat);                  // [n_phi_vox]
                auto aq_k = prob.nl_waveforms.select(0, q)
                                .index_select(0, t_idx)
                                .to(eScalarType::ComplexFloat);                  // [n_phi_t]
                phi_exact = phi_exact.mul(
                    (b_n.unsqueeze(1).mul(aq_k.unsqueeze(0)).mul(pos_i)).exp());
            }

            // ── phi_approx[n, k] = Σ_l Omega[k,l] * Upsilon[bin(n),l]
            auto bins_n    = hist.voxel_to_bin.index_select(0, vox_idx);          // [n_phi_vox]
            auto ups_n     = Upsilon.index_select(0, bins_n);                     // [n_phi_vox, L]
            auto omega_k   = Omega.index_select(0, t_idx);                        // [n_phi_t, L]
            // [n_phi_vox, n_phi_t]
            auto phi_approx = mm(ups_n, omega_k.transpose(0, 1));

            // ── phi error vs B0 (before flatten, while still [n_phi_vox, n_phi_t]) ──
            if (show_phi_error_vs_B0_plots) {
                const float pi_f_b0 = (float)std::numbers::pi_v<double>;

                // phi_exact, phi_approx: [n_phi_vox, n_phi_t] complex on device
                auto abs_err_2d = phi_exact.sub(phi_approx).abs();               // [n_phi_vox, n_phi_t]
                auto rel_err_2d = abs_err_2d.div(
                    phi_exact.abs().add(Scalar(1e-30f)));                          // [n_phi_vox, n_phi_t]

                // B0 [Hz] for each sampled voxel
                auto b0_n = prob.z_map_flat.index_select(0, flat_idx)
                                .imag().div(Scalar(2.0f * pi_f_b0))
                                .cpu().contiguous();                               // [n_phi_vox]

                // Expand b0_n to all (voxel,time) pairs — same layout as flattened phi
                auto b0_flat = b0_n.unsqueeze(1)
                                   .expand({n_phi_vox, n_phi_t})
                                   .contiguous()
                                   .reshape({-1});                                 // [n_phi_vox*n_phi_t]
                auto abs_err_flat = abs_err_2d.reshape({-1}).cpu().contiguous();
                auto rel_err_flat = rel_err_2d.reshape({-1}).cpu().contiguous();

                auto b0_sort_idx = make_sort_idx(b0_flat);
                auto by_b0 = [&](const Tensor& t){ return t.index_select(0, b0_sort_idx); };

                float mean_abs = abs_err_flat.mean().item<float>();
                float mean_rel = rel_err_flat.mean().item<float>();
                std::cout << "  phi B0-err [L=" << L << "]:  mean_abs=" << std::scientific
                          << std::setprecision(3) << mean_abs
                          << "  mean_rel=" << mean_rel
                          << "  (n=" << n_phi_vox*n_phi_t << " pairs)\n";

                hasty::viz::default_line_plots(hasty::viz::DefaultLinePlotsOptions<float>{
                    .lines    = { by_b0(abs_err_flat).spanning_view() },
                    .title    = "|φ_err| vs B0 — " + label + "  L=" + std::to_string(L),
                    .xaxis    = "(voxel,time) pair sorted by B0 Hz",
                    .yaxis    = "|φ_exact - φ_approx|",
                    .legends  = {"abs error"},
                    .markers  = true,
                    .lines_on = false,
                }).show();

                hasty::viz::default_line_plots(hasty::viz::DefaultLinePlotsOptions<float>{
                    .lines    = { by_b0(rel_err_flat).spanning_view() },
                    .title    = "rel |φ_err|/|φ| vs B0 — " + label + "  L=" + std::to_string(L),
                    .xaxis    = "(voxel,time) pair sorted by B0 Hz",
                    .yaxis    = "|φ_exact - φ_approx| / |φ_exact|",
                    .legends  = {"rel error"},
                    .markers  = true,
                    .lines_on = false,
                }).show();
            }

            // Flatten both to [n_phi_vox * n_phi_t]
            phi_exact  = phi_exact.reshape({-1}).cpu().contiguous();
            phi_approx = phi_approx.reshape({-1}).cpu().contiguous();

            if (show_phi_error_plots) {
            auto phi_mag_exact  = phi_exact.abs();
            auto phi_mag_approx = phi_approx.abs();
            auto phi_phase_exact  = [&]() -> Tensor {
                auto re = phi_exact.real().contiguous();  auto im = phi_exact.imag().contiguous();
                auto rv = re.spanning_view(); auto iv = im.spanning_view();
                const float* rp = static_cast<const float*>(rv.data);
                const float* ip = static_cast<const float*>(iv.data);
                i64 M = rv.sizes[0];
                std::vector<float> ph(M);
                for (i64 i = 0; i < M; ++i) ph[i] = std::atan2(ip[i], rp[i]);
                return Tensor::from_blob(ph.data(), {M}, eScalarType::Float,
                                         Device{eDeviceType::CPU}).clone();
            }();
            auto phi_phase_approx = [&]() -> Tensor {
                auto re = phi_approx.real().contiguous(); auto im = phi_approx.imag().contiguous();
                auto rv = re.spanning_view(); auto iv = im.spanning_view();
                const float* rp = static_cast<const float*>(rv.data);
                const float* ip = static_cast<const float*>(iv.data);
                i64 M = rv.sizes[0];
                std::vector<float> ph(M);
                for (i64 i = 0; i < M; ++i) ph[i] = std::atan2(ip[i], rp[i]);
                return Tensor::from_blob(ph.data(), {M}, eScalarType::Float,
                                         Device{eDeviceType::CPU}).clone();
            }();

            // Sort indices by exact magnitude / phase for the scatter
            auto phi_sort_mag   = make_sort_idx(phi_mag_exact);
            auto phi_sort_phase = make_sort_idx(phi_phase_exact);
            auto psm = [&](const Tensor& t){ return t.index_select(0, phi_sort_mag); };
            auto psp = [&](const Tensor& t){ return t.index_select(0, phi_sort_phase); };

            // Print phi approx quality
            float phi_phase_cr = circ_corr(phi_phase_exact, phi_phase_approx);
            float phi_mag_pear = pearson(phi_mag_exact, phi_mag_approx);
            std::cout << "  phi approx [L=" << L << "]:  |φ| Pearson="
                      << std::fixed << std::setprecision(4) << phi_mag_pear
                      << "  phase circ-r=" << phi_phase_cr
                      << "  (n_phi=" << n_phi_vox*n_phi_t << " pairs)\n";

            const std::string phi_lbl = "phi_approx (L=" + std::to_string(L) + ")";

            hasty::viz::default_line_plots(hasty::viz::DefaultLinePlotsOptions<float>{
                .lines    = { psm(phi_mag_exact).spanning_view(),
                              psm(phi_mag_approx).spanning_view() },
                .title    = "|φ(r,t)| sorted — " + label + "  L=" + std::to_string(L),
                .xaxis    = "(voxel,time) pair sorted by |φ_exact|",
                .yaxis    = "|φ|",
                .legends  = {"phi_exact", phi_lbl},
                .markers  = true,
                .lines_on = false,
            }).show();

            hasty::viz::default_line_plots(hasty::viz::DefaultLinePlotsOptions<float>{
                .lines    = { psp(phi_phase_exact).spanning_view(),
                              psp(phi_phase_approx).spanning_view() },
                .title    = "arg(φ(r,t)) sorted — " + label + "  L=" + std::to_string(L),
                .xaxis    = "(voxel,time) pair sorted by arg(φ_exact)",
                .yaxis    = "arg(φ) [rad]",
                .legends  = {"phi_exact", phi_lbl},
                .markers  = true,
                .lines_on = false,
            }).show();
            } // show_phi_error_plots
        } // show_phi_error_plots || show_phi_error_vs_B0_plots

        if (show_fft_error_plots) {
        // Magnitude scatter — x-axis sorted by |S_exact| ascending
        hasty::viz::default_line_plots(hasty::viz::DefaultLinePlotsOptions<float>{
            .lines    = { srt_mag(ref_mag).spanning_view(), srt_mag(mo).spanning_view(),
                          srt_mag(mn).spanning_view(),      srt_mag(mu).spanning_view(),
                          srt_mag(ma).spanning_view(),      srt_mag(md).spanning_view(),
                          srt_mag(mnw).spanning_view(),     srt_mag(mts).spanning_view() },
            .title    = "|S(k,t)| sorted — " + label + "  L=" + std::to_string(L),
            .xaxis    = "sample (sorted by |S_exact|)",
            .yaxis    = "|S|",
            .legends  = sig_labels,
            .markers  = true,
            .lines_on = false,
        }).show();

        // Phase scatter — x-axis sorted by arg(S_exact) ascending
        hasty::viz::default_line_plots(hasty::viz::DefaultLinePlotsOptions<float>{
            .lines    = { srt_phase(ref_phase).spanning_view(), srt_phase(po).spanning_view(),
                          srt_phase(pn).spanning_view(),        srt_phase(pu).spanning_view(),
                          srt_phase(pa).spanning_view(),        srt_phase(pd).spanning_view(),
                          srt_phase(pnw).spanning_view(),       srt_phase(pts).spanning_view() },
            .title    = "arg(S(k,t)) [rad] sorted — " + label + "  L=" + std::to_string(L),
            .xaxis    = "sample (sorted by arg(S_exact))",
            .yaxis    = "arg(S) [rad]",
            .legends  = sig_labels,
            .markers  = true,
            .lines_on = false,
        }).show();
        } // show_fft_error_plots

        if (show_signal_err_vs_mag_plot) {
        // Relative error |S_exact - S_approx| / |S_exact| vs |S_exact|, sorted by |S_exact|.
        // Reveals whether approximation errors concentrate at low-signal k-space points.
        auto rel_err = [&](const Tensor& S_approx_m) -> Tensor {
            // |S_exact - S_approx| / max(|S_exact|, eps) — avoid /0 at zero signal
            auto eps = ones_like(ref_mag).mul(Scalar(1e-8f));
            auto denom = where(ref_mag.gt(Scalar(1e-8f)), ref_mag, eps);
            return S_approx_m.sub(ref_mag).abs().div(denom);
        };
        auto re_offres = rel_err(mo);
        auto re_nonlin = rel_err(mn);
        auto re_nufft  = rel_err(mu);
        auto re_approx = rel_err(ma);
        auto re_dft    = rel_err(md);
        auto re_nw     = rel_err(mnw);
        auto re_ts     = rel_err(mts);

        const std::vector<std::string> err_labels = {
            "exact (off-res only)", "exact (nonlin only)",
            "exact (pure DFT)", approx_lbl, dft_lbl, nw_lbl, ts_lbl
        };
        hasty::viz::default_line_plots(hasty::viz::DefaultLinePlotsOptions<float>{
            .lines    = { srt_mag(re_offres).spanning_view(),
                          srt_mag(re_nonlin).spanning_view(),
                          srt_mag(re_nufft).spanning_view(),
                          srt_mag(re_approx).spanning_view(),
                          srt_mag(re_dft).spanning_view(),
                          srt_mag(re_nw).spanning_view(),
                          srt_mag(re_ts).spanning_view() },
            .title    = "rel err vs |S_exact| sorted — " + label + "  L=" + std::to_string(L),
            .xaxis    = "sample (sorted by |S_exact| ascending)",
            .yaxis    = "|S_approx - S_exact| / |S_exact|",
            .legends  = err_labels,
            .markers  = true,
            .lines_on = false,
        }).show();
        } // show_signal_err_vs_mag_plot

        } // outer plot guard
    }

    return all_pass;
}

static bool test_approx_synth(hasty::i64 n_dim, hasty::i64 n_spokes, hasty::i64 n_samp,
                               hasty::i64 n_sub, std::vector<hasty::i64> L_values,
                               hasty::i64 n_rate, hasty::i64 n_nl,
                               hasty::Device dev)
{
    std::cout << "\n[synth] n_dim=" << n_dim << "  n_spokes=" << n_spokes
              << "  n_samp=" << n_samp << "\n";
    auto prob = make_problem(n_dim, n_spokes, n_samp, dev);
    return run_test(prob, n_sub, L_values, n_rate, n_nl,
                    "synth n_dim=" + std::to_string(n_dim));
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main()
{
    bool show_locally               = false;
    bool show_fft_error_plots        = true;
    bool show_phi_error_plots        = true;
    bool show_phi_error_vs_B0_plots  = true;
    bool show_signal_err_vs_mag_plot = true;

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
        auto b0_mean = b0.mean();
        auto b0_std  = b0.std();
        std::cout << "b0 mean: " << b0_mean.item<hasty::f32>()
                  << "  std: " << b0_std.item<hasty::f32>() << "\n";

        pd = hasty::io::nifti::transform_nifti_data(pd_img);
        mask = pd[0];

        // pixdim[1..3] = x, y, z voxel spacing in mm
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
        mask = hasty::mask_dilate(std::move(mask), 2, 10, {});
        //mask = hasty::mask_erode(std::move(mask), 2, 8, {});
        mask = mask.cpu();

        auto not_mask = mask.logical_not();
        pd[hasty::Slice(),not_mask] = 0.0f;

        hasty::viz::orthoslicer(mask, {"mask_dilated", std::nullopt}, false, show_locally);

        auto b0_uuid = hasty::python::push_nifti_image(b0_img, "b0");
        auto pd_uuid = hasty::python::push_nifti_image(pd_img, "pd");

        auto result = hasty::python::run_script(
            hasty::python::scripts_dir() + "/register_nifti.py",
            {
                "--fixed="    + hasty::python::uuid_to_hex(b0_uuid),
                "--moving="   + hasty::python::uuid_to_hex(pd_uuid),
                "--transform=Rigid",
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

    /*
    // Synthetic sanity check on CUDA
    failures += !test_approx_synth(
        128, 500, 800,
        200,
        {4, 8, 12},
        512, 32,
        cuda0);
    */

    // Real-data test: full-resolution pd / b0 / mask
    {
        // pd may be 4D [n_vols, nx, ny, nz]; first volume = magnetization
        hasty::Tensor pd_3d   = (pd.ndimension() > 3) ? pd.select(0, 0) : pd;
        hasty::Tensor b0_3d   = (b0.ndimension() > 3) ? b0.select(0, 0) : b0;
        hasty::Tensor mask_3d = mask.to(hasty::eScalarType::Bool);

        std::cout << "\nReal-data shape: "
                  << pd_3d.size(0) << " x " << pd_3d.size(1)
                  << " x " << pd_3d.size(2) << "\n";

        // 250 spokes × 1250 samples = K=312,500, 5ms readout @ dt=4µs, TE=3ms start
        // b0_scale=3 → effective B0 std ~75Hz, max phase @T_end=8ms: ~3.8 rad
        auto prob_real = make_problem_real(
            pd_3d, b0_3d, mask_3d,
            pixdim_mm[0], pixdim_mm[1], pixdim_mm[2],
            500, 800,        // n_spokes, n_samp
            cuda0,
            1.0f,            // b0_scale
            3e-3f,           // te_start_s
            0.00f);           // nl_scale  (1.0 → ±π rad peak z² phase)

        failures += !run_test(prob_real,
            300,                // n_sub
            {8,25},             // joint L values (quick comparison)
            16000, 1,            // n_rate, n_nl bins
            "real_data full_res",
            hasty::mri::eBinWeighting::L1Mass,
            8,                  // Lo  (off-res SVD rank)
            3,                  // Lnl (NL SVD rank per field)
            show_fft_error_plots,
            show_phi_error_plots,
            show_phi_error_vs_B0_plots,
            show_signal_err_vs_mag_plot);
    }

    std::cout << "\n=====================================================\n"
              << "  " << failures << " failure(s)\n"
              << "=====================================================\n";
    return failures > 0 ? 1 : 0;
}
