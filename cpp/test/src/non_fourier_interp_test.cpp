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
    hasty::Tensor z_map_flat;      // [N] complex
    hasty::Tensor nl_fields_flat;  // [Q, N] float
    hasty::i64 K, N, C;
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

    // k_traj stored directly as NUFFT coords in [-π, π]:
    //   coord_d = dir_d * k_r * dx_d   (= k_d_rad_m * dx_d, Nyquist → ±π per dim)
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

        float max_alpha_z2  = alpha_z2.abs().max().item<float>();
        float max_alpha_xy  = alpha_xy.abs().max().item<float>();
        float max_field_z2  = field_z2.abs().max().item<float>();
        float max_field_xy  = field_xy.abs().max().item<float>();
        float max_phase_z2  = max_field_z2 * max_alpha_z2;
        float max_phase_xy  = max_field_xy * max_alpha_xy;

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

    return Problem{
        mag_3d, rate_map, sensitivity_maps,
        k_traj, timestamps,
        nl_waveforms, nl_basis,
        mag_flat, pd_flat, z_map_flat, nl_fields_flat,
        K, N, C
    };
}

// ── Exact paired signal via forward_exact ────────────────────────────────────

static hasty::Tensor exact_signal_paired(
    const Problem& prob,
    const hasty::Tensor& sub_idx,  // [M] long — k-space sample indices
    bool apply_ratemap = true,
    bool apply_nonlin  = true)
{
    using namespace hasty;

    auto k_b  = prob.k_traj.index_select(0, sub_idx);       // [M, 3]
    auto t_b  = prob.timestamps.index_select(0, sub_idx);   // [M]
    auto nl_b = prob.nl_waveforms.index_select(1, sub_idx); // [Q, M]

    return mri::forward_exact(
        prob.mag, prob.sensitivity_maps, prob.rate_map,
        t_b, k_b, nl_b, prob.nl_basis,
        apply_ratemap, apply_nonlin);  // [C, M]
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
// Coordinate convention — cuFINUFFT CMCL mode, sign=-1:
//   Plan {nx,ny,nz}: NufftPlan constructor reverse_copies → FINUFFT N1=nz (iz fastest = C-order fastest).
//   coords[0]=kz (paired with k1=iz), coords[1]=ky, coords[2]=kx.
//   DFTConfig centered coords (n - N/2) match CMCL modes [-N/2, N/2-1].
//   Both compute Σ img[ix,iy,iz]·exp(-i·(kx·(ix-nx/2)+ky·(iy-ny/2)+kz·(iz-nz/2))).

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

    auto rho_m = prob.mag_flat.index_select(0, mask_idx).to(eScalarType::ComplexFloat);

    // k_traj C-order: [:,0]=kx, [:,1]=ky, [:,2]=kz.
    // NUFFT Fortran layout: coords[0]→iz (nz modes) → needs kz; coords[2]→ix → needs kx.
    auto ktraj  = prob.k_traj;
    auto coords = zeros({3, K}, opts_f);
    coords.select(0, 0).copy_(ktraj.select(1, 2));  // kz → iz dim
    coords.select(0, 1).copy_(ktraj.select(1, 1));  // ky → iy dim
    coords.select(0, 2).copy_(ktraj.select(1, 0));  // kx → ix dim
    coords = coords.contiguous();

    NufftOptions<cuda_t, f32, UTN> nufft_opts;
    nufft_opts.ntransf    = 1;
    // Plan {nx,ny,nz}: constructor reverse_copies to m_nmodes={nz,ny,nx}, so FINUFFT N1=nz (iz fastest).
    NufftPlan<cuda_t, f32, 3, UTN> plan({nx, ny, nz}, nufft_opts);
    plan.setpts(coords);

    auto signal = zeros({K}, opts_c);
    auto F_l    = zeros({1, K}, opts_c).contiguous();

    for (i64 l = 0; l < L; ++l) {
        auto Upsilon_l   = Upsilon.select(1, l);
        auto Upsilon_vox = Upsilon_l.index_select(0, voxel_to_bin);
        auto src_l       = Upsilon_vox.mul(rho_m);

        auto img_flat = zeros({N}, opts_c);
        img_flat.scatter_add_(0, mask_idx, src_l);

        auto img_l = img_flat.reshape({nx, ny, nz}).contiguous().unsqueeze(0);
        plan.execute(img_l, F_l);

        signal.add_(Omega.select(1, l).mul(F_l.select(0, 0)));
    }

    return signal.unsqueeze(0);  // [1, K]
}

// ── NUFFT-based per-bin weights ───────────────────────────────────────────────
/*
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

    auto mag_masked = prob.mag_flat.index_select(0, hist.mask_idx)
                                    .to(eScalarType::ComplexFloat);

    auto ktraj  = prob.k_traj;
    auto coords = zeros({3, K}, opts_f);
    coords.select(0, 0).copy_(ktraj.select(1, 2));  // kz → iz dim
    coords.select(0, 1).copy_(ktraj.select(1, 1));  // ky → iy dim
    coords.select(0, 2).copy_(ktraj.select(1, 0));  // kx → ix dim
    coords = coords.contiguous();

    NufftOptions<cuda_t, f32, UTN> nufft_opts;
    nufft_opts.ntransf    = 1;
    NufftPlan<cuda_t, f32, 3, UTN> plan({nx, ny, nz}, nufft_opts);
    plan.setpts(coords);

    auto weights = zeros({n_hist}, opts_f);
    auto img     = zeros({1, nx, ny, nz}, opts_c).contiguous();
    auto F_out   = zeros({1, K},          opts_c).contiguous();

    for (i64 h = 0; h < n_hist; ++h) {
        img.zero_();
        auto in_bin = hist.voxel_to_bin.eq(Scalar(h)).to(eScalarType::ComplexFloat);
        img.view({N}).scatter_add_(0, hist.mask_idx, mag_masked.mul(in_bin));
        plan.execute(img, F_out);
        weights.select(0, h).copy_(F_out.abs().pow(2).sum());
    }

    return weights;
}
*/
// ── Approx signal via DFT + SVD interpolants ─────────────────────────────────
//
// S_dft[k] = Σ_l Omega[k,l] · DFT(img_l, xi[k])
//   img_l[n] = Upsilon[bin(n), l] · ρ[n]
//
// Uses the unified hasty::fft::DFTConfig / dft convention:
//   xi[k, d] = 2π k_d / N_d ∈ [−π, π]   →   same kernel as forward_exact.
// Returns [1, n_sub].

static hasty::Tensor approx_signal_dft(
    const hasty::fft::DFTConfig& cfg,    // full 3D DFT config [nx,ny,nz]
    const hasty::Tensor& Omega,           // [K, L]
    const hasty::Tensor& Upsilon,         // [n_hist, L]
    const hasty::Tensor& voxel_to_bin,   // [N_mask] long
    const hasty::Tensor& mask_idx,        // [N_mask] long
    const hasty::Tensor& rho_m,           // [N_mask] ComplexFloat
    hasty::i64 N,                         // total spatial voxels (nx*ny*nz)
    const hasty::Tensor& sub_idx,         // [n_sub] long
    const hasty::Tensor& xi_sub)          // [n_sub, d] float ∈ [−π, π]
{
    using namespace hasty;

    const i64    L     = Omega.size(1);
    const i64    n_sub = sub_idx.size(0);
    const Device dev   = rho_m.device();
    const TensorOptions opts_c{dev, eScalarType::ComplexFloat};

    auto omega_sub = Omega.index_select(0, sub_idx);   // [n_sub, L]
    auto signal    = zeros({n_sub}, opts_c);

    for (i64 l = 0; l < L; ++l) {
        auto ups_vox  = Upsilon.select(1, l).index_select(0, voxel_to_bin);  // [N_mask]
        auto src_l    = ups_vox.mul(rho_m);                                    // [N_mask]
        auto img_flat = zeros({N}, opts_c);
        img_flat.scatter_add_(0, mask_idx, src_l);
        auto img_l = img_flat.reshape(std::vector<i64>(cfg.sizes.begin(), cfg.sizes.end()));
        auto F_l   = fft::dft(cfg, img_l, xi_sub);                            // [n_sub]
        signal.add_(omega_sub.select(1, l).mul(F_l));
    }

    return signal.unsqueeze(0);  // [1, n_sub]
}

// ── Error metric ─────────────────────────────────────────────────────────────

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

static bool run_test(const Problem& prob,
                     hasty::i64 n_sub,
                     const std::vector<hasty::i64>& L_values,
                     hasty::i64 n_rate, hasty::i64 n_nl,
                     const std::string& label,
                     hasty::mri::eBinWeighting weighting = hasty::mri::eBinWeighting::L1Mass,
                     hasty::i64 P = 4,
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
    std::cout << "  n_hist=" << hist.n_hist
              << "  N_mask=" << N_mask
              << "  (" << std::fixed << std::setprecision(1)
              << (100.0f * N_mask / (float)prob.N) << "% of N)\n";

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

    auto t0_exact = std::chrono::steady_clock::now();
    auto S_both   = exact_signal_paired(prob, sub_idx, true,  true);
    auto S_offres = exact_signal_paired(prob, sub_idx, true,  false);
    auto S_nonlin = exact_signal_paired(prob, sub_idx, false, true);
    auto S_nofft  = exact_signal_paired(prob, sub_idx, false, false);
    double exact_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0_exact).count();

    // ── Signal helpers ────────────────────────────────────────────────────────

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

    // Pearson r of two [M] float CPU tensors (double precision to avoid cancellation).
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
    auto circ_corr = [](const Tensor& phase_a, const Tensor& phase_b) -> float {
        auto diff = phase_a.sub(phase_b).contiguous();
        auto dv = diff.spanning_view();
        i64 M = dv.sizes[0];
        const float* d = static_cast<const float*>(dv.data);
        double sum_cos = 0.0;
        for (i64 i = 0; i < M; ++i) sum_cos += std::cos(d[i]);
        return (float)(sum_cos / M);
    };

    auto ref_mag   = to_mag(S_both);
    auto ref_phase = to_phase(S_both);

    const i64   gx  = prob.mag.size(0);
    const i64   gy  = prob.mag.size(1);
    const i64   gz  = prob.mag.size(2);
    auto dft_cfg_full = fft::make_dft_config({gx, gy, gz}, /*batch_size=*/1, prob.mag.device());
    auto xi_sub    = prob.k_traj.index_select(0, sub_idx).contiguous();  // [n_sub, 3]
    auto rho_m_dft = prob.mag_flat.index_select(0, hist.mask_idx).to(eScalarType::ComplexFloat);

    std::cout << "\n  L   n_hist   phase_corr_err   status\n"
              << "  " << std::string(44, '-') << "\n";

    bool all_pass = true;

    for (auto L : L_values) {
        const i64 mpd_val = std::max((i64)8*L, (i64)60);
        const i64 ncv_val = L + mpd_val;

        auto weights = mri::phi_lowrank_weights(weighting, hist, prob.timestamps);

        auto t0_svd = std::chrono::steady_clock::now();
        auto [Omega, S_phi, Upsilon] = mri::phi_lowrank(op, L, weights, ncv_val, mpd_val);
        double svd_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_svd).count();

        // Print singular values for rank-selection guidance.
        {
            auto sv = S_phi.cpu().contiguous();
            const float* sp = static_cast<const float*>(sv.spanning_view().data);
            std::cout << "  phi singular values [L=" << L << "]:";
            for (i64 l = 0; l < L; ++l)
                std::cout << " " << std::scientific << std::setprecision(2) << sp[l];
            std::cout << "\n";
        }

        // ── Normal operator consistency (naive L² vs P×R) ─────────────────────
        {
            using namespace mri;
            const i64 nx = prob.mag.size(0), ny = prob.mag.size(1), nz = prob.mag.size(2);
            const Device dev = prob.mag.device();
            const TensorOptions opts_f{dev, eScalarType::Float};
            const TensorOptions opts_c{dev, eScalarType::ComplexFloat};

            auto ktraj  = prob.k_traj;
            auto coords = zeros({3, prob.K}, opts_f);
            coords.select(0,0).copy_(ktraj.select(1,2));
            coords.select(0,1).copy_(ktraj.select(1,1));
            coords.select(0,2).copy_(ktraj.select(1,0));
            coords = coords.contiguous();

            PhiLowrankResult plr{Omega, S_phi, Upsilon};
            const i64 p_use = std::min(P, L*L);

            auto rho_r = rand({nx, ny, nz}, opts_f);
            auto rho_i = rand({nx, ny, nz}, opts_f);
            auto rho   = view_as_complex(stack({rho_r, rho_i}, -1).contiguous());
            auto mask_3d = zeros({nx*ny*nz}, TensorOptions(dev, eScalarType::Bool));
            mask_3d.scatter_(0, hist.mask_idx,
                ones({hist.mask_idx.size(0)}, TensorOptions(dev, eScalarType::Bool)));
            rho = rho.reshape({nx*ny*nz}).masked_fill(mask_3d.logical_not(), Scalar(0.f))
                     .reshape({nx, ny, nz});

            auto t0n = std::chrono::steady_clock::now();
            auto emb_naive = make_normal_naive_off_fourier_toeplitz_embeddings(
                coords, {nx, ny, nz}, hist.mask_idx, hist.voxel_to_bin, plr,
                eComputeStrategyBuildOffFourierEmbeddings::RUN_ALL_ON_INPUT_DEVICE,
                eStorageStrategyBuildOffFourierEmbeddings::STORE_IN_FILE);
            double t_nb = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0n).count();

            t0n = std::chrono::steady_clock::now();
            auto emb_nh = make_normal_nonhermitian_off_fourier_toeplitz_embeddings(
                coords, {nx, ny, nz}, hist.mask_idx, hist.voxel_to_bin, plr, p_use,
                /*n_als_iter=*/10,
                eComputeStrategyBuildOffFourierEmbeddings::RUN_ALL_ON_INPUT_DEVICE,
                eStorageStrategyBuildOffFourierEmbeddings::STORE_IN_FILE);
            double t_nhb = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0n).count();

            t0n = std::chrono::steady_clock::now();
            auto out_naive = apply_normal_toeplitz_off_fourier_operator(emb_naive, rho);
            double t_na = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0n).count();

            t0n = std::chrono::steady_clock::now();
            auto out_nh = apply_normal_toeplitz_off_fourier_operator(emb_nh, rho);
            double t_nha = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0n).count();

            // Compare non-hermitian (P splits) against naive L² at masked voxels.
            auto flat_ref = out_naive.reshape({nx*ny*nz}).index_select(0, hist.mask_idx).cpu().contiguous();
            auto flat_nh  = out_nh   .reshape({nx*ny*nz}).index_select(0, hist.mask_idx).cpu().contiguous();

            auto mean_rel_err = [&](const Tensor& approx) -> float {
                auto err = flat_ref.sub(approx).abs().contiguous();
                auto ref = flat_ref.abs().contiguous();
                const i64 N_m = err.size(0);
                const float* ep = static_cast<const float*>(err.spanning_view().data);
                const float* rp = static_cast<const float*>(ref.spanning_view().data);
                double sum = 0.0; int cnt = 0;
                for (i64 i = 0; i < N_m; ++i) {
                    float r = rp[i] > 1e-30f ? ep[i] / rp[i] : 0.0f;
                    if (std::isfinite(r)) { sum += r; ++cnt; }
                }
                return cnt ? (float)(sum / cnt) : 0.0f;
            };

            float mre_nh = mean_rel_err(flat_nh);

            std::cout << "\n  Normal op [L=" << L << "]  (ref=naïve L²)\n"
                      << "    non-hermitian CP-ALS (P=" << p_use << "):"
                      << "  mean_rel_err=" << std::scientific << std::setprecision(3) << mre_nh << "\n"
                      << "  timing build:  naive=" << std::setprecision(2) << t_nb
                      << "s  nh=" << t_nhb << "s\n"
                      << "  timing apply:  naive=" << t_na
                      << "s  nh=" << t_nha << "s\n";

            // Sorted rel-err plot: non-hermitian vs naive L² reference.
            {
                auto err = flat_ref.sub(flat_nh).abs().contiguous();
                auto ref = flat_ref.abs().contiguous();
                const i64 N_m = err.size(0);
                const float* ep = static_cast<const float*>(err.spanning_view().data);
                const float* rp = static_cast<const float*>(ref.spanning_view().data);
                std::vector<float> rv;  rv.reserve(N_m);
                for (i64 i = 0; i < N_m; ++i) {
                    float r = rp[i] > 1e-30f ? ep[i] / rp[i] : 0.0f;
                    if (std::isfinite(r)) rv.push_back(r);
                }
                std::sort(rv.begin(), rv.end());
                const i64 np = std::min((i64)2000, (i64)rv.size());
                std::vector<float> pv(np);
                for (i64 i = 0; i < np; ++i) pv[i] = rv[(i64)rv.size() * i / np];
                auto pt = Tensor::from_blob(pv.data(), {np}, eScalarType::Float,
                              Device{eDeviceType::CPU}).clone();
                hasty::viz::default_line_plots(hasty::viz::DefaultLinePlotsOptions<float>{
                    .lines    = { pt.spanning_view() },
                    .title    = "Normal op rel err (sorted) — NH P=" + std::to_string(p_use) +
                                 " L=" + std::to_string(L),
                    .xaxis    = "voxel percentile",
                    .yaxis    = "|out_ref - out_nh| / |out_ref|",
                    .legends  = {"non-hermitian CP-ALS"},
                    .markers  = true,
                    .lines_on = false,
                }).show();
            }

            hasty::viz::orthoslicer(out_naive.abs(), {"normal_naive L²=" + std::to_string(L), std::nullopt}, true, false);
            hasty::viz::orthoslicer(out_nh.abs(),    {"normal_NH P=" + std::to_string(p_use) + " L=" + std::to_string(L), std::nullopt}, true, false);
        }

        // phi_lowrank returns raw U (no S); absorb S into Omega for forward signal:
        // S[k] = sum_l (U[k,l]*S[l]) * NUFFT(Vh^T[:,l]*rho)[k]
        auto Omega_full = Omega.mul(S_phi.unsqueeze(0));  // [K, L]

        auto t0_approx = std::chrono::steady_clock::now();
        auto S_approx_full = approx_signal_nufft(
            prob, Omega_full, Upsilon, hist.voxel_to_bin, hist.mask_idx);
        double approx_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_approx).count();
        auto S_approx = S_approx_full.index_select(1, sub_idx);

        float err  = phase_corrected_rel_err(S_both, S_approx);
        bool  pass = err < 0.05f;
        all_pass   = all_pass && pass;

        std::cout << "  " << std::setw(3) << L
                  << "  " << std::setw(7) << hist.n_hist
                  << "  " << std::scientific << std::setprecision(3) << err
                  << "  " << (pass ? "PASS" : "FAIL") << "\n";

        auto t0_dft = std::chrono::steady_clock::now();
        auto S_dft  = approx_signal_dft(
            dft_cfg_full, Omega_full, Upsilon, hist.voxel_to_bin, hist.mask_idx,
            rho_m_dft, prob.N, sub_idx, xi_sub);
        double dft_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0_dft).count();

        auto mo = to_mag(S_offres);  auto po = to_phase(S_offres);
        auto mn = to_mag(S_nonlin);  auto pn = to_phase(S_nonlin);
        auto mu = to_mag(S_nofft);   auto pu = to_phase(S_nofft);
        auto ma = to_mag(S_approx);  auto pa = to_phase(S_approx);
        auto md = to_mag(S_dft);     auto pd = to_phase(S_dft);

        const std::string approx_lbl = "approx NUFFT (L=" + std::to_string(L) + ")";
        const std::string dft_lbl    = "approx DFT   (L=" + std::to_string(L) + ")";
        const std::vector<std::string> sig_labels = {
            "exact (off-res+nonlin)", "exact (off-res only)",
            "exact (nonlin only)",    "exact (no off-res/nonlin)",
            approx_lbl, dft_lbl
        };

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
            print_row("exact (off-res only)",      mo, po);
            print_row("exact (nonlin only)",        mn, pn);
            print_row("exact (no off-res/nonlin)",  mu, pu);
            print_row(approx_lbl,                   ma, pa);
            print_row(dft_lbl,                      md, pd);
            std::cout << "  timing: SVD=" << std::fixed << std::setprecision(2) << svd_s
                      << "s  approx_nufft=" << approx_s
                      << "s  approx_dft=" << dft_s
                      << "s  (exact variants=" << exact_s << "s)\n\n";

            if (show_phi_error_plots || show_phi_error_vs_B0_plots || show_fft_error_plots) {

            if (show_phi_error_plots || show_phi_error_vs_B0_plots) {
                const i64 n_phi_vox_req = 80;
                const i64 n_phi_t       = 150;

                const i64 N_mask_phi = hist.mask_idx.size(0);
                const i64 n_hist_phi = hist.n_hist;
                const Device dev = prob.mag.device();
                const TensorOptions opts_c{dev, eScalarType::ComplexFloat};
                const TensorOptions opts_f{dev, eScalarType::Float};

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

                std::mt19937 rng_phi(7);
                std::shuffle(nonempty_bins.begin(), nonempty_bins.end(), rng_phi);
                const i64 n_phi_vox = std::min(n_phi_vox_req, (i64)nonempty_bins.size());
                nonempty_bins.resize(n_phi_vox);

                std::vector<i64> vox_buf(n_phi_vox);
                for (i64 i = 0; i < n_phi_vox; ++i) {
                    i64 b = nonempty_bins[i];
                    std::uniform_int_distribution<i64> dvox_b(0, (i64)bin_to_vox[b].size() - 1);
                    vox_buf[i] = bin_to_vox[b][dvox_b(rng_phi)];
                }

                std::vector<i64> t_buf(n_phi_t);
                std::uniform_int_distribution<i64> dt_dist(0, prob.K - 1);
                for (i64 i = 0; i < n_phi_t; ++i) t_buf[i] = dt_dist(rng_phi);

                auto vox_idx = Tensor::from_blob(vox_buf.data(), {n_phi_vox},
                                                  eScalarType::Long, Device{eDeviceType::CPU})
                                   .clone().to(dev);
                auto t_idx   = Tensor::from_blob(t_buf.data(), {n_phi_t},
                                                  eScalarType::Long, Device{eDeviceType::CPU})
                                   .clone().to(dev);

                auto flat_idx = hist.mask_idx.index_select(0, vox_idx);

                auto z_n      = prob.z_map_flat.index_select(0, flat_idx);
                auto t_k      = prob.timestamps.index_select(0, t_idx).to(eScalarType::ComplexFloat);
                auto phi_exact = (z_n.unsqueeze(1).mul(t_k.unsqueeze(0)).mul(Scalar(-1.0f))).exp();

                const Scalar pos_i{std::complex<float>(0.0f, 1.0f)};
                for (i64 q = 0; q < prob.nl_waveforms.size(0); ++q) {
                    auto b_n  = prob.nl_fields_flat.select(0, q)
                                    .index_select(0, flat_idx)
                                    .to(eScalarType::ComplexFloat);
                    auto aq_k = prob.nl_waveforms.select(0, q)
                                    .index_select(0, t_idx)
                                    .to(eScalarType::ComplexFloat);
                    phi_exact = phi_exact.mul(
                        (b_n.unsqueeze(1).mul(aq_k.unsqueeze(0)).mul(pos_i)).exp());
                }

                auto bins_n    = hist.voxel_to_bin.index_select(0, vox_idx);
                auto ups_n     = Upsilon.index_select(0, bins_n);
                auto omega_k   = Omega_full.index_select(0, t_idx);
                auto phi_approx = mm(ups_n, omega_k.transpose(0, 1));

                if (show_phi_error_vs_B0_plots) {
                    const float pi_f_b0 = (float)std::numbers::pi_v<double>;

                    auto abs_err_2d = phi_exact.sub(phi_approx).abs();
                    auto rel_err_2d = abs_err_2d.div(phi_exact.abs().add(Scalar(1e-30f)));

                    auto b0_n = prob.z_map_flat.index_select(0, flat_idx)
                                    .imag().div(Scalar(2.0f * pi_f_b0))
                                    .cpu().contiguous();

                    auto b0_flat = b0_n.unsqueeze(1)
                                       .expand({n_phi_vox, n_phi_t})
                                       .contiguous().reshape({-1});
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

                auto phi_sort_mag   = make_sort_idx(phi_mag_exact);
                auto phi_sort_phase = make_sort_idx(phi_phase_exact);
                auto psm = [&](const Tensor& t){ return t.index_select(0, phi_sort_mag); };
                auto psp = [&](const Tensor& t){ return t.index_select(0, phi_sort_phase); };

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
            hasty::viz::default_line_plots(hasty::viz::DefaultLinePlotsOptions<float>{
                .lines    = { srt_mag(ref_mag).spanning_view(), srt_mag(mo).spanning_view(),
                              srt_mag(mn).spanning_view(),      srt_mag(mu).spanning_view(),
                              srt_mag(ma).spanning_view(),      srt_mag(md).spanning_view() },
                .title    = "|S(k,t)| sorted — " + label + "  L=" + std::to_string(L),
                .xaxis    = "sample (sorted by |S_exact|)",
                .yaxis    = "|S|",
                .legends  = sig_labels,
                .markers  = true,
                .lines_on = false,
            }).show();

            hasty::viz::default_line_plots(hasty::viz::DefaultLinePlotsOptions<float>{
                .lines    = { srt_phase(ref_phase).spanning_view(), srt_phase(po).spanning_view(),
                              srt_phase(pn).spanning_view(),        srt_phase(pu).spanning_view(),
                              srt_phase(pa).spanning_view(),        srt_phase(pd).spanning_view() },
                .title    = "arg(S(k,t)) [rad] sorted — " + label + "  L=" + std::to_string(L),
                .xaxis    = "sample (sorted by arg(S_exact))",
                .yaxis    = "arg(S) [rad]",
                .legends  = sig_labels,
                .markers  = true,
                .lines_on = false,
            }).show();
            } // show_fft_error_plots

            if (show_signal_err_vs_mag_plot) {
            auto rel_err = [&](const Tensor& S_approx_m) -> Tensor {
                auto eps = ones_like(ref_mag).mul(Scalar(1e-8f));
                auto denom = where(ref_mag.gt(Scalar(1e-8f)), ref_mag, eps);
                return S_approx_m.sub(ref_mag).abs().div(denom);
            };
            const std::vector<std::string> err_labels = {
                "exact (off-res only)", "exact (nonlin only)",
                "exact (no off-res/nonlin)", approx_lbl, dft_lbl
            };
            hasty::viz::default_line_plots(hasty::viz::DefaultLinePlotsOptions<float>{
                .lines    = { srt_mag(rel_err(mo)).spanning_view(),
                              srt_mag(rel_err(mn)).spanning_view(),
                              srt_mag(rel_err(mu)).spanning_view(),
                              srt_mag(rel_err(ma)).spanning_view(),
                              srt_mag(rel_err(md)).spanning_view() },
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

// ── Main ──────────────────────────────────────────────────────────────────────

int non_fourier_interp_test()
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

    {
        hasty::Tensor pd_3d   = (pd.ndimension() > 3) ? pd.select(0, 0) : pd;
        hasty::Tensor b0_3d   = (b0.ndimension() > 3) ? b0.select(0, 0) : b0;
        hasty::Tensor mask_3d = mask.to(hasty::eScalarType::Bool);

        std::cout << "\nReal-data shape: "
                  << pd_3d.size(0) << " x " << pd_3d.size(1)
                  << " x " << pd_3d.size(2) << "\n";

        auto prob_real = make_problem_real(
            pd_3d, b0_3d, mask_3d,
            pixdim_mm[0], pixdim_mm[1], pixdim_mm[2],
            500, 800,
            cuda0,
            0.5f,
            1e-3f,
            0.00f);

        failures += !run_test(prob_real,
            300,
            {8},
            14000, 1,
            "real_data full_res",
            hasty::mri::eBinWeighting::L1Mass,
            /*P=*/12,
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

void nufft_dft_consistency_test()
{
    using namespace hasty;
    using namespace hasty::fft;
    using namespace std::numbers;

    const i64 nx = 224, ny = 320, nz = 280;
    const i64 M  = 200;

    auto cuda_dev = Device{eDeviceType::CUDA, 0};
    auto cpu_dev  = Device{eDeviceType::CPU};

    const TensorOptions opts_f{cuda_dev, eScalarType::Float};
    const TensorOptions opts_c{cuda_dev, eScalarType::ComplexFloat};

    std::cout << "NUFFT/DFT consistency test\n"
              << "  image: " << nx << "x" << ny << "x" << nz
              << "  M=" << M << " random frequencies\n\n";

    // Random complex image
    auto img_r = rand({nx, ny, nz}, opts_f);
    auto img_i = rand({nx, ny, nz}, opts_f);
    auto img   = view_as_complex(stack({img_r, img_i}, -1).contiguous());  // [nx,ny,nz] ComplexFloat

    // Random frequencies in (-π, π) per dim
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> udist(
        -(float)pi_v<double> * 0.99f, (float)pi_v<double> * 0.99f);
    std::vector<float> xi_buf(M * 3);
    for (auto& v : xi_buf) v = udist(rng);
    // xi[:,0]=kx, [:,1]=ky, [:,2]=kz — C-order matching DFT cfg.coords dims [ix,iy,iz]
    auto xi = Tensor::from_blob(xi_buf.data(), {M, 3}, eScalarType::Float, cpu_dev)
                  .clone().to(cuda_dev);

    // ── DFT ──────────────────────────────────────────────────────────────────
    // Centered coords [-N/2, N/2-1], xi=[kx,ky,kz], batch_size=1
    auto dft_cfg = make_dft_config({nx, ny, nz}, /*batch_size=*/1, cuda_dev);
    auto t0_dft  = std::chrono::steady_clock::now();
    auto F_dft   = dft(dft_cfg, img, xi);   // [M]
    double dft_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0_dft).count();

    // ── NUFFT ────────────────────────────────────────────────────────────────
    // Plan {nx,ny,nz}: constructor reverse_copies → FINUFFT N1=nz (iz fastest = C-order fastest).
    // coords[0]=kz (paired with k1=iz), [1]=ky, [2]=kx.
    auto coords = zeros({3, M}, opts_f);
    coords.select(0, 0).copy_(xi.select(1, 2));   // kz
    coords.select(0, 1).copy_(xi.select(1, 1));   // ky
    coords.select(0, 2).copy_(xi.select(1, 0));   // kx
    coords = coords.contiguous();

    NufftOptions<cuda_t, f32, UTN> nufft_opts;
    nufft_opts.ntransf = 1;
    NufftPlan<cuda_t, f32, 3, UTN> plan({nx, ny, nz}, nufft_opts);
    plan.setpts(coords);

    auto img_batch   = img.unsqueeze(0).contiguous();   // [1, nx, ny, nz]
    auto F_nufft_out = zeros({1, M}, opts_c).contiguous();
    auto t0_nufft    = std::chrono::steady_clock::now();
    plan.execute(img_batch, F_nufft_out);
    double nufft_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0_nufft).count();
    auto F_nufft = F_nufft_out.select(0, 0);   // [M]

    // ── DC sanity ────────────────────────────────────────────────────────────
    {
        auto dc_coords = zeros({3, 1}, opts_f);
        NufftPlan<cuda_t, f32, 3, UTN> dc_plan({nx, ny, nz}, nufft_opts);
        dc_plan.setpts(dc_coords);
        auto dc_out = zeros({1, 1}, opts_c).contiguous();
        dc_plan.execute(img_batch, dc_out);
        float dc_nufft = dc_out.real().item<float>();

        auto xi_dc  = zeros({1, 3}, opts_f);
        auto F_dc   = dft(dft_cfg, img, xi_dc);
        float dc_dft  = F_dc.real().item<float>();
        float dc_true = img.real().sum().item<float>();

        std::cout << "  DC sanity:\n"
                  << "    sum(img)     = " << std::fixed << std::setprecision(2) << dc_true  << "\n"
                  << "    NUFFT(0,0,0) = " << dc_nufft << "  |err|=" << std::abs(dc_nufft - dc_true) << "\n"
                  << "    DFT(0,0,0)   = " << dc_dft   << "  |err|=" << std::abs(dc_dft   - dc_true) << "\n\n";
    }

    // ── Compare ──────────────────────────────────────────────────────────────
    auto abs_err       = F_dft.sub(F_nufft).abs();
    float mean_abs_err = abs_err.mean().item<float>();
    float max_abs_err  = abs_err.max().item<float>();
    float ref_norm     = F_dft.abs().mean().item<float>();
    float rel_err      = mean_abs_err / (ref_norm + 1e-30f);

    auto phase_diff = F_dft.angle().sub(F_nufft.angle());
    float circ_corr = phase_diff.cos().mean().item<float>();

    std::cout << "  NUFFT vs DFT (" << M << " random freqs):\n"
              << "    mean |err|   = " << std::scientific << std::setprecision(3) << mean_abs_err << "\n"
              << "    max  |err|   = " << max_abs_err << "\n"
              << "    rel err      = " << rel_err << "\n"
              << "    phase circ-r = " << std::fixed << std::setprecision(6) << circ_corr << "\n\n"
              << "  timing: DFT=" << std::setprecision(2) << dft_s
              << "s  NUFFT=" << nufft_s << "s\n\n";

    bool pass = (circ_corr > 0.9999f) && (rel_err < 1e-3f);
    std::cout << (pass ? "PASS" : "FAIL") << "\n";
}


int main()
{
    hasty::InferenceMode im;

    hasty::io::setup_default_dirs();

    //nufft_dft_consistency_test();

    return non_fourier_interp_test();
}
