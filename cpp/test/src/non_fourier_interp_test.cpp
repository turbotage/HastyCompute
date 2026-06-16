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

    // ── warp-validation extras (Phase 1, see notes/MRI_Physics.tex) ──────────
    // Channel Q-1 (field_y2 = y², alpha_y2 = -c_y2*k_y/dy_m) is separable:
    // field_y2(r)*alpha_y2(t) = -k_rad_per_m_y(t)*c_y2*y² = phase shift of
    // exp(i*…) combined with exp(-i*k·r) gives exp(-i*k·u(r)), u_y=y+c_y2*y².
    hasty::Tensor coords_pix_flat; // [N, 3] float — integer-centered pixel coords (ix-nx/2, no +0.5 voxel offset)
    float         nl_y2_c_pix;    // c_y2 * dy_m (dimensionless warp coefficient in pixel space)
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
    const i64 Q  = 4;  // field_z2 (concomitant), field_xy (concomitant), field_xz (concomitant), field_y2 (separable GNL → warp)
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

    // INTEGER-centered pixel-index coords matching fft::dft / forward_exact:
    //   n_d = i_d - N_d/2   (no +0.5 voxel-center offset)
    // This makes exp(-i·k·n_d) == fft::dft's phase kernel exactly.
    // The GNL warp displacement uses (pix_y + 0.5) = phys_y/dy_m as the
    // field argument (see run_test_warp), so u_y = n_y + c_y2_pix*(n_y+0.5)².
    auto pix_x = phys_x.div(Scalar(dx_m)).sub(Scalar(0.5f));   // ix - nx/2
    auto pix_y = phys_y.div(Scalar(dy_m)).sub(Scalar(0.5f));   // iy - ny/2
    auto pix_z = phys_z.div(Scalar(dz_m)).sub(Scalar(0.5f));   // iz - nz/2
    auto coords_pix_flat = stack({pix_x, pix_y, pix_z}, 1).contiguous();  // [N, 3]

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

    // Concomitant-like fields: spatial maps whose waveforms are NOT proportional
    // to any single k_d(t), so they cannot be absorbed into a coordinate warp.
    // field_z2: z²  — waveform ∝ dz²·g_bip (proportional to kz(t) would require
    //   spoke factor dz alone, but we use dz² so alpha ∝ cumsum(dz²·G), NOT ∝ kz).
    // field_xy: x·y — waveform ∝ dx·dy·g_bip, cross-product → non-separable.
    // field_xz: x·z — waveform ∝ dx·dz·g_bip, cross-product → non-separable.
    //   (Represents a concomitant B_con ∝ Gx(t)·Gz(t)·x·z / B0 type coupling.)
    auto field_z2 = phys_z.mul(phys_z);
    auto field_xy = phys_x.mul(phys_y);
    auto field_xz = phys_x.mul(phys_z);   // new concomitant channel

    // Warp-separable GNL channel: field_y2 = y² (in m²), with alpha_y2(t) chosen
    // (below, once k_traj exists) proportional to k_y(t) so that
    // field_y2(r)*alpha_y2(t) == k_y(t)*(c_y2*y²) == k(t)·(u(r)-r) for
    // u_y(r) = y + c_y2*y². Coefficient c_y2 set so the Jacobian perturbation
    // |2*c_y2*y| <= 0.3 over the FOV (det J_u = 1 + 2*c_y2*y ∈ [0.7, 1.3]).
    auto field_y2      = phys_y.mul(phys_y);
    auto field_y2_grad = phys_y.mul(Scalar(2.0f));   // d(y²)/dy
    const float c_y2   = 0.03f / FOV_y;  // realistic: ~7.5 rad max GNL phase (was 0.3 → 75 rad)

    auto nl_basis       = stack({field_z2.reshape({nx, ny, nz}),
                                  field_xy.reshape({nx, ny, nz}),
                                  field_xz.reshape({nx, ny, nz}),
                                  field_y2.reshape({nx, ny, nz})}, 0);
    auto nl_fields_flat = stack({field_z2, field_xy, field_xz, field_y2}, 0);

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
    // G_z2_amp: peak z² phase = ±π·nl_scale rad (polar spoke, edge voxel).
    // G_xy_amp / G_xz_amp: peak x·y and x·z phase = ±0.5π·nl_scale rad.
    // G_xz waveform uses spoke factor dx·dz: NOT proportional to kx, ky, or kz
    // for a general 3D spoke direction → non-separable concomitant coupling.
    i64   half     = n_samp / 2;
    float z_sq_max = 0.5f * FOV_z * 0.5f * FOV_z;
    float xy_max   = 0.5f * FOV_x * 0.5f * FOV_y;
    float xz_max   = 0.5f * FOV_x * 0.5f * FOV_z;
    float G_z2_amp = nl_scale * 0.5f * 2.0f * (float)pi_v<f64> / (gamma * dt * (float)half * z_sq_max);
    float G_xy_amp = nl_scale * 0.5f * (float)pi_v<f64>         / (gamma * dt * (float)half * xy_max  * 0.5f);
    float G_xz_amp = nl_scale * 0.5f * (float)pi_v<f64>         / (gamma * dt * (float)half * xz_max  * 0.5f);

    std::vector<float> g_bip(n_samp);
    for (i64 j = 0; j < n_samp; ++j) g_bip[j] = j < half ? 1.0f : -1.0f;

    std::vector<float> G_z2(K), G_xy(K), G_xz(K);
    for (i64 s = 0; s < n_spokes; ++s) {
        float dz2 = dirs[s*3+2] * dirs[s*3+2];
        float dxy = dirs[s*3+0] * dirs[s*3+1];
        float dxz = dirs[s*3+0] * dirs[s*3+2];   // cross-product → non-separable
        for (i64 j = 0; j < n_samp; ++j) {
            G_z2[s*n_samp+j] = G_z2_amp * dz2 * g_bip[j];
            G_xy[s*n_samp+j] = G_xy_amp * dxy * g_bip[j];
            G_xz[s*n_samp+j] = G_xz_amp * dxz * g_bip[j];
        }
    }
    auto G_z2_t = Tensor::from_blob(G_z2.data(), {K}, eScalarType::Float, cpu).clone().to(dev);
    auto G_xy_t = Tensor::from_blob(G_xy.data(), {K}, eScalarType::Float, cpu).clone().to(dev);
    auto G_xz_t = Tensor::from_blob(G_xz.data(), {K}, eScalarType::Float, cpu).clone().to(dev);

    auto alpha_z2 = G_z2_t.to(eScalarType::Double).cumsum(0).to(eScalarType::Float).mul(Scalar(-gamma * dt));
    auto alpha_xy = G_xy_t.to(eScalarType::Double).cumsum(0).to(eScalarType::Float).mul(Scalar(-gamma * dt));
    auto alpha_xz = G_xz_t.to(eScalarType::Double).cumsum(0).to(eScalarType::Float).mul(Scalar(-gamma * dt));

    // Warp-separable waveform: alpha_y2(t) = -c_y2 * k_y(t) / dy_m  [rad/m²]
    // So that i*field_y2*alpha_y2 = -i*k_rad_per_m_y*c_y2*y² = -i*k·(u(r)-r),
    // combining with exp(-i*k·r) gives exp(-i*k·u(r)) exactly.
    const float c_y2_pix = c_y2 * dy_m;  // dimensionless warp coeff in pixel space
    auto alpha_y2 = k_traj.select(1, 1).div(Scalar(dy_m)).mul(Scalar(-c_y2)).contiguous();

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
        float max_alpha_xz  = alpha_xz.abs().max().item<float>();
        float max_field_z2  = field_z2.abs().max().item<float>();
        float max_field_xy  = field_xy.abs().max().item<float>();
        float max_field_xz  = field_xz.abs().max().item<float>();
        float max_phase_z2  = max_field_z2 * max_alpha_z2;
        float max_phase_xy  = max_field_xy * max_alpha_xy;
        float max_phase_xz  = max_field_xz * max_alpha_xz;

        float max_alpha_y2  = alpha_y2.abs().max().item<float>();
        float max_field_y2  = field_y2.abs().max().item<float>();
        float max_phase_y2  = max_field_y2 * max_alpha_y2;
        auto jac_in_mask = field_y2_grad.masked_select(mask_bool).mul(Scalar(c_y2)).add(Scalar(1.0f));
        float jac_min = jac_in_mask.min().item<float>();
        float jac_max = jac_in_mask.max().item<float>();

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
                  << "  Concomitant fields (nl_scale=" << nl_scale << ", bipolar, non-separable):\n"
                  << "    z²: max|α_z2|=" << std::scientific << std::setprecision(3)
                  << max_alpha_z2 << " rad/m²"
                  << "  max|b_z2|=" << max_field_z2 << " m²"
                  << "  → max phase=" << std::fixed << std::setprecision(2) << max_phase_z2 << " rad\n"
                  << "    x·y: max|α_xy|=" << std::scientific << std::setprecision(3)
                  << max_alpha_xy << " rad/m²"
                  << "  max|b_xy|=" << max_field_xy << " m²"
                  << "  → max phase=" << std::fixed << std::setprecision(2) << max_phase_xy << " rad\n"
                  << "    x·z: max|α_xz|=" << std::scientific << std::setprecision(3)
                  << max_alpha_xz << " rad/m²"
                  << "  max|b_xz|=" << max_field_xz << " m²"
                  << "  → max phase=" << std::fixed << std::setprecision(2) << max_phase_xz << " rad\n"
                  << "\n"
                  << "  Warp-separable channel (y², alpha_y2 = c_y2·k_y, c_y2="
                  << std::scientific << std::setprecision(3) << c_y2 << "):\n"
                  << "    max|α_y2|=" << max_alpha_y2 << " rad/m²"
                  << "  max|b_y2|=" << max_field_y2 << " m²"
                  << "  → max phase=" << std::fixed << std::setprecision(3) << max_phase_y2 << " rad\n"
                  << "    det(J_u) = 1 + 2·c_y2·y  ∈ [" << std::setprecision(3)
                  << jac_min << ", " << jac_max << "]  (diffeomorphism "
                  << ((jac_min > 0.0f) ? "OK" : "VIOLATED") << ")\n"
                  << "  ─────────────────────────────────────────────────────────\n\n";
    }

    auto nl_waveforms = stack({alpha_z2, alpha_xy, alpha_xz, alpha_y2}, 0);

    return Problem{
        mag_3d, rate_map, sensitivity_maps,
        k_traj, timestamps,
        nl_waveforms, nl_basis,
        mag_flat, pd_flat, z_map_flat, nl_fields_flat,
        K, N, C,
        coords_pix_flat, c_y2_pix
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

// ── Core-space CP-rank-P fit diagnostic ────────────────────────────────────────
//
// T[hj,hi,k] = G[hj,k]*conj(G[hi,k]) with G = Upsilon @ diag(S) @ Omega^T.
// Since Upsilon has orthonormal columns, CP-rank-P approximation of T is
// equivalent (same Frobenius error) to CP-rank-P approximation of the tiny
// core tensor M[l,l',k] = W[k,l]*conj(W[k,l']), W = Omega .* S  [K,L].
// This lets us evaluate ALS variants in milliseconds instead of minutes.

// B [rows,P] @ pinv(G) via eigendecomposition. Truncated (Moore-Penrose) pinv:
// eigenvalues below rel_eps*d_max are treated as exact null-space (d_inv=0).
// This is the minimum-norm least-squares solution for rank-deficient G,
// the correct block-coordinate-descent step (non-increasing objective).
static hasty::Tensor cpcore_rsolve_trunc(const hasty::Tensor& B, const hasty::Tensor& G, double rel_eps = 1e-9)
{
    using namespace hasty;
    auto eig = linalg_eigh(G);
    double d_max    = eig.eigenvalues.abs().max().item<double>();
    double d_thresh = d_max * rel_eps + 1e-300;
    auto mask  = eig.eigenvalues.gt(Scalar(d_thresh)).to(eScalarType::Double);
    auto d_inv = (mask / eig.eigenvalues.clamp_min(d_thresh))
                     .to(eScalarType::ComplexDouble)
                     .unsqueeze(0);
    auto BV = mm(B, eig.eigenvectors);
    return mm(BV.mul(d_inv), eig.eigenvectors.conj().transpose(0,1).contiguous());
}

// Relative Frobenius error ||M - Mhat||_F / ||M||_F via gram-trace identities
// (avoids materialising the [L,L,K] tensor).
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

// Fast diagnostic: top-P-pairs init vs damped CP-ALS, in core space.
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

    auto Sc = S_phi.to(eScalarType::Double).to(eScalarType::ComplexDouble);  // [L]
    auto W  = Omega.to(eScalarType::ComplexDouble).mul(Sc.unsqueeze(0)).contiguous();  // [K,L]

    // Top-P pairs (l1,l2) ranked by S[l1]*S[l2], same as the deterministic init.
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
        // Lam update: Lam = (F*C) @ pinv(conj(gram_A) * gram_B)
        //   F = W @ conj(Acore), C = conj(W) @ Bcore
        auto F = mm(W, Acore.conj().contiguous());   // [K,P]
        auto C = mm(W.conj().contiguous(), Bcore);   // [K,P]

        auto gram_A = mm(Acore.conj().transpose(0,1).contiguous(), Acore);
        auto gram_B = mm(Bcore.conj().transpose(0,1).contiguous(), Bcore);
        Lam = cpcore_rsolve_trunc(F.mul(C), gram_A.conj().mul(gram_B), rel_eps);

        auto gram_Lam = mm(Lam.conj().transpose(0,1).contiguous(), Lam);

        // Acore update: Acore = [W^T @ (conj(Lam) * C)] @ pinv(gram_B * conj(gram_Lam))
        //   C = conj(W) @ Bcore (old Bcore)
        Acore = cpcore_rsolve_trunc(
            mm(W.transpose(0,1).contiguous(), Lam.conj().mul(C)),
            gram_B.mul(gram_Lam.conj()), rel_eps);

        gram_A = mm(Acore.conj().transpose(0,1).contiguous(), Acore);

        // Bcore update: Bcore = [W^T @ (Lam * conj(F_new))] @ pinv(gram_A_new * gram_Lam)
        //   F_new = W @ conj(Acore_new)
        auto F_new = mm(W, Acore.conj().contiguous());   // [K,P]
        Bcore = cpcore_rsolve_trunc(
            mm(W.transpose(0,1).contiguous(), Lam.mul(F_new.conj())),
            gram_A.mul(gram_Lam), rel_eps);

        // Final Lam with updated A,B for the error readout.
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

// ── Approx signal via direct DFT at arbitrary (possibly warped) pixel coords ──
//
// S[k] = Σ_l Omega_sub[k,l] · Σ_n Upsilon[bin(n),l] · rho_m[n] · exp(-i·k·pix[n])
//
// pix[n,:] are centered pixel-index coords (= phys/dx per dim) in the same
// convention as fft::dft / forward_exact: phase = exp(-i·k_traj·n_pix).
// For the warp comparison, pix[n] = u(r(n)) (warped by apply_axis_warp in
// pixel-index space). Processed in k-chunks of size k_chunk to bound GPU memory.
static hasty::Tensor approx_signal_direct(
    const hasty::Tensor& Omega_sub,    // [n_sub, L]
    const hasty::Tensor& Upsilon,      // [n_hist, L]
    const hasty::Tensor& voxel_to_bin, // [N_mask] long
    const hasty::Tensor& rho_m,        // [N_mask] ComplexFloat
    const hasty::Tensor& pix_masked,   // [N_mask, 3] float — centered pixel coords (or warped)
    const hasty::Tensor& k_sub,        // [n_sub, 3] float ∈ [-π,π] — k_traj subset
    hasty::i64 k_chunk = 32)
{
    using namespace hasty;
    using namespace std::numbers;

    const i64    L      = Omega_sub.size(1);
    const i64    n_sub  = Omega_sub.size(0);
    const i64    N_mask = rho_m.size(0);
    const Device dev    = rho_m.device();
    const TensorOptions opts_c{dev, eScalarType::ComplexFloat};

    // src[n, l] = Upsilon[bin(n), l] · rho_m[n]  — [N_mask, L]
    auto ups_vox = Upsilon.index_select(0, voxel_to_bin);           // [N_mask, L]
    auto src     = ups_vox.to(eScalarType::ComplexFloat)
                          .mul(rho_m.unsqueeze(1));                  // [N_mask, L]

    const Scalar neg_i = Scalar(std::complex<f32>(0.0f, -1.0f));
    auto pix_t  = pix_masked.transpose(0, 1).contiguous();          // [3, N_mask]
    auto signal = zeros({n_sub}, opts_c);

    for (i64 k0 = 0; k0 < n_sub; k0 += k_chunk) {
        const i64 kc   = std::min(k_chunk, n_sub - k0);
        auto k_c       = k_sub.narrow(0, k0, kc);                  // [kc, 3]
        // phase[kc, N_mask] = k_c @ pix_t;  exp(-i·phase) = DFT kernel
        auto phase     = mm(k_c, pix_t)
                             .to(eScalarType::ComplexFloat)
                             .mul(neg_i).exp();                      // [kc, N_mask]
        // S_sub[kc, L] = phase @ src
        auto S_sub     = mm(phase, src);                            // [kc, L]
        // signal[k0:k0+kc] = Σ_l Omega_sub[k0:k0+kc, l] · S_sub[:, l]
        signal.narrow(0, k0, kc).copy_(
            S_sub.mul(Omega_sub.narrow(0, k0, kc)).sum(1));
    }

    return signal.unsqueeze(0);  // [1, n_sub]
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

            // Fast core-space diagnostic (milliseconds, no embedding build).
            core_cp_diagnostic(Omega, S_phi, p_use, /*n_iter=*/25, /*rel_eps=*/1e-9);

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
            auto emb_nh_init = make_normal_nonhermitian_off_fourier_toeplitz_embeddings(
                coords, {nx, ny, nz}, hist.mask_idx, hist.voxel_to_bin, plr, p_use,
                /*n_als_iter=*/0,
                eComputeStrategyBuildOffFourierEmbeddings::RUN_ALL_ON_INPUT_DEVICE,
                eStorageStrategyBuildOffFourierEmbeddings::STORE_IN_FILE);
            double t_nhb_init = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0n).count();

            t0n = std::chrono::steady_clock::now();
            auto emb_nh = make_normal_nonhermitian_off_fourier_toeplitz_embeddings(
                coords, {nx, ny, nz}, hist.mask_idx, hist.voxel_to_bin, plr, p_use,
                /*n_als_iter=*/50,
                eComputeStrategyBuildOffFourierEmbeddings::RUN_ALL_ON_INPUT_DEVICE,
                eStorageStrategyBuildOffFourierEmbeddings::STORE_IN_FILE);
            double t_nhb = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0n).count();

            t0n = std::chrono::steady_clock::now();
            auto out_naive = apply_normal_toeplitz_off_fourier_operator(emb_naive, rho);
            double t_na = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0n).count();

            t0n = std::chrono::steady_clock::now();
            auto out_nh_init = apply_normal_toeplitz_off_fourier_operator(emb_nh_init, rho);
            double t_nha_init = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0n).count();

            t0n = std::chrono::steady_clock::now();
            auto out_nh = apply_normal_toeplitz_off_fourier_operator(emb_nh, rho);
            double t_nha = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0n).count();

            // Compare non-hermitian (P splits) against naive L² at masked voxels.
            auto flat_ref     = out_naive  .reshape({nx*ny*nz}).index_select(0, hist.mask_idx).cpu().contiguous();
            auto flat_nh_init = out_nh_init.reshape({nx*ny*nz}).index_select(0, hist.mask_idx).cpu().contiguous();
            auto flat_nh      = out_nh     .reshape({nx*ny*nz}).index_select(0, hist.mask_idx).cpu().contiguous();

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

            float mre_init = mean_rel_err(flat_nh_init);
            float mre_nh   = mean_rel_err(flat_nh);

            std::cout << "\n  Normal op [L=" << L << "]  (ref=naïve L²)\n"
                      << "    SVD init, no ALS   (P=" << p_use << "):"
                      << "  mean_rel_err=" << std::scientific << std::setprecision(3) << mre_init << "\n"
                      << "    non-hermitian CP-ALS (P=" << p_use << "):"
                      << "  mean_rel_err=" << std::scientific << std::setprecision(3) << mre_nh << "\n"
                      << "  timing build:  naive=" << std::setprecision(2) << t_nb
                      << "s  init=" << t_nhb_init << "s  als=" << t_nhb << "s\n"
                      << "  timing apply:  naive=" << t_na
                      << "s  init=" << t_nha_init << "s  als=" << t_nha << "s\n";

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

// ── Warp comparison: absorb y² channel into coordinate substitution ───────────
//
// Builds histogram on Q_residual=2 channels only (field_z2, field_xy) and
// evaluates the approximation at warped pixel-index coords u(r) = r + c·r_y²·ĵ,
// using approx_signal_direct for direct comparison against S_both reference.
// This validates that: (a) histogram shrinks, (b) accuracy holds at same/smaller L.
static void run_test_warp(const Problem& prob,
                          hasty::i64 n_sub,
                          const std::vector<hasty::i64>& L_values,
                          hasty::i64 n_rate, hasty::i64 n_nl,
                          const std::string& label,
                          hasty::mri::eBinWeighting weighting = hasty::mri::eBinWeighting::L1Mass)
{
    using namespace hasty;
    std::cout << "\n[warp test: " << label << "]\n";
    std::cout << "  K=" << prob.K << "  N=" << prob.N << "  n_sub=" << n_sub << "\n";

    // Q_residual = 3: drop the separable y² channel (last index = 3).
    // Residual: {field_z2, field_xy, field_xz} — concomitant terms only.
    auto nl_fields_res    = prob.nl_fields_flat.narrow(0, 0, 3);   // [3, N]
    auto nl_waveforms_res = prob.nl_waveforms.narrow(0, 0, 3);     // [3, K]

    auto hist = mri::extract_histogram(
        prob.mag_flat, prob.z_map_flat, nl_fields_res, n_rate, n_nl);
    const i64 N_mask = hist.mask_idx.size(0);
    std::cout << "  n_hist=" << hist.n_hist
              << "  N_mask=" << N_mask
              << "  (" << std::fixed << std::setprecision(1)
              << (100.0f * N_mask / (float)prob.N) << "% of N) — concomitant-only residual (Q=3, GNL absorbed into warp)\n";

    auto op = mri::make_phi_operator(
        hist.z_map_hist, hist.nl_fields_hist,
        nl_waveforms_res, prob.timestamps);

    // Same seed as run_test → identical sub_idx for direct error comparison.
    std::mt19937 rng(42);
    std::uniform_int_distribution<i64> dist(0, prob.K - 1);
    std::vector<i64> idx_buf(n_sub);
    for (i64 i = 0; i < n_sub; ++i) idx_buf[i] = dist(rng);
    auto sub_idx = Tensor::from_blob(idx_buf.data(), {n_sub}, eScalarType::Long,
                                     Device{eDeviceType::CPU})
                       .clone().to(prob.mag.device());

    auto S_both  = exact_signal_paired(prob, sub_idx, true, true);  // ground truth
    auto ref_mag   = S_both.select(0, 0).abs().cpu().contiguous();
    auto ref_phase = [&]() {
        auto s1 = S_both.select(0, 0).cpu().contiguous();
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

    auto circ_corr = [](const Tensor& pa, const Tensor& pb) -> float {
        auto diff = pa.sub(pb).contiguous();
        auto dv = diff.spanning_view();
        i64 M = dv.sizes[0];
        const float* d = static_cast<const float*>(dv.data);
        double sum_cos = 0.0;
        for (i64 i = 0; i < M; ++i) sum_cos += std::cos(d[i]);
        return (float)(sum_cos / M);
    };

    // Pre-compute warped pixel coords for masked voxels.
    // Warp: u_y = n_y + c_y2_pix*(n_y+0.5)²
    // pix_masked stores n_d = i_d - N_d/2 (integer-centered, matching fft::dft).
    // The GNL field in physical space is phys_y² = (n_y+0.5)²*dy_m², so the
    // displacement in pixel space is c_y2_pix*(n_y+0.5)² = c_y2*phys_y²/dy_m.
    // This exactly cancels the GNL phase exp(i*field_y2*alpha_y2) in forward_exact.
    auto pix_masked  = prob.coords_pix_flat.index_select(0, hist.mask_idx);  // [N_mask,3]
    auto pix_y_m     = pix_masked.select(1, 1);                              // n_y [N_mask]
    auto pix_y_c     = pix_y_m.add(Scalar(0.5f));                           // n_y+0.5 = phys_y/dy_m
    auto field_warp  = pix_y_c.mul(pix_y_c);                                // (n_y+0.5)²
    auto fgrad_warp  = pix_y_c.mul(Scalar(2.0f));                           // 2*(n_y+0.5)
    auto warp = mri::apply_axis_warp(pix_masked, field_warp, fgrad_warp, 1, prob.nl_y2_c_pix);

    {
        float jac_min = warp.jacobian_det.min().item<float>();
        float jac_max = warp.jacobian_det.max().item<float>();
        std::cout << "  det(J_u) in masked voxels: [" << std::fixed << std::setprecision(3)
                  << jac_min << ", " << jac_max << "]  (diffeomorphism "
                  << ((jac_min > 0.0f) ? "OK" : "VIOLATED") << ")\n";
    }

    auto rho_m   = prob.mag_flat.index_select(0, hist.mask_idx).to(eScalarType::ComplexFloat);
    auto k_sub   = prob.k_traj.index_select(0, sub_idx);  // [n_sub, 3]
    auto pix_w   = warp.warped_coords;                     // [N_mask, 3]

    std::cout << "\n  L   n_hist   phase_corr_err   status  (warp)\n"
              << "  " << std::string(49, '-') << "\n";

    for (auto L : L_values) {
        const i64 mpd_val = std::max((i64)8*L, (i64)60);
        const i64 ncv_val = L + mpd_val;

        auto weights = mri::phi_lowrank_weights(weighting, hist, prob.timestamps);
        auto [Omega, S_phi, Upsilon] = mri::phi_lowrank(op, L, weights, ncv_val, mpd_val);

        auto Omega_sub = Omega.index_select(0, sub_idx);  // [n_sub, L]

        // Sanity check: approx_signal_direct with UNWARPED pix_masked.
        // Should approximate S_no_y2 (no GNL), revealing if the direct formula is correct.
        auto S_noWarp = approx_signal_direct(Omega_sub, Upsilon,
                                             hist.voxel_to_bin, rho_m,
                                             pix_masked, k_sub);

        auto S_warp = approx_signal_direct(Omega_sub, Upsilon,
                                           hist.voxel_to_bin, rho_m,
                                           pix_w, k_sub);

        float ne = S_both.norm().item<float>();
        float na_warp   = S_warp.norm().item<float>();
        float na_noWarp = S_noWarp.norm().item<float>();

        auto approx_phase_of = [&](const Tensor& S) {
            auto s1 = S.select(0, 0).cpu().contiguous();
            auto re = s1.real().contiguous(); auto im = s1.imag().contiguous();
            auto rv = re.spanning_view(); auto iv = im.spanning_view();
            i64 M = rv.sizes[0];
            std::vector<float> ph(M);
            const float* rp = static_cast<const float*>(rv.data);
            const float* ip = static_cast<const float*>(iv.data);
            for (i64 i = 0; i < M; ++i) ph[i] = std::atan2(ip[i], rp[i]);
            return Tensor::from_blob(ph.data(), {M}, eScalarType::Float,
                                     Device{eDeviceType::CPU}).clone();
        };

        float phase_r_warp   = circ_corr(ref_phase, approx_phase_of(S_warp));
        float phase_r_noWarp = circ_corr(ref_phase, approx_phase_of(S_noWarp));
        float mag_r_warp     = pearson(ref_mag, S_warp.select(0,0).abs().cpu().contiguous());

        float err_warp   = phase_corrected_rel_err(S_both, S_warp);
        float err_noWarp = phase_corrected_rel_err(S_both, S_noWarp);

        bool pass = (phase_r_warp > 0.995f);
        std::cout << "  " << std::setw(2) << L
                  << "  " << std::setw(6) << hist.n_hist
                  << "   na_warp/ne=" << std::fixed << std::setprecision(3) << (na_warp/ne)
                  << "  na_nw/ne=" << (na_noWarp/ne)
                  << "\n"
                  << "       warp:   mag_r=" << std::setprecision(4) << mag_r_warp
                  << "  phase_r=" << phase_r_warp
                  << "  err=" << std::scientific << std::setprecision(3) << err_warp
                  << "\n"
                  << "       noWarp: phase_r=" << std::fixed << std::setprecision(4) << phase_r_noWarp
                  << "  err=" << std::scientific << std::setprecision(3) << err_noWarp
                  << "  " << (pass ? "PASS" : "FAIL") << "\n";
    }
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
            0.5f);

        failures += !run_test(prob_real,
            300,
            {8},
            14000, 1,
            "real_data full_res",
            hasty::mri::eBinWeighting::L1Mass,
            /*P=*/14,
            show_fft_error_plots,
            show_phi_error_plots,
            show_phi_error_vs_B0_plots,
            show_signal_err_vs_mag_plot);

        // Warp comparison: y² GNL absorbed into coord substitution, Q_res=3 concomitant residual.
        // n_nl=2 (vs joint n_nl=1) so the residual histogram captures NL bin variation
        // and n_hist is comparable — shows both rank savings and bin-count behavior.
        run_test_warp(prob_real,
            300,
            {6, 8},   // compare at L=6 (might match L=8 joint) and L=8 (expect better)
            14000, 2,
            "real_data full_res",
            hasty::mri::eBinWeighting::L1Mass);
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
