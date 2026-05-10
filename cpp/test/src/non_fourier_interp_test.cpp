#include <numbers>
#include <cmath>

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_linalg_mod;
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
    hasty::Tensor nl_basis;        // [Q, n, n, n]
    hasty::Tensor mag_flat;        // [N]
    hasty::Tensor z_map_flat;      // [N] complex
    hasty::Tensor nl_fields_flat;  // [Q, N]
    hasty::Tensor coords_flat;     // [N, 3] normalized [-0.5, 0.5]
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
    auto r_norm    = pow(r_sq, 0.5);
    auto mag_flat  = r_norm.lt(Scalar(0.4f)).to(eScalarType::Float);
    auto mag       = mag_flat.reshape({n_dim, n_dim, n_dim});

    auto dB0_hz  = rz.mul(Scalar(1000.0f));
    auto z_imag  = dB0_hz.mul(Scalar(2.0f * (f32)pi_v<f64>));
    auto z_real  = zeros({N}, opts_f);
    auto z_map_flat = view_as_complex(stack({z_real, z_imag}, 1).contiguous());
    auto rate_map   = z_map_flat.reshape({n_dim, n_dim, n_dim});

    auto sensitivity_maps = ones({C, n_dim, n_dim, n_dim}, opts_c);

    // Koosh-ball trajectory (CPU raw math; acos not in Tensor API)
    const float k_max_rad_m = (float)pi_v<f64> / dx;
    const float k_norm_fac  = FOV / (2.0f * (float)pi_v<f64>);  // rad/m → cycles/FOV

    std::vector<float> dirs(n_spokes * 3);
    for (i64 s = 0; s < n_spokes; ++s) {
        float f    = (float)s;
        float th   = std::acos(1.0f - 2.0f * f / (float)n_spokes);
        float ph   = f * (float)pi_v<f64> * (3.0f - std::sqrt(5.0f));
        dirs[s*3+0] = std::sin(th) * std::cos(ph);
        dirs[s*3+1] = std::sin(th) * std::sin(ph);
        dirs[s*3+2] = std::cos(th);
    }

    std::vector<float> k_data(K * 3), t_data(K);
    for (i64 s = 0; s < n_spokes; ++s) {
        for (i64 j = 0; j < n_samp; ++j) {
            float k_r = (-k_max_rad_m) + 2.0f * k_max_rad_m * (float)j / (float)(n_samp - 1);
            i64 idx = s * n_samp + j;
            k_data[idx*3+0] = dirs[s*3+0] * k_r * k_norm_fac;
            k_data[idx*3+1] = dirs[s*3+1] * k_r * k_norm_fac;
            k_data[idx*3+2] = dirs[s*3+2] * k_r * k_norm_fac;
            t_data[idx]     = (float)j * dt;
        }
    }
    auto k_traj    = Tensor::from_blob(k_data.data(), {K, 3}, eScalarType::Float,
                                       Device{eDeviceType::CPU}).clone().to(dev);
    auto timestamps = Tensor::from_blob(t_data.data(), {K}, eScalarType::Float,
                                        Device{eDeviceType::CPU}).clone().to(dev);

    // Nonlinear fields
    auto phys_z    = rz.mul(Scalar(FOV));
    auto phys_x    = rx.mul(Scalar(FOV));
    auto phys_y    = ry.mul(Scalar(FOV));
    auto field_z2  = phys_z.mul(phys_z);
    auto field_xy  = phys_x.mul(phys_y);

    // Biphasic gradient
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

    auto cpu = Device{eDeviceType::CPU};
    auto G_z2_t = Tensor::from_blob(G_z2.data(), {K}, eScalarType::Float, cpu).clone().to(dev);
    auto G_xy_t = Tensor::from_blob(G_xy.data(), {K}, eScalarType::Float, cpu).clone().to(dev);

    auto alpha_z2 = G_z2_t.to(eScalarType::Double).cumsum(0).to(eScalarType::Float)
                           .mul(Scalar(-gamma * dt));
    auto alpha_xy = G_xy_t.to(eScalarType::Double).cumsum(0).to(eScalarType::Float)
                           .mul(Scalar(-gamma * dt));

    auto nl_waveforms   = stack({alpha_z2, alpha_xy}, 0);
    auto nl_basis       = stack({field_z2.reshape({n_dim,n_dim,n_dim}),
                                  field_xy.reshape({n_dim,n_dim,n_dim})}, 0);
    auto nl_fields_flat = stack({field_z2, field_xy}, 0);
    auto coords_flat    = stack({rx, ry, rz}, 1);

    return Problem{
        mag, rate_map, sensitivity_maps,
        k_traj, timestamps,
        nl_waveforms, nl_basis,
        mag_flat, z_map_flat, nl_fields_flat, coords_flat,
        n_dim, K, N, C, FOV, dt
    };
}

// Returns exact signal [C, M] at subsampled (k_m, t_m) pairs.
static hasty::Tensor exact_signal_sub(const Problem& prob,
                                      const hasty::Tensor& sub_idx,
                                      hasty::i64 k_batch = 8)
{
    using namespace hasty;
    const i64 M = sub_idx.size(0);
    const i64 C = prob.C;

    auto k_sub   = prob.k_traj.index_select(0, sub_idx);
    auto t_sub   = prob.timestamps.index_select(0, sub_idx);
    auto nl_sub  = prob.nl_waveforms.index_select(1, sub_idx);

    auto S = mri::forward_exact(
        prob.mag, prob.sensitivity_maps, prob.rate_map,
        t_sub, k_sub, nl_sub, prob.nl_basis,
        true, true, k_batch, 0
    );  // [C, M, M]

    auto result = zeros({C, M}, TensorOptions(prob.mag.device(), eScalarType::ComplexFloat));
    for (i64 c = 0; c < C; ++c)
        for (i64 m = 0; m < M; ++m)
            result.select(0, c).select(0, m).copy_(
                S.select(0, c).select(0, m).select(0, m));
    return result;
}

static bool test_approx(hasty::i64 n_dim, hasty::i64 n_spokes, hasty::i64 n_samp,
                         hasty::i64 n_sub, std::vector<hasty::i64> L_values,
                         hasty::i64 n_rate, hasty::i64 n_nl,
                         hasty::Device dev)
{
    using namespace hasty;
    std::string dev_str = dev.type == eDeviceType::CUDA ? "CUDA" : "CPU";
    std::cout << "\n[test] n_dim=" << n_dim << "  n_spokes=" << n_spokes
              << "  n_samp=" << n_samp << "  dev=" << dev_str << "\n";

    auto prob = make_problem(n_dim, n_spokes, n_samp, dev);
    std::cout << "  K=" << prob.K << "  N=" << prob.N << "\n";

    auto hist = mri::extract_histogram(
        prob.mag_flat, prob.z_map_flat, prob.nl_fields_flat, n_rate, n_nl);
    std::cout << "  n_hist=" << hist.n_hist << "\n";

    auto op = mri::make_phi_operator(
        hist.z_map_hist, hist.nl_fields_hist,
        prob.nl_waveforms, prob.timestamps);

    // Deterministic subsample: first n_sub k-space points
    auto sub_idx = arange(n_sub, TensorOptions(dev, eScalarType::Long));

    std::cout << "  Computing exact signal ...\n";
    auto S_exact    = exact_signal_sub(prob, sub_idx);
    float norm_exact = S_exact.norm().item<float>();
    std::cout << "  |S_exact| = " << norm_exact << "\n";

    // Prepare masked spatial quantities for approx_signal
    auto mag_masked   = prob.mag_flat.to(eScalarType::ComplexFloat)
                                      .index_select(0, hist.mask_idx);
    auto coil_flat    = prob.sensitivity_maps.reshape({prob.C, prob.N});
    auto coil_masked  = coil_flat.index_select(1, hist.mask_idx);
    auto coords_masked = prob.coords_flat.index_select(0, hist.mask_idx);
    auto k_sub        = prob.k_traj.index_select(0, sub_idx);

    std::cout << "\n  L   n_hist   rel_err    status\n"
              << "  " << std::string(36, '-') << "\n";

    bool all_pass = true;
    for (auto L : L_values) {
        auto [Omega, Upsilon] = mri::phi_lowrank(op, L,
            Opt<Tensor>{hist.bin_weights});

        auto Omega_sub = Omega.index_select(0, sub_idx);

        auto S_approx = mri::approx_signal(
            Omega_sub, Upsilon,
            mag_masked, coil_masked, coords_masked,
            k_sub, hist.voxel_to_bin, hist.n_hist);

        float err  = S_exact.sub(S_approx).norm().item<float>() / (norm_exact + 1e-30f);
        bool  pass = err < 0.05f;
        all_pass   = all_pass && pass;

        std::cout << "  " << std::setw(3) << L
                  << "  " << std::setw(7) << hist.n_hist
                  << "  " << std::scientific << std::setprecision(3) << err
                  << "  " << (pass ? "PASS" : "FAIL") << "\n";
    }

    return all_pass;
}

int main()
{
    bool show_locally = false;

    std::string nifti_dir = "/home/turbotage/Documents/GitHub/HastyData/downloads/traveling_heads_7t/TH2_data_ES_s1/upload_ES/ES_20181008/subject1/";

    hasty::io::nifti::NiftiImage b0 = hasty::io::nifti::read_nifti(nifti_dir + "b0fieldHZ.nii.gz");
    hasty::io::nifti::NiftiImage pd = hasty::io::nifti::read_nifti(nifti_dir + "gre_qsm_mag.nii.gz");

    auto mask = hasty::io::nifti::transform_nifti_data(pd);

    auto mask_mean = mask.mean();
    auto mask_std = mask.std();

    std::cout << "mask mean: " << mask_mean.item<hasty::f32>() << "\n";
    std::cout << "mask std: " << mask_std.item<hasty::f32>() << "\n";

    // Diagnostic: print min/max and fraction of voxels above mean so we can
    // check whether the boolean mask is inverted relative to expectations.
    try {
        auto mask_min = mask.min().item<hasty::f32>();
        auto mask_max = mask.max().item<hasty::f32>();
        auto total_elems = mask.numel();
        std::cout << "mask min: " << mask_min << "  max: " << mask_max << "\n";
        std::cout << "total elems: " << total_elems << "\n";
    } catch (...) { }

    mask = mask > (mask_mean);

    try {
        auto total = mask.numel();
        auto n_true = mask.to(hasty::eScalarType::Long).sum().item<hasty::i64>();
        std::cout << "mask true count: " << n_true << " / " << total
                  << " (" << (100.0 * n_true / (double)total) << "% )\n";
    } catch (...) { }

    hasty::viz::orthoslicer(mask, {"mask", std::nullopt}, false, show_locally);

    mask = mask.to(hasty::Device{hasty::eDeviceType::CUDA, 0});

    mask = mask_erode(std::move(mask), 2, {0});
    mask = mask_erode(std::move(mask), 2, {0});

    mask = mask_dilate(std::move(mask), 4, {0});
    mask = mask_dilate(std::move(mask), 4, {0});
    mask = mask_dilate(std::move(mask), 4, {0});
    mask = mask_dilate(std::move(mask), 4, {0});

    mask = mask_erode(std::move(mask), 4, {0});
    mask = mask_erode(std::move(mask), 4, {0});

    mask = mask.cpu();

    hasty::viz::orthoslicer(mask, {"mask_dilated", std::nullopt}, false, show_locally);


    // Push both volumes to the gRPC bank with NIfTI header metadata.
    auto b0_uuid = hasty::python::push_nifti_image(b0, "b0");
    auto pd_uuid = hasty::python::push_nifti_image(pd, "pd");

    // Register pd onto b0's grid via Python/ANTs.
    // Pass --debug-port=5678 as third argument to enable Python debugger attach.
    auto result = hasty::python::run_script(
        hasty::python::scripts_dir() + "/register_nifti.py",
        {
            "--fixed="        + hasty::python::uuid_to_hex(b0_uuid),
            "--moving="       + hasty::python::uuid_to_hex(pd_uuid),
            "--transform=Rigid",   // same-session, cross-modality: rigid only
        }
        // set debug=true, debug_port=5678 to pause for Python debugger:
        // , /*debug=*/true, /*debug_port=*/5678
    );

    if (result.exit_code != 0) {
        std::cerr << "Registration failed (exit " << result.exit_code << ")\n";
        return 1;
    }

    // Fetch registered volume back and display it.
    // The script printed the output UUID hex on stdout.
    std::string reg_uuid_hex = result.output_uuids.at(0);

    hasty::viz::orthoslicer(b0.data, {"b0", std::nullopt}, false, show_locally);
    hasty::viz::orthoslicer(pd.data, {"pd_original", std::nullopt}, false, show_locally);

    // Fetch registered tensor from bank for visualisation.
    auto reg_uuid_arr = hasty::python::hex_to_uuid_array(reg_uuid_hex);
    std::string reg_key(reinterpret_cast<const char*>(reg_uuid_arr.data()), 16);
    hasty::Tensor reg_tensor = hasty::server::global_generic_value_bank
                                   .fetch_value(reg_key).as_tensor();
    hasty::viz::orthoslicer(reg_tensor, {"b0_registered_to_pd", std::nullopt}, true, show_locally);

    std::cout << "=====================================================\n"
              << "  off-Fourier interpolator accuracy test\n"
              << "=====================================================\n";

    hasty::Device cpu{hasty::eDeviceType::CPU};
    int failures = 0;

    // Small CPU test: fast enough for CI
    failures += !test_approx(
        16, 30, 40,        // n_dim, n_spokes, n_samp
        20,                // n_sub (subsample for comparison)
        {4, 8, 12, 20},    // L values
        64, 8,             // n_rate, n_nl bins
        cpu);

    if (cuda_available()) {
        hasty::Device cuda0{hasty::eDeviceType::CUDA, 0};
        failures += !test_approx(
            32, 100, 100,
            30,
            {4, 8, 12, 20},
            256, 16,
            cuda0);
    } else {
        std::cout << "\n[CUDA] not available, skipping GPU test.\n";
    }

    std::cout << "\n=====================================================\n"
              << "  " << failures << " failure(s)\n"
              << "=====================================================\n";
    return failures > 0 ? 1 : 0;
}
