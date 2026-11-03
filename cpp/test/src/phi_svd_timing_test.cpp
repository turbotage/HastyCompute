import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_linalg_mod;
import hasty_mri_mod;

// ── Where does operator_svd's time actually go: our own matvec/rmatvec
// (build_phi's exp() evaluation) or SLEPc/PETSc's own orthogonalization/
// restart bookkeeping? Answers this directly instead of arguing about it --
// builds a production-scale-ish synthetic VoxelFeatures + timestamps, runs
// phi_lowrank once, and reports total wall time vs the matvec/rmatvec time
// accumulated via get_phi_operator_timing_stats() (cpp/lib/src/mri/
// off_fourier_interpolators.cppm). The remainder is PETSc's own overhead.

using namespace hasty;
using namespace hasty::mri;

struct Stopwatch {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

int main()
{
    Device dev{eDeviceType::CUDA, 0};

    const i64 n_voxels = 100000;
    const i64 nspokes   = 500;
    const i64 nsamps    = 4000;
    const i64 Q         = 4;     // concomitant terms
    const i64 L         = 12;

    std::cout << "=== phi_lowrank matvec-vs-PETSc-overhead timing probe ===\n";
    std::cout << "  n_voxels=" << n_voxels << "  K=" << nspokes*nsamps
              << "  Q=" << Q << "  L=" << L << "\n\n";

    auto opts_f = TensorOptions(dev, eScalarType::Float);
    auto opts_l = TensorOptions(dev, eScalarType::Long);

    // Synthetic z_map: off-resonance + decay-rate-like complex feature,
    // smooth-ish random values -- representative SCALE, not real physics.
    auto z_real = rand({n_voxels}, opts_f).mul(Scalar(50.0f));     // ~1/T2-ish
    auto z_imag = rand({n_voxels}, opts_f).sub(Scalar(0.5f)).mul(Scalar(200.0f)); // ~B0-ish
    auto z_map_voxel = view_as_complex(stack({z_real, z_imag}, -1).contiguous())
                           .to(eScalarType::ComplexFloat);

    auto conc_fields_voxel = rand({Q, n_voxels}, opts_f);
    auto conc_waveforms    = rand({Q, nspokes, nsamps}, opts_f);

    auto mask_idx         = arange(n_voxels, opts_l);
    auto voxel_to_feature = arange(n_voxels, opts_l);
    auto voxel_weights    = rand({n_voxels}, opts_f).add(Scalar(0.1f));
    auto voxel_l2_energy  = voxel_weights.mul(voxel_weights);
    auto voxel_count      = ones({n_voxels}, opts_f);

    VoxelFeatures feat{
        mask_idx, voxel_to_feature, n_voxels,
        z_map_voxel, conc_fields_voxel, conc_waveforms,
        voxel_weights, voxel_l2_energy, voxel_count
    };

    auto timestamps = rand({nspokes, nsamps}, opts_f).mul(Scalar(0.01f));  // ~10ms readout

    reset_phi_operator_timing_stats();

    Stopwatch sw;
    auto phi_res = make_phi_operator(feat, timestamps, /*subsample=*/4, 1024);
    double build_op_s = sw.elapsed();

    sw = Stopwatch();
    auto weights = phi_lowrank_weights(eFeatureWeighting::L1Mass, feat, timestamps.reshape({-1}));
    auto plr = phi_lowrank(phi_res.op, L, weights, -1, -1, 1e-5);
    double svd_s = sw.elapsed();

    auto t = get_phi_operator_timing_stats();
    double matvec_total = t.matvec_s + t.rmatvec_s;
    double overhead_s = svd_s - matvec_total;

    std::cout << "  make_phi_operator (build lambdas, no compute yet): " << build_op_s << "s\n\n";
    std::cout << "  phi_lowrank total wall time:        " << std::fixed << std::setprecision(3) << svd_s << "s\n";
    std::cout << "  -- matvec  (forward):  " << t.matvec_s  << "s  (" << t.matvec_calls  << " calls, "
              << (t.matvec_calls  ? t.matvec_s  / t.matvec_calls  * 1000.0 : 0.0) << " ms/call)\n";
    std::cout << "  -- rmatvec (adjoint):  " << t.rmatvec_s << "s  (" << t.rmatvec_calls << " calls, "
              << (t.rmatvec_calls ? t.rmatvec_s / t.rmatvec_calls * 1000.0 : 0.0) << " ms/call)\n";
    std::cout << "  -- our matvec/rmatvec total: " << matvec_total << "s  ("
              << std::setprecision(1) << (100.0 * matvec_total / svd_s) << "% of total)\n";
    std::cout << "  -- SLEPc/PETSc overhead (orthogonalization, restarts, etc.): " << std::setprecision(3)
              << overhead_s << "s  (" << std::setprecision(1) << (100.0 * overhead_s / svd_s) << "% of total)\n\n";

    std::cout << "  Omega: " << plr.Omega.size(0) << "x" << plr.Omega.size(1)
              << "  Upsilon: " << plr.Upsilon.size(0) << "x" << plr.Upsilon.size(1) << "\n";

    std::cout << "\n========================================\n";
    return 0;
}
