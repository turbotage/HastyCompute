import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_mri_mod;

// ── Phase A validation: two-level butterfly factorization vs brute force ──────
//
// Validates cpp/lib/src/mri/{butterfly_tree,butterfly_id,butterfly_factorization}.cppm
// against a DENSE brute-force evaluation of the SAME phase function — proves
// the Chebyshev compression + Lagrange anterpolation/interpolation machinery
// is correct, independent of any CoordinateWarp/GNL specifics (the PhaseFn
// here is a standalone representative quadratic warp, matching
// python/scripts/butterfly_rank_diagnostic.py's model — separable per axis,
// simpler than the fully non-separable warps CoordinateWarp supports (e.g.
// cross terms like xz); sufficient to validate the MACHINERY, not yet a
// substitute for Phase B's integration test against the real warp).

using namespace hasty;
using namespace hasty::mri;

// Integer-centered pixel grid [N,D] for a regular grid_shape, C order.
static Tensor pix_grid(const std::vector<i64>& shape, Device dev)
{
    const i64 D = (i64)shape.size();
    i64 total = 1; for (auto s : shape) total *= s;

    std::vector<float> buf((std::size_t)total * D);
    std::vector<i64> idx(D, 0);
    for (i64 s = 0; s < total; ++s) {
        for (i64 d = 0; d < D; ++d) buf[(std::size_t)s * D + d] = (float)(idx[d] - shape[d] / 2);
        for (i64 d = D - 1; d >= 0; --d) { if (++idx[d] < shape[d]) break; idx[d] = 0; }
    }
    return Tensor::from_blob(buf.data(), {total, D}, eScalarType::Float, Device{eDeviceType::CPU})
               .clone().to(dev);
}

// Representative quadratic GNL-like inverse warp, separable per axis:
// u_inv_axis(q) = q - delta_max*(q/half_n)^2. Phi(k,q) = (2*pi/N)*k.u_inv(q).
static PhaseFn make_quadratic_phase_fn(i64 N, float delta_max)
{
    const float half_n = (float)N / 2.0f;
    using namespace std::numbers;
    const float scale = 2.0f * (float)pi_v<f64> / (float)N;

    return [half_n, delta_max, scale](const Tensor& k_pts, const Tensor& q_pts) -> Tensor {
        const i64 D = k_pts.size(1);
        Tensor Phi;
        for (i64 d = 0; d < D; ++d) {
            auto k_d = k_pts.select(1, d).unsqueeze(1);                 // [Nk,1]
            auto q_d = q_pts.select(1, d);                              // [Nq]
            auto uinv_d = q_d.sub(q_d.mul(q_d).mul(Scalar(delta_max / (half_n * half_n))));
            auto term = k_d.mul(uinv_d.unsqueeze(0));                   // [Nk,Nq]
            Phi = Phi.defined() ? Phi.add(term) : term;
        }
        return Phi.mul(Scalar(scale));
    };
}

static Tensor dense_apply(const PhaseFn& phase_fn, const Tensor& k_pts, const Tensor& q_pts,
                          const Tensor& c_k)
{
    auto Phi = phase_fn(k_pts, q_pts);                                  // [Nk,Nq] Float
    auto M = Phi.to(eScalarType::ComplexFloat)
                 .mul(Scalar(std::complex<f32>(0.0f, 1.0f)))
                 .exp();
    return matmul(c_k.unsqueeze(0), M).reshape({q_pts.size(0)});
}

static bool run_case(i64 N, i64 D, float delta_max, i64 l_k, i64 n_cheb, double tol, i64 max_rank)
{
    Device dev{eDeviceType::CPU};
    std::vector<i64> shape(D, N);

    auto phase_fn = make_quadratic_phase_fn(N, delta_max);
    auto k_pts_full = pix_grid(shape, dev);
    auto q_pts_full = pix_grid(shape, dev);

    i64 N_total = 1; for (i64 d = 0; d < D; ++d) N_total *= N;

    std::mt19937 rng(7);
    std::normal_distribution<float> nd;
    std::vector<float> cr(N_total), ci(N_total);
    for (auto& v : cr) v = nd(rng);
    for (auto& v : ci) v = nd(rng);
    std::vector<std::complex<float>> cbuf(N_total);
    for (i64 i = 0; i < N_total; ++i) cbuf[i] = {cr[i], ci[i]};
    auto c_k = Tensor::from_blob(cbuf.data(), {N_total}, eScalarType::ComplexFloat, Device{eDeviceType::CPU})
                   .clone().to(dev);

    auto m_q_ref = dense_apply(phase_fn, k_pts_full, q_pts_full, c_k);

    // leaf_extent=1 forces L=log2(N) exactly, so box_k*box_q == N at every
    // admissible pairing (matching butterfly_rank_diagnostic.py's validated
    // regime) — anything coarser makes box_k*box_q = N*leaf_extent, growing
    // the phase variation within a box pair past what n_cheb nodes resolve.
    auto part = make_dyadic_partition(ArrayRef<i64>(shape), /*leaf_extent=*/1);
    auto bf = build_two_level(part, l_k, part, phase_fn, n_cheb, tol, max_rank, dev);

    {
        i64 min_r = bf.blocks[0][0].rank, max_r = bf.blocks[0][0].rank;
        for (auto& row : bf.blocks) for (auto& blk : row) {
            min_r = std::min(min_r, blk.rank);
            max_r = std::max(max_r, blk.rank);
        }
        std::cout << "    block rank range: [" << min_r << ", " << max_r << "] (n_cheb=" << n_cheb << ")\n";
    }

    auto m_q_bf = apply_two_level(bf, c_k);

    auto err = m_q_bf.sub(m_q_ref).abs();
    auto ref_norm = m_q_ref.abs().mean().item<float>();
    auto rel_err = err.mean().item<float>() / (ref_norm + 1e-30f);

    std::cout << "  N=" << N << " D=" << D << " delta_max=" << delta_max
              << " l_k=" << l_k << " (l_q=" << admissible_q_level(part.L, l_k) << ", L=" << part.L << ")"
              << " n_cheb=" << n_cheb
              << "  rel_err=" << std::scientific << rel_err
              << (rel_err < 1e-2 ? "  PASS" : "  FAIL") << "\n";

    return rel_err < 1e-2;
}

int main()
{
    std::cout << "=== Two-level butterfly factorization vs brute force ===\n\n";
    int failures = 0;

    if (false) {
        // delta_max=0: PURE plain-FFT phase (Phi=scale*k*q exactly, no warp at
        // all) — isolates the core Chebyshev/Lagrange machinery from anything
        // warp-specific. If this fails too, the bug is in compress_block/
        // apply_two_level's bookkeeping itself, not in resolving the warp's
        // phase variation.
        std::cout << "-- 1D, delta_max=0 (plain FFT phase, simplest possible case) --\n";
        failures += !run_case(64, 1, 0.0f, 2, 8, 1e-6, 16);

        // 1D sanity (cheapest possible — isolates the Chebyshev/Lagrange
        // machinery from D-dim broadcasting bugs before trusting 3D).
        std::cout << "\n-- 1D --\n";
        failures += !run_case(/*N=*/64, /*D=*/1, /*delta_max=*/2.0f, /*l_k=*/2, /*n_cheb=*/8, 1e-6, 16);
        failures += !run_case(64, 1, 8.0f, 2, 8, 1e-6, 16);

        // 3D, small enough for dense brute force (N^3, e.g. 16^3=4096).
        // max_rank bumped to n_cheb^D (216 for n_cheb=6,D=3) -- removes the
        // truncation cap entirely so a FAIL here means a real bug, not rank
        // starvation. Whether genuine compression (rank << n_cheb^D) exists at
        // this box pairing is a SEPARATE open question from correctness.
        std::cout << "\n-- 3D, FOV-center-ish (small delta) --\n";
        failures += !run_case(/*N=*/16, /*D=*/3, /*delta_max=*/1.0f, /*l_k=*/1, /*n_cheb=*/6, 1e-6, 216);

        std::cout << "\n-- 3D, larger displacement (closer to FOV-edge severity) --\n";
        failures += !run_case(16, 3, 4.0f, 1, 6, 1e-6, 216);
        failures += !run_case(16, 3, 4.0f, 2, 6, 1e-6, 216);   // different level pairing, same physics

        // Rank-vs-n_cheb scaling probe: same box pairing, growing n_cheb. If
        // rank tracks n_cheb^D (no plateau), the op is genuinely full-rank on
        // this pairing and butterfly buys nothing here -- not a bug, a fact
        // about this operator/pairing. If rank plateaus, n_cheb=6 was just too
        // coarse to see the real (bounded) admissible rank.
    }
    if (true) {
        std::cout << "\n-- rank-vs-n_cheb scaling probe (N=16,D=3,delta_max=4,l_k=1) --\n";
        //failures += !run_case(16, 3, 4.0f, 1, 4, 1e-6, 64);
        //failures += !run_case(16, 3, 4.0f, 1, 6, 1e-6, 216);
        failures += !run_case(16, 3, 4.0f, 1, 8, 1e-6, 512);
        failures += !run_case(16, 3, 4.0f, 1, 10, 1e-6, 1000);  // settles whether warp's plateau matches plain-FFT's (~190) or sits genuinely higher
        failures += !run_case(16, 3, 4.0f, 1, 12, 1e-6, 1728);  // n_cheb=10 was still at 42% of n_cheb^D, not yet clearly flat -- push one more step
    }

    if (false) {
        // ROOT QUESTION: does even PLAIN FFT (delta_max=0) plateau as n_cheb
        // grows, or does its rank ALSO climb with n_cheb^D? The diagnostic
        // script assumed plain-FFT-on-admissible-box is low rank "by
        // construction" but never tested this convergence -- it only compared
        // warped vs unwarped rank at a FIXED, small sample count (48). If
        // plain FFT itself doesn't plateau here, the diagnostic's whole
        // "EXTRA~=0" framing rests on a false floor.
        std::cout << "\n-- rank-vs-n_cheb probe, PLAIN FFT (N=16,D=3,delta_max=0,l_k=1) --\n";
        failures += !run_case(16, 3, 0.0f, 1, 4, 1e-6, 64);
        failures += !run_case(16, 3, 0.0f, 1, 6, 1e-6, 216);
        failures += !run_case(16, 3, 0.0f, 1, 8, 1e-6, 512);
        failures += !run_case(16, 3, 0.0f, 1, 10, 1e-6, 1000);
    
        std::cout << "\n-- rank-vs-n_cheb probe, PLAIN FFT, 1D (N=64,delta_max=0,l_k=2) --\n";
        failures += !run_case(64, 1, 0.0f, 2, 8, 1e-6, 16);
        failures += !run_case(64, 1, 0.0f, 2, 12, 1e-6, 24);
        failures += !run_case(64, 1, 0.0f, 2, 16, 1e-6, 32);
        failures += !run_case(64, 1, 0.0f, 2, 24, 1e-6, 48);
    }

    std::cout << "\n========================================\n"
              << "  " << failures << " failure(s)\n"
              << "========================================\n";
    return failures > 0 ? 1 : 0;
}
