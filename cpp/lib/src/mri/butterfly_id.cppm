module;

export module hasty_mri_mod:butterfly_id;

import std;
import hasty_util_mod;
import hasty_tensor_mod;

namespace hasty {
namespace mri {

// ── Chebyshev-node low-rank compression for one admissible box pair ──────────
//
// Phase A uses SVD-based low-rank factors rather than a true interpolative
// decomposition (column-pivoted QR skeleton selection) — mathematically
// equivalent asymptotic complexity, what Candes-Demanet-Ying's original 2009
// paper itself used; this codebase has no QR primitive yet, so true ID is a
// stretch goal, not a Phase A blocker (see plan doc).
//
// Generic by design: PhaseFn is a callback, not tied to CoordinateWarp/GNL
// specifics — Phase B plugs in the actual u_inv (Newton-inverted) evaluation;
// Phase A's validation harness can use any phase function, including a
// standalone analytic one matching butterfly_rank_diagnostic.py's model.

// k_pts: [Nk, D], q_pts: [Nq, D] (centered-pixel-units, float) -> Phi: [Nk, Nq]
// (real-valued phase; the caller exponentiates — see compress_block). This is
// (2*pi/N) * k . u_inv(q) for CoordinateWarp's actual operator, but is left
// fully generic here.
export using PhaseFn = std::function<Tensor(const Tensor&, const Tensor&)>;

// Chebyshev-Gauss nodes on [-1,1]: x_j = cos((2j+1)*pi/(2n)), j=0..n-1.
export std::vector<float> chebyshev_nodes_1d(i64 n)
{
    using namespace std::numbers;
    std::vector<float> x(n);
    for (i64 j = 0; j < n; ++j)
        x[j] = (float)std::cos((2.0 * j + 1.0) * pi_v<f64> / (2.0 * n));
    return x;
}

// D-dimensional tensor-product Chebyshev grid, mapped affinely into
// [center-halfwidth, center+halfwidth] per axis. Returns [n^D, D] Float.
export Tensor chebyshev_grid(i64 D, i64 n_per_axis,
                             const std::vector<float>& center,
                             const std::vector<float>& halfwidth, Device dev)
{
    auto nodes = chebyshev_nodes_1d(n_per_axis);

    i64 total = 1;
    for (i64 d = 0; d < D; ++d) total *= n_per_axis;

    std::vector<float> buf((std::size_t)total * D);
    std::vector<i64> idx(D, 0);
    for (i64 s = 0; s < total; ++s) {
        for (i64 d = 0; d < D; ++d)
            buf[(std::size_t)s * D + d] = center[d] + halfwidth[d] * nodes[idx[d]];
        for (i64 d = D - 1; d >= 0; --d) {
            if (++idx[d] < n_per_axis) break;
            idx[d] = 0;
        }
    }

    return Tensor::from_blob(buf.data(), {total, D}, eScalarType::Float, Device{eDeviceType::CPU})
               .clone().to(dev);
}

// ── Lagrange anterpolation/interpolation between actual grid points and a
//    box's own Chebyshev nodes ──────────────────────────────────────────────
//
// C_k/the q-grid live on REGULAR grids, not on Chebyshev nodes — this is
// needed regardless of how many tree levels are used (not specific to
// multilevel recursion). Barycentric form for Chebyshev-Gauss (type-1)
// nodes: standard, numerically stable, avoids explicitly forming
// Lagrange basis polynomials.

// Barycentric weights for Chebyshev-Gauss nodes: w_j = (-1)^j * sin((2j+1)*pi/(2n)).
export std::vector<float> chebyshev_barycentric_weights(i64 n)
{
    using namespace std::numbers;
    std::vector<float> w(n);
    for (i64 j = 0; j < n; ++j) {
        const float sign = (j % 2 == 0) ? 1.0f : -1.0f;
        w[j] = sign * (float)std::sin((2.0 * j + 1.0) * pi_v<f64> / (2.0 * n));
    }
    return w;
}

// 1D barycentric Lagrange interpolation matrix: L[i,j] such that
// f(actual_local[i]) ~= Sum_j L[i,j]*f_cheb[j], for actual_local points
// ALREADY normalized to the same [-1,1] local coordinate as cheb_nodes
// (callers normalize via (global - center)/halfwidth before calling this).
// Returns [n_actual, n_cheb] Float on `dev`.
export Tensor lagrange_matrix_1d(const std::vector<float>& actual_local,
                                 const std::vector<float>& cheb_nodes,
                                 const std::vector<float>& weights, Device dev)
{
    const i64 n_actual = (i64)actual_local.size();
    const i64 n_cheb = (i64)cheb_nodes.size();
    std::vector<float> buf((std::size_t)n_actual * n_cheb);

    for (i64 i = 0; i < n_actual; ++i) {
        i64 exact = -1;
        for (i64 j = 0; j < n_cheb; ++j) {
            if (std::abs(actual_local[i] - cheb_nodes[j]) < 1e-6f) { exact = j; break; }
        }
        if (exact >= 0) {
            for (i64 j = 0; j < n_cheb; ++j) buf[(std::size_t)i * n_cheb + j] = (j == exact) ? 1.0f : 0.0f;
            continue;
        }
        float denom = 0.0f;
        std::vector<float> terms(n_cheb);
        for (i64 j = 0; j < n_cheb; ++j) {
            terms[j] = weights[j] / (actual_local[i] - cheb_nodes[j]);
            denom += terms[j];
        }
        for (i64 j = 0; j < n_cheb; ++j) buf[(std::size_t)i * n_cheb + j] = terms[j] / denom;
    }

    return Tensor::from_blob(buf.data(), {n_actual, n_cheb}, eScalarType::Float, Device{eDeviceType::CPU})
               .clone().to(dev);
}

// D-dimensional tensor-product Lagrange matrix via per-axis combination:
// L_full[point, cheb_multi_idx] = Prod_d L_axis_d[point, cheb_idx_d], built
// via repeated broadcasted elementwise multiply (each step materializes the
// growing dense product — standard tensor broadcast semantics, same pattern
// as make_pix_centered's per-axis expand in coordinate_warp.cppm).
// actual_local: [N, D] Float, each column ALREADY normalized to [-1,1] local
// coords for that point's own box. Returns [N, n_cheb^D] Float.
export Tensor lagrange_matrix_ndim(const Tensor& actual_local, i64 n_cheb_per_axis, i64 D)
{
    const Device dev = actual_local.device();
    const i64 N = actual_local.size(0);
    auto cheb_nodes = chebyshev_nodes_1d(n_cheb_per_axis);
    auto weights = chebyshev_barycentric_weights(n_cheb_per_axis);

    i64 total_cheb = 1;
    for (i64 d = 0; d < D; ++d) total_cheb *= n_cheb_per_axis;

    Tensor result;
    for (i64 d = 0; d < D; ++d) {
        auto col = actual_local.select(1, d).cpu().contiguous();
        const float* col_ptr = col.const_data_ptr<float>();
        std::vector<float> col_vec(col_ptr, col_ptr + N);
        auto Ld = lagrange_matrix_1d(col_vec, cheb_nodes, weights, dev);  // [N, n_cheb]

        std::vector<i64> bshape(D + 1, 1);
        bshape[0] = N;
        bshape[d + 1] = n_cheb_per_axis;
        auto Ld_b = Ld.reshape(ArrayRef<i64>(bshape));

        result = result.defined() ? result.mul(Ld_b) : Ld_b;
    }

    return result.contiguous().reshape({N, total_cheb});
}

// One admissible box pair's compressed factorization: exp(i*Phi) restricted
// to (k_pts, q_pts) (typically Chebyshev nodes of a box pair), truncated-SVD
// compressed. S is folded into U (U_scaled = U * S), so apply() is just
// U_scaled @ Vh — see butterfly_factorization.cppm.
export struct LowRankBlock {
    Tensor U;    // [Nk, rank] ComplexFloat (S already folded in)
    Tensor Vh;   // [rank, Nq] ComplexFloat
    i64 rank = 0;
};

// Compresses the RESIDUAL phase Phi_res(k,q) = Phi(k,q) - Phi(k,qc) -
// Phi(kc,q) + Phi(kc,qc), NOT the raw Phi — this is the key fix found by
// cross-checking ButterflyLab's reference implementation (fastBF.m's
// Ucell/Vcell construction): Phi itself carries the box's large linear-in-k
// and linear-in-q terms (the box can sit far from the origin — tens of
// radians of variation, since those terms scale with the box CENTER's
// distance from 0, not just its width), which Chebyshev interpolation with
// a handful of nodes cannot resolve. Phi_res cancels both linear terms
// EXACTLY (this decomposition is an algebraic identity, true for ANY Phi,
// not just bilinear ones) leaving only the genuinely admissible (~O(1),
// box-WIDTH-only-dependent) residual variation — THAT is what's actually
// low-rank and Chebyshev-compressible. The two cancelled linear terms are
// each exactly/directly evaluable (no interpolation needed) and get
// reapplied around the compressed block in apply_two_level.
// kc/qc: [D] box-center coordinates (NOT included in k_pts/q_pts — those
// are the Chebyshev nodes; kc/qc are the reference points the residual is
// taken relative to).
export LowRankBlock compress_block(const PhaseFn& phase_fn,
                                   const Tensor& k_pts, const Tensor& q_pts,
                                   const Tensor& kc, const Tensor& qc,
                                   double tol, i64 max_rank)
{
    auto Phi_kq   = phase_fn(k_pts, q_pts);   // [Nk, Nq]
    auto Phi_k_qc = phase_fn(k_pts, qc);      // [Nk, 1]
    auto Phi_kc_q = phase_fn(kc, q_pts);      // [1, Nq]
    auto Phi_kc_qc = phase_fn(kc, qc);        // [1, 1]

    auto Phi_res = Phi_kq.sub(Phi_k_qc).sub(Phi_kc_q).add(Phi_kc_qc);  // [Nk,Nq], broadcasts

    auto M = Phi_res.to(eScalarType::ComplexFloat)
                 .mul(Scalar(std::complex<f32>(0.0f, 1.0f)))
                 .exp();                 // [Nk, Nq] ComplexFloat = exp(i*Phi_res)

    // Chebyshev-node chirp matrices (delta_max=0 plain-FFT case especially)
    // are exact Vandermonde-like structured matrices -- LAPACK's gesdd
    // driver (the only one ATen exposes on CPU) can fail to converge on
    // these ("ill-conditioned" error) even though the matrix itself is
    // perfectly well-posed. A small ADDITIVE random perturbation (NOT a
    // uniform scaling, which preserves the exact structure and doesn't
    // help) breaks the degeneracy without affecting the rank decision --
    // magnitude is far below any realistic `tol`.
    {
        // rand_like is uniform [0,1) -- shift to [-1,1) before scaling so the
        // perturbation is centered (not a one-sided bias). Real-only noise is
        // enough to break the exact Vandermonde degeneracy.
        auto noise = rand_like(M.real()).sub(Scalar(0.5f)).mul(Scalar(2e-7f));
        M = M.add(noise.to(eScalarType::ComplexFloat));
    }

    auto [U, S, Vh] = linalg_svd(M, false);   // U:[Nk,r] S:[r] Vh:[r,Nq], r=min(Nk,Nq)

    auto S_cpu = S.cpu().contiguous();
    const i64 max_r = S_cpu.size(0);
    const float* Sp = S_cpu.const_data_ptr<float>();
    const float s0 = Sp[0] + 1e-30f;

    i64 rank = 1;
    for (i64 r = 1; r < max_r; ++r) {
        if (Sp[r] / s0 < (float)tol) break;
        ++rank;
    }
    rank = std::min(rank, max_rank);

    auto U_trunc  = U.narrow(1, 0, rank).contiguous();                         // [Nk, rank]
    auto S_trunc  = S.narrow(0, 0, rank).to(eScalarType::ComplexFloat);        // [rank]
    auto Vh_trunc = Vh.narrow(0, 0, rank).contiguous();                        // [rank, Nq]
    auto U_scaled = U_trunc.mul(S_trunc.unsqueeze(0));                         // [Nk, rank]

    return LowRankBlock{U_scaled, Vh_trunc, rank};
}

}
}
