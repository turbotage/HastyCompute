module;

export module hasty_mri_mod:butterfly_factorization;

import std;
import hasty_util_mod;
import hasty_tensor_mod;

import :butterfly_tree;
import :butterfly_id;

namespace hasty {
namespace mri {

// ── Two-level butterfly factorization (Phase A checkpoint) ──────────────────
//
// Single fixed admissible (l_k, l_q) pairing — block-low-rank exploitation
// of exp(i*Phi(k,q)) at ONE level, not the full O(N log N) recursive
// multilevel merge (that's a planned follow-up once this is verified). Cost
// here is O(n_boxes_k * n_boxes_q * (Chebyshev block cost)), better than
// dense O(Nk*Nq) but not the full butterfly asymptotic — this checkpoint
// validates the core math (Chebyshev compression + Lagrange anterpolation/
// interpolation + the actual CoordinateWarp phase function) end to end
// before adding the recursive merge across levels.
//
// Generic over direction: this same machinery serves BOTH r_to_q and q_to_r
// — they are NOT transposes of one another (confirmed by reading
// coordinate_warp.cppm: r_to_q's query positions are Newton-inverted
// r-space points against padded_r_shape_; q_to_r's are forward-warped
// q-space points against q_shape_ — two genuinely different point sets and
// grid shapes), so callers build TWO separate TwoLevelButterfly instances,
// one per direction, each with its own PhaseFn/k_part/q_part.

export struct TwoLevelButterfly {
    i64 D = 0;
    DyadicPartition k_part, q_part;
    i64 l_k = 0, l_q = 0;
    i64 n_cheb_per_axis = 0;
    PhaseFn phase_fn;   // kept — apply_two_level needs it for the exactly-evaluable reference-phase factors (see compress_block doc)
    std::vector<std::vector<i64>> k_boxes, q_boxes;          // multi-indices at (l_k, l_q)
    std::vector<std::vector<Tensor>> kc, qc;                  // [D]-tensors, box centers, [k_boxes.size()]/[q_boxes.size()]
    std::vector<std::vector<LowRankBlock>> blocks;            // [k_boxes.size()][q_boxes.size()]
};

namespace {
Tensor point_tensor(const std::vector<float>& pt, Device dev)
{
    return Tensor::from_blob((void*)pt.data(), {1, (i64)pt.size()}, eScalarType::Float, Device{eDeviceType::CPU})
               .clone().to(dev);
}
}

// k_part/q_part must share the same total level count L (k_part.L == q_part.L)
// — l_q is derived from l_k via admissible_q_level, not passed independently.
export TwoLevelButterfly build_two_level(
    const DyadicPartition& k_part, i64 l_k,
    const DyadicPartition& q_part,
    const PhaseFn& phase_fn,
    i64 n_cheb_per_axis, double tol, i64 max_rank, Device dev)
{
    const i64 D = k_part.D;
    const i64 l_q = admissible_q_level(k_part.L, l_k);

    TwoLevelButterfly bf;
    bf.D = D; bf.k_part = k_part; bf.q_part = q_part;
    bf.l_k = l_k; bf.l_q = l_q; bf.n_cheb_per_axis = n_cheb_per_axis;
    bf.phase_fn = phase_fn;
    bf.k_boxes = k_part.all_box_indices(l_k);
    bf.q_boxes = q_part.all_box_indices(l_q);

    auto k_ext = k_part.box_extent(l_k);
    auto q_ext = q_part.box_extent(l_q);
    std::vector<float> k_half(D), q_half(D);
    for (i64 d = 0; d < D; ++d) { k_half[d] = 0.5f * (float)k_ext[d]; q_half[d] = 0.5f * (float)q_ext[d]; }

    bf.kc.resize(bf.k_boxes.size());
    bf.qc.resize(bf.k_boxes.size());
    bf.blocks.resize(bf.k_boxes.size());
    for (std::size_t kb = 0; kb < bf.k_boxes.size(); ++kb) {
        auto kc_vec = k_part.box_center(l_k, bf.k_boxes[kb]);
        auto kc_t = point_tensor(kc_vec, dev);
        auto k_cheb = chebyshev_grid(D, n_cheb_per_axis, kc_vec, k_half, dev);

        bf.kc[kb].resize(bf.q_boxes.size());
        bf.qc[kb].resize(bf.q_boxes.size());
        bf.blocks[kb].resize(bf.q_boxes.size());
        for (std::size_t qb = 0; qb < bf.q_boxes.size(); ++qb) {
            auto qc_vec = q_part.box_center(l_q, bf.q_boxes[qb]);
            auto qc_t = point_tensor(qc_vec, dev);
            auto q_cheb = chebyshev_grid(D, n_cheb_per_axis, qc_vec, q_half, dev);
            bf.kc[kb][qb] = kc_t;
            bf.qc[kb][qb] = qc_t;
            bf.blocks[kb][qb] = compress_block(phase_fn, k_cheb, q_cheb, kc_t, qc_t, tol, max_rank);
        }
    }
    return bf;
}

namespace {

// Local (box-relative, [-1,1]-normalized) coords for every actual grid point
// within a box of integer extent `ext` per axis — i.e. the Chebyshev-domain
// position of grid offset 0..ext[d]-1 along axis d.
Tensor box_local_coords(const std::vector<i64>& ext, i64 D, Device dev)
{
    i64 total = 1;
    for (i64 d = 0; d < D; ++d) total *= ext[d];

    // Box center (per box_bounds/box_center in butterfly_tree.cppm) sits at
    // lo + 0.5*ext, in the SAME integer (no half-pixel) convention as the
    // global grid. Grid offset `idx` (0-based within the box) has global
    // coord lo+idx, so local (relative to center, normalized by halfwidth
    // 0.5*ext) = ((lo+idx)-(lo+0.5*ext)) / (0.5*ext) = (idx-0.5*ext)/(0.5*ext)
    // — NO extra "+0.5" term. (An earlier version had a spurious +0.5 here,
    // a half-pixel misalignment against where chebyshev_grid actually places
    // nodes relative to box_center — caught by butterfly_test.cpp's
    // brute-force cross-check.)
    std::vector<float> buf((std::size_t)total * D);
    std::vector<i64> idx(D, 0);
    for (i64 s = 0; s < total; ++s) {
        for (i64 d = 0; d < D; ++d)
            buf[(std::size_t)s * D + d] =
                ((float)idx[d] - 0.5f * (float)ext[d]) / (0.5f * (float)ext[d]);
        for (i64 d = D - 1; d >= 0; --d) {
            if (++idx[d] < ext[d]) break;
            idx[d] = 0;
        }
    }
    return Tensor::from_blob(buf.data(), {total, D}, eScalarType::Float, Device{eDeviceType::CPU})
               .clone().to(dev);
}

// GLOBAL (centered-pixel) coordinates of every actual grid point in a box —
// for evaluating phase_fn directly (the reference-phase factors need real
// coordinates, not box-local [-1,1] ones). bounds: [D] (lo,hi) per axis.
Tensor box_global_coords(const std::vector<std::pair<i64,i64>>& bounds, const std::vector<i64>& ext,
                         i64 D, Device dev)
{
    i64 total = 1;
    for (i64 d = 0; d < D; ++d) total *= ext[d];

    std::vector<float> buf((std::size_t)total * D);
    std::vector<i64> idx(D, 0);
    for (i64 s = 0; s < total; ++s) {
        for (i64 d = 0; d < D; ++d)
            buf[(std::size_t)s * D + d] = (float)(bounds[d].first + idx[d]);
        for (i64 d = D - 1; d >= 0; --d) {
            if (++idx[d] < ext[d]) break;
            idx[d] = 0;
        }
    }
    return Tensor::from_blob(buf.data(), {total, D}, eScalarType::Float, Device{eDeviceType::CPU})
               .clone().to(dev);
}

} // anonymous namespace

// Pure matvec, no normalization applied (matches apply_bandlimited_executor's
// convention in coordinate_warp.cppm — the 1/N inverse-FFT-style scaling is
// the CALLER's responsibility, applied once after this returns).
// c_k_flat: [N_k_total] ComplexFloat, flat in k_part.grid_shape's C order.
// Returns: [N_q_total] ComplexFloat, flat in q_part.grid_shape's C order.
export Tensor apply_two_level(const TwoLevelButterfly& bf, const Tensor& c_k_flat)
{
    const i64 D = bf.D;
    const Device dev = c_k_flat.device();

    auto c_k_nd = c_k_flat.reshape(ArrayRef<i64>(bf.k_part.grid_shape));

    auto k_ext = bf.k_part.box_extent(bf.l_k);
    auto q_ext = bf.q_part.box_extent(bf.l_q);
    i64 n_actual_k = 1; for (i64 d = 0; d < D; ++d) n_actual_k *= k_ext[d];
    i64 n_actual_q = 1; for (i64 d = 0; d < D; ++d) n_actual_q *= q_ext[d];
    i64 n_cheb = 1; for (i64 d = 0; d < D; ++d) n_cheb *= bf.n_cheb_per_axis;

    // Anterpolation (actual k -> k-box Chebyshev) and interpolation
    // (q-box Chebyshev -> actual q) matrices are IDENTICAL across all boxes
    // at a fixed level (same extent => same local-coord pattern) — built
    // ONCE, reused for every box.
    auto Lk = lagrange_matrix_ndim(box_local_coords(k_ext, D, dev), bf.n_cheb_per_axis, D)
                  .to(eScalarType::ComplexFloat);                      // [n_actual_k, n_cheb]
    auto Lq = lagrange_matrix_ndim(box_local_coords(q_ext, D, dev), bf.n_cheb_per_axis, D)
                  .to(eScalarType::ComplexFloat);                      // [n_actual_q, n_cheb]

    auto m_q_nd = zeros(ArrayRef<i64>(bf.q_part.grid_shape), TensorOptions(dev, eScalarType::ComplexFloat));

    // Pre-slice + global-coords for every k-box once (reused across all
    // q-boxes — only the reference-phase factors below depend on qb).
    std::vector<Tensor> c_actual_per_kbox(bf.k_boxes.size());
    std::vector<Tensor> k_global_per_kbox(bf.k_boxes.size());
    for (std::size_t kb = 0; kb < bf.k_boxes.size(); ++kb) {
        Tensor c_slice = c_k_nd;
        auto bounds = bf.k_part.box_bounds(bf.l_k, bf.k_boxes[kb]);
        for (i64 d = 0; d < D; ++d) {
            const i64 start = bounds[d].first + bf.k_part.grid_shape[d] / 2;
            c_slice = c_slice.narrow(d, start, k_ext[d]);
        }
        c_actual_per_kbox[kb] = c_slice.contiguous().reshape({n_actual_k});
        k_global_per_kbox[kb] = box_global_coords(bounds, k_ext, D, dev);
    }

    for (std::size_t qb = 0; qb < bf.q_boxes.size(); ++qb) {
        auto q_bounds = bf.q_part.box_bounds(bf.l_q, bf.q_boxes[qb]);
        auto q_global = box_global_coords(q_bounds, q_ext, D, dev);   // [n_actual_q, D]

        Tensor q_actual_total = zeros({n_actual_q}, TensorOptions(dev, eScalarType::ComplexFloat));

        for (std::size_t kb = 0; kb < bf.k_boxes.size(); ++kb) {
            const auto& blk = bf.blocks[kb][qb];
            const auto& kc = bf.kc[kb][qb];
            const auto& qc = bf.qc[kb][qb];

            // c'[k] = c_actual[k] * exp(i*Phi(k_actual, qc)) — the EXACTLY
            // evaluable (no interpolation) reference-phase factor that
            // compress_block's residual decomposition factored out (see its
            // doc). NOT optional — this is what makes the Chebyshev step
            // resolve the right (small, admissible) quantity instead of the
            // box's full, potentially huge, linear-in-k phase term.
            auto phase_k_qc = bf.phase_fn(k_global_per_kbox[kb], qc).reshape({n_actual_k}); // [n_actual_k]
            auto ref_k = phase_k_qc.to(eScalarType::ComplexFloat)
                             .mul(Scalar(std::complex<f32>(0.0f, 1.0f))).exp();
            auto c_prime = c_actual_per_kbox[kb].mul(ref_k);                 // [n_actual_k]

            auto c_prime_cheb = matmul(c_prime.unsqueeze(0), Lk).reshape({n_cheb});  // [n_cheb]

            auto tmp = matmul(c_prime_cheb.unsqueeze(0), blk.U);            // [1, rank]
            auto partial_cheb = matmul(tmp, blk.Vh).reshape({n_cheb});      // [n_cheb]

            // Interpolate back to actual q points (still missing the
            // reference phase removed above).
            auto partial_actual = matmul(Lq, partial_cheb.unsqueeze(1)).reshape({n_actual_q});

            // Reapply the OTHER exactly-evaluable reference factor:
            // exp(i*(Phi(kc,q_actual) - Phi(kc,qc))).
            auto phase_kc_q  = bf.phase_fn(kc, q_global).reshape({n_actual_q});
            auto phase_kc_qc = bf.phase_fn(kc, qc).reshape({1});
            auto ref_q = phase_kc_q.sub(phase_kc_qc)
                             .to(eScalarType::ComplexFloat)
                             .mul(Scalar(std::complex<f32>(0.0f, 1.0f))).exp();

            q_actual_total = q_actual_total.add(partial_actual.mul(ref_q));
        }

        Tensor dst = m_q_nd;
        for (i64 d = 0; d < D; ++d) {
            const i64 start = q_bounds[d].first + bf.q_part.grid_shape[d] / 2;
            dst = dst.narrow(d, start, q_ext[d]);
        }
        dst.copy_(q_actual_total.reshape(ArrayRef<i64>(q_ext)));
    }

    i64 n_q_total = 1;
    for (i64 d = 0; d < D; ++d) n_q_total *= bf.q_part.grid_shape[d];
    return m_q_nd.reshape({n_q_total});
}

}
}
