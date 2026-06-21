module;

export module hasty_mri_mod:butterfly_tree;

import std;
import hasty_util_mod;
import hasty_tensor_mod;

namespace hasty {
namespace mri {

// ── Dyadic box partitioning for butterfly factorization ──────────────────────
//
// The operator being factored is m_q = M.C_k, M[q,k] = exp(i*(2pi/N)*k.u_inv(q))
// — see coordinate_warp.cppm's class doc for the derivation. Butterfly
// factorization's speedup comes from this matrix being numerically low-rank
// on "admissible" box pairs (box_k, box_q) sized so |box_k|*|box_q| ~ N
// (validated empirically for this specific operator in
// python/scripts/butterfly_rank_diagnostic.py).
//
// Both the k-space frequency grid AND the q-space grid (q_array, the
// REGULAR q-grid index — NOT the irregular u_inv(q) value, which only
// enters the phase function, not the partition coordinate) are already
// regular integer grids by construction (see make_pix_centered in
// coordinate_warp.cppm: same "i - N/2" centered convention used throughout
// this codebase). So partitioning is a fully analytic, deterministic dyadic
// decomposition — no recursive point-based tree-building (median splits,
// k-d trees) is needed, unlike a generic butterfly implementation that must
// handle scattered point sets. This is a meaningful simplification specific
// to this operator's structure.
//
// Levels: level 0 = root (1 box covering the whole grid, per axis), level L
// = finest (2^L boxes per axis). Admissible pairing: k-tree level l_k pairs
// with q-tree level l_q = L - l_k, keeping box_k_size * box_q_size constant
// (~ grid extent) across every pairing — this is the invariant
// butterfly_rank_diagnostic.py's box-size table encodes; see
// admissible_q_level below, unit-test it against that table directly.

export struct DyadicPartition {
    std::vector<i64> grid_shape;  // [D] — same convention as r_shape_/q_shape_
    i64 D = 0;
    i64 L = 0;   // total levels: level 0 (root) .. level L (finest)

    i64 n_boxes_per_axis(i64 level) const {
        return i64(1) << level;
    }

    // Box extent per axis at `level` (grid_shape[d] / 2^level). Requires
    // grid_shape[d] to be divisible by 2^L for all d — callers should pick L
    // accordingly (e.g. L = floor(log2(min_axis_extent))).
    std::vector<i64> box_extent(i64 level) const {
        const i64 n = n_boxes_per_axis(level);
        std::vector<i64> ext(D);
        for (i64 d = 0; d < D; ++d) ext[d] = grid_shape[d] / n;
        return ext;
    }

    // [lo, hi) integer-pixel bounds per axis for box multi-index `box_idx`
    // (length D, each in [0, n_boxes_per_axis(level))) at `level`, in the
    // SAME centered convention as make_pix_centered (coord = i - N/2).
    std::vector<std::pair<i64,i64>> box_bounds(i64 level, ArrayRef<i64> box_idx) const {
        auto ext = box_extent(level);
        std::vector<std::pair<i64,i64>> bounds(D);
        for (i64 d = 0; d < D; ++d) {
            const i64 lo = -(grid_shape[d] / 2) + box_idx[d] * ext[d];
            bounds[d] = {lo, lo + ext[d]};
        }
        return bounds;
    }

    // Center (float, centered-pixel units) of box `box_idx` at `level`.
    std::vector<float> box_center(i64 level, ArrayRef<i64> box_idx) const {
        auto bounds = box_bounds(level, box_idx);
        std::vector<float> c(D);
        for (i64 d = 0; d < D; ++d)
            c[d] = 0.5f * (float)(bounds[d].first + bounds[d].second);
        return c;
    }

    // All box multi-indices at `level`, as a flat [n_boxes, D] list — the
    // Cartesian product {0..n_boxes_per_axis-1}^D. n_boxes = n_boxes_per_axis^D.
    std::vector<std::vector<i64>> all_box_indices(i64 level) const {
        const i64 n = n_boxes_per_axis(level);
        i64 total = 1;
        for (i64 d = 0; d < D; ++d) total *= n;

        std::vector<std::vector<i64>> result;
        result.reserve(total);
        std::vector<i64> idx(D, 0);
        for (i64 s = 0; s < total; ++s) {
            result.push_back(idx);
            for (i64 d = D - 1; d >= 0; --d) {
                if (++idx[d] < n) break;
                idx[d] = 0;
            }
        }
        return result;
    }
};

export DyadicPartition make_dyadic_partition(ArrayRef<i64> grid_shape, i64 leaf_extent)
{
    DyadicPartition p;
    p.grid_shape.assign(grid_shape.begin(), grid_shape.end());
    p.D = (i64)grid_shape.size();

    i64 min_extent = grid_shape[0];
    for (i64 d = 1; d < p.D; ++d) min_extent = std::min(min_extent, grid_shape[d]);

    i64 L = 0;
    while ((min_extent >> (L + 1)) >= leaf_extent && (min_extent % (i64(1) << (L + 1))) == 0)
        ++L;
    p.L = L;
    return p;
}

// Admissible q-tree level for a given k-tree level, keeping
// box_k_extent * box_q_extent ~ const across all pairings (the butterfly
// admissibility condition — see class doc above and
// butterfly_rank_diagnostic.py's box_sizes_k/box_sizes_q sweep, which this
// must match: there, box_k=N/2^l_k pairs with box_q=N/2^(L-l_k)).
export i64 admissible_q_level(i64 L, i64 l_k)
{
    return L - l_k;
}

}
}
