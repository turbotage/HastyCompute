module;

export module hasty_mri_mod:normal_toeplitz;

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_linalg_mod;
import hasty_fft_mod;
import hasty_threading_mod;
import :off_fourier_interpolators;

namespace hasty {
namespace mri {

// ─── NormalOffFourierToeplitzEmbeddings ──────────────────────────────────────
//
// Common struct for both the P×R reduced and naive L² normal operators.
//
//   (A^H A ρ)_i ≈ Σ_{outer,inner}
//       apply(basis_right, conj_right)[i]  ·
//       (h ★ (apply(basis_left, conj_left) · ρ))[i]
//
// Each Split:
//   spatial_basis_left  : (SPtr<CachedTensor>[nx,ny,nz], bool)
//       bool=false → MULT on input;  bool=true → MULT_CONJ on input
//   spatial_basis_right : (SPtr<CachedTensor>[nx,ny,nz], bool)
//       bool=false → MULT on output; bool=true → MULT_CONJ on output
//   kernel              : SPtr<CachedTensor>[2*nx,2*ny,2*nz] full complex kernel
//
// P×R case: basis_left.first == basis_right.first (same SPtr, Θ_rp)
//           basis_left.second=false, basis_right.second=true
// Naive L² : basis_left = (sptr_Υ_l1, false), basis_right = (sptr_Υ_l2, true)
//            When l1==l2: same SPtr — zero copy.

export struct NormalOffFourierToeplitzEmbeddings {
    struct Split {
        Pair<SPtr<CachedTensor>, bool> spatial_basis_left;    // input  mult
        Pair<SPtr<CachedTensor>, bool> spatial_basis_right;   // output mult
        SPtr<CachedTensor>             kernel;                 // [2*nx,2*ny,2*nz]
        double                         eig_val = 1.0;         // λ_{rp}; 1 for naive L²
    };
    Vec<Vec<Split>>   splits;     // [outer][inner]  (P×R or L×L)
    std::array<i64,3> im_size;    // {nx, ny, nz}
};

// ─── Strategy enums ──────────────────────────────────────────────────────────

export enum class eComputeStrategyBuildOffFourierEmbeddings {
    RUN_ALL_ON_INPUT_DEVICE = 0,
    SPLIT_PR_OVER_CUDA      = 1,
    SPLIT_P_OVER_CUDA       = 2,
    SPLIT_R_OVER_CUDA       = 3,
};

export enum class eStorageStrategyBuildOffFourierEmbeddings {
    STORE_ON_CPU  = 0,
    STORE_IN_FILE = 1,
};

export enum class eComputeStrategyApplyOffFourierNormal {
    RUN_ALL_ON_INPUT_DEVICE = 0,
    SPLIT_PR_OVER_CUDA      = 1,
    SPLIT_P_OVER_CUDA       = 2,
};

// ─── Internal helpers ────────────────────────────────────────────────────────

// Wrap a Tensor in a shared CachedTensor and apply the storage strategy.
static SPtr<CachedTensor> make_cached(Tensor t,
                                       eStorageStrategyBuildOffFourierEmbeddings storage)
{
    auto ct = std::make_shared<CachedTensor>(std::move(t));
    switch (storage) {
        case eStorageStrategyBuildOffFourierEmbeddings::STORE_ON_CPU:
            ct->move_to_cpu();   break;  // evicts CUDA copy immediately
        case eStorageStrategyBuildOffFourierEmbeddings::STORE_IN_FILE:
            ct->cache_in_file(); break;
    }
    return ct;
}

// Expand histogram-indexed spatial basis [n_hist] → [N] → [nx,ny,nz].
static Tensor expand_basis(
    const Tensor& basis_hist,    // [n_hist] ComplexFloat on dev
    const Tensor& mask_idx,      // [N_mask] long
    const Tensor& voxel_to_bin,  // [N_mask] long
    i64 N, i64 nx, i64 ny, i64 nz, Device dev)
{
    auto mask_dev  = mask_idx.to(dev);
    auto vtb_dev   = voxel_to_bin.to(dev);
    auto vox       = basis_hist.index_select(0, vtb_dev);  // [N_mask]
    auto flat      = zeros({N}, TensorOptions(dev, eScalarType::ComplexFloat));
    flat.scatter_(0, mask_dev, vox);
    return flat.reshape({nx, ny, nz}).contiguous();
}

// Build a Toeplitz kernel from weights w [K].
// Uses create_toeplitz_kernel_standard which supports complex weights without
// forcing Hermitian symmetry (no hermitify). Returns full [2*nx,2*ny,2*nz] kernel.
static Tensor make_kernel(
    const Tensor& coords,   // [3, K] float on dev
    const Tensor& w,        // [K] ComplexFloat on dev
    ArrayRef<i64> im_size,
    Device dev)
{
    auto kf = fft::create_toeplitz_kernel_standard(coords.to(dev), w.to(eScalarType::ComplexFloat).contiguous().to(dev), im_size);
    //fft::transform_toeplitz_kernel(kf);
    return kf;  // [2*nx, 2*ny, 2*nz] — full complex kernel
}


// ─── make_normal_off_fourier_toeplitz_embeddings ─────────────────────────────
//
// Builds P×R Toeplitz embeddings for the normal operator.
// SVD is run on the input device; kernel/basis building parallelises if requested.

export NormalOffFourierToeplitzEmbeddings make_normal_omega_driven_off_fourier_toeplitz_embeddings(
    const Tensor&                             coords,
    ArrayRef<i64>                             im_size,
    const Tensor&                             mask_idx,
    const Tensor&                             voxel_to_bin,
    const PhiLowrankResult&                   phi_lowrank,
    i64                                       p,
    i64                                       r,
    eComputeStrategyBuildOffFourierEmbeddings compute = eComputeStrategyBuildOffFourierEmbeddings::RUN_ALL_ON_INPUT_DEVICE,
    eStorageStrategyBuildOffFourierEmbeddings storage = eStorageStrategyBuildOffFourierEmbeddings::STORE_ON_CPU
) {
    const Tensor& Omega   = phi_lowrank.Omega;
    const Tensor& S_phi   = phi_lowrank.S;
    const Tensor& Upsilon = phi_lowrank.Upsilon;
    const i64 K = Omega.size(0), L = Omega.size(1);
    const i64 n_hist = Upsilon.size(0);
    const Device device = Omega.device();
    const i64 L2 = L * L;
    const i64 N_mask = mask_idx.size(0);
    const i64 nx = im_size[0], ny = im_size[1], nz = im_size[2];
    const i64 N  = nx * ny * nz;

    if (p < 1 || p > L2) throw std::invalid_argument("p must be in [1, L^2]");
    if (r < 1 || r > L)  throw std::invalid_argument("r must be in [1, L]");

    Tensor HermitianSqueezedNormalOmega = empty({L*(L+1)/2, K}, device, eScalarType::ComplexDouble);
    for (i64 l1 = 0, idx = 0; l1 < L; ++l1) {
        for (i64 l2 = 0; l2 <= l1; ++l2, ++idx) {
            auto weight = Omega.select(1, l1).mul(Omega.select(1, l2).conj())
                            .mul(S_phi.select(0, l1))
                            .mul(S_phi.select(0, l2));
            HermitianSqueezedNormalOmega.select(0, idx).copy_(weight.to(eScalarType::ComplexDouble));
        }
    }

    // op_M: maps [K] → [L*(L+1)/2] via M (mv), and [L*(L+1)/2] → [K] via M^H (rmv).
    auto mv_fn = [&HermitianSqueezedNormalOmega](const Tensor& x) -> Tensor {
        return HermitianSqueezedNormalOmega.mul(x.unsqueeze(0)).sum(1);  // [L*(L+1)/2]
    };
    auto rmv_fn = [&HermitianSqueezedNormalOmega](const Tensor& y) -> Tensor {
        return HermitianSqueezedNormalOmega.conj().mul(y.unsqueeze(1)).sum(0);  // [K]
    };
    const i64 L_tri = L * (L + 1) / 2;
    linalg::LinearOperator op_M(L_tri, K, std::move(mv_fn), std::move(rmv_fn),
                                 eScalarType::ComplexDouble, device);

    // Step 2: Truncated SVD of M → Λ [K,P], C_flat [P, L*(L+1)/2]
    // M = U[L_tri,P] * S[P] * Vh[P,K]
    // Lambda[k,p]     = Vh.T[k,p]  * sqrt(S[p])   — temporal/kernel weights
    // C_flat[p, idx]  = U.T[p,idx] * sqrt(S[p])   — spatial triangle coefficients
    auto svd_M = linalg::operator_svd(op_M, p,
        std::max(2*p+1, (i64)20), p + std::max(p, (i64)10));
    auto sqrt_S_M = svd_M.S.to(eScalarType::Double).clamp_min(0.0).sqrt();

    // Print M singular values (truncated at P — full energy fraction unknown without full SVD).
    {
        auto sv = svd_M.S.cpu().contiguous();
        const double* sp = sv.const_data_ptr<double>();
        std::cout << "  M svd [P=" << p << " L_tri=" << L_tri << "]:";
        for (i64 i = 0; i < sv.size(0); ++i)
            std::cout << " " << std::scientific << std::setprecision(2) << sp[i];
        if (sv.size(0) > 0)
            std::cout << "  (s_min/s_max=" << std::setprecision(3)
                      << sp[sv.size(0)-1] / (sp[0] + 1e-300) << ")";
        std::cout << "\n";
    }

    auto Lambda = svd_M.Vh.transpose(0, 1)
        .mul(sqrt_S_M.to(eScalarType::ComplexDouble).unsqueeze(0))
        .to(eScalarType::ComplexFloat);  // [K, P]
    auto C_flat = svd_M.U.transpose(0, 1)
        .mul(sqrt_S_M.to(eScalarType::ComplexDouble).unsqueeze(1));  // [P, L*(L+1)/2]

    // Steps 3 & 4: per-p SVD of C_p → R spatial bases + 1 kernel per (p,r).
    // Kernel depends only on Λ[:,p] → one kernel per p, shared across R.
    auto build_p_splits = [&](i64 pi, Device dev) -> Vec<NormalOffFourierToeplitzEmbeddings::Split>
    {
        // Expand lower-triangle C_flat[pi,:] → Hermitian [L,L] C_p.
        auto C_p_vec = C_flat.select(0, pi).to(dev).contiguous();  // [L*(L+1)/2]
        auto C_p = zeros({L, L}, TensorOptions(dev, eScalarType::ComplexDouble));
        for (i64 l1_i = 0, fidx = 0; l1_i < L; ++l1_i) {
            for (i64 l2_i = 0; l2_i <= l1_i; ++l2_i, ++fidx) {
                auto val = C_p_vec.select(0, fidx);  // 0-dim
                C_p.select(0, l1_i).select(0, l2_i).copy_(val);
                if (l1_i != l2_i)
                    C_p.select(0, l2_i).select(0, l1_i).copy_(val.conj());
            }
        }

        // Eigh of Hermitian C_p [L,L].
        // C_p = Σ_r λ_r v_r v_r^H (eigendecomposition).  Best rank-R Frobenius approx:
        // keep R eigenvalues of largest |λ| (may be negative for indefinite C_p).
        // Θ_{rp} = Υ v_r  (raw eigenvector, no √|λ| scaling).
        // Apply contribution: λ_r · conj(Θ_r[i]) · (h_p ★ (Θ_r[i] · ρ)).
        // eigh returns eigenvalues ascending → sort by |λ| descending.
        auto [eig_vals, eig_vecs] = linalg_eigh(C_p);
        auto sort_idx      = argsort(eig_vals.abs(), 0, /*descending=*/true);  // [L] long
        auto eig_vals_desc = eig_vals.index_select(0, sort_idx);              // [L] Double
        auto eig_vecs_desc = eig_vecs.index_select(1, sort_idx);              // [L,L] ComplexDouble

        // Print eigenvalue decay — negatives indicate indefinite C_p.
        {
            auto ev = eig_vals_desc.cpu().contiguous();
            const double* ep = ev.const_data_ptr<double>();
            double total = 0.0, captured = 0.0;
            for (i64 i = 0; i < L; ++i) total    += ep[i] * ep[i];
            for (i64 i = 0; i < r; ++i) captured += ep[i] * ep[i];
            std::cout << "  C_p[p=" << pi << "] eigh [R=" << r << " L=" << L << "]:";
            for (i64 i = 0; i < L; ++i)
                std::cout << " " << std::scientific << std::setprecision(2) << ep[i];
            std::cout << "  (rank-" << r << " energy="
                      << std::fixed << std::setprecision(1)
                      << (total > 0.0 ? 100.0 * captured / total : 100.0) << "%)\n";
        }

        // Θ_{jrp} = Σ_l v_r[l] · Υ[bin(j),l]  — raw eigenvector projected onto spatial basis.
        // λ_r stored in split and applied at operator time.
        auto eig_vecs_r = eig_vecs_desc.narrow(1, 0, r).to(eScalarType::ComplexFloat);  // [L, R]
        auto Upsilon_cf = Upsilon.to(eScalarType::ComplexFloat).to(dev);
        auto Upsilon_theta = mm(Upsilon_cf, eig_vecs_r);  // [n_hist, R]

        auto eig_vals_r_cpu = eig_vals_desc.narrow(0, 0, r).cpu().contiguous();
        const double* lam   = eig_vals_r_cpu.const_data_ptr<double>();

        // One kernel per p, shared across all R splits.
        auto lambda_pi = Lambda.select(1, pi).to(dev).contiguous();
        auto kernel_sptr = make_cached(make_kernel(coords, lambda_pi, im_size, dev), storage);

        Vec<NormalOffFourierToeplitzEmbeddings::Split> p_splits;
        p_splits.reserve(r);
        for (i64 ri = 0; ri < r; ++ri) {
            auto theta_hist = Upsilon_theta.select(1, ri).contiguous();
            auto theta_3d   = expand_basis(theta_hist, mask_idx, voxel_to_bin, N, nx, ny, nz, dev);
            auto theta_sptr = make_cached(std::move(theta_3d), storage);

            p_splits.push_back({
                {theta_sptr, false},   // left:  Θ_rp  MULT on input
                {theta_sptr, true},    // right: Θ*_rp MULT_CONJ on output (same sptr)
                kernel_sptr,
                lam[ri]               // eig_val = λ_r, applied at operator time
            });
        }
        return p_splits;
    };

    Vec<Vec<NormalOffFourierToeplitzEmbeddings::Split>> splits;
    splits.resize(p);

    if (compute == eComputeStrategyBuildOffFourierEmbeddings::RUN_ALL_ON_INPUT_DEVICE) {
        for (i64 pi = 0; pi < p; ++pi)
            splits[pi] = build_p_splits(pi, device);
    }
    else {
        auto& lb = global_cuda_load_balancer;
        using Fut = DepFuture<Vec<NormalOffFourierToeplitzEmbeddings::Split>>;
        Vec<Fut> futures;
        futures.reserve(p);
        for (i64 pi = 0; pi < p; ++pi)
            futures.push_back(lb.submit([&, pi](Device dev){ return build_p_splits(pi, dev); }));
        for (i64 pi = 0; pi < p; ++pi)
            splits[pi] = futures[pi].get();
        lb.sync();
    }

    return NormalOffFourierToeplitzEmbeddings{std::move(splits), {nx, ny, nz}};
}


// ─── make_normal_naive_off_fourier_toeplitz_embeddings ───────────────────────
//
// Builds L² Toeplitz embeddings — the exact naive normal operator expansion:
//
//   (A^H A ρ)_i ≈ Σ_{l1=0}^{L-1} Σ_{l2=0}^{L-1}
//       conj(Υ_l2[i]) · (h_{l1,l2} ★ (Υ_l1[i]·ρ))[i]
//
// h_{l1,l2} = F_adj(Ω[:,l1] · conj(Ω[:,l2]))
//
// Spatial bases (L of them) are precomputed and shared via SPtr:
//   splits[l1][l2].spatial_basis_left.first  == splits[l1][l2'].spatial_basis_left.first
//   splits[l1][l1].spatial_basis_left.first  == splits[l1][l1].spatial_basis_right.first (l1==l2)

export NormalOffFourierToeplitzEmbeddings make_normal_naive_off_fourier_toeplitz_embeddings(
    const Tensor&                             coords,
    ArrayRef<i64>                             im_size,
    const Tensor&                             mask_idx,
    const Tensor&                             voxel_to_bin,
    const PhiLowrankResult&                   phi_lowrank,
    eComputeStrategyBuildOffFourierEmbeddings compute = eComputeStrategyBuildOffFourierEmbeddings::RUN_ALL_ON_INPUT_DEVICE,
    eStorageStrategyBuildOffFourierEmbeddings storage = eStorageStrategyBuildOffFourierEmbeddings::STORE_ON_CPU
) {
    const Tensor& Omega   = phi_lowrank.Omega;
    const Tensor& S_phi   = phi_lowrank.S;
    const Tensor& Upsilon = phi_lowrank.Upsilon;
    const i64 K = Omega.size(0), L = Omega.size(1);
    const Device device = Omega.device();
    const i64 N_mask = mask_idx.size(0);
    const i64 nx = im_size[0], ny = im_size[1], nz = im_size[2];
    const i64 N  = nx * ny * nz;

    // Kernel weight: Omega_full[k,l] = Omega[k,l]*S[l]  (absorb full S into kernel).
    // Kernel weight product: S[l1]*S[l2]*Omega[k,l1]*conj(Omega[k,l2]).
    // Spatial bases use raw Upsilon (no S compensation) — S lives entirely in kernel.
    auto S_phi_f = S_phi.to(device);  // [L] Float

    // Precompute L expanded spatial bases using raw Upsilon.
    Vec<SPtr<CachedTensor>> basis_sptrs(L);
    for (i64 l = 0; l < L; ++l) {
        auto v_l = Upsilon.select(1, l).to(eScalarType::ComplexFloat).contiguous();
        auto expanded = expand_basis(v_l, mask_idx, voxel_to_bin, N, nx, ny, nz, device);
        basis_sptrs[l] = make_cached(std::move(expanded), storage);
    }

    // Build L² kernels and splits.
    auto build_l_splits = [&](i64 l1, Device dev)
        -> Vec<NormalOffFourierToeplitzEmbeddings::Split>
    {
        // Omega_full[:,l] = Omega[:,l] * S[l]  (multiply by full S)
        auto omega_full_l1 = Omega.select(1, l1).to(dev)
                                  .mul(S_phi_f.select(0, l1).to(dev))
                                  .to(eScalarType::ComplexFloat).contiguous();

        Vec<NormalOffFourierToeplitzEmbeddings::Split> row;
        row.reserve(L);
        for (i64 l2 = 0; l2 < L; ++l2) {
            auto omega_full_l2 = Omega.select(1, l2).to(dev)
                                      .mul(S_phi_f.select(0, l2).to(dev))
                                      .to(eScalarType::ComplexFloat).contiguous();
            auto weights = omega_full_l1.mul(omega_full_l2.conj()).contiguous();  // [K]
            auto kernel_sptr = make_cached(make_kernel(coords, weights, im_size, dev), storage);

            row.push_back({
                {basis_sptrs[l1], false},  // left:  Υ_l1 MULT on input
                {basis_sptrs[l2], true},   // right: Υ_l2 MULT_CONJ on output
                std::move(kernel_sptr)
            });
        }
        return row;
    };

    Vec<Vec<NormalOffFourierToeplitzEmbeddings::Split>> splits;
    splits.resize(L);

    if (compute == eComputeStrategyBuildOffFourierEmbeddings::RUN_ALL_ON_INPUT_DEVICE) {
        for (i64 l1 = 0; l1 < L; ++l1)
            splits[l1] = build_l_splits(l1, device);
    }
    else {
        auto& lb = global_cuda_load_balancer;
        using Fut = DepFuture<Vec<NormalOffFourierToeplitzEmbeddings::Split>>;
        Vec<Fut> futures;
        futures.reserve(L);
        for (i64 l1 = 0; l1 < L; ++l1)
            futures.push_back(lb.submit([&, l1](Device dev){ return build_l_splits(l1, dev); }));
        for (i64 l1 = 0; l1 < L; ++l1)
            splits[l1] = futures[l1].get();
        lb.sync();
    }

    return NormalOffFourierToeplitzEmbeddings{std::move(splits), {nx, ny, nz}};
}


// ─── make_normal_diagonal_off_fourier_toeplitz_embeddings ────────────────────
//
// SVD-optimal O(L) normal-operator approximation — the diagonal (l1==l2) slice
// of the naïve L² expansion:
//
//   (A^H A ρ)_i ≈ Σ_{l=0}^{L-1}
//       conj(Υ_l[i]) · (h_l ★ (Υ_l[i] · ρ))[i]
//
//   h_l = F_adj(|Ω_full[:,l]|²),   Ω_full[k,l] = Ω[k,l] · S[l]
//
// Equivalent to Θ_{ip} = Υ_{ip},  Λ_{kp} = |Ω_full[k,p]|².
// Optimal for minimising ‖N − N̂‖_F² (Eckart-Young on conj(N)).
// Cost: L Toeplitz convolutions per CG step (vs L² naïve, vs P×R omega-driven).

export NormalOffFourierToeplitzEmbeddings make_normal_diagonal_off_fourier_toeplitz_embeddings(
    const Tensor&                             coords,
    ArrayRef<i64>                             im_size,
    const Tensor&                             mask_idx,
    const Tensor&                             voxel_to_bin,
    const PhiLowrankResult&                   phi_lowrank,
    eComputeStrategyBuildOffFourierEmbeddings compute = eComputeStrategyBuildOffFourierEmbeddings::RUN_ALL_ON_INPUT_DEVICE,
    eStorageStrategyBuildOffFourierEmbeddings storage = eStorageStrategyBuildOffFourierEmbeddings::STORE_ON_CPU
) {
    const Tensor& Omega   = phi_lowrank.Omega;
    const Tensor& S_phi   = phi_lowrank.S;
    const Tensor& Upsilon = phi_lowrank.Upsilon;
    const i64 K = Omega.size(0), L = Omega.size(1);
    const Device device = Omega.device();
    const i64 nx = im_size[0], ny = im_size[1], nz = im_size[2];
    const i64 N  = nx * ny * nz;

    auto S_phi_f = S_phi.to(device);

    // Precompute L spatial bases (shared SPtr per l — left and right are same basis).
    Vec<SPtr<CachedTensor>> basis_sptrs(L);
    for (i64 l = 0; l < L; ++l) {
        auto v_l     = Upsilon.select(1, l).to(eScalarType::ComplexFloat).contiguous();
        auto expanded = expand_basis(v_l, mask_idx, voxel_to_bin, N, nx, ny, nz, device);
        basis_sptrs[l] = make_cached(std::move(expanded), storage);
    }

    auto build_l_split = [&](i64 l, Device dev)
        -> Vec<NormalOffFourierToeplitzEmbeddings::Split>
    {
        // Kernel weight: |Ω_full[k,l]|² = |Ω[k,l]·S[l]|² — real non-negative.
        auto omega_full_l = Omega.select(1, l).to(dev)
                                 .mul(S_phi_f.select(0, l).to(dev))
                                 .to(eScalarType::ComplexFloat).contiguous();
        auto weights = omega_full_l.mul(omega_full_l.conj()).contiguous();  // |Ω_full|² ≥ 0, real
        auto kernel_sptr = make_cached(make_kernel(coords, weights, im_size, dev), storage);

        return { NormalOffFourierToeplitzEmbeddings::Split{
            {basis_sptrs[l], false},  // left:  Υ_l MULT on input
            {basis_sptrs[l], true},   // right: Υ_l MULT_CONJ on output  (same SPtr as left)
            std::move(kernel_sptr)
            // eig_val = 1.0 (default)
        }};
    };

    Vec<Vec<NormalOffFourierToeplitzEmbeddings::Split>> splits(L);

    if (compute == eComputeStrategyBuildOffFourierEmbeddings::RUN_ALL_ON_INPUT_DEVICE) {
        for (i64 l = 0; l < L; ++l)
            splits[l] = build_l_split(l, device);
    } else {
        auto& lb = global_cuda_load_balancer;
        using Fut = DepFuture<Vec<NormalOffFourierToeplitzEmbeddings::Split>>;
        Vec<Fut> futures;
        futures.reserve(L);
        for (i64 l = 0; l < L; ++l)
            futures.push_back(lb.submit([&, l](Device dev){ return build_l_split(l, dev); }));
        for (i64 l = 0; l < L; ++l)
            splits[l] = futures[l].get();
        lb.sync();
    }

    return NormalOffFourierToeplitzEmbeddings{std::move(splits), {nx, ny, nz}};
}


// ─── apply_normal_off_fourier_operator ───────────────────────────────────────
//
// Works for both P×R and naive L² embeddings (same struct, same apply).
//
//   out[i] += Σ_{outer,inner}
//       mult(basis_right, conj_right)[i] ·
//       toeplitz(h, mult(basis_left, conj_left) · ρ)[i]
//
// Scratch [2*nx,2*ny,2*nz] allocated once, reused across all iterations.
// Kernel decompressed from [2*nx,2*ny,nz+1] before each Toeplitz call.

export Tensor apply_normal_toeplitz_off_fourier_operator(
    NormalOffFourierToeplitzEmbeddings&   embeddings,
    const Tensor&                          rho,
    eComputeStrategyApplyOffFourierNormal  compute = eComputeStrategyApplyOffFourierNormal::RUN_ALL_ON_INPUT_DEVICE
) {
    const auto [nx, ny, nz] = embeddings.im_size;
    const Device device = rho.device();
    const i64 P = (i64)embeddings.splits.size();
    const TensorOptions opts_c{device, eScalarType::ComplexFloat};

    auto output    = zeros({1, nx, ny, nz}, opts_c).contiguous();
    auto scratch   = zeros({2*nx, 2*ny, 2*nz}, opts_c).contiguous();
    auto rho_batch = rho.unsqueeze(0).contiguous();

    auto run_split = [&](NormalOffFourierToeplitzEmbeddings::Split& split,
                         Tensor& out, Tensor& scr, const Tensor& rho_b, Device dev)
    {
        auto& [sptr_l, conj_l] = split.spatial_basis_left;
        auto& [sptr_r, conj_r] = split.spatial_basis_right;

        const Tensor& t_left  = sptr_l->get_tensor(dev);
        const Tensor& t_right = sptr_r->get_tensor(dev);
        const Tensor& t_kern  = split.kernel->get_tensor(dev);
        const Tensor& kernel  = t_kern;  // [2*nx,2*ny,2*nz] full complex — no decompress needed

        auto scr_ref   = OptRefW<Tensor>{scr};
        auto left_ref  = OptCRefW<Tensor>{t_left};
        auto right_ref = OptCRefW<Tensor>{t_right};

        auto mult_in  = conj_l ? fft::ToeplitzMultType::MULT_CONJ : fft::ToeplitzMultType::MULT;
        auto mult_out = conj_r ? fft::ToeplitzMultType::MULT_CONJ : fft::ToeplitzMultType::MULT;

        if (split.eig_val == 1.0) {
            // Naive L² case (eig_val default): accumulate directly, no scaling needed.
            fft::toeplitz_multiplication(
                rho_b, out, kernel,
                scr_ref, left_ref, right_ref,
                fft::ToeplitzMultType::NONE,
                mult_in,
                fft::ToeplitzMultType::NONE,
                fft::ToeplitzMultType::NONE,
                mult_out,
                fft::ToeplitzAccumulateType::ACCUMULATE
            );
        } else {
            // P×R case: compute into temp, then out += eig_val * tmp.
            auto tmp = zeros_like(out);
            fft::toeplitz_multiplication(
                rho_b, tmp, kernel,
                scr_ref, left_ref, right_ref,
                fft::ToeplitzMultType::NONE,
                mult_in,
                fft::ToeplitzMultType::NONE,
                fft::ToeplitzMultType::NONE,
                mult_out,
                fft::ToeplitzAccumulateType::ACCUMULATE
            );
            out.add_(tmp, split.eig_val);
        }

        // Stream kernel through GPU — evict immediately after use.
        split.kernel->clear_cache_on_device(dev);
        // Evict basis unless left==right (diagonal, same SPtr).
        sptr_l->clear_cache_on_device(dev);
        if (sptr_l.get() != sptr_r.get())
            sptr_r->clear_cache_on_device(dev);
    };

    if (compute == eComputeStrategyApplyOffFourierNormal::RUN_ALL_ON_INPUT_DEVICE) {
        for (auto& row : embeddings.splits)
            for (auto& split : row)
                run_split(split, output, scratch, rho_batch, device);
    }
    else {
        auto& lb = global_cuda_load_balancer;
        const int N_dev = lb.num_devices();

        Vec<Tensor> partials, scratches;
        partials.reserve(N_dev);  scratches.reserve(N_dev);
        for (int di = 0; di < N_dev; ++di) {
            partials.push_back(zeros({1,nx,ny,nz}, TensorOptions(lb.device(di), eScalarType::ComplexFloat)).contiguous());
            scratches.push_back(zeros({2*nx,2*ny,2*nz}, TensorOptions(lb.device(di), eScalarType::ComplexFloat)).contiguous());
        }

        using Fut = DepFuture<void>;
        Vec<Fut> futures;
        for (i64 pi = 0; pi < P; ++pi)
            for (i64 ri = 0; ri < (i64)embeddings.splits[pi].size(); ++ri)
                futures.push_back(lb.submit([&, pi, ri](Device dev) {
                    int di = (int)dev.index;
                    auto rho_dev = rho_batch.to(dev).contiguous();
                    run_split(embeddings.splits[pi][ri], partials[di], scratches[di], rho_dev, dev);
                }));

        for (auto& f : futures) f.get();
        lb.sync();

        for (int di = 0; di < N_dev; ++di)
            output.add_(partials[di].to(device));
    }

    return output.squeeze(0);  // [nx, ny, nz]
}


}
}
