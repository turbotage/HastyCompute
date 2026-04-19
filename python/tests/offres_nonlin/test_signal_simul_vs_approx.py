"""
test_signal_simul_vs_approx.py

Pipeline
--------
1. Extract voxels in the image support (mask).
2. Compress the spatial problem via feature histogram over (ΔB0, f_z2, f_xy):
     - n_rate bins for ΔB0,  n_nl bins for each nonlinear field.
     - Representative feature values per bin (weighted mean, weight = mag).
     - phi_lowrank runs on n_hist << N histogram bins → huge matvec speedup.
3. Subsample every phi_stride-th sample per spoke for the temporal matvec.
4. Interpolate Omega [K_sub, L] → [K, L] with per-spoke linear interpolation.
5. Evaluate approx signal:
     - omega_k  = Omega_full[k_idx]                                  [L]
     - agg[b]   = Σ_{j: bin[j]=b} mag[j]·coil[j]·exp(-i·k·r[j])   scatter
     - S_approx = (omega_k * (agg @ Upsilon_hist)).sum()
6. Compare with signal_simul (exact, on 50-point subsample).

Run from repo root:
    python -m python.tests.offres_nonlin.test_signal_simul_vs_approx
"""

import math
import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../lib'))

from hastycompute.mri.offres_nonlin import make_phi_matvec, phi_lowrank_krylov
from hastycompute.mri.signal_simul import signal_simul


# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

def make_problem(n_dim=256, n_spokes=500, n_samp=800, device='cpu'):
    """
    Synthetic 3-D MRI problem.  Nonlinear waveform amplitude is spoke-dependent:
      G_z2[s, j] = G_z2_amp * cos²(θ_s)              * g_bip[j]
      G_xy[s, j] = G_xy_amp * sin²(θ_s)·sin(2φ_s)/2  * g_bip[j]
    """
    gamma = 2 * math.pi * 42.577e6
    dt    = 4e-6
    FOV   = 0.08
    dx    = FOV / n_dim
    K     = n_spokes * n_samp

    xi = torch.linspace(-FOV/2 + dx/2, FOV/2 - dx/2, n_dim,
                        dtype=torch.float32, device=device)
    X, Y, Z = torch.meshgrid(xi, xi, xi, indexing='ij')
    coords_phys = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)
    N = coords_phys.shape[0]

    mag_flat = (coords_phys.norm(dim=1) < FOV * 0.4).float()
    mag      = mag_flat.reshape(n_dim, n_dim, n_dim)

    dB0_hz  = 500.0 * coords_phys[:, 2] / (FOV / 2)
    z_map   = (1j * 2 * math.pi * dB0_hz).to(torch.complex64)
    ratemap = dB0_hz.reshape(n_dim, n_dim, n_dim)

    coilmaps = torch.ones(1, n_dim, n_dim, n_dim,
                          dtype=torch.complex64, device=device)

    # Koosh-ball trajectory
    k_max = math.pi / dx
    idx   = torch.arange(n_spokes, dtype=torch.float32, device=device)
    theta = torch.arccos(1.0 - 2.0 * idx / n_spokes)
    phi_a = idx * math.pi * (3.0 - math.sqrt(5.0))
    dirs  = torch.stack([
        torch.sin(theta) * torch.cos(phi_a),
        torch.sin(theta) * torch.sin(phi_a),
        torch.cos(theta),
    ], dim=1)
    k_r          = torch.linspace(-k_max, k_max, n_samp,
                                  dtype=torch.float32, device=device)
    k_traj_rad_m = (dirs.unsqueeze(1) * k_r.unsqueeze(0).unsqueeze(-1)).reshape(-1, 3)
    k_traj_norm  = k_traj_rad_m * (FOV / (2 * math.pi))

    t_spoke = torch.arange(n_samp, dtype=torch.float32, device=device) * dt
    t = t_spoke.unsqueeze(0).expand(n_spokes, n_samp).reshape(-1).contiguous()

    half  = n_samp // 2
    g_bip = torch.cat([
         torch.ones(half,          dtype=torch.float32, device=device),
        -torch.ones(n_samp - half, dtype=torch.float32, device=device),
    ])

    # Nonlinear term 1: z²  (spoke-dependent amplitude via cos²θ)
    z_sq_max = (FOV / 2) ** 2
    G_z2_amp = 0.5 * (2 * math.pi) / (gamma * dt * half * z_sq_max)
    G_z2 = (dirs[:, 2].pow(2).unsqueeze(1)
            * (G_z2_amp * g_bip).unsqueeze(0)).reshape(-1)
    field_z2_flat = coords_phys[:, 2] ** 2
    field_z2_3d   = field_z2_flat.reshape(n_dim, n_dim, n_dim)
    alpha_z2 = (-gamma * dt
                * torch.cumsum(G_z2.to(torch.float64), dim=0)).float()

    # Nonlinear term 2: x·y  (spoke-dependent amplitude via sin²θ·sin2φ/2)
    xy_max   = (FOV / 2) ** 2
    G_xy_amp = 0.5 * math.pi / (gamma * dt * half * xy_max * 0.5)
    G_xy = ((dirs[:, 0] * dirs[:, 1]).unsqueeze(1)
            * (G_xy_amp * g_bip).unsqueeze(0)).reshape(-1)
    field_xy_flat = coords_phys[:, 0] * coords_phys[:, 1]
    field_xy_3d   = field_xy_flat.reshape(n_dim, n_dim, n_dim)
    alpha_xy = (-gamma * dt
                * torch.cumsum(G_xy.to(torch.float64), dim=0)).float()

    spatial_maps_flat = torch.stack([field_z2_flat, field_xy_flat], dim=1).float()
    nonlin_waveforms  = {'z_sq': G_z2, 'xy': G_xy}
    waveform_order    = ['z_sq', 'xy']
    simul_nonlinterms = [(field_z2_3d, alpha_z2), (field_xy_3d, alpha_xy)]

    return dict(
        n_dim=n_dim, N=N, K=K, FOV=FOV, dx=dx, gamma=gamma, dt=dt,
        mag=mag, ratemap=ratemap, coilmaps=coilmaps,
        coords_phys=coords_phys,
        k_traj_rad_m=k_traj_rad_m, k_traj_norm=k_traj_norm,
        t=t, z_map=z_map,
        spatial_maps_flat=spatial_maps_flat,
        nonlin_waveforms=nonlin_waveforms,
        waveform_order=waveform_order,
        simul_nonlinterms=simul_nonlinterms,
        alpha_z2=alpha_z2, alpha_xy=alpha_xy,
        n_spokes=n_spokes, n_samp=n_samp,
    )


# ---------------------------------------------------------------------------
# Histogram feature compression
# ---------------------------------------------------------------------------

def extract_histogram_features(prob, n_rate=64, n_nl=8):
    """
    Compress the spatial problem via a feature histogram over (ΔB0, f_z2, f_xy).

    Steps
    -----
    1. Keep only voxels in the image support (mag > 0).
    2. Bin each feature uniformly: n_rate bins for ΔB0, n_nl each for f_z2, f_xy.
    3. Compute weighted-mean feature representatives per occupied bin
       (weight = mag) for use in make_phi_matvec.
    4. Return voxel→bin mapping for the signal aggregation step.

    Returns
    -------
    mask_idx      [N_mask]       – flat indices of masked voxels in the image
    voxel_to_hist [N_mask]       – which histogram bin each masked voxel belongs to
    n_hist                       – number of occupied bins
    z_map_hist    [n_hist]       – representative z_map per bin  (complex64)
    sm_hist       [n_hist, 2]    – representative spatial fields per bin
    """
    device   = prob['mag'].device
    mag_flat = prob['mag'].flatten()

    mask     = mag_flat > 0
    mask_idx = mask.nonzero(as_tuple=True)[0]   # [N_mask]

    dB0  = prob['ratemap'].flatten()[mask_idx]            # [N_mask]
    f_z2 = prob['spatial_maps_flat'][:, 0][mask_idx]     # [N_mask]
    f_xy = prob['spatial_maps_flat'][:, 1][mask_idx]     # [N_mask]

    def bin_feat(x, n):
        lo, hi = x.min(), x.max()
        return ((x - lo) / (hi - lo + 1e-12) * (n - 1)).long().clamp(0, n - 1)

    bin_flat = (bin_feat(dB0, n_rate) * n_nl * n_nl
                + bin_feat(f_z2, n_nl) * n_nl
                + bin_feat(f_xy, n_nl)).long()

    unique_bins, voxel_to_hist = torch.unique(bin_flat, return_inverse=True)
    n_hist  = unique_bins.shape[0]
    weights = mag_flat[mask_idx]

    def weighted_mean(vals, n_hist):
        """Weighted mean per occupied bin using scatter."""
        wv  = torch.zeros(n_hist, dtype=vals.dtype,    device=device)
        cnt = torch.zeros(n_hist, dtype=weights.dtype, device=device)
        wv.scatter_add_(0, voxel_to_hist, vals * weights)
        cnt.scatter_add_(0, voxel_to_hist, weights)
        return wv / cnt.clamp(min=1e-30)

    dB0_hist  = weighted_mean(dB0,  n_hist)
    f_z2_hist = weighted_mean(f_z2, n_hist)
    f_xy_hist = weighted_mean(f_xy, n_hist)

    z_map_hist = (1j * 2 * math.pi * dB0_hist).to(torch.complex64)
    sm_hist    = torch.stack([f_z2_hist, f_xy_hist], dim=1).float()

    # Bin occupancy weights: Σ_{j∈bin} mag_j
    bin_weights = torch.zeros(n_hist, dtype=weights.dtype, device=device)
    bin_weights.scatter_add_(0, voxel_to_hist, weights)

    return dict(
        mask_idx=mask_idx,
        voxel_to_hist=voxel_to_hist,
        n_hist=n_hist,
        z_map_hist=z_map_hist,
        sm_hist=sm_hist,
        bin_weights=bin_weights,
    )


# ---------------------------------------------------------------------------
# Temporal interpolation of Omega
# ---------------------------------------------------------------------------

def interpolate_omega(Omega_sub, n_spokes, n_samp, phi_stride):
    """
    Linearly interpolate Omega from every phi_stride-th sample to all samples.

    Omega_sub : [K_sub, L]  complex  (K_sub = n_spokes * n_samp // phi_stride)
    Returns   : [K,    L]  complex

    Uses a manual interpolation that maps subsampled index j' exactly to full
    index j' * phi_stride, then linearly interpolates/extrapolates in between.
    F.interpolate(align_corners=True) places j'=n_sub-1 at i=n_samp-1 rather
    than i=(n_sub-1)*phi_stride, causing an end-of-spoke stretch that degrades
    higher-rank modes.
    """
    L     = Omega_sub.shape[1]
    n_sub = n_samp // phi_stride
    dev   = Omega_sub.device

    Ω = Omega_sub.reshape(n_spokes, n_sub, L)   # [n_spokes, n_sub, L]

    # Continuous sub-sampled position for each full index
    j      = torch.arange(n_samp, device=dev, dtype=torch.float32)
    j_cont = j / phi_stride                              # [n_samp]
    j_lo   = j_cont.long().clamp(0, n_sub - 2)          # [n_samp]
    j_hi   = j_lo + 1                                    # [n_samp]
    alpha  = (j_cont - j_lo.float()).unsqueeze(0).unsqueeze(-1)  # [1, n_samp, 1]

    Ω_lo = Ω[:, j_lo, :]                                # [n_spokes, n_samp, L]
    Ω_hi = Ω[:, j_hi, :]                                # [n_spokes, n_samp, L]

    return (Ω_lo + alpha * (Ω_hi - Ω_lo)).reshape(-1, L)  # [K, L]


# ---------------------------------------------------------------------------
# Approximate signal
# ---------------------------------------------------------------------------

def approx_signal_at(prob, hist, Upsilon_hist, Omega_full, subsample_idx):
    """
    Evaluate approx signal at the M subsampled (k_i, t_i) pairs.

        omega_k  = Omega_full[k_idx]                          [L]   temporal factor
        agg[b]   = Σ_{j: bin[j]=b} image_j                   [n_hist]  scatter
        S_approx = (omega_k * (agg @ Upsilon_hist)).sum()

    The scatter aggregation compresses N_mask voxels → n_hist bins before the
    [n_hist, L] dot product, keeping both steps memory-efficient.
    """
    coords_phys   = prob['coords_phys']
    k_traj        = prob['k_traj_rad_m']
    mag_flat      = prob['mag'].flatten().to(torch.complex64)
    coil_flat     = prob['coilmaps'].reshape(-1).to(torch.complex64)
    mask_idx      = hist['mask_idx']
    voxel_to_hist = hist['voxel_to_hist']
    n_hist        = hist['n_hist']

    mag_m  = mag_flat[mask_idx]
    coil_m = coil_flat[mask_idx]
    r_m    = coords_phys[mask_idx]

    C      = prob['coilmaps'].shape[0]
    M      = len(subsample_idx)
    device = prob['mag'].device

    S_approx = torch.zeros(C, M, dtype=torch.complex64, device=device)

    for i, k_idx in enumerate(subsample_idx):
        omega_k  = Omega_full[k_idx]                    # [L]
        phase    = (r_m * k_traj[k_idx].unsqueeze(0)).sum(-1)
        exp_ikr  = torch.exp(-1j * phase.to(torch.complex64))

        for c in range(C):
            image_m = mag_m * coil_m * exp_ikr         # [N_mask]

            # Scatter: N_mask → n_hist
            agg = torch.zeros(n_hist, dtype=torch.complex64, device=device)
            agg.scatter_add_(0, voxel_to_hist, image_m)

            nufft_basis      = agg @ Upsilon_hist       # [L]
            S_approx[c, i]   = (omega_k * nufft_basis).sum()

    return S_approx


# ---------------------------------------------------------------------------
# Exact signal via signal_simul
# ---------------------------------------------------------------------------

def exact_signal_at(prob, subsample_idx, k_batch_size=2, t_batch_size=4):
    """Evaluate signal_simul at M subsampled (k_i, t_i) pairs → [C, M]."""
    M      = len(subsample_idx)
    device = prob['mag'].device
    idx    = torch.tensor(subsample_idx, dtype=torch.long, device=device)

    nonlinterms = [(f3d, alpha[idx])
                   for f3d, alpha in prob['simul_nonlinterms']]

    sig = signal_simul(
        prob['mag'], prob['ratemap'], prob['coilmaps'],
        prob['k_traj_norm'][idx], prob['t'][idx], nonlinterms,
        k_batch_size=k_batch_size, t_batch_size=t_batch_size,
    )   # [C, M, M]

    arange_M = torch.arange(M, device=device)
    return sig[:, arange_M, arange_M]   # [C, M]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_dim       = 256 if device == 'cuda' else 32
    n_spokes    = 500
    n_samp      = 800
    n_subsample = 50
    L_values    = [4, 8, 12, 20]
    phi_stride  = 8      # subsample every 8th sample per spoke for matvec
    n_rate      = 2048    # histogram bins for ΔB0  (more = finer off-resonance grid)
    n_nl        = 512     # histogram bins for each nonlinear field

    torch.manual_seed(42)

    print("=" * 70)
    print("  signal_simul vs phi_lowrank (histogram+temporal) — accuracy test")
    print("=" * 70)

    prob   = make_problem(n_dim=n_dim, n_spokes=n_spokes,
                          n_samp=n_samp, device=device)
    T_ms   = n_samp * prob['dt'] * 1e3
    print(f"\n  device={device},  n_dim={n_dim}³,  N={prob['N']:,},  K={prob['K']:,}")
    print(f"  readout={T_ms:.1f} ms,  off-res peak ≈ ±{2*math.pi*1e3*T_ms*1e-3:.1f} rad")

    # ------------------------------------------------------------------
    # Step 1: histogram feature compression
    # ------------------------------------------------------------------
    print(f"\nExtracting histogram features (n_rate={n_rate}, n_nl={n_nl}) …",
          flush=True)
    hist = extract_histogram_features(prob, n_rate=n_rate, n_nl=n_nl)
    N_mask = hist['mask_idx'].shape[0]
    print(f"  N_mask={N_mask:,},  occupied bins n_hist={hist['n_hist']:,}")

    # ------------------------------------------------------------------
    # Step 2: subsample time points per spoke
    # ------------------------------------------------------------------
    n_samp   = prob['n_samp']
    spoke_sub = torch.arange(0, n_samp, phi_stride, device=device)
    offsets   = torch.arange(n_spokes, device=device) * n_samp
    sub_idx   = (offsets.unsqueeze(1) + spoke_sub.unsqueeze(0)).reshape(-1)
    K_sub     = sub_idx.shape[0]

    t_sub_phi  = prob['t'][sub_idx]

    # Correct subsampling: cumsum must be computed on the full waveform first,
    # then subsampled.  Pass differences so cumsum(w)[i] = cumsum(G_full)[sub_idx[i]].
    nl_sub_phi = {}
    for key, G in prob['nonlin_waveforms'].items():
        cumG     = torch.cumsum(G.double(), dim=0).float()   # [K] full integral
        cumG_sub = cumG[sub_idx]                              # [K_sub] correct K_q
        w = torch.cat([cumG_sub[:1], cumG_sub[1:] - cumG_sub[:-1]])
        nl_sub_phi[key] = w

    # phi_chunk can be large since n_hist << N
    phi_chunk = min(512, K_sub)

    print(f"  K_sub={K_sub:,} ({phi_stride}× subsampled),  phi_chunk={phi_chunk}")

    # ------------------------------------------------------------------
    # Step 3: phi_lowrank on histogram×subsampled grid
    # ------------------------------------------------------------------
    fwd, adj = make_phi_matvec(
        hist['z_map_hist'], hist['sm_hist'],
        nl_sub_phi, prob['waveform_order'],
        t_sub_phi, prob['gamma'], prob['dt'],
        chunk_size=phi_chunk,
    )

    # ------------------------------------------------------------------
    # Step 4: exact signal (reference, independent of L)
    # ------------------------------------------------------------------
    subsample_idx = torch.randperm(prob['K'], device=device)[:n_subsample].tolist()
    print(f"\nComputing exact signal (signal_simul) at {n_subsample} points …",
          flush=True)
    S_exact = exact_signal_at(prob, subsample_idx)
    print(f"  |S_exact| = {S_exact.norm():.4e}")

    # ------------------------------------------------------------------
    # Step 5: L sweep
    # ------------------------------------------------------------------
    print(f"\n{'L':>4}  {'n_hist':>7}  {'K_sub':>8}  {'rel err':>10}  "
          f"{'|S_approx|':>12}  status")
    print("-" * 58)

    for L in L_values:
        # Krylov (Lanczos) decomposition — no random noise, monotone in L
        # Weighted by bin occupancy so high-density bins dominate the SVD
        Omega_sub, Upsilon_hist = phi_lowrank_krylov(
            fwd, adj, hist['n_hist'], L,
            dtype=torch.complex64, device=device,
            weights=hist['bin_weights'],
        )

        # Temporal interpolation: Omega [K_sub, L] → [K, L]
        Omega_full = interpolate_omega(
            Omega_sub, n_spokes, n_samp, phi_stride
        )

        S_approx = approx_signal_at(
            prob, hist, Upsilon_hist, Omega_full, subsample_idx
        )
        err    = (S_exact - S_approx).norm() / S_exact.norm()
        status = "PASS" if err < 0.05 else "FAIL"
        print(f"{L:>4}  {hist['n_hist']:>7,}  {K_sub:>8,}  {err:>10.3e}  "
              f"{S_approx.norm():>12.4e}  {status}")

    print()


if __name__ == '__main__':
    main()
