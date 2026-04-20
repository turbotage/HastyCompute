import math

import torch


def signal_simul(mag, ratemap, coilmaps, coords, timepoints, nonlinterms,
                 k_batch_size=64, t_batch_size=None,
                 apply_ratemap=True, apply_nonlin=True):
    """
    Simulates MRI signal via direct DFT.

    Parameters:
    - mag:          [X, Y, Z]    real magnitude image
    - ratemap:      [X, Y, Z]    off-resonance map (Hz)
    - coilmaps:     [C, X, Y, Z] complex coil sensitivity maps
    - coords:       [num_k, 3]   k-space coordinates (cycles/FOV;
                                 k_norm = k_rad_m * FOV / 2π)
    - timepoints:   [T]          acquisition time points (seconds)
    - nonlinterms:  list of (field [X,Y,Z], alpha [T]) tuples;
                    each contributes phase exp(1j * field(r) * alpha(t))
    - k_batch_size: k-space points processed per iteration  (tune for memory)
    - t_batch_size: timepoints processed per iteration;
                    None = all at once (only safe for small T or small N)

    Returns:
    - signal: [C, num_k, T] complex simulated signal

    Memory footprint per iteration: O(N * max(k_batch_size, t_batch_size))
    For 256³ (N≈16M) use k_batch_size=2, t_batch_size=4 → ~500 MB peak.
    """
    X, Y, Z = mag.shape
    C = coilmaps.shape[0]
    num_k = coords.shape[0]
    T = timepoints.shape[0]
    device = mag.device

    if t_batch_size is None:
        t_batch_size = T

    # Voxel positions at cell centres: (i+0.5)/N - 0.5,  [N, 3]
    # Matches physical linspace(-FOV/2 + dx/2, FOV/2 - dx/2, N) / FOV
    def grid_coords(n):
        return (torch.arange(n, device=device, dtype=torch.float32) + 0.5) / n - 0.5

    gx, gy, gz = torch.meshgrid(grid_coords(X), grid_coords(Y), grid_coords(Z), indexing='ij')
    r = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)  # [N, 3]
    N = r.shape[0]

    # Precompute static spatial quantities
    ratemap_flat = ratemap.flatten().to(torch.complex64)          # [N]
    mag_flat     = mag.flatten().to(torch.complex64)              # [N]
    coil_flat    = coilmaps.reshape(C, N).to(torch.complex64)    # [C, N]
    spatial_wt   = mag_flat.unsqueeze(0) * coil_flat             # [C, N]

    nonlin_fields = [(f.flatten().to(torch.complex64), a) for f, a in nonlinterms] \
                    if nonlinterms else []

    signal = torch.zeros(C, num_k, T, dtype=torch.complex64, device=device)
    coords_f = coords.to(torch.float32)

    for t0 in range(0, T, t_batch_size):
        t1 = min(t0 + t_batch_size, T)
        t_batch = timepoints[t0:t1].to(torch.complex64)          # [t_B]

        # Time-phase for this t-batch: [N, t_B]
        if apply_ratemap:
            time_phase = torch.exp(
                -2j * math.pi * ratemap_flat.unsqueeze(1) * t_batch.unsqueeze(0)
            )
        else:
            time_phase = torch.ones(N, t1 - t0, dtype=torch.complex64, device=device)

        if apply_nonlin:
            for field_flat, alpha in nonlin_fields:
                alpha_batch = alpha[t0:t1].to(torch.complex64)   # [t_B]
                time_phase = time_phase * torch.exp(
                    1j * field_flat.unsqueeze(1) * alpha_batch.unsqueeze(0)
                )

        for k0 in range(0, num_k, k_batch_size):
            k1 = min(k0 + k_batch_size, num_k)
            k_batch = coords_f[k0:k1]                            # [k_B, 3]

            # DFT phase: [k_B, N]
            dft_phase = torch.exp(
                -2j * math.pi * (k_batch @ r.T).to(torch.complex64)
            )

            for c in range(C):
                # [k_B, N] * [N] → [k_B, N] @ [N, t_B] → [k_B, t_B]
                signal[c, k0:k1, t0:t1] = (
                    dft_phase * spatial_wt[c].unsqueeze(0)
                ) @ time_phase

    return signal
