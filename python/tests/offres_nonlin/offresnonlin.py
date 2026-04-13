
import torch
import math
from hastycompute.plot import orthoslicer_old
import h5py

def real_spherical_harmonics(lmin, lmax, theta, phi):
    """
    Compute real spherical harmonics Y_lm^real(theta, phi)
    for lmin <= l <= lmax, -l <= m <= l.

    Args:
        lmin (int): minimum degree (>=0)
        lmax (int): maximum degree
        theta (torch.Tensor): polar angle [0, pi], shape (...)
        phi (torch.Tensor): azimuthal angle [0, 2pi], shape (...)

    Returns:
        dict[(l,m)] -> torch.Tensor with same shape as theta
    """
    x = torch.cos(theta)
    P = {}

    # Base case
    P[(0,0)] = torch.ones_like(x)

    # Diagonal P_m^m
    for m in range(1, lmax+1):
        P[(m,m)] = (-1.0)**m * torch.prod(
            torch.arange(1, 2*m, 2, device=x.device, dtype=x.dtype)
        ) * (1 - x**2).pow(m/2)

    # Off-diagonal P_{m+1}^m
    for m in range(0, lmax):
        P[(m+1, m)] = (2*m+1) * x * P[(m,m)]

    # General recurrence
    for m in range(0, lmax+1):
        for l in range(m+2, lmax+1):
            P[(l,m)] = ((2*l-1) * x * P[(l-1,m)] - (l+m-1) * P[(l-2,m)]) / (l-m)

    Y = {}

    for l in range(lmin, lmax+1):
        for m in range(0, l+1):
            # Normalization factor
            K = math.sqrt((2*l+1)/(4*math.pi) * math.factorial(l-m)/math.factorial(l+m))
            P_lm = P[(l,m)]

            if m == 0:
                Y[(l,0)] = K * P_lm
            else:
                # Positive m -> cos(m phi)
                Y[(l,m)] = math.sqrt(2) * K * P_lm * torch.cos(m*phi)
                # Negative m -> sin(|m| phi)
                Y[(l,-m)] = math.sqrt(2) * K * P_lm * torch.sin(m*phi)

    del P

    return Y


def histogram_lowrank_basis(
    f_maps,        # Tensor of shape (Q, Nx, Ny, Nz): spatial feature maps
    psi_t,         # Tensor of shape (K, Q): temporal coefficients ψ_q(t_k)
    mask,          # Boolean tensor of shape (Nx, Ny, Nz): object support
    n_bins,        # List or tensor of length Q: number of bins per feature
    rank_L,        # Desired rank of the approximation
    device="cuda"  # "cpu" or "cuda"
):
    """
    Computes a histogram-based low-rank approximation of exp(Φ(r, t)).
    
    Returns:
        sigma_kl : (K, L) temporal basis coefficients
        eta_bl   : (B, L) spatial basis values at histogram bin centers
        bin_indices : (N_support,) bin index for each voxel in the support
        bin_centers : (B, Q) representative feature vectors
        weights : (B,) histogram weights
    """
    
    Q = f_maps.shape[0]
    K = psi_t.shape[0]

    # ------------------------------------------------------------
    # 1. Extract feature vectors within the object support
    # ------------------------------------------------------------
    f_flat = f_maps.reshape(Q, -1).T  # Shape: (N_voxels, Q)
    mask_flat = mask.reshape(-1)
    f_support = f_flat[mask_flat]     # Shape: (N_support, Q)

    # ------------------------------------------------------------
    # 2. Construct bin edges and centers for each feature dimension
    # ------------------------------------------------------------
    bin_edges = []
    bin_centers_1d = []
    for q in range(Q):
        f_min = f_support[:, q].min()
        f_max = f_support[:, q].max()
        edges = torch.linspace(f_min, f_max, n_bins[q] + 1, device=device)
        centers = 0.5 * (edges[:-1] + edges[1:])
        bin_edges.append(edges)
        bin_centers_1d.append(centers)

    # ------------------------------------------------------------
    # 3. Assign each voxel to a multidimensional histogram bin
    # ------------------------------------------------------------
    bin_indices_per_dim = []
    for q in range(Q):
        idx = torch.bucketize(f_support[:, q], bin_edges[q]) - 1
        idx = torch.clamp(idx, 0, n_bins[q] - 1)
        bin_indices_per_dim.append(idx)

    # Convert multi-dimensional indices to a single linear index
    multipliers = torch.tensor(
        [torch.prod(torch.tensor(n_bins[q+1:])) if q < Q-1 else 1
         for q in range(Q)],
        device=device
    )
    bin_indices = sum(
        idx * multipliers[q] for q, idx in enumerate(bin_indices_per_dim)
    )

    B = int(torch.prod(torch.tensor(n_bins)))  # Total number of bins

    # ------------------------------------------------------------
    # 4. Compute histogram weights
    # ------------------------------------------------------------
    weights = torch.bincount(bin_indices, minlength=B).float()

    # Identify non-empty bins
    nonzero_bins = weights > 0
    weights = weights[nonzero_bins]
    B_eff = weights.numel()

    # ------------------------------------------------------------
    # 5. Compute bin centers in feature space
    # ------------------------------------------------------------
    # Create grid of all bin centers
    mesh = torch.meshgrid(*bin_centers_1d, indexing='ij')
    bin_centers = torch.stack([m.reshape(-1) for m in mesh], dim=1)
    bin_centers = bin_centers[nonzero_bins]  # Keep only occupied bins

    # ------------------------------------------------------------
    # 6. Construct reduced matrix H_{k b}
    #    H[k, b] = exp(-i * psi_t[k] @ f_b)
    # ------------------------------------------------------------
    phase = psi_t @ bin_centers.T  # Shape: (K, B_eff)
    H = torch.exp(-1j * phase)

    # ------------------------------------------------------------
    # 7. Weighted SVD: SVD of H * sqrt(W)
    # ------------------------------------------------------------
    W_sqrt = torch.sqrt(weights)
    H_weighted = H * W_sqrt.unsqueeze(0)

    # Compute truncated SVD
    U, S, Vh = torch.linalg.svd(H_weighted, full_matrices=False)

    # Retain desired rank
    U = U[:, :rank_L]
    S = S[:rank_L]
    Vh = Vh[:rank_L, :]

    # Temporal basis
    sigma_kl = U * S.unsqueeze(0)  # Shape: (K, L)

    # Spatial basis at bin centers
    eta_bl = (Vh.conj().T) / W_sqrt.unsqueeze(1)  # Shape: (B_eff, L)

    return sigma_kl, eta_bl, bin_indices, bin_centers, weights



def map_eta_to_voxels(eta_bl, bin_indices, mask, volume_shape):
    """
    Maps spatial basis values from histogram bins back to voxels.
    """
    L = eta_bl.shape[1]
    eta_voxels = torch.zeros((*volume_shape, L), dtype=eta_bl.dtype, device=eta_bl.device)

    eta_support = eta_bl[bin_indices]  # Assign basis values
    eta_voxels = eta_voxels.reshape(-1, L)
    eta_voxels[mask.reshape(-1)] = eta_support
    eta_voxels = eta_voxels.reshape(*volume_shape, L)

    return eta_voxels



# # Example parameters
# Q = 4  # e.g., off-resonance + 3 spherical harmonics
# n_bins = [64, 8, 8, 8]  # Finer resolution for off-resonance
# rank_L = 10

# sigma_kl, eta_bl, bin_indices, bin_centers, weights = \
#     histogram_lowrank_basis(
#         f_maps=f_maps,
#         psi_t=psi_t,
#         mask=mask,
#         n_bins=n_bins,
#         rank_L=rank_L,
#         device="cuda"
#     )

# eta_voxels = map_eta_to_voxels(
#     eta_bl, bin_indices, mask, volume_shape=f_maps.shape[1:]
# )



















if False:
    shape = (320,320,320)

    x = torch.ones(shape, dtype=torch.float32).to('cuda')
    idx = x.nonzero(as_tuple=True)

    x = idx[0].float() - 0.5*(shape[0]-1)
    y = idx[1].float() - 0.5*(shape[1]-1)
    z = idx[2].float() - 0.5*(shape[2]-1)

    r = torch.square(x) + torch.square(y) + torch.square(z)
    r.sqrt_()

    mask = r < (shape[0]//2)

    theta = torch.acos(z/(r + 1e-8))
    phi = torch.atan2(y, x)

    Y = real_spherical_harmonics(0, 2, theta, phi)

    ylist = []
    for y in Y.values():
        ylist.append(y.view(shape))
    y = torch.stack(ylist, dim=0)

    off_resonance_map = 1.0/(0.200*torch.exp(-8*torch.square(r/r.max()))) + 1j*torch.exp(-12*torch.square(r/r.max()))

    #with h5py.File('/home/turbotage/Documents/4DRecon/other_data/framed_true.h5', 'r') as f:
    #    img = f['image'][:]

    nspokes = 8000
    nsamp_per_spoke = 2563
    nframes = 20

    #off_resonance_map = None
    #torch.empty((nspokes // nframes, nsamp_per_spoke), dtype=torch.complex64, device='cuda')
    #for spoke in range(nspokes // nframes):
    #    t = 0
    #    for samp in range(nsamp_per_spoke):
    
            
    orthoslicer_old.image_nd(y.cpu().numpy())



    print('Hello')