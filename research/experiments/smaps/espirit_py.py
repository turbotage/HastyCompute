"""Small CPU ESPIRiT (Uecker 2014), same conventions as hasty_recon::espirit (kernels = conj of the right singular
vectors of the window matrix, W(r) = (1/k^3) sum_j v~_j v~_j^H, top eigenpair per voxel, phase aligned to a virtual
reference coil), evaluated exactly (direct separable DFT) on chosen axial slices instead of a coarse grid + trilinear.

calib: [cx,cy,cz,nc] centred calibration k-space (index i <-> k = i - c//2, units 1/FOV_cal).
FOV_cal may be an integer-free multiple f of the reconstruction FOV: positions r are given in units of FOV_cal."""
import numpy as np


def centred_fft(x, axes=(0, 1, 2)):
    return np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(x, axes=axes), axes=axes), axes=axes)


def crop_centre(k, c):
    o = [k.shape[a] // 2 - c[a] // 2 for a in range(3)]
    return k[o[0]:o[0] + c[0], o[1]:o[1] + c[1], o[2]:o[2] + c[2]]


def calibrate(calib, kernel=6, sv_rel=0.02, max_rows=8000, seed=0, iters=5):
    cd = calib.shape[:3]; nc = calib.shape[3]
    kk = [min(kernel, c) for c in cd]
    nw = [cd[a] - kk[a] + 1 for a in range(3)]
    P = nw[0] * nw[1] * nw[2]
    rng = np.random.default_rng(seed)
    pos = np.arange(P) if P <= max_rows else np.sort(rng.choice(P, max_rows, replace=False))
    px, py, pz = np.unravel_index(pos, nw)
    dx, dy, dz = np.meshgrid(*[np.arange(k) for k in kk], indexing="ij")
    # A[p, (dx,dy,dz,c)]
    A = calib[px[:, None, None, None] + dx[None], py[:, None, None, None] + dy[None], pz[:, None, None, None] + dz[None], :]
    A = A.reshape(len(pos), -1).astype(np.complex64)
    G = A.conj().T @ A
    del A
    n = G.shape[0]
    # randomized subspace iteration for the leading eigenpairs above sv_rel^2 * max
    b = min(n, 1024)
    while True:
        Q = np.linalg.qr(G @ (rng.standard_normal((n, b)) + 1j * rng.standard_normal((n, b))).astype(np.complex64))[0]
        for _ in range(iters):
            Q = np.linalg.qr(G @ Q)[0]
        T = Q.conj().T @ (G @ Q)
        e, U = np.linalg.eigh((T + T.conj().T) / 2)
        o = np.argsort(e)[::-1]; e = e[o]; U = U[:, o]
        if b < n and e[-1] > sv_rel ** 2 * e[0]:
            b = min(2 * b, n); continue
        break
    sv = np.sqrt(np.maximum(e, 0))
    keep = int(np.sum(sv > sv_rel * sv[0]))
    V = (Q @ U[:, :keep])                     # n x keep, rows (dx,dy,dz,c)
    kern = np.conj(V).reshape(kk[0], kk[1], kk[2], nc, keep)
    return kern, sv, keep


def maps_on_slices(kern, cdims_fov_mult, out_dims, zs, f=(1.0, 1.0, 1.0)):
    """maps on axial slices zs of the output grid out_dims (which spans FOV); the calibration FOV is f x FOV per axis.
    Returns (maps [len(zs),nx,ny,nc] unaligned, eig [len(zs),nx,ny])."""
    kx, ky, kz, nc, J = kern.shape
    nk = kx * ky * kz
    res_maps, res_eig = [], []
    rx = (np.arange(out_dims[0]) - out_dims[0] // 2) / (out_dims[0] * f[0])
    ry = (np.arange(out_dims[1]) - out_dims[1] // 2) / (out_dims[1] * f[1])
    Ex = np.exp(2j * np.pi * np.outer(rx, np.arange(kx)))
    Ey = np.exp(2j * np.pi * np.outer(ry, np.arange(ky)))
    for z in zs:
        rz = (z - out_dims[2] // 2) / (out_dims[2] * f[2])
        ez = np.exp(2j * np.pi * rz * np.arange(kz))
        t = np.tensordot(kern, ez, axes=([2], [0]))                  # kx,ky,nc,J
        t = np.tensordot(Ey, t, axes=([1], [1]))                     # ny,kx,nc,J
        v = np.tensordot(Ex, t, axes=([1], [1]))                     # nx,ny,nc,J
        v = v.astype(np.complex64)
        W = np.einsum("xyaj,xybj->xyab", v, v.conj(), optimize=True) / nk
        e, U = np.linalg.eigh(W)
        res_maps.append(U[..., -1]); res_eig.append(e[..., -1])
    return np.array(res_maps), np.array(res_eig)


def align(maps, eig, eig_min):
    """phase-align to the dominant direction over voxels with eig > eig_min (as the Rust code does)."""
    sel = eig > eig_min
    s = maps[sel] * np.sqrt(np.maximum(eig[sel], 0))[:, None]
    M = s.T @ s.conj()
    ev, evec = np.linalg.eigh(M)
    u = evec[:, -1]
    phi = np.einsum("...c,c->...", maps, u.conj())
    return maps * (np.conj(phi) / np.maximum(np.abs(phi), 1e-30))[..., None]
