"""Shared helpers for the coil-map diagnostics (CPU only, single thread).

True maps on the reconstruction grid, as recon_p2np builds them (data.rs physics_grid): GNL-aware voxel positions,
trilinear sampling of smaps_true on the native grid, whitening W = Psi^{-1/2}, then the run's coil compression rows
(coil_basis.txt). Cached under the session scratchpad (not git-tracked)."""
import json, os, sys
import numpy as np
from scipy.ndimage import map_coordinates

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, os.path.join(ROOT, "python/lib"))
from hastycompute import store  # noqa: E402

PH = os.path.join(ROOT, "data/hasty/phantom/factory_arms/ph0001")
SCAN = os.path.join(PH, "sim_flow_10min/scan")
RUN = os.path.join(PH, "precond_compare/zmshr1")
CACHE = os.environ.get("SMAPS_CACHE", "/tmp/claude-1000/smaps_cache")
DIMS = (128, 72, 60)


def meta():
    return json.load(open(os.path.join(SCAN, "meta.json")))


def gnl_field(m, x):
    g = m["gnl"]; r2 = g["r0"] ** 2; r4 = r2 * r2
    t3, t5 = g["transverse"]; l3, l5 = g["longitudinal"]
    px, py, pz = x[..., 0], x[..., 1], x[..., 2]
    rho2 = px * px + py * py; z2 = pz * pz
    tr3 = 4 * z2 - rho2; tr5 = 8 * z2 * z2 - 12 * z2 * rho2 + rho2 * rho2
    fxy = 1 + t3 * tr3 / r2 + t5 * tr5 / r4
    fz = 1 + l3 * (2 * z2 - 3 * rho2) / r2 + l5 * (8 * z2 * z2 - 40 * z2 * rho2 + 15 * rho2 * rho2) / r4
    return np.stack([px * fxy, py * fxy, pz * fz], -1)


def gnl_invert(m, u):
    x = u.copy(); h = 1e-4
    for _ in range(8):
        f = gnl_field(m, x)
        J = np.empty(u.shape + (3,))
        for b in range(3):
            e = np.zeros(3); e[b] = h
            J[..., :, b] = (gnl_field(m, x + e) - gnl_field(m, x - e)) / (2 * h)
        x = x - np.linalg.solve(J, (f - u)[..., None])[..., 0]
    return x


def recon_positions(m, dims=DIMS, gnl=True):
    td = m["recon_dims"]; res = m["res"][0]
    vox = [td[a] * res / dims[a] for a in range(3)]
    ax = [(np.arange(dims[a]) - dims[a] // 2) * vox[a] for a in range(3)]
    u = np.stack(np.meshgrid(*ax, indexing="ij"), -1)  # [x,y,z,3]
    return gnl_invert(m, u) if gnl else u, vox


def whitener(m):
    psi = np.array(m["psi"], float)
    e, v = np.linalg.eigh(psi)
    return (v / np.sqrt(np.maximum(e, 1e-12))) @ v.T


def cc_rows(path=os.path.join(RUN, "coil_basis.txt")):
    rows = []
    for line in open(path):
        line = line.strip().rstrip(";")
        if not line:
            continue
        rows.append([complex(float(a), float(b)) for a, b in (p.split(",") for p in line.split(";"))])
    return np.array(rows)  # nvc x nc


def true_maps_cc(dims=DIMS):
    """whitened + compressed true maps on the reconstruction grid, [x,y,z,nvc] complex64 (cached)."""
    os.makedirs(CACHE, exist_ok=True)
    f = os.path.join(CACHE, f"true_cc_{dims[0]}x{dims[1]}x{dims[2]}.npy")
    if os.path.exists(f):
        return np.load(f)
    m = meta()
    x, _ = recon_positions(m, dims)
    res = m["res"][0]; td = m["recon_dims"]
    g = [np.clip(x[..., a] / res + td[a] // 2, 0, td[a] - 1 - 1e-9) for a in range(3)]
    st = store.load_any(os.path.join(SCAN, "smaps_true.hv"))  # [x,y,z,c]
    nc = st.shape[-1]
    s = np.empty(dims + (nc,), np.complex64)
    for c in range(nc):
        a = st[..., c]
        s[..., c] = map_coordinates(a.real, g, order=1, mode="nearest") + 1j * map_coordinates(a.imag, g, order=1, mode="nearest")
    del st
    W = whitener(m); R = cc_rows()
    out = np.einsum("vc,xyzc->xyzv", R @ W, s).astype(np.complex64)
    np.save(f, out)
    return out


def ip_err(est, tru):
    """per voxel 1 - |<est, tru/|tru|>| (recon_p2np's metric) and the orthogonal energy fraction 1 - |<.,.>|^2 / (|est|^2 |tru|^2)."""
    ip = np.abs(np.sum(np.conj(est) * tru, -1))
    nt = np.linalg.norm(tru, axis=-1); ne = np.linalg.norm(est, axis=-1)
    e1 = 1 - ip / np.maximum(nt, 1e-30)
    orth = 1 - (ip / np.maximum(nt * ne, 1e-30)) ** 2
    return e1, orth
