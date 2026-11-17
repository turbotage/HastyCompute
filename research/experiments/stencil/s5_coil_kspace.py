"""How compact are receive-coil maps in k-space, on the oversampled grid?

A k-space coil kernel needs S_c only on the image box: any periodic extension over the sigma-grid period works.  So the
question is a Fourier-extension one: fit S_c on the box by a trig polynomial of period sigma * FOV with |m| <= (K-1)/2
per axis (K^3 coefficients = the k-space kernel support on the sigma-grid), least squares on the box (separable:
Kronecker of per-axis pseudo-inverses).  Report relative rms error on the body region inside the box.
Coils: Biot-Savart loops (B_x - i B_y), 32 loops (4 rings x 8) of radius 40 mm on an elliptic cylinder 170 x 115 mm;
box 198 x 189 x 228 mm (the 1 mm thorax box), 3 mm sample grid.  Virtual coils: SVD of the maps over the body, 99.9 %.
"""
import numpy as np

rng = np.random.default_rng(0)
h = 3.0
box = np.array([198.0, 189.0, 228.0])
ax = [np.arange(-b / 2 + h / 2, b / 2, h) for b in box]
X, Y, Z = np.meshgrid(*ax, indexing="ij")
P = np.stack([X, Y, Z], -1).reshape(-1, 3)
body = ((X / 165.0) ** 2 + (Y / 110.0) ** 2 <= 1).reshape(-1)


def loop_field(c, n, a, nseg=48):
    n = n / np.linalg.norm(n)
    u = np.cross(n, [0, 0, 1.0]); u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-9 else np.array([1.0, 0, 0])
    v = np.cross(n, u)
    th = np.linspace(0, 2 * np.pi, nseg + 1)
    pts = c + a * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
    mid = 0.5 * (pts[1:] + pts[:-1]); dl = pts[1:] - pts[:-1]
    B = np.zeros((len(P), 3))
    for s in range(nseg):
        d = P - mid[s]
        r3 = np.linalg.norm(d, axis=1) ** 3
        B += np.cross(dl[s], d) / r3[:, None]
    return B[:, 0] - 1j * B[:, 1]


coils = []
for zc in [-120.0, -40.0, 40.0, 120.0]:
    for k in range(8):
        t = 2 * np.pi * (k + 0.5) / 8
        c = np.array([175.0 * np.cos(t), 120.0 * np.sin(t), zc])
        nrm = np.array([np.cos(t) / 175.0, np.sin(t) / 120.0, 0.0])
        coils.append(loop_field(c, nrm, 40.0))
S = np.array(coils)                                     # (32, npts)
S = S / np.sqrt((np.abs(S[:, body]) ** 2).sum(0).mean())  # unit rms combined sensitivity over the body

# virtual coils (99.9 % energy over the body)
U, s, Vh = np.linalg.svd(S[:, body], full_matrices=False)
e = np.cumsum(s ** 2) / np.sum(s ** 2)
nv = int(np.searchsorted(e, 0.999) + 1)
Sv = U[:, :nv].conj().T @ S
print(f"physical 32 coils -> {nv} virtual (99.9 %)")
shape = X.shape


def fit_err(Smaps, K, sigma):
    errs = []
    Bp = []
    for a, b in zip(ax, box):
        m = np.arange(K) - (K - 1) / 2
        B = np.exp(2j * np.pi * np.outer(a, m) / (sigma * b))
        Bp.append(np.linalg.pinv(B, rcond=1e-8))
        Bp.append(B)
    for c in range(Smaps.shape[0]):
        f = Smaps[c].reshape(shape)
        coef = np.einsum("ia,jb,kc,abc->ijk", Bp[0], Bp[2], Bp[4], f, optimize=True)
        fit = np.einsum("ai,bj,ck,ijk->abc", Bp[1], Bp[3], Bp[5], coef, optimize=True)
        d = (fit - f).reshape(-1)[body]
        errs.append(np.sqrt(np.mean(np.abs(d) ** 2)) / np.sqrt(np.mean(np.abs(f.reshape(-1)[body]) ** 2)))
    errs = np.array(errs)
    w = (np.abs(Smaps[:, body]) ** 2).mean(1)
    return np.sqrt((errs ** 2 * w).sum() / w.sum()), errs.max()


print("| maps | sigma | K (kernel taps per axis) | energy-weighted rms rel. error | worst coil |")
print("|---|---|---|---|---|")
for nm, Sm in [("physical", S), ("virtual", Sv)]:
    for sigma in [1.25, 2.0]:
        for K in [3, 5, 7, 9, 13, 17, 25]:
            wr, mx = fit_err(Sm, K, sigma)
            print(f"| {nm} | {sigma} | {K} | {wr:.1e} | {mx:.1e} |", flush=True)
