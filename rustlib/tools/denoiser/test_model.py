"""CPU checks of the equivariant denoiser (float64): kernel constraint, O_h / scale / cyclic-time equivariance of the full
model in raw 7p space, identity at initialisation, tiled = whole-volume inference, TorchScript trace with dynamic shapes.

    python -I test_model.py [--compile]
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _env  # noqa: E402,F401  (caches inside the repo; before torch)
import torch  # noqa: E402
from dataset import flip_pair  # noqa: E402
from model import Denoiser7p, denoise_volume, receptive_field  # noqa: E402
from oh import FieldType, OhConv3d, OhKernel, apply_group, group, regular, rep, stencil  # noqa: E402

torch.manual_seed(0)
torch.set_num_threads(4)

# 1. kernel constraint K(g p) = ρ_out(g) K(p) ρ_in(g)⁻¹ for every g
for gname in ["Oh", "O"]:
    t_in = FieldType(gname, [("trivial", 1), ("perm3", 1), ("vector", 1), ("regular", 1)])
    t_out = FieldType(gname, [("vector", 1), ("regular", 1), ("trivial", 1)])
    K = OhKernel(t_in, t_out, 3).double()
    for prm in K.params:
        prm.data.normal_()
    W = K.weight()[:, :, 0].detach().numpy()  # [D_out, D_in, 27]
    mats, _ = group(gname)
    _, act = stencil(3, gname)
    worst = 0.0
    for g in range(len(mats)):
        def mat(ft):
            p, s = ft.channel_perm(g)
            M = np.zeros((ft.dim, ft.dim))
            M[p, np.arange(ft.dim)] = s
            return M
        Mo, Mi = mat(t_out), mat(t_in)
        lhs = W[:, :, act[g]]  # K(g p) at stencil index j
        rhs = np.einsum("ab,bcj,cd->adj", Mo, W, Mi.T)
        worst = max(worst, np.abs(lhs - rhs).max())
    print(f"1. {gname}: kernel constraint max violation {worst:.1e} over {len(mats)} elements, {sum(p.numel() for p in K.params)} params")
    assert worst < 1e-12

# 2. layer: conv(g f) = g conv(f)
t_in = FieldType("Oh", [("trivial", 2), ("perm3", 2), ("vector", 2)])
conv = OhConv3d(t_in, regular("Oh", 1), 3, stride=2).double()
f = torch.randn(1, t_in.dim, 9, 9, 9, dtype=torch.float64)
worst = max(float((conv(apply_group(f, t_in, g, (2, 3, 4))) - apply_group(conv(f), regular("Oh", 1), g, (2, 3, 4))).abs().max()) for g in range(48))
print(f"2. strided conv equivariance (odd size): max error {worst:.1e}")
assert worst < 1e-10


# 3. full model in raw 7p space
def act_raw(x, perm, flip):
    """g on raw x [B, 7, 2, T, X, Y, Z]: axes permuted / flipped, encode pairs follow, flips conjugate c (dataset.py)"""
    xc = torch.complex(x[:, :, 0], x[:, :, 1])  # [B, 7, T, X, Y, Z]
    xc = xc.permute(0, 1, 2, *(3 + p for p in perm))
    xc = xc[:, [0] + [1 + 2 * perm[a] + u for a in range(3) for u in (0, 1)]]
    for a in range(3):
        if flip[a]:
            xc = xc.flip(3 + a)
            xc = torch.stack([flip_pair(xc[b], a) for b in range(xc.shape[0])])
    return torch.stack([xc.real, xc.imag], 2)


model = Denoiser7p(widths=(1, 2, 2), group="Oh", t_levels=(0, 1, 2), t_pad="circular").double().eval()
B, T, n = 1, 4, 17
x = torch.randn(B, 7, 2, T, n, n, n, dtype=torch.float64)
kv = torch.tensor([[0.0, np.pi, np.pi / 2.5] * 2 + [np.pi]], dtype=torch.float64)[:, :7]
kv = torch.tensor([[0.0, np.pi, np.pi / 2.5, np.pi, np.pi / 2.5, np.pi, np.pi / 2.5]], dtype=torch.float64)
h = torch.tensor([2e-3], dtype=torch.float64)
with torch.no_grad():
    ident = float((model(x, kv, h) - x).abs().max())
    print(f"3a. untrained model = identity: max |D(x) − x| {ident:.1e}")
    assert ident < 1e-12
    for prm in model.head.kern.params:  # random head for the equivariance tests
        prm.normal_(0, 0.1)
    for prm in model.cond[-1].parameters():
        prm.normal_(0, 0.1)
    y = model(x, kv, h)
    rel = lambda a, b: float((a - b).abs().max() / b.abs().max())
    worst = 0.0
    for perm, flip in [((1, 0, 2), (0, 0, 0)), ((0, 1, 2), (1, 0, 0)), ((2, 0, 1), (0, 1, 1)), ((0, 2, 1), (1, 1, 1))]:
        worst = max(worst, rel(model(act_raw(x, perm, flip), kv, h), act_raw(y, perm, flip)))
    print(f"3b. O_h equivariance in raw 7p space: max rel. error {worst:.1e}")
    assert worst < 1e-10
    mu = 3.7
    e = rel(model(mu * x, kv, h), mu * y)
    print(f"3c. positive homogeneity D(µx) = µD(x): rel. error {e:.1e}")
    assert e < 1e-12
    e = rel(model(torch.roll(x, 1, dims=3), kv, h), torch.roll(y, 1, dims=3))
    print(f"3d. cyclic time shift: rel. error {e:.1e}")
    assert e < 1e-12

    # 4. tiled = whole volume
    rs, rt = receptive_field(model)
    L = model.pad_multiple
    margin = -(-rs // L) * L
    print(f"4. receptive field radius: {rs} voxels, {rt} frames -> tile margin {margin}")
    xv = torch.randn(7, 2, T, 69, 49, 21, dtype=torch.float64)  # larger than a tile along x and y
    whole = denoise_volume(model, xv, kv[0], 2e-3)
    tiled = denoise_volume(model, xv, kv[0], 2e-3, tile=2 * margin + 1 + 2 * L, margin=margin)
    print(f"   tiles: {len(range(0, 69 - (2 * margin + 1 + 2 * L), 2 * L)) + 1} × {len(range(0, 49 - (2 * margin + 1 + 2 * L), 2 * L)) + 1} × 1")
    e = rel(tiled, whole)
    print(f"   tiled (tile {2 * margin + 1 + 2 * L}) vs whole volume {tuple(xv.shape[-3:])}: rel. error {e:.1e}")
    assert e < 1e-12

# 5. frozen + TorchScript trace, run at another shape
fm = Denoiser7p(widths=(1, 2, 2), group="Oh").float().eval()
for prm in fm.head.kern.params:
    prm.data.normal_(0, 0.1)
fm.freeze()
xs = torch.randn(1, 7, 2, 3, 9, 9, 9)
kvs, hs = kv.float(), h.float()
with torch.no_grad():
    ts = torch.jit.trace(fm, (xs, kvs, hs), check_trace=False)
    x2 = torch.randn(2, 7, 2, 5, 17, 13, 9)
    e = rel(ts(x2, kvs.repeat(2, 1), hs.repeat(2)), fm(x2, kvs.repeat(2, 1), hs.repeat(2)))
print(f"5. TorchScript trace (traced at 1×3×9³, run at 2×5×17×13×9): rel. error {e:.1e}")
assert e < 1e-5

if "--compile" in sys.argv:
    cm = torch.compile(fm)
    t0 = time.time()
    with torch.no_grad():
        e = rel(cm(x2, kvs.repeat(2, 1), hs.repeat(2)), fm(x2, kvs.repeat(2, 1), hs.repeat(2)))
    print(f"6. torch.compile (CPU inductor, {time.time() - t0:.0f} s incl. compile): rel. error {e:.1e}")
    assert e < 1e-4
print("ok")

# 6. velocity-space prior (|z|, p = |z| v): O_h equivariance on trivial ⊕ vector fields, homogeneity, identity at init
from model import DenoiserZP, velocity_from_zp, zp_from_velocity  # noqa: E402

torch.manual_seed(3)
mz = DenoiserZP(widths=(1, 2, 2), group="Oh").double()
with torch.no_grad():
    for prm in mz.head.kern.params:
        prm.normal_(0, 0.05)
    for prm in mz.cond[-1].parameters():
        prm.normal_(0, 0.1)
xz = torch.randn(2, 6, 4, 9, 9, 9, dtype=torch.float64)
hz = torch.tensor([2e-3, 1.6e-3], dtype=torch.float64)
with torch.no_grad():
    yz = mz(xz, hz)
    worst = 0.0
    for g in range(48):
        gx = apply_group(xz.flatten(0, 1), mz.tin, g, (2, 3, 4)).view_as(xz)
        gy = apply_group(yz.flatten(0, 1), mz.tin, g, (2, 3, 4)).view_as(yz)
        worst = max(worst, float((mz(gx, hz) - gy).abs().max() / yz.abs().max()))
    hom = float((mz(2.5 * xz, hz) - 2.5 * yz).abs().max() / yz.abs().max())
    m0 = DenoiserZP(widths=(1, 2, 2)).double()
    ident = float((m0(xz, hz) - xz).abs().max())
    m, v = velocity_from_zp(zp_from_velocity(xz[:, :, 0].abs() + 0.5, xz[:, :, 1:]))
    rt = float((v - xz[:, :, 1:]).abs().max())
print(f"6. velocity-space prior: O_h equivariance (trivial ⊕ vector) {worst:.1e}, homogeneity {hom:.1e}, identity at init {ident:.1e}, (|z|, v) round trip {rt:.1e}")
assert worst < 1e-10 and hom < 1e-10 and ident < 1e-12 and rt < 1e-12
print("ok (velocity-space prior)")
