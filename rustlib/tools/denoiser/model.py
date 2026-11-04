"""O_h-equivariant, positively homogeneous complex 7-point 4D-flow denoiser (research/denoiser_design.md §10).

Input / output (real, export-friendly): x [B, 7, 2, T, X, Y, Z] = (re, im) of [ref, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi],
x_e ∝ m e^{iθ} exp(+i kv_e v_axis(e)), encodes along the grid axes; kv [B, 7] = π/VENC_e; h [B] voxel size (m).

  lift      f = (Re x_ref, Im x_ref | Re c_lo, Re c_hi | Im c_lo, Im c_hi),  c_e = x_e · conj(x_ref)/|x_ref|
            field type 2·trivial ⊕ 2·perm3 ⊕ 2·vector (14 channels): O_h acts linearly — axis permutations permute,
            an axis flip negates Im c along that axis (= conj of c_e, the action of v_a → −v_a)
  network   f' = f + N(f), N a bias-free U-Net of regular O_h fields with ReLU: exactly equivariant to translations
            (by multiples of the coarsest stride), O_h, positive scaling (degree-1 homogeneous) and, with circular
            temporal convs, cyclic time shifts. Conditioning on (VENC_lo, VENC_hi, h) is multiplicative per regular
            field (FiLM γ, degree 0), so homogeneity is kept.
  unlift    x'_ref = f'_ref, x'_e = c'_e · x'_ref/|x'_ref|   (velocity = arg c'_e, independent of the background phase)
The head is zero-initialised: an untrained model is the identity.
Spatial sizes must be ≡ 1 mod 2^(levels−1) for exact flip equivariance (`pad_multiple`); `denoise_volume` pads.
"""
import math

import numpy as np
import _env  # noqa: F401  (caches inside the repo; before torch)
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from oh import FieldType, OhConv3d, OhTemporal, regular

LO, HI = [1, 3, 5], [2, 4, 6]  # encode slots of the x, y, z axes at low / high VENC


def in_type(gname):
    return FieldType(gname, [("trivial", 2), ("perm3", 2), ("vector", 2)])


def inv_abs(re, im):
    """1/|z| (0 at z = 0) with bounded gradients: |z|² is floored at 1e-12 × its mean over each sample (degree 2, so
    scale-homogeneity and O_h invariance stay exact; only |z| < 1e-6 · RMS is affected). Without the floor the
    derivative ~|z|⁻³ overflows near (but not at) zero and 0 · ∞ turns masked-out gradients into NaN."""
    a2 = re ** 2 + im ** 2
    floor = 1e-12 * a2.flatten(1).mean(1).view(-1, *[1] * (a2.dim() - 1))
    nz = a2 > 0
    return torch.where(nz, torch.rsqrt(torch.maximum(torch.where(nz, a2, torch.ones_like(a2)), floor)), torch.zeros_like(a2))


def lift(x):
    """x [B, 7, 2, T, X, Y, Z] -> f [B, T, 14, X, Y, Z]"""
    re, im = x[:, :, 0], x[:, :, 1]  # [B, 7, T, X, Y, Z]
    inv = inv_abs(re[:, 0], im[:, 0])
    ur, ui = re[:, 0] * inv, im[:, 0] * inv  # u = x_ref/|x_ref|
    cr = re[:, 1:] * ur[:, None] + im[:, 1:] * ui[:, None]  # c = x_e conj(u)
    ci = im[:, 1:] * ur[:, None] - re[:, 1:] * ui[:, None]
    lo, hi = [e - 1 for e in LO], [e - 1 for e in HI]
    f = torch.cat([re[:, :1], im[:, :1], cr[:, lo], cr[:, hi], ci[:, lo], ci[:, hi]], 1)
    return f.transpose(1, 2)


def unlift(f):
    """f [B, T, 14, X, Y, Z] -> x [B, 7, 2, T, X, Y, Z]"""
    f = f.transpose(1, 2)  # [B, 14, T, X, Y, Z]
    rr, ri = f[:, 0], f[:, 1]
    inv = inv_abs(rr, ri)
    ur, ui = rr * inv, ri * inv
    cr = torch.stack([f[:, 2], f[:, 5], f[:, 3], f[:, 6], f[:, 4], f[:, 7]], 1)  # slots 1..6: x_lo, x_hi, y_lo, ...
    ci = torch.stack([f[:, 8], f[:, 11], f[:, 9], f[:, 12], f[:, 10], f[:, 13]], 1)
    er = cr * ur[:, None] - ci * ui[:, None]  # x_e = c u
    ei = cr * ui[:, None] + ci * ur[:, None]
    re = torch.cat([rr[:, None], er], 1)
    im = torch.cat([ri[:, None], ei], 1)
    return torch.stack([re, im], 2)


def spatial(m, x):
    """apply a [N, C, X, Y, Z] module to [B, T, C, X, Y, Z]"""
    y = m(x.flatten(0, 1))
    return y.view(x.shape[0], x.shape[1], y.shape[1], y.shape[2], y.shape[3], y.shape[4])


class Block(nn.Module):
    """x + conv(ReLU(t-mix(ReLU(γ · conv(x)))))  on regular fields; bias-free"""

    def __init__(self, ft, temporal, t_pad):
        super().__init__()
        self.c1 = OhConv3d(ft, ft, 3)
        self.t = OhTemporal(ft, ft, 3, pad=t_pad) if temporal else None
        self.c2 = OhConv3d(ft, ft, 3, init="lin")
        self.d = ft.dim // ft.regular_mult()

    def forward(self, x, gamma):
        h = torch.relu(spatial(self.c1, x)) * gamma.repeat_interleave(self.d, 1)[:, None, :, None, None, None]
        if self.t is not None:
            h = torch.relu(self.t(h))
        return x + spatial(self.c2, h) * 0.5


class Denoiser7p(nn.Module):
    def __init__(self, widths=(2, 4, 8), group="Oh", t_levels=(0, 1, 2), t_pad="circular", blocks=1):
        super().__init__()
        self.cfg = dict(widths=list(widths), group=group, t_levels=list(t_levels), t_pad=t_pad, blocks=blocks)
        self.levels = len(widths)
        tin = in_type(group)
        R = [regular(group, w) for w in widths]
        self.stem = OhConv3d(tin, R[0], 3)
        self.enc = nn.ModuleList([nn.ModuleList([Block(R[l], l in t_levels, t_pad) for _ in range(blocks)]) for l in range(self.levels - 1)])
        self.down = nn.ModuleList([OhConv3d(R[l], R[l + 1], 3, stride=2) for l in range(self.levels - 1)])
        self.mid = nn.ModuleList([Block(R[-1], self.levels - 1 in t_levels, t_pad) for _ in range(blocks)])
        self.up = nn.ModuleList([OhConv3d(R[l + 1], R[l], 3, stride=2, transpose=True) for l in range(self.levels - 1)])
        self.merge = nn.ModuleList([OhConv3d(R[l] + R[l], R[l], 1) for l in range(self.levels - 1)])
        self.dec = nn.ModuleList([nn.ModuleList([Block(R[l], False, t_pad) for _ in range(blocks)]) for l in range(self.levels - 1)])
        self.head = OhConv3d(R[0], tin, 3, init="zero")
        n_gamma = sum(widths[l] * blocks for l in range(self.levels - 1)) * 2 + widths[-1] * blocks
        self.cond = nn.Sequential(nn.Linear(3, 32), nn.SiLU(), nn.Linear(32, n_gamma))
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)
        self.pad_multiple = 2 ** (self.levels - 1)

    def gammas(self, kv, h):
        z = torch.stack([torch.log(math.pi / kv[:, LO[0]]), torch.log(math.pi / kv[:, HI[0]]), torch.log(h / 2e-3)], 1)
        g = torch.exp(self.cond(z))  # [B, n_gamma], 1 at initialisation
        return g

    def core(self, f, g):
        """f [B, T, 14, X, Y, Z] -> Δf"""
        x = torch.relu(spatial(self.stem, f))
        gi = 0
        skips = []
        for l in range(self.levels - 1):
            for b in self.enc[l]:
                w = self.cfg["widths"][l]
                x = b(x, g[:, gi:gi + w])
                gi += w
            skips.append(x)
            x = torch.relu(spatial(self.down[l], x))
        for b in self.mid:
            w = self.cfg["widths"][-1]
            x = b(x, g[:, gi:gi + w])
            gi += w
        for l in reversed(range(self.levels - 1)):
            x = torch.relu(spatial(self.up[l], x))
            x = torch.relu(spatial(self.merge[l], torch.cat([x, skips[l]], 2)))
            for b in self.dec[l]:
                w = self.cfg["widths"][l]
                x = b(x, g[:, gi:gi + w])
                gi += w
        return spatial(self.head, x)

    def forward(self, x, kv, h):
        f = lift(x)
        return unlift(f + self.core(f, self.gammas(kv, h)))

    def freeze(self):
        for m in self.modules():
            if hasattr(m, "kern"):
                m.kern.freeze()
        return self


def receptive_field(model, n=None, T=None):
    """structural receptive-field radius (space, time) of `core`: every kernel replaced by its non-zero pattern
    (all parameters 1, absolute value), identity activations, impulse response of the resulting positive network"""
    import copy

    m = copy.deepcopy(model).float().cpu().eval()
    with torch.no_grad():
        for mod in m.modules():
            if hasattr(mod, "kern"):
                for prm in mod.kern.params:
                    prm.fill_(1.0)
                mod.kern.frozen = torch.zeros(0)
                mod.kern.frozen = mod.kern.weight().abs()
    relu = torch.relu
    torch.relu = lambda v: v
    n = n or 8 * m.pad_multiple + 1
    T = T or 15
    try:
        while True:  # grow the probe until the response stays clear of its borders
            f = torch.zeros(1, T, 14, n, n, n)
            c = n // 2
            f[0, T // 2, :, c, c, c] = 1.0
            with torch.no_grad():
                out = m.core(f, torch.ones(1, 10_000)).abs().sum(dim=(0, 2))  # [T, n, n, n]
            nz = torch.nonzero(out > 0)
            rs, rt = int((nz[:, 1:] - c).abs().max()), int((nz[:, 0] - T // 2).abs().max())
            if rs < c and rt < T // 2:
                return rs, rt
            n, T = (n - 1) * 2 + 1 if rs >= c else n, 2 * T + 1 if rt >= T // 2 else T
    finally:
        torch.relu = relu


def pad_to(x, mult):
    """zero-pad the last three dims of x to sizes ≡ 1 mod mult; returns padded tensor and original sizes"""
    sz = x.shape[-3:]
    p = [(-(s - 1)) % mult for s in sz]
    return Fn.pad(x, (0, p[2], 0, p[1], 0, p[0])), sz


@torch.no_grad()
def denoise_volume(model, x, kv, h, tile=None, margin=None):
    """x [7, 2, T, X, Y, Z] (one volume, full cardiac cycle). tile=None: whole volume at once; else overlap-tile with
    tiles of `tile` voxels (≡ 1 mod pad_multiple) and `margin` ≥ receptive-field radius (multiple of pad_multiple),
    keeping only tile centres — identical to the whole-volume result, no blending."""
    L = model.pad_multiple
    xp, sz = pad_to(x[None], L)
    kv, h = kv[None], torch.as_tensor([h], dtype=x.dtype, device=x.device)
    if tile is None:
        y = model(xp, kv, h)
    else:
        assert (tile - 1) % L == 0 and margin % L == 0 and tile > 2 * margin
        N = xp.shape[-3:]
        y = torch.zeros_like(xp)
        step = tile - 1 - 2 * margin  # multiple of pad_multiple: tiles sit on the coarsest lattice
        starts = []
        for a in range(3):
            if N[a] <= tile:
                starts.append([0])
                continue
            s = list(range(0, N[a] - tile, step)) + [N[a] - tile]
            starts.append(sorted(set(s)))
        for s0 in starts[0]:
            for s1 in starts[1]:
                for s2 in starts[2]:
                    s = (s0, s1, s2)
                    t = min(tile, N[0]), min(tile, N[1]), min(tile, N[2])
                    out = model(xp[..., s0:s0 + t[0], s1:s1 + t[1], s2:s2 + t[2]], kv, h)
                    keep = []
                    for a in range(3):
                        lo = 0 if s[a] == 0 else s[a] + margin
                        hi = N[a] if s[a] + t[a] >= N[a] else s[a] + t[a] - margin
                        keep.append((lo, hi))
                    y[..., keep[0][0]:keep[0][1], keep[1][0]:keep[1][1], keep[2][0]:keep[2][1]] = \
                        out[..., keep[0][0] - s0:keep[0][1] - s0, keep[1][0] - s1:keep[1][1] - s1, keep[2][0] - s2:keep[2][1] - s2]
    return y[0, ..., :sz[0], :sz[1], :sz[2]]


class DenoiserZP(Denoiser7p):
    """Prior for the velocity-parametrised reconstruction (research/velocity_model_sr.md §A5): per cardiac node a
    scalar |z| (magnitude) and the vector p = |z|·v (magnitude-weighted velocity, logical axes = spatial axes x, y, z).
    Same bias-free O_h regular-field U-Net core as `Denoiser7p`; input/output field type trivial ⊕ vector, residual,
    zero-initialised head (identity at init). D(µ·) = µD(·) exactly; v = p / |z| is then invariant to intensity
    scaling. Conditioning: voxel size h only (velocities are physical, no VENC).
        x [B, T, 4, X, Y, Z] = [|z|, p_x, p_y, p_z] per node, h [B] (m) -> same layout"""

    def __init__(self, widths=(2, 4, 8), group="Oh", t_levels=(0, 1, 2), t_pad="circular", blocks=1):
        super().__init__(widths, group, t_levels, t_pad, blocks)
        self.cfg["kind"] = "zp"
        tin = FieldType(group, [("trivial", 1), ("vector", 1)])
        R0 = regular(group, widths[0])
        self.stem = OhConv3d(tin, R0, 3)
        self.head = OhConv3d(R0, tin, 3, init="zero")
        n_gamma = self.cond[-1].out_features
        self.cond = nn.Sequential(nn.Linear(1, 32), nn.SiLU(), nn.Linear(32, n_gamma))
        nn.init.zeros_(self.cond[-1].weight)
        nn.init.zeros_(self.cond[-1].bias)
        self.tin = tin

    def gammas_h(self, h):
        return torch.exp(self.cond(torch.log(h / 2e-3).view(-1, 1)))

    def forward(self, x, h):
        return x + self.core(x, self.gammas_h(h))


def zp_from_velocity(mag, v):
    """mag [..., X, Y, Z], v [..., 3, X, Y, Z] -> [..., 4, X, Y, Z] = [|z|, |z| v]"""
    m = mag.abs()
    return torch.cat([m.unsqueeze(-4), m.unsqueeze(-4) * v], dim=-4)


def velocity_from_zp(x, eps=1e-3):
    """[..., 4, X, Y, Z] -> (|z|, v) with v = p / |z| (0 where |z| < eps · max)"""
    m = x[..., 0, :, :, :]
    ok = m > eps * m.amax()
    v = torch.where(ok.unsqueeze(-4), x[..., 1:, :, :, :] / torch.where(ok, m, torch.ones_like(m)).unsqueeze(-4), torch.zeros_like(x[..., 1:, :, :, :]))
    return m, v
