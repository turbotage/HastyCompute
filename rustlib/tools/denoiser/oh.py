"""Octahedral-group equivariant layers in plain PyTorch (O_h: 48 signed axis permutations, or its rotation subgroup O).

Every feature field uses a *monomial* (signed-permutation) representation — trivial (scalar), `perm3` (axis permutation,
unsigned: Re of reference-relative encodes), `vector` (signed permutation: Im of reference-relative encodes, velocity),
`regular` (|G| channels permuted by left multiplication). For monomial representations the equivariance constraint of a
convolution kernel

    K(g p) = ρ_out(g) K(p) ρ_in(g)⁻¹        (p: stencil offset, conv out(x) = Σ_p K(p) in(x + p))

ties kernel entries (o, i, p) into orbits (o, i, p) → (π_out,g(o), π_in,g(i), g p) with sign s_out,g(o) s_in,g(i); an
orbit carries one free parameter, or none if some g fixes it with sign −1. A layer is therefore a signed gather from a
small parameter tensor followed by an ordinary conv3d / conv_transpose3d / einsum: no third-party library, works with
torch.compile, and `freeze()` turns the gathered weights into constants for TorchScript / torch.export.
Spatial action on a field: (g·f)(x) = ρ(g) f(G⁻¹ x), G[new, old] = ±1 (same convention as dataset.py).
"""
import itertools
from functools import lru_cache

import numpy as np
import _env  # noqa: F401  (caches inside the repo; before torch)
import torch
import torch.nn as nn
import torch.nn.functional as Fn

REP_DIMS = {"trivial": 1, "perm3": 3, "vector": 3}


@lru_cache(maxsize=None)
def group(name="Oh"):
    """(matrices [n, 3, 3] int, multiplication table [n, n]: mult[a, b] = index of G_a G_b); identity first"""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            G = np.zeros((3, 3), np.int64)
            for new, old in enumerate(perm):
                G[new, old] = signs[new]
            if name == "O" and round(np.linalg.det(G)) != 1:
                continue
            mats.append(G)
    mats = np.stack(mats)
    key = {m.tobytes(): i for i, m in enumerate(mats)}
    mult = np.array([[key[(a @ b).tobytes()] for b in mats] for a in mats])
    return mats, mult


@lru_cache(maxsize=None)
def rep(name, gname):
    """monomial representation: (perm [n, d], sign [n, d]) with ρ(g) e_i = sign[g, i] e_perm[g, i]"""
    mats, mult = group(gname)
    n = len(mats)
    if name == "trivial":
        return np.zeros((n, 1), np.int64), np.ones((n, 1), np.int64)
    if name in ("perm3", "vector"):
        perm = np.abs(mats).argmax(axis=1)  # row holding column i
        sign = np.take_along_axis(mats, perm[:, None, :], axis=1)[:, 0, :] if name == "vector" else np.ones((n, 3), np.int64)
        return perm, sign
    if name == "regular":
        return mult.copy(), np.ones((n, n), np.int64)
    raise ValueError(name)


def rep_dim(name, gname):
    return len(group(gname)[0]) if name == "regular" else REP_DIMS[name]


@lru_cache(maxsize=None)
def stencil(k, gname):
    """offsets [S, 3] of a k³ stencil (k = 1 or 3), C-order (matches conv3d weight [.., kx, ky, kz]); action [n, S]"""
    r = np.arange(k) - k // 2
    pos = np.stack(np.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    mats, _ = group(gname)
    key = {tuple(p): j for j, p in enumerate(pos)}
    act = np.array([[key[tuple(G @ p)] for p in pos] for G in mats])
    return pos, act


@lru_cache(maxsize=None)
def orbits(rep_out, rep_in, k, gname):
    """parameter id [d_out, d_in, S] (−1: forced zero) and sign [d_out, d_in, S] of an equivariant kernel block"""
    po, so = rep(rep_out, gname)
    pi, si = rep(rep_in, gname)
    _, sa = stencil(k, gname)
    do, di, S = po.shape[1], pi.shape[1], sa.shape[1]
    o, i, j = np.meshgrid(np.arange(do), np.arange(di), np.arange(S), indexing="ij")
    o, i, j = o.ravel(), i.ravel(), j.ravel()
    n = np.arange(do * di * S)
    tgt = (po[:, o] * di + pi[:, i]) * S + sa[:, j]  # [g, N]
    sg = so[:, o] * si[:, i]
    g_star = tgt.argmin(axis=0)
    root = tgt[g_star, n]
    sign = sg[g_star, n]  # K[root] = sign K[n]  =>  K[n] = sign K[root]
    is_root = root == n
    bad = np.zeros(len(n), bool)
    bad[is_root] = ((tgt[:, is_root] == n[is_root]) & (sg[:, is_root] == -1)).any(axis=0)
    roots = np.flatnonzero(is_root & ~bad)
    pid = np.full(len(n), -1)
    pid[roots] = np.arange(len(roots))
    ids = pid[root]
    sign = np.where(ids >= 0, sign, 0)
    return ids.reshape(do, di, S), sign.reshape(do, di, S), len(roots)


class FieldType:
    """direct sum of fields: list of (rep name, multiplicity); channels ordered group by group, copy by copy"""

    def __init__(self, gname, fields):
        self.gname = gname
        self.fields = [(r, int(m)) for r, m in fields if m > 0]
        self.dim = sum(rep_dim(r, gname) * m for r, m in self.fields)

    def __add__(self, other):
        assert self.gname == other.gname
        return FieldType(self.gname, self.fields + other.fields)

    def regular_mult(self):
        return sum(m for r, m in self.fields if r == "regular")

    def channel_perm(self, g):
        """(perm [dim], sign [dim]) of the action of group element g on the channels"""
        perm, sign, off = [], [], 0
        for r, m in self.fields:
            p, s = rep(r, self.gname)
            d = p.shape[1]
            for c in range(m):
                perm.append(off + c * d + p[g])
                sign.append(s[g])
            off += m * d
        return np.concatenate(perm), np.concatenate(sign)


def regular(gname, m):
    return FieldType(gname, [("regular", m)])


class OhKernel(nn.Module):
    """equivariant kernel between field types: weight() -> [D_out, D_in, taps, S]"""

    def __init__(self, t_in, t_out, k=3, taps=1, init="he"):
        super().__init__()
        self.t_in, self.t_out, self.k, self.taps = t_in, t_out, k, taps
        self.params = nn.ParameterList()
        self.blocks = []  # (out offset, in offset, m_o, m_i, d_o, d_i, param index)
        fan_in = t_in.dim * k ** 3 * taps
        std = {"he": np.sqrt(2.0 / fan_in), "lin": np.sqrt(1.0 / fan_in), "zero": 0.0}[init]
        oo = 0
        for ro, mo in t_out.fields:
            do = rep_dim(ro, t_out.gname)
            oi = 0
            for ri, mi in t_in.fields:
                di = rep_dim(ri, t_in.gname)
                ids, sgn, n = orbits(ro, ri, k, t_in.gname)
                if n:
                    pidx = len(self.params)
                    self.params.append(nn.Parameter(torch.randn(mo, mi, n, taps) * std))
                    self.register_buffer(f"ids{pidx}", torch.from_numpy(np.maximum(ids, 0)).long(), persistent=False)
                    self.register_buffer(f"sgn{pidx}", torch.from_numpy(sgn).float(), persistent=False)
                    self.blocks.append((oo, oi, mo, mi, do, di, pidx))
                oi += mi * di
            oo += mo * do
        self.register_buffer("frozen", torch.zeros(0), persistent=True)

    def weight(self):
        if self.frozen.numel():
            return self.frozen
        dev = self.params[0].device if len(self.params) else self.frozen.device
        W = torch.zeros(self.t_out.dim, self.t_in.dim, self.taps, self.k ** 3, device=dev, dtype=self.params[0].dtype if len(self.params) else torch.float32)
        for oo, oi, mo, mi, do, di, p in self.blocks:
            th = self.params[p]  # [mo, mi, n, taps]
            blk = th[:, :, getattr(self, f"ids{p}")] * getattr(self, f"sgn{p}")[None, None, :, :, :, None]  # [mo, mi, do, di, S, taps]
            blk = blk.permute(0, 2, 1, 3, 5, 4).reshape(mo * do, mi * di, self.taps, self.k ** 3)
            W[oo:oo + mo * do, oi:oi + mi * di] = blk
        return W

    def freeze(self):
        with torch.no_grad():
            self.frozen = self.weight().detach().clone()


class OhConv3d(nn.Module):
    """equivariant 3-D convolution on [N, C, X, Y, Z]; stride 2 / transpose keep exact flip equivariance on odd sizes"""

    def __init__(self, t_in, t_out, k=3, stride=1, transpose=False, init="he"):
        super().__init__()
        self.k, self.stride, self.transpose = k, stride, transpose
        self.kern = OhKernel(t_out, t_in, k, init=init) if transpose else OhKernel(t_in, t_out, k, init=init)
        self.d_in, self.d_out = t_in.dim, t_out.dim

    def forward(self, x):
        W = self.kern.weight()[:, :, 0]
        k = self.k
        if self.transpose:
            return Fn.conv_transpose3d(x, W.reshape(self.d_in, self.d_out, k, k, k), stride=self.stride, padding=k // 2)
        return Fn.conv3d(x, W.reshape(self.d_out, self.d_in, k, k, k), stride=self.stride, padding=k // 2)


class OhTemporal(nn.Module):
    """pointwise-in-space equivariant channel mixing with `taps` cardiac-time taps on [B, T, C, X, Y, Z]
    (circular in t for full cycles, zero padding for training windows)"""

    def __init__(self, t_in, t_out, taps=3, pad="circular", init="he"):
        super().__init__()
        self.kern = OhKernel(t_in, t_out, k=1, taps=taps, init=init)
        self.taps, self.pad = taps, pad

    def forward(self, x):
        W = self.kern.weight()[..., 0]  # [D_out, D_in, taps]
        c = self.taps // 2
        y = None
        for tau in range(self.taps):
            s = c - tau  # y[t] += W_tau x[t + tau − c]
            if s == 0:
                xs = x
            elif self.pad == "circular":
                xs = torch.roll(x, shifts=s, dims=1)
            else:
                z = torch.zeros_like(x[:, :abs(s)])
                xs = torch.cat([z, x[:, :-s]], 1) if s > 0 else torch.cat([x[:, -s:], z], 1)
            term = torch.einsum("oi,btixyz->btoxyz", W[:, :, tau], xs)
            y = term if y is None else y + term
        return y


def apply_group(x, ftype, g, spatial_dims):
    """act with element g on a field tensor: channels at dim `ftype_dim`=spatial_dims[0]−1, spatial dims as given"""
    mats, _ = group(ftype.gname)
    G = mats[g]
    perm, sign = ftype.channel_perm(g)
    cdim = spatial_dims[0] - 1
    out = torch.zeros_like(x)
    idx = [slice(None)] * x.dim()
    src = x * torch.tensor(sign, dtype=x.dtype, device=x.device).view(*([1] * cdim), -1, *([1] * (x.dim() - cdim - 1)))
    idx[cdim] = torch.from_numpy(perm).to(x.device)
    out[tuple(idx)] = src
    # spatial: (g·f)(x) = f(G⁻¹ x): new axis a takes old axis b with G[a, b] = ±1, flipped if −1
    old_of_new = np.abs(G).argmax(axis=1)
    out = out.permute(*range(spatial_dims[0]), *[spatial_dims[0] + old_of_new[a] for a in range(3)])
    for a in range(3):
        if G[a, old_of_new[a]] < 0:
            out = out.flip(spatial_dims[0] + a)
    return out
