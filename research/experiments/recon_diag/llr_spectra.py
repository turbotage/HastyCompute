"""Block singular-value spectra of reconstructions as the LLR prior sees them (8³ blocks, Casorati B³ × T, T = frames ×
encodes), against σ̂ (the operator's threshold scale: median over blocks of each block's s[T/4]).

    llr_spectra.py <run dir> <recon npy> [<recon npy> ...]

Per recon: σ̂, the spread of the per-block s[T/4] (how uniform the 'noise' scale is), the number of singular values
above κσ̂ for κ = 1, 3, 10, 30, and where the high-rank blocks are (mean magnitude, inside the lumen). For two recons,
also the spectrum of their difference per block (what the iterations added: low-rank dynamics or high-rank content).
Unaligned 8³ blocks (no shift), blocks with any support voxel (mask_support.npy of the run).
"""
import os
import sys

import numpy as np

run = sys.argv[1]
names = sys.argv[2:]
sup = np.load(os.path.join(run, "mask_support.npy")) > 0.5 if os.path.exists(os.path.join(run, "mask_support.npy")) else None
B = 8


def blocks(shape):
    nx, ny, nz = shape
    for z in range(0, nz, B):
        for y in range(0, ny, B):
            for x in range(0, nx, B):
                yield (slice(x, min(x + B, nx)), slice(y, min(y + B, ny)), slice(z, min(z + B, nz)))


def spectra(x):
    """x: (nx, ny, nz, ne, F) -> list of (block, singular values descending, mean |x|)"""
    nx, ny, nz, ne, nf = x.shape
    out = []
    for bl in blocks((nx, ny, nz)):
        if sup is not None and not sup[bl].any():
            continue
        a = np.asarray(x[bl]).reshape(-1, ne * nf)  # rows: voxels; columns: (encode, frame) -> T
        if a.shape[0] < 2:
            continue
        s = np.linalg.svd(a, compute_uv=False)
        out.append((bl, s, float(np.abs(a).mean())))
    return out


def report(name, sp):
    T = len(sp[0][1])
    q = np.array([np.sort(s)[len(s) // 4] for _, s, _ in sp])
    sig = float(np.sort(q)[len(q) // 2])
    m = np.array([mm for _, _, mm in sp])
    print(f"== {name}: {len(sp)} blocks, T = {T}; sigma_hat {sig:.3e}; per-block s[T/4] / sigma_hat: p10 {np.percentile(q / sig, 10):.2f}, p50 1, p90 {np.percentile(q / sig, 90):.2f}, max {q.max() / sig:.2f}")
    for kap in (1, 3, 10, 30):
        k = np.array([(s > kap * sig).sum() for _, s, _ in sp])
        e_kept = np.array([np.sum(np.maximum(s - kap * sig, 0) ** 2) / np.sum(s ** 2) for _, s, _ in sp])
        print(f"  kappa {kap:>2}: components above threshold per block: median {np.median(k):.0f}, p90 {np.percentile(k, 90):.0f}, max {k.max()}; energy kept (after shrinkage) median {np.median(e_kept):.3f}")
    # high-rank blocks vs brightness
    k10 = np.array([(s > 10 * sig).sum() for _, s, _ in sp])
    order = np.argsort(m)
    third = len(m) // 3
    for nm, idx in [("dim third", order[:third]), ("middle third", order[third:2 * third]), ("bright third", order[2 * third:])]:
        print(f"  {nm}: mean |x| {m[idx].mean():.3e}, s[T/4]/sigma_hat median {np.median(q[idx] / sig):.2f}, components above 10 sigma_hat median {np.median(k10[idx]):.0f}")
    return sig


xs = [np.load(n, mmap_mode="r") for n in names]
sps = [spectra(x) for x in xs]
sigs = [report(os.path.basename(n), sp) for n, sp in zip(names, sps)]
if len(xs) == 2:
    d = np.asarray(xs[1]) - np.asarray(xs[0])
    spd = spectra(d)
    sig = sigs[1]
    fr = []
    for (_, s, _), (_, s1, _) in zip(spd, sps[1]):
        e = np.cumsum(s ** 2) / np.sum(s ** 2)
        fr.append((int(np.searchsorted(e, 0.9)) + 1, float(np.sqrt(np.sum(s ** 2) / np.sum(s1 ** 2)))))
    r90 = np.array([a for a, _ in fr])
    rel = np.array([b for _, b in fr])
    print(f"== difference {os.path.basename(names[1])} - {os.path.basename(names[0])}: |d| / |x| per block median {np.median(rel):.3f}; rank holding 90 % of the difference energy: median {np.median(r90):.0f}, p90 {np.percentile(r90, 90):.0f} (of T = {len(spd[0][1])})")
    s_all = np.concatenate([s for _, s, _ in spd])
    print(f"  difference singular values / sigma_hat of the later recon: p50 {np.median(s_all) / sig:.2f}, p90 {np.percentile(s_all, 90) / sig:.2f}, max {s_all.max() / sig:.2f}")
