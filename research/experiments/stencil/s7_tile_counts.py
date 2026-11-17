"""Points multiplied per sample (per slot, per channel) for tile layouts, Monte Carlo over uniform sample offsets.
dense   : every point of every touched TT^3 tile  (current knot kernels' forward; formula (W-1+TT)^3)
warp222 : 2x2x2 point warps skipped when outside the support box (variant-1 adjoint)
box     : exact support box per (sample, tile) = W^3 (the floor for compact gridding)
stages  : (sample, tile) pairs = how often a sample is staged (weights, a_l, sincos) in the tile-major layout"""
import numpy as np
rng = np.random.default_rng(0)
def axis(W, TT, n=200000):
    sc = rng.uniform(0, TT, n) + 100
    i0 = np.ceil(sc - W / 2).astype(int); i1 = i0 + W - 1            # covered cells
    t0, t1 = i0 // TT, i1 // TT
    tiles = t1 - t0 + 1
    pts = []
    blk2 = []
    for t in range(tiles.max()):
        tt = t0 + t
        lo = np.maximum(i0, tt * TT) - tt * TT; hi = np.minimum(i1, tt * TT + TT - 1) - tt * TT
        valid = (t < tiles)
        e = np.where(valid, hi - lo + 1, 0)
        b = np.where(valid, hi // 2 - lo // 2 + 1, 0)
        pts.append(e); blk2.append(b)
    return tiles, np.array(pts), np.array(blk2)
print("| W | TT | stages/sample | dense pts | warp222 pts | box pts (=W^3) | useful fraction dense / warp222 |")
print("|---|---|---|---|---|---|---|")
for TT in [4, 8]:
    for W in [5, 6, 7, 8, 10]:
        tl, e, b = axis(W, TT)
        st = tl.mean() ** 3
        dense = (tl * TT).mean() ** 3
        w222 = (2 * b.sum(0)).mean() ** 3      # per axis: 2 points per 2-block, summed over tiles (axes independent)
        print(f"| {W} | {TT} | {st:.1f} | {dense:.0f} | {w222:.0f} | {W**3} | {W**3/dense:.2f} / {W**3/w222:.2f} |")
