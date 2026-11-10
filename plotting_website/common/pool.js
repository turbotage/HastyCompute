// Display-only reduction of the PC-MRA volume for the MIP: max-pooling by an integer factor, quantised to u8. Max (not
// mean / stride) so a thin bright vessel stays as bright as in the full-resolution volume; measurements never use this
// volume (they read full-resolution data around the click, see roi.js).

/** pooled size per axis: ceil(n / k) */
export const pooledDims = (dims, k) => dims.map((n) => Math.ceil(n / k));

/** smallest integer factor k ≥ 1 whose pooled volume has at most `maxVox` voxels */
export function autoFactor(dims, maxVox) {
    for (let k = 1; k < 64; k++) { const p = pooledDims(dims, k); if (p[0] * p[1] * p[2] <= maxVox) return k; }
    return 64;
}

/**
 * Max-pool a slab of the volume and quantise it to u8 (255 / scale per unit, clipped).
 * slab: Float32Array [nz, Y, X] holding volume planes z0 … z0 + nz − 1, z0 a multiple of k.
 * out: Uint8Array of the whole pooled volume [ceil(Z/k), ceil(Y/k), ceil(X/k)]; this fills the pooled planes the slab covers.
 */
export function poolSlab(slab, dims, k, z0, nz, scale, out) {
    const [X, Y] = dims, [PX, PY] = pooledDims(dims, k), q = 255 / scale;
    if (z0 % k) throw new Error('slab must start on a pooling boundary');
    const pz0 = z0 / k, npz = Math.ceil(nz / k);
    for (let pz = 0; pz < npz; pz++) {
        const za = pz * k, zb = Math.min(nz, za + k);
        for (let py = 0; py < PY; py++) {
            const ya = py * k, yb = Math.min(Y, ya + k);
            for (let px = 0; px < PX; px++) {
                const xa = px * k, xb = Math.min(X, xa + k);
                let m = 0;
                for (let z = za; z < zb; z++) for (let y = ya; y < yb; y++) {
                    const row = (z * Y + y) * X;
                    for (let x = xa; x < xb; x++) { const v = slab[row + x]; if (v > m) m = v; }
                }
                const c = m * q;
                out[((pz0 + pz) * PY + py) * PX + px] = c >= 255 ? 255 : c > 0 ? (c + 0.5) | 0 : 0;
            }
        }
    }
}

/** map between full-resolution voxel coordinates and pooled ones (voxel centres: full i ↔ pooled (i + ½) / k − ½) */
export const toPooled = (p, k) => p.map((c) => (c + 0.5) / k - 0.5);
export const fromPooled = (p, k) => p.map((c) => (c + 0.5) * k - 0.5);
