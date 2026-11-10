// PC-MRA for the MIP. Small volumes arrive in one piece (exact scaling, float copy kept). Large ones stream in z-slabs
// that are max-pooled by an integer factor and quantised to u8 on arrival, so the browser never holds the float volume.
// The pooling is for DISPLAY only (max keeps thin vessels as bright as at full resolution); the vessel tool measures on
// full-resolution data (common/roi.js). Flags (URL): ?mip=auto|1|2|3|4…  (auto: smallest factor with ≤ ?mipvox= voxels).
import { autoFactor, poolSlab, pooledDims } from '../../common/pool.js';
import { percentile } from '../../common/flowmath.js';

/**
 * @returns {Promise<{u8: Uint8Array, dims: number[], factor: number, scale: number, hot: number[], hotValue: number, full: Float32Array|null, scaleFrom: string}>}
 *  dims: size of u8 (pooled); hot: voxel (full-resolution coords) of the brightest PC-MRA value
 */
export async function loadPcmra(client, pcId, dims, { factor = 'auto', maxVox = 2 ** 27, smallVox = 24e6, chunkBytes = 64 << 20, onProgress, signal } = {}) {
    const [X, Y, Z] = dims, nv = X * Y * Z;
    const k = factor === 'auto' ? autoFactor(dims, maxVox) : Math.max(1, Math.round(Number(factor)) || 1);
    const argmax = (a) => { let m = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[m]) m = i; return m; };
    const voxel = (i) => [i % X, Math.floor(i / X) % Y, Math.floor(i / (X * Y))];
    if (k === 1 && nv <= smallVox) {
        const t = await client.getTensor(pcId, { signal });
        const full = t.dtype === 'float32' ? t.data : Float32Array.from(t.data);
        const scale = percentile(full, 0.999) || 1, q = 255 / scale, u8 = new Uint8Array(nv);
        for (let i = 0; i < nv; i++) { const c = full[i] * q; u8[i] = c >= 255 ? 255 : c > 0 ? (c + 0.5) | 0 : 0; }
        const a = argmax(full);
        return { u8, dims, factor: 1, scale, hot: voxel(a), hotValue: full[a], full, scaleFrom: 'all voxels' };
    }
    // display scale (the value that maps to full brightness) from a strided subsample of at most ~4 M voxels
    const s = Math.max(1, Math.ceil(Math.cbrt(nv / 4e6)));
    const sub = await client.getTensor(pcId, { slice: s > 1 ? `[::${s}, ::${s}, ::${s}]` : undefined, signal });
    const scale = percentile(sub.dtype === 'float32' ? sub.data : Float32Array.from(sub.data), 0.999) || 1;
    const P = pooledDims(dims, k), u8 = new Uint8Array(P[0] * P[1] * P[2]);
    const per = Math.max(k, Math.floor(chunkBytes / (4 * X * Y) / k) * k);
    let hotValue = -Infinity, hot = [0, 0, 0];
    for (let z0 = 0; z0 < Z; z0 += per) {
        const z1 = Math.min(Z, z0 + per);
        const t = await client.getTensor(pcId, { slice: `[${z0}:${z1}]`, signal });
        const slab = t.dtype === 'float32' ? t.data : Float32Array.from(t.data);
        poolSlab(slab, dims, k, z0, z1 - z0, scale, u8);
        const a = argmax(slab);
        if (slab[a] > hotValue) { hotValue = slab[a]; const v = voxel(a); hot = [v[0], v[1], v[2] + z0]; }
        onProgress?.(z1, Z);
    }
    return { u8, dims: P, factor: k, scale, hot, hotValue, full: null, scaleFrom: s > 1 ? `a 1/${s}³ subsample` : 'all voxels' };
}
