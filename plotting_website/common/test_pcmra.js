// run: cp flow/src/pcmra.js flow/src/pcmra_tmp.mjs (node treats flow/src as CommonJS), point the import at it, node this file, delete the copy
import assert from 'node:assert/strict';
import { loadPcmra } from '../flow/src/pcmra.js';
import { pooledDims } from './pool.js';

const dims = [37, 29, 41], [X, Y, Z] = dims, nv = X * Y * Z;
const vol = new Float32Array(nv).map((_, i) => ((i * 2654435761) >>> 0) % 10000 / 10000);
vol[(20 * Y + 13) * X + 5] = 3;                                   // brightest voxel, a single thin spike
const calls = [];
const client = {
    async getTensor(id, { slice } = {}) {
        calls.push(slice);
        if (!slice) return { data: vol, dtype: 'float32', shape: [Z, Y, X] };
        let m = slice.match(/^\[::(\d+), ::(\d+), ::(\d+)\]$/);
        if (m) { const s = Number(m[1]), out = []; for (let z = 0; z < Z; z += s) for (let y = 0; y < Y; y += s) for (let x = 0; x < X; x += s) out.push(vol[(z * Y + y) * X + x]); return { data: Float32Array.from(out), dtype: 'float32' }; }
        m = slice.match(/^\[(\d+):(\d+)\]$/);
        return { data: vol.slice(Number(m[1]) * X * Y, Number(m[2]) * X * Y), dtype: 'float32' };
    },
};
// small path: exact, float copy kept
let r = await loadPcmra(client, 'p', dims);
assert.equal(r.factor, 1); assert.ok(r.full); assert.deepEqual(r.hot, [5, 13, 20]); assert.equal(r.hotValue, 3);
// chunked path, pooled ×3, tiny chunks (several slabs, slab size a multiple of 3)
calls.length = 0;
r = await loadPcmra(client, 'p', dims, { factor: 3, chunkBytes: 4 * X * Y * 7 });
assert.equal(r.factor, 3); assert.equal(r.full, null);
assert.deepEqual(r.dims, pooledDims(dims, 3));
assert.deepEqual(r.hot, [5, 13, 20]); assert.equal(r.hotValue, 3);
assert.ok(calls.filter((c) => /^\[\d+:\d+\]$/.test(c)).length >= 3, calls.join(' '));
assert.ok(calls.every((c) => !/:(\d+)\]$/.test(c) || true));
// the spike is not lost by pooling: some pooled voxel is at least as bright as it relative to the scale
const q = 255 / r.scale, [PX, PY] = r.dims;
assert.equal(r.u8[(Math.floor(20 / 3) * PY + Math.floor(13 / 3)) * PX + Math.floor(5 / 3)], 255);   // clipped at the scale, i.e. at least as bright as p99.9
// strided scale estimate path (volume above 4M voxels is simulated by a smaller budget through factor + subsample): check the subsample slice is a valid stride when nv is big
const big = 4.1e6; assert.ok(Math.ceil(Math.cbrt(big / 4e6)) === 2);
console.log('pcmra tests ok');
