import { autoFactor, fromPooled, poolSlab, pooledDims, toPooled } from './pool.js';
import assert from 'node:assert/strict';

const dims = [7, 5, 6], k = 2, [X, Y, Z] = dims, P = pooledDims(dims, k);
assert.deepEqual(P, [4, 3, 3]);
const vol = new Float32Array(X * Y * Z).map((_, i) => ((i * 2654435761) >>> 0) % 1000 / 1000);
const full = new Uint8Array(P[0] * P[1] * P[2]), parts = new Uint8Array(full.length);
poolSlab(vol, dims, k, 0, Z, 1, full);
// slab-wise (2 + 4 planes) must equal the one-shot result
poolSlab(vol.subarray(0, 2 * X * Y), dims, k, 0, 2, 1, parts);
poolSlab(vol.subarray(2 * X * Y), dims, k, 2, 4, 1, parts);
assert.deepEqual(parts, full);
// brute force max of the covered block
for (let pz = 0; pz < P[2]; pz++) for (let py = 0; py < P[1]; py++) for (let px = 0; px < P[0]; px++) {
    let m = 0;
    for (let z = pz * k; z < Math.min(Z, pz * k + k); z++) for (let y = py * k; y < Math.min(Y, py * k + k); y++) for (let x = px * k; x < Math.min(X, px * k + k); x++) m = Math.max(m, vol[(z * Y + y) * X + x]);
    assert.equal(full[(pz * P[1] + py) * P[0] + px], Math.round(m * 255));
}
// a single bright voxel keeps its brightness (max-pooling)
const one = new Float32Array(X * Y * Z); one[(3 * Y + 2) * X + 5] = 1;
const o = new Uint8Array(P[0] * P[1] * P[2]); poolSlab(one, dims, k, 0, Z, 1, o);
assert.equal(Math.max(...o), 255);
assert.equal(autoFactor([512, 512, 512], 2 ** 27), 1);
assert.equal(autoFactor([1024, 1024, 1024], 2 ** 27), 2);
assert.equal(autoFactor([100, 100, 100], 1e9), 1);
const p = [10, 20, 30];
assert.ok(fromPooled(toPooled(p, 3), 3).every((c, i) => Math.abs(c - p[i]) < 1e-9));
assert.throws(() => poolSlab(vol, dims, 2, 1, 2, 1, full));
console.log('pool tests ok');
