import assert from 'node:assert/strict';
import { FlowField, averageStack, measurePlane } from './flowmath.js';
import { fetchRoi, roiBox, roiContains, shiftResult } from './roi.js';

// synthetic volume: a z-directed tube of radius 3.2 voxels with a parabolic profile, constant in time
const dims = [40, 36, 30], spacing = [1e-3, 1e-3, 1e-3], F = 4, nv = dims[0] * dims[1] * dims[2], cx = 22.3, cy = 17.6, R = 3.2;
const frames = [], mask = new Uint8Array(nv), pc = new Float32Array(nv);
for (let f = 0; f < F; f++) {
    const fr = new Float32Array(3 * nv);
    for (let z = 0; z < dims[2]; z++) for (let y = 0; y < dims[1]; y++) for (let x = 0; x < dims[0]; x++) {
        const r2 = ((x - cx) ** 2 + (y - cy) ** 2) / (R * R), i = (z * dims[1] + y) * dims[0] + x;
        if (r2 < 1) { fr[2 * nv + i] = 1.0 * (1 - r2) * (f + 1) / F; mask[i] = 1; pc[i] = 1 - r2; }
    }
    frames.push(fr);
}
const full = new FlowField({ dims, spacing, dt: 0.05, frames });
const tensor = (a, shape) => ({ data: a, dtype: a instanceof Uint8Array ? 'uint8' : 'float32', shape });

// fake server: implements the slice strings fetchRoi sends
const crop = (arr, c, lo, hi) => {
    const out = new (arr.constructor)((hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]) * c);
    let o = 0;
    for (let k = 0; k < c; k++) for (let z = lo[2]; z < hi[2]; z++) for (let y = lo[1]; y < hi[1]; y++) for (let x = lo[0]; x < hi[0]; x++) out[o++] = arr[k * nv + (z * dims[1] + y) * dims[0] + x];
    return out;
};
const fake = {
    calls: [],
    async getTensor(id, { slice }) {
        this.calls.push([id, slice]);
        const n = [...slice.matchAll(/(\d+):(\d+)/g)].map((m) => [Number(m[1]), Number(m[2])]);
        const [z, y, x] = n.slice(-3), lo = [x[0], y[0], z[0]], hi = [x[1], y[1], z[1]];
        if (slice.startsWith('["vel"]')) {
            const all = new Float32Array(F * 3 * (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]));
            let o = 0;
            for (let f = 0; f < F; f++) { const c = crop(frames[f], 3, lo, hi); all.set(c, o); o += c.length; }
            return tensor(all);
        }
        if (slice.startsWith('["mask"]')) return tensor(crop(mask, 1, lo, hi));
        return tensor(crop(pc, 1, lo, hi));
    },
};
const info = { frames: F, spacing, dt: 0.05, has_mask: true };

const p = [cx, cy, 15], box = roiBox(p, dims, spacing, 8e-3);
assert.deepEqual(box.lo, [14, 9, 7]); assert.deepEqual(box.hi, [31, 26, 24]);
assert.ok(roiContains({ lo: [0, 0, 0], hi: dims }, box) && !roiContains(box, { lo: [0, 0, 0], hi: dims }));
const clipped = roiBox([1, 1, 1], dims, spacing, 8e-3);
assert.deepEqual(clipped.lo, [0, 0, 0]);

const roi = await fetchRoi(fake, { id: 'v', pcId: 'p', info, box });
assert.deepEqual(roi.field.dims, [17, 17, 17]);
assert.ok(fake.calls.some(([, s]) => s === '["vel"][:, :, 7:24, 9:26, 14:31]'), fake.calls.join('|'));

// the measurement in the box (box coordinates) equals the one in the full volume (full coordinates)
const region = roi.mask;
const loc = p.map((c, k) => c - box.lo[k]);
const a = measurePlane(roi.field, { center: loc, normal: [0, 0, 1], radiusM: 6e-3, n: 48, region });
const b = measurePlane(full, { center: p, normal: [0, 0, 1], radiusM: 6e-3, n: 48, region: mask });
assert.ok(a.nlumen > 0);
assert.equal(a.nlumen, b.nlumen);
for (let f = 0; f < F; f++) assert.ok(Math.abs(a.q[f] - b.q[f]) <= 1e-5 * Math.abs(b.q[f]), `q ${f}: ${a.q[f]} vs ${b.q[f]}`);
assert.ok(Math.abs(a.net_ml - b.net_ml) <= 1e-5 * Math.abs(b.net_ml));
// shifting back
const res = { planes: [a], avg: averageStack(roi.field, [a]) };
assert.equal(res.avg.center, a.center);          // shared array, as in flowmath
const aOld = a.center;
shiftResult(res, box.lo);
assert.deepEqual(res.planes[0].center, b.center.map((c, k) => c));
assert.deepEqual(res.avg.center, b.center);
assert.deepEqual(aOld, loc);                     // not mutated in place
console.log('roi tests ok');
