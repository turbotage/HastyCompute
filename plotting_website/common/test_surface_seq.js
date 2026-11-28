import assert from 'node:assert/strict';
import { blend, parseSeq, pick } from './surface_seq.js';

const t = (data) => ({ data });
const mesh = (x) => new Map([['vertices', t(Float32Array.from([x, 0, 0, x + 1, 0, 0, x, 1, 0]))], ['faces', t(Uint32Array.from([0, 1, 2]))]]);
const seqValue = (links) => new Map([['kind', 'mesh_seq'], ['frames', [mesh(0), mesh(2), mesh(4), mesh(100)]], ['links', t(Uint8Array.from(links))], ['dt', 0.05]]);

const s = parseSeq(seqValue([1, 1, 0]));
assert.equal(s.n, 4);
assert.equal(s.dt, 0.05);
assert.deepEqual(s.frames[0].scalars, {});
const withSc = seqValue([1, 1, 0]);
withSc.get('frames')[0].set('scalars', new Map([['support', t(Float32Array.from([1, 0.5, 0]))]]));
assert.deepEqual(Array.from(parseSeq(withSc).frames[0].scalars.support), [1, 0.5, 0]);

// exact frames
assert.deepEqual(pick(s, 2), { i: 2, j: 2, a: 0, linked: false });
// between linked frames: blended, quantised to quarters
let p = pick(s, 0.5);
assert.deepEqual(p, { i: 0, j: 1, a: 0.5, linked: true });
const out = blend(s, p.i, p.j, p.a, new Float32Array(9));
assert.equal(out[0], 1);          // (0 + 2) / 2
p = pick(s, 1.3);
assert.equal(p.a, 0.25);          // 0.3 → nearest quarter
// an unlinked transition (frames 2 → 3) snaps to the nearer frame; no blending across a new surface run
assert.deepEqual(pick(s, 2.3), { i: 2, j: 2, a: 0, linked: false });
assert.deepEqual(pick(s, 2.6), { i: 3, j: 3, a: 0, linked: false });
// the wrap (last → first) is never linked, time is cyclic
assert.deepEqual(pick(s, 3.6), { i: 0, j: 0, a: 0, linked: false });
assert.deepEqual(pick(s, -0.7), { i: 3, j: 3, a: 0, linked: false });
assert.equal(pick(s, 4 + 0.5).i, 0);
// rounding up to the next whole frame
assert.deepEqual(pick(s, 0.9), { i: 1, j: 1, a: 0, linked: false });
// a vertex-count mismatch is never blended
const bad = seqValue([1, 1, 1]);
bad.get('frames')[1] = new Map([['vertices', t(new Float32Array(12))], ['faces', t(Uint32Array.from([0, 1, 2]))]]);
assert.equal(pick(parseSeq(bad), 0.5).linked, false);
assert.throws(() => parseSeq(new Map([['kind', 'mesh']])));
console.log('surface_seq ok');
