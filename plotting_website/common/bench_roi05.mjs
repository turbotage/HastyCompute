// Timing of the vessel-tool pipeline on a 0.5 mm-sized box (synthetic Poiseuille aorta, 106^3 voxels = 53 mm, 17 frames). Not a test.
import { FlowField, averageStack, walkStack } from './flowmath.js';
import { lumenComponent, lumenEnergy, marchingTets, smoothIndicator, wallEnergy, wallNormals, wallShear } from './surface.js';
const n = 106, dims = [n, n, n], nv = n ** 3, sp = [5e-4, 5e-4, 5e-4], F = 17, R = 24, c = (n - 1) / 2;   // 12 mm radius lumen
const mask = new Uint8Array(nv), frames = [];
for (let f = 0; f < F; f++) {
    const fr = new Float32Array(3 * nv), a = 0.3 + 0.7 * Math.sin(Math.PI * f / F) ** 2;
    for (let z = 0; z < n; z++) for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) { const r2 = ((x - c) ** 2 + (y - c) ** 2) / (R * R), q = (z * n + y) * n + x; if (r2 < 1) { fr[2 * nv + q] = a * (1 - r2); mask[q] = 1; } }
    frames.push(fr);
}
const field = new FlowField({ dims, spacing: sp, dt: 0.05, frames });
const T = (l, f) => { const t = performance.now(), r = f(); console.log(l.padEnd(26), `${(performance.now() - t).toFixed(0)} ms`); return r; };
console.log(`box ${n}^3 = ${(nv * (F * 12 + 5) / 1048576).toFixed(0)} MB, ${F} frames`);
const st = T('walkStack (5 planes)', () => walkStack(field, mask, [c, c, c], { nPlanes: 5, stepM: 1.5e-3, radiusM: 18e-3, n: 80 }));
T('averageStack', () => averageStack(field, st));
const comp = T('lumenComponent', () => lumenComponent(mask, dims, [c, c, c]));
const ind = T('smoothIndicator', () => smoothIndicator(comp, dims, 2));
const m = T('marchingTets', () => marchingTets(ind, dims, 0.5));
const nrm = T('wallNormals', () => wallNormals(ind, dims, sp, m.pos));
T('wallShear (2nd order)', () => wallShear(field, m.pos, nrm, { dM: 5e-4, order: 2 }));
T('wallEnergy', () => wallEnergy(field, m.pos, nrm, { dM: 5e-4 }));
T('lumenEnergy', () => lumenEnergy(field, comp));
console.log(`${m.pos.length / 3} vertices, ${m.idx.length / 3} triangles`);
