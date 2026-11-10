import assert from 'node:assert/strict';
import { FlowField } from './flowmath.js';
import { finiteQuantile, lumenComponent, marchingTets, smoothIndicator, wallNormals, wallShear } from './surface.js';

// Poiseuille tube along z, radius R (voxels of 1 mm): v_z = V0 (1 − r²/R²); analytic WSS = μ·2·V0/R
const dims = [40, 40, 30], [nx, ny, nz] = dims, nv = nx * ny * nz, sp = [1e-3, 1e-3, 1e-3], R = 8, V0 = 1.0, mu = 3.5e-3, cx = 19.5, cy = 19.5;
const mask = new Uint8Array(nv), frames = [];
const fr = (scale) => { const f = new Float32Array(3 * nv); for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) { const r2 = ((x - cx) ** 2 + (y - cy) ** 2) / (R * R), q = (z * ny + y) * nx + x; if (r2 < 1) { f[2 * nv + q] = scale * V0 * (1 - r2); mask[q] = 1; } } return f; };
frames.push(fr(1), fr(1));
const field = new FlowField({ dims, spacing: sp, dt: 0.05, frames });
const comp = lumenComponent(mask, dims, [19.5, 19.5, 15]);
assert.ok(comp && comp.reduce((a, b) => a + b, 0) === mask.reduce((a, b) => a + b, 0));
assert.equal(lumenComponent(mask, dims, [2, 2, 2], 1), null);
const ind = smoothIndicator(comp, dims, 2);
const { pos, idx } = marchingTets(ind, dims, 0.5);
assert.ok(pos.length > 3000 && idx.length % 3 === 0);
// the surface is a tube of radius ≈ R and normals point outward
const nrm = wallNormals(ind, dims, sp, pos), V = pos.length / 3;
let rs = 0, cnt = 0, outward = 0;
for (let v = 0; v < V; v++) {
    if (pos[3 * v + 2] < 5 || pos[3 * v + 2] > 24) continue;
    const dx = pos[3 * v] - cx, dy = pos[3 * v + 1] - cy, r = Math.hypot(dx, dy);
    rs += r; cnt++;
    if ((dx * nrm[3 * v] + dy * nrm[3 * v + 1]) / r > 0.95) outward++;
}
assert.ok(Math.abs(rs / cnt - R) < 0.6, `radius ${rs / cnt}`);
assert.ok(outward / cnt > 0.97, `outward ${outward / cnt}`);
// triangle winding agrees with the outward normals
let wind = 0, tot = 0;
for (let t = 0; t < idx.length; t += 3) {
    const [a, b, c] = [idx[t], idx[t + 1], idx[t + 2]].map((i) => [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]);
    if (a[2] < 5 || a[2] > 24) continue;
    const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    const n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    tot++; if (n[0] * (a[0] - cx) + n[1] * (a[1] - cy) > 0) wind++;
}
assert.ok(wind / tot > 0.99, `winding ${wind / tot}`);
// WSS vs analytic (tube radius from the mesh, so use the measured mean radius in the reference)
const Rm = (rs / cnt) * 1e-3, ref = (mu * 2 * V0) / Rm;
for (const [order, tol] of [[2, 0.12], [1, 0.2]]) {
    const w = wallShear(field, pos, nrm, { dM: 1e-3, mu, order });
    let s = 0, n = 0;
    for (let v = 0; v < V; v++) if (pos[3 * v + 2] > 8 && pos[3 * v + 2] < 21 && Number.isFinite(w.tmean[v])) { s += w.tmean[v]; n++; }
    const mean = s / n;
    console.log(`order ${order}: WSS ${mean.toFixed(3)} Pa, analytic ${ref.toFixed(3)} Pa`);
    assert.ok(Math.abs(mean - ref) / ref < tol, `order ${order}: mean WSS ${mean.toFixed(3)} Pa vs analytic ${ref.toFixed(3)} (R=${(Rm * 1e3).toFixed(2)} mm)`);
    // steady flow: OSI ≈ 0 and WSS is axial
    let osi = 0; for (let v = 0; v < V; v++) if (Number.isFinite(w.osi[v])) osi = Math.max(osi, w.osi[v]);
    assert.ok(osi < 1e-4, `osi ${osi}`);
}
// oscillating flow (+, −) with equal magnitude: OSI = ½, time-mean |τ| unchanged
const field2 = new FlowField({ dims, spacing: sp, dt: 0.05, frames: [fr(1), fr(-1)] });
const w2 = wallShear(field2, pos, nrm, { dM: 1e-3, mu, order: 2 });
let ok = 0, tot2 = 0;
for (let v = 0; v < V; v++) if (pos[3 * v + 2] > 8 && pos[3 * v + 2] < 21 && Number.isFinite(w2.osi[v])) { tot2++; if (Math.abs(w2.osi[v] - 0.5) < 0.02) ok++; }
assert.ok(ok / tot2 > 0.97, `osi 0.5: ${ok}/${tot2}`);
// samples outside the box give NaN, not garbage
const edge = wallShear(field, Float32Array.from([0, 0, 0]), Float32Array.from([1, 0, 0]), { dM: 1e-3, mu });
assert.ok(Number.isNaN(edge.tmean[0]));
assert.equal(finiteQuantile([1, 2, NaN, 3, 4], 0.5), 2);
console.log('surface tests ok');

// ── energy: Poiseuille tube, v_z = V0 (1 − r²/R²): Φ(r) = μ (2 V0 r / R²)², KE = ½ρ v²
{
    const { wallEnergy, lumenEnergy, dissipationDensity, velocityGradient } = await import('./surface.js');
    const rho = 1060, G = velocityGradient(field, [cx + 3, cy, 15], 0, 1);
    // ∂v_z/∂x at r = 3 voxels: −2 V0 r / R² per metre (central difference over ±1 voxel of a parabola is exact)
    assert.ok(Math.abs(G[2][0] - (-2 * V0 * 3e-3 / (R * 1e-3) ** 2)) < 1e-6 * 1e6, `G ${G[2][0]}`);
    assert.ok(Math.abs(dissipationDensity([[0, 0, 0], [0, 0, 0], [1, 0, 0]], 2) - 2 * 2 * (0.25 + 0.25)) < 1e-12);   // simple shear γ=1: Φ = μ γ² = 2 → 2μ·(2·¼)... 
    const e = wallEnergy(field, pos, nrm, { dM: 2e-3, mu, rho });
    let ok = 0, tot = 0;
    for (let v = 0; v < V; v++) {
        if (pos[3 * v + 2] < 8 || pos[3 * v + 2] > 21 || !Number.isFinite(e.dissMean[v])) continue;
        const r = (rs / cnt) - 2, exp = mu * (2 * V0 * r * 1e-3 / ((rs / cnt) * 1e-3) ** 2) ** 2, expKE = 0.5 * rho * (V0 * (1 - (r / (rs / cnt)) ** 2)) ** 2;
        tot++; if (Math.abs(e.dissMean[v] - exp) / exp < 0.35 && Math.abs(e.keMean[v] - expKE) / expKE < 0.35) ok++;
    }
    assert.ok(ok / tot > 0.9, `energy at vertices ${ok}/${tot}`);
    // lumen integral vs analytic over the eroded radius (R−1): 2π μ V0² (R−1)⁴ / R⁴ · L
    const en = lumenEnergy(field, comp, { mu, rho }), L = 28e-3, Re = (R - 1) * 1e-3, Rm0 = R * 1e-3;
    const exp = 2 * Math.PI * mu * V0 * V0 * (Re ** 4 / Rm0 ** 4) * L;
    console.log(`lumen dissipation ${en.diss_W[0].toExponential(3)} W vs analytic(eroded) ${exp.toExponential(3)} W`);
    assert.ok(Math.abs(en.diss_W[0] - exp) / exp < 0.2, 'lumen dissipation');
    const keExp = 0.5 * rho * V0 * V0 * Math.PI * Rm0 ** 2 * L / 3 * 1;          // ∫ ½ρ v² dV = ½ρV0² · πR²L/3
    assert.ok(Math.abs(en.ke_J[0] - keExp) / keExp < 0.2, `KE ${en.ke_J[0]} vs ${keExp}`);
}
console.log('energy tests ok');
{
    const { treeMesh } = await import('./surface.js');
    // two separate tubes: only the one at the seed is returned; coordinates are in volume voxels
    const d2 = [60, 30, 30], u8 = new Uint8Array(60 * 30 * 30);
    for (let z = 0; z < 30; z++) for (let y = 0; y < 30; y++) for (let x = 0; x < 60; x++) {
        const a = Math.hypot(y - 10, z - 10) < 4 && x > 3 && x < 25, b = Math.hypot(y - 20, z - 20) < 4 && x > 35 && x < 56;
        if (a || b) u8[(z * 30 + y) * 60 + x] = 200;
    }
    const t = treeMesh(u8, d2, [12, 10, 10], 100);
    assert.ok(t.pos && t.idx.length > 100);
    let mx = 0, mn = 1e9; for (let v = 0; v < t.pos.length; v += 3) { mx = Math.max(mx, t.pos[v]); mn = Math.min(mn, t.pos[v]); }
    assert.ok(mx < 28 && mn > 1, `x range ${mn}..${mx}`);
    assert.ok(treeMesh(u8, d2, [12, 10, 10], 250).error);                    // above everything
    assert.ok(treeMesh(u8, d2, [12, 10, 10], 100, { maxVox: 100 }).error);  // too big is reported, not truncated
    assert.ok(treeMesh(u8, d2, [12, 10, 10], 100, { maxTris: 50 }).error);
}
console.log('tree tests ok');
