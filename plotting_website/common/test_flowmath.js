// node plotting_website/common/test_flowmath.js — analytic checks of flowmath.js (Poiseuille tubes, rigid rotation, uniform drift)
import assert from 'node:assert/strict';
import { FlowField, Centerlines, regionFrom, streamlines, pathlines, measurePlane, measureStack, walkStack, averageStack, derivedScalar, pickRay, percentile, seedsDisc, seedsRegion, lineAttribute, unit } from './flowmath.js';

const ok = (name) => console.log(`ok  ${name}`);
const near = (a, b, tol, what) => assert.ok(Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (tol ${tol})`);

// ── two parallel Poiseuille tubes, oblique to the grid ──────────────────────────────────────────────────────────────
const N = 64, F = 8, dt = 0.05, h = 5e-4;                  // 0.5 mm voxels, 32 mm box
const axis = unit([0.3, 0.5, 0.81]), R = 5e-3, V0 = 1.2;
const off = (() => { const e = unit([axis[1], -axis[0], 0]); return [e[0] * 0.0125, e[1] * 0.0125, e[2] * 0.0125]; })();   // second tube 12.5 mm away
const C1 = [N / 2 * h, N / 2 * h, N / 2 * h], C2 = C1.map((c, i) => c + off[i]);
const modu = (f) => 1 + 0.5 * Math.sin((2 * Math.PI * f) / F);
const nv = N * N * N;
const frames = Array.from({ length: F }, () => new Float32Array(3 * nv));
const region = new Uint8Array(nv), pc = new Float32Array(nv);
for (let k = 0; k < N; k++) for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) {
    const q = (k * N + j) * N + i, p = [i * h, j * h, k * h];
    for (const [c, sgn] of [[C1, 1], [C2, 1]]) {
        const d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]], a = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
        const r2 = d[0] ** 2 + d[1] ** 2 + d[2] ** 2 - a * a, r = Math.sqrt(Math.max(r2, 0));
        if (r < R) {
            region[q] = 1; pc[q] = 1 - 0.2 * (r / R) ** 2;
            for (let f = 0; f < F; f++) { const s = V0 * (1 - (r / R) ** 2) * modu(f) * sgn; for (let cc = 0; cc < 3; cc++) frames[f][cc * nv + q] = s * axis[cc]; }
        }
    }
}
const field = new FlowField({ dims: [N, N, N], spacing: [h, h, h], dt, frames });
const vox = (c) => c.map((x) => x / h);
const Qtrue = (f) => V0 * Math.PI * R * R / 2 * modu(f);

// sampling: trilinear exact on a linear field
{
    const lin = Array.from({ length: 2 }, (_, f) => { const a = new Float32Array(3 * 27); for (let q = 0; q < 27; q++) { const i = q % 3, j = ((q / 3) | 0) % 3, k = (q / 9) | 0; a[q] = i + 2 * j + 3 * k + 10 * f; } return a; });
    const lf = new FlowField({ dims: [3, 3, 3], spacing: [1, 1, 1], dt: 1, frames: lin });
    const v = lf.sample([0.5, 1.25, 0.75], 0.5);
    near(v[0], 0.5 + 2 * 1.25 + 3 * 0.75 + 5, 1e-5, 'trilinear + time');
    assert.equal(lf.sample([2.01, 1, 1], 0), null);
    near(lf.sample([1, 1, 1], 1.5)[0], (6 + 16) / 2, 1e-5, 'time interpolation');
    near(lf.sample([1, 1, 1], 3.5)[0], (16 + 6) / 2, 1e-5, 'cyclic wrap (frame 1 → 0)');
    ok('sampling');
}

// plane quantification: single tube, second tube in the disc is excluded by the connected component
{
    const center = vox(C1), m = measurePlane(field, { center, normal: axis, radiusM: 9e-3, n: 64, region });
    for (let f = 0; f < F; f++) near(m.q[f] / Qtrue(f), 1, 0.04, `Q(t=${f}) vs Poiseuille`);
    near(m.area_mm2, Math.PI * 25, 4, 'lumen area mm²');
    const wide = measurePlane(field, { center, normal: axis, radiusM: 14e-3, n: 64, region });
    const noReg = measurePlane(field, { center, normal: axis, radiusM: 14e-3, n: 64, region: null });
    near(wide.q[2] / Qtrue(2), 1, 0.05, 'wide disc with region still only the clicked vessel');
    assert.ok(noReg.q[2] > 1.6 * wide.q[2], 'without the region the disc picks up the second tube too');
    ok(`plane Q matches Poiseuille (net ${m.net_ml.toFixed(3)} ml/cycle, expect ${(Array.from({ length: F }, (_, f) => Qtrue(f)).reduce((a, b) => a + b) * dt * 1e6).toFixed(3)}), second tube excluded`);
}

// stacks from a centerline, either orientation; multi-plane averaging
{
    const pts = new Float32Array(3 * 61);
    for (let i = 0; i <= 60; i++) { const s = (i - 30) * 0.001; for (let a = 0; a < 3; a++) pts[3 * i + a] = (C1[a] + s * axis[a]) / h; }
    const cl = new Centerlines({ points: pts, offsets: new Uint32Array([0, 61]) }, [h, h, h]);
    const nb = cl.nearest(vox(C1).map((x, i) => x + (i === 0 ? 2.1 : 0)));   // click off-axis
    assert.ok(nb.dist_mm < 2.5);
    const st = cl.stack(vox(C1).map((x, i) => x + (i === 0 ? 2.1 : 0)), 5, 1.0);
    assert.equal(st.planes.length, 5);
    near(Math.abs(st.planes[2].normal[0] * axis[0] + st.planes[2].normal[1] * axis[1] + st.planes[2].normal[2] * axis[2]), 1, 1e-3, 'centerline tangent');
    const planes = measureStack(field, st.planes, { radiusM: 9e-3, n: 64, region });
    const avg = averageStack(field, planes);
    for (let f = 0; f < F; f++) near(avg.q[f] / Qtrue(f), 1, 0.04, `stack mean Q(t=${f})`);
    assert.ok(avg.net_ml > 0, 'orient makes the net flow positive');
    const rev = st.planes.map((d) => ({ center: d.center, normal: d.normal.map((x) => -x) })).reverse();
    const avg2 = averageStack(field, measureStack(field, rev, { radiusM: 9e-3, n: 64, region }));
    near(avg2.net_ml / avg.net_ml, 1, 0.03, 'reversed centerline gives the same oriented flow');
    // the profile is parabolic: centre / mean = 2
    const nn = 64 * 64, f = 2;
    let mx = 0, s = 0, c = 0;
    for (let i = 0; i < nn; i++) if (avg.lumen[i]) { mx = Math.max(mx, avg.img[f * nn + i]); s += avg.img[f * nn + i]; c++; }
    near(mx / (s / c), 2, 0.12, 'peak / mean velocity of a parabolic profile');
    ok('centerline stack + averaging + orientation + parabolic profile');
}

// vessel tool without centerlines: walk along the flow from an off-centre click
{
    const click = vox(C1).map((x, i) => x + (i === 1 ? 3.0 : 0) - (i === 2 ? 1.5 : 0));
    const planes = walkStack(field, region, click, { nPlanes: 5, stepM: 0.002, radiusM: 9e-3, n: 64 });
    assert.ok(planes && planes.length === 5, 'walk produced 5 planes');
    for (const m of planes) {
        const d = m.center.map((x, i) => x * h - C1[i]), a = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
        const r = Math.sqrt(Math.max(0, d[0] ** 2 + d[1] ** 2 + d[2] ** 2 - a * a));
        assert.ok(r < 0.75e-3, `plane centre on the axis (${(r * 1e3).toFixed(2)} mm off)`);
    }
    const avg = averageStack(field, planes);
    for (let f = 0; f < F; f++) near(avg.q[f] / Qtrue(f), 1, 0.05, `walk-stack Q(t=${f})`);
    ok('walkStack follows the vessel from a rim click');
}

// tracing
{
    const seeds = seedsDisc(field, region, vox(C1), axis, 3e-3, 12, 3);
    assert.equal(seeds.length, 12);
    const sl = streamlines(field, 2, seeds, { step: 0.5, maxSteps: 300, stop: region });
    assert.ok(sl.offsets.length - 1 >= 10);
    for (let l = 0; l + 1 < sl.offsets.length; l++) {
        const a = sl.offsets[l], b = sl.offsets[l + 1] - 1;
        const d = [0, 1, 2].map((k) => sl.points[3 * b + k] - sl.points[3 * a + k]), L = Math.hypot(...d);
        assert.ok(L > 20, 'streamline length');
        near(Math.abs(d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2]) / L, 1, 5e-3, 'streamline follows the tube axis');
    }
    assert.equal(lineAttribute(sl, 'age').length, sl.speed.length);
    const rs = seedsRegion(field, region, 30, 1);
    assert.equal(rs.length, 30);
    // uniform drift: pathline displacement = v · duration · dt
    const uni = new FlowField({ dims: [40, 8, 8], spacing: [1e-3, 1e-3, 1e-3], dt: 0.05, frames: Array.from({ length: 4 }, () => { const a = new Float32Array(3 * 2560); a.fill(0.1, 0, 2560); return a; }) });
    const pl = pathlines(uni, 0, 3, [[2, 4, 4]], { step: 4 });
    const e = pl.offsets[1] - 1;
    near(pl.points[3 * e], 2 + 0.1 * 3 * 0.05 / 1e-3, 1e-3, 'pathline end x');       // 15 voxels
    near(pl.time[e] - pl.time[0], 3 * 0.05, 1e-6, 'pathline time span');
    ok(`tracing (${sl.offsets.length - 1} streamlines along the axis, pathline drift exact)`);
}

// derived scalars
{
    const n = 16, Om = 3, W = 0.5, sp = 1e-3, a = new Float32Array(3 * n * n * n);
    for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
        const q = (k * n + j) * n + i, x = (i - 8) * sp, y = (j - 8) * sp;
        a[q] = -Om * y; a[n ** 3 + q] = Om * x; a[2 * n ** 3 + q] = W;
    }
    const f = new FlowField({ dims: [n, n, n], spacing: [sp, sp, sp], dt: 1, frames: [a] });
    const mid = (8 * n + 8) * n + 8;
    near(derivedScalar(f, 0, 'vorticity')[mid], 2 * Om, 1e-3, 'vorticity of rigid rotation');
    near(derivedScalar(f, 0, 'q')[mid], Om * Om, 1e-3, 'Q-criterion of rigid rotation');
    near(derivedScalar(f, 0, 'helicity')[mid], 2 * Om * W, 1e-3, 'helicity');
    near(derivedScalar(f, 0, 'speed')[mid], Math.hypot(0, 0, W), 1e-6, 'speed');
    ok('derived scalars');
}

// ray picking
{
    const origin = vox(C1).map((x, i) => x + (i === 0 ? 40 : 0) + (i === 1 ? 1 : 0));
    const hit = pickRay(pc, [N, N, N], origin, [-1, 0, 0.01], { level: 0.5 });
    assert.ok(hit, 'ray hits the tube');
    const p = hit.p.map((x) => x * h), d = p.map((x, i) => x - C1[i]), a = d[0] * axis[0] + d[1] * axis[1] + d[2] * axis[2];
    assert.ok(Math.sqrt(Math.max(0, d[0] ** 2 + d[1] ** 2 + d[2] ** 2 - a * a)) < R * 1.05, 'hit inside the lumen');
    assert.equal(pickRay(pc, [N, N, N], [100, 100, 100], [1, 0, 0], { level: 0.5 }), null);
    // a non-unit direction (the viewer passes scene rays divided by the voxel size) must pick the same point
    const hit2 = pickRay(pc, [N, N, N], origin, [-0.37, 0, 0.0037], { level: 0.5 });
    assert.ok(hit2 && Math.hypot(...hit2.p.map((x, i) => x - hit.p[i])) < 1, 'scaled direction picks the same point');
    ok('ray picking');
}

// percentile + region
{
    const a = Float32Array.from({ length: 100000 }, (_, i) => i / 1000);
    near(percentile(a, 0.5), 50, 0.05, 'median');
    near(percentile(a, 0.999), 99.9, 0.05, 'p99.9');
    const r = regionFrom({ pc: a, quantile: 0.9 });
    near(r.reduce((s, x) => s + x, 0) / a.length, 0.1, 1e-3, 'region fraction');
    ok('percentile / region');
}
console.log('ok');
