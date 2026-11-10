// Vessel wall surface and wall-derived quantities, computed on the full-resolution box of the vessel tool (roi.js).
//   lumenComponent  3-D connected region of the lumen mask that contains the click
//   smoothIndicator box-blurred 0/1 lumen indicator; its 0.5 iso-surface is the wall
//   marchingTets    iso-surface mesh (Kuhn tetrahedra: small tables, consistent across cells), vertices shared, normals outward
//   wallShear       wall shear stress by one-sided differences of the tangential velocity along the inward normal
// Everything is in box voxel coordinates; spacing (m) converts to physical units. No data other than the velocity and the
// lumen mask enters, so the numbers are exactly as good as those — see the assumptions listed with each quantity in the UI.

/** 6-connected component of region[] (Uint8, [Z,Y,X]) that contains the voxel nearest to `seed` (searched within `maxR` voxels) */
export function lumenComponent(region, dims, seed, maxR = 3) {
    const [nx, ny, nz] = dims, out = new Uint8Array(region.length);
    const si = Math.round(seed[0]), sj = Math.round(seed[1]), sk = Math.round(seed[2]);
    let start = -1, best = Infinity;
    for (let k = Math.max(0, sk - maxR); k <= Math.min(nz - 1, sk + maxR); k++) for (let j = Math.max(0, sj - maxR); j <= Math.min(ny - 1, sj + maxR); j++) for (let i = Math.max(0, si - maxR); i <= Math.min(nx - 1, si + maxR); i++) {
        const q = (k * ny + j) * nx + i;
        if (!region[q]) continue;
        const d = (i - seed[0]) ** 2 + (j - seed[1]) ** 2 + (k - seed[2]) ** 2;
        if (d < best) { best = d; start = q; }
    }
    if (start < 0) return null;
    const stack = [start];
    out[start] = 1;
    while (stack.length) {
        const q = stack.pop(), i = q % nx, j = ((q / nx) | 0) % ny, k = (q / (nx * ny)) | 0;
        if (i > 0 && region[q - 1] && !out[q - 1]) { out[q - 1] = 1; stack.push(q - 1); }
        if (i < nx - 1 && region[q + 1] && !out[q + 1]) { out[q + 1] = 1; stack.push(q + 1); }
        if (j > 0 && region[q - nx] && !out[q - nx]) { out[q - nx] = 1; stack.push(q - nx); }
        if (j < ny - 1 && region[q + nx] && !out[q + nx]) { out[q + nx] = 1; stack.push(q + nx); }
        if (k > 0 && region[q - nx * ny] && !out[q - nx * ny]) { out[q - nx * ny] = 1; stack.push(q - nx * ny); }
        if (k < nz - 1 && region[q + nx * ny] && !out[q + nx * ny]) { out[q + nx * ny] = 1; stack.push(q + nx * ny); }
    }
    return out;
}

/** 0/1 mask → float indicator blurred by `passes` radius-1 box filters per axis (edges clamped, so a vessel cut by the box stays open there) */
export function smoothIndicator(mask, dims, passes = 2) {
    const [nx, ny, nz] = dims;
    let a = Float32Array.from(mask), b = new Float32Array(a.length);
    const line = (src, dst, n, stride, count, base) => {
        for (let c = 0; c < count; c++) {
            const o = base(c);
            for (let i = 0; i < n; i++) {
                const m = o + Math.max(0, i - 1) * stride, p = o + Math.min(n - 1, i + 1) * stride, q = o + i * stride;
                dst[q] = (src[m] + src[q] + src[p]) / 3;
            }
        }
    };
    for (let it = 0; it < passes; it++) {
        line(a, b, nx, 1, ny * nz, (c) => c * nx); [a, b] = [b, a];
        line(a, b, ny, nx, nx * nz, (c) => (c % nx) + ((c / nx) | 0) * nx * ny); [a, b] = [b, a];
        line(a, b, nz, nx * ny, nx * ny, (c) => c); [a, b] = [b, a];
    }
    return a;
}

const CORNER = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
const TETS = [[0, 5, 1, 6], [0, 1, 2, 6], [0, 2, 3, 6], [0, 3, 7, 6], [0, 7, 4, 6], [0, 4, 5, 6]];   // Kuhn: all share the 0–6 diagonal

/**
 * Iso-surface of s (Float32Array [Z,Y,X]) at `iso`; inside = s ≥ iso. Returns {pos: Float32Array [3V] (voxel coords), idx: Uint32Array [3T]};
 * triangles are wound so the geometric normal points from inside to outside.
 */
export function marchingTets(s, dims, iso = 0.5) {
    const [nx, ny, nz] = dims, pos = [], idx = [], cache = new Map();
    const node = (i, j, k) => (k * ny + j) * nx + i;
    const vert = (a, b) => {                       // vertex on the edge between grid nodes a and b (corner records)
        if (a.n > b.n) [a, b] = [b, a];
        const dx = b.c[0] - a.c[0], dy = b.c[1] - a.c[1], dz = b.c[2] - a.c[2], key = a.n * 8 + (dx + 2 * dy + 4 * dz);
        let v = cache.get(key);
        if (v === undefined) {
            const t = (iso - a.s) / (b.s - a.s);
            v = pos.length / 3;
            pos.push(a.c[0] + t * dx, a.c[1] + t * dy, a.c[2] + t * dz);
            cache.set(key, v);
        }
        return v;
    };
    const emit = (i0, i1, i2, dir) => {            // orient the triangle so its normal has a positive component along dir
        const a = [pos[3 * i0], pos[3 * i0 + 1], pos[3 * i0 + 2]], b = [pos[3 * i1], pos[3 * i1 + 1], pos[3 * i1 + 2]], c = [pos[3 * i2], pos[3 * i2 + 1], pos[3 * i2 + 2]];
        const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]], w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        const n = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
        if (i0 === i1 || i1 === i2 || i0 === i2) return;
        if (n[0] * dir[0] + n[1] * dir[1] + n[2] * dir[2] >= 0) idx.push(i0, i1, i2); else idx.push(i0, i2, i1);
    };
    const cn = new Array(8);
    for (let k = 0; k < nz - 1; k++) for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
        let inside = 0;
        for (let c = 0; c < 8; c++) {
            const o = CORNER[c], n = node(i + o[0], j + o[1], k + o[2]);
            cn[c] = { n, s: s[n], c: [i + o[0], j + o[1], k + o[2]] };
            if (cn[c].s >= iso) inside++;
        }
        if (inside === 0 || inside === 8) continue;
        for (const t of TETS) {
            const ins = [], out = [];
            for (const c of t) (cn[c].s >= iso ? ins : out).push(cn[c]);
            if (!ins.length || !out.length) continue;
            const mean = (l) => [0, 1, 2].map((a) => l.reduce((x, p) => x + p.c[a], 0) / l.length);
            const mi = mean(ins), mo = mean(out), dir = [mo[0] - mi[0], mo[1] - mi[1], mo[2] - mi[2]];
            if (ins.length === 1) emit(vert(ins[0], out[0]), vert(ins[0], out[1]), vert(ins[0], out[2]), dir);
            else if (out.length === 1) emit(vert(ins[0], out[0]), vert(ins[1], out[0]), vert(ins[2], out[0]), dir);
            else {                                  // 2 / 2: quad a1–o1, a1–o2, a2–o2, a2–o1
                const q0 = vert(ins[0], out[0]), q1 = vert(ins[0], out[1]), q2 = vert(ins[1], out[1]), q3 = vert(ins[1], out[0]);
                emit(q0, q1, q2, dir); emit(q0, q2, q3, dir);
            }
        }
    }
    return { pos: Float32Array.from(pos), idx: Uint32Array.from(idx) };
}

/** trilinear sample of a scalar [Z,Y,X] */
export function sampleScalar(s, dims, x, y, z) {
    const [nx, ny, nz] = dims;
    x = Math.min(nx - 1, Math.max(0, x)); y = Math.min(ny - 1, Math.max(0, y)); z = Math.min(nz - 1, Math.max(0, z));
    const i = Math.min(x | 0, nx - 2), j = Math.min(y | 0, ny - 2), k = Math.min(z | 0, nz - 2), fx = x - i, fy = y - j, fz = z - k, q = (k * ny + j) * nx + i;
    const c = (o) => s[q + o];
    const a = (c(0) * (1 - fx) + c(1) * fx) * (1 - fy) + (c(nx) * (1 - fx) + c(nx + 1) * fx) * fy;
    const b = (c(nx * ny) * (1 - fx) + c(nx * ny + 1) * fx) * (1 - fy) + (c(nx * ny + nx) * (1 - fx) + c(nx * ny + nx + 1) * fx) * fy;
    return a * (1 - fz) + b * fz;
}

/** unit outward wall normals (physical directions, 3 per vertex) from the gradient of the indicator at each vertex */
export function wallNormals(ind, dims, spacing, pos) {
    const V = pos.length / 3, n = new Float32Array(3 * V), h = 0.75;
    for (let v = 0; v < V; v++) {
        const x = pos[3 * v], y = pos[3 * v + 1], z = pos[3 * v + 2];
        const g = [
            (sampleScalar(ind, dims, x + h, y, z) - sampleScalar(ind, dims, x - h, y, z)) / (2 * h * spacing[0]),
            (sampleScalar(ind, dims, x, y + h, z) - sampleScalar(ind, dims, x, y - h, z)) / (2 * h * spacing[1]),
            (sampleScalar(ind, dims, x, y, z + h) - sampleScalar(ind, dims, x, y, z - h)) / (2 * h * spacing[2]),
        ];
        const l = Math.hypot(g[0], g[1], g[2]) || 1;
        n[3 * v] = -g[0] / l; n[3 * v + 1] = -g[1] / l; n[3 * v + 2] = -g[2] / l;     // indicator falls outward
    }
    return n;
}

/**
 * Wall shear stress vectors (Pa) per frame and vertex: τ = μ ∂v_t/∂n at the wall with v = 0 there (no slip), from the tangential
 * velocity v_t at distance d (order 1: μ v_t(d)/d) or at d and 2d (order 2: μ (4 v_t(d) − v_t(2d)) / 2d, exact for a parabolic profile)
 * inside the lumen. Vertices whose sample points leave the box are NaN.
 * Returns {tau: Float32Array [F, V, 3], mag [F, V], tmean [V] (time-mean |τ|), osi [V] (oscillatory shear index), vmag}.
 */
export function wallShear(field, pos, nrm, { dM = 1e-3, mu = 3.5e-3, order = 2 } = {}) {
    const V = pos.length / 3, F = field.nf, sp = field.spacing, tau = new Float32Array(F * V * 3), mag = new Float32Array(F * V);
    const v1 = [0, 0, 0], v2 = [0, 0, 0], nanV = new Uint8Array(V);
    for (let v = 0; v < V; v++) {
        const n = [nrm[3 * v], nrm[3 * v + 1], nrm[3 * v + 2]];
        const at = (m) => [0, 1, 2].map((a) => pos[3 * v + a] - m * dM * n[a] / sp[a]);
        const p1 = at(1), p2 = order === 2 ? at(2) : null;
        for (let f = 0; f < F; f++) {
            const ok1 = field.sampleInto(p1[0], p1[1], p1[2], f, v1), ok2 = p2 ? field.sampleInto(p2[0], p2[1], p2[2], f, v2) : true;
            if (!ok1 || !ok2) { nanV[v] = 1; break; }
            const t = (u) => { const d = u[0] * n[0] + u[1] * n[1] + u[2] * n[2]; return [u[0] - d * n[0], u[1] - d * n[1], u[2] - d * n[2]]; };
            const a = t(v1), b = p2 ? t(v2) : null;
            for (let c = 0; c < 3; c++) tau[(f * V + v) * 3 + c] = b ? (mu * (4 * a[c] - b[c])) / (2 * dM) : (mu * a[c]) / dM;
        }
    }
    const tmean = new Float32Array(V), osi = new Float32Array(V);
    for (let v = 0; v < V; v++) {
        if (nanV[v]) { for (let f = 0; f < F; f++) { mag[f * V + v] = NaN; for (let c = 0; c < 3; c++) tau[(f * V + v) * 3 + c] = NaN; } tmean[v] = osi[v] = NaN; continue; }
        const sum = [0, 0, 0];
        let sm = 0;
        for (let f = 0; f < F; f++) {
            const a = tau[(f * V + v) * 3], b = tau[(f * V + v) * 3 + 1], c = tau[(f * V + v) * 3 + 2], m = Math.hypot(a, b, c);
            mag[f * V + v] = m; sm += m; sum[0] += a; sum[1] += b; sum[2] += c;
        }
        tmean[v] = sm / F;
        osi[v] = sm > 0 ? 0.5 * (1 - Math.hypot(...sum) / sm) : 0;
    }
    return { tau, mag, tmean, osi };
}

/** p-th quantile of the finite values (for colour scales) */
export function finiteQuantile(a, q, maxN = 200000) {
    const st = Math.max(1, Math.ceil(a.length / maxN)), v = [];
    for (let i = 0; i < a.length; i += st) if (Number.isFinite(a[i])) v.push(a[i]);
    v.sort((x, y) => x - y);
    return v.length ? v[Math.min(v.length - 1, Math.floor(q * (v.length - 1)))] : 0;
}

// ─── energy-based quantities ─────────────────────────────────────────────────────────────────────────────────────

/** velocity gradient tensor G[i][j] = ∂v_i/∂x_j (1/s) at voxel position p, frame f, by central differences of ±h voxels (null if a sample leaves the box) */
export function velocityGradient(field, p, f, h = 1) {
    const sp = field.spacing, a = [0, 0, 0], b = [0, 0, 0], G = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (let j = 0; j < 3; j++) {
        const q = [p[0], p[1], p[2]], r = [p[0], p[1], p[2]];
        q[j] += h; r[j] -= h;
        if (!field.sampleInto(q[0], q[1], q[2], f, a) || !field.sampleInto(r[0], r[1], r[2], f, b)) return null;
        for (let i = 0; i < 3; i++) G[i][j] = (a[i] - b[i]) / (2 * h * sp[j]);
    }
    return G;
}

/** viscous dissipation density Φ = 2 μ S:S (W/m³) of an incompressible Newtonian fluid from the gradient tensor */
export function dissipationDensity(G, mu) {
    let s = 0;
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { const e = 0.5 * (G[i][j] + G[j][i]); s += e * e; }
    return 2 * mu * s;
}

/**
 * Kinetic energy density ½ρ|v|² (J/m³) and viscous dissipation density (W/m³) at the points d inside the wall (same
 * points as the wall shear stress): [F, V] each plus their time means. NaN where a sample leaves the box.
 */
export function wallEnergy(field, pos, nrm, { dM = 1e-3, mu = 3.5e-3, rho = 1060 } = {}) {
    const V = pos.length / 3, F = field.nf, sp = field.spacing, ke = new Float32Array(F * V), diss = new Float32Array(F * V), v = [0, 0, 0];
    const keMean = new Float32Array(V), dissMean = new Float32Array(V);
    for (let i = 0; i < V; i++) {
        const p = [0, 1, 2].map((a) => pos[3 * i + a] - dM * nrm[3 * i + a] / sp[a]);
        let sk = 0, sd = 0, bad = false;
        for (let f = 0; f < F && !bad; f++) {
            const G = velocityGradient(field, p, f);
            if (!G || !field.sampleInto(p[0], p[1], p[2], f, v)) { bad = true; break; }
            const k = 0.5 * rho * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]), d = dissipationDensity(G, mu);
            ke[f * V + i] = k; diss[f * V + i] = d; sk += k; sd += d;
        }
        if (bad) { for (let f = 0; f < F; f++) ke[f * V + i] = diss[f * V + i] = NaN; keMean[i] = dissMean[i] = NaN; } else { keMean[i] = sk / F; dissMean[i] = sd / F; }
    }
    return { ke, diss, keMean, dissMean };
}

/**
 * Whole-lumen energy: kinetic energy (J) and viscous dissipation power (W) per frame, summed over the lumen voxels whose six
 * neighbours are all lumen (the wall-adjacent layer is excluded because velocity gradients there are dominated by the
 * segmentation edge; the dissipation is therefore a lower bound). Returns {ke_J, diss_W (Float64 [F]), nvox, volume_mm3}.
 */
export function lumenEnergy(field, lumen, { mu = 3.5e-3, rho = 1060 } = {}) {
    const [nx, ny, nz] = field.dims, F = field.nf, sp = field.spacing, dV = sp[0] * sp[1] * sp[2], nv = field.nv;
    const ke = new Float64Array(F), diss = new Float64Array(F);
    let nvox = 0;
    const idx = [];
    for (let k = 1; k < nz - 1; k++) for (let j = 1; j < ny - 1; j++) for (let i = 1; i < nx - 1; i++) {
        const q = (k * ny + j) * nx + i;
        if (lumen[q] && lumen[q - 1] && lumen[q + 1] && lumen[q - nx] && lumen[q + nx] && lumen[q - nx * ny] && lumen[q + nx * ny]) idx.push(q);
    }
    nvox = idx.length;
    for (let f = 0; f < F; f++) {
        const fr = field.frames[f], g = [0, 0, 0];
        let sk = 0, sd = 0;
        for (const q of idx) {
            const vx = fr[q], vy = fr[nv + q], vz = fr[2 * nv + q];
            sk += 0.5 * rho * (vx * vx + vy * vy + vz * vz);
            const st = [1, nx, nx * ny], h = [0.5 / sp[0], 0.5 / sp[1], 0.5 / sp[2]];
            let s2 = 0;
            const G = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
            for (let c = 0; c < 3; c++) for (let a = 0; a < 3; a++) G[c][a] = (fr[c * nv + q + st[a]] - fr[c * nv + q - st[a]]) * h[a];
            for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) { const e = 0.5 * (G[a][b] + G[b][a]); s2 += e * e; }
            sd += 2 * mu * s2;
        }
        ke[f] = sk * dV; diss[f] = sd * dV;
    }
    return { ke_J: ke, diss_W: diss, nvox, volume_mm3: nvox * dV * 1e9 };
}

// ─── whole vessel tree (display resolution) ──────────────────────────────────────────────────────────────────────

/**
 * Surface of the vessel tree connected to `seed` in a (possibly max-pooled) u8 volume: voxels ≥ thr, connected component,
 * cropped to its bounding box, blurred, iso 0.5. pos is in the volume's voxel coordinates. Returns {pos, idx, nvox} or
 * {error} when the component is empty or too big (maxVox voxels / maxTris triangles) — it never silently truncates.
 */
export function treeMesh(u8, dims, seed, thr, { maxVox = 3e7, maxTris = 3e6, passes = 1 } = {}) {
    const [nx, ny, nz] = dims, region = new Uint8Array(u8.length);
    for (let i = 0; i < u8.length; i++) region[i] = u8[i] >= thr ? 1 : 0;
    const comp = lumenComponent(region, dims, seed, 3);
    if (!comp) return { error: 'no vessel at the click above the threshold' };
    let x0 = nx, x1 = -1, y0 = ny, y1 = -1, z0 = nz, z1 = -1, nvox = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) if (comp[(k * ny + j) * nx + i]) {
        nvox++;
        if (i < x0) x0 = i; if (i > x1) x1 = i; if (j < y0) y0 = j; if (j > y1) y1 = j; if (k < z0) z0 = k; if (k > z1) z1 = k;
    }
    if (nvox > maxVox) return { error: `the connected vessel tree has ${(nvox / 1e6).toFixed(1)} M voxels (limit ${(maxVox / 1e6).toFixed(0)} M): raise the lumen threshold or the display pooling` };
    const lo = [Math.max(0, x0 - 1), Math.max(0, y0 - 1), Math.max(0, z0 - 1)], hi = [Math.min(nx - 1, x1 + 1), Math.min(ny - 1, y1 + 1), Math.min(nz - 1, z1 + 1)];
    const cd = [hi[0] - lo[0] + 1, hi[1] - lo[1] + 1, hi[2] - lo[2] + 1], crop = new Uint8Array(cd[0] * cd[1] * cd[2]);
    for (let k = 0; k < cd[2]; k++) for (let j = 0; j < cd[1]; j++) {
        const src = ((k + lo[2]) * ny + j + lo[1]) * nx + lo[0], dst = (k * cd[1] + j) * cd[0];
        for (let i = 0; i < cd[0]; i++) crop[dst + i] = comp[src + i];
    }
    const m = marchingTets(smoothIndicator(crop, cd, passes), cd, 0.5);
    if (m.idx.length / 3 > maxTris) return { error: `the vessel tree surface would have ${(m.idx.length / 3e6).toFixed(1)} M triangles (limit ${(maxTris / 1e6).toFixed(0)} M)` };
    for (let v = 0; v < m.pos.length; v += 3) { m.pos[v] += lo[0]; m.pos[v + 1] += lo[1]; m.pos[v + 2] += lo[2]; }
    return { pos: m.pos, idx: m.idx, nvox };
}
