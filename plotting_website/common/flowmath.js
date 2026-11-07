// 4D flow maths in plain JS (no DOM, no three.js): velocity field sampling, vessel regions, streamline / pathline tracing,
// plane quantification with multi-plane averaging, centerline helpers, derived scalars (vorticity, Q-criterion, …) and
// ray picking. Conventions follow rustlib/crates/hasty-server/src/flow.rs so results agree with the server commands:
//   · points are voxel index coordinates (x, y, z), voxel centres at integers, volume box [-0.5, n-0.5];
//   · velocity m/s, spacing m, dt s per frame, frames are cyclic;
//   · frame data is [3, Z, Y, X] (x fastest), one Float32Array per cardiac frame.
// Used by the flow viewer (client-side tracing / vessel tool) and by node tests (common/test_flowmath.js).

// ─── small helpers ───────────────────────────────────────────────────────────────────────────────────────────────

/** q-quantile (0..1) of the non-negative values in `arr` by refining histograms: O(n), no copy, no sort */
export function percentile(arr, q) {
    const n = arr.length;
    let lo = 0, hi = 0;
    for (let i = 0; i < n; i++) { const v = arr[i]; if (v > hi) hi = v; }
    if (!(hi > 0)) return 0;
    const NB = 2048, target = q * (n - 1);
    for (let pass = 0; pass < 3; pass++) {
        const h = new Uint32Array(NB), sc = NB / (hi - lo);
        let cum = 0;
        for (let i = 0; i < n; i++) {
            const v = arr[i];
            if (v < lo) cum++;
            else if (v < hi) h[((v - lo) * sc) | 0]++;
            else if (v === hi) h[NB - 1]++;
        }
        let b = 0;
        while (b < NB && cum + h[b] <= target) { cum += h[b]; b++; }
        if (b >= NB) return hi;
        const nlo = lo + b / sc, nhi = lo + (b + 1) / sc;
        lo = nlo; hi = nhi;
        if (pass === 2) return 0.5 * (lo + hi);
    }
    return hi;
}

export function mulberry32(seed) {
    let a = (seed >>> 0) || 1;
    return () => {
        a = (a + 0x6D2B79F5) >>> 0;
        let t = a;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.sqrt(dot(a, a));
export function unit(a) { const l = norm(a); if (!(l > 0)) throw new Error('zero vector'); return [a[0] / l, a[1] / l, a[2] / l]; }

/** orthonormal basis (u, w) of the plane ⟂ n; `uHint` (if given) fixes u's direction (parallel transport along a vessel) */
export function planeBasis(n, uHint = null) {
    let u;
    if (uHint) {
        const d = dot(uHint, n);
        const t = [uHint[0] - d * n[0], uHint[1] - d * n[1], uHint[2] - d * n[2]];
        if (norm(t) > 1e-6) u = unit(t);
    }
    if (!u) u = unit(cross(n, Math.abs(n[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]));
    return [u, cross(n, u)];
}

// ─── the field ───────────────────────────────────────────────────────────────────────────────────────────────────

export class FlowField {
    /** @param {{dims: number[], spacing: number[], dt: number, frames: Float32Array[]}} o  dims [X, Y, Z]; frames[f] = [3, Z, Y, X] */
    constructor({ dims, spacing, dt, frames }) {
        this.dims = dims;
        this.spacing = spacing;
        this.dt = dt;
        this.frames = frames;
        this.nf = frames.length;
        this.nv = dims[0] * dims[1] * dims[2];
        this._tmp = [0, 0, 0];
    }

    /** trilinear in space, linear in (cyclic) time; false outside [0, n-1]³ (like the server) */
    sampleInto(px, py, pz, t, out) {
        const [nx, ny, nz] = this.dims;
        if (!(px >= 0 && px <= nx - 1 && py >= 0 && py <= ny - 1 && pz >= 0 && pz <= nz - 1)) return false;
        const i0 = Math.min(px | 0, Math.max(nx - 2, 0)), j0 = Math.min(py | 0, Math.max(ny - 2, 0)), k0 = Math.min(pz | 0, Math.max(nz - 2, 0));
        const wx = nx === 1 ? 0 : px - i0, wy = ny === 1 ? 0 : py - j0, wz = nz === 1 ? 0 : pz - k0;
        const dx = nx > 1 ? 1 : 0, dy = ny > 1 ? nx : 0, dz = nz > 1 ? nx * ny : 0;
        const q000 = (k0 * ny + j0) * nx + i0, q100 = q000 + dx, q010 = q000 + dy, q110 = q010 + dx;
        const q001 = q000 + dz, q101 = q001 + dx, q011 = q001 + dy, q111 = q011 + dx;
        const w000 = (1 - wx) * (1 - wy) * (1 - wz), w100 = wx * (1 - wy) * (1 - wz), w010 = (1 - wx) * wy * (1 - wz), w110 = wx * wy * (1 - wz);
        const w001 = (1 - wx) * (1 - wy) * wz, w101 = wx * (1 - wy) * wz, w011 = (1 - wx) * wy * wz, w111 = wx * wy * wz;
        const nf = this.nf, nv = this.nv;
        let tc = t % nf;
        if (tc < 0) tc += nf;
        let f0 = Math.floor(tc);
        if (f0 >= nf) f0 = nf - 1;
        const f1 = (f0 + 1) % nf, wt = tc - f0;
        out[0] = out[1] = out[2] = 0;
        for (let s = 0; s < 2; s++) {
            const wf = s === 0 ? 1 - wt : wt;
            if (wf === 0) continue;
            const fr = this.frames[s === 0 ? f0 : f1];
            for (let c = 0; c < 3; c++) {
                const o = c * nv;
                out[c] += wf * (w000 * fr[o + q000] + w100 * fr[o + q100] + w010 * fr[o + q010] + w110 * fr[o + q110]
                    + w001 * fr[o + q001] + w101 * fr[o + q101] + w011 * fr[o + q011] + w111 * fr[o + q111]);
            }
        }
        return true;
    }

    sample(p, t) { const o = [0, 0, 0]; return this.sampleInto(p[0], p[1], p[2], t, o) ? o : null; }

    /** speed of frame f, [Z, Y, X] */
    speedFrame(f, out = new Float32Array(this.nv)) {
        const fr = this.frames[f], nv = this.nv;
        for (let q = 0; q < nv; q++) out[q] = Math.sqrt(fr[q] * fr[q] + fr[nv + q] * fr[nv + q] + fr[2 * nv + q] * fr[2 * nv + q]);
        return out;
    }

    /** nearest-voxel lookup in a [Z, Y, X] array; -1 outside */
    nearestIndex(px, py, pz) {
        const [nx, ny, nz] = this.dims;
        const i = Math.round(px), j = Math.round(py), k = Math.round(pz);
        if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return -1;
        return (k * ny + j) * nx + i;
    }

    /** voxel coordinates → physical (m) */
    toPhys(p) { return [p[0] * this.spacing[0], p[1] * this.spacing[1], p[2] * this.spacing[2]]; }
    toVox(p) { return [p[0] / this.spacing[0], p[1] / this.spacing[1], p[2] / this.spacing[2]]; }
}

// ─── vessel region ───────────────────────────────────────────────────────────────────────────────────────────────

/** seeding / stopping region [Z, Y, X]: the mask if given, else PC-MRA above its `quantile` */
export function regionFrom({ mask = null, pc = null, quantile = 0.97 }) {
    if (mask) { const r = new Uint8Array(mask.length); for (let i = 0; i < r.length; i++) r[i] = mask[i] ? 1 : 0; return r; }
    if (!pc) throw new Error('region: need a mask or a PC-MRA');
    const thr = percentile(pc, quantile);
    const r = new Uint8Array(pc.length);
    for (let i = 0; i < r.length; i++) r[i] = pc[i] > thr ? 1 : 0;
    return r;
}

/** 6-neighbourhood dilation by one voxel (lines may run along the wall) */
export function dilate(r, [nx, ny, nz]) {
    const d = Uint8Array.from(r);
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const q = (k * ny + j) * nx + i;
        if (r[q]) continue;
        if ((i > 0 && r[q - 1]) || (i < nx - 1 && r[q + 1]) || (j > 0 && r[q - nx]) || (j < ny - 1 && r[q + nx]) || (k > 0 && r[q - nx * ny]) || (k < nz - 1 && r[q + nx * ny])) d[q] = 1;
    }
    return d;
}

// ─── seeds ───────────────────────────────────────────────────────────────────────────────────────────────────────

/** n random points in region voxels, jittered within the voxel */
export function seedsRegion(field, region, n, seed = 1) {
    const idx = [];
    for (let q = 0; q < region.length; q++) if (region[q]) idx.push(q);
    if (!idx.length) throw new Error('seeding region is empty (lower the quantile or give a mask)');
    const rng = mulberry32(seed), [nx, ny, nz] = field.dims, out = [];
    for (let s = 0; s < n; s++) {
        const q = idx[Math.min(idx.length - 1, (rng() * idx.length) | 0)];
        const c = [q % nx, ((q / nx) | 0) % ny, (q / (nx * ny)) | 0];
        out.push([0, 1, 2].map((a) => Math.min(field.dims[a] - 1, Math.max(0, c[a] + rng() - 0.5))));
    }
    return out;
}

/** n points on a disc (centre voxel coords, physical normal, radius m), restricted to the region when given */
export function seedsDisc(field, region, center, normal, radiusM, n, seed = 1) {
    const nrm = unit(normal), [u, w] = planeBasis(nrm), rng = mulberry32(seed), out = [];
    for (let tries = 0; out.length < n && tries < 50 * n; tries++) {
        const a = 2 * rng() - 1, b = 2 * rng() - 1;
        if (a * a + b * b > 1) continue;
        const p = [0, 1, 2].map((k) => center[k] + radiusM * (a * u[k] + b * w[k]) / field.spacing[k]);
        if (region) { const q = field.nearestIndex(p[0], p[1], p[2]); if (q < 0 || !region[q]) continue; }
        out.push(p);
    }
    return out;
}

/** seeds from explicit lumen pixels of a measured plane: up to n jittered points */
export function seedsFromPlane(field, m, n, seed = 1) {
    const idx = [];
    for (let i = 0; i < m.lumen.length; i++) if (m.lumen[i]) idx.push(i);
    const rng = mulberry32(seed), out = [];
    if (!idx.length) return out;
    for (let s = 0; s < n; s++) {
        const i = idx[Math.min(idx.length - 1, (rng() * idx.length) | 0)];
        const a = (i % m.n) + rng() - m.n / 2, b = ((i / m.n) | 0) + rng() - m.n / 2;
        out.push([0, 1, 2].map((k) => m.center[k] + m.ds * (a * m.u[k] + b * m.w[k]) / field.spacing[k]));
    }
    return out;
}

// ─── tracing ─────────────────────────────────────────────────────────────────────────────────────────────────────

/** polylines {points [N*3], speed [N], time [N], offsets [L+1]} (voxel coordinates), as the server's `lines` values */
function packLines(lines) {
    let n = 0;
    for (const l of lines) n += l.sp.length;
    const points = new Float32Array(3 * n), speed = new Float32Array(n), time = new Float32Array(n), offsets = new Uint32Array(lines.length + 1);
    let o = 0;
    lines.forEach((l, i) => {
        points.set(l.pts, 3 * o); speed.set(l.sp, o); time.set(l.tm, o);
        o += l.sp.length;
        offsets[i + 1] = o;
    });
    return { points, speed, time, offsets };
}

/** concatenate several polyline sets */
export function mergeLines(list) {
    const L = list.filter((l) => l.speed.length);
    if (L.length === 1) return L[0];
    const n = L.reduce((a, l) => a + l.speed.length, 0), nl = L.reduce((a, l) => a + l.offsets.length - 1, 0);
    const points = new Float32Array(3 * n), speed = new Float32Array(n), time = new Float32Array(n), offsets = new Uint32Array(nl + 1);
    let o = 0, k = 0;
    for (const l of L) {
        points.set(l.points, 3 * o); speed.set(l.speed, o); time.set(l.time, o);
        for (let i = 1; i < l.offsets.length; i++) offsets[++k] = o + l.offsets[i];
        o += l.speed.length;
    }
    return { points, speed, time, offsets };
}

export const emptyLines = () => ({ points: new Float32Array(0), speed: new Float32Array(0), time: new Float32Array(0), offsets: new Uint32Array(1) });

/**
 * Instantaneous streamlines of frame `frame` (fractional: interpolated), traced both ways from each seed with RK4 on the
 * unit direction field, `step` voxels (arc length) per step. opts: {step = 0.5, maxSteps = 600, minSpeed = 0.02,
 * stop (Uint8 region, stop when outside)}.
 */
export function streamlines(field, frame, seeds, { step = 0.5, maxSteps = 600, minSpeed = 0.02, stop = null } = {}) {
    const sp = field.spacing, v = [0, 0, 0], minS = Math.max(minSpeed, 1e-12), res = [0, 0, 0, 0];
    const inside = (p) => !stop || ((q) => q >= 0 && stop[q] === 1)(field.nearestIndex(p[0], p[1], p[2]));
    /** unit direction in voxel space + speed → res; false if stopped */
    const dir = (x, y, z) => {
        if (!field.sampleInto(x, y, z, frame, v)) return false;
        const s = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        if (s < minS) return false;
        const a = v[0] / sp[0], b = v[1] / sp[1], c = v[2] / sp[2], l = Math.sqrt(a * a + b * b + c * c);
        res[0] = a / l; res[1] = b / l; res[2] = c / l; res[3] = s;
        return true;
    };
    const trace = (p0, sign) => {
        const out = [], h = step * sign;
        let x = p0[0], y = p0[1], z = p0[2];
        for (let it = 0; it < maxSteps; it++) {
            if (!dir(x, y, z)) break;
            if (!inside([x, y, z])) break;
            const k1x = res[0], k1y = res[1], k1z = res[2];
            out.push([x, y, z, res[3]]);
            if (!dir(x + 0.5 * h * k1x, y + 0.5 * h * k1y, z + 0.5 * h * k1z)) break;
            const k2x = res[0], k2y = res[1], k2z = res[2];
            if (!dir(x + 0.5 * h * k2x, y + 0.5 * h * k2y, z + 0.5 * h * k2z)) break;
            const k3x = res[0], k3y = res[1], k3z = res[2];
            if (!dir(x + h * k3x, y + h * k3y, z + h * k3z)) break;
            x += h / 6 * (k1x + 2 * k2x + 2 * k3x + res[0]);
            y += h / 6 * (k1y + 2 * k2y + 2 * k3y + res[1]);
            z += h / 6 * (k1z + 2 * k2z + 2 * k3z + res[2]);
        }
        return out;
    };
    const lines = [];
    for (const s of seeds) {
        const back = trace(s, -1).reverse(), fwd = trace(s, 1);
        const pts = back.concat(fwd.slice(back.length ? 1 : 0));
        if (pts.length < 2) continue;
        const P = new Float32Array(3 * pts.length), S = new Float32Array(pts.length), T = new Float32Array(pts.length).fill(frame * field.dt);
        pts.forEach((q, i) => { P[3 * i] = q[0]; P[3 * i + 1] = q[1]; P[3 * i + 2] = q[2]; S[i] = q[3]; });
        lines.push({ pts: P, sp: S, tm: T });
    }
    return packLines(lines);
}

/**
 * Pathlines: particles released at t0 (frames) from each seed, advected with RK4 through space and cyclic time for
 * `duration` frames, `step` = substeps per frame interval. opts: {step = 4, maxSteps = 2000, stop}.
 */
export function pathlines(field, t0, duration, seeds, { step = 4, maxSteps = 2000, stop = null } = {}) {
    const sp = field.spacing, dtF = field.dt, v = [0, 0, 0];
    const nsteps = Math.min(Math.max(1, Math.ceil(duration * Math.max(step, 1))), maxSteps);
    const h = duration / nsteps, hs = h * dtF;
    const inside = (x, y, z) => !stop || ((q) => q >= 0 && stop[q] === 1)(field.nearestIndex(x, y, z));
    const k = [0, 0, 0, 0];
    const vel = (x, y, z, t) => {
        if (!field.sampleInto(x, y, z, t, v)) return false;
        k[0] = v[0] / sp[0]; k[1] = v[1] / sp[1]; k[2] = v[2] / sp[2]; k[3] = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        return true;
    };
    const lines = [];
    for (const s of seeds) {
        const out = [];
        let x = s[0], y = s[1], z = s[2], t = t0;
        for (let it = 0; it <= nsteps; it++) {
            if (!vel(x, y, z, t)) break;
            if (!inside(x, y, z)) break;
            const k1x = k[0], k1y = k[1], k1z = k[2];
            out.push([x, y, z, k[3], t * dtF]);
            if (!vel(x + 0.5 * hs * k1x, y + 0.5 * hs * k1y, z + 0.5 * hs * k1z, t + h / 2)) break;
            const k2x = k[0], k2y = k[1], k2z = k[2];
            if (!vel(x + 0.5 * hs * k2x, y + 0.5 * hs * k2y, z + 0.5 * hs * k2z, t + h / 2)) break;
            const k3x = k[0], k3y = k[1], k3z = k[2];
            if (!vel(x + hs * k3x, y + hs * k3y, z + hs * k3z, t + h)) break;
            x += hs / 6 * (k1x + 2 * k2x + 2 * k3x + k[0]);
            y += hs / 6 * (k1y + 2 * k2y + 2 * k3y + k[1]);
            z += hs / 6 * (k1z + 2 * k2z + 2 * k3z + k[2]);
            t += h;
        }
        if (out.length < 2) continue;
        const P = new Float32Array(3 * out.length), S = new Float32Array(out.length), T = new Float32Array(out.length);
        out.forEach((q, i) => { P[3 * i] = q[0]; P[3 * i + 1] = q[1]; P[3 * i + 2] = q[2]; S[i] = q[3]; T[i] = q[4]; });
        lines.push({ pts: P, sp: S, tm: T });
    }
    return packLines(lines);
}

/** per-point attribute of polylines used for colouring: 'speed' | 'age' (time since line start, s) | 'line' (hash of the line index) */
export function lineAttribute(lines, kind) {
    const n = lines.speed.length, out = new Float32Array(n), L = lines.offsets.length - 1;
    if (kind === 'speed') { out.set(lines.speed); return out; }
    for (let l = 0; l < L; l++) {
        const a = lines.offsets[l], b = lines.offsets[l + 1];
        for (let i = a; i < b; i++) {
            if (kind === 'age') out[i] = lines.time[i] - lines.time[a];
            else out[i] = ((l * 2654435761) >>> 0) / 4294967296;
        }
    }
    return out;
}

// ─── planes: quantification, lumen, multi-plane averaging ────────────────────────────────────────────────────────

/**
 * Flow through one plane. center: voxel coords, normal: physical, radiusM: disc radius, n: grid size.
 * region: optional [Z, Y, X] vessel mask; the lumen is then the connected component of region pixels containing (or
 * nearest to) the plane centre, so neighbouring vessels in the disc are excluded.
 * sign: +1 / -1 flips the normal (orient with the flow). Returns the through-plane velocity image per frame, the lumen,
 * Q(t) (m³/s) and summary values. Everything is in the plane's own pixel grid: p = center + ds ((ix + .5 − n/2) u + (iy + .5 − n/2) w).
 */
export function measurePlane(field, { center, normal, u: uHint = null, radiusM = 0.018, n = 64, region = null, sign = 1 }) {
    const nrm0 = unit(normal), nrm = [sign * nrm0[0], sign * nrm0[1], sign * nrm0[2]];
    const [u, w] = planeBasis(nrm, uHint);
    const ds = 2 * radiusM / n, sp = field.spacing, F = field.nf, nn = n * n;
    const pos = new Float32Array(3 * nn), cand = new Uint8Array(nn), v = [0, 0, 0];
    for (let i = 0; i < nn; i++) {
        const a = (i % n) + 0.5 - n / 2, b = ((i / n) | 0) + 0.5 - n / 2;
        if ((a * a + b * b) * ds * ds > radiusM * radiusM) continue;
        const x = center[0] + ds * (a * u[0] + b * w[0]) / sp[0], y = center[1] + ds * (a * u[1] + b * w[1]) / sp[1], z = center[2] + ds * (a * u[2] + b * w[2]) / sp[2];
        pos[3 * i] = x; pos[3 * i + 1] = y; pos[3 * i + 2] = z;
        if (!field.sampleInto(x, y, z, 0, v)) continue;
        if (region) { const q = field.nearestIndex(x, y, z); if (q < 0 || !region[q]) continue; }
        cand[i] = 1;
    }
    // lumen = 4-connected component of the candidates containing the pixel nearest the centre
    let lumen = cand;
    if (region) {
        lumen = new Uint8Array(nn);
        let seed = -1, best = Infinity;
        for (let i = 0; i < nn; i++) if (cand[i]) { const a = (i % n) + 0.5 - n / 2, b = ((i / n) | 0) + 0.5 - n / 2, d = a * a + b * b; if (d < best) { best = d; seed = i; } }
        if (seed >= 0 && best <= 36) {                   // the click must be near the lumen (≤ 6 px)
            const stack = [seed];
            lumen[seed] = 1;
            while (stack.length) {
                const i = stack.pop(), x = i % n;
                for (const j of [x > 0 ? i - 1 : -1, x < n - 1 ? i + 1 : -1, i >= n ? i - n : -1, i < nn - n ? i + n : -1]) if (j >= 0 && cand[j] && !lumen[j]) { lumen[j] = 1; stack.push(j); }
            }
        }
    }
    let na = 0, cx = 0, cy = 0;
    for (let i = 0; i < nn; i++) if (lumen[i]) { na++; cx += i % n; cy += (i / n) | 0; }
    const img = new Float32Array(F * nn), q = new Float64Array(F), vmean = new Float64Array(F), vpeak = new Float64Array(F), dA = ds * ds;
    for (let f = 0; f < F; f++) {
        for (let i = 0; i < nn; i++) {
            if (!lumen[i]) continue;
            field.sampleInto(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2], f, v);
            const vn = v[0] * nrm[0] + v[1] * nrm[1] + v[2] * nrm[2];
            img[f * nn + i] = vn;
            q[f] += vn * dA;
            if (Math.abs(vn) > Math.abs(vpeak[f])) vpeak[f] = vn;
        }
        vmean[f] = na ? q[f] / (na * dA) : 0;
    }
    const centroid = na ? (() => {              // lumen centroid back in voxel coords
        const a = cx / na + 0.5 - n / 2, b = cy / na + 0.5 - n / 2;
        return [0, 1, 2].map((k) => center[k] + ds * (a * u[k] + b * w[k]) / sp[k]);
    })() : null;
    return { n, ds, center: Array.from(center), normal: nrm, u, w, lumen, nlumen: na, area_mm2: na * dA * 1e6, img, q, vmean, vpeak, centroid, ...flowStats(q, field.dt) };
}

/** forward / backward / net volume (ml) over the cycle and regurgitant fraction from Q(t) in m³/s */
export function flowStats(q, dt) {
    let fwd = 0, bwd = 0, net = 0;
    for (const x of q) { if (x >= 0) fwd += x * dt; else bwd -= x * dt; net += x * dt; }
    const ml = 1e6;
    return { forward_ml: fwd * ml, backward_ml: bwd * ml, net_ml: net * ml, regurgitant_fraction: fwd > 0 ? bwd / fwd : 0, flow_ml_s: Array.from(q, (x) => x * ml) };
}

/**
 * Average over a stack of planes (same grid size, parallel-transported bases so pixel (ix, iy) is the same offset in each
 * cross-section): mean Q(t) with min / max band, mean through-plane velocity image, and a majority lumen.
 */
export function averageStack(field, planes) {
    const P = planes.length, n = planes[0].n, nn = n * n, F = field.nf;
    const q = new Float64Array(F), qmin = new Float64Array(F).fill(Infinity), qmax = new Float64Array(F).fill(-Infinity);
    const img = new Float32Array(F * nn), cnt = new Uint16Array(nn);
    for (const m of planes) {
        for (let f = 0; f < F; f++) { q[f] += m.q[f] / P; qmin[f] = Math.min(qmin[f], m.q[f]); qmax[f] = Math.max(qmax[f], m.q[f]); }
        for (let i = 0; i < nn; i++) if (m.lumen[i]) cnt[i]++;
        for (let k = 0; k < F * nn; k++) img[k] += m.img[k];
    }
    const lumen = new Uint8Array(nn);
    for (let i = 0; i < nn; i++) {
        lumen[i] = cnt[i] * 2 >= P ? 1 : 0;
        if (cnt[i]) for (let f = 0; f < F; f++) img[f * nn + i] /= cnt[i];
    }
    for (let i = 0; i < nn; i++) if (!lumen[i]) for (let f = 0; f < F; f++) img[f * nn + i] = 0;
    const mid = planes[(P - 1) >> 1];
    let na = 0;
    for (let i = 0; i < nn; i++) na += lumen[i];
    const vpeak = new Float64Array(F), vmean = new Float64Array(F);
    for (let f = 0; f < F; f++) {
        let s = 0;
        for (let i = 0; i < nn; i++) if (lumen[i]) { const x = img[f * nn + i]; s += x; if (Math.abs(x) > Math.abs(vpeak[f])) vpeak[f] = x; }
        vmean[f] = na ? s / na : 0;
    }
    return {
        planes: P, n, ds: mid.ds, center: mid.center, normal: mid.normal, u: mid.u, w: mid.w, lumen, nlumen: na, area_mm2: planes.reduce((a, m) => a + m.area_mm2, 0) / P,
        img, q, qmin, qmax, vmean, vpeak, ...flowStats(q, field.dt), band_ml_s: { min: Array.from(qmin, (x) => x * 1e6), max: Array.from(qmax, (x) => x * 1e6) },
    };
}

/** time-mean velocity direction (unit, physical) averaged over a plane's lumen; null if there is no flow */
export function meanFlowDirection(field, m) {
    const v = [0, 0, 0], acc = [0, 0, 0], sp = field.spacing, n = m.n;
    for (let i = 0; i < n * n; i++) {
        if (!m.lumen[i]) continue;
        const a = (i % n) + 0.5 - n / 2, b = ((i / n) | 0) + 0.5 - n / 2;
        const x = m.center[0] + m.ds * (a * m.u[0] + b * m.w[0]) / sp[0], y = m.center[1] + m.ds * (a * m.u[1] + b * m.w[1]) / sp[1], z = m.center[2] + m.ds * (a * m.u[2] + b * m.w[2]) / sp[2];
        for (let f = 0; f < field.nf; f++) if (field.sampleInto(x, y, z, f, v)) { acc[0] += v[0]; acc[1] += v[1]; acc[2] += v[2]; }
    }
    return norm(acc) > 0 ? unit(acc) : null;
}

/**
 * Plane stack for the vessel tool without centerline data: starting at `p0` the vessel direction is the time-mean flow
 * direction; the walk steps `stepM` along it, re-centres on the lumen centroid of each plane and re-estimates the
 * direction from that plane. Returns planes (upstream → downstream) with normals along the flow.
 */
export function walkStack(field, region, p0, { nPlanes = 5, stepM = 0.002, radiusM = 0.018, n = 64 } = {}) {
    const sp = field.spacing;
    let first = measurePlane(field, { center: p0, normal: [0, 0, 1], radiusM, n, region });
    // direction from a small isotropic probe: the time-mean velocity of the lumen of a disc ⟂ z, refined once ⟂ to the result
    let d = meanFlowDirection(field, first);
    if (!d) return null;
    let m0 = measurePlane(field, { center: first.centroid ?? p0, normal: d, radiusM, n, region });
    d = meanFlowDirection(field, m0) ?? d;
    m0 = measurePlane(field, { center: m0.centroid ?? p0, normal: d, radiusM, n, region });
    if (!m0.nlumen) return null;
    const half = (nPlanes - 1) / 2, nUp = Math.floor(half), nDown = Math.ceil(half);
    const chain = (dirSign, count) => {
        const out = [];
        let prev = m0, dd = d;
        for (let k = 0; k < count; k++) {
            const c = [0, 1, 2].map((a) => prev.center[a] + dirSign * stepM * dd[a] / sp[a]);
            let m = measurePlane(field, { center: c, normal: dd, u: prev.u, radiusM, n, region });
            if (!m.nlumen) break;
            m = measurePlane(field, { center: m.centroid, normal: dd, u: prev.u, radiusM, n, region });   // re-centre once
            if (!m.nlumen) break;
            const d2 = meanFlowDirection(field, m);
            if (d2 && dot(d2, dd) > 0.5) { dd = unit([dd[0] + d2[0], dd[1] + d2[1], dd[2] + d2[2]]); }  // smooth direction update
            out.push(m);
            prev = m;
        }
        return out;
    };
    return [...chain(-1, nUp).reverse(), m0, ...chain(1, nDown)];
}

// ─── centerlines ─────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * Centerlines as polylines in voxel coordinates ({points [N*3], offsets [L+1]}, optional radius [N] in voxels or mm —
 * see CENTERLINES.md). Arc length is physical (mm).
 */
export class Centerlines {
    constructor({ points, offsets, radius = null }, spacing) {
        this.points = points;
        this.offsets = offsets;
        this.radius = radius;
        this.spacing = spacing;
        this.nLines = offsets.length - 1;
        this.arc = new Float64Array(points.length / 3);               // cumulative arc length per point (mm), restarting per line
        for (let l = 0; l < this.nLines; l++) {
            let s = 0;
            for (let i = offsets[l]; i < offsets[l + 1]; i++) {
                if (i > offsets[l]) {
                    const dx = (points[3 * i] - points[3 * i - 3]) * spacing[0], dy = (points[3 * i + 1] - points[3 * i - 2]) * spacing[1], dz = (points[3 * i + 2] - points[3 * i - 1]) * spacing[2];
                    s += 1e3 * Math.sqrt(dx * dx + dy * dy + dz * dz);
                }
                this.arc[i] = s;
            }
        }
    }

    /** nearest centerline point to voxel position p: {line, i (global point index), dist_mm} */
    nearest(p) {
        let best = Infinity, bi = -1;
        const P = this.points, sp = this.spacing;
        for (let i = 0; i < P.length / 3; i++) {
            const dx = (P[3 * i] - p[0]) * sp[0], dy = (P[3 * i + 1] - p[1]) * sp[1], dz = (P[3 * i + 2] - p[2]) * sp[2];
            const d = dx * dx + dy * dy + dz * dz;
            if (d < best) { best = d; bi = i; }
        }
        if (bi < 0) return null;
        let line = 0;
        while (this.offsets[line + 1] <= bi) line++;
        return { line, i: bi, dist_mm: 1e3 * Math.sqrt(best) };
    }

    /** position (voxel coords) and unit tangent (physical, along increasing arc length) at arc length s (mm) on a line */
    frameAt(line, s) {
        const a = this.offsets[line], b = this.offsets[line + 1] - 1;
        const total = this.arc[b];
        const at = (ss) => {
            ss = Math.min(Math.max(ss, 0), total);
            let lo = a, hi = b;
            while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (this.arc[mid] <= ss) lo = mid; else hi = mid; }
            const span = this.arc[hi] - this.arc[lo], t = span > 0 ? (ss - this.arc[lo]) / span : 0;
            return [0, 1, 2].map((k) => this.points[3 * lo + k] + t * (this.points[3 * hi + k] - this.points[3 * lo + k]));
        };
        const w = 2;                                                  // tangent from ±2 mm
        const p0 = at(s - w), p1 = at(s + w), sp = this.spacing;
        const t = [(p1[0] - p0[0]) * sp[0], (p1[1] - p0[1]) * sp[1], (p1[2] - p0[2]) * sp[2]];
        return { p: at(s), t: norm(t) > 0 ? unit(t) : [0, 0, 1], total_mm: total };
    }

    /** `nPlanes` plane definitions centred on the point of the centerline nearest to p, every `stepMm` along the line */
    stack(p, nPlanes, stepMm) {
        const nb = this.nearest(p);
        if (!nb) return null;
        const s0 = this.arc[nb.i], out = [];
        for (let k = 0; k < nPlanes; k++) {
            const s = s0 + (k - (nPlanes - 1) / 2) * stepMm;
            const f = this.frameAt(nb.line, s);
            out.push({ center: f.p, normal: f.t });
        }
        return { planes: out, line: nb.line, dist_mm: nb.dist_mm };
    }
}

/** measure a stack of plane definitions (parallel-transported bases, orientation by the centre plane's net flow) */
export function measureStack(field, defs, { radiusM = 0.018, n = 64, region = null, orient = true } = {}) {
    const mid = defs[(defs.length - 1) >> 1];
    const probe = measurePlane(field, { center: mid.center, normal: mid.normal, radiusM, n, region });
    const sign = orient && probe.net_ml < 0 ? -1 : 1;
    const out = [];
    let uHint = null;
    for (const d of defs) {
        const m = measurePlane(field, { center: d.center, normal: d.normal, u: uHint, radiusM, n, region, sign });
        uHint = m.u;
        out.push(m);
    }
    return out;
}

// ─── derived scalar volumes ──────────────────────────────────────────────────────────────────────────────────────

/**
 * Scalar volume [Z, Y, X] of frame f. kind: 'speed' (m/s), 'vorticity' (|∇×v|, 1/s), 'q' (Q-criterion ½(|Ω|²−|S|²), 1/s², clamped ≥ 0),
 * 'helicity' (|v·ω|, m/s²). Central differences (one-sided at the borders), physical spacing.
 */
export function derivedScalar(field, f, kind, out = new Float32Array(field.nv)) {
    if (kind === 'speed') return field.speedFrame(f, out);
    const [nx, ny, nz] = field.dims, sp = field.spacing, fr = field.frames[f], nv = field.nv;
    const sx = nx * 1, sy = nx, sz = nx * ny;
    const grad = (c, q, i, j, k) => {
        const o = c * nv;
        const gx = nx === 1 ? 0 : (fr[o + q + (i < nx - 1 ? 1 : 0)] - fr[o + q - (i > 0 ? 1 : 0)]) / (((i < nx - 1) + (i > 0)) * sp[0]);
        const gy = ny === 1 ? 0 : (fr[o + q + (j < ny - 1 ? sy : 0)] - fr[o + q - (j > 0 ? sy : 0)]) / (((j < ny - 1) + (j > 0)) * sp[1]);
        const gz = nz === 1 ? 0 : (fr[o + q + (k < nz - 1 ? sz : 0)] - fr[o + q - (k > 0 ? sz : 0)]) / (((k < nz - 1) + (k > 0)) * sp[2]);
        return [gx, gy, gz];
    };
    void sx;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
        const q = (k * ny + j) * nx + i;
        const gu = grad(0, q, i, j, k), gv = grad(1, q, i, j, k), gw = grad(2, q, i, j, k);   // rows: ∇u, ∇v, ∇w
        const wx = gw[1] - gv[2], wy = gu[2] - gw[0], wz = gv[0] - gu[1];                       // ω = ∇×v
        if (kind === 'vorticity') out[q] = Math.sqrt(wx * wx + wy * wy + wz * wz);
        else if (kind === 'helicity') out[q] = Math.abs(fr[q] * wx + fr[nv + q] * wy + fr[2 * nv + q] * wz);
        else if (kind === 'q') {
            const J = [gu, gv, gw];
            let s2 = 0, o2 = 0;
            for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
                const S = 0.5 * (J[a][b] + J[b][a]), O = 0.5 * (J[a][b] - J[b][a]);
                s2 += S * S; o2 += O * O;
            }
            out[q] = Math.max(0, 0.5 * (o2 - s2));
        } else throw new Error(`derivedScalar: unknown kind ${kind}`);
    }
    return out;
}

// ─── ray picking ─────────────────────────────────────────────────────────────────────────────────────────────────

/**
 * March a ray through a scalar volume `vol` [Z, Y, X] (dims [X, Y, Z]). Ray in voxel coordinates. Returns the first sample
 * ≥ `level`, refined to the local maximum within `refine` voxels along the ray (the lumen centre rather than its rim),
 * or null. mode 'max' returns the global maximum along the ray instead (what a MIP shows).
 */
export function pickRay(vol, dims, origin, dir, { level, mode = 'first', refine = 4, stepVox = 0.5 } = {}) {
    const [nx, ny, nz] = dims;
    const lo = [-0.5, -0.5, -0.5], hi = [nx - 0.5, ny - 0.5, nz - 0.5];
    let t0 = 0, t1 = Infinity;
    for (let a = 0; a < 3; a++) {
        if (Math.abs(dir[a]) < 1e-12) { if (origin[a] < lo[a] || origin[a] > hi[a]) return null; continue; }
        let ta = (lo[a] - origin[a]) / dir[a], tb = (hi[a] - origin[a]) / dir[a];
        if (ta > tb) [ta, tb] = [tb, ta];
        t0 = Math.max(t0, ta); t1 = Math.min(t1, tb);
    }
    if (!(t0 < t1)) return null;
    const dl = norm(dir), d = [dir[0] / dl, dir[1] / dl, dir[2] / dl];
    t0 *= dl; t1 *= dl;                                  // parameter of the unit direction: origin + dir·ta = origin + d·(ta·|dir|)
    const val = (t) => {
        const x = Math.min(nx - 1, Math.max(0, origin[0] / 1 + d[0] * t)), y = Math.min(ny - 1, Math.max(0, origin[1] + d[1] * t)), z = Math.min(nz - 1, Math.max(0, origin[2] + d[2] * t));
        const i0 = Math.min(x | 0, Math.max(nx - 2, 0)), j0 = Math.min(y | 0, Math.max(ny - 2, 0)), k0 = Math.min(z | 0, Math.max(nz - 2, 0));
        const wx = nx === 1 ? 0 : x - i0, wy = ny === 1 ? 0 : y - j0, wz = nz === 1 ? 0 : z - k0;
        const dx = nx > 1 ? 1 : 0, dy = ny > 1 ? nx : 0, dz = nz > 1 ? nx * ny : 0, q = (k0 * ny + j0) * nx + i0;
        const c00 = vol[q] * (1 - wx) + vol[q + dx] * wx, c10 = vol[q + dy] * (1 - wx) + vol[q + dy + dx] * wx;
        const c01 = vol[q + dz] * (1 - wx) + vol[q + dz + dx] * wx, c11 = vol[q + dz + dy] * (1 - wx) + vol[q + dz + dy + dx] * wx;
        return (c00 * (1 - wy) + c10 * wy) * (1 - wz) + (c01 * (1 - wy) + c11 * wy) * wz;
    };
    let tBest = -1, vBest = -Infinity;
    for (let t = t0; t <= t1; t += stepVox) {
        const v = val(t);
        if (mode === 'max') { if (v > vBest) { vBest = v; tBest = t; } continue; }
        if (v >= level) {
            tBest = t; vBest = v;
            for (let r = stepVox; r <= refine; r += stepVox) { const v2 = val(t + r); if (v2 > vBest) { vBest = v2; tBest = t + r; } }
            break;
        }
    }
    if (tBest < 0 || vBest < level) return null;
    return { p: [origin[0] + d[0] * tBest, origin[1] + d[1] * tBest, origin[2] + d[2] * tBest], value: vBest, t: tBest };
}
