// Vessel tool: click a vessel in the 3D view → a ray is marched through the PC-MRA to find the lumen, a stack of
// analysis planes is placed along the vessel (⟂ to the nearest centerline when the flow value has centerlines, else by
// following the time-mean flow from the click), each plane's lumen is the connected region around the click, and the
// through-plane flow is averaged over the N planes (noise ↓ ~ √N). All maths is in common/flowmath.js and runs here, in
// the browser, on a full-resolution box around the click (common/roi.js) — the whole velocity volume is never needed.
import * as THREE from 'three';
import { averageStack, measureStack, pickRay, walkStack, seedsFromPlane } from '../../common/flowmath.js';
import { PlaneDisc, lumenContour, segments } from './layers.js';
import { fromPooled, toPooled } from '../../common/pool.js';
import { roiBox, roiBytes, roiContains, shiftResult } from '../../common/roi.js';
import { finiteQuantile, lumenComponent, lumenEnergy, marchingTets, smoothIndicator, treeMesh, wallEnergy, wallNormals, wallShear } from '../../common/surface.js';

const CYAN = 0x4fc3f7, YELLOW = 0xffd23f;

export class VesselTool {
    /**
     * @param {object} o  scene, spacingMm [dx,dy,dz], divArr (256×RGBA coolwarm), invalidate(),
     *   vol: {u8, dims, factor, scale}  the (possibly max-pooled) u8 PC-MRA the clicks are traced through,
     *   info: {dims, spacing (m), frames},  fetchRoi(box) → Promise<{field, mask, pc, lo, hi}>  full-resolution box,
     *   getCenterlines() (Centerlines|null), maxRoiBytes
     */
    constructor(o) {
        Object.assign(this, o);
        this.group = new THREE.Group();
        o.scene.add(this.group);
        this.result = null;
        this.disc = null;
        this.roi = null;                  // last fetched box (re-used while clicks stay inside it)
        this.gen = 0;
        this.wall = null;
        this.tree = null;
    }

    // ── wall surface ───────────────────────────────────────────────────────────────────────────────────────────────
    /**
     * Lumen surface (iso 0.5 of the blurred lumen component around the click) with wall-shear quantities, for the last
     * measurement. opts: {dMm (sampling distance inside the wall), mu (Pa·s), order 1|2, lut (texture)}. Computation is
     * synchronous; returns {vertices, triangles, wss: [p2, p98] of time-mean WSS (Pa)} or null if there is no lumen.
     */
    buildWall(res, { dMm, mu = 3.5e-3, order = 2, rho = 1060, lut }) {
        this.clearWall();
        const b = res?.box;
        if (!b) return null;
        const comp = lumenComponent(b.region, b.field.dims, b.p);
        if (!comp) return null;
        const ind = smoothIndicator(comp, b.field.dims, 2), mesh = marchingTets(ind, b.field.dims, 0.5);
        if (!mesh.idx.length) return null;
        const sp = b.field.spacing, nrm = wallNormals(ind, b.field.dims, sp, mesh.pos), V = mesh.pos.length / 3;
        const q = wallShear(b.field, mesh.pos, nrm, { dM: dMm * 1e-3, mu, order });
        const world = new Float32Array(3 * V);
        for (let v = 0; v < V; v++) for (let a = 0; a < 3; a++) world[3 * v + a] = (mesh.pos[3 * v + a] + b.lo[a]) * this.spacingMm[a];
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(world, 3));
        g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3));
        const val = new Float32Array(V).fill(-1);
        g.setAttribute('val', new THREE.BufferAttribute(val, 1));
        g.setIndex(new THREE.BufferAttribute(mesh.idx, 1));
        const uniforms = { lut: { value: lut }, vmax: { value: 1 }, opacity: { value: 0.7 } };
        const m = new THREE.Mesh(g, new THREE.ShaderMaterial({
            uniforms, transparent: true, depthWrite: false, side: THREE.DoubleSide,
            vertexShader: 'attribute float val; varying float vVal; varying vec3 vN; void main() { vVal = val; vN = normalMatrix * normal; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
            fragmentShader: 'uniform sampler2D lut; uniform float vmax; uniform float opacity; varying float vVal; varying vec3 vN; void main() { float sh = 0.35 + 0.65 * abs(normalize(vN).z); vec3 c = vVal < 0.0 ? vec3(0.62) : texture2D(lut, vec2(clamp(vVal / vmax, 0.0, 1.0), 0.5)).rgb; gl_FragColor = vec4(c * sh, opacity); }',
        }));
        m.renderOrder = 1;
        this.group.add(m);
        this.wall = { mesh: m, uniforms, val, V, q, en: null, lum: null, F: b.field.nf, b, comp, nrm, pos: mesh.pos, dM: dMm * 1e-3, mu, rho };
        const p98 = (a) => finiteQuantile(a, 0.98);
        this.wall.p2 = finiteQuantile(q.tmean, 0.02); this.wall.p98 = p98(q.tmean);
        this.wall.auto = { twss: p98(q.tmean), wss: p98(q.mag), osi: 0.5, none: 1, ke: 0, ke_f: 0, diss: 0, diss_f: 0 };
        this.invalidate();
        return { vertices: V, triangles: mesh.idx.length / 3, wss: [this.wall.p2, this.wall.p98], osi: [finiteQuantile(q.osi, 0.02), finiteQuantile(q.osi, 0.98)], nan: q.tmean.reduce((a, x) => a + (Number.isFinite(x) ? 0 : 1), 0) };
    }

    /** energy quantities are computed on first use (they cost ~2 s at 0.5 mm): per-vertex kinetic energy / dissipation and the whole-lumen totals */
    ensureEnergy() {
        const w = this.wall;
        if (!w || w.en) return w?.lum ?? null;
        w.en = wallEnergy(w.b.field, w.pos, w.nrm, { dM: w.dM, mu: w.mu, rho: w.rho });
        w.lum = lumenEnergy(w.b.field, w.comp, { mu: w.mu, rho: w.rho });
        const p98 = (a) => finiteQuantile(a, 0.98);
        Object.assign(w.auto, { ke: p98(w.en.keMean), ke_f: p98(w.en.ke), diss: p98(w.en.dissMean), diss_f: p98(w.en.diss) });
        return w.lum;
    }

    /** colour the wall: by 'none' | 'twss' | 'wss' (frame) | 'osi' | 'ke' | 'ke_f' | 'diss' | 'diss_f' (_f = this frame); max = colour-scale end (same unit as the quantity) */
    setWallColour(by, frame, max, opacity) {
        const w = this.wall;
        if (!w) return;
        const f = Math.min(frame, w.F - 1), at = (a) => a.subarray(f * w.V, (f + 1) * w.V);
        if (/^(ke|diss)/.test(by)) this.ensureEnergy();
        const src = { twss: w.q.tmean, osi: w.q.osi, wss: at(w.q.mag), ke: w.en?.keMean, ke_f: w.en && at(w.en.ke), diss: w.en?.dissMean, diss_f: w.en && at(w.en.diss) }[by] ?? null;
        for (let v = 0; v < w.V; v++) w.val[v] = src && Number.isFinite(src[v]) ? src[v] : -1;
        w.mesh.geometry.attributes.val.needsUpdate = true;
        w.uniforms.vmax.value = Math.max(1e-9, max);
        w.uniforms.opacity.value = opacity;
        this.invalidate();
    }

    /** the whole vessel tree connected to the click, at the resolution of the display volume (max-pooled when ?mip>1, so vessels can look up to
     *  one pooled voxel wider). frac × the value at the click is the lumen level, as for the measurement. Returns {triangles, nvox, factor} or {error}. */
    buildTree(hit, frac, opacity = 0.25) {
        this.clearTree();
        const { u8, dims, factor: k, scale } = this.vol;
        const thr = Math.max(1, Math.round(frac * (hit.value / scale) * 255));
        const m = treeMesh(u8, dims, toPooled(hit.p, k), thr);
        if (m.error) return { error: m.error };
        const V = m.pos.length / 3, world = new Float32Array(3 * V);
        for (let v = 0; v < V; v++) { const p = fromPooled([m.pos[3 * v], m.pos[3 * v + 1], m.pos[3 * v + 2]], k); for (let a = 0; a < 3; a++) world[3 * v + a] = p[a] * this.spacingMm[a]; }
        const g = new THREE.BufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(world, 3));
        g.setIndex(new THREE.BufferAttribute(m.idx, 1));
        g.computeVertexNormals();
        const mat = new THREE.ShaderMaterial({
            uniforms: { opacity: { value: opacity } }, transparent: true, depthWrite: false, side: THREE.DoubleSide,
            vertexShader: 'varying vec3 vN; void main() { vN = normalMatrix * normal; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
            fragmentShader: 'uniform float opacity; varying vec3 vN; void main() { float sh = 0.35 + 0.65 * abs(normalize(vN).z); gl_FragColor = vec4(vec3(0.75, 0.8, 0.9) * sh, opacity); }',
        });
        this.tree = new THREE.Mesh(g, mat);
        this.tree.renderOrder = 0;
        this.group.add(this.tree);
        this.invalidate();
        return { triangles: m.idx.length / 3, nvox: m.nvox, factor: k, thr };
    }
    setTreeOpacity(o) { if (this.tree) { this.tree.material.uniforms.opacity.value = o; this.invalidate(); } }
    clearTree() {
        if (!this.tree) return;
        this.group.remove(this.tree);
        this.tree.geometry.dispose(); this.tree.material.dispose();
        this.tree = null;
    }

    clearWall() {
        if (!this.wall) return;
        this.group.remove(this.wall.mesh);
        this.wall.mesh.geometry.dispose(); this.wall.mesh.material.dispose();
        this.wall = null;
    }

    /** voxel-space ray → hit {p (full-resolution voxel coords), value (PC-MRA units)} on the PC-MRA (first sample ≥ level, refined to the local maximum), or null.
     *  level is in PC-MRA units. The MIP volume may be max-pooled; the hit is then refined again on the full-resolution box in measure(). */
    pick(originVox, dirVox, level) {
        const { u8, dims, factor: k, scale } = this.vol;
        const o = toPooled(originVox, k), d = dirVox.map((c) => c / k);
        const h = pickRay(u8, dims, o, d, { level: (level / scale) * 255, mode: 'first', refine: 4 });
        return h ? { p: fromPooled(h.p, k), value: (h.value / 255) * scale } : null;
    }

    /** reach (m) of the box a measurement needs around the click */
    reachM(o) {
        const vox = Math.max(...this.info.spacing);
        return o.radiusMm * 1e-3 + Math.ceil(o.nPlanes / 2) * o.stepMm * 1e-3 + 3 * vox;
    }

    /**
     * Measure at a PC-MRA hit. opts: {nPlanes, stepMm, radiusMm, frac, orient}. Fetches the full-resolution box around the
     * click (cached), measures in it and maps the planes back to volume coordinates.
     * Returns {planes, avg, source, hit, opts} or null (no lumen / no flow); null also if a newer measure() superseded this one.
     */
    async measure(hit0, { nPlanes = 5, stepMm = 1.5, radiusMm = 18, frac = 0.4, orient = true } = {}) {
        const gen = ++this.gen, dims = this.info.dims, spacing = this.info.spacing, o = { nPlanes, stepMm, radiusMm };
        const box = roiBox(hit0.p, dims, spacing, this.reachM(o));
        if (!(this.roi && roiContains(this.roi, box))) {
            const need = roiBytes(box, this.info.frames);
            if (need > this.maxRoiBytes) throw new Error(`the region around the click needs ${(need / 1048576).toFixed(0)} MB (limit ${(this.maxRoiBytes / 1048576).toFixed(0)}); lower the radius / planes or raise ?roimb=`);
            const r = await this.fetchRoi(box);
            if (gen !== this.gen) return undefined;
            this.roi = r;
        }
        const roi = this.roi, lo = roi.lo, field = roi.field, [nx, ny, nz] = field.dims;
        let p = hit0.p.map((c, k) => c - lo[k]);
        // refine on the full-resolution data: snap to the strongest voxel within the pooling cell (only when the MIP volume is pooled)
        const w = this.vol.factor > 1 ? this.vol.factor : 0;
        const i0 = Math.round(p[0]), j0 = Math.round(p[1]), k0 = Math.round(p[2]), at = (x, y, z) => roi.pc[(z * ny + y) * nx + x];
        let value = at(Math.min(nx - 1, Math.max(0, i0)), Math.min(ny - 1, Math.max(0, j0)), Math.min(nz - 1, Math.max(0, k0)));
        if (w) {
            value = -1;
            for (let z = Math.max(0, k0 - w); z <= Math.min(nz - 1, k0 + w); z++) for (let y = Math.max(0, j0 - w); y <= Math.min(ny - 1, j0 + w); y++) for (let x = Math.max(0, i0 - w); x <= Math.min(nx - 1, i0 + w); x++) {
                const v = at(x, y, z);
                if (v > value) { value = v; p = [x, y, z]; }
            }
        }
        if (!(value > 0)) return null;
        const thr = frac * value, region = roi.mask ?? roi.pc.map((v) => (v >= thr ? 1 : 0));
        const radiusM = radiusMm * 1e-3;
        const n = Math.max(24, Math.min(80, 2 * Math.round(radiusMm / Math.min(...spacing.map((s) => s * 1e3)) * 2)));   // pixel ≈ ½ voxel
        const cl = this.getCenterlines?.();
        let planes = null, source = 'flow';
        if (cl) {
            const st = cl.stack(p.map((c, k) => c + lo[k]), nPlanes, stepMm);
            if (st && st.dist_mm < Math.max(8, radiusMm * 0.6)) {
                planes = measureStack(field, st.planes.map((d) => ({ ...d, center: d.center.map((c, k) => c - lo[k]) })), { radiusM, n, region, orient });
                source = 'centerline';
            }
        }
        if (!planes) planes = walkStack(field, region, p, { nPlanes, stepM: stepMm * 1e-3, radiusM, n });
        planes = (planes ?? []).filter((m) => m.nlumen > 0);
        if (!planes.length) return null;
        const res = shiftResult({ planes, avg: averageStack(field, planes), source, opts: { nPlanes, stepMm, radiusMm, frac } }, lo);
        res.hit = { p: p.map((c, k) => c + lo[k]), value };
        res.box = { field, region, lo, p };       // box-local data the wall surface is built from
        return res;
    }

    /** seeds on the averaged lumen of the centre plane */
    seeds(res, n, seed = 5) {
        const mid = res.planes[(res.planes.length - 1) >> 1];
        return seedsFromPlane({ spacing: this.info.spacing }, { ...mid, lumen: res.avg.lumen }, n, seed);
    }

    /** draw the measurement: lumen outline of every plane (the region being averaged), the through-plane velocity on the
     *  centre plane, and the path of the plane centres */
    show(res, frame, vmax) {
        this.clear();
        this.result = res;
        const sp = this.spacingMm;
        res.planes.forEach((m, k) => {
            const mid = k === (res.planes.length - 1) >> 1;
            this.group.add(segments(lumenContour(m, sp), mid ? YELLOW : CYAN, { opacity: mid ? 1 : 0.55 }));
        });
        const path = new Float32Array(3 * res.planes.length);
        res.planes.forEach((m, k) => { for (let c = 0; c < 3; c++) path[3 * k + c] = m.center[c] * sp[c]; });
        if (res.planes.length > 1) {
            const g = new THREE.BufferGeometry();
            g.setAttribute('position', new THREE.BufferAttribute(path, 3));
            const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color: YELLOW, depthTest: false }));
            l.renderOrder = 3;
            this.group.add(l);
        }
        const a = res.avg;
        this.disc = new PlaneDisc({ center: a.center, normal: a.normal, u: a.u, w: a.w, grid: a.n, pixel_mm: a.ds * 1e3 }, a.img, a.lumen, sp, this.divArr);
        this.group.add(this.disc.group);
        this.disc.frame(frame, vmax);
        this.invalidate();
    }

    setFrame(frame, vmax) { if (this.disc) { this.disc.frame(frame, vmax); this.invalidate(); } }

    clear(keepWall = false) {
        if (!keepWall) { this.clearWall(); this.clearTree(); }
        for (const o of [...this.group.children]) {
            if (o === this.wall?.mesh || o === this.tree) continue;
            this.group.remove(o);
            o.traverse?.((x) => { x.geometry?.dispose(); x.material?.dispose?.(); });
        }
        this.disc = null;
        this.result = null;
        this.invalidate();
    }
}

// ── 2D views of a measurement ───────────────────────────────────────────────────────────────────────────────────────

/** through-plane velocity of the averaged lumen at `frame` as an image (nearest-neighbour upscaled), with a ±vmax colour bar */
export function drawProfile(canvas, avg, frame, divArr, vmax) {
    const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = canvas.clientHeight;
    if (canvas.width !== Math.round(W * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
    const g = canvas.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    if (!avg) return;
    const n = avg.n, nn = n * n, off = document.createElement('canvas');
    off.width = off.height = n;
    const im = new ImageData(n, n);
    for (let i = 0; i < nn; i++) {
        if (avg.lumen[i]) {
            const j = 4 * Math.round(255 * Math.min(1, Math.max(0, 0.5 + 0.5 * avg.img[frame * nn + i] / vmax)));
            im.data[4 * i] = divArr[j]; im.data[4 * i + 1] = divArr[j + 1]; im.data[4 * i + 2] = divArr[j + 2]; im.data[4 * i + 3] = 255;
        } else { im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = 28; im.data[4 * i + 3] = 255; }
    }
    off.getContext('2d').putImageData(im, 0, 0);
    // crop to the lumen's bounding box (+ margin) so small vessels fill the panel
    let x0 = n, x1 = 0, y0 = n, y1 = 0;
    for (let i = 0; i < nn; i++) if (avg.lumen[i]) { const x = i % n, y = (i / n) | 0; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    if (x1 < x0) return;
    const m = 3, cx = Math.max(0, x0 - m), cy = Math.max(0, y0 - m), cw = Math.min(n, x1 + m + 1) - cx, ch = Math.min(n, y1 + m + 1) - cy;
    const bar = 14, s = Math.min((W - bar - 8) / cw, H / ch);
    g.imageSmoothingEnabled = false;
    g.drawImage(off, cx, cy, cw, ch, 2, (H - ch * s) / 2, cw * s, ch * s);
    for (let k = 0; k < H - 8; k++) {                                    // colour bar
        const j = 4 * Math.round(255 * (1 - k / (H - 9)));
        g.fillStyle = `rgb(${divArr[j]},${divArr[j + 1]},${divArr[j + 2]})`;
        g.fillRect(W - bar, 4 + k, bar - 4, 1);
    }
    g.fillStyle = '#999'; g.font = '10px sans-serif';
    g.fillText(`+${vmax.toFixed(1)}`, W - 44, 12);
    g.fillText(`−${vmax.toFixed(1)}`, W - 44, H - 6);
    g.fillText(`${(avg.ds * 1e3 * cw).toFixed(0)} mm`, 4, H - 3);
}

/** velocity along the two in-plane axes through the lumen centroid at `frame`: series for the Plot class (x in mm) */
export function profileSeries(avg, frame) {
    const n = avg.n, nn = n * n;
    let cx = 0, cy = 0, c = 0;
    for (let i = 0; i < nn; i++) if (avg.lumen[i]) { cx += i % n; cy += (i / n) | 0; c++; }
    if (!c) return [];
    const ix = Math.round(cx / c), iy = Math.round(cy / c), mm = (k) => (k + 0.5 - n / 2) * avg.ds * 1e3;
    const row = { name: 'u', color: '#4fc3f7', x: [], y: [] }, col = { name: 'w', color: '#ffb74d', x: [], y: [] };
    for (let k = 0; k < n; k++) {
        if (avg.lumen[iy * n + k]) { row.x.push(mm(k)); row.y.push(avg.img[frame * nn + iy * n + k]); }
        if (avg.lumen[k * n + ix]) { col.x.push(mm(k)); col.y.push(avg.img[frame * nn + k * n + ix]); }
    }
    return [row, col].filter((s) => s.x.length > 1);
}
