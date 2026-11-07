// Vessel tool: click a vessel in the 3D view → a ray is marched through the PC-MRA to find the lumen, a stack of
// analysis planes is placed along the vessel (⟂ to the nearest centerline when the flow value has centerlines, else by
// following the time-mean flow from the click), each plane's lumen is the connected region around the click, and the
// through-plane flow is averaged over the N planes (noise ↓ ~ √N). All maths is in common/flowmath.js and runs here, in
// the browser, on the velocity volume that field.js loaded.
import * as THREE from 'three';
import { averageStack, measureStack, pickRay, walkStack, seedsFromPlane } from '../../common/flowmath.js';
import { PlaneDisc, lumenContour, segments } from './layers.js';

const CYAN = 0x4fc3f7, YELLOW = 0xffd23f;

export class VesselTool {
    /**
     * @param {object} o  scene, spacingMm [dx,dy,dz], divArr (256×RGBA coolwarm), getField(), getPc() (Float32Array [Z,Y,X]),
     *                    getMask() (Uint8Array|null), getCenterlines() (Centerlines|null), invalidate()
     */
    constructor(o) {
        Object.assign(this, o);
        this.group = new THREE.Group();
        o.scene.add(this.group);
        this.result = null;
        this.disc = null;
    }

    /** voxel-space ray → hit {p, value} on the PC-MRA volume (first sample ≥ level, refined to the local maximum), or null */
    pick(originVox, dirVox, level) {
        const f = this.getField();
        return pickRay(this.getPc(), f.dims, originVox, dirVox, { level, mode: 'first', refine: 4 });
    }

    /** lumen candidates: the mask, or the PC-MRA above `frac` × its value at the click (adapts to bright and dim vessels) */
    region(hit, { frac = 0.4, global = null }) {
        const mask = this.getMask();
        if (mask) return mask;
        const pc = this.getPc(), thr = global ?? frac * hit.value, r = new Uint8Array(pc.length);
        for (let i = 0; i < pc.length; i++) r[i] = pc[i] >= thr ? 1 : 0;
        return r;
    }

    /**
     * Measure at a PC-MRA hit. opts: {nPlanes, stepMm, radiusMm, frac, global, orient}.
     * Returns {planes, avg, source, region, hit} or null (no lumen / no flow).
     */
    measure(hit, { nPlanes = 5, stepMm = 1.5, radiusMm = 18, frac = 0.4, global = null, orient = true } = {}) {
        const field = this.getField(), region = this.region(hit, { frac, global });
        const radiusM = radiusMm * 1e-3;
        const n = Math.max(24, Math.min(80, 2 * Math.round(radiusMm / Math.min(...field.spacing.map((s) => s * 1e3)) * 2)));   // pixel ≈ ½ voxel
        const cl = this.getCenterlines?.();
        let planes = null, source = 'flow';
        if (cl) {
            const st = cl.stack(hit.p, nPlanes, stepMm);
            if (st && st.dist_mm < Math.max(8, radiusMm * 0.6)) {
                planes = measureStack(field, st.planes, { radiusM, n, region, orient });
                source = 'centerline';
            }
        }
        if (!planes) planes = walkStack(field, region, hit.p, { nPlanes, stepM: stepMm * 1e-3, radiusM, n });
        planes = (planes ?? []).filter((m) => m.nlumen > 0);
        if (!planes.length) return null;
        return { planes, avg: averageStack(field, planes), source, region, hit, opts: { nPlanes, stepMm, radiusMm, frac } };
    }

    /** seeds on the averaged lumen of the centre plane */
    seeds(res, n, seed = 5) {
        const mid = res.planes[(res.planes.length - 1) >> 1];
        return seedsFromPlane(this.getField(), { ...mid, lumen: res.avg.lumen }, n, seed);
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

    clear() {
        for (const o of [...this.group.children]) {
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
