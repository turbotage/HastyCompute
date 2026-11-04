// three.js layers of the flow viewer. World coordinates are millimetres: voxel (i, j, k) sits at
// (i·dx, j·dy, k·dz); line points arrive in voxel coordinates and are scaled in the shaders.
import * as THREE from 'three';
import { lut } from './colormaps.js';

export function lutTexture(name) {
    const t = new THREE.DataTexture(lut(name), 256, 1, THREE.RGBAFormat);
    t.magFilter = THREE.LinearFilter;
    t.minFilter = THREE.LinearFilter;
    t.colorSpace = THREE.NoColorSpace;
    t.needsUpdate = true;
    return t;
}

// ─── screen-space wide line segments (streamlines, pathlines with comet tails) ──────────────────────────────────
const LINE_VS = /* glsl */ `
attribute vec3 iA; attribute vec3 iB; attribute vec2 iS; attribute vec2 iT;
uniform vec3 spacing; uniform vec2 resolution; uniform float width;
uniform float tNow; uniform float tail; uniform float cycle; uniform int mode;
varying float vSpeed; varying float vAlpha;
void main() {
    vec4 a = projectionMatrix * modelViewMatrix * vec4(iA * spacing, 1.0);
    vec4 b = projectionMatrix * modelViewMatrix * vec4(iB * spacing, 1.0);
    vec2 d = (b.xy / b.w - a.xy / a.w) * resolution;
    float L = length(d);
    d = L > 1e-6 ? d / L : vec2(1.0, 0.0);
    vec2 n = vec2(-d.y, d.x);
    vec4 p = position.x < 0.5 ? a : b;
    p.xy += n * position.y * width / resolution * p.w;
    gl_Position = p;
    vSpeed = mix(iS.x, iS.y, position.x);
    float t = mix(iT.x, iT.y, position.x);
    if (mode == 1) {
        float dt = mod(tNow - t, cycle);
        vAlpha = dt <= tail ? 1.0 - dt / tail : 0.0;
    } else vAlpha = 1.0;
}`;
const LINE_FS = /* glsl */ `
uniform sampler2D lut; uniform float vmax; uniform float opacity;
varying float vSpeed; varying float vAlpha;
void main() {
    if (vAlpha <= 0.002) discard;
    vec3 c = texture2D(lut, vec2(clamp(vSpeed / vmax, 0.0, 1.0), 0.5)).rgb;
    gl_FragColor = vec4(c, vAlpha * opacity);
}`;
const HEAD_VS = /* glsl */ `
attribute float aS; attribute float aT; attribute float aStep;
uniform vec3 spacing; uniform float tNow; uniform float cycle; uniform float size;
varying float vSpeed;
void main() {
    vSpeed = aS;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position * spacing, 1.0);
    float dt = mod(tNow - aT, cycle);
    gl_PointSize = dt < aStep ? size : 0.0;
}`;
const HEAD_FS = /* glsl */ `
uniform sampler2D lut; uniform float vmax;
varying float vSpeed;
void main() {
    vec2 q = gl_PointCoord * 2.0 - 1.0;
    float r = dot(q, q);
    if (r > 1.0) discard;
    vec3 c = texture2D(lut, vec2(clamp(vSpeed / vmax, 0.0, 1.0), 0.5)).rgb;
    gl_FragColor = vec4(mix(c, vec3(1.0), 0.35 * (1.0 - r)), 1.0);
}`;

const QUAD = new Float32Array([0, -1, 0, 1, -1, 0, 1, 1, 0, 0, -1, 0, 1, 1, 0, 0, 1, 0]);

/** lines = {points Float32Array [N*3], speed [N], time [N], offsets Uint32Array [L+1]} (voxel coordinates) */
export class LineLayer {
    constructor(spacing, lutTex, { pathlines = false } = {}) {
        this.group = new THREE.Group();
        this.pathlines = pathlines;
        this.uniforms = {
            spacing: { value: new THREE.Vector3(...spacing) }, resolution: { value: new THREE.Vector2(1, 1) },
            width: { value: pathlines ? 2.0 : 1.6 }, tNow: { value: 0 }, tail: { value: 0.15 }, cycle: { value: 1 },
            mode: { value: pathlines ? 1 : 0 }, lut: { value: lutTex }, vmax: { value: 1 }, opacity: { value: pathlines ? 1.0 : 0.85 },
        };
        this.headUniforms = {
            spacing: this.uniforms.spacing, tNow: this.uniforms.tNow, cycle: this.uniforms.cycle,
            size: { value: 5.0 }, lut: { value: lutTex }, vmax: this.uniforms.vmax,
        };
        this.mesh = null;
        this.heads = null;
        this.nLines = 0;
    }

    set(lines) {
        this.clear();
        const { points, speed, time, offsets } = lines;
        const L = offsets.length - 1;
        let nseg = 0;
        for (let l = 0; l < L; l++) nseg += Math.max(0, offsets[l + 1] - offsets[l] - 1);
        const A = new Float32Array(nseg * 3), B = new Float32Array(nseg * 3), S = new Float32Array(nseg * 2), T = new Float32Array(nseg * 2);
        let s = 0;
        for (let l = 0; l < L; l++) {
            for (let k = offsets[l]; k + 1 < offsets[l + 1]; k++, s++) {
                for (let c = 0; c < 3; c++) { A[3 * s + c] = points[3 * k + c]; B[3 * s + c] = points[3 * (k + 1) + c]; }
                S[2 * s] = speed[k]; S[2 * s + 1] = speed[k + 1];
                T[2 * s] = time[k]; T[2 * s + 1] = time[k + 1];
            }
        }
        const g = new THREE.InstancedBufferGeometry();
        g.setAttribute('position', new THREE.BufferAttribute(QUAD, 3));
        g.setAttribute('iA', new THREE.InstancedBufferAttribute(A, 3));
        g.setAttribute('iB', new THREE.InstancedBufferAttribute(B, 3));
        g.setAttribute('iS', new THREE.InstancedBufferAttribute(S, 2));
        g.setAttribute('iT', new THREE.InstancedBufferAttribute(T, 2));
        g.instanceCount = nseg;
        const m = new THREE.ShaderMaterial({
            uniforms: this.uniforms, vertexShader: LINE_VS, fragmentShader: LINE_FS,
            transparent: this.pathlines, depthWrite: !this.pathlines, side: THREE.DoubleSide,
        });
        this.mesh = new THREE.Mesh(g, m);
        this.mesh.frustumCulled = false;
        this.group.add(this.mesh);
        if (this.pathlines) {
            // particle heads: every sample point, drawn when it is the newest within one sampling step
            const N = points.length / 3;
            const step = new Float32Array(N);
            for (let l = 0; l < L; l++) {
                for (let k = offsets[l]; k < offsets[l + 1]; k++) {
                    const k1 = Math.min(k + 1, offsets[l + 1] - 1);
                    step[k] = k1 > k ? time[k1] - time[k] : 1e-6;
                }
            }
            const hg = new THREE.BufferGeometry();
            hg.setAttribute('position', new THREE.BufferAttribute(points, 3));
            hg.setAttribute('aS', new THREE.BufferAttribute(speed, 1));
            hg.setAttribute('aT', new THREE.BufferAttribute(time, 1));
            hg.setAttribute('aStep', new THREE.BufferAttribute(step, 1));
            this.heads = new THREE.Points(hg, new THREE.ShaderMaterial({ uniforms: this.headUniforms, vertexShader: HEAD_VS, fragmentShader: HEAD_FS }));
            this.heads.frustumCulled = false;
            this.group.add(this.heads);
        }
        this.nLines = L;
    }

    clear() {
        for (const o of [this.mesh, this.heads]) {
            if (o) { this.group.remove(o); o.geometry.dispose(); o.material.dispose(); }
        }
        this.mesh = this.heads = null;
        this.nLines = 0;
    }
}

// ─── PC-MRA maximum intensity projection (ray marching a 3D texture) ─────────────────────────────────────────────
const MIP_VS = /* glsl */ `
varying vec3 vPos;
void main() { vPos = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const MIP_FS = /* glsl */ `
precision highp sampler3D;
uniform sampler3D vol; uniform sampler2D lut; uniform vec3 camObj; uniform float lo; uniform float hi;
uniform float opacity; uniform int steps;
varying vec3 vPos;
vec2 box(vec3 o, vec3 d) {
    vec3 inv = 1.0 / d;
    vec3 t0 = (vec3(0.0) - o) * inv, t1 = (vec3(1.0) - o) * inv;
    vec3 tn = min(t0, t1), tf = max(t0, t1);
    return vec2(max(max(tn.x, tn.y), tn.z), min(min(tf.x, tf.y), tf.z));
}
void main() {
    vec3 d = normalize(vPos - camObj);
    vec2 t = box(camObj, d);
    t.x = max(t.x, 0.0);
    if (t.x >= t.y) discard;
    float m = 0.0;
    float dt = (t.y - t.x) / float(steps);
    for (int i = 0; i < 512; i++) {
        if (i >= steps) break;
        m = max(m, texture(vol, camObj + d * (t.x + (float(i) + 0.5) * dt)).r);
    }
    float v = clamp((m - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
    if (v <= 0.0) discard;
    gl_FragColor = vec4(texture2D(lut, vec2(v, 0.5)).rgb, v * opacity);
}`;

export class MipLayer {
    /** data: Float32Array [Z, Y, X] (C-order, x fastest) */
    constructor(dims, spacing, data, lutTex) {
        const [X, Y, Z] = dims;
        let mx = 0;
        const sorted = Float32Array.from(data).sort();
        const p999 = sorted[Math.floor((sorted.length - 1) * 0.999)] || 1;
        const u8 = new Uint8Array(data.length);
        for (let i = 0; i < data.length; i++) { u8[i] = Math.min(255, Math.round(255 * data[i] / p999)); mx = Math.max(mx, data[i]); }
        const tex = new THREE.Data3DTexture(u8, X, Y, Z);
        tex.format = THREE.RedFormat;
        tex.type = THREE.UnsignedByteType;
        tex.minFilter = tex.magFilter = THREE.LinearFilter;
        tex.unpackAlignment = 1;
        tex.needsUpdate = true;
        this.uniforms = {
            vol: { value: tex }, lut: { value: lutTex }, camObj: { value: new THREE.Vector3() },
            lo: { value: 0.15 }, hi: { value: 1.0 }, opacity: { value: 0.6 }, steps: { value: 256 },
        };
        const g = new THREE.BoxGeometry(1, 1, 1);
        g.translate(0.5, 0.5, 0.5);
        this.mesh = new THREE.Mesh(g, new THREE.ShaderMaterial({
            uniforms: this.uniforms, vertexShader: MIP_VS, fragmentShader: MIP_FS,
            transparent: true, depthWrite: false, side: THREE.BackSide,
        }));
        // unit cube → the voxel-centre box [−½, n − ½] · spacing
        this.mesh.scale.set(X * spacing[0], Y * spacing[1], Z * spacing[2]);
        this.mesh.position.set(-0.5 * spacing[0], -0.5 * spacing[1], -0.5 * spacing[2]);
        this.mesh.renderOrder = -1;
        this.mesh.onBeforeRender = (r, s, camera) => {
            this.mesh.updateMatrixWorld();
            const inv = this.mesh.matrixWorld.clone().invert();
            this.uniforms.camObj.value.copy(camera.position).applyMatrix4(inv);
        };
    }
}

// ─── slice planes (magnitude + speed overlay, CPU-composited RGBA8) and vector glyphs ────────────────────────────
function quad(p00, p10, p01, p11) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array([...p00, ...p10, ...p01, ...p11]), 3));
    g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), 2));
    g.setIndex([0, 1, 3, 0, 3, 2]);
    return g;
}

/** axis 2: axial (z = idx, image x × y), 1: coronal (y = idx, x × z), 0: sagittal (x = idx, y × z) */
export function sliceAxes(axis) {
    return axis === 2 ? [0, 1] : axis === 1 ? [0, 2] : [1, 2];
}

export class SlicePlane {
    constructor(axis, dims, spacing) {
        this.axis = axis;
        this.dims = dims;
        this.spacing = spacing;
        const [ua, va] = sliceAxes(axis);
        this.nu = dims[ua];
        this.nv = dims[va];
        this.rgba = new Uint8Array(this.nu * this.nv * 4);
        this.tex = new THREE.DataTexture(this.rgba, this.nu, this.nv, THREE.RGBAFormat);
        this.tex.minFilter = this.tex.magFilter = THREE.LinearFilter;
        this.tex.colorSpace = THREE.SRGBColorSpace;
        this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({ map: this.tex, side: THREE.DoubleSide, transparent: true }));
        this.glyphs = null;
        this.index = 0;
    }

    /** move to slice `idx`: rebuild the quad (voxel-centre extent) */
    place(idx) {
        this.index = idx;
        const [ua, va] = sliceAxes(this.axis);
        const s = this.spacing;
        const lo = [0, 1, 2].map((a) => -0.5 * s[a]);
        const hi = [0, 1, 2].map((a) => (this.dims[a] - 0.5) * s[a]);
        const pt = (u, v) => {
            const p = [0, 0, 0];
            p[this.axis] = idx * s[this.axis];
            p[ua] = u ? hi[ua] : lo[ua];
            p[va] = v ? hi[va] : lo[va];
            return p;
        };
        this.mesh.geometry.dispose();
        this.mesh.geometry = quad(pt(0, 0), pt(1, 0), pt(0, 1), pt(1, 1));
    }

    /** composite: grayscale magnitude (window [0, magHi]) + speed colormap where speed > thr (alpha ramp) */
    paint(mag, vel, { magHi, vmax, thr, overlay, lutArr, alpha = 0.9 }) {
        const n = this.nu * this.nv;
        const out = this.rgba;
        for (let i = 0; i < n; i++) {
            const g = mag ? Math.min(1, Math.max(0, mag[i] / magHi)) : 0;
            let r = g, gg = g, b = g;
            if (overlay && vel) {
                const vx = vel[i], vy = vel[n + i], vz = vel[2 * n + i];
                let val, a;
                if (overlay === 'speed') {
                    val = Math.sqrt(vx * vx + vy * vy + vz * vz) / vmax;
                    a = Math.min(1, Math.max(0, (val * vmax - thr) / Math.max(1e-6, 0.5 * thr + 1e-3))) * alpha;
                } else {
                    const comp = overlay === 'vx' ? vx : overlay === 'vy' ? vy : vz;
                    val = 0.5 + 0.5 * comp / vmax;
                    a = Math.min(1, Math.abs(comp) / Math.max(thr, 1e-3)) * alpha;
                }
                const j = 4 * Math.round(255 * Math.min(1, Math.max(0, val)));
                r = g * (1 - a) + a * lutArr[j] / 255;
                gg = g * (1 - a) + a * lutArr[j + 1] / 255;
                b = g * (1 - a) + a * lutArr[j + 2] / 255;
            }
            out[4 * i] = 255 * r; out[4 * i + 1] = 255 * gg; out[4 * i + 2] = 255 * b; out[4 * i + 3] = 255;
        }
        this.tex.needsUpdate = true;
    }
}

/** instanced cones along v on a slice: vel [3, nv, nu] (slice layout), every `stride` pixels with speed > thr */
export class Glyphs {
    constructor(lutArr) {
        const cone = new THREE.ConeGeometry(0.28, 1, 10);
        cone.translate(0, 0.5, 0);
        this.geo = cone;
        this.mat = new THREE.MeshBasicMaterial();
        this.mesh = null;
        this.lutArr = lutArr;
        this.group = new THREE.Group();
    }

    set(plane, vel, { stride, thr, vmax, lenMm }) {
        if (this.mesh) { this.group.remove(this.mesh); this.mesh.dispose(); this.mesh = null; }
        if (!vel) return;
        const [ua, va] = sliceAxes(plane.axis);
        const n = plane.nu * plane.nv;
        const items = [];
        for (let v = 0; v < plane.nv; v += stride) {
            for (let u = 0; u < plane.nu; u += stride) {
                const i = v * plane.nu + u;
                const vv = [vel[i], vel[n + i], vel[2 * n + i]];
                const sp = Math.hypot(...vv);
                if (sp > thr) items.push([u, v, vv, sp]);
            }
        }
        if (!items.length) return;
        const m = new THREE.InstancedMesh(this.geo, this.mat, items.length);
        const Y = new THREE.Vector3(0, 1, 0), dir = new THREE.Vector3(), q = new THREE.Quaternion(), M = new THREE.Matrix4();
        const col = new THREE.Color();
        const s = plane.spacing;
        items.forEach(([u, v, vv, sp], k) => {
            const p = [0, 0, 0];
            p[plane.axis] = plane.index * s[plane.axis];
            p[ua] = u * s[ua];
            p[va] = v * s[va];
            dir.set(vv[0], vv[1], vv[2]).normalize();
            q.setFromUnitVectors(Y, dir);
            const len = lenMm * Math.min(1, sp / vmax) + 0.25 * lenMm;
            M.compose(new THREE.Vector3(...p), q, new THREE.Vector3(len * 0.6, len, len * 0.6));
            m.setMatrixAt(k, M);
            const j = 4 * Math.round(255 * Math.min(1, sp / vmax));
            col.setRGB(this.lutArr[j] / 255, this.lutArr[j + 1] / 255, this.lutArr[j + 2] / 255, THREE.SRGBColorSpace);
            m.setColorAt(k, col);
        });
        m.instanceMatrix.needsUpdate = true;
        m.instanceColor.needsUpdate = true;
        this.mesh = m;
        this.group.add(m);
    }
}

// ─── analysis disc: through-plane velocity per frame (signed colormap) ───────────────────────────────────────────
export class PlaneDisc {
    /** info: flow_plane JSON (center voxel coords, u, w, grid, pixel_mm); img Float32Array [F, n, n]; region Uint8Array [n, n] */
    constructor(info, img, region, spacing, lutArr) {
        this.info = info;
        this.img = img;
        this.region = region;
        this.n = info.grid;
        this.lutArr = lutArr;
        this.rgba = new Uint8Array(this.n * this.n * 4);
        this.tex = new THREE.DataTexture(this.rgba, this.n, this.n, THREE.RGBAFormat);
        this.tex.minFilter = this.tex.magFilter = THREE.NearestFilter;
        this.tex.colorSpace = THREE.SRGBColorSpace;
        const c = info.center.map((v, a) => v * spacing[a]);
        const R = 0.5 * this.n * info.pixel_mm;
        const at = (a, b) => [0, 1, 2].map((k) => c[k] + R * (a * info.u[k] + b * info.w[k]));
        this.mesh = new THREE.Mesh(quad(at(-1, -1), at(1, -1), at(-1, 1), at(1, 1)), new THREE.MeshBasicMaterial({ map: this.tex, side: THREE.DoubleSide, transparent: true, depthWrite: false }));
        this.mesh.renderOrder = 2;
        // rim and normal arrow
        const ring = [];
        for (let i = 0; i <= 64; i++) { const t = (2 * Math.PI * i) / 64; ring.push(new THREE.Vector3(...at(Math.cos(t), Math.sin(t)))); }
        this.rim = new THREE.Line(new THREE.BufferGeometry().setFromPoints(ring), new THREE.LineBasicMaterial({ color: 0xffd23f }));
        const nrm = new THREE.Vector3(...info.normal);
        this.arrow = new THREE.ArrowHelper(nrm, new THREE.Vector3(...c), R * 0.9, 0xffd23f, R * 0.25, R * 0.15);
        this.group = new THREE.Group();
        this.group.add(this.mesh, this.rim, this.arrow);
    }

    frame(f, vmax) {
        const n2 = this.n * this.n;
        const base = (f % (this.img.length / n2)) * n2;
        for (let i = 0; i < n2; i++) {
            const inside = this.region[i];
            const x = 0.5 + 0.5 * this.img[base + i] / vmax;
            const j = 4 * Math.round(255 * Math.min(1, Math.max(0, x)));
            this.rgba[4 * i] = this.lutArr[j]; this.rgba[4 * i + 1] = this.lutArr[j + 1]; this.rgba[4 * i + 2] = this.lutArr[j + 2];
            this.rgba[4 * i + 3] = inside ? 235 : 0;
        }
        this.tex.needsUpdate = true;
    }
}
