// three.js layers of the flow viewer. World coordinates are millimetres: voxel (i, j, k) sits at
// (i·dx, j·dy, k·dz); line points arrive in voxel coordinates and are scaled in the shaders.
import * as THREE from 'three';
import { lut } from './colormaps.js';
import { percentile } from '../../common/flowmath.js';

export { percentile };

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

    /** lines = {points, speed, time, offsets}; `attr` (optional Float32Array per point, in speed units) replaces speed as the colour value */
    set(lines, attr = null) {
        this.clear();
        const { points, time, offsets } = lines;
        const speed = attr ?? lines.speed;
        const L = offsets.length - 1;
        if (!this.pathlines) { this.use(this.prepare(lines, attr), true); return; }
        const mesh = this.prepare(lines, attr);
        this.use(mesh, true);
        this.#heads(lines, speed, L);
    }

    /** build the GPU geometry of a set of lines without showing it (cache it, then use() it: switching is free) */
    prepare(lines, attr = null) {
        const { points, time, offsets } = lines;
        const speed = attr ?? lines.speed;
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
        if (!this.material) this.material = new THREE.ShaderMaterial({
            uniforms: this.uniforms, vertexShader: LINE_VS, fragmentShader: LINE_FS,
            transparent: this.pathlines, depthWrite: !this.pathlines, side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(g, this.material);
        mesh.frustumCulled = false;
        mesh.userData.nLines = L;
        return mesh;
    }

    /** show a prepared mesh (kept alive for reuse: use dispose(mesh) when dropping it from a cache) */
    use(mesh, owned = false) {
        this.clear();
        this.mesh = mesh;
        mesh.userData.owned = owned;
        this.group.add(mesh);
        this.nLines = mesh.userData.nLines;
    }

    dispose(mesh) { mesh.geometry.dispose(); }

    #heads(lines, speed, L) {
        const { points, time, offsets } = lines;
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

    clear() {
        for (const o of [this.mesh, this.heads]) {
            if (!o) continue;
            this.group.remove(o);
            if (o === this.heads) { o.geometry.dispose(); o.material.dispose(); }
            else if (o.userData.owned) o.geometry.dispose();       // meshes from set() are ours; cached ones (use(m)) are not
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
uniform float opacity; uniform int steps; uniform int maxSteps; uniform float stepVox; uniform vec3 dims;
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
    // steps = 0: one sample per stepVox texture voxels of the ray's actual length inside the volume (a corner-to-corner
    // ray is sqrt(3)·N voxels long, so it gets sqrt(3)·N/stepVox samples); steps > 0: that many samples whatever the size
    int n = steps;
    if (n <= 0) n = int(ceil((t.y - t.x) * length(d * dims) / stepVox));
    n = clamp(n, 1, maxSteps);
    float dt = (t.y - t.x) / float(n);
    for (int i = 0; i < 4096; i++) {
        if (i >= n) break;
        m = max(m, texture(vol, camObj + d * (t.x + (float(i) + 0.5) * dt)).r);
    }
    float v = clamp((m - lo) / max(hi - lo, 1e-6), 0.0, 1.0);
    if (v <= 0.0) discard;
    gl_FragColor = vec4(texture2D(lut, vec2(v, 0.5)).rgb, v * opacity);
}`;

export class MipLayer {
    /** replace the volume by a new u8 [Z, Y, X] (same size): derived quantities per frame */
    setData(u8) { this.tex.image.data = u8; this.tex.needsUpdate = true; }

    /** data: Float32Array [Z, Y, X] (C-order, x fastest), or null with `pre` = {u8, dims, factor, scale}: a volume that was already
     *  max-pooled by `factor` and quantised (common/pool.js); dims are then the pooled sizes and `spacing` the full-resolution one. */
    constructor(dims, spacing, data, lutTex, pre = null) {
        let u8, scale;
        const factor = pre?.factor ?? 1;
        if (pre) { ({ u8, scale } = pre); dims = pre.dims; } else {
            scale = percentile(data, 0.999) || 1;             // PC-MRA value that maps to 1 in the shader
            const k = 255 / scale;
            u8 = new Uint8Array(data.length);
            for (let i = 0; i < data.length; i++) { const q = data[i] * k; u8[i] = q >= 255 ? 255 : q > 0 ? (q + 0.5) | 0 : 0; }
        }
        const [X, Y, Z] = dims;
        this.scale = scale;
        this.factor = factor;
        const tex = new THREE.Data3DTexture(u8, X, Y, Z);
        tex.format = THREE.RedFormat;
        tex.type = THREE.UnsignedByteType;
        tex.minFilter = tex.magFilter = THREE.LinearFilter;
        tex.unpackAlignment = 1;
        tex.needsUpdate = true;
        this.tex = tex;
        this.u8 = u8;                                     // the PC-MRA volume (restored by setData(layer.u8))
        this.uniforms = {
            vol: { value: tex }, lut: { value: lutTex }, camObj: { value: new THREE.Vector3() },
            lo: { value: 0.15 }, hi: { value: 1.0 }, opacity: { value: 0.6 },
            steps: { value: 0 }, maxSteps: { value: 2048 }, stepVox: { value: 1 }, dims: { value: new THREE.Vector3(X, Y, Z) },
        };
        const g = new THREE.BoxGeometry(1, 1, 1);
        g.translate(0.5, 0.5, 0.5);
        this.mesh = new THREE.Mesh(g, new THREE.ShaderMaterial({
            uniforms: this.uniforms, vertexShader: MIP_VS, fragmentShader: MIP_FS,
            transparent: true, depthWrite: false, side: THREE.BackSide,
        }));
        // unit cube → the voxel-centre box [−½, n − ½] · spacing
        this.mesh.scale.set(X * factor * spacing[0], Y * factor * spacing[1], Z * factor * spacing[2]);   // pooled: the last cell may reach past the data (zeros)
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

// Slice planes keep every cardiac frame of their current slice in one RGBA texture array (r,g,b = vx,vy,vz in m/s,
// a = magnitude / magHi). Everything that used to be composited per pixel on the CPU (anatomy grey, speed / component
// overlay, threshold, colormap) happens in the fragment shader, so scrubbing frames, moving the threshold or vmax and
// switching the colormap are uniform changes. The frames arrive as one request per slice (the whole cycle).
const SLICE_VS = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const SLICE_FS = /* glsl */ `
precision highp sampler2DArray;
uniform sampler2DArray tex; uniform sampler2D lut;
uniform float layer; uniform float ready; uniform float vmax; uniform float thr; uniform int mode;
uniform float opacity; uniform float alpha;
varying vec2 vUv;
void main() {
    if (ready < 0.5) discard;
    vec4 d = texture(tex, vec3(vUv, layer));
    vec3 rgb = vec3(clamp(d.a, 0.0, 1.0));
    float sp = length(d.rgb);
    if (mode == 1) {
        float a = clamp((sp - thr) / max(1e-6, 0.5 * thr + 1e-3), 0.0, 1.0) * alpha;
        rgb = mix(rgb, texture(lut, vec2(clamp(sp / vmax, 0.0, 1.0), 0.5)).rgb, a);
    } else if (mode >= 2) {
        float c = mode == 2 ? d.r : (mode == 3 ? d.g : d.b);
        float a = clamp(abs(c) / max(thr, 1e-3), 0.0, 1.0) * alpha;
        rgb = mix(rgb, texture(lut, vec2(clamp(0.5 + 0.5 * c / vmax, 0.0, 1.0), 0.5)).rgb, a);
    }
    gl_FragColor = vec4(rgb, opacity);
}`;

export class SlicePlane {
    /** @param {number} F frames; @param {boolean} floatTex RGBA32F (needs OES_texture_float_linear) else RGBA16F */
    constructor(axis, dims, spacing, F, floatTex) {
        this.axis = axis;
        this.dims = dims;
        this.spacing = spacing;
        this.F = F;
        this.floatTex = floatTex;
        const [ua, va] = sliceAxes(axis);
        this.nu = dims[ua];
        this.nv = dims[va];
        this.tex = null;                       // allocated on first use
        this.texData = null;
        this.vel = new Array(F).fill(null);    // CPU views of the velocity of each loaded frame ([3, nv, nu])
        this.has = new Array(F).fill(false);
        this.nHave = 0;
        this.magHi = 0;
        this.shown = -1;                       // frame currently displayed (the requested one, or the nearest loaded)
        this.uniforms = {
            tex: { value: null }, lut: { value: null }, layer: { value: 0 }, ready: { value: 0 }, vmax: { value: 1 }, thr: { value: 0.1 },
            mode: { value: 1 }, opacity: { value: 1 }, alpha: { value: 0.9 },
        };
        this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.ShaderMaterial({
            uniforms: this.uniforms, vertexShader: SLICE_VS, fragmentShader: SLICE_FS, side: THREE.DoubleSide, transparent: true,
        }));
        this.index = 0;
        this.cycleRequested = false;
    }

    #alloc() {
        if (this.tex) return;
        const n = this.nu * this.nv * 4 * this.F;
        this.texData = this.floatTex ? new Float32Array(n) : new Uint16Array(n);
        this.tex = new THREE.DataArrayTexture(this.texData, this.nu, this.nv, this.F);
        this.tex.format = THREE.RGBAFormat;
        this.tex.type = this.floatTex ? THREE.FloatType : THREE.HalfFloatType;
        this.tex.minFilter = this.tex.magFilter = THREE.LinearFilter;
        this.tex.generateMipmaps = false;
        this.tex.unpackAlignment = 1;
        this.tex.colorSpace = THREE.NoColorSpace;
        this.uniforms.tex.value = this.tex;
    }

    /** move to slice `idx`: rebuild the quad (voxel-centre extent) and forget the loaded frames */
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
        this.vel.fill(null);
        this.has.fill(false);
        this.nHave = 0;
        this.cycleRequested = false;
        this.shown = -1;
        // keep showing the previous slice's pixels (stale, replaced as soon as the new ones arrive) instead of blanking
        // the plane while the slider moves
        if (!this.tex) this.uniforms.ready.value = 0;
    }

    /**
     * store frames f0 … f0+nf−1: vel Float32Array [nf, 3, nv, nu], mag Float32Array [nf, nv, nu] or null (then the speed
     * is the anatomy channel). Writes the texture layers and records the velocity views for readouts.
     */
    ingest(f0, nf, vel, mag) {
        this.#alloc();
        const n = this.nu * this.nv;
        if (!this.magHi) {                                     // anatomy window: p99.5 of the first frame's magnitude, once per plane
            let ref;
            if (mag) { ref = new Float32Array(n); for (let i = 0; i < n; i++) ref[i] = Math.abs(mag[i]); }
            else { ref = new Float32Array(n); for (let i = 0; i < n; i++) ref[i] = Math.hypot(vel[i], vel[n + i], vel[2 * n + i]); }
            this.magHi = percentile(ref, 0.995) || 1;
        }
        const inv = 1 / this.magHi, A = this.texData, half = !this.floatTex;
        const toHalf = THREE.DataUtils.toHalfFloat;
        for (let k = 0; k < nf; k++) {
            const f = f0 + k;
            if (f >= this.F) break;
            const vb = k * 3 * n, mb = k * n, o = f * n * 4;
            for (let i = 0; i < n; i++) {
                const vx = vel[vb + i], vy = vel[vb + n + i], vz = vel[vb + 2 * n + i];
                const m = (mag ? Math.abs(mag[mb + i]) : Math.sqrt(vx * vx + vy * vy + vz * vz)) * inv;
                const j = o + 4 * i;
                if (half) { A[j] = toHalf(vx); A[j + 1] = toHalf(vy); A[j + 2] = toHalf(vz); A[j + 3] = toHalf(m); }
                else { A[j] = vx; A[j + 1] = vy; A[j + 2] = vz; A[j + 3] = m; }
            }
            this.vel[f] = vel.subarray(vb, vb + 3 * n);
            if (!this.has[f]) { this.has[f] = true; this.nHave++; }
            this.tex.addLayerUpdate(f);
        }
        this.tex.needsUpdate = true;
    }

    /** display frame f, or the nearest loaded one while f is still on its way; false when nothing is loaded yet */
    show(f) {
        let g = -1;
        if (this.nHave) for (let d = 0; d < this.F; d++) { if (this.has[(f + d) % this.F]) { g = (f + d) % this.F; break; } if (this.has[(f - d + this.F) % this.F]) { g = (f - d + this.F) % this.F; break; } }
        this.shown = g;
        if (g >= 0) { this.uniforms.ready.value = 1; this.uniforms.layer.value = g; }
        else if (!this.tex) this.uniforms.ready.value = 0;     // otherwise keep the stale content on screen
        return g >= 0;
    }

    /** [vx, vy, vz] at slice pixel (u, v) of frame f (nearest loaded frame), or null */
    velocityAt(f, u, v) {
        const g = this.has[f] ? f : this.shown;
        const vel = g >= 0 ? this.vel[g] : null;
        if (!vel) return null;
        const n = this.nu * this.nv, i = v * this.nu + u;
        return [vel[i], vel[n + i], vel[2 * n + i]];
    }

    style({ vmax, thr, mode, lutTex }) {
        this.uniforms.vmax.value = vmax;
        this.uniforms.thr.value = thr;
        this.uniforms.mode.value = mode;
        this.uniforms.lut.value = lutTex;
    }

    setOpacity(o) { this.uniforms.opacity.value = o; this.mesh.material.depthWrite = o > 0.99; }
}

// ─── vector glyphs: instanced cones whose direction, length and colour come from the velocity texture ─────────────
const GLYPH_VS = /* glsl */ `
precision highp sampler2DArray;
attribute vec2 iUV;
uniform sampler2DArray tex; uniform sampler2D lut;
uniform float layer; uniform float ready; uniform float thr; uniform float vmax; uniform float lenMm;
uniform vec3 pOrigin; uniform vec3 pU; uniform vec3 pV;
varying vec3 vColor;
vec3 rotY(vec3 v, vec3 d) {                      // rotate the cone axis (0,1,0) onto d
    if (d.y < -0.9999) return vec3(v.x, -v.y, -v.z);
    vec4 q = vec4(d.z, 0.0, -d.x, 1.0 + d.y);
    q /= length(q);
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
void main() {
    vec3 vel = texelFetch(tex, ivec3(int(iUV.x), int(iUV.y), int(layer)), 0).rgb;
    float sp = length(vel);
    if (ready < 0.5 || sp <= thr) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); vColor = vec3(0.0); return; }   // clipped
    float len = lenMm * min(1.0, sp / vmax) + 0.25 * lenMm;
    vec3 local = vec3(position.x * len * 0.6, position.y * len, position.z * len * 0.6);
    vec3 world = pOrigin + pU * iUV.x + pV * iUV.y + rotY(local, vel / sp);
    vColor = texture(lut, vec2(clamp(sp / vmax, 0.0, 1.0), 0.5)).rgb;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(world, 1.0);
}`;
const GLYPH_FS = /* glsl */ `
varying vec3 vColor;
void main() { gl_FragColor = vec4(vColor, 1.0); }`;

export class Glyphs {
    constructor(lutTex) {
        const cone = new THREE.ConeGeometry(0.28, 1, 10);
        cone.translate(0, 0.5, 0);
        this.cone = cone;
        this.uniforms = {
            tex: { value: null }, lut: { value: lutTex }, layer: { value: 0 }, ready: { value: 0 }, thr: { value: 0.1 }, vmax: { value: 1 },
            lenMm: { value: 1 }, pOrigin: { value: new THREE.Vector3() }, pU: { value: new THREE.Vector3() }, pV: { value: new THREE.Vector3() },
        };
        this.material = new THREE.ShaderMaterial({ uniforms: this.uniforms, vertexShader: GLYPH_VS, fragmentShader: GLYPH_FS });
        this.mesh = null;
        this.built = '';
        this.group = new THREE.Group();
    }

    /** show glyphs on `plane` (every `stride` pixels with speed > thr); the instance grid is rebuilt only when it changes */
    set(plane, { stride, thr, vmax, lenMm }) {
        const key = `${plane.axis}:${plane.nu}x${plane.nv}:${stride}`;
        if (key !== this.built) {
            this.#clearMesh();
            const cu = Math.ceil(plane.nu / stride), cv = Math.ceil(plane.nv / stride);
            const uv = new Float32Array(cu * cv * 2);
            let k = 0;
            for (let v = 0; v < plane.nv; v += stride) for (let u = 0; u < plane.nu; u += stride) { uv[k++] = u; uv[k++] = v; }
            const g = new THREE.InstancedBufferGeometry();
            g.index = this.cone.index;
            g.setAttribute('position', this.cone.getAttribute('position'));
            g.setAttribute('iUV', new THREE.InstancedBufferAttribute(uv, 2));
            g.instanceCount = cu * cv;
            this.mesh = new THREE.Mesh(g, this.material);
            this.mesh.frustumCulled = false;
            this.group.add(this.mesh);
            this.built = key;
        }
        const [ua, va] = sliceAxes(plane.axis), s = plane.spacing;
        const o = [0, 0, 0]; o[plane.axis] = plane.index * s[plane.axis];
        const U = [0, 0, 0]; U[ua] = s[ua];
        const V = [0, 0, 0]; V[va] = s[va];
        const un = this.uniforms;
        un.pOrigin.value.set(...o); un.pU.value.set(...U); un.pV.value.set(...V);
        un.thr.value = thr; un.vmax.value = vmax; un.lenMm.value = lenMm;
        this.sync(plane);
    }

    /** follow the plane's texture and displayed frame */
    sync(plane) {
        this.uniforms.tex.value = plane.tex;
        this.uniforms.layer.value = Math.max(0, plane.shown);
        this.uniforms.ready.value = plane.tex && plane.shown >= 0 ? 1 : 0;
        if (this.mesh) this.mesh.visible = this.uniforms.ready.value > 0;
    }

    hide() { if (this.mesh) this.mesh.visible = false; this.uniforms.ready.value = 0; }

    #clearMesh() {
        if (this.mesh) { this.group.remove(this.mesh); this.mesh.geometry.dispose(); this.mesh = null; }
        this.built = '';
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

// ─── centerlines (optional "centerlines" entry of the flow value; see flow/CENTERLINES.md) ───────────────────────────
const CENTERLINE_PALETTE = [0xffd23f, 0x4fc3f7, 0x81c784, 0xf06292, 0xba68c8, 0xffb74d, 0x4db6ac, 0xe0e0e0];

export class CenterlineLayer {
    constructor() { this.group = new THREE.Group(); this.group.visible = false; this.n = 0; }

    /** points: Float32Array [N*3] voxel coordinates, offsets: Uint32Array [L+1]; spacing in mm */
    set({ points, offsets }, spacing) {
        this.clear();
        const L = offsets.length - 1;
        for (let l = 0; l < L; l++) {
            const a = offsets[l], b = offsets[l + 1];
            if (b - a < 2) continue;
            const pos = new Float32Array(3 * (b - a));
            for (let i = a; i < b; i++) for (let c = 0; c < 3; c++) pos[3 * (i - a) + c] = points[3 * i + c] * spacing[c];
            const g = new THREE.BufferGeometry();
            g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
            const line = new THREE.Line(g, new THREE.LineBasicMaterial({ color: CENTERLINE_PALETTE[l % CENTERLINE_PALETTE.length], depthTest: false, transparent: true }));
            line.renderOrder = 3;
            this.group.add(line);
            this.n++;
        }
    }

    clear() {
        for (const o of [...this.group.children]) { this.group.remove(o); o.geometry.dispose(); o.material.dispose(); }
        this.n = 0;
    }
}

/** line segments from a flat position array [x0,y0,z0, x1,y1,z1, …] (world mm) */
export function segments(positions, color, { opacity = 1, depthTest = false } = {}) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const m = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color, transparent: opacity < 1, opacity, depthTest }));
    m.renderOrder = 3;
    return m;
}

/** boundary edges of the lumen pixels of a measured plane as world-mm line segments */
export function lumenContour(m, spacingMm) {
    const { n, lumen, ds, u, w, center } = m, out = [];
    const sp = spacingMm, dsMm = ds * 1e3;
    const at = (a, b) => [0, 1, 2].map((k) => (center[k] + (ds * (a * u[k] + b * w[k])) / (sp[k] * 1e-3)) * sp[k]);
    void dsMm;
    const edge = (a0, b0, a1, b1) => { out.push(...at(a0, b0), ...at(a1, b1)); };
    for (let i = 0; i < n * n; i++) {
        if (!lumen[i]) continue;
        const ix = i % n, iy = (i / n) | 0, a = ix - n / 2, b = iy - n / 2;
        if (ix === 0 || !lumen[i - 1]) edge(a, b, a, b + 1);
        if (ix === n - 1 || !lumen[i + 1]) edge(a + 1, b, a + 1, b + 1);
        if (iy === 0 || !lumen[i - n]) edge(a, b, a + 1, b);
        if (iy === n - 1 || !lumen[i + n]) edge(a, b + 1, a + 1, b + 1);
    }
    return new Float32Array(out);
}
