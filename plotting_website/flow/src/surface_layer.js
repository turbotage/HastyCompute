// Moving blood-pool surface in the flow viewer's scene: one mesh whose vertices follow the cardiac time. Shaded as a
// headlight (no scene lights needed), double-sided (inside darker), with an optional cut plane applied in the fragment
// shader — a cut-through shows the inside of the far wall. Nothing here touches the other layers.
import * as THREE from 'three';
import { blend, parseSeq, pick } from '../../common/surface_seq.js';

const VERT = `
attribute float support;
varying vec3 vN; varying vec3 vW; varying vec3 vView; varying float vSup;
void main() {
    vSup = support;
    vec4 w = modelMatrix * vec4(position, 1.0);
    vW = w.xyz;
    vN = normalize(normalMatrix * normal);
    vec4 mv = viewMatrix * w;
    vView = -mv.xyz;
    gl_Position = projectionMatrix * mv;
}`;
const FRAG = `
uniform vec3 color; uniform float opacity; uniform vec4 clip; uniform float clipOn; uniform float supMin;
varying vec3 vN; varying vec3 vW; varying vec3 vView; varying float vSup;
void main() {
    if (clipOn > 0.5 && dot(vW, clip.xyz) > clip.w) discard;
    if (vSup < supMin) discard;                       // wall shell: hide what the data does not support (supMin = 0 shows everything)
    vec3 n = normalize(vN), v = normalize(vView);
    if (!gl_FrontFacing) n = -n;
    float shade = 0.35 + 0.65 * abs(dot(n, v));
    vec3 c = gl_FrontFacing ? color : mix(color, vec3(0.75, 0.82, 0.95), 0.45);   // the inside of the wall is lighter and cooler
    gl_FragColor = vec4(c * shade, gl_FrontFacing ? opacity : min(1.0, opacity + 0.35));
}`;

export class SurfaceLayer {
    constructor() {
        this.group = new THREE.Group();
        this.group.visible = false;
        this.seq = null;
        this.uniforms = { color: { value: new THREE.Color(0xe0654f) }, opacity: { value: 1 }, clip: { value: new THREE.Vector4(1, 0, 0, 0) }, clipOn: { value: 0 }, supMin: { value: 0 } };
        this.material = new THREE.ShaderMaterial({ vertexShader: VERT, fragmentShader: FRAG, uniforms: this.uniforms, side: THREE.DoubleSide, transparent: false });
        this.geometry = new THREE.BufferGeometry();
        this.mesh = new THREE.Mesh(this.geometry, this.material);
        this.mesh.frustumCulled = false;
        this.mesh.renderOrder = 1;
        this.group.add(this.mesh);
        this.key = '';
        this.scratch = null;
    }

    get loaded() { return !!this.seq; }

    /** @param {Map} v mesh_seq value */
    set(v) {
        this.seq = parseSeq(v);
        this.key = '';
        this.scratch = null;
        return this.seq;
    }

    clear() {
        this.seq = null; this.key = '';
        this.geometry.dispose();
        this.geometry = new THREE.BufferGeometry();
        this.mesh.geometry = this.geometry;
        this.group.visible = false;
    }

    /** show cardiac time `tf` (frames, fractional); returns true when the geometry changed */
    setTime(tf) {
        if (!this.seq || !this.group.visible) return false;
        const p = pick(this.seq, tf);
        const key = `${p.i}:${p.j}:${p.a}`;
        if (key === this.key) return false;
        this.key = key;
        const fi = this.seq.frames[p.i];
        if (p.linked) {
            const n = fi.pos.length;
            if (!this.scratch || this.scratch.length !== n) this.scratch = new Float32Array(n);
            blend(this.seq, p.i, p.j, p.a, this.scratch);
            this.setPositions(this.scratch, fi.idx, null, fi.scalars.support);
        } else {
            this.setPositions(fi.pos, fi.idx, fi.nrm, fi.scalars.support);
        }
        return true;
    }

    /** copy into private buffers (the frame arrays of the sequence are never handed to the GPU attribute, so blending cannot overwrite them) */
    setPositions(pos, idx, nrm, support) {
        const g = this.geometry;
        let a = g.getAttribute('position');
        if (!a || a.array.length !== pos.length) {                 // a new surface run with another vertex count: fresh buffers
            g.dispose();
            a = new THREE.BufferAttribute(new Float32Array(pos.length), 3);
            g.setAttribute('position', a);
            g.deleteAttribute('normal');
            g.deleteAttribute('support');
        }
        a.array.set(pos);
        a.needsUpdate = true;
        if (g.getIndex()?.array !== idx) g.setIndex(new THREE.BufferAttribute(idx, 1));
        if (nrm && nrm.length === pos.length) {
            let n = g.getAttribute('normal');
            if (!n) { n = new THREE.BufferAttribute(new Float32Array(nrm.length), 3); g.setAttribute('normal', n); }
            n.array.set(nrm);
            n.needsUpdate = true;
        } else g.computeVertexNormals();
        // per-vertex support (wall shell); a surface without it is always shown (support 1)
        let sa = g.getAttribute('support');
        if (!sa || sa.array.length !== pos.length / 3) { sa = new THREE.BufferAttribute(new Float32Array(pos.length / 3), 1); g.setAttribute('support', sa); }
        if (support && support.length === sa.array.length) sa.array.set(support); else sa.array.fill(1);
        sa.needsUpdate = true;
        g.computeBoundingSphere();
    }

    setOpacity(o) {
        this.uniforms.opacity.value = o;
        this.material.transparent = o < 0.99;
        this.material.depthWrite = o >= 0.99;
        this.material.needsUpdate = true;
    }

    /** wall shell: show only vertices with support ≥ s (0…1) */
    setMinSupport(s) { this.uniforms.supMin.value = s; }

    setColor(hex) { this.uniforms.color.value.set(hex); }

    /** cut plane: keep the half-space dot(p, axis·sign) ≤ posMm; off = null */
    setClip(axis, posMm, flip) {
        if (axis == null) { this.uniforms.clipOn.value = 0; return; }
        const s = flip ? -1 : 1, n = [0, 0, 0];
        n[axis] = s;
        this.uniforms.clip.value.set(n[0], n[1], n[2], s * posMm);
        this.uniforms.clipOn.value = 1;
    }
}
