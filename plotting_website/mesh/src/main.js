// Geometry viewer: triangle meshes (hasty mesh value, rustlib/crates/hasty-server/src/mesh.rs) and line sets (kind "lines", or the
// centerlines of a flow value) from the bank. Several values in one scene, per-vertex scalars as colour, clip plane, point picking
// with distances, PNG export.
//   mesh/index.html?id=<value id>[&id=<second>…][&scalar=<name>][&selftest=1]       (also ?ids=a,b,c)
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { HastyClient } from '../../common/hasty_client.js';
import { COLORMAPS, cssGradient, lut } from '../../flow/src/colormaps.js';
import { meshValue } from '../../common/meshio.js';
import { installHelp } from '../../common/help.js';
import { HELP } from './help.js';
installHelp(HELP);

const client = new HastyClient();
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const status = (s) => { $('status').textContent = s; };
const PALETTE = ['#d9534f', '#5bc0de', '#f0ad4e', '#5cb85c', '#9b6bd8', '#e8e8e8'];

const renderer = new THREE.WebGLRenderer({ canvas: $('canvas'), antialias: true, preserveDrawingBuffer: params.get('selftest') === '1' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.localClippingEnabled = true;
const scene = new THREE.Scene();
scene.background = new THREE.Color($('bg').value);
const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 1e5);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, renderer.domElement);
const head = new THREE.DirectionalLight(0xffffff, 2.2);   // headlight: shading follows the view
scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.1), head);
const helpers = new THREE.Group();
scene.add(helpers);
const picks = new THREE.Group();
scene.add(picks);
const clip = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
let dirty = true, raf = 0;
const invalidate = () => { dirty = true; if (!raf) raf = requestAnimationFrame(loop); };
controls.addEventListener('change', invalidate);

/** @type {Array<object>} */
const parts = [];
const bbox = new THREE.Box3();
let cmapName = 'turbo';
for (const c of COLORMAPS) $('cmap').append(new Option(c, c));

function loop() {
    raf = 0;
    if (!dirty) return;
    dirty = false;
    head.position.copy(camera.position).sub(controls.target).normalize().multiplyScalar(10).add(controls.target);
    renderer.render(scene, camera);
}

function resize() {
    const v = $('view'), w = v.clientWidth, h = v.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(1, h);
    camera.updateProjectionMatrix();
    invalidate();
}
new ResizeObserver(resize).observe($('view'));

// ── loading ─────────────────────────────────────────────────────────────────────────────────────────────────────
const arr = (v, k) => v.get(k)?.data;

function meshParts(v) {
    const pos = arr(v, 'vertices'), idx = arr(v, 'faces');
    if (!pos || !idx) throw new Error('mesh value without vertices / faces');
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(new THREE.BufferAttribute(idx instanceof Uint32Array ? idx : Uint32Array.from(idx), 1));
    const nrm = arr(v, 'normals');
    if (nrm) g.setAttribute('normal', new THREE.BufferAttribute(nrm, 3)); else g.computeVertexNormals();
    g.computeBoundingBox();
    const scalars = {};
    const sc = v.get('scalars');
    if (sc instanceof Map) for (const [k, t] of sc) scalars[k] = t.data;
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.55, metalness: 0.05, side: THREE.DoubleSide, clippingPlanes: [], transparent: false });
    const obj = new THREE.Mesh(g, mat);
    let area = 0;
    for (let t = 0; t < idx.length; t += 3) {
        const a = idx[t] * 3, b = idx[t + 1] * 3, c = idx[t + 2] * 3;
        const ux = pos[b] - pos[a], uy = pos[b + 1] - pos[a + 1], uz = pos[b + 2] - pos[a + 2], wx = pos[c] - pos[a], wy = pos[c + 1] - pos[a + 1], wz = pos[c + 2] - pos[a + 2];
        area += 0.5 * Math.hypot(uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx);
    }
    return { kind: 'mesh', space: v.get('space') ?? '', obj, geometry: g, material: mat, scalars, stats: `${pos.length / 3} vertices, ${idx.length / 3} triangles, ${area.toFixed(0)} mm², space ${v.get('space') ?? '?'}`, pos, idx };
}

function meshPart(v) {
    if (v.get('kind') !== 'mesh_seq') return meshParts(v);
    const fr = v.get('frames');
    if (!Array.isArray(fr) || !fr.length) throw new Error('mesh sequence without frames');
    const first = meshParts(fr[0]);
    first.frames = fr.map((f) => ({ pos: arr(f, 'vertices'), idx: arr(f, 'faces'), nrm: arr(f, 'normals'), parent: arr(f, 'parent') ?? null }));
    first.dt = Number(v.get('dt')) || 0.04;
    first.links = v.get('links')?.data ?? null;     // links[i] = 1: frame i+1 has the vertices of frame i (blend between them), 0: a new surface
    first.coherent = v.get('coherent') === true && fr.every((f) => arr(f, 'vertices').length === first.pos.length);
    first.stats = `sequence of ${fr.length} frames: ` + first.stats;
    return first;
}

function linesPart(v) {
    const pts = arr(v, 'points'), off = arr(v, 'offsets');
    if (!pts || !off) throw new Error('line value without points / offsets');
    const seg = [], vi = [];
    for (let l = 0; l + 1 < off.length; l++) for (let i = off[l]; i + 1 < off[l + 1]; i++) vi.push(i, i + 1);
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pts, 3));
    g.setIndex(vi);
    const mat = new THREE.LineBasicMaterial({ color: 0xffffff, clippingPlanes: [] });
    const scalars = {};
    for (const k of ['speed', 'time', 'radius', 'arclength', 'branch', 'area_mm2', 'eq_diameter_mm']) if (arr(v, k)?.length === pts.length / 3) scalars[k] = arr(v, k);
    g.computeBoundingBox();
    const w = arr(v, 'pwv_waveforms');
    const pwv = w ? { w, P: v.get('pwv_waveforms').shape[0], F: v.get('pwv_waveforms').shape[1], s: arr(v, 'pwv_arclength'), delay: arr(v, 'pwv_delay_ms'), dt: Number(v.get('pwv_dt_s')) || 0.04, fwd: arr(v, 'pwv_forward') ?? null, bwd: arr(v, 'pwv_backward') ?? null, ri: Number(v.get('pwv_reflection_index')) } : null;
    return { kind: 'lines', obj: new THREE.LineSegments(g, mat), geometry: g, material: mat, scalars, pwv, stats: `${off.length - 1} lines, ${pts.length / 3} points${pwv ? `, ${pwv.P} waveforms` : ''}`, seg };
}

async function loadPart(id) {
    const entry = await client.info(id);
    let v = await client.get(id);
    if (!(v instanceof Map)) throw new Error(`${entry.name}: not a mesh / lines value`);
    let kind = v.get('kind');
    if (kind === 'flow4d' && v.get('centerlines') instanceof Map) { v = v.get('centerlines'); kind = 'lines'; }
    const p = kind === 'mesh' || kind === 'mesh_seq' ? meshPart(v) : kind === 'lines' ? linesPart(v) : (() => { throw new Error(`${entry.name}: kind ${kind} is not a mesh or lines`); })();
    return { id, name: entry.name, visible: true, opacity: 1, color: PALETTE[parts.length % PALETTE.length], scalar: '', range: null, wire: false, flat: false, ...p };
}

// ── appearance ──────────────────────────────────────────────────────────────────────────────────────────────────
function finiteRange(a) {
    const f = [];
    const step = Math.max(1, Math.floor(a.length / 200000));
    for (let i = 0; i < a.length; i += step) if (Number.isFinite(a[i])) f.push(a[i]);
    if (!f.length) return [0, 1];
    f.sort((x, y) => x - y);
    const lo = f[Math.floor(f.length * 0.02)], hi = f[Math.min(f.length - 1, Math.floor(f.length * 0.98))];
    return hi > lo ? [lo, hi] : [f[0], f[0] + 1];
}

function applyPart(p) {
    p.obj.visible = p.visible;
    const m = p.material;
    m.opacity = p.opacity;
    m.transparent = p.opacity < 1;
    m.depthWrite = p.opacity >= 1;
    if (p.kind === 'mesh') { m.wireframe = p.wire; m.flatShading = p.flat; m.side = $('two-sided').checked ? THREE.DoubleSide : THREE.FrontSide; }
    m.clippingPlanes = $('clip-on').checked ? [clip] : [];
    const g = p.geometry;
    if (p.scalar && p.scalars[p.scalar]) {
        const s = p.scalars[p.scalar], [lo, hi] = p.range ?? (p.range = finiteRange(s)), t = lut(cmapName), col = new Float32Array(s.length * 3);
        for (let i = 0; i < s.length; i++) {
            const x = s[i];
            if (!Number.isFinite(x)) { col[3 * i] = col[3 * i + 1] = col[3 * i + 2] = 0.35; continue; }
            const q = 4 * Math.round(255 * Math.min(1, Math.max(0, (x - lo) / (hi - lo))));
            col[3 * i] = t[q] / 255; col[3 * i + 1] = t[q + 1] / 255; col[3 * i + 2] = t[q + 2] / 255;
        }
        g.setAttribute('color', new THREE.BufferAttribute(col, 3));
        m.vertexColors = true;
        m.color.set(0xffffff);
    } else {
        m.vertexColors = false;
        m.color.set(p.color);
    }
    m.needsUpdate = true;
    legend();
    invalidate();
}

function legend() {
    const p = parts.find((q) => q.visible && q.scalar && q.scalars[q.scalar]);
    $('legend').hidden = !p;
    if (!p) return;
    $('legend-bar').style.background = cssGradient(cmapName);
    $('legend-lo').textContent = (+p.range[0].toPrecision(3)).toString();
    $('legend-hi').textContent = (+p.range[1].toPrecision(3)).toString();
    $('legend-title').textContent = `${p.name}: ${p.scalar}`;
}

function partUI(p) {
    const d = document.createElement('div');
    d.className = 'part';
    const names = Object.keys(p.scalars);
    d.innerHTML = `<div class="head"><input type="checkbox" class="vis" checked><input type="color" class="col"><span class="pname"></span></div>
        <div class="row"><span>opacity</span><input type="range" class="op" min="0.05" max="1" step="0.05" value="1">
            ${p.kind === 'mesh' ? '<label><input type="checkbox" class="wire"> wire</label><label><input type="checkbox" class="flat"> flat</label>' : ''}</div>
        ${names.length ? `<div class="row"><span>colour by</span><select class="sc"><option value="">solid</option>${names.map((n) => `<option>${n}</option>`).join('')}</select>
            <span>min</span><input type="number" class="lo" step="any"><span>max</span><input type="number" class="hi" step="any"></div>` : ''}
        <div class="stats"></div>`;
    d.querySelector('.pname').textContent = p.name;
    d.querySelector('.col').value = p.color;
    d.querySelector('.stats').textContent = p.stats;
    const q = (s) => d.querySelector(s);
    q('.vis').onchange = (e) => { p.visible = e.target.checked; applyPart(p); };
    q('.col').oninput = (e) => { p.color = e.target.value; applyPart(p); };
    q('.op').oninput = (e) => { p.opacity = +e.target.value; applyPart(p); };
    q('.wire')?.addEventListener('change', (e) => { p.wire = e.target.checked; applyPart(p); });
    q('.flat')?.addEventListener('change', (e) => { p.flat = e.target.checked; applyPart(p); });
    if (names.length) {
        const sync = () => { if (p.range) { q('.lo').value = +p.range[0].toPrecision(4); q('.hi').value = +p.range[1].toPrecision(4); } };
        q('.sc').onchange = (e) => { p.scalar = e.target.value; p.range = null; applyPart(p); sync(); };
        q('.lo').onchange = (e) => { p.range = [+e.target.value, p.range?.[1] ?? 1]; applyPart(p); };
        q('.hi').onchange = (e) => { p.range = [p.range?.[0] ?? 0, +e.target.value]; applyPart(p); };
        p.syncRange = sync;
    }
    $('parts').append(d);
}

// ── scene helpers ───────────────────────────────────────────────────────────────────────────────────────────────
function fitView(dir = [0.9, -1.6, 0.8]) {
    const c = bbox.getCenter(new THREE.Vector3()), s = bbox.getSize(new THREE.Vector3()), r = Math.max(s.x, s.y, s.z, 1e-3);
    const d = new THREE.Vector3(...dir).normalize().multiplyScalar(r * 2.1);
    camera.position.copy(c).add(d);
    camera.near = r / 500; camera.far = r * 60;
    camera.updateProjectionMatrix();
    controls.target.copy(c);
    controls.update();
    invalidate();
}

function buildHelpers() {
    helpers.clear();
    if (!$('box-on').checked || bbox.isEmpty()) return invalidate();
    helpers.add(new THREE.Box3Helper(bbox, 0x3a4a5a));
    const o = bbox.min, s = bbox.getSize(new THREE.Vector3()), L = Math.max(s.x, s.y, s.z) * 0.15;
    const ax = new THREE.AxesHelper(L);
    ax.position.copy(o);
    helpers.add(ax);
    invalidate();
}

function updateClip() {
    const ax = +$('clip-axis').value, f = $('clip-flip').checked ? -1 : 1;
    const lo = bbox.min.getComponent(ax), hi = bbox.max.getComponent(ax), pos = lo + (hi - lo) * +$('clip-pos').value;
    const n = new THREE.Vector3().setComponent(ax, -f);       // keeps the side where (coordinate · f) ≤ pos·f
    clip.set(n, f * pos);
    $('clip-val').textContent = `${'xyz'[ax]} = ${pos.toFixed(1)} mm`;
    parts.forEach(applyPart);
}

// ── picking ─────────────────────────────────────────────────────────────────────────────────────────────────────
const ray = new THREE.Raycaster();
const pickPts = [];
let down = null, profileMark = null;
renderer.domElement.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
renderer.domElement.addEventListener('pointerup', (e) => {
    if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
    const r = renderer.domElement.getBoundingClientRect();
    ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
    const meshes = parts.filter((p) => p.visible && p.kind === 'mesh').map((p) => p.obj);
    const size = Math.max(...bbox.getSize(new THREE.Vector3()).toArray());
    ray.params.Line = { threshold: size * 0.012 };
    const hit = ray.intersectObjects(meshes, false).find((h) => !$('clip-on').checked || clip.distanceToPoint(h.point) >= 0);
    const lh = ray.intersectObjects(parts.filter((p) => p.visible && p.kind === 'lines').map((p) => p.obj), false)[0];
    let p, near, point;
    if (lh && (!hit || lh.distance < hit.distance * 1.02)) {            // a line is the nearest thing under the cursor
        p = parts.find((q) => q.obj === lh.object);
        const a = p.geometry.index.getX(lh.index), b = p.geometry.index.getX(lh.index + 1);
        near = lh.point.distanceToSquared(new THREE.Vector3().fromBufferAttribute(p.geometry.attributes.position, a)) <= lh.point.distanceToSquared(new THREE.Vector3().fromBufferAttribute(p.geometry.attributes.position, b)) ? a : b;
        point = new THREE.Vector3().fromBufferAttribute(p.geometry.attributes.position, near);
        if (p.scalars.arclength && p.scalars.radius) profileMark = { s: p.scalars.arclength[near], d: p.scalars.eq_diameter_mm ? p.scalars.eq_diameter_mm[near] : 2 * p.scalars.radius[near], b: p.scalars.branch?.[near] };
    } else if (hit) {
        p = parts.find((q) => q.obj === hit.object);
        const f = hit.face;
        near = [f.a, f.b, f.c].map((i) => [i, hit.point.distanceToSquared(new THREE.Vector3().fromBufferAttribute(p.geometry.attributes.position, i))]).sort((a, b) => a[1] - b[1])[0][0];
        point = hit.point;
    } else return;
    const vals = Object.entries(p.scalars).map(([k, a]) => `${k} = ${(+a[near].toPrecision(4))}`).join(', ');
    drawProfile();
    pickPts.push({ pt: point.clone(), text: `${p.name}: (${point.x.toFixed(1)}, ${point.y.toFixed(1)}, ${point.z.toFixed(1)}) mm${vals ? '  ' + vals : ''}` });
    if (pickPts.length > 8) pickPts.shift();
    drawPicks();
});

function drawPicks() {
    picks.clear();
    const r = Math.max(...bbox.getSize(new THREE.Vector3()).toArray()) * 0.008;
    for (const k of pickPts) {
        const m = new THREE.Mesh(new THREE.SphereGeometry(r, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffee33, depthTest: false }));
        m.position.copy(k.pt);
        m.renderOrder = 9;
        picks.add(m);
    }
    const lines = pickPts.map((k, i) => `${i + 1}  ${k.text}`);
    if (pickPts.length > 1) { const [a, b] = pickPts.slice(-2); lines.push(`distance ${pickPts.length - 1}→${pickPts.length}: ${a.pt.distanceTo(b.pt).toFixed(2)} mm`); }
    $('pick-out').textContent = lines.join('\n');
    $('pick-out').style.whiteSpace = 'pre-wrap';
    invalidate();
}


// ── mesh sequences (one surface per frame) ───────────────────────────────────────────────────────────────────────
let frame = 0, playing = false, lastT = 0, acc = 0, tfloat = 0, shown = -1;
const seqParts = () => parts.filter((p) => p.frames);

function setFrame(f) {
    const n = Math.max(...seqParts().map((p) => p.frames.length));
    frame = ((f % n) + n) % n;
    for (const p of seqParts()) {
        const fd = p.frames[Math.min(frame, p.frames.length - 1)];
        p.geometry.dispose();
        p.geometry.setAttribute('position', new THREE.BufferAttribute(p.coherent ? Float32Array.from(fd.pos) : fd.pos, 3));
        p.geometry.setIndex(new THREE.BufferAttribute(fd.idx instanceof Uint32Array ? fd.idx : Uint32Array.from(fd.idx), 1));
        if (fd.nrm) p.geometry.setAttribute('normal', new THREE.BufferAttribute(fd.nrm, 3)); else p.geometry.computeVertexNormals();
        p.geometry.computeBoundingBox();
        p.geometry.computeBoundingSphere();
        p.pos = p.geometry.attributes.position.array; p.idx = fd.idx;
    }
    $('frame').value = frame;
    $('frame-label').textContent = `${frame + 1} / ${n}`;
    invalidate();
}

/** coherent sequences (same vertices in every frame): positions are blended between neighbouring frames, so playback is smooth */
function setTime(tf) {
    const n = Math.max(...seqParts().map((p) => p.frames.length));
    const t = ((tf % n) + n) % n, i = Math.floor(t), a = t - i;
    const linked = seqParts().every((p) => (p.links ? i < p.links.length && p.links[i] === 1 : p.coherent));
    if (!linked && a >= 1e-6 && seqParts().every((p) => p.frames[Math.min((i + 1) % n, p.frames.length - 1)].parent)) {
        // a split / merge: the next frame's surface, with every vertex starting at the position of its parent vertex in this frame
        shown = -2;
        for (const p of seqParts()) {
            const f0 = p.frames[Math.min(i, p.frames.length - 1)], f1 = p.frames[Math.min(i + 1, p.frames.length - 1)];
            const nv = f1.pos.length / 3, dst = new Float32Array(f1.pos.length);
            for (let v = 0; v < nv; v++) { const q = f1.parent[v] * 3; for (let c = 0; c < 3; c++) dst[3 * v + c] = f0.pos[q + c] * (1 - a) + f1.pos[3 * v + c] * a; }
            p.geometry.dispose();
            p.geometry.setAttribute('position', new THREE.BufferAttribute(dst, 3));
            p.geometry.setIndex(new THREE.BufferAttribute(f1.idx instanceof Uint32Array ? f1.idx : Uint32Array.from(f1.idx), 1));
            p.geometry.computeVertexNormals();
            p.geometry.computeBoundingBox();
            p.geometry.computeBoundingSphere();
            p.pos = dst; p.idx = f1.idx;
        }
        frame = i;
        $('frame').value = i;
        $('frame-label').textContent = `${i + 1} / ${n}  (morph +${a.toFixed(2)})`;
        invalidate();
        return;
    }
    if (!linked || a < 1e-6) { if (i !== shown) { setFrame(i); shown = i; } return; }       // no vertex correspondence across a topology change: show the frames as they are
    shown = -1;
    for (const p of seqParts()) {
        const f0 = p.frames[Math.min(i, p.frames.length - 1)], f1 = p.frames[Math.min((i + 1) % n, p.frames.length - 1)];
        if (p.geometry.attributes.position.array.length !== f0.pos.length) setFrame(i);        // the previous frame had another topology: rebuild the buffers first
        const dst = p.geometry.attributes.position.array;
        for (let k = 0; k < dst.length; k++) dst[k] = f0.pos[k] * (1 - a) + f1.pos[k] * a;
        p.geometry.attributes.position.needsUpdate = true;
        p.geometry.computeVertexNormals();
        p.geometry.computeBoundingBox();
        p.geometry.computeBoundingSphere();
    }
    frame = i;
    $('frame').value = i;
    $('frame-label').textContent = `${i + 1} / ${n}  (+${a.toFixed(2)})`;
    invalidate();
}

function tick(t) {
    if (!playing) return;
    const dt = Math.min(0.2, (t - lastT) / 1000) * +$('rate').value;
    lastT = t;
    acc += dt;
    const per = Math.min(...seqParts().map((p) => p.dt));
    if (seqParts().every((p) => p.coherent || p.links)) { tfloat += dt / per; setTime(tfloat); }
    else while (acc >= per) { acc -= per; setFrame(frame + 1); }
    requestAnimationFrame(tick);
}

$('play').onclick = () => {
    playing = !playing;
    tfloat = frame;
    $('play').textContent = playing ? '❚❚' : '▶';
    if (playing) { lastT = performance.now(); requestAnimationFrame(tick); }
};
$('frame').oninput = (e) => setFrame(+e.target.value);

// ── editing: erase around the last pick, keep one connected piece, save a copy ───────────────────────────────────
const history = [];
const editPart = () => parts.filter((p) => p.kind === 'mesh')[+$('edit-part').value] ?? null;
const editMsg = (t) => { $('edit-out').textContent = t; };

function setIndex(p, idx) {
    p.idx = idx;
    p.geometry.setIndex(new THREE.BufferAttribute(idx, 1));
    p.stats = p.stats.replace(/[\d]+ triangles/, `${idx.length / 3} triangles`);
    invalidate();
}

function editable() {
    const p = editPart();
    if (!p) { editMsg('no mesh to edit'); return null; }
    if (p.frames) { editMsg('sequences cannot be edited'); return null; }
    if (!pickPts.length) { editMsg('pick a point on the surface first'); return null; }
    return p;
}

$('edit-erase').onclick = () => {
    const p = editable();
    if (!p) return;
    const c = pickPts.at(-1).pt, r2 = (+$('edit-r').value) ** 2, pos = p.pos, out = [];
    for (let t = 0; t < p.idx.length; t += 3) {
        const a = p.idx[t] * 3, b = p.idx[t + 1] * 3, d = p.idx[t + 2] * 3;
        const x = (pos[a] + pos[b] + pos[d]) / 3 - c.x, y = (pos[a + 1] + pos[b + 1] + pos[d + 1]) / 3 - c.y, z = (pos[a + 2] + pos[b + 2] + pos[d + 2]) / 3 - c.z;
        if (x * x + y * y + z * z > r2) out.push(p.idx[t], p.idx[t + 1], p.idx[t + 2]);
    }
    history.push({ p, idx: p.idx });
    editMsg(`erased ${(p.idx.length - out.length) / 3} triangles`);
    setIndex(p, Uint32Array.from(out));
};

$('edit-keep').onclick = () => {
    const p = editable();
    if (!p) return;
    const nv = p.pos.length / 3, par = Int32Array.from({ length: nv }, (_, i) => i);
    const find = (x) => { while (par[x] !== x) { par[x] = par[par[x]]; x = par[x]; } return x; };
    for (let t = 0; t < p.idx.length; t += 3) for (let k = 1; k < 3; k++) { const a = find(p.idx[t]), b = find(p.idx[t + k]); if (a !== b) par[a] = b; }
    const c = pickPts.at(-1).pt;
    let best = -1, bd = Infinity;
    for (let t = 0; t < p.idx.length; t++) {
        const v = p.idx[t] * 3, d = (p.pos[v] - c.x) ** 2 + (p.pos[v + 1] - c.y) ** 2 + (p.pos[v + 2] - c.z) ** 2;
        if (d < bd) { bd = d; best = p.idx[t]; }
    }
    const root = find(best), out = [];
    for (let t = 0; t < p.idx.length; t += 3) if (find(p.idx[t]) === root) out.push(p.idx[t], p.idx[t + 1], p.idx[t + 2]);
    history.push({ p, idx: p.idx });
    editMsg(`kept ${out.length / 3} of ${p.idx.length / 3} triangles`);
    setIndex(p, Uint32Array.from(out));
};

$('edit-undo').onclick = () => {
    const h = history.pop();
    if (!h) return;
    if (h.pos) { h.p.pos.set(h.pos); h.p.geometry.attributes.position.needsUpdate = true; h.p.geometry.computeVertexNormals(); invalidate(); } else setIndex(h.p, h.idx);
    editMsg('undone');
};

$('edit-save').onclick = async () => {
    const p = editPart();
    if (!p || p.frames) return editMsg('nothing to save');
    const map = new Int32Array(p.pos.length / 3).fill(-1), verts = [], keep = [];
    const faces = new Uint32Array(p.idx.length);
    for (let i = 0; i < p.idx.length; i++) {
        const v = p.idx[i];
        if (map[v] < 0) { map[v] = verts.length / 3; verts.push(p.pos[3 * v], p.pos[3 * v + 1], p.pos[3 * v + 2]); keep.push(v); }
        faces[i] = map[v];
    }
    const scalars = Object.fromEntries(Object.entries(p.scalars).map(([k, a]) => [k, Float32Array.from(keep, (v) => a[v])]));
    const id = await client.put(meshValue({ vertices: Float32Array.from(verts), faces }, { space: p.space || 'voxel_mm', scalars: Object.keys(scalars).length ? scalars : null }), { name: `${p.name} (edited)`, metadata: { edited_from: p.id } });
    editMsg(`saved ${verts.length / 3} vertices, ${faces.length / 3} triangles as ${id}`);
};


// ── profile of a centerline tree: diameter along the vessel, one curve per branch ──────────────────────────────────────────
function drawProfile() {
    const p = parts.find((q) => q.kind === 'lines' && q.scalars.radius && q.scalars.arclength && q.scalars.branch);
    $('profile-sec').hidden = !p;
    if (!p) return;
    const c = $('profile'), g = c.getContext('2d'), W = c.width, H = c.height, m = { l: 34, r: 8, t: 8, b: 20 };
    const s = p.scalars.arclength, rr = p.scalars.radius, b = p.scalars.branch, eq = p.scalars.eq_diameter_mm;
    const r = { length: rr.length, get: (i) => (eq ? eq[i] : 2 * rr[i]) };
    let smax = 1e-9, dmax = 1e-9;
    for (let i = 0; i < s.length; i++) { smax = Math.max(smax, s[i]); dmax = Math.max(dmax, r.get(i)); }
    dmax *= 1.1;
    const X = (v) => m.l + (v / smax) * (W - m.l - m.r), Y = (v) => H - m.b - (v / dmax) * (H - m.t - m.b);
    g.fillStyle = '#0e0e0e'; g.fillRect(0, 0, W, H);
    g.strokeStyle = '#333'; g.fillStyle = '#888'; g.font = '10px monospace';
    g.beginPath(); g.moveTo(m.l, m.t); g.lineTo(m.l, H - m.b); g.lineTo(W - m.r, H - m.b); g.stroke();
    g.fillText('0', m.l - 8, H - m.b + 12); g.fillText(smax.toFixed(0) + ' mm', W - m.r - 44, H - m.b + 12); g.fillText(dmax.toFixed(1), 2, m.t + 8); g.fillText(eq ? 'eq ⌀ mm' : '⌀ mm', 2, H / 2);
    const off = p.geometry.index ? null : null;
    let cur = -1;
    for (let i = 0; i < s.length; i++) {
        if (b[i] !== cur) { if (cur >= 0) g.stroke(); cur = b[i]; g.strokeStyle = PALETTE[cur % PALETTE.length]; g.beginPath(); g.moveTo(X(s[i]), Y(r.get(i))); } else g.lineTo(X(s[i]), Y(r.get(i)));
    }
    g.stroke();
    if (profileMark) {
        g.strokeStyle = '#ffee33'; g.beginPath(); g.moveTo(X(profileMark.s), m.t); g.lineTo(X(profileMark.s), H - m.b); g.stroke();
        g.beginPath(); g.arc(X(profileMark.s), Y(profileMark.d), 3, 0, 7); g.fillStyle = '#ffee33'; g.fill();
        g.fillText(`${profileMark.d.toFixed(1)} mm @ ${profileMark.s.toFixed(0)}`, Math.min(X(profileMark.s) + 5, W - 90), m.t + 10);
    }
}


// ── pulse wave plot: the waveforms at the cross-sections of a centerline and their delay against arc length ────────────
function drawWaves() {
    const p = parts.find((q) => q.pwv);
    $('pwv-sec').hidden = !p;
    if (!p) return;
    const { w, P, F, s, delay, dt } = p.pwv, c = $('pwv'), g = c.getContext('2d'), W = c.width, H = c.height;
    g.fillStyle = '#0e0e0e'; g.fillRect(0, 0, W, H);
    g.font = '10px monospace';
    const top = { l: 30, r: 8, t: 8, b: 14, h: 112 };
    const t = lut('turbo');
    const smin = Math.min(...s), smax = Math.max(...s) || 1;
    for (let k = 0; k < P; k++) {
        let lo = Infinity, hi = -Infinity;
        for (let f = 0; f < F; f++) { const x = w[k * F + f]; lo = Math.min(lo, x); hi = Math.max(hi, x); }
        const q = 4 * Math.round(255 * (s[k] - smin) / (smax - smin || 1));
        g.strokeStyle = `rgb(${t[q]},${t[q + 1]},${t[q + 2]})`;
        g.beginPath();
        for (let f = 0; f <= F; f++) {
            const x = w[k * F + (f % F)], px = top.l + (f / F) * (W - top.l - top.r), py = top.t + (1 - (x - lo) / (hi - lo || 1)) * (top.h - top.t);
            if (f === 0) g.moveTo(px, py); else g.lineTo(px, py);
        }
        g.stroke();
    }
    g.fillStyle = '#888';
    g.fillText('0', top.l - 4, top.h + 11); g.fillText(`${(F * dt).toFixed(2)} s`, W - top.r - 40, top.h + 11); g.fillText('Q', 4, top.t + 10);
    // delay against arc length with a least-squares line
    const bx = { l: 30, r: 8, t: top.h + 28, b: H - 16 };
    const dmin = Math.min(...delay), dmax = Math.max(...delay), dr = (dmax - dmin) || 1;
    const X = (v) => bx.l + ((v - smin) / (smax - smin || 1)) * (W - bx.l - bx.r), Y = (v) => bx.b - ((v - dmin) / dr) * (bx.b - bx.t);
    g.strokeStyle = '#333'; g.beginPath(); g.moveTo(bx.l, bx.t); g.lineTo(bx.l, bx.b); g.lineTo(W - bx.r, bx.b); g.stroke();
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (let k = 0; k < P; k++) { sx += s[k]; sy += delay[k]; sxx += s[k] * s[k]; sxy += s[k] * delay[k]; }
    const slope = (P * sxy - sx * sy) / (P * sxx - sx * sx || 1), icpt = (sy - slope * sx) / P;
    g.strokeStyle = '#7ab4d8'; g.beginPath(); g.moveTo(X(smin), Y(icpt + slope * smin)); g.lineTo(X(smax), Y(icpt + slope * smax)); g.stroke();
    for (let k = 0; k < P; k++) { const q = 4 * Math.round(255 * (s[k] - smin) / (smax - smin || 1)); g.fillStyle = `rgb(${t[q]},${t[q + 1]},${t[q + 2]})`; g.beginPath(); g.arc(X(s[k]), Y(delay[k]), 3, 0, 7); g.fill(); }
    g.fillStyle = '#888';
    g.fillText(`${dmin.toFixed(1)} ms`, 2, bx.b); g.fillText(`${dmax.toFixed(1)}`, 2, bx.t + 8); g.fillText(`${smin.toFixed(0)}`, bx.l, H - 3); g.fillText(`${smax.toFixed(0)} mm`, W - bx.r - 44, H - 3);
    g.fillStyle = '#ffee33';
    g.fillText(Math.abs(slope) > 1e-9 ? `PWV ${(1 / Math.abs(slope)).toFixed(2)} m/s` : 'no delay', bx.l + 6, bx.t + 8);
    // forward / backward components at the first cross-section (flow-only wave separation)
    const { fwd, bwd, ri } = p.pwv;
    if (fwd && bwd && fwd.length === F) {
        const y0 = H - 100, h = 80;
        const all = [...fwd, ...bwd], lo = Math.min(...all), hi = Math.max(...all), Yw = (x) => y0 + h - ((x - lo) / (hi - lo || 1)) * h;
        g.strokeStyle = '#333'; g.strokeRect(top.l, y0, W - top.l - top.r, h);
        for (const [a, col] of [[fwd, '#5bc0de'], [bwd, '#f0ad4e']]) {
            g.strokeStyle = col; g.beginPath();
            for (let f = 0; f <= F; f++) { const px = top.l + (f / F) * (W - top.l - top.r); if (f === 0) g.moveTo(px, Yw(a[0])); else g.lineTo(px, Yw(a[f % F])); }
            g.stroke();
        }
        g.fillStyle = '#5bc0de'; g.fillText('forward', top.l + 4, y0 + 10);
        g.fillStyle = '#f0ad4e'; g.fillText('backward', top.l + 60, y0 + 10);
        g.fillStyle = '#ffee33'; g.fillText(`reflection index ${ri.toFixed(2)}`, W - top.r - 120, y0 + 10);
    }
}

// ── sculpting: push / pull / smooth around the last pick ──────────────────────────────────────────────────────────────────
function neighbours(p) {
    if (p.adj) return p.adj;
    const nv = p.pos.length / 3, adj = Array.from({ length: nv }, () => new Set());
    for (let t = 0; t < p.idx.length; t += 3) for (let k = 0; k < 3; k++) { adj[p.idx[t + k]].add(p.idx[t + (k + 1) % 3]); adj[p.idx[t + (k + 1) % 3]].add(p.idx[t + k]); }
    return (p.adj = adj.map((x) => [...x]));
}

function sculpt(kind) {
    const p = editable();
    if (!p) return;
    const c = pickPts.at(-1).pt, r = +$('edit-r').value, pos = p.pos, nv = pos.length / 3, nrm = p.geometry.attributes.normal.array;
    history.push({ p, pos: Float32Array.from(pos) });
    const w = new Float32Array(nv);
    let moved = 0;
    for (let v = 0; v < nv; v++) {
        const d2 = (pos[3 * v] - c.x) ** 2 + (pos[3 * v + 1] - c.y) ** 2 + (pos[3 * v + 2] - c.z) ** 2;
        if (d2 < r * r) { w[v] = Math.exp(-2 * d2 / (r * r)); moved++; }
    }
    if (kind === 'smooth') {
        const adj = neighbours(p);
        for (let it = 0; it < 3; it++) {
            const prev = Float32Array.from(pos);
            for (let v = 0; v < nv; v++) {
                if (!w[v] || !adj[v].length) continue;
                for (let a = 0; a < 3; a++) {
                    let m = 0;
                    for (const u of adj[v]) m += prev[3 * u + a];
                    pos[3 * v + a] += 0.6 * w[v] * (m / adj[v].length - prev[3 * v + a]);
                }
            }
        }
    } else {
        const amt = (kind === 'push' ? 1 : -1) * 0.25 * r;
        for (let v = 0; v < nv; v++) if (w[v]) for (let a = 0; a < 3; a++) pos[3 * v + a] += amt * w[v] * nrm[3 * v + a];
    }
    p.geometry.attributes.position.needsUpdate = true;
    p.geometry.computeVertexNormals();
    p.geometry.computeBoundingBox();
    p.geometry.computeBoundingSphere();
    editMsg(`${kind}: ${moved} vertices`);
    invalidate();
}

// ── grab tool: drag the surface around the pressed point in the screen plane ───────────────────────────────────────────────
let grab = null;
const ndc = (e) => { const r = renderer.domElement.getBoundingClientRect(); return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1); };
renderer.domElement.addEventListener('pointerdown', (e) => {
    if (!$('edit-grab').checked) return;
    const p = editPart();
    if (!p || p.frames) return;
    ray.setFromCamera(ndc(e), camera);
    const hit = ray.intersectObject(p.obj, false)[0];
    if (!hit) return;
    e.stopImmediatePropagation();
    e.preventDefault();
    down = null;
    controls.enabled = false;
    const r = +$('edit-r').value, pos = p.pos, nv = pos.length / 3, w = new Float32Array(nv);
    let n = 0;
    for (let v = 0; v < nv; v++) {
        const d2 = (pos[3 * v] - hit.point.x) ** 2 + (pos[3 * v + 1] - hit.point.y) ** 2 + (pos[3 * v + 2] - hit.point.z) ** 2;
        if (d2 < r * r) { w[v] = Math.exp(-2 * d2 / (r * r)); n++; }
    }
    grab = { p, w, start: Float32Array.from(pos), origin: hit.point.clone(), plane: new THREE.Plane().setFromNormalAndCoplanarPoint(camera.getWorldDirection(new THREE.Vector3()), hit.point), n };
    history.push({ p, pos: grab.start });
    try { renderer.domElement.setPointerCapture(e.pointerId); } catch { /* synthetic events have no active pointer */ }
}, true);
renderer.domElement.addEventListener('pointermove', (e) => {
    if (!grab) return;
    ray.setFromCamera(ndc(e), camera);
    const pt = new THREE.Vector3();
    if (!ray.ray.intersectPlane(grab.plane, pt)) return;
    const d = pt.sub(grab.origin), pos = grab.p.pos;
    for (let v = 0; v < grab.w.length; v++) if (grab.w[v]) for (let a = 0; a < 3; a++) pos[3 * v + a] = grab.start[3 * v + a] + grab.w[v] * d.getComponent(a);
    grab.p.geometry.attributes.position.needsUpdate = true;
    grab.p.geometry.computeVertexNormals();
    invalidate();
}, true);
const endGrab = () => {
    if (!grab) return;
    grab.p.geometry.computeBoundingBox();
    grab.p.geometry.computeBoundingSphere();
    editMsg(`grabbed ${grab.n} vertices`);
    grab = null;
    controls.enabled = true;
    invalidate();
};
renderer.domElement.addEventListener('pointerup', endGrab, true);
renderer.domElement.addEventListener('pointercancel', endGrab, true);
$('edit-push').onclick = () => sculpt('push');
$('edit-pull').onclick = () => sculpt('pull');
$('edit-smooth').onclick = () => sculpt('smooth');
// ── page ────────────────────────────────────────────────────────────────────────────────────────────────────────
async function pick() {
    const list = (await client.list()).filter((e) => e.summary?.type === 'dict' && (e.summary.items?.faces || e.summary.items?.offsets));
    const p = $('picker');
    p.hidden = false;
    p.innerHTML = '<h2>Geometry values in the bank</h2>' + (list.length ? '' : '<p class="hint">None yet: build one in the node editor (Geometry/Mesh From Mask), or upload an OBJ / STL / PLY with Geometry/Mesh Load.</p>');
    for (const e of list.reverse()) {
        const l = document.createElement('label');
        l.innerHTML = '<input type="checkbox"> ';
        l.append(`${e.name} (${e.summary.items.faces ? 'mesh ' + e.summary.items.vertices.shape[0] + ' vertices' : 'lines'})`);
        l.firstChild.value = e.id;
        p.append(l);
    }
    if (list.length) {
        const b = document.createElement('button');
        b.textContent = 'open selected';
        b.onclick = () => { const ids = [...p.querySelectorAll('input:checked')].map((i) => i.value); if (ids.length) location.search = ids.map((i) => 'id=' + i).join('&'); };
        p.append(b);
    }
}

async function main() {
    const ids = [...params.getAll('id'), ...(params.get('ids') ?? '').split(',').filter(Boolean)];
    if (!ids.length) { await pick(); return; }
    status('loading…');
    const errors = [];
    for (const id of ids) {
        try { parts.push(await loadPart(id)); } catch (e) { errors.push(String(e.message ?? e)); }
    }
    for (const p of parts) {
        scene.add(p.obj);
        p.geometry.boundingBox.isEmpty() || bbox.union(p.geometry.boundingBox);
        for (const fd of p.frames ?? []) for (let i = 0; i < fd.pos.length; i += 3) bbox.expandByPoint(new THREE.Vector3(fd.pos[i], fd.pos[i + 1], fd.pos[i + 2]));
        partUI(p);
    }
    $('name').textContent = parts.map((p) => p.name).join(', ');
    parts.filter((p) => p.kind === 'mesh').forEach((p, i) => $('edit-part').append(new Option(p.name, i)));
    if (seqParts().length) { $('anim').hidden = false; $('frame').max = Math.max(...seqParts().map((p) => p.frames.length)) - 1; setFrame(0); }
    document.title = `${parts[0]?.name ?? 'geometry'} — geometry`;
    const want = params.get('scalar');
    for (const p of parts) { if (want && p.scalars[want]) { p.scalar = want; const sel = $('parts').children[parts.indexOf(p)].querySelector('.sc'); if (sel) sel.value = want; } applyPart(p); p.syncRange?.(); }
    buildHelpers();
    updateClip();
    resize();
    fitView();
    drawProfile();
    drawWaves();
    status(errors.length ? errors.join('; ') : `${parts.length} object(s)`);
    if (params.get('selftest') === '1') {
        loop();
        const px = new Uint8Array(4);
        const gl = renderer.getContext();
        gl.readPixels(Math.floor(gl.drawingBufferWidth / 2), Math.floor(gl.drawingBufferHeight / 2), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
        const pre = document.createElement('pre');
        pre.id = 'selftest';
        pre.hidden = true;
        pre.textContent = JSON.stringify({ ok: parts.length > 0 && errors.length === 0, parts: parts.map((p) => ({ name: p.name, kind: p.kind, frames: p.frames?.length ?? null, stats: p.stats, scalars: Object.keys(p.scalars), scalar: p.scalar })), errors, centre_pixel: [...px], bbox: [bbox.min.toArray(), bbox.max.toArray()] });
        document.body.append(pre);
    }
}

$('bg').onchange = (e) => { scene.background = new THREE.Color(e.target.value); invalidate(); };
$('box-on').onchange = buildHelpers;
$('cmap').onchange = (e) => { cmapName = e.target.value; parts.forEach(applyPart); };
$('two-sided').onchange = () => parts.forEach(applyPart);
for (const id of ['clip-on', 'clip-axis', 'clip-flip', 'clip-pos']) $(id).addEventListener('input', updateClip);
$('pick-clear').onclick = () => { pickPts.length = 0; drawPicks(); };
$('view-reset').onclick = () => fitView();
$('view-x').onclick = () => fitView([1, 0, 0.0001]);
$('view-y').onclick = () => fitView([0, -1, 0.0001]);
$('view-z').onclick = () => fitView([0.0001, 0.0001, 1]);
$('view-shot').onclick = () => {
    renderer.render(scene, camera);
    renderer.domElement.toBlob((b) => { const a = document.createElement('a'); a.href = URL.createObjectURL(b); a.download = 'geometry.png'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000); });
};
main().catch((e) => status(String(e.message ?? e)));
