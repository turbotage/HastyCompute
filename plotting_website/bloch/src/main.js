// Bloch parcel viewer: a small parcel of spins is carried along pathlines of a 4D-flow value while a designed flow
// sequence is played (server command bloch_parcel, rustlib/crates/hasty-server/src/bloch.rs). Middle: every spin as an
// arrow on the unit sphere + the ensemble mean; right: MIP of the vessel tree with the parcel; bottom: the played TR
// (RF, gradients, mean magnetisation) and the echo / M_z over all TRs.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { HastyClient } from '../../common/hasty_client.js';
import { loadPcmra } from '../../flow/src/pcmra.js';
import { installHelp } from '../../common/help.js';
import { HELP } from './help.js';
installHelp(HELP);

const client = new HastyClient().enableWorkers();
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const status = (s) => { $('status').textContent = s; };
const num = (id) => Number($(id).value);
const temps = new Set();
window.addEventListener('pagehide', () => { for (const t of temps) client.delete(t).catch(() => {}); });

const hsl = (h, s, l) => new THREE.Color().setHSL(((h % 1) + 1) % 1, s, l);

async function pick() {
    const list = (await client.list()).filter((e) => e.summary?.type === 'dict' && e.summary.items?.vel);
    const p = $('picker');
    p.hidden = false;
    p.innerHTML = '<h2>Flow values in the bank</h2>';
    for (const e of list.reverse()) {
        const a = document.createElement('a');
        a.href = `?id=${e.id}`;
        a.textContent = `${e.name} — ${e.summary.items.vel.shape.join('×')}`;
        p.append(a);
    }
    const a = document.createElement('a');
    a.href = '?plug=1';
    a.textContent = 'No flow value: plug-flow parcel';
    p.append(a);
    if (!list.length) p.insertAdjacentHTML('beforeend', '<p class="hint">No flow value in the bank yet (show_flow …).</p>');
}

async function main() {
    const id = params.get('id');
    const plug = params.get('plug') === '1';
    if (!id && !plug) { await pick(); return; }

    // ── flow value, PC-MRA ───────────────────────────────────────────────────────────────────────────────────
    let info = null, pcm = null;
    const mips = {};            // axis → {img, depth, nu, nv}
    if (id) {
        const entry = await client.info(id);
        $('name').textContent = entry.name;
        document.title = `${entry.name} — Bloch parcel`;
        status('flow_info…');
        const r = await client.execute('flow_info', {}, [id], 'flow_info (bloch viewer)');
        info = JSON.parse(r.message);
        r.outputs.forEach((o) => client.delete(o).catch(() => {}));
        $('t0').max = info.frames - 1;
        status('PC-MRA…');
        const pcOut = (await client.execute('flow_pcmra', {}, [id], 'flow_pcmra (bloch viewer)')).outputs[0];
        temps.add(pcOut);
        pcm = await loadPcmra(client, pcOut, info.dims, { factor: params.get('mip') ?? 'auto', maxVox: Number(params.get('mipvox') ?? 2 ** 26), onProgress: (a, b) => status(`PC-MRA: ${a}/${b} planes`) });
        client.delete(pcOut).catch(() => {}); temps.delete(pcOut);
        status('');
    } else {
        $('src-flow').hidden = true;
        $('src-plug').hidden = false;
        $('name').textContent = 'plug flow';
    }

    // designs
    {
        const r = await client.execute('bloch_designs', {}, [], 'bloch_designs');
        const m = JSON.parse(r.message);
        for (const d of m.designs) $('design').append(new Option(`${d.name} (TR ${(d.tr * 1e3).toFixed(2)} ms, TE ${(d.te * 1e3).toFixed(2)} ms, ${d.n_enc} encodes)`, d.name));
        const v2 = m.designs.find((d) => d.name === 'flow_thorax_v2_1mm') ?? m.designs.find((d) => d.name === 'flow_thorax_v2');
        if (v2) $('design').value = v2.name;
        if (!m.designs.length) status(`no sequence designs in ${m.dir}`);
    }

    const TISSUE = { gad: [0.388, 0.111], native: [1.7, 0.2], muscle: [1.4, 0.05], fat: [0.38, 0.07] };
    $('contrast').addEventListener('change', () => { const t = TISSUE[$('contrast').value]; if (t) { $('t1').value = t[0]; $('t2').value = t[1]; } });
    for (const i of ['t1', 't2']) $(i).addEventListener('input', () => { $('contrast').value = 'custom'; });
    for (const i of ['rf', 'slab-thk', 'slab-c']) $(i).addEventListener('input', () => draw());

    // ── MIP ──────────────────────────────────────────────────────────────────────────────────────────────────
    const AX = { 1: [0, 2], 0: [1, 2], 2: [0, 1] };   // projection axis → [horizontal, vertical] array axes
    function mipOf(a) {
        if (mips[a]) return mips[a];
        const [X, Y, Z] = pcm.dims, d = [X, Y, Z];
        const [ua, va] = AX[a];
        const nu = d[ua], nv = d[va];
        const img = new Uint8Array(nu * nv), depth = new Uint16Array(nu * nv);
        const u8 = pcm.u8;
        for (let k = 0; k < Z; k++) for (let j = 0; j < Y; j++) for (let i = 0; i < X; i++) {
            const v = u8[i + X * (j + Y * k)];
            const c = [i, j, k];
            const q = c[ua] + nu * c[va];
            if (v > img[q]) { img[q] = v; depth[q] = c[a]; }
        }
        return (mips[a] = { img, depth, nu, nv, ua, va });
    }
    const mipCanvas = $('mip'), mctx = mipCanvas.getContext('2d');
    const off = document.createElement('canvas');
    let lay = null;
    let center = null;       // voxel coordinates (full resolution)
    let axisSel = 1;
    if (pcm) { center = pcm.hot.slice(); }
    const spacing = info ? info.spacing : null;     // m
    const layout = () => {
        const W = mipCanvas.width, H = mipCanvas.height, [ua, va] = AX[axisSel];
        const wmm = info.dims[ua] * spacing[ua], hmm = info.dims[va] * spacing[va];
        const s = Math.min((W - 16) / wmm, (H - 16) / hmm);
        lay = { s, x0: (W - wmm * s) / 2, y0: (H - hmm * s) / 2, ua, va, hmm, wmm };
    };
    const toScr = (v) => [lay.x0 + (v[lay.ua] + 0.5) * spacing[lay.ua] * lay.s, lay.y0 + (info.dims[lay.va] - v[lay.va] - 0.5) * spacing[lay.va] * lay.s];
    const fromScr = (px, py) => [(px - lay.x0) / (spacing[lay.ua] * lay.s) - 0.5, info.dims[lay.va] - (py - lay.y0) / (spacing[lay.va] * lay.s) - 0.5];
    function drawMip(k) {
        const W = mipCanvas.width, H = mipCanvas.height;
        mctx.fillStyle = '#080808'; mctx.fillRect(0, 0, W, H);
        if (!pcm) { mctx.fillStyle = '#777'; mctx.font = '13px sans-serif'; mctx.fillText('plug flow: no vessel tree', 16, 24); return; }
        const m = mipOf(axisSel);
        if (off.width !== m.nu || off.height !== m.nv || off.dataset.axis !== String(axisSel)) {
            off.width = m.nu; off.height = m.nv; off.dataset.axis = String(axisSel);
            const c = off.getContext('2d'), im = c.createImageData(m.nu, m.nv);
            for (let y = 0; y < m.nv; y++) for (let x = 0; x < m.nu; x++) {
                const v = Math.round(255 * Math.sqrt(m.img[x + m.nu * (m.nv - 1 - y)] / 255));
                const o = 4 * (x + m.nu * y);
                im.data[o] = im.data[o + 1] = im.data[o + 2] = v; im.data[o + 3] = 255;
            }
            c.putImageData(im, 0, 0);
        }
        layout();
        mctx.imageSmoothingEnabled = true;
        mctx.drawImage(off, lay.x0, lay.y0, lay.wmm * lay.s, lay.hmm * lay.s);
        mctx.strokeStyle = '#2c3a48'; mctx.strokeRect(lay.x0, lay.y0, lay.wmm * lay.s, lay.hmm * lay.s);
        // the excited slab (along z) on the views where z is vertical
        if (info && $('rf').value === 'slab' && axisSel !== 2) {
            const iso = res ? res.meta.iso : info.dims.map((n) => (n - 1) / 2), sz = spacing[2];
            const zc = iso[2] + num('slab-c') * 1e-3 / sz, half = num('slab-thk') * 1e-3 / sz / 2;
            const [, y0] = toScr([0, 0, zc + half]), [, y1] = toScr([0, 0, zc - half]);
            const sp = res?.meta.slab_profile;
            if (sp && sp.z.length) {
                // the actual flip-angle profile of the pulse: opacity ∝ flip / nominal flip (the dashed box is the 50 % level)
                const fmax = Math.max(...sp.flip_deg, 1e-9);
                for (let q = 0; q < sp.z.length; q++) {
                    const [, ya] = toScr([0, 0, iso[2] + (sp.z[q] + 1e-3) / sz]), [, yb] = toScr([0, 0, iso[2] + (sp.z[q] - 1e-3) / sz]);
                    mctx.fillStyle = `rgba(232,192,74,${(0.42 * sp.flip_deg[q] / fmax).toFixed(3)})`;
                    mctx.fillRect(lay.x0, ya, lay.wmm * lay.s, Math.max(1, yb - ya));
                }
            } else { mctx.fillStyle = 'rgba(232,192,74,0.07)'; mctx.fillRect(lay.x0, y0, lay.wmm * lay.s, y1 - y0); }
            mctx.strokeStyle = 'rgba(232,192,74,0.6)'; mctx.setLineDash([6, 4]); mctx.strokeRect(lay.x0, y0, lay.wmm * lay.s, y1 - y0); mctx.setLineDash([]);
            mctx.fillStyle = 'rgba(232,192,74,0.8)'; mctx.font = '11px sans-serif';
            mctx.fillText(sp ? 'slab: shading = flip angle, dashed = 50 % edges' : 'slab: 50 % edges (run for the profile)', lay.x0 + 4, y0 - 4);
        }
        // start marker
        if (center) {
            const [x, y] = toScr(center);
            mctx.strokeStyle = '#ffb347'; mctx.lineWidth = 1;
            mctx.beginPath(); mctx.moveTo(x - 9, y); mctx.lineTo(x + 9, y); mctx.moveTo(x, y - 9); mctx.lineTo(x, y + 9); mctx.stroke();
        }
        if (res) {
            // path of the parcel centre (all samples), the part played so far, the spins and the mean
            const vox = (p) => [0, 1, 2].map((a) => p[a] / spacing[a] + res.meta.iso[a]);
            mctx.lineWidth = 1.5;
            mctx.strokeStyle = 'rgba(120,160,200,0.45)'; mctx.beginPath();
            for (let q = 0; q < res.n_out; q += 1) { const [x, y] = toScr(vox(res.meanPos[q])); q ? mctx.lineTo(x, y) : mctx.moveTo(x, y); }
            mctx.stroke();
            mctx.strokeStyle = '#ffb347'; mctx.beginPath();
            for (let q = 0; q <= k; q += 1) { const [x, y] = toScr(vox(res.meanPos[q])); q ? mctx.lineTo(x, y) : mctx.moveTo(x, y); }
            mctx.stroke();
            const N = res.N, base = k * N * 3;
            mctx.fillStyle = 'rgba(120,230,255,0.55)';
            for (let j = 0; j < N; j++) {
                const [x, y] = toScr(vox([res.pos[base + 3 * j], res.pos[base + 3 * j + 1], res.pos[base + 3 * j + 2]]));
                mctx.fillRect(x - 1, y - 1, 2, 2);
            }
            const [x, y] = toScr(vox(res.meanPos[k]));
            mctx.strokeStyle = '#fff'; mctx.lineWidth = 2; mctx.beginPath(); mctx.arc(x, y, 7, 0, 2 * Math.PI); mctx.stroke();
        }
        $('mip-info').textContent = `${['sagittal (along x)', 'coronal (along y)', 'axial (along z)'][axisSel]}; click to place the parcel${center ? `; start voxel ${center.map((c) => c.toFixed(0)).join(', ')}` : ''}`;
    }
    mipCanvas.addEventListener('click', (ev) => {
        if (!pcm) return;
        const r = mipCanvas.getBoundingClientRect();
        const [vu, vv] = fromScr((ev.clientX - r.left) * mipCanvas.width / r.width, (ev.clientY - r.top) * mipCanvas.height / r.height);
        const m = mipOf(axisSel), kf = pcm.factor;
        const iu = Math.floor(vu / kf), iv = Math.floor(vv / kf);
        if (iu < 0 || iv < 0 || iu >= m.nu || iv >= m.nv) return;
        const c = [0, 0, 0];
        c[m.ua] = vu; c[m.va] = vv;
        const hit = m.img[iu + m.nu * iv];
        // no vessel on the ray: the MIP knows no depth, so use the middle of the volume (the parcel is then in tissue, not in a vessel)
        c[axisSel] = hit ? m.depth[iu + m.nu * iv] * kf + kf / 2 : info.dims[axisSel] / 2;
        setCenter(c, hit ? `vessel on the ray (MIP value ${hit}/255) at depth ${c[axisSel].toFixed(0)}: the parcel starts inside it and flows` : `no vessel on the ray: depth set to the volume middle (${c[axisSel].toFixed(0)}), parcel in tissue. Set x, y, z by hand to choose the depth.`);
        draw();
    });
    $('mip-axis').addEventListener('change', () => { axisSel = Number($('mip-axis').value); draw(); });
    function setCenter(c, note) {
        center = c;
        ['cx', 'cy', 'cz'].forEach((id, a) => { $(id).value = Math.round(c[a]); });
        $('ray-info').textContent = note || '';
    }
    for (const [a, id] of ['cx', 'cy', 'cz'].entries()) $(id).addEventListener('input', () => {
        if (!center) center = [0, 0, 0];
        center = center.slice(); center[a] = Number($(id).value);
        $('ray-info').textContent = 'set by hand';
        draw();
    });
    if (center) setCenter(center, 'brightest PC-MRA voxel');
    $('t0').addEventListener('input', () => { $('t0-label').textContent = `frame ${$('t0').value} of ${info.frames} (${(Number($('t0').value) * info.dt * 1e3).toFixed(0)} ms)`; });
    if (info) $('t0-label').textContent = `frame 0 of ${info.frames}`;

    // ── three.js: the spin ensemble ──────────────────────────────────────────────────────────────────────────
    const canvas = $('canvas');
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: params.get('selftest') === '1' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101214);
    const camera = new THREE.PerspectiveCamera(35, 1, 0.05, 50);
    camera.up.set(0, 0, 1);
    camera.position.set(2.6, -3.1, 1.9);
    const controls = new OrbitControls(camera, canvas);
    controls.target.set(0, 0, 0.1);
    controls.update();
    const sph = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 20), new THREE.MeshBasicMaterial({ color: 0x4a6a8a, wireframe: true, transparent: true, opacity: 0.1 }));
    scene.add(sph);
    const ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array.from({ length: 128 }, (_, i) => new THREE.Vector3(Math.cos(i / 128 * 2 * Math.PI), Math.sin(i / 128 * 2 * Math.PI), 0))), new THREE.LineBasicMaterial({ color: 0x6a8aa8, transparent: true, opacity: 0.5 }));
    scene.add(ring);
    const axes = [[1, 0, 0, 0xd05050], [0, 1, 0, 0x50c070], [0, 0, 1, 0x5080e0]];
    for (const [x, y, z, c] of axes) scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-x * 1.15, -y * 1.15, -z * 1.15 * 0), new THREE.Vector3(x * 1.15, y * 1.15, z * 1.15)]), new THREE.LineBasicMaterial({ color: c })));
    const rfAxis = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), new THREE.Vector3(1.15, 0, 0)]), new THREE.LineBasicMaterial({ color: 0xffd23a }));
    rfAxis.visible = false; scene.add(rfAxis);
    let spinLines = null, spinTips = null;
    const mean = new THREE.Group();
    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.014, 0.014, 1, 12).translate(0, 0.5, 0), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    const head = new THREE.Mesh(new THREE.ConeGeometry(0.045, 1, 16).translate(0, 0.5, 0), new THREE.MeshBasicMaterial({ color: 0xffffff }));
    mean.add(shaft, head);
    scene.add(mean);
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(201 * 3), 3));
    const trail = new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ color: 0xffb347 }));
    trail.frustumCulled = false;
    scene.add(trail);
    function setMean(v) {
        const L = Math.hypot(v[0], v[1], v[2]);
        mean.visible = L > 1e-4;
        if (!mean.visible) return;
        const dir = new THREE.Vector3(v[0], v[1], v[2]).normalize();
        mean.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir);
        const hl = Math.min(0.1, L * 0.45);
        shaft.scale.set(1, L - hl, 1);
        head.position.set(0, L - hl, 0);
        head.scale.set(1, hl, 1);
    }
    function resize() {
        for (const [c, p] of [[canvas, $('view')], [$('scope'), $('scope')], [mipCanvas, $('mip-wrap')], [$('timeline'), $('bottom')], [$('history'), $('bottom')]]) {
            const w = Math.max(10, Math.floor(c === canvas ? p.clientWidth : c.clientWidth || p.clientWidth)), h = Math.max(10, Math.floor(c.clientHeight || p.clientHeight));
            if (c === canvas) { renderer.setSize(p.clientWidth, p.clientHeight, false); camera.aspect = p.clientWidth / Math.max(1, p.clientHeight); camera.updateProjectionMatrix(); continue; }
            const dpr = Math.min(window.devicePixelRatio || 1, 2);
            if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
        }
        draw();
    }

    // ── result ───────────────────────────────────────────────────────────────────────────────────────────────
    let res = null, k = 0, playing = false, last = 0, simT = 0;
    function buildSpins(N) {
        if (spinLines) { scene.remove(spinLines, spinTips); spinLines.geometry.dispose(); spinTips.geometry.dispose(); }
        const lp = new Float32Array(N * 6), lc = new Float32Array(N * 6), tp = new Float32Array(N * 3), tc = new Float32Array(N * 3);
        const lg = new THREE.BufferGeometry();
        lg.setAttribute('position', new THREE.BufferAttribute(lp, 3)); lg.setAttribute('color', new THREE.BufferAttribute(lc, 3));
        const tg = new THREE.BufferGeometry();
        tg.setAttribute('position', new THREE.BufferAttribute(tp, 3)); tg.setAttribute('color', new THREE.BufferAttribute(tc, 3));
        spinLines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.85 }));
        spinTips = new THREE.Points(tg, new THREE.PointsMaterial({ size: 4, sizeAttenuation: false, vertexColors: true }));
        spinLines.frustumCulled = spinTips.frustumCulled = false;
        scene.add(spinLines, spinTips);
    }
    const tOf = (q) => (res.out_tr[q] * res.meta.n_steps + res.out_step[q]) * res.meta.dt;
    function vecAt(q, j, c, s) {
        // the frame rotated by the RF phase φ (c = cos φ, s = sin φ): x' = x c + y s, y' = −x s + y c
        const o = (q * res.N + j) * 3, x = res.m[o], y = res.m[o + 1];
        return [x * c + y * s, -x * s + y * c, res.m[o + 2]];
    }
    function phaseRot(q) {
        if ($('frame').value !== 'rf') return [1, 0];
        const ph = res.meta.rf_phase[res.out_tr[q]] ?? 0;
        return [Math.cos(ph), Math.sin(ph)];
    }
    function updateScene() {
        if (!res) return;
        const N = res.N, [c, s] = phaseRot(k);
        const lp = spinLines.geometry.attributes.position.array, lc = spinLines.geometry.attributes.color.array;
        const tp = spinTips.geometry.attributes.position.array, tc = spinTips.geometry.attributes.color.array;
        const comp = $('comp').value, mode = $('color').value, show = $('show-spins').checked;
        let mx = 0, my = 0, mz = 0, sumAbs = 0, mxy = 0;
        for (let j = 0; j < N; j++) {
            let [x, y, z] = vecAt(k, j, c, s);
            mx += x; my += y; mz += z; sumAbs += Math.hypot(x, y, z); mxy += Math.hypot(x, y);
            if (comp === 'xy') z = 0; else if (comp === 'z') { x = 0; y = 0; }
            lp[6 * j] = lp[6 * j + 1] = lp[6 * j + 2] = 0;
            lp[6 * j + 3] = tp[3 * j] = x; lp[6 * j + 4] = tp[3 * j + 1] = y; lp[6 * j + 5] = tp[3 * j + 2] = z;
            let col;
            if (mode === 'plain') col = new THREE.Color(0x8fd0ff);
            else if (mode === 'offset') col = hsl(res.offs[j] * 0.8, 0.8, 0.55);
            else { const r = Math.hypot(x, y); col = r < 1e-5 ? new THREE.Color(0x555555) : hsl(Math.atan2(y, x) / (2 * Math.PI), Math.min(1, 0.25 + 40 * r), 0.55); }
            lc[6 * j] = col.r * 0.35; lc[6 * j + 1] = col.g * 0.35; lc[6 * j + 2] = col.b * 0.35;
            lc[6 * j + 3] = tc[3 * j] = col.r; lc[6 * j + 4] = tc[3 * j + 1] = col.g; lc[6 * j + 5] = tc[3 * j + 2] = col.b;
        }
        spinLines.visible = spinTips.visible = show;
        spinLines.geometry.attributes.position.needsUpdate = spinLines.geometry.attributes.color.needsUpdate = true;
        spinTips.geometry.attributes.position.needsUpdate = spinTips.geometry.attributes.color.needsUpdate = true;
        mx /= N; my /= N; mz /= N;
        const rfAng = $('frame').value === 'rf' ? 0 : (res.meta.rf_phase[res.out_tr[k]] ?? 0);
        rfAxis.visible = true; rfAxis.rotation.z = rfAng;
        setMean(comp === 'xy' ? [mx, my, 0] : comp === 'z' ? [0, 0, mz] : [mx, my, mz]);
        drawScope(c, s);
        // trail of the mean tip
        const nt = Math.min(num('trail'), k), tpos = trail.geometry.attributes.position.array;
        for (let q = 0; q <= nt; q++) {
            const [cc, ss] = phaseRot(k - nt + q); let x = 0, y = 0, z = 0;
            for (let j = 0; j < N; j++) { const v = vecAt(k - nt + q, j, cc, ss); x += v[0]; y += v[1]; z += v[2]; }
            tpos[3 * q] = x / N; tpos[3 * q + 1] = y / N; tpos[3 * q + 2] = z / N;
        }
        trail.geometry.setDrawRange(0, nt > 0 ? nt + 1 : 0);
        trail.geometry.attributes.position.needsUpdate = true;
        // readout
        const m = res.meta, tr = res.out_tr[k], st = res.out_step[k];
        const part = (partsOf(m).find(([a, b]) => st >= a && st < b) ?? partsOf(m)[5])[2];
        const e = m.echo[tr] ?? [0, 0];
        let slabTxt = '';
        if (m.slab_profile?.z.length && res.meanPos) {
            const pz = res.meanPos[k][m.slab.axis], z = m.slab_profile.z, f = m.slab_profile.flip_deg;
            const t = (pz - z[0]) / (z[1] - z[0]);
            const fl = t < 0 || t > z.length - 1 ? 0 : f[Math.floor(t)] + (f[Math.min(z.length - 1, Math.floor(t) + 1)] - f[Math.floor(t)]) * (t - Math.floor(t));
            slabTxt = `\nslab flip at the parcel centre (z ${(pz * 1e3).toFixed(0)} mm): ${fl.toFixed(2)}° of ${m.flip_deg.toFixed(0)}°`;
        }
        $('readout').textContent =
            `TR ${tr}${tr === 0 ? ' (first)' : ''}   encode ${m.encode[tr]}   rotation ${m.rot[tr]}\n` +
            `t = ${(tOf(k) * 1e3).toFixed(3)} ms   step ${st}/${m.n_steps}   ${part}\n` +
            `mean  M⊥ ${Math.hypot(mx, my).toFixed(4)}   Mz ${mz.toFixed(4)}   phase ${(Math.atan2(my, mx) * 180 / Math.PI).toFixed(0)}°\n` +
            `coherence |⟨M⟩| / ⟨|M|⟩ ${(Math.hypot(mx, my, mz) / Math.max(1e-12, sumAbs / N)).toFixed(3)}   ⟨|M⊥|⟩ ${(mxy / N).toFixed(4)}\n` +
            `echo of this TR |S| ${Math.hypot(e[0], e[1]).toFixed(4)}  (Ernst ${m.ernst.toFixed(4)})` + slabTxt;
    }

    // ── timeline of the played TR and history over all TRs ──────────────────────────────────────────────────
    const C = (n) => { const ct = $(n).getContext('2d'); return [ct, $(n).width, $(n).height]; };
    // regions of the TR (raster steps): label, colour
    function partsOf(m) {
        const [ra, rb] = m.rf_steps, sel = !!m.slab;
        return [[0, ra, 'dead time', '#1a1a1a'], [ra, rb, sel ? 'RF + slice select' : 'RF', '#3a2a1a'], [rb, m.i_ro, 'rephase + encode', '#16222c'],
            [m.i_ro, m.i_peak, 'readout out', '#1c2e1c'], [m.i_peak, m.i_tail, 'readout in', '#2c2c16'], [m.i_tail, m.n_steps, 'spoiler', '#2a1c28']];
    }
    function timelineRange(m) {
        const z = $('tl-zoom').value, us = (x) => Math.round(x * 1e-6 / m.dt);
        if (z === 'rf') return [0, m.rf_steps[1] + us(400)];
        if (z === 'enc') return [m.rf_steps[1], m.i_ro];
        if (z === 'out') return [m.i_ro, m.i_peak];
        if (z === 'in') return [m.i_peak, m.i_tail];
        if (z === 'spoil') return [m.i_tail, m.n_steps];
        return [0, m.n_steps];
    }
    function drawTimeline() {
        const [g, W, H] = C('timeline');
        g.fillStyle = '#0e0e0e'; g.fillRect(0, 0, W, H);
        if (!res) { g.fillStyle = '#666'; g.font = `${12 * (W / $('timeline').clientWidth || 1)}px sans-serif`; g.fillText('press Simulate', 12, 22); return; }
        const m = res.meta, tr = res.out_tr[k], dpr = W / $('timeline').clientWidth;
        const L = 56 * dpr, R = W - 8 * dpr, T = 6 * dpr, B = H - 4 * dpr;
        const [s0, s1] = timelineRange(m);
        const X = (s) => L + (R - L) * (s - s0) / Math.max(1, s1 - s0);
        const lanes = [['RF', 0.0, 0.2], ['Gx', 0.2, 0.4], ['Gy', 0.4, 0.6], ['Gz', 0.6, 0.8], ['⟨M⟩', 0.8, 1.0]];
        g.font = `${11 * dpr}px sans-serif`;
        g.save(); g.beginPath(); g.rect(L, T, R - L, B - T); g.clip();
        for (const [a, b2, name, c] of partsOf(m)) {
            if (b2 <= s0 || a >= s1) continue;
            g.fillStyle = c; g.fillRect(X(a), T, Math.max(1, X(b2) - X(a)), B - T);
            g.strokeStyle = '#333'; g.beginPath(); g.moveTo(X(a), T); g.lineTo(X(a), B); g.stroke();
            if (X(b2) - X(a) > (name.length * 6.2 + 8) * dpr) { g.fillStyle = '#9a9a9a'; g.fillText(name, Math.max(X(a), L) + 4 * dpr, T + 10 * dpr); }
        }
        // gradient scale over the visible part
        const ng = m.n_g, gbase = tr * ng * 3;
        const mxg = [1e-9, 1e-9, 1e-9];      // per axis, over the visible range
        for (let q = 0; q < ng; q++) { const s = q * m.g_stride; if (s < s0 || s > s1) continue; for (let a = 0; a < 3; a++) mxg[a] = Math.max(mxg[a], Math.abs(res.g[gbase + 3 * q + a])); }
        const labels = [];
        lanes.forEach(([n, a, b2], i) => {
            const y0 = T + (B - T) * a, y1 = T + (B - T) * b2, ym = (y0 + y1) / 2;
            labels.push([n, ym, i === 0 && m.slab ? `G${'xyz'[m.slab.axis]} ${(m.slab.g_ss * 1e3).toFixed(2)} mT/m` : i >= 1 && i <= 3 ? `±${(mxg[i - 1] * 1e3).toFixed(mxg[i - 1] < 0.0095 ? 2 : 1)} mT/m` : '']);
            g.strokeStyle = '#2a2a2a'; g.beginPath(); g.moveTo(L, ym); g.lineTo(R, ym); g.stroke();
            if (i === 0) {
                // the RF envelope of the slab pulse (sinc) / the hard pulse, to scale
                const [ra, rb] = m.rf_steps, env = m.rf_env;
                g.fillStyle = '#e8c04a'; g.beginPath(); g.moveTo(X(ra), ym);
                const nn = env.length;
                for (let q = 0; q <= nn; q++) g.lineTo(X(ra + q), ym - (y1 - y0) * 0.42 * (env[Math.min(q, nn - 1)] ?? 1));
                g.lineTo(X(rb), ym); g.closePath(); g.fill();
                if (m.slab) {
                    // the slice-select gradient (slab axis) over the pulse and the gap before the encode block, to its own scale
                    const ax = m.slab.axis; let gm = 1e-12;
                    for (let q = 0; q < ng; q++) if (q * m.g_stride <= m.rf_steps[1]) gm = Math.max(gm, Math.abs(res.g[gbase + 3 * q + ax]));
                    g.strokeStyle = '#6fd'; g.lineWidth = 1.5 * dpr; g.beginPath();
                    for (let q = 0; q < ng && q * m.g_stride <= m.rf_steps[1] + Math.round(30e-6 / m.dt); q++) { const x = X(q * m.g_stride), y = ym - (res.g[gbase + 3 * q + ax] / gm) * (y1 - y0) * 0.3; q ? g.lineTo(x, y) : g.moveTo(x, y); }
                    g.stroke(); g.lineWidth = 1;
                }
            } else if (i <= 3) {
                const ax = i - 1;
                g.strokeStyle = ['#d05050', '#50c070', '#5080e0'][ax]; g.beginPath();
                for (let q = 0; q < ng; q++) { const x = X(q * m.g_stride), y = ym - (res.g[gbase + 3 * q + ax] / mxg[ax]) * (y1 - y0) * 0.45; q ? g.lineTo(x, y) : g.moveTo(x, y); }
                g.stroke();
            } else {
                // ensemble mean over the TR: M_z (orange, 0..1) and |M⊥| (cyan, own scale)
                const np = m.n_mean_per_tr, base = tr * np * 3;
                let mxp = 1e-6; for (let q = 0; q < np; q++) mxp = Math.max(mxp, Math.hypot(res.mean[base + 3 * q], res.mean[base + 3 * q + 1]), res.minc[tr * np + q]);
                g.strokeStyle = '#ffb347'; g.beginPath();
                for (let q = 0; q < np; q++) { const x = X(q * m.mean_stride), y = y1 - res.mean[base + 3 * q + 2] * (y1 - y0); q ? g.lineTo(x, y) : g.moveTo(x, y); }
                g.stroke();
                g.strokeStyle = '#5ad'; g.beginPath();
                for (let q = 0; q < np; q++) { const x = X(q * m.mean_stride), y = y1 - Math.hypot(res.mean[base + 3 * q], res.mean[base + 3 * q + 1]) / mxp * (y1 - y0); q ? g.lineTo(x, y) : g.moveTo(x, y); }
                g.stroke();
                const bi = tr * np;
                g.strokeStyle = '#ffb347'; g.setLineDash([3 * dpr, 3 * dpr]); g.beginPath();
                for (let q = 0; q < np; q++) { const x = X(q * m.mean_stride), y = y1 - res.minc[bi + q] / mxp * (y1 - y0); q ? g.lineTo(x, y) : g.moveTo(x, y); }
                g.stroke(); g.setLineDash([]);
                g.fillStyle = '#5ad'; g.fillText(`M⊥ (max ${mxp.toFixed(3)})`, R - 90 * dpr, y0 + 12 * dpr);
                g.fillStyle = '#ffb347'; g.fillText('Mz 0–1', R - 90 * dpr, y0 + 24 * dpr);
                g.fillText('dashed: per-spin |M⊥| (no spread loss)', L + 120 * dpr, y0 + 12 * dpr);
            }
        });
        const xc = X(res.out_step[k]);
        g.strokeStyle = '#fff'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(xc, T); g.lineTo(xc, B); g.stroke(); g.lineWidth = 1;
        g.restore();
        g.fillStyle = '#777';
        for (const [n, ym, sc] of labels) { g.fillStyle = '#9a9a9a'; g.fillText(n, 6 * dpr, ym + 1 * dpr); g.fillStyle = '#666'; g.font = `${9.5 * dpr}px sans-serif`; g.fillText(sc, 6 * dpr, ym + 12 * dpr); g.font = `${11 * dpr}px sans-serif`; }
        g.fillStyle = '#777';
        g.fillText(`TR ${tr}: ${(m.tr * 1e3).toFixed(2)} ms · window ${(s0 * m.dt * 1e3).toFixed(2)}–${(s1 * m.dt * 1e3).toFixed(2)} ms · each gradient lane has its own scale`, L + 4 * dpr, B - 2 * dpr);
    }
    $('tl-zoom').addEventListener('input', () => draw());
    function drawHistory() {
        const [g, W, H] = C('history');
        g.fillStyle = '#0c0c0c'; g.fillRect(0, 0, W, H);
        if (!res) return;
        const m = res.meta, dpr = W / $('history').clientWidth, L = 56 * dpr, R = W - 8 * dpr, T = 8 * dpr, B = H - 16 * dpr;
        const nt = m.n_tr, X = (n) => L + (R - L) * (n + 0.5) / nt;
        const S = m.echo.map((e) => Math.hypot(e[0], e[1]));
        const smax = Math.max(m.ernst * 1.15, ...S, ...m.echo_inc.map((e) => e[0])) || 1;
        g.font = `${11 * dpr}px sans-serif`;
        g.strokeStyle = '#444'; g.setLineDash([4 * dpr, 4 * dpr]); g.beginPath(); const ye = B - m.ernst / smax * (B - T); g.moveTo(L, ye); g.lineTo(R, ye); g.stroke(); g.setLineDash([]);
        g.fillStyle = '#777'; g.fillText('Ernst', 6 * dpr, ye + 4 * dpr);
        g.strokeStyle = '#5ad'; g.beginPath();
        S.forEach((s, n) => { const x = X(n), y = B - s / smax * (B - T); n ? g.lineTo(x, y) : g.moveTo(x, y); });
        g.stroke();
        // M_z of the mean before each TR's RF pulse
        const np = m.n_mean_per_tr;
        g.strokeStyle = '#ffb347'; g.beginPath();
        for (let n = 0; n < nt; n++) { const x = X(n), y = B - res.mean[n * np * 3 + 2] * (B - T); n ? g.lineTo(x, y) : g.moveTo(x, y); }
        g.stroke();
        g.strokeStyle = '#6c6'; g.setLineDash([3 * dpr, 3 * dpr]); g.beginPath();
        m.echo_inc.forEach((e, n) => { const x = X(n), y = B - e[0] / smax * (B - T); n ? g.lineTo(x, y) : g.moveTo(x, y); });
        g.stroke(); g.setLineDash([]);
        g.fillStyle = '#6c6'; g.fillText('mean per-spin |M⊥| at echo (no spread loss)', L + 6 * dpr, T + 24 * dpr);
        g.fillStyle = '#5ad'; g.fillText('|echo| per TR (demodulated, vector sum)', L + 6 * dpr, T + 10 * dpr);
        g.fillStyle = '#ffb347'; g.fillText('⟨Mz⟩ before the RF', L + 300 * dpr, T + 10 * dpr);
        g.fillStyle = '#777'; g.fillText(`TR 0`, L, H - 3 * dpr); g.fillText(`${nt}`, R - 20 * dpr, H - 3 * dpr);
        const xc = X(res.out_tr[k]);
        g.strokeStyle = '#fff'; g.beginPath(); g.moveTo(xc, T); g.lineTo(xc, B); g.stroke();
        res.histL = L; res.histR = R; res.histNt = nt;
    }
    $('history').addEventListener('click', (ev) => {
        if (!res) return;
        const r = $('history').getBoundingClientRect(), px = (ev.clientX - r.left) * $('history').width / r.width;
        const n = Math.max(0, Math.min(res.histNt - 1, Math.floor((px - res.histL) / (res.histR - res.histL) * res.histNt)));
        const q = res.out_tr.findIndex((t) => t === n);
        if (q >= 0) seek(q);
    });

    const jit = Array.from({ length: 4000 }, () => Math.random());
    function drawScope(c, s) {
        const cv = $('scope'); cv.style.display = $('scope-on').checked ? '' : 'none';
        if (!res || !$('scope-on').checked) return;
        const g = cv.getContext('2d'), W = cv.width, H = cv.height, dpr = W / cv.clientWidth, N = res.N;
        g.fillStyle = '#0a0c0e'; g.fillRect(0, 0, W, H);
        g.font = `${11 * dpr}px sans-serif`;
        let mxp = 0.05, mx = 0, my = 0, mz = 0, inc = 0;
        for (let j = 0; j < N; j++) { const [x, y, z] = vecAt(k, j, c, s); mxp = Math.max(mxp, Math.hypot(x, y)); mx += x; my += y; mz += z; inc += Math.hypot(x, y); }
        mx /= N; my /= N; mz /= N; inc /= N;
        const R = Math.min(1, mxp * 1.15), cx = W * 0.27, cy = H * 0.52, rr = Math.min(W * 0.24, H * 0.4);
        g.strokeStyle = '#2a3a4a'; g.beginPath(); g.arc(cx, cy, rr, 0, 7); g.stroke();
        g.beginPath(); g.moveTo(cx - rr, cy); g.lineTo(cx + rr, cy); g.moveTo(cx, cy - rr); g.lineTo(cx, cy + rr); g.stroke();
        g.fillStyle = '#6a7a8a'; g.fillText(`transverse plane, edge = ${R.toFixed(3)}`, 6 * dpr, 13 * dpr);
        const mode = $('color').value;
        for (let j = 0; j < N; j++) {
            const [x, y] = vecAt(k, j, c, s);
            const col = mode === 'offset' ? hsl(res.offs[j] * 0.8, 0.8, 0.55) : mode === 'plain' ? new THREE.Color(0x8fd0ff) : hsl(Math.atan2(y, x) / (2 * Math.PI), 0.8, 0.55);
            g.fillStyle = `rgb(${col.r * 255 | 0},${col.g * 255 | 0},${col.b * 255 | 0})`;
            g.fillRect(cx + x / R * rr - 1.5 * dpr, cy - y / R * rr - 1.5 * dpr, 3 * dpr, 3 * dpr);
        }
        const ra = $('frame').value === 'rf' ? 0 : (res.meta.rf_phase[res.out_tr[k]] ?? 0);
        g.strokeStyle = '#ffd23a'; g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + rr * 1.1 * Math.cos(ra), cy - rr * 1.1 * Math.sin(ra)); g.stroke();
        g.strokeStyle = '#fff'; g.lineWidth = 2 * dpr; g.beginPath(); g.moveTo(cx, cy); g.lineTo(cx + mx / R * rr, cy - my / R * rr); g.stroke(); g.lineWidth = 1;
        // mean of the lengths: a circle of that radius (the vector sum is always inside it; the gap is dephasing)
        g.strokeStyle = '#ffb347'; g.setLineDash([3 * dpr, 3 * dpr]); g.beginPath(); g.arc(cx, cy, inc / R * rr, 0, 7); g.stroke(); g.setLineDash([]);
        g.fillStyle = '#ffb347'; g.fillText(`dashed: mean per-spin |M⊥| ${inc.toFixed(3)}`, 6 * dpr, H - 20 * dpr);
        g.fillStyle = '#fff'; g.fillText(`white: |mean M⊥| ${Math.hypot(mx, my).toFixed(3)}   yellow: RF axis`, 6 * dpr, H - 6 * dpr);
        // longitudinal panel: Mz of every spin (x = spin index, shuffled), mean line
        const x0 = W * 0.58, x1 = W - 10 * dpr, y0 = 20 * dpr, y1 = H - 10 * dpr, Y = (z) => y1 - (z + 1) / 2 * (y1 - y0);
        g.strokeStyle = '#2a3a4a'; g.strokeRect(x0, y0, x1 - x0, y1 - y0);
        g.fillStyle = '#6a7a8a'; g.fillText('Mz of every spin (−1 … 1)', x0, 13 * dpr);
        g.beginPath(); g.moveTo(x0, Y(0)); g.lineTo(x1, Y(0)); g.stroke();
        g.fillStyle = '#5080e0';
        for (let j = 0; j < N; j++) { const z = vecAt(k, j, c, s)[2]; g.fillRect(x0 + jit[j % 4000] * (x1 - x0 - 3 * dpr), Y(z) - 1.5 * dpr, 3 * dpr, 3 * dpr); }
        g.strokeStyle = '#fff'; g.beginPath(); g.moveTo(x0, Y(mz)); g.lineTo(x1, Y(mz)); g.stroke();
        g.fillStyle = '#fff'; g.fillText(`⟨Mz⟩ ${mz.toFixed(3)}`, x0 + 4 * dpr, Y(mz) + (mz > 0.5 ? 14 : -4) * dpr);
    }
    $('scope-on').addEventListener('input', () => draw());
    $('comp').addEventListener('input', () => draw());
    $('cam').addEventListener('input', () => {
        const v = $('cam').value, d = camera.position.distanceTo(controls.target);
        if (v === 'top') camera.position.set(0, -0.001, d); else if (v === 'side') camera.position.set(0, -d, controls.target.z); else camera.position.set(2.6, -3.1, 1.9);
        camera.up.set(0, v === 'top' ? 1 : 0, v === 'top' ? 0 : 1); camera.lookAt(controls.target); controls.update();
    });
    function draw() { updateScene(); drawMip(k); drawTimeline(); drawHistory(); }
    function seek(q) {
        k = Math.max(0, Math.min(res.n_out - 1, q));
        $('pos').value = k;
        simT = tOf(k);
        $('pos-label').textContent = `${k + 1} / ${res.n_out}  (TR ${res.out_tr[k]})`;
        draw();
    }
    $('pos').addEventListener('input', () => seek(Number($('pos').value)));
    for (const i of ['color', 'frame', 'trail', 'show-spins']) $(i).addEventListener('input', () => draw());
    $('play').addEventListener('click', () => { playing = !playing; $('play').textContent = playing ? '❚❚' : '▶'; last = performance.now(); if (playing) requestAnimationFrame(tick); });
    window.addEventListener('keydown', (ev) => { if (ev.code === 'Space' && !/INPUT|SELECT/.test(ev.target.tagName)) { ev.preventDefault(); $('play').click(); } });
    let strobeAcc = 0;
    function tick(now) {
        if (!playing || !res) return;
        if ($('strobe').checked) {
            strobeAcc += (now - last) * 1e-3 * num('strobe-rate'); last = now;
            while (strobeAcc >= 1) { strobeAcc -= 1; if (!seekTR(1)) { const f = res.out_tr.findIndex((t) => t === 0); let b = f, bd = 1e9; for (let q = 0; q < res.n_out; q++) if (res.out_tr[q] === 0 && Math.abs(res.out_step[q] - res.out_step[k]) < bd) { bd = Math.abs(res.out_step[q] - res.out_step[k]); b = q; } seek(b); } }
            simT = tOf(k);
            requestAnimationFrame(tick); return;
        }
        simT += (now - last) * 1e-3 * num('rate') * 1e-3; last = now;
        const tEnd = tOf(res.n_out - 1);
        if ($('loop-tr').checked) {
            const tr = res.out_tr[k]; let a = k, b = k;
            while (a > 0 && res.out_tr[a - 1] === tr) a--;
            while (b < res.n_out - 1 && res.out_tr[b + 1] === tr) b++;
            if (simT > tOf(b) || simT < tOf(a)) simT = tOf(a);
        }
        if (simT > tEnd) simT = 0;
        let lo = 0, hi = res.n_out - 1;
        while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (tOf(mid) <= simT) lo = mid; else hi = mid - 1; }
        if (lo !== k) { k = lo; $('pos').value = k; $('pos-label').textContent = `${k + 1} / ${res.n_out}  (TR ${res.out_tr[k]})`; draw(); }
        requestAnimationFrame(tick);
    }

    function seekTR(d) {
        if (!res) return;
        const t = res.out_tr[k] + d, st = res.out_step[k];
        if (t < 0 || t >= res.meta.n_tr) return false;
        let best = -1, bd = 1e9;
        for (let q = 0; q < res.n_out; q++) if (res.out_tr[q] === t) { const dd = Math.abs(res.out_step[q] - st); if (dd < bd) { bd = dd; best = q; } }
        if (best < 0) return false;
        seek(best); return true;
    }
    $('tr-b').addEventListener('click', () => seekTR(-1));
    $('tr-f').addEventListener('click', () => seekTR(1));
    $('step-b').addEventListener('click', () => seek(k - 1));
    $('step-f').addEventListener('click', () => seek(k + 1));
    window.addEventListener('keydown', (ev) => { if (!/INPUT|SELECT/.test(ev.target.tagName) && res) { if (ev.code === 'ArrowLeft') seek(k - 1); if (ev.code === 'ArrowRight') seek(k + 1); if (ev.code === 'PageUp') { ev.preventDefault(); seekTR(-1); } if (ev.code === 'PageDown') { ev.preventDefault(); seekTR(1); } } });
    $('zoom').addEventListener('click', async () => {
        const tr = res.out_tr[k];
        $('d-a').value = tr; $('d-b').value = tr + 1; $('d-s').value = 1;
        await run();
        const q = res?.out_tr.findIndex((t, i) => t === tr && res.out_step[i] === 0);
        if (q >= 0) seek(q);
    });

    // ── simulate ─────────────────────────────────────────────────────────────────────────────────────────────
    async function run() {
        const o = {
            design: $('design').value, start: $('start').value, n_tr: num('n-tr'), per_tr: num('per-tr'),
            flip_deg: num('flip'), rf_inc_deg: num('rfinc'), t1: num('t1'), t2: num('t2'), df_hz: num('df'), b1_scale: num('b1s'),
            concomitant: $('conc').checked, left_handed: $('lh').checked,
            dt_us: num('dt-us'), rf: $('rf').value, slab_thk_mm: num('slab-thk'), slab_center_mm: num('slab-c'), tbw: num('tbw'), rf_dur_us: num('rf-dur'),
        };
        if ($('start').value === 'steady') o.preroll = num('preroll');
        if (num('d-b') > num('d-a')) o.detail = [num('d-a'), Math.min(num('d-b'), o.n_tr), Math.max(1, Math.round(num('d-s') / num('dt-us')))];
        let inputs = [];
        if (id) {
            if (!center) { status('click the vessel MIP first'); return; }
            Object.assign(o, { center, n_spins: num('n-spins'), radius_mm: num('radius'), advect: $('advect').value, t0: num('t0') });
            inputs = [id];
        } else {
            Object.assign(o, { center: [num('px'), num('py'), num('pz')], velocity: [num('vx'), num('vy'), num('vz')], n_spins: num('n-spins-p'), radius_mm: num('radius-p') });
        }
        $('run').disabled = true;
        const t0 = performance.now();
        const timer = setInterval(() => { $('progress').textContent = `simulating… ${((performance.now() - t0) / 1e3).toFixed(0)} s`; }, 500);
        try {
            const r = await client.execute('bloch_parcel', o, inputs, 'bloch_parcel (viewer)');
            r.outputs.forEach((x) => temps.add(x));
            const meta = JSON.parse(r.message);
            $('progress').textContent = 'downloading…';
            const [m, p, mean, g, mi] = await Promise.all(r.outputs.map((x) => client.getTensor(x)));
            r.outputs.forEach((x) => { client.delete(x).catch(() => {}); temps.delete(x); });
            const f = (t) => (t.data instanceof Float32Array ? t.data : Float32Array.from(t.data));
            const N = meta.n_spins, nOut = meta.n_out;
            const pos = f(p), meanPos = new Array(nOut);
            for (let q = 0; q < nOut; q++) { let x = 0, y = 0, z = 0; for (let j = 0; j < N; j++) { const o3 = (q * N + j) * 3; x += pos[o3]; y += pos[o3 + 1]; z += pos[o3 + 2]; } meanPos[q] = [x / N, y / N, z / N]; }
            // start offsets along the direction of travel (colour mode "start position")
            const dir = [0, 1, 2].map((a) => meanPos[nOut - 1][a] - meanPos[0][a]); const dn = Math.hypot(...dir) || 1;
            const dv = Math.abs(dn) < 1e-9 ? [0, 0, 1] : dir.map((x) => x / dn);
            const proj = Array.from({ length: N }, (_, j) => [0, 1, 2].reduce((s, a) => s + (pos[j * 3 + a] - meanPos[0][a]) * dv[a], 0));
            const pmin = Math.min(...proj), pmax = Math.max(...proj);
            const offs = proj.map((x) => (pmax > pmin ? (x - pmin) / (pmax - pmin) : 0.5));
            res = { meta, N, n_out: nOut, out_tr: meta.out_tr, out_step: meta.out_step, m: f(m), pos, mean: f(mean), minc: f(mi), g: f(g), meanPos, offs };
            buildSpins(N);
            $('pos').max = nOut - 1; $('pos').disabled = false; for (const b of ['play', 'step-b', 'step-f', 'zoom', 'tr-b', 'tr-f']) $(b).disabled = false;
            $('info').innerHTML = `<b>${meta.design}</b>: ${meta.n_enc} encodes, ${meta.n_rot} rotations<br>` +
                `${meta.n_spins} spins, ${meta.n_tr} TRs${meta.preroll ? ` after ${meta.preroll} pre-roll TRs` : ''}, start: ${meta.start}<br>` +
                `${meta.slab ? `slab ${(meta.slab.thickness * 1e3).toFixed(0)} mm, G_ss ${(meta.slab.g_ss * 1e3).toFixed(2)} mT/m, ` : 'hard pulse, '}TR ${(meta.tr * 1e3).toFixed(2)} ms (with the RF block), TE ${(meta.te * 1e3).toFixed(2)} ms<br>flip ${meta.flip_deg.toFixed(1)}°, RF spoil ${meta.rf_inc_deg.toFixed(0)}°, T1 ${meta.t1} s, T2 ${meta.t2} s<br>` +
                `ideal spoiled echo (Ernst) ${meta.ernst.toFixed(4)}<br>${meta.spins_left_volume ? `⚠ ${meta.spins_left_volume} spins left the flow volume (held in place)<br>` : ''}` +
                `computed in ${meta.seconds.toFixed(1)} s`;
            seek(0);
            status('');
            $('progress').textContent = `done in ${((performance.now() - t0) / 1e3).toFixed(1)} s`;
        } catch (e) {
            console.error(e);
            status(`error: ${e.message ?? e}`);
            $('progress').textContent = '';
        } finally {
            clearInterval(timer);
            $('run').disabled = false;
        }
    }
    $('run').addEventListener('click', run);
    window.addEventListener('resize', resize);
    window.__bloch = { run, seek, get res() { return res; } };
    resize();
    renderer.setAnimationLoop(() => renderer.render(scene, camera));
    // URL presets: ?start=steady&ntr=150&preroll=600&seek=130&auto=1 (the run starts at once, then seeks to that sample)
    if (params.get('start')) $('start').value = params.get('start');
    if (params.get('ntr')) $('n-tr').value = params.get('ntr');
    if (params.get('preroll')) $('preroll').value = params.get('preroll');
    if (params.get('auto') === '1') run().then(() => { if (res && params.get('seek')) seek(Number(params.get('seek'))); });
}

main().catch((e) => { console.error(e); status(`error: ${e.message ?? e}`); });
