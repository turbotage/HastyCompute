// 4D flow viewer: PC-MRA MIP, slice planes with velocity overlay and glyphs, streamlines, animated pathlines and
// analysis planes with flow curves, on a flow value in the hasty bank (schema: rustlib/crates/hasty-server/src/flow.rs).
// Heavy lifting (tracing, quantification) runs in the server's flow_* commands; this page fetches and renders.
//   flow/index.html?id=<value id>[&frame=n][&follow=0][&selftest=1]
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { HastyClient } from '../../common/hasty_client.js';
import { COLORMAPS, cssGradient, lut } from './colormaps.js';
import { Glyphs, LineLayer, MipLayer, PlaneDisc, SlicePlane, lutTexture, sliceAxes } from './layers.js';
import { Plot } from './plot.js';

const client = new HastyClient();
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const selftest = params.get('selftest') === '1';
const status = (s) => { $('status').textContent = s; };
const temps = new Set();
const forget = (id) => { temps.delete(id); client.delete(id).catch(() => {}); };
window.addEventListener('pagehide', () => { for (const t of temps) client.delete(t).catch(() => {}); });

/** run a flow command; fetch its outputs and delete them from the bank */
async function run(cmd, opts, inputs) {
    const r = await client.execute(cmd, opts, inputs, `${cmd} (flow viewer)`);
    r.outputs.forEach((o) => temps.add(o));
    const vals = await Promise.all(r.outputs.map((o) => client.get(o)));
    r.outputs.forEach(forget);
    return { message: r.message ? JSON.parse(r.message) : {}, values: vals };
}

const linesOf = (v) => ({ points: v.get('points').data, speed: v.get('speed').data, time: v.get('time').data, offsets: v.get('offsets').data });

async function pick() {
    const list = (await client.list()).filter((e) => e.summary?.type === 'dict' && e.summary.items?.vel);
    const p = $('picker');
    p.hidden = false;
    p.innerHTML = '<h2>Flow values in the bank</h2>' + (list.length ? '' : '<p class="hint">None yet: <code>hastycompute.v2.show_flow(…)</code>, <code>hasty_server::viz::show_flow(…)</code> or <code>tools/flow/show_run.py</code>.</p>');
    for (const e of list.reverse()) {
        const a = document.createElement('a');
        a.href = `?id=${e.id}`;
        a.textContent = `${e.name} — ${e.summary.items.vel.shape.join('×')}`;
        p.append(a);
    }
}

async function main() {
    const id = params.get('id');
    if (!id) { await pick(); return; }
    const entry = await client.info(id);
    $('name').textContent = entry.name;
    document.title = `${entry.name} — flow`;
    status('flow_info…');
    const info = (await run('flow_info', {}, [id])).message;
    const [X, Y, Z] = info.dims, F = info.frames, dt = info.dt;
    const sp = info.spacing.map((s) => s * 1e3); // mm
    const cycle = F * dt;
    const vmaxDefault = Math.max(0.1, Number((info.p999_speed || info.max_speed || 1).toFixed(2)));
    $('vmax').value = vmaxDefault;
    $('frame').max = F - 1;
    for (const c of COLORMAPS) $('cmap').append(new Option(c, c));

    // ── three.js scene ───────────────────────────────────────────────────────────────────────────────────────────
    const canvas = $('canvas');
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: selftest });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x101214);
    const ext = [X * sp[0], Y * sp[1], Z * sp[2]];
    const centre = new THREE.Vector3(...ext.map((e) => e / 2 - 0.5));
    const camera = new THREE.PerspectiveCamera(35, 1, 1, 10 * Math.max(...ext));
    camera.up.set(0, 0, 1);
    camera.position.copy(centre).add(new THREE.Vector3(0.2, -1.6, 0.9).multiplyScalar(Math.max(...ext)));
    const controls = new OrbitControls(camera, canvas);
    controls.target.copy(centre);
    controls.update();
    const box = new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(...sp.map((s) => -s / 2)), new THREE.Vector3(...ext.map((e, a) => e - sp[a] / 2))), 0x2c3a48);
    scene.add(box);

    let cmapName = 'turbo';
    let lutTex = lutTexture(cmapName), lutArr = lut(cmapName), divArr = lut('coolwarm');
    const vmax = () => Math.max(1e-3, Number($('vmax').value));
    const legend = () => {
        $('legend-bar').style.background = cssGradient(cmapName);
        $('legend-lo').textContent = '0';
        $('legend-hi').textContent = vmax().toFixed(2);
    };
    legend();

    // ── PC-MRA ─────────────────────────────────────────────────────────────────────────────────────────────────
    status('PC-MRA…');
    const pc = (await run('flow_pcmra', {}, [id])).values[0];
    const mip = new MipLayer(info.dims, sp, pc.data, lutTexture('gray'));
    scene.add(mip.mesh);
    let amax = 0;
    for (let i = 1; i < pc.data.length; i++) if (pc.data[i] > pc.data[amax]) amax = i;
    const hot = [amax % X, Math.floor(amax / X) % Y, Math.floor(amax / (X * Y))];

    // ── slices ─────────────────────────────────────────────────────────────────────────────────────────────────
    const planes = [0, 1, 2].map((a) => new SlicePlane(a, info.dims, sp));
    const glyphs = new Glyphs(lutArr);
    scene.add(glyphs.group);
    planes.forEach((p, a) => {
        $(`s${a}`).max = info.dims[a] - 1;
        $(`s${a}`).value = hot[a];
        p.place(hot[a]);
        scene.add(p.mesh);
    });
    const cache = new Map(); // `${axis}:${idx}:${t}` -> {vel, mag}
    const pending = new Map();
    function slicePaths(axis, idx, t) {
        const v = axis === 2 ? `["vel"][${t}, :, ${idx}, :, :]` : axis === 1 ? `["vel"][${t}, :, :, ${idx}, :]` : `["vel"][${t}, :, :, :, ${idx}]`;
        const m = axis === 2 ? `["mag"][${t}, ${idx}, :, :]` : axis === 1 ? `["mag"][${t}, :, ${idx}, :]` : `["mag"][${t}, :, :, ${idx}]`;
        return [v, m];
    }
    function fetchSlice(axis, idx, t) {
        const key = `${axis}:${idx}:${t}`;
        if (cache.has(key)) return Promise.resolve(cache.get(key));
        if (pending.has(key)) return pending.get(key);
        const [pv, pm] = slicePaths(axis, idx, t);
        const pr = Promise.all([client.get(id, { slice: pv }), info.has_mag ? client.get(id, { slice: pm }) : null]).then(([v, m]) => {
            let mag = m ? m.data : null;
            if (!mag) { // speed as the anatomy channel
                const n = v.data.length / 3;
                mag = new Float32Array(n);
                for (let i = 0; i < n; i++) mag[i] = Math.hypot(v.data[i], v.data[n + i], v.data[2 * n + i]);
            } else mag = Float32Array.from(mag, Math.abs);
            const r = { vel: v.data, mag };
            cache.set(key, r);
            pending.delete(key);
            return r;
        });
        pending.set(key, pr);
        return pr;
    }
    const magHi = [null, null, null];
    function windowOf(mag) {
        const s = Float32Array.from(mag).sort();
        return s[Math.floor(0.995 * (s.length - 1))] || 1;
    }

    // ── state ───────────────────────────────────────────────────────────────────────────────────────────────────
    let tau = (Number(params.get('frame') ?? info.peak_frame) || 0) * dt; // cardiac time, s
    let frame = Math.floor(tau / dt) % F;
    let playing = false;
    let cursor = null; // voxel coords
    const cursorMesh = new THREE.Mesh(new THREE.SphereGeometry(Math.min(...sp) * 0.9, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffd23f }));
    cursorMesh.visible = false;
    scene.add(cursorMesh);
    const stream = new LineLayer(sp, lutTex);
    const path = new LineLayer(sp, lutTex, { pathlines: true });
    scene.add(stream.group, path.group);
    const streamCache = new Map();
    const discs = [];
    let activeDisc = null;
    const plot = new Plot($('plot'));

    async function paintSlices() {
        const thr = Number($('thr').value), overlay = $('overlay').value;
        await Promise.all(planes.map(async (p, a) => {
            const on = $(`s${a}-on`).checked;
            p.mesh.visible = on;
            if (!on) return;
            const d = await fetchSlice(a, p.index, frame);
            if (magHi[a] === null) magHi[a] = windowOf(d.mag);
            p.paint(d.mag, d.vel, { magHi: magHi[a], vmax: overlay === 'speed' ? vmax() : vmax(), thr, overlay, lutArr: overlay === 'speed' ? lutArr : divArr });
        }));
        // glyphs on the first visible slice
        const ga = [2, 1, 0].find((a) => $(`s${a}-on`).checked);
        if ($('glyph-on').checked && ga !== undefined) {
            const d = await fetchSlice(ga, planes[ga].index, frame);
            glyphs.lutArr = lutArr;
            glyphs.set(planes[ga], d.vel, { stride: Math.max(1, Number($('stride').value)), thr: Number($('thr').value), vmax: vmax(), lenMm: 2.2 * Math.max(...sp) * Math.max(1, Number($('stride').value)) / 2 });
        } else glyphs.set(planes[0], null, {});
    }

    function prefetch() { // the whole cycle of the visible slices, a few requests at a time
        const jobs = [];
        planes.forEach((p, a) => { if ($(`s${a}-on`).checked) for (let t = 0; t < F; t++) jobs.push([a, p.index, t]); });
        let i = 0;
        const next = () => { if (i < jobs.length) { const j = jobs[i++]; fetchSlice(...j).then(next, next); } };
        for (let k = 0; k < 4; k++) next();
    }

    function discOpts() {
        if (!activeDisc) return {};
        const i = activeDisc.info;
        return { center: i.center, normal: i.normal, radius_mm: 0.5 * i.grid * i.pixel_mm };
    }

    async function streamlines(f) {
        const key = `${f}:${$('sl-n').value}:${$('sl-q').value}:${$('sl-disc').checked && activeDisc ? discs.indexOf(activeDisc) : -1}`;
        if (!streamCache.has(key)) {
            streamCache.set(key, run('flow_streamlines', { frame: f, n: Number($('sl-n').value), quantile: Number($('sl-q').value), seed: 7, ...($('sl-disc').checked ? discOpts() : {}) }, [id]).then((r) => linesOf(r.values[0])));
        }
        return streamCache.get(key);
    }

    let streamShown = null;
    async function updateStreamlines() {
        if (!$('sl-on').checked) { stream.clear(); streamShown = null; return; }
        const f = frame;
        const l = await streamlines(f);
        if (f === frame && streamShown !== l) { stream.set(l); streamShown = l; }
    }

    async function updatePathlines() {
        if (!$('pl-on').checked) { path.clear(); return; }
        status('pathlines…');
        const r = await run('flow_pathlines', { n: Number($('pl-n').value), releases: Number($('pl-rel').value), quantile: Number($('sl-q').value), seed: 11, ...($('pl-disc').checked ? discOpts() : {}) }, [id]);
        path.set(linesOf(r.values[0]));
        status(`pathlines: ${r.message.lines} traces from ${r.message.seeds} seeds × ${r.message.releases} releases`);
        return r.message;
    }

    function velocityAt(p) { // from the slice cache of the current frame (nearest pixel of a slice through p)
        for (const a of [2, 1, 0]) {
            const d = cache.get(`${a}:${Math.round(p[a])}:${frame}`);
            if (!d) continue;
            const [ua, va] = sliceAxes(a);
            const nu = info.dims[ua], n = nu * info.dims[va];
            const i = Math.round(p[va]) * nu + Math.round(p[ua]);
            return [d.vel[i], d.vel[n + i], d.vel[2 * n + i]];
        }
        return null;
    }

    async function addPlane(center, normalHint) {
        let nrm = normalHint;
        if (!nrm) {
            // mean velocity direction in a small neighbourhood over the cycle peak: the cursor's slice data
            const v = velocityAt(center);
            nrm = v && Math.hypot(...v) > 0.05 * vmax() ? v : [0, 0, 1];
        }
        status('flow_plane…');
        const r = await run('flow_plane', { center, normal: nrm, radius_mm: Number($('pa-r').value), grid: 64, quantile: Number($('pa-q').value) }, [id]);
        const j = r.message;
        const disc = new PlaneDisc(j, r.values[0].data, r.values[1].data, sp, divArr);
        disc.color = ['#4fc3f7', '#ffb74d', '#81c784', '#f06292', '#ba68c8'][discs.length % 5];
        disc.frame(frame, vmax());
        scene.add(disc.group);
        discs.push(disc);
        activeDisc = disc;
        refreshPlot();
        status(`plane ${discs.length}: net ${j.net_ml.toFixed(1)} ml/cycle, area ${j.area_mm2.toFixed(0)} mm²`);
        return j;
    }

    function refreshPlot() {
        const t = [...Array(F).keys()].map((f) => f * dt * 1e3);
        plot.set(discs.map((d, k) => ({ name: `plane ${k + 1}`, x: t, y: d.info.flow_ml_s, color: d.color })), { xlabel: 't (ms)', ylabel: 'Q (ml/s)' });
        plot.setCursor(frame * dt * 1e3);
        const rows = [];
        discs.forEach((d, k) => {
            const j = d.info;
            const pk = j.peak_velocity_m_s.reduce((a, b) => (Math.abs(b) > Math.abs(a) ? b : a), 0);
            rows.push(`<tr><td style="color:${d.color}">plane ${k + 1}</td><td>net ${j.net_ml.toFixed(1)} ml</td><td>fwd ${j.forward_ml.toFixed(1)}</td><td>RF ${(100 * j.regurgitant_fraction).toFixed(1)}%</td></tr>`,
                `<tr><td></td><td>area ${j.area_mm2.toFixed(0)} mm²</td><td colspan="2">peak ${pk.toFixed(2)} m/s</td></tr>`);
        });
        $('pa-table').innerHTML = rows.join('');
    }

    function setFrame(f, fromPlayback = false) {
        f = ((f % F) + F) % F;
        if (!fromPlayback) tau = f * dt;
        if (f === frame && fromPlayback) return;
        frame = f;
        $('frame').value = f;
        $('frame-label').textContent = `t ${f} / ${F} (${(f * dt * 1e3).toFixed(0)} ms)`;
        paintSlices();
        updateStreamlines();
        discs.forEach((d) => d.frame(f, vmax()));
        plot.setCursor(f * dt * 1e3);
    }

    // ── UI wiring ───────────────────────────────────────────────────────────────────────────────────────────────
    $('frame').oninput = () => setFrame(Number($('frame').value));
    $('play').onclick = () => { playing = !playing; $('play').textContent = playing ? '❚❚' : '▶'; if (playing) prefetch(); };
    window.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.code === 'Space') { $('play').click(); e.preventDefault(); }
        if (e.code === 'ArrowRight') setFrame(frame + 1);
        if (e.code === 'ArrowLeft') setFrame(frame - 1);
    });
    $('cmap').onchange = () => {
        cmapName = $('cmap').value;
        lutTex = lutTexture(cmapName); lutArr = lut(cmapName);
        stream.uniforms.lut.value = lutTex; path.uniforms.lut.value = lutTex; path.headUniforms.lut.value = lutTex;
        legend(); paintSlices();
    };
    $('vmax').onchange = () => { stream.uniforms.vmax.value = vmax(); legend(); paintSlices(); discs.forEach((d) => d.frame(frame, vmax())); };
    stream.uniforms.vmax.value = vmax();
    $('mip-on').onchange = () => { mip.mesh.visible = $('mip-on').checked; };
    $('mip-lo').oninput = () => { mip.uniforms.lo.value = Number($('mip-lo').value); };
    $('mip-op').oninput = () => { mip.uniforms.opacity.value = Number($('mip-op').value); };
    for (const a of [0, 1, 2]) {
        $(`s${a}`).oninput = () => { planes[a].place(Number($(`s${a}`).value)); paintSlices(); };
        $(`s${a}-on`).onchange = () => paintSlices();
    }
    for (const k of ['overlay', 'thr', 'glyph-on', 'stride']) $(k).onchange = () => paintSlices();
    $('s-op').oninput = () => { for (const p of planes) { p.mesh.material.opacity = Number($('s-op').value); p.mesh.material.depthWrite = Number($('s-op').value) > 0.99; } };
    $('sl-on').onchange = updateStreamlines;
    $('sl-go').onclick = () => { streamCache.clear(); $('sl-on').checked = true; updateStreamlines(); };
    $('pl-on').onchange = updatePathlines;
    $('pl-go').onclick = () => { $('pl-on').checked = true; updatePathlines(); };
    $('pl-tail').oninput = () => { path.uniforms.tail.value = Number($('pl-tail').value) * cycle; };
    path.uniforms.tail.value = Number($('pl-tail').value) * cycle;
    path.uniforms.cycle.value = cycle;
    $('pa-add').onclick = () => { if (cursor) addPlane(cursor); else status('click a slice first'); };
    $('pa-clear').onclick = () => { for (const d of discs) scene.remove(d.group); discs.length = 0; activeDisc = null; refreshPlot(); };

    // picking on the slices: cursor + readout
    const ray = new THREE.Raycaster();
    let down = null;
    canvas.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
    canvas.addEventListener('pointerup', (e) => {
        if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
        const r = canvas.getBoundingClientRect();
        ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
        const hit = ray.intersectObjects(planes.filter((p) => p.mesh.visible).map((p) => p.mesh))[0];
        if (!hit) return;
        cursor = [0, 1, 2].map((a) => Math.min(info.dims[a] - 1, Math.max(0, hit.point.getComponent(a) / sp[a])));
        cursorMesh.position.copy(hit.point);
        cursorMesh.visible = true;
        const v = velocityAt(cursor);
        $('readout').hidden = false;
        $('readout').textContent = `voxel (${cursor.map((c) => c.toFixed(1)).join(', ')})` + (v ? `\nv = (${v.map((x) => x.toFixed(3)).join(', ')}) m/s, |v| ${Math.hypot(...v).toFixed(3)}` : '');
    });

    function resize() {
        const w = canvas.clientWidth, h = canvas.clientHeight;
        renderer.setSize(w, h, false);
        camera.aspect = w / Math.max(h, 1);
        camera.updateProjectionMatrix();
        const res = new THREE.Vector2(w, h);
        stream.uniforms.resolution.value.copy(res);
        path.uniforms.resolution.value.copy(res);
        path.headUniforms.size.value = 5 * (window.devicePixelRatio || 1);
        plot.draw();
    }
    window.addEventListener('resize', resize);
    resize();

    let last = performance.now(), frames = 0, fps = 0, fpsT = last;
    function loop(now) {
        const dtReal = Math.min(0.1, (now - last) / 1000);
        last = now;
        if (playing) {
            tau = (tau + dtReal * Number($('rate').value)) % cycle;
            setFrame(Math.floor(tau / dt) % F, true);
        }
        path.uniforms.tNow.value = tau;
        renderer.render(scene, camera);
        frames++;
        if (now - fpsT > 1000) { fps = (frames * 1000) / (now - fpsT); frames = 0; fpsT = now; }
        requestAnimationFrame(loop);
    }

    // live: follow newly shown flow values, reload on updates of this one
    if (params.get('follow') !== '0') {
        client.watch((e) => {
            if (e.kind === 'show' && e.id && e.id !== id && e.message === 'flow') location.search = `?id=${e.id}`;
            else if (e.kind === 'update' && e.id === id) setTimeout(() => location.reload(), 300);
        });
    }

    frame = -1;
    setFrame(Math.floor(tau / dt) % F);
    await paintSlices();
    status(`${X}×${Y}×${Z}, ${F} frames × ${(dt * 1e3).toFixed(1)} ms, voxel ${sp.map((s) => s.toFixed(2)).join('×')} mm, p99.9 speed ${vmaxDefault} m/s`);
    requestAnimationFrame(loop);
    window.hasty = { client, scene, camera, renderer, info, planes, stream, path, discs, setFrame, addPlane, updatePathlines, updateStreamlines };

    if (selftest) {
        const out = { ok: false };
        const layers = (params.get('layers') ?? 'mip,slices,glyphs,stream,path,plane').split(',');
        try {
            mip.mesh.visible = $('mip-on').checked = layers.includes('mip');
            $('glyph-on').checked = layers.includes('glyphs');
            if (!layers.includes('slices')) $('s2-on').checked = false;
            if (layers.includes('coronal')) $('s1-on').checked = true;
            await paintSlices();
            $('sl-on').checked = layers.includes('stream');
            await updateStreamlines();
            out.streamlines = stream.nLines;
            $('pl-on').checked = layers.includes('path');
            const pm = (await updatePathlines()) ?? { lines: -1 };
            out.pathlines = pm.lines;
            const j = layers.includes('plane') ? await addPlane(hot) : { net_ml: 0, area_mm2: 0, flow_ml_s: [0] };
            out.plane = { net_ml: j.net_ml, area_mm2: j.area_mm2, peak_flow_ml_s: Math.max(...j.flow_ml_s) };
            tau = (info.peak_frame + 0.3) * dt;
            path.uniforms.tNow.value = tau;
            await new Promise((r) => setTimeout(r, 1500));
            out.fps = Number(fps.toFixed(1));
            out.gl = renderer.getContext().getParameter(renderer.getContext().VERSION);
            out.ok = (!layers.includes('stream') || stream.nLines > 0) && (!layers.includes('path') || pm.lines > 0) && Number.isFinite(j.net_ml);
        } catch (e) { out.error = String(e.stack || e); }
        $('selftest').textContent = JSON.stringify(out);
    }
}

main().catch((e) => { status(`error: ${e.message}`); console.error(e); if (selftest) $('selftest').textContent = JSON.stringify({ ok: false, error: String(e.stack || e) }); });
