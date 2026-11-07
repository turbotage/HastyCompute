// 4D flow viewer: PC-MRA MIP, slice planes with velocity overlay and glyphs, streamlines, animated pathlines and
// analysis planes with flow curves, on a flow value in the hasty bank (schema: rustlib/crates/hasty-server/src/flow.rs).
// Particle tracing and plane quantification run in the browser once the velocity volume is loaded (WebGPU compute
// tracer, JS fallback, common/flowmath.js); until then — or when the volume exceeds the memory budget, or with &client=0 —
// the server's flow_* commands do the work.
//   flow/index.html?id=<value id>[&frame=n][&follow=0][&client=0][&maxmb=1500][&dpr=1.5][&selftest=1]
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { HastyClient } from '../../common/hasty_client.js';
import { Centerlines, derivedScalar, dilate, lineAttribute, mergeLines, meanFlowDirection, measurePlane, pathlines as cpuPathlines, percentile as pctl, regionFrom, seedsDisc, seedsRegion, streamlines as cpuStreamlines } from '../../common/flowmath.js';
import { COLORMAPS, cssGradient, lut } from './colormaps.js';
import { CenterlineLayer, Glyphs, LineLayer, MipLayer, PlaneDisc, SlicePlane, lutTexture } from './layers.js';
import { fieldBytes, loadCenterlines, loadField } from './field.js';
import { GpuTracer } from './gputrace.js';
import { VesselTool, drawProfile, profileSeries } from './vessel.js';
import { Plot } from './plot.js';

const client = new HastyClient().enableWorkers();   // slice / volume fetches are decoded off the main thread
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
    const pixelRatio = Math.min(window.devicePixelRatio || 1, Number(params.get('dpr') ?? 1.5)); // ?dpr=2 for full Retina
    renderer.setPixelRatio(pixelRatio);
    // hardware acceleration check: a software GL (SwiftShader / llvmpipe) makes every layer crawl
    const gl0 = renderer.getContext(), dbg = gl0.getExtension('WEBGL_debug_renderer_info');
    const gpuName = String(dbg ? gl0.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl0.getParameter(gl0.RENDERER));
    if (/swiftshader|llvmpipe|softpipe|software|basic render|microsoft/i.test(gpuName) || gl0.isContextLost()) {
        const b = $('hwwarn');
        b.hidden = false;
        b.querySelector('span').textContent = `No GPU acceleration: WebGL is rendered in software (${gpuName}). Playback will be slow. In Chrome: Settings → System → "Use graphics acceleration when available", then check chrome://gpu.`;
    }
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
    let ageMax = 1;
    const legend = () => {
        const mode = $('lcolor').value;
        $('legend-bar').style.background = cssGradient(cmapName);
        $('legend-lo').textContent = '0';
        $('legend-hi').textContent = mode === 'speed' ? vmax().toFixed(2) : mode === 'age' ? (ageMax > 0 ? ageMax.toFixed(2) : 'end') : '';
        $('legend-title').textContent = mode === 'speed' ? 'speed m/s' : mode === 'age' ? (ageMax > 0 ? 'age s' : 'position along line') : 'seed';
    };
    legend();

    // ── PC-MRA ─────────────────────────────────────────────────────────────────────────────────────────────────
    status('PC-MRA…');
    const pcOut = (await client.execute('flow_pcmra', {}, [id], 'flow_pcmra (flow viewer)')).outputs[0];
    temps.add(pcOut);
    const pc = await client.getTensor(pcOut);
    forget(pcOut);
    const grayTex = lutTexture('gray');
    const mip = new MipLayer(info.dims, sp, pc.data, grayTex);
    const pcData = pc.data;
    scene.add(mip.mesh);
    let amax = 0;
    for (let i = 1; i < pc.data.length; i++) if (pc.data[i] > pc.data[amax]) amax = i;
    const hot = [amax % X, Math.floor(amax / X) % Y, Math.floor(amax / (X * Y))];

    // ── slices ─────────────────────────────────────────────────────────────────────────────────────────────────
    // Each visible slice keeps its whole cardiac cycle in one GPU texture array; frames are requested as the current
    // frame first and the full cycle in one more request (debounced while a slider is dragged, aborted when it moves).
    const floatTex = renderer.extensions.has('OES_texture_float_linear');
    const planes = [0, 1, 2].map((a) => new SlicePlane(a, info.dims, sp, F, floatTex));
    const divTex = lutTexture('coolwarm');
    const glyphs = new Glyphs(lutTex);
    scene.add(glyphs.group);
    const perf = { fetchMs: 0, buildMs: 0, setMs: 0, renderMs: 0, gpuMs: 0, stalls: 0, fps: 0 };
    let dirty = true, raf = 0, interacting = false;
    const alwaysRender = selftest;
    const invalidate = () => { dirty = true; if (!raf) raf = requestAnimationFrame(loop); };
    const sgen = [0, 0, 0], sac = [new AbortController(), new AbortController(), new AbortController()], stimer = [0, 0, 0], sload = [null, null, null];
    function slicePaths(axis, idx, t) { // t: frame number, or null for every frame
        const T = t === null ? ':' : String(t);
        const v = axis === 2 ? `["vel"][${T}, :, ${idx}, :, :]` : axis === 1 ? `["vel"][${T}, :, :, ${idx}, :]` : `["vel"][${T}, :, :, :, ${idx}]`;
        const m = axis === 2 ? `["mag"][${T}, ${idx}, :, :]` : axis === 1 ? `["mag"][${T}, :, ${idx}, :]` : `["mag"][${T}, :, :, ${idx}]`;
        return [v, m];
    }
    /** all frames of slice idx of axis a straight from the in-memory volume (no server request) */
    function localSlice(a, idx) {
        const p = planes[a], n = p.nu * p.nv, vel = new Float32Array(F * 3 * n), mag = field.mag ? new Float32Array(F * n) : null, nv = X * Y * Z;
        const base = new Uint32Array(n);
        for (let v = 0; v < p.nv; v++) for (let u = 0; u < p.nu; u++) base[v * p.nu + u] = a === 2 ? (idx * Y + v) * X + u : a === 1 ? (v * Y + idx) * X + u : (v * Y + u) * X + idx;
        for (let f = 0; f < F; f++) {
            const fr = field.frames[f];
            for (let c = 0; c < 3; c++) { const o = (f * 3 + c) * n, co = c * nv; for (let i = 0; i < n; i++) vel[o + i] = fr[co + base[i]]; }
            if (mag) { const mf = field.mag[f]; for (let i = 0; i < n; i++) mag[f * n + i] = mf[base[i]]; }
        }
        return { vel, mag };
    }
    function fetchFrames(a, t) {
        if (field) {
            const p = planes[a], { vel, mag } = localSlice(a, p.index);
            p.ingest(0, F, vel, mag);
            p.cycleRequested = true;
            paintSlices();
            return Promise.resolve();
        }
        const p = planes[a], idx = p.index, signal = t === null ? sac[a].signal : undefined;
        const [pv, pm] = slicePaths(a, idx, t);
        const f0 = t === null ? 0 : t, nf = t === null ? F : 1;
        return Promise.all([client.getTensor(id, { slice: pv, signal }), info.has_mag ? client.getTensor(id, { slice: pm, signal }) : null]).then(([v, m]) => {
            if (idx !== p.index) return;                       // the plane moved on while this was in flight
            p.ingest(f0, nf, v.data, m ? m.data : null);
            paintSlices();
        }).catch((e) => { if (e?.name !== 'AbortError') console.error('slice fetch', e); });
    }
    // Slice loading: at most one current-frame request in flight per plane; while the slider moves the newest position is
    // fetched as soon as the previous request returned (never cancelled, so the plane keeps updating while dragging).
    // The rest of the cycle follows in one request once the plane rests; that one is cancelled when the plane moves.
    const ctimer = [0, 0, 0], busy = [false, false, false], want = [false, false, false];
    function pump(a) {
        const p = planes[a];
        if (busy[a] || !want[a] || !$(`s${a}-on`).checked) return;
        want[a] = false;
        if (p.has[frame]) { scheduleCycle(a); return; }
        busy[a] = true;
        sload[a] = fetchFrames(a, frame).finally(() => { busy[a] = false; if (want[a]) pump(a); else scheduleCycle(a); });
    }
    function scheduleCycle(a, delay = 2000, force = false) {   // the whole cycle is ~5 MB per plane: only once the user rests (or plays)
        const p = planes[a];
        if ((p.cycleRequested && !force) || p.nHave >= F || !$(`s${a}-on`).checked) return;
        p.cycleRequested = true;
        clearTimeout(ctimer[a]);
        ctimer[a] = setTimeout(() => {
            if (busy[a] || want[a]) { p.cycleRequested = false; return; }
            busy[a] = true;
            sload[a] = fetchFrames(a, null).finally(() => { busy[a] = false; if (want[a]) pump(a); });
        }, delay);
    }
    function loadSlice(a, delay = 0) {
        clearTimeout(stimer[a]);
        want[a] = true;
        if (delay) stimer[a] = setTimeout(() => pump(a), delay); else pump(a);
    }
    function moveSlice(a, idx) {
        sac[a].abort();                                        // only a running whole-cycle request is worth cancelling
        sac[a] = new AbortController();
        clearTimeout(ctimer[a]);
        planes[a].place(idx);
        loadSlice(a);
        paintSlices();
    }
    const slicesReady = () => Promise.all(planes.map((p, a) => ($(`s${a}-on`).checked ? sload[a] : null)));
    planes.forEach((p, a) => {
        $(`s${a}`).max = info.dims[a] - 1;
        $(`s${a}`).value = hot[a];
        p.place(hot[a]);
        scene.add(p.mesh);
    });
    const OVERLAY_MODE = { '': 0, speed: 1, vx: 2, vy: 3, vz: 4 };
    function glyphPlane() { const ga = [2, 1, 0].find((a) => $(`s${a}-on`).checked); return ga === undefined ? null : planes[ga]; }
    /** push UI state + loaded data to the slice and glyph shaders (no data movement) */
    function paintSlices() {
        const thr = Number($('thr').value), overlay = $('overlay').value;
        planes.forEach((p, a) => {
            const on = $(`s${a}-on`).checked;
            p.mesh.visible = on;
            if (!on) return;
            p.show(frame);
            p.style({ vmax: vmax(), thr, mode: OVERLAY_MODE[overlay] ?? 0, lutTex: overlay === 'speed' ? lutTex : divTex });
        });
        const gp = glyphPlane();
        if ($('glyph-on').checked && gp && gp.tex && gp.shown >= 0) {
            const stride = Math.max(1, Number($('stride').value));
            glyphs.uniforms.lut.value = lutTex;
            glyphs.set(gp, { stride, thr, vmax: vmax(), lenMm: 2.2 * Math.max(...sp) * stride / 2 });
        } else glyphs.hide();
        invalidate();
    }

    // ── state ───────────────────────────────────────────────────────────────────────────────────────────────────
    let tau = (Number(params.get('frame') ?? info.peak_frame) || 0) * dt; // cardiac time, s
    let frame = Math.floor(tau / dt) % F;
    let playing = false;
    let cursor = null; // voxel coords
    const cursorMesh = new THREE.Mesh(new THREE.SphereGeometry(Math.min(...sp) * 0.9, 12, 8), new THREE.MeshBasicMaterial({ color: 0xffd23f }));
    cursorMesh.visible = false;
    scene.add(cursorMesh);
    const centerLayer = new CenterlineLayer();
    scene.add(centerLayer.group);
    const stream = new LineLayer(sp, lutTex);
    const path = new LineLayer(sp, lutTex, { pathlines: true });
    scene.add(stream.group, path.group);
    const streamCache = new Map(), streamDone = new Set();
    const discs = [];
    let activeDisc = null;
    const plot = new Plot($('plot'));

    function discOpts() {
        if (!activeDisc) return {};
        const i = activeDisc.info;
        return { center: i.center, normal: i.normal, radius_mm: 0.5 * i.grid * i.pixel_mm };
    }

    // ── client-side compute state ───────────────────────────────────────────────────────────────────────────────
    // The velocity volume is fetched once (field.js); then tracing runs on the GPU / in JS and planes are measured locally.
    let localTrace = params.get('client') === '1';       // trace + measure in the browser; otherwise the volume (if loaded for the vessel tool) is only used by the vessel tool
    let field = null, fieldMask = null, gpu = null, centerlines = null, vessel = null, gpuStop = null;
    const maxBytes = Number(params.get('maxmb') ?? 1500) * 1048576;
    // opt-in: ?client=1, or click the "server" label. Loading the whole volume costs memory + a GPU upload, so it is never automatic.
    const clientFits = fieldBytes(info) <= maxBytes;
    let clientWanted = params.get('client') === '1' && clientFits, clientStarted = false;
    const setCompute = (extra = '') => {
        const el = $('compute');
        el.textContent = (field && localTrace ? `client · ${gpu ? 'WebGPU' : 'JS'}` : field ? 'server · volume in browser (vessel tool)' : clientStarted ? 'server · loading volume…' : clientFits ? 'server · click to load volume into the browser' : 'server') + extra;
        el.style.cursor = !field && !clientStarted && clientFits ? 'pointer' : '';
    };
    $('compute').addEventListener('click', () => { if (!field && !clientStarted && clientFits) { localTrace = true; clientWanted = true; startClientLoad(); } });
    const local = () => field && localTrace;
    setCompute();
    const regionCache = new Map();
    const regionFor = (q) => { const k = `r${q.toFixed(3)}`; if (!regionCache.has(k)) regionCache.set(k, regionFrom({ mask: fieldMask, pc: pcData, quantile: q })); return regionCache.get(k); };
    const stopFor = (q) => { const k = `s${q.toFixed(3)}`; if (!regionCache.has(k)) regionCache.set(k, dilate(regionFor(Math.max(0, q - 0.05)), info.dims)); return regionCache.get(k); };
    const mmToM = 1e-3;

    function clientSeeds(kind) {          // kind: 'sl' | 'pl'
        const n = Number($(`${kind}-n`).value), q = Number($('sl-q').value), seed = kind === 'sl' ? 7 : 11;
        if ($(`${kind}-vessel`).checked && vessel?.result) return vessel.seeds(vessel.result, n, seed);
        if ($(`${kind}-disc`).checked && activeDisc) { const i = activeDisc.info; return seedsDisc(field, regionFor(q), i.center, i.normal, 0.5 * i.grid * i.pixel_mm * mmToM, n, seed); }
        return seedsRegion(field, regionFor(q), n, seed);
    }
    function useGpuStop(stop) { if (gpu && gpuStop !== stop) { gpu.setRegion(stop); gpuStop = stop; } }

    async function clientStreamlines(f) {
        const stop = stopFor(Number($('sl-q').value)), seeds = clientSeeds('sl'), opts = { step: 0.5, maxSteps: 600, minSpeed: 0.02 };
        if (gpu) { useGpuStop(stop); return gpu.streamlines(f, seeds, opts); }
        return cpuStreamlines(field, f, seeds, { ...opts, stop });
    }
    async function clientPathlines() {
        const stop = stopFor(Number($('sl-q').value)), seeds = clientSeeds('pl'), rel = Math.max(1, Number($('pl-rel').value)), all = [];
        for (let r = 0; r < rel; r++) {
            const t0 = (r * F) / rel;
            all.push(gpu ? (useGpuStop(stop), await gpu.pathlines(t0, F, seeds, { step: 4 })) : cpuPathlines(field, t0, F, seeds, { step: 4, maxSteps: 100000, stop }));
        }
        const lines = mergeLines(all);
        return { lines, message: { lines: lines.offsets.length - 1, seeds: seeds.length, releases: rel } };
    }

    // colour of lines: speed (default), age (time since the line start / release) or one colour per line
    function colourAttr(lines) {
        const mode = $('lcolor').value;
        if (mode === 'speed') return null;
        const a = lineAttribute(lines, mode);
        if (mode === 'age') {
            let m = 1e-9; for (const v of a) if (v > m) m = v;
            if (m <= 1e-9) {                                   // instantaneous streamlines have no time: use the position along the line instead
                ageMax = 0;
                for (let l = 0; l + 1 < lines.offsets.length; l++) { const o0 = lines.offsets[l], o1 = lines.offsets[l + 1], len = Math.max(1, o1 - o0 - 1); for (let i = o0; i < o1; i++) a[i] = ((i - o0) / len) * vmax(); }
            } else { ageMax = m; for (let i = 0; i < a.length; i++) a[i] = (a[i] / m) * vmax(); }
        }
        else for (let i = 0; i < a.length; i++) a[i] *= vmax();
        return a;
    }
    let lastStream = null, lastPath = null;
    // Streamline sets are turned into GPU geometry once (WeakMap by lines object); showing a cached frame is a pointer swap.
    const prepared = new WeakMap(), preparedAll = new Set();
    function prepareStream(l) {
        let m = prepared.get(l);
        if (!m) { const t = performance.now(); m = stream.prepare(l); prepared.set(l, m); preparedAll.add(m); perf.buildMs = performance.now() - t; }
        return m;
    }
    function clearStreamCache() { streamCache.clear(); streamDone.clear(); for (const m of preparedAll) stream.dispose(m); preparedAll.clear(); }
    const setStream = (l) => {
        const t = performance.now();
        lastStream = l;
        if ($('lcolor').value === 'speed') stream.use(prepareStream(l)); else stream.set(l, colourAttr(l));
        perf.setMs = performance.now() - t;
        legend(); invalidate();
    };
    const setPath = (l) => { lastPath = l; path.set(l, colourAttr(l)); legend(); invalidate(); };

    const streamKey = (f) => `${f}:${$('sl-n').value}:${$('sl-q').value}:${$('sl-disc').checked && activeDisc ? discs.indexOf(activeDisc) : -1}`;
    function streamlines(f) {
        if (local()) return clientStreamlines(f);
        const key = streamKey(f);
        if (!streamCache.has(key)) {
            const t = performance.now();
            streamCache.set(key, run('flow_streamlines', { frame: f, n: Number($('sl-n').value), quantile: Number($('sl-q').value), seed: 7, ...($('sl-disc').checked ? discOpts() : {}) }, [id]).then((r) => { perf.fetchMs = performance.now() - t; streamDone.add(key); return linesOf(r.values[0]); }));
        }
        return streamCache.get(key);
    }

    // Server mode: while playing only frames already computed are shown (no new server work per frame); scrubbing is
    // debounced. Client mode: tracing is local, so "update every frame" re-traces each frame on the GPU.
    let streamShown = null, streamTimer = 0, streamBusy = false, streamAgain = false;
    async function updateStreamlines({ cachedOnly = false } = {}) {
        if (!$('sl-on').checked) { stream.clear(); streamShown = null; invalidate(); return; }
        const f = frame;
        if (cachedOnly && local() && !$('sl-live').checked) return;
        if (local()) {                                           // local tracing: one at a time, newest frame wins
            if (streamBusy) { streamAgain = true; return; }
            streamBusy = true;
            try {
                const l = await streamlines(f);
                if (f === frame && streamShown !== l) { setStream(l); streamShown = l; }
            } finally { streamBusy = false; if (streamAgain) { streamAgain = false; updateStreamlines(); } }
            return;
        }
        const l = await streamlines(f);                        // server: cached per frame, requests overlap like before
        if (f === frame && streamShown !== l) { setStream(l); streamShown = l; }
        prefetchStreams();
    }
    // server mode: compute + upload the streamlines of every frame ahead of playback (4 requests in flight, nearest frames first)
    let prefetchRun = 0, prefetching = false;
    async function prefetchStreams() {
        if (local() || !$('sl-on').checked || prefetching || cachedStreams() >= F) return;
        prefetching = true;
        const my = ++prefetchRun, order = Array.from({ length: F }, (_, k) => (frame + k) % F);
        const worker = async () => {
            while (my === prefetchRun && order.length) {
                const f = order.shift();
                try { prepareStream(await streamlines(f)); } catch (e) { console.warn('stream prefetch', e); return; }
                await new Promise((r) => setTimeout(r, 0));
            }
        };
        try { await Promise.all([worker(), worker(), worker(), worker()]); } finally { prefetching = false; }
    }
    const cachedStreams = () => { let n = 0; for (let f = 0; f < F; f++) if (streamCache.has(streamKey(f))) n++; return n; };
    function scheduleStreamlines() {
        clearTimeout(streamTimer);
        if (playing) { updateStreamlines({ cachedOnly: true }); return; }
        streamTimer = setTimeout(updateStreamlines, local() ? 150 : 60);
    }

    async function updatePathlines() {
        if (!$('pl-on').checked) { path.clear(); invalidate(); return; }
        status('pathlines…');
        let lines, message;
        if (local()) {
            const t = performance.now();
            ({ lines, message } = await clientPathlines());
            status(`pathlines: ${message.lines} traces from ${message.seeds} seeds × ${message.releases} releases (${(performance.now() - t).toFixed(0)} ms, ${gpu ? 'WebGPU' : 'JS'})`);
        } else {
            const r = await run('flow_pathlines', { n: Number($('pl-n').value), releases: Number($('pl-rel').value), quantile: Number($('sl-q').value), seed: 11, ...($('pl-disc').checked ? discOpts() : {}) }, [id]);
            lines = linesOf(r.values[0]); message = r.message;
            status(`pathlines: ${message.lines} traces from ${message.seeds} seeds × ${message.releases} releases`);
        }
        setPath(lines);
        return message;
    }

    function velocityAt(p) { // nearest pixel of a loaded slice through p (current frame, or the nearest loaded one)
        for (const a of [2, 1, 0]) {
            const pl = planes[a];
            if (!pl.mesh.visible || pl.index !== Math.round(p[a]) || pl.shown < 0) continue;
            const [ua, va] = a === 2 ? [0, 1] : a === 1 ? [0, 2] : [1, 2];
            return pl.velocityAt(frame, Math.min(pl.nu - 1, Math.max(0, Math.round(p[ua]))), Math.min(pl.nv - 1, Math.max(0, Math.round(p[va]))));
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
        let j, img, reg;
        if (local()) {
            const m = measurePlane(field, { center, normal: nrm, radiusM: Number($('pa-r').value) * mmToM, n: 64, region: regionFor(Number($('pa-q').value)) });
            j = { center: m.center, normal: m.normal, u: m.u, w: m.w, grid: m.n, pixel_mm: m.ds * 1e3, flow_ml_s: m.flow_ml_s, area_mm2: m.area_mm2, net_ml: m.net_ml, forward_ml: m.forward_ml, backward_ml: m.backward_ml, regurgitant_fraction: m.regurgitant_fraction, peak_velocity_m_s: Array.from(m.vpeak) };
            img = m.img; reg = m.lumen;
        } else {
            status('flow_plane…');
            const r = await run('flow_plane', { center, normal: nrm, radius_mm: Number($('pa-r').value), grid: 64, quantile: Number($('pa-q').value) }, [id]);
            j = r.message; img = r.values[0].data; reg = r.values[1].data;
        }
        const disc = new PlaneDisc(j, img, reg, sp, divArr);
        disc.color = ['#4fc3f7', '#ffb74d', '#81c784', '#f06292', '#ba68c8'][discs.length % 5];
        disc.frame(frame, vmax());
        scene.add(disc.group);
        invalidate();
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
        invalidate();
    }

    function setFrame(f, fromPlayback = false) {
        f = ((f % F) + F) % F;
        if (!fromPlayback) tau = f * dt;
        if (f === frame && fromPlayback) return;
        frame = f;
        $('frame').value = f;
        $('frame-label').textContent = `t ${f} / ${F} (${(f * dt * 1e3).toFixed(0)} ms)`;
        paintSlices();
        planes.forEach((p, a) => { if ($(`s${a}-on`).checked && !p.has[f] && !p.cycleRequested) loadSlice(a, 0); });
        scheduleStreamlines();
        discs.forEach((d) => d.frame(f, vmax()));
        plot.setCursor(f * dt * 1e3);
        vessel?.setFrame(f, vmax());
        renderVesselFrame();
        if ($('vq').value !== 'pcmra') applyQuantity();
    }

    // ── volume quantity (MIP of PC-MRA, or speed / vorticity / Q-criterion / helicity of the current frame) ──────────────
    const qCache = {};
    function applyQuantity() {
        const kind = $('vq').value;
        if (kind === 'pcmra' || !field) return;
        const c = (qCache[kind] ??= { scale: 0, frames: new Map() });
        if (!c.scale) {                                          // common scale: p99.5 over a few frames of the cycle
            let m = 0;
            for (const f of new Set([info.peak_frame, Math.floor(F / 4), Math.floor(F / 2), Math.floor((3 * F) / 4)])) m = Math.max(m, pctl(derivedScalar(field, f, kind), 0.995));
            c.scale = m || 1;
        }
        let u = c.frames.get(frame);
        if (!u) {
            const v = derivedScalar(field, frame, kind), k = 255 / c.scale;
            u = new Uint8Array(v.length);
            for (let i = 0; i < v.length; i++) { const q = v[i] * k; u[i] = q >= 255 ? 255 : q > 0 ? (q + 0.5) | 0 : 0; }
            c.frames.set(frame, u);
        }
        mip.setData(u);
        invalidate();
    }

    // ── vessel tool ─────────────────────────────────────────────────────────────────────────────────────────────
    const vtPlot = new Plot($('vt-plot')), vtDiam = new Plot($('vt-diam'));
    let lastHit = null;
    const vesselOpts = () => ({
        nPlanes: Math.max(1, Math.min(31, Math.round(Number($('vt-n').value)))), stepMm: Math.max(0.1, Number($('vt-step').value)),
        radiusMm: Math.max(3, Number($('vt-r').value)), frac: Math.min(0.95, Math.max(0.05, Number($('vt-frac').value))), orient: $('vt-orient').checked,
    });
    function renderVesselFrame() {
        const res = vessel?.result;
        drawProfile($('vt-profile'), res?.avg ?? null, frame, divArr, vmax());
        vtDiam.set(res ? profileSeries(res.avg, frame) : [], { xlabel: 'mm across the lumen', ylabel: 'v (m/s)' });
        vtPlot.setCursor(frame * dt * 1e3);
    }
    function renderVessel() {
        const res = vessel?.result;
        if (!res) { vtPlot.set([]); $('vt-table').innerHTML = ''; return; }
        const t = [...Array(F).keys()].map((f) => f * dt * 1e3), a = res.avg;
        const series = res.planes.map((m) => ({ x: t, y: m.flow_ml_s, color: 'rgba(79,195,247,0.35)' }));
        series.push({ x: t, y: a.flow_ml_s, color: '#ffd23f' });
        vtPlot.set(series, { xlabel: 't (ms)', ylabel: 'Q ml/s (yellow: mean)' });
        const pk = a.vpeak.reduce((x, y) => (Math.abs(y) > Math.abs(x) ? y : x), 0), qpk = a.flow_ml_s.reduce((x, y) => (Math.abs(y) > Math.abs(x) ? y : x), 0);
        const spread = res.planes.length > 1 ? Math.max(...a.band_ml_s.max.map((v, i) => v - a.band_ml_s.min[i])) : 0;
        $('vt-table').innerHTML = [
            ['planes', `${res.planes.length} of ${res.opts.nPlanes} (${res.source}), ${((res.planes.length - 1) * res.opts.stepMm).toFixed(1)} mm`],
            ['net flow', `${a.net_ml.toFixed(2)} ml/cycle`], ['forward / backward', `${a.forward_ml.toFixed(2)} / ${a.backward_ml.toFixed(2)} ml`],
            ['regurgitant fraction', `${(100 * a.regurgitant_fraction).toFixed(1)} %`], ['peak flow', `${qpk.toFixed(1)} ml/s`],
            ['lumen area', `${a.area_mm2.toFixed(0)} mm² (⌀ ${(2 * Math.sqrt(a.area_mm2 / Math.PI)).toFixed(1)} mm)`],
            ['peak velocity', `${pk.toFixed(2)} m/s`], ['plane-to-plane spread', `${spread.toFixed(1)} ml/s`],
        ].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join('');
    }
    function runVessel(hit) {
        const t = performance.now(), res = vessel.measure(hit, vesselOpts());
        if (!res) { status('vessel tool: no lumen / flow at that point — lower the threshold or click nearer the vessel centre'); return null; }
        lastHit = hit;
        vessel.show(res, frame, vmax());
        renderVessel();
        renderVesselFrame();
        status(`vessel: ${res.planes.length} planes (${res.source}), net ${res.avg.net_ml.toFixed(2)} ml/cycle, ${(performance.now() - t).toFixed(0)} ms`);
        return res;
    }
    function vesselClick(e, r) {
        if (!field || !vessel) { status(clientStarted ? 'vessel tool: the velocity volume is still loading into the browser…' : 'vessel tool: the velocity volume is not loaded (too large? see ?maxmb=)'); return; }
        ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
        const o = ray.ray.origin, d = ray.ray.direction;
        const hit = vessel.pick([o.x / sp[0], o.y / sp[1], o.z / sp[2]], [d.x / sp[0], d.y / sp[1], d.z / sp[2]], Math.max(0.25, mip.uniforms.lo.value) * mip.scale);
        if (!hit) { status('vessel tool: nothing bright under the cursor (raise the volume level or click on a vessel)'); return; }
        runVessel(hit);
    }

    // ── UI wiring ───────────────────────────────────────────────────────────────────────────────────────────────
    $('frame').oninput = () => setFrame(Number($('frame').value));
    $('play').onclick = () => { playing = !playing; $('play').textContent = playing ? '❚❚' : '▶'; if (playing) { prefetchStreams(); planes.forEach((p, a) => scheduleCycle(a, 0, true)); invalidate(); } else scheduleStreamlines(); };
    window.addEventListener('keydown', (e) => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.code === 'Space') { $('play').click(); e.preventDefault(); }
        if (e.code === 'ArrowRight') setFrame(frame + 1);
        if (e.code === 'ArrowLeft') setFrame(frame - 1);
    });
    $('cmap').onchange = () => {
        cmapName = $('cmap').value;
        lutTex = lutTexture(cmapName); lutArr = lut(cmapName);
        stream.uniforms.lut.value = lutTex; path.uniforms.lut.value = lutTex; path.headUniforms.lut.value = lutTex; glyphs.uniforms.lut.value = lutTex;
        if (lastStream) setStream(lastStream);
        if (lastPath) setPath(lastPath);
        if ($('vq').value !== 'pcmra') mip.uniforms.lut.value = lutTex;
        legend(); paintSlices();
    };
    $('lcolor').onchange = () => { if (lastStream) setStream(lastStream); if (lastPath) setPath(lastPath); legend(); };
    $('vmax').onchange = () => { stream.uniforms.vmax.value = vmax(); path.uniforms.vmax.value = vmax(); if (lastStream) setStream(lastStream); if (lastPath) setPath(lastPath); legend(); paintSlices(); discs.forEach((d) => d.frame(frame, vmax())); vessel?.setFrame(frame, vmax()); renderVesselFrame(); };
    stream.uniforms.vmax.value = vmax();
    path.uniforms.vmax.value = vmax();
    $('mip-on').onchange = () => { mip.mesh.visible = $('mip-on').checked; invalidate(); };
    $('mip-lo').oninput = () => { mip.uniforms.lo.value = Number($('mip-lo').value); invalidate(); };
    $('mip-op').oninput = () => { mip.uniforms.opacity.value = Number($('mip-op').value); invalidate(); };
    for (const a of [0, 1, 2]) {
        $(`s${a}`).oninput = () => moveSlice(a, Number($(`s${a}`).value));
        $(`s${a}-on`).onchange = () => { loadSlice(a, 0); paintSlices(); };
    }
    for (const k of ['overlay', 'thr', 'glyph-on', 'stride']) $(k).onchange = () => paintSlices();
    $('s-op').oninput = () => { for (const p of planes) p.setOpacity(Number($('s-op').value)); invalidate(); };
    $('sl-on').onchange = () => updateStreamlines();
    $('sl-go').onclick = () => { clearStreamCache(); $('sl-on').checked = true; updateStreamlines(); };
    $('pl-on').onchange = updatePathlines;
    $('pl-go').onclick = () => { $('pl-on').checked = true; updatePathlines(); };
    $('pl-tail').oninput = () => { path.uniforms.tail.value = Number($('pl-tail').value) * cycle; invalidate(); };
    path.uniforms.tail.value = Number($('pl-tail').value) * cycle;
    path.uniforms.cycle.value = cycle;
    $('vq').onchange = () => {
        if ($('vq').value === 'pcmra') { mip.setData(mip.u8); mip.uniforms.lut.value = grayTex; invalidate(); }
        else { mip.uniforms.lut.value = lutTex; applyQuantity(); }
    };
    $('vt-on').onchange = () => { document.body.classList.toggle('vessel-tool', $('vt-on').checked); if ($('vt-on').checked && !field) {
            if (clientFits) { status('vessel tool: loading the velocity volume into the browser (streamlines stay on the server)…'); clientWanted = true; startClientLoad(); }
            else status(`vessel tool: the velocity volume is too large for the browser (${(fieldBytes(info) / 1048576).toFixed(0)} MB; raise ?maxmb=)`);
        } };
    for (const k of ['vt-n', 'vt-step', 'vt-frac', 'vt-r', 'vt-orient']) $(k).onchange = () => { if (lastHit) runVessel(lastHit); };
    $('vt-redo').onclick = () => { if (lastHit) runVessel(lastHit); };
    $('vt-clear').onclick = () => { vessel?.clear(); lastHit = null; renderVessel(); renderVesselFrame(); };
    $('vt-seed').onclick = () => {
        if (!vessel?.result) { status('measure a vessel first'); return; }
        $('sl-vessel').checked = $('pl-vessel').checked = true;
        $('sl-on').checked = true; streamShown = null; updateStreamlines();
        if ($('pl-on').checked) updatePathlines();
    };
    $('cl-on').onchange = () => { centerLayer.group.visible = $('cl-on').checked; invalidate(); };
    $('pa-add').onclick = () => { if (cursor) addPlane(cursor); else status('click a slice first'); };
    $('pa-clear').onclick = () => { for (const d of discs) scene.remove(d.group); discs.length = 0; activeDisc = null; refreshPlot(); invalidate(); };

    // picking on the slices: cursor + readout (or, with the vessel tool on, a ray through the PC-MRA)
    const ray = new THREE.Raycaster();
    let down = null;
    canvas.addEventListener('pointerdown', (e) => { down = [e.clientX, e.clientY]; });
    canvas.addEventListener('pointerup', (e) => {
        if (!down || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
        const r = canvas.getBoundingClientRect();
        if ($('vt-on').checked) { vesselClick(e, r); return; }
        ray.setFromCamera(new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1), camera);
        const hit = ray.intersectObjects(planes.filter((p) => p.mesh.visible).map((p) => p.mesh))[0];
        if (!hit) return;
        cursor = [0, 1, 2].map((a) => Math.min(info.dims[a] - 1, Math.max(0, hit.point.getComponent(a) / sp[a])));
        cursorMesh.position.copy(hit.point);
        cursorMesh.visible = true;
        invalidate();
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
        path.headUniforms.size.value = 5 * pixelRatio;
        plot.draw(); vtPlot.draw(); vtDiam.draw(); renderVesselFrame();
        invalidate();
    }
    window.addEventListener('resize', resize);
    new ResizeObserver(resize).observe(canvas);
    resize();

    // Render on demand: a frame is drawn only when something changed (camera, UI, data) or while playing. The MIP's
    // ray-marching steps drop while the camera is being dragged and refine once it rests.
    const MIP_STEPS = 256, MIP_STEPS_MOVING = 96;
    controls.addEventListener('change', invalidate);
    controls.addEventListener('start', () => { interacting = true; mip.uniforms.steps.value = MIP_STEPS_MOVING; invalidate(); });
    controls.addEventListener('end', () => { interacting = false; mip.uniforms.steps.value = MIP_STEPS; invalidate(); });
    for (const ev of ['input', 'change']) $('side').addEventListener(ev, invalidate);

    let last = performance.now(), frames = 0, fps = 0, fpsT = last;
    // Playback waits for a frame's streamlines (server mode) instead of racing ahead of them: the first cycle fills the
    // cache (prefetchStreams), later cycles run at full speed.
    const holdFor = (nf) => $('sl-on').checked && !local() && nf !== frame && !streamDone.has(streamKey(nf));
    const glc = renderer.getContext(), px = new Uint8Array(4);
    function loop(now) {
        raf = 0;
        const dtReal = Math.min(0.1, (now - last) / 1000);
        last = now;
        if (playing) {
            const nt = (tau + dtReal * Number($('rate').value)) % cycle, nf = Math.floor(nt / dt) % F;
            if (holdFor(nf)) { streamlines(nf).catch(() => {}); perf.stalls++; }
            else { tau = nt; setFrame(nf, true); }
        }
        path.uniforms.tNow.value = tau;
        if (dirty || playing || alwaysRender) {
            dirty = false;
            const t0 = performance.now();
            renderer.render(scene, camera);
            perf.renderMs = performance.now() - t0;
            if (!$('perf').hidden) { glc.readPixels(0, 0, 1, 1, glc.RGBA, glc.UNSIGNED_BYTE, px); perf.gpuMs = performance.now() - t0; }   // readPixels waits for the GPU
            frames++;
        }
        if (now - fpsT > 1000) { fps = (frames * 1000) / (now - fpsT); frames = 0; fpsT = now; perf.fps = fps; showPerf(); }
        if ((playing || alwaysRender) && !raf) raf = requestAnimationFrame(loop);   // invalidate() inside this frame may already have queued one
    }
    function showPerf() {
        const el = $('perf');
        if (el.hidden) return;
        el.textContent = `${perf.fps.toFixed(0)} fps · render cpu ${perf.renderMs.toFixed(1)} ms, cpu+gpu ${perf.gpuMs.toFixed(1)} ms\n` +
            `streamlines ${local() ? 'local' : `server ${cachedStreams()}/${F} frames cached`} · last fetch ${perf.fetchMs.toFixed(0)} ms · build ${perf.buildMs.toFixed(1)} ms · show ${perf.setMs.toFixed(1)} ms · waited ${perf.stalls}×\n` +
            `slices ${planes.map((p) => `${p.nHave}/${F}`).join(' ')} · ${gpuName}`;
    }
    $('perf').hidden = params.get('perf') !== '1';
    window.addEventListener('keydown', (e) => { if (e.key === 'p' && !e.target.matches('input, select, textarea')) { $('perf').hidden = !$('perf').hidden; showPerf(); invalidate(); } });

    // live: follow newly shown flow values, reload on updates of this one
    if (params.get('follow') !== '0') {
        client.watch((e) => {
            if (e.kind === 'show' && e.id && e.id !== id && e.message === 'flow') location.search = `?id=${e.id}`;
            else if (e.kind === 'update' && e.id === id) setTimeout(() => location.reload(), 300);
        });
    }

    // ── client-side volume, tracer, vessel tool, centerlines (background; the viewer is usable meanwhile) ────────────────
    async function startClientLoad() {
        if (!centerlines) loadCenterlines(client, id, entry.summary).then((c) => {
            if (!c) return;
            centerlines = new Centerlines(c, info.spacing);
            centerLayer.set(c, sp);
            $('cl-section').hidden = false;
            $('cl-info').textContent = `(${centerlines.nLines} lines)`;
        }).catch((e) => console.warn('centerlines', e));
        if (!clientWanted || clientStarted) { setCompute(); return; }
        clientStarted = true; setCompute();
        try {
            const velRank = entry.summary?.items?.vel?.shape?.length ?? 5;
            const r = await loadField(client, id, info, { velRank, onProgress: (a, b) => status(`loading the velocity volume to the browser: ${a}/${b} frames`) });
            field = r.field; fieldMask = r.mask;
            regionCache.clear();
            gpu = localTrace ? await GpuTracer.create() : null;
            if (gpu && gpu.fits(field)) await gpu.setField(field, (a, b) => status(`uploading the velocity volume to the GPU: ${a}/${b}`));
            else gpu = null;
            gpuStop = null;
            vessel = new VesselTool({ scene, spacingMm: sp, divArr, getField: () => field, getPc: () => pcData, getMask: () => fieldMask, getCenterlines: () => centerlines, invalidate });
            for (const [k, label] of [['speed', 'speed (this frame)'], ['vorticity', 'vorticity'], ['q', 'Q-criterion (vortices)'], ['helicity', 'helicity density']]) $('vq').append(new Option(label, k));
            if (localTrace) clearStreamCache();
            for (let a = 0; a < 3; a++) {          // from now on the slice planes come from memory
                sac[a].abort(); sac[a] = new AbortController(); sgen[a]++; clearTimeout(ctimer[a]);
                planes[a].place(planes[a].index); loadSlice(a, 0);
            }
            status(`${X}×${Y}×${Z}, ${F} frames: velocity volume is in the browser (${localTrace ? (gpu ? 'WebGPU' : 'JS') + ' tracing, ' : 'streamlines stay on the server, '}vessel tool ready)`);
        } catch (e) { console.error('client volume', e); field = null; gpu = null; status(`client-side volume unavailable (${e.message}); using the server`); }
        clientStarted = false; setCompute();
        if (local() && ($('sl-on').checked)) updateStreamlines();
    }
    const clientReady = startClientLoad();

    frame = -1;
    setFrame(Math.floor(tau / dt) % F);
    for (let a = 0; a < 3; a++) loadSlice(a, 0);
    await new Promise((r) => setTimeout(r, 0));
    await slicesReady();
    status(`${X}×${Y}×${Z}, ${F} frames × ${(dt * 1e3).toFixed(1)} ms, voxel ${sp.map((s) => s.toFixed(2)).join('×')} mm, p99.9 speed ${vmaxDefault} m/s`);
    invalidate();
    window.hasty = { client, scene, camera, renderer, info, planes, stream, path, discs, setFrame, addPlane, updatePathlines, updateStreamlines, get field() { return field; }, get gpu() { return gpu; }, get vessel() { return vessel; }, runVessel, clientReady };

    if (selftest) {
        const out = { ok: false };
        const layers = (params.get('layers') ?? 'mip,slices,glyphs,stream,path,plane').split(',');
        try {
            await clientReady;
            out.client = !!field; out.gpu = !!gpu;
            mip.mesh.visible = $('mip-on').checked = layers.includes('mip');
            $('glyph-on').checked = layers.includes('glyphs');
            if (!layers.includes('slices')) $('s2-on').checked = false;
            if (layers.includes('coronal')) $('s1-on').checked = true;
            for (let a = 0; a < 3; a++) loadSlice(a, 0);
            await new Promise((r) => setTimeout(r, 0));
            await slicesReady();
            paintSlices();
            $('sl-on').checked = layers.includes('stream');
            await updateStreamlines();
            out.streamlines = stream.nLines;
            $('pl-on').checked = layers.includes('path');
            const pm = (await updatePathlines()) ?? { lines: -1 };
            out.pathlines = pm.lines;
            const j = layers.includes('plane') ? await addPlane(hot) : { net_ml: 0, area_mm2: 0, flow_ml_s: [0] };
            out.plane = { net_ml: j.net_ml, area_mm2: j.area_mm2, peak_flow_ml_s: Math.max(...j.flow_ml_s) };
            if (field && layers.includes('vessel')) {
                const i0 = hot[0] + X * (hot[1] + Y * hot[2]), v = runVessel({ p: hot, value: pcData[i0] });
                out.vessel = v ? { planes: v.planes.length, source: v.source, net_ml: v.avg.net_ml, area_mm2: v.avg.area_mm2 } : null;
            }
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
