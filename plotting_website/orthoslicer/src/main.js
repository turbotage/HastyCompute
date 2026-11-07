import { LazyVolume } from './volume.js';
import { PlaneStore, LazySource } from './planes.js';
import { Renderer, normalizeComprepConfig, POOL_LAYERS } from './renderer.js';
import { HastyClient, firstTensor } from '../../common/hasty_client.js';

const client = new HastyClient().enableWorkers();     // slices are fetched and decoded off the main thread

// ─── Entry point ──────────────────────────────────────────────────────────────

async function init() {
    // ── WebGPU availability check ────────────────────────────────────────────
    if (!navigator.gpu) {
        showError('WebGPU is not supported in this browser.\nTry Chrome 113+, Edge 113+ or Safari 26+.');
        return;
    }
    let adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) adapter = await navigator.gpu.requestAdapter();
    if (!adapter) adapter = await navigator.gpu.requestAdapter({ forceFallbackAdapter: true });
    if (!adapter) {
        showError('No suitable GPU adapter found.\nOn Linux try: google-chrome --enable-features=Vulkan --enable-unsafe-webgpu');
        return;
    }
    const device = await adapter.requestDevice();
    device.lost.then(info => showError(`GPU device lost: ${info.message}`));

    // ── Resolve the value: ?id=<hex> (or legacy ?uuid=), optional &path=["key"][0] ────────────────────────────
    const params = new URLSearchParams(window.location.search);
    const idHex  = params.get('id') ?? params.get('uuid');
    const temps  = [];                                       // derived values, deleted when the page closes
    let source;
    if (idHex) {
        let info;
        try { info = await client.info(idHex); } catch (e) { showError(`No value ${idHex}: ${e.message}`); return; }
        let path = params.get('path') ?? '';
        let sum  = info.summary;
        if (!path) { const ft = firstTensor(sum); if (!ft) { showError(`Value ${info.name} contains no tensor`); return; } [path, sum] = ft; }
        else { const ft = firstTensor(sum); sum = ft && ft[0] === path ? ft[1] : null; }
        let tid = idHex;
        if (path) {
            setStatus(`Selecting ${path}…`);
            tid = (await client.execute('slice', { path }, [idHex], `${info.name}${path}`)).outputs[0];
            temps.push(tid);
            if (!sum) sum = (await client.info(tid)).summary;
        }
        const complex = sum.dtype.startsWith('complex');
        const mode = params.get('mode') ?? (complex ? 'abs' : 'real');
        // compressed u16 representation (comprep): ?config={...} full or partial (focus, left/right_mode, ...), else auto.
        // &prepped=1: the value already is a comprep u16 tensor and ?config is its full config (node editor)
        const userCfg = params.get('config') ? JSON.parse(decodeURIComponent(params.get('config'))) : {};
        const prepped = params.get('prepped') === '1';
        let dispId, config, stats;
        if (prepped) {
            if (sum.dtype !== 'uint16' || !('a' in userCfg)) { showError('prepped=1 needs a uint16 comprep tensor and its full config'); return; }
            dispId = tid;
            config = userCfg;
            stats = JSON.parse((await client.execute('stats', { mode: 'real' }, [tid])).message);
        } else {
            setStatus('Building the u16 display representation…');
            const [q, st] = await Promise.all([
                client.execute('comprep_u16', { mode, ...userCfg }, [tid], `${info.name} (display)`),
                client.execute('stats', { mode }, [tid]),
            ]);
            temps.push(q.outputs[0]);
            dispId = q.outputs[0];
            config = JSON.parse(q.message);
            stats = JSON.parse(st.message);
        }
        document.getElementById('win-min').value = Number(config.a).toFixed(4);
        document.getElementById('win-max').value = Number(config.b).toFixed(4);
        window.addEventListener('pagehide', () => { for (const t of temps) client.delete(t).catch(() => {}); });
        source = new PlaneStore(client, dispId, sum.shape, { dtype: 'i16', comprepConfig: config });
        setStatus(`${info.name}${path}: ${sum.shape.join('×')} ${sum.dtype}${complex ? ` (${mode})` : ''} — planes on demand`);
        document.title = `${info.name} — orthoslicer`;
        showStats(stats);
        // follow: switch to every newly shown value, reload when this one changes
        const follow = params.get('follow') !== '0';
        let reloadTimer = null;
        client.watch((e) => {
            if (follow && e.kind === 'show' && e.id && e.id !== idHex && (e.message || 'orthoslicer') === 'orthoslicer') {
                window.location.search = `?id=${e.id}`;
            } else if (e.kind === 'update' && e.id === idHex) {
                clearTimeout(reloadTimer);
                reloadTimer = setTimeout(() => window.location.reload(), 300);
            } else if (e.kind === 'delete' && e.id === idHex) {
                setStatus(`${info.name} was deleted on the server`);
            }
        });
    } else {
        const shape = [8, 3, 400, 400, 400];
        source = new LazySource(new LazyVolume(shape));
        setStatus(`Synthetic ${shape.join('×')} — GPU pool: ${POOL_LAYERS} planes per view (open with ?id=<value id>)`);
    }
    await nextFrame();

    // ── Create renderer ───────────────────────────────────────────────────────
    const renderer = new Renderer(device, source);
    window.hasty = { source, renderer, client };   // debug / headless-test handle

    // ── Attach canvases ───────────────────────────────────────────────────────
    const viewIds = [
        { key: 'axial',    id: 'axial-canvas'    },
        { key: 'sagittal', id: 'sagittal-canvas'  },
        { key: 'coronal',  id: 'coronal-canvas'   },
        { key: 'extra',    id: 'extra-canvas'     },
    ];
    for (const { key, id } of viewIds) {
        const canvas = document.getElementById(id);
        fitCanvas(canvas);                                     // GPU framebuffer = CSS size × devicePixelRatio
        renderer.attachCanvas(key, canvas);
        fitCanvas(document.getElementById(`${key}-overlay`));
    }

    // ── Cursor overlay ────────────────────────────────────────────────────────
    let showCursor = true;
    const cursorBtn = document.getElementById('cursor-toggle');
    function drawAllCursors() {
        for (const { key } of viewIds) {
            const { nx, ny } = viewFraction(key, renderer.pos, source);
            drawCrosshair(document.getElementById(`${key}-overlay`), nx, ny, showCursor);
        }
    }
    cursorBtn.addEventListener('click', () => {
        showCursor = !showCursor;
        cursorBtn.textContent = `Cursor: ${showCursor ? 'ON' : 'OFF'}`;
        drawAllCursors();
    });

    // ── Voxel readout while dragging ──────────────────────────────────────────
    let dragging = false;
    function updateVoxelValueUI() {
        const el = document.getElementById('voxel-value');
        if (!el) return;
        if (!dragging) { el.textContent = 'Value: —'; return; }
        const slice = renderer.shownPlane('axial');            // only a plane that is really on screen (no zeros)
        if (!slice) { el.textContent = 'Value: —'; return; }
        const { pos } = renderer;
        let val = slice[pos.y * source.X + pos.x] ?? 0;
        if (source.dtype === 'i16') val = inverseToneMapU16(val, normalizeComprepConfig(source.comprepConfig ?? {}));
        el.textContent = `Value: ${val.toFixed(4)}`;
    }

    // ── Per-frame UI (position text, crosshairs, perf line) ───────────────────
    let ready = !idHex;
    renderer.onFrame = (stats, posChanged) => {
        if (!ready && renderer.shownPlane('axial')) { ready = true; setStatus('Ready'); }
        if (posChanged) {
            updatePositionUI(renderer.pos);
            drawAllCursors();
        }
        if (dragging) updateVoxelValueUI();
        updateStatsUI(stats, source);
    };

    renderer.setPosition({ t: 0 });
    renderer.requestFrame();
    if (idHex) setStatus('Loading planes…'); else setStatus('Ready');

    // ── Controls ──────────────────────────────────────────────────────────────
    document.addEventListener('keydown', e => {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        if (e.key === 'c' || e.key === 'C') cursorBtn.click();
        if (e.key === 'i' || e.key === 'I') { renderer.smooth = !renderer.smooth; renderer.invalidateAll(); setStatus(`interpolation ${renderer.smooth ? 'on' : 'off'}`); }
    });
    document.getElementById('colormap-select').addEventListener('change', e => {
        renderer.colormap = parseInt(e.target.value, 10);
        renderer.invalidateAll();
    });
    document.getElementById('win-min').addEventListener('change', e => { renderer.windowMin = parseFloat(e.target.value); renderer.invalidateAll(); });
    document.getElementById('win-max').addEventListener('change', e => { renderer.windowMax = parseFloat(e.target.value); renderer.invalidateAll(); });
    document.getElementById('auto-range-btn').addEventListener('click', () => {
        // the u16 → value map is monotonic, so min/max of the codes of the three planes on screen are enough
        let mn = Infinity, mx = -Infinity;
        const isComprep = source.dtype === 'i16';
        for (const name of ['axial', 'sagittal', 'coronal']) {
            const d = renderer.shownPlane(name);
            if (!d) continue;
            for (let i = 0; i < d.length; i++) { const v = d[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
        }
        if (!(mn <= mx)) return;
        if (isComprep) {
            const p = normalizeComprepConfig(source.comprepConfig ?? {});
            mn = inverseToneMapU16(mn, p); mx = inverseToneMapU16(mx, p);
        }
        if (mn < mx) {
            renderer.windowMin = mn;
            renderer.windowMax = mx;
            document.getElementById('win-min').value = mn.toFixed(4);
            document.getElementById('win-max').value = mx.toFixed(4);
            renderer.invalidateAll();
        }
    });

    setupInteraction(renderer, source, (d) => { dragging = d; updateVoxelValueUI(); });

    // ── Resize (layout changes, window resize, devicePixelRatio) ──────────────
    const refit = () => {
        for (const { key, id } of viewIds) { fitCanvas(document.getElementById(id)); fitCanvas(document.getElementById(`${key}-overlay`)); }
        renderer.invalidateAll();
        drawAllCursors();
    };
    new ResizeObserver(refit).observe(document.getElementById('viewport-grid'));
    window.addEventListener('resize', refit);

    window._renderer = renderer;
    window._volume   = source;
}

/** inverse of the u16 comprep tone map (mirrors the WGSL decode) */
function inverseToneMapU16(u16, p) {
    if (!p) return u16 / 65535.0;
    const y = u16 / 65535.0;
    const t0 = p.t0, t1 = p.t1;
    const inv = (uv, mode, gamma, logc) => {
        if (mode === 1) return Math.pow(Math.max(uv, 0.0), gamma);
        if (mode === 2) { const c = Math.max(logc, 1e-6); return (Math.exp(uv * Math.log(1.0 + c)) - 1.0) / c; }
        return uv;
    };
    if (y < t0 && t0 > 0.0) return inv(y / t0, p.left_mode, p.left_gamma, p.left_logc) * (p.a - p.clampa) + p.clampa;
    if (y > t1 && t1 < 1.0) return inv((y - t1) / (1.0 - t1), p.right_mode, p.right_gamma, p.right_logc) * (p.clampb - p.b) + p.b;
    const span = t1 - t0;
    return (span < 1e-6 ? 0.0 : (y - t0) / span) * (p.b - p.a) + p.a;
}

// ─── Interaction ──────────────────────────────────────────────────────────────

function setupInteraction(renderer, source, setDragging) {
    const px = (v, n) => Math.min(Math.floor(clamp01(v) * n), n - 1);
    const views = [
        { id: 'axial-canvas',    fn: (nx, ny) => ({ x: px(nx, source.X), y: px(ny, source.Y) }) },
        { id: 'sagittal-canvas', fn: (nx, ny) => ({ y: px(nx, source.Y), z: px(ny, source.Z) }) },
        { id: 'coronal-canvas',  fn: (nx, ny) => ({ x: px(nx, source.X), z: px(ny, source.Z) }) },
        // Extra view: horizontal = e (echo), vertical = t (time)
        { id: 'extra-canvas',    fn: (nx, ny) => ({ e: px(nx, source.E), t: px(ny, source.T) }) },
    ];

    function moveTo(ev, canvas, fn) {
        const rect = canvas.getBoundingClientRect();
        renderer.setPosition(fn((ev.clientX - rect.left) / rect.width, (ev.clientY - rect.top) / rect.height));   // coalesced per frame
    }

    for (const { id, fn } of views) {
        const canvas = document.getElementById(id);
        canvas.style.touchAction = 'none';
        let active = false;
        canvas.addEventListener('pointerdown', ev => {
            if (ev.button !== 0) return;
            active = true;
            canvas.setPointerCapture(ev.pointerId);
            setDragging(true);
            moveTo(ev, canvas, fn);
            ev.preventDefault();
        });
        canvas.addEventListener('pointermove', ev => { if (active) moveTo(ev, canvas, fn); });
        const end = () => { if (active) { active = false; setDragging(false); } };
        canvas.addEventListener('pointerup', end);
        canvas.addEventListener('pointercancel', end);
    }

    // Scroll wheel / trackpad: accumulate the delta, one slice per ~40 px so a trackpad flick does not skip through the volume.
    function addWheelHandler(id, axis, max) {
        let acc = 0;
        document.getElementById(id).addEventListener('wheel', ev => {
            ev.preventDefault();
            acc += ev.deltaY * (ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? 400 : 1);
            const steps = Math.trunc(acc / 40);
            if (!steps) return;
            acc -= steps * 40;
            const cur = renderer.pos[axis];
            renderer.setPosition({ [axis]: Math.min(Math.max(cur + steps, 0), max - 1) });
        }, { passive: false });
    }
    addWheelHandler('axial-canvas',    'z', source.Z);
    addWheelHandler('sagittal-canvas', 'x', source.X);
    addWheelHandler('coronal-canvas',  'y', source.Y);
    addWheelHandler('extra-canvas',    't', source.T);
}

// ─── Cursor helpers ───────────────────────────────────────────────────────────

/**
 * Maps the current voxel position to fractional (nx, ny) for a given view.
 *   Axial    (z fixed) → nx = x/X,  ny = y/Y
 *   Sagittal (x fixed) → nx = y/Y,  ny = z/Z
 *   Coronal  (y fixed) → nx = x/X,  ny = z/Z
 */
function viewFraction(key, pos, v) {
    switch (key) {
        case 'axial':    return { nx: (pos.x + 0.5) / v.X, ny: (pos.y + 0.5) / v.Y };
        case 'sagittal': return { nx: (pos.y + 0.5) / v.Y, ny: (pos.z + 0.5) / v.Z };
        case 'coronal':  return { nx: (pos.x + 0.5) / v.X, ny: (pos.z + 0.5) / v.Z };
        case 'extra':    return { nx: (pos.e + 0.5) / v.E, ny: (pos.t + 0.5) / v.T };
        default: return { nx: 0.5, ny: 0.5 };
    }
}

/** Draw (or clear) the crosshair on a 2-D overlay canvas (sized in device pixels). */
function drawCrosshair(overlay, nx, ny, visible) {
    const w = overlay.width, h = overlay.height;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    if (!visible) return;
    const dpr = window.devicePixelRatio || 1;
    const cx = nx * w, cy = ny * h, gap = 10 * dpr;
    ctx.save();
    ctx.strokeStyle = 'rgba(0, 215, 255, 0.80)';
    ctx.lineWidth = dpr;
    ctx.beginPath();
    ctx.moveTo(0, cy);        ctx.lineTo(cx - gap, cy);
    ctx.moveTo(cx + gap, cy); ctx.lineTo(w, cy);
    ctx.moveTo(cx, 0);        ctx.lineTo(cx, cy - gap);
    ctx.moveTo(cx, cy + gap); ctx.lineTo(cx, h);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(cx, cy, 3 * dpr, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0, 215, 255, 0.90)';
    ctx.fill();
    ctx.restore();
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Size a canvas's pixel buffer to its CSS layout size × devicePixelRatio. */
function fitCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.min(4096, Math.max(1, Math.round((canvas.clientWidth  || 512) * dpr)));
    const h = Math.min(4096, Math.max(1, Math.round((canvas.clientHeight || 512) * dpr)));
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
}

function updatePositionUI(pos) {
    document.getElementById('position-info').textContent =
        `(t=${pos.t}, e=${pos.e}, z=${pos.z}, y=${pos.y}, x=${pos.x})`;
}

let lastStatsText = 0;
function updateStatsUI(stats, source) {
    const now = performance.now();
    if (now - lastStatsText < 250) return;            // text updates are not free either
    lastStatsText = now;
    const el = document.getElementById('perf-stats');
    if (!el) return;
    const mb = (source.bytes ?? 0) / 1048576;
    el.textContent = `frame: ${stats.frameMs.toFixed(1)} ms | net: ${source.active} active, ${source.queued} queued | cache: ${mb.toFixed(0)} MB, ${source.stats.requests} req`;
}

function setStatus(msg) { document.getElementById('status-msg').textContent = msg; }

function showError(msg) {
    const el = document.getElementById('error-overlay');
    el.textContent = msg;
    el.classList.add('visible');
}

function clamp01(v) { return Math.max(0, Math.min(1, v)); }

/** Resolves after the next animation frame (lets the browser paint). */
function nextFrame() { return new Promise(resolve => requestAnimationFrame(resolve)); }

function showStats(st) {
    const el = document.getElementById('image-stats');
    if (!el) return;
    const f = (v) => (typeof v === 'number' && Number.isFinite(v) ? v.toPrecision(5) : '—');
    el.textContent = `min: ${f(st.min)} max: ${f(st.max)} mean: ${f(st.mean)} std: ${f(st.std)}`;
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

init().catch(err => {
    console.error(err);
    const msg = err.message ?? String(err);
    showError(`Initialisation failed:\n${msg.slice(0, 300)}${msg.length > 300 ? '\n…(see console for full error)' : ''}`);
});
