import { LazyVolume, RemoteVolume } from './volume.js';
import { Renderer, normalizeComprepConfig }   from './renderer.js';
import { CACHE_DEPTH, CACHE_TE } from './renderer.js';
import {
    hexToUuid, fetchMetaString, initCommandIds,
    deleteValue, compressUi16Default, execute, fetchRaw, COMMAND_IDS,
} from '../../node_editor/src/grpc_client.js';

// ─── Entry point ──────────────────────────────────────────────────────────────

async function init() {
    // ── WebGPU availability check ────────────────────────────────────────────
    if (!navigator.gpu) {
        showError('WebGPU is not supported in this browser.\nTry Chrome 113+ or Edge 113+.');
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

    // ── Resolve command IDs from server ──────────────────────────────────────
    await initCommandIds();

    // ── Build volume ──────────────────────────────────────────────────────────
    const params    = new URLSearchParams(window.location.search);
    const uuidHex   = params.get('uuid');
    let originalUuid = null;
    const configStr = params.get('config');
    const comprepConfig = configStr ? JSON.parse(decodeURIComponent(configStr)) : null;
    let volume;
    if (uuidHex) {
        let uuid16 = hexToUuid(uuidHex);
        originalUuid = uuid16;
        const meta = await fetchMetaString(uuid16);
        if (!meta) { showError('Could not fetch tensor metadata for UUID: ' + uuidHex); return; }

        let resolvedUuid   = uuid16;
        let resolvedDtype  = meta.dtype;
        let resolvedConfig = comprepConfig;
        let tempUuid       = null;  // UUID to delete on page unload

        // Auto-comprep: delegate entirely to compress_ui16_default (C++ computes
        // min/max and q01/q99 internally, returns config as JSON in msg).
        const AUTOCOMPRESS = new Set(['f32', 'f64', 'f16', 'i8', 'u8', 'i16', 'i32', 'i64']);
        if (!resolvedConfig && AUTOCOMPRESS.has(meta.dtype)) {
            setStatus('Auto-compressing…');
            const { compressedUuid, config } = await compressUi16Default(uuid16);
            resolvedUuid   = compressedUuid;
            resolvedDtype  = 'i16';
            resolvedConfig = config;
            tempUuid       = resolvedUuid;

            document.getElementById('win-min').value = (config.a ?? 0).toFixed(4);
            document.getElementById('win-max').value = (config.b ?? 1).toFixed(4);
        }

        window.addEventListener('unload', () => {
            if (tempUuid) deleteValue(tempUuid).catch(() => {});
        });

        volume = new RemoteVolume(resolvedUuid, meta.shape, { dtype: resolvedDtype, comprepConfig: resolvedConfig });
        setStatus(`Remote tensor ${meta.shape.join('×')} [${meta.dtype}${resolvedConfig ? ' → comprep' : ''}] — fetching slices on demand`);
    } else {
        const shape = [8, 3, 400, 400, 400];
        volume = new LazyVolume(shape);
        setStatus(`LazyVolume ${shape.join('×')} — TE cache slots: ${CACHE_TE}, spatial depth: ${CACHE_DEPTH}`);
    }
    await nextFrame();

    // ── Create renderer ───────────────────────────────────────────────────────
    const renderer = new Renderer(device, volume);
    if (volume instanceof RemoteVolume) {
        volume.onUpdate = () => { renderer.invalidateCache(); renderer.setPosition({...renderer.pos}); };
    }

    // ── Attach canvases ───────────────────────────────────────────────────────
    const viewIds = [
        { key: 'axial',    id: 'axial-canvas'    },
        { key: 'sagittal', id: 'sagittal-canvas'  },
        { key: 'coronal',  id: 'coronal-canvas'   },
        { key: 'extra',    id: 'extra-canvas'     },
    ];

    for (const { key, id } of viewIds) {
        const canvas = document.getElementById(id);
        // Set the GPU framebuffer size to the CSS layout size.
        fitCanvas(canvas);
        renderer.attachCanvas(key, canvas);

        const overlay = document.getElementById(`${key}-overlay`);
        fitCanvas(overlay);
    }

    // ── Initial render (centre of volume) ────────────────────────────────────
    renderer.setPosition({
        t: 0,
        e: Math.floor(volume.E / 2),
        z: Math.floor(volume.Z / 2),
        y: Math.floor(volume.Y / 2),
        x: Math.floor(volume.X / 2),
    });
    updatePositionUI(renderer.pos);
    updateStatsUI(renderer.lastStats);
    setStatus('Ready');

    // Fetch full-volume statistics once (non-compressed units) for remote tensors.
    if (originalUuid) {
        try {
            await fetchAndDisplayStatistics(originalUuid);
        } catch (e) {
            console.warn('Failed to fetch statistics:', e);
        }
    }

    // ── Cursor state ──────────────────────────────────────────────────────────
    let showCursor = true;
    const cursorBtn = document.getElementById('cursor-toggle');

    function drawAllCursors() {
        const views = [
            { key: 'axial',    id: 'axial-overlay'    },
            { key: 'sagittal', id: 'sagittal-overlay'  },
            { key: 'coronal',  id: 'coronal-overlay'   },
            { key: 'extra',    id: 'extra-overlay'     },
        ];
        for (const { key, id } of views) {
            const overlay = document.getElementById(id);
            const { nx, ny } = viewFraction(key, renderer.pos, volume);
            drawCrosshair(overlay, nx, ny, showCursor);
        }
    }

    renderer.onPositionChange = () => {
        drawAllCursors();
    };

    cursorBtn.addEventListener('click', () => {
        showCursor = !showCursor;
        cursorBtn.textContent = `Cursor: ${showCursor ? 'ON' : 'OFF'}`;
        drawAllCursors();
    });
    document.addEventListener('keydown', e => {
        if (e.key === 'c' || e.key === 'C') cursorBtn.click();
    });

    // ── Colormap selector ─────────────────────────────────────────────────────
    document.getElementById('colormap-select').addEventListener('change', e => {
        renderer.colormap = parseInt(e.target.value, 10);
        renderer.render();
    });

    // ── Window / level ────────────────────────────────────────────────────────
    document.getElementById('win-min').addEventListener('change', e => {
        renderer.windowMin = parseFloat(e.target.value);
        renderer.render();
    });
    document.getElementById('win-max').addEventListener('change', e => {
        renderer.windowMax = parseFloat(e.target.value);
        renderer.render();
    });
    document.getElementById('auto-range-btn').addEventListener('click', () => {
        const cache = volume._cache;
        if (!cache || cache.size === 0) return;
        let mn = Infinity, mx = -Infinity;
        const isComprep = (volume.dtype === 'i16');
        const normCfg = isComprep ? normalizeComprepConfig(volume.comprepConfig ?? {}) : null;
        for (const data of cache.values()) {
            if (isComprep) {
                for (let i = 0; i < data.length; i++) {
                    const v = inverseToneMapU16(data[i], normCfg);
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
            } else {
                for (let i = 0; i < data.length; i++) {
                    const v = data[i];
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
            }
        }
        if (mn < mx) {
            renderer.windowMin = mn;
            renderer.windowMax = mx;
            document.getElementById('win-min').value = mn.toFixed(4);
            document.getElementById('win-max').value = mx.toFixed(4);
            renderer.render();
        }
    });

    // Info UI helpers: voxel value and basic cached-stats (non-compressed units)
    function inverseToneMapU16(u16, p) {
        // p is normalized comprep params from normalizeComprepConfig
        if (!p) return u16 / 65535.0;
        const y = u16 / 65535.0;
        const t0 = p.t0, t1 = p.t1;
        function inv_shape(uv, mode, gamma, logc) {
            if (mode === 1) return Math.pow(Math.max(uv, 0.0), gamma);
            if (mode === 2) {
                const c = Math.max(logc, 1e-6);
                return (Math.exp(uv * Math.log(1.0 + c)) - 1.0) / c;
            }
            return uv;
        }
        let x;
        if (y < t0 && t0 > 0.0) {
            const uv = inv_shape(y / t0, p.left_mode, p.left_gamma, p.left_logc);
            x = uv * (p.a - p.clampa) + p.clampa;
        } else if (y > t1 && t1 < 1.0) {
            const uv = inv_shape((y - t1) / (1.0 - t1), p.right_mode, p.right_gamma, p.right_logc);
            x = uv * (p.clampb - p.b) + p.b;
        } else {
            const span = t1 - t0;
            const uv = span < 1e-6 ? 0.0 : (y - t0) / span;
            x = uv * (p.b - p.a) + p.a;
        }
        return x;
    }

    function updateVoxelValueUI(pos, show) {
        const el = document.getElementById('voxel-value');
        if (!el) return;
        if (!show) { el.textContent = 'Value: —'; return; }
        // Only show a value if the exact axial slice is present in the JS cache.
        // RemoteVolume.getAxialSlice() returns zeros on cache-miss which would
        // otherwise mislead the UI — prefer an explicit "—" until the slice
        // has arrived.
        const key = `ax:${pos.t},${pos.e},${pos.z}`;
        if (!volume._cache || !volume._cache.has(key)) {
            el.textContent = 'Value: —';
            return;
        }
        const slice = volume._cache.get(key);
        const idx = pos.y * volume.X + pos.x;
        let val = slice[idx] ?? 0;
        if (volume.dtype === 'i16') {
            const p = normalizeComprepConfig(volume.comprepConfig ?? {});
            val = inverseToneMapU16(val, p);
        }
        el.textContent = `Value: ${val.toFixed(4)}`;
    }

    function updateImageStatsUI() {
        // Intentionally left blank. Full-volume statistics are fetched once
        // via gRPC and displayed by `fetchAndDisplayStatistics` to avoid
        // repeated expensive computations on mouse move.
    }

    // Parse a GenericValue string raw response (same format used in grpc_client).
    function parseGVStringRaw(raw) {
        if (!raw || raw.length < 9 || raw[0] !== 5) return null;
        const dv = new DataView(raw.buffer, raw.byteOffset + 1);
        const len = Number(dv.getBigUint64(0, true));
        return new TextDecoder().decode(raw.subarray(9, 9 + len));
    }

    async function fetchAndDisplayStatistics(uuid16) {
        if (!uuid16) return;
        if (!('statistics_string' in COMMAND_IDS)) {
            // COMMAND_IDS is in grpc_client; if not available, skip.
            // Try to refer to it dynamically to avoid import cycle.
        }
        try {
            const outIds = await execute(COMMAND_IDS.statistics_string, [uuid16]);
            if (!outIds || outIds.length === 0) throw new Error('No output from statistics_string');
            const tmp = outIds[0];
            const raw = await fetchRaw(tmp);
            const s = parseGVStringRaw(raw) ?? '';
            try { deleteValue(tmp).catch(() => {}); } catch {}
            // Parse numbers from the statistics string.
            const m = { min: '—', max: '—', mean: '—', std: '—' };
            const minM = s.match(/min=([^\n]+)/);
            const maxM = s.match(/max=([^\n]+)/);
            const meanM = s.match(/mean=([^\n]+)/);
            const stdM = s.match(/std=([^\n]+)/);
            if (minM) m.min = parseFloat(minM[1]).toFixed(4);
            if (maxM) m.max = parseFloat(maxM[1]).toFixed(4);
            if (meanM) m.mean = parseFloat(meanM[1]).toFixed(4);
            if (stdM) m.std = parseFloat(stdM[1]).toFixed(4);
            const el = document.getElementById('image-stats');
            if (el) el.textContent = `min: ${m.min} max: ${m.max} mean: ${m.mean} std: ${m.std}`;
        } catch (e) {
            const el = document.getElementById('image-stats');
            if (el) el.textContent = 'min: — max: — mean: — std: —';
            throw e;
        }
    }

    // ── Mouse interaction ─────────────────────────────────────────────────────
    setupInteraction(renderer, volume, drawAllCursors);

    // ── Resize handler ────────────────────────────────────────────────────────
    window.addEventListener('resize', () => {
        for (const { id } of viewIds) {
            fitCanvas(document.getElementById(id));
        }
        for (const { id } of [
            { id: 'axial-overlay' },
            { id: 'sagittal-overlay' },
            { id: 'coronal-overlay' },
            { id: 'extra-overlay' },
        ]) {
            fitCanvas(document.getElementById(id));
        }
        renderer.render();
        drawAllCursors();
    });

    // Initial cursor draw.
    drawAllCursors();

    // Expose for console debugging.
    window._renderer = renderer;
    window._volume   = volume;
}

// ─── Interaction ──────────────────────────────────────────────────────────────

function setupInteraction(renderer, volume, drawAllCursors) {
    const views = [
        {
            id: 'axial-canvas',
            fn: (nx, ny) => ({
                x: Math.min(Math.floor(clamp01(nx) * volume.X), volume.X - 1),
                y: Math.min(Math.floor(clamp01(ny) * volume.Y), volume.Y - 1),
            }),
        },
        {
            id: 'sagittal-canvas',
            fn: (nx, ny) => ({
                y: Math.min(Math.floor(clamp01(nx) * volume.Y), volume.Y - 1),
                z: Math.min(Math.floor(clamp01(ny) * volume.Z), volume.Z - 1),
            }),
        },
        {
            id: 'coronal-canvas',
            fn: (nx, ny) => ({
                x: Math.min(Math.floor(clamp01(nx) * volume.X), volume.X - 1),
                z: Math.min(Math.floor(clamp01(ny) * volume.Z), volume.Z - 1),
            }),
        },
        {
            // Extra view: horizontal = e (echo), vertical = t (time)
            id: 'extra-canvas',
            fn: (nx, ny) => ({
                e: Math.min(Math.floor(clamp01(nx) * volume.E), volume.E - 1),
                t: Math.min(Math.floor(clamp01(ny) * volume.T), volume.T - 1),
            }),
        },
    ];

    let activeView = null;  // { canvas, fn } while dragging

    function handleMove(e, canvas, fn) {
        const rect = canvas.getBoundingClientRect();
        const nx = (e.clientX - rect.left) / rect.width;
        const ny = (e.clientY - rect.top)  / rect.height;
        renderer.setPosition(fn(nx, ny));
        updatePositionUI(renderer.pos);
        updateStatsUI(renderer.lastStats);
        updateImageStatsUI();
    }

    for (const { id, fn } of views) {
        const canvas = document.getElementById(id);
        canvas.addEventListener('mousedown', e => {
            if (e.button !== 0) return;
            activeView = { canvas, fn };
            handleMove(e, canvas, fn);
            updateVoxelValueUI(renderer.pos, true);
            e.preventDefault();  // prevent text selection while dragging
        });

        // Throttled mousemove to update voxel value while left button pressed.
        canvas.addEventListener('mousemove', e => {
            if (!(e.buttons & 1)) return; // only when left button held
            const now = performance.now();
            const last = canvas._lastValTime || 0;
            if (now - last < 250) return;
            canvas._lastValTime = now;
            // If dragging, position already updated via global mousemove; just show value.
            updateVoxelValueUI(renderer.pos, true);
        });
    }

    document.addEventListener('mousemove', e => {
        if (!activeView) return;
        handleMove(e, activeView.canvas, activeView.fn);
    });

    document.addEventListener('mouseup', e => {
        if (e.button === 0) activeView = null;
        // Stop any polling and hide the voxel value display.
        for (const { id } of views) {
            const canvas = document.getElementById(id);
            if (canvas && typeof canvas._voxelPollId === 'number') {
                clearInterval(canvas._voxelPollId);
                canvas._voxelPollId = undefined;
            }
        }
        updateVoxelValueUI(renderer.pos, false);
    });

    // Scroll wheel to step through the perpendicular axis.
    function addWheelHandler(id, axis, max) {
        document.getElementById(id).addEventListener('wheel', e => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 1 : -1;
            const cur = renderer.pos[axis];
            renderer.setPosition({ [axis]: Math.min(Math.max(cur + delta, 0), max - 1) });
            updatePositionUI(renderer.pos);
            updateStatsUI(renderer.lastStats);
        }, { passive: false });
    }

    addWheelHandler('axial-canvas',    'z', volume.Z);
    addWheelHandler('sagittal-canvas', 'x', volume.X);
    addWheelHandler('coronal-canvas',  'y', volume.Y);
    addWheelHandler('extra-canvas',    't', volume.T);
}

// ─── Cursor helpers ───────────────────────────────────────────────────────────

/**
 * Maps the current voxel position to fractional (nx, ny) for a given view.
 * nx = horizontal fraction (column), ny = vertical fraction (row).
 *
 *   Axial    (z fixed) → nx = x/X,  ny = y/Y
 *   Sagittal (x fixed) → nx = y/Y,  ny = z/Z
 *   Coronal  (y fixed) → nx = x/X,  ny = z/Z
 */
function viewFraction(key, pos, volume) {
    switch (key) {
        case 'axial':    return { nx: (pos.x + 0.5) / volume.X, ny: (pos.y + 0.5) / volume.Y };
        case 'sagittal': return { nx: (pos.y + 0.5) / volume.Y, ny: (pos.z + 0.5) / volume.Z };
        case 'coronal':  return { nx: (pos.x + 0.5) / volume.X, ny: (pos.z + 0.5) / volume.Z };
        case 'extra':    return { nx: (pos.e + 0.5) / volume.E, ny: (pos.t + 0.5) / volume.T };
        default: return { nx: 0.5, ny: 0.5 };
    }
}

/**
 * Draw (or clear) the crosshair on a 2-D overlay canvas.
 * @param {HTMLCanvasElement} overlay
 * @param {number} nx  horizontal fraction [0, 1]
 * @param {number} ny  vertical   fraction [0, 1]
 * @param {boolean} visible
 */
function drawCrosshair(overlay, nx, ny, visible) {
    const w = overlay.width;
    const h = overlay.height;
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    if (!visible) return;

    const cx = nx * w;
    const cy = ny * h;
    const gap = 10;  // pixel gap around centre so dot stays readable

    ctx.save();
    ctx.strokeStyle = 'rgba(0, 215, 255, 0.80)';
    ctx.lineWidth = 1;

    // Horizontal line — two segments with a gap at the centre
    ctx.beginPath();
    ctx.moveTo(0,      cy);  ctx.lineTo(cx - gap, cy);
    ctx.moveTo(cx + gap, cy); ctx.lineTo(w,       cy);
    ctx.stroke();

    // Vertical line — two segments with a gap at the centre
    ctx.beginPath();
    ctx.moveTo(cx, 0);       ctx.lineTo(cx, cy - gap);
    ctx.moveTo(cx, cy + gap); ctx.lineTo(cx, h);
    ctx.stroke();

    // Centre dot
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0, 215, 255, 0.90)';
    ctx.fill();

    ctx.restore();
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/** Resize a canvas's pixel buffer to its current CSS layout size. */
function fitCanvas(canvas) {
    const w = canvas.offsetWidth  || 512;
    const h = canvas.offsetHeight || 512;
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width  = w;
        canvas.height = h;
    }
}

function updatePositionUI(pos) {
    document.getElementById('position-info').textContent =
        `(t=${pos.t}, e=${pos.e}, z=${pos.z}, y=${pos.y}, x=${pos.x})`;
}

function updateStatsUI(stats) {
    const el = document.getElementById('perf-stats');
    if (!el) return;
    el.textContent = `fill: ${stats.fillMs.toFixed(1)} ms | uploads: ${stats.uploadCount} slices`;
}

function setStatus(msg) {
    document.getElementById('status-msg').textContent = msg;
}

function showError(msg) {
    const el = document.getElementById('error-overlay');
    el.textContent = msg;
    el.classList.add('visible');
}

function clamp01(v) { return Math.max(0, Math.min(1, v)); }

/** Resolves after the next animation frame (lets the browser paint). */
function nextFrame() {
    return new Promise(resolve => requestAnimationFrame(resolve));
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

init().catch(err => {
    console.error(err);
    const msg = err.message ?? String(err);
    showError(`Initialisation failed:\n${msg.slice(0, 300)}${msg.length > 300 ? '\n…(see console for full error)' : ''}`);
});
