import { LazyVolume } from './volume.js';
import { Renderer }   from './renderer.js';
import { CACHE_DEPTH, CACHE_TE } from './renderer.js';

// ─── Entry point ──────────────────────────────────────────────────────────────

async function init() {
    // ── WebGPU availability check ────────────────────────────────────────────
    if (!navigator.gpu) {
        showError('WebGPU is not supported in this browser.\nTry Chrome 113+ or Edge 113+.');
        return;
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
        showError('No suitable GPU adapter found.');
        return;
    }
    const device = await adapter.requestDevice();
    device.lost.then(info => showError(`GPU device lost: ${info.message}`));

    // ── Build lazy volume (no pre-allocation — slices computed on demand) ─────
    // Shape: T=8, E=3, Z=400, Y=400, X=400
    // Full array would be ~6 GB; LazyVolume uses zero memory for the data.
    const shape = [8, 3, 400, 400, 400];
    const volume = new LazyVolume(shape);
    setStatus(`LazyVolume ${shape.join('×')} — TE cache slots: ${CACHE_TE}, spatial depth: ${CACHE_DEPTH}`);
    await nextFrame();

    // ── Create renderer ───────────────────────────────────────────────────────
    const renderer = new Renderer(device, volume);

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
    }

    // ── Initial render (centre of volume) ────────────────────────────────────
    renderer.setPosition({
        t: 0,
        e: 0,
        z: Math.floor(volume.Z / 2),
        y: Math.floor(volume.Y / 2),
        x: Math.floor(volume.X / 2),
    });
    updatePositionUI(renderer.pos);
    updateStatsUI(renderer.lastStats);
    setStatus('Ready');

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
            fitCanvas(overlay);
            const { nx, ny } = viewFraction(key, renderer.pos, volume);
            drawCrosshair(overlay, nx, ny, showCursor);
        }
    }

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

    // ── Mouse interaction ─────────────────────────────────────────────────────
    setupInteraction(renderer, volume, drawAllCursors);

    // ── Resize handler ────────────────────────────────────────────────────────
    window.addEventListener('resize', () => {
        for (const { id } of viewIds) {
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
        drawAllCursors();
    }

    for (const { id, fn } of views) {
        const canvas = document.getElementById(id);
        canvas.addEventListener('mousedown', e => {
            if (e.button !== 0) return;
            activeView = { canvas, fn };
            handleMove(e, canvas, fn);
            e.preventDefault();  // prevent text selection while dragging
        });
    }

    document.addEventListener('mousemove', e => {
        if (!activeView) return;
        handleMove(e, activeView.canvas, activeView.fn);
    });

    document.addEventListener('mouseup', e => {
        if (e.button === 0) activeView = null;
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
            drawAllCursors();
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
    showError(`Initialisation failed:\n${err.message}`);
});
