// ─── GPU plane pool ───────────────────────────────────────────────────────────
//
// Each spatial view owns one texture_2d_array of POOL_LAYERS layers; a layer holds one plane. Which plane sits in
// which layer is an LRU map (key → layer). Planes are uploaded exactly once, when they arrive from the PlaneStore
// (or when a cached plane becomes wanted again), and are never re-uploaded just because the cursor moved. Showing a
// plane is a uniform change (the layer index). Rendering is coalesced to one pass per animation frame and only the
// views that changed are redrawn.

import { computeWanted, planeKey, VIEWS, VIEW_AXIS } from './planes.js';

/** layers per spatial view (≈ 128 × 320 KB = 40 MB at 400×400) */
export const POOL_LAYERS = 128;
/** plane uploads per frame (the displayed planes always go through) */
const UPLOAD_BUDGET = 12;
/** prefetch radii: while the cursor moves / after it rested */
const ACTIVE_PREFETCH = { radius: 3, tRadius: 0, eRadius: 0 };
const IDLE_PREFETCH   = { radius: 10, tRadius: 2, eRadius: 1 };
const IDLE_MS = 250;

// ─── Float32 → Float16 ───────────────────────────────────────────────────────

const HAS_F16 = typeof Float16Array !== 'undefined';

function f32ToF16Array(src) {
    if (HAS_F16) { const h = new Float16Array(src); return new Uint16Array(h.buffer, h.byteOffset, h.length); }
    const n = src.length, dst = new Uint16Array(n);
    const buf = new ArrayBuffer(4), fv = new Float32Array(buf), iv = new Int32Array(buf);
    for (let i = 0; i < n; i++) {
        fv[0] = src[i];
        const x = iv[0], sign = (x >>> 16) & 0x8000, exp = (x >>> 23) & 0xff, mant = x & 0x007fffff;
        let bits;
        if (exp === 0xff) bits = sign | 0x7c00 | (mant ? 0x0200 : 0);
        else if (exp === 0) bits = sign;
        else {
            const e16 = exp - 127 + 15;
            if (e16 >= 31) bits = sign | 0x7c00;
            else if (e16 <= 0) bits = sign;
            else bits = sign | (e16 << 10) | (mant >>> 13);
        }
        dst[i] = bits;
    }
    return dst;
}

// ─── normalizeComprepConfig ───────────────────────────────────────────────────
//
// Converts a raw comprep JSON config (from URL or user input) into the flat
// numeric object used by ViewState.writeUniforms.  All fields have defaults
// matching the C++ compress_ui16_config defaults.

export function normalizeComprepConfig(cfg = {}) {
    const modeMap = { linear: 0, gamma: 1, log: 2 };
    const a      = cfg.a      ?? 0.0;
    const b      = cfg.b      ?? 1.0;
    const clampa = cfg.clampa ?? a;
    const clampb = cfg.clampb ?? b;
    const focus  = cfg.focus  ?? 1.0;
    const t0     = (1.0 - focus) * 0.5;
    const t1     = 1.0 - t0;
    return {
        a, b, clampa, clampb, t0, t1,
        left_mode:   modeMap[cfg.left_mode   ?? 'linear'] ?? 0,
        right_mode:  modeMap[cfg.right_mode  ?? 'linear'] ?? 0,
        left_gamma:  cfg.left_gamma  ?? 1.0,
        right_gamma: cfg.right_gamma ?? 1.0,
        left_logc:   cfg.left_logc   ?? 1.0,
        right_logc:  cfg.right_logc  ?? 1.0,
    };
}

// ─── Shaders ──────────────────────────────────────────────────────────────────
//
// Both render one layer of a texture_2d_array onto a full-screen quad with textureLoad (no sampler, so r16float and
// r16uint need no filtering features). `smooth` = 1 interpolates bilinearly between texels in the shader (after the
// u16 → physical decode in comprep mode, so the tone curve is respected); 0 = nearest.
// Colormap IDs:  0=Gray  1=Viridis  2=Plasma  3=Hot  4=Cool

const COMMON_WGSL = /* wgsl */`
struct VOut {
    @builtin(position) pos : vec4f,
    @location(0)       uv  : vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VOut {
    var pos = array<vec2f, 6>(
        vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
        vec2f(-1.0, -1.0), vec2f( 1.0,  1.0), vec2f(-1.0,  1.0)
    );
    var uvs = array<vec2f, 6>(
        vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0),
        vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(0.0, 0.0)
    );
    var out : VOut;
    out.pos = vec4f(pos[vi], 0.0, 1.0);
    out.uv  = uvs[vi];
    return out;
}

fn cm_viridis(t: f32) -> vec3f {
    let c0 = vec3f( 0.27773, 0.00541, 0.33410);
    let c1 = vec3f( 0.10509, 1.40461, 1.38459);
    let c2 = vec3f(-0.33086, 0.21485, 0.09510);
    let c3 = vec3f(-4.63423,-5.79910,-19.33244);
    let c4 = vec3f( 6.22827,14.17993, 56.69055);
    let c5 = vec3f( 4.77638,-13.74515,-65.35303);
    let c6 = vec3f(-5.43546, 4.64585, 26.31244);
    return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3f(0.0), vec3f(1.0));
}
fn cm_plasma(t: f32) -> vec3f {
    let c0 = vec3f( 0.05873, 0.02334, 0.54334);
    let c1 = vec3f( 2.17651, 0.23838, 0.75396);
    let c2 = vec3f(-2.68946,-7.45585, 3.11080);
    let c3 = vec3f( 6.13035,42.34619,-28.51885);
    let c4 = vec3f(-11.10744,-82.66631, 60.13985);
    let c5 = vec3f(10.02307, 71.41362,-54.07219);
    let c6 = vec3f(-3.65871,-22.93153, 18.19191);
    return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3f(0.0), vec3f(1.0));
}
fn cm_hot(t: f32) -> vec3f { return clamp(vec3f(t * 3.0, t * 3.0 - 1.0, t * 3.0 - 2.0), vec3f(0.0), vec3f(1.0)); }
fn cm_cool(t: f32) -> vec3f { return vec3f(t, 1.0 - t, 1.0); }
fn apply_colormap(v: f32, cm: u32) -> vec3f {
    if cm == 1u { return cm_viridis(v); }
    if cm == 2u { return cm_plasma(v);  }
    if cm == 3u { return cm_hot(v);     }
    if cm == 4u { return cm_cool(v);    }
    return vec3f(v, v, v);
}
`;

// float mode: r16float, values are the data itself.  Uniforms (32 bytes).
const SLICE_SHADER = /* wgsl */`
struct Uniforms {
    slice_layer : u32,
    tex_width   : u32,
    tex_height  : u32,
    colormap    : u32,
    window_min  : f32,
    window_max  : f32,
    smooth_     : u32,
    _pad        : f32,
};
@group(0) @binding(0) var<uniform> u   : Uniforms;
@group(0) @binding(1) var          tex : texture_2d_array<f32>;
${COMMON_WGSL}
fn texel(px: i32, py: i32) -> f32 {
    let cx = clamp(px, 0, i32(u.tex_width)  - 1);
    let cy = clamp(py, 0, i32(u.tex_height) - 1);
    return textureLoad(tex, vec2i(cx, cy), i32(u.slice_layer), 0).r;
}

@fragment
fn fs_main(in : VOut) -> @location(0) vec4f {
    var raw : f32;
    if u.smooth_ == 0u {
        raw = texel(i32(in.uv.x * f32(u.tex_width)), i32(in.uv.y * f32(u.tex_height)));
    } else {
        let p  = in.uv * vec2f(f32(u.tex_width), f32(u.tex_height)) - vec2f(0.5);
        let f  = floor(p);
        let fr = p - f;
        let x0 = i32(f.x); let y0 = i32(f.y);
        raw = mix(mix(texel(x0, y0),     texel(x0 + 1, y0),     fr.x),
                  mix(texel(x0, y0 + 1), texel(x0 + 1, y0 + 1), fr.x), fr.y);
    }
    let v = clamp((raw - u.window_min) / (u.window_max - u.window_min), 0.0, 1.0);
    return vec4f(apply_colormap(v, u.colormap), 1.0);
}
`;

// comprep mode: r16uint, u16 → y=[0,1] → piecewise inverse tone-map → physical x → window/level → colormap.
// Uniforms (80 bytes = 20 × 4).
const COMPREP_SHADER = /* wgsl */`
struct Uniforms {
    slice_layer  : u32,
    tex_width    : u32,
    tex_height   : u32,
    colormap     : u32,
    window_min   : f32,
    window_max   : f32,
    a            : f32,
    b            : f32,
    clampa       : f32,
    clampb       : f32,
    t0           : f32,
    t1           : f32,
    left_mode    : u32,
    right_mode   : u32,
    left_gamma   : f32,
    right_gamma  : f32,
    left_logc    : f32,
    right_logc   : f32,
    smooth_      : u32,
    _pad1        : f32,
};
@group(0) @binding(0) var<uniform> u   : Uniforms;
@group(0) @binding(1) var          tex : texture_2d_array<u32>;
${COMMON_WGSL}
// inverse of the shaping applied per region: linear identity, gamma pow(u,γ), log (exp(u·log(1+c))−1)/c
fn inv_shape(uv: f32, mode: u32, gamma: f32, logc: f32) -> f32 {
    if mode == 1u { return pow(max(uv, 0.0), gamma); }
    if mode == 2u {
        let c = max(logc, 1e-6);
        return (exp(uv * log(1.0 + c)) - 1.0) / c;
    }
    return uv;
}

fn decode(raw: u32) -> f32 {
    let y = f32(raw) / 65535.0;
    if y < u.t0 && u.t0 > 0.0 {
        let uv = inv_shape(y / u.t0, u.left_mode, u.left_gamma, u.left_logc);
        return uv * (u.a - u.clampa) + u.clampa;
    } else if y > u.t1 && u.t1 < 1.0 {
        let uv = inv_shape((y - u.t1) / (1.0 - u.t1), u.right_mode, u.right_gamma, u.right_logc);
        return uv * (u.clampb - u.b) + u.b;
    }
    let span = u.t1 - u.t0;
    let uv   = select((y - u.t0) / span, 0.0, span < 1e-6);
    return uv * (u.b - u.a) + u.a;
}

fn texel(px: i32, py: i32) -> f32 {
    let cx = clamp(px, 0, i32(u.tex_width)  - 1);
    let cy = clamp(py, 0, i32(u.tex_height) - 1);
    return decode(textureLoad(tex, vec2i(cx, cy), i32(u.slice_layer), 0).r);
}

@fragment
fn fs_main(in : VOut) -> @location(0) vec4f {
    var x : f32;
    if u.smooth_ == 0u {
        x = texel(i32(in.uv.x * f32(u.tex_width)), i32(in.uv.y * f32(u.tex_height)));
    } else {
        let p  = in.uv * vec2f(f32(u.tex_width), f32(u.tex_height)) - vec2f(0.5);
        let f  = floor(p);
        let fr = p - f;
        let x0 = i32(f.x); let y0 = i32(f.y);
        x = mix(mix(texel(x0, y0),     texel(x0 + 1, y0),     fr.x),
                mix(texel(x0, y0 + 1), texel(x0 + 1, y0 + 1), fr.x), fr.y);
    }
    let v = clamp((x - u.window_min) / (u.window_max - u.window_min), 0.0, 1.0);
    return vec4f(apply_colormap(v, u.colormap), 1.0);
}
`;

// ─── ViewState ────────────────────────────────────────────────────────────────

class ViewState {
    /**
     * @param {number} numLayers  pool size (1 for the extra T×E view)
     * @param {'float'|'comprep'} mode  'float' = r16float, 'comprep' = r16uint + inverse tone-map shader
     */
    constructor(device, canvas, label, texWidth, texHeight, numLayers, mode, comprepParams) {
        this.device = device;
        this.label = label;
        this.texWidth = texWidth;
        this.texHeight = texHeight;
        this.numLayers = numLayers;
        this.mode = mode;
        this.comprepParams = comprepParams;
        this.dirty = true;

        // LRU of resident planes: key → layer (Map order = least → most recently used)
        this.resident = new Map();
        this.freeLayers = Array.from({ length: numLayers }, (_, i) => numLayers - 1 - i);
        this.shownKey = null;
        this.shownLayer = 0;

        this.ctx = canvas.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.ctx.configure({ device, format: this.format, alphaMode: 'opaque' });

        this.cacheTex = device.createTexture({
            label: `pool-${label}`,
            size: { width: texWidth, height: texHeight, depthOrArrayLayers: numLayers },
            format: mode === 'comprep' ? 'r16uint' : 'r16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        const shaderModule = device.createShaderModule({ label: `shader-${label}`, code: mode === 'comprep' ? COMPREP_SHADER : SLICE_SHADER });
        this.bgl = device.createBindGroupLayout({
            label: `bgl-${label}`,
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.FRAGMENT,
                  texture: { sampleType: mode === 'comprep' ? 'uint' : 'unfilterable-float', viewDimension: '2d-array', multisampled: false } },
            ],
        });
        this.pipeline = device.createRenderPipeline({
            label: `pipeline-${label}`,
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.bgl] }),
            vertex: { module: shaderModule, entryPoint: 'vs_main' },
            fragment: { module: shaderModule, entryPoint: 'fs_main', targets: [{ format: this.format }] },
            primitive: { topology: 'triangle-list' },
        });
        const ubytes = mode === 'comprep' ? 80 : 32;
        this.uniformBuf = device.createBuffer({ label: `uniform-${label}`, size: ubytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this._ubuf = new ArrayBuffer(ubytes);
        this._u32 = new Uint32Array(this._ubuf);
        this._f32 = new Float32Array(this._ubuf);
        this.bindGroup = device.createBindGroup({
            label: `bg-${label}`,
            layout: this.bgl,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuf } },
                { binding: 1, resource: this.cacheTex.createView({ dimension: '2d-array', baseArrayLayer: 0, arrayLayerCount: numLayers }) },
            ],
        });
    }

    has(key) { return this.resident.has(key); }
    layerOf(key) { return this.resident.get(key); }
    touch(key) { const l = this.resident.get(key); if (l !== undefined) { this.resident.delete(key); this.resident.set(key, l); } }

    /** make `key` resident (uploading `data` if it is not); returns the layer */
    ensure(key, data) {
        const have = this.resident.get(key);
        if (have !== undefined) { this.touch(key); return have; }
        let layer = this.freeLayers.pop();
        if (layer === undefined) {
            for (const [k, l] of this.resident) {          // least recently used first; the plane on screen stays
                if (k === this.shownKey) continue;
                this.resident.delete(k);
                layer = l;
                break;
            }
        }
        this.resident.set(key, layer);
        this.uploadLayer(layer, data);
        return layer;
    }

    uploadLayer(layer, data) {
        const bytes = data instanceof Uint16Array ? data : f32ToF16Array(data);
        this.device.queue.writeTexture(
            { texture: this.cacheTex, origin: { x: 0, y: 0, z: layer } },
            bytes,
            { bytesPerRow: this.texWidth * 2, rowsPerImage: this.texHeight },
            { width: this.texWidth, height: this.texHeight, depthOrArrayLayers: 1 },
        );
    }

    writeUniforms(layer, windowMin, windowMax, colormap, smooth) {
        const u32 = this._u32, f32 = this._f32;
        u32[0] = layer; u32[1] = this.texWidth; u32[2] = this.texHeight; u32[3] = colormap | 0;
        f32[4] = windowMin; f32[5] = windowMax;
        if (this.mode === 'comprep') {
            const p = this.comprepParams;
            f32[6] = p.a; f32[7] = p.b; f32[8] = p.clampa; f32[9] = p.clampb; f32[10] = p.t0; f32[11] = p.t1;
            u32[12] = p.left_mode; u32[13] = p.right_mode;
            f32[14] = p.left_gamma; f32[15] = p.right_gamma; f32[16] = p.left_logc; f32[17] = p.right_logc;
            u32[18] = smooth ? 1 : 0;
        } else u32[6] = smooth ? 1 : 0;
        this.device.queue.writeBuffer(this.uniformBuf, 0, this._ubuf);
    }

    destroy() { this.cacheTex.destroy(); this.uniformBuf.destroy(); }
}

// ─── Renderer ────────────────────────────────────────────────────────────────

/**
 * Four WebGPU views: axial, sagittal, coronal and extra (T×E at the cursor voxel).
 *
 * `source` supplies planes (PlaneStore / LazySource: peek, setWanted, peekExtra, requestExtra, onPlane, onExtra).
 * setPosition() only records the position; the work happens once per animation frame in _frame():
 *   1. tell the source which planes are wanted (cursor planes first, then neighbours, widening once the cursor rests),
 *   2. upload newly available wanted planes to the GPU pool (bounded per frame),
 *   3. point each view at its current plane (held at the last good plane while the new one is still on its way),
 *   4. draw only the views whose plane / window / colormap changed.
 */
export class Renderer {
    constructor(device, source) {
        this.device = device;
        this.source = source;
        this.volume = source;                        // legacy alias
        this.pos = { t: Math.floor(source.T / 2), e: Math.floor(source.E / 2), z: Math.floor(source.Z / 2), y: Math.floor(source.Y / 2), x: Math.floor(source.X / 2) };
        this._views = new Map();
        this.colormap = 0;
        this.smooth = true;
        this.onFrame = null;                         // (stats, posChanged) after every frame that did work
        this.lastStats = { uploadCount: 0, frameMs: 0 };

        this._mode = source.dtype === 'i16' ? 'comprep' : 'float';
        this._comprepParams = this._mode === 'comprep' ? normalizeComprepConfig(source.comprepConfig ?? {}) : null;
        if (this._comprepParams) { this.windowMin = this._comprepParams.a; this.windowMax = this._comprepParams.b; }
        else { this.windowMin = 0.0; this.windowMax = 1.0; }

        this._raf = 0;
        this._posChanged = true;
        this._motion = {};
        this._prefetch = ACTIVE_PREFETCH;
        this._idleTimer = 0;
        this._uploadsThisFrame = 0;
        this._more = false;
        this._extraKey = null;
        source.onPlane = () => this.requestFrame();
        source.onExtra = () => this.requestFrame();
    }

    attachCanvas(key, canvas) {
        const s = this.source;
        const dimMap = {
            axial:    [s.X, s.Y, POOL_LAYERS],
            sagittal: [s.Y, s.Z, POOL_LAYERS],
            coronal:  [s.X, s.Z, POOL_LAYERS],
            extra:    [s.E, s.T, 1],
        };
        if (!dimMap[key]) throw new Error(`Unknown view key: ${key}`);
        const [tw, th, nl] = dimMap[key];
        this._views.set(key, new ViewState(this.device, canvas, key, tw, th, nl, this._mode, this._comprepParams));
    }

    // ── scheduling ────────────────────────────────────────────────────────────

    requestFrame() {
        if (this._raf) return;
        this._raf = requestAnimationFrame(() => { this._raf = 0; this._frame(); });
    }

    /** mark every view for redraw (window, colormap, resize) */
    invalidateAll() { for (const v of this._views.values()) v.dirty = true; this.requestFrame(); }
    render() { this.invalidateAll(); }

    /** record a (partial) new position; the frame loop does the rest */
    setPosition(partial) {
        const old = this.pos, next = { ...old, ...partial };
        const m = {};
        for (const k of ['t', 'e', 'z', 'y', 'x']) if (next[k] !== old[k]) m[k] = Math.sign(next[k] - old[k]);
        if (Object.keys(m).length === 0 && this._extraKey !== null) return;
        this.pos = next;
        this._motion = { ...this._motion, ...m };
        this._posChanged = true;
        this._prefetch = ACTIVE_PREFETCH;
        clearTimeout(this._idleTimer);
        this._idleTimer = setTimeout(() => { this._prefetch = IDLE_PREFETCH; this._motion = {}; this.requestFrame(); }, IDLE_MS);
        this.requestFrame();
    }

    // ── per-frame work ────────────────────────────────────────────────────────

    _frame() {
        const t0 = performance.now();
        const { source, pos } = this;
        const wanted = computeWanted(pos, source, { ...this._prefetch, motion: this._motion });
        source.setWanted(wanted);

        // upload newly available wanted planes (cursor planes first), bounded per frame
        wanted.sort((a, b) => a.pri - b.pri);
        let uploads = 0;
        this._more = false;
        for (const it of wanted) {
            const view = this._views.get(it.view);
            if (!view) continue;
            const key = planeKey(it.view, it.t, it.e, it.i);
            if (view.has(key)) { view.touch(key); continue; }
            if (it.pri >= 1 && uploads >= UPLOAD_BUDGET) { this._more = true; continue; }
            const data = source.peek(it.view, it.t, it.e, it.i);
            if (!data) continue;
            view.ensure(key, data);
            uploads++;
        }

        // point each spatial view at its current plane
        for (const name of VIEWS) {
            const view = this._views.get(name);
            if (!view) continue;
            const key = planeKey(name, pos.t, pos.e, pos[VIEW_AXIS[name]]);
            if (view.has(key)) {
                view.touch(key);
                const layer = view.layerOf(key);
                if (view.shownKey !== key || view.shownLayer !== layer) { view.shownKey = key; view.shownLayer = layer; view.dirty = true; }
            }
        }

        // extra view: the T×E series at the cursor voxel (held at the last one until the new one arrives)
        const extra = this._views.get('extra');
        if (extra) {
            const ek = `${pos.z},${pos.y},${pos.x}`;
            if (ek !== this._extraKey) {
                const d = source.peekExtra(pos.z, pos.y, pos.x);
                if (d) { extra.uploadLayer(0, d); this._extraKey = ek; extra.dirty = true; }
                else source.requestExtra(pos.z, pos.y, pos.x);
            }
        }

        const drawn = this._draw();
        const posChanged = this._posChanged;
        this._posChanged = false;
        this.lastStats = { uploadCount: uploads, frameMs: performance.now() - t0, drawn };
        this.onFrame?.(this.lastStats, posChanged);
        if (this._more) this.requestFrame();
    }

    _draw() {
        const dirty = [...this._views.values()].filter((v) => v.dirty);
        if (!dirty.length) return 0;
        const encoder = this.device.createCommandEncoder({ label: 'slicer-frame' });
        for (const view of dirty) {
            view.dirty = false;
            view.writeUniforms(view.shownLayer, this.windowMin, this.windowMax, this.colormap, this.smooth);
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: view.ctx.getCurrentTexture().createView(),
                    clearValue: { r: 0.04, g: 0.04, b: 0.05, a: 1.0 },
                    loadOp: 'clear', storeOp: 'store',
                }],
            });
            pass.setPipeline(view.pipeline);
            pass.setBindGroup(0, view.bindGroup);
            pass.draw(6);
            pass.end();
        }
        this.device.queue.submit([encoder.finish()]);
        return dirty.length;
    }

    /** the plane data currently on screen for a spatial view, or null while it is still on its way */
    shownPlane(name) {
        const view = this._views.get(name);
        const { pos } = this;
        if (!view || view.shownKey !== planeKey(name, pos.t, pos.e, pos[VIEW_AXIS[name]])) return null;
        return this.source.peek(name, pos.t, pos.e, pos[VIEW_AXIS[name]]);
    }

    destroy() {
        cancelAnimationFrame(this._raf);
        clearTimeout(this._idleTimer);
        this.source.destroy?.();
        for (const view of this._views.values()) view.destroy();
        this._views.clear();
    }
}
