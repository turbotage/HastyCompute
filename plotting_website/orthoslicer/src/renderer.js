// ─── Constants ────────────────────────────────────────────────────────────────

/** Number of spatial slices per (t,e) slot in the ring-buffer cache. */
export const CACHE_DEPTH = 12;
const HALF_CACHE = CACHE_DEPTH >>> 1;  // = 6

/**
 * Number of (t,e) pair slots cached per spatial view.
 * Switching to a cached (t,e) combination costs 0 uploads (cache hit).
 * Total GPU array layers per spatial view = CACHE_DEPTH × CACHE_TE.
 * At 400×400 pixels: 12×12 = 144 layers ≈ 92 MB per view.
 */
export const CACHE_TE = 12;

// ─── Float32 → Float16 conversion ───────────────────────────────────────────
//
// Textures are stored as r16float (half the VRAM of r32float).
// JavaScript has no native Float16Array, so we bit-pack manually.
// For voxel data in [0, 1] the only branches exercised are the normal path.

function f32ToF16Array(src) {
    const n   = src.length;
    const dst = new Uint16Array(n);
    const buf = new ArrayBuffer(4);
    const fv  = new Float32Array(buf);
    const iv  = new Int32Array(buf);
    for (let i = 0; i < n; i++) {
        fv[0] = src[i];
        const x    = iv[0];
        const sign = (x >>> 16) & 0x8000;
        const exp  = (x >>> 23) & 0xff;
        const mant = x & 0x007fffff;
        let bits;
        if (exp === 0xff) {
            bits = sign | 0x7c00 | (mant ? 0x0200 : 0);   // Inf / NaN
        } else if (exp === 0) {
            bits = sign;                                    // zero / subnormal
        } else {
            const e16 = exp - 127 + 15;
            if      (e16 >= 31) bits = sign | 0x7c00;      // overflow → Inf
            else if (e16 <= 0)  bits = sign;               // underflow → 0
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

function normalizeComprepConfig(cfg = {}) {
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

// ─── CacheTracker — ring-buffer sliding-window cache ─────────────────────────
//
// Maps physical slice indices → GPU texture array layers via a ring buffer.
// Only the newly-needed layers are re-uploaded when the window slides.
//
//   physMin                            physMin + CACHE_DEPTH - 1
//     │◄──────────── CACHE_DEPTH ──────────────────►│
//     layer: ringOffset  ringOffset+1  … (ringOffset+N-1)%N
//
// Scroll up  → evict layer ringOffset (lowest phys), write new high slice to it,
//              ringOffset = (ringOffset+1)%N, physMin++
// Scroll down → evict layer (ringOffset-1+N)%N (highest phys), write new low,
//              ringOffset = (ringOffset-1+N)%N, physMin--

class CacheTracker {
    /**
     * @param {number} physCenter  physical slice to center the cache on initially
     * @param {number} total       total number of slices available on this axis
     */
    constructor(physCenter, total) {
        this.total      = Math.max(1, total);
        this.ringOffset = 0;
        this.physMin    = this._clamp(physCenter - HALF_CACHE);
    }

    _clamp(v) {
        return Math.max(0, Math.min(v, Math.max(0, this.total - CACHE_DEPTH)));
    }

    get physMax() { return this.physMin + CACHE_DEPTH - 1; }

    has(p) { return p >= this.physMin && p <= this.physMax; }

    /** GPU array-layer index for a cached physical slice (only valid when has(p)). */
    layerFor(p) {
        return (this.ringOffset + p - this.physMin) % CACHE_DEPTH;
    }

    /**
     * Completely re-center on physCenter (used when image data changes, e.g. t/e).
     * Returns all { layer, phys } pairs that must be uploaded.
     */
    refill(physCenter) {
        this.physMin    = this._clamp(physCenter - HALF_CACHE);
        this.ringOffset = 0;
        return Array.from({ length: CACHE_DEPTH }, (_, i) => ({
            layer: i,
            phys:  Math.min(this.physMin + i, this.total - 1),
        }));
    }

    /**
     * Ensure physSlice is in cache.  Returns only the newly-needed
     * { layer, phys } upload pairs (empty array on cache hit).
     * Automatically falls back to refill() for jumps ≥ CACHE_DEPTH.
     */
    stream(physSlice) {
        if (this.has(physSlice)) return [];

        const delta = physSlice - (this.physMin + HALF_CACHE);
        if (Math.abs(delta) >= CACHE_DEPTH) return this.refill(physSlice);

        const uploads = [];

        if (physSlice > this.physMax) {
            // Slide window upward.
            while (!this.has(physSlice)) {
                const evictLayer = this.ringOffset;
                const newPhys    = Math.min(this.physMax + 1, this.total - 1);
                uploads.push({ layer: evictLayer, phys: newPhys });
                this.physMin++;
                this.ringOffset = (this.ringOffset + 1) % CACHE_DEPTH;
            }
        } else {
            // Slide window downward.
            while (!this.has(physSlice)) {
                const newLayer = (this.ringOffset - 1 + CACHE_DEPTH) % CACHE_DEPTH;
                const newPhys  = Math.max(this.physMin - 1, 0);
                uploads.push({ layer: newLayer, phys: newPhys });
                this.physMin--;
                this.ringOffset = newLayer;
            }
        }

        return uploads;
    }
}

// ─── TERingBuffer ─────────────────────────────────────────────────────────────
//
// Maintains a FIFO ring of CACHE_TE slots, each slot holds one (t, e) pair.
// `getOrAlloc(t, e)` returns { slot, isNew }:
//   isNew = false → the (t,e) combination is already cached in `slot`; the
//                   spatial CacheTracker for that slot can be queried cheaply.
//   isNew = true  → slot was just (re-)allocated; caller must do a full spatial
//                   cache fill for that slot.

class TERingBuffer {
    /** @param {number} maxSlots */
    constructor(maxSlots) {
        this.maxSlots   = maxSlots;
        /** @type {Map<string, number>}  key `${t},${e}` → slot index */
        this._map       = new Map();
        /** @type {Array<{t:number,e:number}|null>}  slot → (t,e) or null */
        this._slots     = new Array(maxSlots).fill(null);
        this._nextEvict = 0;
    }

    /**
     * Return an existing slot for (t,e), or allocate a new one (FIFO eviction).
     * @returns {{ slot: number, isNew: boolean }}
     */
    getOrAlloc(t, e) {
        const key = `${t},${e}`;
        if (this._map.has(key)) return { slot: this._map.get(key), isNew: false };

        // Evict the oldest slot.
        const slot = this._nextEvict;
        const old  = this._slots[slot];
        if (old !== null) this._map.delete(`${old.t},${old.e}`);
        this._slots[slot] = { t, e };
        this._map.set(key, slot);
        this._nextEvict = (this._nextEvict + 1) % this.maxSlots;
        return { slot, isNew: true };
    }
}
//
// Renders a single layer from a texture_2d_array onto a full-screen quad.
// Uniforms (32 bytes):
//   u32  slice_layer  – GPU array layer to sample
//   u32  tex_width    – texture logical width  (columns)
//   u32  tex_height   – texture logical height (rows)
//   u32  _pad0
//   f32  window_min
//   f32  window_max
//   f32  _pad1, _pad2
//
// textureLoad is used (no sampler) so r16float works without float32-filterable.
// Colormap IDs:  0=Gray  1=Viridis  2=Plasma  3=Hot  4=Cool

const SLICE_SHADER = /* wgsl */`

struct Uniforms {
    slice_layer : u32,
    tex_width   : u32,
    tex_height  : u32,
    colormap    : u32,   // 0=Gray 1=Viridis 2=Plasma 3=Hot 4=Cool
    window_min  : f32,
    window_max  : f32,
    _pad1       : f32,
    _pad2       : f32,
};

@group(0) @binding(0) var<uniform> u   : Uniforms;
@group(0) @binding(1) var          tex : texture_2d_array<f32>;

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

// ── Colormap functions ────────────────────────────────────────────────────────

fn cm_gray(t: f32) -> vec3f {
    return vec3f(t, t, t);
}

// Polynomial approximation of Matplotlib viridis (Björn Ottosson / matplotlab)
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

// Polynomial approximation of Matplotlib plasma
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

// Black → red → yellow → white
fn cm_hot(t: f32) -> vec3f {
    return clamp(vec3f(t * 3.0, t * 3.0 - 1.0, t * 3.0 - 2.0), vec3f(0.0), vec3f(1.0));
}

// Cyan → magenta
fn cm_cool(t: f32) -> vec3f {
    return vec3f(t, 1.0 - t, 1.0);
}

fn apply_colormap(v: f32, cm: u32) -> vec3f {
    if cm == 1u { return cm_viridis(v); }
    if cm == 2u { return cm_plasma(v);  }
    if cm == 3u { return cm_hot(v);     }
    if cm == 4u { return cm_cool(v);    }
    return cm_gray(v);   // 0 = default
}

// ── Fragment shader ───────────────────────────────────────────────────────────

@fragment
fn fs_main(in : VOut) -> @location(0) vec4f {
    let px  = min(u32(in.uv.x * f32(u.tex_width)),  u.tex_width  - 1u);
    let py  = min(u32(in.uv.y * f32(u.tex_height)), u.tex_height - 1u);
    let raw = textureLoad(tex, vec2i(i32(px), i32(py)), i32(u.slice_layer), 0).r;
    let v   = clamp((raw - u.window_min) / (u.window_max - u.window_min), 0.0, 1.0);
    return vec4f(apply_colormap(v, u.colormap), 1.0);
}
`;

// ─── COMPREP_SHADER ───────────────────────────────────────────────────────────
//
// Renders a u16-quantised comprep tensor (r16uint → texture_2d_array<u32>).
// Uniforms (80 bytes = 20 × 4):
//   u32  slice_layer, tex_width, tex_height, colormap
//   f32  window_min, window_max, a, b, clampa, clampb, t0, t1
//   u32  left_mode, right_mode   (0=linear 1=gamma 2=log)
//   f32  left_gamma, right_gamma, left_logc, right_logc, _pad0, _pad1
//
// u16 value 0..65535 → y=[0,1] → piecewise inverse tone-map → physical x
// → window/level → colormap

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
    _pad0        : f32,
    _pad1        : f32,
};

@group(0) @binding(0) var<uniform> u   : Uniforms;
@group(0) @binding(1) var          tex : texture_2d_array<u32>;

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

fn cm_gray_c(t: f32) -> vec3f { return vec3f(t, t, t); }
fn cm_viridis_c(t: f32) -> vec3f {
    let c0 = vec3f( 0.27773, 0.00541, 0.33410);
    let c1 = vec3f( 0.10509, 1.40461, 1.38459);
    let c2 = vec3f(-0.33086, 0.21485, 0.09510);
    let c3 = vec3f(-4.63423,-5.79910,-19.33244);
    let c4 = vec3f( 6.22827,14.17993, 56.69055);
    let c5 = vec3f( 4.77638,-13.74515,-65.35303);
    let c6 = vec3f(-5.43546, 4.64585, 26.31244);
    return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3f(0.0), vec3f(1.0));
}
fn cm_plasma_c(t: f32) -> vec3f {
    let c0 = vec3f( 0.05873, 0.02334, 0.54334);
    let c1 = vec3f( 2.17651, 0.23838, 0.75396);
    let c2 = vec3f(-2.68946,-7.45585, 3.11080);
    let c3 = vec3f( 6.13035,42.34619,-28.51885);
    let c4 = vec3f(-11.10744,-82.66631, 60.13985);
    let c5 = vec3f(10.02307, 71.41362,-54.07219);
    let c6 = vec3f(-3.65871,-22.93153, 18.19191);
    return clamp(c0+t*(c1+t*(c2+t*(c3+t*(c4+t*(c5+t*c6))))), vec3f(0.0), vec3f(1.0));
}
fn cm_hot_c(t: f32) -> vec3f {
    return clamp(vec3f(t*3.0, t*3.0-1.0, t*3.0-2.0), vec3f(0.0), vec3f(1.0));
}
fn cm_cool_c(t: f32) -> vec3f { return vec3f(t, 1.0-t, 1.0); }
fn apply_colormap_c(v: f32, cm: u32) -> vec3f {
    if cm == 1u { return cm_viridis_c(v); }
    if cm == 2u { return cm_plasma_c(v);  }
    if cm == 3u { return cm_hot_c(v);     }
    if cm == 4u { return cm_cool_c(v);    }
    return cm_gray_c(v);
}

// Inverse of shaping applied per region:
//   linear: identity
//   gamma:  forward=pow(u,1/γ)  →  inverse=pow(u,γ)
//   log:    forward=log1p(u*c)/log(1+c)  →  inverse=(exp(u*log(1+c))-1)/c
fn inv_shape(uv: f32, mode: u32, gamma: f32, logc: f32) -> f32 {
    if mode == 1u { return pow(max(uv, 0.0), gamma); }
    if mode == 2u {
        let c = max(logc, 1e-6);
        return (exp(uv * log(1.0 + c)) - 1.0) / c;
    }
    return uv;
}

@fragment
fn fs_main(in : VOut) -> @location(0) vec4f {
    let px  = min(u32(in.uv.x * f32(u.tex_width)),  u.tex_width  - 1u);
    let py  = min(u32(in.uv.y * f32(u.tex_height)), u.tex_height - 1u);
    let raw = textureLoad(tex, vec2i(i32(px), i32(py)), i32(u.slice_layer), 0).r;
    let y   = f32(raw) / 65535.0;

    var x : f32;
    if y < u.t0 && u.t0 > 0.0 {
        // LEFT region  y ∈ [0, t0]  →  x ∈ [clampa, a]
        let uv = inv_shape(y / u.t0, u.left_mode, u.left_gamma, u.left_logc);
        x = uv * (u.a - u.clampa) + u.clampa;
    } else if y > u.t1 && u.t1 < 1.0 {
        // RIGHT region  y ∈ [t1, 1]  →  x ∈ [b, clampb]
        let uv = inv_shape((y - u.t1) / (1.0 - u.t1), u.right_mode, u.right_gamma, u.right_logc);
        x = uv * (u.clampb - u.b) + u.b;
    } else {
        // MIDDLE region  y ∈ [t0, t1]  →  x ∈ [a, b]
        let span = u.t1 - u.t0;
        let uv   = select((y - u.t0) / span, 0.0, span < 1e-6);
        x = uv * (u.b - u.a) + u.a;
    }

    let v = clamp((x - u.window_min) / (u.window_max - u.window_min), 0.0, 1.0);
    return vec4f(apply_colormap_c(v, u.colormap), 1.0);
}
`;

// ─── ViewState ────────────────────────────────────────────────────────────────

/**
 * Per-view GPU state: WebGPU context, render pipeline, uniform buffer,
 * bind group, and cache texture (r32float 2d-array).
 *
 * numLayers = CACHE_DEPTH for spatial views; 1 for the extra (T×E) view.
 */
class ViewState {
    /**
     * @param {GPUDevice}         device
     * @param {HTMLCanvasElement} canvas
     * @param {string}            label
     * @param {number}            texWidth   texture width  (columns)
     * @param {number}            texHeight  texture height (rows)
     * @param {number}            numLayers  array depth; default = CACHE_DEPTH
     * @param {'float'|'comprep'} mode       'float' = r16float, 'comprep' = r16uint + inverse shader
     * @param {object|null}       comprepParams  normalised comprep config (required when mode='comprep')
     */
    constructor(device, canvas, label, texWidth, texHeight, numLayers = CACHE_DEPTH, mode = 'float', comprepParams = null) {
        this.device         = device;
        this.label          = label;
        this.texWidth       = texWidth;
        this.texHeight      = texHeight;
        this.numLayers      = numLayers;
        this.mode           = mode;
        this.comprepParams  = comprepParams;

        /** GPU layer index to display; updated by Renderer before each draw. */
        this.currentLayer = 0;
        /**
         * Per-(t,e)-slot spatial ring buffers.
         * Index = slot from teRing; each entry is a CacheTracker or null.
         * Only used for spatial views (numLayers > 1).
         */
        this.teRing        = new TERingBuffer(CACHE_TE);
        this.spatialTrackers = new Array(CACHE_TE).fill(null);

        // ── WebGPU canvas context ────────────────────────────────────────────
        this.ctx    = canvas.getContext('webgpu');
        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.ctx.configure({ device, format: this.format, alphaMode: 'opaque' });

        // ── Cache texture ────────────────────────────────────────────────────
        this.cacheTex = device.createTexture({
            label: `cache-${label}`,
            size: { width: texWidth, height: texHeight, depthOrArrayLayers: numLayers },
            format: mode === 'comprep' ? 'r16uint' : 'r16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });

        // ── Shader module ────────────────────────────────────────────────────
        const shaderModule = device.createShaderModule({
            label: `shader-${label}`,
            code: mode === 'comprep' ? COMPREP_SHADER : SLICE_SHADER,
        });

        // ── Bind group layout ────────────────────────────────────────────────
        this.bgl = device.createBindGroupLayout({
            label: `bgl-${label}`,
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                    buffer: { type: 'uniform' },
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.FRAGMENT,
                    texture: {
                        sampleType:    mode === 'comprep' ? 'uint' : 'unfilterable-float',
                        viewDimension: '2d-array',
                        multisampled:  false,
                    },
                },
            ],
        });

        // ── Render pipeline ───────────────────────────────────────────────────
        this.pipeline = device.createRenderPipeline({
            label:    `pipeline-${label}`,
            layout:   device.createPipelineLayout({ bindGroupLayouts: [this.bgl] }),
            vertex:   { module: shaderModule, entryPoint: 'vs_main' },
            fragment: {
                module:     shaderModule,
                entryPoint: 'fs_main',
                targets:    [{ format: this.format }],
            },
            primitive: { topology: 'triangle-list' },
        });

        // ── Uniform buffer ────────────────────────────────────────────────────
        // float mode: 32 bytes (8 × 4).  comprep mode: 80 bytes (20 × 4).
        this.uniformBuf = device.createBuffer({
            label: `uniform-${label}`,
            size:  mode === 'comprep' ? 80 : 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // ── Bind group ────────────────────────────────────────────────────────
        this.bindGroup = device.createBindGroup({
            label:   `bg-${label}`,
            layout:  this.bgl,
            entries: [
                { binding: 0, resource: { buffer: this.uniformBuf } },
                {
                    binding: 1,
                    resource: this.cacheTex.createView({
                        dimension:       '2d-array',
                        baseArrayLayer:  0,
                        arrayLayerCount: numLayers,
                    }),
                },
            ],
        });
    }

    /**
     * Upload one texture array layer.
     * Float32Array is converted to float16; Uint16Array is uploaded as-is
     * (used for both r16float-with-f16-bits and r16uint comprep data).
     * @param {number}                     layer  0 .. numLayers-1
     * @param {Float32Array|Uint16Array}   data   length = texWidth * texHeight
     */
    uploadLayer(layer, data) {
        const bytes = data instanceof Uint16Array ? data : f32ToF16Array(data);
        this.device.queue.writeTexture(
            { texture: this.cacheTex, origin: { x: 0, y: 0, z: layer } },
            bytes,
            { bytesPerRow: this.texWidth * 2, rowsPerImage: this.texHeight },
            { width: this.texWidth, height: this.texHeight, depthOrArrayLayers: 1 },
        );
    }

    /**
     * Write uniform values for this draw call.
     * @param {number} layer      GPU layer to display (0 .. numLayers-1)
     * @param {number} windowMin
     * @param {number} windowMax
     * @param {number} colormap   0=Gray 1=Viridis 2=Plasma 3=Hot 4=Cool
     */
    writeUniforms(layer, windowMin, windowMax, colormap) {
        if (this.mode === 'comprep') {
            const p   = this.comprepParams;
            const buf = new ArrayBuffer(80);
            const u32 = new Uint32Array(buf);
            const f32 = new Float32Array(buf);
            u32[0]  = layer;
            u32[1]  = this.texWidth;
            u32[2]  = this.texHeight;
            u32[3]  = colormap | 0;
            f32[4]  = windowMin;
            f32[5]  = windowMax;
            f32[6]  = p.a;
            f32[7]  = p.b;
            f32[8]  = p.clampa;
            f32[9]  = p.clampb;
            f32[10] = p.t0;
            f32[11] = p.t1;
            u32[12] = p.left_mode;
            u32[13] = p.right_mode;
            f32[14] = p.left_gamma;
            f32[15] = p.right_gamma;
            f32[16] = p.left_logc;
            f32[17] = p.right_logc;
            // f32[18] = 0; f32[19] = 0; (already zeroed)
            this.device.queue.writeBuffer(this.uniformBuf, 0, buf);
        } else {
            const buf = new ArrayBuffer(32);
            const u32 = new Uint32Array(buf);
            const f32 = new Float32Array(buf);
            u32[0] = layer;
            u32[1] = this.texWidth;
            u32[2] = this.texHeight;
            u32[3] = colormap | 0;
            f32[4] = windowMin;
            f32[5] = windowMax;
            this.device.queue.writeBuffer(this.uniformBuf, 0, buf);
        }
    }

    destroy() {
        this.cacheTex.destroy();
        this.uniformBuf.destroy();
    }
}

// ─── Renderer ────────────────────────────────────────────────────────────────

/**
 * Manages four WebGPU views: axial, sagittal, coronal, and extra (T×E).
 *
 * Spatial views (axial/sagittal/coronal) each hold a CACHE_DEPTH-layer ring-
 * buffer texture.  Only the layers that slide into view as the cursor moves
 * are re-uploaded; unchanged layers are served from GPU memory at zero cost.
 *
 * When t or e changes the spatial caches are fully invalidated (the underlying
 * image data changes).  The extra view texture is re-uploaded only when the
 * spatial position (z,y,x) changes; t/e changes only move its crosshair.
 *
 * API:
 *   renderer.attachCanvas('axial',    canvas);
 *   renderer.attachCanvas('sagittal', canvas);
 *   renderer.attachCanvas('coronal',  canvas);
 *   renderer.attachCanvas('extra',    canvas);
 *   renderer.setPosition({ t, e, z, y, x });  // partial updates accepted
 *   renderer.render();                         // re-render without data change
 */
export class Renderer {
    constructor(device, volume) {
        this.device  = device;
        this.volume  = volume;
        this.pos = {
            t: Math.floor(volume.T / 2),
            e: Math.floor(volume.E / 2),
            z: Math.floor(volume.Z / 2),
            y: Math.floor(volume.Y / 2),
            x: Math.floor(volume.X / 2),
        };
        /** @type {Map<string, ViewState>} */
        this._views = new Map();

        this.windowMin = 0.0;
        this.windowMax = 1.0;
        /** 0=Gray 1=Viridis 2=Plasma 3=Hot 4=Cool */
        this.colormap  = 0;

        this._extraInitialized = false;
        /** @type {{ uploadCount: number, fillMs: number }} */
        this.lastStats = { uploadCount: 0, fillMs: 0 };

        // Determine rendering mode from volume dtype.
        // 'i16' = compress_ui16_config output → inverse tone-map shader + r16uint texture.
        this._mode          = volume.dtype === 'i16' ? 'comprep' : 'float';
        this._comprepParams = this._mode === 'comprep'
            ? normalizeComprepConfig(volume.comprepConfig ?? {})
            : null;
    }

    // ── Canvas attachment ─────────────────────────────────────────────────────

    /**
     * Attach a canvas to a named view and create all GPU objects for it.
     * Must be called after canvas.width / canvas.height have been set.
     *
     * @param {'axial'|'sagittal'|'coronal'|'extra'} key
     * @param {HTMLCanvasElement} canvas
     */
    attachCanvas(key, canvas) {
        const { volume } = this;
        // Spatial views: CACHE_DEPTH × CACHE_TE layers.
        // Extra view:    1 layer, width=E (echo), height=T (time-point).
        const dimMap = {
            axial:    [volume.X, volume.Y, CACHE_DEPTH * CACHE_TE],
            sagittal: [volume.Y, volume.Z, CACHE_DEPTH * CACHE_TE],
            coronal:  [volume.X, volume.Z, CACHE_DEPTH * CACHE_TE],
            extra:    [volume.E, volume.T, 1],
        };
        if (!dimMap[key]) throw new Error(`Unknown view key: ${key}`);
        const [tw, th, nl] = dimMap[key];
        this._views.set(key, new ViewState(this.device, canvas, key, tw, th, nl, this._mode, this._comprepParams));
    }

    // ── Smart cache management ────────────────────────────────────────────────

    /**
     * Update one spatial view for the given (t, e, physSlice).
     *
     * • If (t,e) is already cached in a slot: only stream the physSlice into
     *   that slot's CacheTracker (0 uploads on spatial cache-hit).
     * • If (t,e) is new: allocate a slot (evicting the oldest), then do a full
     *   CACHE_DEPTH-slice spatial fill centred on physSlice.
     *
     * Returns the number of slice uploads performed.
     *
     * @param {string}                      key       view key
     * @param {number}                      t
     * @param {number}                      e
     * @param {number}                      physSlice physical axis index
     * @param {number}                      total     total slices on this axis
     * @param {function(number):Float32Array} getSlice (phys) → Float32Array
     * @returns {number} upload count
     */
    _updateSpatialViewTE(key, t, e, physSlice, total, getSlice) {
        const view = this._views.get(key);
        if (!view) return 0;

        const { slot, isNew } = view.teRing.getOrAlloc(t, e);
        let uploads;

        if (isNew || !view.spatialTrackers[slot]) {
            // New (t,e) slot: create tracker and do full spatial fill.
            const tracker = new CacheTracker(physSlice, total);
            view.spatialTrackers[slot] = tracker;
            uploads = tracker.refill(physSlice);
        } else {
            // Existing (t,e) slot: only stream what's missing.
            uploads = view.spatialTrackers[slot].stream(physSlice);
        }

        const base = slot * CACHE_DEPTH;
        for (const { layer, phys } of uploads) {
            view.uploadLayer(base + layer, getSlice(phys));
        }

        view.currentLayer = base + view.spatialTrackers[slot].layerFor(physSlice);
        return uploads.length;
    }

    /** Re-upload the extra (T×E) texture for the current spatial voxel. */
    _updateExtraView() {
        const extra = this._views.get('extra');
        if (!extra) return;
        extra.uploadLayer(0, this.volume.getExtraSlice(this.pos.z, this.pos.y, this.pos.x));
        extra.currentLayer = 0;
    }

    // ── Position update ───────────────────────────────────────────────────────

    /**
     * Move to a (partial) new voxel position, stream in new slices, render.
     * The TE-ring handles t/e changes transparently — no full cache invalidation.
     *
     * Exposes `this.lastStats` after each call:
     *   { uploadCount: number, fillMs: number }
     *
     * @param {Partial<{t,e,z,y,x}>} partial
     */
    setPosition(partial) {
        const t0 = performance.now();
        this.pos = { ...this.pos, ...partial };
        const { pos, volume } = this;

        let uploadCount = 0;
        uploadCount += this._updateSpatialViewTE(
            'axial',    pos.t, pos.e, pos.z, volume.Z,
            (p) => volume.getAxialSlice(pos.t, pos.e, p));
        uploadCount += this._updateSpatialViewTE(
            'sagittal', pos.t, pos.e, pos.x, volume.X,
            (p) => volume.getSagittalSlice(pos.t, pos.e, p));
        uploadCount += this._updateSpatialViewTE(
            'coronal',  pos.t, pos.e, pos.y, volume.Y,
            (p) => volume.getCoronalSlice(pos.t, pos.e, p));

        // Extra view: re-upload only when spatial position changes.
        if ((partial.z !== undefined) || (partial.y !== undefined) ||
            (partial.x !== undefined) || !this._extraInitialized) {
            this._updateExtraView();
            this._extraInitialized = true;
        }

        this.lastStats = { uploadCount, fillMs: performance.now() - t0 };
        this.render();
    }

    // ── Rendering ─────────────────────────────────────────────────────────────

    /** Record and submit one render pass per attached view. */
    render() {
        const { device } = this;
        const encoder = device.createCommandEncoder({ label: 'slicer-frame' });

        for (const [, view] of this._views) {
            view.writeUniforms(view.currentLayer, this.windowMin, this.windowMax, this.colormap);

            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view:       view.ctx.getCurrentTexture().createView(),
                    clearValue: { r: 0.04, g: 0.04, b: 0.05, a: 1.0 },
                    loadOp:     'clear',
                    storeOp:    'store',
                }],
            });
            pass.setPipeline(view.pipeline);
            pass.setBindGroup(0, view.bindGroup);
            pass.draw(6);
            pass.end();
        }

        device.queue.submit([encoder.finish()]);
    }

    /**
     * Invalidate all TE slot caches so the next setPosition() call re-uploads
     * all slices from the volume.  Use this when the volume's underlying data
     * has changed (e.g. RemoteVolume received new fetched slices).
     */
    invalidateCache() {
        for (const [, view] of this._views) {
            view.teRing._map.clear();
            view.teRing._slots.fill(null);
            view.teRing._nextEvict = 0;
            view.spatialTrackers.fill(null);
        }
        this._extraInitialized = false;
    }

    destroy() {
        for (const view of this._views.values()) view.destroy();
        this._views.clear();
    }
}
