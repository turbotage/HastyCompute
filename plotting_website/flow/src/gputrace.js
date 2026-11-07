// WebGPU compute tracer for the flow viewer: streamlines and pathlines of a velocity field that lives on the GPU as one
// rgba16float 3D texture per cardiac frame (trilinear sampling in hardware, frames blended in the shader). One thread per
// seed; RK4 exactly as in common/flowmath.js (which is the reference implementation and the CPU fallback), so the results
// agree up to float16 storage of the velocity. Output is the same polyline structure ({points, speed, time, offsets}).
//
// Streamlines: one dispatch per direction, each thread walks its whole line.
// Pathlines:   one dispatch per time step (bind group = the three frames the step touches), particle state in a buffer;
//              all steps are recorded in a single submit.

const GROUP0 = /* wgsl */`
struct G {
    dims    : vec4f,      // X, Y, Z, 0
    spacing : vec4f,      // m
    a       : vec4f,      // seeds, maxSteps, sign, step
    b       : vec4f,      // minSpeed, useStop, frameW, 0
};
@group(0) @binding(0) var<uniform> g : G;
@group(0) @binding(1) var smp : sampler;
@group(0) @binding(2) var tA : texture_3d<f32>;
@group(0) @binding(3) var tB : texture_3d<f32>;
@group(0) @binding(4) var tC : texture_3d<f32>;
@group(0) @binding(5) var stopT : texture_3d<u32>;

fn inb(p : vec3f) -> bool {
    return all(p >= vec3f(0.0)) && all(p <= g.dims.xyz - vec3f(1.0));
}
fn tex_v(t : texture_3d<f32>, p : vec3f) -> vec3f {
    return textureSampleLevel(t, smp, (p + vec3f(0.5)) / g.dims.xyz, 0.0).xyz;
}
fn inside(p : vec3f) -> bool {
    if g.b.y < 0.5 { return true; }
    let c = vec3i(floor(p + vec3f(0.5)));
    let d = vec3i(g.dims.xyz);
    if any(c < vec3i(0)) || any(c >= d) { return false; }
    return textureLoad(stopT, c, 0).r == 1u;
}
`;

const STREAM_WGSL = GROUP0 + /* wgsl */`
@group(1) @binding(0) var<storage, read> seeds : array<vec4f>;
@group(1) @binding(1) var<storage, read_write> outp : array<vec4f>;
@group(1) @binding(2) var<storage, read_write> counts : array<u32>;

// unit direction in voxel space (xyz) and speed m/s (w); w < 0: stop
fn sdir(p : vec3f) -> vec4f {
    if !inb(p) { return vec4f(0.0, 0.0, 0.0, -1.0); }
    let v = mix(tex_v(tA, p), tex_v(tB, p), g.b.z);
    let s = length(v);
    if s < max(g.b.x, 1e-12) { return vec4f(0.0, 0.0, 0.0, -1.0); }
    let u = v / g.spacing.xyz;
    return vec4f(u / length(u), s);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3u) {
    let i = id.x;
    let n = u32(g.a.x);
    if i >= n { return; }
    let maxs = u32(g.a.y);
    let h = g.a.w * g.a.z;
    var p = seeds[i].xyz;
    var cnt = 0u;
    for (var it = 0u; it < maxs; it++) {
        let k1 = sdir(p);
        if k1.w < 0.0 { break; }
        if !inside(p) { break; }
        outp[i * maxs + cnt] = vec4f(p, k1.w);
        cnt++;
        let k2 = sdir(p + 0.5 * h * k1.xyz);
        if k2.w < 0.0 { break; }
        let k3 = sdir(p + 0.5 * h * k2.xyz);
        if k3.w < 0.0 { break; }
        let k4 = sdir(p + h * k3.xyz);
        if k4.w < 0.0 { break; }
        p = p + h / 6.0 * (k1.xyz + 2.0 * k2.xyz + 2.0 * k3.xyz + k4.xyz);
    }
    counts[i] = cnt;
}
`;

const PATH_WGSL = GROUP0 + /* wgsl */`
struct SP { a : vec4f };      // tau0 (frames since frame f0), h (frames per step), hs (s per step), step index
@group(1) @binding(0) var<uniform> sp : SP;
@group(1) @binding(1) var<storage, read_write> state : array<vec4f>;
@group(1) @binding(2) var<storage, read_write> trace : array<vec4f>;

// velocity in voxels / s (xyz) and speed m/s (w); w < 0: outside
fn pvel(p : vec3f, tau : f32) -> vec4f {
    if !inb(p) { return vec4f(0.0, 0.0, 0.0, -1.0); }
    var v : vec3f;
    if tau < 1.0 { v = mix(tex_v(tA, p), tex_v(tB, p), tau); }
    else { v = mix(tex_v(tB, p), tex_v(tC, p), tau - 1.0); }
    return vec4f(v / g.spacing.xyz, length(v));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3u) {
    let i = id.x;
    let n = u32(g.a.x);
    if i >= n { return; }
    let idx = u32(sp.a.w) * n + i;
    var st = state[i];
    if st.w < 0.5 { trace[idx] = vec4f(0.0, 0.0, 0.0, -1.0); return; }
    let p = st.xyz;
    let tau = sp.a.x;
    let h = sp.a.y;
    let hs = sp.a.z;
    let k1 = pvel(p, tau);
    if k1.w < 0.0 || !inside(p) {
        st.w = 0.0; state[i] = st; trace[idx] = vec4f(0.0, 0.0, 0.0, -1.0); return;
    }
    trace[idx] = vec4f(p, k1.w);
    let k2 = pvel(p + 0.5 * hs * k1.xyz, tau + 0.5 * h);
    if k2.w < 0.0 { st.w = 0.0; state[i] = st; return; }
    let k3 = pvel(p + 0.5 * hs * k2.xyz, tau + 0.5 * h);
    if k3.w < 0.0 { st.w = 0.0; state[i] = st; return; }
    let k4 = pvel(p + hs * k3.xyz, tau + h);
    if k4.w < 0.0 { st.w = 0.0; state[i] = st; return; }
    state[i] = vec4f(p + hs / 6.0 * (k1.xyz + 2.0 * k2.xyz + 2.0 * k3.xyz + k4.xyz), 1.0);
}
`;

// ── float32 → float16 bits ──────────────────────────────────────────────────────────────────────────────────────────
const HAS_F16 = typeof Float16Array !== 'undefined';
const _f = new Float32Array(1), _i = new Int32Array(_f.buffer);
function toHalfBits(x) {
    _f[0] = x;
    const b = _i[0], sign = (b >>> 16) & 0x8000, exp = (b >>> 23) & 0xff, mant = b & 0x7fffff;
    if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
    if (exp === 0) return sign;
    const e = exp - 127 + 15;
    if (e >= 31) return sign | 0x7c00;
    if (e <= 0) return sign;                     // flush denormals (velocities are ≥ 1e-4 m/s or exactly 0)
    return sign | (e << 10) | (mant >>> 13);
}

/** [3, Z, Y, X] float32 → interleaved rgba16float bits [Z, Y, X, 4] (a = 0) */
export function packRGBA16(frame, nv, out = new Uint16Array(4 * nv)) {
    if (HAS_F16) {
        const h = new Float16Array(out.buffer, out.byteOffset, out.length);
        for (let q = 0; q < nv; q++) { h[4 * q] = frame[q]; h[4 * q + 1] = frame[nv + q]; h[4 * q + 2] = frame[2 * nv + q]; }
    } else {
        for (let q = 0; q < nv; q++) { out[4 * q] = toHalfBits(frame[q]); out[4 * q + 1] = toHalfBits(frame[nv + q]); out[4 * q + 2] = toHalfBits(frame[2 * nv + q]); }
    }
    return out;
}

export class GpuTracer {
    /** a tracer on its own device, or null if WebGPU is unavailable */
    static async create() {
        if (!globalThis.navigator?.gpu) return null;
        try {
            const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
            if (!adapter) return null;
            const lim = adapter.limits;
            const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize, maxBufferSize: lim.maxBufferSize, maxTextureDimension3D: lim.maxTextureDimension3D } });
            return new GpuTracer(device);
        } catch (e) { console.warn('GpuTracer unavailable', e); return null; }
    }

    constructor(device) {
        this.device = device;
        this.field = null;
        this.textures = [];
        this.stopTex = this.#makeStop(null, [1, 1, 1]);
        this.useStop = false;
        this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge', addressModeW: 'clamp-to-edge' });
        const tex3 = { sampleType: 'float', viewDimension: '3d' };
        this.bgl0 = device.createBindGroupLayout({ entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: tex3 },
            { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: tex3 },
            { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: tex3 },
            { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'uint', viewDimension: '3d' } },
        ] });
        const st = (type) => ({ visibility: GPUShaderStage.COMPUTE, buffer: { type } });
        this.bglStream = device.createBindGroupLayout({ entries: [{ binding: 0, ...st('read-only-storage') }, { binding: 1, ...st('storage') }, { binding: 2, ...st('storage') }] });
        this.bglPath = device.createBindGroupLayout({ entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } }, { binding: 1, ...st('storage') }, { binding: 2, ...st('storage') }] });
        const pipe = (code, bgl1) => device.createComputePipeline({
            layout: device.createPipelineLayout({ bindGroupLayouts: [this.bgl0, bgl1] }),
            compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
        });
        this.streamPipe = pipe(STREAM_WGSL, this.bglStream);
        this.pathPipe = pipe(PATH_WGSL, this.bglPath);
        this.uniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }

    get maxDim() { return this.device.limits.maxTextureDimension3D; }

    /** can this field be held on the GPU? */
    fits(field) { return Math.max(...field.dims) <= this.maxDim; }

    #makeStop(region, dims) {
        const d = this.device;
        const [x, y, z] = dims;
        const t = d.createTexture({ size: { width: x, height: y, depthOrArrayLayers: z }, dimension: '3d', format: 'r8uint', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
        d.queue.writeTexture({ texture: t }, region ?? new Uint8Array(x * y * z), { bytesPerRow: x, rowsPerImage: y }, { width: x, height: y, depthOrArrayLayers: z });
        return t;
    }

    /** stopping region ([Z, Y, X] Uint8; null = none) */
    setRegion(region) {
        this.stopTex.destroy();
        this.stopTex = this.#makeStop(region, region ? this.field.dims : [1, 1, 1]);
        this.useStop = !!region;
        this._bgCache = new Map();
    }

    /** upload every frame as an rgba16float 3D texture (yields between frames so the page stays responsive) */
    async setField(field, onProgress) {
        if (!this.fits(field)) throw new Error(`volume ${field.dims} exceeds the GPU's 3D texture limit ${this.maxDim}`);
        for (const t of this.textures) t.destroy();
        this.textures = [];
        this.field = field;
        const [x, y, z] = field.dims, buf = new Uint16Array(4 * field.nv);
        for (let f = 0; f < field.nf; f++) {
            const t = this.device.createTexture({ size: { width: x, height: y, depthOrArrayLayers: z }, dimension: '3d', format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
            this.device.queue.writeTexture({ texture: t }, packRGBA16(field.frames[f], field.nv, buf), { bytesPerRow: 8 * x, rowsPerImage: y }, { width: x, height: y, depthOrArrayLayers: z });
            this.textures.push(t);
            onProgress?.(f + 1, field.nf);
            await new Promise((r) => setTimeout(r, 0));
        }
        this._bgCache = new Map();
    }

    #group0(f0, f1, f2) {
        const key = `${f0},${f1},${f2}`;
        if (!this._bgCache) this._bgCache = new Map();
        let bg = this._bgCache.get(key);
        if (!bg) {
            const v = (f) => this.textures[f].createView({ dimension: '3d' });
            bg = this.device.createBindGroup({ layout: this.bgl0, entries: [
                { binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: this.sampler },
                { binding: 2, resource: v(f0) }, { binding: 3, resource: v(f1) }, { binding: 4, resource: v(f2) },
                { binding: 5, resource: this.stopTex.createView({ dimension: '3d' }) },
            ] });
            this._bgCache.set(key, bg);
        }
        return bg;
    }

    #buffer(size, usage, data = null) {
        const b = this.device.createBuffer({ size: Math.max(16, size), usage });
        if (data) this.device.queue.writeBuffer(b, 0, data);
        return b;
    }

    async #read(buf, size) {
        const rb = this.device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(buf, 0, rb, 0, size);
        this.device.queue.submit([enc.finish()]);
        await rb.mapAsync(GPUMapMode.READ);
        const out = rb.getMappedRange().slice(0);
        rb.unmap(); rb.destroy();
        return out;
    }

    #writeG(nSeeds, maxSteps, sign, step, minSpeed, frameW) {
        const f = this.field, g = new Float32Array(16);
        g.set([f.dims[0], f.dims[1], f.dims[2], 0, f.spacing[0], f.spacing[1], f.spacing[2], 0, nSeeds, maxSteps, sign, step, minSpeed, this.useStop ? 1 : 0, frameW, 0]);
        this.device.queue.writeBuffer(this.uniform, 0, g);
    }

    static #seedsData(seeds) {
        const a = new Float32Array(4 * seeds.length);
        seeds.forEach((s, i) => { a[4 * i] = s[0]; a[4 * i + 1] = s[1]; a[4 * i + 2] = s[2]; a[4 * i + 3] = 1; });
        return a;
    }

    /** same contract as flowmath.streamlines: {step, maxSteps, minSpeed}; the stopping region comes from setRegion() */
    async streamlines(frame, seeds, { step = 0.5, maxSteps = 600, minSpeed = 0.02 } = {}) {
        const d = this.device, F = this.field.nf, S = seeds.length;
        if (!S) return { points: new Float32Array(0), speed: new Float32Array(0), time: new Float32Array(0), offsets: new Uint32Array(1) };
        let tc = frame % F; if (tc < 0) tc += F;
        const f0 = Math.floor(tc) % F, f1 = (f0 + 1) % F, wt = tc - Math.floor(tc);
        const seedBuf = this.#buffer(16 * S, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, GpuTracer.#seedsData(seeds));
        const res = [];
        for (const sign of [-1, 1]) {
            // the uniform is shared: record each direction in its own submit
            this.#writeG(S, maxSteps, sign, step, minSpeed, wt);
            const out = this.#buffer(16 * S * maxSteps, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            const cnt = this.#buffer(4 * S, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            const g1 = d.createBindGroup({ layout: this.bglStream, entries: [{ binding: 0, resource: { buffer: seedBuf } }, { binding: 1, resource: { buffer: out } }, { binding: 2, resource: { buffer: cnt } }] });
            const enc = d.createCommandEncoder();
            const pass = enc.beginComputePass();
            pass.setPipeline(this.streamPipe);
            pass.setBindGroup(0, this.#group0(f0, f1, f1));
            pass.setBindGroup(1, g1);
            pass.dispatchWorkgroups(Math.ceil(S / 64));
            pass.end();
            d.queue.submit([enc.finish()]);
            const [pts, counts] = await Promise.all([this.#read(out, 16 * S * maxSteps), this.#read(cnt, 4 * S)]);
            res.push({ pts: new Float32Array(pts), counts: new Uint32Array(counts) });
            out.destroy(); cnt.destroy();
        }
        seedBuf.destroy();
        const [back, fwd] = res, dt = this.field.dt, lines = [];
        for (let i = 0; i < S; i++) {
            const nb = back.counts[i], nf = fwd.counts[i], skip = nb ? 1 : 0, n = nb + Math.max(0, nf - skip);
            if (n < 2) continue;
            const P = new Float32Array(3 * n), Sp = new Float32Array(n), T = new Float32Array(n).fill(frame * dt);
            let o = 0;
            for (let k = nb - 1; k >= 0; k--, o++) { const a = 4 * (i * maxSteps + k); P.set(back.pts.subarray(a, a + 3), 3 * o); Sp[o] = back.pts[a + 3]; }
            for (let k = skip; k < nf; k++, o++) { const a = 4 * (i * maxSteps + k); P.set(fwd.pts.subarray(a, a + 3), 3 * o); Sp[o] = fwd.pts[a + 3]; }
            lines.push({ P, Sp, T });
        }
        return GpuTracer.#pack(lines);
    }

    /** same contract as flowmath.pathlines: {step (substeps per frame), maxSteps} */
    async pathlines(t0, duration, seeds, { step = 4, maxSteps = 2000 } = {}) {
        const d = this.device, F = this.field.nf, dt = this.field.dt, S = seeds.length;
        if (!S) return { points: new Float32Array(0), speed: new Float32Array(0), time: new Float32Array(0), offsets: new Uint32Array(1) };
        const nsteps = Math.min(Math.max(1, Math.ceil(duration * Math.max(step, 1))), maxSteps), h = duration / nsteps, hs = h * dt, steps = nsteps + 1;
        this.#writeG(S, 0, 0, 0, 0, 0);
        const params = new Float32Array(64 * steps), f0s = new Uint32Array(steps);     // 256-byte stride per step
        for (let s = 0; s < steps; s++) {
            let tm = (t0 + s * h) % F; if (tm < 0) tm += F;
            const f0 = Math.floor(tm) % F;
            f0s[s] = f0;
            params.set([tm - Math.floor(tm), h, hs, s], 64 * s);
        }
        const spBuf = this.#buffer(256 * steps, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, params);
        const init = GpuTracer.#seedsData(seeds);
        const state = this.#buffer(16 * S, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, init);
        const trace = this.#buffer(16 * S * steps, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
        const g1 = d.createBindGroup({ layout: this.bglPath, entries: [{ binding: 0, resource: { buffer: spBuf, size: 16 } }, { binding: 1, resource: { buffer: state } }, { binding: 2, resource: { buffer: trace } }] });
        const enc = d.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(this.pathPipe);
        for (let s = 0; s < steps; s++) {
            const f0 = f0s[s];
            pass.setBindGroup(0, this.#group0(f0, (f0 + 1) % F, (f0 + 2) % F));
            pass.setBindGroup(1, g1, [256 * s]);
            pass.dispatchWorkgroups(Math.ceil(S / 64));
        }
        pass.end();
        d.queue.submit([enc.finish()]);
        const tr = new Float32Array(await this.#read(trace, 16 * S * steps));
        spBuf.destroy(); state.destroy(); trace.destroy();
        const lines = [];
        for (let i = 0; i < S; i++) {
            let n = 0;
            while (n < steps && tr[4 * (n * S + i) + 3] >= 0) n++;
            if (n < 2) continue;
            const P = new Float32Array(3 * n), Sp = new Float32Array(n), T = new Float32Array(n);
            for (let k = 0; k < n; k++) { const a = 4 * (k * S + i); P.set(tr.subarray(a, a + 3), 3 * k); Sp[k] = tr[a + 3]; T[k] = (t0 + k * h) * dt; }
            lines.push({ P, Sp, T });
        }
        return GpuTracer.#pack(lines);
    }

    static #pack(lines) {
        let n = 0;
        for (const l of lines) n += l.Sp.length;
        const points = new Float32Array(3 * n), speed = new Float32Array(n), time = new Float32Array(n), offsets = new Uint32Array(lines.length + 1);
        let o = 0;
        lines.forEach((l, i) => { points.set(l.P, 3 * o); speed.set(l.Sp, o); time.set(l.T, o); o += l.Sp.length; offsets[i + 1] = o; });
        return { points, speed, time, offsets };
    }

    destroy() { for (const t of this.textures) t.destroy(); this.stopTex.destroy(); this.uniform.destroy(); this.device.destroy(); }
}
