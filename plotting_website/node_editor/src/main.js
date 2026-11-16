/**
 * main.js — HastyEdit node editor entry point.
 *
 * litegraph.js is loaded as a plain <script> tag (sets window.LiteGraph)
 * before this ES module is evaluated.
 *
 * Node categories and their backend mapping:
 *   IO     — LoadTensor, SaveTensor
 *   Create — RandTensor, ZerosTensor, OnesTensor  (cmd 6, 7, 8)
 *   Math   — Add, Sub, Mult, Div, Neg, Abs        (cmd 0-5)
 *            Scalar  (no backend, plain number)
 *   FFT    — FFT, IFFT, NUFFT                     (stubs)
 *   MRI    — CoilSense, SensitivityMaps, Toeplitz (stubs)
 *   Viz    — OrthoSlicer
 */

import { runGraph }                               from './node_runner.js';
import { RunController, call, compilePlan }      from './server_run.js';
import { client, fetchMeta, run, isId, DTYPE, SHORT } from './hasty.js';
import { readGeometry, writeGeometry, geometryMetadata } from './nifti.js';
import { HTensor }                                from '../../common/value_format.js';
import { WRITERS, meshValue, parseMesh }         from '../../common/meshio.js';

const LG = window.LiteGraph;

// -- Theme --------------------------------------------------------------------
LG.NODE_DEFAULT_COLOR        = "#141820";
LG.NODE_DEFAULT_BGCOLOR      = "#0e1218";
LG.NODE_DEFAULT_BOXCOLOR     = "#3a6090";
LG.NODE_SELECTED_TITLE_COLOR = "#7ab4d8";
LG.DEFAULT_SHADOW_COLOR      = "transparent";
LG.CONNECTING_LINK_COLOR     = "#5a9abc";
LG.LINK_COLOR                = "#3a6090";
LG.EVENT_LINK_COLOR          = "#60b070";
LG.RENDER_CONNECTIONS_BORDER = false;

// -- Shared helpers -----------------------------------------------------------

const STATUS_H = 22;   // execution status bar, bottom of the node body
const INFO_H   = 14;   // optional info line (node._info) just above the status bar, for nodes with static infoLine = true

/** cut `text` with an ellipsis so it is at most `maxW` px wide in the current ctx font */
function ellipsize(ctx, text, maxW) {
    if (ctx.measureText(text).width <= maxW) return text;
    let lo = 0, hi = text.length;
    while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (ctx.measureText(text.slice(0, mid) + '\u2026').width <= maxW) lo = mid; else hi = mid - 1;
    }
    return text.slice(0, lo) + '\u2026';
}

/**
 * Draw the node's info line (node._info) and execution status bar in the bottom of the node body.
 * Call from onDrawForeground (coordinate system: node body, top-left origin). Text is clipped to the node width.
 */
function drawStatusBar(ctx, node) {
    const w = node.size[0];
    const h = node.size[1];
    ctx.font = '10px monospace';
    if (node.constructor.infoLine && node._info) {
        ctx.fillStyle    = '#88aacc';
        ctx.textBaseline = 'middle';
        ctx.fillText(ellipsize(ctx, node._info, w - 12), 6, h - STATUS_H - INFO_H / 2);
    }
    if (!node._execStatus) return;
    const y = h - STATUS_H;

    const BG = { running: '#2a1800', done: '#001810', error: '#1a0006', queued: '#14181e', waiting: '#2a2200', skipped: '#16161a' };
    const FG = { running: '#cc8800', done:  '#44cc88', error:  '#ee4444', queued: '#7a8aa0', waiting: '#e0b020', skipped: '#707070' };
    ctx.fillStyle = BG[node._execStatus] ?? '#111';
    ctx.fillRect(3, y + 2, w - 6, STATUS_H - 4);
    if (node._execStatus === 'running' && node._execProgress > 0) {
        ctx.fillStyle = '#4a3000';
        ctx.fillRect(3, y + 2, (w - 6) * Math.min(1, node._execProgress), STATUS_H - 4);
    }

    ctx.fillStyle    = FG[node._execStatus] ?? '#888';
    ctx.textBaseline = 'middle';

    let text = '';
    if (node._execStatus === 'running') {
        text = node._execNote ? `${Math.round((node._execProgress ?? 0) * 100)}% ${node._execNote}` : 'running...';
    } else if (node._execStatus === 'queued') {
        text = 'queued';
    } else if (node._execStatus === 'waiting') {
        text = 'waiting for a browser';
    } else if (node._execStatus === 'skipped') {
        text = 'skipped';
    } else if (node._execStatus === 'done') {
        const uuid = node._execResult?.[0];
        if (uuid) {
            const hex  = String(uuid).slice(0, 8);
            const meta = node._execMeta;
            text = meta
                ? hex + '... ' + meta.dtype + '[' + meta.shape.join('x') + ']'
                : hex + '...';
        } else {
            text = 'done';
        }
    } else if (node._execStatus === 'error') {
        text = 'ERR: ' + (node._execError ?? 'error');
    }

    ctx.fillText(ellipsize(ctx, text, w - 14), 6, y + 2 + (STATUS_H - 4) / 2);
}

/**
 * Base of all editor nodes.
 *  - the title comes from the class (`static title`): litegraph passes it to the constructor, which the subclasses'
 *    argument-less super() calls used to drop (every node was titled "Unnamed")
 *  - size is never smaller than slots + widgets (+ the status bar for executable nodes, + the info line when
 *    static infoLine = true): fixed sizes in the constructors let widgets spill over the status bar and out of the node
 */
class HNode extends LG.LGraphNode {
    constructor() { super(new.target.title); }
    fit() {
        const m = this.computeSize();
        const extra = (typeof this.executeAsync === 'function' ? STATUS_H : 0) + (this.constructor.infoLine ? INFO_H : 0);
        this.size = [Math.max(this.size[0], m[0]), Math.max(this.size[1], m[1] + extra)];
    }
    onAdded() { this.fit(); this.tint(); }
    configure(info) { super.configure(info); this.fit(); this.tint(); }
    /** title colour shows an explicit placement: green = browser, blue = server */
    tint() {
        const p = this.properties?.placement;
        this.color = p === 'client' ? '#1c3a22' : p === 'server' ? '#1c2c4a' : undefined;
    }
    getExtraMenuOptions() {
        const canRun = (typeof this.plan === 'function') + (typeof this.executeAsync === 'function');
        if (canRun < 2) return [];   // only nodes that can run in both places have a choice
        return ['auto', 'server', 'client'].map((p) => ({
            content: `Run on: ${p}${(this.properties.placement ?? 'auto') === p ? '  \u2713' : ''}`,
            callback: () => { this.properties.placement = p; this.tint(); this.setDirtyCanvas(true, true); },
        }));
    }
}

/** tensor-creation options for the server's zeros / ones / rand: { shape, dtype } (device: CPU for now) */
function makeTensorOptions(dtype, device, shapeStr) {
    const shape = shapeStr.trim().replace(/^\(|\)$/g, '').split(/[\s,]+/).filter(Boolean).map(Number);
    return { shape, dtype: DTYPE[dtype] ?? dtype, device };
}

// -- NIfTI helpers ------------------------------------------------------------

/** ATen scalar_type index for each hasty dtype string. */
const DTYPE_SCALAR_TYPE = { u8:0, i8:1, i16:2, i32:3, i64:4, f16:5, f32:6, f64:7 };
/** Element byte width for each hasty dtype string. */
const DTYPE_BYTES        = { u8:1, i8:1, i16:2, i32:4, i64:8, f16:2, f32:4, f64:8 };

/**
 * Decompress a gzip ArrayBuffer using the browser's DecompressionStream API.
 * Returns a new ArrayBuffer with the decompressed bytes.
 */
async function decompressGzip(buf) {
    const ds     = new DecompressionStream('gzip');
    const writer = ds.writable.getWriter();
    const reader = ds.readable.getReader();
    writer.write(new Uint8Array(buf));
    writer.close();
    const chunks = [];
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
    }
    const total = chunks.reduce((s, c) => s + c.length, 0);
    const out   = new Uint8Array(total);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    return out.buffer;
}

/**
 * Parse a NIfTI-1 file (.nii or .nii.gz) into tensor metadata + raw bytes.
 *
 * Returns { shape, dtype, data } where:
 *   shape — C-order (slowest-first) array, i.e. NIfTI dims reversed
 *   dtype — hasty dtype string (f32, i16, …)
 *   data  — Uint8Array of raw element bytes (no reordering needed)
 *
 * NIfTI dim[1]=X varies fastest (Fortran-style); reversing the shape vector
 * gives C-order so the raw bytes match a contiguous PyTorch tensor.
 *
 * @param {File} file
 * @returns {Promise<{shape: number[], dtype: string, data: Uint8Array, geometry: {spacing_mm: number[], affine: number[][], source: string}}>}
 */
async function parseNifti(file) {
    let buf = await file.arrayBuffer();
    if (file.name.toLowerCase().endsWith('.gz')) {
        buf = await decompressGzip(buf);
    }
    const view = new DataView(buf);
    // Detect endianness: sizeof_hdr must be 348
    const le = view.getInt32(0, true) === 348;
    if (!le && view.getInt32(0, false) !== 348)
        throw new Error('Not a valid NIfTI-1 file (sizeof_hdr ≠ 348)');

    const ndim = view.getInt16(40, le); // dim[0]
    if (ndim < 1 || ndim > 7) throw new Error('NIfTI ndim out of range: ' + ndim);
    const niftiDims = [];
    for (let i = 1; i <= ndim; i++) {
        niftiDims.push(view.getInt16(40 + i * 2, le));
    }

    const datatype  = view.getInt16(70, le);
    const voxOffset = Math.max(352, Math.floor(view.getFloat32(108, le)));

    // NIfTI-1 datatype → hasty dtype + bytes per element
    const NIFTI_DTYPE_MAP = {
          2: { dtype: 'u8',  bytes: 1 },
          4: { dtype: 'i16', bytes: 2 },
          8: { dtype: 'i32', bytes: 4 },
         16: { dtype: 'f32', bytes: 4 },
         64: { dtype: 'f64', bytes: 8 },
        256: { dtype: 'i8',  bytes: 1 },
        512: { dtype: 'u16', bytes: 2 },
         32: { dtype: 'c32', bytes: 8 }, // complex64
    };
    const dtInfo = NIFTI_DTYPE_MAP[datatype];
    if (!dtInfo) throw new Error('Unsupported NIfTI datatype code: ' + datatype);

    // Reverse dims: NIfTI [X,Y,Z,T] → hasty C-order [T,Z,Y,X]
    const shape     = niftiDims.slice().reverse();
    const totalElem = shape.reduce((a, b) => a * b, 1);
    let   data      = new Uint8Array(buf, voxOffset, totalElem * dtInfo.bytes);

    // Byte-swap multi-byte elements if the file is big-endian
    if (!le && dtInfo.bytes > 1) {
        data = data.slice(); // own copy
        const b = dtInfo.bytes;
        for (let i = 0; i < totalElem; i++) {
            let lo = i * b, hi = lo + b - 1;
            while (lo < hi) {
                const t = data[lo]; data[lo] = data[hi]; data[hi] = t;
                lo++; hi--;
            }
        }
    }

    return { shape, dtype: dtInfo.dtype, data, geometry: readGeometry(view, le) };
}


/**
 * Convert a packed Uint16Array of IEEE 754 half-precision values to Float32Array.
 */
function f16ArrayToF32(u16) {
    const f32 = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
        const v  = u16[i];
        const s  = (v >>> 15) & 1;
        const e  = (v >>> 10) & 0x1f;
        const m  = v & 0x3ff;
        if (e === 0)  { f32[i] = (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024); continue; }
        if (e === 31) { f32[i] = m ? NaN : (s ? -Infinity : Infinity);          continue; }
        f32[i] = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
    }
    return f32;
}

/**
 * Build a NIfTI-1 (.nii) single-file from tensor data.
 *
 * Shape must be in hasty C-order (slowest first); dims are reversed for NIfTI.
 * Data must be raw element bytes (no reordering needed).
 *
 * @param {number[]}  shape    C-order, e.g. [Z, Y, X] or [T, Z, Y, X]
 * @param {string}    dtype    hasty dtype string
 * @param {Uint8Array} data
 * @returns {ArrayBuffer}
 */
function buildNifti1(shape, dtype, data, geometry = null) {
    const DTYPE_TO_NIFTI  = { u8:2, i8:256, i16:4, u16:512, i32:8, f32:16, f64:64, c32:32 };
    const DTYPE_TO_BITPIX = { u8:8, i8:8,   i16:16, u16:16, i32:32, f32:32, f64:64, c32:64 };
    const niftiType = DTYPE_TO_NIFTI[dtype];
    if (!niftiType) throw new Error('Cannot save dtype "' + dtype + '" as NIfTI-1');

    // Reverse C-order shape to NIfTI convention (X fastest)
    const niftiDims = shape.slice().reverse();
    const ndim      = niftiDims.length;

    const voxOffset  = 352; // 348-byte header + 4-byte extension block
    const totalBytes = voxOffset + data.byteLength;
    const buf  = new ArrayBuffer(totalBytes);
    const view = new DataView(buf);
    const u8   = new Uint8Array(buf);

    view.setInt32(0, 348, true);          // sizeof_hdr
    view.setInt16(40, ndim, true);         // dim[0] = ndim
    for (let i = 0; i < Math.min(ndim, 7); i++) {
        view.setInt16(40 + (i + 1) * 2, niftiDims[i], true); // dim[1..ndim]
    }
    view.setInt16(70, niftiType, true);                     // datatype
    view.setInt16(72, DTYPE_TO_BITPIX[dtype], true);        // bitpix
    view.setFloat32(76, 1.0, true);                         // pixdim[0]
    for (let i = 1; i <= Math.min(ndim, 7); i++) {
        view.setFloat32(76 + i * 4, 1.0, true);             // pixdim[1..ndim] = 1 mm
    }
    view.setFloat32(108, voxOffset, true);                  // vox_offset
    if (geometry) writeGeometry(view, geometry);            // pixdim[1..3], sform, qform
    // magic: "n+1\0"
    u8[344] = 110; u8[345] = 43; u8[346] = 49; u8[347] = 0;
    // extension block (4 bytes, all zero = no extensions)
    // data starts at voxOffset
    u8.set(data, voxOffset);
    return buf;
}

/**
 * FFT / IFFT on the server with numpy-style norm: the server's fft is unnormalised and ifft scales by 1/N
 * ("backward"); "ortho" and "forward" rescale afterwards. axesStr: comma-separated axes (negative allowed), empty = all.
 */
async function fftExec(cmd, x, norm, axesStr) {
    const axes = axesStr.trim() ? axesStr.split(/[\s,]+/).filter(Boolean).map(Number) : null;
    let y = (await run(cmd, [x], axes ? { axes } : {})).outputs[0];
    if (norm === 'ortho' || norm === 'forward') {
        const { shape } = await fetchMeta(x);
        const nd = shape.length;
        const n = (axes ?? shape.map((_, i) => i)).map(a => shape[(a + nd) % nd]).reduce((p, q) => p * q, 1);
        const scale = norm === 'ortho' ? (cmd === 'fft' ? 1 / Math.sqrt(n) : Math.sqrt(n)) : (cmd === 'fft' ? 1 / n : n);
        const tmp = y;
        y = (await run('mul', [tmp], { value: scale }, cmd)).outputs[0];
        client.delete(tmp).catch(() => {});
    }
    return [y];
}

// -- Node definitions ---------------------------------------------------------

function registerNodes() {

    // IO -----------------------------------------------------------------------

    class LoadTensorNode extends HNode {
        static title    = "Load Tensor";
        static category = "IO";
        constructor() {
            super();
            this.addWidget("text", "id (hex)", "", v => { this.properties.uuid = v; });
            this.addOutput("tensor", "tensor_uuid");
            this.size = [240, 60];
        }
        async executeAsync(_inputs) {
            const hex = (this.properties.uuid ?? '').replace(/\s/g, '').toLowerCase();
            if (!isId(hex)) throw new Error('id must be 32 hex chars');
            this._execMeta = await fetchMeta(hex);
            return [hex];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class SaveTensorNode extends HNode {
        static title    = "Save Tensor";
        static category = "IO";
        constructor() {
            super();
            this.addInput("tensor", "tensor_uuid");
            this.addWidget("text", "label", "result", v => { this.properties.label = v; });
            this.size = [200, 60];
        }
        async executeAsync([uuid]) {
            if (!uuid) throw new Error('No tensor connected');
            this._execMeta = await fetchMeta(uuid);
            return [uuid];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    // Tensor creation ----------------------------------------------------------

    class TensorCreationBase extends HNode {
        constructor(cmdId) {
            super();
            this._cmdId = cmdId;
            this.addWidget("combo", "dtype",  "f32", v => { this.properties.dtype  = v; },
                { values: ["f32","f64","c32","c64"] });
            this.addWidget("combo", "device", "cpu", v => { this.properties.device = v; },
                { values: ["cpu","cuda:0"] });
            this.addWidget("text",  "shape",  "4,4", v => { this.properties.shape  = v; });
            this.addOutput("tensor", "tensor_uuid");
            this.size = [200, 102];
        }
        async executeAsync(_inputs) {
            const opts = makeTensorOptions(
                this.properties.dtype  ?? 'f32',
                this.properties.device ?? 'cpu',
                this.properties.shape  ?? '4,4');
            return (await run(this._cmdId, [], opts)).outputs;
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class RandTensorNode extends TensorCreationBase {
        static title = "Rand Tensor";  static category = "Create";
        constructor() { super('rand'); }
    }
    class ZerosTensorNode extends TensorCreationBase {
        static title = "Zeros Tensor"; static category = "Create";
        constructor() { super('zeros'); }
    }
    class OnesTensorNode extends TensorCreationBase {
        static title = "Ones Tensor";  static category = "Create";
        constructor() { super('ones'); }
    }

    // Binary arithmetic --------------------------------------------------------

    class BinaryArithBase extends HNode {
        constructor(cmdId) {
            super();
            this._cmdId = cmdId;
            this.addInput("A",    "tensor_uuid");
            this.addInput("B",    "tensor_uuid");
            this.addOutput("out", "tensor_uuid");
            this.size = [150, 62];
        }
        async executeAsync([a, b]) {
            if (!a) throw new Error('Input A not connected');
            if (!b) throw new Error('Input B not connected');
            return (await run(this._cmdId, [a, b])).outputs;
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class AddNode  extends BinaryArithBase {
        static title = "Add";  static category = "Math";
        constructor() { super('add'); }
    }
    class SubNode  extends BinaryArithBase {
        static title = "Sub";  static category = "Math";
        constructor() { super('sub'); }
    }
    class MultNode extends BinaryArithBase {
        static title = "Mult"; static category = "Math";
        constructor() { super('mul'); }
    }
    class DivNode  extends BinaryArithBase {
        static title = "Div";  static category = "Math";
        constructor() { super('div'); }
    }

    // Unary arithmetic ---------------------------------------------------------

    class UnaryArithBase extends HNode {
        constructor(cmdId) {
            super();
            this._cmdId = cmdId;
            this.addInput("in",   "tensor_uuid");
            this.addOutput("out", "tensor_uuid");
            this.size = [140, 56];
        }
        async executeAsync([a]) {
            if (!a) throw new Error('Input not connected');
            return (await run(this._cmdId, [a])).outputs;
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class NegNode extends UnaryArithBase {
        static title = "Neg"; static category = "Math";
        constructor() { super('neg'); }
    }
    class AbsNode extends UnaryArithBase {
        static title = "Abs"; static category = "Math";
        constructor() { super('abs'); }
    }

    // Scalar (no backend) ------------------------------------------------------

    class ScalarNode extends HNode {
        static title = "Scalar"; static category = "Math";
        constructor() {
            super();
            this.addWidget("number", "value", 1.0, v => { this.properties.value = v; });
            this.addOutput("out", "number");
            this.size = [160, 50];
        }
        onExecute() { this.setOutputData(0, this.properties.value ?? 1.0); }
    }

    // FFT stubs ----------------------------------------------------------------

    class FFTNode extends HNode {
        static title = "FFT"; static category = "FFT";
        constructor() {
            super();
            this.addInput("in",      "tensor_uuid");
            this.addOutput("kspace", "tensor_uuid");
            this.addWidget("combo", "norm", "ortho", v => { this.properties.norm = v; },
                { values: ["backward","ortho","forward"] });
            this.addWidget("text", "axes (empty: all)", "", v => { this.properties.axes = v; });
            this.size = [190, 96];
        }
        async executeAsync([x]) {
            if (!x) throw new Error('Input not connected');
            return await fftExec(this.constructor.title === 'FFT' ? 'fft' : 'ifft', x, this.properties.norm ?? 'ortho', this.properties.axes ?? '');
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }
    class IFFTNode extends HNode {
        static title = "IFFT"; static category = "FFT";
        constructor() {
            super();
            this.addInput("kspace", "tensor_uuid");
            this.addOutput("out",   "tensor_uuid");
            this.addWidget("combo", "norm", "ortho", v => { this.properties.norm = v; },
                { values: ["backward","ortho","forward"] });
            this.addWidget("text", "axes (empty: all)", "", v => { this.properties.axes = v; });
            this.size = [190, 96];
        }
        async executeAsync([x]) {
            if (!x) throw new Error('Input not connected');
            return await fftExec(this.constructor.title === 'FFT' ? 'fft' : 'ifft', x, this.properties.norm ?? 'ortho', this.properties.axes ?? '');
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }
    class NUFFTNode extends HNode {
        static title = "NUFFT"; static category = "FFT";
        constructor() {
            super();
            this.addInput("image",  "tensor_uuid");
            this.addInput("coords", "tensor_uuid");
            this.addOutput("kspace","tensor_uuid");
            this.size = [160, 70];
        }
    }

    // MRI stubs ----------------------------------------------------------------

    class CoilSenseNode extends HNode {
        static title = "Coil Sense"; static category = "MRI";
        constructor() {
            super();
            this.addInput("image",      "tensor_uuid");
            this.addInput("smaps",      "tensor_uuid");
            this.addOutput("coil_imgs", "tensor_uuid");
            this.size = [170, 70];
        }
    }
    class SensitivityMapNode extends HNode {
        static title = "Sensitivity Maps"; static category = "MRI";
        constructor() {
            super();
            this.addInput("kspace", "tensor_uuid");
            this.addOutput("smaps", "tensor_uuid");
            this.size = [170, 56];
        }
    }
    class ToeplitzNode extends HNode {
        static title = "Toeplitz"; static category = "MRI";
        constructor() {
            super();
            this.addInput("ktraj",   "tensor_uuid");
            this.addInput("weights", "tensor_uuid");
            this.addOutput("kernel", "tensor_uuid");
            this.size = [170, 70];
        }
    }

    // Viz ----------------------------------------------------------------------

    class CompressUi16ConfigNode extends HNode {
        static title = "Compress UI16"; static category = "Viz";
        constructor() {
            super();
            this.addInput("tensor", "tensor_uuid");
            this.addOutput("compressed", "tensor_uuid");
            this.addOutput("config",     "comprep_config");
            this.addWidget("text", "a (empty: p1)",  "",    v => { this.properties.a           = v; });
            this.addWidget("text", "b (empty: p99)", "",    v => { this.properties.b           = v; });
            this.addWidget("text", "clampa",      "",       v => { this.properties.clampa      = v; });
            this.addWidget("text", "clampb",      "",       v => { this.properties.clampb      = v; });
            this.addWidget("text", "focus",       "1.0",    v => { this.properties.focus       = v; });
            this.addWidget("combo", "left_mode",  "linear", v => { this.properties.left_mode  = v; },
                { values: ["linear", "gamma", "log"] });
            this.addWidget("combo", "right_mode", "linear", v => { this.properties.right_mode = v; },
                { values: ["linear", "gamma", "log"] });
            this.addWidget("text", "left_gamma",  "1.0",    v => { this.properties.left_gamma  = v; });
            this.addWidget("text", "right_gamma", "1.0",    v => { this.properties.right_gamma = v; });
            this.addWidget("text", "left_logc",   "1.0",    v => { this.properties.left_logc   = v; });
            this.addWidget("text", "right_logc",  "1.0",    v => { this.properties.right_logc  = v; });
            this.size = [220, 290];
        }
        _buildConfig() {
            const p = this.properties;
            const n = s => { const v = parseFloat(s); return isNaN(v) ? null : v; };
            const cfg = {};
            const a = n(p.a); if (a !== null) cfg.a = a;
            const b = n(p.b); if (b !== null) cfg.b = b;
            const ca = n(p.clampa); if (ca !== null) cfg.clampa = ca;
            const cb = n(p.clampb); if (cb !== null) cfg.clampb = cb;
            const f  = n(p.focus);  if (f  !== null) cfg.focus  = f;
            if (p.left_mode  && p.left_mode  !== 'linear') cfg.left_mode  = p.left_mode;
            if (p.right_mode && p.right_mode !== 'linear') cfg.right_mode = p.right_mode;
            const lg = n(p.left_gamma);  if (lg !== null && lg !== 1.0) cfg.left_gamma  = lg;
            const rg = n(p.right_gamma); if (rg !== null && rg !== 1.0) cfg.right_gamma = rg;
            const ll = n(p.left_logc);   if (ll !== null && ll !== 1.0) cfg.left_logc   = ll;
            const rl = n(p.right_logc);  if (rl !== null && rl !== 1.0) cfg.right_logc  = rl;
            return cfg;
        }
        async executeAsync([uuid]) {
            if (!uuid) throw new Error('No tensor connected');
            const cfg    = this._buildConfig();
            this._lastConfig = cfg;
            const r = await run('comprep_u16', [uuid], cfg, 'comprep');
            return [r.outputs[0], JSON.parse(r.message)];     // the full config (auto values filled in)
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class OrthoSlicerNode extends HNode {
        static title = "Ortho Slicer"; static category = "Viz";
        constructor() {
            super();
            this.addInput("volume", "tensor_uuid");
            this.addInput("config", "comprep_config");
            this._uuid        = null;
            this._wiredConfig = null;
            this._configStr   = '';
            this.addWidget("text", "Comprep Config JSON", '', v => { this._configStr = v; });
            this.addWidget("button", "Open Viewer", null, () => {
                if (!this._uuid) { alert('Connect a tensor and run the graph first.'); return; }
                let url = '../orthoslicer/index.html?id=' + this._uuid;
                if (this._wiredConfig) url += '&prepped=1&config=' + encodeURIComponent(JSON.stringify(this._wiredConfig));
                else if (this._configStr.trim()) url += '&config=' + encodeURIComponent(this._configStr.trim());
                window.open(url, '_blank');
            });
            this.size = [220, 110];
        }
        /** the viewer builds the comprep display copy itself; a wired config means the input already is one */
        async executeAsync([uuid, config]) {
            this._uuid        = uuid ?? null;
            this._wiredConfig = config ?? null;
            return this._uuid ? [this._uuid] : [];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /**
     * NiftiLoadNode — browse for a .nii/.nii.gz file, upload it to the server,
     * and output the UUID.
     *
     * File dialog is opened on "Browse" button click (user gesture required).
     * On subsequent graph runs with no new file selected, the cached UUID is
     * returned without re-uploading.
     *
     * static _skipCache = true so node_runner always calls executeAsync, letting
     * the node manage its own file-change detection internally.
     */
    class NiftiLoadNode extends HNode {
        static title    = "NIfTI Load";
        static category = "IO";
        static _skipCache = true;
        static infoLine = true;

        constructor() {
            super();
            this.addOutput("tensor", "tensor_uuid");
            this._uuid        = null;
            this._pendingFile = null;

            // Hidden file input for the browser dialog
            this._fileInput = document.createElement('input');
            this._fileInput.type   = 'file';
            this._fileInput.accept = '.nii,.nii.gz,.gz';
            this._fileInput.addEventListener('change', () => {
                const f = this._fileInput.files?.[0];
                if (f) {
                    this._pendingFile = f;
                    this._info        = 'pending: ' + f.name;
                    this._uuid        = null; // invalidate cache
                    this._execStatus  = null;
                }
            });

            this.addWidget('button', 'Browse…', null, () => {
                this._fileInput.click();
            });

            this.size = [200, 72];
        }

        async executeAsync(_inputs) {
            if (this._pendingFile) {
                const file = this._pendingFile;
                this._pendingFile = null;
                this._info = file.name;

                const { shape, dtype, data, geometry } = await parseNifti(file);
                const tensor    = HTensor.fromBytes(DTYPE[dtype], shape, data);
                const axes      = shape.map((_, i) => ['x', 'y', 'z', 't', 'e'][shape.length - 1 - i] ?? `d${i}`);
                const uuid      = await client.put(tensor, { name: file.name, metadata: { source: file.name, axes, ...geometryMetadata(geometry) } });
                this._uuid      = uuid;
                this._execMeta  = { dtype, shape, device: 'cpu' };
                return [uuid];
            }
            if (this._uuid) {
                return [this._uuid];
            }
            // No file loaded yet — return empty (don't throw so graph can still run)
            return [];
        }

        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /**
     * NiftiSaveNode — accepts a tensor UUID input and provides a "Download NIfTI"
     * button that fetches the tensor from the server and triggers a browser download
     * of a NIfTI-1 (.nii) file.
     *
     * f16 tensors are automatically converted to f32 (NIfTI-1 has no f16 type).
     */
    class NiftiSaveNode extends HNode {
        static title    = "NIfTI Save";
        static category = "IO";

        constructor() {
            super();
            this.addInput("tensor", "tensor_uuid");
            this.addWidget('text', 'filename', 'tensor.nii',
                v => { this.properties.filename = v; });
            this.addWidget('button', 'Download NIfTI', null, () => {
                if (!this._uuid) { alert('Connect a tensor and run the graph first.'); return; }
                this._doSave().catch(e => alert('NIfTI save failed:\n' + e.message));
            });
            this._uuid = null;
            this.size  = [220, 88];
        }

        async executeAsync([uuid]) {
            if (uuid) this._uuid = uuid;
            return this._uuid ? [this._uuid] : [];
        }

        async _doSave() {
            const t = await client.get(this._uuid, { codec: 'lz4' });
            if (!(t instanceof HTensor)) throw new Error('value is not a tensor');
            const shape   = t.shape;
            let saveDtype = SHORT[t.dtype];
            let data      = t.bytes;
            if (saveDtype === 'f16') {
                data      = new Uint8Array(f16ArrayToF32(new Uint16Array(t.data.buffer, t.data.byteOffset, t.numel)).buffer);
                saveDtype = 'f32';
            }
            const md = (await client.info(this._uuid))?.metadata;
            const geometry = md?.affine || md?.spacing_mm ? { affine: md.affine, spacing_mm: md.spacing_mm } : null;
            const niftiBytes = buildNifti1(shape, saveDtype, data, geometry);
            const filename   = (this.properties.filename ?? 'tensor.nii')
                .replace(/\.gz$/i, '').replace(/\.nii$/i, '') + '.nii';
            const a = document.createElement('a');
            a.href     = URL.createObjectURL(new Blob([niftiBytes], { type: 'application/octet-stream' }));
            a.download = filename;
            a.click();
            setTimeout(() => URL.revokeObjectURL(a.href), 10000);
        }

        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }


    /**
     * ModelRunNode — run a segmentation model installed on the server (data/hasty/models, see model_list).
     * Geometry (spacing_mm, affine) is taken from the input entry's metadata (set by NIfTI Load, or by any
     * producer); the "spacing_mm" widget overrides it for data without a header. The label output carries the
     * geometry and the label table in its metadata, so NIfTI Save writes an aligned label map.
     */
    class ModelRunNode extends HNode {
        static title    = "Run Model";
        static category = "Models";
        static infoLine = true;

        constructor() {
            super();
            this.addInput("image", "tensor_uuid");
            this.addInput("image 2", "tensor_uuid");   // further channels of multi-contrast models (same grid), in model order
            this.addInput("image 3", "tensor_uuid");
            this.addOutput("labels", "tensor_uuid");
            this.addOutput("probabilities", "tensor_uuid");
            this._models = [];
            this._wModel = this.addWidget("combo", "model", "", v => { this.properties.model = v; }, { values: [] });
            this.addWidget("combo", "device", "cpu", v => { this.properties.device = v; }, { values: ["cuda", "cpu"] });
            this.addWidget("number", "overlap (0 = model)", 0, v => { this.properties.overlap = v; }, { min: 0, max: 0.9, step: 0.05, precision: 2 });
            this.addWidget("toggle", "mirror TTA", false, v => { this.properties.tta = v; });
            this.addWidget("text", "prob classes", "", v => { this.properties.prob_classes = v; });
            this.addWidget("text", "spacing_mm (x,y,z)", "", v => { this.properties.spacing = v; });
            this.addWidget("button", "Refresh models", null, () => { this.refreshModels(); });
            this.size = [260, 210];
            this.refreshModels();
        }

        async refreshModels() {
            try {
                const { message } = await run("model_list", [], {});
                const j = JSON.parse(message);
                this._models = j.models;
                this._problems = j.problems;
                const names = j.models.filter(m => m.kind === "segmentation").map(m => m.name);  // flow_sr models need a different node
                this._wModel.options.values = names;
                if (!names.includes(this._wModel.value)) this._wModel.value = names[0] ?? "";
                this.properties.model = this._wModel.value;
            } catch (e) {
                this._problems = [String(e.message ?? e)];
            }
            this.setDirtyCanvas(true, true);
        }

        async executeAsync(inputs) {
            const images = inputs.filter(isId);
            const image = images[0];
            if (!image) throw new Error('connect an image tensor');
            const model = this.properties.model || this._wModel.value;
            if (!model) throw new Error('no model installed on the server (data/hasty/models)');
            const md = (await client.info(image))?.metadata ?? {};
            const options = { model, device: this.properties.device ?? "cpu", tta: !!this.properties.tta };
            if (this.properties.overlap > 0) options.overlap = this.properties.overlap;
            const sp = (this.properties.spacing ?? "").split(/[\s,]+/).filter(Boolean).map(Number);
            if (sp.length === 3 && sp.every(v => v > 0)) options.spacing_mm = sp;
            else { options.spacing_mm = md.spacing_mm; options.affine = md.affine; }
            if (!options.spacing_mm && !options.affine)
                throw new Error('image has no spacing metadata: fill the spacing_mm widget');
            const pc = (this.properties.prob_classes ?? "").split(/[\s,]+/).filter(Boolean).map(Number);
            if (pc.length) options.prob_classes = pc;
            const { outputs, message } = await run("model_run", images, options, model);
            const info = JSON.parse(message);
            const geom = { spacing_mm: options.spacing_mm, affine: options.affine, axes: md.axes };
            await client.setMetadata(outputs[0], { ...geom, labels: info.labels, segmentation_of: image, model, counts: info.counts });
            if (outputs[1]) await client.setMetadata(outputs[1], { ...geom, prob_classes: pc, model });
            this._info = `${info.backend}${info.cuda ? "/cuda" : "/cpu"} ${info.seconds.toFixed(1)} s`;
            return outputs;
        }

        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }


    // Flow (4D flow values on the server: dict kind "flow4d", see rustlib/crates/hasty-server/src/flow.rs) ----------

    const nums = (str) => String(str ?? '').split(/[\s,]+/).filter(Boolean).map(Number);
    const vec3 = (str, what) => { const v = nums(str); if (v.length !== 3 || v.some(Number.isNaN)) throw new Error(`${what}: three numbers x,y,z`); return v; };

    /** value ids of flow values in the bank: [{ label, id, dims }] */
    async function listFlows() {
        return (await client.list()).filter((e) => e.summary?.type === 'dict' && e.summary.items?.vel)
            .map((e) => ({ id: e.id, label: `${e.name} ${e.summary.items.vel.shape.join('x')} ${String(e.id).slice(0, 6)}` }));
    }

    /** brings an existing flow value of the bank into the graph (show_flow / recon runs put them there) */
    class FlowLoadNode extends HNode {
        static title    = "Flow Load";
        static category = "Flow";
        static infoLine = true;
        static _skipCache = true;
        constructor() {
            super();
            this.addOutput("flow", "flow_uuid");
            this._flows = [];
            this._wFlow = this.addWidget("combo", "flow", "", v => { this.properties.flow = v; }, { values: [] });
            this.addWidget("button", "Refresh", null, () => { this.refresh(); });
            this.size = [270, 90];
            this.refresh();
        }
        async refresh() {
            try {
                this._flows = await listFlows();
                this._wFlow.options.values = this._flows.map(f => f.label);
                if (!this._flows.some(f => f.label === this._wFlow.value)) this._wFlow.value = this._flows.at(-1)?.label ?? '';
                this.properties.flow = this._wFlow.value;
                this._info = this._flows.length ? `${this._flows.length} flow value(s) on the server` : 'no flow values on the server';
            } catch (e) { this._info = 'server: ' + String(e.message ?? e).slice(0, 40); }
            this.setDirtyCanvas(true, true);
        }
        async executeAsync() {
            await this.refresh();
            const want = this.properties.flow ?? this._wFlow.value;
            // saved graphs keep the label; the id suffix differs after a server restart, so fall back to the name part
            const f = this._flows.find(x => x.label === want) ?? this._flows.find(x => x.label.split(' ').slice(0, -1).join(' ') === String(want).split(' ').slice(0, -1).join(' '));
            if (!f) throw new Error('no such flow value on the server: ' + want);
            return [f.id];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /** flow value from tensors: vel [F,3,Z,Y,X] (hasty [X,Y,Z,3,F], m/s), optional mag [F,Z,Y,X] and mask [Z,Y,X] */
    class FlowMakeNode extends HNode {
        static title    = "Flow Make";
        static category = "Flow";
        static infoLine = true;
        constructor() {
            super();
            this.addInput("vel", "tensor_uuid");
            this.addInput("mag", "tensor_uuid");
            this.addInput("mask", "tensor_uuid");
            this.addOutput("flow", "flow_uuid");
            this.addWidget("text", "spacing_mm (empty: from vel)", "", v => { this.properties.spacing = v; });
            this.addWidget("number", "dt (s per frame)", 0.04, v => { this.properties.dt = v; }, { min: 0.0001, step: 0.1, precision: 4 });
            this.addWidget("number", "venc (m/s, 0 = none)", 0, v => { this.properties.venc = v; }, { min: 0, step: 0.5, precision: 2 });
            this.size = [270, 150];
        }
        async executeAsync([vel, mag, mask]) {
            if (!vel) throw new Error('connect vel');
            if (mask && !mag) throw new Error('mask needs the mag input as well (inputs are positional)');
            const md = (await client.info(vel))?.metadata ?? {};
            const mm = (this.properties.spacing ?? '').trim() ? vec3(this.properties.spacing, 'spacing_mm') : (md.spacing_mm ?? [2, 2, 2]);
            const options = { spacing: mm.map(v => v / 1000), dt: this.properties.dt ?? 0.04 };
            if (this.properties.venc > 0) options.venc = this.properties.venc;
            const inputs = [vel, mag, mask].filter(isId);
            const { outputs } = await run('flow_make', inputs, options, 'flow');
            this._info = `spacing ${mm.join('x')} mm, dt ${(options.dt * 1e3).toFixed(0)} ms`;
            return outputs;
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /** summary of a flow value; passes the flow through. info output: { dims, frames, spacing (m), dt, venc, p99_speed, peak_frame, ... } */
    class FlowInfoNode extends HNode {
        static title    = "Flow Info";
        static category = "Flow";
        static infoLine = true;
        constructor() {
            super();
            this.addInput("flow", "flow_uuid");
            this.addOutput("flow", "flow_uuid");
            this.addOutput("info", "flow_info");
            this.size = [320, 60];
        }
        async executeAsync([flow]) {
            if (!flow) throw new Error('connect a flow value');
            const j = JSON.parse((await run('flow_info', [flow], {})).message);
            this._info = `${j.dims.join('x')} x${j.frames}f, dt ${(j.dt * 1e3).toFixed(0)} ms, p99 ${j.p99_speed.toFixed(2)} m/s, peak f${j.peak_frame}`;
            return [flow, j];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /** time-averaged PC-MRA of a flow value as a [Z,Y,X] tensor (carries spacing_mm): feed it to Ortho Slicer / NIfTI Save */
    class PcmraNode extends HNode {
        static title    = "PC-MRA";
        static category = "Flow";
        constructor() {
            super();
            this.addInput("flow", "flow_uuid");
            this.addOutput("pcmra", "tensor_uuid");
            this.addWidget("combo", "method", "mag_speed", v => { this.properties.method = v; }, { values: ["mag_speed", "speed", "speed_max"] });
            this.size = [220, 70];
        }
        async executeAsync([flow]) {
            if (!flow) throw new Error('connect a flow value');
            const { outputs } = await run('flow_pcmra', [flow], { method: this.properties.method ?? 'mag_speed' }, 'pcmra');
            const j = JSON.parse((await run('flow_info', [flow], {})).message);
            await client.setMetadata(outputs[0], { spacing_mm: j.spacing.map(v => v * 1000), axes: ['x', 'y', 'z'], source: 'flow_pcmra' });
            return outputs;
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /** flow through an analysis disc (flow_plane). center is in voxel coordinates; empty = middle of the volume. */
    class FlowPlaneNode extends HNode {
        static title    = "Flow Plane";
        static category = "Flow";
        static infoLine = true;
        constructor() {
            super();
            this.addInput("flow", "flow_uuid");
            this.addOutput("velocity", "tensor_uuid");   // through-plane velocity [F, n, n], m/s
            this.addOutput("region", "tensor_uuid");     // lumen mask [n, n]
            this.addOutput("summary", "flow_summary");   // flow_ml_s per frame, net / forward / backward ml, peak and mean velocity, area
            this.addWidget("text", "center x,y,z (voxel)", "", v => { this.properties.center = v; });
            this.addWidget("text", "normal x,y,z", "0,0,1", v => { this.properties.normal = v; });
            this.addWidget("number", "radius (mm)", 15, v => { this.properties.radius = v; }, { min: 1, step: 5, precision: 1 });
            this.addWidget("number", "grid", 64, v => { this.properties.grid = Math.round(v); }, { min: 16, max: 256, step: 160, precision: 0 });
            this.addWidget("text", "lumen quantile (empty: whole disc)", "", v => { this.properties.quantile = v; });
            this.size = [290, 200];
        }
        async executeAsync([flow]) {
            if (!flow) throw new Error('connect a flow value');
            let center = (this.properties.center ?? '').trim() ? vec3(this.properties.center, 'center') : null;
            if (!center) { const j = JSON.parse((await run('flow_info', [flow], {})).message); center = j.dims.map(d => (d - 1) / 2); }
            const options = { center, normal: vec3(this.properties.normal ?? '0,0,1', 'normal'), radius_mm: this.properties.radius ?? 15, grid: this.properties.grid ?? 64 };
            const q = parseFloat(this.properties.quantile);
            if (!Number.isNaN(q)) options.quantile = q;
            const { outputs, message } = await run('flow_plane', [flow], options, 'flow plane');
            const j = JSON.parse(message);
            this._info = `net ${j.net_ml.toFixed(2)} ml, fwd ${j.forward_ml.toFixed(2)}, bwd ${j.backward_ml.toFixed(2)}, ${j.area_mm2.toFixed(0)} mm2`;
            return [outputs[0], outputs[1], j];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /**
     * Opens the flow viewer (3D PC-MRA, slices, glyphs, stream / pathlines, vessel tool) on the connected flow value in a new
     * tab. The viewer reads the value from the server itself; nothing is computed in the graph.
     */
    class FlowViewerNode extends HNode {
        static title    = "Flow Viewer";
        static category = "Flow";
        constructor() {
            super();
            this.addInput("flow", "flow_uuid");
            this._uuid = null;
            this.addWidget("number", "start frame", 0, v => { this.properties.frame = Math.round(v); }, { min: 0, step: 10, precision: 0 });
            this.addWidget("combo", "particles", "server", v => { this.properties.compute = v; }, { values: ["server", "browser (WebGPU)"] });   // browser = loads the whole volume into the tab
            this.addWidget("button", "Open Viewer", null, () => {
                if (!this._uuid) { alert('Connect a flow value and run the graph first.'); return; }
                const p = new URLSearchParams({ id: this._uuid });
                if (this.properties.frame > 0) p.set('frame', String(this.properties.frame));
                if ((this.properties.compute ?? '').startsWith('browser')) p.set('client', '1'); else p.set('client', '0');
                window.open('../flow/index.html?' + p, '_blank');
            });
            this.size = [240, 110];
        }
        async executeAsync([flow]) {
            this._uuid = flow ?? null;
            return this._uuid ? [this._uuid] : [];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }


    // Server plans and run results (see server_run.js) --------------------------------------------------------------
    // Each node that can run on the server describes one command step; nodes without plan() run in the browser.

    const geomKeys = ['spacing_mm', 'affine', 'axes'];
    const needAll = (node, n) => { for (let i = 0; i < n; i++) if (node.inputs[i]?.link == null) throw new Error(`input ${node.inputs[i]?.name ?? i} not connected`); };

    Object.assign(LoadTensorNode.prototype, {
        async plan() {
            const hex = (this.properties.uuid ?? '').replace(/\s/g, '').toLowerCase();
            if (!isId(hex)) throw new Error('id must be 32 hex chars');
            return { cmd: '_const', options: { value: hex }, inputs: [] };
        },
    });
    for (const C of [SaveTensorNode, NiftiSaveNode, FlowViewerNode]) {
        Object.assign(C.prototype, {
            async plan() { return { cmd: '_pass' }; },
            onRunResult(outs) { this._uuid = outs?.[0] ?? null; },
        });
    }
    Object.assign(OrthoSlicerNode.prototype, {
        async plan() { return { cmd: '_pass' }; },
        onRunResult(outs) { this._uuid = outs?.[0] ?? null; this._wiredConfig = outs?.[1] ?? null; },
    });
    Object.assign(NiftiLoadNode.prototype, {
        async plan() {
            // a file chosen in this tab must be uploaded by this tab: a browser step; afterwards the id is a constant
            if (this._pendingFile) return { client: true };
            return { cmd: '_const', options: { value: this._uuid ?? null }, inputs: [] };
        },
    });
    Object.assign(ScalarNode.prototype, {
        async plan() { return { cmd: '_const', options: { value: this.properties.value ?? 1.0 }, inputs: [] }; },
        async executeAsync() { return [this.properties.value ?? 1.0]; },
    });
    Object.assign(TensorCreationBase.prototype, {
        async plan() {
            return { cmd: this._cmdId, options: makeTensorOptions(this.properties.dtype ?? 'f32', this.properties.device ?? 'cpu', this.properties.shape ?? '4,4'), inputs: [] };
        },
    });
    Object.assign(BinaryArithBase.prototype, { async plan() { needAll(this, 2); return { cmd: this._cmdId }; } });
    Object.assign(UnaryArithBase.prototype, { async plan() { needAll(this, 1); return { cmd: this._cmdId }; } });
    for (const [C, cmd] of [[FFTNode, 'fft'], [IFFTNode, 'ifft']]) {
        Object.assign(C.prototype, {
            async plan() {
                needAll(this, 1);
                const axes = (this.properties.axes ?? '').trim() ? nums(this.properties.axes) : null;
                return { cmd, options: { norm: this.properties.norm ?? 'ortho', ...(axes ? { axes } : {}) } };
            },
        });
    }
    Object.assign(CompressUi16ConfigNode.prototype, {
        async plan() {
            needAll(this, 1);
            return { cmd: 'comprep_u16', options: this._buildConfig(), name: 'comprep', outs: [{ output: 0 }, { message: true }] };   // 2nd output: the full config (auto values filled in)
        },
    });
    Object.assign(ModelRunNode.prototype, {
        async plan() {
            needAll(this, 1);
            const model = this.properties.model || this._wModel.value;
            if (!model) throw new Error('no model installed on the server (data/hasty/models)');
            const options = { model, device: this.properties.device ?? 'cpu', tta: !!this.properties.tta };
            if (this.properties.overlap > 0) options.overlap = this.properties.overlap;
            const sp = nums(this.properties.spacing);
            const own = sp.length === 3 && sp.every(v => v > 0);
            if (own) options.spacing_mm = sp;
            const pc = nums(this.properties.prob_classes);
            if (pc.length) options.prob_classes = pc;
            const geom = own ? { set: { spacing_mm: sp } } : { from_input: { input: 0, keys: geomKeys } };
            return {
                cmd: 'model_run', options, name: model,
                meta_in: own ? [] : [{ input: 0, keys: { spacing_mm: 'spacing_mm', affine: 'affine' } }],   // the input's own header, unless the widget overrides it
                meta_out: [
                    { output: 0, ...geom, from_message: { labels: 'labels', counts: 'counts' }, set_input_id: { segmentation_of: 0 }, set: { model, ...(geom.set ?? {}) } },
                    { output: 1, ...geom, set: { prob_classes: pc, model, ...(geom.set ?? {}) } },
                ],
            };
        },
        onRunResult(_outs, msg) { if (msg?.backend) this._info = `${msg.backend}${msg.cuda ? '/cuda' : '/cpu'} ${msg.seconds.toFixed(1)} s`; },
    });
    Object.assign(FlowLoadNode.prototype, {
        async plan() { return { cmd: '_const', options: { value: (await this.executeAsync())[0] }, inputs: [] }; },
    });
    Object.assign(FlowMakeNode.prototype, {
        async plan() {
            needAll(this, 1);
            if (this.inputs[2]?.link != null && this.inputs[1]?.link == null) throw new Error('mask needs the mag input as well (inputs are positional)');
            const options = { dt: this.properties.dt ?? 0.04 };
            if ((this.properties.spacing ?? '').trim()) options.spacing_mm = vec3(this.properties.spacing, 'spacing_mm');
            if (this.properties.venc > 0) options.venc = this.properties.venc;
            return { cmd: 'flow_make', options, name: 'flow', meta_in: [{ input: 0, keys: { spacing_mm: 'spacing_mm' } }] };
        },
    });
    Object.assign(FlowInfoNode.prototype, {
        async plan() { needAll(this, 1); return { cmd: 'flow_info', outs: [{ input: 0 }, { message: true }] }; },
        onRunResult(_o, j) { if (j?.dims) this._info = `${j.dims.join('x')} x${j.frames}f, dt ${(j.dt * 1e3).toFixed(0)} ms, p99 ${j.p99_speed.toFixed(2)} m/s, peak f${j.peak_frame}`; },
    });
    Object.assign(PcmraNode.prototype, {
        async plan() {
            needAll(this, 1);
            return { cmd: 'flow_pcmra', options: { method: this.properties.method ?? 'mag_speed' }, name: 'pcmra',
                     meta_out: [{ output: 0, from_message: { spacing_mm: 'spacing_mm' }, set: { axes: ['x', 'y', 'z'], source: 'flow_pcmra' } }], outs: [{ output: 0 }] };
        },
    });
    Object.assign(FlowPlaneNode.prototype, {
        async plan() {
            needAll(this, 1);
            const options = { normal: vec3(this.properties.normal ?? '0,0,1', 'normal'), radius_mm: this.properties.radius ?? 15, grid: this.properties.grid ?? 64 };
            if ((this.properties.center ?? '').trim()) options.center = vec3(this.properties.center, 'center');   // else the server uses the middle of the volume
            const q = parseFloat(this.properties.quantile);
            if (!Number.isNaN(q)) options.quantile = q;
            return { cmd: 'flow_plane', options, name: 'flow plane', outs: [{ output: 0 }, { output: 1 }, { message: true }] };
        },
        onRunResult(_o, j) { if (j?.net_ml != null) this._info = `net ${j.net_ml.toFixed(2)} ml, fwd ${j.forward_ml.toFixed(2)}, bwd ${j.backward_ml.toFixed(2)}, ${j.area_mm2.toFixed(0)} mm2`; },
    });


    // Geometry (triangle meshes: hasty mesh value, see rustlib/crates/hasty-server/src/mesh.rs) --------------------------

    /** iso-surface of a mask / label volume (vessel tree, organ, ...) */
    class MeshFromMaskNode extends HNode {
        static title    = "Mesh From Mask";
        static category = "Geometry";
        static infoLine = true;
        constructor() {
            super();
            this.addInput("mask", "tensor_uuid");
            this.addOutput("mesh", "mesh_uuid");
            this.addOutput("info", "mesh_info");
            this.addWidget("text", "label (empty: threshold)", "", v => { this.properties.label = v; });
            this.addWidget("number", "threshold", 0.5, v => { this.properties.threshold = v; }, { step: 0.5, precision: 3 });
            this.addWidget("number", "blur passes", 1, v => { this.properties.blur = Math.round(v); }, { min: 0, max: 6, step: 10, precision: 0 });
            this.addWidget("number", "smooth iterations", 6, v => { this.properties.smooth = Math.round(v); }, { min: 0, max: 100, step: 10, precision: 0 });
            this.addWidget("combo", "keep", "largest", v => { this.properties.keep = v; }, { values: ["largest", "all"] });
            this.size = [280, 170];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class MeshSmoothNode extends HNode {
        static title    = "Mesh Smooth";
        static category = "Geometry";
        constructor() {
            super();
            this.addInput("mesh", "mesh_uuid");
            this.addOutput("mesh", "mesh_uuid");
            this.addWidget("number", "iterations", 10, v => { this.properties.iterations = Math.round(v); }, { min: 1, max: 200, step: 10, precision: 0 });
            this.size = [220, 70];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class MeshLargestNode extends HNode {
        static title    = "Mesh Largest Part";
        static category = "Geometry";
        constructor() {
            super();
            this.addInput("mesh", "mesh_uuid");
            this.addOutput("mesh", "mesh_uuid");
            this.size = [200, 50];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /** sample a volume onto the mesh as a per-vertex scalar (e.g. PC-MRA, speed, a probability map) */
    class MeshSampleNode extends HNode {
        static title    = "Mesh Sample";
        static category = "Geometry";
        constructor() {
            super();
            this.addInput("mesh", "mesh_uuid");
            this.addInput("volume", "tensor_uuid");
            this.addOutput("mesh", "mesh_uuid");
            this.addWidget("text", "scalar name", "value", v => { this.properties.name = v; });
            this.size = [230, 90];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class MeshInfoNode extends HNode {
        static title    = "Mesh Info";
        static category = "Geometry";
        static infoLine = true;
        constructor() {
            super();
            this.addInput("mesh", "mesh_uuid");
            this.addOutput("mesh", "mesh_uuid");
            this.addOutput("info", "mesh_info");
            this.size = [320, 60];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /** opens the geometry viewer on up to four meshes / line sets */
    class MeshViewerNode extends HNode {
        static title    = "Mesh Viewer";
        static category = "Geometry";
        constructor() {
            super();
            for (const n of ["geometry", "geometry 2", "geometry 3", "geometry 4"]) this.addInput(n, "mesh_uuid,lines_uuid,meshseq_uuid");
            this._ids = [];
            this.addWidget("text", "colour by (scalar)", "", v => { this.properties.scalar = v; });
            this.addWidget("button", "Open Viewer", null, () => {
                if (!this._ids.length) { alert('Connect a mesh and run the graph first.'); return; }
                const p = new URLSearchParams();
                this._ids.forEach(i => p.append('id', i));
                if ((this.properties.scalar ?? '').trim()) p.set('scalar', this.properties.scalar.trim());
                window.open('../mesh/index.html?' + p, '_blank');
            });
            this.size = [240, 150];
        }
        async executeAsync(inputs) { this._ids = inputs.filter(isId); return this._ids; }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /** upload an OBJ / STL / PLY file as a mesh value (converted to mm from the chosen file units) */
    class MeshLoadNode extends HNode {
        static title    = "Mesh Load";
        static category = "Geometry";
        static _skipCache = true;
        static infoLine = true;
        constructor() {
            super();
            this.addOutput("mesh", "mesh_uuid");
            this._uuid = null; this._pendingFile = null;
            this._fileInput = document.createElement('input');
            this._fileInput.type = 'file';
            this._fileInput.accept = '.obj,.stl,.ply';
            this._fileInput.addEventListener('change', () => {
                const f = this._fileInput.files?.[0];
                if (f) { this._pendingFile = f; this._uuid = null; this._execStatus = null; this._info = 'pending: ' + f.name; }
            });
            this.addWidget('button', 'Browse…', null, () => this._fileInput.click());
            this.addWidget('combo', 'file units', 'mm', v => { this.properties.units = v; }, { values: ['mm', 'cm', 'm', 'um'] });
            this.size = [220, 110];
        }
        async executeAsync() {
            if (this._pendingFile) {
                const file = this._pendingFile;
                this._pendingFile = null;
                const m = parseMesh(await file.arrayBuffer(), file.name);
                const k = { mm: 1, cm: 10, m: 1000, um: 0.001 }[this.properties.units ?? 'mm'] ?? 1;   // the mesh value is in mm
                if (k !== 1) for (let i = 0; i < m.vertices.length; i++) m.vertices[i] *= k;
                this._uuid = await client.put(meshValue(m, { space: 'voxel_mm' }), { name: file.name, metadata: { source: file.name } });
                this._info = `${m.vertices.length / 3} vertices, ${m.faces.length / 3} triangles`;
            }
            return this._uuid ? [this._uuid] : [];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /** download a mesh as OBJ / STL / PLY */
    class MeshSaveNode extends HNode {
        static title    = "Mesh Save";
        static category = "Geometry";
        constructor() {
            super();
            this.addInput("mesh", "mesh_uuid");
            this.addWidget('text', 'filename', 'mesh', v => { this.properties.filename = v; });
            this.addWidget('combo', 'format', 'obj', v => { this.properties.format = v; }, { values: ['obj', 'stl', 'ply'] });
            this.addWidget('button', 'Download', null, () => {
                if (!this._uuid) { alert('Connect a mesh and run the graph first.'); return; }
                this._save().catch(e => alert('Mesh save failed:\n' + e.message));
            });
            this._uuid = null;
            this.size = [230, 120];
        }
        async executeAsync([uuid]) { if (uuid) this._uuid = uuid; return this._uuid ? [this._uuid] : []; }
        async _save() {
            const v = await client.get(this._uuid);
            if (!(v instanceof Map) || v.get('kind') !== 'mesh') throw new Error('value is not a mesh');
            const fmt = this.properties.format ?? 'obj';
            const bytes = WRITERS[fmt]({ vertices: v.get('vertices').data, faces: v.get('faces').data });
            const a = document.createElement('a');
            a.href = URL.createObjectURL(new Blob([bytes], { type: 'application/octet-stream' }));
            a.download = (this.properties.filename || 'mesh').replace(/\.(obj|stl|ply)$/i, '') + '.' + fmt;
            a.click();
            setTimeout(() => URL.revokeObjectURL(a.href), 10000);
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    const meshText = (j) => j?.vertices != null ? `${j.vertices} vertices, ${j.faces} triangles, ${Number(j.area_mm2).toFixed(0)} mm2${Math.abs(j.volume_mm3) > 0 ? ', ' + (Math.abs(j.volume_mm3) / 1000).toFixed(2) + ' ml' : ''}` : '';
    Object.assign(MeshFromMaskNode.prototype, {
        async plan() {
            needAll(this, 1);
            const options = { blur: this.properties.blur ?? 1, smooth: this.properties.smooth ?? 6, keep: this.properties.keep ?? 'largest' };
            const lab = parseFloat(this.properties.label);
            if (!Number.isNaN(lab)) options.label = lab; else options.threshold = this.properties.threshold ?? 0.5;
            return { cmd: 'mesh_from_mask', options, name: 'mesh', meta_in: [{ input: 0, keys: { spacing_mm: 'spacing_mm', affine: 'affine' } }], outs: [{ output: 0 }, { message: true }] };
        },
        onRunResult(_o, j) { this._info = meshText(j); },
    });
    Object.assign(MeshSmoothNode.prototype, { async plan() { needAll(this, 1); return { cmd: 'mesh_smooth', options: { iterations: this.properties.iterations ?? 10 }, name: 'mesh' }; } });
    Object.assign(MeshLargestNode.prototype, { async plan() { needAll(this, 1); return { cmd: 'mesh_largest', name: 'mesh' }; } });
    Object.assign(MeshSampleNode.prototype, {
        async plan() {
            needAll(this, 2);
            return { cmd: 'mesh_sample', options: { name: (this.properties.name ?? 'value').trim() || 'value' }, name: 'mesh', meta_in: [{ input: 1, keys: { spacing_mm: 'spacing_mm', affine: 'affine' } }] };
        },
    });
    Object.assign(MeshInfoNode.prototype, {
        async plan() {
            needAll(this, 1);
            return { cmd: 'mesh_info', outs: [{ input: 0 }, { message: true }] };
        },
        onRunResult(_o, j) {
            if (!j) return;
            const sc = Object.keys(j.scalars ?? {});
            this._info = meshText(j) + `, ${j.components} part(s)${j.boundary_edges ? ', open edges ' + j.boundary_edges : ', closed'}${sc.length ? ', scalars: ' + sc.join(',') : ''}`;
        },
    });
    Object.assign(MeshViewerNode.prototype, {
        async plan() { return { cmd: '_pass' }; },
        onRunResult(outs) { this._ids = (outs ?? []).filter(isId); },
    });
    Object.assign(MeshSaveNode.prototype, {
        async plan() { return { cmd: '_pass' }; },
        onRunResult(outs) { this._uuid = outs?.[0] ?? null; },
    });
    Object.assign(MeshLoadNode.prototype, {
        async plan() {
            if (this._pendingFile) return { client: true };
            return { cmd: '_const', options: { value: this._uuid ?? null }, inputs: [] };
        },
    });


    class MeshDecimateNode extends HNode {
        static title    = "Mesh Decimate";
        static category = "Geometry";
        static infoLine = true;
        constructor() {
            super();
            this.addInput("mesh", "mesh_uuid");
            this.addOutput("mesh", "mesh_uuid");
            this.addWidget("number", "keep ratio", 0.25, v => { this.properties.ratio = v; }, { min: 0.01, max: 1, step: 0.05, precision: 2 });
            this.addWidget("number", "or target triangles (0: ratio)", 0, v => { this.properties.target = Math.round(v); }, { min: 0, step: 1000, precision: 0 });
            this.addWidget("toggle", "keep scalar extremes", true, v => { this.properties.preserve = v; });
            this.addWidget("number", "scalar edge weight", 0.5, v => { this.properties.scalar_weight = v; }, { min: 0, step: 5, precision: 2 });
            this.size = [270, 140];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class MeshCutNode extends HNode {
        static title    = "Mesh Cut";
        static category = "Geometry";
        constructor() {
            super();
            this.addInput("mesh", "mesh_uuid");
            this.addOutput("mesh", "mesh_uuid");
            this.addWidget("text", "origin x,y,z mm (empty: centre)", "", v => { this.properties.origin = v; });
            this.addWidget("text", "normal x,y,z", "0,0,1", v => { this.properties.normal = v; });
            this.addWidget("combo", "keep side", "positive", v => { this.properties.keep = v; }, { values: ["positive", "negative"] });
            this.addWidget("toggle", "cap the cut", false, v => { this.properties.cap = v; });
            this.size = [280, 130];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /** centerlines of a vessel tree mask (TEASAR): a line set with the lumen radius per point */
    class CenterlineNode extends HNode {
        static title    = "Centerline";
        static category = "Geometry";
        static infoLine = true;
        constructor() {
            super();
            this.addInput("mask", "tensor_uuid");
            this.addOutput("lines", "lines_uuid");
            this.addOutput("info", "centerline_info");   // branch table: [{id, parent, length_mm, mean / min / max / tip radius, ...}]
            this.addWidget("text", "label (empty: threshold)", "", v => { this.properties.label = v; });
            this.addWidget("number", "threshold", 0.5, v => { this.properties.threshold = v; }, { step: 0.5, precision: 3 });
            this.addWidget("number", "min branch (mm)", 4, v => { this.properties.min_length = v; }, { min: 0, step: 10, precision: 1 });
            this.addWidget("combo", "coordinates", "mm", v => { this.properties.coords = v; }, { values: ["mm", "voxel"] });   // voxel: what the flow viewer's centerlines use
            this.size = [270, 140];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /** one surface per frame of a 4-D mask (moving walls) */
    class MeshSeqNode extends HNode {
        static title    = "Mesh Sequence";
        static category = "Geometry";
        static infoLine = true;
        constructor() {
            super();
            this.addInput("mask 4D", "tensor_uuid");
            this.addOutput("meshes", "meshseq_uuid");
            this.addWidget("text", "label (empty: threshold)", "", v => { this.properties.label = v; });
            this.addWidget("number", "threshold", 0.5, v => { this.properties.threshold = v; }, { step: 0.5, precision: 3 });
            this.addWidget("number", "dt (s per frame)", 0.04, v => { this.properties.dt = v; }, { min: 0.001, step: 0.1, precision: 4 });
            this.addWidget("toggle", "coherent (one topology)", true, v => { this.properties.coherent = v; });
            this.addWidget("number", "blur passes", 1, v => { this.properties.blur = Math.round(v); }, { min: 0, max: 6, step: 10, precision: 0 });
            this.addWidget("number", "smooth iterations", 6, v => { this.properties.smooth = Math.round(v); }, { min: 0, max: 100, step: 10, precision: 0 });
            this.size = [270, 190];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    const maskSelect = (p) => { const lab = parseFloat(p.label); return Number.isNaN(lab) ? { threshold: p.threshold ?? 0.5 } : { label: lab }; };
    Object.assign(MeshDecimateNode.prototype, {
        async plan() {
            needAll(this, 1);
            const t = this.properties.target ?? 0;
            return { cmd: 'mesh_decimate', options: { ...(t > 0 ? { target_faces: t } : { ratio: this.properties.ratio ?? 0.25 }), preserve_scalars: this.properties.preserve ?? true, scalar_weight: this.properties.scalar_weight ?? 0.5 }, name: 'mesh', outs: [{ output: 0 }, { message: true }] };
        },
        onRunResult(_o, j) { if (j?.faces_after != null) this._info = `${j.faces_before} -> ${j.faces_after} triangles`; },
    });
    Object.assign(MeshCutNode.prototype, {
        async plan() {
            needAll(this, 1);
            const options = { normal: vec3(this.properties.normal ?? '0,0,1', 'normal'), keep: this.properties.keep ?? 'positive', cap: !!this.properties.cap };
            if ((this.properties.origin ?? '').trim()) options.origin = vec3(this.properties.origin, 'origin');
            return { cmd: 'mesh_cut', options, name: 'mesh' };
        },
    });
    Object.assign(CenterlineNode.prototype, {
        async plan() {
            needAll(this, 1);
            const options = { ...maskSelect(this.properties), min_length_mm: this.properties.min_length ?? 4 };
            if (this.properties.coords === 'voxel') options.coordinates = 'voxel';
            return { cmd: 'centerline_from_mask', options, name: 'centerline', meta_in: [{ input: 0, keys: { spacing_mm: 'spacing_mm', affine: 'affine' } }],
                     meta_out: [], outs: [{ output: 0 }, { message: true }] };
        },
        onRunResult(_o, j) { if (j?.branches != null) this._info = `${j.branches} branches, ${j.length_mm.toFixed(0)} mm, max radius ${j.max_radius_mm.toFixed(1)} mm`; },
    });
    Object.assign(MeshSeqNode.prototype, {
        async plan() {
            needAll(this, 1);
            const p = this.properties;
            return { cmd: (p.coherent ?? true) ? 'meshseq_track' : 'meshseq_from_mask', options: { ...maskSelect(p), dt: p.dt ?? 0.04, blur: p.blur ?? 1, smooth: p.smooth ?? 6 }, name: 'mesh sequence',
                     meta_in: [{ input: 0, keys: { spacing_mm: 'spacing_mm', affine: 'affine' } }], outs: [{ output: 0 }, { message: true }] };
        },
        onRunResult(_o, j) { if (j?.frames) this._info = `${j.frames} frames, area ${Math.min(...j.area_mm2).toFixed(0)}..${Math.max(...j.area_mm2).toFixed(0)} mm2`; },
    });


    class MeshFillHolesNode extends HNode {
        static title    = "Mesh Fill Holes";
        static category = "Geometry";
        static infoLine = true;
        constructor() {
            super();
            this.addInput("mesh", "mesh_uuid");
            this.addOutput("mesh", "mesh_uuid");
            this.addWidget("number", "max loop edges", 80, v => { this.properties.max_edges = Math.round(v); }, { min: 3, step: 100, precision: 0 });
            this.addWidget("toggle", "fair the patch", true, v => { this.properties.fair = v; });
            this.addWidget("toggle", "cotangent weights", true, v => { this.properties.cotan = v; });
            this.size = [250, 120];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /** union / intersection / difference of two closed meshes through a shared voxel grid (accurate to about one voxel) */
    class MeshBooleanNode extends HNode {
        static title    = "Mesh Boolean";
        static category = "Geometry";
        static infoLine = true;
        constructor() {
            super();
            this.addInput("A", "mesh_uuid");
            this.addInput("B", "mesh_uuid");
            this.addOutput("mesh", "mesh_uuid");
            this.addWidget("combo", "operation", "union", v => { this.properties.op = v; }, { values: ["union", "intersection", "difference"] });
            this.addWidget("number", "voxel mm (0: auto)", 0, v => { this.properties.voxel_mm = v; }, { min: 0, step: 1, precision: 3 });
            this.addWidget("toggle", "snap onto the inputs", true, v => { this.properties.refine = v; });
            this.addWidget("combo", "method", "voxel", v => { this.properties.method = v; }, { values: ["voxel", "exact"] });     // exact: corefinement, watertight for meshes in general position
            this.size = [250, 150];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /** inside / outside mask of a closed mesh (carries spacing and affine, so Mesh From Mask / Ortho Slicer / NIfTI Save line up) */
    class MeshVoxelizeNode extends HNode {
        static title    = "Mesh Voxelize";
        static category = "Geometry";
        static infoLine = true;
        constructor() {
            super();
            this.addInput("mesh", "mesh_uuid");
            this.addOutput("mask", "tensor_uuid");
            this.addWidget("number", "voxel mm (0: auto)", 0, v => { this.properties.voxel_mm = v; }, { min: 0, step: 1, precision: 3 });
            this.size = [250, 80];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }
    Object.assign(MeshFillHolesNode.prototype, {
        async plan() { needAll(this, 1); return { cmd: 'mesh_fill_holes', options: { max_edges: this.properties.max_edges ?? 80, fair: this.properties.fair ?? true, cotan: this.properties.cotan ?? true }, name: 'mesh', outs: [{ output: 0 }, { message: true }] }; },
        onRunResult(_o, j) { if (j?.filled != null) this._info = `filled ${j.filled}, ${j.remaining} larger loop(s) left open`; },
    });
    Object.assign(MeshBooleanNode.prototype, {
        async plan() {
            needAll(this, 2);
            const options = { op: this.properties.op ?? 'union', refine: this.properties.refine ?? true };
            if (this.properties.voxel_mm > 0) options.voxel_mm = this.properties.voxel_mm;
            if (this.properties.method === 'exact') return { cmd: 'mesh_csg', options: { op: options.op }, name: 'mesh', outs: [{ output: 0 }, { message: true }] };
            return { cmd: 'mesh_boolean', options, name: 'mesh', outs: [{ output: 0 }, { message: true }] };
        },
        onRunResult(_o, j) { if (j?.faces != null) this._info = `${j.op}${j.voxel_mm == null ? ' (exact)' : ''}: ${j.faces} triangles, ${j.crease_vertices != null ? j.crease_vertices + ' crease vertices, ' : ''}${(j.volume_mm3 / 1000).toFixed(2)} ml${j.voxel_mm != null ? ', voxel ' + j.voxel_mm.toFixed(2) + ' mm' : ''}`; },
    });
    Object.assign(MeshVoxelizeNode.prototype, {
        async plan() {
            needAll(this, 1);
            const options = {};
            if (this.properties.voxel_mm > 0) options.voxel_mm = this.properties.voxel_mm;
            return { cmd: 'mesh_voxelize', options, name: 'mesh mask', meta_out: [{ output: 0, from_message: { spacing_mm: 'spacing_mm', affine: 'affine' }, set: { axes: ['x', 'y', 'z'], source: 'mesh_voxelize' } }], outs: [{ output: 0 }, { message: true }] };
        },
        onRunResult(_o, j) { if (j?.voxels != null) this._info = `${j.dims.join('x')} voxels of ${j.spacing_mm[0].toFixed(2)} mm, ${j.voxels} inside`; },
    });


    /** cross-section area, tortuosity and bifurcation angles of a centerline, measured on the mask it came from */
    class CenterlineMeasureNode extends HNode {
        static title    = "Centerline Measure";
        static category = "Geometry";
        static infoLine = true;
        constructor() {
            super();
            this.addInput("lines", "lines_uuid");
            this.addInput("mask", "tensor_uuid");
            this.addOutput("lines", "lines_uuid");
            this.addOutput("info", "centerline_info");    // {branches: [...tortuosity, diameters...], junctions: [...angles]}
            this.addWidget("text", "label (empty: threshold)", "", v => { this.properties.label = v; });
            this.addWidget("number", "threshold", 0.5, v => { this.properties.threshold = v; }, { step: 0.5, precision: 3 });
            this.size = [270, 130];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }
    Object.assign(CenterlineMeasureNode.prototype, {
        async plan() {
            needAll(this, 2);
            return { cmd: 'centerline_measure', options: maskSelect(this.properties), name: 'measured centerline',
                     meta_in: [{ input: 1, keys: { spacing_mm: 'spacing_mm', affine: 'affine' } }], outs: [{ output: 0 }, { message: true }] };
        },
        onRunResult(_o, j) {
            if (!j?.branches) return;
            const b = j.branches, sg = j.segments ?? [], a = (j.junctions ?? [])[0];
            this._info = `${b.length} branches, ⌀ ${Math.min(...b.map(x => x.min_eq_diameter_mm)).toFixed(1)}..${Math.max(...b.map(x => x.max_eq_diameter_mm)).toFixed(1)} mm, tortuosity ${Math.max(1, ...sg.map(x => x.tortuosity)).toFixed(2)}${a ? ', bifurcation ' + a.angle_between_daughters_deg.toFixed(0) + '°' : ''}`;
        },
    });


    /** wall shear stress on a vessel mesh from a flow value (time-averaged, peak, oscillatory shear index as vertex scalars) */
    class MeshWssNode extends HNode {
        static title    = "Mesh Wall Shear";
        static category = "Geometry";
        static infoLine = true;
        constructor() {
            super();
            this.addInput("mesh", "mesh_uuid");
            this.addInput("flow", "flow_uuid");
            this.addOutput("mesh", "mesh_uuid");        // with scalars tawss, peak_wss, osi
            this.addOutput("info", "wss_info");
            this.addWidget("number", "viscosity (Pa s)", 0.0035, v => { this.properties.mu = v; }, { min: 0.001, step: 0.1, precision: 4 });
            this.addWidget("number", "depth mm (0: 1 voxel)", 0, v => { this.properties.depth_mm = v; }, { min: 0, step: 1, precision: 2 });
            this.addWidget("combo", "profile", "3 depths", v => { this.properties.profile = v; }, { values: ["3 depths", "quadratic", "linear"] });
            this.size = [280, 130];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    /** pulse wave velocity along a centerline branch from the flow waveforms at cross-sections */
    class CenterlinePwvNode extends HNode {
        static title    = "Pulse Wave Velocity";
        static category = "Geometry";
        static infoLine = true;
        constructor() {
            super();
            this.addInput("lines", "lines_uuid");
            this.addInput("flow", "flow_uuid");
            this.addOutput("lines", "lines_uuid");
            this.addOutput("info", "pwv_info");         // {pwv_m_s, r2, arclength_mm, delay_ms, flow_ml_s: waveforms}
            this.addWidget("number", "cross-sections", 8, v => { this.properties.points = Math.round(v); }, { min: 3, max: 64, step: 10, precision: 0 });
            this.addWidget("number", "branch (-1: longest)", -1, v => { this.properties.branch = Math.round(v); }, { min: -1, step: 10, precision: 0 });
            this.addWidget("combo", "method", "consensus", v => { this.properties.method = v; }, { values: ["consensus", "foot", "halfmax", "maxslope", "xcorr", "phase"] });
            this.addWidget("combo", "waveform", "flow", v => { this.properties.signal = v; }, { values: ["flow", "velocity"] });
            this.addWidget("toggle", "all branches", false, v => { this.properties.all = v; });
            this.addWidget("toggle", "separate forward / backward", true, v => { this.properties.separate = v; });
            this.addWidget("text", "lumen quantile (empty: disc)", "", v => { this.properties.quantile = v; });
            this.size = [290, 230];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }
    Object.assign(MeshWssNode.prototype, {
        async plan() {
            needAll(this, 2);
            const options = { mu: this.properties.mu ?? 0.0035, order: { 'linear': 1, 'quadratic': 2, '3 depths': 3 }[this.properties.profile ?? '3 depths'] };
            if (this.properties.depth_mm > 0) options.depth_mm = this.properties.depth_mm;
            return { cmd: 'mesh_wss', options, name: 'mesh with wall shear', outs: [{ output: 0 }, { message: true }] };
        },
        onRunResult(_o, j) { if (j?.mean_tawss_pa != null) this._info = `TAWSS ${j.mean_tawss_pa.toFixed(2)} Pa, peak ${j.mean_peak_pa.toFixed(2)} Pa, OSI ${j.mean_osi.toFixed(3)} (${j.valid_vertices}/${j.vertices} vertices)`; },
    });
    Object.assign(CenterlinePwvNode.prototype, {
        async plan() {
            needAll(this, 2);
            const options = { points: this.properties.points ?? 8, method: this.properties.method ?? 'consensus', signal: this.properties.signal ?? 'flow', all: !!this.properties.all, separate: this.properties.separate ?? true };
            if ((this.properties.branch ?? -1) >= 0) options.branch = this.properties.branch;
            const q = parseFloat(this.properties.quantile);
            if (!Number.isNaN(q)) options.quantile = q;
            return { cmd: 'centerline_pwv', options, name: 'centerline', outs: [{ output: 0 }, { message: true }] };
        },
        onRunResult(_o, j) { if (j?.pwv_m_s != null) this._info = `PWV ${j.pwv_m_s.toFixed(2)} m/s [${j.ci_m_s.map(x => x.toFixed(1)).join('..')}], r2 ${j.r2.toFixed(3)}, ${j.method}${j.separation?.reflection_index != null ? ', reflection ' + j.separation.reflection_index.toFixed(2) : ''}${j.warnings?.length ? ', ' + j.warnings.length + ' warning(s)' : ''}`; },
    });

    // Register all -------------------------------------------------------------

    for (const cls of [
        LoadTensorNode, SaveTensorNode, NiftiLoadNode, NiftiSaveNode,
        RandTensorNode, ZerosTensorNode, OnesTensorNode,
        AddNode, SubNode, MultNode, DivNode, NegNode, AbsNode, ScalarNode,
        FFTNode, IFFTNode, NUFFTNode,
        CoilSenseNode, SensitivityMapNode, ToeplitzNode,
        CompressUi16ConfigNode, OrthoSlicerNode, ModelRunNode,
        FlowLoadNode, FlowMakeNode, FlowInfoNode, PcmraNode, FlowPlaneNode, FlowViewerNode,
        MeshFromMaskNode, MeshSmoothNode, MeshLargestNode, MeshSampleNode, MeshInfoNode, MeshViewerNode, MeshLoadNode, MeshSaveNode,
        MeshDecimateNode, MeshCutNode, CenterlineNode, MeshSeqNode, MeshFillHolesNode, MeshBooleanNode, MeshVoxelizeNode, CenterlineMeasureNode, MeshWssNode, CenterlinePwvNode,
    ]) {
        LG.registerNodeType(cls.category + '/' + cls.title, cls);
    }
}

// -- Sidebar ------------------------------------------------------------------

let _lgCanvas;

function buildSidebar() {
    const container = document.getElementById("node-list");
    const groups = {};
    for (const [type, cls] of Object.entries(LG.registered_node_types)) {
        const [group] = type.split("/");
        (groups[group] ??= []).push({ type, title: cls.title ?? type.split("/")[1] });
    }
    for (const [group, nodes] of Object.entries(groups)) {
        const label = document.createElement("div");
        label.className   = "node-group-label";
        label.textContent = group;
        container.appendChild(label);
        for (const { type, title } of nodes) {
            const entry       = document.createElement("div");
            entry.className   = "node-entry";
            entry.textContent = title;
            entry.title       = type;
            entry.draggable   = true;
            entry.addEventListener("dragstart", e => {
                e.dataTransfer.setData("nodetype", type);
            });
            entry.addEventListener("dblclick", () => {
                const node = LG.createNode(type);
                const rect = _lgCanvas.canvas.getBoundingClientRect();
                node.pos = _lgCanvas.convertOffsetToCanvas(
                    [rect.width / 2, rect.height / 2]);
                _lgCanvas.graph.add(node);
            });
            container.appendChild(entry);
        }
    }
}

// -- Drop handler -------------------------------------------------------------

function setupDrop(lgCanvas) {
    const el = lgCanvas.canvas;
    el.addEventListener("dragover", e => e.preventDefault());
    el.addEventListener("drop", e => {
        e.preventDefault();
        const type = e.dataTransfer.getData("nodetype");
        if (!type) return;
        const rect = el.getBoundingClientRect();
        const pos  = lgCanvas.convertOffsetToCanvas(
            [e.clientX - rect.left, e.clientY - rect.top]);
        const node = LG.createNode(type);
        node.pos = pos;
        lgCanvas.graph.add(node);
    });
}

// -- Toolbar ------------------------------------------------------------------

/** the graph on the server: id comes from ?graph=, the first Save / Run creates one */
const doc = { id: new URLSearchParams(location.search).get('graph'), lastJson: '' };

async function saveGraph(graph, quiet = false) {
    const status = document.getElementById('save-status');
    const json = JSON.stringify(graph.serialize());
    const name = document.getElementById('graph-name').value.trim() || 'untitled';
    try {
        const r = await call('graph_save', { id: doc.id ?? undefined, name, graph: JSON.parse(json) });
        doc.id = r.id;
        doc.lastJson = json + '|' + name;
        const u = new URL(location.href);
        if (u.searchParams.get('graph') !== r.id) { u.searchParams.set('graph', r.id); history.replaceState(null, '', u); }
        status.textContent = 'saved ' + new Date().toLocaleTimeString();
    } catch (e) {
        status.textContent = 'save failed: ' + String(e.message ?? e).slice(0, 50);
        if (!quiet) alert('Save failed: ' + e.message);
    }
}

function setupToolbar(graph, lgCanvas) {
    const btnRun = document.getElementById("btn-run");
    const banner = document.getElementById('run-banner'), bText = document.getElementById('run-text');
    const bCont = document.getElementById('btn-continue'), bCancel = document.getElementById('btn-cancel');
    const ctl = new RunController(graph, lgCanvas, {
        banner(text, b) {
            banner.hidden = !text;
            bText.textContent = text ?? '';
            bCont.hidden = !b?.continue;
            bCancel.hidden = !b?.cancel;
        },
        onStatus(st) {
            const active = st === 'running' || st === 'waiting_client';
            btnRun.disabled = active;
            btnRun.textContent = active ? 'Running...' : '\u25B6 Run';
        },
    });
    window._runCtl = ctl;
    bCont.addEventListener('click', () => ctl.proceed());
    bCancel.addEventListener('click', () => ctl.cancel());

    btnRun.addEventListener("click", async () => {
        const { plan, errors } = await compilePlan(graph);
        for (const n of graph._nodes) { if (n._execStatus === 'error' && n._planError) { n._execStatus = null; n._planError = false; } }
        if (errors.length) {
            for (const { node, error } of errors) { node._execStatus = 'error'; node._execError = error; node._planError = true; node.boxcolor = '#cc2200'; }
            lgCanvas.setDirty(true, true);
            return;
        }
        if (!plan.length) return;
        await saveGraph(graph, true);   // the run is listed under this graph on the home page
        try {
            const { run: runId } = await call('run_submit', { graph: doc.id ?? undefined, name: document.getElementById('graph-name').value, plan });
            await ctl.attach(runId, { auto: true });
        } catch (e) { alert('Run failed: ' + e.message); }
    });

    document.getElementById("btn-clear").addEventListener("click", () => {
        if (confirm("Clear graph?")) graph.clear();
    });

    document.getElementById("btn-save").addEventListener("click", () => saveGraph(graph));

    document.getElementById("btn-export").addEventListener("click", () => {
        const json = JSON.stringify(graph.serialize(), null, 2);
        const a    = document.createElement("a");
        a.href     = URL.createObjectURL(new Blob([json], { type: "application/json" }));
        a.download = (document.getElementById('graph-name').value || 'graph') + ".json";
        a.click();
    });

    document.getElementById("btn-load").addEventListener("click", () => {
        const inp    = document.createElement("input");
        inp.type     = "file";
        inp.accept   = ".json";
        inp.onchange = async () => {
            const text = await inp.files[0].text();
            graph.configure(JSON.parse(text));
        };
        inp.click();
    });

    // autosave: once the graph has an id, changes are written to the server every few seconds
    setInterval(() => {
        if (!doc.id) return;
        const name = document.getElementById('graph-name').value.trim() || 'untitled';
        if (JSON.stringify(graph.serialize()) + '|' + name !== doc.lastJson) saveGraph(graph, true);
    }, 3000);

    const btnBankMeta = document.getElementById("btn-bank-meta");
    btnBankMeta.addEventListener("click", async () => {
        btnBankMeta.disabled    = true;
        btnBankMeta.textContent = 'Loading…';
        try {
            const entries = await client.list();
            showBankModal(entries);
        } catch (e) {
            alert('Bank meta error: ' + e.message);
        } finally {
            btnBankMeta.disabled    = false;
            btnBankMeta.textContent = 'Bank Meta';
        }
    });

    const btnBankClear = document.getElementById("btn-bank-clear");
    btnBankClear.addEventListener("click", async () => {
        if (!confirm('Delete ALL values from the server bank? This cannot be undone.')) return;
        btnBankClear.disabled    = true;
        btnBankClear.textContent = 'Clearing…';
        try {
            const entries = await client.list();
            await Promise.all(entries.map(e => client.delete(e.id)));
        } catch (e) {
            alert('Clear bank error: ' + e.message);
        } finally {
            btnBankClear.disabled    = false;
            btnBankClear.textContent = 'Clear Bank';
        }
    });
}

function showBankModal(entries) {
    document.getElementById('bank-modal-overlay')?.remove();

    const overlay = document.createElement('div');
    overlay.id = 'bank-modal-overlay';
    Object.assign(overlay.style, {
        position: 'fixed', inset: '0', background: 'rgba(0,0,0,0.65)',
        zIndex: '1000', display: 'flex', alignItems: 'center', justifyContent: 'center',
    });

    const box = document.createElement('div');
    Object.assign(box.style, {
        background: '#0e1218', border: '1px solid #2a3548', borderRadius: '6px',
        padding: '1rem', maxWidth: '80vw', maxHeight: '80vh',
        display: 'flex', flexDirection: 'column', gap: '0.5rem',
        minWidth: '520px',
    });

    const hdr = document.createElement('div');
    Object.assign(hdr.style, { display: 'flex', justifyContent: 'space-between', alignItems: 'center' });
    const ttl = document.createElement('span');
    Object.assign(ttl.style, { color: '#9ab8d0', fontSize: '0.85rem', fontWeight: '600' });
    ttl.textContent = `Bank Contents  (${entries.length} value${entries.length !== 1 ? 's' : ''})`;
    const closeBtn = document.createElement('button');
    closeBtn.textContent = '×';
    Object.assign(closeBtn.style, {
        background: 'none', border: 'none', color: '#7090b0',
        fontSize: '1.2rem', cursor: 'pointer', lineHeight: '1',
    });
    closeBtn.onclick = () => overlay.remove();
    hdr.appendChild(ttl);
    hdr.appendChild(closeBtn);

    const pre = document.createElement('pre');
    Object.assign(pre.style, {
        overflow: 'auto', color: '#a0c8e0', fontSize: '0.72rem',
        fontFamily: 'monospace', background: '#080a0e',
        padding: '0.7rem', borderRadius: '4px', maxHeight: '65vh',
    });
    pre.textContent = JSON.stringify(entries, null, 2);

    box.appendChild(hdr);
    box.appendChild(pre);
    overlay.appendChild(box);
    overlay.addEventListener('click', e => { if (e.target === overlay) overlay.remove(); });
    document.body.appendChild(overlay);
}

// -- Init ---------------------------------------------------------------------

async function init() {
    registerNodes();

    const graph    = new LG.LGraph();
    const canvasEl = document.getElementById("graph-canvas");

    function resize() {
        const main      = canvasEl.parentElement;
        canvasEl.width  = main.clientWidth;
        canvasEl.height = main.clientHeight;
    }
    resize();
    window.addEventListener("resize", () => { resize(); _lgCanvas?.setDirty(true, true); });

    _lgCanvas = new LG.LGraphCanvas(canvasEl, graph);
    _lgCanvas.background_image     = null;
    _lgCanvas.render_shadows        = false;
    _lgCanvas.render_canvas_border  = false;
    _lgCanvas.show_info             = false;   // the T/I/N/V/FPS debug overlay
    _lgCanvas.node_title_color      = "#9ab8d0";
    _lgCanvas.default_connection_color.input_on  = "#5a9abc";
    _lgCanvas.default_connection_color.output_on = "#5a9abc";

    buildSidebar();
    setupDrop(_lgCanvas);
    setupToolbar(graph, _lgCanvas);

    graph.start();

    // open a saved graph (?graph=<id>) and re-attach to its latest run, if the server still has one
    let opened = false;
    if (doc.id) {
        try {
            const d = await call('graph_get', { id: doc.id });
            graph.configure(d.graph);
            document.getElementById('graph-name').value = d.name;
            doc.lastJson = JSON.stringify(graph.serialize()) + '|' + d.name;
            document.getElementById('save-status').textContent = 'opened ' + d.id;
            opened = true;
            const last = await window._runCtl.latestFor(doc.id);
            if (last) await window._runCtl.attach(last.run);
        } catch (e) {
            document.getElementById('save-status').textContent = 'could not open graph: ' + String(e.message ?? e).slice(0, 60);
            doc.id = null;
        }
    }

    // Starter note
    const note = opened ? null : LG.createNode("Note");
    if (note) {
        note.pos = [60, 60];
        note.properties.text =
            "Drag nodes from the sidebar or double-click them.\n" +
            "Connect outputs to inputs, then press Run\n" +
            "to execute on the hasty server.";
        graph.add(note);
    }
    const q = new URLSearchParams(location.search);
    if (q.get('selftest') === '1') selfTest(graph, _lgCanvas);
    if (['1', '2', '3'].includes(q.get('runtest'))) {
        const out = document.createElement('pre'); out.id = 'selftest'; document.body.appendChild(out);
        const dev = await import('./devtools.js');
        try {
            out.textContent = JSON.stringify(q.get('runtest') === '1'
                ? await dev.runtest1(graph, _lgCanvas, { save: (g) => saveGraph(g, true), compile: compilePlan, call, doc, RunController })
                : q.get('runtest') === '3' ? await dev.runtest3(graph, _lgCanvas, { compile: compilePlan, call, doc, RunController })
                : await dev.runtest2(graph, _lgCanvas));
        } catch (e) { out.textContent = JSON.stringify({ ok: false, errors: [String(e.stack ?? e)] }); }
    }
    if (q.get('meshtest') === '1') {
        const out = document.createElement('pre'); out.id = 'selftest'; document.body.appendChild(out);
        const dev = await import('./devtools.js');
        try { out.textContent = JSON.stringify(await dev.meshtest(graph, _lgCanvas, { compile: compilePlan, call, RunController }, client, HTensor)); } catch (e) { out.textContent = JSON.stringify({ ok: false, errors: [String(e.stack ?? e)] }); }
    }
    if (q.get('geomtest') === '1') {
        const out = document.createElement('pre'); out.id = 'selftest'; document.body.appendChild(out);
        const dev = await import('./devtools.js');
        try { out.textContent = JSON.stringify(await dev.geomtest(graph, _lgCanvas, { compile: compilePlan, call, RunController }, client, HTensor, parseMesh)); } catch (e) { out.textContent = JSON.stringify({ ok: false, errors: [String(e.stack ?? e)] }); }
    }
    if (q.get('geomtest') === '2') {
        const out = document.createElement('pre'); out.id = 'selftest'; document.body.appendChild(out);
        const dev = await import('./devtools.js');
        try { out.textContent = JSON.stringify(await dev.geomtest2(graph, _lgCanvas, { compile: compilePlan, call, RunController }, client, HTensor)); } catch (e) { out.textContent = JSON.stringify({ ok: false, errors: [String(e.stack ?? e)] }); }
    }
    if (q.get('geomtest') === '3') {
        const out = document.createElement('pre'); out.id = 'selftest'; document.body.appendChild(out);
        const dev = await import('./devtools.js');
        try { out.textContent = JSON.stringify(await dev.geomtest3(graph, _lgCanvas, { compile: compilePlan, call, RunController }, client, HTensor)); } catch (e) { out.textContent = JSON.stringify({ ok: false, errors: [String(e.stack ?? e)] }); }
    }
    if (q.get('flowtest') === '1') {
        const out = document.createElement('pre'); out.id = 'selftest'; document.body.appendChild(out);
        const dev = await import('./devtools.js');
        try { out.textContent = JSON.stringify(await dev.flowtest(graph, _lgCanvas, runGraph, client)); } catch (e) { out.textContent = JSON.stringify({ ok: false, errors: [String(e.stack ?? e)] }); }
    }
    if (q.get('gallery') === '1' || q.get('perf')) {
        const dev = await import('./devtools.js');
        const out = document.createElement('pre');
        out.id = 'selftest';
        out.style.display = 'none';
        document.body.appendChild(out);
        graph.stop();
        graph.clear();
        if (q.get('gallery') === '1') {
            const rows = dev.gallery(graph, _lgCanvas);
            if (q.get('states') === '1') dev.gallery_states(graph, _lgCanvas);
            out.textContent = JSON.stringify({ ok: true, rows });
        } else {
            out.textContent = JSON.stringify({ ok: true, ...(await dev.perf(graph, _lgCanvas, Number(q.get('perf')))) });
        }
    }
}

/**
 * ?selftest=1: build Rand → Add(·, Ones) → FFT(ortho) → IFFT(ortho) → Sub(·, sum) and Rand → Compress UI16 →
 * Ortho Slicer, run the graph, verify the round trip on the server, and report JSON in <pre id="selftest">
 * (used by the headless browser test).
 */
async function selfTest(graph, canvas) {
    const out = document.createElement('pre');
    out.id = 'selftest';
    document.body.appendChild(out);
    const report = (r) => { out.textContent = JSON.stringify(r); };
    window.addEventListener('error', (e) => report({ ok: false, errors: [`uncaught: ${e.message}`] }));
    window.addEventListener('unhandledrejection', (e) => report({ ok: false, errors: [`unhandled: ${e.reason}`] }));
    report({ stage: 'building graph' });
    try {
        const add = (type, props = {}) => { const n = LG.createNode(type); Object.assign(n.properties, props); graph.add(n); return n; };
        const rand = add('Create/Rand Tensor', { shape: '4,6,8', dtype: 'c32' });
        const ones = add('Create/Ones Tensor', { shape: '4,6,8', dtype: 'c32' });
        const sum = add('Math/Add');
        const fft = add('FFT/FFT', { norm: 'ortho', axes: '-1,-2' });
        const ifft = add('FFT/IFFT', { norm: 'ortho', axes: '-1,-2' });
        const diff = add('Math/Sub');
        const comp = add('Viz/Compress UI16', { focus: '0.9', right_mode: 'log', right_logc: '10' });
        const view = add('Viz/Ortho Slicer');
        rand.connect(0, sum, 0); ones.connect(0, sum, 1);
        sum.connect(0, fft, 0); fft.connect(0, ifft, 0);
        ifft.connect(0, diff, 0); sum.connect(0, diff, 1);
        rand.connect(0, comp, 0); comp.connect(0, view, 0); comp.connect(1, view, 1);
        report({ stage: 'running graph' });
        await runGraph(graph, canvas);
        report({ stage: 'checking results' });
        const nodes = [rand, ones, sum, fft, ifft, diff, comp, view];
        const errors = nodes.filter((n) => n._execStatus !== 'done').map((n) => `${n.title}: ${n._execError}`);
        const d = await client.get(diff._execResult[0]);
        const s = await client.get(sum._execResult[0]);
        let maxDiff = 0, maxSum = 0;
        for (let i = 0; i < d.data.length; i++) { maxDiff = Math.max(maxDiff, Math.abs(d.data[i])); maxSum = Math.max(maxSum, Math.abs(s.data[i])); }
        const q = await client.get(comp._execResult[0]);
        report({ ok: errors.length === 0 && maxDiff < 1e-5 * maxSum && q.dtype === 'uint16', errors, maxDiff, maxSum, comprep: comp._execResult[1], viewerId: view._uuid, meta: sum._execMeta });
    } catch (e) {
        report({ ok: false, errors: [String(e.stack ?? e)] });
    }
}

init();
