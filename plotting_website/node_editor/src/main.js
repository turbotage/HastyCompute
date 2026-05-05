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
import { execute, executeWithMsg, fetchMeta, fetchMetaString,
         fetchSliceData, fetchRaw, uploadValue,
         bankQuery, deleteValue, fetchGVMetadata,
         readMetadata, writeMetadata, deleteMetadata,
         compressUi16Default, decompressUi16,
         initCommandIds, uuidToHex,
         hexToUuid, COMMAND_IDS }                 from './grpc_client.js';

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

/**
 * Draw an execution status bar in the bottom 22px of the node body.
 * Call from onDrawForeground (coordinate system: node body, top-left origin).
 */
function drawStatusBar(ctx, node) {
    if (!node._execStatus) return;
    const w    = node.size[0];
    const h    = node.size[1];
    const barH = 22;
    const y    = h - barH;

    const BG = { running: '#2a1800', done: '#001810', error: '#1a0006' };
    const FG = { running: '#cc8800', done:  '#44cc88', error:  '#ee4444' };
    ctx.fillStyle = BG[node._execStatus] ?? '#111';
    ctx.fillRect(3, y + 2, w - 6, barH - 4);

    ctx.fillStyle    = FG[node._execStatus] ?? '#888';
    ctx.font         = '10px monospace';
    ctx.textBaseline = 'middle';

    let text = '';
    if (node._execStatus === 'running') {
        text = 'running...';
    } else if (node._execStatus === 'done') {
        const uuid = node._execResult?.[0];
        if (uuid) {
            const hex  = uuidToHex(uuid).slice(0, 8);
            const meta = node._execMeta;
            text = meta
                ? hex + '... ' + meta.dtype + '[' + meta.shape.join('x') + ']'
                : hex + '...';
        } else {
            text = 'done';
        }
    } else if (node._execStatus === 'error') {
        text = 'ERR: ' + (node._execError ?? 'error').slice(0, 28);
    }

    ctx.fillText(text, 6, y + 2 + (barH - 4) / 2);
}

/**
 * Build a tensor-creation options string for from_metadata_string().
 * Format: "Tensor[dtype=f32,device=cpu,shape=(4,4)]"
 */
function makeTensorOptions(dtype, device, shapeStr) {
    const s = shapeStr.trim().replace(/^\(|\)$/g, '').replace(/\s+/g, ',');
    return 'Tensor[dtype=' + dtype + ',device=' + device + ',shape=(' + s + ')]';
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
 * @returns {Promise<{shape: number[], dtype: string, data: Uint8Array}>}
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
        512: { dtype: 'i16', bytes: 2 }, // UINT16 → i16 (same bit pattern)
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

    return { shape, dtype: dtInfo.dtype, data };
}

/**
 * Build the HastyCompute GenericValue wire format for a tensor.
 *
 * Wire layout:
 *   [0]       tag           u8  = 1 (TENSOR)
 *   [1]       device_type   u8  = 0 (CPU)
 *   [2]       scalar_type   u8
 *   [3]       ndim          u8
 *   [4]       device_index  i8  = 0
 *   [5..8]    padding       4 bytes (struct alignment)
 *   [9..16]   total_elements  i64 LE
 *   [17..]    shape           ndim × i64 LE
 *   [17+ndim*8..] element bytes
 *
 * @param {number[]}  shape
 * @param {string}    dtype   hasty dtype string
 * @param {Uint8Array} data   raw element bytes (C-order, matching shape)
 * @returns {Uint8Array}
 */
function buildTensorWireFormat(shape, dtype, data) {
    const scalarType = DTYPE_SCALAR_TYPE[dtype];
    if (scalarType === undefined) throw new Error('Unknown dtype: ' + dtype);
    const ndim      = shape.length;
    const totalElem = shape.reduce((a, b) => a * b, 1);
    const hdrSize   = 1 + 16 + ndim * 8; // tag + SerializedTensorHeader + shape
    const out       = new Uint8Array(hdrSize + data.byteLength);
    const dv        = new DataView(out.buffer);
    out[0] = 1;           // TENSOR tag
    out[1] = 0;           // device_type = CPU
    out[2] = scalarType;
    out[3] = ndim;
    out[4] = 0xFF;        // device_index = -1 (CPU alias)
    // out[5..8] = 0 (padding, already zeroed)
    dv.setBigInt64(9,  BigInt(totalElem), true); // total_elements at raw[9]
    for (let i = 0; i < ndim; i++) {
        dv.setBigInt64(17 + i * 8, BigInt(shape[i]), true); // shape at raw[17+i*8]
    }
    out.set(data, hdrSize);
    return out;
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
function buildNifti1(shape, dtype, data) {
    const DTYPE_TO_NIFTI  = { u8:2, i8:256, i16:4, i32:8, f32:16, f64:64 };
    const DTYPE_TO_BITPIX = { u8:8, i8:8,   i16:16, i32:32, f32:32, f64:64 };
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
    // magic: "n+1\0"
    u8[344] = 110; u8[345] = 43; u8[346] = 49; u8[347] = 0;
    // extension block (4 bytes, all zero = no extensions)
    // data starts at voxOffset
    u8.set(data, voxOffset);
    return buf;
}

// -- Node definitions ---------------------------------------------------------

function registerNodes() {

    // IO -----------------------------------------------------------------------

    class LoadTensorNode extends LG.LGraphNode {
        static title    = "Load Tensor";
        static category = "IO";
        constructor() {
            super();
            this.addWidget("text", "uuid (hex)", "", v => { this.properties.uuid = v; });
            this.addOutput("tensor", "tensor_uuid");
            this.size = [240, 60];
        }
        async executeAsync(_inputs) {
            const hex = (this.properties.uuid ?? '').replace(/\s/g, '');
            if (hex.length !== 32) throw new Error('UUID must be 32 hex chars');
            const uuid = hexToUuid(hex);
            this._execMeta = await fetchMeta(uuid);
            return [uuid];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class SaveTensorNode extends LG.LGraphNode {
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

    class TensorCreationBase extends LG.LGraphNode {
        constructor(cmdId) {
            super();
            this._cmdId = cmdId;
            this.addWidget("combo", "dtype",  "f32", v => { this.properties.dtype  = v; },
                { values: ["f32","f64","c32","c64","i32","i64"] });
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
            return await execute(this._cmdId, [], opts);
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class RandTensorNode extends TensorCreationBase {
        static title = "Rand Tensor";  static category = "Create";
        constructor() { super(COMMAND_IDS.rand); }
    }
    class ZerosTensorNode extends TensorCreationBase {
        static title = "Zeros Tensor"; static category = "Create";
        constructor() { super(COMMAND_IDS.zeros); }
    }
    class OnesTensorNode extends TensorCreationBase {
        static title = "Ones Tensor";  static category = "Create";
        constructor() { super(COMMAND_IDS.ones); }
    }

    // Binary arithmetic --------------------------------------------------------

    class BinaryArithBase extends LG.LGraphNode {
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
            return await execute(this._cmdId, [a, b]);
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class AddNode  extends BinaryArithBase {
        static title = "Add";  static category = "Math";
        constructor() { super(COMMAND_IDS.add); }
    }
    class SubNode  extends BinaryArithBase {
        static title = "Sub";  static category = "Math";
        constructor() { super(COMMAND_IDS.sub); }
    }
    class MultNode extends BinaryArithBase {
        static title = "Mult"; static category = "Math";
        constructor() { super(COMMAND_IDS.mult); }
    }
    class DivNode  extends BinaryArithBase {
        static title = "Div";  static category = "Math";
        constructor() { super(COMMAND_IDS.div); }
    }

    // Unary arithmetic ---------------------------------------------------------

    class UnaryArithBase extends LG.LGraphNode {
        constructor(cmdId) {
            super();
            this._cmdId = cmdId;
            this.addInput("in",   "tensor_uuid");
            this.addOutput("out", "tensor_uuid");
            this.size = [140, 56];
        }
        async executeAsync([a]) {
            if (!a) throw new Error('Input not connected');
            return await execute(this._cmdId, [a]);
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class NegNode extends UnaryArithBase {
        static title = "Neg"; static category = "Math";
        constructor() { super(COMMAND_IDS.neg); }
    }
    class AbsNode extends UnaryArithBase {
        static title = "Abs"; static category = "Math";
        constructor() { super(COMMAND_IDS.abs); }
    }

    // Scalar (no backend) ------------------------------------------------------

    class ScalarNode extends LG.LGraphNode {
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

    class FFTNode extends LG.LGraphNode {
        static title = "FFT"; static category = "FFT";
        constructor() {
            super();
            this.addInput("in",      "tensor_uuid");
            this.addOutput("kspace", "tensor_uuid");
            this.addWidget("combo", "norm", "ortho", null,
                { values: ["none","ortho","forward","backward"] });
            this.size = [160, 70];
        }
    }
    class IFFTNode extends LG.LGraphNode {
        static title = "IFFT"; static category = "FFT";
        constructor() {
            super();
            this.addInput("kspace", "tensor_uuid");
            this.addOutput("out",   "tensor_uuid");
            this.addWidget("combo", "norm", "ortho", null,
                { values: ["none","ortho","forward","backward"] });
            this.size = [160, 70];
        }
    }
    class NUFFTNode extends LG.LGraphNode {
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

    class CoilSenseNode extends LG.LGraphNode {
        static title = "Coil Sense"; static category = "MRI";
        constructor() {
            super();
            this.addInput("image",      "tensor_uuid");
            this.addInput("smaps",      "tensor_uuid");
            this.addOutput("coil_imgs", "tensor_uuid");
            this.size = [170, 70];
        }
    }
    class SensitivityMapNode extends LG.LGraphNode {
        static title = "Sensitivity Maps"; static category = "MRI";
        constructor() {
            super();
            this.addInput("kspace", "tensor_uuid");
            this.addOutput("smaps", "tensor_uuid");
            this.size = [170, 56];
        }
    }
    class ToeplitzNode extends LG.LGraphNode {
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

    class CompressUi16ConfigNode extends LG.LGraphNode {
        static title = "Compress UI16"; static category = "Viz";
        constructor() {
            super();
            this.addInput("tensor", "tensor_uuid");
            this.addOutput("compressed", "tensor_uuid");
            this.addOutput("config",     "comprep_config");
            this.addWidget("text", "a",           "0.0",    v => { this.properties.a           = v; });
            this.addWidget("text", "b",           "1.0",    v => { this.properties.b           = v; });
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
            const result = await execute(COMMAND_IDS.compress_ui16_config, [uuid], JSON.stringify(cfg));
            return [result[0], cfg];
        }
        onDrawForeground(ctx) { drawStatusBar(ctx, this); }
    }

    class OrthoSlicerNode extends LG.LGraphNode {
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
                let url = '../orthoslicer/?uuid=' + uuidToHex(this._uuid);
                const cfg = this._effectiveConfigStr();
                if (cfg) url += '&config=' + encodeURIComponent(cfg);
                window.open(url, '_blank');
            });
            this.size = [220, 110];
        }
        _effectiveConfigStr() {
            if (this._wiredConfig) return JSON.stringify(this._wiredConfig);
            return this._configStr.trim();
        }
        async executeAsync([uuid, config]) {
            if (uuid && !config) {
                // fetchMetaString: metadata only, no tensor download, cleans temp
                const meta = await fetchMetaString(uuid);
                if (meta) {
                    const AUTOCOMPRESS = new Set(['f32','f64','f16','i8','u8','i16','i32','i64']);
                    if (AUTOCOMPRESS.has(meta.dtype)) {
                        // Convert everything to f32 first:
                        // - fetchSliceData only correctly reads f32 scalars
                        // - compress_ui16_config needs float arithmetic
                        // - min/max on non-f32 returns wrong dtype scalar
                        let workUuid  = uuid;
                        let f32TempId = null;
                        if (meta.dtype !== 'f32') {
                            const opts = makeTensorOptions('f32', meta.device, meta.shape.join(','));
                            const convIds = await execute(COMMAND_IDS.to, [uuid], opts);
                            if (convIds[0]) { workUuid = convIds[0]; f32TempId = convIds[0]; }
                        }

                        const [minIds, maxIds] = await Promise.all([
                            execute(COMMAND_IDS.min, [workUuid]),
                            execute(COMMAND_IDS.max, [workUuid]),
                        ]);
                        const [minRes, maxRes] = await Promise.all([
                            fetchSliceData(minIds[0], ''),
                            fetchSliceData(maxIds[0], ''),
                        ]);
                        [...minIds, ...maxIds].forEach(id => deleteValue(id).catch(() => {}));

                        const cfg = { a: minRes.data[0], b: maxRes.data[0] };
                        const compressed = await execute(COMMAND_IDS.compress_ui16_config, [workUuid], JSON.stringify(cfg));
                        if (f32TempId) deleteValue(f32TempId).catch(() => {});

                        this._uuid        = compressed[0];
                        this._wiredConfig = cfg;
                        return [this._uuid];
                    }
                }
            }
            if (uuid)   this._uuid        = uuid;
            if (config) this._wiredConfig = config;
            return this._uuid ? [this._uuid] : [];
        }
    }

    // NIfTI IO -----------------------------------------------------------------

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
    class NiftiLoadNode extends LG.LGraphNode {
        static title    = "NIfTI Load";
        static category = "IO";
        static _skipCache = true;

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

                const { shape, dtype, data } = await parseNifti(file);
                // Pad shape to 5D (T,E,Z,Y,X) with leading 1s.
                // RemoteVolume always slices with [t,e,z,:,:] — a 3D tensor
                // would get INVALID_ARGUMENT on the server for those 5 indices.
                // Raw bytes are C-order contiguous, so prepending 1s is safe.
                while (shape.length < 5) shape.unshift(1);
                const wireBytes = buildTensorWireFormat(shape, dtype, data);
                const uuid      = await uploadValue(wireBytes);
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

        onDrawForeground(ctx) {
            drawStatusBar(ctx, this);
            // Show the loaded filename under the status bar if available
            if (this._pendingFile) {
                const w = this.size[0];
                ctx.fillStyle = '#88aacc';
                ctx.font      = '10px monospace';
                ctx.textBaseline = 'top';
                ctx.fillText('pending: ' + this._pendingFile.name.slice(0, 24), 6, this.size[1] - 42);
            }
        }
    }

    /**
     * NiftiSaveNode — accepts a tensor UUID input and provides a "Download NIfTI"
     * button that fetches the tensor from the server and triggers a browser download
     * of a NIfTI-1 (.nii) file.
     *
     * f16 tensors are automatically converted to f32 (NIfTI-1 has no f16 type).
     */
    class NiftiSaveNode extends LG.LGraphNode {
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
            const raw  = await fetchRaw(this._uuid);
            // Parse wire format: tag(1) + header(16) + shape(ndim*8) + data
            if (raw[0] !== 1) throw new Error('Not a tensor GV');
            const ndim     = raw[3];
            const dv       = new DataView(raw.buffer, raw.byteOffset + 1);
            const dtype    = Object.entries(DTYPE_SCALAR_TYPE)
                .find(([, v]) => v === raw[2])?.[0];
            if (!dtype) throw new Error('Unknown scalar type: ' + raw[2]);
            const shape    = [];
            for (let i = 0; i < ndim; i++) {
                shape.push(Number(dv.getBigInt64(16 + i * 8, true)));
            }
            const dataStart = 1 + 16 + ndim * 8;
            let   data      = raw.subarray(dataStart);

            // NIfTI-1 has no f16; promote to f32
            let saveDtype = dtype;
            if (dtype === 'f16') {
                const u16 = new Uint16Array(data.buffer, data.byteOffset, data.byteLength / 2);
                data      = new Uint8Array(f16ArrayToF32(u16).buffer);
                saveDtype = 'f32';
            }

            const niftiBytes = buildNifti1(shape, saveDtype, data);
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

    // Register all -------------------------------------------------------------

    for (const cls of [
        LoadTensorNode, SaveTensorNode, NiftiLoadNode, NiftiSaveNode,
        RandTensorNode, ZerosTensorNode, OnesTensorNode,
        AddNode, SubNode, MultNode, DivNode, NegNode, AbsNode, ScalarNode,
        FFTNode, IFFTNode, NUFFTNode,
        CoilSenseNode, SensitivityMapNode, ToeplitzNode,
        CompressUi16ConfigNode, OrthoSlicerNode,
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

function setupToolbar(graph, lgCanvas) {
    const btnRun = document.getElementById("btn-run");

    btnRun.addEventListener("click", async () => {
        btnRun.disabled    = true;
        btnRun.textContent = 'Running...';
        try {
            await runGraph(graph, lgCanvas);
        } finally {
            btnRun.disabled    = false;
            btnRun.textContent = '\u25B6 Run';
        }
    });

    document.getElementById("btn-clear").addEventListener("click", () => {
        if (confirm("Clear graph?")) graph.clear();
    });

    document.getElementById("btn-save").addEventListener("click", () => {
        const json = JSON.stringify(graph.serialize(), null, 2);
        const a    = document.createElement("a");
        a.href     = URL.createObjectURL(new Blob([json], { type: "application/json" }));
        a.download = "graph.json";
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

    const btnBankMeta = document.getElementById("btn-bank-meta");
    btnBankMeta.addEventListener("click", async () => {
        btnBankMeta.disabled    = true;
        btnBankMeta.textContent = 'Loading…';
        try {
            const { success, error_msg, uuids } = await bankQuery(0);
            if (!success) { alert('BankQuery failed: ' + error_msg); return; }
            const entries = await Promise.all(uuids.map(async uuid => {
                const hex  = uuidToHex(uuid);
                const meta = await fetchGVMetadata(uuid);
                return { uuid: hex, meta };
            }));
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
            const { success, error_msg, uuids } = await bankQuery(0);
            if (!success) { alert('BankQuery failed: ' + error_msg); return; }
            await Promise.all(uuids.map(uuid => deleteValue(uuid)));
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
    await initCommandIds();
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
    _lgCanvas.node_title_color      = "#9ab8d0";
    _lgCanvas.default_connection_color.input_on  = "#5a9abc";
    _lgCanvas.default_connection_color.output_on = "#5a9abc";

    buildSidebar();
    setupDrop(_lgCanvas);
    setupToolbar(graph, _lgCanvas);

    graph.start();

    // Starter note
    const note = LG.createNode("Note");
    if (note) {
        note.pos = [60, 60];
        note.properties.text =
            "Drag nodes from the sidebar or double-click them.\n" +
            "Connect outputs to inputs, then press Run\n" +
            "to execute against the C++ backend.";
        graph.add(note);
    }
}

init();
