/**
 * grpc_client.js — gRPC-web client for HastyService.
 * Self-contained: zero external libraries. Works fully offline once loaded.
 *
 * Protobuf wire types used:
 *   0  VARINT  — int32, bool
 *   2  LEN     — string, bytes, embedded messages, repeated fields
 *
 * gRPC-web framing:
 *   [flags: u8][length: u32 big-endian][proto bytes]
 *   flags 0x00 = data frame, 0x80 = trailer frame
 */

const BASE_URL = window.location.origin;

// ── Protobuf encoding ────────────────────────────────────────────────────────

function encodeVarint(n) {
    n = n >>> 0;
    const out = [];
    while (n > 0x7F) { out.push((n & 0x7F) | 0x80); n >>>= 7; }
    out.push(n & 0x7F);
    return new Uint8Array(out);
}

function concatBytes(...arrays) {
    const total = arrays.reduce((s, a) => s + a.length, 0);
    const out = new Uint8Array(total);
    let off = 0;
    for (const a of arrays) { out.set(a, off); off += a.length; }
    return out;
}

const _ENC = new TextEncoder();

function fieldTag(fieldNum, wireType) { return encodeVarint((fieldNum << 3) | wireType); }
function encodeVarField(fn, v)  { return concatBytes(fieldTag(fn, 0), encodeVarint(v)); }
function encodeLenField(fn, b)  { return concatBytes(fieldTag(fn, 2), encodeVarint(b.length), b); }
function encodeStringField(fn, s) { return encodeLenField(fn, _ENC.encode(s)); }
function encodeMsgField(fn, b)  { return encodeLenField(fn, b); }

// ── Message encoders ─────────────────────────────────────────────────────────

/** Encode:  Uuid { bytes value = 1; } */
function encodeUuid(bytes16) { return encodeLenField(1, bytes16); }

/**
 * Encode:
 *   ExecuteCommandRequest {
 *     int32         function_id = 1;
 *     string        options     = 2;
 *     repeated Uuid input_ids   = 3;
 *   }
 */
function buildExecuteCommandRequest(functionId, options, inputUuids) {
    const parts = [encodeVarField(1, functionId)];
    if (options) parts.push(encodeStringField(2, options));
    for (const u of inputUuids) parts.push(encodeMsgField(3, encodeUuid(u)));
    return concatBytes(...parts);
}

/**
 * Encode:
 *   ReadRequest { Uuid id = 1; string slice_info = 2; }
 */
function buildReadRequest(uuid16, sliceInfo = '') {
    const parts = [encodeMsgField(1, encodeUuid(uuid16))];
    if (sliceInfo) parts.push(encodeStringField(2, sliceInfo));
    return concatBytes(...parts);
}

/**
 * Encode:
 *   MetadataReadRequest { Uuid id = 1; }
 * (same wire format as a bare Uuid message)
 */
function buildMetadataReadRequest(uuid16) { return encodeMsgField(1, encodeUuid(uuid16)); }

/**
 * Encode:
 *   MetadataWriteRequest { Uuid id = 1; string metadata_string = 2; }
 */
function buildMetadataWriteRequest(uuid16, metaStr) {
    return concatBytes(encodeMsgField(1, encodeUuid(uuid16)), encodeStringField(2, metaStr));
}

/**
 * Encode:
 *   MetadataDeleteRequest { Uuid id = 1; }
 */
function buildMetadataDeleteRequest(uuid16) { return encodeMsgField(1, encodeUuid(uuid16)); }

// ── Protobuf decoding ────────────────────────────────────────────────────────

/**
 * Decode a binary protobuf message.
 * Returns Map<fieldNum, Array<number|Uint8Array>>:
 *   wire type 0 (VARINT)  → number
 *   wire type 2 (LEN)     → Uint8Array
 */
function decodeProto(bytes) {
    const fields = new Map();
    let i = 0;
    while (i < bytes.length) {
        let tag = 0, shift = 0;
        while (i < bytes.length) {
            const b = bytes[i++];
            tag |= (b & 0x7F) << shift;
            if (!(b & 0x80)) break;
            shift += 7;
        }
        const fn = tag >>> 3, wt = tag & 0x7;
        const arr = fields.get(fn) ?? [];
        if (wt === 0) {
            let v = 0, s = 0;
            while (i < bytes.length) {
                const b = bytes[i++];
                v |= (b & 0x7F) << s;
                if (!(b & 0x80)) break;
                s += 7;
            }
            arr.push(v >>> 0);
        } else if (wt === 2) {
            let len = 0, s = 0;
            while (i < bytes.length) {
                const b = bytes[i++];
                len |= (b & 0x7F) << s;
                if (!(b & 0x80)) break;
                s += 7;
            }
            arr.push(bytes.slice(i, i + len));
            i += len;
        } else break;
        fields.set(fn, arr);
    }
    return fields;
}

const _DEC = new TextDecoder();

/**
 * Decode:
 *   ExecuteCommandResponse {
 *     bool           success    = 1;
 *     string         error_msg  = 2;
 *     string         msg        = 3;
 *     repeated Uuid  output_ids = 4;
 *   }
 */
function decodeExecuteCommandResponse(bytes) {
    const f = decodeProto(bytes);
    return {
        success:    !!(f.get(1)?.[0] ?? 0),
        error_msg:  f.has(2) ? _DEC.decode(f.get(2)[0]) : '',
        msg:        f.has(3) ? _DEC.decode(f.get(3)[0]) : '',
        output_ids: (f.get(4) ?? []).map(msgBytes => {
            const inner = decodeProto(msgBytes);
            return inner.get(1)?.[0]; // Uint8Array(16)
        }),
    };
}

/**
 * Decode a single ReadResponse frame.
 * Returns { type: 'header', success, msg } or { type: 'data', data: Uint8Array }
 *
 *   ReadResponse {
 *     oneof payload {
 *       ReadResponseHeader header = 1;  // { bool success=1; string msg=2; }
 *       DataChunk          data   = 2;  // { bytes data=1; }
 *     }
 *   }
 */
function decodeReadResponse(bytes) {
    const f = decodeProto(bytes);
    if (f.has(1)) {
        const hf = decodeProto(f.get(1)[0]);
        return { type: 'header', success: !!(hf.get(1)?.[0] ?? 0), msg: hf.has(2) ? _DEC.decode(hf.get(2)[0]) : '' };
    }
    if (f.has(2)) {
        const cf = decodeProto(f.get(2)[0]);
        return { type: 'data', data: cf.get(1)?.[0] ?? new Uint8Array(0) };
    }
    return null;
}

// ── gRPC-web framing ──────────────────────────────────────────────────────────

function makeGrpcFrame(protoBytes) {
    const n = protoBytes.length;
    const frame = new Uint8Array(5 + n);
    frame[0] = 0;
    frame[1] = (n >>> 24) & 0xFF;
    frame[2] = (n >>> 16) & 0xFF;
    frame[3] = (n >>>  8) & 0xFF;
    frame[4] =  n         & 0xFF;
    frame.set(protoBytes, 5);
    return frame;
}

function parseGrpcFrames(bytes) {
    const frames = [];
    let i = 0;
    while (i + 5 <= bytes.length) {
        if (bytes[i] & 0x80) break; // trailer frame — stop
        const len = (bytes[i+1] << 24) | (bytes[i+2] << 16) | (bytes[i+3] << 8) | bytes[i+4];
        i += 5;
        if (i + len > bytes.length) break;
        frames.push(bytes.slice(i, i + len));
        i += len;
    }
    return frames;
}

function parseGrpcTrailer(bytes) {
    let i = 0;
    while (i + 5 <= bytes.length) {
        const flags = bytes[i];
        const len   = (bytes[i+1] << 24) | (bytes[i+2] << 16) | (bytes[i+3] << 8) | bytes[i+4];
        i += 5;
        if (i + len > bytes.length) break;
        if (flags & 0x80) {
            const trailer = _DEC.decode(bytes.slice(i, i + len));
            const statusM = trailer.match(/grpc-status:(\d+)/);
            const msgM    = trailer.match(/grpc-message:([^\r\n]*)/);
            return {
                status: statusM ? parseInt(statusM[1], 10) : -1,
                message: msgM ? decodeURIComponent(msgM[1].trim()) : trailer.trim(),
            };
        }
        i += len;
    }
    return null;
}

// ── HTTP transport ────────────────────────────────────────────────────────────

async function grpcPost(method, protoBytes) {
    const resp = await fetch(`${BASE_URL}/${method}`, {
        method:  'POST',
        headers: {
            'Content-Type': 'application/grpc-web+proto',
            'x-grpc-web':   '1',
        },
        body: makeGrpcFrame(protoBytes),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status} calling /${method}`);
    const bytes  = new Uint8Array(await resp.arrayBuffer());
    const frames = parseGrpcFrames(bytes);
    const trailer = parseGrpcTrailer(bytes);
    if (trailer && trailer.status !== 0) {
        throw new Error(`gRPC error from ${method} (status ${trailer.status}): ${trailer.message}`);
    }
    return frames;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Execute a registered command on values in the bank.
 *
 * @param {number}       functionId
 * @param {Uint8Array[]} inputUuids
 * @param {string}       options
 * @returns {Promise<Uint8Array[]>}  Output UUID Uint8Arrays
 */
export async function execute(functionId, inputUuids = [], options = '') {
    const frames = await grpcPost(
        'hasty.HastyService/ExecuteCommand',
        buildExecuteCommandRequest(functionId, options, inputUuids));
    if (!frames.length) throw new Error('Empty ExecuteCommand response');
    const ack = decodeExecuteCommandResponse(frames[0]);
    if (!ack.success) throw new Error(ack.error_msg || 'ExecuteCommand failed');
    return ack.output_ids;
}

/**
 * Like execute() but also returns the msg string from the response.
 *
 * @returns {Promise<{output_ids: Uint8Array[], msg: string}>}
 */
export async function executeWithMsg(functionId, inputUuids = [], options = '') {
    const frames = await grpcPost(
        'hasty.HastyService/ExecuteCommand',
        buildExecuteCommandRequest(functionId, options, inputUuids));
    if (!frames.length) throw new Error('Empty ExecuteCommand response');
    const ack = decodeExecuteCommandResponse(frames[0]);
    if (!ack.success) throw new Error(ack.error_msg || 'ExecuteCommand failed');
    return { output_ids: ack.output_ids, msg: ack.msg };
}

/**
 * Compress tensor to UI16 using auto-selected config.
 * Returns the compressed UUID and the chosen config (parsed from response msg).
 *
 * @param {Uint8Array} uuid16
 * @returns {Promise<{compressedUuid: Uint8Array, config: object}>}
 */
export async function compressUi16Default(uuid16) {
    const { output_ids, msg } = await executeWithMsg(COMMAND_IDS.compress_ui16_default, [uuid16]);
    if (!output_ids.length) throw new Error('compress_ui16_default: no output');
    const config = msg ? JSON.parse(msg) : {};
    // Track this intermediate UUID so the frontend can free it when the
    // page/window is closed.
    try { trackCreatedUuid(output_ids[0]); } catch {}
    return { compressedUuid: output_ids[0], config };
}

/**
 * Decompress a UI16-compressed tensor back to float32.
 *
 * @param {Uint8Array} uuid16   - UUID of the compressed (i16) tensor
 * @param {object}     config   - Config object (from compress_ui16_default or manual)
 * @returns {Promise<Uint8Array>}  UUID of the decompressed f32 tensor
 */
export async function decompressUi16(uuid16, config) {
    const options = JSON.stringify(config);
    const output_ids = await execute(COMMAND_IDS.decompress_ui16_config, [uuid16], options);
    if (!output_ids.length) throw new Error('decompress_ui16_config: no output');
    return output_ids[0];
}

/**
 * Fetch the raw serialised GenericValue bytes for a UUID.
 * Reassembles all DataChunk frames into one contiguous buffer.
 *
 * @param {Uint8Array} uuid16
 * @returns {Promise<Uint8Array>}
 */
export async function fetchRaw(uuid16) {
    const frames = await grpcPost('hasty.HastyService/Read', buildReadRequest(uuid16));
    if (!frames.length) throw new Error('Empty Read response');
    // First frame is ReadResponseHeader
    const hdr = decodeReadResponse(frames[0]);
    if (!hdr || hdr.type !== 'header') throw new Error('Read: missing response header');
    if (!hdr.success) throw new Error('Read failed: ' + hdr.msg);
    // Remaining frames are data chunks
    const parts = frames.slice(1).map(f => {
        const r = decodeReadResponse(f);
        return (r && r.type === 'data') ? r.data : new Uint8Array(0);
    });
    const total = parts.reduce((s, a) => s + a.length, 0);
    const out   = new Uint8Array(total);
    let off = 0;
    for (const p of parts) { out.set(p, off); off += p.length; }
    return out;
}

/**
 * Fetch a float32 slice of a tensor using Python-style slice notation.
 *
 * @param {Uint8Array} uuid16
 * @param {string}     sliceInfo
 * @returns {Promise<{data: Float32Array|Uint16Array, dtype: string}>}
 */
export async function fetchSliceData(uuid16, sliceInfo) {
    const frames = await grpcPost('hasty.HastyService/Read', buildReadRequest(uuid16, sliceInfo));
    if (!frames.length) throw new Error('Empty Read response');
    const hdr = decodeReadResponse(frames[0]);
    if (!hdr || hdr.type !== 'header') throw new Error('Read: missing response header');
    if (!hdr.success) throw new Error('Read failed: ' + hdr.msg);
    const parts = frames.slice(1).map(f => {
        const r = decodeReadResponse(f);
        return (r && r.type === 'data') ? r.data : new Uint8Array(0);
    });
    const total = parts.reduce((s, a) => s + a.length, 0);
    const raw   = new Uint8Array(total);
    let off = 0;
    for (const p of parts) { raw.set(p, off); off += p.length; }
    if (raw.length < 17 || raw[0] !== 1) throw new Error('Response is not a tensor');
    const dv         = new DataView(raw.buffer, raw.byteOffset + 1);
    const scalarType = dv.getUint8(1);
    const dtype      = _SCALAR_DTYPE[scalarType] ?? `scalar(${scalarType})`;
    const ndim       = dv.getUint8(2);
    const totalElem  = Number(dv.getBigInt64(8, true));
    const dataOff    = 16 + ndim * 8;
    if (dtype === 'f16' || dtype === 'i16') {
        const byteBase = 1 + dataOff;
        const u16 = new Uint16Array(totalElem);
        new Uint8Array(u16.buffer).set(
            new Uint8Array(raw.buffer, raw.byteOffset + byteBase, totalElem * 2));
        return { data: u16, dtype };
    }
    // Boolean (b8) tensors are encoded as one byte per element. Convert to
    // Float32Array of 0.0/1.0 so the renderer can upload as float textures.
    if (dtype === 'b8') {
        const byteBase = 1 + dataOff;
        const u8view = new Uint8Array(raw.buffer, raw.byteOffset + byteBase, totalElem);
        const f32 = new Float32Array(totalElem);
        for (let i = 0; i < totalElem; ++i) f32[i] = u8view[i] ? 1.0 : 0.0;
        return { data: f32, dtype: 'f32' };
    }
    const f32 = new Float32Array(totalElem);
    if (dtype === 'f64') {
        for (let i = 0; i < totalElem; i++)
            f32[i] = dv.getFloat64(dataOff + i * 8, true);
    } else {
        for (let i = 0; i < totalElem; i++)
            f32[i] = dv.getFloat32(dataOff + i * 4, true);
    }
    return { data: f32, dtype: 'f32' };
}

// ATen scalar_type index → hasty dtype string
const _SCALAR_DTYPE = {
    0:'u8', 1:'i8', 2:'i16', 3:'i32', 4:'i64',
    5:'f16', 6:'f32', 7:'f64',
    9:'c32', 10:'c64', 11:'b8',
};

/**
 * Parse tensor metadata from raw GenericValue wire bytes.
 *
 * @param {Uint8Array} raw
 * @returns {{ device, dtype, shape } | null}
 */
export function parseTensorMeta(raw) {
    if (!raw || raw.length < 17 || raw[0] !== 1) return null;
    const dv  = new DataView(raw.buffer, raw.byteOffset + 1);
    const ndim = dv.getUint8(2);
    if (raw.length < 17 + ndim * 8) return null;
    const shape = [];
    for (let i = 0; i < ndim; i++) {
        shape.push(Number(dv.getBigInt64(16 + i * 8, true)));
    }
    return {
        device: dv.getUint8(0) === 1 ? 'cuda' : 'cpu',
        dtype:  _SCALAR_DTYPE[dv.getUint8(1)] ?? `scalar(${dv.getUint8(1)})`,
        shape,
    };
}

/**
 * Fetch and parse tensor metadata (dtype, shape, device) for a UUID.
 *
 * @param {Uint8Array} uuid16
 * @returns {Promise<{device, dtype, shape} | null>}
 */
export async function fetchMeta(uuid16) {
    try {
        return parseTensorMeta(await fetchRaw(uuid16));
    } catch {
        return null;
    }
}

/** Convert a 16-byte Uint8Array UUID to a lowercase 32-char hex string. */
export function uuidToHex(bytes16) {
    return Array.from(bytes16, b => b.toString(16).padStart(2, '0')).join('');
}

/** Parse a 32-char hex string into a 16-byte Uint8Array. */
export function hexToUuid(hex) {
    const out = new Uint8Array(16);
    for (let i = 0; i < 16; i++) out[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
    return out;
}

// Track created intermediate UUIDs in localStorage so we can attempt to
// delete them when the user closes the page. Stored as array of 32-char hex
// strings under key 'hasty_created_uuids'.
function _getTrackedHex() {
    try { return JSON.parse(localStorage.getItem('hasty_created_uuids') || '[]'); } catch { return []; }
}
function _saveTrackedHex(arr) {
    try { localStorage.setItem('hasty_created_uuids', JSON.stringify(arr)); } catch {}
}

export function trackCreatedUuid(uuid16) {
    const hex = uuidToHex(uuid16);
    const arr = _getTrackedHex();
    if (!arr.includes(hex)) { arr.push(hex); _saveTrackedHex(arr); }
}

export function untrackCreatedUuid(uuid16_or_hex) {
    const hex = (uuid16_or_hex instanceof Uint8Array) ? uuidToHex(uuid16_or_hex) : String(uuid16_or_hex);
    const arr = _getTrackedHex().filter(h => h !== hex);
    _saveTrackedHex(arr);
}

// Attempt best-effort cleanup on unload. We don't await deletions because
// unload handlers cannot reliably run async to completion; still this will
// trigger a request in most browsers.
window.addEventListener('unload', () => {
    const arr = _getTrackedHex();
    for (const hex of arr) {
        try { deleteValue(hexToUuid(hex)).catch(() => {}); } catch { /* ignore */ }
    }
    try { localStorage.removeItem('hasty_created_uuids'); } catch {}
});

function _parseRawString(raw) {
    if (!raw || raw.length < 9 || raw[0] !== 5) return null;
    const dv  = new DataView(raw.buffer, raw.byteOffset + 1);
    const len = Number(dv.getBigUint64(0, true));
    return _DEC.decode(raw.subarray(9, 9 + len));
}

/**
 * Execute get_gv_metadata_string on uuid16, read the result, delete the temp.
 *
 * @param {Uint8Array} uuid16
 * @returns {Promise<string|null>}
 */
export async function fetchGVMetadata(uuid16) {
    let tempId;
    try {
        const outIds = await execute(COMMAND_IDS.get_gv_metadata_string, [uuid16]);
        if (!outIds.length) return null;
        tempId = outIds[0];
        return _parseRawString(await fetchRaw(tempId));
    } catch {
        return null;
    } finally {
        if (tempId) deleteValue(tempId).catch(() => {});
    }
}

/**
 * Fetch tensor metadata (dtype, shape, device) for a UUID via metadata string command.
 *
 * @param {Uint8Array} uuid16
 * @returns {Promise<{device: string, dtype: string, shape: number[]} | null>}
 */
export async function fetchMetaString(uuid16) {
    try {
        const str = await fetchGVMetadata(uuid16);
        if (!str) return null;
        const dtypeM  = str.match(/dtype=([^,\]]+)/);
        const deviceM = str.match(/device=([^,\]]+)/);
        const shapeM  = str.match(/shape=\(([^)]+)\)/);
        if (!dtypeM || !deviceM || !shapeM) return null;
        return { dtype: dtypeM[1], device: deviceM[1], shape: shapeM[1].split(',').map(Number) };
    } catch {
        return null;
    }
}

/**
 * Fetch all registered command IDs from the server and populate COMMAND_IDS.
 *
 * @returns {Promise<void>}
 */
export async function initCommandIds() {
    const outIds = await execute(0, []);
    const strings = await Promise.all(outIds.map(async id => {
        try { return _parseRawString(await fetchRaw(id)); } catch { return null; }
    }));
    await Promise.all(outIds.map(id => deleteValue(id).catch(() => {})));
    for (const s of strings) {
        if (!s) continue;
        const colon = s.indexOf(':');
        if (colon < 0) continue;
        COMMAND_IDS[s.slice(colon + 1)] = parseInt(s.slice(0, colon), 10);
    }
}

/**
 * Upload a serialized GenericValue to the server via the Push streaming RPC.
 *
 * @param {Uint8Array} gvBytes
 * @returns {Promise<Uint8Array>}  16-byte UUID Uint8Array
 */
export async function uploadValue(gvBytes) {
    const CHUNK = 1 * 1024 * 1024;
    const frames = [];
    for (let off = 0; off < gvBytes.length; off += CHUNK) {
        const slice = gvBytes.subarray(off, Math.min(off + CHUNK, gvBytes.length));
        frames.push(makeGrpcFrame(encodeLenField(1, slice)));
    }
    const body = concatBytes(...frames);
    const resp = await fetch(`${BASE_URL}/hasty.HastyService/Push`, {
        method:  'POST',
        headers: {
            'Content-Type': 'application/grpc-web+proto',
            'x-grpc-web':   '1',
        },
        body,
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status} calling Push`);
    const respBytes  = new Uint8Array(await resp.arrayBuffer());
    const respFrames = parseGrpcFrames(respBytes);
    if (!respFrames.length) {
        const trailer = parseGrpcTrailer(respBytes);
        const msg = trailer?.message || 'empty response';
        throw new Error(`Push gRPC error (status ${trailer?.status ?? '?'}): ${msg}`);
    }
    // PushResponse { bool success=1; string msg=2; Uuid id=3; }
    const f = decodeProto(respFrames[0]);
    const success = !!(f.get(1)?.[0] ?? 0);
    if (!success) {
        const errMsg = f.has(2) ? _DEC.decode(f.get(2)[0]) : 'unknown error';
        throw new Error('Push failed: ' + errMsg);
    }
    const uuidMsg  = f.get(3)?.[0];
    if (!uuidMsg) throw new Error('Push: missing UUID in response');
    const uuidBytes = decodeProto(uuidMsg).get(1)?.[0];
    if (!uuidBytes || uuidBytes.length !== 16) throw new Error('Push: invalid UUID response');
    return uuidBytes;
}

/**
 * Query the bank (unary).
 * queryType 0 = LIST_UUIDS → returns all UUIDs currently in the bank.
 *
 * @param {number} queryType
 * @returns {Promise<{success: boolean, error_msg: string, uuids: Uint8Array[]}>}
 */
export async function bankQuery(queryType = 0) {
    const reqBytes = encodeVarField(1, queryType);
    const frames   = await grpcPost('hasty.HastyService/BankQuery', reqBytes);
    if (!frames.length) return { success: false, error_msg: 'No response', uuids: [] };
    const f = decodeProto(frames[0]);
    return {
        success:   !!(f.get(1)?.[0] ?? 0),
        error_msg: f.has(2) ? _DEC.decode(f.get(2)[0]) : '',
        uuids:     (f.get(4) ?? []).map(b => decodeProto(b).get(1)?.[0]).filter(Boolean),
    };
}

/**
 * Delete a value from the bank by UUID.
 *
 * @param {Uint8Array} uuid16
 * @returns {Promise<{success: boolean, msg: string}>}
 */
export async function deleteValue(uuid16) {
    const frames = await grpcPost('hasty.HastyService/Delete', encodeUuid(uuid16));
    if (!frames.length) return { success: false, msg: 'No response' };
    const f = decodeProto(frames[0]);
    return {
        success: !!(f.get(1)?.[0] ?? 0),
        msg:     f.has(2) ? _DEC.decode(f.get(2)[0]) : '',
    };
}

/**
 * Read metadata string for a UUID from the bank's metadata map.
 *
 * @param {Uint8Array} uuid16
 * @returns {Promise<{success: boolean, msg: string, metadata_string: string}>}
 */
export async function readMetadata(uuid16) {
    const frames = await grpcPost('hasty.HastyService/ReadMetadata', buildMetadataReadRequest(uuid16));
    if (!frames.length) return { success: false, msg: 'No response', metadata_string: '' };
    const f = decodeProto(frames[0]);
    return {
        success:         !!(f.get(1)?.[0] ?? 0),
        msg:             f.has(2) ? _DEC.decode(f.get(2)[0]) : '',
        metadata_string: f.has(3) ? _DEC.decode(f.get(3)[0]) : '',
    };
}

/**
 * Write metadata string for a UUID into the bank's metadata map.
 *
 * @param {Uint8Array} uuid16
 * @param {string}     metaStr
 * @returns {Promise<{success: boolean, msg: string}>}
 */
export async function writeMetadata(uuid16, metaStr) {
    const frames = await grpcPost('hasty.HastyService/WriteMetadata', buildMetadataWriteRequest(uuid16, metaStr));
    if (!frames.length) return { success: false, msg: 'No response' };
    const f = decodeProto(frames[0]);
    return {
        success: !!(f.get(1)?.[0] ?? 0),
        msg:     f.has(2) ? _DEC.decode(f.get(2)[0]) : '',
    };
}

/**
 * Delete metadata for a UUID from the bank's metadata map.
 *
 * @param {Uint8Array} uuid16
 * @returns {Promise<{success: boolean, msg: string}>}
 */
export async function deleteMetadata(uuid16) {
    const frames = await grpcPost('hasty.HastyService/DeleteMetadata', buildMetadataDeleteRequest(uuid16));
    if (!frames.length) return { success: false, msg: 'No response' };
    const f = decodeProto(frames[0]);
    return {
        success: !!(f.get(1)?.[0] ?? 0),
        msg:     f.has(2) ? _DEC.decode(f.get(2)[0]) : '',
    };
}

/**
 * Integer command IDs — populated at runtime by initCommandIds().
 * Do not use until initCommandIds() has resolved.
 */
export const COMMAND_IDS = {};
