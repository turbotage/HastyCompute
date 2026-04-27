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

// Base URL for all gRPC-web calls.  The C++ HttpServer serves both the
// static website files and acts as a gRPC-web reverse proxy, so
// same-origin requests reach the gRPC server transparently.
const BASE_URL = window.location.origin;

// ── Protobuf encoding ────────────────────────────────────────────────────────

function encodeVarint(n) {
    n = n >>> 0; // treat as unsigned 32-bit integer
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
 *   ExecuteCommand {
 *     int32         function_id = 1;
 *     string        options     = 2;
 *     repeated Uuid input_ids   = 3;
 *   }
 */
function buildExecuteCommand(functionId, options, inputUuids) {
    const parts = [encodeVarField(1, functionId)];
    if (options) parts.push(encodeStringField(2, options));
    for (const u of inputUuids) parts.push(encodeMsgField(3, encodeUuid(u)));
    return concatBytes(...parts);
}

/**
 * Encode:
 *   FetchRequest { Uuid id = 1; string slice_info = 2; }
 */
function buildFetchRequest(uuid16, sliceInfo = '') {
    const parts = [encodeMsgField(1, encodeUuid(uuid16))];
    if (sliceInfo) parts.push(encodeStringField(2, sliceInfo));
    return concatBytes(...parts);
}

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
        } else break; // unsupported wire type
        fields.set(fn, arr);
    }
    return fields;
}

const _DEC = new TextDecoder();

/**
 * Decode:
 *   ExecuteAck {
 *     bool           success    = 1;
 *     string         error_msg  = 2;
 *     string         msg        = 3;
 *     repeated Uuid  output_ids = 4;
 *   }
 */
function decodeExecuteAck(bytes) {
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

/** Extract data bytes from:  DataChunk { bytes data = 1; } */
function decodeDataChunk(bytes) {
    return decodeProto(bytes).get(1)?.[0] ?? new Uint8Array(0);
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
    return parseGrpcFrames(new Uint8Array(await resp.arrayBuffer()));
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Execute a registered command on values in the bank.
 *
 * @param {number}       functionId  - Command ID (see COMMAND_IDS)
 * @param {Uint8Array[]} inputUuids  - 16-byte UUID Uint8Arrays of input tensors
 * @param {string}       options     - Options string, e.g.
 *                                     "Tensor[dtype=f32,device=cpu,shape=(4,4)]"
 * @returns {Promise<Uint8Array[]>}  Output UUID Uint8Arrays
 */
export async function execute(functionId, inputUuids = [], options = '') {
    const frames = await grpcPost(
        'hasty.HastyService/Execute',
        buildExecuteCommand(functionId, options, inputUuids));
    if (!frames.length) throw new Error('Empty Execute response');
    const ack = decodeExecuteAck(frames[0]);
    if (!ack.success) throw new Error(ack.error_msg || 'Execute failed');
    return ack.output_ids;
}

/**
 * Fetch the raw serialised GenericValue bytes for a UUID.
 * Reassembles all DataChunk frames into one contiguous buffer.
 *
 * @param {Uint8Array} uuid16
 * @returns {Promise<Uint8Array>}
 */
export async function fetchRaw(uuid16) {
    const frames = await grpcPost('hasty.HastyService/FetchValue', buildFetchRequest(uuid16));
    const parts  = frames.map(f => decodeDataChunk(f));
    const total  = parts.reduce((s, a) => s + a.length, 0);
    const out    = new Uint8Array(total);
    let off = 0;
    for (const p of parts) { out.set(p, off); off += p.length; }
    return out;
}

// ATen scalar_type index → hasty dtype string (matches tensor_background.cppm)
const _SCALAR_DTYPE = {
    0:'u8', 1:'i8', 2:'i16', 3:'i32', 4:'i64',
    5:'f16', 6:'f32', 7:'f64',
    9:'c32', 10:'c64', 11:'b8',
};

/**
 * Parse tensor metadata from raw GenericValue wire bytes.
 *
 * Wire format (from generic_value.py):
 *   tag (u8 = 1 for TENSOR)
 *   SerializedTensorHeader (16 bytes):
 *     offset  0: device_type  (u8)   — 0=cpu, 1=cuda
 *     offset  1: scalar_type  (u8)   — ATen ScalarType enum
 *     offset  2: ndim         (u8)
 *     offset  3: device_index (i8)
 *     offset  4-7: padding    (4 bytes)
 *     offset  8: total_elements (i64 LE)
 *   shape: ndim × i64 LE
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
 * Returns null if the value is not a tensor or if any error occurs.
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

/**
 * Integer command IDs — must match the C++ CommandRegistry construction order
 * in cpp/lib/src/server/cmd_registry_impl.cpp.
 *
 * base_arithmetic: add(0) sub(1) mult(2) div(3) neg(4) abs(5)
 * base_creation:   rand(6) zeros(7) ones(8)
 */
export const COMMAND_IDS = {
    add:   0,
    sub:   1,
    mult:  2,
    div:   3,
    neg:   4,
    abs:   5,
    rand:  6,
    zeros: 7,
    ones:  8,
};
