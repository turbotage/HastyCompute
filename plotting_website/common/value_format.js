// hasty value format v1 (spec: rustlib/crates/hasty-value/FORMAT.md) for the browser.
//
// Decoding is incremental: `new Decoder()`, `push(Uint8Array)` as chunks arrive (each complete block is decoded at
// once), `finish()` returns the value. Codecs: stored and LZ4 (zstd blocks are rejected — ask the server for lz4).
// Encoding writes stored blocks (uploads from the browser are small).
//
// JS mapping: null, boolean, number (Int / Float; Ints beyond 2^53 become BigInt), string, Array (List),
// HTuple (Tuple), Map (Dict, insertion order), HTensor (Tensor).

export const DTYPES = [
    // name, bytes, filter word, TypedArray (complex: interleaved re/im)
    ['bool', 1, 1, Uint8Array], ['uint8', 1, 1, Uint8Array], ['int8', 1, 1, Int8Array], ['uint16', 2, 2, Uint16Array],
    ['int16', 2, 2, Int16Array], ['uint32', 4, 4, Uint32Array], ['int32', 4, 4, Int32Array],
    ['uint64', 8, 8, BigUint64Array], ['int64', 8, 8, BigInt64Array], ['float16', 2, 2, Uint16Array],
    ['bfloat16', 2, 2, Uint16Array], ['float32', 4, 4, Float32Array], ['float64', 8, 8, Float64Array],
    ['complex64', 8, 4, Float32Array], ['complex128', 16, 8, Float64Array],
];
const CODE = Object.fromEntries(DTYPES.map((d, i) => [d[0], i]));

export class HTuple {
    constructor(items) { this.items = items; }
}

export class HTensor {
    /** @param {string} dtype  @param {number[]} shape  @param {ArrayBufferView} data (C-order) */
    constructor(dtype, shape, data) {
        this.dtype = dtype;
        this.shape = shape;
        this.data = data;
    }
    get numel() { return this.shape.reduce((a, b) => a * b, 1); }
    get bytes() { return new Uint8Array(this.data.buffer, this.data.byteOffset, this.data.byteLength); }
    static fromBytes(dtype, shape, bytes) {
        const [, size, , T] = DTYPES[CODE[dtype]];
        const buf = new ArrayBuffer(bytes.length);
        new Uint8Array(buf).set(bytes);
        return new HTensor(dtype, shape, new T(buf, 0, bytes.length / (T.BYTES_PER_ELEMENT || size)));
    }
}

// ── block codec ─────────────────────────────────────────────────────────────────────────────────────────────────
export function lz4Decompress(src, dst) {
    let s = 0, d = 0;
    const n = src.length;
    while (s < n) {
        const token = src[s++];
        let lit = token >>> 4;
        if (lit === 15) { let b; do { b = src[s++]; lit += b; } while (b === 255); }
        dst.set(src.subarray(s, s + lit), d);
        s += lit; d += lit;
        if (s >= n) break;                       // the last sequence carries literals only
        const off = src[s] | (src[s + 1] << 8);
        s += 2;
        let ml = token & 15;
        if (ml === 15) { let b; do { b = src[s++]; ml += b; } while (b === 255); }
        ml += 4;
        let m = d - off;
        if (off === 0 || m < 0) throw new Error('lz4: bad match offset');
        if (off >= ml) { dst.copyWithin(d, m, m + ml); d += ml; }
        else { for (let k = 0; k < ml; k++) dst[d++] = dst[m++]; }   // overlapping: repeat the pattern
    }
    if (d !== dst.length) throw new Error(`lz4: decoded ${d} of ${dst.length} bytes`);
}

function unshuffle(src, word, dst) {
    const n = Math.floor(src.length / word);
    for (let b = 0; b < word; b++) {
        const o = b * n;
        for (let i = 0; i < n; i++) dst[i * word + b] = src[o + i];
    }
    dst.set(src.subarray(n * word), n * word);
}

function undelta(buf, word) {
    const n = Math.floor(buf.length / word);
    const a = new Uint8Array(n * word);
    a.set(buf.subarray(0, n * word));
    if (word === 1) { for (let i = 1; i < n; i++) a[i] = (a[i] + a[i - 1]) & 255; }
    else if (word === 2 || word === 4) {
        const v = word === 2 ? new Uint16Array(a.buffer) : new Uint32Array(a.buffer);
        for (let i = 1; i < n; i++) v[i] = v[i] + v[i - 1];
    } else if (word === 8) {
        const v = new BigUint64Array(a.buffer);
        for (let i = 1; i < n; i++) v[i] = BigInt.asUintN(64, v[i] + v[i - 1]);
    }
    buf.set(a);
}

let CRC_TABLE = null;
export function crc32(b) {
    if (!CRC_TABLE) {
        CRC_TABLE = new Uint32Array(256);
        for (let i = 0; i < 256; i++) { let c = i; for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1; CRC_TABLE[i] = c >>> 0; }
    }
    let c = 0xFFFFFFFF;
    for (let i = 0; i < b.length; i++) c = CRC_TABLE[(c ^ b[i]) & 255] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
}

function decodeBlock(codec, filter, word, payload, out) {
    let data;
    if (codec === 0) data = payload;
    else if (codec === 1) { data = filter ? new Uint8Array(out.length) : out; lz4Decompress(payload, data); }
    else if (codec === 2) throw new Error('zstd blocks cannot be decoded in the browser (request codec "lz4")');
    else throw new Error(`unknown codec ${codec}`);
    if (filter & 1) unshuffle(data, word, out);
    else if (data !== out) out.set(data);
    if (filter & 2) undelta(out, word);
}

// ── incremental decoder (a generator parser that pauses whenever it needs more bytes) ───────────────────────────
const NEED = Symbol('need');

export class Decoder {
    constructor() {
        this.buf = new Uint8Array(0);
        this.pos = 0;
        this.value = undefined;
        this.done = false;
        this.gen = this.#stream();
        this.gen.next();
    }

    push(chunk) {
        if (this.done) { if (chunk.length) throw new Error('data after the end of the value'); return; }
        const rest = this.buf.length - this.pos;
        const nb = new Uint8Array(rest + chunk.length);
        nb.set(this.buf.subarray(this.pos));
        nb.set(chunk, rest);
        this.buf = nb;
        this.pos = 0;
        const r = this.gen.next();
        if (r.done) { this.done = true; this.value = r.value; }
    }

    finish() {
        if (!this.done) throw new Error('truncated stream');
        if (this.pos !== this.buf.length) throw new Error('trailing bytes');
        return this.value;
    }

    *#need(n) {
        while (this.buf.length - this.pos < n) yield NEED;
        const s = this.buf.subarray(this.pos, this.pos + n);
        this.pos += n;
        return s;
    }

    *#stream() {
        const h = yield* this.#need(8);
        if (h[0] !== 0x48 || h[1] !== 0x53 || h[2] !== 0x54 || h[3] !== 0x56) throw new Error('not a hasty value stream');
        if (h[4] !== 1) throw new Error(`unsupported format version ${h[4]}`);
        return yield* this.#value();
    }

    *#u64() {
        const b = yield* this.#need(8);
        const v = new DataView(b.buffer, b.byteOffset, 8).getBigUint64(0, true);
        return v <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(v) : v;
    }

    *#value() {
        const tag = (yield* this.#need(1))[0];
        const td = new TextDecoder();
        switch (tag) {
            case 0: return null;
            case 1: return (yield* this.#need(1))[0] !== 0;
            case 2: {
                const b = yield* this.#need(8);
                const v = new DataView(b.buffer, b.byteOffset, 8).getBigInt64(0, true);
                return (v <= BigInt(Number.MAX_SAFE_INTEGER) && v >= BigInt(Number.MIN_SAFE_INTEGER)) ? Number(v) : v;
            }
            case 3: { const b = yield* this.#need(8); return new DataView(b.buffer, b.byteOffset, 8).getFloat64(0, true); }
            case 4: { const n = yield* this.#u64(); return td.decode((yield* this.#need(n)).slice()); }
            case 5: case 6: {
                const n = yield* this.#u64();
                const items = [];
                for (let i = 0; i < n; i++) items.push(yield* this.#value());
                return tag === 5 ? items : new HTuple(items);
            }
            case 7: {
                const n = yield* this.#u64();
                const m = new Map();
                for (let i = 0; i < n; i++) {
                    const kb = yield* this.#need(4);
                    const kl = new DataView(kb.buffer, kb.byteOffset, 4).getUint32(0, true);
                    const k = td.decode((yield* this.#need(kl)).slice());
                    m.set(k, yield* this.#value());
                }
                return m;
            }
            case 8: {
                const hb = (yield* this.#need(7)).slice();
                const code = hb[0], ndim = hb[1];
                const blockSize = new DataView(hb.buffer, 3, 4).getUint32(0, true);
                const shape = [];
                for (let d = 0; d < ndim; d++) shape.push(Number(yield* this.#u64()));
                const [name, size, , T] = DTYPES[code] ?? (() => { throw new Error(`unknown dtype ${code}`); })();
                const nbytes = shape.reduce((a, b) => a * b, 1) * size;
                const out = new Uint8Array(nbytes);
                for (let off = 0; off < nbytes;) {
                    const bh = (yield* this.#need(12)).slice();
                    const dv = new DataView(bh.buffer);
                    const [codec, filter, word, flags] = bh;
                    const raw = dv.getUint32(4, true), comp = dv.getUint32(8, true);
                    const crc = flags & 1 ? new DataView((yield* this.#need(4)).slice().buffer).getUint32(0, true) : null;
                    if (raw !== Math.min(blockSize, nbytes - off)) throw new Error('block raw length mismatch');
                    const payload = yield* this.#need(comp);
                    if (crc !== null && crc32(payload) !== crc) throw new Error('block checksum mismatch');
                    decodeBlock(codec, filter, word, payload, out.subarray(off, off + raw));
                    off += raw;
                }
                return new HTensor(name, shape, new T(out.buffer, 0, nbytes / T.BYTES_PER_ELEMENT));
            }
            default: throw new Error(`unknown value tag ${tag}`);
        }
    }
}

export function decode(bytes) {
    const d = new Decoder();
    d.push(bytes);
    return d.finish();
}

// ── encoder (stored blocks) ───────────────────────────────────────────────────────────────────────────────────
export function encode(value, { blockSize = 1 << 20 } = {}) {
    const parts = [];
    const te = new TextEncoder();
    const u8 = (...b) => parts.push(Uint8Array.from(b));
    const u32 = (v) => { const a = new Uint8Array(4); new DataView(a.buffer).setUint32(0, v, true); parts.push(a); };
    const u64 = (v) => { const a = new Uint8Array(8); new DataView(a.buffer).setBigUint64(0, BigInt(v), true); parts.push(a); };
    const str = (s) => { const b = te.encode(s); u64(b.length); parts.push(b); };
    const enc = (v) => {
        if (v === null || v === undefined) u8(0);
        else if (typeof v === 'boolean') u8(1, v ? 1 : 0);
        else if (typeof v === 'bigint') { u8(2); const a = new Uint8Array(8); new DataView(a.buffer).setBigInt64(0, v, true); parts.push(a); }
        else if (typeof v === 'number') {
            const a = new Uint8Array(8);
            if (Number.isInteger(v)) { u8(2); new DataView(a.buffer).setBigInt64(0, BigInt(v), true); }
            else { u8(3); new DataView(a.buffer).setFloat64(0, v, true); }
            parts.push(a);
        } else if (typeof v === 'string') { u8(4); str(v); }
        else if (Array.isArray(v)) { u8(5); u64(v.length); v.forEach(enc); }
        else if (v instanceof HTuple) { u8(6); u64(v.items.length); v.items.forEach(enc); }
        else if (v instanceof HTensor) {
            const bs = Math.max(16, Math.floor(blockSize / 16) * 16);
            u8(8, CODE[v.dtype], v.shape.length, 0); u32(bs);
            v.shape.forEach(u64);
            const b = v.bytes;
            for (let o = 0; o < b.length; o += bs) {
                const blk = b.subarray(o, Math.min(o + bs, b.length));
                u8(0, 0, DTYPES[CODE[v.dtype]][2], 0); u32(blk.length); u32(blk.length);
                parts.push(blk);
            }
        } else if (v instanceof Map || typeof v === 'object') {
            const entries = v instanceof Map ? [...v.entries()] : Object.entries(v);
            u8(7); u64(entries.length);
            for (const [k, x] of entries) { const kb = te.encode(String(k)); u32(kb.length); parts.push(kb); enc(x); }
        } else throw new Error(`cannot encode ${typeof v}`);
    };
    parts.push(Uint8Array.from([0x48, 0x53, 0x54, 0x56, 1, 0, 0, 0]));
    enc(value);
    const n = parts.reduce((a, p) => a + p.length, 0);
    const out = new Uint8Array(n);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
}
