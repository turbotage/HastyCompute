// Browser client of the hasty server (proto: rustlib/crates/hasty-server/proto/hasty.proto) over gRPC-web.
// Values arrive as hasty value streams (value_format.js); server streams are decoded frame by frame as they arrive.
// Ids are 32-character hex strings.
import { Decoder, encode } from './value_format.js';

// ── minimal protobuf ────────────────────────────────────────────────────────────────────────────────────────────
class W {
    constructor() { this.parts = []; this.n = 0; }
    #push(a) { this.parts.push(a); this.n += a.length; }
    varint(v) {
        let x = BigInt(v);
        const out = [];
        do { let b = Number(x & 0x7Fn); x >>= 7n; if (x) b |= 0x80; out.push(b); } while (x);
        this.#push(Uint8Array.from(out));
    }
    tag(f, wire) { this.varint((f << 3) | wire); }
    bytes(f, b) { if (!b || !b.length) return this; this.tag(f, 2); this.varint(b.length); this.#push(b); return this; }
    str(f, s) { return s ? this.bytes(f, new TextEncoder().encode(s)) : this; }
    msg(f, w) { if (w) { const b = w.finish(); this.tag(f, 2); this.varint(b.length); this.#push(b); } return this; }
    uint(f, v) { if (v) { this.tag(f, 0); this.varint(v); } return this; }
    bool(f, v) { if (v) { this.tag(f, 0); this.varint(1); } return this; }
    finish() { const o = new Uint8Array(this.n); let p = 0; for (const a of this.parts) { o.set(a, p); p += a.length; } return o; }
}

/** parse a message into Map<field, Array<number | Uint8Array>> */
function parse(b) {
    const m = new Map();
    let p = 0;
    const varint = () => { let x = 0n, s = 0n, c; do { c = b[p++]; x |= BigInt(c & 0x7F) << s; s += 7n; } while (c & 0x80); return x; };
    while (p < b.length) {
        const t = Number(varint());
        const f = t >>> 3, wire = t & 7;
        let v;
        if (wire === 0) { const x = varint(); v = x <= BigInt(Number.MAX_SAFE_INTEGER) ? Number(x) : x; }
        else if (wire === 2) { const n = Number(varint()); v = b.subarray(p, p + n); p += n; }
        else if (wire === 5) { v = b.subarray(p, p + 4); p += 4; }
        else if (wire === 1) { v = b.subarray(p, p + 8); p += 8; }
        else throw new Error(`protobuf: wire type ${wire}`);
        if (!m.has(f)) m.set(f, []);
        m.get(f).push(v);
    }
    return m;
}
const td = new TextDecoder();
const one = (m, f, d) => (m.has(f) ? m.get(f)[0] : d);
const str = (m, f) => (m.has(f) ? td.decode(m.get(f)[0]) : '');
const many = (m, f) => m.get(f) ?? [];

/** first tensor in a value summary (Entry.summary): [slice path, tensor summary] or null */
export function firstTensor(sum, path = '') {
    if (!sum || typeof sum !== 'object') return null;
    if (sum.type === 'tensor') return [path, sum];
    if (sum.type === 'dict') {
        for (const [k, v] of Object.entries(sum.items)) { const r = firstTensor(v, `${path}["${k}"]`); if (r) return r; }
    }
    if (sum.type === 'list' || sum.type === 'tuple') {
        for (let i = 0; i < sum.items.length; i++) { const r = firstTensor(sum.items[i], `${path}[${i}]`); if (r) return r; }
    }
    return null;
}

export const hexToBytes = (h) => Uint8Array.from(h.match(/../g).map((x) => parseInt(x, 16)));
export const bytesToHex = (b) => [...b].map((x) => x.toString(16).padStart(2, '0')).join('');
const idMsg = (hex) => new W().bytes(1, hexToBytes(hex));
const idOf = (b) => bytesToHex(one(parse(b), 1, new Uint8Array()));

function entry(b) {
    const m = parse(b);
    const json = (s) => { try { return JSON.parse(s); } catch { return s; } };
    return {
        id: m.has(1) ? idOf(one(m, 1)) : '', name: str(m, 2), summary: json(str(m, 3)), metadata: json(str(m, 4) || '{}'),
        nbytes: one(m, 5, 0), created_ms: one(m, 6, 0), updated_ms: one(m, 7, 0),
    };
}

// ── gRPC-web transport ─────────────────────────────────────────────────────────────────────────────────────────
export class GrpcError extends Error {
    constructor(code, message) { super(`grpc ${code}: ${message}`); this.code = code; }
}

function frame(msg) {
    const out = new Uint8Array(5 + msg.length);
    new DataView(out.buffer).setUint32(1, msg.length, false);
    out.set(msg, 5);
    return out;
}

function trailerStatus(text) {
    const t = Object.fromEntries(text.split('\r\n').filter(Boolean).map((l) => { const i = l.indexOf(':'); return [l.slice(0, i).trim().toLowerCase(), l.slice(i + 1).trim()]; }));
    return [Number(t['grpc-status'] ?? 0), decodeURIComponent(t['grpc-message'] ?? '')];
}

export class HastyClient {
    /** @param {string} base  server origin; '' = the page's own origin */
    constructor(base = '') { this.base = base.replace(/\/$/, ''); }

    /** call a method; onMessage(Uint8Array) per response message; resolves when the stream ends */
    async #call(method, req, onMessage, signal) {
        const r = await fetch(`${this.base}/hasty.v2.Hasty/${method}`, {
            method: 'POST', body: frame(req.finish()), signal,
            headers: { 'content-type': 'application/grpc-web+proto', 'x-grpc-web': '1', accept: 'application/grpc-web+proto' },
        });
        if (!r.ok) throw new GrpcError(-1, `HTTP ${r.status}`);
        const hs = r.headers.get('grpc-status');
        if (hs !== null && hs !== '0') throw new GrpcError(Number(hs), decodeURIComponent(r.headers.get('grpc-message') ?? ''));
        const reader = r.body.getReader();
        let buf = new Uint8Array(0);
        for (;;) {
            const { value, done } = await reader.read();
            if (value) { const nb = new Uint8Array(buf.length + value.length); nb.set(buf); nb.set(value, buf.length); buf = nb; }
            let p = 0;
            while (buf.length - p >= 5) {
                const flag = buf[p];
                const n = new DataView(buf.buffer, buf.byteOffset + p + 1, 4).getUint32(0, false);
                if (buf.length - p - 5 < n) break;
                const payload = buf.subarray(p + 5, p + 5 + n);
                p += 5 + n;
                if (flag & 0x80) {
                    const [code, msg] = trailerStatus(td.decode(payload));
                    if (code !== 0) throw new GrpcError(code, msg);
                } else onMessage(payload.slice());
            }
            buf = buf.subarray(p);
            if (done) break;
        }
    }

    async #unary(method, req) {
        let out = null;
        await this.#call(method, req, (m) => { out = m; });
        return out ?? new Uint8Array(0);
    }

    async list() { return many(parse(await this.#unary('List', new W())), 1).map(entry); }
    async info(id) { return entry(await this.#unary('Info', idMsg(id))); }
    async delete(id) { await this.#unary('Delete', idMsg(id)); }
    async setMetadata(id, meta) { return entry(await this.#unary('SetMetadata', new W().msg(1, idMsg(id)).str(2, JSON.stringify(meta)))); }
    async commands() { return many(parse(await this.#unary('Commands', new W())), 1).map((b) => { const m = parse(b); return { name: str(m, 1), doc: str(m, 2) }; }); }
    async show(id, view = 'orthoslicer') { await this.#unary('Show', new W().msg(1, idMsg(id)).str(2, view)); }

    /** fetch a value (or a slice of it); decoded incrementally as chunks arrive. onBytes(n) reports progress */
    async get(id, { slice = '', codec = 'lz4', delta = false, onBytes } = {}) {
        const enc = new W().str(1, codec).bool(3, delta);
        const d = new Decoder();
        let n = 0;
        await this.#call('Get', new W().msg(1, idMsg(id)).str(2, slice).msg(3, enc), (m) => {
            const data = one(parse(m), 1, new Uint8Array());
            n += data.length;
            d.push(data);
            if (onBytes) onBytes(n);
        });
        return d.finish();
    }

    /** upload a value (stored blocks); returns its id */
    async put(value, { name = '', metadata = {} } = {}) {
        const h = new W().str(3, name).str(4, JSON.stringify(metadata));
        return idOf(one(parse(await this.#unary('PutUnary', new W().msg(1, h).bytes(2, encode(value)))), 1));
    }

    /** overwrite an entry, or the part of it selected by `slice` */
    async write(id, value, slice = '') {
        const h = new W().msg(1, idMsg(id)).str(2, slice);
        await this.#unary('PutUnary', new W().msg(1, h).bytes(2, encode(value)));
    }

    async execute(command, options = {}, inputs = [], name = '') {
        const w = new W().str(1, command).str(2, JSON.stringify(options));
        for (const i of inputs) w.msg(3, idMsg(i));
        w.str(4, name);
        const m = parse(await this.#unary('Execute', w));
        return { outputs: many(m, 1).map(idOf), message: str(m, 2) };
    }

    /** subscribe to bank events {kind, id, name, message}; returns {close()}; reconnects after errors */
    watch(onEvent, onError) {
        const ac = new AbortController();
        const loop = async () => {
            while (!ac.signal.aborted) {
                try {
                    await this.#call('Watch', new W(), (b) => {
                        const m = parse(b);
                        onEvent({ kind: str(m, 1), id: m.has(2) ? idOf(one(m, 2)) : '', name: str(m, 3), message: str(m, 4) });
                    }, ac.signal);
                } catch (e) {
                    if (ac.signal.aborted) return;
                    if (onError) onError(e);
                }
                await new Promise((r) => setTimeout(r, 1000));
            }
        };
        loop();
        return { close: () => ac.abort() };
    }
}
