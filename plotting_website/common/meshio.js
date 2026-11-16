// Triangle mesh file I/O (OBJ, STL ascii + binary, PLY ascii + binary) and the hasty mesh value (rustlib/crates/hasty-server/src/mesh.rs).
// Parsers return { vertices: Float32Array [3N], faces: Uint32Array [3M] } with shared vertices (STL vertices are welded).
// Files are data: nothing is executed, sizes are validated against the buffer.
import { HTensor } from './value_format.js';

const MAX_VERTS = 2 ** 28;

/** hasty mesh value (a plain object, encodes as a dict) from arrays; vertices in mm */
export function meshValue({ vertices, faces }, { space = 'voxel_mm', scalars = null } = {}) {
    const v = { kind: 'mesh', vertices: new HTensor('float32', [vertices.length / 3, 3], vertices), faces: new HTensor('uint32', [faces.length / 3, 3], faces), space, unit: 'mm' };
    if (scalars) v.scalars = Object.fromEntries(Object.entries(scalars).map(([k, a]) => [k, new HTensor('float32', [a.length], a)]));
    return v;
}

function weld(pos) {                                   // exact duplicate vertices (STL) → shared vertices
    const map = new Map(), out = [], idx = new Uint32Array(pos.length / 3);
    for (let i = 0; i < idx.length; i++) {
        const k = `${pos[3 * i]},${pos[3 * i + 1]},${pos[3 * i + 2]}`;
        let j = map.get(k);
        if (j === undefined) { j = out.length / 3; map.set(k, j); out.push(pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]); }
        idx[i] = j;
    }
    return { vertices: Float32Array.from(out), faces: idx };
}

function parseOBJ(text) {
    const v = [], f = [];
    for (const line of text.split(/\r?\n/)) {
        if (line.startsWith('v ')) { const p = line.trim().split(/\s+/); v.push(+p[1], +p[2], +p[3]); }
        else if (line.startsWith('f ')) {
            const ids = line.trim().split(/\s+/).slice(1).map((t) => { const i = parseInt(t.split('/')[0], 10); return i < 0 ? v.length / 3 + i : i - 1; });
            for (let k = 1; k + 1 < ids.length; k++) f.push(ids[0], ids[k], ids[k + 1]);          // fan for polygons
        }
    }
    if (v.some(Number.isNaN)) throw new Error('OBJ: a vertex is not a number');
    if (f.some((i) => !(i >= 0 && i < v.length / 3))) throw new Error('OBJ: a face refers to a missing vertex');
    return { vertices: Float32Array.from(v), faces: Uint32Array.from(f) };
}

function parseSTL(buf) {
    const head = new TextDecoder().decode(new Uint8Array(buf, 0, Math.min(buf.byteLength, 512)));
    const dv = new DataView(buf);
    const binary = buf.byteLength >= 84 && 84 + dv.getUint32(80, true) * 50 === buf.byteLength;
    const pos = [];
    if (!binary && /^\s*solid/.test(head) && /facet/.test(head)) {
        for (const m of new TextDecoder().decode(buf).matchAll(/vertex\s+(\S+)\s+(\S+)\s+(\S+)/g)) pos.push(+m[1], +m[2], +m[3]);
        if (pos.length % 9) throw new Error('STL: vertex count is not a multiple of 3');
    } else {
        if (!binary) throw new Error('STL: neither ASCII nor a binary file of the stated size');
        const n = dv.getUint32(80, true);
        for (let t = 0; t < n; t++) for (let k = 0; k < 3; k++) for (let c = 0; c < 3; c++) pos.push(dv.getFloat32(84 + t * 50 + 12 + k * 12 + c * 4, true));
    }
    return weld(pos);
}

const PLY_TYPES = { char: 1, int8: 1, uchar: 1, uint8: 1, short: 2, int16: 2, ushort: 2, uint16: 2, int: 4, int32: 4, uint: 4, uint32: 4, float: 4, float32: 4, double: 8, float64: 8 };
function plyRead(dv, o, t, le) {
    switch (t) {
        case 'char': case 'int8': return dv.getInt8(o);
        case 'uchar': case 'uint8': return dv.getUint8(o);
        case 'short': case 'int16': return dv.getInt16(o, le);
        case 'ushort': case 'uint16': return dv.getUint16(o, le);
        case 'int': case 'int32': return dv.getInt32(o, le);
        case 'uint': case 'uint32': return dv.getUint32(o, le);
        case 'float': case 'float32': return dv.getFloat32(o, le);
        default: return dv.getFloat64(o, le);
    }
}

function parsePLY(buf) {
    const bytes = new Uint8Array(buf);
    const hdrEnd = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 65536))).indexOf('end_header');
    if (hdrEnd < 0) throw new Error('PLY: no end_header');
    let bodyStart = hdrEnd + 'end_header'.length;
    while (bytes[bodyStart] === 13) bodyStart++;
    bodyStart++;                                               // the newline after end_header
    const lines = new TextDecoder().decode(bytes.subarray(0, hdrEnd)).split(/\r?\n/).map((l) => l.trim().split(/\s+/));
    const format = lines.find((l) => l[0] === 'format')?.[1];
    const elems = [];
    for (const l of lines) {
        if (l[0] === 'element') elems.push({ name: l[1], count: +l[2], props: [] });
        else if (l[0] === 'property') elems.at(-1).props.push(l[1] === 'list' ? { list: true, ct: l[2], it: l[3], name: l[4] } : { type: l[1], name: l[2] });
    }
    const vertices = [], faces = [];
    const text = format === 'ascii' ? new TextDecoder().decode(bytes.subarray(bodyStart)).split(/\r?\n/) : null;
    const dv = new DataView(buf), le = format !== 'binary_big_endian';
    if (!['ascii', 'binary_little_endian', 'binary_big_endian'].includes(format)) throw new Error(`PLY: format ${format}`);
    let o = bodyStart, ln = 0;
    for (const e of elems) {
        if (e.count > MAX_VERTS) throw new Error('PLY: element count too large');
        for (let r = 0; r < e.count; r++) {
            const tok = text ? (text[ln++] ?? '').trim().split(/\s+/) : null;
            let ti = 0;
            const vals = {};
            for (const p of e.props) {
                if (p.list) {
                    const n = text ? +tok[ti++] : (() => { const x = plyRead(dv, o, p.ct, le); o += PLY_TYPES[p.ct]; return x; })();
                    const arr = [];
                    for (let k = 0; k < n; k++) {
                        if (text) arr.push(+tok[ti++]); else { arr.push(plyRead(dv, o, p.it, le)); o += PLY_TYPES[p.it]; }
                    }
                    vals[p.name] = arr;
                } else if (text) vals[p.name] = +tok[ti++];
                else { vals[p.name] = plyRead(dv, o, p.type, le); o += PLY_TYPES[p.type]; }
            }
            if (e.name === 'vertex') vertices.push(vals.x, vals.y, vals.z);
            else if (e.name === 'face') {
                const ids = vals.vertex_indices ?? vals.vertex_index ?? [];
                for (let k = 1; k + 1 < ids.length; k++) faces.push(ids[0], ids[k], ids[k + 1]);
            }
        }
    }
    if (o > buf.byteLength) throw new Error('PLY: file is shorter than its header says');
    if (vertices.some(Number.isNaN)) throw new Error('PLY: bad vertex data');
    if (faces.some((i) => !(i >= 0 && i < vertices.length / 3))) throw new Error('PLY: a face refers to a missing vertex');
    return { vertices: Float32Array.from(vertices), faces: Uint32Array.from(faces) };
}

/** parse by file name extension (.obj .stl .ply) */
export function parseMesh(buf, filename) {
    const ext = filename.toLowerCase().replace(/^.*\./, '');
    if (ext === 'obj') return parseOBJ(new TextDecoder().decode(buf));
    if (ext === 'stl') return parseSTL(buf);
    if (ext === 'ply') return parsePLY(buf);
    throw new Error(`unsupported mesh format .${ext} (obj, stl, ply)`);
}

export function writeOBJ({ vertices, faces }) {
    const out = [];
    for (let i = 0; i < vertices.length; i += 3) out.push(`v ${vertices[i]} ${vertices[i + 1]} ${vertices[i + 2]}`);
    for (let i = 0; i < faces.length; i += 3) out.push(`f ${faces[i] + 1} ${faces[i + 1] + 1} ${faces[i + 2] + 1}`);
    return new TextEncoder().encode(out.join('\n') + '\n');
}

export function writeSTL({ vertices, faces }) {
    const n = faces.length / 3, buf = new ArrayBuffer(84 + 50 * n), dv = new DataView(buf);
    new TextEncoder().encodeInto('hasty mesh', new Uint8Array(buf, 0, 80));
    dv.setUint32(80, n, true);
    for (let t = 0; t < n; t++) {
        const p = [0, 1, 2].map((k) => faces[3 * t + k] * 3);
        const a = [0, 1, 2].map((c) => vertices[p[1] + c] - vertices[p[0] + c]), b = [0, 1, 2].map((c) => vertices[p[2] + c] - vertices[p[0] + c]);
        let nx = a[1] * b[2] - a[2] * b[1], ny = a[2] * b[0] - a[0] * b[2], nz = a[0] * b[1] - a[1] * b[0];
        const l = Math.hypot(nx, ny, nz) || 1; nx /= l; ny /= l; nz /= l;
        const o = 84 + 50 * t;
        [nx, ny, nz].forEach((x, c) => dv.setFloat32(o + 4 * c, x, true));
        p.forEach((pi, k) => [0, 1, 2].forEach((c) => dv.setFloat32(o + 12 + 12 * k + 4 * c, vertices[pi + c], true)));
    }
    return new Uint8Array(buf);
}

export function writePLY({ vertices, faces }) {
    const hdr = new TextEncoder().encode(`ply\nformat binary_little_endian 1.0\nelement vertex ${vertices.length / 3}\nproperty float x\nproperty float y\nproperty float z\nelement face ${faces.length / 3}\nproperty list uchar uint vertex_indices\nend_header\n`);
    const out = new Uint8Array(hdr.length + vertices.length * 4 + (faces.length / 3) * 13), dv = new DataView(out.buffer);
    out.set(hdr);
    let o = hdr.length;
    for (const x of vertices) { dv.setFloat32(o, x, true); o += 4; }
    for (let t = 0; t < faces.length; t += 3) { dv.setUint8(o++, 3); for (let k = 0; k < 3; k++) { dv.setUint32(o, faces[t + k], true); o += 4; } }
    return out;
}

export const WRITERS = { obj: writeOBJ, stl: writeSTL, ply: writePLY };
