// node plotting_website/common/test_value_format.js — decode the Rust test files (hvtool gen), re-encode, compare digests
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { Decoder, decode, encode, crc32, HTensor, HTuple } from './value_format.js';

const OUT = new URL('../../rustlib/target/hv-test/', import.meta.url).pathname;
const TOOL = new URL('../../rustlib/target/release/hvtool', import.meta.url).pathname;
const want = JSON.parse(readFileSync(OUT + 'digest.json', 'utf8'));

function digest(v) {
    if (v instanceof HTensor) return { tensor: v.dtype, shape: v.shape, crc32: crc32(v.bytes) };
    if (Array.isArray(v)) return { list: v.map(digest) };
    if (v instanceof HTuple) return { tuple: v.items.map(digest) };
    if (v instanceof Map) return { dict: [...v.entries()].map(([k, x]) => [k, digest(x)]) };
    if (v === null) return { none: null };
    if (typeof v === 'boolean') return { bool: v };
    if (typeof v === 'number') return Number.isInteger(v) && !(Object.is(v, 0.125)) ? { int: v } : { float: v };
    return { str: v };
}
const canon = (o) => JSON.stringify(o, (k, v) => (v && typeof v === 'object' && !Array.isArray(v)) ? Object.fromEntries(Object.entries(v).sort()) : v);

for (const f of readdirSync(OUT).filter((f) => f.endsWith('.hv') && !/^(py_|js_|chunked)/.test(f)).sort()) {
    const bytes = new Uint8Array(readFileSync(OUT + f));
    if (f.startsWith('zstd')) {
        try { decode(bytes); throw new Error('zstd must be rejected'); } catch (e) { if (!/zstd/.test(e.message)) throw e; }
        console.log(`Rust -> JS  ${f.padEnd(20)} rejected (zstd) as intended`);
        continue;
    }
    const d = new Decoder();                                   // feed in odd-sized chunks
    for (let i = 0; i < bytes.length; i += 333) d.push(bytes.subarray(i, i + 333));
    const v = d.finish();
    if (canon(digest(v)) !== canon(want)) throw new Error(`${f}: digest differs\n${canon(digest(v))}\n${canon(want)}`);
    console.log(`Rust -> JS  ${f.padEnd(20)} ${String(bytes.length).padStart(7)} B  ok`);
}
const v = decode(new Uint8Array(readFileSync(OUT + 'lz4.hv')));
writeFileSync(OUT + 'js_stored.hv', encode(v, { blockSize: 777 }));
const got = JSON.parse(execFileSync(TOOL, ['digest', OUT + 'js_stored.hv'], { encoding: 'utf8' }));
if (canon(got) !== canon(want)) throw new Error('JS encoder: Rust digest differs');
console.log('JS -> Rust  js_stored.hv          ok');
console.log('ok');
