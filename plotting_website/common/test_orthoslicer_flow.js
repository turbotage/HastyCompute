// Replays the orthoslicer's server calls for one value and prints a check against the expected centre slice.
//   NODE_EXTRA_CA_CERTS=data/hasty/certs/ca.pem node plotting_website/common/test_orthoslicer_flow.js <base> <id>
import { HastyClient } from './hasty_client.js';
const [base, id] = process.argv.slice(2);
const c = new HastyClient(base);
const info = await c.info(id);
const sum = info.summary;
const mode = sum.dtype.startsWith('complex') ? 'abs' : 'real';
const [q, st] = await Promise.all([c.execute('comprep_u16', { mode, focus: 0.9, right_mode: 'log', right_logc: 10 }, [id]), c.execute('stats', { mode }, [id])]);
const qc = JSON.parse(q.message), stats = JSON.parse(st.message);
const z = Math.floor(sum.shape[0] / 2);
const t0 = Date.now();
const sl = await c.get(q.outputs[0], { slice: `[${z}, :, :]`, codec: 'lz4', delta: true });
const ms = Date.now() - t0;
const raw = await c.get(id, { slice: `[${z}, :, :]`, codec: 'lz4' });
let worst = 0;
const v = (i) => raw.dtype === 'float16' ? f16(raw.data[i]) : raw.data[i];
for (let i = 0; i < sl.data.length; i++) { const x = Math.min(Math.max(v(i), qc.clampa), qc.clampb); if (x >= qc.a && x <= qc.b) worst = Math.max(worst, Math.abs(inv(sl.data[i] / 65535, qc) - x)); }
console.log(`${info.name}: ${sum.dtype} ${sum.shape.join('×')}, stats min ${stats.min.toFixed(3)} max ${stats.max.toFixed(3)} p99.5 ${stats.p99_5.toFixed(3)}; axial slice ${sl.shape.join('×')} ${sl.dtype} in ${ms} ms, window [${qc.a.toFixed(3)}, ${qc.b.toFixed(3)}] decode error ${worst.toExponential(2)} (step ${((qc.b - qc.a) / (0.9 * 65535)).toExponential(2)})`);
await c.delete(q.outputs[0]);
function f16(h) { const s = h >> 15 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023; return e === 0 ? s * m * 2 ** -24 : e === 31 ? NaN : s * (1 + m / 1024) * 2 ** (e - 15); }
// the orthoslicer shader's inverse (renderer.js fs_main)
function inv(y, c) {
    const t0 = (1 - c.focus) / 2, t1 = 1 - t0;
    const sh = (u, m, g, l) => m === 'gamma' ? Math.max(u, 0) ** g : m === 'log' ? Math.expm1(u * Math.log1p(Math.max(l, 1e-6))) / Math.max(l, 1e-6) : u;
    if (y < t0 && t0 > 0) return sh(y / t0, c.left_mode, c.left_gamma, c.left_logc) * (c.a - c.clampa) + c.clampa;
    if (y > t1 && t1 < 1) return sh((y - t1) / (1 - t1), c.right_mode, c.right_gamma, c.right_logc) * (c.clampb - c.b) + c.b;
    return ((y - t0) / (t1 - t0)) * (c.b - c.a) + c.a;
}
