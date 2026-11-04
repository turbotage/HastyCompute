// End-to-end check of the browser client against a running hasty-serve over TLS (CA trusted via NODE_EXTRA_CA_CERTS):
//   NODE_EXTRA_CA_CERTS=<cert dir>/ca.pem node plotting_website/common/test_client.js https://<host>:<port>
import { HastyClient } from './hasty_client.js';
import { HTensor } from './value_format.js';

const base = process.argv[2];
const c = new HastyClient(base);
const page = await fetch(`${base}/orthoslicer/index.html`);
if (!page.ok || !(await page.text()).includes('<')) throw new Error('static website not served');
const caPem = await (await fetch(`${base}/hasty-ca.pem`)).text();
if (!caPem.includes('BEGIN CERTIFICATE')) throw new Error('CA not served');
console.log(`static site + CA over ${base}: ok`);

const events = [];
const w = c.watch((e) => events.push(e), (e) => console.error('watch error', e.message));
await new Promise((r) => setTimeout(r, 300));
const data = new Float32Array(4 * 5 * 6).map((_, i) => Math.sin(i / 7));
const id = await c.put(new Map([['vol', new HTensor('float32', [4, 5, 6], data)], ['note', 'from js']]), { name: 'js upload', metadata: { axes: ['z', 'y', 'x'] } });
const list = await c.list();
if (!list.some((e) => e.id === id && e.name === 'js upload' && e.metadata.axes)) throw new Error('list misses the upload');
const back = await c.get(id, { slice: '["vol"][1, :, 2:4]' });
if (back.shape.join() !== '5,2' || Math.abs(back.data[0] - data[1 * 30 + 0 * 6 + 2]) > 1e-7) throw new Error('slice wrong');
const full = await c.get(id, { codec: 'lz4' });
if (full.get('note') !== 'from js' || full.get('vol').data.some((x, i) => x !== data[i])) throw new Error('round trip wrong');
const vid = await c.put(new HTensor('float32', [4, 5, 6], data), { name: 'vol' });
const { outputs, message } = await c.execute('comprep_u16', { mode: 'real' }, [vid]);
const q = await c.get(outputs[0], { codec: 'lz4', delta: true });
if (q.dtype !== 'uint16' || q.shape.join() !== '4,5,6' || !('clampb' in JSON.parse(message))) throw new Error('comprep wrong');
const cmds = await c.commands();
if (!cmds.some((x) => x.name === 'fft')) throw new Error('commands missing');
await c.show(vid);
try { await c.execute('nope', {}, []); throw new Error('expected error'); } catch (e) { if (!/grpc 3/.test(e.message)) throw e; }
await c.delete(id);
await new Promise((r) => setTimeout(r, 300));
w.close();
const kinds = new Set(events.map((e) => e.kind));
for (const k of ['put', 'show', 'delete']) if (!kinds.has(k)) throw new Error(`missing ${k} event (${[...kinds]})`);
console.log(`grpc-web: put/list/get(slice)/execute/commands/show/delete + watch events (${events.length}): ok`);
