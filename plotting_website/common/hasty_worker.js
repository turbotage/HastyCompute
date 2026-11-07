// Decode worker: fetches and decodes tensor slices off the main thread and transfers the buffers back (no copy).
// Started by HastyClient.enableWorkers(); requests are {op: 'get', rid, base, id, slice, codec, delta, verify} / {op: 'abort', rid}.
import { HastyClient } from './hasty_client.js';
import { HTensor } from './value_format.js';

const clients = new Map();
const aborts = new Map();

self.onmessage = async (ev) => {
    const m = ev.data;
    if (m.op === 'abort') { aborts.get(m.rid)?.abort(); return; }
    if (m.op !== 'get') return;
    const ac = new AbortController();
    aborts.set(m.rid, ac);
    try {
        if (!clients.has(m.base)) clients.set(m.base, new HastyClient(m.base));
        const v = await clients.get(m.base).get(m.id, { slice: m.slice, codec: m.codec, delta: m.delta, verify: m.verify, signal: ac.signal });
        if (!(v instanceof HTensor)) throw new Error('not a tensor');
        const buf = v.data.buffer;
        self.postMessage({ rid: m.rid, dtype: v.dtype, shape: v.shape, buf }, [buf]);
    } catch (e) {
        self.postMessage({ rid: m.rid, error: String(e.message ?? e), aborted: e?.name === 'AbortError' || ac.signal.aborted });
    } finally { aborts.delete(m.rid); }
};
