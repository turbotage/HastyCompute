// Client-side velocity volume: the whole [F, 3, Z, Y, X] float32 cycle (and the optional mask / centerlines) fetched
// once, in a few contiguous frame-range requests decoded in the worker pool. With it the viewer traces streamlines /
// pathlines on the GPU (or in JS) and quantifies planes locally, so exploring does not load the server any more.
import { FlowField } from '../../common/flowmath.js';

/** bytes of the velocity volume in float32 */
export const fieldBytes = (info) => info.frames * (3 + (info.has_mag ? 1 : 0)) * info.dims[0] * info.dims[1] * info.dims[2] * 4;

/**
 * @param {import('../../common/hasty_client.js').HastyClient} client
 * @param {string} id  flow value id;  @param {object} info  flow_info message
 * @returns {Promise<{field: FlowField, mask: Uint8Array|null}>}
 */
export async function loadField(client, id, info, { onProgress, signal, chunkBytes = 96 << 20, velRank = 5 } = {}) {
    const [X, Y, Z] = info.dims, F = info.frames, nv = X * Y * Z;
    const per = Math.max(1, Math.floor(chunkBytes / (12 * nv)));
    const frames = [];
    if (velRank === 4) {                                         // a single frame stored as [3, Z, Y, X]
        const t = await client.getTensor(id, { slice: '["vel"]', signal });
        frames.push(f32(t));
    } else {
        for (let f0 = 0; f0 < F; f0 += per) {
            const f1 = Math.min(F, f0 + per);
            const t = await client.getTensor(id, { slice: `["vel"][${f0}:${f1}]`, signal });
            const d = f32(t);
            for (let k = 0; k < f1 - f0; k++) frames.push(d.subarray(k * 3 * nv, (k + 1) * 3 * nv));
            onProgress?.(f1, F);
        }
    }
    let mag = null;
    if (info.has_mag) {                                          // anatomy channel: with it in memory the slice planes need no server requests either
        mag = [];
        const perM = Math.max(1, Math.floor(chunkBytes / (4 * nv)));
        for (let f0 = 0; f0 < F; f0 += perM) {
            const f1 = Math.min(F, f0 + perM);
            const t = await client.getTensor(id, { slice: `["mag"][${f0}:${f1}]`, signal });
            const d = f32(t);
            for (let k = 0; k < f1 - f0; k++) mag.push(d.subarray(k * nv, (k + 1) * nv));
        }
    }
    let mask = null;
    if (info.has_mask) {
        const t = await client.getTensor(id, { slice: '["mask"]', signal });
        mask = Uint8Array.from(t.data, (v) => (v ? 1 : 0));
    }
    const field = new FlowField({ dims: info.dims, spacing: info.spacing, dt: info.dt, frames });
    field.mag = mag;                                             // [F] of [Z, Y, X], or null (speed is then the anatomy channel)
    return { field, mask };
}

function f32(t) {
    if (t.dtype === 'float32') return t.data;
    return Float32Array.from(t.data);                            // float64 etc.; float16 would need decoding — the server stores f32
}

/** optional centerlines: {points Float32Array [N*3] voxel coords, offsets Uint32Array [L+1], radius?}; null if absent */
export async function loadCenterlines(client, id, entrySummary, { signal } = {}) {
    const items = entrySummary?.items?.centerlines?.items;
    if (!items?.points || !items?.offsets) return null;
    const [p, o] = await Promise.all([
        client.getTensor(id, { slice: '["centerlines"]["points"]', signal }),
        client.getTensor(id, { slice: '["centerlines"]["offsets"]', signal }),
    ]);
    const r = items.radius ? await client.getTensor(id, { slice: '["centerlines"]["radius"]', signal }) : null;
    return { points: Float32Array.from(p.data), offsets: Uint32Array.from(o.data), radius: r ? Float32Array.from(r.data) : null };
}
