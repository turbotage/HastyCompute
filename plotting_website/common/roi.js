// Region of interest around a vessel click: the velocity (all frames), mask and PC-MRA of a sub-box at FULL resolution
// are fetched with the server's range slices, so the vessel measurement never depends on the (possibly huge) whole
// volume. The box is a sub-volume; its measurements are made in its own voxel coordinates and mapped back by `lo`.
import { FlowField } from './flowmath.js';

/** voxel box [lo, hi) around p (voxel coords) reaching `reachM` metres in every direction, clipped to the volume */
export function roiBox(p, dims, spacing, reachM) {
    const lo = [], hi = [];
    for (let a = 0; a < 3; a++) {
        const r = Math.ceil(reachM / spacing[a]);
        lo.push(Math.max(0, Math.floor(p[a]) - r));
        hi.push(Math.min(dims[a], Math.floor(p[a]) + r + 1));
    }
    return { lo, hi };
}

export const roiBytes = (box, frames) => (box.hi[0] - box.lo[0]) * (box.hi[1] - box.lo[1]) * (box.hi[2] - box.lo[2]) * (frames * 12 + 5);
export const roiContains = (outer, inner) => [0, 1, 2].every((a) => outer.lo[a] <= inner.lo[a] && outer.hi[a] >= inner.hi[a]);

const zyx = (b) => `${b.lo[2]}:${b.hi[2]}, ${b.lo[1]}:${b.hi[1]}, ${b.lo[0]}:${b.hi[0]}`;

/**
 * @param client HastyClient;  @param id flow value;  @param pcId PC-MRA value id (tensor [Z, Y, X]);  @param info flow_info message
 * @returns {{field: FlowField, mask: Uint8Array|null, pc: Float32Array, lo: number[], hi: number[]}}  (field.dims = box size)
 */
export async function fetchRoi(client, { id, pcId, info, velRank = 5, box, signal }) {
    const dims = [box.hi[0] - box.lo[0], box.hi[1] - box.lo[1], box.hi[2] - box.lo[2]], nv = dims[0] * dims[1] * dims[2];
    const velSlice = velRank === 4 ? `["vel"][:, ${zyx(box)}]` : `["vel"][:, :, ${zyx(box)}]`;
    const [v, pc, m] = await Promise.all([
        client.getTensor(id, { slice: velSlice, signal }),
        client.getTensor(pcId, { slice: `[${zyx(box)}]`, signal }),
        info.has_mask ? client.getTensor(id, { slice: `["mask"][${zyx(box)}]`, signal }) : null,
    ]);
    const data = v.dtype === 'float32' ? v.data : Float32Array.from(v.data);
    const F = velRank === 4 ? 1 : info.frames;
    if (data.length !== F * 3 * nv) throw new Error(`region velocity has ${data.length} values, expected ${F * 3 * nv}`);
    const frames = [];
    for (let f = 0; f < F; f++) frames.push(data.subarray(f * 3 * nv, (f + 1) * 3 * nv));
    if (pc.data.length !== nv) throw new Error('region PC-MRA has the wrong size');
    return {
        field: new FlowField({ dims, spacing: info.spacing, dt: info.dt, frames }),
        mask: m ? Uint8Array.from(m.data, (x) => (x ? 1 : 0)) : null,
        pc: pc.dtype === 'float32' ? pc.data : Float32Array.from(pc.data),
        lo: box.lo.slice(), hi: box.hi.slice(),
    };
}

/** shift a measurement made in box coordinates back to volume coordinates (new arrays: planes share `center` with the average) */
export function shiftResult(res, lo) {
    const sh = (c) => (c ? c.map((x, k) => x + lo[k]) : c);
    for (const m of res.planes) { m.center = sh(m.center); m.centroid = sh(m.centroid); }
    res.avg.center = sh(res.avg.center);
    return res;
}
