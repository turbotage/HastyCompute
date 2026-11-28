// Moving surface (a `mesh_seq` value from flow_bloodpool / meshseq_track): parsing and the frame / blend logic, no three.js.
// Vertices are in mm with voxel centres at index × spacing, i.e. the flow viewer's world coordinates.
//
// links[i] = 1: frame i+1 has the vertices (and triangles) of frame i, so positions can be blended between them;
// 0: a new surface run (split / merge, a valve opening or closing): the viewer snaps to the nearer frame.
// links has F − 1 entries; the wrap from the last frame to the first is never linked.

function scalarsOf(f) {
    const sc = f.get('scalars'), out = {};
    if (sc instanceof Map) for (const [k, t] of sc) out[k] = Float32Array.from(t.data);
    return out;
}

/** @param {Map} v mesh_seq value as decoded by the client */
export function parseSeq(v) {
    if (!(v instanceof Map) || v.get('kind') !== 'mesh_seq') throw new Error('not a mesh sequence');
    const fr = v.get('frames');
    if (!Array.isArray(fr) || !fr.length) throw new Error('mesh sequence without frames');
    const frames = fr.map((f) => ({
        pos: f.get('vertices').data,
        idx: f.get('faces').data instanceof Uint32Array ? f.get('faces').data : Uint32Array.from(f.get('faces').data),
        nrm: f.get('normals')?.data ?? null,
        scalars: scalarsOf(f),                       // {name: Float32Array} per vertex (wall shell: thickness_mm, confidence, support, …)
    }));
    const links = v.get('links')?.data ?? null;
    return { frames, links: links ? Uint8Array.from(links) : new Uint8Array(Math.max(0, frames.length - 1)), dt: Number(v.get('dt')) || 0.04, n: frames.length };
}

/**
 * Which frame(s) a cardiac time (in frames, fractional, cyclic) shows.
 * Returns {i, j, a, linked}: show frame i, or (linked) positions blended between i and j by a ∈ [0, 1).
 * `steps` quantises the blend (fewer geometry updates while playing).
 */
export function pick(seq, tf, steps = 4) {
    const n = seq.n;
    const t = ((tf % n) + n) % n;
    let i = Math.floor(t), a = Math.round((t - i) * steps) / steps;
    if (a >= 1) { i = (i + 1) % n; a = 0; }
    const j = (i + 1) % n;
    const linked = a > 0 && i + 1 < n && seq.links[i] === 1 && seq.frames[i].pos.length === seq.frames[j].pos.length;
    if (linked) return { i, j, a, linked: true };
    return { i: a >= 0.5 ? j : i, j: a >= 0.5 ? j : i, a: 0, linked: false };
}

/** out = (1 − a) · pos_i + a · pos_j */
export function blend(seq, i, j, a, out) {
    const p = seq.frames[i].pos, q = seq.frames[j].pos;
    for (let k = 0; k < p.length; k++) out[k] = p[k] * (1 - a) + q[k] * a;
    return out;
}
