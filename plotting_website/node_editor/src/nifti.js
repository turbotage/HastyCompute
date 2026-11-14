/**
 * NIfTI-1 spatial geometry <-> hasty entry metadata. Pure functions (no DOM), unit-tested in nifti.test.mjs.
 *
 * Metadata convention (axes named x, y, z in file order, x varies fastest; tensors are C-order [.., z, y, x]):
 *   spacing_mm : [sx, sy, sz]
 *   affine     : 4x4 row-major, voxel index (i = x, j = y, k = z) -> RAS+ millimetres (NIfTI world)
 * The affine is optional for every consumer; without it the data is treated as axis-aligned RAS with spacing_mm.
 */

/** qform (quaternion b, c, d + offset + qfac) -> 4x4 affine for voxel sizes dx, dy, dz. */
export function qformToAffine(b, c, d, qx, qy, qz, dx, dy, dz, qfac) {
    let a = 1 - (b * b + c * c + d * d);
    a = a < 1e-7 ? 0 : Math.sqrt(a);
    if (qfac < 0) dz = -dz;
    return [
        [(a * a + b * b - c * c - d * d) * dx, 2 * (b * c - a * d) * dy, 2 * (b * d + a * c) * dz, qx],
        [2 * (b * c + a * d) * dx, (a * a + c * c - b * b - d * d) * dy, 2 * (c * d - a * b) * dz, qy],
        [2 * (b * d - a * c) * dx, 2 * (c * d + a * b) * dy, (a * a + d * d - c * c - b * b) * dz, qz],
        [0, 0, 0, 1],
    ];
}

/** affine -> qform (b, c, d, qfac, voxel sizes); inverse of qformToAffine for a proper rotation + scale. */
export function affineToQform(M) {
    const col = j => [M[0][j], M[1][j], M[2][j]];
    const norm = v => Math.hypot(v[0], v[1], v[2]);
    const dx = norm(col(0)), dy = norm(col(1)), dz = norm(col(2));
    const R = [0, 1, 2].map(i => [M[i][0] / dx, M[i][1] / dy, M[i][2] / dz]);
    const det = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
    let qfac = 1;
    if (det < 0) { qfac = -1; for (let i = 0; i < 3; i++) R[i][2] = -R[i][2]; }
    // rotation matrix -> quaternion
    const tr = R[0][0] + R[1][1] + R[2][2];
    let a, b, c, d;
    if (tr > 0) {
        const s = Math.sqrt(tr + 1) * 2;
        a = 0.25 * s; b = (R[2][1] - R[1][2]) / s; c = (R[0][2] - R[2][0]) / s; d = (R[1][0] - R[0][1]) / s;
    } else if (R[0][0] > R[1][1] && R[0][0] > R[2][2]) {
        const s = Math.sqrt(1 + R[0][0] - R[1][1] - R[2][2]) * 2;
        a = (R[2][1] - R[1][2]) / s; b = 0.25 * s; c = (R[0][1] + R[1][0]) / s; d = (R[0][2] + R[2][0]) / s;
    } else if (R[1][1] > R[2][2]) {
        const s = Math.sqrt(1 + R[1][1] - R[0][0] - R[2][2]) * 2;
        a = (R[0][2] - R[2][0]) / s; b = (R[0][1] + R[1][0]) / s; c = 0.25 * s; d = (R[1][2] + R[2][1]) / s;
    } else {
        const s = Math.sqrt(1 + R[2][2] - R[0][0] - R[1][1]) * 2;
        a = (R[1][0] - R[0][1]) / s; b = (R[0][2] + R[2][0]) / s; c = (R[1][2] + R[2][1]) / s; d = 0.25 * s;
    }
    if (a < 0) { b = -b; c = -c; d = -d; }
    return { b, c, d, qfac, dx, dy, dz };
}

/**
 * Read spatial geometry from a NIfTI-1 header.
 * @param {DataView} view  header bytes
 * @param {boolean}  le    little endian
 * @returns {{spacing_mm: number[], affine: number[][], source: 'sform'|'qform'|'pixdim'}}
 */
export function readGeometry(view, le) {
    const f = o => view.getFloat32(o, le);
    const unit = view.getUint8(123) & 7; // 1 m, 2 mm, 3 um
    const us = unit === 1 ? 1000 : unit === 3 ? 1e-3 : 1;
    const qcode = view.getInt16(252, le), scode = view.getInt16(254, le);
    const pix = [f(80) * us, f(84) * us, f(88) * us].map(v => (v > 0 ? v : 1));
    let affine, source;
    if (scode > 0) {
        affine = [[f(280), f(284), f(288), f(292)], [f(296), f(300), f(304), f(308)], [f(312), f(316), f(320), f(324)], [0, 0, 0, 1]];
        source = 'sform';
        const sp = [0, 1, 2].map(j => Math.hypot(affine[0][j], affine[1][j], affine[2][j]));
        if (sp.every(v => v > 0)) pix.splice(0, 3, ...sp);
    } else if (qcode > 0) {
        const qfac = f(76) < 0 ? -1 : 1;
        affine = qformToAffine(f(256), f(260), f(264), f(268) * us, f(272) * us, f(276) * us, pix[0], pix[1], pix[2], qfac);
        source = 'qform';
    } else {
        affine = [[pix[0], 0, 0, 0], [0, pix[1], 0, 0], [0, 0, pix[2], 0], [0, 0, 0, 1]];
        source = 'pixdim';
    }
    return { spacing_mm: pix, affine, source };
}

/**
 * Write spatial geometry into a NIfTI-1 header: pixdim, units (mm), sform (code 2) and the matching qform (code 1).
 * `geom` = {spacing_mm?, affine?}; whichever is missing is derived from the other, default 1 mm axis-aligned.
 */
export function writeGeometry(view, geom) {
    const sp = geom?.spacing_mm ?? [1, 1, 1];
    const A = geom?.affine ?? [[sp[0], 0, 0, 0], [0, sp[1], 0, 0], [0, 0, sp[2], 0], [0, 0, 0, 1]];
    const spacing = geom?.spacing_mm ?? [0, 1, 2].map(j => Math.hypot(A[0][j], A[1][j], A[2][j]));
    const w = (o, v) => view.setFloat32(o, v, true);
    for (let i = 0; i < 3; i++) w(80 + i * 4, spacing[i]);
    view.setUint8(123, 2); // xyzt_units: mm
    view.setInt16(254, geom?.affine ? 2 : 0, true); // sform_code: aligned
    for (let i = 0; i < 3; i++) for (let j = 0; j < 4; j++) w(280 + i * 16 + j * 4, A[i][j]);
    const q = affineToQform(A);
    view.setInt16(252, 1, true); // qform_code: scanner
    w(76, q.qfac); w(256, q.b); w(260, q.c); w(264, q.d); w(268, A[0][3]); w(272, A[1][3]); w(276, A[2][3]);
}

/** Entry metadata fragment for a parsed NIfTI geometry. */
export function geometryMetadata(g) {
    return { spacing_mm: g.spacing_mm, affine: g.affine, geometry_source: g.source };
}
