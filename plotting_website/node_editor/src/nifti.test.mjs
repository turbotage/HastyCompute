// node --test plotting_website/node_editor/src/nifti.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { readGeometry, writeGeometry, qformToAffine, affineToQform } from './nifti.js';

const close = (a, b, tol = 1e-4) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);
const matClose = (A, B) => A.forEach((r, i) => r.forEach((v, j) => close(v, B[i][j])));
const hdr = () => { const b = new ArrayBuffer(352); new DataView(b).setInt32(0, 348, true); return new DataView(b); };

test('affine round trip through sform + qform (oblique, flipped x)', () => {
    const c = Math.cos(0.3), s = Math.sin(0.3);
    const A = [[-1.1 * c, 1.2 * -s, 0, 90], [-1.1 * s, 1.2 * c, 0, -120], [0, 0, 1.5, -60], [0, 0, 0, 1]];
    const h = hdr();
    writeGeometry(h, { affine: A });
    const g = readGeometry(h, true);
    assert.equal(g.source, 'sform');
    matClose(g.affine, A);
    [1.1, 1.2, 1.5].forEach((v, i) => close(g.spacing_mm[i], v));
    // qform alone reproduces it
    h.setInt16(254, 0, true);
    const q = readGeometry(h, true);
    assert.equal(q.source, 'qform');
    matClose(q.affine, A);
});

test('spacing only -> diagonal', () => {
    const h = hdr();
    writeGeometry(h, { spacing_mm: [0.5, 0.5, 2] });
    h.setInt16(252, 0, true);
    const g = readGeometry(h, true);
    assert.equal(g.source, 'pixdim');
    matClose(g.affine, [[0.5, 0, 0, 0], [0, 0.5, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]);
});

test('qform inverse', () => {
    const A = qformToAffine(0.1, -0.2, 0.3, 5, 6, 7, 1, 2, 3, -1);
    const q = affineToQform(A);
    matClose(qformToAffine(q.b, q.c, q.d, 5, 6, 7, q.dx, q.dy, q.dz, q.qfac), A);
});
