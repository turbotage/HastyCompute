import assert from 'node:assert/strict';
import { parseMesh, writeOBJ, writePLY, writeSTL } from './meshio.js';

// a tetrahedron: 4 vertices, 4 faces
const m = { vertices: Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]), faces: Uint32Array.from([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]) };
const ab = (u8) => u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
const same = (a, b) => { assert.deepEqual([...a.vertices], [...b.vertices]); assert.deepEqual([...a.faces], [...b.faces]); };
same(parseMesh(ab(writeOBJ(m)), 'a.obj'), m);
same(parseMesh(ab(writePLY(m)), 'a.ply'), m);
const stl = parseMesh(ab(writeSTL(m)), 'a.stl');           // vertices are welded back to 4 (order of first use)
assert.equal(stl.vertices.length, 12);
assert.equal(stl.faces.length, 12);
// ASCII STL + ASCII PLY + OBJ with quads, negative indices and v/vt/vn
const asciiStl = 'solid t\n facet normal 0 0 1\n  outer loop\n   vertex 0 0 0\n   vertex 1 0 0\n   vertex 0 1 0\n  endloop\n endfacet\nendsolid t\n';
const s2 = parseMesh(new TextEncoder().encode(asciiStl).buffer, 'x.stl');
assert.equal(s2.faces.length, 3);
const ply = 'ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n';
assert.deepEqual([...parseMesh(new TextEncoder().encode(ply).buffer, 'x.ply').faces], [0, 1, 2]);
const quad = parseMesh(new TextEncoder().encode('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1/1/1 2/2/2 3/3/3 4/4/4\nf -4 -3 -2\n').buffer, 'q.obj');
assert.deepEqual([...quad.faces], [0, 1, 2, 0, 2, 3, 0, 1, 2]);
// malformed input is rejected, not trusted
assert.throws(() => parseMesh(new TextEncoder().encode('v 0 0 0\nf 1 2 3\n').buffer, 'bad.obj'), /missing vertex/);
assert.throws(() => parseMesh(new ArrayBuffer(10), 'bad.stl'), /STL/);
assert.throws(() => parseMesh(new TextEncoder().encode('ply\nformat binary_little_endian 1.0\nelement vertex 100\nproperty float x\nproperty float y\nproperty float z\nend_header\n').buffer, 'b.ply'));
assert.throws(() => parseMesh(new ArrayBuffer(4), 'x.xyz'), /unsupported/);
console.log('meshio ok');
