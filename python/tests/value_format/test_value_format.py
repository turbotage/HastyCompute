"""Cross-language check of the hasty value format: Rust-written files decode in Python, Python-written files decode in
Rust, digests (structure + CRC32 of tensor bytes) match. Needs `cargo build --release -p hasty-value --bin hvtool`.

    rustlib/.venv/bin/python -I python/tests/value_format/test_value_format.py
"""
import json
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "python/lib"))
from hastycompute import value_format as vf  # noqa: E402

TOOL = REPO / "rustlib/target/release/hvtool"
OUT = REPO / "rustlib/target/hv-test"


def sample():
    i = np.arange(5 * 6 * 7, dtype=np.float32)
    a = np.empty(i.shape, np.complex64)
    a.real, a.imag = i * np.float32(0.5), -i * np.float32(0.25)  # keeps −0.0 like Rust
    a = a.reshape(5, 6, 7)
    b = ((np.arange(33 * 31, dtype=np.uint64) * 7919) % 65536).astype(np.uint16).reshape(33, 31)
    k = np.arange(64 * 64)
    smooth = (np.sin((k % 64).astype(np.float32) / np.float32(9.0)) * np.cos((k // 64).astype(np.float32) / np.float32(13.0))).astype(np.float32).reshape(64, 64)
    return {
        "a": a, "b": b, "smooth": smooth, "scalar": np.array(3.5), "empty": np.zeros((0, 3), np.float32),
        "mask": np.array([True, False, True, True]), "name": "hasty ✓", "n": -42, "x": 0.125, "flag": True,
        "nested": [None, (1, "two"), {}, []],
    }


def digest(v):
    if isinstance(v, np.ndarray):
        code = vf.CODE_OF[(v.dtype.kind, v.dtype.itemsize)]
        return {"tensor": vf.NAMES[code], "shape": list(v.shape), "crc32": zlib.crc32(np.ascontiguousarray(v).tobytes())}
    if isinstance(v, list):
        return {"list": [digest(x) for x in v]}
    if isinstance(v, tuple):
        return {"tuple": [digest(x) for x in v]}
    if isinstance(v, dict):
        return {"dict": [[k, digest(x)] for k, x in v.items()]}
    if v is None:
        return {"none": None}
    if isinstance(v, bool):
        return {"bool": v}
    if isinstance(v, int):
        return {"int": v}
    if isinstance(v, float):
        return {"float": v}
    return {"str": v}


subprocess.run([str(TOOL), "gen", str(OUT)], check=True)
want = json.loads((OUT / "digest.json").read_text())
smooth_rust = vf.read(OUT / "none.hv")["smooth"]
s = sample()
# float32 sin/cos may differ in the last ulp between libm implementations: compare, then use Rust's values
assert np.abs(s["smooth"] - smooth_rust).max() < 1e-6
s["smooth"] = smooth_rust
assert digest(s) == want, "Python sample differs from the Rust sample"
for f in sorted(OUT.glob("*.hv")):
    if f.name.startswith(("py_", "js_", "chunked")):
        continue
    got = digest(vf.read(f))
    assert got == want, f"{f.name}: decoded digest differs"
    print(f"Rust -> Python  {f.name:22s} {f.stat().st_size:7d} B  ok")
for name, kw in [("none", dict(codec="none")), ("lz4", dict(codec="lz4", block_size=4096)), ("lz4_delta_crc", dict(codec="lz4", block_size=1000, delta=True, checksum=True)),
                 ("zstd", dict(codec="zstd", level=5, block_size=2048)), ("lz4_noshuffle", dict(codec="lz4", shuffle=False))]:
    p = OUT / f"py_{name}.hv"
    vf.write(p, s, **kw)
    got = json.loads(subprocess.run([str(TOOL), "digest", str(p)], check=True, capture_output=True, text=True).stdout)
    assert got == want, f"py_{name}: Rust decode differs"
    rust_size = (OUT / f"{name}.hv").stat().st_size
    print(f"Python -> Rust  py_{name + '.hv':19s} {p.stat().st_size:7d} B  ok  (Rust encoder: {rust_size} B)")
# chunked decoding with odd chunk sizes
data = vf.encode(s, block_size=512)
chunks = [data[i:i + 37] for i in range(0, len(data), 37)]
assert digest(vf.decode_chunks(chunks)) == want
print("ok")

# chunked tensors written by Rust (cargo test -p hasty-value: chunked_tensors_stream_and_random_access)
for name in ["chunked_false.hv", "chunked_true.hv"]:
    f = OUT / name
    if not f.exists():
        print(f"(skip {name}: run cargo test -p hasty-value first)")
        continue
    full = vf.read(f)
    x, y, z, fr, e = 37, 29, 21, 5, 7
    i = np.arange(x * y * z * fr * e * 2, dtype=np.uint64)
    want_x = ((((i * np.uint64(2654435761)) >> np.uint64(7)) & np.uint64(0xFFFF)) ^ ((i & np.uint64(0xFFFF)) // np.uint64(3))).astype(np.uint16).reshape(x, y, z, fr, e, 2)
    assert np.array_equal(full["x"], want_x), "chunked stream decode differs"
    lz = vf.open_lazy(f)
    assert lz["meta"] == "case" and lz["x"].chunk == (16, 16, 16, 1, 7, 2)
    for key in [np.s_[5:33, 15:18, 3:, 2:4], np.s_[36, -1, ::4, 4, 6], np.s_[..., 1], np.s_[10:11, :, :, 1:2, 2:3, :], np.s_[::-3, 2]]:
        assert np.array_equal(lz["x"][key], want_x[key]), f"lazy {key}"
    assert np.array_equal(lz["mask"].read(), full["mask"]) and np.array_equal(lz["plain"][1:, 2], full["plain"][1:, 2])
    print(f"Rust chunked -> Python  {name}: stream decode and lazy boxes ok")
print("chunked ok")
