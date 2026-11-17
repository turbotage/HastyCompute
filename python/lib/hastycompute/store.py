"""Array files with a precision policy (Python side of `hasty_value::store`; policy: research/io_policy.md).

    from hastycompute.store import load_any, save_array, resolve
    a = load_any("run/recon_x.npy")          # .npy, or the .hv written instead of it (same array as np.load gave)
    save_array("run/x.npy", a, "image")       # -> run/x.hv: quantisation noise rms = 1e-4 · rms(a)

Single-array `.hv` files hold {"array": chunked tensor, "order": "C" | "F", "loss": "<json>"}; a Fortran-ordered
array is stored as the C-order tensor of reversed shape (no copy), and comes back Fortran-ordered as from `np.load`.

Policies (float32 / float64 / complex only; integer, bool, float16 data and float arrays holding small integers are
always stored losslessly; chunks with NaN / inf are stored losslessly):
- "lossless";
- "image" or ("image", rel): real data: step = rel·√12·rms (default rel 1e-4), quantisation noise rms = rel·rms,
  |err| ≤ step/2; complex data: mantissa rounding to ⌈log2(1/rel)⌉ − 1 bits (pointwise relative ≤ rel: phase);
- ("uniform", rel): the uniform step for any float dtype (checkpoints);
- ("relative", rel, floor): asinh transfer, knee k = floor·rms: |err| ≲ rel·√(k² + x²);
- ("abs", step): |err| ≤ step/2 (k-space: kspace_step(sigma, rms));
- ("bits", B): per chunk B-bit signed range of the chunk's max |x|;
- ("mantissa", m): keep m mantissa bits (pointwise relative error ≤ 2^-(m+1)).
Environment: HASTY_IO=npy writes plain .npy instead; HASTY_IO_LOSSLESS=1 disables quantisation; HASTY_IO_THREADS.
"""
import json
import os
import struct
from pathlib import Path

import numpy as np

from . import value_format as vf

IMAGE_REL = 1e-4
KSPACE_NOISE_FRAC = 1.0 / 8.0
KSPACE_REL = 1e-4
AUTO_CHUNK_BYTES = 4 << 20
GROUP_BYTES = 256 << 20


def kspace_step(sigma, rms):
    """uniform k-space step for complex data of noise std `sigma` and rms `rms` per complex sample (Rust: same rule)"""
    a = KSPACE_REL * 12 ** 0.5 * rms / 2 ** 0.5
    return min(KSPACE_NOISE_FRAC * sigma / 2 ** 0.5, a) if sigma > 0 else a


def resolve(path):
    """`path` if it exists, else the same name with the other extension (.npy <-> .hv); None if neither exists"""
    p = Path(path)
    if p.exists():
        return p
    alt = {".npy": ".hv", ".hv": ".npy"}.get(p.suffix)
    if alt and p.with_suffix(alt).exists():
        return p.with_suffix(alt)
    return None


def exists_any(path):
    return resolve(path) is not None


def glob_any(pattern):
    """glob for .npy and .hv versions of a '*.npy' (or '*.hv') pattern; one path per stem, .hv preferred"""
    import glob

    base = str(pattern)
    stem = base[:-4] if base.endswith(".npy") else base[:-3] if base.endswith(".hv") else base
    out = {}
    for ext in (".npy", ".hv"):
        for f in glob.glob(stem + ext):
            out[f[: -len(ext)]] = f if ext == ".hv" or f[: -len(ext)] not in out else out[f[: -len(ext)]]
    return sorted(out.values())


def _is_hv(p):
    with open(p, "rb") as f:
        return f.read(4) == vf.MAGIC


def _lazy_to_value(v):
    if isinstance(v, vf.LazyTensor):
        return v.read()
    if isinstance(v, dict):
        return {k: _lazy_to_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_lazy_to_value(x) for x in v]
    if isinstance(v, tuple):
        return tuple(_lazy_to_value(x) for x in v)
    return v


class LazyArray:
    """Read-on-index view of a single-array .hv file, what `load_any(path, mmap_mode=...)` returns for .hv (the
    counterpart of a memory map): `a[i]`, `a[a:b, ..., k]` decode only the chunks they overlap (k-space: one readout
    per chunk); other indexing (index arrays, masks, None) and `np.asarray(a)` read the whole array. Shape, dtype and
    index order are those of the array as np.load returned it (Fortran-ordered arrays included)."""

    def __init__(self, t, order="C"):
        self.t, self.order = t, order
        self.dtype = t.dtype
        self.shape = tuple(reversed(t.shape)) if order == "F" else tuple(t.shape)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape)) if self.shape else 1
        self.nbytes = self.size * self.dtype.itemsize

    def __repr__(self):
        return f"LazyArray({self.dtype}, {self.shape}, order {self.order})"

    def __len__(self):
        return self.shape[0]

    def read(self):
        a = self.t.read()
        return a.T if self.order == "F" else a

    def __array__(self, dtype=None, copy=None):
        a = self.read()
        return a if dtype is None else a.astype(dtype)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if not all(isinstance(k, (slice, int, np.integer)) or k is Ellipsis for k in key) or sum(k is Ellipsis for k in key) > 1:
            return self.read()[key]
        if any(k is Ellipsis for k in key):
            i = key.index(Ellipsis)
            key = key[:i] + (slice(None),) * (self.ndim - len(key) + 1) + key[i + 1:]
        key = key + (slice(None),) * (self.ndim - len(key))
        if self.order == "F":
            r = self.t[tuple(reversed(key))]
            return r.T
        return self.t[key]


def load_any(path, mmap_mode=None, lazy=False, **kw):
    """np.load for .npy / .npz, and single-array or tree .hv files (resolving .npy <-> .hv). For a single-array .hv,
    `mmap_mode` (or `lazy`) gives a LazyArray (chunks decoded on indexing, like a memory map); otherwise arrays are
    read whole (multithreaded)."""
    p = resolve(path)
    if p is None:
        raise FileNotFoundError(f"{path} (nor its .npy / .hv sibling)")
    if not _is_hv(p):
        return np.load(p, mmap_mode=mmap_mode, **kw)
    tree = vf.open_lazy(p)
    if isinstance(tree, vf.LazyTensor):
        return LazyArray(tree) if (lazy or mmap_mode) else tree.read()
    if isinstance(tree, dict) and isinstance(tree.get("array"), vf.LazyTensor):
        t, order = tree["array"], tree.get("order", "C")
        if lazy or mmap_mode:
            return LazyArray(t, order)
        a = t.read()
        return a.T if order == "F" else a
    return _lazy_to_value(tree)


def load_meta(path):
    """the "loss" record of a single-array .hv file (dict), {} for .npy"""
    p = resolve(path)
    if p is None or not _is_hv(p):
        return {}
    tree = vf.open_lazy(p)
    return json.loads(tree.get("loss", "{}")) if isinstance(tree, dict) else {}


# ── writer ──

def _scalars(a):
    """float view of the scalar components (complex: interleaved re, im)"""
    return a.view(np.float32 if a.dtype in (np.dtype("<f4"), np.dtype("<c8")) else np.float64).reshape(-1)


def _scan(flat, piece=1 << 24):
    """rms / max over the finite scalars, and whether all are small integers (in pieces: bounded memory)"""
    n, ss, mx, integral = 0, 0.0, 0.0, True
    for o in range(0, flat.size, piece):
        x = flat[o:o + piece].astype(np.float64)
        x = x[np.isfinite(x)]
        if x.size == 0:
            continue
        n += x.size
        ss += float(np.dot(x, x))
        mx = max(mx, float(np.abs(x).max()))
        integral = integral and bool(np.all((x == np.round(x)) & (np.abs(x) < 16777216.0)))
    return dict(n=n, rms=(ss / n) ** 0.5 if n else 0.0, max=mx, integral=integral)


def _norm_policy(policy):
    if isinstance(policy, str):
        return (policy,)
    return tuple(policy)


def resolve_policy(policy, a):
    """(quantiser dict or None, record) for policy on array a (as hasty_value::store::resolve_policy)"""
    pol = _norm_policy(policy)
    if os.environ.get("HASTY_IO_LOSSLESS", "") in ("1", "true", "yes") or pol[0] == "lossless":
        return None, {"policy": "lossless", "why": "requested"}
    if a.dtype not in (np.dtype("<f4"), np.dtype("<f8"), np.dtype("<c8"), np.dtype("<c16")):
        return None, {"policy": "lossless", "why": "not a float32/64 or complex dtype"}
    st = _scan(_scalars(np.ascontiguousarray(a)))
    if pol[0] in ("image", "uniform", "relative") and (st["n"] == 0 or st["max"] == 0.0):
        return None, {"policy": "lossless", "why": "no nonzero finite values"}
    if pol[0] in ("image", "uniform", "relative") and st["integral"]:
        return None, {"policy": "lossless", "why": "integer-valued (mask / labels)"}
    if pol[0] == "image" and a.dtype.kind == "c":
        # complex: pointwise-relative mantissa rounding (the phase of low-magnitude voxels survives)
        rel = pol[1] if len(pol) > 1 else IMAGE_REL
        lim = 52 if a.dtype == np.complex128 else 23
        m = int(min(lim, max(1, int(np.ceil(np.log2(1.0 / rel))) - 1)))
        return {"kind": "mantissa", "m": m}, {"policy": f"Image {{ rel: {rel} }}", "mantissa_bits": m, "max_rel_err": 0.5 ** (m + 1), "why": "complex: pointwise relative (phase)"}
    if pol[0] in ("image", "uniform"):
        rel = pol[1] if len(pol) > 1 else IMAGE_REL
        step = rel * 12 ** 0.5 * st["rms"]
        return {"kind": "abs", "step": step, "knee": 0.0}, {"policy": f"Image {{ rel: {rel} }}", "rms": st["rms"], "step": step, "max_abs_err": step / 2}
    if pol[0] == "relative":
        rel, floor = pol[1], pol[2]
        k = max(floor * st["rms"], 1e-300)
        return {"kind": "abs", "step": 2 * rel * k, "knee": k}, {"policy": f"Relative {{ rel: {rel}, floor: {floor} }}", "rms": st["rms"], "knee": k, "step": 2 * rel * k}
    if pol[0] == "abs":
        return {"kind": "abs", "step": float(pol[1]), "knee": 0.0}, {"policy": f"Abs({pol[1]})", "step": float(pol[1]), "max_abs_err": pol[1] / 2}
    if pol[0] == "bits":
        return {"kind": "bits", "bits": int(pol[1]), "knee": 0.0}, {"policy": f"Bits({pol[1]})", "bits": int(pol[1])}
    if pol[0] == "mantissa":
        return {"kind": "mantissa", "m": int(pol[1])}, {"policy": f"Mantissa({pol[1]})", "max_rel_err": 0.5 ** (int(pol[1]) + 1)}
    raise ValueError(f"unknown policy {policy}")


def chunk_shape(shape, itemsize, target=AUTO_CHUNK_BYTES, rows=None):
    """contiguous slabs [1, …, 1, c, full, …] of about `target` bytes (or `rows` entries of axis 0)"""
    shape = list(shape)
    if not shape:
        return []
    if rows is not None:
        return [max(1, min(rows, max(shape[0], 1)))] + [max(1, s) for s in shape[1:]]
    ch = [1] * len(shape)
    inner = itemsize
    for d in range(len(shape) - 1, -1, -1):
        n = max(shape[d], 1)
        if inner * n <= target:
            ch[d] = n
            inner *= n
        else:
            ch[d] = max(1, min(n, target // inner))
            break
    return ch


def _round_mantissa(flat, m):
    if flat.dtype == np.float32:
        b, drop, ebits, emask = flat.view(np.uint32), 23 - m, 23, 0xFF
        U = np.uint32
    else:
        b, drop, ebits, emask = flat.view(np.uint64), 52 - m, 52, 0x7FF
        U = np.uint64
    if drop <= 0:
        return flat
    one = U(1)
    r = (b + (one << U(drop - 1)) - one + ((b >> U(drop)) & one)) & ~((one << U(drop)) - one)
    special = ((b >> U(ebits)) & U(emask)) == U(emask)
    return np.where(special, b, r).astype(U).view(flat.dtype)


def _encode_chunk(raw_arr, code, q, codec, level, bs, delta):
    """C-order chunk -> its bytes in the file (tag 9 chunk, or tag 10 prefix + blocks)"""
    word = vf.DTYPES[code][1]

    def blocks(b, w, integer):
        return b"".join(vf._encode_block(b[o:o + bs], w, integer, codec, level, True, delta, False) for o in range(0, len(b), bs)) if b else b""

    raw = raw_arr.tobytes()
    if q is None or q["kind"] == "mantissa":
        if q is not None:
            raw = _round_mantissa(_scalars(raw_arr), q["m"]).tobytes()
        return blocks(raw, word, code in vf.INTEGER)
    flat = _scalars(raw_arr).astype(np.float64)
    lossless = struct.pack("<BBHdd", 0, 0, 0, 0.0, 0.0) + blocks(raw, word, False)
    if flat.size == 0 or not np.all(np.isfinite(flat)):
        return lossless
    knee = q.get("knee", 0.0)
    u = knee * np.arcsinh(flat / knee) if knee > 0 else flat
    if q["kind"] == "bits":
        mx = float(np.abs(u).max())
        step = mx / ((1 << (q["bits"] - 1)) - 1) if mx > 0 else 1.0
    else:
        step = q["step"]
    qmax = float(np.abs(u).max()) / step
    if not (np.isfinite(step) and step > 0 and qmax < 2.0e9):
        return lossless
    v = u * (1.0 / step)
    qi = np.trunc(v + np.copysign(0.5, v)).astype(np.int64)
    z = ((qi << 1) ^ (qi >> 63)).astype(np.uint64)
    zb = 2 * int(qmax + 0.5 + 1e-6)
    qw = 1 if zb < (1 << 8) else 2 if zb < (1 << 16) else 4
    words = z.astype(vf._UINT[qw]).astype(f"<u{qw}").tobytes()
    return struct.pack("<BBHdd", qw, 1 if knee > 0 else 0, 0, step, knee) + blocks(words, qw, True)


def save_array(path, a, policy="image", codec="zstd", level=1, rows=None, chunk_bytes=AUTO_CHUNK_BYTES, delta=False, chunks=None, drop_stale=True):
    """Save array `a` with a precision policy. Writes `path` with suffix .hv (or .npy with HASTY_IO=npy); returns the
    path. Fortran-ordered arrays are stored without a copy (order "F"). `rows`: entries of axis 0 per chunk (k-space:
    1 readout); `chunks`: explicit C-order chunk shape of the stored tensor (e.g. [1, n_adc, 1]: one field of one
    readout, so every field gets its own scale under ("bits", B)); `delta`: delta filter on quantised words."""
    a = np.asarray(a)
    p = Path(path)
    if os.environ.get("HASTY_IO", "") == "npy":
        p = p.with_suffix(".npy")
        np.save(p, a)
        if p.with_suffix(".hv").exists():  # an older version of the same array would shadow nothing, but mislead
            p.with_suffix(".hv").unlink()
        return p
    p = p.with_suffix(".hv")
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    order = "F" if (a.flags.f_contiguous and not a.flags.c_contiguous and a.ndim > 1) else "C"
    c = a.T if order == "F" else np.ascontiguousarray(a)  # C-order view of the stored tensor
    if c.ndim == 0:
        c = c.reshape(1)
    code = vf.CODE_OF.get((c.dtype.kind, c.dtype.itemsize))
    if code is None:
        raise TypeError(f"unsupported dtype {c.dtype}")
    q, rec = resolve_policy(policy, c)
    shape = list(c.shape)
    ch = [max(1, min(int(k), max(n, 1))) for k, n in zip(chunks, shape)] if chunks is not None else chunk_shape(shape, c.dtype.itemsize, chunk_bytes, rows)
    grid = [-(-s // k) for s, k in zip(shape, ch)]
    n = int(np.prod(grid)) if all(s > 0 for s in shape) else 0
    bs = 1 << 20
    tag = vf.TAG_QCHUNKED if q is not None and q["kind"] in ("abs", "bits") else vf.TAG_CHUNKED
    head = vf.MAGIC + struct.pack("<BBH", vf.VERSION, 0, 0) + struct.pack("<BQ", vf.TAG_DICT, 3) + struct.pack("<I", 5) + b"array"
    head += struct.pack("<BBBBI", tag, code, len(shape), 0, bs) + struct.pack(f"<{2 * len(shape)}Q", *shape, *ch) + struct.pack("<Q", n)
    tmp = p.with_name(p.name + ".tmp")
    index = np.zeros((n, 2), "<u8")

    def enc(ci):
        lo, ext = vf._chunk_box(ci, grid, ch, shape)
        sub = np.ascontiguousarray(c[tuple(slice(x, x + e) for x, e in zip(lo, ext))])
        return _encode_chunk(sub, code, q, codec, level, bs, delta)

    per = max(1, int(np.prod(ch)) * c.dtype.itemsize)
    group = max(1, GROUP_BYTES // per)
    from concurrent.futures import ThreadPoolExecutor

    with open(tmp, "wb") as f, ThreadPoolExecutor(vf._threads()) as ex:
        f.write(head)
        ipos = f.tell()
        f.write(b"\0" * (16 * n))
        off = 0
        for g in range(0, n, group):
            for k, b in enumerate(ex.map(enc, range(g, min(n, g + group)))):
                index[g + k] = (off, len(b))
                off += len(b)
                f.write(b)
        for k, v in (("order", order), ("loss", json.dumps(rec))):
            kb, vb = k.encode(), v.encode()
            f.write(struct.pack("<I", len(kb)) + kb + struct.pack("<BQ", vf.TAG_STR, len(vb)) + vb)
        f.seek(ipos)
        f.write(index.tobytes())
    os.replace(tmp, p)
    # the array was rewritten: an older .npy of the same name would be found first by resolve() / np.load
    # (drop_stale=False: the caller owns the .npy, e.g. convert.py deletes it only after verifying the .hv)
    if drop_stale and p.with_suffix(".npy").exists():
        p.with_suffix(".npy").unlink()
    return p
