"""hasty value format v1 (spec: rustlib/crates/hasty-value/FORMAT.md): streaming, block-compressed value trees.

Python mapping: None, bool, int, float, str, numpy.ndarray (C-order; torch tensors are converted on encode), list,
tuple, dict (str keys, order kept). Codecs: stored, LZ4 (package `lz4`), zstd (package `zstandard`).

    data = encode(value, codec="lz4")          # bytes
    for chunk in iter_encode(value): ...       # streaming
    value = decode(data)
    value = decode_chunks(iterable_of_bytes)   # e.g. a gRPC response stream
    write("x.hv", value, codec="zstd"); value = read("x.hv")
    tree = open_lazy("case.hv"); tree["x"][10:40, :, :, 3:5]   # random access (chunked tensors decode only what is needed)
"""
import struct
import zlib

import numpy as np

MAGIC = b"HSTV"
VERSION = 1
TAG_NONE, TAG_BOOL, TAG_INT, TAG_FLOAT, TAG_STR, TAG_LIST, TAG_TUPLE, TAG_DICT, TAG_TENSOR, TAG_CHUNKED = range(10)
CODEC_NONE, CODEC_LZ4, CODEC_ZSTD = 0, 1, 2
FILTER_SHUFFLE, FILTER_DELTA = 1, 2
FLAG_CRC = 1

# wire code -> (numpy dtype, filter word size)
DTYPES = {
    0: (np.dtype("bool"), 1), 1: (np.dtype("uint8"), 1), 2: (np.dtype("int8"), 1), 3: (np.dtype("<u2"), 2),
    4: (np.dtype("<i2"), 2), 5: (np.dtype("<u4"), 4), 6: (np.dtype("<i4"), 4), 7: (np.dtype("<u8"), 8),
    8: (np.dtype("<i8"), 8), 9: (np.dtype("<f2"), 2), 10: (np.dtype("<u2"), 2),  # bfloat16: raw bits as uint16
    11: (np.dtype("<f4"), 4), 12: (np.dtype("<f8"), 8), 13: (np.dtype("<c8"), 4), 14: (np.dtype("<c16"), 8),
}
CODE_OF = {(d.kind, d.itemsize): c for c, (d, _) in DTYPES.items() if c != 10}  # bfloat16 has no numpy dtype
NAMES = ["bool", "uint8", "int8", "uint16", "int16", "uint32", "int32", "uint64", "int64", "float16", "bfloat16", "float32", "float64", "complex64", "complex128"]
INTEGER = {0, 1, 2, 3, 4, 5, 6, 7, 8}


def _shuffle(b: bytes, w: int) -> bytes:
    a = np.frombuffer(b, np.uint8)
    n = len(a) // w
    return a[: n * w].reshape(n, w).T.tobytes() + a[n * w:].tobytes()


def _unshuffle(b: bytes, w: int) -> bytes:
    a = np.frombuffer(b, np.uint8)
    n = len(a) // w
    return a[: n * w].reshape(w, n).T.tobytes() + a[n * w:].tobytes()


_UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def _delta(b: bytes, w: int, encode: bool) -> bytes:
    n = len(b) // w
    x = np.frombuffer(b, _UINT[w], count=n)
    if encode:
        y = x.copy()
        y[1:] = x[1:] - x[:-1]
    else:
        y = np.cumsum(x, dtype=_UINT[w])
    return y.astype(f"<u{w}").tobytes() + b[n * w:]


def _compress(codec, level, data):
    if codec == "lz4":
        import lz4.block

        return CODEC_LZ4, lz4.block.compress(data, store_size=False)
    if codec == "zstd":
        import zstandard

        return CODEC_ZSTD, zstandard.ZstdCompressor(level=level or 3).compress(data)
    return CODEC_NONE, data


def _encode_block(raw: bytes, word: int, integer: bool, codec, level, shuffle, delta, checksum) -> bytes:
    filt, work = 0, raw
    if codec != "none":
        if delta and integer and word in _UINT:
            work, filt = _delta(work, word, True), filt | FILTER_DELTA
        if shuffle and word > 1:
            work, filt = _shuffle(work, word), filt | FILTER_SHUFFLE
    c, payload = _compress(codec, level, work)
    if c == CODEC_NONE or len(payload) >= len(raw):
        c, filt, payload = CODEC_NONE, 0, raw
    flags = FLAG_CRC if checksum else 0
    h = struct.pack("<BBBBII", c, filt, word, flags, len(raw), len(payload))
    if checksum:
        h += struct.pack("<I", zlib.crc32(payload))
    return h + payload


def _as_array(v):
    try:
        import torch

        if isinstance(v, torch.Tensor):
            return v.detach().cpu().contiguous().numpy()
    except ImportError:
        pass
    return v


def iter_encode(value, codec="lz4", level=0, block_size=1 << 20, shuffle=True, delta=False, checksum=False, chunk=1 << 20):
    """yield byte chunks of the encoded stream"""
    block_size = max(16, block_size // 16 * 16)
    buf = bytearray(MAGIC + struct.pack("<BBH", VERSION, 0, 0))

    def emit(force=False):
        nonlocal buf
        if len(buf) >= chunk or (force and buf):
            out, buf = bytes(buf), bytearray()
            return out
        return None

    stack = [value]
    while stack:
        v = _as_array(stack.pop())
        if isinstance(v, _Key):
            kb = v.k.encode()
            buf += struct.pack("<I", len(kb)) + kb
        elif v is None:
            buf.append(TAG_NONE)
        elif isinstance(v, (bool, np.bool_)):
            buf += struct.pack("<BB", TAG_BOOL, int(v))
        elif isinstance(v, int):
            buf += struct.pack("<Bq", TAG_INT, v)
        elif isinstance(v, float):
            buf += struct.pack("<Bd", TAG_FLOAT, v)
        elif isinstance(v, str):
            b = v.encode()
            buf += struct.pack("<BQ", TAG_STR, len(b)) + b
        elif isinstance(v, (list, tuple)):
            buf += struct.pack("<BQ", TAG_LIST if isinstance(v, list) else TAG_TUPLE, len(v))
            stack.extend(reversed(v))
        elif isinstance(v, dict):
            buf += struct.pack("<BQ", TAG_DICT, len(v))
            for k, x in reversed(list(v.items())):
                stack.append(x)
                stack.append(_Key(str(k)))
        elif isinstance(v, (np.ndarray, np.generic)):
            a = np.asarray(v, order="C")  # (ascontiguousarray would make 0-d arrays 1-d)
            code = CODE_OF.get((a.dtype.kind, a.dtype.itemsize))
            if code is None:
                raise TypeError(f"unsupported dtype {a.dtype}")
            word = DTYPES[code][1]
            raw = a.astype(DTYPES[code][0], copy=False).tobytes()  # little-endian
            buf += struct.pack("<BBBBI", TAG_TENSOR, code, a.ndim, 0, block_size)
            buf += struct.pack(f"<{a.ndim}Q", *a.shape)
            for o in range(0, len(raw), block_size):
                buf += _encode_block(raw[o:o + block_size], word, code in INTEGER, codec, level, shuffle, delta, checksum)
                c = emit()
                if c:
                    yield c
        else:
            raise TypeError(f"cannot encode {type(v).__name__}")
        c = emit()
        if c:
            yield c
    c = emit(True)
    if c:
        yield c


class _Key:
    __slots__ = ("k",)

    def __init__(self, k):
        self.k = k


def encode(value, **kw) -> bytes:
    return b"".join(iter_encode(value, **kw))


class _Reader:
    def __init__(self, chunks):
        self.it = iter(chunks)
        self.buf = bytearray()
        self.pos = 0

    def read(self, n: int) -> memoryview:
        while len(self.buf) - self.pos < n:
            try:
                c = next(self.it)
            except StopIteration:
                raise ValueError("truncated stream") from None
            if self.pos > (1 << 23):
                del self.buf[: self.pos]
                self.pos = 0
            self.buf += c
        m = memoryview(self.buf)[self.pos:self.pos + n]
        self.pos += n
        return m

    def at_end(self) -> bool:
        if len(self.buf) > self.pos:
            return False
        for c in self.it:
            if c:
                self.buf += c
                return False
        return True


def _decode_block(h, payload: bytes, out: memoryview):
    codec, filt, word, flags, raw_len, comp_len = h
    if codec == CODEC_NONE:
        data = payload
    elif codec == CODEC_LZ4:
        import lz4.block

        data = lz4.block.decompress(payload, uncompressed_size=raw_len)
    elif codec == CODEC_ZSTD:
        import zstandard

        data = zstandard.ZstdDecompressor().decompress(payload, max_output_size=raw_len)
    else:
        raise ValueError(f"unknown codec {codec}")
    if len(data) != raw_len:
        raise ValueError("block length mismatch")
    if filt & FILTER_SHUFFLE:
        data = _unshuffle(data, word)
    if filt & FILTER_DELTA:
        data = _delta(data, word, False)
    out[:] = data


def _value(r: _Reader):
    tag = r.read(1)[0]
    if tag == TAG_NONE:
        return None
    if tag == TAG_BOOL:
        return bool(r.read(1)[0])
    if tag == TAG_INT:
        return struct.unpack("<q", r.read(8))[0]
    if tag == TAG_FLOAT:
        return struct.unpack("<d", r.read(8))[0]
    if tag == TAG_STR:
        (n,) = struct.unpack("<Q", r.read(8))
        return bytes(r.read(n)).decode()
    if tag in (TAG_LIST, TAG_TUPLE):
        (n,) = struct.unpack("<Q", r.read(8))
        items = [_value(r) for _ in range(n)]
        return items if tag == TAG_LIST else tuple(items)
    if tag == TAG_DICT:
        (n,) = struct.unpack("<Q", r.read(8))
        d = {}
        for _ in range(n):
            (k,) = struct.unpack("<I", r.read(4))
            key = bytes(r.read(k)).decode()
            d[key] = _value(r)
        return d
    if tag == TAG_TENSOR:
        code, ndim, _, block_size = struct.unpack("<BBBI", r.read(7))
        shape = struct.unpack(f"<{ndim}Q", r.read(8 * ndim)) if ndim else ()
        dt = DTYPES[code][0]
        nbytes = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
        out = bytearray(nbytes)
        mv = memoryview(out)
        off = 0
        while off < nbytes:
            h = struct.unpack("<BBBBII", r.read(12))
            crc = struct.unpack("<I", r.read(4))[0] if h[3] & FLAG_CRC else None
            if h[4] != min(block_size, nbytes - off):
                raise ValueError("block raw length mismatch")
            payload = bytes(r.read(h[5]))
            if crc is not None and zlib.crc32(payload) != crc:
                raise ValueError("block checksum mismatch")
            _decode_block(h, payload, mv[off:off + h[4]])
            off += h[4]
        return np.frombuffer(out, dt).reshape(shape)
    if tag == TAG_CHUNKED:
        code, ndim, _, block_size = struct.unpack("<BBBI", r.read(7))
        shape = struct.unpack(f"<{ndim}Q", r.read(8 * ndim)) if ndim else ()
        chunk = struct.unpack(f"<{ndim}Q", r.read(8 * ndim)) if ndim else ()
        (n,) = struct.unpack("<Q", r.read(8))
        index = np.frombuffer(bytes(r.read(16 * n)), "<u8").reshape(n, 2) if n else np.zeros((0, 2), "<u8")
        data = bytes(r.read(int(index[:, 1].sum())))
        dt = DTYPES[code][0]
        out = np.zeros(shape, dt)
        grid = [-(-s // c) for s, c in zip(shape, chunk)]
        for ci in range(n):
            lo, ext = _chunk_box(ci, grid, chunk, shape)
            raw = _decode_blocks(memoryview(data)[index[ci, 0]:index[ci, 0] + index[ci, 1]], int(np.prod(ext)) * dt.itemsize, block_size)
            out[tuple(slice(a, a + e) for a, e in zip(lo, ext))] = np.frombuffer(raw, dt).reshape(ext)
        return out
    raise ValueError(f"unknown value tag {tag}")


def _chunk_box(ci, grid, chunk, shape):
    lo = [0] * len(grid)
    for d in range(len(grid) - 1, -1, -1):
        lo[d] = (ci % grid[d]) * chunk[d]
        ci //= grid[d]
    return lo, [min(c, s - a) for c, s, a in zip(chunk, shape, lo)]


def _decode_blocks(buf, raw_len: int, block_size: int) -> bytearray:
    """the consecutive blocks of a plain tensor / chunk in `buf` -> raw bytes"""
    out = bytearray(raw_len)
    mv = memoryview(out)
    p = off = 0
    while off < raw_len:
        h = struct.unpack_from("<BBBBII", buf, p)
        p += 12
        crc = None
        if h[3] & FLAG_CRC:
            (crc,) = struct.unpack_from("<I", buf, p)
            p += 4
        if h[4] != min(block_size, raw_len - off):
            raise ValueError("block raw length mismatch")
        payload = bytes(buf[p:p + h[5]])
        if crc is not None and zlib.crc32(payload) != crc:
            raise ValueError("block checksum mismatch")
        _decode_block(h, payload, mv[off:off + h[4]])
        p += h[5]
        off += h[4]
    return out


_CAPI = None
_HANDLES = {}


def _capi():
    """the Rust reader (rustlib/target/release/libhasty_value_capi.so or HASTY_VALUE_LIB), or None"""
    global _CAPI
    if _CAPI is None:
        import ctypes
        import os
        from pathlib import Path

        _CAPI = False
        for c in [os.environ.get("HASTY_VALUE_LIB"), str(Path(__file__).resolve().parents[3] / "rustlib/target/release/libhasty_value_capi.so")]:
            if c and Path(c).exists():
                lib = ctypes.CDLL(c)
                vp, cp, u64p = ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)
                lib.hv_open.argtypes, lib.hv_open.restype = [cp, ctypes.c_uint32], vp
                lib.hv_close.argtypes, lib.hv_close.restype = [vp], None
                lib.hv_read_box.argtypes, lib.hv_read_box.restype = [vp, cp, u64p, u64p, vp, ctypes.c_uint64], ctypes.c_int32
                lib.hv_last_error.argtypes, lib.hv_last_error.restype = [ctypes.c_char_p, ctypes.c_uint32], ctypes.c_int32
                _CAPI = lib
                break
    return _CAPI or None


def _handle(path):
    """per-process handle (DataLoader workers are forked: never reuse the parent's thread pool)"""
    import os

    key = (os.getpid(), str(path))
    if key not in _HANDLES:
        h = _capi().hv_open(str(path).encode(), int(os.environ.get("HASTY_VALUE_THREADS", "4")))
        if not h:
            raise OSError(_capi_error())
        _HANDLES[key] = h
    return _HANDLES[key]


def _capi_error():
    import ctypes

    b = ctypes.create_string_buffer(1024)
    _capi().hv_last_error(b, 1024)
    return b.value.decode()


class LazyTensor:
    """A tensor in a memory-mapped .hv file. `t[...]` / `read_box(lo, hi)` read only what is needed (chunked tensors
    decode just the overlapping chunks); `read()` reads everything."""

    def __init__(self, mm, code, shape, block_size, offset, end=None, chunk=None, index=None, file=None, key=""):
        self.mm, self.code, self.shape, self.block_size = mm, code, tuple(shape), block_size
        self.file, self.key = file, key
        self.dtype = DTYPES[code][0]
        self.offset, self.end, self.chunk, self.index = offset, end, chunk, index
        self.grid = [-(-s // c) for s, c in zip(self.shape, chunk)] if chunk else None
        self.ndim = len(self.shape)

    def __repr__(self):
        return f"LazyTensor({self.dtype}, {self.shape}{', chunks ' + str(tuple(self.chunk)) if self.chunk else ''})"

    def read(self):
        return self.read_box([0] * self.ndim, list(self.shape))

    def read_box(self, lo, hi):
        lo, hi = [int(x) for x in lo], [int(x) for x in hi]
        ext = [b - a for a, b in zip(lo, hi)]
        if any(e < 0 for e in ext) or any(b > s for b, s in zip(hi, self.shape)):
            raise IndexError(f"box {lo}..{hi} outside {self.shape}")
        if self.file is not None and _capi() is not None:  # Rust: parallel, multi-GB/s
            import ctypes

            out = np.empty(ext, self.dtype)
            arr = ctypes.c_uint64 * max(self.ndim, 1)
            rc = _capi().hv_read_box(_handle(self.file), self.key.encode(), arr(*lo), arr(*hi), out.ctypes.data_as(ctypes.c_void_p), out.nbytes)
            if rc != 0:
                raise OSError(_capi_error())
            return out
        if self.chunk is None:
            raw = _decode_blocks(self.mm[self.offset:self.end], int(np.prod(self.shape)) * self.dtype.itemsize, self.block_size)
            full = np.frombuffer(raw, self.dtype).reshape(self.shape)
            return np.array(full[tuple(slice(a, b) for a, b in zip(lo, hi))])
        out = np.zeros(ext, self.dtype)
        if any(e == 0 for e in ext):
            return out
        ranges = [range(a // c, (b - 1) // c + 1) for a, b, c in zip(lo, hi, self.chunk)]
        for g in np.ndindex(*[len(r) for r in ranges]):
            gi = [r[k] for r, k in zip(ranges, g)]
            ci = 0
            for x, n in zip(gi, self.grid):
                ci = ci * n + x
            clo, cext = _chunk_box(ci, self.grid, self.chunk, self.shape)
            off, ln = int(self.index[ci, 0]), int(self.index[ci, 1])
            raw = _decode_blocks(self.mm[self.offset + off:self.offset + off + ln], int(np.prod(cext)) * self.dtype.itemsize, self.block_size)
            cdat = np.frombuffer(raw, self.dtype).reshape(cext)
            a = [max(c, l) for c, l in zip(clo, lo)]
            b = [min(c + e, h) for c, e, h in zip(clo, cext, hi)]
            out[tuple(slice(x - l, y - l) for x, y, l in zip(a, b, lo))] = cdat[tuple(slice(x - c, y - c) for x, y, c in zip(a, b, clo))]
        return out

    def __getitem__(self, key):
        """basic numpy indexing; the bounding box is read, steps / integer indices applied afterwards"""
        if not isinstance(key, tuple):
            key = (key,)
        if any(k is Ellipsis for k in key):
            i = key.index(Ellipsis)
            key = key[:i] + (slice(None),) * (self.ndim - len(key) + 1) + key[i + 1:]
        key = key + (slice(None),) * (self.ndim - len(key))
        lo, hi, post = [], [], []
        for k, n in zip(key, self.shape):
            if isinstance(k, slice):
                a, b, st = k.indices(n)
                if st > 0:
                    lo.append(a), hi.append(max(a, b)), post.append(slice(0, max(a, b) - a, st))
                else:
                    lo.append(b + 1), hi.append(a + 1), post.append(slice(a - b - 1, None, st))
            else:
                i = int(k) + (n if int(k) < 0 else 0)
                if not 0 <= i < n:
                    raise IndexError(f"index {k} out of range for {n}")
                lo.append(i), hi.append(i + 1), post.append(0)
        return self.read_box(lo, hi)[tuple(post)]


def open_lazy(path):
    """the value tree of an .hv file with tensors as LazyTensor (memory-mapped; nothing decoded yet)"""
    import mmap

    f = open(path, "rb")
    mm = memoryview(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
    if bytes(mm[:4]) != MAGIC or mm[4] != VERSION:
        raise ValueError(f"{path}: not a hasty value file")
    pos = 8
    fpath = str(path)

    def u(fmt, n):
        nonlocal pos
        v = struct.unpack_from(fmt, mm, pos)
        pos += n
        return v

    def val(key=""):
        nonlocal pos
        tag = mm[pos]
        pos += 1
        if tag == TAG_NONE:
            return None
        if tag == TAG_BOOL:
            return bool(u("<B", 1)[0])
        if tag == TAG_INT:
            return u("<q", 8)[0]
        if tag == TAG_FLOAT:
            return u("<d", 8)[0]
        if tag == TAG_STR:
            (n,) = u("<Q", 8)
            s_ = bytes(mm[pos:pos + n]).decode()
            pos += n
            return s_
        if tag in (TAG_LIST, TAG_TUPLE):
            (n,) = u("<Q", 8)
            items = [val(f"{key}[{i}]") for i in range(n)]
            return items if tag == TAG_LIST else tuple(items)
        if tag == TAG_DICT:
            (n,) = u("<Q", 8)
            d = {}
            for _ in range(n):
                (k,) = u("<I", 4)
                name = bytes(mm[pos:pos + k]).decode()
                pos += k
                d[name] = val(f'{key}["{name}"]')
            return d
        if tag in (TAG_TENSOR, TAG_CHUNKED):
            code, ndim, _, bs = u("<BBBI", 7)
            shape = u(f"<{ndim}Q", 8 * ndim) if ndim else ()
            if tag == TAG_TENSOR:
                nbytes = int(np.prod(shape, dtype=np.int64)) * DTYPES[code][0].itemsize
                off = pos
                for _ in range(-(-nbytes // bs) if nbytes else 0):
                    h = struct.unpack_from("<BBBBII", mm, pos)
                    pos += 12 + (4 if h[3] & FLAG_CRC else 0) + h[5]
                return LazyTensor(mm, code, shape, bs, off, end=pos, file=fpath, key=key)
            chunk = u(f"<{ndim}Q", 8 * ndim) if ndim else ()
            (n,) = u("<Q", 8)
            index = np.frombuffer(mm[pos:pos + 16 * n], "<u8").reshape(n, 2)
            pos += 16 * n
            t = LazyTensor(mm, code, shape, bs, pos, chunk=chunk, index=index, file=fpath, key=key)
            pos += int(index[:, 1].sum())
            return t
        raise ValueError(f"unknown value tag {tag}")

    return val()


def decode_chunks(chunks):
    r = _Reader(chunks)
    head = bytes(r.read(8))
    if head[:4] != MAGIC:
        raise ValueError("not a hasty value stream (bad magic)")
    if head[4] != VERSION:
        raise ValueError(f"unsupported format version {head[4]}")
    v = _value(r)
    if not r.at_end():
        raise ValueError("data after the end of the value")
    return v


def decode(data: bytes):
    return decode_chunks([data])


def write(path, value, **kw):
    with open(path, "wb") as f:
        for c in iter_encode(value, **kw):
            f.write(c)


def read(path):
    def chunks():
        with open(path, "rb") as f:
            while True:
                c = f.read(8 << 20)
                if not c:
                    return
                yield c

    return decode_chunks(chunks())
