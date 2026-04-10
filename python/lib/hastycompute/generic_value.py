"""
Python mirror of hasty::GenericValue serialization.

Wire format matches the C++ implementation exactly:
  - Every value is prefixed by a 1-byte type tag (eType enum)
  - TENSOR:  tag | SerializedTensorHeader (16 bytes, see below) | shape (ndim*i64) | raw data
  - STRING:  tag | length (u64 LE) | utf-8 bytes
  - VECTOR:  tag | count (u64 LE) | elements...
  - TUPLE:   tag | count (u64 LE) | elements...
  - DICT:    tag | count (u64 LE) | (STRING key, value) pairs...
  - NONE:    tag only

SerializedTensorHeader layout (matches C++ struct with natural alignment):
  offset 0 : device_type  (u8)
  offset 1 : scalar_type  (u8)
  offset 2 : ndim         (u8)
  offset 3 : device_index (i8)
  offset 4-7: padding     (4 bytes, natural alignment of i64)
  offset 8 : total_elements (i64 LE)
  total: 16 bytes
"""

import struct
import io
from enum import IntEnum
from typing import Union

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Type enum — must match C++ eType
# ---------------------------------------------------------------------------

class eType(IntEnum):
    NONE   = 0
    TENSOR = 1
    VECTOR = 2
    DICT   = 3
    TUPLE  = 4
    STRING = 5


# ---------------------------------------------------------------------------
# Scalar type mappings — must match ATen scalar type enum
# ---------------------------------------------------------------------------

_TORCH_TO_SCALAR: dict[torch.dtype, int] = {
    torch.uint8:     0,
    torch.int8:      1,
    torch.int16:     2,
    torch.int32:     3,
    torch.int64:     4,
    torch.float16:   5,
    torch.float32:   6,
    torch.float64:   7,
    torch.complex64:  9,
    torch.complex128: 10,
    torch.bool:      11,
}

_SCALAR_TO_TORCH: dict[int, torch.dtype] = {v: k for k, v in _TORCH_TO_SCALAR.items()}

_SCALAR_ITEMSIZE: dict[int, int] = {
    0: 1, 1: 1, 2: 2, 3: 4, 4: 8,
    5: 2, 6: 4, 7: 8,
    8: 4, 9: 8, 10: 16,
    11: 1,
}

_NUMPY_DTYPE: dict[int, np.dtype] = {
    0:  np.dtype('uint8'),
    1:  np.dtype('int8'),
    2:  np.dtype('int16'),
    3:  np.dtype('int32'),
    4:  np.dtype('int64'),
    5:  np.dtype('float16'),
    6:  np.dtype('float32'),
    7:  np.dtype('float64'),
    9:  np.dtype('complex64'),
    10: np.dtype('complex128'),
    11: np.dtype('bool'),
}

# Device type — matches ATen DeviceType
DEVICE_CPU  = 0
DEVICE_CUDA = 1

# SerializedTensorHeader struct format (little-endian, with 4-byte padding before i64)
_HEADER_FMT  = '<BBBb4xq'   # B B B b [4 pad] q  →  16 bytes
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)  # 16


# ---------------------------------------------------------------------------
# GenericValue
# ---------------------------------------------------------------------------

class GenericValue:
    """Mirrors hasty::GenericValue.  Holds one of: None, Tensor, list, dict, str."""

    def __init__(self, data=None, gtype: eType = eType.NONE):
        self._type = gtype
        self._data = data

    # --- constructors -------------------------------------------------------

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> 'GenericValue':
        gv = cls.__new__(cls)
        gv._type = eType.TENSOR
        gv._data = t
        return gv

    @classmethod
    def from_string(cls, s: str) -> 'GenericValue':
        gv = cls.__new__(cls)
        gv._type = eType.STRING
        gv._data = s
        return gv

    @classmethod
    def from_vector(cls, items: list) -> 'GenericValue':
        gv = cls.__new__(cls)
        gv._type = eType.VECTOR
        gv._data = list(items)
        return gv

    @classmethod
    def from_dict(cls, d: dict) -> 'GenericValue':
        gv = cls.__new__(cls)
        gv._type = eType.DICT
        gv._data = dict(d)
        return gv

    # --- accessors ----------------------------------------------------------

    def is_tensor(self) -> bool: return self._type == eType.TENSOR
    def is_string(self) -> bool: return self._type == eType.STRING
    def is_vector(self) -> bool: return self._type == eType.VECTOR
    def is_dict(self)   -> bool: return self._type == eType.DICT

    def as_tensor(self) -> torch.Tensor:
        assert self._type == eType.TENSOR
        return self._data

    def as_string(self) -> str:
        assert self._type == eType.STRING
        return self._data

    # --- streaming serialization (avoids full-copy buffer) ------------------

    def iter_chunks(self, chunk_size: int):
        """Yield raw serialized bytes in chunk_size pieces.
        For tensors, streams directly from the tensor's memory via memoryview
        instead of copying everything into a BytesIO buffer first.
        """
        if self._type == eType.TENSOR:
            yield from GenericValue._iter_tensor_chunks(self._data, chunk_size)
        else:
            data = self.serialize()
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]

    @staticmethod
    def _iter_tensor_chunks(t: torch.Tensor, chunk_size: int):
        t = t.cpu().contiguous()
        scalar_type = _TORCH_TO_SCALAR[t.dtype]
        # Build header + shape into a small buffer (< 100 bytes typically)
        hdr = io.BytesIO()
        hdr.write(struct.pack('B', eType.TENSOR))
        hdr.write(struct.pack(_HEADER_FMT, DEVICE_CPU, scalar_type, t.dim(), -1, t.numel()))
        for s in t.shape:
            hdr.write(struct.pack('<q', s))
        header_bytes = hdr.getvalue()

        # Stream tensor bytes via memoryview — no 10 GiB copy into BytesIO
        mv = memoryview(t.numpy()).cast('B')
        total = len(mv)

        if total == 0:
            yield header_bytes
            return

        # Combine header with the leading bytes of tensor data into the first chunk
        first_data = min(chunk_size - len(header_bytes), total)
        yield header_bytes + bytes(mv[0:first_data])
        off = first_data
        while off < total:
            end = min(off + chunk_size, total)
            yield bytes(mv[off:end])
            off = end

    # --- streaming deserialization (pre-allocated numpy for tensors) --------

    @classmethod
    def deserialize_chunks(cls, chunks_iter) -> 'GenericValue':
        """Deserialize from an iterator of bytes chunks.
        For tensors: pre-allocates the output numpy array so each incoming
        gRPC chunk is copied directly into it — no b''.join and no arr.copy().
        """
        chunks_iter = iter(chunks_iter)
        buf = bytearray()

        def _fill_to(n: int):
            while len(buf) < n:
                try:
                    buf.extend(next(chunks_iter))
                except StopIteration:
                    raise RuntimeError("Stream ended prematurely")

        _fill_to(1)
        type_byte = buf[0]

        if type_byte != eType.TENSOR:
            # Non-tensor: collect everything then fall back to normal deserialize
            for chunk in chunks_iter:
                buf.extend(chunk)
            return cls.deserialize(bytes(buf))

        # Parse header
        _fill_to(1 + _HEADER_SIZE)
        device_type, scalar_type, ndim, device_index, total_elements = \
            struct.unpack(_HEADER_FMT, bytes(buf[1:1 + _HEADER_SIZE]))

        # Parse shape
        shape_end = 1 + _HEADER_SIZE + 8 * ndim
        _fill_to(shape_end)
        shape = struct.unpack(f'<{ndim}q', bytes(buf[1 + _HEADER_SIZE:shape_end]))

        # Pre-allocate writable numpy array — no extra copy needed after receive
        itemsize = _SCALAR_ITEMSIZE[scalar_type]
        total_bytes = total_elements * itemsize
        arr = np.empty(total_elements, dtype=_NUMPY_DTYPE[scalar_type])
        out = memoryview(arr).cast('B')

        # Copy any tensor data already in buf (past shape_end)
        written = len(buf) - shape_end
        if written > 0:
            out[0:written] = buf[shape_end:]

        # Stream remaining chunks directly into the array
        for chunk in chunks_iter:
            n = len(chunk)
            out[written:written + n] = chunk
            written += n

        if written != total_bytes:
            raise RuntimeError(
                f"Tensor data size mismatch: expected {total_bytes}, got {written}"
            )

        t = torch.from_numpy(arr.reshape(shape))
        return cls.from_tensor(t)

    # --- serialization ------------------------------------------------------

    def serialize(self) -> bytes:
        buf = io.BytesIO()
        self._write(buf)
        return buf.getvalue()

    def _write(self, buf: io.BytesIO) -> None:
        if self._type == eType.NONE:
            buf.write(struct.pack('B', eType.NONE))

        elif self._type == eType.TENSOR:
            self._write_tensor(buf, self._data)

        elif self._type == eType.STRING:
            self._write_string(buf, self._data)

        elif self._type in (eType.VECTOR, eType.TUPLE):
            buf.write(struct.pack('B', int(self._type)))
            buf.write(struct.pack('<Q', len(self._data)))
            for item in self._data:
                item._write(buf)

        elif self._type == eType.DICT:
            buf.write(struct.pack('B', eType.DICT))
            buf.write(struct.pack('<Q', len(self._data)))
            for key, val in self._data.items():
                GenericValue.from_string(key)._write(buf)
                val._write(buf)

    @staticmethod
    def _write_tensor(buf: io.BytesIO, t: torch.Tensor) -> None:
        buf.write(struct.pack('B', eType.TENSOR))
        t = t.cpu().contiguous()
        scalar_type = _TORCH_TO_SCALAR[t.dtype]
        ndim = t.dim()
        device_type = DEVICE_CPU
        device_index = -1
        total_elements = t.numel()
        buf.write(struct.pack(_HEADER_FMT,
            device_type, scalar_type, ndim, device_index, total_elements))
        for dim in t.shape:
            buf.write(struct.pack('<q', dim))
        buf.write(t.numpy().tobytes())

    @staticmethod
    def _write_string(buf: io.BytesIO, s: str) -> None:
        encoded = s.encode('utf-8')
        buf.write(struct.pack('B', eType.STRING))
        buf.write(struct.pack('<Q', len(encoded)))
        buf.write(encoded)

    # --- deserialization ----------------------------------------------------

    @classmethod
    def deserialize(cls, data: bytes) -> 'GenericValue':
        buf = io.BytesIO(data)
        return cls._read(buf)

    @classmethod
    def _read(cls, buf: io.BytesIO) -> 'GenericValue':
        (type_byte,) = struct.unpack('B', buf.read(1))
        t = eType(type_byte)

        if t == eType.NONE:
            return cls()

        elif t == eType.TENSOR:
            return cls._read_tensor(buf)

        elif t == eType.STRING:
            return cls._read_string(buf)

        elif t in (eType.VECTOR, eType.TUPLE):
            (count,) = struct.unpack('<Q', buf.read(8))
            items = [cls._read(buf) for _ in range(count)]
            gv = cls.__new__(cls)
            gv._type = t
            gv._data = items
            return gv

        elif t == eType.DICT:
            (count,) = struct.unpack('<Q', buf.read(8))
            d = {}
            for _ in range(count):
                key = cls._read_string(buf).as_string()
                val = cls._read(buf)
                d[key] = val
            return cls.from_dict(d)

        else:
            raise ValueError(f"Unknown type byte: {type_byte}")

    @classmethod
    def _read_tensor(cls, buf: io.BytesIO) -> 'GenericValue':
        device_type, scalar_type, ndim, device_index, total_elements = \
            struct.unpack(_HEADER_FMT, buf.read(_HEADER_SIZE))
        shape = struct.unpack(f'<{ndim}q', buf.read(8 * ndim))
        itemsize = _SCALAR_ITEMSIZE[scalar_type]
        raw = buf.read(total_elements * itemsize)
        arr = np.frombuffer(raw, dtype=_NUMPY_DTYPE[scalar_type]).reshape(shape)
        t = torch.from_numpy(arr.copy())
        return cls.from_tensor(t)

    @classmethod
    def _read_string(cls, buf: io.BytesIO) -> 'GenericValue':
        # type byte already consumed by caller only when called from _read;
        # but _write_string writes the tag, so we must NOT consume it again here.
        # When called directly from _read_dict we need the tag already consumed.
        (length,) = struct.unpack('<Q', buf.read(8))
        s = buf.read(length).decode('utf-8')
        return cls.from_string(s)
