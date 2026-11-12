# hasty value format, version 1

The format carries one value tree per stream, used both on the wire (hasty server: gRPC chunks) and as files
(`.hv`). Implementations:
- Rust: `hasty-value::format`
- Python: `python/lib/hastycompute/value_format.py`
- JS: `plotting_website/common/value_format.js` (decodes LZ4 and stored blocks only)

**Design goals**
- streaming in both directions;
- compression native to the format;
- independent blocks, which give parallel (de)compression and bounded memory;
- self-describing;
- little-endian throughout.

## Stream

```
magic   "HSTV"            4 bytes
version u8 = 1
flags   u8                integrity algorithm: 0 none, 1 XXH3-128 (16-byte digest), 2 BLAKE3-256 (32 bytes)
pad     u16 = 0
value                     (exactly one, see below)
digest                    only if flags ≠ 0: the digest of every preceding byte of the stream (header included)
                          nothing may follow
```

## Values

The first byte is a tag.

| tag | type | payload |
|---|---|---|
| 0 | None | — |
| 1 | Bool | u8 (0 / 1) |
| 2 | Int | i64 |
| 3 | Float | f64 |
| 4 | Str | u64 length, UTF-8 bytes |
| 5 | List | u64 n, then n values |
| 6 | Tuple | u64 n, then n values |
| 7 | Dict | u64 n, then n × (u32 key length, UTF-8 key, value); insertion order is preserved |
| 8 | Tensor | header, then data blocks |
| 9 | Chunked tensor | header, chunk index, chunk data (random access; files) |
| 10 | Quantised chunked tensor | as 9; every chunk starts with a 20-byte quantiser prefix (lossy, float dtypes only) |

### Tensor

```
u8  tag = 8
u8  dtype                (code table below)
u8  ndim                 (≤ 255; 0 = scalar)
u8  reserved = 0
u32 block_size           raw bytes per block (a multiple of 16); every block but the last is exactly this size
u64 shape[ndim]          C-order (last index fastest)
blocks                   nblocks = ceil(nbytes / block_size), nbytes = prod(shape) · dtype size (0 → no blocks)
```

dtype codes:

| code | dtype | bytes | | code | dtype | bytes |
|---|---|---|---|---|---|---|
| 0 | bool | 1 | | 8 | int64 | 8 |
| 1 | uint8 | 1 | | 9 | float16 | 2 |
| 2 | int8 | 1 | | 10 | bfloat16 | 2 |
| 3 | uint16 | 2 | | 11 | float32 | 4 |
| 4 | int16 | 2 | | 12 | float64 | 8 |
| 5 | uint32 | 4 | | 13 | complex64 | 8 |
| 6 | int32 | 4 | | 14 | complex128 | 16 |
| 7 | uint64 | 8 | | | | |

### Block

```
u8  codec      0 stored, 1 LZ4 block format (no size prefix), 2 zstd frame
u8  filter     bit 0: byte shuffle, bit 1: delta (applied before shuffle)
u8  word       filter word size in bytes (dtype size; complex64 → 4, complex128 → 8: real/imag shuffle separately)
u8  flags      bit 0: a CRC32 (IEEE) of the payload follows
u32 raw_len    decoded size (= block_size, or the remainder for the last block)
u32 comp_len   payload size
[u32 crc32]    if flags bit 0
payload        comp_len bytes
```

**Encoding a block**
1. raw → delta (if bit 1) → shuffle (if bit 0) → codec.
2. If the codec output is not smaller than raw_len, the block is written stored (codec 0, filter 0).

**Decoding a block** reverses the steps:
1. decompress;
2. unshuffle;
3. prefix-sum (delta).

**Filters**
- **Shuffle with word w over L bytes.** Let n = ⌊L/w⌋. Byte b of word i goes to b·n + i. The last L − n·w bytes (a ragged tail) are copied unchanged at the end.
- **Delta.** Wrapping difference of consecutive little-endian unsigned words of size w (1, 2, 4 or 8), first word unchanged. It is only produced for integer dtypes.

Because each block carries its own sizes, a reader decodes block by block as bytes arrive, and blocks decode
independently (in parallel).

### Chunked tensor (tag 9)

A chunked tensor is cut into N-D chunks along a regular grid. Each chunk is compressed independently, so a reader
can fetch any box by decoding only the chunks it overlaps.

```
u8  tag = 9, u8 dtype, u8 ndim, u8 reserved = 0, u32 block_size
u64 shape[ndim]
u64 chunk[ndim]          chunk shape (≥ 1 per dimension; edge chunks are clipped to the tensor)
u64 nchunks              = prod(ceil(shape / chunk)), 0 if any shape entry is 0
nchunks × (u64 offset, u64 length)     chunk data location, relative to the end of this index
chunk data               chunks in C-order of the chunk grid
```

- **Chunk data:** each chunk holds the C-order bytes of its own (clipped) box, stored as the blocks of a plain tensor (same block header, filters and codecs; `block_size` raw bytes per block).
- **Writing:** writers compress all chunks first, so the index can precede the data.
- **Reading:** stream decoders assemble the full tensor once all chunk data has arrived. Random-access readers (`LazyFile` in Rust, `open_lazy` in Python) read the header and the index, then only the chunks they need.

### Quantised chunked tensor (tag 10)

Same header, index and chunk layout as tag 9 (float32 / float64 / complex dtypes only). Each chunk's data starts with a
20-byte prefix, followed by blocks of the quantised words:

```
u8  qw         word size of the quantised values: 1, 2 or 4; 0 = the chunk is stored losslessly (blocks of raw dtype bytes follow)
u8  transfer   0 linear, 1 asinh
u16 reserved = 0
f64 step       quantiser step in transfer space
f64 knee       k of the asinh transfer (0 if linear)
blocks         one qw-byte little-endian word per scalar (complex: re, im), block word = qw
```

Per scalar x: u = x (linear) or k·asinh(x/k); q = round(u / step); the word is zigzag(q) = (q << 1) ^ (q >> 63).
Decoding: q = unzigzag(word), x = q·step (linear) or k·sinh(q·step / k), cast to the dtype. Chunks of one tensor may use
different `step` and `qw` (scale scope = chunk) or share the step (scope = tensor). Quantisation specs and their
guarantees: `hasty_value::lossy`. Mantissa rounding produces ordinary tag 8 / 9 tensors and needs no format support.
