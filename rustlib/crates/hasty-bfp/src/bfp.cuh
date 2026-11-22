// bfp.cuh: block-floating-point (BFP) storage codec for NVRTC kernels (research/bfp_storage.md).
//
// No std headers; include by string concatenation (hasty_bfp::BFP_CUH) before the kernels that use it. Compile without
// --use_fast_math / -ftz=true: decoded values of tiny blocks may be subnormal, and the CPU codec (hasty_bfp) keeps them.
//
// Format (identical to the Rust CPU codec, bit for bit):
//   * A block is 32 consecutive reals of a buffer (16 complex values, re/im interleaved), aligned to 32 reals in the
//     buffer's flat index. A tail block is padded with zeros by the writer.
//   * Mantissas: B-bit two's complement integers q_i, 8 <= B <= 24. Value i of a block occupies bits [i*B, i*B + B) of
//     the block's bit stream; the stream is B little-endian 32-bit words, so block k's words are mant[k*B .. k*B + B).
//     Values straddle words (for B not dividing 32), never blocks.
//   * Exponent: one byte per block in a separate array (ex[k]; 4 blocks per 32-bit word, little-endian).
//     es = e + BFP_EBIAS with every |x_i| of the block < 2^e, and x_i ~ q_i * 2^(e - B + 1).
//     es = BFP_ENAN marks a block holding an Inf or NaN: it decodes as all-NaN (non-finite values are never saturated).
//   * Choice of e: the smallest e in [B - 127, 128] with max|x| < 2^e and round(max|x| * 2^(B-1-e)) <= 2^(B-1) - 1,
//     so no value saturates and every value obeys |x - deq(x)| <= 2^(e - B) (half a step). The limits keep the scale
//     factors 2^(B-1-e) and 2^(e-B+1) normal floats and every decoded value finite; es <= 253. Only values within half
//     a step of 2^128 (|x| > 2^128 (1 - 2^-B)) saturate, with an error below one step.
//   * Rounding: round to nearest, ties to even (__float2int_rn), of x * 2^(B-1-e) (an exact power-of-two scaling).
//
// Access patterns over the same bytes (the consumer chooses; the memory format does not change):
//   * warp block   (bfp_warp_*):  lane j of a warp holds value j of one block; lane j < B loads/stores word j (one
//                                 coalesced 4B-byte access). Unpacking value i takes words floor(i*B/32) and the next:
//                                 two shuffles, one funnel shift, one sign extension, for any B.
//   * half warp, complex (bfp_hw_*): lane t of a 16-lane group holds complex t of one block (the strided FFT passes'
//                                 16 lines per CUDA block); lane t stores words t and t + 16.
//   * per thread   (bfp_run_*):   a thread owns V = 32 / gcd(B, 32) consecutive values, which fill exactly
//                                 W = V * B / 32 whole words: no shuffles for the mantissas; the block exponent is
//                                 shared by 32 / V lanes (group max for packing).

#ifndef HASTY_BFP_CUH
#define HASTY_BFP_CUH

#define BFP_VALS 32
#define BFP_EBIAS 125
#define BFP_ENAN 255
#define BFP_FULL 0xffffffffu

template <int B> struct BfpT {
    static_assert(B >= 8 && B <= 24, "BFP mantissa bits must be in 8..24");
    static constexpr int LIM = (1 << (B - 1)) - 1;
    static constexpr unsigned MASK = (1u << B) - 1u;
    static constexpr int EMIN = B - 127;
    static constexpr int GCD32 = (B & 31) == 0 ? 32 : (B & 15) == 0 ? 16 : (B & 7) == 0 ? 8 : (B & 3) == 0 ? 4 : (B & 1) == 0 ? 2 : 1;
    // per-thread run: V values in W words
    static constexpr int V = 32 / GCD32;
    static constexpr int W = V * B / 32;
};

// 2^k as a float, k in [-126, 127]
__device__ __forceinline__ float bfp_pow2(int k) { return __int_as_float((k + 127) << 23); }

// |x| as ordered integer bits (NaN > Inf > finite), so an integer max never drops a NaN (fmaxf would)
__device__ __forceinline__ unsigned bfp_abs_bits(float x) { return __float_as_uint(x) & 0x7fffffffu; }

// exponent byte of a block with max |x| bits `mbits`
template <int B> __device__ __forceinline__ int bfp_estore(unsigned mbits) {
    if (mbits >= 0x7f800000u) return BFP_ENAN;
    const int E = (int)(mbits >> 23);
    int e = E == 0 ? -126 : E - 126;
    if (e < BfpT<B>::EMIN) e = BfpT<B>::EMIN;
    if (e < 128 && __float2int_rn(__uint_as_float(mbits) * bfp_pow2(B - 1 - e)) > BfpT<B>::LIM) ++e;
    return e + BFP_EBIAS;
}

template <int B> __device__ __forceinline__ int bfp_quant(float x, int es) {
    const int q = __float2int_rn(x * bfp_pow2(B - 1 - (es - BFP_EBIAS)));
    return max(-BfpT<B>::LIM, min(BfpT<B>::LIM, q));
}

template <int B> __device__ __forceinline__ float bfp_dequant(int q, int es) {
    if (es == BFP_ENAN) return __int_as_float(0x7fc00000);
    return (float)q * bfp_pow2(es - BFP_EBIAS - B + 1);
}

template <int B> __device__ __forceinline__ int bfp_sext(unsigned u) { return ((int)(u << (32 - B))) >> (32 - B); }

// Word k (0 <= k < B) of a block's mantissa stream, straight from the block's 32 values x[0..32) and its exponent
// byte: value i occupies bits [i B, i B + B) of the block's B words, the same layout bfp_run_pack produces run by run
// (run j starts at bit j V B, a word boundary). Lets a store write a block as whole vector pieces (contiguous lanes,
// full sectors) instead of one run of W scattered words per lane.
template <int B> __device__ __forceinline__ unsigned bfp_block_word(const float* __restrict__ x, int es, int k) {
    unsigned w = 0u;
    const int b0 = 32 * k, i0 = b0 / B, i1 = (b0 + 31) / B;
#pragma unroll
    for (int d = 0; d < 32 / B + 2; ++d) {
        const int i = i0 + d;
        if (i <= i1 && i < BFP_VALS) {
            const unsigned u = (unsigned)(es == BFP_ENAN ? 0 : bfp_quant<B>(x[i], es)) & BfpT<B>::MASK;
            const int sh = i * B - b0;
            w |= sh >= 0 ? (u << sh) : (u >> (-sh));
        }
    }
    return w;
}

// bfp_block_word from the block's already quantised values q[0..32) (bfp_quant, 0 for a NaN block)
template <int B> __device__ __forceinline__ unsigned bfp_block_word_q(const int* __restrict__ q, int k) {
    unsigned w = 0u;
    const int b0 = 32 * k, i0 = b0 / B, i1 = (b0 + 31) / B;
#pragma unroll
    for (int d = 0; d < 32 / B + 2; ++d) {
        const int i = i0 + d;
        if (i <= i1 && i < BFP_VALS) {
            const unsigned u = (unsigned)q[i] & BfpT<B>::MASK;
            const int sh = i * B - b0;
            w |= sh >= 0 ? (u << sh) : (u >> (-sh));
        }
    }
    return w;
}

// ---------------------------------------------------------------------------------------------------------------------
// Reductions of the block max (integer bits of |x|)

// max over aligned groups of G lanes (G a power of two <= 32)
template <int G> __device__ __forceinline__ unsigned bfp_group_max(unsigned m) {
#if __CUDA_ARCH__ >= 800
    if constexpr (G == 32) {
        return __reduce_max_sync(BFP_FULL, m);
    } else
#endif
    {
#pragma unroll
        for (int o = G / 2; o > 0; o >>= 1) m = max(m, __shfl_xor_sync(BFP_FULL, m, o));
        return m;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Per-thread runs: values q[0..V) start at a word boundary and fill W whole words (V * B % 32 == 0)

template <int B, int V> __device__ __forceinline__ void bfp_run_pack(const int (&q)[V], unsigned (&w)[V * B / 32]) {
    static_assert((V * B) % 32 == 0, "a run must fill whole words");
#pragma unroll
    for (int k = 0; k < V * B / 32; ++k) w[k] = 0u;
#pragma unroll
    for (int i = 0; i < V; ++i) {
        const unsigned u = (unsigned)q[i] & BfpT<B>::MASK;
        const int bit = i * B, k = bit >> 5, s = bit & 31;
        w[k] |= u << s;
        if (s + B > 32) w[k + 1] |= u >> (32 - s);
    }
}

template <int B, int V> __device__ __forceinline__ void bfp_run_unpack(const unsigned (&w)[V * B / 32], int (&q)[V]) {
    static_assert((V * B) % 32 == 0, "a run must fill whole words");
#pragma unroll
    for (int i = 0; i < V; ++i) {
        const int bit = i * B, k = bit >> 5, s = bit & 31;
        const unsigned hi = (s + B > 32) ? w[k + 1] : 0u;
        q[i] = bfp_sext<B>(__funnelshift_r(w[k], hi, s));
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Warp block: lane j holds value j; lane j < B holds word j. All 32 lanes must call (full mask).

template <int B> __device__ __forceinline__ unsigned bfp_warp_pack_word(int q) {
    const int lane = threadIdx.x & 31;
    const unsigned u = (unsigned)q & BfpT<B>::MASK;
    // values touching word `lane`: ilo .. ilo + NMAX - 1 (bit offset relative to the word in (-B, 32))
    constexpr int NMAX = (32 % B == 0) ? 32 / B : (32 + B - 1) / B + 1;
    const int ilo = (32 * lane) / B;
    unsigned word = 0u;
#pragma unroll
    for (int k = 0; k < NMAX; ++k) {
        const int i = ilo + k;
        const unsigned v = __shfl_sync(BFP_FULL, u, i & 31);
        const int off = i * B - 32 * lane;
        if (i < 32 && off < 32) word |= off >= 0 ? (v << off) : (v >> (-off));
    }
    return word;
}

template <int B> __device__ __forceinline__ int bfp_warp_unpack_q(unsigned word) {
    const int lane = threadIdx.x & 31;
    const int bit = lane * B, k = bit >> 5, s = bit & 31;
    const unsigned lo = __shfl_sync(BFP_FULL, word, k);
    unsigned hi = 0u;
    if constexpr (32 % B != 0) hi = __shfl_sync(BFP_FULL, word, min(k + 1, B - 1));
    return bfp_sext<B>(__funnelshift_r(lo, hi, s));
}

// block exponent of the warp's 32 values
template <int B> __device__ __forceinline__ int bfp_warp_estore(float x) { return bfp_estore<B>(bfp_group_max<32>(bfp_abs_bits(x))); }

// store value x (lane j = value j) as block `blk`; returns the exponent byte (all lanes) so callers may combine
// four blocks' exponents into one 32-bit store (pass write_ex = false then)
template <int B> __device__ __forceinline__ int bfp_warp_store(float x, unsigned* __restrict__ mant, unsigned char* __restrict__ ex, long blk, bool write_ex = true) {
    const int lane = threadIdx.x & 31;
    const int es = bfp_warp_estore<B>(x);
    const unsigned w = bfp_warp_pack_word<B>(es == BFP_ENAN ? 0 : bfp_quant<B>(x, es));
    if (lane < B) mant[blk * B + lane] = w;
    if (write_ex && lane == 0) ex[blk] = (unsigned char)es;
    return es;
}

template <int B> __device__ __forceinline__ float bfp_warp_load(const unsigned* __restrict__ mant, int es, long blk) {
    const int lane = threadIdx.x & 31;
    const unsigned w = lane < B ? mant[blk * B + lane] : 0u;
    return bfp_dequant<B>(bfp_warp_unpack_q<B>(w), es);
}

// ---------------------------------------------------------------------------------------------------------------------
// Half warp, complex: lane t of a 16-lane group holds complex t of one block (values 2t, 2t + 1); lane t holds words
// t and t + 16 (the latter only for B > 16). Both half warps of a warp work on their own blocks; the shuffles use the
// half warp's own mask, so the two halves may diverge (e.g. one half outside a pass's input range), but all 16 lanes
// of a half must call.

__device__ __forceinline__ unsigned bfp_hw_mask() { return 0xffffu << (threadIdx.x & 16); }

template <int B> __device__ __forceinline__ void bfp_hw_pack_c(int q0, int q1, unsigned& w0, unsigned& w1) {
    const int t = threadIdx.x & 15;
    const unsigned hm = bfp_hw_mask();
    const unsigned long long c = (unsigned long long)((unsigned)q0 & BfpT<B>::MASK) | ((unsigned long long)((unsigned)q1 & BfpT<B>::MASK) << B);
    const unsigned clo = (unsigned)c, chi = (unsigned)(c >> 32);
    // complex lanes touching a word: jlo .. jlo + NJ - 1, each a 2B-bit chunk at bit 2jB
    constexpr int NJ = (32 % (2 * B) == 0) ? 32 / (2 * B) : (32 + 2 * B - 1) / (2 * B) + 1;
    constexpr int NH = B > 16 ? 2 : 1;
    unsigned out[2] = {0u, 0u};
#pragma unroll
    for (int h = 0; h < NH; ++h) {
        const int w = t + 16 * h;
        const int jlo = (32 * w) / (2 * B);
#pragma unroll
        for (int k = 0; k < NJ; ++k) {
            const int j = jlo + k;
            const unsigned vlo = __shfl_sync(hm, clo, j & 15, 16);
            unsigned vhi = 0u;
            if constexpr (2 * B > 32) vhi = __shfl_sync(hm, chi, j & 15, 16);
            const unsigned long long v = ((unsigned long long)vhi << 32) | vlo;
            const int off = j * 2 * B - 32 * w;
            if (w < B && j < 16 && off < 32) out[h] |= (unsigned)(off >= 0 ? (v << off) : (v >> (-off)));
        }
    }
    w0 = out[0];
    w1 = out[1];
}

template <int B> __device__ __forceinline__ void bfp_hw_unpack_c(unsigned w0, unsigned w1, int& q0, int& q1) {
    const int t = threadIdx.x & 15;
    const unsigned hm = bfp_hw_mask();
    const int bit = 2 * t * B, k0 = bit >> 5, s = bit & 31;
    // words k0, k0 + 1, k0 + 2 cover bits [s, s + 2B) (2B <= 48, s <= 31)
    constexpr int NW = (32 % (2 * B) == 0) ? 1 : (2 * B <= 32 ? 2 : 3);
    unsigned wd[3] = {0u, 0u, 0u};
#pragma unroll
    for (int k = 0; k < NW; ++k) {
        const int kk = min(k0 + k, B - 1);
        const unsigned a = __shfl_sync(hm, w0, kk & 15, 16);
        unsigned b = 0u;
        if constexpr (B > 16) b = __shfl_sync(hm, w1, kk & 15, 16);
        wd[k] = kk < 16 ? a : b;
    }
    q0 = bfp_sext<B>(__funnelshift_r(wd[0], wd[1], s));
    const int s1 = s + B, r1 = s1 & 31;
    const unsigned lo1 = (s1 >> 5) ? wd[1] : wd[0], hi1 = (s1 >> 5) ? wd[2] : wd[1];
    q1 = bfp_sext<B>(__funnelshift_r(lo1, hi1, r1));
}

template <int B> __device__ __forceinline__ int bfp_hw_estore_c(float re, float im) {
    const unsigned hm = bfp_hw_mask();
    unsigned m = max(bfp_abs_bits(re), bfp_abs_bits(im));
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) m = max(m, __shfl_xor_sync(hm, m, o));
    return bfp_estore<B>(m);
}

// store complex z (lane t of its half warp = complex t) as block `blk` (one block per half warp)
template <int B> __device__ __forceinline__ int bfp_hw_store_c(float2 z, unsigned* __restrict__ mant, unsigned char* __restrict__ ex, long blk, bool valid, bool write_ex = true) {
    const int t = threadIdx.x & 15;
    const int es = bfp_hw_estore_c<B>(z.x, z.y);
    const bool nan = es == BFP_ENAN;
    unsigned w0, w1;
    bfp_hw_pack_c<B>(nan ? 0 : bfp_quant<B>(z.x, es), nan ? 0 : bfp_quant<B>(z.y, es), w0, w1);
    if (valid) {
        if (t < B) mant[blk * B + t] = w0;
        if (B > 16 && t + 16 < B) mant[blk * B + t + 16] = w1;
        if (write_ex && t == 0) ex[blk] = (unsigned char)es;
    }
    return es;
}

template <int B> __device__ __forceinline__ float2 bfp_hw_load_c(const unsigned* __restrict__ mant, int es, long blk, bool valid) {
    const int t = threadIdx.x & 15;
    const unsigned w0 = (valid && t < B) ? mant[blk * B + t] : 0u;
    const unsigned w1 = (valid && B > 16 && t + 16 < B) ? mant[blk * B + t + 16] : 0u;
    int q0, q1;
    bfp_hw_unpack_c<B>(w0, w1, q0, q1);
    return make_float2(bfp_dequant<B>(q0, es), bfp_dequant<B>(q1, es));
}

// ---------------------------------------------------------------------------------------------------------------------
// Per-thread runs in global memory: run r covers values [r*V, (r+1)*V) of the buffer, words [r*W, (r+1)*W), block r*V/32.

// quantise and pack a run with a known exponent byte
template <int B> __device__ __forceinline__ void bfp_run_store(const float (&x)[BfpT<B>::V], int es, unsigned* __restrict__ mant, long run) {
    constexpr int V = BfpT<B>::V, W = BfpT<B>::W;
    int q[V];
#pragma unroll
    for (int i = 0; i < V; ++i) q[i] = es == BFP_ENAN ? 0 : bfp_quant<B>(x[i], es);
    unsigned w[W];
    bfp_run_pack<B, V>(q, w);
#pragma unroll
    for (int k = 0; k < W; ++k) mant[run * W + k] = w[k];
}

template <int B> __device__ __forceinline__ void bfp_run_load(const unsigned* __restrict__ mant, int es, long run, float (&x)[BfpT<B>::V]) {
    constexpr int V = BfpT<B>::V, W = BfpT<B>::W;
    unsigned w[W];
#pragma unroll
    for (int k = 0; k < W; ++k) w[k] = mant[run * W + k];
    int q[V];
    bfp_run_unpack<B, V>(w, q);
#pragma unroll
    for (int i = 0; i < V; ++i) x[i] = bfp_dequant<B>(q[i], es);
}

// exponent byte of the block a run belongs to, from the run's values: max over the 32 / V lanes sharing the block
// (consecutive runs in consecutive lanes, groups aligned to 32 / V lanes; all lanes call)
template <int B> __device__ __forceinline__ int bfp_run_estore(const float (&x)[BfpT<B>::V]) {
    constexpr int V = BfpT<B>::V;
    unsigned m = 0u;
#pragma unroll
    for (int i = 0; i < V; ++i) m = max(m, bfp_abs_bits(x[i]));
    if constexpr (V < 32) m = bfp_group_max<32 / V>(m);
    return bfp_estore<B>(m);
}

// ---------------------------------------------------------------------------------------------------------------------
// Random access (no shuffles, any lane pattern): decode real value `i` or complex value `c` of a buffer from global
// memory. A complex value needs the words covering bits [32 c mod 16 * B ...): at most 3 word loads (2 for B <= 16),
// all inside its block; for consumers whose lane groups do not line up with blocks (a gather reading the channels of
// one cell when the channel count is not a multiple of 16).

template <int B> __device__ __forceinline__ float bfp_load_r(const unsigned* __restrict__ mant, const unsigned char* __restrict__ ex, long i) {
    const long k = i >> 5;
    const int bit = (int)(i & 31) * B, s = bit & 31;
    const unsigned* w = mant + k * B + (bit >> 5);
    const unsigned lo = w[0], hi = (s + B > 32) ? w[1] : 0u;
    return bfp_dequant<B>(bfp_sext<B>(__funnelshift_r(lo, hi, s)), ex[k]);
}

template <int B> __device__ __forceinline__ float2 bfp_load_c(const unsigned* __restrict__ mant, const unsigned char* __restrict__ ex, long c) {
    const long k = c >> 4;
    const int bit = (int)(c & 15) * 2 * B, s = bit & 31;
    const unsigned* w = mant + k * B + (bit >> 5);
    // the value pair ends at bit 2B (pos + 1) <= 32 B: w[1] / w[2] only when the pair reaches them
    const unsigned w0 = w[0], w1 = (s + 2 * B > 32) ? w[1] : 0u, w2 = (s + 2 * B > 64) ? w[2] : 0u;
    const int es = ex[k];
    const int q0 = bfp_sext<B>(__funnelshift_r(w0, w1, s));
    const int s1 = s + B;
    const int q1 = bfp_sext<B>((s1 >> 5) ? __funnelshift_r(w1, w2, s1 & 31) : __funnelshift_r(w0, w1, s1));
    return make_float2(bfp_dequant<B>(q0, es), bfp_dequant<B>(q1, es));
}

#endif // HASTY_BFP_CUH
