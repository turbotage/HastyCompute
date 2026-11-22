// Test and benchmark kernels of the BFP codec (compile with -DBFP_B=<bits> after bfp.cuh). One packed buffer: mant
// (nblk * B words) and ex (nblk bytes, padded to 4-byte words); n reals, the tail block zero-padded.
//
//   bfp_pack_warp / bfp_unpack_warp : warp block layout, 4 blocks per warp iteration, exponents as one 32-bit word
//   bfp_pack_hw   / bfp_unpack_hw   : half warp complex layout (lane = complex), 2 x 2 blocks per warp iteration
//   bfp_pack_run  / bfp_unpack_run  : per-thread runs (V values, W words per thread)
//   bfp_unpack_rand(_r)             : random access decode, one complex (real) value per thread
//   copy_f32 / copy_u32             : plain copies (bandwidth reference)

#ifndef BFP_B
#error "define BFP_B"
#endif
#define BB BFP_B

#define GTID ((long)blockIdx.x * blockDim.x + threadIdx.x)
#define GSTRIDE ((long)gridDim.x * blockDim.x)

extern "C" __global__ void __launch_bounds__(256) bfp_pack_warp(const float* __restrict__ x, unsigned* __restrict__ mant, unsigned char* __restrict__ ex, long n) {
    const int lane = threadIdx.x & 31;
    const long nblk = (n + 31) >> 5;
    for (long g = GTID >> 5; g * 4 < nblk; g += GSTRIDE >> 5) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long i = (g * 4 + j) * 32 + lane;
            v[j] = i < n ? x[i] : 0.f;
        }
        unsigned ew = 0u;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long blk = g * 4 + j;
            const int es = bfp_warp_estore<BB>(v[j]);
            const unsigned w = bfp_warp_pack_word<BB>(es == BFP_ENAN ? 0 : bfp_quant<BB>(v[j], es));
            if (blk < nblk && lane < BB) mant[blk * BB + lane] = w;
            ew |= (unsigned)es << (8 * j);
        }
        if (lane == 0) {
            if (g * 4 + 3 < nblk) reinterpret_cast<unsigned*>(ex)[g] = ew;
            else
                for (int j = 0; j < 4; ++j)
                    if (g * 4 + j < nblk) ex[g * 4 + j] = (unsigned char)(ew >> (8 * j));
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) bfp_unpack_warp(const unsigned* __restrict__ mant, const unsigned char* __restrict__ ex, float* __restrict__ out, long n) {
    const int lane = threadIdx.x & 31;
    const long nblk = (n + 31) >> 5;
    for (long g = GTID >> 5; g * 4 < nblk; g += GSTRIDE >> 5) {
        unsigned ew = 0u;
        if (g * 4 + 3 < nblk) ew = reinterpret_cast<const unsigned*>(ex)[g];
        else
            for (int j = 0; j < 4; ++j)
                if (g * 4 + j < nblk) ew |= (unsigned)ex[g * 4 + j] << (8 * j);
        unsigned wv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long blk = g * 4 + j;
            wv[j] = (blk < nblk && lane < BB) ? mant[blk * BB + lane] : 0u;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const long i = (g * 4 + j) * 32 + lane;
            const float v = bfp_dequant<BB>(bfp_warp_unpack_q<BB>(wv[j]), (ew >> (8 * j)) & 255);
            if (i < n) out[i] = v;
        }
    }
}

// n reals (even): complex c = (x[2c], x[2c+1])
extern "C" __global__ void __launch_bounds__(256) bfp_pack_hw(const float2* __restrict__ x, unsigned* __restrict__ mant, unsigned char* __restrict__ ex, long n) {
    const int lane = threadIdx.x & 31, t = lane & 15, h = lane >> 4;
    const long nblk = (n + 31) >> 5;
    for (long g = GTID >> 5; g * 4 < nblk; g += GSTRIDE >> 5) {
        float2 z[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const long c = (g * 4 + 2 * r + h) * 16 + t;
            z[r] = 2 * c < n ? x[c] : make_float2(0.f, 0.f);
        }
        unsigned ew = 0u;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const long blk = g * 4 + 2 * r + h;
            // half-warp-uniform branch: at the tail one half skips the codec (tests the half-warp shuffle masks)
            if (blk < nblk) {
                const int es = bfp_hw_store_c<BB>(z[r], mant, ex, blk, true, false);
                ew |= (unsigned)es << (8 * (2 * r + h));
            }
        }
        ew |= __shfl_xor_sync(BFP_FULL, ew, 16);
        if (lane == 0) {
            if (g * 4 + 3 < nblk) reinterpret_cast<unsigned*>(ex)[g] = ew;
            else
                for (int j = 0; j < 4; ++j)
                    if (g * 4 + j < nblk) ex[g * 4 + j] = (unsigned char)(ew >> (8 * j));
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) bfp_unpack_hw(const unsigned* __restrict__ mant, const unsigned char* __restrict__ ex, float2* __restrict__ out, long n) {
    const int lane = threadIdx.x & 31, t = lane & 15, h = lane >> 4;
    const long nblk = (n + 31) >> 5;
    for (long g = GTID >> 5; g * 4 < nblk; g += GSTRIDE >> 5) {
        unsigned ew = 0u;
        if (g * 4 + 3 < nblk) ew = reinterpret_cast<const unsigned*>(ex)[g];
        else
            for (int j = 0; j < 4; ++j)
                if (g * 4 + j < nblk) ew |= (unsigned)ex[g * 4 + j] << (8 * j);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const long blk = g * 4 + 2 * r + h;
            if (blk < nblk) {
                const float2 z = bfp_hw_load_c<BB>(mant, (ew >> (8 * (2 * r + h))) & 255, blk, true);
                const long c = blk * 16 + t;
                if (2 * c < n) out[c] = z;
            }
        }
    }
}

// per-thread runs: thread p owns run p (values [pV, pV + V)); the 32 / V lanes of a block are consecutive
extern "C" __global__ void __launch_bounds__(256) bfp_pack_run(const float* __restrict__ x, unsigned* __restrict__ mant, unsigned char* __restrict__ ex, long n) {
    constexpr int V = BfpT<BB>::V, G = 32 / V;
    const int lane = threadIdx.x & 31;
    const long nrun = ((n + 31) >> 5) * G;
    for (long p = GTID; p - lane < nrun; p += GSTRIDE) {
        float v[V];
        const long i0 = p * V;
        if (p < nrun && i0 + V <= n) {
            if constexpr (V % 4 == 0) {
#pragma unroll
                for (int k = 0; k < V / 4; ++k) {
                    const float4 f = reinterpret_cast<const float4*>(x + i0)[k];
                    v[4 * k] = f.x; v[4 * k + 1] = f.y; v[4 * k + 2] = f.z; v[4 * k + 3] = f.w;
                }
            } else {
#pragma unroll
                for (int k = 0; k < V / 2; ++k) {
                    const float2 f = reinterpret_cast<const float2*>(x + i0)[k];
                    v[2 * k] = f.x; v[2 * k + 1] = f.y;
                }
            }
        } else {
#pragma unroll
            for (int k = 0; k < V; ++k) v[k] = (p < nrun && i0 + k < n) ? x[i0 + k] : 0.f;
        }
        const int es = bfp_run_estore<BB>(v);
        if (p < nrun) {
            bfp_run_store<BB>(v, es, mant, p);
            if (p % G == 0) ex[p / G] = (unsigned char)es;
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) bfp_unpack_run(const unsigned* __restrict__ mant, const unsigned char* __restrict__ ex, float* __restrict__ out, long n) {
    constexpr int V = BfpT<BB>::V, G = 32 / V;
    const long nrun = ((n + 31) >> 5) * G;
    for (long p = GTID; p < nrun; p += GSTRIDE) {
        float v[V];
        bfp_run_load<BB>(mant, ex[p / G], p, v);
        const long i0 = p * V;
        if (i0 + V <= n) {
            if constexpr (V % 4 == 0) {
#pragma unroll
                for (int k = 0; k < V / 4; ++k) reinterpret_cast<float4*>(out + i0)[k] = make_float4(v[4 * k], v[4 * k + 1], v[4 * k + 2], v[4 * k + 3]);
            } else {
#pragma unroll
                for (int k = 0; k < V / 2; ++k) reinterpret_cast<float2*>(out + i0)[k] = make_float2(v[2 * k], v[2 * k + 1]);
            }
        } else {
            for (int k = 0; k < V; ++k)
                if (i0 + k < n) out[i0 + k] = v[k];
        }
    }
}

// random access: thread c decodes complex c (n reals, even)
extern "C" __global__ void __launch_bounds__(256) bfp_unpack_rand(const unsigned* __restrict__ mant, const unsigned char* __restrict__ ex, float2* __restrict__ out, long n) {
    for (long c = GTID; 2 * c < n; c += GSTRIDE) out[c] = bfp_load_c<BB>(mant, ex, c);
}

extern "C" __global__ void __launch_bounds__(256) bfp_unpack_rand_r(const unsigned* __restrict__ mant, const unsigned char* __restrict__ ex, float* __restrict__ out, long n) {
    for (long i = GTID; i < n; i += GSTRIDE) out[i] = bfp_load_r<BB>(mant, ex, i);
}

extern "C" __global__ void __launch_bounds__(256) copy_f32(const float4* __restrict__ x, float4* __restrict__ y, long n4) {
    for (long i = GTID; i < n4; i += GSTRIDE) y[i] = x[i];
}

extern "C" __global__ void __launch_bounds__(256) copy_u32(const unsigned* __restrict__ x, unsigned* __restrict__ y, long n) {
    for (long i = GTID; i < n; i += GSTRIDE) y[i] = x[i];
}
