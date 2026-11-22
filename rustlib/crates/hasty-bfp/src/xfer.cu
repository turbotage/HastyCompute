// xfer.cu: BFP pack before a device -> host copy and decode after a host -> device copy (research/bfp_storage.md §5:
// the host link is the one place where a standalone conversion pass pays, its DRAM time is ~1/40 of the PCIe time it
// saves). Prepended by bfp.cuh; compiled once per width with -DBFP_B=<bits>.
//
// Layout: `rows` rows of `n` reals, row r at x + r * n; every row is coded as its own block sequence of
// rb = ceil(n / 32) blocks: mantissa words mant[(r * rb + k) * B ..], exponent bytes ex[r * rb + k]. One row is a
// plain BFP buffer (the gradient chunk of a set); several rows are the coil rows of a k-space set, so a coil subset is
// a row subset. Bit-identical to the host codec (hasty_bfp::Bfp::encode_serial per row).
//
// Per-thread runs (bfp_run_*): thread p owns run pr = p mod (rb * G) of row p / (rb * G), V values in W words; the
// block exponent is the max over the G = 32 / V lanes of the block (aligned lane groups; rows hold whole blocks, so a
// group never spans two rows). All lanes of a warp take part in every iteration (warp-uniform loop condition).

#ifndef BFP_B
#error "define BFP_B"
#endif
#define XB BFP_B

#define XGTID ((long)blockIdx.x * blockDim.x + threadIdx.x)
#define XGSTRIDE ((long)gridDim.x * blockDim.x)

extern "C" __global__ void __launch_bounds__(256) bfp_xfer_pack(const float* __restrict__ x, unsigned* __restrict__ mant, unsigned char* __restrict__ ex, long n, long rows) {
    constexpr int V = BfpT<XB>::V, G = 32 / V, W = BfpT<XB>::W;
    const int lane = threadIdx.x & 31;
    const long rb = (n + 31) >> 5, rr = rb * G, nrun = rows * rr;
    for (long p = XGTID; p - lane < nrun; p += XGSTRIDE) {
        float v[V];
        const long row = p / rr, pr = p - row * rr, i0 = pr * V;
        const float* xr = x + row * n;
        if (p < nrun && i0 + V <= n && (n & 1) == 0) {
#pragma unroll
            for (int k = 0; k < V / 2; ++k) {
                const float2 f = reinterpret_cast<const float2*>(xr + i0)[k];
                v[2 * k] = f.x;
                v[2 * k + 1] = f.y;
            }
        } else {
#pragma unroll
            for (int k = 0; k < V; ++k) v[k] = (p < nrun && i0 + k < n) ? xr[i0 + k] : 0.f;
        }
        const int es = bfp_run_estore<XB>(v);
        if (p < nrun) {
            bfp_run_store<XB>(v, es, mant + row * rb * XB, pr);
            if (pr % G == 0) ex[row * rb + pr / G] = (unsigned char)es;
        }
        (void)W;
    }
}

extern "C" __global__ void __launch_bounds__(256) bfp_xfer_unpack(const unsigned* __restrict__ mant, const unsigned char* __restrict__ ex, float* __restrict__ out, long n, long rows) {
    constexpr int V = BfpT<XB>::V, G = 32 / V;
    const long rb = (n + 31) >> 5, rr = rb * G, nrun = rows * rr;
    for (long p = XGTID; p < nrun; p += XGSTRIDE) {
        const long row = p / rr, pr = p - row * rr, i0 = pr * V;
        float v[V];
        bfp_run_load<XB>(mant + row * rb * XB, ex[row * rb + pr / G], pr, v);
        float* o = out + row * n;
        if (i0 + V <= n && (n & 1) == 0) {
#pragma unroll
            for (int k = 0; k < V / 2; ++k) reinterpret_cast<float2*>(o + i0)[k] = make_float2(v[2 * k], v[2 * k + 1]);
        } else {
            for (int k = 0; k < V; ++k)
                if (i0 + k < n) o[i0 + k] = v[k];
        }
    }
}
