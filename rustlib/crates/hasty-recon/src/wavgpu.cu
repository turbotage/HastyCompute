// GPU smoothed wavelet prior of knotncg (wavgpu.rs): db4 separable periodic 3-D transform on the zero-padded box and the
// Huber reductions over its detail coefficients. Conventions as sepwt.rs / knotncg::WavHuber: per image a box p (x
// fastest, every axis a multiple of 2^levels), complex as float2 (= 2 interleaved lanes), in-place Mallat layout (level
// l works on the sub-box s = p >> l, low band first along each axis), periodic boundary, 8-tap orthonormal bank
// (analysis lo[k] = sum_t h[t] x[(2k + t) mod n], hi with g[t] = (-1)^t h[7 - t]; synthesis = the transpose).
// Images of a stack are np apart (np = p0 p1 p2 complex); the knot index is blockIdx.z (transforms) or blockIdx.y
// (pointwise kernels and reductions).
#define NT 256
#define MAXM 8

__constant__ float H[8] = {(float)(-0.010597401784997278), (float)(0.032883011666982945), (float)(0.030841381835986965), (float)(-0.18703481171888114),
                           (float)(-0.02798376941698385), (float)(0.6308807679295904), (float)(0.7148465705525415), (float)(0.23037781330885523)};

__device__ __forceinline__ float Gh(int t) { const float v = H[7 - t]; return (t & 1) ? -v : v; }

// output m of a line of n elements (stride st in shared memory): analysis (lo -> m < n/2, hi -> m >= n/2) or synthesis
// from [lo | hi]; taps accumulated in the host's order (sepwt.rs: analysis t = 0..7, synthesis the lo taps then the hi)
__device__ __forceinline__ float2 line_out(const float2* a, int n, int st, int m, int inv)
{
    float2 r = make_float2(0.f, 0.f);
    const int h = n >> 1;
    if (!inv) {
        const bool hi = m >= h;
        const int b = 2 * (hi ? m - h : m);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            int i = b + t;
            while (i >= n) i -= n;
            const float c = hi ? Gh(t) : H[t];
            const float2 v = a[i * st];
            r.x = fmaf(c, v.x, r.x);
            r.y = fmaf(c, v.y, r.y);
        }
    } else {
        const int i0 = m >> 1, par = m & 1;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int k = i0 - j;
            while (k < 0) k += h;
            const float2 v = a[k * st];
            const float c = H[par + 2 * j];
            r.x = fmaf(c, v.x, r.x);
            r.y = fmaf(c, v.y, r.y);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int k = i0 - j;
            while (k < 0) k += h;
            const float2 v = a[(h + k) * st];
            const float c = Gh(par + 2 * j);
            r.x = fmaf(c, v.x, r.x);
            r.y = fmaf(c, v.y, r.y);
        }
    }
    return r;
}

// x pass of the sub-box (s0 along x) on rows y < s1 of plane blockIdx.y: lpb rows per block (coalesced loads of whole
// rows into shared memory, outputs written in place). grid (ceil(s1 / lpb), s2, nk), shared lpb * s0 float2
extern "C" __global__ void wt_x(float2* __restrict__ buf, long long np, int p0, int p1, int s0, int s1, int lpb, int inv)
{
    extern __shared__ float2 sm[];
    const int y0 = blockIdx.x * lpb;
    const int nl = min(lpb, s1 - y0);
    float2* img = buf + (long long)blockIdx.z * np + ((long long)blockIdx.y * p1 + y0) * p0;
    const int tot = nl * s0;
    for (int q = threadIdx.x; q < tot; q += blockDim.x) {
        const int l = q / s0, x = q - l * s0;
        sm[q] = img[(long long)l * p0 + x];
    }
    __syncthreads();
    for (int q = threadIdx.x; q < tot; q += blockDim.x) {
        const int l = q / s0, x = q - l * s0;
        img[(long long)l * p0 + x] = line_out(sm + l * s0, s0, 1, x, inv);
    }
}

// y or z pass: lines of n elements at stride es (p0 for y, p0 p1 for z) through a tile of tx consecutive x (< s0, rows
// of the tile coalesced); the remaining in-plane axis is blockIdx.y at stride os. grid (ceil(s0 / tx), s_other, nk),
// shared n * tx float2 (element i of column xx at i * tx + xx: conflict-free over a warp)
extern "C" __global__ void wt_yz(float2* __restrict__ buf, long long np, int s0, int n, long long es, long long os, int tx, int inv)
{
    extern __shared__ float2 sm[];
    const int x0 = blockIdx.x * tx;
    const int w = min(tx, s0 - x0);
    float2* img = buf + (long long)blockIdx.z * np + (long long)blockIdx.y * os + x0;
    const int tot = n * tx;
    for (int q = threadIdx.x; q < tot; q += blockDim.x) {
        const int i = q / tx, xx = q - i * tx;
        if (xx < w) sm[q] = img[i * es + xx];
    }
    __syncthreads();
    for (int q = threadIdx.x; q < tot; q += blockDim.x) {
        const int i = q / tx, xx = q - i * tx;
        if (xx < w) img[i * es + xx] = line_out(sm + xx, n, tx, i, inv);
    }
}

// box[q] = image q (dims d) zero-padded into the box p. grid (.., nk)
extern "C" __global__ void pad_zero(const float2* __restrict__ img, float2* __restrict__ box, int d0, int d1, int d2, int p0, int p1, long long nv, long long np)
{
    const long long q = blockIdx.y;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < np; i += (long long)gridDim.x * blockDim.x) {
        const int x = (int)(i % p0);
        const long long r = i / p0;
        const int y = (int)(r % p1), z = (int)(r / p1);
        box[q * np + i] = (x < d0 && y < d1 && z < d2) ? img[q * nv + x + (long long)d0 * (y + (long long)d1 * z)] : make_float2(0.f, 0.f);
    }
}

// out[q][v] = a box[q][S v] + b out[q][v] (b = 0: out is not read). grid (.., nk)
extern "C" __global__ void crop_axpby(const float2* __restrict__ box, float2* __restrict__ out, int d0, int d1, int p0, int p1, long long nv, long long np, float a, float b)
{
    const long long q = blockIdx.y;
    for (long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x; v < nv; v += (long long)gridDim.x * blockDim.x) {
        const int x = (int)(v % d0);
        const long long r = v / d0;
        const int y = (int)(r % d1), z = (int)(r / d1);
        const float2 c = box[q * np + x + (long long)p0 * (y + (long long)p1 * z)];
        float2 o = make_float2(a * c.x, a * c.y);
        if (b != 0.f) {
            const float2 u = out[q * nv + v];
            o.x = fmaf(b, u.x, o.x);
            o.y = fmaf(b, u.y, o.y);
        }
        out[q * nv + v] = o;
    }
}

// detail coefficient? (the approximation block is x < a0, y < a1, z < a2 of the box)
__device__ __forceinline__ bool detail(long long i, int p0, int p1, int a0, int a1, int a2)
{
    const int x = (int)(i % p0);
    const long long r = i / p0;
    const int y = (int)(r % p1), z = (int)(r / p1);
    return !(x < a0 && y < a1 && z < a2);
}

// h_gamma(|c|) with theta = th, 1/gamma = ig (as knotncg::huber, same operation order, no contraction); *w = psi(c) / c
__device__ __forceinline__ float huber(float cr, float ci, float th, float ig, float* w)
{
    const float m = sqrtf(__fadd_rn(__fmul_rn(cr, cr), __fmul_rn(ci, ci)));
    if (m <= th) {
        *w = ig;
        return __fmul_rn(__fmul_rn(__fmul_rn(0.5f, m), m), ig);
    }
    *w = __fdiv_rn(__fmul_rn(th, ig), m);
    return __fmul_rn(__fsub_rn(__fmul_rn(th, m), __fmul_rn(__fmul_rn(0.5f, th), th)), ig);
}

// block sum of nv doubles per thread (v[0..nv)) into out[0..nv) (blockDim = NT)
template <int NV>
__device__ __forceinline__ void block_sum(double (&v)[NV], int nv, double* out)
{
    __shared__ double sh[NV][NT / 32];
    const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < NV; ++j) {
        if (j >= nv) break;
        double s = v[j];
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if (lane == 0) sh[j][wp] = s;
    }
    __syncthreads();
    if (wp == 0) {
        for (int j = 0; j < nv; ++j) {
            double s = lane < NT / 32 ? sh[j][lane] : 0.0;
            for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
            if (lane == 0) out[j] = s;
        }
    }
}

// sum of h over the detail coefficients of c (partial per block at part[blockIdx.y * gridDim.x + blockIdx.x]); write:
// psi = w c on the detail coefficients, 0 on the approximation (psi may be c itself). grid (nbx, nk)
extern "C" __global__ void huber_psi(const float2* c, float2* psi, double* __restrict__ part, int p0, int p1, int a0, int a1, int a2, long long np, float th, float ig, int write)
{
    const long long q = blockIdx.y;
    double v[1] = {0.0};
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < np; i += (long long)gridDim.x * blockDim.x) {
        const float2 z = c[q * np + i];
        float2 o = make_float2(0.f, 0.f);
        if (detail(i, p0, p1, a0, a1, a2)) {
            float w;
            v[0] += (double)huber(z.x, z.y, th, ig, &w);
            o = make_float2(w * z.x, w * z.y);
        }
        if (write) psi[q * np + i] = o;
    }
    block_sum<1>(v, 1, part + (q * gridDim.x + blockIdx.x));
}

// along c(a) = cx + a cd at up to MAXM steps a_j: partial (sum h(c(a_j)), sum Re<psi(c(a_j)), cd>) per block at
// part[(blockIdx.y * gridDim.x + blockIdx.x) * 2m + {2j, 2j + 1}]; one read of cx, cd for all steps. grid (nbx, nk)
extern "C" __global__ void huber_line(const float2* __restrict__ cx, const float2* __restrict__ cd, double* __restrict__ part, int p0, int p1, int a0, int a1, int a2, long long np, float th, float ig, int m,
                                      float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7)
{
    const long long q = blockIdx.y;
    const float a[MAXM] = {s0, s1, s2, s3, s4, s5, s6, s7};
    double v[2 * MAXM];
#pragma unroll
    for (int j = 0; j < 2 * MAXM; ++j) v[j] = 0.0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < np; i += (long long)gridDim.x * blockDim.x) {
        if (!detail(i, p0, p1, a0, a1, a2)) continue;
        const float2 x = cx[q * np + i], d = cd[q * np + i];
#pragma unroll
        for (int j = 0; j < MAXM; ++j) {
            if (j >= m) break;
            const float cr = __fadd_rn(x.x, __fmul_rn(a[j], d.x)), ci = __fadd_rn(x.y, __fmul_rn(a[j], d.y));
            float w;
            v[2 * j] += (double)huber(cr, ci, th, ig, &w);
            v[2 * j + 1] += (double)__fmul_rn(w, __fadd_rn(__fmul_rn(cr, d.x), __fmul_rn(ci, d.y)));
        }
    }
    block_sum<2 * MAXM>(v, 2 * m, part + (q * gridDim.x + blockIdx.x) * 2 * m);
}

// out[j] (+)= sum_b part[b * nval + j], one block per value j
extern "C" __global__ void sum_parts(const double* __restrict__ part, int nparts, int nval, double* __restrict__ out, int acc)
{
    const int j = blockIdx.x;
    double v[1] = {0.0};
    for (int b = threadIdx.x; b < nparts; b += blockDim.x) v[0] += part[(long long)b * nval + j];
    double r[1];
    block_sum<1>(v, 1, r);
    if (threadIdx.x == 0) out[j] = acc ? out[j] + r[0] : r[0];
}

// soft threshold of the detail coefficients at t (the exact l1 prox); the approximation is kept. grid (.., nk)
extern "C" __global__ void huber_soft(float2* __restrict__ c, int p0, int p1, int a0, int a1, int a2, long long np, float t)
{
    const long long q = blockIdx.y;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < np; i += (long long)gridDim.x * blockDim.x) {
        if (!detail(i, p0, p1, a0, a1, a2)) continue;
        const float2 z = c[q * np + i];
        const float m = sqrtf(__fadd_rn(__fmul_rn(z.x, z.x), __fmul_rn(z.y, z.y)));
        const float s = m > t ? 1.f - t / m : 0.f;
        c[q * np + i] = make_float2(s * z.x, s * z.y);
    }
}

// ADMM wavelet split in the coefficient domain (knotncg::admm): t = c + v, u = soft_t(t) (detail; approximation u = t),
// v <- t - u, c <- u - v (the centre part). grid (.., nk)
extern "C" __global__ void admm_wav(float2* __restrict__ c, float2* __restrict__ vw, int p0, int p1, int a0, int a1, int a2, long long np, float t)
{
    const long long q = blockIdx.y;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < np; i += (long long)gridDim.x * blockDim.x) {
        const long long j = q * np + i;
        const float2 a = c[j], b = vw[j];
        const float2 tt = make_float2(a.x + b.x, a.y + b.y);
        float2 u = tt;
        if (detail(i, p0, p1, a0, a1, a2)) {
            const float m = sqrtf(tt.x * tt.x + tt.y * tt.y);
            const float s = m > t ? 1.f - t / m : 0.f;
            u = make_float2(s * tt.x, s * tt.y);
        }
        const float2 v = make_float2(tt.x - u.x, tt.y - u.y);
        vw[j] = v;
        c[j] = make_float2(u.x - v.x, u.y - v.y);
    }
}
