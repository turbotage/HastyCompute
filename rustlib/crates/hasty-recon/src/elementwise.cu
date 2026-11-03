// Elementwise / reduction kernels for the reconstruction (interleaved complex float2).
struct __align__(8) cx { float x, y; };
__device__ __forceinline__ cx cmul(cx a, cx b) { cx r; r.x = a.x * b.x - a.y * b.y; r.y = a.x * b.y + a.y * b.x; return r; }
__device__ __forceinline__ cx cmulc(cx a, cx b) { cx r; r.x = a.x * b.x + a.y * b.y; r.y = a.x * b.y - a.y * b.x; return r; } // conj(a) * b

// out[c][v] = s[c][v] * (d ? d[v] : 1) * x[v]
extern "C" __global__ void coil_expand(const cx* __restrict__ x, const cx* __restrict__ d, const cx* __restrict__ s, cx* __restrict__ out, long long nv, int nc)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= nv * nc) return;
    long long v = i % nv;
    cx xv = x[v];
    if (d) xv = cmul(d[v], xv);
    out[i] = cmul(s[i], xv);
}

// out[v] (+)= conj(d[v]) * sum_c conj(s[c][v]) y[c][v]
extern "C" __global__ void coil_reduce(const cx* __restrict__ y, const cx* __restrict__ d, const cx* __restrict__ s, cx* __restrict__ out, long long nv, int nc, int acc)
{
    long long v = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (v >= nv) return;
    cx a; a.x = 0.f; a.y = 0.f;
    for (int c = 0; c < nc; c++) { cx t = cmulc(s[c * nv + v], y[c * nv + v]); a.x += t.x; a.y += t.y; }
    if (d) a = cmulc(d[v], a);
    if (acc) { cx o = out[v]; a.x += o.x; a.y += o.y; }
    out[v] = a;
}

// y[c][j] (+)= w[j] * z[c][j]   (conj_w: conj(w))
extern "C" __global__ void sample_weight(const cx* __restrict__ z, const cx* __restrict__ w, cx* __restrict__ y, long long m, int nc, int conj_w, int acc)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= m * nc) return;
    long long j = i % m;
    cx t = conj_w ? cmulc(w[j], z[i]) : cmul(w[j], z[i]);
    if (acc) { cx o = y[i]; t.x += o.x; t.y += o.y; }
    y[i] = t;
}

// y = a*x + b*y (complex scalars as float2)
extern "C" __global__ void axpby(const cx* __restrict__ x, cx* __restrict__ y, cx a, cx b, long long n)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    cx t = cmul(a, x[i]); cx u = cmul(b, y[i]);
    y[i].x = t.x + u.x; y[i].y = t.y + u.y;
}

// partial sums of conj(x).y in double, one per block (blockDim = 256)
extern "C" __global__ void dot_partial(const cx* __restrict__ x, const cx* __restrict__ y, double* __restrict__ part, long long n)
{
    __shared__ double sr[256], si[256];
    double ar = 0.0, ai = 0.0;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < n; i += (long long)gridDim.x * blockDim.x) {
        cx t = cmulc(x[i], y[i]); ar += t.x; ai += t.y;
    }
    sr[threadIdx.x] = ar; si[threadIdx.x] = ai;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) { sr[threadIdx.x] += sr[threadIdx.x + s]; si[threadIdx.x] += si[threadIdx.x + s]; }
        __syncthreads();
    }
    if (threadIdx.x == 0) { part[2 * blockIdx.x] = sr[0]; part[2 * blockIdx.x + 1] = si[0]; }
}

// out[i] = d[i].x * x[i]  (real diagonal stored in .x)
extern "C" __global__ void mul_real(const cx* __restrict__ x, const cx* __restrict__ d, cx* __restrict__ out, long long n)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = d[i].x; cx v = x[i]; v.x *= s; v.y *= s; out[i] = v;
}
