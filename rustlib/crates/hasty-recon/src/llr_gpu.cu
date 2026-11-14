// GPU locally-low-rank denoiser helpers (batched Casorati blocks, column-major R x T per block).
struct __align__(8) cx { float x, y; };
struct __align__(16) zx { double x, y; };

// b = (double) a
extern "C" __global__ void c2z(const cx* __restrict__ a, zx* __restrict__ b, long long n)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    cx v = a[i];
    zx o; o.x = v.x; o.y = v.y;
    b[i] = o;
}

// Columns [t - k, t) of every block's T x T eigenvector matrix (f64), as f32: out[j][r + t * c] = V[j][r + t * (t - k + c)]
extern "C" __global__ void top_cols(const zx* __restrict__ v, cx* __restrict__ out, int t, int k, long long nb)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long per = (long long)t * k;
    if (i >= per * nb) return;
    long long j = i / per, e = i % per;
    zx z = v[j * (long long)t * t + (long long)t * (t - k) + e];
    cx o; o.x = (float)z.x; o.y = (float)z.y;
    out[i] = o;
}

// P[j][r + R * c] *= w[j * k + c]  (P: R x k per block)
extern "C" __global__ void scale_cols(cx* __restrict__ p, const float* __restrict__ w, int r, int k, long long nb)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long per = (long long)r * k;
    if (i >= per * nb) return;
    long long j = i / per, c = (i % per) / r;
    float s = w[j * k + c];
    p[i].x *= s;
    p[i].y *= s;
}
