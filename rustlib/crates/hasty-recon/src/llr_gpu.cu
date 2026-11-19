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

// Device-resident stack (LlrGpu::apply_dev): idx[b][row] = voxel of row `row` of block b (0xffffffff: zero row).
// dc[j][row + R t] = x[t nv + idx[b0 + j][row]], j < nb
extern "C" __global__ void gather_blocks(const cx* __restrict__ x, const unsigned* __restrict__ idx, cx* __restrict__ dc, int r, int t, long long nv, long long b0, long long nb)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long per = (long long)r * t;
    if (i >= per * nb) return;
    long long j = i / per, e = i % per;
    int tt = (int)(e / r), row = (int)(e % r);
    unsigned q = idx[(b0 + j) * r + row];
    cx o; o.x = 0.f; o.y = 0.f;
    if (q != 0xffffffffu) o = x[tt * nv + q];
    dc[i] = o;
}

// x[t nv + idx[b0 + j][row]] = dc[j][row + R t] for blocks with ok[j] != 0
extern "C" __global__ void scatter_blocks(const cx* __restrict__ dc, const unsigned* __restrict__ idx, const int* __restrict__ ok, cx* __restrict__ x, int r, int t, long long nv, long long b0, long long nb)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    long long per = (long long)r * t;
    if (i >= per * nb) return;
    long long j = i / per, e = i % per;
    if (!ok[j]) return;
    int tt = (int)(e / r), row = (int)(e % r);
    unsigned q = idx[(b0 + j) * r + row];
    if (q != 0xffffffffu) x[tt * nv + q] = dc[i];
}
