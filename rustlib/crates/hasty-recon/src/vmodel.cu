// Velocity-parametrised 4D flow model (vmodel.rs): pointwise evaluation, vector-Jacobian and Jacobian-vector products,
// Gauss-Newton metric pieces. Parameters per voxel: z_k (complex, K_z nodes), v_k (3 reals, K_v nodes, component-major
// [3][nv] per node), global residual-phase coefficients a[(e-1)*np + p] (Legendre, total degree <= d).
//   x_{e,f}(r) = z_f(r) exp(i (phi_e(r) - kenc_e . v_f(r))),  z_f = sum_k wz[f,k] z_k,  v_f = sum_k wv[f,k] v_k
struct __align__(8) cx { float x, y; };
typedef unsigned long long u64;

__device__ __forceinline__ float leg(int n, float u)
{
    return n == 0 ? 1.f : n == 1 ? u : n == 2 ? 0.5f * (3.f * u * u - 1.f) : 0.5f * u * (5.f * u * u - 3.f);
}
__device__ __forceinline__ float poly(const int* ex, int p, long long r, int nx, int ny, int nz)
{
    int i = (int)(r % nx), j = (int)((r / nx) % ny), k = (int)(r / ((long long)nx * ny));
    float ux = (2.f * i + 1.f) / nx - 1.f, uy = (2.f * j + 1.f) / ny - 1.f, uz = (2.f * k + 1.f) / nz - 1.f;
    return leg(ex[3 * p], ux) * leg(ex[3 * p + 1], uy) * leg(ex[3 * p + 2], uz);
}

// z_f, v_f and the phase of encode e at voxel r
__device__ __forceinline__ void state(const u64* zp, const float* wz, int kz, const u64* vp, const float* wv, int kv,
                                      const float* a, const int* ex, int np, int e, const float* ke,
                                      long long r, long long nv, int nx, int ny, int nz, cx* z, float* v, float* ph)
{
    z->x = 0.f; z->y = 0.f;
    for (int k = 0; k < kz; k++) {
        float w = wz[k];
        if (w != 0.f) { cx zk = ((const cx*)zp[k])[r]; z->x += w * zk.x; z->y += w * zk.y; }
    }
    v[0] = v[1] = v[2] = 0.f;
    for (int k = 0; k < kv; k++) {
        float w = wv[k];
        if (w != 0.f) {
            const float* vk = (const float*)vp[k];
            v[0] += w * vk[r]; v[1] += w * vk[nv + r]; v[2] += w * vk[2 * nv + r];
        }
    }
    float p = -(ke[0] * v[0] + ke[1] * v[1] + ke[2] * v[2]);
    if (e > 0) for (int q = 0; q < np; q++) p += a[(e - 1) * np + q] * poly(ex, q, r, nx, ny, nz);
    *ph = p;
}

extern "C" __global__ void vm_eval(const u64* zp, const float* wz, int kz, const u64* vp, const float* wv, int kv,
                                   const float* a, const int* ex, int np, int e, const float* ke,
                                   int nx, int ny, int nz, cx* x)
{
    long long nv = (long long)nx * ny * nz;
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (r >= nv) return;
    cx z; float v[3], ph;
    state(zp, wz, kz, vp, wv, kv, a, ex, np, e, ke, r, nv, nx, ny, nz, &z, v, &ph);
    float s, c;
    sincosf(ph, &s, &c);
    cx o; o.x = z.x * c - z.y * s; o.y = z.x * s + z.y * c;
    x[r] = o;
}

// gradient accumulation for one (encode, frame) from g = dF/dx* (df = Re <g, dx>):
//   Gz_k += wz[k] conj(u) g,  Gv_k[a] += wv[k] ke[a] Im(conj(g) x),  sacc += Im(conj(g) x)  (residual phase, later
//   reduced against the polynomials with a minus sign); `scale` multiplies g (set weight)
extern "C" __global__ void vm_vjp(const u64* zp, const float* wz, int kz, const u64* vp, const float* wv, int kv,
                                  const float* a, const int* ex, int np, int e, const float* ke,
                                  int nx, int ny, int nz, const cx* g, float scale, const u64* gzp, const u64* gvp, float* sacc)
{
    long long nv = (long long)nx * ny * nz;
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (r >= nv) return;
    cx z; float v[3], ph;
    state(zp, wz, kz, vp, wv, kv, a, ex, np, e, ke, r, nv, nx, ny, nz, &z, v, &ph);
    float s, c;
    sincosf(ph, &s, &c);
    cx gr = g[r]; gr.x *= scale; gr.y *= scale;
    cx t; t.x = c * gr.x + s * gr.y; t.y = c * gr.y - s * gr.x;   // conj(u) g
    cx x; x.x = z.x * c - z.y * s; x.y = z.x * s + z.y * c;
    float im = gr.x * x.y - gr.y * x.x;                              // Im(conj(g) x)
    for (int k = 0; k < kz; k++) {
        float w = wz[k];
        if (w != 0.f) { cx* G = (cx*)gzp[k]; cx o = G[r]; o.x += w * t.x; o.y += w * t.y; G[r] = o; }
    }
    for (int k = 0; k < kv; k++) {
        float w = wv[k];
        if (w != 0.f) {
            float* G = (float*)gvp[k];
            G[r] += w * ke[0] * im; G[nv + r] += w * ke[1] * im; G[2 * nv + r] += w * ke[2] * im;
        }
    }
    if (sacc) sacc[r] += im;
}

// linearised model: dx = u dz_f + x i (dphi_e - ke . dv_f)
extern "C" __global__ void vm_jvp(const u64* zp, const float* wz, int kz, const u64* vp, const float* wv, int kv,
                                  const float* a, const int* ex, int np, int e, const float* ke,
                                  int nx, int ny, int nz, const u64* dzp, const u64* dvp, const float* da, cx* dx)
{
    long long nv = (long long)nx * ny * nz;
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (r >= nv) return;
    cx z; float v[3], ph;
    state(zp, wz, kz, vp, wv, kv, a, ex, np, e, ke, r, nv, nx, ny, nz, &z, v, &ph);
    cx dz; float dv[3], dph;
    state(dzp, wz, kz, dvp, wv, kv, da, ex, np, e, ke, r, nv, nx, ny, nz, &dz, dv, &dph);  // linear in the parameters
    float s, c;
    sincosf(ph, &s, &c);
    cx x; x.x = z.x * c - z.y * s; x.y = z.x * s + z.y * c;
    cx o;
    o.x = dz.x * c - dz.y * s - x.y * dph;
    o.y = dz.x * s + dz.y * c + x.x * dph;
    dx[r] = o;
}

// partial sums of sum_r P_p(r)^pw s[r] for every p (blockIdx.y = p), double accumulation
extern "C" __global__ void vm_poly_reduce(const float* sv, const int* ex, int np, int pw, int nx, int ny, int nz, double* part)
{
    __shared__ double sh[256];
    long long nv = (long long)nx * ny * nz;
    int p = blockIdx.y;
    double acc = 0.0;
    for (long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x; r < nv; r += (long long)gridDim.x * blockDim.x) {
        float q = poly(ex, p, r, nx, ny, nz);
        acc += (double)(pw == 2 ? q * q : q) * sv[r];
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) part[p * gridDim.x + blockIdx.x] = sh[0];
}

// Gauss-Newton weights: sig[k][r] = sum_f wv[f,k] |z_f(r)|^2 (all frames), zz[r] = sum_f |z_f(r)|^2
extern "C" __global__ void vm_sigma(const u64* zp, const float* wz_all, int kz, const float* wv_all, int kv, int nf,
                                    long long nv, float* sig, float* zz)
{
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (r >= nv) return;
    float acc[64];
    for (int k = 0; k < kv; k++) acc[k] = 0.f;
    float tot = 0.f;
    for (int f = 0; f < nf; f++) {
        cx z; z.x = 0.f; z.y = 0.f;
        for (int k = 0; k < kz; k++) {
            float w = wz_all[f * kz + k];
            if (w != 0.f) { cx zk = ((const cx*)zp[k])[r]; z.x += w * zk.x; z.y += w * zk.y; }
        }
        float m = z.x * z.x + z.y * z.y;
        tot += m;
        for (int k = 0; k < kv; k++) acc[k] += wv_all[f * kv + k] * m;
    }
    for (int k = 0; k < kv; k++) sig[k * nv + r] = acc[k];
    zz[r] = tot;
}

// z-chunk metric: out = r / b (inverse: r * b), b = wsum * c[r] + damp
extern "C" __global__ void vm_metric_z(const cx* r_, cx* out, const float* c, float wsum, float damp, int inverse, long long nv)
{
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (r >= nv) return;
    float b = wsum * (c ? c[r] : 1.f) + damp;
    float f = inverse ? b : 1.f / b;
    cx v = r_[r]; v.x *= f; v.y *= f;
    out[r] = v;
}

// v-chunk metric: out = Minv r / b (inverse: b M r), b = c[r] sig_k[r] + damp, per voxel 3-vectors [3][nv]
// (+ aw * w[r]: the static-tissue anchor's curvature, trace-matched onto M)
extern "C" __global__ void vm_metric_v(const float* r_, float* out, const float* c, const float* sig, const float* M, const float* Minv,
                                       float damp, int inverse, long long nv, const float* w, float aw)
{
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (r >= nv) return;
    float b = (c ? c[r] : 1.f) * sig[r] + damp + (w ? aw * w[r] : 0.f);
    float v0 = r_[r], v1 = r_[nv + r], v2 = r_[2 * nv + r];
    const float* A = inverse ? M : Minv;
    float f = inverse ? b : 1.f / b;
    out[r] = f * (A[0] * v0 + A[1] * v1 + A[2] * v2);
    out[nv + r] = f * (A[3] * v0 + A[4] * v1 + A[5] * v2);
    out[2 * nv + r] = f * (A[6] * v0 + A[7] * v1 + A[8] * v2);
}

// static-tissue anchor on one velocity node: f += 1/2 lam sum_r w[r] |v(r)|^2, grad += lam w v  ([3][nv] floats)
extern "C" __global__ void vm_anchor(const float* v, const float* w, float lam, float* grad, long long nv, double* part)
{
    __shared__ double sh[256];
    double acc = 0.0;
    for (long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x; r < nv; r += (long long)gridDim.x * blockDim.x) {
        float wr = w[r];
        if (wr == 0.f) continue;
        float a = v[r], b = v[nv + r], c = v[2 * nv + r];
        acc += 0.5 * lam * wr * ((double)a * a + (double)b * b + (double)c * c);
        if (grad) { grad[r] += lam * wr * a; grad[nv + r] += lam * wr * b; grad[2 * nv + r] += lam * wr * c; }
    }
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0 && part) part[blockIdx.x] = sh[0];
}
