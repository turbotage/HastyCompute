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

// at-knots mode: |z|^2 of one node chunk, out[r] = w |z|^2 (out != 0), acc[r] += w |z|^2 (acc != 0)
extern "C" __global__ void vm_absq(const cx* z, long long nv, float w, float* out, float* acc)
{
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (r >= nv) return;
    cx v = z[r];
    float m = w * (v.x * v.x + v.y * v.y);
    if (out) out[r] = m;
    if (acc) acc[r] += m;
}

// ─── parameter-spline mode (research/velocity_model_sr.md §F) ─────────────────────────────────────────────────────
// Node chunk k = (m_k, psi_k) packed as cx (x = magnitude, y = phase), v_k [3][nv]. One respiratory row: zrow[c], vrow[c]
// = the chunks of cardiac knots c = 0..mc. Operator frame f of the row reads the <= nn cardiac knots fi[f*nn + j]
// (weights fw; fi < 0: unused):
//   m = sum w m_c, psi = sum w psi_c, v = sum w v_c,  x_e = m exp(i (psi + phi_e - ke'.v))   (ke' = k_e - mean_e k_e)

__device__ __forceinline__ void pstate(const u64* zrow, const u64* vrow, const int* fi, const float* fw, int nn,
                                       const float* a, const int* ex, int np, int e, const float* ke,
                                       long long r, long long nv, int nx, int ny, int nz, float* m, float* psi, float* v, float* ph)
{
    *m = 0.f; *psi = 0.f; v[0] = v[1] = v[2] = 0.f;
    for (int j = 0; j < nn; j++) {
        int c = fi[j];
        if (c < 0) continue;
        float w = fw[j];
        cx z = ((const cx*)zrow[c])[r];
        *m += w * z.x; *psi += w * z.y;
        const float* vk = (const float*)vrow[c];
        v[0] += w * vk[r]; v[1] += w * vk[nv + r]; v[2] += w * vk[2 * nv + r];
    }
    float p = *psi - (ke[0] * v[0] + ke[1] * v[1] + ke[2] * v[2]);
    if (e > 0) for (int q = 0; q < np; q++) p += a[(e - 1) * np + q] * poly(ex, q, r, nx, ny, nz);
    *ph = p;
}

// out[f * nv + r] = x_e at frame f (grid.y = frame)
extern "C" __global__ void vp_eval(const u64* zrow, const u64* vrow, const int* fi, const float* fw, int nn,
                                   const float* a, const int* ex, int np, int e, const float* ke,
                                   int nx, int ny, int nz, cx* out)
{
    long long nv = (long long)nx * ny * nz;
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int f = blockIdx.y;
    if (r >= nv) return;
    float m, psi, v[3], ph;
    pstate(zrow, vrow, fi + f * nn, fw + f * nn, nn, a, ex, np, e, ke, r, nv, nx, ny, nz, &m, &psi, v, &ph);
    float s, c;
    sincosf(ph, &s, &c);
    cx o; o.x = m * c; o.y = m * s;
    out[(long long)f * nv + r] = o;
}

// linearised: dx = u dm + i x (dpsi + dphi_e - ke'.dv)
extern "C" __global__ void vp_jvp(const u64* zrow, const u64* vrow, const u64* dzrow, const u64* dvrow, const int* fi, const float* fw, int nn,
                                  const float* a, const float* da, const int* ex, int np, int e, const float* ke,
                                  int nx, int ny, int nz, cx* out)
{
    long long nv = (long long)nx * ny * nz;
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int f = blockIdx.y;
    if (r >= nv) return;
    float m, psi, v[3], ph, dm, dpsi, dv[3], dph;
    pstate(zrow, vrow, fi + f * nn, fw + f * nn, nn, a, ex, np, e, ke, r, nv, nx, ny, nz, &m, &psi, v, &ph);
    pstate(dzrow, dvrow, fi + f * nn, fw + f * nn, nn, da, ex, np, e, ke, r, nv, nx, ny, nz, &dm, &dpsi, dv, &dph);
    float s, c;
    sincosf(ph, &s, &c);
    cx o;
    o.x = dm * c - m * s * dph;
    o.y = dm * s + m * c * dph;
    out[(long long)f * nv + r] = o;
}

// gradient of one row from g = dF/dx* at the row's frames (g[f * nv + r]), one thread per (voxel, cardiac knot c =
// blockIdx.y) over the frames that read c (ki / kw, <= nkf, -1 padded):
//   G_m += w Re(conj(u) g), G_psi += -w Im(conj(g) x), G_v += w ke' Im(conj(g) x);
// sacc[r] += Im(conj(g) x) of frame f by the thread of knot prim[f] (residual phase, reduced later with a minus sign)
extern "C" __global__ void vp_vjp(const u64* zrow, const u64* vrow, const int* fi, const float* fw, int nn,
                                  const int* ki, const float* kw, int nkf, const int* prim,
                                  const float* a, const int* ex, int np, int e, const float* ke,
                                  int nx, int ny, int nz, const cx* g, float scale, const u64* gzrow, const u64* gvrow, float* sacc)
{
    long long nv = (long long)nx * ny * nz;
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int kc = blockIdx.y;
    if (r >= nv) return;
    float gm = 0.f, gp = 0.f, gv0 = 0.f, gv1 = 0.f, gv2 = 0.f, sa = 0.f;
    for (int j = 0; j < nkf; j++) {
        int f = ki[kc * nkf + j];
        if (f < 0) break;
        float w = kw[kc * nkf + j];
        float m, psi, v[3], ph;
        pstate(zrow, vrow, fi + f * nn, fw + f * nn, nn, a, ex, np, e, ke, r, nv, nx, ny, nz, &m, &psi, v, &ph);
        float s, c;
        sincosf(ph, &s, &c);
        cx gr = g[(long long)f * nv + r]; gr.x *= scale; gr.y *= scale;
        float re = c * gr.x + s * gr.y;                 // Re(conj(u) g)
        float xx = m * c, xy = m * s;
        float im = gr.x * xy - gr.y * xx;               // Im(conj(g) x)
        gm += w * re; gp -= w * im;
        gv0 += w * ke[0] * im; gv1 += w * ke[1] * im; gv2 += w * ke[2] * im;
        if (prim[f] == kc) sa += im;
    }
    cx* G = (cx*)gzrow[kc];
    cx o = G[r]; o.x += gm; o.y += gp; G[r] = o;
    float* V = (float*)gvrow[kc];
    V[r] += gv0; V[nv + r] += gv1; V[2 * nv + r] += gv2;
    if (sacc && sa != 0.f) atomicAdd(&sacc[r], sa);
}

// periodic cardiac interpolation filter in place on `rows` rows of F frames (x[(row * F + f) * nv + r]):
// c_f = sum_n C[n] x_{f-n}, taps |n| <= T (T >= F/2: all F offsets once); one thread per (voxel, row = blockIdx.y)
extern "C" __global__ void vp_prefilter(cx* x, int F, const float* C, int T, long long nv)
{
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (r >= nv) return;
    cx* row = x + (long long)blockIdx.y * F * nv;
    cx buf[128];
    for (int f = 0; f < F; f++) buf[f] = row[(long long)f * nv + r];
    bool full = 2 * T + 1 >= F;
    for (int f = 0; f < F; f++) {
        float ox = 0.f, oy = 0.f;
        if (full) {
            for (int n = 0; n < F; n++) { float w = C[n]; cx b = buf[(f - n + F) % F]; ox += w * b.x; oy += w * b.y; }
        } else {
            for (int n = -T; n <= T; n++) { float w = C[(n + F) % F]; cx b = buf[((f - n) % F + F) % F]; ox += w * b.x; oy += w * b.y; }
        }
        cx o; o.x = ox; o.y = oy;
        row[(long long)f * nv + r] = o;
    }
}

// (m, psi) chunk metric: re / (bm), im / (bp); bm = wm * c[r] + dm, bp = we * (c[r] * sig[r] + dp) (inverse: *)
extern "C" __global__ void vp_metric_mp(const cx* r_, cx* out, const float* c, const float* sig, float wm, float dm, float we, float dp, int inverse, long long nv)
{
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (r >= nv) return;
    float cc = c ? c[r] : 1.f;
    float bm = wm * cc + dm, bp = we * (cc * sig[r] + dp);
    cx v = r_[r];
    if (inverse) { v.x *= bm; v.y *= bp; } else { v.x /= bm; v.y /= bp; }
    out[r] = v;
}

// complex knot images of one respiratory row at the operator frames (the robust start's linear model):
// out[f * nv + r] = sum_j fw[f, j] C_{fi[f, j]}[r]   (grid.y = frame)
extern "C" __global__ void vp_cframes(const u64* crow, const int* fi, const float* fw, int nn, long long nv, cx* out)
{
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int f = blockIdx.y;
    if (r >= nv) return;
    cx o; o.x = 0.f; o.y = 0.f;
    for (int j = 0; j < nn; j++) {
        int c = fi[f * nn + j];
        if (c < 0) continue;
        float w = fw[f * nn + j];
        cx z = ((const cx*)crow[c])[r];
        o.x += w * z.x; o.y += w * z.y;
    }
    out[(long long)f * nv + r] = o;
}

// its adjoint, accumulated: C_c[r] += sum_{f reading c} w g[f * nv + r]   (grid.y = cardiac knot c)
extern "C" __global__ void vp_cframes_adj(const cx* g, const int* ki, const float* kw, int nkf, long long nv, const u64* crow)
{
    long long r = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    int c = blockIdx.y;
    if (r >= nv) return;
    float ox = 0.f, oy = 0.f;
    for (int j = 0; j < nkf; j++) {
        int f = ki[c * nkf + j];
        if (f < 0) break;
        float w = kw[c * nkf + j];
        cx v = g[(long long)f * nv + r];
        ox += w * v.x; oy += w * v.y;
    }
    cx* C = (cx*)crow[c];
    cx o = C[r]; o.x += ox; o.y += oy; C[r] = o;
}
