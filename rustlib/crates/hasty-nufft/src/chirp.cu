// Chirp-aware (modulated-row) gather / scatter for 3-D f32. Defines: W (ES width), S (stencil half-width), BETA, DIM=3.
// One thread block per (sorted) sample: builds the sample's stencil Gamma = phi (*) (delta + sum_l c_l(w) h_l) in shared memory, then
// gathers / scatters it for every coil with the shared stencil.
typedef float real;
struct cx { real x, y; };
#define NS (2 * S + 1)
#define JW (W + 2 * S)
#define BLK 256
struct Dims { int nf0, nf1, nf2, nb0, nb1, nb2; };

__device__ __forceinline__ cx cmul_(cx a, cx b) { return cx{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
__device__ __forceinline__ real es_eval(real z) { real a = 1.0f - z * z; return a > 0.0f ? expf((real)BETA * (sqrtf(a) - 1.0f)) : 0.0f; }
__device__ __forceinline__ int wrap(int i, int n) { int r = i % n; return r < 0 ? r + n : r; }

// builds Gamma in smem (T2, JW^3, layout [uz][uy][ux]); returns base cell (i0 - S) per axis in base[3]
__device__ void build_stencil(const real* __restrict__ sc, int M, int j, real w1, real w2, real w3, real w4, int L,
                              const real* __restrict__ pts, const real* __restrict__ ptsw, const cx* __restrict__ binv, const cx* __restrict__ H,
                              cx* fvec, cx* cvec, cx* T1, cx* T2, real* wgt, int* base) {
  const int tid = threadIdx.x;
  // (1) values at the DEIM points
  for (int i = tid; i < L; i += BLK) {
    const real x = pts[3 * i], y = pts[3 * i + 1], z = pts[3 * i + 2];
    const real ph = w1 * (z * z) + w2 * (x * x + y * y) + w3 * (x * z) + w4 * (y * z);
    real sn, cs; sincosf(ph, &sn, &cs);
    fvec[i] = cx{(cs - 1.0f) * ptsw[i], -sn * ptsw[i]};
  }
  __syncthreads();
  for (int l = tid; l < L; l += BLK) {                      // (2) coefficients c = Binv f
    real ax = 0, ay = 0;
    for (int i = 0; i < L; ++i) { const cx b = binv[l * L + i]; const cx f = fvec[i]; ax += b.x * f.x - b.y * f.y; ay += b.x * f.y + b.y * f.x; }
    cvec[l] = cx{ax, ay};
  }
  if (tid < 3) {                                            // ES weights per axis
    const real s = sc[tid * (size_t)M + j];
    const int i0 = (int)ceilf(s - (real)W * 0.5f);
    base[tid] = i0 - S;
    for (int t = 0; t < W; ++t) wgt[tid * W + t] = es_eval(((real)(i0 + t) - s) * (2.0f / (real)W));
  }
  __syncthreads();
  for (int s = tid; s < NS * NS * NS; s += BLK) {           // (3) ghat(s) = delta + sum_l c_l h_l(s)
    real ax = (s == (S * NS + S) * NS + S) ? 1.0f : 0.0f, ay = 0.0f;
    for (int l = 0; l < L; ++l) { const cx c = cvec[l]; const cx h = H[(size_t)l * (NS * NS * NS) + s]; ax += c.x * h.x - c.y * h.y; ay += c.x * h.y + c.y * h.x; }
    T1[s] = cx{ax, ay};                                     // layout [sz][sy][sx]
  }
  __syncthreads();
  // (4) sweep z:  A1[(sy,sx)][uz] = sum_sz g[sz][sy][sx] wz[uz - sz]      (T2 holds A1)
  for (int o = tid; o < NS * NS * JW; o += BLK) {
    const int uz = o % JW, q = o / JW;                      // q = sy*NS + sx
    real ax = 0, ay = 0;
    for (int sz = 0; sz < NS; ++sz) { const int t = uz - sz; if (t >= 0 && t < W) { const cx g = T1[sz * NS * NS + q]; const real w = wgt[2 * W + t]; ax += g.x * w; ay += g.y * w; } }
    T2[o] = cx{ax, ay};
  }
  __syncthreads();
  // (5) sweep y:  A2[sx][uy][uz] = sum_sy A1[(sy,sx)][uz] wy[uy - sy]       (T1 holds A2)
  for (int o = tid; o < NS * JW * JW; o += BLK) {
    const int uz = o % JW, uy = (o / JW) % JW, sx = o / (JW * JW);
    real ax = 0, ay = 0;
    for (int sy = 0; sy < NS; ++sy) { const int t = uy - sy; if (t >= 0 && t < W) { const cx a = T2[(sy * NS + sx) * JW + uz]; const real w = wgt[W + t]; ax += a.x * w; ay += a.y * w; } }
    T1[o] = cx{ax, ay};
  }
  __syncthreads();
  // (6) sweep x:  Gamma[uz][uy][ux] = sum_sx A2[sx][uy][uz] wx[ux - sx]     (T2 holds Gamma)
  for (int o = tid; o < JW * JW * JW; o += BLK) {
    const int ux = o % JW, uy = (o / JW) % JW, uz = o / (JW * JW);
    real ax = 0, ay = 0;
    for (int sx = 0; sx < NS; ++sx) { const int t = ux - sx; if (t >= 0 && t < W) { const cx a = T1[(sx * JW + uy) * JW + uz]; const real w = wgt[t]; ax += a.x * w; ay += a.y * w; } }
    T2[o] = cx{ax, ay};
  }
  __syncthreads();
}

#define SMEM_CX(Lmax) ((2 * (Lmax)) + (NS * JW * JW > NS * NS * NS ? NS * JW * JW : NS * NS * NS) + (JW * JW * JW))

extern "C" __global__ void __launch_bounds__(BLK) chirp_interp(const real* __restrict__ sc, const unsigned int* __restrict__ perm,
    const real* __restrict__ w1a, const real* __restrict__ w2a, const real* __restrict__ w3a, const real* __restrict__ w4a,
    const cx* __restrict__ grid, cx* __restrict__ cout, int M, int ntr, size_t gstride, Dims d, int L,
    const real* __restrict__ pts, const real* __restrict__ ptsw, const cx* __restrict__ binv, const cx* __restrict__ H, int lmax) {
  extern __shared__ cx smem[];
  __shared__ real wgt[3 * W];
  __shared__ int base[3];
  __shared__ real red[BLK / 32 * 2];
  cx* fvec = smem; cx* cvec = smem + lmax;
  cx* T1 = smem + 2 * lmax;
  const int t1n = (NS * JW * JW > NS * NS * NS) ? NS * JW * JW : NS * NS * NS;
  cx* T2 = T1 + t1n;
  const int i = blockIdx.x; if (i >= M) return;
  const int j = perm[i];
  build_stencil(sc, M, j, w1a[j], w2a[j], w3a[j], w4a[j], L, pts, ptsw, binv, H, fvec, cvec, T1, T2, wgt, base);
  const int tid = threadIdx.x;
  const int b0 = base[0], b1 = base[1], b2 = base[2];
  for (int tr = 0; tr < ntr; ++tr) {
    const cx* g = grid + (size_t)tr * gstride;
    real ax = 0, ay = 0;
    for (int o = tid; o < JW * JW * JW; o += BLK) {
      const int ux = o % JW, uy = (o / JW) % JW, uz = o / (JW * JW);
      const cx v = g[((size_t)wrap(b2 + uz, d.nf2) * d.nf1 + wrap(b1 + uy, d.nf1)) * d.nf0 + wrap(b0 + ux, d.nf0)];
      const cx gm = T2[o];
      ax += gm.x * v.x - gm.y * v.y; ay += gm.x * v.y + gm.y * v.x;
    }
    for (int off = 16; off > 0; off >>= 1) { ax += __shfl_down_sync(0xffffffff, ax, off); ay += __shfl_down_sync(0xffffffff, ay, off); }
    if ((tid & 31) == 0) { red[(tid >> 5) * 2] = ax; red[(tid >> 5) * 2 + 1] = ay; }
    __syncthreads();
    if (tid == 0) { real sx = 0, sy = 0; for (int w = 0; w < BLK / 32; ++w) { sx += red[2 * w]; sy += red[2 * w + 1]; } cout[(size_t)tr * M + j] = cx{sx, sy}; }
    __syncthreads();
  }
}

// adjoint: grid += conj(Gamma) * c   (grid pre-zeroed)
extern "C" __global__ void __launch_bounds__(BLK) chirp_spread(const real* __restrict__ sc, const unsigned int* __restrict__ perm,
    const real* __restrict__ w1a, const real* __restrict__ w2a, const real* __restrict__ w3a, const real* __restrict__ w4a,
    cx* __restrict__ grid, const cx* __restrict__ cin, int M, int ntr, size_t gstride, Dims d, int L,
    const real* __restrict__ pts, const real* __restrict__ ptsw, const cx* __restrict__ binv, const cx* __restrict__ H, int lmax) {
  extern __shared__ cx smem[];
  __shared__ real wgt[3 * W];
  __shared__ int base[3];
  cx* fvec = smem; cx* cvec = smem + lmax;
  cx* T1 = smem + 2 * lmax;
  const int t1n = (NS * JW * JW > NS * NS * NS) ? NS * JW * JW : NS * NS * NS;
  cx* T2 = T1 + t1n;
  const int i = blockIdx.x; if (i >= M) return;
  const int j = perm[i];
  build_stencil(sc, M, j, w1a[j], w2a[j], w3a[j], w4a[j], L, pts, ptsw, binv, H, fvec, cvec, T1, T2, wgt, base);
  const int tid = threadIdx.x;
  const int b0 = base[0], b1 = base[1], b2 = base[2];
  for (int tr = 0; tr < ntr; ++tr) {
    cx* g = grid + (size_t)tr * gstride;
    const cx cv = cin[(size_t)tr * M + j];
    for (int o = tid; o < JW * JW * JW; o += BLK) {
      const int ux = o % JW, uy = (o / JW) % JW, uz = o / (JW * JW);
      const cx gm = T2[o];
      const real vx = gm.x * cv.x + gm.y * cv.y, vy = gm.x * cv.y - gm.y * cv.x;       // conj(Gamma) * c
      cx* p = g + ((size_t)wrap(b2 + uz, d.nf2) * d.nf1 + wrap(b1 + uy, d.nf1)) * d.nf0 + wrap(b0 + ux, d.nf0);
      atomicAdd(&p->x, vx); atomicAdd(&p->y, vy);
    }
  }
}

// ================================================================== stored-stencil path
// Stencils are generated once per trajectory ("precompute"), stored as packed fp16 (re low, im high) in bin-sorted order, and reused by
// shared-memory-tiled gather / scatter kernels for every coil and every iteration.
#define JW3 (JW * JW * JW)
#define TT0 (BSZ + JW)
#define TT3 (TT0 * TT0 * TT0)
__device__ __forceinline__ unsigned int pack_h2(real a, real b) {
  unsigned short ha, hb;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(ha) : "f"(a));
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(hb) : "f"(b));
  return (unsigned int)ha | ((unsigned int)hb << 16);
}
__device__ __forceinline__ cx unpack_h2(unsigned int u) {
  float a, b;
  asm("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"((unsigned short)(u & 0xffffu)));
  asm("cvt.f32.f16 %0, %1;" : "=f"(b) : "h"((unsigned short)(u >> 16)));
  return cx{a, b};
}

extern "C" __global__ void __launch_bounds__(BLK) chirp_gen(const real* __restrict__ sc, const unsigned int* __restrict__ perm,
    const real* __restrict__ w1a, const real* __restrict__ w2a, const real* __restrict__ w3a, const real* __restrict__ w4a,
    unsigned int* __restrict__ stencils, int M, int L, const real* __restrict__ pts, const real* __restrict__ ptsw,
    const cx* __restrict__ binv, const cx* __restrict__ H, int lmax) {
  extern __shared__ cx smem[];
  __shared__ real wgt[3 * W];
  __shared__ int base[3];
  cx* fvec = smem; cx* cvec = smem + lmax;
  cx* T1 = smem + 2 * lmax;
  const int t1n = (NS * JW * JW > NS * NS * NS) ? NS * JW * JW : NS * NS * NS;
  cx* T2 = T1 + t1n;
  const int i = blockIdx.x; if (i >= M) return;
  const int j = perm[i];
  build_stencil(sc, M, j, w1a[j], w2a[j], w3a[j], w4a[j], L, pts, ptsw, binv, H, fvec, cvec, T1, T2, wgt, base);
  unsigned int* out = stencils + (size_t)i * JW3;
  for (int o = threadIdx.x; o < JW3; o += BLK) out[o] = pack_h2(T2[o].x, T2[o].y);
}

// bin geometry for the tiled kernels: tile origin per axis = b*BSZ - W/2 - S
extern "C" __global__ void __launch_bounds__(256) chirp_gather_sm(const real* __restrict__ sc, const unsigned int* __restrict__ perm,
    const unsigned int* __restrict__ stencils, const cx* __restrict__ grid, cx* __restrict__ cout,
    const unsigned int* __restrict__ sub_bin, const unsigned int* __restrict__ sub_start, const unsigned int* __restrict__ sub_cnt,
    int M, int ntr, size_t gstride, Dims d) {
  extern __shared__ cx tile[];
  const unsigned int sb = sub_bin[blockIdx.x], start = sub_start[blockIdx.x], cnt = sub_cnt[blockIdx.x];
  const int b0 = sb % d.nb0, b1 = (sb / d.nb0) % d.nb1, b2 = sb / (d.nb0 * d.nb1);
  const int o0 = b0 * BSZ - W / 2 - S, o1 = b1 * BSZ - W / 2 - S, o2 = b2 * BSZ - W / 2 - S;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int tr = 0; tr < ntr; ++tr) {
    const cx* g = grid + (size_t)tr * gstride;
    for (int k = tid; k < TT3; k += 256) {
      const int k0 = k % TT0, k1 = (k / TT0) % TT0, k2 = k / (TT0 * TT0);
      tile[k] = g[((size_t)wrap(o2 + k2, d.nf2) * d.nf1 + wrap(o1 + k1, d.nf1)) * d.nf0 + wrap(o0 + k0, d.nf0)];
    }
    __syncthreads();
    for (unsigned int u = warp; u < cnt; u += 8) {
      const unsigned int i = start + u;
      const int j = perm[i];
      const int l0 = (int)ceilf(sc[j] - (real)W * 0.5f) - S - o0;
      const int l1 = (int)ceilf(sc[(size_t)M + j] - (real)W * 0.5f) - S - o1;
      const int l2 = (int)ceilf(sc[2 * (size_t)M + j] - (real)W * 0.5f) - S - o2;
      const unsigned int* st = stencils + (size_t)i * JW3;
      real ax = 0, ay = 0;
      for (int o = lane; o < JW3; o += 32) {
        const int ux = o % JW, uy = (o / JW) % JW, uz = o / (JW * JW);
        const cx gm = unpack_h2(st[o]);
        const cx v = tile[((l2 + uz) * TT0 + (l1 + uy)) * TT0 + l0 + ux];
        ax += gm.x * v.x - gm.y * v.y; ay += gm.x * v.y + gm.y * v.x;
      }
      for (int off = 16; off > 0; off >>= 1) { ax += __shfl_down_sync(0xffffffff, ax, off); ay += __shfl_down_sync(0xffffffff, ay, off); }
      if (lane == 0) cout[(size_t)tr * M + j] = cx{ax, ay};
    }
    __syncthreads();
  }
}

extern "C" __global__ void __launch_bounds__(256) chirp_scatter_sm(const real* __restrict__ sc, const unsigned int* __restrict__ perm,
    const unsigned int* __restrict__ stencils, cx* __restrict__ grid, const cx* __restrict__ cin,
    const unsigned int* __restrict__ sub_bin, const unsigned int* __restrict__ sub_start, const unsigned int* __restrict__ sub_cnt,
    int M, int ntr, size_t gstride, Dims d) {
  extern __shared__ cx tile[];
  const unsigned int sb = sub_bin[blockIdx.x], start = sub_start[blockIdx.x], cnt = sub_cnt[blockIdx.x];
  const int b0 = sb % d.nb0, b1 = (sb / d.nb0) % d.nb1, b2 = sb / (d.nb0 * d.nb1);
  const int o0 = b0 * BSZ - W / 2 - S, o1 = b1 * BSZ - W / 2 - S, o2 = b2 * BSZ - W / 2 - S;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int tr = 0; tr < ntr; ++tr) {
    for (int k = tid; k < TT3; k += 256) { tile[k].x = 0; tile[k].y = 0; }
    __syncthreads();
    for (unsigned int u = warp; u < cnt; u += 8) {
      const unsigned int i = start + u;
      const int j = perm[i];
      const int l0 = (int)ceilf(sc[j] - (real)W * 0.5f) - S - o0;
      const int l1 = (int)ceilf(sc[(size_t)M + j] - (real)W * 0.5f) - S - o1;
      const int l2 = (int)ceilf(sc[2 * (size_t)M + j] - (real)W * 0.5f) - S - o2;
      const unsigned int* st = stencils + (size_t)i * JW3;
      const cx cv = cin[(size_t)tr * M + j];
      for (int o = lane; o < JW3; o += 32) {
        const int ux = o % JW, uy = (o / JW) % JW, uz = o / (JW * JW);
        const cx gm = unpack_h2(st[o]);
        cx* p = &tile[((l2 + uz) * TT0 + (l1 + uy)) * TT0 + l0 + ux];
        atomicAdd(&p->x, gm.x * cv.x + gm.y * cv.y);       // conj(Gamma) * c
        atomicAdd(&p->y, gm.x * cv.y - gm.y * cv.x);
      }
    }
    __syncthreads();
    cx* g = grid + (size_t)tr * gstride;
    for (int k = tid; k < TT3; k += 256) {
      const cx v = tile[k];
      if (v.x == 0.0f && v.y == 0.0f) continue;
      const int k0 = k % TT0, k1 = (k / TT0) % TT0, k2 = k / (TT0 * TT0);
      cx* p = g + ((size_t)wrap(o2 + k2, d.nf2) * d.nf1 + wrap(o1 + k1, d.nf1)) * d.nf0 + wrap(o0 + k0, d.nf0);
      atomicAdd(&p->x, v.x); atomicAdd(&p->y, v.y);
    }
    __syncthreads();
  }
}
