// hasty-nufft CUDA kernels. Defines supplied by the host: REAL (float|double), DIM (1..3), W (kernel width), BETA, BS (bin size per dim),
// REAL_IS_DOUBLE (0|1).
// Layout: fine grid / mode arrays row-major, dim0 (x) fastest: idx = (k2 * n1 + k1) * n0 + k0. Points are SoA: sc[d * M + j].
typedef REAL real;
struct cx { real x, y; };
#if REAL_IS_DOUBLE
#define RCEIL ceil
#define RFLOOR floor
#define REXP exp
#define RSQRT sqrt
#define RFABS fabs
#else
#define RCEIL ceilf
#define RFLOOR floorf
#define REXP expf
#define RSQRT sqrtf
#define RFABS fabsf
#endif
#define N1D (DIM >= 2 ? W : 1)
#define N2D (DIM >= 3 ? W : 1)

__device__ __forceinline__ real es_eval(real z) {
  real a = (real)1 - z * z;
  return a > (real)0 ? REXP((real)BETA * (RSQRT(a) - (real)1)) : (real)0;
}
// weights of the W cells starting at i0 = ceil(sc - W/2); returns i0
__device__ __forceinline__ int kernel_weights(real sc, real* wv) {
  const int i0 = (int)RCEIL(sc - (real)W * (real)0.5);
  #pragma unroll
  for (int t = 0; t < W; ++t) {
    const real z = ((real)(i0 + t) - sc) * ((real)2 / (real)W);
    wv[t] = es_eval(z);
  }
  return i0;
}
__device__ __forceinline__ int wrap(int i, int n) { int r = i % n; return r < 0 ? r + n : r; }

struct Dims { int nf0, nf1, nf2, nb0, nb1, nb2; };

// ---------------------------------------------------------------- binning
extern "C" __global__ void prep_points(const real* __restrict__ x0, const real* __restrict__ x1, const real* __restrict__ x2, int M,
                                       real* __restrict__ sc, unsigned int* __restrict__ binid, unsigned int* __restrict__ hist, Dims d) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= M) return;
  const real inv2pi = (real)0.15915494309189533577;
  unsigned int b = 0, stride = 1;
  const int nf[3] = {d.nf0, d.nf1, d.nf2};
  const int nb[3] = {d.nb0, d.nb1, d.nb2};
  #pragma unroll
  for (int a = 0; a < DIM; ++a) {
    const real x = a == 0 ? x0[j] : (a == 1 ? x1[j] : x2[j]);
    real v = x * inv2pi;
    v -= RFLOOR(v);                       // [0,1)
    real s = v * (real)nf[a];
    if (s >= (real)nf[a]) s -= (real)nf[a];
    sc[a * (size_t)M + j] = s;
    int cell = (int)s; if (cell >= nf[a]) cell = nf[a] - 1;
    b += (unsigned int)(cell / BS) * stride;
    stride *= nb[a];
  }
  binid[j] = b;
  atomicAdd(&hist[b], 1u);
}

extern "C" __global__ void scan_local(const unsigned int* __restrict__ in, unsigned int* __restrict__ out, unsigned int* __restrict__ sums, int n) {
  __shared__ unsigned int s[1024];
  const int tid = threadIdx.x;
  const int base = blockIdx.x * 1024;
  unsigned int v0 = (base + 2 * tid < n) ? in[base + 2 * tid] : 0u;
  unsigned int v1 = (base + 2 * tid + 1 < n) ? in[base + 2 * tid + 1] : 0u;
  s[2 * tid] = v0; s[2 * tid + 1] = v1;
  for (int off = 1; off < 1024; off <<= 1) {   // up-sweep
    __syncthreads();
    int i = (tid + 1) * off * 2 - 1;
    if (i < 1024) s[i] += s[i - off];
  }
  __syncthreads();
  if (tid == 0) { sums[blockIdx.x] = s[1023]; s[1023] = 0; }
  for (int off = 512; off >= 1; off >>= 1) {   // down-sweep
    __syncthreads();
    int i = (tid + 1) * off * 2 - 1;
    if (i < 1024) { unsigned int t = s[i - off]; s[i - off] = s[i]; s[i] += t; }
  }
  __syncthreads();
  if (base + 2 * tid < n) out[base + 2 * tid] = s[2 * tid];
  if (base + 2 * tid + 1 < n) out[base + 2 * tid + 1] = s[2 * tid + 1];
}
extern "C" __global__ void scan_sums(unsigned int* __restrict__ sums, int n) {   // single block, exclusive scan of n values
  __shared__ unsigned int s[1024];
  unsigned int carry = 0;
  for (int base = 0; base < n; base += 1024) {
    const int idx = base + threadIdx.x;
    unsigned int v = idx < n ? sums[idx] : 0u;
    s[threadIdx.x] = v;
    for (int off = 1; off < 1024; off <<= 1) {
      __syncthreads();
      unsigned int t = threadIdx.x >= off ? s[threadIdx.x - off] : 0u;
      __syncthreads();
      s[threadIdx.x] += t;
    }
    __syncthreads();
    unsigned int incl = s[threadIdx.x];
    if (idx < n) sums[idx] = carry + incl - v;
    unsigned int total = s[1023];
    __syncthreads();
    carry += total;
  }
}
extern "C" __global__ void scan_add(unsigned int* __restrict__ out, const unsigned int* __restrict__ sums, int n) {
  const int i = blockIdx.x * 1024 + 2 * threadIdx.x;
  const unsigned int a = sums[blockIdx.x];
  if (i < n) out[i] += a;
  if (i + 1 < n) out[i + 1] += a;
}
extern "C" __global__ void scatter_perm(const unsigned int* __restrict__ binid, const unsigned int* __restrict__ offsets, unsigned int* __restrict__ counter,
                                        unsigned int* __restrict__ perm, int M) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= M) return;
  const unsigned int b = binid[j];
  perm[offsets[b] + atomicAdd(&counter[b], 1u)] = (unsigned int)j;
}
extern "C" __global__ void count_subprobs(const unsigned int* __restrict__ hist, unsigned int* __restrict__ nsub, int nbins, int maxsub) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < nbins) nsub[b] = (hist[b] + maxsub - 1) / maxsub;
}
extern "C" __global__ void build_subprobs(const unsigned int* __restrict__ hist, const unsigned int* __restrict__ offsets, const unsigned int* __restrict__ sub_off,
                                          unsigned int* __restrict__ sub_bin, unsigned int* __restrict__ sub_start, unsigned int* __restrict__ sub_cnt, int nbins, int maxsub) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= nbins) return;
  const unsigned int cnt = hist[b];
  const unsigned int ns = (cnt + maxsub - 1) / maxsub;
  for (unsigned int s = 0; s < ns; ++s) {
    const unsigned int k = sub_off[b] + s;
    sub_bin[k] = b; sub_start[k] = offsets[b] + s * maxsub;
    sub_cnt[k] = min((unsigned int)maxsub, cnt - s * maxsub);
  }
}

// ---------------------------------------------------------------- spreading / interpolation
// per-sample base indices and weights
struct Stencil { int i0[3]; real w0[W]; real w1[N1D]; real w2[N2D]; };
__device__ __forceinline__ void make_stencil(const real* __restrict__ sc, int M, int j, Stencil& s) {
  s.i0[0] = kernel_weights(sc[j], s.w0);
  s.i0[1] = 0; s.i0[2] = 0;
  if (DIM >= 2) s.i0[1] = kernel_weights(sc[(size_t)M + j], s.w1); else s.w1[0] = (real)1;
  if (DIM >= 3) s.i0[2] = kernel_weights(sc[2 * (size_t)M + j], s.w2); else s.w2[0] = (real)1;
}

__device__ __forceinline__ void atomic_add_cx(cx* p, real vx, real vy) { atomicAdd(&p->x, vx); atomicAdd(&p->y, vy); }

// type-1 spreading, global atomics, thread per (sorted) point
extern "C" __global__ void spread_gm(const real* __restrict__ sc, const unsigned int* __restrict__ perm, const cx* __restrict__ c, cx* __restrict__ grid,
                                     int M, int ntr, size_t grid_stride, Dims d) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  const int j = perm[i];
  Stencil s; make_stencil(sc, M, j, s);
  int ix[W]; int iy[N1D]; int iz[N2D];
  #pragma unroll
  for (int t = 0; t < W; ++t) ix[t] = wrap(s.i0[0] + t, d.nf0);
  #pragma unroll
  for (int t = 0; t < N1D; ++t) iy[t] = DIM >= 2 ? wrap(s.i0[1] + t, d.nf1) : 0;
  #pragma unroll
  for (int t = 0; t < N2D; ++t) iz[t] = DIM >= 3 ? wrap(s.i0[2] + t, d.nf2) : 0;
  for (int tr = 0; tr < ntr; ++tr) {
    const cx cv = c[(size_t)tr * M + j];
    cx* g = grid + (size_t)tr * grid_stride;
    for (int tz = 0; tz < N2D; ++tz) {
      const real wz = s.w2[tz] ;
      for (int ty = 0; ty < N1D; ++ty) {
        const real wzy = wz * s.w1[ty];
        cx* row = g + ((size_t)iz[tz] * d.nf1 + iy[ty]) * d.nf0;
        #pragma unroll
        for (int tx = 0; tx < W; ++tx) {
          const real wgt = wzy * s.w0[tx];
          atomic_add_cx(row + ix[tx], cv.x * wgt, cv.y * wgt);
        }
      }
    }
  }
}

// type-1 spreading, shared-memory tile per subproblem (bin x chunk of points)
#define T0 (BS + W)
#define T1 (DIM >= 2 ? BS + W : 1)
#define T2 (DIM >= 3 ? BS + W : 1)
extern "C" __global__ void spread_sm(const real* __restrict__ sc, const unsigned int* __restrict__ perm, const cx* __restrict__ c, cx* __restrict__ grid,
                                     const unsigned int* __restrict__ sub_bin, const unsigned int* __restrict__ sub_start, const unsigned int* __restrict__ sub_cnt,
                                     int M, int ntr, size_t grid_stride, Dims d) {
  extern __shared__ cx tile[];
  const unsigned int sb = sub_bin[blockIdx.x], start = sub_start[blockIdx.x], cnt = sub_cnt[blockIdx.x];
  int b0 = sb % d.nb0, b1 = (sb / d.nb0) % d.nb1, b2 = sb / (d.nb0 * d.nb1);
  if (DIM < 3) b2 = 0;
  if (DIM < 2) b1 = 0;
  const int o0 = b0 * BS - W / 2, o1 = DIM >= 2 ? b1 * BS - W / 2 : 0, o2 = DIM >= 3 ? b2 * BS - W / 2 : 0;
  const int ncell = T0 * T1 * T2;
  for (int tr = 0; tr < ntr; ++tr) {
    for (int k = threadIdx.x; k < ncell; k += blockDim.x) { tile[k].x = (real)0; tile[k].y = (real)0; }
    __syncthreads();
    for (unsigned int u = threadIdx.x; u < cnt; u += blockDim.x) {
      const int j = perm[start + u];
      Stencil s; make_stencil(sc, M, j, s);
      const cx cv = c[(size_t)tr * M + j];
      const int l0 = s.i0[0] - o0, l1 = DIM >= 2 ? s.i0[1] - o1 : 0, l2 = DIM >= 3 ? s.i0[2] - o2 : 0;
      for (int tz = 0; tz < N2D; ++tz) {
        const real wz = s.w2[tz];
        for (int ty = 0; ty < N1D; ++ty) {
          const real wzy = wz * s.w1[ty];
          cx* row = tile + ((l2 + tz) * T1 + (l1 + ty)) * T0 + l0;
          #pragma unroll
          for (int tx = 0; tx < W; ++tx) {
            const real wgt = wzy * s.w0[tx];
            atomicAdd(&row[tx].x, cv.x * wgt); atomicAdd(&row[tx].y, cv.y * wgt);
          }
        }
      }
    }
    __syncthreads();
    cx* g = grid + (size_t)tr * grid_stride;
    for (int k = threadIdx.x; k < ncell; k += blockDim.x) {
      const cx v = tile[k];
      if (v.x == (real)0 && v.y == (real)0) continue;
      const int k0 = k % T0, k1 = (k / T0) % T1, k2 = k / (T0 * T1);
      const int g0 = wrap(o0 + k0, d.nf0), g1 = DIM >= 2 ? wrap(o1 + k1, d.nf1) : 0, g2 = DIM >= 3 ? wrap(o2 + k2, d.nf2) : 0;
      atomic_add_cx(g + ((size_t)g2 * d.nf1 + g1) * d.nf0 + g0, v.x, v.y);
    }
    __syncthreads();
  }
}

// type-2 interpolation, thread per (sorted) point
extern "C" __global__ void interp_gm(const real* __restrict__ sc, const unsigned int* __restrict__ perm, const cx* __restrict__ grid, cx* __restrict__ c,
                                     int M, int ntr, size_t grid_stride, Dims d) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M) return;
  const int j = perm[i];
  Stencil s; make_stencil(sc, M, j, s);
  int ix[W]; int iy[N1D]; int iz[N2D];
  #pragma unroll
  for (int t = 0; t < W; ++t) ix[t] = wrap(s.i0[0] + t, d.nf0);
  #pragma unroll
  for (int t = 0; t < N1D; ++t) iy[t] = DIM >= 2 ? wrap(s.i0[1] + t, d.nf1) : 0;
  #pragma unroll
  for (int t = 0; t < N2D; ++t) iz[t] = DIM >= 3 ? wrap(s.i0[2] + t, d.nf2) : 0;
  for (int tr = 0; tr < ntr; ++tr) {
    const cx* g = grid + (size_t)tr * grid_stride;
    real ax = 0, ay = 0;
    for (int tz = 0; tz < N2D; ++tz) {
      for (int ty = 0; ty < N1D; ++ty) {
        const cx* row = g + ((size_t)iz[tz] * d.nf1 + iy[ty]) * d.nf0;
        real rx = 0, ry = 0;
        #pragma unroll
        for (int tx = 0; tx < W; ++tx) { const cx v = row[ix[tx]]; rx += v.x * s.w0[tx]; ry += v.y * s.w0[tx]; }
        const real wzy = s.w2[tz] * s.w1[ty];
        ax += rx * wzy; ay += ry * wzy;
      }
    }
    c[(size_t)tr * M + j] = cx{ax, ay};
  }
}

// ---------------------------------------------------------------- mode <-> fine-grid transfer with deconvolution
struct MDims { int N0, N1, N2, fftord; };
// type 2: fine grid (zeroed beforehand) <- modes * inv_phi
extern "C" __global__ void pad_deconv(const cx* __restrict__ f, cx* __restrict__ grid, const real* __restrict__ i0v, const real* __restrict__ i1v, const real* __restrict__ i2v,
                                      int ntr, size_t grid_stride, MDims m, Dims d) {
  const size_t total = (size_t)m.N0 * m.N1 * m.N2;
  const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= total) return;
  const int a0 = t % m.N0, a1 = (t / m.N0) % m.N1, a2 = t / ((size_t)m.N0 * m.N1);   // centred mode index
  const real fac = i0v[a0] * (DIM >= 2 ? i1v[a1] : (real)1) * (DIM >= 3 ? i2v[a2] : (real)1);
  const int g0 = wrap(a0 - m.N0 / 2, d.nf0), g1 = DIM >= 2 ? wrap(a1 - m.N1 / 2, d.nf1) : 0, g2 = DIM >= 3 ? wrap(a2 - m.N2 / 2, d.nf2) : 0;
  const size_t gi = ((size_t)g2 * d.nf1 + g1) * d.nf0 + g0;
  // array position of this mode: centred (default) or FFT order (k mod N)
  size_t ti = t;
  if (m.fftord) { const int b0 = wrap(a0 - m.N0 / 2, m.N0), b1 = DIM >= 2 ? wrap(a1 - m.N1 / 2, m.N1) : 0, b2 = DIM >= 3 ? wrap(a2 - m.N2 / 2, m.N2) : 0; ti = ((size_t)b2 * m.N1 + b1) * m.N0 + b0; }
  for (int tr = 0; tr < ntr; ++tr) {
    const cx v = f[(size_t)tr * total + ti];
    grid[(size_t)tr * grid_stride + gi] = cx{v.x * fac, v.y * fac};
  }
}
// type 1: modes <- fine grid * inv_phi
extern "C" __global__ void crop_deconv(const cx* __restrict__ grid, cx* __restrict__ f, const real* __restrict__ i0v, const real* __restrict__ i1v, const real* __restrict__ i2v,
                                       int ntr, size_t grid_stride, MDims m, Dims d) {
  const size_t total = (size_t)m.N0 * m.N1 * m.N2;
  const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= total) return;
  const int a0 = t % m.N0, a1 = (t / m.N0) % m.N1, a2 = t / ((size_t)m.N0 * m.N1);
  const real fac = i0v[a0] * (DIM >= 2 ? i1v[a1] : (real)1) * (DIM >= 3 ? i2v[a2] : (real)1);
  const int g0 = wrap(a0 - m.N0 / 2, d.nf0), g1 = DIM >= 2 ? wrap(a1 - m.N1 / 2, d.nf1) : 0, g2 = DIM >= 3 ? wrap(a2 - m.N2 / 2, d.nf2) : 0;
  const size_t gi = ((size_t)g2 * d.nf1 + g1) * d.nf0 + g0;
  size_t ti = t;
  if (m.fftord) { const int b0 = wrap(a0 - m.N0 / 2, m.N0), b1 = DIM >= 2 ? wrap(a1 - m.N1 / 2, m.N1) : 0, b2 = DIM >= 3 ? wrap(a2 - m.N2 / 2, m.N2) : 0; ti = ((size_t)b2 * m.N1 + b1) * m.N0 + b0; }
  for (int tr = 0; tr < ntr; ++tr) {
    const cx v = grid[(size_t)tr * grid_stride + gi];
    f[(size_t)tr * total + ti] = cx{v.x * fac, v.y * fac};
  }
}
