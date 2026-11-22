// Knot-major temporal tile kernels, appended to segnufft.cu (compiled with NF = 1: one grid slot) for engines built
// with `SegNufft::new_temporal(.., knot_major = true)`. Defines: KS = knot support (B-spline degree + 1: 2 linear,
// 4 cubic); the other defines as segnufft.cu.
//
// The interval-major kernels (seg_forward_t / seg_adjoint_t, NF = KS) run one launch per frame interval i and touch
// the KS slots of the frames i + KOFF .. i + KOFF + KS − 1: every slot is read (forward) or read-modified-written
// (adjoint) by KS launches, and KS slots of cb · NSEG channels must be resident. Here one launch serves one knot b:
//   forward: y_c(j) += β_b(t_j) Σ_l a_l(τ_j) Σ_p w_j(p) G_{b,c,l}(p)        for the samples j with β_b(t_j) ≠ 0
//   adjoint: G_{b,c,l}(p) (=)  Σ_j conj(w_j(p)) β_b(t_j) ā_l(τ_j) y_c(j)    (one store per grid point)
// so each knot's slot is transformed, read / written and transformed back once, and only that slot is resident.
// The per-sample work (staging, MACs) runs KS times per sample, once per knot of its support.
//
// Tables (`SegNufft::points` of a knot-major engine): one tile list for the whole set, each tile's entries sorted by
// frame interval, entry = (interval << 25) | sample; work items split long tile lists as usual. A block filters its
// item's entries to the knot's window (intervals b − KOFF − KS + 1 .. b − KOFF, periodic), which is at most two
// contiguous runs because the entries are interval-sorted, and compacts them into the shared list.
#ifndef KS
#define KS 2
#endif
#define KOFF (-((KS - 2) / 2))
#define KEY_SHIFT 25
#define KEY_MASK 0x1FFFFFFu

__device__ __forceinline__ float bspline_k(int o, float u) {
#if KS == 1
  return 1.f;   // degree 0: bins
#elif KS == 2
  return o == 0 ? 1.f - u : u;
#elif KS == 4
  const float v = 1.f - u, u2 = u * u, u3 = u2 * u;
  return o == 0 ? v * v * v * (1.f / 6.f) : o == 1 ? (3.f * u3 - 6.f * u2 + 4.f) * (1.f / 6.f) : o == 2 ? (-3.f * u3 + 3.f * u2 + 3.f * u + 1.f) * (1.f / 6.f) : u3 * (1.f / 6.f);
#else
#error "KS must be 1, 2 or 4"
#endif
}

// compact the entries of tl[first, first + count) whose interval lies in knot b's window into sl; returns their number
// (block-uniform; contains barriers, call from all threads)
__device__ __forceinline__ int knot_window(const unsigned int* __restrict__ tl, int first, int count, int b, int nint, int* sl, int nthr) {
  const int khi = ((b - KOFF) % nint + nint) % nint, klo = ((b - KOFF - KS + 1) % nint + nint) % nint;
  int clo = 0, chi = 0;   // entries with key < klo, key ≤ khi
  for (int base = 0; base < count; base += nthr) {
    const int q = base + threadIdx.x;
    const int key = q < count ? (int)(tl[first + q] >> KEY_SHIFT) : 0x7fffffff;
    clo += __syncthreads_count(key < klo);
    chi += __syncthreads_count(key <= khi);
  }
  const bool wrapw = klo > khi || KS >= nint;   // window wraps the period (or covers it)
  const int nwin = wrapw ? (KS >= nint ? count : chi + (count - clo)) : chi - clo;
  for (int q = threadIdx.x; q < nwin; q += nthr) {
    const int pos = KS >= nint ? q : (wrapw ? (q < chi ? q : clo + (q - chi)) : clo + q);
    sl[q] = (int)tl[first + pos];
  }
  __syncthreads();
  return nwin;
}

// β_b(t_j) of the staged samples, folded into their coefficients aa[s][0][l] (same q → (s, l) mapping as stage_samples,
// so each thread rescales what it wrote)
__device__ __forceinline__ void stage_knot(Stage& S, const int* sl, int base, int nload, int b, int nint) {
  for (int q = threadIdx.x; q < nload * NSEG; q += blockDim.x) {
    const int sm = q / NSEG, l = q % NSEG;
    const int key = (int)((unsigned int)sl[base + sm] >> KEY_SHIFT);
    const int o = ((b - KOFF - key) % nint + nint) % nint;
    const float beta = bspline_k(o, S.rec[sm][2].w);
    S.aa[sm][0][l] = cx{beta * S.aa[sm][0][l].x, beta * S.aa[sm][0][l].y};
  }
}

// ---------------------------------------------------------------- 2-D knots: respiratory × cardiac (RDEG ≥ 0)
// Engine `SegNufft::new_temporal2d`: knot b = k · nint + bc (respiratory knot k, cardiac knot bc). Respiratory basis of
// degree RDEG over NRI intervals, not periodic: RDEG 0 = bins (interval i ↔ knot i, weight 1), RDEG 1 = linear hats
// (interval i between knots i and i + 1, weights 1 − v, v). Cardiac as above (periodic B-spline of support KS).
// Entry key = ri · nint + ic (< 256), tile lists sorted by key; record float 11 holds (u, v) packed as 16-bit fractions.
// A knot's window is ≤ 2 respiratory intervals × ≤ 2 cardiac key runs = ≤ 4 contiguous runs of the sorted list: their
// bounds by binary search (one thread per bound), no scan of the whole list.
#ifndef RDEG
#define RDEG -1
#endif
#if RDEG >= 0
__device__ __forceinline__ void unpack_uv(float f, float& u, float& v) {
  const unsigned int b = __float_as_uint(f);
  u = (float)(b >> 16) * (1.f / 65535.f);
  v = (float)(b & 0xFFFFu) * (1.f / 65535.f);
}
// first position in tl[first, first + count) whose key ≥ key (keys sorted)
__device__ __forceinline__ int lower_key(const unsigned int* __restrict__ tl, int first, int count, int key) {
  int lo = 0, hi = count;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if ((int)(tl[first + mid] >> KEY_SHIFT) < key) lo = mid + 1; else hi = mid;
  }
  return lo;
}
__device__ __forceinline__ int knot_window2d(const unsigned int* __restrict__ tl, int first, int count, int b, int nint, int nri, int* sl, int nthr) {
  __shared__ int rb[8][2];   // runs [lo, hi)
  __shared__ int nr;
  const int k = b / nint, bc = b % nint;
  const int khi = ((bc - KOFF) % nint + nint) % nint, klo = ((bc - KOFF - KS + 1) % nint + nint) % nint;
  // respiratory intervals of knot k
  const int r0 = RDEG == 0 ? k : max(k - 1, 0), r1 = RDEG == 0 ? k : min(k, nri - 1);
  if (threadIdx.x == 0) nr = 0;
  __syncthreads();
  if (threadIdx.x < 8) {
    const int q = threadIdx.x;               // run q / 2 of the list, bound q % 2
    const int ri = r0 + (q >> 2);            // ≤ 2 respiratory intervals
    const int part = (q >> 1) & 1;           // cardiac part 0: [klo, khi] or [0, khi]; 1: [klo, nint − 1] when wrapping
    const bool wrapw = klo > khi || KS >= nint;
    bool live = ri <= r1 && (part == 0 || (wrapw && KS < nint));
    if (live) {
      int a, z;   // key range [a, z] within interval ri
      if (KS >= nint) { a = 0; z = nint - 1; }
      else if (!wrapw) { a = klo; z = khi; }
      else if (part == 0) { a = 0; z = khi; }
      else { a = klo; z = nint - 1; }
      const int key = ri * nint + ((q & 1) ? z + 1 : a);
      rb[q >> 1][q & 1] = lower_key(tl, first, count, key);
    } else {
      rb[q >> 1][q & 1] = 0;
    }
  }
  __syncthreads();
  // compact the runs (≤ 4) into sl
  int base = 0;
  for (int r = 0; r < 4; ++r) {
    const int lo = rb[r][0], n = max(0, rb[r][1] - rb[r][0]);
    for (int q = threadIdx.x; q < n; q += nthr) sl[base + q] = (int)tl[first + lo + q];
    base += n;
  }
  __syncthreads();
  return base;
}
__device__ __forceinline__ void stage_knot2d(Stage& S, const int* sl, int base, int nload, int b, int nint) {
  const int k = b / nint, bc = b % nint;
  for (int q = threadIdx.x; q < nload * NSEG; q += blockDim.x) {
    const int sm = q / NSEG, l = q % NSEG;
    const int key = (int)((unsigned int)sl[base + sm] >> KEY_SHIFT), ri = key / nint, ic = key % nint;
    float u, v; unpack_uv(S.rec[sm][2].w, u, v);
    const int o = ((bc - KOFF - ic) % nint + nint) % nint;
    const float g = RDEG == 0 ? 1.f : (ri == k ? 1.f - v : v);
    const float beta = g * bspline_k(o, u);
    S.aa[sm][0][l] = cx{beta * S.aa[sm][0][l].x, beta * S.aa[sm][0][l].y};
  }
}
__device__ __forceinline__ float knot_beta(const Stage& S, const int* sl, int base, int sm, int b, int nint) {
  const int k = b / nint, bc = b % nint;
  const int key = (int)((unsigned int)sl[base + sm] >> KEY_SHIFT), ri = key / nint, ic = key % nint;
  float u, v; unpack_uv(S.rec[sm][2].w, u, v);
  const int o = ((bc - KOFF - ic) % nint + nint) % nint;
  return (RDEG == 0 ? 1.f : (ri == k ? 1.f - v : v)) * bspline_k(o, u);
}
#define KNOT_WINDOW(tl, first, count, b, nint, sl, nthr) knot_window2d(tl, first, count, b, nint, nri, sl, nthr)
#define STAGE_KNOT(S, sl, base, nload, b, nint) stage_knot2d(S, sl, base, nload, b, nint)
#else
__device__ __forceinline__ float knot_beta(const Stage& S, const int* sl, int base, int sm, int b, int nint) {
  const int key = (int)((unsigned int)sl[base + sm] >> KEY_SHIFT);
  return bspline_k(((b - KOFF - key) % nint + nint) % nint, S.rec[sm][2].w);
}
#define KNOT_WINDOW(tl, first, count, b, nint, sl, nthr) knot_window(tl, first, count, b, nint, sl, nthr)
#define STAGE_KNOT(S, sl, base, nload, b, nint) stage_knot(S, sl, base, nload, b, nint)
#endif

// ---------------------------------------------------------------- correction order 3 (CORR 3)
// As the multi-image kernels (segnufft_multi.cu, same arithmetic): the CORR 2 terms plus a per-sample degree-3/4
// polynomial phase to first order, c_j(ρ) = Σ_α hc_jα ρ^α, hc_jα = Σ_q u_jq H'_qα (u_j per ORIGINAL sample index as
// float4, H' in kernel-argument units, host `points_hi` / `points2d_hi`), image factor 1 − iψ − ½ψ² − i c_j(ρ):
// degree 3 → + c ∂^α φ (real part), degree 4 → −i c ∂^α φ (imaginary part); monomials x^a y^b z^c, degree 3 (10) then
// 4 (15), a outer, b inner. The coefficients of a chunk live in shared memory after the list (KHC bytes; the regions
// after it shift by KHC, 0 below CORR 3, so the CORR 0/1/2 layouts and kernels are unchanged). uw / htab are extra
// trailing kernel arguments at CORR 3 only (null: c = 0, the order-2 operator).
#define KBASE ((sizeof(Stage) + 4 * LISTMAX + 15) & ~(size_t)15)
#if CORR >= 3
#define NH 25
#define KHC (((size_t)CHUNK * NH * 4 + 15) & ~(size_t)15)
// degree-3/4 coefficients of the chunk's samples (sl entries: key << KEY_SHIFT | original sample index)
__device__ __forceinline__ void stage_hc_k(float (*hc)[NH], const int* sl, int base, int nload, const float4* __restrict__ uw, const float* __restrict__ htab) {
  for (int q = threadIdx.x; q < nload * NH; q += blockDim.x) {
    const int s = q / NH, a = q % NH;
    float v = 0.f;
    if (uw && htab) {
      const float4 u = uw[(unsigned int)sl[base + s] & KEY_MASK];
      v = fmaf(u.x, htab[a], fmaf(u.y, htab[NH + a], fmaf(u.z, htab[2 * NH + a], u.w * htab[3 * NH + a])));
    }
    hc[s][a] = v;
  }
}
// stage_weights (CORR 2 terms, same arithmetic) plus the first-order degree-3 / degree-4 terms (= stage_weights_hi of
// segnufft_multi.cu with T.hc → hc)
__device__ __forceinline__ void stage_weights_hk(Stage& S, const float (*hc)[NH], int nload) {
  for (int q = threadIdx.x; q < nload * TT * TT; q += blockDim.x) {
    const int s = q / (TT * TT), row = q % (TT * TT), py = row % TT, pz = row / TT;
    float R[NK], Q[NK];
    #pragma unroll
    for (int a = 0; a < NK; ++a) { R[a] = 0.f; Q[a] = 0.f; }
    float wy[NK], wz[NK];
    #pragma unroll
    for (int k = 0; k < NK; ++k) { wy[k] = S.ax[s][1][k][py]; wz[k] = S.ax[s][2][k][pz]; }
    R[0] = wy[0] * wz[0];
    const float4 r1 = S.rec[s][1], r2 = S.rec[s][2];   // [xx, yy, zz] = r1.yzw, [xy, xz, yz] = r2.xyz
    Q[0] = r1.z * wy[2] * wz[0] + r1.w * wy[0] * wz[2] + r2.z * wy[1] * wz[1];
    Q[1] = r2.x * wy[1] * wz[0] + r2.y * wy[0] * wz[1];
    Q[2] = r1.y * wy[0] * wz[0];
    {
      int idx = 0;
      #pragma unroll
      for (int a = 0; a <= 4; ++a) {
        #pragma unroll
        for (int b = 0; b <= 4 - a; ++b) { R[a] += S.ee[s][idx] * wy[b] * wz[4 - a - b]; ++idx; }
      }
    }
    {
      int idx = 0;
      #pragma unroll
      for (int a = 0; a <= 3; ++a) {   // degree 3: + c ∂^α φ (real)
        #pragma unroll
        for (int b = 0; b <= 3 - a; ++b) { R[a] = fmaf(hc[s][idx], wy[b] * wz[3 - a - b], R[a]); ++idx; }
      }
      #pragma unroll
      for (int a = 0; a <= 4; ++a) {   // degree 4: −i c ∂^α φ (imaginary)
        #pragma unroll
        for (int b = 0; b <= 4 - a; ++b) { Q[a] = fmaf(-hc[s][idx], wy[b] * wz[4 - a - b], Q[a]); ++idx; }
      }
    }
    #pragma unroll
    for (int px = 0; px < TT; ++px) {
      float wr = 0.f, wi = 0.f;
      #pragma unroll
      for (int a = 0; a < NK; ++a) { const float xa = S.ax[s][0][a][px]; wr = fmaf(xa, R[a], wr); wi = fmaf(xa, Q[a], wi); }
      S.wt[s][row * TT + px] = cx{wr, wi};
    }
  }
}
#define KHI_ARGS , const float4* __restrict__ uw, const float* __restrict__ htab
#define KHC_DECL float (*HC)[NH] = reinterpret_cast<float (*)[NH]>(smem + KBASE);
#define STAGE_HC(sl, base, nload) stage_hc_k(HC, sl, base, nload, uw, htab)
#define STAGE_WEIGHTS_K(S, n) stage_weights_hk(S, HC, n)
#else
#define KHC ((size_t)0)
#define KHI_ARGS
#define KHC_DECL
#define STAGE_HC(sl, base, nload)
#define STAGE_WEIGHTS_K(S, n) stage_weights(S, n)
#endif


#ifndef KV
#define KV 0
#endif
#if KV == 0
// ---------------------------------------------------------------- variant 0: the original kernels (dense tile loops)
// forward: samples of knot b's support (+=, y initialised by the host) <- knot b's grid slot
extern "C" __global__ void __launch_bounds__(NT_F) seg_forward_k(const float4* __restrict__ rec, const unsigned int* __restrict__ tl, const Item* __restrict__ items,
                                             const cx* __restrict__ atab, int ntab, float t0, float dt,
                                             Grids g, int b, int nint, int nri, cx* __restrict__ y, int M, int c0, int ncb, Dims d KHI_ARGS) {
  extern __shared__ __align__(16) unsigned char smem[];   // dynamic: Stage | list | tile values (host: TileCfg::smem)
  Stage& S = *reinterpret_cast<Stage*>(smem);
  int* sl = reinterpret_cast<int*>(smem + sizeof(Stage));
  cx* gt = reinterpret_cast<cx*>(smem + KBASE + KHC);
  KHC_DECL
  const Item it = items[blockIdx.x];
  if (it.count == 0) return;
  const int nwin = KNOT_WINDOW(tl, it.first, it.count, b, nint, sl, NT_F);
  (void)nri;
  if (nwin == 0) return;
  const int cg = blockIdx.y * CPBF;
  int o0, o1, o2; tile_origin(it.tile, d, o0, o1, o2);
  for (int q = threadIdx.x; q < TT3 * GN; q += NT_F) {
    const int p = q / GN, c = (q / NSEG) % CPBF, l = q % NSEG;
    const int gx = o0 + p % TT, gy = o1 + (p / TT) % TT, gz = o2 + p / (TT * TT);
    cx v = cx{0.f, 0.f};
    if (cg + c < ncb && gx < d.nf0 && gy < d.nf1 && gz < d.nf2) v = GLOAD(g.p[0], (((size_t)gz * d.nf1 + gy) * d.nf0 + gx) * NCH + (cg + c) * NSEG + l);
    gt[p * GTS + c * NSEG + l] = v;
  }
  constexpr int NR = (3 * CHUNK + NT_F - 1) / NT_F;
  float4 pr[NR];
  auto prefetch = [&](int base) {
    const int nl = min(CHUNK, nwin - base);
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
      const int q = threadIdx.x + r * NT_F;
      if (q < 3 * nl) pr[r] = rec[(size_t)((unsigned int)sl[base + q / 3] & KEY_MASK) * 3 + q % 3];
    }
  };
  __syncthreads();
  prefetch(0);
  const int pp = threadIdx.x % PPF, fo = 0, ss = (threadIdx.x / PPF) % SGF, cl = (threadIdx.x / (PPF * SGF)) * FCB;   // FCB coils per lane
  for (int base = 0; base < nwin; base += CHUNK) {
    const int nload = min(CHUNK, nwin - base);
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < NR; ++r) { const int q = threadIdx.x + r * NT_F; if (q < 3 * nload) S.rec[q / 3][q % 3] = pr[r]; }
    __syncthreads();
    if (base + CHUNK < nwin) prefetch(base + CHUNK);
#ifndef PROF_NOSTAGE
    stage_samples<1>(S, nload, o0, o1, o2, d, atab, ntab, t0, dt);
    STAGE_KNOT(S, sl, base, nload, b, nint);
    STAGE_HC(sl, base, nload);
    __syncthreads();
    STAGE_WEIGHTS_K(S, nload);
#endif
    __syncthreads();
    FWD_MAC(0, 0)
    (void)fo;
    if (cg + cl < ncb && pp == 0) {
      #pragma unroll
      for (int k = 0; k < FSB; ++k) {
        if (ss + k * SGF < nload) {
          const unsigned int j = (unsigned int)sl[base + ss + k * SGF] & KEY_MASK;
          #pragma unroll
          for (int a = 0; a < FCB; ++a) {
            if (cg + cl + a < ncb) { cx* dst = y + (size_t)(c0 + cg + cl + a) * M + j; FWD_ATOMIC(dst, o[k][a]); }
          }
        }
      }
    }
  }
}

// adjoint: knot b's grid slot (NCH channels of the batch) <- samples of its support; bit 0 of addmask: add, not store
extern "C" __global__ void __launch_bounds__(NT_A) seg_adjoint_k(const float4* __restrict__ rec, const unsigned int* __restrict__ tl, const Item* __restrict__ items,
                                             const cx* __restrict__ atab, int ntab, float t0, float dt,
                                             const cx* __restrict__ y, int M, int c0, int ncb, Grids g, int b, int nint, int nri, int addmask, Dims d KHI_ARGS) {
  extern __shared__ __align__(16) unsigned char smem[];   // dynamic: Stage | list | coil values (host: TileCfg::smem)
  Stage& S = *reinterpret_cast<Stage*>(smem);
  int* sl = reinterpret_cast<int*>(smem + sizeof(Stage));
  cx (*yv)[CPBA] = reinterpret_cast<cx (*)[CPBA]>(smem + KBASE + KHC);
  KHC_DECL
  Item it = items[blockIdx.x];
  const int cg = blockIdx.y * CPBA;
  const int p = threadIdx.x % TT3, cl = (threadIdx.x / TT3) * ACB;   // first of ACB coils
  int o0, o1, o2; tile_origin(it.tile, d, o0, o1, o2);
  const int nwin = it.count > 0 ? KNOT_WINDOW(tl, it.first, it.count, b, nint, sl, NT_A) : 0;
  (void)nri;
  constexpr int NR = (3 * CHUNK + NT_A - 1) / NT_A, NY = (CHUNK * CPBA + NT_A - 1) / NT_A;
  float4 pr[NR]; cx py[NY];
  auto prefetch = [&](int base) {
    const int nl = min(CHUNK, nwin - base);
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
      const int q = threadIdx.x + r * NT_A;
      if (q < 3 * nl) pr[r] = rec[(size_t)((unsigned int)sl[base + q / 3] & KEY_MASK) * 3 + q % 3];
    }
    #pragma unroll
    for (int r = 0; r < NY; ++r) {
      const int q = threadIdx.x + r * NT_A, coil = cg + q % CPBA;
      py[r] = cx{0.f, 0.f};
      if (q < nl * CPBA && coil < ncb) py[r] = y[(size_t)(c0 + coil) * M + ((unsigned int)sl[base + q / CPBA] & KEY_MASK)];
    }
  };
  if (nwin > 0) prefetch(0);
  cx acc[ACB][NSEG];
  #pragma unroll
  for (int a = 0; a < ACB; ++a) {
    #pragma unroll
    for (int l = 0; l < NSEG; ++l) acc[a][l] = cx{0.f, 0.f};
  }
  for (int base = 0; base < nwin; base += CHUNK) {
    const int nload = min(CHUNK, nwin - base);
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < NR; ++r) { const int q = threadIdx.x + r * NT_A; if (q < 3 * nload) S.rec[q / 3][q % 3] = pr[r]; }
    #pragma unroll
    for (int r = 0; r < NY; ++r) { const int q = threadIdx.x + r * NT_A; if (q < CHUNK * CPBA) yv[q / CPBA][q % CPBA] = py[r]; }
    __syncthreads();
    if (base + CHUNK < nwin) prefetch(base + CHUNK);
#ifndef PROF_NOSTAGE
    stage_samples<-1>(S, nload, o0, o1, o2, d, atab, ntab, t0, dt);
    STAGE_KNOT(S, sl, base, nload, b, nint);
    STAGE_HC(sl, base, nload);
    __syncthreads();
    STAGE_WEIGHTS_K(S, nload);
#endif
    __syncthreads();
#ifdef PROF_NOMAC
    for (int s = 0; s < 0; ++s) {
#else
    for (int s = 0; s < nload; ++s) {
#endif
#if TT3 % 32 == 0
      {   // skip samples whose support box misses this warp's 32 tile points (warp-uniform)
        const int bz = S.box[s] >> 16, zlo = (p & ~31) / (TT * TT), zhi = (p | 31) / (TT * TT);
        if ((bz & 15) > zhi || ((bz >> 4) & 15) < zlo) continue;
      }
#endif
      const cx w = S.wt[s][p];
      cx u[ACB];
      #pragma unroll
      for (int a = 0; a < ACB; ++a) { const cx yc = yv[s][cl + a]; u[a] = cx{fmaf(w.x, yc.x, w.y * yc.y), fmaf(w.x, yc.y, -w.y * yc.x)}; }   // conj(w) y
      #pragma unroll
      for (int l = 0; l < NSEG; ++l) {   // explicit FMA chains (no fast-math reassociation): 4 FFMA per complex MAC
        const cx c = S.aa[s][0][l];
        #pragma unroll
        for (int a = 0; a < ACB; ++a) {
          acc[a][l].x = fmaf(u[a].x, c.x, acc[a][l].x); acc[a][l].x = fmaf(-u[a].y, c.y, acc[a][l].x);
          acc[a][l].y = fmaf(u[a].x, c.y, acc[a][l].y); acc[a][l].y = fmaf(u[a].y, c.x, acc[a][l].y);
        }
      }
    }
  }
  it.count = nwin;   // an item with nothing in the window adds nothing to a slot in add mode
  adj_store<1, CPBA, ACB>(smem, acc, p, 0, cl, NT_A, it, cg, ncb, g.p, addmask, o0, o1, o2, d);
}
#endif   // KV == 0

#if KV >= 1
// ---------------------------------------------------------------- variants ≥ 1 (dense regime, 2026-11-16)
// Forward, variant 1: the original sample-lane layout (lanes = (sample group, coil, point stripe), tile values broadcast
// from shared memory), with the tile values read as float4 pairs (NSEG even) and, at order 0, real weights: the weight
// of an order-0 kernel has no imaginary part, so 2 of the 4 FFMA of each complex MAC multiply an exact zero (same sums,
// bit for bit). Variant 2: point-lane layout, see seg_forward_k below.
// Adjoint (all variants ≥ 1): warps cover a spatial sub-box of the tile (2×2×2 points × 4 coils when the coil group
// allows, else 4×2×2 × 2 or 4×4×2 × 1) instead of 32 consecutive points, so a sample whose support box misses the
// sub-box is skipped by the whole warp (§17 of research/temporal_operator.md: 64 → ≈ 36 points per (sample, tile) at
// W = 8 vs ≈ 53 for the z-slab skip of variant 0). The coil values are folded into the coefficients once per staged
// (sample, coil, segment), z = y a_l, so the inner MAC is conj(w) z (order 0: real w, 2 FFMA).
#if TT != 4
#error "kernel variants >= 1 need TT = 4"
#endif
// adjoint coil register blocking (2026-11-19): a thread holds ACB consecutive coils (ACB · NSEG accumulators), so each
// staged weight load serves ACB · NSEG MACs; the warp's coil groups count thread groups of ACB coils
#if CPBA % ACB != 0
#error "ACB must divide CPBA"
#endif
#define CGA (CPBA / ACB)                                     // thread coil groups per block
#define CWA (CGA % 4 == 0 ? 4 : (CGA % 2 == 0 ? 2 : 1))      // thread coil groups per warp
#define PGA (32 / CWA)                                       // points per warp: 8, 16, 32
#define SXA (PGA == 8 ? 2 : 4)
#define SYA (PGA == 32 ? 4 : 2)
#define SZA 2
// zc: [CHUNK][CPBA][NSEG] cx (16-aligned) after the list and the coil values
#define ZC_OFF (((KBASE + KHC) + 8 * CHUNK * CPBA + 15) & ~(size_t)15)

// stage A of variants ≥ 1: stage_samples with the knot weight β_b(t) folded into the one pass over (sample, segment)
// (no separate STAGE_KNOT pass); the adjoint (ZC) writes z = y a β for the CPBA coils directly instead of aa
template <int DIR, bool ZC>
__device__ __forceinline__ void stage_kv(Stage& S, const int* sl, int base, int nload, int b, int nint, int o0, int o1, int o2, const Dims& d,
                                         const cx* __restrict__ atab, int ntab, float t0, float dt, cx* zc, const cx (*yv)[CPBA]) {
  for (int q = threadIdx.x; q < nload * 3 * TT; q += blockDim.x) {
    const int s = q / (3 * TT), a = (q / TT) % 3, cell = q % TT;
    const float4 r = S.rec[s][0];
    const float sc = a == 0 ? r.x : (a == 1 ? r.y : r.z);
    const int og = a == 0 ? o0 : (a == 1 ? o1 : o2), n = a == 0 ? d.nf0 : (a == 1 ? d.nf1 : d.nf2);
    const float z = wrapdiff((float)(og + cell) - sc, n) * (2.f / (float)W);
    float o[5];
    kern_derivs(z, o);
    #pragma unroll
    for (int k = 0; k < NK; ++k) S.ax[s][a][k][cell] = o[k];
  }
  for (int q = threadIdx.x; q < nload * NSEG; q += blockDim.x) {
    const int sm = q / NSEG, l = q % NSEG;
    float u = (S.rec[sm][0].w - t0) / dt;
    u = fminf(fmaxf(u, 0.f), (float)(ntab - 1) - 1e-3f);
    const int i = (int)u;
    const float fr = u - (float)i;
    const cx pa = atab[i * NSEG + l], pb = atab[(i + 1) * NSEG + l];
    const cx a = cx{fmaf(fr, pb.x - pa.x, pa.x), fmaf(fr, pb.y - pa.y, pa.y)};
    float s0, c1; sincosf(S.rec[sm][1].x, &s0, &c1);
    const cx v = DIR > 0 ? cmul(a, cx{c1, -s0}) : cmulc(a, cx{c1, s0});
    const float beta = knot_beta(S, sl, base, sm, b, nint);
    const cx w = cx{beta * v.x, beta * v.y};
    if (ZC) {
      #pragma unroll
      for (int c = 0; c < CPBA; ++c) zc[(sm * CPBA + c) * NSEG + l] = cmul(yv[sm][c], w);
    } else {
      S.aa[sm][0][l] = w;
    }
  }
  for (int q = threadIdx.x; q < nload; q += blockDim.x) {
    const float4 r = S.rec[q][0];
    int bx = 0;
    #pragma unroll
    for (int a = 0; a < 3; ++a) {
      const float sc = a == 0 ? r.x : (a == 1 ? r.y : r.z);
      const int og = a == 0 ? o0 : (a == 1 ? o1 : o2), n = a == 0 ? d.nf0 : (a == 1 ? d.nf1 : d.nf2);
      const float dd = wrapdiff((float)og - sc, n);
      const int lo = max(0, (int)floorf(-0.5f * W - dd) + 1), hi = min(TT - 1, (int)ceilf(0.5f * W - dd) - 1);
      bx |= (lo | (hi << 4)) << (8 * a);
    }
    S.box[q] = bx;
#if CORR >= 2
    const float4 r1 = S.rec[q][1], r2 = S.rec[q][2];
    const float dd[6] = {r1.y, r1.z, r1.w, r2.x, r2.y, r2.z};
    const int mx[6] = {2, 0, 0, 1, 1, 0}, my[6] = {0, 2, 0, 1, 0, 1};
    float e[NE];
    #pragma unroll
    for (int k = 0; k < NE; ++k) e[k] = 0.f;
    #pragma unroll
    for (int p = 0; p < 6; ++p) {
      #pragma unroll
      for (int pq = 0; pq < 6; ++pq) {
        const int ea = mx[p] + mx[pq], eb = my[p] + my[pq];
        e[ea * (11 - ea) / 2 + eb] = fmaf(-0.5f * dd[p], dd[pq], e[ea * (11 - ea) / 2 + eb]);
      }
    }
    #pragma unroll
    for (int k = 0; k < NE; ++k) S.ee[q][k] = e[k];
#endif
  }
}

// forward, variants 1 and 2: sample lanes. Variant 2 (opt-in, HASTY_SEG_KV=2 with HASTY_SEG_BOXSORT=1): lanes ordered
// point stripe, coil, sample group (a warp holds few, box-sorted samples) and the p loop runs over the union of the
// warp's support boxes only; measured slower than the dense unrolled loop of variant 1 (§17).
#if KV == 1 || KV == 2
extern "C" __global__ void __launch_bounds__(NT_F) seg_forward_k(const float4* __restrict__ rec, const unsigned int* __restrict__ tl, const Item* __restrict__ items,
                                             const cx* __restrict__ atab, int ntab, float t0, float dt,
                                             Grids g, int b, int nint, int nri, cx* __restrict__ y, int M, int c0, int ncb, Dims d KHI_ARGS) {
  extern __shared__ __align__(16) unsigned char smem[];   // dynamic: Stage | list | tile values (host: TileCfg::smem)
  Stage& S = *reinterpret_cast<Stage*>(smem);
  int* sl = reinterpret_cast<int*>(smem + sizeof(Stage));
  cx* gt = reinterpret_cast<cx*>(smem + KBASE + KHC);
  KHC_DECL
  const Item it = items[blockIdx.x];
  if (it.count == 0) return;
  const int nwin = KNOT_WINDOW(tl, it.first, it.count, b, nint, sl, NT_F);
  (void)nri;
  if (nwin == 0) return;
  const int cg = blockIdx.y * CPBF;
  int o0, o1, o2; tile_origin(it.tile, d, o0, o1, o2);
  for (int q = threadIdx.x; q < TT3 * GN; q += NT_F) {
    const int p = q / GN, c = (q / NSEG) % CPBF, l = q % NSEG;
    const int gx = o0 + p % TT, gy = o1 + (p / TT) % TT, gz = o2 + p / (TT * TT);
    cx v = cx{0.f, 0.f};
    if (cg + c < ncb && gx < d.nf0 && gy < d.nf1 && gz < d.nf2) v = GLOAD(g.p[0], (((size_t)gz * d.nf1 + gy) * d.nf0 + gx) * NCH + (cg + c) * NSEG + l);
    gt[p * GTS + c * NSEG + l] = v;
  }
  constexpr int NR = (3 * CHUNK + NT_F - 1) / NT_F;
  float4 pr[NR];
  auto prefetch = [&](int base) {
    const int nl = min(CHUNK, nwin - base);
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
      const int q = threadIdx.x + r * NT_F;
      if (q < 3 * nl) pr[r] = rec[(size_t)((unsigned int)sl[base + q / 3] & KEY_MASK) * 3 + q % 3];
    }
  };
  __syncthreads();
  prefetch(0);
#if KV == 2
#if FCB != 1
#error "kernel variant 2: FCB = 1"
#endif
  // lanes: point stripe fastest, then coil, then sample group (a warp holds few samples: its union box stays small);
  // lane group ss handles the FSB adjacent samples ss · FSB + k
  const int pp = threadIdx.x % PPF, cl = (threadIdx.x / PPF) % CPBF, ss = threadIdx.x / (PPF * CPBF);
#define SIDX(k) (ss * FSB + (k))
#else
  // lanes as variant 0: point stripe, sample group, coil group; lane group ss handles samples ss + k · SGF and the FCB
  // coils cl .. cl + FCB − 1 (their FCB · NSEG tile values are contiguous)
  const int pp = threadIdx.x % PPF, ss = (threadIdx.x / PPF) % SGF, cl = (threadIdx.x / (PPF * SGF)) * FCB;
#define SIDX(k) (ss + (k) * SGF)
#endif
#define NJ (FCB * NSEG)
  for (int base = 0; base < nwin; base += CHUNK) {
    const int nload = min(CHUNK, nwin - base);
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < NR; ++r) { const int q = threadIdx.x + r * NT_F; if (q < 3 * nload) S.rec[q / 3][q % 3] = pr[r]; }
    __syncthreads();
    if (base + CHUNK < nwin) prefetch(base + CHUNK);
#ifndef PROF_NOSTAGE
    stage_kv<1, false>(S, sl, base, nload, b, nint, o0, o1, o2, d, atab, ntab, t0, dt, nullptr, nullptr);
    STAGE_HC(sl, base, nload);
    __syncthreads();
    STAGE_WEIGHTS_K(S, nload);
#endif
    __syncthreads();
    cx o[FSB][FCB];
    #pragma unroll
    for (int k = 0; k < FSB; ++k) {
      #pragma unroll
      for (int a = 0; a < FCB; ++a) o[k][a] = cx{0.f, 0.f};
    }
#if KV == 2
    // union of the warp's support boxes (lists are box-sorted within a key, `box_sort`): loop only over its points
    unsigned int ulo[3] = {15u, 15u, 15u}, uhi[3] = {0u, 0u, 0u};
    #pragma unroll
    for (int k = 0; k < FSB; ++k) {
      if (SIDX(k) < nload) {
        const unsigned int bx = (unsigned int)S.box[SIDX(k)];
        #pragma unroll
        for (int a = 0; a < 3; ++a) { ulo[a] = min(ulo[a], (bx >> (8 * a)) & 15u); uhi[a] = max(uhi[a], (bx >> (8 * a + 4)) & 15u); }
      }
    }
    #pragma unroll
    for (int a = 0; a < 3; ++a) { ulo[a] = __reduce_min_sync(0xffffffffu, ulo[a]); uhi[a] = __reduce_max_sync(0xffffffffu, uhi[a]); }
#endif
#ifdef PROF_NOMAC
    if (false) {
#else
    if (SIDX(0) < nload && cg + cl < ncb) {
#endif
      // acc[k][j], j = a · NSEG + l (coil cl + a, segment l); FCB = 1: the original [k][l]
      cx acc[FSB][NJ];
      #pragma unroll
      for (int k = 0; k < FSB; ++k) {
        #pragma unroll
        for (int j = 0; j < NJ; ++j) acc[k][j] = cx{0.f, 0.f};
      }
#if KV == 2
      for (int pz = (int)ulo[2]; pz <= (int)uhi[2]; ++pz)
      for (int py = (int)ulo[1]; py <= (int)uhi[1]; ++py)
      for (int px = (int)ulo[0] + pp; px <= (int)uhi[0]; px += PPF) {
        const int p = (pz * TT + py) * TT + px;
#else
      for (int p = pp; p < TT3; p += PPF) {
#endif
        const cx* gv = gt + p * GTS + cl * NSEG;
#if CORR == 0
        float w[FSB];   // order 0: real weights
        #pragma unroll
        for (int k = 0; k < FSB; ++k) w[k] = S.wt[SIDX(k)][p];
#if NJ % 2 == 0
        #pragma unroll
        for (int j2 = 0; j2 < NJ / 2; ++j2) {
          const float4 v = reinterpret_cast<const float4*>(gv)[j2];
          #pragma unroll
          for (int k = 0; k < FSB; ++k) {
            acc[k][2 * j2].x = fmaf(w[k], v.x, acc[k][2 * j2].x); acc[k][2 * j2].y = fmaf(w[k], v.y, acc[k][2 * j2].y);
            acc[k][2 * j2 + 1].x = fmaf(w[k], v.z, acc[k][2 * j2 + 1].x); acc[k][2 * j2 + 1].y = fmaf(w[k], v.w, acc[k][2 * j2 + 1].y);
          }
        }
#else
        #pragma unroll
        for (int j = 0; j < NJ; ++j) {
          const cx v = gv[j];
          #pragma unroll
          for (int k = 0; k < FSB; ++k) { acc[k][j].x = fmaf(w[k], v.x, acc[k][j].x); acc[k][j].y = fmaf(w[k], v.y, acc[k][j].y); }
        }
#endif
#else
        cx w[FSB];
        #pragma unroll
        for (int k = 0; k < FSB; ++k) w[k] = S.wt[SIDX(k)][p];
#if NJ % 2 == 0
        #pragma unroll
        for (int j2 = 0; j2 < NJ / 2; ++j2) {
          const float4 v4 = reinterpret_cast<const float4*>(gv)[j2];
          #pragma unroll
          for (int h = 0; h < 2; ++h) {
            const cx v = h == 0 ? cx{v4.x, v4.y} : cx{v4.z, v4.w};
            const int j = 2 * j2 + h;
            #pragma unroll
            for (int k = 0; k < FSB; ++k) {
              acc[k][j].x = fmaf(w[k].x, v.x, acc[k][j].x); acc[k][j].x = fmaf(-w[k].y, v.y, acc[k][j].x);
              acc[k][j].y = fmaf(w[k].x, v.y, acc[k][j].y); acc[k][j].y = fmaf(w[k].y, v.x, acc[k][j].y);
            }
          }
        }
#else
        #pragma unroll
        for (int j = 0; j < NJ; ++j) {
          const cx v = gv[j];
          #pragma unroll
          for (int k = 0; k < FSB; ++k) {
            acc[k][j].x = fmaf(w[k].x, v.x, acc[k][j].x); acc[k][j].x = fmaf(-w[k].y, v.y, acc[k][j].x);
            acc[k][j].y = fmaf(w[k].x, v.y, acc[k][j].y); acc[k][j].y = fmaf(w[k].y, v.x, acc[k][j].y);
          }
        }
#endif
#endif
      }
      #pragma unroll
      for (int k = 0; k < FSB; ++k) {
        #pragma unroll
        for (int a = 0; a < FCB; ++a) {
          #pragma unroll
          for (int l = 0; l < NSEG; ++l) { const cx v = cmul(S.aa[SIDX(k)][0][l], acc[k][a * NSEG + l]); o[k][a].x += v.x; o[k][a].y += v.y; }
        }
      }
    }
    #pragma unroll
    for (int k = 0; k < FSB; ++k) {
      #pragma unroll
      for (int a = 0; a < FCB; ++a) {
        #pragma unroll
        for (int sh = PPF / 2; sh > 0; sh >>= 1) { o[k][a].x += __shfl_xor_sync(0xffffffffu, o[k][a].x, sh); o[k][a].y += __shfl_xor_sync(0xffffffffu, o[k][a].y, sh); }
      }
    }
    if (cg + cl < ncb && pp == 0) {
      #pragma unroll
      for (int k = 0; k < FSB; ++k) {
        if (SIDX(k) < nload) {
          const unsigned int j = (unsigned int)sl[base + SIDX(k)] & KEY_MASK;
          #pragma unroll
          for (int a = 0; a < FCB; ++a) {
            if (cg + cl + a < ncb) { cx* dst = y + (size_t)(c0 + cg + cl + a) * M + j; FWD_ATOMIC(dst, o[k][a]); }
          }
        }
      }
    }
  }
}
#undef NJ
#undef SIDX
#endif   // KV == 1 || KV == 2 (forward)

// adjoint, variants ≥ 1: sub-box warps, z = y a_l
extern "C" __global__ void __launch_bounds__(NT_A) seg_adjoint_k(const float4* __restrict__ rec, const unsigned int* __restrict__ tl, const Item* __restrict__ items,
                                             const cx* __restrict__ atab, int ntab, float t0, float dt,
                                             const cx* __restrict__ y, int M, int c0, int ncb, Grids g, int b, int nint, int nri, int addmask, Dims d KHI_ARGS) {
  extern __shared__ __align__(16) unsigned char smem[];   // dynamic: Stage | list | coil values | z (host: TileCfg::smem)
  Stage& S = *reinterpret_cast<Stage*>(smem);
  int* sl = reinterpret_cast<int*>(smem + sizeof(Stage));
  cx (*yv)[CPBA] = reinterpret_cast<cx (*)[CPBA]>(smem + KBASE + KHC);
  KHC_DECL
  cx* zc = reinterpret_cast<cx*>(smem + ZC_OFF);
  Item it = items[blockIdx.x];
  const int cg = blockIdx.y * CPBA;
  // warp → (point group, coil group); lane → (point in group, coil in group)
  const int lane = threadIdx.x & 31, wp = threadIdx.x >> 5;
  constexpr int NGX = 4 / SXA, NGY = 4 / SYA, NPG = 64 / PGA;
  const int gi = wp % NPG, pi = lane % PGA;
  const int cl = ((wp / NPG) * CWA + lane / PGA) * ACB;   // first of the thread's ACB coils
  const int gx = gi % NGX, gy = (gi / NGX) % NGY, gz = gi / (NGX * NGY);
  const int p = ((gz * SZA + pi / (SXA * SYA)) * TT + gy * SYA + (pi / SXA) % SYA) * TT + gx * SXA + pi % SXA;
  // the group's box, packed as the support boxes: lo | hi << 4 per axis
  const int x0 = gx * SXA, x1 = x0 + SXA - 1, y0 = gy * SYA, y1 = y0 + SYA - 1, z0 = gz * SZA, z1 = z0 + SZA - 1;
  int o0, o1, o2; tile_origin(it.tile, d, o0, o1, o2);
  const int nwin = it.count > 0 ? KNOT_WINDOW(tl, it.first, it.count, b, nint, sl, NT_A) : 0;
  (void)nri;
  constexpr int NR = (3 * CHUNK + NT_A - 1) / NT_A, NY = (CHUNK * CPBA + NT_A - 1) / NT_A;
  float4 pr[NR]; cx py[NY];
  auto prefetch = [&](int base) {
    const int nl = min(CHUNK, nwin - base);
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
      const int q = threadIdx.x + r * NT_A;
      if (q < 3 * nl) pr[r] = rec[(size_t)((unsigned int)sl[base + q / 3] & KEY_MASK) * 3 + q % 3];
    }
    #pragma unroll
    for (int r = 0; r < NY; ++r) {
      const int q = threadIdx.x + r * NT_A, coil = cg + q % CPBA;
      py[r] = cx{0.f, 0.f};
      if (q < nl * CPBA && coil < ncb) py[r] = y[(size_t)(c0 + coil) * M + ((unsigned int)sl[base + q / CPBA] & KEY_MASK)];
    }
  };
  if (nwin > 0) prefetch(0);
  cx acc[ACB][NSEG];
  #pragma unroll
  for (int a = 0; a < ACB; ++a) {
    #pragma unroll
    for (int l = 0; l < NSEG; ++l) acc[a][l] = cx{0.f, 0.f};
  }
  for (int base = 0; base < nwin; base += CHUNK) {
    const int nload = min(CHUNK, nwin - base);
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < NR; ++r) { const int q = threadIdx.x + r * NT_A; if (q < 3 * nload) S.rec[q / 3][q % 3] = pr[r]; }
    #pragma unroll
    for (int r = 0; r < NY; ++r) { const int q = threadIdx.x + r * NT_A; if (q < CHUNK * CPBA) yv[q / CPBA][q % CPBA] = py[r]; }
    __syncthreads();
    if (base + CHUNK < nwin) prefetch(base + CHUNK);
#ifndef PROF_NOSTAGE
    stage_kv<-1, true>(S, sl, base, nload, b, nint, o0, o1, o2, d, atab, ntab, t0, dt, zc, yv);
    STAGE_HC(sl, base, nload);
    __syncthreads();
    STAGE_WEIGHTS_K(S, nload);
#endif
    __syncthreads();
#ifndef PROF_NOMAC
    for (int s0 = 0; s0 < nload; s0 += 32) {
      bool hit = false;
      if (s0 + lane < nload) {
        const int bx = S.box[s0 + lane];
        hit = (bx & 15) <= x1 && ((bx >> 4) & 15) >= x0 && ((bx >> 8) & 15) <= y1 && ((bx >> 12) & 15) >= y0 && ((bx >> 16) & 15) <= z1 && ((bx >> 20) & 15) >= z0;
      }
      unsigned int msk = __ballot_sync(0xffffffffu, hit);
      while (msk) {
        const int s = s0 + __ffs(msk) - 1;
        msk &= msk - 1;
        const cx* zp = zc + (s * CPBA + cl) * NSEG;   // the thread's ACB · NSEG coefficients, contiguous
#define ACC(J) acc[(J) / NSEG][(J) % NSEG]
#if CORR == 0
        const float w = S.wt[s][p];
#if (ACB * NSEG) % 2 == 0
        #pragma unroll
        for (int j2 = 0; j2 < ACB * NSEG / 2; ++j2) {
          const float4 z = reinterpret_cast<const float4*>(zp)[j2];
          ACC(2 * j2).x = fmaf(w, z.x, ACC(2 * j2).x); ACC(2 * j2).y = fmaf(w, z.y, ACC(2 * j2).y);
          ACC(2 * j2 + 1).x = fmaf(w, z.z, ACC(2 * j2 + 1).x); ACC(2 * j2 + 1).y = fmaf(w, z.w, ACC(2 * j2 + 1).y);
        }
#else
        #pragma unroll
        for (int j = 0; j < ACB * NSEG; ++j) { const cx z = zp[j]; ACC(j).x = fmaf(w, z.x, ACC(j).x); ACC(j).y = fmaf(w, z.y, ACC(j).y); }
#endif
#else
        const cx w = S.wt[s][p];
        #pragma unroll
        for (int j = 0; j < ACB * NSEG; ++j) {   // conj(w) z
          const cx z = zp[j];
          ACC(j).x = fmaf(w.x, z.x, ACC(j).x); ACC(j).x = fmaf(w.y, z.y, ACC(j).x);
          ACC(j).y = fmaf(w.x, z.y, ACC(j).y); ACC(j).y = fmaf(-w.y, z.x, ACC(j).y);
        }
#endif
#undef ACC
      }
    }
#endif
  }
  it.count = nwin;
  adj_store<1, CPBA, ACB>(smem, acc, p, 0, cl, NT_A, it, cg, ncb, g.p, addmask, o0, o1, o2, d);
}
#endif   // KV >= 1
