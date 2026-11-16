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
#define KNOT_WINDOW(tl, first, count, b, nint, sl, nthr) knot_window2d(tl, first, count, b, nint, nri, sl, nthr)
#define STAGE_KNOT(S, sl, base, nload, b, nint) stage_knot2d(S, sl, base, nload, b, nint)
#else
#define KNOT_WINDOW(tl, first, count, b, nint, sl, nthr) knot_window(tl, first, count, b, nint, sl, nthr)
#define STAGE_KNOT(S, sl, base, nload, b, nint) stage_knot(S, sl, base, nload, b, nint)
#endif


// forward: samples of knot b's support (+=, y initialised by the host) <- knot b's grid slot
extern "C" __global__ void __launch_bounds__(NT_F) seg_forward_k(const float4* __restrict__ rec, const unsigned int* __restrict__ tl, const Item* __restrict__ items,
                                             const cx* __restrict__ atab, int ntab, float t0, float dt,
                                             Grids g, int b, int nint, int nri, cx* __restrict__ y, int M, int c0, int ncb, Dims d) {
  extern __shared__ __align__(16) unsigned char smem[];   // dynamic: Stage | list | tile values (host: TileCfg::smem)
  Stage& S = *reinterpret_cast<Stage*>(smem);
  int* sl = reinterpret_cast<int*>(smem + sizeof(Stage));
  cx* gt = reinterpret_cast<cx*>(smem + ((sizeof(Stage) + 4 * LISTMAX + 15) & ~(size_t)15));
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
    if (cg + c < ncb && gx < d.nf0 && gy < d.nf1 && gz < d.nf2) v = g.p[0][(((size_t)gz * d.nf1 + gy) * d.nf0 + gx) * NCH + (cg + c) * NSEG + l];
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
  const int pp = threadIdx.x % PPF, fo = 0, ss = (threadIdx.x / PPF) % SGF, cl = threadIdx.x / (PPF * SGF);
  for (int base = 0; base < nwin; base += CHUNK) {
    const int nload = min(CHUNK, nwin - base);
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < NR; ++r) { const int q = threadIdx.x + r * NT_F; if (q < 3 * nload) S.rec[q / 3][q % 3] = pr[r]; }
    __syncthreads();
    if (base + CHUNK < nwin) prefetch(base + CHUNK);
    stage_samples<1>(S, nload, o0, o1, o2, d, atab, ntab, t0, dt);
    STAGE_KNOT(S, sl, base, nload, b, nint);
    __syncthreads();
    stage_weights(S, nload);
    __syncthreads();
    FWD_MAC(0, 0)
    (void)fo;
    if (cg + cl < ncb && pp == 0) {
      #pragma unroll
      for (int k = 0; k < FSB; ++k) {
        if (ss + k * SGF < nload) {
          cx* dst = y + (size_t)(c0 + cg + cl) * M + ((unsigned int)sl[base + ss + k * SGF] & KEY_MASK);
          atomicAdd(&dst->x, o[k].x);
          atomicAdd(&dst->y, o[k].y);
        }
      }
    }
  }
}

// adjoint: knot b's grid slot (NCH channels of the batch) <- samples of its support; bit 0 of addmask: add, not store
extern "C" __global__ void __launch_bounds__(NT_A) seg_adjoint_k(const float4* __restrict__ rec, const unsigned int* __restrict__ tl, const Item* __restrict__ items,
                                             const cx* __restrict__ atab, int ntab, float t0, float dt,
                                             const cx* __restrict__ y, int M, int c0, int ncb, Grids g, int b, int nint, int nri, int addmask, Dims d) {
  extern __shared__ __align__(16) unsigned char smem[];   // dynamic: Stage | list | coil values (host: TileCfg::smem)
  Stage& S = *reinterpret_cast<Stage*>(smem);
  int* sl = reinterpret_cast<int*>(smem + sizeof(Stage));
  cx (*yv)[CPBA] = reinterpret_cast<cx (*)[CPBA]>(smem + ((sizeof(Stage) + 4 * LISTMAX + 15) & ~(size_t)15));
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
    stage_samples<-1>(S, nload, o0, o1, o2, d, atab, ntab, t0, dt);
    STAGE_KNOT(S, sl, base, nload, b, nint);
    __syncthreads();
    stage_weights(S, nload);
    __syncthreads();
    for (int s = 0; s < nload; ++s) {
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
