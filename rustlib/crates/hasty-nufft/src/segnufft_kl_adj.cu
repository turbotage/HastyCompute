// Knot-path adjoint `kl` of the segmented NUFFT (research/sample_kernels.md §17; host segnufft/kl.rs): the ra adjoint
// (segnufft_ra.cu, design 7; same block / warp / lane mapping, accumulators, staging arithmetic and output park) with
//   * whole tiles: one block per tile and knot, over the tile's whole key-sorted list (all parts of a split tile; parts
//     with split == 1 exit, the first part (split == 2) reads the tile's total from `tcount`), so every tile is stored
//     by exactly one block: no atomics and no zeroing of split tiles (at the real scan density nearly every tile of the
//     thorax grid is split at max_item 1024, §17);
//   * the knot window read in place: its ≤ 2 (1-D) / ≤ 4 (2-D) contiguous key runs of the tile list found by binary
//     search (no scan of the whole list, no shared-memory copy, no LISTMAX limit);
//   * the degree-3/4 table htab in shared memory once per block and u (uw) prefetched with the records.
// Output: store (bit 0 of addmask: read-modify-write). An empty window stores zeros (store mode). Defines as ra.
// Phase-2 variants (research/sample_kernels.md §19; defaults = the kernel above): RA_STAGES 2 copies chunk k + 1's
// records, coil values and uw by cp.async (sm_80+; plain copies below) into a second shared buffer while chunk k is
// staged and MACed (list entries prefetched into registers one chunk further ahead); LBMIN m > 0:
// __launch_bounds__(NTA, m). Smaller chunks (CHK) are host candidates; the code is CHK-generic.
struct cx { float x, y; };
__device__ __forceinline__ cx cmul(cx a, cx b) { return cx{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
__device__ __forceinline__ cx cmulc(cx a, cx b) { return cx{a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x}; } // conj(a)*b
struct Dims { int nf0, nf1, nf2, nb0, nb1, nb2; };
struct Grids { cx* p[4]; };
struct RaItem { int tile, first, count, split; };
#define NCH (NCB * NSEG)
#ifndef TT
#define TT 4
#endif
#if TT != 4
#error "kl: TT = 4 (the engine's tile lists)"
#endif

// kernel window and its z-derivatives (copies of segnufft.cu es3 / es5, ES and KB)
#ifdef KER_KB
__device__ __forceinline__ void es3(float z, float& f, float& d1, float& d2) { float o[3]; kb_eval<3>(z, o); f = o[0]; d1 = o[1]; d2 = o[2]; }
#if CORR >= 2
__device__ __forceinline__ void es5(float z, float* o) { kb_eval<5>(z, o); }
#endif
#else
__device__ __forceinline__ void es3(float z, float& f, float& d1, float& d2) {
  const float a = 1.f - z * z;
  if (a <= 0.f) { f = 0.f; d1 = 0.f; d2 = 0.f; return; }
  const float s = sqrtf(a);
  f = expf((float)BETA * (s - 1.f));
  if (fabsf(z) > 0.98f) { d1 = 0.f; d2 = 0.f; return; }
  const float is = rsqrtf(a), bz = (float)BETA * z * is;
  d1 = -f * bz;
  d2 = f * (bz * bz - (float)BETA * is * __frcp_rn(a));
}
#if CORR >= 2
__device__ __forceinline__ void es5(float z, float* o) {
  const float a = 1.f - z * z;
  if (a <= 0.f) { o[0] = o[1] = o[2] = o[3] = o[4] = 0.f; return; }
  const float s = sqrtf(a);
  const float f = expf((float)BETA * (s - 1.f));
  o[0] = f;
  if (fabsf(z) > 0.98f) { o[1] = o[2] = o[3] = o[4] = 0.f; return; }
  const float b = (float)BETA, ia = __frcp_rn(a), is = rsqrtf(a), is3 = is * ia, is5 = is3 * ia, is7 = is5 * ia;
  const float u1 = -b * z * is, u2 = -b * is3, u3 = -3.f * b * z * is5, u4 = -3.f * b * (1.f + 4.f * z * z) * is7;
  o[1] = u1 * f;
  o[2] = (u2 + u1 * u1) * f;
  o[3] = (u3 + 3.f * u1 * u2 + u1 * u1 * u1) * f;
  o[4] = (u4 + 4.f * u1 * u3 + 3.f * u2 * u2 + 6.f * u1 * u1 * u2 + u1 * u1 * u1 * u1) * f;
}
#endif
#endif
#if CORR >= 2
#define NKA 5
#elif CORR == 1
#define NKA 3
#else
#define NKA 1
#endif
// as segnufft.cu kern_derivs
__device__ __forceinline__ void kern_derivs(float z, float* o) {
#if CORR >= 2
  es5(z, o);
#else
  float f, d1, d2; es3(z, f, d1, d2); o[0] = f;
#if CORR == 1
  o[1] = d1; o[2] = d2;
#endif
#endif
}
__device__ __forceinline__ float wrapdiff(float a, int n) { // nearest periodic difference (segnufft.cu)
  if (a > 0.5f * n) a -= n;
  if (a < -0.5f * n) a += n;
  return a;
}

// ---------------------------------------------------------------- knot windows and weights β_b(t) (segnufft_knot.cu)
#ifndef KS
#define KS 0
#endif
#ifndef RDEG
#define RDEG -1
#endif
#if KS > 0
#define KOFF (-((KS - 2) / 2))
#define KEY_SHIFT 25
#define KEY_MASK 0x1FFFFFFu
__device__ __forceinline__ float bspline_k(int o, float u) {
#if KS == 1
  return 1.f;
#elif KS == 2
  return o == 0 ? 1.f - u : u;
#elif KS == 4
  const float v = 1.f - u, u2 = u * u, u3 = u2 * u;
  return o == 0 ? v * v * v * (1.f / 6.f) : o == 1 ? (3.f * u3 - 6.f * u2 + 4.f) * (1.f / 6.f) : o == 2 ? (-3.f * u3 + 3.f * u2 + 3.f * u + 1.f) * (1.f / 6.f) : u3 * (1.f / 6.f);
#else
#error "KS must be 1, 2 or 4"
#endif
}
#if RDEG >= 0
__device__ __forceinline__ int lower_key(const unsigned int* __restrict__ tl, int first, int count, int key) {
  int lo = 0, hi = count;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if ((int)(tl[first + mid] >> KEY_SHIFT) < key) lo = mid + 1; else hi = mid;
  }
  return lo;
}
// knot b's window in the tile's key-sorted list tl[first, first + count): ≤ 4 contiguous runs (as segnufft_knot.cu
// knot_window2d), bounds by binary search; run r = [RLO[r], RLO[r] + RN[r]). Returns the window length (block-uniform;
// contains a barrier, call from all threads).
__device__ __forceinline__ int kl_runs(const unsigned int* __restrict__ tl, int first, int count, int b, int nint, int nri, int* RLO, int* RN) {
  __shared__ int rb[8][2];
  const int k = b / nint, bc = b % nint;
  const int khi = ((bc - KOFF) % nint + nint) % nint, klo = ((bc - KOFF - KS + 1) % nint + nint) % nint;
  const int r0 = RDEG == 0 ? k : max(k - 1, 0), r1 = RDEG == 0 ? k : min(k, nri - 1);
  if (threadIdx.x < 8) {
    const int q = threadIdx.x;
    const int ri = r0 + (q >> 2);
    const int part = (q >> 1) & 1;
    const bool wrapw = klo > khi || KS >= nint;
    bool live = ri <= r1 && (part == 0 || (wrapw && KS < nint));
    if (live) {
      int a, z;
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
  int tot = 0;
  #pragma unroll
  for (int r = 0; r < 4; ++r) { RLO[r] = rb[r][0]; RN[r] = max(0, rb[r][1] - rb[r][0]); tot += RN[r]; }
  return tot;
}
__device__ __forceinline__ float ra_beta(unsigned int e, float w11, int b, int nint) {
  const int k = b / nint, bc = b % nint;
  const int key = (int)(e >> KEY_SHIFT), ri = key / nint, ic = key % nint;
  const unsigned int bits = __float_as_uint(w11);
  const float u = (float)(bits >> 16) * (1.f / 65535.f), v = (float)(bits & 0xFFFFu) * (1.f / 65535.f);
  const int o = ((bc - KOFF - ic) % nint + nint) % nint;
  return (RDEG == 0 ? 1.f : (ri == k ? 1.f - v : v)) * bspline_k(o, u);
}
#else
__device__ __forceinline__ int lower_key(const unsigned int* __restrict__ tl, int first, int count, int key) {
  int lo = 0, hi = count;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if ((int)(tl[first + mid] >> KEY_SHIFT) < key) lo = mid + 1; else hi = mid;
  }
  return lo;
}
// knot b's window (intervals b − KOFF − KS + 1 .. b − KOFF, periodic; as segnufft_knot.cu knot_window): ≤ 2 runs of
// the interval-sorted list, bounds by binary search (no scan of the list)
__device__ __forceinline__ int kl_runs(const unsigned int* __restrict__ tl, int first, int count, int b, int nint, int nri, int* RLO, int* RN) {
  (void)nri;
  __shared__ int rb[4];
  const int khi = ((b - KOFF) % nint + nint) % nint, klo = ((b - KOFF - KS + 1) % nint + nint) % nint;
  if (threadIdx.x < 2) rb[threadIdx.x] = lower_key(tl, first, count, threadIdx.x == 0 ? klo : khi + 1);
  __syncthreads();
  const int clo = rb[0], chi = rb[1];   // entries with key < klo, key ≤ khi
  RLO[2] = RLO[3] = 0; RN[2] = RN[3] = 0;
  if (KS >= nint) { RLO[0] = 0; RN[0] = count; RLO[1] = 0; RN[1] = 0; }
  else if (klo > khi) { RLO[0] = 0; RN[0] = chi; RLO[1] = clo; RN[1] = count - clo; }   // wrapped window
  else { RLO[0] = clo; RN[0] = chi - clo; RLO[1] = 0; RN[1] = 0; }
  return RN[0] + RN[1];
}
__device__ __forceinline__ float ra_beta(unsigned int e, float w11, int b, int nint) {
  const int key = (int)(e >> KEY_SHIFT);
  return bspline_k(((b - KOFF - key) % nint + nint) % nint, w11);
}
#endif
#define SAMPLE_OF(e) ((e) & KEY_MASK)
#else
#error "kl: knot-major engines only (KS > 0)"
#endif
// window entry i: through the runs (registers)
__device__ __forceinline__ unsigned int kl_entry(const unsigned int* __restrict__ tl, int first, const int* RLO, const int* RN, int i) {
  #pragma unroll
  for (int r = 0; r < 4; ++r) {
    if (i < RN[r]) return tl[first + RLO[r] + i];
    i -= RN[r];
  }
  return 0u;
}

#ifndef RA_STAGES
#define RA_STAGES 1
#endif
#ifndef LBMIN
#define LBMIN 0
#endif
#if RA_STAGES != 1 && RA_STAGES != 2
#error "kl: RA_STAGES 1 | 2"
#endif
#if RA_STAGES == 2
// cp.async (LDGSTS, sm_80+): global → shared without registers; plain copies below sm_80 (same structure)
__device__ __forceinline__ void cpa16(void* s, const void* g) {
#if __CUDA_ARCH__ >= 800
  const unsigned int sa = (unsigned int)__cvta_generic_to_shared(s);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(__cvta_generic_to_global(g)) : "memory");
#else
  *reinterpret_cast<float4*>(s) = __ldg(reinterpret_cast<const float4*>(g));
#endif
}
__device__ __forceinline__ void cpa8(void* s, const void* g) {
#if __CUDA_ARCH__ >= 800
  const unsigned int sa = (unsigned int)__cvta_generic_to_shared(s);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(sa), "l"(__cvta_generic_to_global(g)) : "memory");
#else
  *reinterpret_cast<float2*>(s) = __ldg(reinterpret_cast<const float2*>(g));
#endif
}
__device__ __forceinline__ void cpa_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void cpa_wait_all() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}
#endif

// ---------------------------------------------------------------- layout
#define PT (TT / NXG)
#define NCL (32 / NXG)
#define CBP ((CB + 1) / 2 * 2)
#define CPW (NCL * CB)
#define NTA (TT * 32)
#define TROWS (TT * TT)
#define NCBRM ((CPW + NSEG - 1) / NSEG + 1 < NCB ? (CPW + NSEG - 1) / NSEG + 1 : NCB)
#define NE 15
#define NH 25
#define A16(x) (((x) + 15) & ~(size_t)15)
// shared memory (bytes): list | records | y | w_l | kernel values | box | e | hc | z | weights
#define OFF_SE ((size_t)0)
#define OFF_HT A16(OFF_SE + (size_t)RA_STAGES * 4 * CHK)
#define OFF_UW A16(OFF_HT + (CORR >= 3 ? (size_t)4 * 4 * NH : 0))
#define OFF_REC A16(OFF_UW + (CORR >= 3 ? (size_t)RA_STAGES * 16 * CHK : 0))
#define OFF_RY A16(OFF_REC + (size_t)RA_STAGES * CHK * 48)
#define OFF_RA A16(OFF_RY + (size_t)RA_STAGES * CHK * NCBRM * 8)
#define OFF_AX A16(OFF_RA + (size_t)CHK * NSEG * 8)
#define OFF_BOX A16(OFF_AX + (size_t)CHK * 3 * NKA * TT * 4)
#define OFF_E A16(OFF_BOX + (size_t)CHK * 4)
#define OFF_H A16(OFF_E + (CORR >= 2 ? (size_t)CHK * NE * 4 : 0))
#define OFF_Z A16(OFF_H + (CORR >= 3 ? (size_t)CHK * NH * 4 : 0))
#define OFF_W A16(OFF_Z + (size_t)CHK * (CBP / 2) * NCL * 16)
#define RA_STAGE (OFF_W + (CORR >= 1 ? (size_t)CHK * TROWS * TT * 8 : 0))
#define OBS (CPW | 1)
#define RA_PARK ((size_t)TT * TT * OBS * 8)
#define DIVUP(a, b) (((a) + (b) - 1) / (b))
#if CORR >= 3
#define RHI_ARGS , const float4* __restrict__ uw, const float* __restrict__ htab
#else
#define RHI_ARGS
#endif
#if PT * CB > 16
#error "kl: TT / NXG * CB <= 16 (accumulator registers)"
#endif
#if LBMIN > 0
#define RA_LB __launch_bounds__(NTA, LBMIN)
#else
#define RA_LB __launch_bounds__(NTA)
#endif

extern "C" __global__ void RA_LB seg_adjoint_kl(const float4* __restrict__ rec, const unsigned int* __restrict__ tl, const RaItem* __restrict__ items, const int* __restrict__ tcount,
                                                               const cx* __restrict__ atab, int ntab, float t0, float dt, const cx* __restrict__ y, int M, int c0, int ncb,
                                                               Grids g, int b, int nint, int nri, int addmask, Dims d RHI_ARGS) {
  extern __shared__ __align__(16) unsigned char smem[];
  unsigned int* SE = reinterpret_cast<unsigned int*>(smem + OFF_SE);
#if CORR >= 3
  float* HT = reinterpret_cast<float*>(smem + OFF_HT);
  float4* UW = reinterpret_cast<float4*>(smem + OFF_UW);
#endif
  float4* REC = reinterpret_cast<float4*>(smem + OFF_REC);
  cx* RY = reinterpret_cast<cx*>(smem + OFF_RY);
#if RA_STAGES == 2
  unsigned int* const SE0 = SE;
  float4* const REC0 = REC;
  cx* const RY0 = RY;
#if CORR >= 3
  float4* const UW0 = UW;
#endif
#endif
  cx* RA = reinterpret_cast<cx*>(smem + OFF_RA);
  float* AX = reinterpret_cast<float*>(smem + OFF_AX);   // [s][axis][k][cell]
  int* BOX = reinterpret_cast<int*>(smem + OFF_BOX);
#if CORR >= 2
  float* EE = reinterpret_cast<float*>(smem + OFF_E);
#endif
#if CORR >= 3
  float* HC = reinterpret_cast<float*>(smem + OFF_H);
#endif
  cx* ZS = reinterpret_cast<cx*>(smem + OFF_Z);
#if CORR >= 1
  cx* WS = reinterpret_cast<cx*>(smem + OFF_W);          // [s][row = pz · TT + py][px]
#endif
  (void)b; (void)nint; (void)nri;
  const RaItem it = items[blockIdx.x];
  if (it.split == 1) return;   // block-uniform: a later part of a split tile (its first part's block owns the tile)
  const int tcnt = it.split ? tcount[it.tile] : it.count;
  const int ntx = (d.nf0 + TT - 1) / TT, nty = (d.nf1 + TT - 1) / TT;
  const int o0 = (it.tile % ntx) * TT, o1 = ((it.tile / ntx) % nty) * TT, o2 = (it.tile / (ntx * nty)) * TT;
  const int tz = threadIdx.x >> 5, lane = threadIdx.x & 31, gx = lane / NCL, cb = lane % NCL;
  const int nchv = ncb * NSEG;
  const int chb = blockIdx.y * CPW;
  const int clo = chb / NSEG, ncbr = min(NCBRM, ncb - clo);
  // knot b's window of the tile's whole list (every part), read in place
  int RLO[4], RN[4];
  const int n = tcnt > 0 ? kl_runs(tl, it.first, tcnt, b, nint, nri, RLO, RN) : 0;
  if (tcnt == 0) { RN[0] = RN[1] = RN[2] = RN[3] = 0; RLO[0] = RLO[1] = RLO[2] = RLO[3] = 0; }
  const bool add = addmask & 1;
  if (n == 0 && add) return;   // block-uniform: nothing to add
#if CORR >= 3
  if (uw && htab) for (int q = threadIdx.x; q < 4 * NH; q += NTA) HT[q] = htab[q];
#endif
  cx acc[TT][PT][CB];
  #pragma unroll
  for (int r = 0; r < TT; ++r) {
    #pragma unroll
    for (int i = 0; i < PT; ++i) {
      #pragma unroll
      for (int k = 0; k < CB; ++k) acc[r][i][k] = cx{0.f, 0.f};
    }
  }
  constexpr int NRP = DIVUP(3 * CHK, NTA), NRY = DIVUP(CHK * NCBRM, NTA), NRE = DIVUP(CHK, NTA);
#if RA_STAGES == 2
  // cp.async pipeline (as segnufft_ra.cu RA_STAGES 2): entries of chunk k + 1 in registers at chunk k's head
  unsigned int eR[NRP], eY[NRY], eS[NRE];
  auto entries = [&](int base) {
    const int nl = min(CHK, n - base);
    #pragma unroll
    for (int r = 0; r < NRP; ++r) { const int q = threadIdx.x + r * NTA; if (q < 3 * nl) eR[r] = kl_entry(tl, it.first, RLO, RN, base + q / 3); }
    #pragma unroll
    for (int r = 0; r < NRY; ++r) { const int q = threadIdx.x + r * NTA; if (q < nl * NCBRM && q % NCBRM < ncbr) eY[r] = kl_entry(tl, it.first, RLO, RN, base + q / NCBRM); }
    #pragma unroll
    for (int r = 0; r < NRE; ++r) { const int q = threadIdx.x + r * NTA; if (q < nl) eS[r] = kl_entry(tl, it.first, RLO, RN, base + q); }
  };
  auto issue = [&](int base, int buf) {
    const int nl = min(CHK, n - base);
    float4* Rb = REC0 + (size_t)buf * 3 * CHK;
    cx* Yb = RY0 + (size_t)buf * CHK * NCBRM;
    #pragma unroll
    for (int r = 0; r < NRP; ++r) { const int q = threadIdx.x + r * NTA; if (q < 3 * nl) cpa16(Rb + q, rec + (size_t)SAMPLE_OF(eR[r]) * 3 + q % 3); }
    #pragma unroll
    for (int r = 0; r < NRY; ++r) {
      const int q = threadIdx.x + r * NTA;
      if (q < nl * NCBRM && q % NCBRM < ncbr) cpa8(Yb + q, y + (size_t)(c0 + clo + q % NCBRM) * M + SAMPLE_OF(eY[r]));
      else if (q < nl * NCBRM) Yb[q] = cx{0.f, 0.f};   // coils past the batch: never read; zero as the register path
    }
    #pragma unroll
    for (int r = 0; r < NRE; ++r) {
      const int q = threadIdx.x + r * NTA;
      if (q < nl) {
        SE0[(size_t)buf * CHK + q] = eS[r];
#if CORR >= 3
        if (uw && htab) cpa16(UW0 + (size_t)buf * CHK + q, uw + SAMPLE_OF(eS[r]));
        else UW0[(size_t)buf * CHK + q] = make_float4(0.f, 0.f, 0.f, 0.f);
#endif
      }
    }
    cpa_commit();
  };
  if (n > 0) {
    entries(0);
    issue(0, 0);
    if (CHK < n) entries(CHK);
  }
  int kch = 0;
  for (int base = 0; base < n; base += CHK, ++kch) {
    const int nl = min(CHK, n - base);
    const int buf = kch & 1;
    cpa_wait_all();    // this thread's copies of chunk k
    __syncthreads();   // everyone's copies (and htab) visible; the previous chunk's MAC is done
    if (base + CHK < n) {
      issue(base + CHK, buf ^ 1);   // the other buffer was last read before the previous chunk's pre-MAC barrier
      if (base + 2 * CHK < n) entries(base + 2 * CHK);
    }
    SE = SE0 + (size_t)buf * CHK;
    REC = REC0 + (size_t)buf * 3 * CHK;
    RY = RY0 + (size_t)buf * CHK * NCBRM;
#if CORR >= 3
    UW = UW0 + (size_t)buf * CHK;
#endif
#else
  // register prefetch of one chunk's records and coil values
  float4 pfr[NRP]; cx pfy[NRY]; unsigned int pfe[NRE];
#if CORR >= 3
  float4 pfu[NRE];
#endif
  auto prefetch = [&](int base) {
    const int nl = min(CHK, n - base);
    #pragma unroll
    for (int r = 0; r < NRE; ++r) {
      const int q = threadIdx.x + r * NTA;
      if (q < nl) {
        pfe[r] = kl_entry(tl, it.first, RLO, RN, base + q);
#if CORR >= 3
        pfu[r] = (uw && htab) ? uw[SAMPLE_OF(pfe[r])] : make_float4(0.f, 0.f, 0.f, 0.f);
#endif
      }
    }
    #pragma unroll
    for (int r = 0; r < NRP; ++r) {
      const int q = threadIdx.x + r * NTA;
      if (q < 3 * nl) pfr[r] = rec[(size_t)SAMPLE_OF(kl_entry(tl, it.first, RLO, RN, base + q / 3)) * 3 + q % 3];
    }
    #pragma unroll
    for (int r = 0; r < NRY; ++r) {
      const int q = threadIdx.x + r * NTA;
      pfy[r] = cx{0.f, 0.f};
      if (q < nl * NCBRM && q % NCBRM < ncbr) pfy[r] = y[(size_t)(c0 + clo + q % NCBRM) * M + SAMPLE_OF(kl_entry(tl, it.first, RLO, RN, base + q / NCBRM))];
    }
  };
  if (n > 0) prefetch(0);
  for (int base = 0; base < n; base += CHK) {
    const int nl = min(CHK, n - base);
    __syncthreads();   // previous chunk consumed
    #pragma unroll
    for (int r = 0; r < NRP; ++r) { const int q = threadIdx.x + r * NTA; if (q < 3 * nl) REC[q] = pfr[r]; }
    #pragma unroll
    for (int r = 0; r < NRY; ++r) { const int q = threadIdx.x + r * NTA; if (q < nl * NCBRM) RY[q] = pfy[r]; }
    #pragma unroll
    for (int r = 0; r < NRE; ++r) {
      const int q = threadIdx.x + r * NTA;
      if (q < nl) {
        SE[q] = pfe[r];
#if CORR >= 3
        UW[q] = pfu[r];
#endif
      }
    }
    __syncthreads();
    if (base + CHK < n) prefetch(base + CHK);
#endif
    // ---- stage A: kernel values on the tile's cells (as stage_kv), coefficients w_l, support boxes, e, hc
    for (int q = threadIdx.x; q < nl * 3 * TT; q += NTA) {
      const int s = q / (3 * TT), a = (q / TT) % 3, cell = q % TT;
      const float4 r = REC[s * 3];
      const float sc = a == 0 ? r.x : (a == 1 ? r.y : r.z);
      const int og = a == 0 ? o0 : (a == 1 ? o1 : o2), nn = a == 0 ? d.nf0 : (a == 1 ? d.nf1 : d.nf2);
      const float z = wrapdiff((float)(og + cell) - sc, nn) * (2.f / (float)W);
      float o[5];
      kern_derivs(z, o);
      #pragma unroll
      for (int k = 0; k < NKA; ++k) AX[((s * 3 + a) * NKA + k) * TT + cell] = o[k];
    }
    for (int q = threadIdx.x; q < nl * NSEG; q += NTA) {
      const int sm = q / NSEG, l = q % NSEG;
      float u = (REC[sm * 3].w - t0) / dt;
      u = fminf(fmaxf(u, 0.f), (float)(ntab - 1) - 1e-3f);
      const int i = (int)u;
      const float fr = u - (float)i;
      const cx pa = atab[i * NSEG + l], pb = atab[(i + 1) * NSEG + l];
      const cx a = cx{fmaf(fr, pb.x - pa.x, pa.x), fmaf(fr, pb.y - pa.y, pa.y)};
      float s0, c1; sincosf(REC[sm * 3 + 1].x, &s0, &c1);
      const cx v = cmulc(a, cx{c1, s0});
#if KS > 0
      const float beta = ra_beta(SE[sm], REC[sm * 3 + 2].w, b, nint);
#else
      const float beta = 1.f;
#endif
      RA[q] = cx{beta * v.x, beta * v.y};
    }
    for (int q = threadIdx.x; q < nl; q += NTA) {
      const float4 r = REC[q * 3];
      int bx = 0;
      #pragma unroll
      for (int a = 0; a < 3; ++a) {
        const float sc = a == 0 ? r.x : (a == 1 ? r.y : r.z);
        const int og = a == 0 ? o0 : (a == 1 ? o1 : o2), nn = a == 0 ? d.nf0 : (a == 1 ? d.nf1 : d.nf2);
        const float dd = wrapdiff((float)og - sc, nn);
        const int lo = max(0, (int)floorf(-0.5f * W - dd) + 1), hi = min(TT - 1, (int)ceilf(0.5f * W - dd) - 1);
        bx |= (lo | (hi << 4)) << (8 * a);
      }
      BOX[q] = bx;
#if CORR >= 2
      const float4 r1 = REC[q * 3 + 1], r2 = REC[q * 3 + 2];
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
      for (int k = 0; k < NE; ++k) EE[q * NE + k] = e[k];
#endif
    }
#if CORR >= 3
    for (int q = threadIdx.x; q < nl * NH; q += NTA) {   // degree-3/4 coefficients (stage_hc_k)
      const int s = q / NH, a = q % NH;
      float v = 0.f;
      if (uw && htab) {
        const float4 u = UW[s];
        v = fmaf(u.x, HT[a], fmaf(u.y, HT[NH + a], fmaf(u.z, HT[2 * NH + a], u.w * HT[3 * NH + a])));
      }
      HC[q] = v;
    }
#endif
    __syncthreads();
    // ---- stage B: z = y_c w_l for the block's channels (zero past the batch), one thread per channel (index math out
    // of the sample loop); layout [s][pair kp][cb][2]: a lane's float4 loads of one pair are consecutive
    for (int k = threadIdx.x; k < CPW; k += NTA) {
      const int ch = chb + k, kk = k % CB;
      const bool ok = ch < nchv;
      const int coil = ok ? ch / NSEG : clo, l = ok ? ch % NSEG : 0;
      const cx* ap = RA + l;
      const cx* yp = RY + (coil - clo);
      cx* zp = ZS + ((kk / 2) * NCL + k / CB) * 2 + (kk & 1);
      #pragma unroll 4
      for (int s = 0; s < nl; ++s) zp[(size_t)s * (CBP / 2) * NCL * 2] = ok ? cmul(yp[s * NCBRM], ap[s * NSEG]) : cx{0.f, 0.f};
    }
#if CB % 2 == 1
    // odd CB: the pad channel of the last pair is zero
    for (int q = threadIdx.x; q < nl * NCL; q += NTA) ZS[((size_t)(q / NCL) * (CBP / 2) * NCL + (CBP / 2 - 1) * NCL + q % NCL) * 2 + 1] = cx{0.f, 0.f};
#endif
#if CORR >= 1
    // complex weights of the covered (sample, tile row) pairs: stage_weights (CORR ≤ 2) / stage_weights_hk (CORR 3)
    for (int q = threadIdx.x; q < nl * TROWS; q += NTA) {
      const int s = q / TROWS, row = q % TROWS, py = row % TT, pz = row / TT;
      const int bx = BOX[s];
      if (py < ((bx >> 8) & 15) || py > ((bx >> 12) & 15) || pz < ((bx >> 16) & 15) || pz > ((bx >> 20) & 15)) continue;
      const float* ax = AX + (size_t)s * 3 * NKA * TT;
      float R[NKA], Q[NKA];
      #pragma unroll
      for (int a = 0; a < NKA; ++a) { R[a] = 0.f; Q[a] = 0.f; }
      float wy[NKA], wz[NKA];
      #pragma unroll
      for (int k = 0; k < NKA; ++k) { wy[k] = ax[(1 * NKA + k) * TT + py]; wz[k] = ax[(2 * NKA + k) * TT + pz]; }
      R[0] = wy[0] * wz[0];
      const float4 r1 = REC[s * 3 + 1], r2 = REC[s * 3 + 2];   // [xx, yy, zz] = r1.yzw, [xy, xz, yz] = r2.xyz
      Q[0] = r1.z * wy[2] * wz[0] + r1.w * wy[0] * wz[2] + r2.z * wy[1] * wz[1];
      Q[1] = r2.x * wy[1] * wz[0] + r2.y * wy[0] * wz[1];
      Q[2] = r1.y * wy[0] * wz[0];
#if CORR >= 2
      {
        int idx = 0;
        #pragma unroll
        for (int a = 0; a <= 4; ++a) {
          #pragma unroll
          for (int bb = 0; bb <= 4 - a; ++bb) { R[a] += EE[s * NE + idx] * wy[bb] * wz[4 - a - bb]; ++idx; }
        }
      }
#endif
#if CORR >= 3
      {
        const float* hc = HC + s * NH;
        int idx = 0;
        #pragma unroll
        for (int a = 0; a <= 3; ++a) {   // degree 3: + c ∂^α φ (real)
          #pragma unroll
          for (int bb = 0; bb <= 3 - a; ++bb) { R[a] = fmaf(hc[idx], wy[bb] * wz[3 - a - bb], R[a]); ++idx; }
        }
        #pragma unroll
        for (int a = 0; a <= 4; ++a) {   // degree 4: −i c ∂^α φ (imaginary)
          #pragma unroll
          for (int bb = 0; bb <= 4 - a; ++bb) { Q[a] = fmaf(-hc[idx], wy[bb] * wz[4 - a - bb], Q[a]); ++idx; }
        }
      }
#endif
      cx* wp = WS + ((size_t)s * TROWS + row) * TT;
      #pragma unroll
      for (int px = 0; px < TT; ++px) {
        float wr = 0.f, wi = 0.f;
        #pragma unroll
        for (int a = 0; a < NKA; ++a) { const float xa = ax[(0 * NKA + a) * TT + px]; wr = fmaf(xa, R[a], wr); wi = fmaf(xa, Q[a], wi); }
        wp[px] = cx{wr, wi};
      }
    }
#endif
    __syncthreads();
    // ---- MAC: the warp's plane tz, covered rows only
    for (int s = 0; s < nl; ++s) {
      const int bx = BOX[s];
      if (tz < ((bx >> 16) & 15) || tz > ((bx >> 20) & 15)) continue;   // warp-uniform
      const int ylo = (bx >> 8) & 15, yhi = (bx >> 12) & 15, xlo = bx & 15, xhi = (bx >> 4) & 15;
      (void)xlo; (void)xhi;
      cx zv[CBP];
      {
        const float4* zp = reinterpret_cast<const float4*>(ZS) + (size_t)s * (CBP / 2) * NCL + cb;
        #pragma unroll
        for (int k = 0; k < CBP / 2; ++k) { const float4 u = zp[k * NCL]; zv[2 * k] = cx{u.x, u.y}; zv[2 * k + 1] = cx{u.z, u.w}; }
      }
#if CORR == 0
      const float* ax = AX + (size_t)s * 3 * TT;
      const float fz = ax[2 * TT + tz];
      float wx[PT];
      #pragma unroll
      for (int i = 0; i < PT; ++i) wx[i] = ax[gx * PT + i];
      const float4 kyv = *reinterpret_cast<const float4*>(ax + TT);
      const float ky[4] = {kyv.x, kyv.y, kyv.z, kyv.w};
      #pragma unroll
      for (int r = 0; r < TT; ++r) {
        if (r < ylo || r > yhi) continue;   // warp-uniform: exact rows
        const float wr = ky[r] * fz;
        #pragma unroll
        for (int i = 0; i < PT; ++i) {
          if (NXG == 1 && (i < xlo || i > xhi)) continue;   // NXG 1: warp-uniform exact x
          const float w = wx[i] * wr;
          #pragma unroll
          for (int k = 0; k < CB; ++k) { acc[r][i][k].x = fmaf(w, zv[k].x, acc[r][i][k].x); acc[r][i][k].y = fmaf(w, zv[k].y, acc[r][i][k].y); }
        }
      }
#else
      #pragma unroll
      for (int r = 0; r < TT; ++r) {
        if (r < ylo || r > yhi) continue;
        const cx* wp = WS + ((size_t)s * TROWS + tz * TT + r) * TT + gx * PT;
        cx wv[PT];
#if PT % 2 == 0
        #pragma unroll
        for (int i = 0; i < PT; i += 2) { const float4 u = *reinterpret_cast<const float4*>(wp + i); wv[i] = cx{u.x, u.y}; wv[i + 1] = cx{u.z, u.w}; }
#else
        #pragma unroll
        for (int i = 0; i < PT; ++i) wv[i] = wp[i];
#endif
        #pragma unroll
        for (int i = 0; i < PT; ++i) {
          if (NXG == 1 && (i < xlo || i > xhi)) continue;
          #pragma unroll
          for (int k = 0; k < CB; ++k) {   // conj(w) z
            acc[r][i][k].x = fmaf(wv[i].x, zv[k].x, acc[r][i][k].x); acc[r][i][k].x = fmaf(wv[i].y, zv[k].y, acc[r][i][k].x);
            acc[r][i][k].y = fmaf(wv[i].x, zv[k].y, acc[r][i][k].y); acc[r][i][k].y = fmaf(-wv[i].y, zv[k].x, acc[r][i][k].y);
          }
        }
      }
#endif
    }
  }
  // ---- output: per warp, row by row, park the TT × CPW values in shared memory and write runs of CPW channels
  __syncthreads();
  cx* ob = reinterpret_cast<cx*>(smem) + (size_t)tz * TT * OBS;
  const int gz = o2 + tz;
  const int run = min(CPW, nchv - chb);
  cx* gbase = g.p[0];
  #pragma unroll
  for (int r = 0; r < TT; ++r) {
    #pragma unroll
    for (int i = 0; i < PT; ++i) {
      #pragma unroll
      for (int k = 0; k < CB; ++k) ob[(gx * PT + i) * OBS + cb * CB + k] = acc[r][i][k];
    }
    __syncwarp();
    const int gy = o1 + r;
    if (gy < d.nf1 && gz < d.nf2) {
      for (int q = lane; q < TT * run; q += 32) {
        const int t = q / run, k = q % run, gxx = o0 + t;
        if (gxx >= d.nf0) continue;
        const cx v = ob[t * OBS + k];
        cx* gp = gbase + (((size_t)gz * d.nf1 + gy) * d.nf0 + gxx) * NCH + chb + k;
        if (add) { const cx o = *gp; *gp = cx{o.x + v.x, o.y + v.y}; }
        else *gp = v;
      }
    }
    __syncwarp();
  }
}
