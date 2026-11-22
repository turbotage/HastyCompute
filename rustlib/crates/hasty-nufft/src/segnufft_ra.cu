// Adjoint variant `ra` of the segmented NUFFT: design 7 of research/sample_kernels.md (row-blocked output-tile
// adjoint; §15 the prototype `pg_adj_ra` in proto_gather.cu, §16 this engine kernel). Standalone NVRTC module (own
// helpers, no Stage struct), on the engine's own TT = 4 tile lists, work items and split flags: no new tables.
// Defines from the host (`segnufft/ra.rs`): W, BETA, NSEG, CORR (0..3), NCB (coils per batch: grid channel stride
// NCH = NCB · NSEG), LISTMAX (longest item list), NXG (x groups per warp: 1 | 2 | 4), CB (channels per lane), CHK
// (samples staged per chunk), KS (knot support; 0 = static image), RDEG (2-D knots; −1: 1-D), optionally KER_KB.
//
//   G_{b,c,l}(p) (=)  Σ_j conj(w_j(p)) β_b(t_j) conj(a_l(τ_j)) e^{iφ0_j} y_c(j)       (the knot / tile adjoint)
//
// Block = (work item of a TT³ tile, channel-block row blockIdx.y); warp w = tile plane tz = w (TZ = TT warps); lane =
// (x-group g < NXG, channel block cb < NCL = 32 / NXG); the thread holds acc[TT rows][PT = TT / NXG points][CB
// channels] in registers (channels blockIdx.y · CPW + cb · CB + k). Per chunk of CHK samples the block stages the
// records (prefetched into registers one chunk ahead), the kernel values on the tile's cells (wrapdiff, as the tile
// kernels), the support box, the coefficient w_l = β conj(a_l) e^{iφ0} per (sample, segment), z = y_c w_l per (sample,
// channel) in a conflict-free layout [s][pair][cb][2], and at orders ≥ 1 the complex weights of every covered (sample,
// tile row) with exactly the terms of segnufft.cu stage_weights / segnufft_knot.cu stage_weights_hk. The MAC skips
// samples whose z box misses the warp's plane and rows outside the y box (warp-uniform; outside the box the weights
// are exact zeros). Output: per warp row by row, parked in shared memory, written as runs of CPW channels per point:
// split items add atomically, unsplit items store (bit 0 of addmask: read-modify-write), so the contract of
// HASTY_SEG_ZERO=tiles holds (only split tiles are zeroed before the scatter). An item with nothing in the knot window
// stores zeros (store mode) and writes nothing in add mode or when split.
//
// Phase-2 variants (research/sample_kernels.md §19, research/arch_targets.md §4 items 2 / 3); defaults = the kernel above:
//   RA_INPLACE 1: the item's list (knot engines: knot b's window, ≤ 1 / 2 / 4 contiguous runs found by binary search)
//                 is read in place from global memory instead of being copied into shared memory (no LISTMAX buffer);
//                 the chunk's entries, uw (order 3) and htab live in shared memory as in segnufft_kl_adj.cu.
//   RA_STAGES 2:  (needs RA_INPLACE) records, coil values and uw of chunk k + 1 are copied by cp.async (sm_80+; plain
//                 loads / stores below sm_80) into the second of two shared buffers while chunk k is staged and
//                 MACed; only the list entries are prefetched into registers (one chunk further ahead). One barrier
//                 per chunk less than the register prefetch.
//   LBMIN m > 0:  __launch_bounds__(NTA, m) (per-shape candidate; the host screens spills).
//   RA_STAGES 3:  RA_STAGES 2 plus dynamic warp-unit staging: each staging loop of a chunk is cut into units of 32
//                 items that warps take round robin (RA_GRAB 1: from a shared counter), heavy units first, z per
//                 channel in RA_ZSPL sample slices; the MAC loads the next sample's box ahead. Same items, same MAC
//                 order: bitwise.
//   RA_STAGES 4:  RA_STAGES 3 with the MAC of chunk k − 1 and staging phase 1 of chunk k between the same two barriers
//                 (orders ≥ 1: the previous chunk's boxes in registers, read by shuffle; order 0: box / kernel-value
//                 buffers doubled): two barriers per chunk instead of three.
//   RA_WS p > 0:  (needs RA_STAGES 2) warp-specialised (research/sample_kernels.md §22): p producer warps (after the TT
//                 MAC warps) run the whole staging of chunk k + 1 (cp.async raw copies, kernel values, boxes, w_l, z,
//                 weights) while the MAC warps work on chunk k; the MAC inputs (boxes, z, weights; order 0: kernel
//                 values) are double-buffered; hand-over by named hardware barriers (bar.arrive / bar.sync: full / free
//                 per buffer parity), the producers' own phases by a producer-only named barrier. No block barrier in
//                 the chunk loop. Same items, same MAC order: bitwise.
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
#error "ra: TT = 4 (the engine's tile lists)"
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
// segnufft_knot.cu knot_window2d: ≤ 4 contiguous key runs of the sorted list, compacted into sl
__device__ __forceinline__ int ra_window(const unsigned int* __restrict__ tl, int first, int count, int b, int nint, int nri, int* sl, int nthr) {
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
  int base = 0;
  for (int r = 0; r < 4; ++r) {
    const int lo = rb[r][0], n = max(0, rb[r][1] - rb[r][0]);
    for (int q = threadIdx.x; q < n; q += nthr) sl[base + q] = (int)tl[first + lo + q];
    base += n;
  }
  __syncthreads();
  return base;
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
// segnufft_knot.cu knot_window: the entries whose interval lies in knot b's window (≤ 2 contiguous runs)
__device__ __forceinline__ int ra_window(const unsigned int* __restrict__ tl, int first, int count, int b, int nint, int nri, int* sl, int nthr) {
  (void)nri;
  const int khi = ((b - KOFF) % nint + nint) % nint, klo = ((b - KOFF - KS + 1) % nint + nint) % nint;
  int clo = 0, chi = 0;
  for (int base = 0; base < count; base += nthr) {
    const int q = base + threadIdx.x;
    const int key = q < count ? (int)(tl[first + q] >> KEY_SHIFT) : 0x7fffffff;
    clo += __syncthreads_count(key < klo);
    chi += __syncthreads_count(key <= khi);
  }
  const bool wrapw = klo > khi || KS >= nint;
  const int nwin = wrapw ? (KS >= nint ? count : chi + (count - clo)) : chi - clo;
  for (int q = threadIdx.x; q < nwin; q += nthr) {
    const int pos = KS >= nint ? q : (wrapw ? (q < chi ? q : clo + (q - chi)) : clo + q);
    sl[q] = (int)tl[first + pos];
  }
  __syncthreads();
  return nwin;
}
__device__ __forceinline__ float ra_beta(unsigned int e, float w11, int b, int nint) {
  const int key = (int)(e >> KEY_SHIFT);
  return bspline_k(((b - KOFF - key) % nint + nint) % nint, w11);
}
#endif
#define SAMPLE_OF(e) ((e) & KEY_MASK)
#else
#define SAMPLE_OF(e) (e)
#endif

// ---------------------------------------------------------------- phase-2 variants: in-place lists, cp.async, bounds
#ifndef RA_INPLACE
#define RA_INPLACE 0
#endif
#ifndef RA_STAGES
#define RA_STAGES 1
#endif
#ifndef LBMIN
#define LBMIN 0
#endif
#if RA_STAGES < 1 || RA_STAGES > 4
#error "ra: RA_STAGES 1 | 2 | 3 | 4"
#endif
#if RA_STAGES >= 2 && !RA_INPLACE
#error "ra: RA_STAGES 2 / 3 / 4 need RA_INPLACE 1"
#endif
// stage buffers of the raw inputs: 1 (register prefetch) or 2 (cp.async double buffer; RA_STAGES 2 / 3 / 4)
#define RA_NBUF (RA_STAGES >= 2 ? 2 : 1)
#ifndef RA_WS
#define RA_WS 0
#endif
#if RA_WS && RA_STAGES != 2
#error "ra: RA_WS needs RA_STAGES 2"
#endif
// staging thread index / count: all threads, or (RA_WS) the producer warps after the TT MAC warps
#if RA_WS
#define RA_TID ((int)threadIdx.x - NTA)
#define RA_NTS (32 * RA_WS)
#define RA_PSYNC() ra_nb_sync(1, RA_NTS)
#define RB_F 2                          // + parity: MAC inputs full (producers arrive, MAC warps sync)
#define RB_E 4                          // + parity: MAC inputs free (MAC warps arrive, producers sync)
__device__ __forceinline__ void ra_nb_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void ra_nb_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
#else
#define RA_TID threadIdx.x
#define RA_NTS NTA
#define RA_PSYNC() __syncthreads()
#endif
#ifndef RA_GRAB
#define RA_GRAB 0   // RA_STAGES 3 / 4: 0 = warp round robin over the units, 1 = shared counter (dynamic)
#endif
#ifndef RA_ZSPL
#define RA_ZSPL 2   // RA_STAGES 3 / 4: sample slices of the per-channel z tasks
#endif
#if RA_INPLACE
// runs of the item's list that make up the (window of the) list: static 1, 1-D knots 2, 2-D knots 4
#if KS == 0
#define NRUN 1
#elif RDEG >= 0
#define NRUN 4
#else
#define NRUN 2
#endif
// The list of the item tl[first, first + count) as ≤ NRUN runs [RLO[r], RLO[r] + RN[r]) (positions relative to first);
// returns the total. Knot engines: knot b's window, by binary search of the key-sorted list (the same entries in the
// same order as ra_window's compaction). Relies on every item's list being sorted by key, as segnufft.rs tile_lists
// builds them (the 1-D ra_window scan does not need it; kl does). Contains a barrier on knot engines: call from all
// threads (block-uniform).
__device__ __forceinline__ int ra_runs(const unsigned int* __restrict__ tl, int first, int count, int b, int nint, int nri, int* RLO, int* RN) {
#if KS == 0
  (void)tl; (void)first; (void)b; (void)nint; (void)nri;
  RLO[0] = 0; RN[0] = count;
  return count;
#elif RDEG >= 0
  __shared__ int rq[8][2];
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
      rq[q >> 1][q & 1] = lower_key(tl, first, count, key);
    } else {
      rq[q >> 1][q & 1] = 0;
    }
  }
  __syncthreads();
  int tot = 0;
  #pragma unroll
  for (int r = 0; r < 4; ++r) { RLO[r] = rq[r][0]; RN[r] = max(0, rq[r][1] - rq[r][0]); tot += RN[r]; }
  return tot;
#else
  __shared__ int rq[2];
  (void)nri;
  const int khi = ((b - KOFF) % nint + nint) % nint, klo = ((b - KOFF - KS + 1) % nint + nint) % nint;
  if (threadIdx.x < 2) {
    const int key = threadIdx.x == 0 ? klo : khi + 1;
    int lo = 0, hi = count;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if ((int)(tl[first + mid] >> KEY_SHIFT) < key) lo = mid + 1; else hi = mid;
    }
    rq[threadIdx.x] = lo;
  }
  __syncthreads();
  const int clo = rq[0], chi = rq[1];   // entries with key < klo, key ≤ khi
  if (KS >= nint) { RLO[0] = 0; RN[0] = count; RLO[1] = 0; RN[1] = 0; }
  else if (klo > khi) { RLO[0] = 0; RN[0] = chi; RLO[1] = clo; RN[1] = count - clo; }   // wrapped window
  else { RLO[0] = clo; RN[0] = chi - clo; RLO[1] = 0; RN[1] = 0; }
  return RN[0] + RN[1];
#endif
}
// list entry i (0 ≤ i < total) through the runs (registers)
__device__ __forceinline__ unsigned int ra_entry(const unsigned int* __restrict__ tl, int first, const int* RLO, const int* RN, int i) {
  #pragma unroll
  for (int r = 0; r < NRUN; ++r) {
    if (i < RN[r]) return tl[first + RLO[r] + i];
    i -= RN[r];
  }
  return 0u;
}
#endif
#if RA_STAGES >= 2
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
// wait for every outstanding group of this thread (the barrier after it publishes all threads' copies)
__device__ __forceinline__ void cpa_wait_all() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
}
#endif
#if RA_WS
// 2 blocks per SM: ≤ 128 registers at 4 producers (16 warps, the 4-warp kernel's 4 blocks); without a block minimum
// ptxas takes ≈ 160 registers (3 blocks of the 2-producer shape spill at 96)
#define RA_LB __launch_bounds__(NTA + 32 * RA_WS, LBMIN > 0 ? LBMIN : 2)
#elif LBMIN > 0
#define RA_LB __launch_bounds__(NTA, LBMIN)
#else
#define RA_LB __launch_bounds__(NTA)
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
// (RA_INPLACE: entries [stages] | htab | uw [stages] | records [stages] | y [stages] | w_l | ...)
#if RA_INPLACE
#define OFF_SE ((size_t)0)
#define OFF_HT A16(OFF_SE + (size_t)RA_NBUF * 4 * CHK)
#define OFF_UW A16(OFF_HT + (CORR >= 3 ? (size_t)4 * 4 * NH : 0))
#define OFF_REC A16(OFF_UW + (CORR >= 3 ? (size_t)RA_NBUF * 16 * CHK : 0))
#define OFF_RY A16(OFF_REC + (size_t)RA_NBUF * CHK * 48)
#define OFF_RA A16(OFF_RY + (size_t)RA_NBUF * CHK * NCBRM * 8)
#else
#define OFF_SL ((size_t)0)
#define OFF_REC A16(OFF_SL + (size_t)4 * LISTMAX)
#define OFF_RY A16(OFF_REC + (size_t)CHK * 48)
#define OFF_RA A16(OFF_RY + (size_t)CHK * NCBRM * 8)
#endif
#define OFF_AX A16(OFF_RA + (size_t)CHK * NSEG * 8)
#define OFF_BOX A16(OFF_AX + (size_t)CHK * 3 * NKA * TT * 4)
#define OFF_E A16(OFF_BOX + (size_t)CHK * 4)
#define OFF_H A16(OFF_E + (CORR >= 2 ? (size_t)CHK * NE * 4 : 0))
#define OFF_Z A16(OFF_H + (CORR >= 3 ? (size_t)CHK * NH * 4 : 0))
#define OFF_W A16(OFF_Z + (size_t)CHK * (CBP / 2) * NCL * 16)
#if RA_STAGES == 4 && CORR == 0
// fused pipeline at order 0 (the MAC reads kernel values): second support-box and kernel-value buffers (orders ≥ 1
// keep the previous chunk's boxes in registers)
#define OFF_BOX2 A16(OFF_W + (CORR >= 1 ? (size_t)CHK * TROWS * TT * 8 : 0))
#define OFF_AX2 A16(OFF_BOX2 + (size_t)CHK * 4)
#define RA_STAGE (OFF_AX2 + (size_t)CHK * 3 * NKA * TT * 4)
#elif RA_WS
// warp-specialised: second MAC-input buffers (box, z, weights; order 0: kernel values)
#define OFF_BOXB A16(OFF_W + (CORR >= 1 ? (size_t)CHK * TROWS * TT * 8 : 0))
#define OFF_ZB A16(OFF_BOXB + (size_t)CHK * 4)
#define OFF_WB A16(OFF_ZB + (size_t)CHK * (CBP / 2) * NCL * 16)
#define OFF_AXB A16(OFF_WB + (CORR >= 1 ? (size_t)CHK * TROWS * TT * 8 : 0))
#define RA_STAGE (OFF_AXB + (CORR == 0 ? (size_t)CHK * 3 * NKA * TT * 4 : 0))
#else
#define RA_STAGE (OFF_W + (CORR >= 1 ? (size_t)CHK * TROWS * TT * 8 : 0))
#endif
#define OBS (CPW | 1)
#define RA_PARK ((size_t)TT * TT * OBS * 8)
#define DIVUP(a, b) (((a) + (b) - 1) / (b))
#if CORR >= 3
#define RHI_ARGS , const float4* __restrict__ uw, const float* __restrict__ htab
#else
#define RHI_ARGS
#endif
#if PT * CB > 16
#error "ra: TT / NXG * CB <= 16 (accumulator registers)"
#endif

extern "C" __global__ void RA_LB seg_adjoint_ra(const float4* __restrict__ rec, const unsigned int* __restrict__ tl, const RaItem* __restrict__ items,
                                                               const cx* __restrict__ atab, int ntab, float t0, float dt, const cx* __restrict__ y, int M, int c0, int ncb,
                                                               Grids g, int b, int nint, int nri, int addmask, Dims d RHI_ARGS) {
  extern __shared__ __align__(16) unsigned char smem[];
#if RA_INPLACE
  unsigned int* SE0 = reinterpret_cast<unsigned int*>(smem + OFF_SE);
  float4* REC0 = reinterpret_cast<float4*>(smem + OFF_REC);
  cx* RY0 = reinterpret_cast<cx*>(smem + OFF_RY);
  unsigned int* SE = SE0;
  float4* REC = REC0;
  cx* RY = RY0;
#if CORR >= 3
  const float* HT = reinterpret_cast<const float*>(smem + OFF_HT);
  float4* UW0 = reinterpret_cast<float4*>(smem + OFF_UW);
  float4* UW = UW0;
#endif
#else
  int* sl = reinterpret_cast<int*>(smem + OFF_SL);
  float4* REC = reinterpret_cast<float4*>(smem + OFF_REC);
  cx* RY = reinterpret_cast<cx*>(smem + OFF_RY);
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
  const int ntx = (d.nf0 + TT - 1) / TT, nty = (d.nf1 + TT - 1) / TT;
  const int o0 = (it.tile % ntx) * TT, o1 = ((it.tile / ntx) % nty) * TT, o2 = (it.tile / (ntx * nty)) * TT;
  const int tz = threadIdx.x >> 5, lane = threadIdx.x & 31, gx = lane / NCL, cb = lane % NCL;
  const int nchv = ncb * NSEG;
  const int chb = blockIdx.y * CPW;
  const int clo = chb / NSEG, ncbr = min(NCBRM, ncb - clo);
  // the item's sample list (knot engines: the entries of knot b's window)
#if RA_INPLACE
  int RLO[NRUN], RN[NRUN];
  #pragma unroll
  for (int r = 0; r < NRUN; ++r) { RLO[r] = 0; RN[r] = 0; }
  const int n = it.count > 0 ? ra_runs(tl, it.first, it.count, b, nint, nri, RLO, RN) : 0;
#if CORR >= 3
  // htab once per block (published by the first chunk-head barrier; unused when n == 0)
#if RA_WS
  if (threadIdx.x >= NTA)   // producers only (the producer barrier publishes it)
  if (uw && htab) for (int q = RA_TID; q < 4 * NH; q += RA_NTS) reinterpret_cast<float*>(smem + OFF_HT)[q] = htab[q];
#else
  if (uw && htab) for (int q = threadIdx.x; q < 4 * NH; q += NTA) reinterpret_cast<float*>(smem + OFF_HT)[q] = htab[q];
#endif
#endif
#elif KS > 0
  const int n = it.count > 0 ? ra_window(tl, it.first, it.count, b, nint, nri, sl, NTA) : 0;
#else
  const int n = it.count;
  for (int q = threadIdx.x; q < n; q += NTA) sl[q] = (int)tl[it.first + q];
  __syncthreads();
#endif
  const bool add = addmask & 1, split = it.split != 0;
  if (n == 0 && (add || split)) return;   // block-uniform: nothing to add
  cx acc[TT][PT][CB];
  #pragma unroll
  for (int r = 0; r < TT; ++r) {
    #pragma unroll
    for (int i = 0; i < PT; ++i) {
      #pragma unroll
      for (int k = 0; k < CB; ++k) acc[r][i][k] = cx{0.f, 0.f};
    }
  }
  constexpr int NRP = DIVUP(3 * CHK, RA_NTS), NRY = DIVUP(CHK * NCBRM, RA_NTS), NRE = DIVUP(CHK, RA_NTS);
#if RA_STAGES >= 2
  // cp.async pipeline: the list entries of chunk k + 1 are in registers (eR: record slots, eY: coil-value slots, eS:
  // entry slots) when chunk k's head issues its copies; then the entries of chunk k + 2 are loaded
  unsigned int eR[NRP], eY[NRY], eS[NRE];
  auto entries = [&](int base) {
    const int nl = min(CHK, n - base);
    #pragma unroll
    for (int r = 0; r < NRP; ++r) { const int q = RA_TID + r * RA_NTS; if (q < 3 * nl) eR[r] = ra_entry(tl, it.first, RLO, RN, base + q / 3); }
    #pragma unroll
    for (int r = 0; r < NRY; ++r) { const int q = RA_TID + r * RA_NTS; if (q < nl * NCBRM && q % NCBRM < ncbr) eY[r] = ra_entry(tl, it.first, RLO, RN, base + q / NCBRM); }
    #pragma unroll
    for (int r = 0; r < NRE; ++r) { const int q = RA_TID + r * RA_NTS; if (q < nl) eS[r] = ra_entry(tl, it.first, RLO, RN, base + q); }
  };
  // copies of the chunk at `base` (entries in registers) into stage buffer `buf`
  auto issue = [&](int base, int buf) {
    const int nl = min(CHK, n - base);
    float4* Rb = REC0 + (size_t)buf * 3 * CHK;
    cx* Yb = RY0 + (size_t)buf * CHK * NCBRM;
    #pragma unroll
    for (int r = 0; r < NRP; ++r) { const int q = RA_TID + r * RA_NTS; if (q < 3 * nl) cpa16(Rb + q, rec + (size_t)SAMPLE_OF(eR[r]) * 3 + q % 3); }
    #pragma unroll
    for (int r = 0; r < NRY; ++r) {
      const int q = RA_TID + r * RA_NTS;
      if (q < nl * NCBRM && q % NCBRM < ncbr) cpa8(Yb + q, y + (size_t)(c0 + clo + q % NCBRM) * M + SAMPLE_OF(eY[r]));
      else if (q < nl * NCBRM) Yb[q] = cx{0.f, 0.f};   // coils past the batch: never read; zero as the register path
    }
    #pragma unroll
    for (int r = 0; r < NRE; ++r) {
      const int q = RA_TID + r * RA_NTS;
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
#if RA_STAGES >= 3
  // ---- RA_STAGES 3 / 4 (research/sample_kernels.md §20): the staging loops of one chunk are cut into warp units of
  // 32 consecutive items that warps take round robin (RA_GRAB 1: from a shared counter, dynamic balance: a warp with
  // fewer covered samples in its MAC takes more units), heavy units first; the per-channel z loop is split into RA_ZSPL
  // sample slices so it spreads over all warps. Every item is computed exactly as in the loops of RA_STAGES 1 / 2 and
  // the MAC adds the samples in the same order, so the result is bitwise the same. RA_STAGES 4 also fuses the MAC of
  // chunk k − 1 with the first staging phase of chunk k (between the same two barriers; support boxes and, at order 0,
  // kernel values double-buffered): two barriers per chunk instead of three.
#if RA_GRAB
  // RA_GRAB 1: units from a shared counter (the next unit is taken before the current one runs)
  __shared__ int ra_ctr;
  if (threadIdx.x == 0) ra_ctr = 0;   // published by the first chunk-head barrier
  int cbase = 0;                      // block-uniform: units taken before this phase (each warp overshoots once)
  auto grab = [&]() -> int {
    int u = 0;
    if (lane == 0) u = atomicAdd(&ra_ctr, 1);
    return __shfl_sync(0xffffffffu, u, 0) - cbase;
  };
#define RA_UNIT_LOOP(U) for (int u = grab(), unx = 0; u < (U); u = unx) if ((unx = grab()) >= 0 || true)
#define RA_UNITS_DONE(U) cbase += (U) + TT
#else
  // RA_GRAB 0 (default): warp w takes units w, w + TT, ... of the phase's unit list (heavy units first)
#define RA_UNIT_LOOP(U) for (int u = tz; u < (U); u += TT)
#define RA_UNITS_DONE(U) (void)0
#endif
  int* BOX0 = BOX;
  float* AX0 = AX;
#if RA_STAGES == 4 && CORR == 0
  int* BOX1 = reinterpret_cast<int*>(smem + OFF_BOX2);
  float* AX1 = reinterpret_cast<float*>(smem + OFF_AX2);
#else
  int* BOX1 = BOX;
  float* AX1 = AX;
#endif
  // staging phase 1 of a chunk (nl samples; stage buffers SE / REC / RY / UW, outputs AXw / BOXw): support boxes and
  // E (heaviest units first), kernel values, coefficients w_l, degree-3/4 coefficients
  auto phase1 = [&](const int nl, float* AXw, int* BOXw) {
    const int uB = DIVUP(nl, 32), uA = DIVUP(nl * 3 * TT, 32), uR = DIVUP(nl * NSEG, 32);
#if CORR >= 3
    const int uH = DIVUP(nl * NH, 32);
#else
    const int uH = 0;
#endif
    const int U = uB + uA + uR + uH;
    RA_UNIT_LOOP(U) {
      if (u < uB) {
        const int q = u * 32 + lane;
        if (q < nl) {
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
          BOXw[q] = bx;
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
      } else if (u < uB + uA) {
        const int q = (u - uB) * 32 + lane;
        if (q < nl * 3 * TT) {
          const int s = q / (3 * TT), a = (q / TT) % 3, cell = q % TT;
          const float4 r = REC[s * 3];
          const float sc = a == 0 ? r.x : (a == 1 ? r.y : r.z);
          const int og = a == 0 ? o0 : (a == 1 ? o1 : o2), nn = a == 0 ? d.nf0 : (a == 1 ? d.nf1 : d.nf2);
          const float z = wrapdiff((float)(og + cell) - sc, nn) * (2.f / (float)W);
          float o[5];
          kern_derivs(z, o);
          #pragma unroll
          for (int k = 0; k < NKA; ++k) AXw[((s * 3 + a) * NKA + k) * TT + cell] = o[k];
        }
      } else if (u < uB + uA + uR) {
        const int q = (u - uB - uA) * 32 + lane;
        if (q < nl * NSEG) {
          const int sm = q / NSEG, l = q % NSEG;
          float uu = (REC[sm * 3].w - t0) / dt;
          uu = fminf(fmaxf(uu, 0.f), (float)(ntab - 1) - 1e-3f);
          const int i = (int)uu;
          const float fr = uu - (float)i;
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
      }
#if CORR >= 3
      else {
        const int q = (u - uB - uA - uR) * 32 + lane;
        if (q < nl * NH) {   // degree-3/4 coefficients (stage_hc_k)
          const int s = q / NH, a = q % NH;
          float v = 0.f;
          if (uw && htab) {
            const float4 w4 = UW[s];
            v = fmaf(w4.x, HT[a], fmaf(w4.y, HT[NH + a], fmaf(w4.z, HT[2 * NH + a], w4.w * HT[3 * NH + a])));
          }
          HC[q] = v;
        }
      }
#endif
    }
    RA_UNITS_DONE(U);
  };
  // staging phase 2: tile-row weights (heaviest first), z = y_c w_l in RA_ZSPL sample slices per channel, odd-CB pad
  auto phase2 = [&](const int nl, const float* AXr, const int* BOXr) {
#if CORR >= 1
    const int uW = DIVUP(nl * TROWS, 32);
#else
    const int uW = 0;
    (void)AXr; (void)BOXr;
#endif
    const int uZ = DIVUP(RA_ZSPL * CPW, 32);
#if CB % 2 == 1
    const int uP = DIVUP(nl * NCL, 32);
#else
    const int uP = 0;
#endif
    const int U = uW + uZ + uP;
    RA_UNIT_LOOP(U) {
#if CORR >= 1
      if (u < uW) {
        const int q = u * 32 + lane;
        const int s = q / TROWS, row = q % TROWS, py = row % TT, pz = row / TT;
        if (q < nl * TROWS) {
          const int bx = BOXr[s];
          if (!(py < ((bx >> 8) & 15) || py > ((bx >> 12) & 15) || pz < ((bx >> 16) & 15) || pz > ((bx >> 20) & 15))) {
            const float* ax = AXr + (size_t)s * 3 * NKA * TT;
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
        }
      } else
#endif
      if (u < uW + uZ) {
        const int t = (u - uW) * 32 + lane;
        if (t < RA_ZSPL * CPW) {
          const int k = t % CPW, sl = t / CPW;
          const int zh = DIVUP(nl, RA_ZSPL), s0 = sl * zh, s1 = min(nl, s0 + zh);
          const int ch = chb + k, kk = k % CB;
          const bool ok = ch < nchv;
          const int coil = ok ? ch / NSEG : clo, l = ok ? ch % NSEG : 0;
          const cx* ap = RA + l;
          const cx* yp = RY + (coil - clo);
          cx* zp = ZS + ((kk / 2) * NCL + k / CB) * 2 + (kk & 1);
          #pragma unroll 4
          for (int s = s0; s < s1; ++s) zp[(size_t)s * (CBP / 2) * NCL * 2] = ok ? cmul(yp[s * NCBRM], ap[s * NSEG]) : cx{0.f, 0.f};
        }
      }
#if CB % 2 == 1
      else {   // odd CB: the pad channel of the last pair is zero
        const int q = (u - uW - uZ) * 32 + lane;
        if (q < nl * NCL) ZS[((size_t)(q / NCL) * (CBP / 2) * NCL + (CBP / 2 - 1) * NCL + q % NCL) * 2 + 1] = cx{0.f, 0.f};
      }
#endif
    }
    RA_UNITS_DONE(U);
  };
  // MAC of one staged chunk (nl samples; support boxes BOXr, order-0 kernel values AXr): the warp's plane tz, covered
  // rows only; the next sample's box is loaded ahead
#if RA_STAGES == 4 && CORR >= 1
  // fused, orders ≥ 1: lane i holds the boxes of samples i and 32 + i of the chunk the next MAC adds (loaded after
  // barrier B, before phase 1 of the following chunk overwrites BOX)
  int mbx0 = 0, mbx1 = 0;
  auto boxat = [&](const int* BOXr, int s) -> int { (void)BOXr; return __shfl_sync(0xffffffffu, s < 32 ? mbx0 : mbx1, s & 31); };
#else
  auto boxat = [&](const int* BOXr, int s) -> int { return BOXr[s]; };
#endif
  auto mac = [&](const int nl, const float* AXr, const int* BOXr) {
    (void)AXr;
    int bxn = boxat(BOXr, 0);
    for (int s = 0; s < nl; ++s) {
      const int bx = bxn;
      if (s + 1 < nl) bxn = boxat(BOXr, s + 1);
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
      const float* ax = AXr + (size_t)s * 3 * TT;
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
  };
  auto chunk_bufs = [&](const int buf) {
    SE = SE0 + (size_t)buf * CHK;
    REC = REC0 + (size_t)buf * 3 * CHK;
    RY = RY0 + (size_t)buf * CHK * NCBRM;
#if CORR >= 3
    UW = UW0 + (size_t)buf * CHK;
#endif
  };
  const int nch = DIVUP(n, CHK);
  if (n > 0) {
    entries(0);
    issue(0, 0);
    if (CHK < n) entries(CHK);
  }
#if RA_STAGES == 3
  for (int kch = 0; kch < nch; ++kch) {
    const int base = kch * CHK, nl = min(CHK, n - base), buf = kch & 1;
    cpa_wait_all();    // this thread's copies of chunk k
    __syncthreads();   // everyone's copies visible; the previous chunk's MAC is done (stage buffers free)
    if (base + CHK < n) {
      issue(base + CHK, buf ^ 1);
      if (base + 2 * CHK < n) entries(base + 2 * CHK);
    }
    chunk_bufs(buf);
    phase1(nl, AX0, BOX0);
    __syncthreads();
    phase2(nl, AX0, BOX0);
    __syncthreads();
    mac(nl, AX0, BOX0);
  }
#else
  // iteration k: [barrier A] MAC of chunk k − 1, phase 1 of chunk k [barrier B] phase 2 of chunk k. Barrier A orders
  // phase 2 of k − 1 (z, weights) before its MAC and before phase 1 of k overwrites the coefficients / E / H; barrier B
  // orders the MAC of k − 1 (z, weights read) before phase 2 of k writes them. The MAC reads only z, weights and its own
  // box (/ order-0 kernel value) buffer, so the cp.async copy of chunk k + 1 into k − 1's stage buffer may start at A.
  for (int kch = 0; kch <= nch && nch > 0; ++kch) {
    const int base = kch * CHK, buf = kch & 1;
    if (kch < nch) cpa_wait_all();
    __syncthreads();   // barrier A
    if (base + CHK < n) {
      issue(base + CHK, buf ^ 1);
      if (base + 2 * CHK < n) entries(base + 2 * CHK);
    }
    if (kch > 0) {
      const int pb = (kch - 1) & 1;
      mac(min(CHK, n - (base - CHK)), pb ? AX1 : AX0, pb ? BOX1 : BOX0);
    }
    if (kch < nch) {
      const int nl = min(CHK, n - base);
      chunk_bufs(buf);
      phase1(nl, buf ? AX1 : AX0, buf ? BOX1 : BOX0);
      __syncthreads();   // barrier B
#if CORR >= 1
      mbx0 = lane < nl ? BOX0[lane] : 0;
      mbx1 = (CHK > 32 && lane + 32 < nl) ? BOX0[lane + 32] : 0;
#endif
      phase2(nl, buf ? AX1 : AX0, buf ? BOX1 : BOX0);
    }
  }
#endif
#else
#if RA_WS
  const bool prod = threadIdx.x >= NTA;
  int* const BOXA = BOX;
  cx* const ZSA = ZS;
  int* const BOXB = reinterpret_cast<int*>(smem + OFF_BOXB);
  cx* const ZSB = reinterpret_cast<cx*>(smem + OFF_ZB);
#if CORR >= 1
  cx* const WSA = WS;
  cx* const WSB = reinterpret_cast<cx*>(smem + OFF_WB);
#else
  float* const AXA = AX;
  float* const AXB = reinterpret_cast<float*>(smem + OFF_AXB);
#endif
  if (n > 0 && prod) {
#else
  if (n > 0) {
#endif
    entries(0);
    issue(0, 0);
    if (CHK < n) entries(CHK);
  }
  int kch = 0;
  for (int base = 0; base < n; base += CHK, ++kch) {
    const int nl = min(CHK, n - base);
    const int buf = kch & 1;
#if RA_WS
    // MAC inputs of this chunk: buffer parity kch & 1
    BOX = (kch & 1) ? BOXB : BOXA;
    ZS = (kch & 1) ? ZSB : ZSA;
#if CORR >= 1
    WS = (kch & 1) ? WSB : WSA;
#else
    AX = (kch & 1) ? AXB : AXA;
#endif
    if (prod) {
    cpa_wait_all();    // this thread's copies of chunk k
    RA_PSYNC();        // all producers' copies visible; their staging of the previous chunk is done
    if (kch >= 2) ra_nb_sync(RB_E + (kch & 1), NTA + RA_NTS);   // the MAC warps released chunk k − 2's inputs
#else
    cpa_wait_all();    // this thread's copies of chunk k
    __syncthreads();   // everyone's copies visible; the previous chunk's MAC is done (stage buffers free)
#endif
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
#endif
#else
  // register prefetch of one chunk's records and coil values
  float4 pfr[NRP]; cx pfy[NRY];
#if RA_INPLACE
  unsigned int pfe[NRE];
#if CORR >= 3
  float4 pfu[NRE];
#endif
#endif
  auto prefetch = [&](int base) {
    const int nl = min(CHK, n - base);
#if RA_INPLACE
    #pragma unroll
    for (int r = 0; r < NRE; ++r) {
      const int q = threadIdx.x + r * NTA;
      if (q < nl) {
        pfe[r] = ra_entry(tl, it.first, RLO, RN, base + q);
#if CORR >= 3
        pfu[r] = (uw && htab) ? uw[SAMPLE_OF(pfe[r])] : make_float4(0.f, 0.f, 0.f, 0.f);
#endif
      }
    }
#define RA_SAMPLE(i) SAMPLE_OF(ra_entry(tl, it.first, RLO, RN, base + (i)))
#else
#define RA_SAMPLE(i) SAMPLE_OF((unsigned int)sl[base + (i)])
#endif
    #pragma unroll
    for (int r = 0; r < NRP; ++r) {
      const int q = threadIdx.x + r * NTA;
      if (q < 3 * nl) pfr[r] = rec[(size_t)RA_SAMPLE(q / 3) * 3 + q % 3];
    }
    #pragma unroll
    for (int r = 0; r < NRY; ++r) {
      const int q = threadIdx.x + r * NTA;
      pfy[r] = cx{0.f, 0.f};
      if (q < nl * NCBRM && q % NCBRM < ncbr) pfy[r] = y[(size_t)(c0 + clo + q % NCBRM) * M + RA_SAMPLE(q / NCBRM)];
    }
#undef RA_SAMPLE
  };
  (void)NRE;
  if (n > 0) prefetch(0);
  for (int base = 0; base < n; base += CHK) {
    const int nl = min(CHK, n - base);
    __syncthreads();   // previous chunk consumed
    #pragma unroll
    for (int r = 0; r < NRP; ++r) { const int q = threadIdx.x + r * NTA; if (q < 3 * nl) REC[q] = pfr[r]; }
    #pragma unroll
    for (int r = 0; r < NRY; ++r) { const int q = threadIdx.x + r * NTA; if (q < nl * NCBRM) RY[q] = pfy[r]; }
#if RA_INPLACE
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
#endif
    __syncthreads();
    if (base + CHK < n) prefetch(base + CHK);
#endif
#if RA_STAGES <= 2
    for (int q = RA_TID; q < nl * 3 * TT; q += RA_NTS) {
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
    for (int q = RA_TID; q < nl * NSEG; q += RA_NTS) {
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
#if RA_INPLACE
      const float beta = ra_beta(SE[sm], REC[sm * 3 + 2].w, b, nint);
#else
      const float beta = ra_beta((unsigned int)sl[base + sm], REC[sm * 3 + 2].w, b, nint);
#endif
#else
      const float beta = 1.f;
#endif
      RA[q] = cx{beta * v.x, beta * v.y};
    }
    for (int q = RA_TID; q < nl; q += RA_NTS) {
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
    for (int q = RA_TID; q < nl * NH; q += RA_NTS) {   // degree-3/4 coefficients (stage_hc_k)
      const int s = q / NH, a = q % NH;
      float v = 0.f;
      if (uw && htab) {
#if RA_INPLACE
        const float4 u = UW[s];
        v = fmaf(u.x, HT[a], fmaf(u.y, HT[NH + a], fmaf(u.z, HT[2 * NH + a], u.w * HT[3 * NH + a])));
#else
        const float4 u = uw[SAMPLE_OF((unsigned int)sl[base + s])];
        v = fmaf(u.x, htab[a], fmaf(u.y, htab[NH + a], fmaf(u.z, htab[2 * NH + a], u.w * htab[3 * NH + a])));
#endif
      }
      HC[q] = v;
    }
#endif
    RA_PSYNC();
    // ---- stage B: z = y_c w_l for the block's channels (zero past the batch), one thread per channel (index math out
    // of the sample loop); layout [s][pair kp][cb][2]: a lane's float4 loads of one pair are consecutive
    for (int k = RA_TID; k < CPW; k += RA_NTS) {
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
    for (int q = RA_TID; q < nl * NCL; q += RA_NTS) ZS[((size_t)(q / NCL) * (CBP / 2) * NCL + (CBP / 2 - 1) * NCL + q % NCL) * 2 + 1] = cx{0.f, 0.f};
#endif
#if CORR >= 1
    // complex weights of the covered (sample, tile row) pairs: stage_weights (CORR ≤ 2) / stage_weights_hk (CORR 3)
    for (int q = RA_TID; q < nl * TROWS; q += RA_NTS) {
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
#if RA_WS
    ra_nb_arrive(RB_F + (kch & 1), NTA + RA_NTS);   // chunk k's MAC inputs complete
    } else {
    ra_nb_sync(RB_F + (kch & 1), NTA + RA_NTS);
#else
    __syncthreads();
#endif
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
#if RA_WS
    ra_nb_arrive(RB_E + (kch & 1), NTA + RA_NTS);   // chunk k's MAC inputs free
    }
#endif
  }
#endif
  // ---- output: per warp, row by row, park the TT × CPW values in shared memory and write runs of CPW channels
  __syncthreads();
#if RA_WS
  if (threadIdx.x >= NTA) return;   // producers: no output
#endif
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
        if (split) { atomicAdd(&gp->x, v.x); atomicAdd(&gp->y, v.y); }
        else if (add) { const cx o = *gp; *gp = cx{o.x + v.x, o.y + v.y}; }
        else *gp = v;
      }
    }
    __syncwarp();
  }
}
