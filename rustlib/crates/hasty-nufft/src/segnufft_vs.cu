// Forward variant `vs` of the segmented NUFFT: design 6 of research/sample_kernels.md (value-stationary rows, lanes
// own channels; §12 the prototype `pg_fwd_vs` in proto_gather.cu, §14 this engine kernel). Standalone module (own
// helpers, no Stage / tile-list code). Defines from the host (`segnufft/vs.rs`): W (≤ 10), BETA, NSEG, CORR (0 | 1),
// NCB (coils per batch: grid channel stride NCH = NCB · NSEG), TX, TY, TZ (home tile), SMAX (samples per item), NWC
// (16-channel groups per block), NWS (window-position splits per block), VXPAD (1: zero-padded x weights over HXV),
// KS (knot support; 0 = static image), RDEG (2-D knots; −1: 1-D), optionally KER_KB + kb_eval (Kaiser–Bessel window).
//
//   y_c(j) += β_b(t_j) e^{−iφ0_j} Σ_l a_l(τ_j) Σ_p w_j(p) G[p][c · NSEG + l]        (plain read-modify-write)
//
// Every (sample, coil) of a launch has exactly one owner thread (the host only launches channel blocks that hold
// whole coils), so the output is y += o without atomics; y is initialised by the host (0 or −data), exactly as the
// tile kernels' atomic accumulation expects.
// Weights: CORR 0 the real separable φ(zx) φ(zy) φ(zz); CORR 1 the first-order corrected complex weights of
// segnufft.cu (stage_weights): w = φx R0 + i Σ_a φx^(a) Q_a with R0 = φy φz and Q_a from ΔC and the y / z derivatives.
// CORR 2 / 3 (`vt`, research/sample_kernels.md §13 prototype, §16 engine): per (sample, plane) the warp builds the
// sample's 10 × 10 complex weights (lanes 0..29 = row × x group of 4 points; exactly the terms of segnufft.cu
// stage_weights and, at CORR 3, segnufft_knot.cu stage_weights_hk with hc = Σ_q u_q H'_q (stage_hc_k), wz folded in)
// into a warp-private double-buffered 800 B table after the accumulators; each lane contracts its half's 5 rows against
// the half's 50 weights (25 broadcast LDS.128). VXPAD 0 only. CORR 3: trailing arguments uw / htab (null: c = 0).
//
// Tables (host): per knot b (static: one list) the samples of the knot's window, grouped by home tile (support corner
// i0 = ceil(sc − W/2), wrapped; tile floor(i0 / T) per axis), cut into items of ≤ SMAX samples in (oz, oy, ox) order
// and re-sorted inside an item by class (oy · TX + ox) then oz; entries key << 25 | j (knot engines) or j (static), j
// the ORIGINAL sample index of the records. Item = {tile, first entry, count, zlo | zhi << 16} (tile-relative oz range).
//
// Phase-2 variant (research/sample_kernels.md §19): VSCH < SMAX stages an item in sub-chunks of ≤ VSCH samples (the
// records, the class table and the per-sample accumulators are sized for VSCH, not SMAX), so the block's shared memory
// shrinks (vt 4 × 4 × 16, SMAX 48, nwc 4: 76 KB → ≈ 42 KB at VSCH 24, 2 blocks per SM on 100 KB parts). Same tables.
// Each sub-chunk is a contiguous, still class-sorted part of the item's list; every sample's sum is formed exactly as
// in one pass (same planes, rows and order), so the result is bitwise the same; only the window loads repeat per
// sub-chunk (items longer than VSCH). Default VSCH = SMAX: the kernel above.
struct cx { float x, y; };
__device__ __forceinline__ cx cmul(cx a, cx b) { return cx{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
struct Dims { int nf0, nf1, nf2, nb0, nb1, nb2; };
struct Grids { cx* p[4]; };
struct VsItem { int tile, first, count, zr; };
#define NCH (NCB * NSEG)

// BFP grid slots (research/bfp_storage.md §10, HASTY_SEG_GRID_STORE): with GBFP (mantissa bits) the forward z pass
// stored block k = q / 16 (16 consecutive channel-interleaved values of a (z, y) row) in place: its GBFP mantissa words
// are the first words of the block's own 16 cells, its exponent byte is byte k after the slot's nf0 · nf1 · nf2 · NCH
// cells. Exponent 0 marks a block the pass did not write (outside the k support): zeros.
#ifdef GBFP
__device__ __forceinline__ cx gload_bfp(const cx* __restrict__ g, size_t q, const Dims& d) {
  const size_t k = q >> 4;
  const unsigned char* ex = reinterpret_cast<const unsigned char*>(g + (size_t)d.nf0 * d.nf1 * d.nf2 * NCH);
  if (ex[k] == 0) return cx{0.f, 0.f};
  const float2 v = bfp_load_c<GBFP>(reinterpret_cast<const unsigned*>(g + (k << 4)), ex + k, (long)(q & 15));
  return cx{v.x, v.y};
}
#define GLOAD(g, q) gload_bfp((g), (q), d)
#else
#define GLOAD(g, q) ((g)[q])
#endif
__device__ __forceinline__ int wrapi(int i, int n) { int r = i % n; return r < 0 ? r + n : r; }

// kernel window and its first two z-derivatives (copies of segnufft.cu es3, ES and KB)
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
// derivatives up to order 4 (copy of segnufft.cu es5)
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

#if CORR > 3
#error "vs forward: correction order <= 3"
#endif
#if W > 10
#error "vs forward: the two row halves hold 2 x 5 support rows (W <= 10)"
#endif
#if TY % NWS != 0
#error "vs: NWS must divide TY"
#endif

// ---------------------------------------------------------------- knot weights β_b(t) (as segnufft_knot.cu)
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
// key: interval (1-D) or ri · nint + ic (2-D); w11: record float 11 (u, or (u, v) packed as 16-bit fractions)
__device__ __forceinline__ float vs_beta(int key, float w11, int b, int nint) {
#if RDEG >= 0
  const int k = b / nint, bc = b % nint, ri = key / nint, ic = key % nint;
  const unsigned int bits = __float_as_uint(w11);
  const float u = (float)(bits >> 16) * (1.f / 65535.f), v = (float)(bits & 0xFFFFu) * (1.f / 65535.f);
  const int o = ((bc - KOFF - ic) % nint + nint) % nint;
  return (RDEG == 0 ? 1.f : (ri == k ? 1.f - v : v)) * bspline_k(o, u);
#else
  return bspline_k(((b - KOFF - key) % nint + nint) % nint, w11);
#endif
}
#endif

// ---------------------------------------------------------------- per-sample shared record (flat float words)
#define HXV (TX + W - 1)
#define NCLS (TX * TY)
#define NPLM (TZ + W - 1)
#ifndef VW
#define VW 0
#endif
#if VW == 1
#ifndef VWP
#define VWP 1
#endif
#define NTV ((VWC + VWB + VWP) * 32)    // vw: VWC consumer warps, VWB weight-build warps, VWP window-copy warps
#elif VW == 2
#define NTV ((NWC + VWB) * 32)          // vw 2: the vt channel-group warps + VWB weight-build warps
#else
#define NTV (NWC * NWS * 32)
#endif
#ifndef VXPAD
#define VXPAD 0
#endif
#if VXPAD
#define KXN HXV
#else
#define KXN W
#endif
#define W4 ((W + 3) / 4 * 4)
#if CORR >= 2 && VXPAD
#error "vt (CORR >= 2): VXPAD 0"
#endif
#if CORR == 0
#define KXS (((KXN + 2) + 3) / 4 * 4)   // x weights, oz (int bits), φy(5h + 4)
#define VR_KX 0                         // [2][KXS]: the x row duplicated per half
#define VR_Y4 (2 * KXS)                 // [2][4]: φy(5h .. 5h + 3)
#define VR_WZ (VR_Y4 + 8)               // [W]: φz
#define VR_M (VR_WZ + W4)
#elif CORR == 1
#define KXS (((KXN + 1) + 3) / 4 * 4)   // [3][KXS]: φx, φx', φx''; row 0 slot KXN: oz (int bits)
#define VR_KX 0
#define VR_WY (3 * KXS)                 // [2][16]: wy_k(5h + i) at k * 5 + i (0 beyond the support)
#define VR_WZ (VR_WY + 32)              // [W][4]: wz_k(t) at t * 4 + k
#define VR_CC (VR_WZ + 4 * W)           // ΔC: xx, yy, zz, xy, xz, yz (kernel-argument units, off-diagonals doubled)
#define VR_M (VR_CC + 8)
#endif
#if CORR <= 1
#define VR_F (VR_M + 2)                 // M: ox, oy (int); F: table index (int), fraction, cos φ0, sin φ0, β
#define VR_J (VR_F + 5)                 // original sample index (int)
#define VR_SC (VR_J + 1)                // grid coordinates (staging only)
#define VSW ((VR_SC + 3 + 3) / 4 * 4)
#define VR_OZP (VR_KX + KXN)            // oz (int)
#else
// vt record: kx[a][12] (t padded to 12, zero for t ≥ W), wy[i][8] = wy_0..4(i), wz_0..2(i) (zero for i ≥ W),
// wz_3..4 [i][2], e[15] d[6] sc[3] (grid coordinates, staging only), CORR 3: hc[25] (+3), M: ox, oy, oz, j (int),
// F: table index (int), fraction, cos φ0, sin φ0, β
#define VR_KX 0
#define VR_WY 60
#define VR_WZH 140
#define VR_E 160
#define VR_D (VR_E + 15)
#define VR_SC (VR_E + 21)
#if CORR >= 3
#define VR_H 184
#define VR_M 212
#else
#define VR_M 184
#endif
#define VR_OZ (VR_M + 2)
#define VR_J (VR_M + 3)
#define VR_F (VR_M + 4)
#define VSW ((VR_F + 5 + 3) / 4 * 4)
#define VR_OZP VR_OZ
#endif
#if CORR >= 3
#define VHI_ARGS , const float4* __restrict__ uw, const float* __restrict__ htab
#else
#define VHI_ARGS
#endif
#ifndef VSCH
#define VSCH SMAX
#endif
#if VSCH > SMAX || VSCH < 1
#error "vs: 1 <= VSCH <= SMAX"
#endif
#define APL (VSCH * 16 + 16)            // accumulator plane stride (re / im planes 16 banks apart)
// ---------------------------------------------------------------- vtc: tensor-core contraction (VTC 1; §21)
#ifndef VTC
#define VTC 0
#endif
#if VTC
#if CORR < 2 || VXPAD || NWS > 2 || SMAX > 64 || VSCH > 32 || defined(GBFP)
#error "vtc: CORR >= 2, VXPAD 0, NWS <= 2 (chunk splits), SMAX <= 64, VSCH <= 32, no BFP grid"
#endif
#define UX (TX + W - 1)
#define UY (TY + W - 1)
#define NPU (UX * UY)                   // union-window points of a plane
#define NKC ((NPU + 15) / 16)           // 16-point K chunks
#define NPV (NKC * 16)
#define BWS (NKC * 8 + 4)               // uint4 per sample in Bw (≡ 4 mod 8: conflict-free B-fragment loads)
#define VSCH8 ((VSCH + 7) / 8 * 8)
#define NT8 (VSCH8 / 8)
#define NRNG ((SMAX + NPLM + 1) / 2 * 2 - SMAX)   // rng words (poff 8-byte aligned: int2 loads)
#define VTC_BOFF ((((size_t)VSCH * VSW + SMAX + NRNG + NPV) * 4 + 15) & ~(size_t)15)
// bf16x2 hi / lo split of (a, b) (a in the low half): hi = rn(a), rn(b); lo = rn(a − hi_a), rn(b − hi_b)
__device__ __forceinline__ void bsplit(float a, float b, unsigned& hi, unsigned& lo) {
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(hi) : "f"(b), "f"(a));
  const float ha = __uint_as_float(hi << 16), hb = __uint_as_float(hi & 0xffff0000u);
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(lo) : "f"(b - hb), "f"(a - ha));
}
__device__ __forceinline__ void mma_bf16(float* c, const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// VTC 2: fp16 hi / lo (11 + 11 bits, ≈ 2^-22 per product). fp16 has 5 exponent bits, so the operands are scaled by
// powers of two: weights by 2^8 (|w| < 255), grid values per (channel, plane) by 2^E with max |G| 2^E < 2^15 (E
// lowered chunk by chunk as larger values appear; the accumulators are rescaled exactly by the same powers of two)
__device__ __forceinline__ void hsplit(float a, float b, unsigned& hi, unsigned& lo) {
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(hi) : "f"(b), "f"(a));
  float ha, hb;
  asm("{ .reg .b16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h; }" : "=f"(ha), "=f"(hb) : "r"(hi));
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(lo) : "f"(b - hb), "f"(a - ha));
}
__device__ __forceinline__ void mma_f16(float* c, const unsigned (&a)[4], unsigned b0, unsigned b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#if VTC == 2
#define VSPLIT hsplit
#define VMMA mma_f16
#define WSC 256.f
#define WSE 8
#else
#define VSPLIT bsplit
#define VMMA mma_bf16
#define WSC 1.f
#define WSE 0
#endif
__device__ __forceinline__ float pow2i(int e) { e = max(-126, min(127, e)); return __int_as_float((e + 127) << 23); }
#endif
#define NOXV (VXPAD ? 1 : TX)
// ---------------------------------------------------------------- vw: warp-specialised vt (VW 1; research/sample_kernels.md §22)
// Block = one item sub-chunk × the channel row (NWC 16-channel slices). Roles: VWB build warps write each active
// (sample, plane)'s 10 × 10 complex weight table ONCE per block into a ring of VWT tables (vt_build); VWP copy warps
// stage the plane's union window (VUX × VUY points × 16 channels of one slice, wrapped point offsets computed once per
// item) into VWS shared stages by cp.async; VWC consumer warps contract (lane = channel × row half, as vt) with every
// grid value and weight read from shared memory at compile-time offsets (no register window). Hand-over by named
// hardware barriers (bar.arrive / bar.sync, no polling: mbarrier waits are shared-memory spin loops on sm_80 / sm_89
// and took a third of the shared-memory bandwidth): tables full / free per plane parity, window full / free per stage
// (no block barrier in the main loop; every barrier phase is completed exactly once, in order). Sample s of a
// sub-chunk is always contracted by consumer s mod VWC (no accumulator races); per sample the planes are summed in
// increasing order with vt's own row / point order, so the result equals vt's.
#if VW == 1
#if CORR < 2 || VXPAD || NWS != 1 || VTC || defined(GBFP)
#error "vw: CORR >= 2, VXPAD 0, NWS 1, no VTC, no BFP grid"
#endif
#ifndef VWS
#define VWS 2
#endif
#ifndef VWT
#define VWT VSCH
#endif
#if VWT < VSCH
#error "vw: VWT >= VSCH (one plane's tables must fit)"
#endif
#define VUX (TX + W - 1)
#define VUY (TY + 9)                    // rows of 10-row supports (W < 10: zero-weight rows read real grid values)
#define VNP (VUX * VUY)
#define VNPO (VNP > SMAX ? VNP : SMAX)  // point offsets; stage 0 keeps the entries' oz there
#define VSL (VNP * 16)                  // float2 per window stage
#define VW_ENT ((size_t)VSCH * VSW * 4)
#define VW_RNG (VW_ENT + (size_t)SMAX * 4)
#define VW_PRE (VW_RNG + (size_t)NPLM * 4)
#define VW_POFF (VW_PRE + (size_t)NPLM * 4)
#define VW_ACC ((VW_POFF + (size_t)VNPO * 4 + 15) & ~(size_t)15)
#define VW_WT (VW_ACC + (size_t)NWC * 2 * APL * 4)
#define VW_WIN (VW_WT + (size_t)VWT * 800)
#define VW_SMEM (VW_WIN + (size_t)VWS * VSL * 8)
#if NCH % 2 == 0
#define VCP 2                           // channels per cp.async (16 B; point stride NCH · 8 B is 16-byte aligned)
#else
#define VCP 1
#endif
__device__ __forceinline__ unsigned vw_sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
// named hardware barriers (bar.sync / bar.arrive id, n: no polling; ids 1 .. 4 + 2 VWS, 0 is __syncthreads)
__device__ __forceinline__ void nb_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void nb_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
#define VB_WF 1                         // + plane parity: tables full (build warps arrive, consumers sync)
#define VB_WE 3                         // + plane parity: tables free (consumers arrive, build warps sync)
#define VB_SF 5                         // + stage: window full (copy warps arrive, consumers sync)
#define VB_SE (5 + VWS)                 // + stage: window free (consumers arrive, copy warps sync)
#define VN_W (32 * (VWC + VWB))
#define VN_S (32 * (VWC + VWP))
#if 5 + 2 * VWS > 16
#error "vw: VWS <= 5 (named barrier ids)"
#endif
__device__ __forceinline__ void vw_cp(void* d, const void* s) {
#if VCP == 2
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(vw_sa(d)), "l"(__cvta_generic_to_global(s)) : "memory");
#else
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" :: "r"(vw_sa(d)), "l"(__cvta_generic_to_global(s)) : "memory");
#endif
}
__device__ __forceinline__ void vw_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void vw_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }
#endif

// ---------------------------------------------------------------- vw 2: vt with the weight tables built once per block
// (VW 2; research/sample_kernels.md §22): the vt kernel's channel-group warps keep their register window and contract
// as before, but read each (sample, plane) table from a ring of VWT tables in shared memory that VWB build warps fill one
// plane ahead (vt_build, once per block instead of once per channel group); per plane a byte map gives every active
// sample its ring slot. Hand-over by named barriers per plane parity (tables full / free). Same sums as vt: bitwise.
#if VW == 2
#if CORR < 2 || VXPAD || NWS != 1 || VTC || defined(GBFP)
#error "vw 2: CORR >= 2, VXPAD 0, NWS 1, no VTC, no BFP grid"
#endif
#ifndef VWT
#define VWT VSCH
#endif
#if VWT < VSCH || VWT > 255
#error "vw 2: VSCH <= VWT <= 255 (a plane's tables must fit; byte slots)"
#endif
__device__ __forceinline__ void nb_sync(int id, int n) { asm volatile("bar.sync %0, %1;" :: "r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void nb_arrive(int id, int n) { asm volatile("bar.arrive %0, %1;" :: "r"(id), "r"(n) : "memory"); }
#define VB_WF 1
#define VB_WE 3
#define VN_W (32 * (NWC + VWB))
#endif
#if VW == 1
#define VLB __launch_bounds__(NTV, 1)   // without the block minimum ptxas targets 64 registers at 512 threads and spills
#elif VW == 2 && defined(VWMR)
// register cap from the host (2 blocks per SM: 64 K / (2 · threads)); __launch_bounds__(NTV, 2) rounds the threads up
// to 192 and caps at 168, which spills the register window (36 B) and ran 2 × slower
#define VLB __maxnreg__(VWMR)
#elif VW == 2
#ifndef VWLB
#define VWLB 2
#endif
#define VLB __launch_bounds__(NTV, VWLB)
#else
#define VLB __launch_bounds__(NTV)
#endif
extern "C" __global__ void VLB seg_forward_vs(const float4* __restrict__ rec, const unsigned int* __restrict__ vl, const VsItem* __restrict__ items,
                                                                 const cx* __restrict__ atab, int ntab, float t0, float dt, Grids g, int b, int nint, int nri,
                                                                 cx* __restrict__ y, int M, int c0, int ncb, Dims d VHI_ARGS) {
  extern __shared__ __align__(16) unsigned char smem[];
  float* S = reinterpret_cast<float*>(smem);
#if VW == 1
  // vw layout: records | ent[SMAX] | rng[NPLM] | pre[NPLM] | poff[VNPO] | accb | Wt[VWT][50] | Win[VWS][VSL]
  unsigned int* ent = reinterpret_cast<unsigned int*>(smem + VW_ENT);
  unsigned int* rng = reinterpret_cast<unsigned int*>(smem + VW_RNG);
  int* pre = reinterpret_cast<int*>(smem + VW_PRE);
  int* poff = reinterpret_cast<int*>(smem + VW_POFF);
  float* accb = reinterpret_cast<float*>(smem + VW_ACC);
  float4* Wt = reinterpret_cast<float4*>(smem + VW_WT);
  float2* Win = reinterpret_cast<float2*>(smem + VW_WIN);
  // plane / window-slice sequence numbers across sub-chunks (barrier ids by parity / stage); planes whose table release
  // the build warps have taken (each named-barrier phase is completed exactly once, in order)
  unsigned vw_pseq = 0, vw_tseq = 0, vw_rel = 0;
#elif VTC
  // vtc layout: records | ent[SMAX] | rng[NPLM] | poff[NPV] | Bw[2][VSCH8][BWS] (accb aliases Bw after the main loop)
  unsigned int* ent = reinterpret_cast<unsigned int*>(smem + (size_t)VSCH * VSW * 4);
  unsigned int* rng = ent + SMAX;
  int* poff = reinterpret_cast<int*>(rng + NRNG);
  uint4* Bw = reinterpret_cast<uint4*>(smem + VTC_BOFF);
  float* accb = reinterpret_cast<float*>(Bw);
#else
  unsigned int* tab = reinterpret_cast<unsigned int*>(smem + (size_t)VSCH * VSW * 4);
  float* accb = reinterpret_cast<float*>(smem + (size_t)VSCH * VSW * 4 + (((size_t)NPLM * NCLS * 4 + 15) & ~(size_t)15));
#if VW == 2
  // vw 2: ring of VWT weight tables after the accumulators (instead of the per-warp tables), then the slot map
  float4* Wt = reinterpret_cast<float4*>(accb + (size_t)NWC * 2 * APL);
  unsigned char* smap = reinterpret_cast<unsigned char*>(Wt + (size_t)VWT * 50);   // [plane parity][sample]
  unsigned vw_pseq = 0, vw_rel = 0;   // plane sequence (barrier phases) across sub-chunks; planes released to the builders
#endif
#endif
  (void)b; (void)nint; (void)nri;
  const VsItem it = items[blockIdx.x];
  const int ntx = (d.nf0 + TX - 1) / TX, nty = (d.nf1 + TY - 1) / TY;
  const int o0 = (it.tile % ntx) * TX, o1 = ((it.tile / ntx) % nty) * TY, o2 = (it.tile / (ntx * nty)) * TZ;
  const int zlo = it.zr & 0xffff, npl = (it.zr >> 16) - zlo + W;
  const int nchv = ncb * NSEG;
#if VTC || VW == 1
  // ---- stage 0 (vtc): the item's entries in (oz, list position) order, so that every sub-chunk and every plane's
  // active samples are contiguous (poff holds the entries' oz meanwhile)
  {
    for (int q = threadIdx.x; q < it.count; q += NTV) {
      const unsigned int e = vl[it.first + q];
#if KS > 0
      const unsigned int j = e & KEY_MASK;
#else
      const unsigned int j = e;
#endif
      poff[q] = wrapi((int)ceilf(rec[(size_t)j * 3].z - (float)W * 0.5f), d.nf2) - o2;
    }
    __syncthreads();
    for (int q = threadIdx.x; q < it.count; q += NTV) {
      const int oz = poff[q];
      int r = 0;
      for (int u = 0; u < it.count; ++u) { const int o = poff[u]; r += (o < oz) || (o == oz && u < q); }
      ent[r] = vl[it.first + q];
    }
    __syncthreads();
  }
#endif
#if VSCH < SMAX
  // sub-chunks of ≤ VSCH samples of the item (block-uniform loop; one barrier between sub-chunks)
  for (int sq0 = 0; sq0 < it.count; sq0 += VSCH) {
  const int cnt = min(VSCH, it.count - sq0);
#else
  {
  const int sq0 = 0;
  const int cnt = it.count;
#endif
  // ---- stage 1, per sample: index, coefficient terms, coordinates
  for (int q = threadIdx.x; q < cnt; q += NTV) {
    float* P = S + (size_t)q * VSW;
#if VTC || VW == 1
    const unsigned int e = ent[sq0 + q];
#else
    const unsigned int e = vl[it.first + sq0 + q];
#endif
#if KS > 0
    const unsigned int j = e & KEY_MASK;
#else
    const unsigned int j = e;
#endif
    const float4 r0 = rec[(size_t)j * 3], r1 = rec[(size_t)j * 3 + 1], r2 = rec[(size_t)j * 3 + 2];
    (void)r2;
    P[VR_SC] = r0.x; P[VR_SC + 1] = r0.y; P[VR_SC + 2] = r0.z;
    // a_l(τ) table position (as stage_samples / stage_kv)
    float u = (r0.w - t0) / dt;
    u = fminf(fmaxf(u, 0.f), (float)(ntab - 1) - 1e-3f);
    const int ti = (int)u;
    reinterpret_cast<int*>(P)[VR_F] = ti;
    P[VR_F + 1] = u - (float)ti;
    float s0, c1; sincosf(r1.x, &s0, &c1);
    P[VR_F + 2] = c1; P[VR_F + 3] = s0;
#if KS > 0
    P[VR_F + 4] = vs_beta((int)(e >> KEY_SHIFT), r2.w, b, nint);
#else
    P[VR_F + 4] = 1.f;
#endif
    reinterpret_cast<unsigned int*>(P)[VR_J] = j;
#if CORR == 1
    P[VR_CC] = r1.y; P[VR_CC + 1] = r1.z; P[VR_CC + 2] = r1.w; P[VR_CC + 3] = r2.x; P[VR_CC + 4] = r2.y; P[VR_CC + 5] = r2.z;
#elif CORR >= 2
    // ΔC [xx, yy, zz, xy, xz, yz] and the second-order coefficients e_α = −½ Σ_pq d_p d_q (as stage_samples)
    const float dd[6] = {r1.y, r1.z, r1.w, r2.x, r2.y, r2.z};
    const int mx[6] = {2, 0, 0, 1, 1, 0}, my[6] = {0, 2, 0, 1, 0, 1};
    float ec[15];
    #pragma unroll
    for (int k = 0; k < 15; ++k) ec[k] = 0.f;
    #pragma unroll
    for (int p = 0; p < 6; ++p) {
      P[VR_D + p] = dd[p];
      #pragma unroll
      for (int pq = 0; pq < 6; ++pq) {
        const int ea = mx[p] + mx[pq], eb = my[p] + my[pq];
        ec[ea * (11 - ea) / 2 + eb] = fmaf(-0.5f * dd[p], dd[pq], ec[ea * (11 - ea) / 2 + eb]);
      }
    }
    #pragma unroll
    for (int k = 0; k < 15; ++k) P[VR_E + k] = ec[k];
#endif
  }
  __syncthreads();
  // ---- stage 2, per (sample, axis, cell): kernel values and derivatives, support offsets on the tile
#if CORR >= 2
  for (int q = threadIdx.x; q < cnt * 3 * 12; q += NTV) {
    const int s = q / 36, a = (q / 12) % 3, t = q % 12;
    if (a != 0 && t >= 10) continue;
    float* P = S + (size_t)s * VSW;
    const float sc = P[VR_SC + a];
    const int i0 = (int)ceilf(sc - (float)W * 0.5f);
    const int n = a == 0 ? d.nf0 : (a == 1 ? d.nf1 : d.nf2), og = a == 0 ? o0 : (a == 1 ? o1 : o2);
    float o[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    if (t < W) es5(((float)(i0 + t) - sc) * (2.f / (float)W), o);
    if (a == 0) {
      #pragma unroll
      for (int k = 0; k < 5; ++k) P[VR_KX + k * 12 + t] = o[k];
    } else if (a == 1) {
      #pragma unroll
      for (int k = 0; k < 5; ++k) P[VR_WY + t * 8 + k] = o[k];
    } else {
      #pragma unroll
      for (int k = 0; k < 3; ++k) P[VR_WY + t * 8 + 5 + k] = o[k];
      P[VR_WZH + t * 2] = o[3]; P[VR_WZH + t * 2 + 1] = o[4];
    }
    if (t == 0) reinterpret_cast<int*>(P)[VR_M + a] = wrapi(i0, n) - og;
  }
#if CORR >= 3
  // degree-3/4 coefficients hc = Σ_q u_q H'_q of the ORIGINAL sample index (segnufft_knot.cu stage_hc_k)
  for (int q = threadIdx.x; q < cnt * 28; q += NTV) {
    const int s = q / 28, a = q % 28;
    float* P = S + (size_t)s * VSW;
    float v = 0.f;
    if (a < 25 && uw && htab) {
      const float4 u = uw[reinterpret_cast<const unsigned int*>(P)[VR_J]];
      v = fmaf(u.x, htab[a], fmaf(u.y, htab[25 + a], fmaf(u.z, htab[50 + a], u.w * htab[75 + a])));
    }
    P[VR_H + a] = v;
  }
#endif
#else
  constexpr int WQ = KXN > 10 ? KXN : 10;
  for (int q = threadIdx.x; q < cnt * 3 * WQ; q += NTV) {
    const int s = q / (3 * WQ), a = (q / WQ) % 3, t = q % WQ;
    if ((a == 0 && t >= KXN) || (a == 1 && t >= 10) || (a == 2 && t >= W)) continue;
    float* P = S + (size_t)s * VSW;
    const float sc = P[VR_SC + a];
    const int i0 = (int)ceilf(sc - (float)W * 0.5f);
    const int n = a == 0 ? d.nf0 : (a == 1 ? d.nf1 : d.nf2), og = a == 0 ? o0 : (a == 1 ? o1 : o2);
    const int off = wrapi(i0, n) - og;
    const int tt = (a == 0 && VXPAD) ? t - off : t;
    float f = 0.f, d1 = 0.f, d2 = 0.f;
    if (tt >= 0 && tt < W) es3(((float)(i0 + tt) - sc) * (2.f / (float)W), f, d1, d2);
    if (a == 0) {
#if CORR == 0
      P[VR_KX + t] = f; P[VR_KX + KXS + t] = f;
#else
      P[VR_KX + t] = f; P[VR_KX + KXS + t] = d1; P[VR_KX + 2 * KXS + t] = d2;
#endif
      if (t == 0) reinterpret_cast<int*>(P)[VR_M] = off;
    } else if (a == 1) {
      // rows t ≥ W (W < 10): zero weights (the halves always cover 10 rows)
      const int h = t / 5, i = t % 5;
#if CORR == 0
      if (i < 4) P[VR_Y4 + h * 4 + i] = f; else P[VR_KX + h * KXS + KXN + 1] = f;
#else
      P[VR_WY + h * 16 + i] = f; P[VR_WY + h * 16 + 5 + i] = d1; P[VR_WY + h * 16 + 10 + i] = d2;
#endif
      if (t == 0) reinterpret_cast<int*>(P)[VR_M + 1] = off;
    } else {
#if CORR == 0
      P[VR_WZ + t] = f;
#else
      P[VR_WZ + t * 4] = f; P[VR_WZ + t * 4 + 1] = d1; P[VR_WZ + t * 4 + 2] = d2;
#endif
      if (t == 0) {
        reinterpret_cast<int*>(P)[VR_KX + KXN] = off;
#if CORR == 0
        reinterpret_cast<int*>(P)[VR_KX + KXS + KXN] = off;
#endif
      }
    }
  }
#endif   // CORR >= 2
#if VW == 1
  __syncthreads();   // stage 2's offsets (oz) are read below
  // ---- vw tables: per plane the active range [a, b) of the oz-sorted sub-chunk, union-window point offsets (wrapped
  // once here), zeroed accumulators; then the table-ring start pre[hz] of every plane
  for (int q = threadIdx.x; q < npl; q += NTV) {
    const int z = zlo + q;
    int a = 0, bb = 0;
    for (int s = 0; s < cnt; ++s) { const int oz = reinterpret_cast<const int*>(S + (size_t)s * VSW)[VR_OZ]; a += oz < z - (W - 1); bb += oz <= z; }
    rng[q] = (unsigned)a | ((unsigned)bb << 16);
  }
  for (int q = threadIdx.x; q < VNP; q += NTV) {
    int gy = o1 + q / VUX, gx = o0 + q % VUX;
    gy -= gy >= d.nf1 ? d.nf1 : 0;
    gx -= gx >= d.nf0 ? d.nf0 : 0;
    poff[q] = (gy * d.nf0 + gx) * NCH;
  }
  for (int q = threadIdx.x; q < NWC * 2 * APL; q += NTV) accb[q] = 0.f;
  __syncthreads();
  if (threadIdx.x == 0) {
    int acc = 0;
    for (int hz = 0; hz < npl; ++hz) { pre[hz] = acc; acc += (int)(rng[hz] >> 16) - (int)(rng[hz] & 0xffffu); }
  }
  __syncthreads();
#elif VTC
  __syncthreads();   // stage 2's offsets (oz) are read below
  // ---- vtc tables: per plane the active range [a, b) of the oz-sorted sub-chunk; offsets of the union-window points
  for (int q = threadIdx.x; q < npl; q += NTV) {
    const int z = zlo + q;
    int a = 0, bb = 0;
    for (int s = 0; s < cnt; ++s) { const int oz = reinterpret_cast<const int*>(S + (size_t)s * VSW)[VR_OZ]; a += oz < z - (W - 1); bb += oz <= z; }
    rng[q] = (unsigned)a | ((unsigned)bb << 16);
  }
  for (int q = threadIdx.x; q < NPV; q += NTV) {
    int o = -1;
    if (q < NPU) {
      int gy = o1 + q / UX, gx = o0 + q % UX;
      gy -= gy >= d.nf1 ? d.nf1 : 0;
      gx -= gx >= d.nf0 ? d.nf0 : 0;
      o = (gy * d.nf0 + gx) * NCH;
    }
    poff[q] = o;
  }
  __syncthreads();
#else
  for (int q = threadIdx.x; q < NWC * 2 * APL; q += NTV) accb[q] = 0.f;
  __syncthreads();
  // ---- per-item table: active range of each class in each plane (samples sorted by (class key, oz))
  for (int q = threadIdx.x; q < NCLS; q += NTV) {
    auto key = [&](int s) { const int* m = reinterpret_cast<const int*>(S + (size_t)s * VSW) + VR_M; return m[1] * TX + m[0]; };
    auto ozs = [&](int s) { return reinterpret_cast<const int*>(S + (size_t)s * VSW)[VR_OZP]; };
    int cb = 0;
    while (cb < cnt && key(cb) < q) ++cb;
    int ce = cb;
    while (ce < cnt && key(ce) == q) ++ce;
    int a = cb, bb = cb;
    for (int hz = 0; hz < npl; ++hz) {
      const int z = zlo + hz;
      while (bb < ce && ozs(bb) <= z) ++bb;
      while (a < bb && ozs(a) < z - (W - 1)) ++a;
      tab[hz * NCLS + q] = (unsigned)a | ((unsigned)bb << 16);
    }
  }
  __syncthreads();
#endif
  // ---- main loop (per warp, no barriers)
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int cg = warp % NWC, sp = warp / NWC;
  const int h = lane >> 4, c = lane & 15;
  const int gbase = (blockIdx.y * NWC + cg) * 16;
  const int ch = gbase + c;
  const bool valid = ch < nchv;
  float* accm = accb + (size_t)(cg * 2 + h) * APL + c;   // half 0: real plane, half 1: imaginary plane
  const int stb = NWS == 1 ? 0 : sp * (TY / NWS);
#if CORR >= 2
  // vt: warp-private double-buffered weight table (2 × 50 float4) after the accumulators; builder lane (bi, bg) → row
  // bi, x points 4 bg .. 4 bg + 3 (lanes 30, 31 compute on in-bounds padding and store nothing)
  float4* tabw = reinterpret_cast<float4*>(accb + (size_t)NWC * 2 * APL) + (size_t)warp * 100;
  const int bi = lane % 10, bg = lane / 10;
  int par = 0;
  auto vt_build = [&](int s, int hz, float4* Tb) {
    const float* P = S + (size_t)s * VSW;
    const int tz = zlo + hz - reinterpret_cast<const int*>(P)[VR_OZ];
    const float4 za = *reinterpret_cast<const float4*>(P + VR_WY + tz * 8 + 4);
    const float2 zb = *reinterpret_cast<const float2*>(P + VR_WZH + tz * 2);
    const float wz[5] = {za.y, za.z, za.w, zb.x, zb.y};
    float ev[24];
    #pragma unroll
    for (int k = 0; k < 6; ++k) { const float4 u = *reinterpret_cast<const float4*>(P + VR_E + 4 * k); ev[4 * k] = u.x; ev[4 * k + 1] = u.y; ev[4 * k + 2] = u.z; ev[4 * k + 3] = u.w; }
    const float* dv = ev + 15;   // [xx, yy, zz, xy, xz, yz]
    // plane coefficients: ER_ab = e_ab wz_{4−a−b} (+ wz_0 at (0, 0)), EQ: the six ΔC terms (stage_weights)
    float ER[15], EQ[15];
    {
      int idx = 0;
      #pragma unroll
      for (int a = 0; a <= 4; ++a) {
        #pragma unroll
        for (int bb = 0; bb <= 4 - a; ++bb) { ER[idx] = ev[idx] * wz[4 - a - bb]; EQ[idx] = 0.f; ++idx; }
      }
    }
    ER[0] += wz[0];
    EQ[0] = dv[2] * wz[2]; EQ[1] = dv[5] * wz[1]; EQ[2] = dv[1] * wz[0];
    EQ[5] = dv[4] * wz[1]; EQ[6] = dv[3] * wz[0]; EQ[9] = dv[0] * wz[0];
#if CORR >= 3
    {
      float hv[28];
      #pragma unroll
      for (int k = 0; k < 7; ++k) { const float4 u = *reinterpret_cast<const float4*>(P + VR_H + 4 * k); hv[4 * k] = u.x; hv[4 * k + 1] = u.y; hv[4 * k + 2] = u.z; hv[4 * k + 3] = u.w; }
      int j = 0;
      #pragma unroll
      for (int a = 0; a <= 3; ++a) {   // degree 3: + c ∂^α φ (real)
        #pragma unroll
        for (int bb = 0; bb <= 3 - a; ++bb) { const int id = a * (11 - a) / 2 + bb; ER[id] = fmaf(hv[j], wz[3 - a - bb], ER[id]); ++j; }
      }
      #pragma unroll
      for (int a = 0; a <= 4; ++a) {   // degree 4: −i c ∂^α φ (imaginary)
        #pragma unroll
        for (int bb = 0; bb <= 4 - a; ++bb) {
          const int id = a * (11 - a) / 2 + bb;
          const bool nz = id == 0 || id == 1 || id == 2 || id == 5 || id == 6 || id == 9;
          EQ[id] = nz ? fmaf(-hv[j], wz[4 - a - bb], EQ[id]) : -hv[j] * wz[4 - a - bb];
          ++j;
        }
      }
    }
#endif
    // row factors R_a = Σ_b ER_ab wy_b, Q_a = Σ_b EQ_ab wy_b of the builder's row
    const float4 ya = *reinterpret_cast<const float4*>(P + VR_WY + bi * 8);
    const float wy[5] = {ya.x, ya.y, ya.z, ya.w, P[VR_WY + bi * 8 + 4]};
    float R[5], Q[5];
    {
      int idx = 0;
      #pragma unroll
      for (int a = 0; a <= 4; ++a) {
        R[a] = ER[idx] * wy[0];
        #pragma unroll
        for (int bb = 1; bb <= 4 - a; ++bb) R[a] = fmaf(ER[idx + bb], wy[bb], R[a]);
#if CORR >= 3
        Q[a] = EQ[idx] * wy[0];
        #pragma unroll
        for (int bb = 1; bb <= 4 - a; ++bb) Q[a] = fmaf(EQ[idx + bb], wy[bb], Q[a]);
#endif
        idx += 5 - a;
      }
    }
#if CORR < 3
    Q[0] = fmaf(EQ[2], wy[2], fmaf(EQ[1], wy[1], EQ[0] * wy[0]));
    Q[1] = fmaf(EQ[6], wy[1], EQ[5] * wy[0]);
    Q[2] = EQ[9] * wy[0];
    Q[3] = 0.f; Q[4] = 0.f;
#endif
    constexpr int NQ = CORR >= 3 ? 5 : 3;
    float wr[4], wi[4];
    #pragma unroll
    for (int a = 0; a < 5; ++a) {
      const float4 u = *reinterpret_cast<const float4*>(P + VR_KX + a * 12 + bg * 4);
      const float kk[4] = {u.x, u.y, u.z, u.w};
      #pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        wr[jj] = a == 0 ? kk[jj] * R[0] : fmaf(kk[jj], R[a], wr[jj]);
        if (a < NQ) wi[jj] = a == 0 ? kk[jj] * Q[0] : fmaf(kk[jj], Q[a], wi[jj]);
      }
    }
    if (lane < 30) {
      Tb[bi * 5 + bg * 2] = make_float4(wr[0], wi[0], wr[1], wi[1]);
      if (bg < 2) Tb[bi * 5 + bg * 2 + 1] = make_float4(wr[2], wi[2], wr[3], wi[3]);
    }
  };
#endif
#if VW == 1
  // ---- vw main loop (warp-specialised; no block barrier): consumers | build warps | window-copy warp
  {
    const int gb0 = blockIdx.y * NWC * 16;
    const int nsl = (min(gb0 + NWC * 16, nchv) - gb0 + 15) / 16;   // 16-channel slices with channels (≥ 1)
    if (warp < VWC) {
      for (int hz = 0; hz < npl; ++hz) {
        const unsigned rgw = rng[hz];
        const int sa = (int)(rgw & 0xffffu), sb = (int)(rgw >> 16);
        if (sa == sb) continue;
        const unsigned pq = vw_pseq++;
        nb_sync(VB_WF + (pq & 1), VN_W);   // the plane's weight tables
        const int rb = pre[hz] % VWT;
        const int s0 = sa + (warp - sa % VWC + VWC) % VWC;   // first sample ≥ sa owned by this consumer
        for (int k = 0; k < nsl; ++k) {
          const unsigned tq = vw_tseq++;
          const int stg = (int)(tq % VWS);
          nb_sync(VB_SF + stg, VN_S);   // the slice's window
          const float2* Wn = Win + (size_t)stg * VSL + c + h * (5 * VUX * 16);
          const bool valid = gb0 + 16 * k + c < nchv;
          float* accm = accb + (size_t)(k * 2 + h) * APL + c;
          for (int s = s0; s < sb; s += VWC) {
            const int* Mo = reinterpret_cast<const int*>(S + (size_t)s * VSW) + VR_M;
            const float2* gp = Wn + (Mo[1] * VUX + Mo[0]) * 16;
            int slot = rb + s - sa;
            slot -= slot >= VWT ? VWT : 0;
            const float4* Th = Wt + (size_t)slot * 50 + h * 25;
            // vt's contraction: the lane's half's 5 rows × W points, same FMA order
            float ar[5], ai[5];
            #pragma unroll
            for (int r = 0; r < 5; ++r) {
              ar[r] = 0.f; ai[r] = 0.f;
              #pragma unroll
              for (int kk = 0; kk < 5; ++kk) {
                if (2 * kk < W) {
                  const float4 q = Th[r * 5 + kk];
                  const float2 v0 = gp[(r * VUX + 2 * kk) * 16];
                  ar[r] = fmaf(q.x, v0.x, ar[r]); ai[r] = fmaf(q.x, v0.y, ai[r]);
                  ar[r] = fmaf(-q.y, v0.y, ar[r]); ai[r] = fmaf(q.y, v0.x, ai[r]);
                  if (2 * kk + 1 < W) {
                    const float2 v1 = gp[(r * VUX + 2 * kk + 1) * 16];
                    ar[r] = fmaf(q.z, v1.x, ar[r]); ai[r] = fmaf(q.z, v1.y, ai[r]);
                    ar[r] = fmaf(-q.w, v1.y, ar[r]); ai[r] = fmaf(q.w, v1.x, ai[r]);
                  }
                }
              }
            }
            const float pr = (ar[0] + ar[1]) + (ar[2] + ar[3]) + ar[4];
            const float pi = (ai[0] + ai[1]) + (ai[2] + ai[3]) + ai[4];
            const float snd = h ? pr : pi;
            const float rcv = __shfl_xor_sync(0xffffffffu, snd, 16);
            const float mine = (h ? pi : pr) + rcv;
            if (valid) accm[s * 16] = fmaf(1.f, mine, accm[s * 16]);
          }
          nb_arrive(VB_SE + stg, VN_S);   // window stage free
        }
        nb_arrive(VB_WE + (pq & 1), VN_W);   // the plane's tables free
      }
    } else if (warp < VWC + VWB) {
      const int bw = warp - VWC;
      int nprev = 0;
      for (int hz = 0; hz < npl; ++hz) {
        const unsigned rgw = rng[hz];
        const int sa = (int)(rgw & 0xffffu), sb = (int)(rgw >> 16);
        if (sa == sb) continue;
        const unsigned pq = vw_pseq++;
        // ring: planes ≤ pq − 2 released always; plane pq − 1 too when both planes' tables do not fit together
        const unsigned need = (pq >= 1 && nprev + (sb - sa) > VWT) ? pq : (pq >= 2 ? pq - 1 : 0);
        for (; vw_rel < need; ++vw_rel) nb_sync(VB_WE + (vw_rel & 1), VN_W);
        const int rb = pre[hz] % VWT;
        for (int s = sa + bw; s < sb; s += VWB) {
          int slot = rb + s - sa;
          slot -= slot >= VWT ? VWT : 0;
          vt_build(s, hz, Wt + (size_t)slot * 50);
        }
        nb_arrive(VB_WF + (pq & 1), VN_W);
        nprev = sb - sa;
      }
    } else {
      constexpr int NCK = 16 / VCP;   // copies per point and slice
      const int ct = (warp - VWC - VWB) * 32 + lane;   // copy thread
      const int cj = ct % NCK;                         // fixed channel pair of this thread (32 · VWP ≡ 0 mod NCK)
      for (int hz = 0; hz < npl; ++hz) {
        const unsigned rgw = rng[hz];
        if ((rgw & 0xffffu) == (rgw >> 16)) continue;
        int gz = o2 + zlo + hz;
        gz -= gz >= d.nf2 ? d.nf2 : 0;
        const cx* gpl = g.p[0] + (size_t)gz * d.nf1 * d.nf0 * NCH;
        for (int k = 0; k < nsl; ++k) {
          const unsigned tq = vw_tseq++;
          const int stg = (int)(tq % VWS);
          if (tq >= VWS) nb_sync(VB_SE + stg, VN_S);   // slice tq − VWS released
          const int c0k = gb0 + 16 * k, nck = min(16, nchv - c0k);
          float2* dst = Win + (size_t)stg * VSL;
          const cx* src = gpl + c0k;
          if (cj * VCP < nck) {
            for (int p = ct / NCK; p < VNP; p += 32 * VWP / NCK) vw_cp(dst + p * 16 + cj * VCP, src + poff[p] + cj * VCP);
          }
          vw_commit();
          // publish as soon as this thread's copies landed: the consumers work on the previous stages meanwhile (a
          // lag of one slice here would hold slice tq − 1 back until the consumers free slice tq − VWS)
          vw_wait<0>();
          nb_arrive(VB_SF + stg, VN_S);
        }
      }
    }
  }
#elif VTC
  // ---- vtc main loop (research/sample_kernels.md §21): per plane the block builds the active samples' weights once
  // (warp per sample, vt_build) into the union-window table Bw[plane parity][sample][kc][q] (bf16 hi / lo of
  // (Re w, Im w), point pairs of 16-point chunks kc in mma B-fragment order), one barrier per plane (double buffer);
  // then warp (channel group cg, chunk split sp) contracts Y[(c, re | im)][s] += Σ_p G[(c, re | im)][p] W[p][s] with
  // mma.m16n8k16 bf16 (rows of an m16 tile: Re of 8 channels, then Im of the same 8; hi·hi + hi·lo + lo·hi, f32
  // accumulators in registers across the planes). Re y = Σ Gr Wr − Gi Wi, Im y = Σ Gr Wi + Gi Wr from the two products.
  {
    const int g8 = lane >> 2, t4 = lane & 3;
    float acc[NT8][2][8];
    #pragma unroll
    for (int nt = 0; nt < NT8; ++nt) {
      #pragma unroll
      for (int mt = 0; mt < 2; ++mt) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) acc[nt][mt][k] = 0.f;
      }
    }
    bool mv[2], okc[2];
    int chm[2];
    int ec[2] = {40, 40}, ey[2] = {40, 40};   // VTC 2: the channel's current grid exponent and the accumulators' unit
    (void)ec; (void)ey;
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      mv[mt] = gbase + 8 * mt < nchv;
      okc[mt] = gbase + 8 * mt + g8 < nchv;
      chm[mt] = okc[mt] ? gbase + 8 * mt + g8 : 0;
    }
    for (int hz = 0; hz < npl; ++hz) {
      const unsigned rg = rng[hz];
      const int ra = (int)(rg & 0xffffu), rb = (int)(rg >> 16);
      uint4* Bslot = Bw + (size_t)(hz & 1) * VSCH8 * BWS;
      const int nt0 = ra >> 3, nt1 = (rb + 7) >> 3;
      if (ra < rb) {
        for (int qs = nt0 * 8 + warp; qs < nt1 * 8; qs += NTV / 32) {
          uint4* Bs = Bslot + (size_t)qs * BWS;
          const bool act = qs >= ra && qs < rb;   // warp-uniform
          float4 e0 = make_float4(0.f, 0.f, 0.f, 0.f), e1 = e0;
          if (act) {
            vt_build(qs, hz, reinterpret_cast<float4*>(Bs));
            __syncwarp();
            if (lane < 25) { e0 = reinterpret_cast<const float4*>(Bs)[lane]; e1 = reinterpret_cast<const float4*>(Bs)[lane + 25]; }
            __syncwarp();
          }
          for (int i = lane; i < BWS; i += 32) Bs[i] = make_uint4(0u, 0u, 0u, 0u);
          __syncwarp();
          if (act && lane < 25) {
            const int* Mo = reinterpret_cast<const int*>(S + (size_t)qs * VSW) + VR_M;
            const int ox = Mo[0], oy = Mo[1];
            unsigned short* Bh = reinterpret_cast<unsigned short*>(Bs);
            auto put = [&](int e, float4 q) {
              const int r = e / 5, x = 2 * (e % 5);
              if (r >= W) return;
              #pragma unroll
              for (int u = 0; u < 2; ++u) {
                if (x + u < W) {
                  const int p = (oy + r) * UX + ox + x + u;
                  unsigned hi, lo;
                  VSPLIT((u ? q.z : q.x) * WSC, (u ? q.w : q.y) * WSC, hi, lo);
                  unsigned short* bp = Bh + ((p >> 4) * 8 + ((p & 15) >> 1)) * 8 + (p & 1);
                  bp[0] = (unsigned short)(hi & 0xffffu); bp[2] = (unsigned short)(lo & 0xffffu);
                  bp[4] = (unsigned short)(hi >> 16); bp[6] = (unsigned short)(lo >> 16);
                }
              }
            };
            put(lane, e0);
            put(lane + 25, e1);
          }
        }
      }
      __syncthreads();
      if (ra < rb && gbase < nchv) {   // block- / warp-uniform
        int gz = o2 + zlo + hz;
        gz -= gz >= d.nf2 ? d.nf2 : 0;
        const cx* gpl = g.p[0] + (size_t)gz * d.nf1 * d.nf0 * NCH;
        ec[0] = 40; ec[1] = 40;   // VTC 2: each plane starts from the largest scale
        auto ldA = [&](int kc, float2 (&v)[2][4]) {
          const int p0 = kc * 16 + 2 * t4;
          const int2 oa = *reinterpret_cast<const int2*>(poff + p0), ob = *reinterpret_cast<const int2*>(poff + p0 + 8);
          const int o[4] = {oa.x, oa.y, ob.x, ob.y};
          #pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[mt][i] = (okc[mt] && o[i] >= 0) ? __ldg(reinterpret_cast<const float2*>(gpl + chm[mt] + o[i])) : make_float2(0.f, 0.f);
          }
        };
        float2 vn[2][4];
        if (sp < NKC) ldA(sp, vn);
        for (int kc = sp; kc < NKC; kc += NWS) {
          float2 v[2][4];
          #pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) v[mt][i] = vn[mt][i];
          }
          if (kc + NWS < NKC) ldA(kc + NWS, vn);
          unsigned ah[2][4], al[2][4];
          #pragma unroll
          for (int mt = 0; mt < 2; ++mt) {
#if VTC == 2
            // the channel's exponent: max |G| of the chunk over the quad (the 4 lanes of channel g8)
            if (mv[mt]) {
              unsigned mb = 0;
              #pragma unroll
              for (int i = 0; i < 4; ++i) mb = max(mb, max(__float_as_uint(v[mt][i].x) & 0x7fffffffu, __float_as_uint(v[mt][i].y) & 0x7fffffffu));
              mb = max(mb, __shfl_xor_sync(0xffffffffu, mb, 1));
              mb = max(mb, __shfl_xor_sync(0xffffffffu, mb, 2));
              const int eb = (int)(mb >> 23);
              const int en = eb == 0 ? ec[mt] : min(ec[mt], 141 - eb);
              ec[mt] = en;
              if (__any_sync(0xffffffffu, en != ey[mt])) {
                const float f = pow2i(en - ey[mt]);
                #pragma unroll
                for (int nt = 0; nt < NT8; ++nt) {
                  #pragma unroll
                  for (int k = 0; k < 8; ++k) acc[nt][mt][k] *= f;
                }
                ey[mt] = en;
              }
              const float sc = pow2i(en);
              #pragma unroll
              for (int i = 0; i < 4; ++i) { v[mt][i].x *= sc; v[mt][i].y *= sc; }
            }
#endif
            VSPLIT(v[mt][0].x, v[mt][1].x, ah[mt][0], al[mt][0]);
            VSPLIT(v[mt][0].y, v[mt][1].y, ah[mt][1], al[mt][1]);
            VSPLIT(v[mt][2].x, v[mt][3].x, ah[mt][2], al[mt][2]);
            VSPLIT(v[mt][2].y, v[mt][3].y, ah[mt][3], al[mt][3]);
          }
          #pragma unroll
          for (int nt = 0; nt < NT8; ++nt) {
            if (nt < nt0 || nt >= nt1) continue;
            const uint4* Bq = Bslot + (size_t)(nt * 8 + g8) * BWS + kc * 8 + t4;
            const uint4 u0 = Bq[0], u1 = Bq[4];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
              if (!mv[mt]) continue;
              float* cr = acc[nt][mt];
              float* ci = acc[nt][mt] + 4;
              VMMA(cr, al[mt], u0.x, u1.x); VMMA(cr, ah[mt], u0.y, u1.y); VMMA(cr, ah[mt], u0.x, u1.x);
              VMMA(ci, al[mt], u0.z, u1.z); VMMA(ci, ah[mt], u0.w, u1.w); VMMA(ci, ah[mt], u0.z, u1.z);
            }
          }
        }
      }
    }
    __syncthreads();   // Bw free: accb aliases it
    for (int q = threadIdx.x; q < NWC * 2 * APL; q += NTV) accb[q] = 0.f;
    __syncthreads();
    if (gbase < nchv) {
      #pragma unroll
      for (int nt = 0; nt < NT8; ++nt) {
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
          if (!okc[mt]) continue;
          const float* A = acc[nt][mt];
          const float un = VTC == 2 ? pow2i(-ey[mt] - WSE) : 1.f;   // VTC 2: back from 2^(E + 8) units
          #pragma unroll
          for (int jj = 0; jj < 2; ++jj) {
            const int s = nt * 8 + 2 * t4 + jj, c = 8 * mt + g8;
            if (s < cnt) {
              // NWS ≤ 2 chunk splits: 0 + a + b is the same in either order
              atomicAdd(accb + (size_t)(cg * 2) * APL + s * 16 + c, (A[jj] - A[6 + jj]) * un);
              atomicAdd(accb + (size_t)(cg * 2 + 1) * APL + s * 16 + c, (A[4 + jj] + A[2 + jj]) * un);
            }
          }
        }
      }
    }
  }
#else
#if VW == 2
  if (warp >= NWC) {
    // build warps: per plane the active samples (ballot over the sub-chunk), their ring slots, their tables
    const int bw = warp - NWC;
    int nprev = 0, rpos = 0;
    for (int hz = 0; hz < npl; ++hz) {
      const unsigned pq = vw_pseq++;
      const int z = zlo + hz;
      constexpr int NM = (VSCH + 31) / 32;
      unsigned m[NM];
      int n = 0;
      #pragma unroll
      for (int w = 0; w < NM; ++w) {
        const int s = w * 32 + lane;
        bool act = false;
        if (s < cnt) { const int oz = reinterpret_cast<const int*>(S + (size_t)s * VSW)[VR_OZ]; act = oz >= z - (W - 1) && oz <= z; }
        m[w] = __ballot_sync(0xffffffffu, act);
        n += __popc(m[w]);
      }
      // ring: planes ≤ pq − 2 released always; plane pq − 1 too when both planes' tables do not fit together
      const unsigned need = (pq >= 1 && nprev + n > VWT) ? pq : (pq >= 2 ? pq - 1 : 0);
      for (; vw_rel < need; ++vw_rel) nb_sync(VB_WE + (vw_rel & 1), VN_W);
      int r = 0;
      #pragma unroll
      for (int w = 0; w < NM; ++w) {
        unsigned mm = m[w];
        while (mm) {
          const int s = w * 32 + __ffs(mm) - 1;
          mm &= mm - 1;
          int slot = rpos + r;
          slot -= slot >= VWT ? VWT : 0;
          if (bw == 0 && lane == 0) smap[(pq & 1) * VSCH + s] = (unsigned char)slot;
          if (r % VWB == bw) vt_build(s, hz, Wt + (size_t)slot * 50);
          ++r;
        }
      }
      rpos += n;
      rpos -= rpos >= VWT ? VWT : 0;
      nb_arrive(VB_WF + (pq & 1), VN_W);
      nprev = n;
    }
  } else {
  const bool live = gbase < nchv;   // groups past the batch's channels only take part in the hand-over
  for (int hz = 0; hz < npl; ++hz) {
    const unsigned pq = vw_pseq++;
    nb_sync(VB_WF + (pq & 1), VN_W);   // the plane's tables
    if (live) {
#else
  if (gbase < nchv) {   // warp-uniform: groups past the batch's channels have nothing to do
  for (int hz = 0; hz < npl; ++hz) {
#endif
    int gz = o2 + zlo + hz;
    gz -= gz >= d.nf2 ? d.nf2 : 0;
    const cx* gpl = g.p[0] + (size_t)gz * d.nf1 * d.nf0 * NCH + (valid ? ch : 0);
    float2 V[5][HXV];
    // zop: an opaque zero produced right before the loads of each window step, so that the (row, x) address offsets
    // are computed per step instead of being hoisted out of the plane loop and kept live (prototype §12.5)
    auto loadrow = [&](float2 (&Vr)[HXV], int yrel, int zop) {
      int gy = o1 + yrel + zop;
      gy -= gy >= d.nf1 ? d.nf1 : 0;
      const cx* rp = gpl + (size_t)gy * d.nf0 * NCH;
      #pragma unroll
      for (int t = 0; t < HXV; ++t) {
        int gx = o0 + t + zop;
        gx -= gx >= d.nf0 ? d.nf0 : 0;
#ifdef GBFP
        if (valid) { const cx w = gload_bfp(g.p[0], (size_t)(rp + (size_t)gx * NCH - g.p[0]), d); Vr[t] = make_float2(w.x, w.y); } else Vr[t] = make_float2(0.f, 0.f);
#else
        Vr[t] = valid ? __ldg(reinterpret_cast<const float2*>(rp + (size_t)gx * NCH)) : make_float2(0.f, 0.f);
#endif
      }
    };
    #pragma unroll
    for (int st = 0; st < TY; ++st) {
      if (NWS > 1 && (st < stb || st >= stb + TY / NWS)) continue;
      // compiler memory barrier: keeps the window loads of this step from being hoisted above the previous step's
      // sample loops (else every step's rows are live at once: 255 registers and spills)
      int zop;
      asm volatile("mov.u32 %0, 0;" : "=r"(zop) :: "memory");
      if (NWS == 1 ? st == 0 : st == stb) {
        #pragma unroll
        for (int i = 0; i < 5; ++i) loadrow(V[(st + i) % 5], st + 5 * h + i, zop);
      } else {
        loadrow(V[(st + 4) % 5], st + 4 + 5 * h, zop);
      }
      #pragma unroll
      for (int oxv = 0; oxv < NOXV; ++oxv) {
        const int xsh = VXPAD ? 0 : oxv;   // compile-time after unrolling
        for (int ox = (VXPAD ? 0 : oxv); ox < (VXPAD ? TX : oxv + 1); ++ox) {
          const unsigned rg = tab[hz * NCLS + st * TX + ox];
          const int sa = (int)(rg & 0xffffu), sb = (int)(rg >> 16);
          for (int s = sa; s < sb; ++s) {
            const float* P = S + (size_t)s * VSW;
            float pr = 0.f, pi = 0.f;
#if CORR >= 2
            // vt: this (sample, plane)'s weight table, then the lane's 5 rows × W points of its half
#if VW == 2
            const float4* Tb = Wt + (size_t)smap[(pq & 1) * VSCH + s] * 50;
#else
            float4* Tb = tabw + (par ? 50 : 0);
            par ^= 1;
            vt_build(s, hz, Tb);
            __syncwarp();
#endif
            {
              const float4* Th = Tb + h * 25;
              float ar[5], ai[5];
              #pragma unroll
              for (int r = 0; r < 5; ++r) {
                const int slot = (st + r) % 5;
                ar[r] = 0.f; ai[r] = 0.f;
                #pragma unroll
                for (int k = 0; k < 5; ++k) {
                  if (2 * k < W) {
                    const float4 q = Th[r * 5 + k];
                    const float2 v0 = V[slot][2 * k + xsh];
                    ar[r] = fmaf(q.x, v0.x, ar[r]); ai[r] = fmaf(q.x, v0.y, ai[r]);
                    ar[r] = fmaf(-q.y, v0.y, ar[r]); ai[r] = fmaf(q.y, v0.x, ai[r]);
                    if (2 * k + 1 < W) {
                      const float2 v1 = V[slot][2 * k + 1 + xsh];
                      ar[r] = fmaf(q.z, v1.x, ar[r]); ai[r] = fmaf(q.z, v1.y, ai[r]);
                      ar[r] = fmaf(-q.w, v1.y, ar[r]); ai[r] = fmaf(q.w, v1.x, ai[r]);
                    }
                  }
                }
              }
              pr = (ar[0] + ar[1]) + (ar[2] + ar[3]) + ar[4];
              pi = (ai[0] + ai[1]) + (ai[2] + ai[3]) + ai[4];
            }
            const float fz = 1.f;
            (void)P;
#elif CORR == 0
            float kx[KXS];
            #pragma unroll
            for (int k = 0; k < KXS / 4; ++k) { const float4 u = *reinterpret_cast<const float4*>(P + VR_KX + h * KXS + 4 * k); kx[4 * k] = u.x; kx[4 * k + 1] = u.y; kx[4 * k + 2] = u.z; kx[4 * k + 3] = u.w; }
            const int oz = __float_as_int(kx[KXN]);
            const float4 yv = *reinterpret_cast<const float4*>(P + VR_Y4 + 4 * h);
            const float wy[5] = {yv.x, yv.y, yv.z, yv.w, kx[KXN + 1]};
            const float fz = P[VR_WZ + zlo + hz - oz];
            #pragma unroll
            for (int i = 0; i < 5; ++i) {
              const int slot = (st + i) % 5;
              float mr = 0.f, mi = 0.f;
              #pragma unroll
              for (int t = 0; t < KXN; ++t) { mr = fmaf(kx[t], V[slot][t + xsh].x, mr); mi = fmaf(kx[t], V[slot][t + xsh].y, mi); }
              pr = fmaf(wy[i], mr, pr); pi = fmaf(wy[i], mi, pi);
            }
#else
            // first order: per (sample, plane) the z-scaled ΔC factors, per row R0 = φy φz and Q_a, per x-moment a the
            // complex contraction m_a = Σ_t φx^(a)(t) v(t): w·v = R0 m_0 + i (Q_0 m_0 + Q_1 m_1 + Q_2 m_2)
            const int oz = reinterpret_cast<const int*>(P)[VR_KX + KXN];
            const float4 wz = *reinterpret_cast<const float4*>(P + VR_WZ + 4 * (zlo + hz - oz));
            const float4 ca = *reinterpret_cast<const float4*>(P + VR_CC);       // xx, yy, zz, xy
            const float2 cb2 = *reinterpret_cast<const float2*>(P + VR_CC + 4);  // xz, yz
            // Q_0 = cyy φy'' φz + czz φy φz'' + cyz φy' φz';  Q_1 = cxy φy' φz + cxz φy φz';  Q_2 = cxx φy φz
            const float q0y0 = ca.z * wz.z, q0y1 = cb2.y * wz.y, q0y2 = ca.y * wz.x;
            const float q1y0 = cb2.x * wz.y, q1y1 = ca.w * wz.x, q2y0 = ca.x * wz.x;
            float wyb[16];
            #pragma unroll
            for (int k = 0; k < 4; ++k) { const float4 u = *reinterpret_cast<const float4*>(P + VR_WY + h * 16 + 4 * k); wyb[4 * k] = u.x; wyb[4 * k + 1] = u.y; wyb[4 * k + 2] = u.z; wyb[4 * k + 3] = u.w; }
            #pragma unroll
            for (int a = 0; a < 3; ++a) {
              float kx[KXS];
              #pragma unroll
              for (int k = 0; k < KXS / 4; ++k) { const float4 u = *reinterpret_cast<const float4*>(P + VR_KX + a * KXS + 4 * k); kx[4 * k] = u.x; kx[4 * k + 1] = u.y; kx[4 * k + 2] = u.z; kx[4 * k + 3] = u.w; }
              #pragma unroll
              for (int i = 0; i < 5; ++i) {
                const int slot = (st + i) % 5;
                float mr = 0.f, mi = 0.f;
                #pragma unroll
                for (int t = 0; t < KXN; ++t) { mr = fmaf(kx[t], V[slot][t + xsh].x, mr); mi = fmaf(kx[t], V[slot][t + xsh].y, mi); }
                const float y0 = wyb[i], y1 = wyb[5 + i], y2 = wyb[10 + i];
                float Q;
                if (a == 0) {
                  const float R = y0 * wz.x;
                  Q = fmaf(q0y0, y0, fmaf(q0y1, y1, q0y2 * y2));
                  pr = fmaf(R, mr, pr); pi = fmaf(R, mi, pi);
                } else if (a == 1) {
                  Q = fmaf(q1y0, y0, q1y1 * y1);
                } else {
                  Q = q2y0 * y0;
                }
                pr = fmaf(-Q, mi, pr); pi = fmaf(Q, mr, pi);
              }
            }
            const float fz = 1.f;
#endif
            const float snd = h ? pr : pi;
            const float rcv = __shfl_xor_sync(0xffffffffu, snd, 16);
            const float mine = (h ? pi : pr) + rcv;
            if (valid) accm[s * 16] = fmaf(fz, mine, accm[s * 16]);
          }
        }
      }
    }
#if VW == 2
    }
    nb_arrive(VB_WE + (pq & 1), VN_W);   // the plane's tables free
#endif
  }
  }
#endif   // VTC
  __syncthreads();
  // ---- epilogue: y[c0 + coil][j] += Σ_l β e^{−iφ0} a_l(τ) acc[coil · NSEG + l] (one owner per (sample, coil))
  const int base = blockIdx.y * NWC * 16;
  const int bend = min(base + NWC * 16, nchv);
  if (base >= bend) return;
  const int clo = base / NSEG, chi = (bend - 1) / NSEG, ncoil = chi - clo + 1;
  for (int q = threadIdx.x; q < cnt * ncoil; q += NTV) {
    const int s = q / ncoil, coil = clo + q % ncoil;
    const float* P = S + (size_t)s * VSW;
    const unsigned int j = reinterpret_cast<const unsigned int*>(P)[VR_J];
    const int ti = reinterpret_cast<const int*>(P)[VR_F];
    const float fr = P[VR_F + 1], c1 = P[VR_F + 2], s0 = P[VR_F + 3], beta = P[VR_F + 4];
    cx o = cx{0.f, 0.f};
    #pragma unroll
    for (int l = 0; l < NSEG; ++l) {
      const int k = coil * NSEG + l - base, gg = k >> 4, cc = k & 15;
      const float* ap = accb + (size_t)gg * 2 * APL + s * 16 + cc;
      const cx pa = atab[ti * NSEG + l], pb = atab[(ti + 1) * NSEG + l];
      const cx al = cx{fmaf(fr, pb.x - pa.x, pa.x), fmaf(fr, pb.y - pa.y, pa.y)};
      const cx v = cmul(al, cx{c1, -s0});
      const cx w = cx{beta * v.x, beta * v.y};
      const cx r = cmul(w, cx{ap[0], ap[APL]});
      o.x += r.x; o.y += r.y;
    }
    cx* dst = y + (size_t)(c0 + coil) * M + j;
    const cx cur = *dst;
    *dst = cx{cur.x + o.x, cur.y + o.y};
  }
#if VSCH < SMAX
  __syncthreads();   // the next sub-chunk overwrites the records and accumulators read above
#endif
  }
}
