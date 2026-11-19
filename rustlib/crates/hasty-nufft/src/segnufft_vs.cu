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
// CORR ≥ 2: not implemented here (hook below; the order-2 layout is being redesigned in the prototype, §12.9 item 3).
//
// Tables (host): per knot b (static: one list) the samples of the knot's window, grouped by home tile (support corner
// i0 = ceil(sc − W/2), wrapped; tile floor(i0 / T) per axis), cut into items of ≤ SMAX samples in (oz, oy, ox) order
// and re-sorted inside an item by class (oy · TX + ox) then oz; entries key << 25 | j (knot engines) or j (static), j
// the ORIGINAL sample index of the records. Item = {tile, first entry, count, zlo | zhi << 16} (tile-relative oz range).
struct cx { float x, y; };
__device__ __forceinline__ cx cmul(cx a, cx b) { return cx{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
struct Dims { int nf0, nf1, nf2, nb0, nb1, nb2; };
struct Grids { cx* p[4]; };
struct VsItem { int tile, first, count, zr; };
#define NCH (NCB * NSEG)
__device__ __forceinline__ int wrapi(int i, int n) { int r = i % n; return r < 0 ? r + n : r; }

// kernel window and its first two z-derivatives (copies of segnufft.cu es3, ES and KB)
#ifdef KER_KB
__device__ __forceinline__ void es3(float z, float& f, float& d1, float& d2) { float o[3]; kb_eval<3>(z, o); f = o[0]; d1 = o[1]; d2 = o[2]; }
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
#endif

#if CORR >= 2
#error "vs forward: correction order >= 2 is not implemented in the engine (hook: research/sample_kernels.md §12.9 item 3 / §14)"
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
#define NTV (NWC * NWS * 32)
#ifndef VXPAD
#define VXPAD 0
#endif
#if VXPAD
#define KXN HXV
#else
#define KXN W
#endif
#define W4 ((W + 3) / 4 * 4)
#if CORR == 0
#define KXS (((KXN + 2) + 3) / 4 * 4)   // x weights, oz (int bits), φy(5h + 4)
#define VR_KX 0                         // [2][KXS]: the x row duplicated per half
#define VR_Y4 (2 * KXS)                 // [2][4]: φy(5h .. 5h + 3)
#define VR_WZ (VR_Y4 + 8)               // [W]: φz
#define VR_M (VR_WZ + W4)
#else
#define KXS (((KXN + 1) + 3) / 4 * 4)   // [3][KXS]: φx, φx', φx''; row 0 slot KXN: oz (int bits)
#define VR_KX 0
#define VR_WY (3 * KXS)                 // [2][16]: wy_k(5h + i) at k * 5 + i (0 beyond the support)
#define VR_WZ (VR_WY + 32)              // [W][4]: wz_k(t) at t * 4 + k
#define VR_CC (VR_WZ + 4 * W)           // ΔC: xx, yy, zz, xy, xz, yz (kernel-argument units, off-diagonals doubled)
#define VR_M (VR_CC + 8)
#endif
#define VR_F (VR_M + 2)                 // M: ox, oy (int); F: table index (int), fraction, cos φ0, sin φ0, β
#define VR_J (VR_F + 5)                 // original sample index (int)
#define VR_SC (VR_J + 1)                // grid coordinates (staging only)
#define VSW ((VR_SC + 3 + 3) / 4 * 4)
#define APL (SMAX * 16 + 16)            // accumulator plane stride (re / im planes 16 banks apart)
#define NOXV (VXPAD ? 1 : TX)

extern "C" __global__ void __launch_bounds__(NTV) seg_forward_vs(const float4* __restrict__ rec, const unsigned int* __restrict__ vl, const VsItem* __restrict__ items,
                                                                 const cx* __restrict__ atab, int ntab, float t0, float dt, Grids g, int b, int nint, int nri,
                                                                 cx* __restrict__ y, int M, int c0, int ncb, Dims d) {
  extern __shared__ __align__(16) unsigned char smem[];
  float* S = reinterpret_cast<float*>(smem);
  unsigned int* tab = reinterpret_cast<unsigned int*>(smem + (size_t)SMAX * VSW * 4);
  float* accb = reinterpret_cast<float*>(smem + (size_t)SMAX * VSW * 4 + (((size_t)NPLM * NCLS * 4 + 15) & ~(size_t)15));
  (void)b; (void)nint; (void)nri;
  const VsItem it = items[blockIdx.x];
  const int ntx = (d.nf0 + TX - 1) / TX, nty = (d.nf1 + TY - 1) / TY;
  const int o0 = (it.tile % ntx) * TX, o1 = ((it.tile / ntx) % nty) * TY, o2 = (it.tile / (ntx * nty)) * TZ;
  const int zlo = it.zr & 0xffff, npl = (it.zr >> 16) - zlo + W;
  const int cnt = it.count;
  const int nchv = ncb * NSEG;
  // ---- stage 1, per sample: index, coefficient terms, coordinates
  for (int q = threadIdx.x; q < cnt; q += NTV) {
    float* P = S + (size_t)q * VSW;
    const unsigned int e = vl[it.first + q];
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
#endif
  }
  __syncthreads();
  // ---- stage 2, per (sample, axis, cell): kernel values and derivatives, support offsets on the tile
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
  for (int q = threadIdx.x; q < NWC * 2 * APL; q += NTV) accb[q] = 0.f;
  __syncthreads();
  // ---- per-item table: active range of each class in each plane (samples sorted by (class key, oz))
  for (int q = threadIdx.x; q < NCLS; q += NTV) {
    auto key = [&](int s) { const int* m = reinterpret_cast<const int*>(S + (size_t)s * VSW) + VR_M; return m[1] * TX + m[0]; };
    auto ozs = [&](int s) { return reinterpret_cast<const int*>(S + (size_t)s * VSW)[VR_KX + KXN]; };
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
  // ---- main loop (per warp, no barriers)
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int cg = warp % NWC, sp = warp / NWC;
  const int h = lane >> 4, c = lane & 15;
  const int gbase = (blockIdx.y * NWC + cg) * 16;
  const int ch = gbase + c;
  const bool valid = ch < nchv;
  float* accm = accb + (size_t)(cg * 2 + h) * APL + c;   // half 0: real plane, half 1: imaginary plane
  const int stb = NWS == 1 ? 0 : sp * (TY / NWS);
  if (gbase < nchv) {   // warp-uniform: groups past the batch's channels have nothing to do
  for (int hz = 0; hz < npl; ++hz) {
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
        Vr[t] = valid ? __ldg(reinterpret_cast<const float2*>(rp + (size_t)gx * NCH)) : make_float2(0.f, 0.f);
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
#if CORR == 0
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
  }
  }
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
}
