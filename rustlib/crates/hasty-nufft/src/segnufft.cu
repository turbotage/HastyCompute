// Segmented physics NUFFT (3-D, single precision). Defines from the host: W (kernel width), BETA, NSEG (segments),
// CORR (0: none, 1: first-order quadratic-phase correction in the interpolation kernel).
//
// Forward:  y_c(j) = e^{-i phi0_j} * sum_l a_l(tau_j) * sum_u w_j(u) G_{c,l}(u),
//           G_{c,l} = FFT_sigma( pad( S_c * P_l * x / phihat ) ),
//           w_j(u)  = phi(u) + i * sum_pq dc_j,pq * d_p d_q phi(u)   (dc pre-scaled to kernel-argument units)
// Adjoint:  exact transpose (output-driven tile scatter, inverse FFT, fused multi-input crop).
// Grid layout: channel-interleaved, ((iz * nf1 + iy) * nf0 + ix) * NCH + coil_in_batch * NSEG + l, NCH = NCB * NSEG
// (one partial-axis FFT over all channels; pad/crop fuse all channels of a voxel).
// Further defines: NCB (coils per batch), TT (tile edge), CPBA/CHUNKA, CPBF/CHUNKF (coils per block / staged samples).
struct cx { float x, y; };
__device__ __forceinline__ cx cmul(cx a, cx b) { return cx{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
__device__ __forceinline__ cx cmulc(cx a, cx b) { return cx{a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x}; } // conj(a)*b
__device__ __forceinline__ int wrap(int i, int n) { int r = i % n; return r < 0 ? r + n : r; }
struct Dims { int nf0, nf1, nf2, nb0, nb1, nb2; };
struct MDims { int N0, N1, N2, fftord; };

// ES kernel and its derivatives with respect to the kernel argument z ∈ (-1, 1)
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
// derivatives up to order 4 (Faà di Bruno on φ = e^u, u = β(√(1−z²) − 1))
__device__ __forceinline__ void es5(float z, float* o) {
  const float a = 1.f - z * z;
  if (a <= 0.f) { o[0] = o[1] = o[2] = o[3] = o[4] = 0.f; return; }
  const float s = sqrtf(a);
  const float f = expf((float)BETA * (s - 1.f));
  o[0] = f;
  // ES derivatives are unbounded towards the support edge (√ cut-off): drop them for |z| > 0.98, where φ < 1e-7
  // (measured: 4th-derivative interpolation 46× → 1e-3 at σ = 1.25, W = 10; 2nd derivative unaffected)
  if (fabsf(z) > 0.98f) { o[1] = o[2] = o[3] = o[4] = 0.f; return; }
  const float b = (float)BETA, ia = __frcp_rn(a), is = rsqrtf(a), is3 = is * ia, is5 = is3 * ia, is7 = is5 * ia;
  const float u1 = -b * z * is, u2 = -b * is3, u3 = -3.f * b * z * is5, u4 = -3.f * b * (1.f + 4.f * z * z) * is7;
  o[1] = u1 * f;
  o[2] = (u2 + u1 * u1) * f;
  o[3] = (u3 + 3.f * u1 * u2 + u1 * u1 * u1) * f;
  o[4] = (u4 + 4.f * u1 * u3 + 3.f * u2 * u2 + 6.f * u1 * u1 * u2 + u1 * u1 * u1 * u1) * f;
}
#endif
// weights of the W cells starting at i0 = ceil(sc - W/2); z = (cell - sc) * 2/W; returns i0
__device__ __forceinline__ int weights3(float sc, float* f, float* d1, float* d2) {
  const int i0 = (int)ceilf(sc - (float)W * 0.5f);
  #pragma unroll
  for (int t = 0; t < W; ++t) {
    const float z = ((float)(i0 + t) - sc) * (2.f / (float)W);
    es3(z, f[t], d1[t], d2[t]);
  }
  return i0;
}

// interpolator a_l(tau) from a table [ntab][NSEG], linear in tau
__device__ __forceinline__ void interp_a(const cx* __restrict__ atab, int ntab, float t0, float dt, float tau, cx* a) {
  float u = (tau - t0) / dt;
  u = fminf(fmaxf(u, 0.f), (float)(ntab - 1) - 1e-3f);
  const int i = (int)u;
  const float fr = u - (float)i;
  #pragma unroll
  for (int l = 0; l < NSEG; ++l) {
    const cx p = atab[i * NSEG + l], q = atab[(i + 1) * NSEG + l];
    a[l] = cx{p.x + fr * (q.x - p.x), p.y + fr * (q.y - p.y)};
  }
}

#if CORR >= 2
#define NK 5
#elif CORR == 1
#define NK 3
#else
#define NK 1
#endif
#define NE 15
struct SampleKernel {
  int i0, j0, k0;
  float dx[NK][W], dy[NK][W], dz[NK][W];
#if CORR >= 1
  float c[6];
#endif
#if CORR >= 2
  float e[NE];
#endif
};
__device__ __forceinline__ int axis_weights(float sc, float (*o)[W]) {
  const int i0 = (int)ceilf(sc - (float)W * 0.5f);
  #pragma unroll
  for (int t = 0; t < W; ++t) {
    const float z = ((float)(i0 + t) - sc) * (2.f / (float)W);
#if CORR >= 2
    float v[5]; es5(z, v);
    #pragma unroll
    for (int q = 0; q < NK; ++q) o[q][t] = v[q];
#else
    float f, d1, d2; es3(z, f, d1, d2);
    o[0][t] = f;
#if CORR == 1
    o[1][t] = d1; o[2][t] = d2;
#endif
#endif
  }
  return i0;
}
__device__ __forceinline__ void make_kernel(const float* __restrict__ sc, const float* __restrict__ dc, const float* __restrict__ de, int M, int j, SampleKernel& k) {
  k.i0 = axis_weights(sc[j], k.dx);
  k.j0 = axis_weights(sc[(size_t)M + j], k.dy);
  k.k0 = axis_weights(sc[2 * (size_t)M + j], k.dz);
#if CORR >= 1
  #pragma unroll
  for (int q = 0; q < 6; ++q) k.c[q] = dc[(size_t)q * M + j];
#endif
#if CORR >= 2
  #pragma unroll
  for (int q = 0; q < NE; ++q) k.e[q] = de[(size_t)q * M + j];
#endif
}
// per-row factors: real part wr = Σ_a dx_a[tx] R[a], imaginary wi = Σ_a dx_a[tx] Q[a]
struct RowFactors { float R[NK]; float Q[NK]; };
__device__ __forceinline__ void row_factors(const SampleKernel& k, int ty, int tz, RowFactors& r) {
  #pragma unroll
  for (int a = 0; a < NK; ++a) { r.R[a] = 0.f; r.Q[a] = 0.f; }
  r.R[0] = k.dy[0][ty] * k.dz[0][tz];
#if CORR >= 1
  // c = [xx, yy, zz, xy, xz, yz]
  r.Q[0] = k.c[1] * k.dy[2][ty] * k.dz[0][tz] + k.c[2] * k.dy[0][ty] * k.dz[2][tz] + k.c[5] * k.dy[1][ty] * k.dz[1][tz];
  r.Q[1] = k.c[3] * k.dy[1][ty] * k.dz[0][tz] + k.c[4] * k.dy[0][ty] * k.dz[1][tz];
  r.Q[2] = k.c[0] * k.dy[0][ty] * k.dz[0][tz];
#endif
#if CORR >= 2
  // e indexed by (a, b) with a = x-order, b = y-order, c = 4 − a − b; flat index in host order (a outer, b inner)
  int idx = 0;
  #pragma unroll
  for (int a = 0; a <= 4; ++a) {
    #pragma unroll
    for (int b = 0; b <= 4 - a; ++b) {
      r.R[a] += k.e[idx] * k.dy[b][ty] * k.dz[4 - a - b][tz];
      ++idx;
    }
  }
#endif
}
__device__ __forceinline__ void weight(const SampleKernel& k, const RowFactors& r, int tx, float& wr, float& wi) {
  wr = 0.f; wi = 0.f;
  #pragma unroll
  for (int a = 0; a < NK; ++a) { wr += k.dx[a][tx] * r.R[a]; wi += k.dx[a][tx] * r.Q[a]; }
}

#define NCH (NCB * NSEG)
// ---------------------------------------------------------------- fused multi-output pad: one image -> NCH channels (grid zeroed by the host)
extern "C" __global__ void seg_pad(const cx* __restrict__ x, const cx* __restrict__ smaps, const cx* __restrict__ segph, cx* __restrict__ grid,
                                   const float* __restrict__ i0v, const float* __restrict__ i1v, const float* __restrict__ i2v,
                                   int c0, int ncb, MDims m, Dims d) {
  const size_t nv = (size_t)m.N0 * m.N1 * m.N2;
  const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= nv) return;
  const int a0 = t % m.N0, a1 = (t / m.N0) % m.N1, a2 = t / ((size_t)m.N0 * m.N1);
  const float fac = i0v[a0] * i1v[a1] * i2v[a2];
  const size_t gi = ((size_t)wrap(a2 - m.N2 / 2, d.nf2) * d.nf1 + wrap(a1 - m.N1 / 2, d.nf1)) * d.nf0 + wrap(a0 - m.N0 / 2, d.nf0);
  cx xv = x[t]; xv.x *= fac; xv.y *= fac;
  cx ph[NSEG];
  #pragma unroll
  for (int l = 0; l < NSEG; ++l) ph[l] = cmul(segph[(size_t)l * nv + t], xv);
  cx* g = grid + gi * NCH;
  for (int c = 0; c < ncb; ++c) {
    const cx s = smaps[(size_t)(c0 + c) * nv + t];
    #pragma unroll
    for (int l = 0; l < NSEG; ++l) g[c * NSEG + l] = cmul(s, ph[l]);
  }
}

// ---------------------------------------------------------------- fused multi-input crop: NCH channels -> image (accumulate)
extern "C" __global__ void seg_crop(const cx* __restrict__ grid, const cx* __restrict__ smaps, const cx* __restrict__ segph, cx* __restrict__ x,
                                    const float* __restrict__ i0v, const float* __restrict__ i1v, const float* __restrict__ i2v,
                                    int c0, int ncb, int accumulate, MDims m, Dims d) {
  const size_t nv = (size_t)m.N0 * m.N1 * m.N2;
  const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= nv) return;
  const int a0 = t % m.N0, a1 = (t / m.N0) % m.N1, a2 = t / ((size_t)m.N0 * m.N1);
  const float fac = i0v[a0] * i1v[a1] * i2v[a2];
  const size_t gi = ((size_t)wrap(a2 - m.N2 / 2, d.nf2) * d.nf1 + wrap(a1 - m.N1 / 2, d.nf1)) * d.nf0 + wrap(a0 - m.N0 / 2, d.nf0);
  const cx* g = grid + gi * NCH;
  cx acc = cx{0.f, 0.f};
  for (int c = 0; c < ncb; ++c) {
    cx sl = cx{0.f, 0.f};
    #pragma unroll
    for (int l = 0; l < NSEG; ++l) {
      const cx v = cmulc(segph[(size_t)l * nv + t], g[c * NSEG + l]);
      sl.x += v.x; sl.y += v.y;
    }
    const cx v = cmulc(smaps[(size_t)(c0 + c) * nv + t], sl);
    acc.x += v.x; acc.y += v.y;
  }
  acc.x *= fac; acc.y *= fac;
  if (accumulate) { const cx o = x[t]; acc.x += o.x; acc.y += o.y; }
  x[t] = acc;
}

// ---------------------------------------------------------------- tiled gather / scatter on exact tile -> sample lists
// Fine grid split into TT^3 tiles (shared memory is dynamic, sized per launch shape up to the device's opt-in limit).
// The host lists, per tile, every sample whose W^3 support meets it (CSR `tl`, original
// sample indices) and cuts long lists into work items of <= LISTMAX samples. A block is one (item, coil group). It loads
// the item's list into shared memory, then streams CHUNK samples at a time: per-sample records (prefetched into
// registers one chunk ahead) -> kernel factors on the tile -> weight of every (sample, tile point), evaluated once and
// reused for all coils, frame slots and segments.
// Temporal interpolation: samples of one launch lie in one frame interval; frame offset o in [0, NF) reads/writes grid
// slot g.p[o] with coefficient a_l(tau) b_o(u) (B-spline of order NF - 1 in the interval fraction u). NF = 1: static.
//   adjoint: thread per (tile point, frame offset, coil): NSEG accumulators; per-offset mode: store (fresh slot, no
//            split items) / add (read-modify-write; atomics for split items).
//   forward: tile values (offsets x coil group x NSEG) in shared memory, PPF*NF lanes per (sample, coil) split the tile
//            points / offsets and reduce with shuffles, one atomic per (sample, coil, tile).
// Supports are not box-restricted inside a tile (TT = 4, W = 10: ~45% of the MACs hit zero weights): restricting the
// forward loop desynchronises the lanes of a warp (tile values stop being broadcast) and measured slower; the adjoint
// skips whole 32-point slabs outside the box (warp-uniform).
// Sample record (3 x float4): {sc0, sc1, sc2, tau}, {phi0, d_xx, d_yy, d_zz}, {d_xy', d_xz', d_yz', u}, d = ΔC in
// kernel-argument units (off-diagonals pre-doubled), u = fraction of the frame interval. Second-order coefficients
// e_α = −½ Σ_pq d_p d_q (monomial sums).
#ifndef TT
#define TT 4
#endif
#ifndef PPF
#define PPF 2
#endif
#ifndef NF
#define NF 1
#endif
#define TT3 (TT * TT * TT)
#define WTS (TT3 + PPF * NF)             // weight row stride (cx): conflict-free for both MAC layouts
#define GN (NF * CPBF * NSEG)
#define GTS (GN + ((2 - GN % 4) + 4) % 4)   // forward tile point stride (cx), ≡ 2 mod 4: lanes on distinct banks
struct Item { int tile, first, count, split; };
struct Grids { cx* p[4]; };

__device__ __forceinline__ float wrapdiff(float a, int n) { // nearest periodic difference
  if (a > 0.5f * n) a -= n;
  if (a < -0.5f * n) a += n;
  return a;
}
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
// uniform B-spline weights of the NF frames touching interval fraction u (linear: frames i, i+1; cubic: i-1 .. i+2)
__device__ __forceinline__ float bspline(int o, float u) {
#if NF == 1
  return 1.f;
#elif NF == 2
  return o == 0 ? 1.f - u : u;
#elif NF == 4
  const float v = 1.f - u, u2 = u * u, u3 = u2 * u;
  return o == 0 ? v * v * v * (1.f / 6.f) : o == 1 ? (3.f * u3 - 6.f * u2 + 4.f) * (1.f / 6.f) : o == 2 ? (-3.f * u3 + 3.f * u2 + 3.f * u + 1.f) * (1.f / 6.f) : u3 * (1.f / 6.f);
#else
#error "NF must be 1, 2 or 4"
#endif
}
__device__ __forceinline__ void tile_origin(int tile, const Dims& d, int& o0, int& o1, int& o2) {
  const int ntx = (d.nf0 + TT - 1) / TT, nty = (d.nf1 + TT - 1) / TT;
  o0 = (tile % ntx) * TT; o1 = ((tile / ntx) % nty) * TT; o2 = (tile / (ntx * nty)) * TT;
}
struct Stage {   // per-chunk shared state common to both directions
  float4 rec[CHUNK][3];
  float ax[CHUNK][3][NK][TT];
  float ee[CORR >= 2 ? CHUNK : 1][NE];
  int box[CHUNK];        // support box on the tile: lo/hi per axis, 4 bits each (x lo, x hi, y lo, y hi, z lo, z hi)
  cx aa[CHUNK][NF][NSEG];
  cx wt[CHUNK][WTS];
};
// stage A: kernel factors on the tile, coefficients aa[o][l] = a_l(tau) b_o(u) (forward: e^{-i phi0}; adjoint: conj,
// e^{+i phi0}), support box, second-order coefficients
template <int DIR>
__device__ __forceinline__ void stage_samples(Stage& S, int nload, int o0, int o1, int o2, const Dims& d, const cx* __restrict__ atab, int ntab, float t0, float dt) {
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
    const float uf = S.rec[sm][2].w;
    #pragma unroll
    for (int o = 0; o < NF; ++o) { const float b = bspline(o, uf); S.aa[sm][o][l] = cx{b * v.x, b * v.y}; }
  }
  for (int q = threadIdx.x; q < nload; q += blockDim.x) {
    // support box: cells c with |o + c - sc| < W/2
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
// stage B: weight of every (sample, tile point), p = (pz * TT + py) * TT + px; one task per (sample, row)
__device__ __forceinline__ void stage_weights(Stage& S, int nload) {
  for (int q = threadIdx.x; q < nload * TT * TT; q += blockDim.x) {
    const int s = q / (TT * TT), row = q % (TT * TT), py = row % TT, pz = row / TT;
    float R[NK], Q[NK];
    #pragma unroll
    for (int a = 0; a < NK; ++a) { R[a] = 0.f; Q[a] = 0.f; }
    float wy[NK], wz[NK];
    #pragma unroll
    for (int k = 0; k < NK; ++k) { wy[k] = S.ax[s][1][k][py]; wz[k] = S.ax[s][2][k][pz]; }
    R[0] = wy[0] * wz[0];
#if CORR >= 1
    const float4 r1 = S.rec[s][1], r2 = S.rec[s][2];   // [xx, yy, zz] = r1.yzw, [xy, xz, yz] = r2.xyz
    Q[0] = r1.z * wy[2] * wz[0] + r1.w * wy[0] * wz[2] + r2.z * wy[1] * wz[1];
    Q[1] = r2.x * wy[1] * wz[0] + r2.y * wy[0] * wz[1];
    Q[2] = r1.y * wy[0] * wz[0];
#endif
#if CORR >= 2
    {
      int idx = 0;
      #pragma unroll
      for (int a = 0; a <= 4; ++a) {
        #pragma unroll
        for (int b = 0; b <= 4 - a; ++b) { R[a] += S.ee[s][idx] * wy[b] * wz[4 - a - b]; ++idx; }
      }
    }
#endif
    #pragma unroll
    for (int px = 0; px < TT; ++px) {
      float wr = 0.f, wi = 0.f;
      #pragma unroll
      for (int a = 0; a < NK; ++a) { const float xa = S.ax[s][0][a][px]; wr = fmaf(xa, R[a], wr); wi = fmaf(xa, Q[a], wi); }
      S.wt[s][row * TT + px] = cx{wr, wi};
    }
  }
}
#define NT_A (TT3 * NF * CPBA)
#define NT_F (CHUNK * CPBF * PPF * NF)

// adjoint: grid slots (NCH channels of the batch each) <- samples of one frame interval
extern "C" __global__ void __launch_bounds__(NT_A) seg_adjoint_t(const float4* __restrict__ rec, const unsigned int* __restrict__ tl, const Item* __restrict__ items,
                                             const cx* __restrict__ atab, int ntab, float t0, float dt,
                                             const cx* __restrict__ y, int M, int c0, int ncb, Grids g, int addmask, Dims d) {
  extern __shared__ __align__(16) unsigned char smem[];   // dynamic: Stage | list | coil values (host: TileCfg::smem)
  Stage& S = *reinterpret_cast<Stage*>(smem);
  int* sl = reinterpret_cast<int*>(smem + sizeof(Stage));
  cx (*yv)[CPBA] = reinterpret_cast<cx (*)[CPBA]>(smem + ((sizeof(Stage) + 4 * LISTMAX + 15) & ~(size_t)15));
  const Item it = items[blockIdx.x];
  const int cg = blockIdx.y * CPBA;
  const int p = threadIdx.x % TT3, fo = (threadIdx.x / TT3) % NF, cl = threadIdx.x / (TT3 * NF);
  int o0, o1, o2; tile_origin(it.tile, d, o0, o1, o2);
  for (int q = threadIdx.x; q < it.count; q += NT_A) sl[q] = tl[it.first + q];
  constexpr int NR = (3 * CHUNK + NT_A - 1) / NT_A, NY = (CHUNK * CPBA + NT_A - 1) / NT_A;
  float4 pr[NR]; cx py[NY];
  auto prefetch = [&](int base) {
    const int nl = min(CHUNK, it.count - base);
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
      const int q = threadIdx.x + r * NT_A;
      if (q < 3 * nl) pr[r] = rec[(size_t)sl[base + q / 3] * 3 + q % 3];
    }
    #pragma unroll
    for (int r = 0; r < NY; ++r) {
      const int q = threadIdx.x + r * NT_A, coil = cg + q % CPBA;
      py[r] = cx{0.f, 0.f};
      if (q < nl * CPBA && coil < ncb) py[r] = y[(size_t)(c0 + coil) * M + sl[base + q / CPBA]];
    }
  };
  __syncthreads();
  if (it.count > 0) prefetch(0);
  cx acc[NSEG];
  #pragma unroll
  for (int l = 0; l < NSEG; ++l) acc[l] = cx{0.f, 0.f};
  for (int base = 0; base < it.count; base += CHUNK) {
    const int nload = min(CHUNK, it.count - base);
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < NR; ++r) { const int q = threadIdx.x + r * NT_A; if (q < 3 * nload) S.rec[q / 3][q % 3] = pr[r]; }
    #pragma unroll
    for (int r = 0; r < NY; ++r) { const int q = threadIdx.x + r * NT_A; if (q < CHUNK * CPBA) yv[q / CPBA][q % CPBA] = py[r]; }
    __syncthreads();
    if (base + CHUNK < it.count) prefetch(base + CHUNK);
    stage_samples<-1>(S, nload, o0, o1, o2, d, atab, ntab, t0, dt);
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
      const cx w = S.wt[s][p], yc = yv[s][cl];
      const cx u = cx{fmaf(w.x, yc.x, w.y * yc.y), fmaf(w.x, yc.y, -w.y * yc.x)};   // conj(w) y
      #pragma unroll
      for (int l = 0; l < NSEG; ++l) {   // explicit FMA chains (no fast-math reassociation): 4 FFMA per complex MAC
        const cx a = S.aa[s][fo][l];
        acc[l].x = fmaf(u.x, a.x, acc[l].x); acc[l].x = fmaf(-u.y, a.y, acc[l].x);
        acc[l].y = fmaf(u.x, a.y, acc[l].y); acc[l].y = fmaf(u.y, a.x, acc[l].y);
      }
    }
  }
  const int gx = o0 + p % TT, gy = o1 + (p / TT) % TT, gz = o2 + p / (TT * TT);
  if (cg + cl < ncb && gx < d.nf0 && gy < d.nf1 && gz < d.nf2) {
    cx* gp = g.p[fo] + (((size_t)gz * d.nf1 + gy) * d.nf0 + gx) * NCH + (cg + cl) * NSEG;
    if (it.split) {
      #pragma unroll
      for (int l = 0; l < NSEG; ++l) { atomicAdd(&gp[l].x, acc[l].x); atomicAdd(&gp[l].y, acc[l].y); }
    } else if ((addmask >> fo) & 1) {
      #pragma unroll
      for (int l = 0; l < NSEG; ++l) { const cx v = gp[l]; gp[l] = cx{v.x + acc[l].x, v.y + acc[l].y}; }
    } else {
      #pragma unroll
      for (int l = 0; l < NSEG; ++l) gp[l] = acc[l];
    }
  }
}

// forward: samples of one frame interval (+=, y zeroed by the host) <- grid slots
extern "C" __global__ void __launch_bounds__(NT_F) seg_forward_t(const float4* __restrict__ rec, const unsigned int* __restrict__ tl, const Item* __restrict__ items,
                                             const cx* __restrict__ atab, int ntab, float t0, float dt,
                                             Grids g, cx* __restrict__ y, int M, int c0, int ncb, Dims d) {
  extern __shared__ __align__(16) unsigned char smem[];   // dynamic: Stage | list | tile values (host: TileCfg::smem)
  Stage& S = *reinterpret_cast<Stage*>(smem);
  int* sl = reinterpret_cast<int*>(smem + sizeof(Stage));
  cx* gt = reinterpret_cast<cx*>(smem + ((sizeof(Stage) + 4 * LISTMAX + 15) & ~(size_t)15));
  const Item it = items[blockIdx.x];
  if (it.count == 0) return;
  const int cg = blockIdx.y * CPBF;
  int o0, o1, o2; tile_origin(it.tile, d, o0, o1, o2);
  for (int q = threadIdx.x; q < it.count; q += NT_F) sl[q] = tl[it.first + q];
  for (int q = threadIdx.x; q < TT3 * GN; q += NT_F) {
    const int p = q / GN, fo = q / (CPBF * NSEG) % NF, c = (q / NSEG) % CPBF, l = q % NSEG;
    const int gx = o0 + p % TT, gy = o1 + (p / TT) % TT, gz = o2 + p / (TT * TT);
    cx v = cx{0.f, 0.f};
    if (cg + c < ncb && gx < d.nf0 && gy < d.nf1 && gz < d.nf2) v = g.p[fo][(((size_t)gz * d.nf1 + gy) * d.nf0 + gx) * NCH + (cg + c) * NSEG + l];
    gt[p * GTS + (fo * CPBF + c) * NSEG + l] = v;
  }
  constexpr int NR = (3 * CHUNK + NT_F - 1) / NT_F;
  float4 pr[NR];
  auto prefetch = [&](int base) {
    const int nl = min(CHUNK, it.count - base);
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
      const int q = threadIdx.x + r * NT_F;
      if (q < 3 * nl) pr[r] = rec[(size_t)sl[base + q / 3] * 3 + q % 3];
    }
  };
  __syncthreads();
  prefetch(0);
  // lanes: pp (tile-point stripe) fastest, then frame offset, then sample, then coil
  const int pp = threadIdx.x % PPF, fo = (threadIdx.x / PPF) % NF, ss = (threadIdx.x / (PPF * NF)) % CHUNK, cl = threadIdx.x / (PPF * NF * CHUNK);
  for (int base = 0; base < it.count; base += CHUNK) {
    const int nload = min(CHUNK, it.count - base);
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < NR; ++r) { const int q = threadIdx.x + r * NT_F; if (q < 3 * nload) S.rec[q / 3][q % 3] = pr[r]; }
    __syncthreads();
    if (base + CHUNK < it.count) prefetch(base + CHUNK);
    stage_samples<1>(S, nload, o0, o1, o2, d, atab, ntab, t0, dt);
    __syncthreads();
    stage_weights(S, nload);
    __syncthreads();
    const bool live = ss < nload && cg + cl < ncb;
    cx o = cx{0.f, 0.f};
    if (live) {
      cx acc[NSEG];
      #pragma unroll
      for (int l = 0; l < NSEG; ++l) acc[l] = cx{0.f, 0.f};
      for (int p = pp; p < TT3; p += PPF) {   // lanes of one (sample, coil, offset) read the same tile values (broadcast)
        const cx w = S.wt[ss][p];
        const cx* gv = gt + p * GTS + (fo * CPBF + cl) * NSEG;
        #pragma unroll
        for (int l = 0; l < NSEG; ++l) {
          const cx v = gv[l];
          acc[l].x = fmaf(w.x, v.x, acc[l].x); acc[l].x = fmaf(-w.y, v.y, acc[l].x);
          acc[l].y = fmaf(w.x, v.y, acc[l].y); acc[l].y = fmaf(w.y, v.x, acc[l].y);
        }
      }
      #pragma unroll
      for (int l = 0; l < NSEG; ++l) { const cx v = cmul(S.aa[ss][fo][l], acc[l]); o.x += v.x; o.y += v.y; }
    }
    #pragma unroll
    for (int k = PPF * NF / 2; k > 0; k >>= 1) { o.x += __shfl_xor_sync(0xffffffffu, o.x, k); o.y += __shfl_xor_sync(0xffffffffu, o.y, k); }
    if (live && pp == 0 && fo == 0) {
      cx* dst = y + (size_t)(c0 + cg + cl) * M + sl[base + ss];
      atomicAdd(&dst->x, o.x);
      atomicAdd(&dst->y, o.y);
    }
  }
}
