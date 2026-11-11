// Multi-image tile kernels of the segmented physics NUFFT, appended to segnufft.cu (all its helpers and defines) for the
// modules of `SegNufft::forward_multi` / `adjoint_multi`. Defines: KM (images per pass, 1..4), NCB = the multi path's
// coil batch; the other defines as segnufft.cu.
//
// KM images on the same samples share one tile pass: sample lists, records, kernel factors, interpolators a_l(τ),
// phases and the weights on the tile are staged once per (item, coil group); image k reads / writes its own grid slots
// g.p[k * NF + o] and carries a per-sample complex weight w_k (null pointer: 1):
//   forward: y_c(j) += Σ_k w_k(j) Σ_o Σ_l aa_ol(j) Σ_p wt_j(p) G_{k,o,c,l}(p)      (one atomic per (sample, coil, tile))
//   adjoint: G_{k,o,c,l}(p) (+)= Σ_j conj(wt_j(p)) aa_ol(j) conj(w_k(j)) y_c(j)    (thread per (point, offset, coil, image))
// Only nk ≤ KM images are live (the last pass of a larger set).
//
// CORR 3: the second-order correction of CORR 2 plus a per-sample degree-3/4 polynomial phase to first order,
//   c_j(ρ) = Σ_α hc_jα ρ^α,  hc_jα = Σ_q u_jq H'_qα  (u_j per sample as float4, H' in kernel-argument units, host),
// image factor 1 − iψ − ½ψ² − i c_j(ρ): ρ^α ↔ (i ∂_k)^α and ∂_k = −g ∂_z give degree 3 → + c ∂^α φ (real part),
// degree 4 → −i c ∂^α φ (imaginary part). Monomials x^a y^b z^c, degree 3 (10) then 4 (15), a outer, b inner.
#ifndef KM
#define KM 1
#endif
#define NH 25
struct GridsM { cx* p[16]; };          // [k * NF + o]
struct WPtrs { const cx* p[4]; };      // per image, null = weight 1
#define GNM (KM * NF * CPBF * NSEG)
#define GTSM (GNM + ((2 - GNM % 4) + 4) % 4)   // forward tile point stride (cx), ≡ 2 mod 4 as GTS
#define NT_AM (TT3 * NF * CPBA * KM)
struct StageM {   // per-chunk state of the multi kernels (after Stage | list)
  cx ws[CHUNK][KM];
#if CORR >= 3
  float hc[CHUNK][NH];
#endif
};
#define SMEM_BASE ((sizeof(Stage) + 4 * LISTMAX + 15) & ~(size_t)15)
#define SMEM_M ((sizeof(StageM) + 15) & ~(size_t)15)

// image weights of the chunk's samples and (CORR 3) their degree-3/4 coefficients
__device__ __forceinline__ void stage_multi(StageM& T, const int* sl, int base, int nload, int nk, const WPtrs& wp,
                                            const float4* __restrict__ uw, const float* __restrict__ htab) {
  for (int q = threadIdx.x; q < nload * KM; q += blockDim.x) {
    const int s = q / KM, k = q % KM;
    const cx* w = k < nk ? wp.p[k] : nullptr;
    T.ws[s][k] = k >= nk ? cx{0.f, 0.f} : (w ? w[sl[base + s]] : cx{1.f, 0.f});
  }
#if CORR >= 3
  for (int q = threadIdx.x; q < nload * NH; q += blockDim.x) {
    const int s = q / NH, a = q % NH;
    float v = 0.f;
    if (uw && htab) {
      const float4 u = uw[sl[base + s]];
      v = fmaf(u.x, htab[a], fmaf(u.y, htab[NH + a], fmaf(u.z, htab[2 * NH + a], u.w * htab[3 * NH + a])));
    }
    T.hc[s][a] = v;
  }
#endif
}

#if CORR >= 3
// stage_weights (CORR 2 terms, same arithmetic) plus the first-order degree-3 / degree-4 terms
__device__ __forceinline__ void stage_weights_hi(Stage& S, const StageM& T, int nload) {
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
        for (int b = 0; b <= 3 - a; ++b) { R[a] = fmaf(T.hc[s][idx], wy[b] * wz[3 - a - b], R[a]); ++idx; }
      }
      #pragma unroll
      for (int a = 0; a <= 4; ++a) {   // degree 4: −i c ∂^α φ (imaginary)
        #pragma unroll
        for (int b = 0; b <= 4 - a; ++b) { Q[a] = fmaf(-T.hc[s][idx], wy[b] * wz[4 - a - b], Q[a]); ++idx; }
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
#define STAGE_WEIGHTS_M(S, T, n) stage_weights_hi(S, T, n)
#else
#define STAGE_WEIGHTS_M(S, T, n) stage_weights(S, n)
#endif

// forward: samples of one frame interval (+=, y zeroed by the host) <- KM images' grid slots
extern "C" __global__ void __launch_bounds__(NT_F) seg_forward_mt(const float4* __restrict__ rec, const unsigned int* __restrict__ tl, const Item* __restrict__ items,
                                              const cx* __restrict__ atab, int ntab, float t0, float dt, GridsM g, WPtrs wp, int nk,
                                              const float4* __restrict__ uw, const float* __restrict__ htab,
                                              cx* __restrict__ y, int M, int c0, int ncb, Dims d) {
  extern __shared__ __align__(16) unsigned char smem[];   // dynamic: Stage | list | StageM | tile values (host: TileCfg::smem_multi)
  Stage& S = *reinterpret_cast<Stage*>(smem);
  int* sl = reinterpret_cast<int*>(smem + sizeof(Stage));
  StageM& T = *reinterpret_cast<StageM*>(smem + SMEM_BASE);
  cx* gt = reinterpret_cast<cx*>(smem + SMEM_BASE + SMEM_M);
  const Item it = items[blockIdx.x];
  if (it.count == 0) return;
  const int cg = blockIdx.y * CPBF;
  int o0, o1, o2; tile_origin(it.tile, d, o0, o1, o2);
  for (int q = threadIdx.x; q < it.count; q += NT_F) sl[q] = tl[it.first + q];
  for (int q = threadIdx.x; q < TT3 * GNM; q += NT_F) {
    const int p = q / GNM, r = q % GNM;
    const int k = r / (NF * CPBF * NSEG), fo = (r / (CPBF * NSEG)) % NF, c = (r / NSEG) % CPBF, l = r % NSEG;
    const int gx = o0 + p % TT, gy = o1 + (p / TT) % TT, gz = o2 + p / (TT * TT);
    cx v = cx{0.f, 0.f};
    if (k < nk && cg + c < ncb && gx < d.nf0 && gy < d.nf1 && gz < d.nf2) v = g.p[k * NF + fo][(((size_t)gz * d.nf1 + gy) * d.nf0 + gx) * NCH + (cg + c) * NSEG + l];
    gt[p * GTSM + r] = v;
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
  // lanes: pp (tile-point stripe) fastest, then frame offset, then sample, then coil; images looped inside the lane
  const int pp = threadIdx.x % PPF, fo = (threadIdx.x / PPF) % NF, ss = (threadIdx.x / (PPF * NF)) % CHUNK, cl = threadIdx.x / (PPF * NF * CHUNK);
  for (int base = 0; base < it.count; base += CHUNK) {
    const int nload = min(CHUNK, it.count - base);
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < NR; ++r) { const int q = threadIdx.x + r * NT_F; if (q < 3 * nload) S.rec[q / 3][q % 3] = pr[r]; }
    __syncthreads();
    if (base + CHUNK < it.count) prefetch(base + CHUNK);
    stage_samples<1>(S, nload, o0, o1, o2, d, atab, ntab, t0, dt);
    stage_multi(T, sl, base, nload, nk, wp, uw, htab);
    __syncthreads();
    STAGE_WEIGHTS_M(S, T, nload);
    __syncthreads();
    const bool live = ss < nload && cg + cl < ncb;
    cx o = cx{0.f, 0.f};
    if (live) {
      for (int k = 0; k < nk; ++k) {
        cx acc[NSEG];
        #pragma unroll
        for (int l = 0; l < NSEG; ++l) acc[l] = cx{0.f, 0.f};
        for (int p = pp; p < TT3; p += PPF) {
          const cx w = S.wt[ss][p];
          const cx* gv = gt + p * GTSM + ((k * NF + fo) * CPBF + cl) * NSEG;
          #pragma unroll
          for (int l = 0; l < NSEG; ++l) {
            const cx v = gv[l];
            acc[l].x = fmaf(w.x, v.x, acc[l].x); acc[l].x = fmaf(-w.y, v.y, acc[l].x);
            acc[l].y = fmaf(w.x, v.y, acc[l].y); acc[l].y = fmaf(w.y, v.x, acc[l].y);
          }
        }
        cx t = cx{0.f, 0.f};
        #pragma unroll
        for (int l = 0; l < NSEG; ++l) { const cx v = cmul(S.aa[ss][fo][l], acc[l]); t.x += v.x; t.y += v.y; }
        const cx v = cmul(T.ws[ss][k], t);
        o.x += v.x; o.y += v.y;
      }
    }
    #pragma unroll
    for (int sh = PPF * NF / 2; sh > 0; sh >>= 1) { o.x += __shfl_xor_sync(0xffffffffu, o.x, sh); o.y += __shfl_xor_sync(0xffffffffu, o.y, sh); }
    if (live && pp == 0 && fo == 0) {
      cx* dst = y + (size_t)(c0 + cg + cl) * M + sl[base + ss];
      atomicAdd(&dst->x, o.x);
      atomicAdd(&dst->y, o.y);
    }
  }
}

// adjoint: KM images' grid slots (NCH channels of the batch each) <- samples of one frame interval
extern "C" __global__ void __launch_bounds__(NT_AM) seg_adjoint_mt(const float4* __restrict__ rec, const unsigned int* __restrict__ tl, const Item* __restrict__ items,
                                               const cx* __restrict__ atab, int ntab, float t0, float dt, WPtrs wp, int nk,
                                               const float4* __restrict__ uw, const float* __restrict__ htab,
                                               const cx* __restrict__ y, int M, int c0, int ncb, GridsM g, int addmask, Dims d) {
  extern __shared__ __align__(16) unsigned char smem[];   // dynamic: Stage | list | StageM | coil values (host: TileCfg::smem_multi)
  Stage& S = *reinterpret_cast<Stage*>(smem);
  int* sl = reinterpret_cast<int*>(smem + sizeof(Stage));
  StageM& T = *reinterpret_cast<StageM*>(smem + SMEM_BASE);
  cx (*yv)[CPBA] = reinterpret_cast<cx (*)[CPBA]>(smem + SMEM_BASE + SMEM_M);
  const Item it = items[blockIdx.x];
  const int cg = blockIdx.y * CPBA;
  const int p = threadIdx.x % TT3, fo = (threadIdx.x / TT3) % NF, cl = (threadIdx.x / (TT3 * NF)) % CPBA, km = threadIdx.x / (TT3 * NF * CPBA);
  int o0, o1, o2; tile_origin(it.tile, d, o0, o1, o2);
  for (int q = threadIdx.x; q < it.count; q += NT_AM) sl[q] = tl[it.first + q];
  constexpr int NR = (3 * CHUNK + NT_AM - 1) / NT_AM, NY = (CHUNK * CPBA + NT_AM - 1) / NT_AM;
  float4 pr[NR]; cx py[NY];
  auto prefetch = [&](int base) {
    const int nl = min(CHUNK, it.count - base);
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
      const int q = threadIdx.x + r * NT_AM;
      if (q < 3 * nl) pr[r] = rec[(size_t)sl[base + q / 3] * 3 + q % 3];
    }
    #pragma unroll
    for (int r = 0; r < NY; ++r) {
      const int q = threadIdx.x + r * NT_AM, coil = cg + q % CPBA;
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
    for (int r = 0; r < NR; ++r) { const int q = threadIdx.x + r * NT_AM; if (q < 3 * nload) S.rec[q / 3][q % 3] = pr[r]; }
    #pragma unroll
    for (int r = 0; r < NY; ++r) { const int q = threadIdx.x + r * NT_AM; if (q < CHUNK * CPBA) yv[q / CPBA][q % CPBA] = py[r]; }
    __syncthreads();
    if (base + CHUNK < it.count) prefetch(base + CHUNK);
    stage_samples<-1>(S, nload, o0, o1, o2, d, atab, ntab, t0, dt);
    stage_multi(T, sl, base, nload, nk, wp, uw, htab);
    __syncthreads();
    STAGE_WEIGHTS_M(S, T, nload);
    __syncthreads();
    for (int s = 0; s < nload; ++s) {
#if TT3 % 32 == 0
      {   // skip samples whose support box misses this warp's 32 tile points (warp-uniform)
        const int bz = S.box[s] >> 16, zlo = (p & ~31) / (TT * TT), zhi = (p | 31) / (TT * TT);
        if ((bz & 15) > zhi || ((bz >> 4) & 15) < zlo) continue;
      }
#endif
      const cx w = S.wt[s][p], yc = cmulc(T.ws[s][km], yv[s][cl]);   // conj(w_k) y
      const cx u = cx{fmaf(w.x, yc.x, w.y * yc.y), fmaf(w.x, yc.y, -w.y * yc.x)};   // conj(wt) y
      #pragma unroll
      for (int l = 0; l < NSEG; ++l) {
        const cx a = S.aa[s][fo][l];
        acc[l].x = fmaf(u.x, a.x, acc[l].x); acc[l].x = fmaf(-u.y, a.y, acc[l].x);
        acc[l].y = fmaf(u.x, a.y, acc[l].y); acc[l].y = fmaf(u.y, a.x, acc[l].y);
      }
    }
  }
  const int gx = o0 + p % TT, gy = o1 + (p / TT) % TT, gz = o2 + p / (TT * TT);
  // an empty tile adds nothing to a slot holding partial sums
  const bool noop = it.count == 0 && ((addmask >> fo) & 1);
  if (!noop && km < nk && cg + cl < ncb && gx < d.nf0 && gy < d.nf1 && gz < d.nf2) {
    cx* gp = g.p[km * NF + fo] + (((size_t)gz * d.nf1 + gy) * d.nf0 + gx) * NCH + (cg + cl) * NSEG;
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

// ---------------------------------------------------------------- joint water + fat diagonals with a separate fat map
// P_l(v) (1 + ff_v (c_l − 1)): the water diagonal times the voxel's water / fat mix at node l (c_l = fat spectrum at
// τ_l), ff = the frame's fat-fraction map (`SegNufft::forward_fat`). Otherwise seg_pad / seg_crop.
__device__ __forceinline__ cx fatmix(float f, cx c) { return cx{1.f + f * (c.x - 1.f), f * c.y}; }
extern "C" __global__ void seg_pad_ff(const cx* __restrict__ x, const cx* __restrict__ smaps, const cx* __restrict__ segph, const float* __restrict__ ff,
                                      const cx* __restrict__ cl, cx* __restrict__ grid,
                                      const float* __restrict__ i0v, const float* __restrict__ i1v, const float* __restrict__ i2v,
                                      int c0, int ncb, MDims m, Dims d) {
  const size_t nv = (size_t)m.N0 * m.N1 * m.N2;
  const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= nv) return;
  const int a0 = t % m.N0, a1 = (t / m.N0) % m.N1, a2 = t / ((size_t)m.N0 * m.N1);
  const float fac = i0v[a0] * i1v[a1] * i2v[a2];
  const size_t gi = ((size_t)wrap(a2 - m.N2 / 2, d.nf2) * d.nf1 + wrap(a1 - m.N1 / 2, d.nf1)) * d.nf0 + wrap(a0 - m.N0 / 2, d.nf0);
  cx xv = x[t]; xv.x *= fac; xv.y *= fac;
  const float f = ff[t];
  cx ph[NSEG];
  #pragma unroll
  for (int l = 0; l < NSEG; ++l) ph[l] = cmul(cmul(segph[(size_t)l * nv + t], fatmix(f, cl[l])), xv);
  cx* g = grid + gi * NCH;
  for (int c = 0; c < ncb; ++c) {
    const cx s = smaps[(size_t)(c0 + c) * nv + t];
    #pragma unroll
    for (int l = 0; l < NSEG; ++l) g[c * NSEG + l] = cmul(s, ph[l]);
  }
}
extern "C" __global__ void seg_crop_ff(const cx* __restrict__ grid, const cx* __restrict__ smaps, const cx* __restrict__ segph, const float* __restrict__ ff,
                                       const cx* __restrict__ cl, cx* __restrict__ x,
                                       const float* __restrict__ i0v, const float* __restrict__ i1v, const float* __restrict__ i2v,
                                       int c0, int ncb, int accumulate, MDims m, Dims d) {
  const size_t nv = (size_t)m.N0 * m.N1 * m.N2;
  const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= nv) return;
  const int a0 = t % m.N0, a1 = (t / m.N0) % m.N1, a2 = t / ((size_t)m.N0 * m.N1);
  const float fac = i0v[a0] * i1v[a1] * i2v[a2];
  const size_t gi = ((size_t)wrap(a2 - m.N2 / 2, d.nf2) * d.nf1 + wrap(a1 - m.N1 / 2, d.nf1)) * d.nf0 + wrap(a0 - m.N0 / 2, d.nf0);
  const float f = ff[t];
  cx p[NSEG];
  #pragma unroll
  for (int l = 0; l < NSEG; ++l) p[l] = cmul(segph[(size_t)l * nv + t], fatmix(f, cl[l]));
  const cx* g = grid + gi * NCH;
  cx acc = cx{0.f, 0.f};
  for (int c = 0; c < ncb; ++c) {
    cx sl = cx{0.f, 0.f};
    #pragma unroll
    for (int l = 0; l < NSEG; ++l) {
      const cx v = cmulc(p[l], g[c * NSEG + l]);
      sl.x += v.x; sl.y += v.y;
    }
    const cx v = cmulc(smaps[(size_t)(c0 + c) * nv + t], sl);
    acc.x += v.x; acc.y += v.y;
  }
  acc.x *= fac; acc.y *= fac;
  if (accumulate) { const cx o = x[t]; acc.x += o.x; acc.y += o.y; }
  x[t] = acc;
}
