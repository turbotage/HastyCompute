// Prototype sample-side kernels of the segmented NUFFT (research/sample_kernels.md). NOT used by the engine: a
// standalone microbenchmark target (hasty-bench `proto_gather`). Exact W^3 supports, no zero-weight MACs, no per-tile
// atomics.
//
// Forward  y[c][j] = sum_l a_jl sum_{p in supp(j)} w_j(p) G[p][c * NSEG + l]          (a_jl includes e^{-i phi0_j})
// Adjoint  G[p][c * NSEG + l] += sum_j conj(w_j(p)) conj(a_jl) y[c][j]
// Grid: channel-interleaved, G[((z * nf1 + y) * nf0 + x) * NCH + ch], NCH = NCB * NSEG (as segnufft.cu).
// Weights: CORR 0: real separable phi(zx) phi(zy) phi(zz); CORR 2: the second-order corrected complex weights of
// segnufft.cu (stage_weights; same arithmetic).
//
// Home tile: sample j with support corner i0 = ceil(sc - W/2) (per axis, wrapped) belongs to tile floor(i0 / TT); its
// support lies in the tile's halo box [TT * t, TT * t + HW), HW = TT + W - 1. Work item = (tile, first, count) over
// samples sorted by home tile (count <= SMAX).
//
// Defines: W, BETA, CORR (0|2), NCB, NSEG, TT, CH (channels per chunk, divides NCH), TPS (threads per sample, divides
// CH), NT (threads per block), SMAX (samples per item), HALO (forward: 1 = halo staged in shared memory, 0 = read the
// grid through L1/L2), PADC (cx padding of the halo point stride).
struct cx { float x, y; };
__device__ __forceinline__ cx cmul(cx a, cx b) { return cx{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
__device__ __forceinline__ cx ldcx(const cx* p) { const float2 v = __ldg(reinterpret_cast<const float2*>(p)); return cx{v.x, v.y}; }

#ifndef PADC
#define PADC 0
#endif
#ifndef HALO
#define HALO 1
#endif
#define NCH (NCB * NSEG)
#define HW (TT + W - 1)
#define FC (CH / TPS)
#define CHS (CH + PADC)
#ifndef SB
#define SB 1
#endif
#ifndef RX
#define RX 1
#endif
#define WX (W + RX - 1)
#if HALO && TT % RX != 0
#error "halo units need TT % RX == 0"
#endif
#if CH % TPS != 0
#error "TPS must divide CH"
#endif
#if NCH % CH != 0
#error "CH must divide NCH"
#endif
#if CORR >= 2
#define NK 5
#else
#define NK 1
#endif
// one thread owns whole coils (all NSEG segments of its channels): plain stores, else one atomic per (sample, coil
// piece)
#define OWN ((FC % NSEG == 0) && (CH % NSEG == 0))

// ES kernel and derivatives (copies of segnufft.cu es3 / es5)
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
__device__ __forceinline__ void kern(float z, float* o) {
#if CORR >= 2
  es5(z, o);
#else
  const float a = 1.f - z * z;
  o[0] = a <= 0.f ? 0.f : expf((float)BETA * (sqrtf(a) - 1.f));
#endif
}

struct Item { int tile, first, count, ufirst; };   // ufirst: first work unit (items[n] is a sentinel)
struct Dims { int nf0, nf1, nf2; };

// per-sample shared record, flat float words (stride SW odd: lanes of different samples on distinct banks)
#define K_OFF 0
#define O_OFF (3 * NK * W)
#define D_OFF (O_OFF + 3)
#define E_OFF (D_OFF + (CORR >= 2 ? 6 : 0))
#define A_OFF (E_OFF + (CORR >= 2 ? 15 : 0))
// PS_ACFG 1 (ps only): the segment coefficients stay in global memory (read once per (sample, segment) at the output),
// shrinking the record by 2 NSEG words (more resident blocks when NSEG is large)
#ifndef PS_ACFG
#define PS_ACFG 0
#endif
#if defined(PS) && PS_ACFG
#define SW0 A_OFF
#else
#define SW0 (A_OFF + 2 * NSEG)
#endif
#define SW (SW0 | 1)
#define KX(P, ax, k, t) (P)[K_OFF + ((ax) * NK + (k)) * W + (t)]

__device__ __forceinline__ int wrapi(int i, int n) { int r = i % n; return r < 0 ? r + n : r; }
__device__ __forceinline__ void tile_org(int tile, const Dims& d, int& o0, int& o1, int& o2) {
  const int ntx = (d.nf0 + TT - 1) / TT, nty = (d.nf1 + TT - 1) / TT;
  o0 = (tile % ntx) * TT; o1 = ((tile / ntx) % nty) * TT; o2 = (tile / (ntx * nty)) * TT;
}

// stage the item's samples: kernel values per axis, support offsets (in the halo with HALO / USE_HALO, else wrapped
// grid indices), ΔC terms, segment coefficients (conjugated for the adjoint)
template <int CONJ, int REL>
__device__ __forceinline__ void stage(float* S, const float4* __restrict__ pos, const float* __restrict__ dco, const cx* __restrict__ acf, int M,
                                      const Item& it, int o0, int o1, int o2, const Dims& d) {
  for (int q = threadIdx.x; q < it.count * 3 * W; q += NT) {
    const int s = q / (3 * W), a = (q / W) % 3, t = q % W;
    const float4 r = pos[it.first + s];
    const float sc = a == 0 ? r.x : (a == 1 ? r.y : r.z);
    const int i0 = (int)ceilf(sc - (float)W * 0.5f);
    const float z = ((float)(i0 + t) - sc) * (2.f / (float)W);
    float o[5];
    kern(z, o);
    float* P = S + (size_t)s * SW;
    #pragma unroll
    for (int k = 0; k < NK; ++k) KX(P, a, k, t) = o[k];
    if (t == 0) {
      const int n = a == 0 ? d.nf0 : (a == 1 ? d.nf1 : d.nf2), og = a == 0 ? o0 : (a == 1 ? o1 : o2);
      const int wv = wrapi(i0, n);
      reinterpret_cast<int*>(P)[O_OFF + a] = REL ? wv - og : wv;
    }
  }
  for (int q = threadIdx.x; q < it.count * NSEG; q += NT) {
    const int s = q / NSEG, l = q % NSEG;
    const cx v = acf[(size_t)(it.first + s) * NSEG + l];
    float* P = S + (size_t)s * SW;
    P[A_OFF + 2 * l] = v.x; P[A_OFF + 2 * l + 1] = CONJ ? -v.y : v.y;
  }
#if CORR >= 2
  for (int q = threadIdx.x; q < it.count; q += NT) {
    float* P = S + (size_t)q * SW;
    float dd[6];
    #pragma unroll
    for (int p = 0; p < 6; ++p) { dd[p] = dco[(size_t)p * M + it.first + q]; P[D_OFF + p] = dd[p]; }
    const int mx[6] = {2, 0, 0, 1, 1, 0}, my[6] = {0, 2, 0, 1, 0, 1};
    float e[15];
    #pragma unroll
    for (int k = 0; k < 15; ++k) e[k] = 0.f;
    #pragma unroll
    for (int p = 0; p < 6; ++p) {
      #pragma unroll
      for (int pq = 0; pq < 6; ++pq) {
        const int ea = mx[p] + mx[pq], eb = my[p] + my[pq];
        e[ea * (11 - ea) / 2 + eb] = fmaf(-0.5f * dd[p], dd[pq], e[ea * (11 - ea) / 2 + eb]);
      }
    }
    #pragma unroll
    for (int k = 0; k < 15; ++k) P[E_OFF + k] = e[k];
  }
#endif
}

// row factors of the weights of row (ty, tz): CORR 0 r[0] = phi_y phi_z; CORR 2 as segnufft.cu stage_weights
struct Row { float R[NK]; float Q[CORR >= 2 ? 3 : 1]; };
__device__ __forceinline__ void row_factors(const float* P, int ty, int tz, Row& r) {
#if CORR >= 2
  float wy[NK], wz[NK];
  #pragma unroll
  for (int k = 0; k < NK; ++k) { wy[k] = KX(P, 1, k, ty); wz[k] = KX(P, 2, k, tz); }
  #pragma unroll
  for (int a = 0; a < NK; ++a) r.R[a] = 0.f;
  r.R[0] = wy[0] * wz[0];
  const float* c = P + D_OFF;   // [xx, yy, zz, xy, xz, yz]
  r.Q[0] = c[1] * wy[2] * wz[0] + c[2] * wy[0] * wz[2] + c[5] * wy[1] * wz[1];
  r.Q[1] = c[3] * wy[1] * wz[0] + c[4] * wy[0] * wz[1];
  r.Q[2] = c[0] * wy[0] * wz[0];
  int idx = 0;
  #pragma unroll
  for (int a = 0; a <= 4; ++a) {
    #pragma unroll
    for (int b = 0; b <= 4 - a; ++b) { r.R[a] += P[E_OFF + idx] * wy[b] * wz[4 - a - b]; ++idx; }
  }
#else
  r.R[0] = KX(P, 1, 0, ty) * KX(P, 2, 0, tz);
  r.Q[0] = 0.f;
#endif
}

#if HALO
#define FWD_NAME pg_fwd_halo
#else
#define FWD_NAME pg_fwd_gm
#endif

// forward. Block = item; channel chunks of CH outer; thread = (sample, lane q of TPS) with FC channels
// q * FC .. q * FC + FC - 1 of the chunk; samples s = tid / TPS, += NT / TPS.
extern "C" __global__ void __launch_bounds__(NT) FWD_NAME(const float4* __restrict__ pos, const float* __restrict__ dco, const cx* __restrict__ acf,
                                                          const unsigned int* __restrict__ perm, const Item* __restrict__ items, const unsigned int* __restrict__ units,
                                                          const cx* __restrict__ grid, cx* __restrict__ y, int M, Dims d) {
  extern __shared__ __align__(16) unsigned char smem[];
  float* S = reinterpret_cast<float*>(smem);
  cx* halo = reinterpret_cast<cx*>(smem + (((size_t)SMAX * SW * 4 + 15) & ~(size_t)15));
  const Item it = items[blockIdx.x];
  int o0 = 0, o1 = 0, o2 = 0;
#if HALO
  tile_org(it.tile, d, o0, o1, o2);
#endif
  stage<0, HALO>(S, pos, dco, acf, M, it, o0, o1, o2, d);
  const int q = threadIdx.x % TPS;
  for (int ch0 = 0; ch0 < NCH; ch0 += CH) {
#if HALO
    __syncthreads();   // previous chunk's readers done (and the stage)
    // halo [HW^3][CHS] <- grid, runs of CH channels per point (float4 = 2 channels when CH is even)
#if CH % 2 == 0
    for (int qq = threadIdx.x; qq < HW * HW * HW * (CH / 2); qq += NT) {
      const int c2 = qq % (CH / 2), h = qq / (CH / 2);
      const int hx = h % HW, hy = (h / HW) % HW, hz = h / (HW * HW);
      int gx = o0 + hx, gy = o1 + hy, gz = o2 + hz;
      gx -= gx >= d.nf0 ? d.nf0 : 0; gy -= gy >= d.nf1 ? d.nf1 : 0; gz -= gz >= d.nf2 ? d.nf2 : 0;
      const float4 v = *reinterpret_cast<const float4*>(grid + (((size_t)gz * d.nf1 + gy) * d.nf0 + gx) * NCH + ch0 + 2 * c2);
      *reinterpret_cast<float4*>(halo + (size_t)h * CHS + 2 * c2) = v;
    }
#else
    for (int qq = threadIdx.x; qq < HW * HW * HW * CH; qq += NT) {
      const int c = qq % CH, h = qq / CH;
      const int hx = h % HW, hy = (h / HW) % HW, hz = h / (HW * HW);
      int gx = o0 + hx, gy = o1 + hy, gz = o2 + hz;
      gx -= gx >= d.nf0 ? d.nf0 : 0; gy -= gy >= d.nf1 ? d.nf1 : 0; gz -= gz >= d.nf2 ? d.nf2 : 0;
      halo[(size_t)h * CHS + c] = grid[(((size_t)gz * d.nf1 + gy) * d.nf0 + gx) * NCH + ch0 + c];
    }
#endif
#endif
    __syncthreads();
    // work units: up to SB samples with the same (j0, k0) support rows and i0 in one RX-wide bin (host-sorted,
    // consecutive in the item); the unit loops over the union WX = W + RX - 1 of their x supports, every value load
    // serves all SB samples (dummy slots past n carry zero weights)
    const int nun = items[blockIdx.x + 1].ufirst - it.ufirst;
    for (int un = threadIdx.x / TPS; un < nun; un += NT / TPS) {
      const unsigned int ud = units[it.ufirst + un];
      const int s0 = (int)(ud >> 8), n = (int)(ud & 255u);
      const int* off0 = reinterpret_cast<const int*>(S + (size_t)s0 * SW) + O_OFF;
      const int ox = off0[0] - off0[0] % RX, oy = off0[1], oz = off0[2];   // x origin: start of the RX bin
      float kx[SB][NK][WX];
      #pragma unroll
      for (int b = 0; b < SB; ++b) {
        const bool live = b < n;
        const float* Pb = S + (size_t)(s0 + (live ? b : 0)) * SW;
        const int db = live ? reinterpret_cast<const int*>(Pb)[O_OFF] - ox : 0;
        #pragma unroll
        for (int k = 0; k < NK; ++k) {
          #pragma unroll
          for (int t = 0; t < WX; ++t) {
            const int tt = t - db;
            kx[b][k][t] = (live && tt >= 0 && tt < W) ? KX(Pb, 0, k, tt) : 0.f;
          }
        }
      }
      cx acc[SB][FC];
      #pragma unroll
      for (int b = 0; b < SB; ++b) {
        #pragma unroll
        for (int f = 0; f < FC; ++f) acc[b][f] = cx{0.f, 0.f};
      }
#if !HALO
      const bool xwrap = ox + WX > d.nf0;
#endif
      for (int tz = 0; tz < W; ++tz) {
        for (int ty = 0; ty < W; ++ty) {
          Row r[SB];
          #pragma unroll
          for (int b = 0; b < SB; ++b) row_factors(S + (size_t)(s0 + (b < n ? b : 0)) * SW, ty, tz, r[b]);
#if HALO
          const cx* hp = halo + ((size_t)((oz + tz) * HW + (oy + ty)) * HW + ox) * CHS + q * FC;
#else
          int gy = oy + ty, gz = oz + tz;
          gy -= gy >= d.nf1 ? d.nf1 : 0; gz -= gz >= d.nf2 ? d.nf2 : 0;
          const cx* rowp = grid + ((size_t)gz * d.nf1 + gy) * d.nf0 * NCH + ch0 + q * FC;
#endif
          #pragma unroll
          for (int t = 0; t < WX; ++t) {
#if HALO
            const cx* vp = hp + t * CHS;
#else
            int gx = ox + t;
            if (xwrap) gx -= gx >= d.nf0 ? d.nf0 : 0;
            const cx* vp = rowp + (size_t)gx * NCH;
#endif
            cx v[FC];
#if FC % 2 == 0
            #pragma unroll
            for (int f = 0; f < FC; f += 2) {
#if HALO
              const float4 u = *reinterpret_cast<const float4*>(vp + f);
#else
              const float4 u = __ldg(reinterpret_cast<const float4*>(vp + f));
#endif
              v[f] = cx{u.x, u.y}; v[f + 1] = cx{u.z, u.w};
            }
#else
            #pragma unroll
            for (int f = 0; f < FC; ++f) v[f] = vp[f];
#endif
            #pragma unroll
            for (int b = 0; b < SB; ++b) {
              float wr = 0.f, wi = 0.f;
              #pragma unroll
              for (int a = 0; a < NK; ++a) wr = fmaf(kx[b][a][t], r[b].R[a], wr);
#if CORR >= 2
              #pragma unroll
              for (int a = 0; a < 3; ++a) wi = fmaf(kx[b][a][t], r[b].Q[a], wi);
#endif
              #pragma unroll
              for (int f = 0; f < FC; ++f) {
                acc[b][f].x = fmaf(wr, v[f].x, acc[b][f].x);
                acc[b][f].y = fmaf(wr, v[f].y, acc[b][f].y);
#if CORR >= 2
                acc[b][f].x = fmaf(-wi, v[f].y, acc[b][f].x);
                acc[b][f].y = fmaf(wi, v[f].x, acc[b][f].y);
#endif
              }
            }
          }
        }
      }
      // segment combination and output
      const int cfirst = ch0 + q * FC;
      #pragma unroll
      for (int b = 0; b < SB; ++b) {
        if (b >= n) break;
        const float* P = S + (size_t)(s0 + b) * SW;
        const unsigned int j = perm[it.first + s0 + b];
#if OWN
        #pragma unroll
        for (int cc = 0; cc < FC / NSEG; ++cc) {
          cx o = cx{0.f, 0.f};
          #pragma unroll
          for (int l = 0; l < NSEG; ++l) {
            const cx a = cx{P[A_OFF + 2 * l], P[A_OFF + 2 * l + 1]};
            const cx v = cmul(a, acc[b][cc * NSEG + l]);
            o.x += v.x; o.y += v.y;
          }
          y[(size_t)(cfirst / NSEG + cc) * M + j] = o;
        }
#else
        int cur = cfirst / NSEG;
        cx o = cx{0.f, 0.f};
        #pragma unroll
        for (int f = 0; f < FC; ++f) {
          const int ch = cfirst + f, coil = ch / NSEG, l = ch % NSEG;
          if (coil != cur) { cx* dst = y + (size_t)cur * M + j; atomicAdd(&dst->x, o.x); atomicAdd(&dst->y, o.y); o = cx{0.f, 0.f}; cur = coil; }
          const cx a = cx{P[A_OFF + 2 * l], P[A_OFF + 2 * l + 1]};
          const cx v = cmul(a, acc[b][f]);
          o.x += v.x; o.y += v.y;
        }
        cx* dst = y + (size_t)cur * M + j; atomicAdd(&dst->x, o.x); atomicAdd(&dst->y, o.y);
#endif
      }
    }
  }
}

#if HALO
// adjoint, shared-memory spreading (dual of the halo forward): per chunk the block spreads its samples into a
// [HW^3][CHS] accumulator with shared atomics, then adds the halo to the grid with global atomics (grid zeroed first).
extern "C" __global__ void __launch_bounds__(NT) pg_adj_sm(const float4* __restrict__ pos, const float* __restrict__ dco, const cx* __restrict__ acf,
                                                           const unsigned int* __restrict__ perm, const Item* __restrict__ items,
                                                           const cx* __restrict__ y, cx* __restrict__ grid, int M, Dims d) {
  extern __shared__ __align__(16) unsigned char smem[];
  float* S = reinterpret_cast<float*>(smem);
  cx* halo = reinterpret_cast<cx*>(smem + (((size_t)SMAX * SW * 4 + 15) & ~(size_t)15));
  const Item it = items[blockIdx.x];
  int o0, o1, o2;
  tile_org(it.tile, d, o0, o1, o2);
  stage<1, 1>(S, pos, dco, acf, M, it, o0, o1, o2, d);
  const int q = threadIdx.x % TPS;
  for (int ch0 = 0; ch0 < NCH; ch0 += CH) {
    __syncthreads();
    for (int qq = threadIdx.x; qq < HW * HW * HW * CHS; qq += NT) halo[qq] = cx{0.f, 0.f};
    __syncthreads();
    for (int s = threadIdx.x / TPS; s < it.count; s += NT / TPS) {
      const float* P = S + (size_t)s * SW;
      const int* off = reinterpret_cast<const int*>(P) + O_OFF;
      const int ox = off[0], oy = off[1], oz = off[2];
      const unsigned int j = perm[it.first + s];
      cx u[FC];
      #pragma unroll
      for (int f = 0; f < FC; ++f) {
        const int ch = ch0 + q * FC + f, coil = ch / NSEG, l = ch % NSEG;
        u[f] = cmul(cx{P[A_OFF + 2 * l], P[A_OFF + 2 * l + 1]}, y[(size_t)coil * M + j]);
      }
      float kx[NK][W];
      #pragma unroll
      for (int k = 0; k < NK; ++k) {
        #pragma unroll
        for (int t = 0; t < W; ++t) kx[k][t] = KX(P, 0, k, t);
      }
      for (int tz = 0; tz < W; ++tz) {
        for (int ty = 0; ty < W; ++ty) {
          Row r;
          row_factors(P, ty, tz, r);
          cx* hp = halo + ((size_t)((oz + tz) * HW + (oy + ty)) * HW + ox) * CHS + q * FC;
          #pragma unroll
          for (int t = 0; t < W; ++t) {
            float wr = 0.f, wi = 0.f;
            #pragma unroll
            for (int a = 0; a < NK; ++a) wr = fmaf(kx[a][t], r.R[a], wr);
#if CORR >= 2
            #pragma unroll
            for (int a = 0; a < 3; ++a) wi = fmaf(kx[a][t], r.Q[a], wi);
#endif
            #pragma unroll
            for (int f = 0; f < FC; ++f) {   // conj(w) u
              atomicAdd(&hp[t * CHS + f].x, fmaf(wr, u[f].x, wi * u[f].y));
              atomicAdd(&hp[t * CHS + f].y, fmaf(wr, u[f].y, -wi * u[f].x));
            }
          }
        }
      }
    }
    __syncthreads();
    for (int qq = threadIdx.x; qq < HW * HW * HW * CH; qq += NT) {
      const int c = qq % CH, h = qq / CH;
      const int hx = h % HW, hy = (h / HW) % HW, hz = h / (HW * HW);
      int gx = o0 + hx, gy = o1 + hy, gz = o2 + hz;
      gx -= gx >= d.nf0 ? d.nf0 : 0; gy -= gy >= d.nf1 ? d.nf1 : 0; gz -= gz >= d.nf2 ? d.nf2 : 0;
      const cx v = halo[(size_t)h * CHS + c];
      cx* gp = grid + (((size_t)gz * d.nf1 + gy) * d.nf0 + gx) * NCH + ch0 + c;
      atomicAdd(&gp->x, v.x); atomicAdd(&gp->y, v.y);
    }
  }
}
#endif

#ifdef PS
// ---------------------------------------------------------------- design 5 `ps`: plane-streamed home tile
// Home tile TX × TY × TZ (sample j belongs to the tile holding its wrapped support corner); the block streams the
// (TZ + W − 1) halo z-planes of one channel chunk (CHP = NCG · FC channels) through a double-buffered shared plane of
// (TX + W − 1)(TY + W − 1) points (register prefetch of the next plane, one barrier per plane). Lanes = (sample slot,
// channel group), channel group fastest: a warp holds 32 / NCG slots of FSB samples each and the NCG groups' FC
// contiguous channels, so every plane-value load (float4 = 2 channels) is broadcast to the warp's slots. Per sample the
// x loop is the full halo row (unrolled HX, x weights in registers with zeros outside the support), the y loop the
// warp's union of support rows, the z loop the planes where the sample is active; contraction separable: row partial
// Σ_x φx v, then × φy φz. Accumulators stay in registers across the planes: one store per (sample, coil) (atomics only
// when FC holds partial coils). Order 0 (real weights); CORR 2: the complex weights per point (row factors R, Q as
// stage_weights).
#define HXP (TX + W - 1)
#define HYP (TY + W - 1)
#define HZP (TZ + W - 1)
#define NSL (NT / NCG)
#define CHP (NCG * FC)
#define PSMAX (NSL * FSB)
#define PLN (HXP * HYP)
#if FC % 2 != 0
#error "ps: FC must be even (float4 loads)"
#endif
#if 32 % NCG != 0
#error "ps: NCG must divide 32"
#endif
#define POWN (FC % NSEG == 0)
// Timing-only ablation switches (wrong results; research/sample_kernels.md §11), each removes one part of the kernel:
//   PS_NOLOAD   no global plane loads (the prefetch registers hold a per-thread constant; the shared stores and barriers
//               stay, so the planes hold stale values)
//   PS_NOSYNC   no per-plane __syncthreads (races; the chunk-start barrier stays)
//   PS_NOMAC    no MAC block (plane loads, stores and barriers stay; accumulators stay zero)
//   PS_NOSTORE  no final output store (accumulators kept live by a never-true compare)
//   PS_FMAONLY  x loop reads its FC / 2 float4 once per row (at the row start) and reuses them for every x: the same FFMA
//               count with HXP times fewer shared loads
// Tunable PS_PFS (0 | 1): 0 = the next plane is prefetched whole into NPF float4 registers per thread, live across the
// MAC loop; 1 = it is copied global -> shared one float4 at a time, interleaved with the warp's row loop (load at one
// row, store at the next; leftovers after the loop), so only one float4 is live: fewer registers, more resident warps.
// The barrier at the start of plane hz makes plane hz visible and frees buffer (hz + 1) & 1 for the copy of plane hz + 1.
// (The ablation switches above apply to PS_PFS 0.)
#ifndef PS_PFS
#define PS_PFS 0
#endif
// Tunable PS_MINB: minimum resident blocks per SM asked of ptxas (__launch_bounds__ second argument; caps registers at
// 64 K / (PS_MINB NT), may spill); 0 = none
#ifndef PS_MINB
#define PS_MINB 0
#endif
#if PS_MINB > 0
#define PS_LB __launch_bounds__(NT, PS_MINB)
#else
#define PS_LB __launch_bounds__(NT)
#endif
extern "C" __global__ void PS_LB pg_fwd_ps(const float4* __restrict__ pos, const float* __restrict__ dco, const cx* __restrict__ acf,
                                                           const unsigned int* __restrict__ perm, const Item* __restrict__ items, const unsigned int* __restrict__ units,
                                                           const cx* __restrict__ grid, cx* __restrict__ y, int M, Dims d) {
  extern __shared__ __align__(16) unsigned char smem[];
  float* S = reinterpret_cast<float*>(smem);
  cx* pl = reinterpret_cast<cx*>(smem + (((size_t)PSMAX * SW * 4 + 15) & ~(size_t)15));
  const Item it = items[blockIdx.x];
  const int ntx = (d.nf0 + TX - 1) / TX, nty = (d.nf1 + TY - 1) / TY;
  const int o0 = (it.tile % ntx) * TX, o1 = ((it.tile / ntx) % nty) * TY, o2 = (it.tile / (ntx * nty)) * TZ;
  // stage: kernel values per axis, offsets relative to the tile origin (in [0, T)), coefficients
  for (int q = threadIdx.x; q < it.count * 3 * W; q += NT) {
    const int s = q / (3 * W), a = (q / W) % 3, t = q % W;
    const float4 r = pos[it.first + s];
    const float sc = a == 0 ? r.x : (a == 1 ? r.y : r.z);
    const int i0 = (int)ceilf(sc - (float)W * 0.5f);
    const float z = ((float)(i0 + t) - sc) * (2.f / (float)W);
    float o[5];
    kern(z, o);
    float* P = S + (size_t)s * SW;
    #pragma unroll
    for (int k = 0; k < NK; ++k) KX(P, a, k, t) = o[k];
    if (t == 0) {
      const int n = a == 0 ? d.nf0 : (a == 1 ? d.nf1 : d.nf2), og = a == 0 ? o0 : (a == 1 ? o1 : o2);
      reinterpret_cast<int*>(P)[O_OFF + a] = wrapi(i0, n) - og;
    }
  }
#if !PS_ACFG
  for (int q = threadIdx.x; q < it.count * NSEG; q += NT) {
    const int s = q / NSEG, l = q % NSEG;
    const cx v = acf[(size_t)(it.first + s) * NSEG + l];
    float* P = S + (size_t)s * SW;
    P[A_OFF + 2 * l] = v.x; P[A_OFF + 2 * l + 1] = v.y;
  }
#endif
#if CORR >= 2
  for (int q = threadIdx.x; q < it.count; q += NT) {
    float* P = S + (size_t)q * SW;
    float dd[6];
    #pragma unroll
    for (int p = 0; p < 6; ++p) { dd[p] = dco[(size_t)p * M + it.first + q]; P[D_OFF + p] = dd[p]; }
    const int mx[6] = {2, 0, 0, 1, 1, 0}, my[6] = {0, 2, 0, 1, 0, 1};
    float e[15];
    #pragma unroll
    for (int k = 0; k < 15; ++k) e[k] = 0.f;
    #pragma unroll
    for (int p = 0; p < 6; ++p) {
      #pragma unroll
      for (int pq = 0; pq < 6; ++pq) {
        const int ea = mx[p] + mx[pq], eb = my[p] + my[pq];
        e[ea * (11 - ea) / 2 + eb] = fmaf(-0.5f * dd[p], dd[pq], e[ea * (11 - ea) / 2 + eb]);
      }
    }
    #pragma unroll
    for (int k = 0; k < 15; ++k) P[E_OFF + k] = e[k];
  }
#endif
  __syncthreads();
  const int cg = threadIdx.x % NCG, slot = threadIdx.x / NCG;
  // my samples slot · FSB + k: x weights over the halo row (zeros outside the support), y / z offsets
  float kxh[FSB][NK][HXP];
  int oy[FSB], oz[FSB];
  bool live[FSB];
  int ylo = 1 << 20, yhi = -1, zlo = 1 << 20, zhi = -1;
  #pragma unroll
  for (int k = 0; k < FSB; ++k) {
    const int s = slot * FSB + k;
    live[k] = s < it.count;
    const float* P = S + (size_t)(live[k] ? s : 0) * SW;
    const int* off = reinterpret_cast<const int*>(P) + O_OFF;
    const int ox = live[k] ? off[0] : 0;
    oy[k] = live[k] ? off[1] : 0; oz[k] = live[k] ? off[2] : 0;
    #pragma unroll
    for (int a = 0; a < NK; ++a) {
      #pragma unroll
      for (int t = 0; t < HXP; ++t) { const int tt = t - ox; kxh[k][a][t] = (live[k] && tt >= 0 && tt < W) ? KX(P, 0, a, tt) : 0.f; }
    }
    if (live[k]) { ylo = min(ylo, oy[k]); yhi = max(yhi, oy[k]); zlo = min(zlo, oz[k]); zhi = max(zhi, oz[k]); }
  }
  ylo = __reduce_min_sync(0xffffffffu, ylo); yhi = __reduce_max_sync(0xffffffffu, yhi);
  zlo = __reduce_min_sync(0xffffffffu, zlo); zhi = __reduce_max_sync(0xffffffffu, zhi);
  constexpr int NPF = (PLN * (CHP / 2) + NT - 1) / NT;   // float4 of a plane per thread
  for (int ch0 = 0; ch0 < NCH; ch0 += CHP) {
    cx acc[FSB][FC];
    #pragma unroll
    for (int k = 0; k < FSB; ++k) {
      #pragma unroll
      for (int f = 0; f < FC; ++f) acc[k][f] = cx{0.f, 0.f};
    }
#if PS_PFS
    // in-plane offsets (float4 units, wrapped; -1 = no element) of this thread's NPF phases, and the plane base
    int poff[NPF];
    #pragma unroll
    for (int r = 0; r < NPF; ++r) {
      const int e = threadIdx.x + r * NT;
      const int c2 = e % (CHP / 2), h = e / (CHP / 2), hx = h % HXP, hy = h / HXP;
      int gx = o0 + hx, gy = o1 + hy;
      gx -= gx >= d.nf0 ? d.nf0 : 0; gy -= gy >= d.nf1 ? d.nf1 : 0;
      poff[r] = e < PLN * (CHP / 2) ? (gy * d.nf0 + gx) * (NCH / 2) + c2 : -1;
    }
    auto pbase = [&](int hz) -> const float4* {
      int gz = o2 + hz; gz -= gz >= d.nf2 ? d.nf2 : 0;
      return reinterpret_cast<const float4*>(grid + (size_t)gz * d.nf1 * d.nf0 * NCH + ch0);
    };
    __syncthreads();   // previous chunk's readers done with both buffers
    {
      float4* b0 = reinterpret_cast<float4*>(pl);
      const float4* g0 = pbase(0);
      #pragma unroll
      for (int r = 0; r < NPF; ++r) if (poff[r] >= 0) b0[threadIdx.x + r * NT] = __ldg(g0 + poff[r]);
    }
    for (int hz = 0; hz < HZP; ++hz) {
      float4* cur = reinterpret_cast<float4*>(pl + (size_t)(hz & 1) * PLN * CHP);
      float4* nxt = reinterpret_cast<float4*>(pl + (size_t)((hz + 1) & 1) * PLN * CHP);
      __syncthreads();
      // copy of plane hz + 1: phase ph loaded next, pv pending store of phase ph - 1
      const int nph = hz + 1 < HZP ? NPF : 0;
      const float4* gn = pbase(hz + 1 < HZP ? hz + 1 : hz);
      int ph = 0;
      float4 pv = make_float4(0.f, 0.f, 0.f, 0.f);
      bool pend = false;
      auto step = [&]() {
        if (pend) { nxt[threadIdx.x + (ph - 1) * NT] = pv; pend = false; }
        if (ph < nph) {
          #pragma unroll
          for (int r = 0; r < NPF; ++r) if (r == ph && poff[r] >= 0) { pv = __ldg(gn + poff[r]); pend = true; }   // compile-time index
          ++ph;
        }
      };
#else
    float4 pf[NPF];
#endif
#if !PS_PFS
#ifdef PS_NOLOAD
    #pragma unroll
    for (int r = 0; r < NPF; ++r) pf[r] = make_float4(1e-3f * threadIdx.x, 1e-3f * r, -1e-3f * threadIdx.x, 0.5f);
    auto load = [&](int hz) { (void)hz; };
#else
    auto load = [&](int hz) {
      int gz = o2 + hz; gz -= gz >= d.nf2 ? d.nf2 : 0;
      #pragma unroll
      for (int r = 0; r < NPF; ++r) {
        const int e = threadIdx.x + r * NT;
        if (e < PLN * (CHP / 2)) {
          const int c2 = e % (CHP / 2), h = e / (CHP / 2), hx = h % HXP, hy = h / HXP;
          int gx = o0 + hx, gy = o1 + hy;
          gx -= gx >= d.nf0 ? d.nf0 : 0; gy -= gy >= d.nf1 ? d.nf1 : 0;
          pf[r] = __ldg(reinterpret_cast<const float4*>(grid + (((size_t)gz * d.nf1 + gy) * d.nf0 + gx) * NCH + ch0) + c2);
        }
      }
    };
#endif
    __syncthreads();   // previous chunk's readers done with both buffers
    load(0);
    for (int hz = 0; hz < HZP; ++hz) {
      float4* cur = reinterpret_cast<float4*>(pl + (size_t)(hz & 1) * PLN * CHP);
      #pragma unroll
      for (int r = 0; r < NPF; ++r) { const int e = threadIdx.x + r * NT; if (e < PLN * (CHP / 2)) cur[e] = pf[r]; }
#ifndef PS_NOSYNC
      __syncthreads();
#endif
      if (hz + 1 < HZP) load(hz + 1);
#endif
#ifdef PS_NOMAC
      if (false) {
#else
      if (hz >= zlo && hz < zhi + W) {   // warp-uniform: some sample of the warp is active in this plane
#endif
        float rz[FSB][NK];
        #pragma unroll
        for (int k = 0; k < FSB; ++k) {
          const int tz = hz - oz[k];
          const bool on = live[k] && tz >= 0 && tz < W;
          const float* P = S + (size_t)(slot * FSB + k) * SW;
          #pragma unroll
          for (int a = 0; a < NK; ++a) rz[k][a] = on ? KX(P, 2, a, tz) : 0.f;
        }
        const cx* cp = reinterpret_cast<const cx*>(cur) + cg * FC;
        for (int hy = ylo; hy < yhi + W; ++hy) {
#if PS_PFS
          step();
#endif
#if CORR >= 2
          // complex weights per point: row factors R (5), Q (3) of the sample at (ty, tz)
          float R[FSB][NK], Q[FSB][3];
          #pragma unroll
          for (int k = 0; k < FSB; ++k) {
            const int ty = hy - oy[k];
            const bool on = ty >= 0 && ty < W;
            const float* P = S + (size_t)(slot * FSB + k) * SW;
            float wy[NK];
            #pragma unroll
            for (int a = 0; a < NK; ++a) wy[a] = on ? KX(P, 1, a, ty) : 0.f;
            const float* c = P + D_OFF;
            R[k][0] = wy[0] * rz[k][0];
            #pragma unroll
            for (int a = 1; a < NK; ++a) R[k][a] = 0.f;
            Q[k][0] = c[1] * wy[2] * rz[k][0] + c[2] * wy[0] * rz[k][2] + c[5] * wy[1] * rz[k][1];
            Q[k][1] = c[3] * wy[1] * rz[k][0] + c[4] * wy[0] * rz[k][1];
            Q[k][2] = c[0] * wy[0] * rz[k][0];
            int idx = 0;
            #pragma unroll
            for (int a = 0; a <= 4; ++a) {
              #pragma unroll
              for (int b = 0; b <= 4 - a; ++b) { R[k][a] += P[E_OFF + idx] * wy[b] * rz[k][4 - a - b]; ++idx; }
            }
          }
          const cx* rp = cp + (size_t)hy * HXP * CHP;
          #pragma unroll
          for (int t = 0; t < HXP; ++t) {
            cx v[FC];
            #pragma unroll
            for (int f = 0; f < FC; f += 2) { const float4 u = *reinterpret_cast<const float4*>(rp + t * CHP + f); v[f] = cx{u.x, u.y}; v[f + 1] = cx{u.z, u.w}; }
            #pragma unroll
            for (int k = 0; k < FSB; ++k) {
              float wr = 0.f, wi = 0.f;
              #pragma unroll
              for (int a = 0; a < NK; ++a) wr = fmaf(kxh[k][a][t], R[k][a], wr);
              #pragma unroll
              for (int a = 0; a < 3; ++a) wi = fmaf(kxh[k][a][t], Q[k][a], wi);
              #pragma unroll
              for (int f = 0; f < FC; ++f) {
                acc[k][f].x = fmaf(wr, v[f].x, acc[k][f].x); acc[k][f].x = fmaf(-wi, v[f].y, acc[k][f].x);
                acc[k][f].y = fmaf(wr, v[f].y, acc[k][f].y); acc[k][f].y = fmaf(wi, v[f].x, acc[k][f].y);
              }
            }
          }
#else
          cx racc[FSB][FC];
          #pragma unroll
          for (int k = 0; k < FSB; ++k) {
            #pragma unroll
            for (int f = 0; f < FC; ++f) racc[k][f] = cx{0.f, 0.f};
          }
          const cx* rp = cp + (size_t)hy * HXP * CHP;
#ifdef PS_FMAONLY
          float4 ur[FC / 2];
          #pragma unroll
          for (int f = 0; f < FC; f += 2) ur[f / 2] = *reinterpret_cast<const float4*>(rp + f);
#endif
          #pragma unroll
          for (int t = 0; t < HXP; ++t) {
            #pragma unroll
            for (int f = 0; f < FC; f += 2) {
#ifdef PS_FMAONLY
              const float4 u = ur[f / 2];
#else
              const float4 u = *reinterpret_cast<const float4*>(rp + t * CHP + f);
#endif
              #pragma unroll
              for (int k = 0; k < FSB; ++k) {
                racc[k][f].x = fmaf(kxh[k][0][t], u.x, racc[k][f].x); racc[k][f].y = fmaf(kxh[k][0][t], u.y, racc[k][f].y);
                racc[k][f + 1].x = fmaf(kxh[k][0][t], u.z, racc[k][f + 1].x); racc[k][f + 1].y = fmaf(kxh[k][0][t], u.w, racc[k][f + 1].y);
              }
            }
          }
          #pragma unroll
          for (int k = 0; k < FSB; ++k) {
            const int ty = hy - oy[k];
            const float* P = S + (size_t)(slot * FSB + k) * SW;
            const float ryz = (ty >= 0 && ty < W) ? KX(P, 1, 0, ty) * rz[k][0] : 0.f;
            #pragma unroll
            for (int f = 0; f < FC; ++f) { acc[k][f].x = fmaf(ryz, racc[k][f].x, acc[k][f].x); acc[k][f].y = fmaf(ryz, racc[k][f].y, acc[k][f].y); }
          }
#endif
        }
      }
#if PS_PFS
      while (pend || ph < nph) step();   // rest of the copy (all of it in planes where the warp is idle)
#endif
    }
    // segment combination and output
    const int cfirst = ch0 + cg * FC;
#ifdef PS_NOSTORE
    {
      float sacc = 0.f;
      #pragma unroll
      for (int k = 0; k < FSB; ++k) {
        #pragma unroll
        for (int f = 0; f < FC; ++f) sacc += acc[k][f].x + acc[k][f].y;
      }
      if (sacc == 1.2345e-30f) y[threadIdx.x] = cx{sacc, 0.f};   // never true: keeps the accumulators live
      continue;
    }
#endif
    #pragma unroll
    for (int k = 0; k < FSB; ++k) {
      if (!live[k]) continue;
      const int s = slot * FSB + k;
      const float* P = S + (size_t)s * SW;
      const unsigned int j = perm[it.first + s];
#if POWN
      #pragma unroll
      for (int cc = 0; cc < FC / NSEG; ++cc) {
        cx o = cx{0.f, 0.f};
        #pragma unroll
        for (int l = 0; l < NSEG; ++l) {
#if PS_ACFG
          const cx v = cmul(ldcx(acf + (size_t)(it.first + s) * NSEG + l), acc[k][cc * NSEG + l]);
#else
          const cx v = cmul(cx{P[A_OFF + 2 * l], P[A_OFF + 2 * l + 1]}, acc[k][cc * NSEG + l]);
#endif
          o.x += v.x; o.y += v.y;
        }
        y[(size_t)(cfirst / NSEG + cc) * M + j] = o;
      }
#else
      int cur = cfirst / NSEG;
      cx o = cx{0.f, 0.f};
      #pragma unroll
      for (int f = 0; f < FC; ++f) {
        const int ch = cfirst + f, coil = ch / NSEG, l = ch % NSEG;
        if (coil != cur) { cx* dst = y + (size_t)cur * M + j; atomicAdd(&dst->x, o.x); atomicAdd(&dst->y, o.y); o = cx{0.f, 0.f}; cur = coil; }
#if PS_ACFG
        const cx v = cmul(ldcx(acf + (size_t)(it.first + s) * NSEG + l), acc[k][f]);
#else
        const cx v = cmul(cx{P[A_OFF + 2 * l], P[A_OFF + 2 * l + 1]}, acc[k][f]);
#endif
        o.x += v.x; o.y += v.y;
      }
      cx* dst = y + (size_t)cur * M + j; atomicAdd(&dst->x, o.x); atomicAdd(&dst->y, o.y);
#endif
    }
  }
}
#endif

#ifdef VS
// ---------------------------------------------------------------- design 6 `vs`: value-stationary rows
// (research/sample_kernels.md §12). Home tile TX × TY × TZ; item = <= SMAX samples of one tile (cut in z order), sorted
// inside by class key (oy * TX + ox) then oz, so the samples of one class active in a plane form one contiguous range
// (table tab[plane][class], built per item). Warp = 16 channel lanes × 2 row halves (lane = h * 16 + c): lane (h, c)
// holds in registers, for channel c, the values of 5 grid rows × HXV = TX + W − 1 points of the current plane; half 0
// rows st..st+4, half 1 rows st+5..st+9 of the window st = oy of the class being processed (the 10 support rows of
// every sample of class oy). The window slides one row per step (row r in register slot r mod 5, static in the
// unrolled st loop). Per sample (all lanes the same sample: weights are broadcast loads; the x shift ox is a
// compile-time variant, VXPAD 1 = zero-padded weights over HXV instead): x contraction of the lane's 5 rows, y
// combination, z scale; one SHFL combines the halves (half 0 keeps the real, half 1 the imaginary part) and the lane
// read-modify-writes one float of the shared accumulator plane (acc re / im planes, [s][16]). Grid values come straight
// from global memory (no staging, no barriers in the main loop). Warps: NWC channel groups (16 channels each, block
// channel range blockIdx.y * NWC * 16 ...) × NWS splits of the window positions (different classes = different samples:
// no races on the accumulator rows). Epilogue after one barrier: y[coil][j] = Σ_l a_jl acc (plain store when the
// block holds the whole coil, else one atomic per (sample, coil piece)). CORR 2: the same body with x-moments: per
// (sample, plane) E_ab = e_ab wz_{4−a−b}(tz) (+ wz0 for the leading term) and Q coefficients, per row R_a = Σ_b E_ab
// wy_b, Q_a; per derivative a: m_a = Σ_t kx_a(t) v, r += (R_a + i Q_a) m_a.
//
// Tunables of §13 (all default 0 = the §12 kernel):
//   VT 1 (CORR >= 2, VXPAD 0; CORR 3 only with VT): per (sample, plane) the warp builds the 10 × 10 complex weights
//        (lanes 0..29 = row × x group, terms of segnufft_knot.cu stage_weights / stage_weights_hk, wz folded in) into a
//        warp-private double-buffered 800 B table; each lane reads its half's 50 weights as 25 broadcast LDS.128.
//        CORR 3: + per-sample degree-3/4 coefficients hc = Σ_q uw_q htab_q (engine stage_hc_k), trailing kernel args.
//   VTP 1 (with VT): software pipelining: the next sample's table is built (into the other buffer) during this
//        sample's MACs, across class ranges of the plane.
//   VSHR 1: half 1 hands the row it evicts (row st + 4) to half 0 by SHFL; half 0 loads only at the window start
//        (L2 rows per plane and warp TY + 9 instead of 2 (TY + 4)); half 1's loads are predicated.
//   VPF 1: the next window step's row is loaded one step ahead into HXV spare float2 registers.
//   VS2 1 (CORR 0): two samples per inner iteration (two independent bodies).
//   VPT 1: the range table is built in parallel (one thread per (plane, class), binary searches).
//   VMINB n: __launch_bounds__(NTV, n) (register cap).
#define HXV (TX + W - 1)
#define NCLS (TX * TY)
#define NPLM (TZ + W - 1)
#define NWV (NWC * NWS)
#define NTV (NWV * 32)
#ifndef VXPAD
#define VXPAD 0
#endif
#ifndef VT
#define VT 0
#endif
#ifndef VSHR
#define VSHR 0
#endif
#ifndef VTP
#define VTP 0
#endif
#ifndef VPF
#define VPF 0
#endif
#ifndef VS2
#define VS2 0
#endif
#ifndef VPT
#define VPT 0
#endif
#if TY % NWS != 0
#error "vs: NWS must divide TY"
#endif
#if VT && (CORR < 2 || VXPAD)
#error "vs: VT needs CORR >= 2 and VXPAD 0"
#endif
#if CORR >= 3 && !VT
#error "vs: CORR 3 only with VT"
#endif
#if VS2 && CORR >= 2
#error "vs: VS2 is order 0 only"
#endif
#if CORR >= 2
#define VK 5
#else
#define VK 1
#endif
#if VXPAD
#define KXN HXV
#else
#define KXN W
#endif
#define KXS (((KXN + 2) + 3) / 4 * 4)   // x row stride: KXN weights, oz (int bits), o0: φy of row 4 of the half
#define VR_KX 0
#if VT
// kx[a][12] (t padded to 12), wy[i][8] = wy_0..4(i), wz_0..2(i); wz_3..4 [i][2]; e[15], d[6] (+3); CORR 3 hc[25] (+3)
#define VR_WY 60
#define VR_WZH 140
#define VR_E 160
#define VR_D (VR_E + 15)
#if CORR >= 3
#define VR_H 184
#define VSW0 212
#else
#define VSW0 184
#endif
#define VR_M VSW0                        // ox, oy, oz (int), pad
#define VR_OZ (VR_M + 2)
#define VSW (VSW0 + 4)
#else
#if CORR >= 2
#define VR_WY (VK * KXS)                 // [h][28]: wy_b(5h + i) at i * 5 + b
#define VR_WZ (VR_WY + 56)               // [tz][5] (+2 pad)
#define VR_E (VR_WZ + 52)                // e[15], then d[6] (d0 = ev[15]); 16 B aligned
#define VR_D (VR_E + 15)
#define VSW0 (VR_D + 6)
#else
#define VR_Y4 (2 * KXS)                  // [h][4]: φy(5h .. 5h + 3); x row duplicated per half (+ φy(5h + 4))
#define VR_WZ (VR_Y4 + 8)
#define VSW0 (VR_WZ + 10)
#endif
#define VR_M VSW0                        // ox, oy (int)
#define VR_OZ (VR_KX + KXN)
#define VSW ((VSW0 + 2 + 3) / 4 * 4)
#endif
#define APL (SMAX * 16 + 16)             // accumulator plane stride (re / im planes 16 banks apart)
#define NOXV (VXPAD ? 1 : TX)
#if CORR >= 3
#define VHI_ARGS , const float4* __restrict__ uw, const float* __restrict__ htab
#else
#define VHI_ARGS
#endif
// v = pr ? *p : v, as one predicated non-coherent global load (lanes without pr issue no memory request)
__device__ __forceinline__ void ldg2_if(const float2* p, bool pr, float2& v) {
  asm("{\n\t.reg .pred q;\n\t.reg .u64 g;\n\tsetp.ne.b32 q, %3, 0;\n\tcvta.to.global.u64 g, %2;\n\t@q ld.global.nc.v2.f32 {%0, %1}, [g];\n\t}"
      : "+f"(v.x), "+f"(v.y) : "l"(p), "r"((int)pr));
}
// VMINB > 0: __launch_bounds__ minimum resident blocks (register cap 64 K / (VMINB · NTV)), for the small-window
// variants that should fit 12–16 warps per SM
#ifndef VMINB
#define VMINB 0
#endif
#if VMINB > 0
#define VS_LB __launch_bounds__(NTV, VMINB)
#else
#define VS_LB __launch_bounds__(NTV)
#endif
extern "C" __global__ void VS_LB pg_fwd_vs(const float4* __restrict__ pos, const float* __restrict__ dco, const cx* __restrict__ acf,
                                                            const unsigned int* __restrict__ perm, const Item* __restrict__ items, const unsigned int* __restrict__ units,
                                                            const cx* __restrict__ grid, cx* __restrict__ y, int M, Dims d VHI_ARGS) {
  extern __shared__ __align__(16) unsigned char smem[];
  float* S = reinterpret_cast<float*>(smem);
  unsigned int* tab = reinterpret_cast<unsigned int*>(smem + (size_t)SMAX * VSW * 4);
  float* accb = reinterpret_cast<float*>(smem + (size_t)SMAX * VSW * 4 + (((size_t)NPLM * NCLS * 4 + 15) & ~(size_t)15));
#if VT
  float4* tabw_all = reinterpret_cast<float4*>(accb + (size_t)NWC * 2 * APL);   // NWV × 2 × 50 float4
#endif
  const Item it = items[blockIdx.x];
  const int ntx = (d.nf0 + TX - 1) / TX, nty = (d.nf1 + TY - 1) / TY;
  const int o0 = (it.tile % ntx) * TX, o1 = ((it.tile / ntx) % nty) * TY, o2 = (it.tile / (ntx * nty)) * TZ;
  const int zlo = it.ufirst & 0xffff, npl = (it.ufirst >> 16) - zlo + W;
  const int cnt = it.count;
  // ---- stage the sample records
#if VT
  for (int q = threadIdx.x; q < cnt * 3 * W; q += NTV) {
    const int s = q / (3 * W), a = (q / W) % 3, t = q % W;
    const float4 r = pos[it.first + s];
    const float sc = a == 0 ? r.x : (a == 1 ? r.y : r.z);
    const int i0 = (int)ceilf(sc - (float)W * 0.5f);
    const int n = a == 0 ? d.nf0 : (a == 1 ? d.nf1 : d.nf2), og = a == 0 ? o0 : (a == 1 ? o1 : o2);
    const int off = wrapi(i0, n) - og;
    float* P = S + (size_t)s * VSW;
    float o[5];
    kern(((float)(i0 + t) - sc) * (2.f / (float)W), o);
    if (a == 0) {
      #pragma unroll
      for (int k = 0; k < 5; ++k) P[VR_KX + k * 12 + t] = o[k];
      if (t < 2) {
        #pragma unroll
        for (int k = 0; k < 5; ++k) P[VR_KX + k * 12 + 10 + t] = 0.f;
      }
    } else if (a == 1) {
      #pragma unroll
      for (int k = 0; k < 5; ++k) P[VR_WY + t * 8 + k] = o[k];
    } else {
      #pragma unroll
      for (int k = 0; k < 3; ++k) P[VR_WY + t * 8 + 5 + k] = o[k];
      P[VR_WZH + t * 2] = o[3]; P[VR_WZH + t * 2 + 1] = o[4];
    }
    if (t == 0) reinterpret_cast<int*>(P)[VR_M + a] = off;
  }
#if CORR >= 3
  // degree-3/4 coefficients (segnufft_knot.cu stage_hc_k; uw in the sorted sample order here)
  for (int q = threadIdx.x; q < cnt * 28; q += NTV) {
    const int s = q / 28, a = q % 28;
    float v = 0.f;
    if (a < 25) {
      const float4 u = uw[it.first + s];
      v = fmaf(u.x, htab[a], fmaf(u.y, htab[25 + a], fmaf(u.z, htab[50 + a], u.w * htab[75 + a])));
    }
    S[(size_t)s * VSW + VR_H + a] = v;
  }
#endif
#else
  constexpr int WQ = KXN > W ? KXN : W;
  for (int q = threadIdx.x; q < cnt * 3 * WQ; q += NTV) {
    const int s = q / (3 * WQ), a = (q / WQ) % 3, t = q % WQ;
    if (a != 0 && t >= W) continue;
    const float4 r = pos[it.first + s];
    const float sc = a == 0 ? r.x : (a == 1 ? r.y : r.z);
    const int i0 = (int)ceilf(sc - (float)W * 0.5f);
    const int n = a == 0 ? d.nf0 : (a == 1 ? d.nf1 : d.nf2), og = a == 0 ? o0 : (a == 1 ? o1 : o2);
    const int off = wrapi(i0, n) - og;
    float* P = S + (size_t)s * VSW;
    const int tt = (a == 0 && VXPAD) ? t - off : t;
    float o[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
    if (tt >= 0 && tt < W) kern(((float)(i0 + tt) - sc) * (2.f / (float)W), o);
    if (a == 0) {
#if CORR >= 2
      #pragma unroll
      for (int k = 0; k < VK; ++k) P[VR_KX + k * KXS + t] = o[k];   // (oz at slot KXN: written by the z thread)
#else
      P[VR_KX + t] = o[0]; P[VR_KX + KXS + t] = o[0];
#endif
      if (t == 0) reinterpret_cast<int*>(P)[VR_M] = off;
    } else if (a == 1) {
      const int h = t / 5, i = t % 5;
#if CORR >= 2
      #pragma unroll
      for (int k = 0; k < VK; ++k) P[VR_WY + h * 28 + i * 5 + k] = o[k];
#else
      if (i < 4) P[VR_Y4 + h * 4 + i] = o[0]; else P[VR_KX + h * KXS + KXN + 1] = o[0];
#endif
      if (t == 0) reinterpret_cast<int*>(P)[VR_M + 1] = off;
    } else {
#if CORR >= 2
      #pragma unroll
      for (int k = 0; k < VK; ++k) P[VR_WZ + t * 5 + k] = o[k];
#else
      P[VR_WZ + t] = o[0];
#endif
      if (t == 0) {
        reinterpret_cast<int*>(P)[VR_KX + KXN] = off;
#if CORR < 2
        reinterpret_cast<int*>(P)[VR_KX + KXS + KXN] = off;
#endif
      }
    }
  }
#endif
#if CORR >= 2
  for (int q = threadIdx.x; q < cnt; q += NTV) {
    float* P = S + (size_t)q * VSW;
    float dd[6];
    #pragma unroll
    for (int p = 0; p < 6; ++p) { dd[p] = dco[(size_t)p * M + it.first + q]; P[VR_D + p] = dd[p]; }
    const int mx[6] = {2, 0, 0, 1, 1, 0}, my[6] = {0, 2, 0, 1, 0, 1};
    float e[15];
    #pragma unroll
    for (int k = 0; k < 15; ++k) e[k] = 0.f;
    #pragma unroll
    for (int p = 0; p < 6; ++p) {
      #pragma unroll
      for (int pq = 0; pq < 6; ++pq) {
        const int ea = mx[p] + mx[pq], eb = my[p] + my[pq];
        e[ea * (11 - ea) / 2 + eb] = fmaf(-0.5f * dd[p], dd[pq], e[ea * (11 - ea) / 2 + eb]);
      }
    }
    #pragma unroll
    for (int k = 0; k < 15; ++k) P[VR_E + k] = e[k];
#if VT
    P[VR_E + 21] = 0.f; P[VR_E + 22] = 0.f; P[VR_E + 23] = 0.f;
#endif
  }
#endif
  for (int q = threadIdx.x; q < NWC * 2 * APL; q += NTV) accb[q] = 0.f;
  __syncthreads();
  // ---- per-item table: active range of each class in each plane (samples sorted by (class key, oz))
  auto key = [&](int s) { const int* m = reinterpret_cast<const int*>(S + (size_t)s * VSW) + VR_M; return m[1] * TX + m[0]; };
  auto ozs = [&](int s) { return reinterpret_cast<const int*>(S + (size_t)s * VSW)[VR_OZ]; };
#if VPT
  for (int p = threadIdx.x; p < npl * NCLS; p += NTV) {
    const int hz = p / NCLS, q = p % NCLS, z = zlo + hz;
    int lo = 0, hi = cnt;   // class range: first key >= q, first key > q
    while (lo < hi) { const int mid = (lo + hi) >> 1; if (key(mid) >= q) hi = mid; else lo = mid + 1; }
    const int cb = lo;
    hi = cnt;
    while (lo < hi) { const int mid = (lo + hi) >> 1; if (key(mid) > q) hi = mid; else lo = mid + 1; }
    const int ce = lo;
    lo = cb; hi = ce;       // active: oz in [z − (W − 1), z]
    while (lo < hi) { const int mid = (lo + hi) >> 1; if (ozs(mid) >= z - (W - 1)) hi = mid; else lo = mid + 1; }
    const int a = lo;
    hi = ce;
    while (lo < hi) { const int mid = (lo + hi) >> 1; if (ozs(mid) > z) hi = mid; else lo = mid + 1; }
    tab[hz * NCLS + q] = (unsigned)a | ((unsigned)lo << 16);
  }
#else
  if (threadIdx.x < NCLS) {
    const int q = threadIdx.x;
    int cb = 0;
    while (cb < cnt && key(cb) < q) ++cb;
    int ce = cb;
    while (ce < cnt && key(ce) == q) ++ce;
    int a = cb, b = cb;
    for (int hz = 0; hz < npl; ++hz) {
      const int z = zlo + hz;
      while (b < ce && ozs(b) <= z) ++b;
      while (a < b && ozs(a) < z - (W - 1)) ++a;
      tab[hz * NCLS + q] = (unsigned)a | ((unsigned)b << 16);
    }
  }
#endif
  __syncthreads();
  // ---- main loop (per warp, no barriers)
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int cg = warp % NWC, sp = warp / NWC;
  const int h = lane >> 4, c = lane & 15;
  const int ch = (blockIdx.y * NWC + cg) * 16 + c;
  const bool valid = ch < NCH;
  float* accm = accb + (size_t)(cg * 2 + h) * APL + c;   // half 0: real plane, half 1: imaginary plane
  const int stb = NWS == 1 ? 0 : sp * (TY / NWS);
#if VT
  float4* tabw = tabw_all + (size_t)warp * 100;
  const int bi = lane % 10, bg = lane / 10;   // builder lane: row bi, x group bg (lanes 30, 31 idle)
  int par = 0;
#endif
  // combine the two halves' partials of sample s (one SHFL) and add into the accumulator
  auto fin = [&](int s, float pr, float pi, float fz) {
    const float snd = h ? pr : pi;
    const float rcv = __shfl_xor_sync(0xffffffffu, snd, 16);
    const float mine = (h ? pi : pr) + rcv;
    if (valid) accm[s * 16] = fmaf(fz, mine, accm[s * 16]);
  };
#if VT
  // VT: build the (sample s, plane hz) weight table into Tb (all lanes compute; lane (bi, bg) → row bi, x points
  // 4 bg .. 4 bg + 3; lanes 30, 31 and store = false compute on clamped indices and store nothing)
  auto vt_build = [&](int s, int hz, float4* Tb, bool store) {
    const float* P = S + (size_t)s * VSW;
    const int tz = zlo + hz - reinterpret_cast<const int*>(P)[VR_OZ];
    const float4 za = *reinterpret_cast<const float4*>(P + VR_WY + tz * 8 + 4);
    const float2 zb = *reinterpret_cast<const float2*>(P + VR_WZH + tz * 2);
    const float wz[5] = {za.y, za.z, za.w, zb.x, zb.y};
    float ev[24];
    #pragma unroll
    for (int k = 0; k < 6; ++k) { const float4 u = *reinterpret_cast<const float4*>(P + VR_E + 4 * k); ev[4 * k] = u.x; ev[4 * k + 1] = u.y; ev[4 * k + 2] = u.z; ev[4 * k + 3] = u.w; }
    const float* dv = ev + 15;   // [xx, yy, zz, xy, xz, yz]
    float ER[15], EQ[15];
    {
      int idx = 0;
      #pragma unroll
      for (int a = 0; a <= 4; ++a) {
        #pragma unroll
        for (int b = 0; b <= 4 - a; ++b) { ER[idx] = ev[idx] * wz[4 - a - b]; EQ[idx] = 0.f; ++idx; }
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
        for (int b = 0; b <= 3 - a; ++b) { const int id = a * (11 - a) / 2 + b; ER[id] = fmaf(hv[j], wz[3 - a - b], ER[id]); ++j; }
      }
      #pragma unroll
      for (int a = 0; a <= 4; ++a) {   // degree 4: −i c ∂^α φ (imaginary)
        #pragma unroll
        for (int b = 0; b <= 4 - a; ++b) {
          const int id = a * (11 - a) / 2 + b;
          // (zero entries of EQ: a plain product, no fma with a constant 0)
          const bool nz = id == 0 || id == 1 || id == 2 || id == 5 || id == 6 || id == 9;
          EQ[id] = nz ? fmaf(-hv[j], wz[4 - a - b], EQ[id]) : -hv[j] * wz[4 - a - b];
          ++j;
        }
      }
    }
#endif
    const float4 ya = *reinterpret_cast<const float4*>(P + VR_WY + bi * 8);
    const float wy[5] = {ya.x, ya.y, ya.z, ya.w, P[VR_WY + bi * 8 + 4]};
    float R[5], Q[5];
    {
      int idx = 0;
      #pragma unroll
      for (int a = 0; a <= 4; ++a) {
        R[a] = ER[idx] * wy[0];
        #pragma unroll
        for (int b = 1; b <= 4 - a; ++b) R[a] = fmaf(ER[idx + b], wy[b], R[a]);
#if CORR >= 3
        Q[a] = EQ[idx] * wy[0];
        #pragma unroll
        for (int b = 1; b <= 4 - a; ++b) Q[a] = fmaf(EQ[idx + b], wy[b], Q[a]);
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
      for (int j = 0; j < 4; ++j) {
        wr[j] = a == 0 ? kk[j] * R[0] : fmaf(kk[j], R[a], wr[j]);
        if (a < NQ) wi[j] = a == 0 ? kk[j] * Q[0] : fmaf(kk[j], Q[a], wi[j]);
      }
    }
    if (store && lane < 30) {
      Tb[bi * 5 + bg * 2] = make_float4(wr[0], wi[0], wr[1], wi[1]);
      if (bg < 2) Tb[bi * 5 + bg * 2 + 1] = make_float4(wr[2], wi[2], wr[3], wi[3]);
    }
  
  };
  const int qbeg = stb * TX, qend = (NWS == 1 ? TY : stb + TY / NWS) * TX;
  (void)qbeg; (void)qend;
#endif
  for (int hz = 0; hz < npl; ++hz) {
    int gz = o2 + zlo + hz;
    gz -= gz >= d.nf2 ? d.nf2 : 0;
    const cx* gpl = grid + (size_t)gz * d.nf1 * d.nf0 * NCH + (valid ? ch : 0);
    float2 V[5][HXV];
#if VPF
    float2 NX[HXV];
    #pragma unroll
    for (int t = 0; t < HXV; ++t) NX[t] = make_float2(0.f, 0.f);
#endif
#if VT && VTP
    {   // the table of this warp's first active sample in the plane
      int f = -1;
      for (int q = qbeg; q < qend; ++q) { const unsigned r2 = tab[hz * NCLS + q]; if ((r2 & 0xffffu) < (r2 >> 16)) { f = (int)(r2 & 0xffffu); break; } }
      vt_build(f >= 0 ? f : 0, hz, tabw + (par ? 50 : 0), f >= 0);
      __syncwarp();
    }
#endif
    // zop: an opaque zero produced right before the loads of each window step, so that the (row, x) address offsets
    // are computed per step instead of being hoisted out of the plane loop and kept live (they do not depend on the
    // plane: 22 x 13 64-bit offsets, spills)
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
    // the same, only lanes with pr load (others keep Vr); rows past the tile edge wrap as above
    auto loadrow_if = [&](float2 (&Vr)[HXV], int yrel, int zop, bool pr) {
      int gy = o1 + yrel + zop;
      gy -= gy >= d.nf1 ? d.nf1 : 0;
      const cx* rp = gpl + (size_t)gy * d.nf0 * NCH;
      #pragma unroll
      for (int t = 0; t < HXV; ++t) {
        int gx = o0 + t + zop;
        gx -= gx >= d.nf0 ? d.nf0 : 0;
        ldg2_if(reinterpret_cast<const float2*>(rp + (size_t)gx * NCH), pr && valid, Vr[t]);
      }
    };
    (void)loadrow_if;
    #pragma unroll
    for (int st = 0; st < TY; ++st) {
      if (NWS > 1 && (st < stb || st >= stb + TY / NWS)) continue;
      // compiler memory barrier: keeps the window loads of this step from being hoisted above the previous step's
      // sample loops (else every step's rows are live at once: 255 registers and spills)
      int zop;
      asm volatile("mov.u32 %0, 0;" : "=r"(zop) :: "memory");
      const bool wfirst = NWS == 1 ? st == 0 : st == stb;
      if (wfirst) {
        #pragma unroll
        for (int i = 0; i < 5; ++i) loadrow(V[(st + i) % 5], st + 5 * h + i, zop);
      } else {
        const int k = (st + 4) % 5;
#if VSHR
        // half 0 takes row st + 4 from half 1 (which drops it now); half 1 gets row st + 9
        #pragma unroll
        for (int t = 0; t < HXV; ++t) {
          float2 x;
          x.x = __shfl_xor_sync(0xffffffffu, V[k][t].x, 16);
          x.y = __shfl_xor_sync(0xffffffffu, V[k][t].y, 16);
#if VPF
          V[k][t] = h ? NX[t] : x;
#else
          if (!h) V[k][t] = x;
#endif
        }
#if !VPF
        loadrow_if(V[k], st + 9, zop, h != 0);
#endif
#else
#if VPF
        #pragma unroll
        for (int t = 0; t < HXV; ++t) V[k][t] = NX[t];
#else
        loadrow(V[k], st + 4 + 5 * h, zop);
#endif
#endif
      }
#if VPF
      // the next step's row, one step ahead
      if (NWS == 1 ? st + 1 < TY : st + 1 < stb + TY / NWS) {
#if VSHR
        loadrow_if(NX, st + 10, zop, h != 0);
#else
        loadrow(NX, st + 5 + 5 * h, zop);
#endif
      }
#endif
      #pragma unroll
      for (int oxv = 0; oxv < NOXV; ++oxv) {
        const int xsh = VXPAD ? 0 : oxv;   // compile-time after unrolling
        for (int ox = (VXPAD ? 0 : oxv); ox < (VXPAD ? TX : oxv + 1); ++ox) {
          const unsigned rg = tab[hz * NCLS + st * TX + ox];
          const int sa = (int)(rg & 0xffffu), sb = (int)(rg >> 16);
#if CORR < 2
          // order-0 body of sample s: x contraction of the lane's 5 rows, y combination, z factor
          auto body0 = [&](int s, float& pr, float& pi, float& fz) {
            const float* P = S + (size_t)s * VSW;
            pr = 0.f; pi = 0.f;
            float kx[KXS];
            #pragma unroll
            for (int k = 0; k < KXS / 4; ++k) { const float4 u = *reinterpret_cast<const float4*>(P + VR_KX + h * KXS + 4 * k); kx[4 * k] = u.x; kx[4 * k + 1] = u.y; kx[4 * k + 2] = u.z; kx[4 * k + 3] = u.w; }
            const int oz = __float_as_int(kx[KXN]);
            const float4 yv = *reinterpret_cast<const float4*>(P + VR_Y4 + 4 * h);
            const float wy[5] = {yv.x, yv.y, yv.z, yv.w, kx[KXN + 1]};
            fz = P[VR_WZ + zlo + hz - oz];
            #pragma unroll
            for (int i = 0; i < 5; ++i) {
              const int slot = (st + i) % 5;
              float mr = 0.f, mi = 0.f;
              #pragma unroll
              for (int t = 0; t < KXN; ++t) { mr = fmaf(kx[t], V[slot][t + xsh].x, mr); mi = fmaf(kx[t], V[slot][t + xsh].y, mi); }
              pr = fmaf(wy[i], mr, pr); pi = fmaf(wy[i], mi, pi);
            }
          };
#if VS2
          int s = sa;
          for (; s + 1 < sb; s += 2) {
            float ar, ai, az, br, bi2, bz;
            body0(s, ar, ai, az);
            body0(s + 1, br, bi2, bz);
            fin(s, ar, ai, az);
            fin(s + 1, br, bi2, bz);
          }
          if (s < sb) { float ar, ai, az; body0(s, ar, ai, az); fin(s, ar, ai, az); }
#else
          for (int s = sa; s < sb; ++s) { float pr, pi, fz; body0(s, pr, pi, fz); fin(s, pr, pi, fz); }
#endif
#else
#if VT && VTP
          int nq = -1;   // first sample of the next non-empty class of this warp's range in this plane (−1: none)
          if (sa < sb) {
            for (int q = st * TX + ox + 1; q < qend; ++q) { const unsigned r2 = tab[hz * NCLS + q]; if ((r2 & 0xffffu) < (r2 >> 16)) { nq = (int)(r2 & 0xffffu); break; } }
          }
#endif
          for (int s = sa; s < sb; ++s) {
            const float* P = S + (size_t)s * VSW;
            float pr = 0.f, pi = 0.f;
#if VT
#if VTP
            // pipelined: the next sample's table (next in this range, else the next non-empty class's first) is built
            // into the other buffer while this sample's MACs run (MAC loads first in program order, so the build's
            // loads and arithmetic can interleave with the MAC FFMAs; its stores follow); one __syncwarp per sample
            const float4* Tb = tabw + (par ? 50 : 0);
#else
            float4* Tb = tabw + (par ? 50 : 0);
            par ^= 1;
            vt_build(s, hz, Tb, true);
            __syncwarp();
#endif
            // -- MAC: the lane's 5 rows × 10 points against the half's 50 table weights
            {
              const float4* Th = Tb + h * 25;
              float ar[5], ai[5];
              #pragma unroll
              for (int r = 0; r < 5; ++r) {
                const int slot = (st + r) % 5;
                #pragma unroll
                for (int k = 0; k < 5; ++k) {
                  const float4 q = Th[r * 5 + k];
                  const float2 v0 = V[slot][2 * k + xsh], v1 = V[slot][2 * k + 1 + xsh];
                  if (k == 0) { ar[r] = q.x * v0.x; ai[r] = q.x * v0.y; } else { ar[r] = fmaf(q.x, v0.x, ar[r]); ai[r] = fmaf(q.x, v0.y, ai[r]); }
                  ar[r] = fmaf(-q.y, v0.y, ar[r]); ai[r] = fmaf(q.y, v0.x, ai[r]);
                  ar[r] = fmaf(q.z, v1.x, ar[r]); ai[r] = fmaf(q.z, v1.y, ai[r]);
                  ar[r] = fmaf(-q.w, v1.y, ar[r]); ai[r] = fmaf(q.w, v1.x, ai[r]);
                }
              }
              pr = (ar[0] + ar[1]) + (ar[2] + ar[3]) + ar[4];
              pi = (ai[0] + ai[1]) + (ai[2] + ai[3]) + ai[4];
            }
#if VTP
            {
              const int nx = s + 1 < sb ? s + 1 : nq;
              vt_build(nx >= 0 ? nx : s, hz, tabw + (par ? 0 : 50), nx >= 0);
            }
            __syncwarp();
            par ^= 1;
#endif
            const float fz = 1.f;
#else
            const int oz = reinterpret_cast<const int*>(P)[VR_KX + KXN];
            const int tz = zlo + hz - oz;
            float wz[5];
            #pragma unroll
            for (int k = 0; k < 5; ++k) wz[k] = P[VR_WZ + tz * 5 + k];
            float ev[16];
            #pragma unroll
            for (int k = 0; k < 4; ++k) { const float4 u = *reinterpret_cast<const float4*>(P + VR_E + 4 * k); ev[4 * k] = u.x; ev[4 * k + 1] = u.y; ev[4 * k + 2] = u.z; ev[4 * k + 3] = u.w; }
            // ev[15] = d0 (VR_D = VR_E + 15)
            float dv[6];
            dv[0] = ev[15];
            {
              const float4 u = *reinterpret_cast<const float4*>(P + VR_D + 1);
              dv[1] = u.x; dv[2] = u.y; dv[3] = u.z; dv[4] = u.w; dv[5] = P[VR_D + 5];
            }
            float E[5][5];
            #pragma unroll
            for (int a = 0; a < 5; ++a) {
              #pragma unroll
              for (int b = 0; b <= 4 - a; ++b) E[a][b] = ev[a * (11 - a) / 2 + b] * wz[4 - a - b];
            }
            E[0][0] += wz[0];
            const float q00 = dv[2] * wz[2], q01 = dv[5] * wz[1], q02 = dv[1] * wz[0], q10 = dv[4] * wz[1], q11 = dv[3] * wz[0], q20 = dv[0] * wz[0];
            float wyb[28];
            #pragma unroll
            for (int k = 0; k < 7; ++k) { const float4 u = *reinterpret_cast<const float4*>(P + VR_WY + h * 28 + 4 * k); wyb[4 * k] = u.x; wyb[4 * k + 1] = u.y; wyb[4 * k + 2] = u.z; wyb[4 * k + 3] = u.w; }
            #pragma unroll
            for (int a = 0; a < 5; ++a) {
              float kx[KXS];
              #pragma unroll
              for (int k = 0; k < KXS / 4; ++k) { const float4 u = *reinterpret_cast<const float4*>(P + VR_KX + a * KXS + 4 * k); kx[4 * k] = u.x; kx[4 * k + 1] = u.y; kx[4 * k + 2] = u.z; kx[4 * k + 3] = u.w; }
              #pragma unroll
              for (int i = 0; i < 5; ++i) {
                const int slot = (st + i) % 5;
                float mr = 0.f, mi = 0.f;
                #pragma unroll
                for (int t = 0; t < KXN; ++t) { mr = fmaf(kx[t], V[slot][t + xsh].x, mr); mi = fmaf(kx[t], V[slot][t + xsh].y, mi); }
                float R = 0.f;
                #pragma unroll
                for (int b = 0; b <= 4 - a; ++b) R = fmaf(E[a][b], wyb[i * 5 + b], R);
                float Q = 0.f;
                if (a == 0) Q = q00 * wyb[i * 5] + q01 * wyb[i * 5 + 1] + q02 * wyb[i * 5 + 2];
                if (a == 1) Q = q10 * wyb[i * 5] + q11 * wyb[i * 5 + 1];
                if (a == 2) Q = q20 * wyb[i * 5];
                pr = fmaf(R, mr, pr); pi = fmaf(R, mi, pi);
                if (a < 3) { pr = fmaf(-Q, mi, pr); pi = fmaf(Q, mr, pi); }
              }
            }
            const float fz = 1.f;
#endif
            fin(s, pr, pi, fz);
          }
#endif
        }
      }
    }
  }
  __syncthreads();
  // ---- epilogue: y[coil][j] = Σ_l a_jl acc[ch]
  const int base = blockIdx.y * NWC * 16;
  const int bend = min(base + NWC * 16, NCH);
  const int clo = base / NSEG, chi = (bend - 1) / NSEG, ncoil = chi - clo + 1;
  for (int q = threadIdx.x; q < cnt * ncoil; q += NTV) {
    const int s = q / ncoil, coil = clo + q % ncoil;
    const unsigned int j = perm[it.first + s];
    const int l0 = max(0, base - coil * NSEG), l1 = min(NSEG, bend - coil * NSEG);
    cx o = cx{0.f, 0.f};
    for (int l = l0; l < l1; ++l) {
      const int k = coil * NSEG + l - base, g = k >> 4, cc = k & 15;
      const float* ap = accb + (size_t)g * 2 * APL + s * 16 + cc;
      const cx v = cmul(ldcx(acf + (size_t)(it.first + s) * NSEG + l), cx{ap[0], ap[APL]});
      o.x += v.x; o.y += v.y;
    }
    cx* dst = y + (size_t)coil * M + j;
    if (l0 == 0 && l1 == NSEG) *dst = o; else { atomicAdd(&dst->x, o.x); atomicAdd(&dst->y, o.y); }
  }
}
#endif

// deterministic grid values: hash of (point, channel) -> [-1, 1)^2 (the host reference uses the same function)
__device__ __forceinline__ unsigned int hmix(unsigned int x) {
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;
}
extern "C" __global__ void pg_fill(cx* __restrict__ grid, unsigned long long n) {
  for (unsigned long long t = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; t < n; t += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned int a = hmix((unsigned int)t * 2u + 1u), b = hmix((unsigned int)t * 2u + 2u + (unsigned int)(t >> 32));
    grid[t] = cx{(float)(a >> 8) * (1.f / 8388608.f) - 1.f, (float)(b >> 8) * (1.f / 8388608.f) - 1.f};
  }
}

#ifdef LDSB
// shared-load microbenchmark (research/sample_kernels.md §11): warp-wide LDS.128 where the 32 lanes read `ncg`
// distinct consecutive float4 (lane % ncg), KF independent FFMA per loaded float4, 16 loads per unrolled step.
// Measures the shared-memory cost of a broadcast float4 load as used by pg_fwd_ps (lanes = slot x channel group).
#ifndef KF
#define KF 0
#endif
#ifndef LDSW
#define LDSW 4
#endif
#if LDSW != 4
// LDSW 1 | 2: the same benchmark with LDS.32 / LDS.64 (lanes read `ncg` distinct consecutive words / float2; `ncg` 33
// means every lane distinct and consecutive: lane i reads element i), KF FFMA per loaded word / float2 component
extern "C" __global__ void __launch_bounds__(256) pg_lds_bench(float* __restrict__ out, int iters, int ncg) {
  __shared__ float buf[8192];   // 16 phases x 16 loads: not hoistable into registers
  for (int i = threadIdx.x; i < 8192; i += blockDim.x) buf[i] = 1e-3f * i;
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int idx = ncg >= 32 ? lane : lane % ncg;
  float acc[8];
  #pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = 0.f;
  float wq = 1.f + 1e-7f * threadIdx.x;
  for (int it = 0; it < iters; ++it) {
    #pragma unroll
    for (int u = 0; u < 16; ++u) {
#if LDSW == 1
      const float v = buf[idx + u * 64 + (it & 15) * 128];
#if KF == 0
      acc[u & 7] += v;
#else
      #pragma unroll
      for (int q = 0; q < KF; ++q) acc[(u * KF + q) & 7] = fmaf(v, wq, acc[(u * KF + q) & 7]);
#endif
#else
      const float2 v = reinterpret_cast<const float2*>(buf)[idx + u * 64 + (it & 15) * 128];
#if KF == 0
      acc[u & 7] += v.x + v.y;
#else
      #pragma unroll
      for (int q = 0; q < KF; ++q) acc[(u * KF + q) & 7] = fmaf((q & 1) ? v.y : v.x, wq, acc[(u * KF + q) & 7]);
#endif
#endif
    }
  }
  float s = 0.f;
  #pragma unroll
  for (int k = 0; k < 8; ++k) s += acc[k];
  if (s == 1.2345e-30f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
#else
extern "C" __global__ void __launch_bounds__(256) pg_lds_bench(float* __restrict__ out, int iters, int ncg) {
  __shared__ float4 buf[1024];
  for (int i = threadIdx.x; i < 1024; i += blockDim.x) buf[i] = make_float4(1e-3f * i, 1e-3f, 2e-3f, 3e-3f);
  __syncthreads();
  const int lane = threadIdx.x & 31;
#if defined(LDSBLK) && LDSBLK
  // blocked (§13): lanes 32/ncg·g .. read address g (one address per quarter-warp when ncg <= 4)
  const int idx = lane / (32 / ncg);
#else
  const int idx = lane % ncg;
#endif
  float acc[8];
  #pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = 0.f;
  float wq = 1.f + 1e-7f * threadIdx.x;
  for (int it = 0; it < iters; ++it) {
    #pragma unroll
    for (int u = 0; u < 16; ++u) {
      const float4 v = buf[idx + u * 32 + (it & 1) * 512];   // < 1024
#if KF == 0
      acc[u & 7] += v.x + v.y + v.z + v.w;   // keep the load live (3 FADD + 1 FADD)
#else
      #pragma unroll
      for (int q = 0; q < KF; ++q) acc[(u * KF + q) & 7] = fmaf((q & 3) == 0 ? v.x : (q & 3) == 1 ? v.y : (q & 3) == 2 ? v.z : v.w, wq, acc[(u * KF + q) & 7]);
#endif
    }
  }
  float s = 0.f;
  #pragma unroll
  for (int k = 0; k < 8; ++k) s += acc[k];
  if (s == 1.2345e-30f) out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}
#endif
#endif

#ifdef REDB
// global float-RED microbenchmark (research/sample_kernels.md §15): the flush pattern of an accumulator-stationary
// adjoint. Block t flushes the footprint box (TXB + W − 1) × (TYB + W − 1) × (TZB + W − 1) points of home tile t (tiles
// x fastest, wrapped), NWB warps = channel groups of 16 channels, lane (h, c): half h takes every other row, channel c.
// RED_MODE 0: RED.ADD.F32 on .x and .y (16 lanes per 128 B span, stride 8 B); 1: packed (lane l adds float l of the
// 32-float span of its half's 16 channels; the same floats, every RED instruction covers whole sectors); 2: plain
// read-modify-write (races ignored: reference for the byte traffic); 3: plain store.
#ifndef RED_MODE
#define RED_MODE 0
#endif
extern "C" __global__ void pg_red_bench(cx* __restrict__ g, int nf, int nch, int ntx, int nty) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, h = lane >> 4, c = lane & 15;
  const int t = blockIdx.x;
  const int bx = (t % ntx) * TXB, by = ((t / ntx) % nty) * TYB, bz = (t / (ntx * nty)) * TZB;
  const float v = 1e-3f * (float)(lane + 1);
  for (int z = 0; z < TZB + W - 1; ++z) {
    const int gz = (bz + z) % nf;
    for (int y = h; y < TYB + W - 1; y += 2) {
      const int gy = (by + y) % nf;
      #pragma unroll 4
      for (int x = 0; x < TXB + W - 1; ++x) {
        const int gx = (bx + x) % nf;
        float* p = reinterpret_cast<float*>(g + (((size_t)gz * nf + gy) * nf + gx) * nch + warp * 16);
#if RED_MODE == 0
        atomicAdd(p + 2 * c, v); atomicAdd(p + 2 * c + 1, v);
#elif RED_MODE == 1
        atomicAdd(p + c, v); atomicAdd(p + 16 + c, v);
#elif RED_MODE == 2
        p[2 * c] += v; p[2 * c + 1] += v;
#else
        p[2 * c] = v; p[2 * c + 1] = v;
#endif
      }
    }
  }
}
#endif

#ifdef RA
// ---------------------------------------------------------------- design 7 `ra`: row-blocked output adjoint (§15)
// Output tile TX × TY × TZ, block = (item of the tile's sample list, channel-block row blockIdx.y). Warp w = tile plane
// tz = w; lane = (x-group g < NXG, channel block cb < NCL = 32 / NXG); the thread holds acc[TY rows][PT = TX / NXG
// points][CB channels] (channels blockIdx.y · CPW + cb · CB + k) in registers. Samples are staged per chunk of CHK:
// kernel values on the tile's cells, relative support corners (ox, oy, oz), z = conj(a_l) y_c for the block's channels;
// CORR >= 1: the complex weights of every covered (sample, tile row) (stage_weights / stage_weights_hk terms). A warp
// skips a sample whose z support misses its plane (warp-uniform) and applies it only to the covered rows (exact y and
// z); x: zero weights outside the support (NXG 1: all lanes the same points, warp-uniform per-point skip, exact x).
// Output: unsplit items store, split items (tile list longer than one item) atomically add (grid zeroed by the host).
// Grid G[((z nf1 + y) nf0 + x) NCH + ch]; tile list tl = original sample indices; pos / dco / acf / y / uw in the
// original order.
#define PT (TX / NXG)
#define NCL (32 / NXG)
#define CBP ((CB + 1) / 2 * 2)
#define CPW (NCL * CB)
#define NTA (TZ * 32)
#define TROWS (TY * TZ)
#if CORR >= 2
#define NKA 5
#elif CORR == 1
#define NKA 3
#else
#define NKA 1
#endif
#define R4(n) (((n) + 3) / 4 * 4)
#define RR_KX 4                               // ints ox, oy, oz, pad at 0..3
#define RR_KY (RR_KX + NKA * R4(TX))
#define RR_KZ (RR_KY + NKA * R4(TY))
#define RR_D (RR_KZ + NKA * R4(TZ))
#define RR_E (RR_D + (CORR >= 1 ? 6 : 0))
#define RR_H (RR_E + (CORR >= 2 ? 15 : 0))
#define RSW R4(RR_H + (CORR >= 3 ? 25 : 0))
#define RZ_OFF ((size_t)CHK * RSW * 4)
#define RW_OFF (RZ_OFF + (size_t)CHK * NCL * CBP * 8)
// raw per-sample inputs of the chunk (prefetched into registers one chunk ahead, then parked here): positions, ΔC,
// u (CORR 3), a_l (NSEG), y of the block row's coils (≤ NCBRM); the chunk's sample indices, double-buffered
#define NCBRM ((CPW + NSEG - 1) / NSEG + 1 < NCB ? (CPW + NSEG - 1) / NSEG + 1 : NCB)
#define RAW_OFF (RW_OFF + (CORR >= 1 ? (size_t)CHK * TROWS * TX * 8 : 0))
#define RAW_POS RAW_OFF
#define RAW_U (RAW_POS + (size_t)CHK * 16)
#define RAW_A (RAW_U + (CORR >= 3 ? (size_t)CHK * 16 : 0))
#define RAW_Y (RAW_A + (size_t)CHK * NSEG * 8)
#define RAW_D (RAW_Y + (size_t)CHK * NCBRM * 8)
#define RAW_J (RAW_D + (CORR >= 1 ? (size_t)CHK * 6 * 4 : 0))
#define RAW_PL (RAW_J + (size_t)2 * CHK * 4)
#define RA_SMEM (RAW_PL + (CORR >= 1 ? (size_t)CHK * TZ * 32 * 4 : 0))   // CORR >= 1: per (sample, plane) ER[15] | EQ[15]
#define DIVUP(a, b) (((a) + (b) - 1) / (b))
#if CORR >= 3
#define RHI_ARGS , const float4* __restrict__ uw, const float* __restrict__ htab
#else
#define RHI_ARGS
#endif
#ifndef RA_NOMAC
#define RA_NOMAC 0
#endif
#ifndef RA_SPLITB
#define RA_SPLITB 0   // CORR >= 1: 1 = weights in two steps (per-plane ER / EQ, then rows); measured slower (§15)
#endif
struct RItem { int tile, first, count, split; };
__device__ __forceinline__ void ra_kern(float z, float* o) {
#if CORR >= 1
  es5(z, o);
#else
  const float a = 1.f - z * z;
  o[0] = a <= 0.f ? 0.f : expf((float)BETA * (sqrtf(a) - 1.f));
#endif
}
extern "C" __global__ void __launch_bounds__(NTA) pg_adj_ra(const float4* __restrict__ pos, const float* __restrict__ dco, const cx* __restrict__ acf,
                                                           const unsigned int* __restrict__ tl, const RItem* __restrict__ items, const cx* __restrict__ y,
                                                           cx* __restrict__ grid, int M, Dims d RHI_ARGS) {
  extern __shared__ __align__(16) unsigned char smem[];
  float* RS = reinterpret_cast<float*>(smem);
  cx* ZS = reinterpret_cast<cx*>(smem + RZ_OFF);
#if CORR >= 1
  cx* WS = reinterpret_cast<cx*>(smem + RW_OFF);   // [CHK][TROWS][TX]
#endif
  const RItem it = items[blockIdx.x];
  const int ntx = (d.nf0 + TX - 1) / TX, nty = (d.nf1 + TY - 1) / TY;
  const int o0 = (it.tile % ntx) * TX, o1 = ((it.tile / ntx) % nty) * TY, o2 = (it.tile / (ntx * nty)) * TZ;
  const int tz = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane / NCL, cb = lane % NCL;
  const int chb = blockIdx.y * CPW;
  cx acc[TY][PT][CB];
  #pragma unroll
  for (int r = 0; r < TY; ++r) {
    #pragma unroll
    for (int i = 0; i < PT; ++i) {
      #pragma unroll
      for (int k = 0; k < CB; ++k) acc[r][i][k] = cx{0.f, 0.f};
    }
  }
  float4* RP = reinterpret_cast<float4*>(smem + RAW_POS);
  cx* RAa = reinterpret_cast<cx*>(smem + RAW_A);
  cx* RYy = reinterpret_cast<cx*>(smem + RAW_Y);
  unsigned int* SJ = reinterpret_cast<unsigned int*>(smem + RAW_J);
#if CORR >= 1
  float* RDd = reinterpret_cast<float*>(smem + RAW_D);
#endif
#if CORR >= 3
  float4* RU = reinterpret_cast<float4*>(smem + RAW_U);
#endif
  const int clo = chb / NSEG, ncbr = min(NCBRM, NCB - clo);
  // register prefetch of one chunk's raw inputs (sample indices from SJ[buf])
  constexpr int NRP = DIVUP(CHK, NTA), NRA = DIVUP(CHK * NSEG, NTA), NRY = DIVUP(CHK * NCBRM, NTA);
  float4 pp[NRP]; cx pa[NRA], py[NRY];
#if CORR >= 1
  constexpr int NRD = DIVUP(CHK * 6, NTA);
  float pd[NRD];
#endif
#if CORR >= 3
  float4 pu[NRP];
#endif
  auto prefetch = [&](int buf, int nl) {
    const unsigned int* sj = SJ + buf * CHK;
    #pragma unroll
    for (int r = 0; r < NRP; ++r) {
      const int q = threadIdx.x + r * NTA;
      if (q < nl) { pp[r] = pos[sj[q]];
#if CORR >= 3
        pu[r] = uw[sj[q]];
#endif
      }
    }
    #pragma unroll
    for (int r = 0; r < NRA; ++r) { const int q = threadIdx.x + r * NTA; if (q < nl * NSEG) pa[r] = ldcx(acf + (size_t)sj[q / NSEG] * NSEG + q % NSEG); }
    #pragma unroll
    for (int r = 0; r < NRY; ++r) { const int q = threadIdx.x + r * NTA; if (q < nl * NCBRM && q % NCBRM < ncbr) py[r] = ldcx(y + (size_t)(clo + q % NCBRM) * M + sj[q / NCBRM]); }
#if CORR >= 1
    #pragma unroll
    for (int r = 0; r < NRD; ++r) { const int q = threadIdx.x + r * NTA; if (q < nl * 6) pd[r] = dco[(size_t)(q % 6) * M + sj[q / 6]]; }
#endif
  };
  // prologue: j lists of chunks 0 and 1, raw inputs of chunk 0
  for (int q = threadIdx.x; q < min(2 * CHK, it.count); q += NTA) SJ[q] = tl[it.first + q];
  __syncthreads();
  if (it.count > 0) prefetch(0, min(CHK, it.count));
  int buf = 0;
  for (int base = 0; base < it.count; base += CHK) {
    const int n = min(CHK, it.count - base);
    __syncthreads();   // previous chunk consumed
    // park the prefetched raw inputs
    #pragma unroll
    for (int r = 0; r < NRP; ++r) { const int q = threadIdx.x + r * NTA; if (q < n) { RP[q] = pp[r];
#if CORR >= 3
      RU[q] = pu[r];
#endif
    } }
    #pragma unroll
    for (int r = 0; r < NRA; ++r) { const int q = threadIdx.x + r * NTA; if (q < n * NSEG) RAa[q] = pa[r]; }
    #pragma unroll
    for (int r = 0; r < NRY; ++r) { const int q = threadIdx.x + r * NTA; if (q < n * NCBRM) RYy[q] = py[r]; }
#if CORR >= 1
    #pragma unroll
    for (int r = 0; r < NRD; ++r) { const int q = threadIdx.x + r * NTA; if (q < n * 6) RDd[q] = pd[r]; }
#endif
    __syncthreads();
    // next chunk: raw inputs in flight during this chunk's staging and MACs; j list of the chunk after it
    if (base + CHK < it.count) prefetch(buf ^ 1, min(CHK, it.count - base - CHK));
    for (int q = threadIdx.x; q < CHK && base + 2 * CHK + q < it.count; q += NTA) SJ[buf * CHK + q] = tl[it.first + base + 2 * CHK + q];
    buf ^= 1;
    // ---- stage A: kernel values on the tile's cells and relative corners (per (sample, axis, cell))
    constexpr int NCELL = TX + TY + TZ;
    for (int q = threadIdx.x; q < n * NCELL; q += NTA) {
      const int s = q / NCELL, r = q % NCELL;
      const int a = r < TX ? 0 : (r < TX + TY ? 1 : 2), c = r - (a == 0 ? 0 : (a == 1 ? TX : TX + TY));
      const float4 p4 = RP[s];
      const float sc = a == 0 ? p4.x : (a == 1 ? p4.y : p4.z);
      const int nn = a == 0 ? d.nf0 : (a == 1 ? d.nf1 : d.nf2), og = a == 0 ? o0 : (a == 1 ? o1 : o2);
      const int i0 = (int)ceilf(sc - (float)W * 0.5f);
      int rel = i0 - og;
      if (rel > nn / 2) rel -= nn;
      if (rel < -nn / 2) rel += nn;
      const int t = c - rel;
      float o[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
      if (t >= 0 && t < W && og + c < nn) ra_kern(((float)(i0 + t) - sc) * (2.f / (float)W), o);
      float* P = RS + (size_t)s * RSW;
      const int off = a == 0 ? RR_KX : (a == 1 ? RR_KY : RR_KZ), str = a == 0 ? R4(TX) : (a == 1 ? R4(TY) : R4(TZ));
      #pragma unroll
      for (int k = 0; k < NKA; ++k) P[off + k * str + c] = o[k];
      if (c == 0) reinterpret_cast<int*>(P)[a] = rel;
    }
#if CORR >= 1
    for (int q = threadIdx.x; q < n; q += NTA) {
      float* P = RS + (size_t)q * RSW;
      float dd[6];
      #pragma unroll
      for (int p = 0; p < 6; ++p) { dd[p] = RDd[q * 6 + p]; P[RR_D + p] = dd[p]; }
#if CORR >= 2
      const int mx[6] = {2, 0, 0, 1, 1, 0}, my[6] = {0, 2, 0, 1, 0, 1};
      float e[15];
      #pragma unroll
      for (int k = 0; k < 15; ++k) e[k] = 0.f;
      #pragma unroll
      for (int p = 0; p < 6; ++p) {
        #pragma unroll
        for (int pq = 0; pq < 6; ++pq) {
          const int ea = mx[p] + mx[pq], eb = my[p] + my[pq];
          e[ea * (11 - ea) / 2 + eb] = fmaf(-0.5f * dd[p], dd[pq], e[ea * (11 - ea) / 2 + eb]);
        }
      }
      #pragma unroll
      for (int k = 0; k < 15; ++k) P[RR_E + k] = e[k];
#endif
    }
#if CORR >= 3
    for (int q = threadIdx.x; q < n * 25; q += NTA) {   // degree-3/4 coefficients (segnufft_knot.cu stage_hc_k)
      const int s = q / 25, a = q % 25;
      const float4 u = RU[s];
      RS[(size_t)s * RSW + RR_H + a] = fmaf(u.x, htab[a], fmaf(u.y, htab[25 + a], fmaf(u.z, htab[50 + a], u.w * htab[75 + a])));
    }
#endif
#endif
    // z = conj(a_l) y_c for the block's channels (zero past NCH); RA_NOZ (timing ablation, wrong results): not built
#ifndef RA_NOZ
    // one thread per channel k (index math hoisted out of the sample loop); layout [s][pair kp][cb][2]: the lanes'
    // float4 loads of one pair are consecutive (conflict-free)
    for (int k = threadIdx.x; k < CPW; k += NTA) {
      const int ch = chb + k, kk = k % CB;
      const bool ok = ch < NCH;
      const int coil = ok ? ch / NSEG : clo, l = ok ? ch % NSEG : 0;
      const cx* ap = RAa + l;
      const cx* yp = RYy + (coil - clo);
      cx* zp = ZS + ((kk / 2) * NCL + k / CB) * 2 + (kk & 1);
      #pragma unroll 4
      for (int s = 0; s < n; ++s) {
        const cx a = ap[s * NSEG], yy = yp[s * NCBRM];
        zp[(size_t)s * (CBP / 2) * NCL * 2] = ok ? cx{fmaf(a.x, yy.x, a.y * yy.y), fmaf(a.x, yy.y, -a.y * yy.x)} : cx{0.f, 0.f};
      }
    }
#endif
    __syncthreads();
#if CORR >= 1
#if !RA_SPLITB
    // ---- stage B (default): complex weights of the covered (sample, tile row) pairs, one task per pair with the
    // stage_weights / stage_weights_hk terms inline
    for (int q = threadIdx.x; q < n * TROWS; q += NTA) {
      const int s = q / TROWS, row = q % TROWS, ty = row % TY, tzr = row / TY;
      const float* P = RS + (size_t)s * RSW;
      const int* oo = reinterpret_cast<const int*>(P);
      if (ty < oo[1] || ty > oo[1] + W - 1 || tzr < oo[2] || tzr > oo[2] + W - 1) continue;
      float wy[5], wz[5];
      #pragma unroll
      for (int k = 0; k < NKA; ++k) { wy[k] = P[RR_KY + k * R4(TY) + ty]; wz[k] = P[RR_KZ + k * R4(TZ) + tzr]; }
      float R[5] = {0.f, 0.f, 0.f, 0.f, 0.f}, Q[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
      R[0] = wy[0] * wz[0];
      const float* c = P + RR_D;   // [xx, yy, zz, xy, xz, yz]
      Q[0] = c[1] * wy[2] * wz[0] + c[2] * wy[0] * wz[2] + c[5] * wy[1] * wz[1];
      Q[1] = c[3] * wy[1] * wz[0] + c[4] * wy[0] * wz[1];
      Q[2] = c[0] * wy[0] * wz[0];
#if CORR >= 2
      {
        int idx = 0;
        #pragma unroll
        for (int a = 0; a <= 4; ++a) {
          #pragma unroll
          for (int b = 0; b <= 4 - a; ++b) { R[a] += P[RR_E + idx] * wy[b] * wz[4 - a - b]; ++idx; }
        }
      }
#endif
#if CORR >= 3
      {
        int idx = 0;
        #pragma unroll
        for (int a = 0; a <= 3; ++a) {
          #pragma unroll
          for (int b = 0; b <= 3 - a; ++b) { R[a] = fmaf(P[RR_H + idx], wy[b] * wz[3 - a - b], R[a]); ++idx; }
        }
        #pragma unroll
        for (int a = 0; a <= 4; ++a) {
          #pragma unroll
          for (int b = 0; b <= 4 - a; ++b) { Q[a] = fmaf(-P[RR_H + idx], wy[b] * wz[4 - a - b], Q[a]); ++idx; }
        }
      }
#endif
      cx* wp = WS + ((size_t)s * TROWS + row) * TX;
      #pragma unroll
      for (int t = 0; t < TX; ++t) {
        float wr = 0.f, wi = 0.f;
        #pragma unroll
        for (int a = 0; a < NKA; ++a) { const float xa = P[RR_KX + a * R4(TX) + t]; wr = fmaf(xa, R[a], wr); wi = fmaf(xa, Q[a], wi); }
        wp[t] = cx{wr, wi};
      }
    }
    __syncthreads();
#else
    // ---- RA_SPLITB 1 (measured slower, §15): stage B1: per covered (sample, plane) the plane coefficients ER_ab = e_ab wz_{4-a-b} (+ wz_0 at (0, 0))
    // [+ hc_ab wz_{3-a-b}] and EQ_ab (the six ΔC terms [- hc_ab wz_{4-a-b}]), wz folded in (the vt terms of §13.1)
    float* PL = reinterpret_cast<float*>(smem + RAW_PL);
    for (int q = threadIdx.x; q < n * TZ; q += NTA) {
      const int s = q / TZ, tzr = q % TZ;
      const float* P = RS + (size_t)s * RSW;
      const int* oo = reinterpret_cast<const int*>(P);
      if (tzr < oo[2] || tzr > oo[2] + W - 1) continue;
      float wz[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int k = 0; k < NKA; ++k) wz[k] = P[RR_KZ + k * R4(TZ) + tzr];
      float ER[15], EQ[15];
      #pragma unroll
      for (int k = 0; k < 15; ++k) { ER[k] = 0.f; EQ[k] = 0.f; }
#if CORR >= 2
      {
        int idx = 0;
        #pragma unroll
        for (int a = 0; a <= 4; ++a) {
          #pragma unroll
          for (int b = 0; b <= 4 - a; ++b) { ER[idx] = P[RR_E + idx] * wz[4 - a - b]; ++idx; }
        }
      }
#endif
      ER[0] += wz[0];
      const float* dv = P + RR_D;   // [xx, yy, zz, xy, xz, yz]
      EQ[0] = dv[2] * wz[2]; EQ[1] = dv[5] * wz[1]; EQ[2] = dv[1] * wz[0];
      EQ[5] = dv[4] * wz[1]; EQ[6] = dv[3] * wz[0]; EQ[9] = dv[0] * wz[0];
#if CORR >= 3
      {
        int j = 0;
        #pragma unroll
        for (int a = 0; a <= 3; ++a) {   // degree 3: + c ∂^α φ (real)
          #pragma unroll
          for (int b = 0; b <= 3 - a; ++b) { const int id = a * (11 - a) / 2 + b; ER[id] = fmaf(P[RR_H + j], wz[3 - a - b], ER[id]); ++j; }
        }
        #pragma unroll
        for (int a = 0; a <= 4; ++a) {   // degree 4: −i c ∂^α φ (imaginary)
          #pragma unroll
          for (int b = 0; b <= 4 - a; ++b) { const int id = a * (11 - a) / 2 + b; EQ[id] = fmaf(-P[RR_H + j], wz[4 - a - b], EQ[id]); ++j; }
        }
      }
#endif
      float* pl = PL + ((size_t)s * TZ + tzr) * 32;
      #pragma unroll
      for (int k = 0; k < 15; ++k) { pl[k] = ER[k]; pl[16 + k] = EQ[k]; }
    }
    __syncthreads();
    // ---- stage B2: complex weights of the covered (sample, tile row) pairs: R_a = Σ_b ER_ab wy_b, Q_a = Σ_b EQ_ab
    // wy_b, w(t) = Σ_a kx_a(t) (R_a + i Q_a)
    for (int q = threadIdx.x; q < n * TROWS; q += NTA) {
      const int s = q / TROWS, row = q % TROWS, ty = row % TY, tzr = row / TY;
      const float* P = RS + (size_t)s * RSW;
      const int* oo = reinterpret_cast<const int*>(P);
      if (ty < oo[1] || ty > oo[1] + W - 1 || tzr < oo[2] || tzr > oo[2] + W - 1) continue;
      float wy[5] = {0.f, 0.f, 0.f, 0.f, 0.f};
      #pragma unroll
      for (int k = 0; k < NKA; ++k) wy[k] = P[RR_KY + k * R4(TY) + ty];
      const float* pl = PL + ((size_t)s * TZ + tzr) * 32;
      float R[5], Q[5];
      {
        int idx = 0;
        #pragma unroll
        for (int a = 0; a <= 4; ++a) {
          R[a] = pl[idx] * wy[0]; Q[a] = pl[16 + idx] * wy[0];
          #pragma unroll
          for (int b = 1; b <= 4 - a; ++b) { R[a] = fmaf(pl[idx + b], wy[b], R[a]); Q[a] = fmaf(pl[16 + idx + b], wy[b], Q[a]); }
          idx += 5 - a;
        }
      }
      cx* wp = WS + ((size_t)s * TROWS + row) * TX;
      #pragma unroll
      for (int t = 0; t < TX; ++t) {
        float wr = 0.f, wi = 0.f;
        #pragma unroll
        for (int a = 0; a < NKA; ++a) { const float xa = P[RR_KX + a * R4(TX) + t]; wr = fmaf(xa, R[a], wr); wi = fmaf(xa, Q[a], wi); }
        wp[t] = cx{wr, wi};
      }
    }
    __syncthreads();
#endif   // RA_SPLITB
#endif   // CORR >= 1
    // ---- MAC: the warp's plane tz, covered rows only
#if !RA_NOMAC
    for (int s = 0; s < n; ++s) {
      const float* P = RS + (size_t)s * RSW;
      const int4 oo = *reinterpret_cast<const int4*>(P);
      if (tz < oo.z || tz > oo.z + W - 1) continue;   // warp-uniform
      cx zv[CBP];
      {
        const float4* zp = reinterpret_cast<const float4*>(ZS) + (size_t)s * (CBP / 2) * NCL + cb;
        #pragma unroll
        for (int k = 0; k < CBP / 2; ++k) { const float4 u = zp[k * NCL]; zv[2 * k] = cx{u.x, u.y}; zv[2 * k + 1] = cx{u.z, u.w}; }
      }
#if CORR == 0
      const float fz = P[RR_KZ + tz];
      float wx[PT], ky[R4(TY)];
      #pragma unroll
      for (int i = 0; i < PT; ++i) wx[i] = P[RR_KX + g * PT + i];
      #pragma unroll
      for (int k = 0; k < R4(TY) / 4; ++k) { const float4 u = reinterpret_cast<const float4*>(P + RR_KY)[k]; ky[4 * k] = u.x; ky[4 * k + 1] = u.y; ky[4 * k + 2] = u.z; ky[4 * k + 3] = u.w; }
      #pragma unroll
      for (int r = 0; r < TY; ++r) {
        if (r < oo.y || r > oo.y + W - 1) continue;   // warp-uniform: exact rows
        const float wr = ky[r] * fz;
        #pragma unroll
        for (int i = 0; i < PT; ++i) {
          if (NXG == 1 && (i < oo.x || i > oo.x + W - 1)) continue;   // NXG 1: warp-uniform exact x
          const float w = wx[i] * wr;
          #pragma unroll
          for (int k = 0; k < CB; ++k) { acc[r][i][k].x = fmaf(w, zv[k].x, acc[r][i][k].x); acc[r][i][k].y = fmaf(w, zv[k].y, acc[r][i][k].y); }
        }
      }
#else
      #pragma unroll
      for (int r = 0; r < TY; ++r) {
        if (r < oo.y || r > oo.y + W - 1) continue;
        const cx* wp = WS + ((size_t)s * TROWS + tz * TY + r) * TX + g * PT;
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
          if (NXG == 1 && (i < oo.x || i > oo.x + W - 1)) continue;
          #pragma unroll
          for (int k = 0; k < CB; ++k) {   // conj(w) z
            acc[r][i][k].x = fmaf(wv[i].x, zv[k].x, acc[r][i][k].x); acc[r][i][k].x = fmaf(wv[i].y, zv[k].y, acc[r][i][k].x);
            acc[r][i][k].y = fmaf(wv[i].x, zv[k].y, acc[r][i][k].y); acc[r][i][k].y = fmaf(-wv[i].y, zv[k].x, acc[r][i][k].y);
          }
        }
      }
#endif
    }
#endif
  }
  // ---- output: per warp, row by row, park the TX × CPW values in shared memory and write coalesced runs of CPW channels
  __syncthreads();
  constexpr int OBS = CPW | 1;
  cx* ob = reinterpret_cast<cx*>(smem) + (size_t)tz * TX * OBS;
  const int gz = o2 + tz;
  #pragma unroll
  for (int r = 0; r < TY; ++r) {
    #pragma unroll
    for (int i = 0; i < PT; ++i) {
      #pragma unroll
      for (int k = 0; k < CB; ++k) ob[(g * PT + i) * OBS + cb * CB + k] = acc[r][i][k];
    }
    __syncwarp();
    const int gy = o1 + r;
    if (gy < d.nf1 && gz < d.nf2) {
      for (int q = lane; q < TX * CPW; q += 32) {
        const int t = q / CPW, k = q % CPW, gx = o0 + t, ch = chb + k;
        if (gx >= d.nf0 || ch >= NCH) continue;
        const cx v = ob[t * OBS + k];
        cx* gp = grid + (((size_t)gz * d.nf1 + gy) * d.nf0 + gx) * NCH + ch;
        if (it.split) { atomicAdd(&gp->x, v.x); atomicAdd(&gp->y, v.y); } else *gp = v;
      }
    }
    __syncwarp();
  }
}
#endif
