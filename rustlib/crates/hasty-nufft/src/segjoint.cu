// Joint coil × time-segment channels of the segmented NUFFT (research/coil_segment_compression.md §2, §8).
//
//   E_c(r, τ) = S_c(r) g(r, τ) ≈ Σ_q B_cq(τ) Φ_q(r),   Φ_q(r) = Σ_{c,k} conj(U_{(c,k),q}) S_c(r) h_k(r)
//
// h_k(r) = the voxel's time function in the time basis V (K functions), U = the eigenvectors of the reduced Gram
// H = Σ_r w_r z_r z_rᴴ, z_r = S(r) ⊗ h_r. The NUFFT engine sees the R channels Φ_q (coil maps of an nseg = 1 engine)
// in batches; the mixing kernels map the batch's channel samples g_q(j) to the coils:
//
//   forward  y_c(j) += Σ_{q ∈ batch} B_cq(τ_j) g_q(j)
//   adjoint  g_q(j)  = Σ_c conj(B_cq(τ_j)) y_c(j)
//
// Defines from the host: NC (coils), NK (time-basis functions), QT (channels per Φ register tile), NN (ADC samples per
// mixing block). The mixing kernels need the regular sample layout j = readout · na + n (n = ADC sample index) and a
// B table with one row per ADC sample: B[(n · R + q) · NC + c].
struct cx { float x, y; };
struct cxd { double x, y; };

__device__ __forceinline__ cx cmul(cx a, cx b) { return cx{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
__device__ __forceinline__ void cfma(cx& acc, cx a, cx b) {
  acc.x = fmaf(a.x, b.x, fmaf(-a.y, b.y, acc.x));
  acc.y = fmaf(a.x, b.y, fmaf(a.y, b.x, acc.y));
}
// acc += conj(a) b
__device__ __forceinline__ void cfmac(cx& acc, cx a, cx b) {
  acc.x = fmaf(a.x, b.x, fmaf(a.y, b.y, acc.x));
  acc.y = fmaf(a.x, b.y, fmaf(-a.y, b.x, acc.y));
}

// Φ batch: phi[q · nv + r] = Σ_{c,k} Uc[(c · NK + k) · ldu + q0 + q] S[c · nv + r] h[k · nv + r], q < nq
// (Uc = conj(U), columns zero-padded to a multiple of QT: the tile loads stay in bounds)
extern "C" __global__ void joint_phi(const cx* __restrict__ S, const cx* __restrict__ h, const cx* __restrict__ Uc, cx* __restrict__ phi, long nv, int ldu, int q0, int nq) {
  const long r = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (r >= nv) return;
  cx s[NC];
#pragma unroll
  for (int c = 0; c < NC; ++c) s[c] = S[(long)c * nv + r];
  for (int qb = 0; qb < nq; qb += QT) {
    cx acc[QT];
#pragma unroll
    for (int t = 0; t < QT; ++t) acc[t] = cx{0.f, 0.f};
    for (int k = 0; k < NK; ++k) {
      const cx hk = h[(long)k * nv + r];
      const cx* u = Uc + (long)k * ldu + q0 + qb;
#pragma unroll 2
      for (int c = 0; c < NC; ++c) {
        const cx a = cmul(s[c], hk);
        const cx* uc = u + (long)c * NK * ldu;
#pragma unroll
        for (int t = 0; t < QT; ++t) cfma(acc[t], uc[t], a);
      }
    }
#pragma unroll
    for (int t = 0; t < QT; ++t)
      if (qb + t < nq) phi[(long)(qb + t) * nv + r] = acc[t];
  }
}

// Φ as a GEMM (default, `JointSegNufft::phi_gemm`): Φ[v0 + r, q] = Σ_{ck} Z[r, ck] Uc[ck, q0 + q] with the Khatri–Rao
// rows Z[r + ck · nvc] = S[c · nv + v0 + r] h[k · nv + v0 + r], ck = c · NK + k, over a voxel chunk r < nvc
// (cuBLAS CGEMM on the host side)
extern "C" __global__ void joint_zkr(const cx* __restrict__ S, const cx* __restrict__ h, cx* __restrict__ Z, long nv, long v0, int nvc) {
  const long e = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (e >= (long)NC * NK * nvc) return;
  const int r = (int)(e % nvc);
  const int ck = (int)(e / nvc), c = ck / NK, k = ck - c * NK;
  Z[e] = cmul(S[(long)c * nv + v0 + r], h[(long)k * nv + v0 + r]);
}

// mixing with a chunk of the set's samples: g holds `mg` samples (leading dimension, the chunk), y the whole set (M)
// with the chunk at sample offset j0 (whole readouts: j0 = readout0 · na); otherwise as joint_mix_fwd / joint_mix_adj
// (forward: always accumulates into y)
extern "C" __global__ void joint_mix_fwd_o(const cx* __restrict__ g, const cx* __restrict__ B, cx* __restrict__ y, long M, long mg, long j0, int na, int nro, int R, int q0, int nq, int rit) {
  extern __shared__ cx Bs[];
  const int ld = nq * NC + 1;
  const int n0 = blockIdx.x * NN;
  for (int e = threadIdx.x; e < NN * nq * NC; e += blockDim.x) {
    const int a = e / (nq * NC), rem = e - a * (nq * NC);
    const int n = n0 + a;
    Bs[a * ld + rem] = n < na ? B[((long)n * R + q0) * NC + rem] : cx{0.f, 0.f};
  }
  __syncthreads();
  const int nl = threadIdx.x % NN, rl = threadIdx.x / NN;
  const int n = n0 + nl;
  if (n >= na) return;
  const cx* bs = Bs + nl * ld;
  for (int it = 0; it < rit; ++it) {
    const long ro = ((long)blockIdx.y * rit + it) * (blockDim.x / NN) + rl;
    if (ro >= nro) return;
    const long j = ro * na + n;
    cx acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = cx{0.f, 0.f};
    for (int q = 0; q < nq; ++q) {
      const cx gq = g[(long)q * mg + j];
#pragma unroll
      for (int c = 0; c < NC; ++c) cfma(acc[c], bs[q * NC + c], gq);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      cx* yp = y + (long)c * M + j0 + j;
      const cx p = *yp;
      *yp = cx{p.x + acc[c].x, p.y + acc[c].y};
    }
  }
}
extern "C" __global__ void joint_mix_adj_o(const cx* __restrict__ y, const cx* __restrict__ B, cx* __restrict__ g, long M, long mg, long j0, int na, int nro, int R, int q0, int nq, int rit) {
  extern __shared__ cx Bs[];
  const int ld = nq * NC + 1;
  const int n0 = blockIdx.x * NN;
  for (int e = threadIdx.x; e < NN * nq * NC; e += blockDim.x) {
    const int a = e / (nq * NC), rem = e - a * (nq * NC);
    const int n = n0 + a;
    Bs[a * ld + rem] = n < na ? B[((long)n * R + q0) * NC + rem] : cx{0.f, 0.f};
  }
  __syncthreads();
  const int nl = threadIdx.x % NN, rl = threadIdx.x / NN;
  const int n = n0 + nl;
  if (n >= na) return;
  const cx* bs = Bs + nl * ld;
  for (int it = 0; it < rit; ++it) {
    const long ro = ((long)blockIdx.y * rit + it) * (blockDim.x / NN) + rl;
    if (ro >= nro) return;
    const long j = ro * na + n;
    cx yv[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) yv[c] = y[(long)c * M + j0 + j];
    for (int q = 0; q < nq; ++q) {
      cx acc = cx{0.f, 0.f};
#pragma unroll
      for (int c = 0; c < NC; ++c) cfmac(acc, bs[q * NC + c], yv[c]);
      g[(long)q * mg + j] = acc;
    }
  }
}

// forward mixing, regular layout: block = NN ADC samples × 64 readout lanes, each lane loops over `rit` readouts;
// the block's B rows (NN × nq × NC) sit in dynamic shared memory (row stride nq·NC + 1)
extern "C" __global__ void joint_mix_fwd(const cx* __restrict__ g, const cx* __restrict__ B, cx* __restrict__ y, long M, int na, int nro, int R, int q0, int nq, int rit, int accum) {
  extern __shared__ cx Bs[];
  const int ld = nq * NC + 1;
  const int n0 = blockIdx.x * NN;
  for (int e = threadIdx.x; e < NN * nq * NC; e += blockDim.x) {
    const int a = e / (nq * NC), rem = e - a * (nq * NC);
    const int n = n0 + a;
    Bs[a * ld + rem] = n < na ? B[((long)n * R + q0) * NC + rem] : cx{0.f, 0.f};
  }
  __syncthreads();
  const int nl = threadIdx.x % NN, rl = threadIdx.x / NN;
  const int n = n0 + nl;
  if (n >= na) return;
  const cx* bs = Bs + nl * ld;
  for (int it = 0; it < rit; ++it) {
    const long ro = ((long)blockIdx.y * rit + it) * (blockDim.x / NN) + rl;
    if (ro >= nro) return;
    const long j = ro * na + n;
    cx acc[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) acc[c] = cx{0.f, 0.f};
    for (int q = 0; q < nq; ++q) {
      const cx gq = g[(long)q * M + j];
#pragma unroll
      for (int c = 0; c < NC; ++c) cfma(acc[c], bs[q * NC + c], gq);
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      cx o = acc[c];
      if (accum) { const cx p = y[(long)c * M + j]; o.x += p.x; o.y += p.y; }
      y[(long)c * M + j] = o;
    }
  }
}

// adjoint mixing, regular layout: g[q · M + j] = Σ_c conj(B_cq(τ_n)) y[c · M + j]
extern "C" __global__ void joint_mix_adj(const cx* __restrict__ y, const cx* __restrict__ B, cx* __restrict__ g, long M, int na, int nro, int R, int q0, int nq, int rit) {
  extern __shared__ cx Bs[];
  const int ld = nq * NC + 1;
  const int n0 = blockIdx.x * NN;
  for (int e = threadIdx.x; e < NN * nq * NC; e += blockDim.x) {
    const int a = e / (nq * NC), rem = e - a * (nq * NC);
    const int n = n0 + a;
    Bs[a * ld + rem] = n < na ? B[((long)n * R + q0) * NC + rem] : cx{0.f, 0.f};
  }
  __syncthreads();
  const int nl = threadIdx.x % NN, rl = threadIdx.x / NN;
  const int n = n0 + nl;
  if (n >= na) return;
  const cx* bs = Bs + nl * ld;
  for (int it = 0; it < rit; ++it) {
    const long ro = ((long)blockIdx.y * rit + it) * (blockDim.x / NN) + rl;
    if (ro >= nro) return;
    const long j = ro * na + n;
    cx yv[NC];
#pragma unroll
    for (int c = 0; c < NC; ++c) yv[c] = y[(long)c * M + j];
    for (int q = 0; q < nq; ++q) {
      cx acc = cx{0.f, 0.f};
#pragma unroll
      for (int c = 0; c < NC; ++c) cfmac(acc, bs[q * NC + c], yv[c]);
      g[(long)q * M + j] = acc;
    }
  }
}

// ---- basis build (fp64) ----

// Z[(c · NK + k) + j · NC·NK] = sw[c + j · NC] · h[k + cols[j] · NK], j < nfit (fit voxels of a chunk)
extern "C" __global__ void joint_kron(const cxd* __restrict__ sw, const int* __restrict__ cols, const cxd* __restrict__ h, cxd* __restrict__ Z, int nfit) {
  const long e = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const long ck = (long)NC * NK;
  if (e >= ck * nfit) return;
  const long j = e / ck;
  const int i = (int)(e - j * ck), c = i / NK, k = i - c * NK;
  const cxd a = sw[c + j * NC], b = h[k + (long)cols[j] * NK];
  Z[e] = cxd{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

// device h table (fp32, K × nv): hd[k · nv + v0 + col] = h[k + col · NK], col < ncol
extern "C" __global__ void joint_hstore(const cxd* __restrict__ h, cx* __restrict__ hd, long nv, long v0, int ncol) {
  const long e = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (e >= (long)NK * ncol) return;
  const long col = e / NK;
  const int k = (int)(e - col * NK);
  const cxd z = h[e];
  hd[(long)k * nv + v0 + col] = cx{(float)z.x, (float)z.y};
}

// fp32 TSQR helper: A[i + j · lda] for i < CK of the stacked [R_prev; Zᴴ chunk] (column j = one (c,k) row of Z):
// a[(rows0 + v) + col · lda] = conj(Z[col + v · CK]) as fp32, v < nfit
extern "C" __global__ void joint_zt32(const cxd* __restrict__ Z, cx* __restrict__ A, int lda, int rows0, int nfit) {
  const long e = blockIdx.x * (long)blockDim.x + threadIdx.x;
  const long ck = (long)NC * NK;
  if (e >= ck * nfit) return;
  const long v = e / ck;
  const int col = (int)(e - v * ck);
  const cxd z = Z[e];
  A[(rows0 + v) + (long)col * lda] = cx{(float)z.x, (float)(-z.y)};
}

// TSQR: keep R (upper triangle of the top ck rows) of a column-major lda × ck block, zero everything else
extern "C" __global__ void joint_keep_r(cx* __restrict__ A, int lda, int ck) {
  const long e = blockIdx.x * (long)blockDim.x + threadIdx.x;
  if (e >= (long)lda * ck) return;
  const int row = (int)(e % lda), col = (int)(e / lda);
  if (row > col) A[e] = cx{0.f, 0.f};
}
