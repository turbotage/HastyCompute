typedef REAL real;
struct cx { real x, y; };
__device__ __forceinline__ real ipow_(real x, int p) { real r = (real)1; for (int i = 0; i < p; ++i) r *= x; return r; }
// out_j = wt_j * prod_q w_q,j ^ mu_q * extra_j
extern "C" __global__ void make_weights(const real* __restrict__ wt, const real* __restrict__ w1, const real* __restrict__ w2, const real* __restrict__ w3, const real* __restrict__ w4,
                                        int m1, int m2, int m3, int m4, const cx* __restrict__ extra, int has_extra, cx* __restrict__ out, int M) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= M) return;
  real a = wt[j] * ipow_(w1[j], m1) * ipow_(w2[j], m2) * ipow_(w3[j], m3) * ipow_(w4[j], m4);
  cx e = has_extra ? extra[j] : cx{(real)1, (real)0};
  out[j] = cx{a * e.x, a * e.y};
}
// dst = src * (kx/(Nx/2))^gx (ky/(Ny/2))^gy (kz/(Nz/2))^gz with k the signed offset of FFT-order index p of an axis of length 2N
extern "C" __global__ void delta_mul(const cx* __restrict__ src, cx* __restrict__ dst, int P0, int P1, int P2, int g0, int g1, int g2) {
  const size_t total = (size_t)P0 * P1 * P2;
  const size_t t = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  if (t >= total) return;
  const int p0 = t % P0, p1 = (t / P0) % P1, p2 = t / ((size_t)P0 * P1);
  const real k0 = (real)(p0 < P0 / 2 ? p0 : p0 - P0), k1 = (real)(p1 < P1 / 2 ? p1 : p1 - P1), k2 = (real)(p2 < P2 / 2 ? p2 : p2 - P2);
  const real f = ipow_(k0 / ((real)P0 * (real)0.25), g0) * ipow_(k1 / ((real)P1 * (real)0.25), g1) * ipow_(k2 / ((real)P2 * (real)0.25), g2);
  const cx v = src[t];
  dst[t] = cx{v.x * f, v.y * f};
}
