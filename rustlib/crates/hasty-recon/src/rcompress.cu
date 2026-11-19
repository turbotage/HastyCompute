// Banded R step of the per-readout sample compression (rcompress.rs), in place.
// Set layout: coil-major y[c * m + s]; readout r owns the virtual samples [ro_off[r], ro_off[r + 1]); its R rows start
// at R + tab_off[r]: row i holds R[i][i + d], d < q (upper band, q taps).
//   forward: y_r <- R y_r       (y[i] = sum_d R[i][d] y[i + d], i + d < len)
//   adjoint: y_r <- R^H y_r     (y[i] = sum_d conj(R[i - d][d]) y[i - d], d <= i)
// One block per readout; the block walks its readout in chunks of blockDim.x outputs (forward: ascending, adjoint:
// descending, so a chunk only reads values not yet overwritten), all coils per chunk (the R chunk stays in L1).
struct __align__(8) cx { float x, y; };

extern "C" __global__ void rc_apply(cx* __restrict__ y, const cx* __restrict__ R, const unsigned* __restrict__ ro_off,
                                    const unsigned long long* __restrict__ tab_off, long long m, int q, int nc, int adjoint)
{
    extern __shared__ cx sh[];
    const int r = blockIdx.x;
    const long long a0 = ro_off[r];
    const int len = (int)(ro_off[r + 1] - ro_off[r]);
    if (len <= 0) return;
    const cx* Rr = R + tab_off[r];
    const int T = blockDim.x, tid = threadIdx.x;
    if (!adjoint) {
        for (int base = 0; base < len; base += T) {
            const int nload = min(T + q - 1, len - base);
            const int i = base + tid;
            for (int c = 0; c < nc; c++) {
                cx* v = y + (long long)c * m + a0;
                for (int k = tid; k < nload; k += T) sh[k] = v[base + k];
                __syncthreads();
                float ax = 0.f, ay = 0.f;
                if (i < len) {
                    const cx* row = Rr + (long long)i * q;
                    const int dmax = min(q, len - i);
                    for (int d = 0; d < dmax; d++) {
                        const cx a = row[d], b = sh[tid + d];
                        ax += a.x * b.x - a.y * b.y;
                        ay += a.x * b.y + a.y * b.x;
                    }
                }
                __syncthreads();
                if (i < len) { cx o; o.x = ax; o.y = ay; v[i] = o; }
            }
        }
    } else {
        for (int base = ((len - 1) / T) * T; base >= 0; base -= T) {
            const int lo = max(0, base - (q - 1));
            const int nload = min(base + T, len) - lo;
            const int i = base + tid;
            for (int c = 0; c < nc; c++) {
                cx* v = y + (long long)c * m + a0;
                for (int k = tid; k < nload; k += T) sh[k] = v[lo + k];
                __syncthreads();
                float ax = 0.f, ay = 0.f;
                if (i < len) {
                    const int dmax = min(q - 1, i);
                    for (int d = 0; d <= dmax; d++) {
                        const cx a = Rr[(long long)(i - d) * q + d], b = sh[i - d - lo];
                        // conj(a) * b
                        ax += a.x * b.x + a.y * b.y;
                        ay += a.x * b.y - a.y * b.x;
                    }
                }
                __syncthreads();
                if (i < len) { cx o; o.x = ax; o.y = ay; v[i] = o; }
            }
        }
    }
}

// out[i] = x[i] * m[i % nv] (real mask per voxel, every image of a stack); in place when out == x
extern "C" __global__ void rc_mask(const cx* x, const float* __restrict__ m, cx* out, long long n, long long nv)
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float w = m[i % nv];
    cx v = x[i];
    v.x *= w; v.y *= w;
    out[i] = v;
}
