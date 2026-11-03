// Direct-sum MR signal: out[i][c] += sum_p amp[p][c] * exp(-r2s_p tau_i) * exp(-i phi_ip)
//   phi = k.u + m1.(vt + delta vtt) + m2h.at + w.mq + omega tau
// sample layout (PF floats): k[3] m1[3] m2h[3] w[4] tau delta pad
// source layout (SF floats): u[3] mq[4] vt[3] vtt[3] at[3] omega r2s pad pad
// One thread per sample; sources streamed through shared memory; f32 partial sums flushed to f64 every FLUSH sources.
#ifndef NC
#define NC 1
#endif
#ifndef TS
#define TS 32
#endif
#ifndef FLUSH
#define FLUSH 512
#endif
#define PF 16
#define SF 20

extern "C" __global__ void __launch_bounds__(128) direct_sum(const float* __restrict__ smp, int ns, const float* __restrict__ src,
                                                             const float2* __restrict__ amp, int np, double2* __restrict__ out)
{
    __shared__ float s_src[TS * SF];
    __shared__ float2 s_amp[TS * NC];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < ns;
    float p[PF];
#pragma unroll
    for (int q = 0; q < PF; q++) p[q] = active ? smp[(size_t)i * PF + q] : 0.f;
    float ar[NC], ai[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) { ar[c] = 0.f; ai[c] = 0.f; }
    int pending = 0;
    for (int base = 0; base < np; base += TS) {
        const int cnt = min(TS, np - base);
        __syncthreads();
        for (int q = threadIdx.x; q < cnt * SF; q += blockDim.x) s_src[q] = src[(size_t)base * SF + q];
        for (int q = threadIdx.x; q < cnt * NC; q += blockDim.x) s_amp[q] = amp[(size_t)base * NC + q];
        __syncthreads();
        if (active) {
            for (int j = 0; j < cnt; j++) {
                const float* s = s_src + j * SF;
                float ph = p[0] * s[0] + p[1] * s[1] + p[2] * s[2];
                ph += p[3] * fmaf(p[14], s[10], s[7]) + p[4] * fmaf(p[14], s[11], s[8]) + p[5] * fmaf(p[14], s[12], s[9]);
                ph += p[6] * s[13] + p[7] * s[14] + p[8] * s[15];
                ph += p[9] * s[3] + p[10] * s[4] + p[11] * s[5] + p[12] * s[6];
                ph += s[16] * p[13];
                const float decay = __expf(-s[17] * p[13]);
                float r = ph * 0.15915494309189535f;
                r -= rintf(r);
                float sn, cs;
                __sincosf(r * 6.283185307179586f, &sn, &cs);
                const float er = decay * cs, ei = -decay * sn;
                const float2* a = s_amp + j * NC;
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    const float2 av = a[c];
                    ar[c] = fmaf(er, av.x, fmaf(-ei, av.y, ar[c]));
                    ai[c] = fmaf(er, av.y, fmaf(ei, av.x, ai[c]));
                }
            }
        }
        pending += cnt;
        if (pending >= FLUSH || base + TS >= np) {
            if (active) {
                double2* o = out + (size_t)i * NC;
#pragma unroll
                for (int c = 0; c < NC; c++) {
                    double2 v = o[c];
                    v.x += (double)ar[c];
                    v.y += (double)ai[c];
                    o[c] = v;
                    ar[c] = 0.f;
                    ai[c] = 0.f;
                }
            }
            pending = 0;
        }
    }
}
