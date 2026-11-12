//! k-space of a moving factory phantom (direct summation on the GPU), from `tools/phantom/prep_sim.py`.
//!   simphantom <sim_input.hv> <out_dir> [--bin-ms 10] [--venc 1.5] [--venc-hi v] [--noise 0.02] [--rr-std 0] [--flip 12]
//!              [--ideal] [--static] [--write-noisy] [--fat six|single] [--scan-time s] [--plan]
//!              [--seq spiral | b0 [--b0-res 2e-3] [--b0-k 24] [--b0-pns 80] [--dwell 3e-6]
//!                     | flow [--flow-run data/hasty/seq/flow_thorax] [--girf ideal|clinical|<variant>|<girf.json>] [--girf-hist N] [--dwell 2e-6]]
//! `--seq` picks the schedule (`hasty_mrsim::sequence::Schedule`): the spiral 4D-flow `Acquisition` (default) or the
//! hasty-seq B0 / R2* / fat train (`hasty_bench::seqsched::EchoSchedule`, also writes samples.npy [readout][adc][4] =
//! k (rad/m, physical) and the time since excitation). Fat sources (shift ≠ 0) are split into the six-peak spectrum
//! (hasty-qfit `FatSpectrum::six_peak`, the fit's model) unless `--fat single`. `--seq flow`: the final 4D-flow yarn TR
//! of a seqdesign run (`FlowSchedule`), optionally played through a GIRF (`--girf`: per-sample moments of the played
//! sequence, previous TR included; the B0 terms as a constant phase) — samples.npy then holds the nominal trajectory,
//! samples_played.npy [readout][adc][8] the played one (γM0 = k (rad/m), γM1 (rad·s/m), τ, B0-term phase), physical axes.
//! Flow sims also write sched_nominal.npy (and sched_played.npy) [readout][adc][16]: the full per-sample records the
//! reconstruction loader (hasty-recon `SimData`, `HASTY_TRAJ=nominal|played`) uses.
//! `--girf <file.json>`: a measured / estimated chain (`hasty-girf-1` table file, hasty_seq `Girf::from_measurement`).
//! `--girf-hist N`: play every TR with the chain's N previous TRs (hasty_seq `PlayedStream`, exact LTI superposition;
//! research/girf_sensitivity.md: ≈ 10 needed for 10–30 ms eddy terms); default 0 = one previous TR (the original path).
//! The meta records the design's reconstruction window (`sequence.recon_adc`: ADC-index ranges; seqdesign `recon`):
//! the reconstruction loader keeps those samples, the rest (e.g. v2's rewinder) is navigator data.
//! `--plan`: build the schedule, write readouts.npy, samples*.npy and plan.json (cost estimate), no GPU work.
//! Readouts are grouped into absolute-time bins; per bin (cardiac time from the ECG, breathing state r, ṙ):
//! - tissue sources (Lagrangian, reference pose) are moved by the cardiac displacement (periodic Catmull-Rom of the
//!   stored samples, trilinear) and the respiratory field: x = X + u_c(X, tc) + r S(X + u_c), v = u̇_c + ṙ S;
//! - blood sources come from the nearest stored cardiac frame (Eulerian, r = 0): x = p + v Δ + r S(p), v + ṙ S;
//! - then gradient nonlinearity (u = f(x − iso) − c, J v), slab profile, coil sensitivities at the moved position;
//! - direct sum over all sources for the bin's samples, split over source chunks (f64 atomics), with the position
//!   advanced to each readout's excitation: φ = k·(u + δ J v) + m1·J v + m2h·J a + w·m(x) + ω τ, δ = t_exc − t_bin.
//! Writes data_clean.npy (complex64 [readout][adc][coil]), readouts.npy, meta.json (incl. Ψ, noise_sigma),
//! smaps_true.npy and field_ppm_recon.npy on the reconstruction grid (axial logical frame). Interruptible / resumable
//! (STOP file, Ctrl-C): chunks of bins are kept as `state_sim_bins_<b0>_<b1>.hv`.

#![recursion_limit = "256"]

use std::path::PathBuf;
use std::time::Instant;

use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, C32};
use hasty_mrsim::gnl::GradientCoils;
use hasty_mrsim::physio::Ecg;
use hasty_bench::seqsched::{EchoSchedule, FlowSchedule};
use hasty_mrsim::sequence::{Acquisition, FlowSpiralParams, Schedule, SAMPLE_FLOATS};
use hasty_qfit::waterfat::FatSpectrum;
use hasty_seq::multiecho::{fit_slew_to_pns, EchoSpec, EchoTrain};
use hasty_phantom::coils::{body_arrays, noise_covariance, SensitivityTable};
use hasty_phantom::volume::{npy_header, write_npy, Volume};
use hasty_value::Value;
use num_complex::Complex64;

/// floats per source record of the kernels (`SF` in SRC)
const SRC_FLOATS: usize = 24;

const SRC: &str = r#"
#ifndef NC
#define NC 1
#endif
#define PF 16
#define SF 24
#define TS 32

struct Grid { const float* f; int nx, ny, nz, nc; float ox, oy, oz, h; int clamp; };

// trilinear at index coordinates (x − o) / h (o = centre of voxel 0); components c0 .. c0 + n − 1 → out
__device__ void tri(const Grid g, float x, float y, float z, int c0, int n, float* out)
{
    float fx = (x - g.ox) / g.h, fy = (y - g.oy) / g.h, fz = (z - g.oz) / g.h;
    if (g.clamp) {
        fx = fminf(fmaxf(fx, 0.f), g.nx - 1.001f); fy = fminf(fmaxf(fy, 0.f), g.ny - 1.001f); fz = fminf(fmaxf(fz, 0.f), g.nz - 1.001f);
    } else if (fx < 0.f || fy < 0.f || fz < 0.f || fx > g.nx - 1.001f || fy > g.ny - 1.001f || fz > g.nz - 1.001f) {
        for (int c = 0; c < n; c++) out[c] = 0.f;
        return;
    }
    int i = (int)fx, j = (int)fy, k = (int)fz;
    float a = fx - i, b = fy - j, d = fz - k;
    long long sx = 1, sy = g.nx, sz = (long long)g.nx * g.ny, sc = sz * g.nz;
    long long q = i + sy * j + sz * k;
    for (int c = 0; c < n; c++) {
        const float* F = g.f + (c0 + c) * sc + q;
        float v00 = F[0] * (1 - a) + F[sx] * a, v10 = F[sy] * (1 - a) + F[sy + sx] * a;
        float v01 = F[sz] * (1 - a) + F[sz + sx] * a, v11 = F[sz + sy] * (1 - a) + F[sz + sy + sx] * a;
        out[c] = ((v00 * (1 - b) + v10 * b) * (1 - d) + (v01 * (1 - b) + v11 * b) * d);
    }
}

extern "C" __global__ void combine(const float* U, long long nv, int k0, int k1, int k2, int k3, float w0, float w1, float w2, float w3,
                                   float d0, float d1, float d2, float d3, float* Uc, float* Vc)
{
    long long q = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (q >= nv) return;
    float a = U[k0 * nv + q], b = U[k1 * nv + q], c = U[k2 * nv + q], d = U[k3 * nv + q];
    Uc[q] = w0 * a + w1 * b + w2 * c + w3 * d;
    Vc[q] = d0 * a + d1 * b + d2 * c + d3 * d;
}

// mode 0: tissue (reference pose + cardiac grids), mode 1: blood (frame positions + velocities)
extern "C" __global__ void prep(int mode, long long n, const float* pos, const float* vel, const float* m, const float* omega, const float* r2s, const float* fat,
                                const float* e1, const float* zoff, const float* elh, const float* elg, int el,
                                Grid Ug, Grid Vg, Grid Sg, Grid Gg, Grid Cg, float r, float rdot, float dlt,
                                float isox, float isoy, float isoz, float fcx, float fcy, float fcz,
                                float pcx, float pcy, float pcz, const float* atab, int ntab, float az0, float adz, float vs, float* src, float2* amp)
{
    long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (p >= n) return;
    float x[3] = {pos[3 * p], pos[3 * p + 1], pos[3 * p + 2]}, v[3], t[3];
    if (mode == 0) {
        tri(Ug, x[0], x[1], x[2], 0, 3, t);
        tri(Vg, x[0], x[1], x[2], 0, 3, v);
        for (int c = 0; c < 3; c++) x[c] += t[c];
    } else {
        // vs: the template-to-heartbeat velocity scale (period / RR); dlt is physical time
        for (int c = 0; c < 3; c++) { v[c] = vel[3 * p + c] * vs; x[c] += v[c] * dlt; }
    }
    if (r != 0.f || rdot != 0.f) {
        tri(Sg, x[0], x[1], x[2], 0, 3, t);
        for (int c = 0; c < 3; c++) { x[c] += r * t[c]; v[c] += rdot * t[c]; }
    }
    float xs[3] = {x[0] - isox, x[1] - isoy, x[2] - isoz};
    float g[12];
    tri(Gg, xs[0], xs[1], xs[2], 0, 12, g);
    float* s = src + p * SF;
    s[0] = g[0] - fcx; s[1] = g[1] - fcy; s[2] = g[2] - fcz;
    s[3] = xs[2] * xs[2]; s[4] = xs[0] * xs[0] + xs[1] * xs[1]; s[5] = xs[0] * xs[2]; s[6] = xs[1] * xs[2];
    for (int a = 0; a < 3; a++) s[7 + a] = g[3 + 3 * a] * v[0] + g[4 + 3 * a] * v[1] + g[5 + 3 * a] * v[2];
    for (int a = 10; a < 16; a++) s[a] = 0.f;
    s[16] = omega[p]; s[17] = r2s[p]; s[18] = fat[p];
    // box element (multi-resolution sources): edge h (m, 0 = point) and the field gradient across it (rad/s/m)
    s[19] = el ? elh[p] : 0.f;
    s[20] = el ? elg[3 * p] : 0.f; s[21] = el ? elg[3 * p + 1] : 0.f; s[22] = el ? elg[3 * p + 2] : 0.f; s[23] = 0.f;
    // axial slab (logical z = subject z): the local flip angle from the Bloch-simulated RF profile, the source's
    // chemical-shift displacement of the slab zoff; steady-state spoiled GRE (e1 = E1 ≥ 0) or a relative longitudinal
    // magnetization s = −e1 from the spin history (blood): M0 s sin α
    float zr = x[2] - pcz - zoff[p];
    float fa = (zr - az0) / adz, al = 0.f;
    if (fa >= 0.f && fa <= ntab - 1.001f) { int ia = (int)fa; float aa = fa - ia; al = atab[ia] * (1.f - aa) + atab[ia + 1] * aa; }
    float sa = sinf(al), ca = cosf(al), E = e1[p];
    float w = E >= 0.f ? sa * (1.f - E) / fmaxf(1.f - ca * E, 1e-12f) : -E * sa;
    float cs[2 * NC];
    tri(Cg, x[0], x[1], x[2], 0, 2 * NC, cs);
    float mw = m[p] * w;
    for (int c = 0; c < NC; c++) amp[p * NC + c] = make_float2(mw * cs[2 * c], mw * cs[2 * c + 1]);
}

__device__ float h2f(unsigned short h)
{
    unsigned int s = (h >> 15) & 1u, e = (h >> 10) & 31u, m = h & 1023u, f;
    if (e == 0) {
        if (m == 0) f = s << 31;
        else { int ee = 113; while (!(m & 1024u)) { m <<= 1; ee--; } m &= 1023u; f = (s << 31) | ((unsigned)ee << 23) | (m << 13); }
    } else if (e == 31) f = (s << 31) | 0x7f800000u | (m << 13);
    else f = (s << 31) | ((e + 112u) << 23) | (m << 13);
    return __int_as_float((int)f);
}

// symmetric 3×3 eigen-decomposition (cyclic Jacobi): A destroyed, eigenvalues d, eigenvectors as columns of V
__device__ void jacobi3(float A[3][3], float V[3][3], float d[3])
{
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) V[i][j] = i == j ? 1.f : 0.f;
    for (int sweep = 0; sweep < 10; sweep++) {
        float off = fabsf(A[0][1]) + fabsf(A[0][2]) + fabsf(A[1][2]);
        if (off < 1e-12f) break;
        for (int pq = 0; pq < 3; pq++) {
            int p = pq == 2 ? 1 : 0, q = pq == 0 ? 1 : 2;
            if (fabsf(A[p][q]) < 1e-20f) continue;
            float th = (A[q][q] - A[p][p]) / (2.f * A[p][q]);
            float t = (th >= 0.f ? 1.f : -1.f) / (fabsf(th) + sqrtf(th * th + 1.f));
            float c = rsqrtf(t * t + 1.f), s = t * c;
            for (int k = 0; k < 3; k++) {
                float akp = A[k][p], akq = A[k][q];
                A[k][p] = c * akp - s * akq;
                A[k][q] = s * akp + c * akq;
            }
            for (int k = 0; k < 3; k++) {
                float apk = A[p][k], aqk = A[q][k];
                A[p][k] = c * apk - s * aqk;
                A[q][k] = s * apk + c * aqk;
            }
            for (int k = 0; k < 3; k++) {
                float vkp = V[k][p], vkq = V[k][q];
                V[k][p] = c * vkp - s * vkq;
                V[k][q] = s * vkp + c * vkq;
            }
        }
    }
    for (int i = 0; i < 3; i++) d[i] = A[i][i];
}

// LBM beat-library sources: each frame source with a moment record (row ≥ 0; mom: f16 [rows][24] = w, mean, cov
// (xx yy zz xy xz yz), third central (xxx yyy zzz xxy xxz xyy yyz xzz yzz xyz), E[s], cov(s, u)) becomes six sub-sources
// at its position: along each principal axis of the covariance two velocities μ + x± e with weights reproducing the axis
// variance and third moment (an asymmetric two-point rule, weights 1/3 per axis); without a record the frame's own
// velocity (slot 0). spin = 1: the library's spin history replaces the steady-state signal: each sub-source carries the
// relative M_z s = E[s] + cov(s,u)ᵀ Σ⁻¹ (u − μ) (exact E[s] and E[s u] over the six points; e16 = −s selects the
// relative-M_z amplitude in prep) and its PD loses the constant inflow gain (pd_scale = 1 / gain)
extern "C" __global__ void bsig(long long n, const float* pos, const float* vel, const float* m, const float* om, const float* r2,
                                const float* e1, const float* zo, const int* row, const unsigned short* mom, float vs, int spin, float pd_scale,
                                float* pos6, float* vel6, float* m6, float* om6, float* r26, float* e16, float* zo6)
{
    long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (p >= n) return;
    int r = row[p];
    for (int j = 0; j < 6; j++) {
        long long q = 6 * p + j;
        for (int c = 0; c < 3; c++) { pos6[3 * q + c] = pos[3 * p + c]; vel6[3 * q + c] = 0.f; }
        m6[q] = 0.f; om6[q] = om[p]; r26[q] = r2[p]; e16[q] = e1[p]; zo6[q] = zo[p];
    }
    if (r < 0) {
        for (int c = 0; c < 3; c++) vel6[18 * p + c] = vel[3 * p + c] * vs;
        m6[6 * p] = m[p];
        return;
    }
    const unsigned short* M = mom + 24LL * r;
    float smean = h2f(M[20]), csu[3] = {h2f(M[21]), h2f(M[22]), h2f(M[23])};
    float mp = spin ? m[p] * pd_scale : m[p];
    float mu[3] = {h2f(M[1]), h2f(M[2]), h2f(M[3])};
    float A[3][3] = {{h2f(M[4]), h2f(M[7]), h2f(M[8])}, {h2f(M[7]), h2f(M[5]), h2f(M[9])}, {h2f(M[8]), h2f(M[9]), h2f(M[6])}};
    float t3[10];
    for (int k = 0; k < 10; k++) t3[k] = h2f(M[10 + k]);
    float V[3][3], d[3];
    jacobi3(A, V, d);
    for (int a = 0; a < 3; a++) {
        float e0 = V[0][a], e1 = V[1][a], e2 = V[2][a];
        float lam = fmaxf(d[a], 0.f), sd = sqrtf(lam);
        float sk = t3[0] * e0 * e0 * e0 + t3[1] * e1 * e1 * e1 + t3[2] * e2 * e2 * e2
                 + 3.f * (t3[3] * e0 * e0 * e1 + t3[4] * e0 * e0 * e2 + t3[5] * e0 * e1 * e1 + t3[6] * e1 * e1 * e2 + t3[7] * e0 * e2 * e2 + t3[8] * e1 * e2 * e2)
                 + 6.f * t3[9] * e0 * e1 * e2;
        float xp, xm, wp, wm;
        if (sd > 1e-6f) {
            float g = fminf(fmaxf(sk / (sd * sd * sd), -3.f), 3.f);
            float rr = sqrtf(0.25f * g * g + 3.f);
            xp = sd * (0.5f * g + rr);
            xm = sd * (0.5f * g - rr);
            wp = lam / (xp * (xp - xm));
            wm = -wp * xp / xm;
        } else {
            xp = xm = 0.f;
            wp = wm = 1.f / 6.f;
        }
        long long qp = 6 * p + 2 * a, qm = qp + 1;
        for (int c = 0; c < 3; c++) {
            float ec = V[c][a];
            vel6[3 * qp + c] = (mu[c] + xp * ec) * vs;
            vel6[3 * qm + c] = (mu[c] + xm * ec) * vs;
        }
        m6[qp] = mp * wp;
        m6[qm] = mp * wm;
        if (spin) {
            float g = lam > 1e-8f ? (csu[0] * e0 + csu[1] * e1 + csu[2] * e2) / lam : 0.f;
            e16[qp] = -fminf(fmaxf(smean + g * xp, 0.f), 1.f);
            e16[qm] = -fminf(fmaxf(smean + g * xm, 0.f), 1.f);
        }
    }
}

// out[i][c] += Σ_p amp[p][c] e^{−r2s τ} e^{−iφ}, sources split over blockIdx.y chunks (f64 atomics)
extern "C" __global__ void __launch_bounds__(128) dsum(const float* __restrict__ smp, int ns, const float* __restrict__ src,
                                                       const float2* __restrict__ amp, long long np, long long chunk, double* __restrict__ out,
                                                       const float* __restrict__ fatpk, int npk, int yoff)
{
    __shared__ float s_src[TS * SF];
    __shared__ float2 s_amp[TS * NC];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < ns;
    const long long p0 = (blockIdx.y + (long long)yoff) * chunk, p1 = min(np, p0 + chunk);
    float p[PF];
#pragma unroll
    for (int q = 0; q < PF; q++) p[q] = active ? smp[(size_t)i * PF + q] : 0.f;
    float ar[NC], ai[NC];
#pragma unroll
    for (int c = 0; c < NC; c++) { ar[c] = 0.f; ai[c] = 0.f; }
    // fat spectrum of this sample, c(τ) = Σ_p α_p e^{−i ω_p τ} (multiplies fat sources, flag s[18])
    float cfr = 0.f, cfi = 0.f;
    for (int k = 0; k < npk; k++) {
        float sn, cs;
        sincosf(fatpk[npk + k] * p[13], &sn, &cs);
        cfr += fatpk[k] * cs;
        cfi -= fatpk[k] * sn;
    }
    for (long long base = p0; base < p1; base += TS) {
        const int cnt = (int)min((long long)TS, p1 - base);
        __syncthreads();
        for (int q = threadIdx.x; q < cnt * SF; q += blockDim.x) s_src[q] = src[(size_t)base * SF + q];
        for (int q = threadIdx.x; q < cnt * NC; q += blockDim.x) s_amp[q] = amp[(size_t)base * NC + q];
        __syncthreads();
        if (!active) continue;
        for (int j = 0; j < cnt; j++) {
            const float* s = s_src + j * SF;
            float ph = p[0] * fmaf(p[14], s[7], s[0]) + p[1] * fmaf(p[14], s[8], s[1]) + p[2] * fmaf(p[14], s[9], s[2]);
            ph += p[3] * fmaf(p[14], s[10], s[7]) + p[4] * fmaf(p[14], s[11], s[8]) + p[5] * fmaf(p[14], s[12], s[9]);
            ph += p[6] * s[13] + p[7] * s[14] + p[8] * s[15];
            ph += p[9] * s[3] + p[10] * s[4] + p[11] * s[5] + p[12] * s[6];
            ph += s[16] * p[13] + p[15];
            const float decay = __expf(-s[17] * p[13]);
            float rr = ph * 0.15915494309189535f;
            rr -= rintf(rr);
            float sn, cs;
            __sincosf(rr * 6.283185307179586f, &sn, &cs);
            float er = decay * cs, ei = -decay * sn;
            // box element: ∫ over the cube of uniform magnetization with a linear field = h³ Π_a sinc(h (k_a + g_a τ) / 2)
            // (amplitudes carry PD × h³); the branch is uniform over the block (every thread has the same source j)
            if (s[19] > 0.f) {
                const float hh = 0.5f * s[19];
                float fe = 1.f;
#pragma unroll
                for (int a = 0; a < 3; a++) {
                    const float xa = hh * fmaf(s[20 + a], p[13], p[a]);
                    fe *= fabsf(xa) < 1e-3f ? 1.f - xa * xa * (1.f / 6.f) : __sinf(xa) / xa;
                }
                er *= fe;
                ei *= fe;
            }
            if (s[18] != 0.f) {
                const float tr = er * cfr - ei * cfi;
                ei = er * cfi + ei * cfr;
                er = tr;
            }
            const float2* a = s_amp + j * NC;
#pragma unroll
            for (int c = 0; c < NC; c++) {
                ar[c] = fmaf(er, a[c].x, fmaf(-ei, a[c].y, ar[c]));
                ai[c] = fmaf(er, a[c].y, fmaf(ei, a[c].x, ai[c]));
            }
        }
    }
    if (active)
#pragma unroll
        for (int c = 0; c < NC; c++) {
            atomicAdd(out + 2 * ((size_t)i * NC + c), (double)ar[c]);
            atomicAdd(out + 2 * ((size_t)i * NC + c) + 1, (double)ai[c]);
        }
}
"#;

fn tf32(v: &Value, k: &str) -> Vec<f32> {
    v.get(k).and_then(|x| x.as_tensor()).unwrap_or_else(|| panic!("{k}")).to_vec::<f32>().unwrap()
}
fn fl(v: &Value, k: &str) -> f64 {
    match v.get(k) {
        Some(Value::Float(x)) => *x,
        Some(Value::Int(x)) => *x as f64,
        _ => panic!("{k}"),
    }
}
fn fls(v: &Value, k: &str) -> Vec<f64> {
    match v.get(k) {
        Some(Value::List(l)) => l.iter().map(|x| match x { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => panic!("{k}") }).collect(),
        Some(Value::Tensor(t)) => (0..t.numel()).map(|i| t.get_f64(i).0).collect(),
        _ => panic!("{k}"),
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Grid {
    f: u64,
    nx: i32,
    ny: i32,
    nz: i32,
    nc: i32,
    ox: f32,
    oy: f32,
    oz: f32,
    h: f32,
    clamp: i32,
}
unsafe impl cudarc::driver::DeviceRepr for Grid {}

fn grid(f: &CudaSlice<f32>, st: &std::sync::Arc<cudarc::driver::CudaStream>, dims: [usize; 3], nc: usize, o: [f64; 3], h: f64, clamp: bool) -> Grid {
    use cudarc::driver::DevicePtr;
    let (ptr, _sync) = f.device_ptr(st);
    Grid { f: ptr, nx: dims[0] as i32, ny: dims[1] as i32, nz: dims[2] as i32, nc: nc as i32, ox: o[0] as f32, oy: o[1] as f32, oz: o[2] as f32, h: h as f32, clamp: clamp as i32 }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let bin_ms: f64 = take("--bin-ms").map(|s| s.parse().unwrap()).unwrap_or(10.0);
    let venc: f64 = take("--venc").map(|s| s.parse().unwrap()).unwrap_or(1.5);
    let venc_hi: Option<f64> = take("--venc-hi").map(|s| s.parse().unwrap());
    let noise_rel: f64 = take("--noise").map(|s| s.parse().unwrap()).unwrap_or(0.02);
    let rr_std: f64 = take("--rr-std").map(|s| s.parse().unwrap()).unwrap_or(0.0);
    let flip: f64 = take("--flip").map(|s| s.parse::<f64>().unwrap()).unwrap_or(12.0).to_radians();
    let seq_kind = take("--seq").unwrap_or_else(|| "spiral".into());
    let fat_mode = take("--fat").unwrap_or_else(|| "six".into());
    let fat_six = fat_mode != "single";
    // "six": fat sources carry the six-peak spectrum c(τ) evaluated per sample in the kernel (exact, one source per
    // fat voxel); "six-expand": six sub-sources per fat voxel (the older, 2.4× slower equivalent, for checks)
    let fat_expand = fat_mode == "six-expand";
    let scan_time_arg: Option<f64> = take("--scan-time").map(|s| s.parse().unwrap());
    let b0_res: f64 = take("--b0-res").map(|s| s.parse().unwrap()).unwrap_or(2e-3);
    let b0_k: usize = take("--b0-k").map(|s| s.parse().unwrap()).unwrap_or(24);
    let b0_pns: f64 = take("--b0-pns").map(|s| s.parse().unwrap()).unwrap_or(80.0);
    let dwell_opt: Option<f64> = take("--dwell").map(|s| s.parse().unwrap());
    let flow_run = take("--flow-run").map(PathBuf::from).unwrap_or_else(|| hasty_site::data().join("seq/flow_thorax"));
    let girf_name = take("--girf").unwrap_or_else(|| "ideal".into());
    let girf_hist: usize = take("--girf-hist").map(|s| s.parse().expect("--girf-hist N")).unwrap_or(0);
    // --beats <lbm_run>: per-heartbeat chamber flow from the LBM beat library (hasty_bench::beats), each heartbeat a
    // library beat (seeded shuffle, the first --beat-max of the library), the intravoxel distribution as six
    // moment-matched sub-sources per covered voxel (coverage ≥ --beat-cover); a beat's moments are computed on first use
    // and kept in an LRU of --beat-cache beats
    let beats_run: Option<PathBuf> = take("--beats").map(PathBuf::from);
    let beat_seed: u64 = take("--beat-seed").map(|s| s.parse().unwrap()).unwrap_or(1);
    let beat_cover: f32 = take("--beat-cover").map(|s| s.parse().unwrap()).unwrap_or(0.5);
    let beat_max: usize = take("--beat-max").map(|s| s.parse().unwrap()).unwrap_or(30);
    let beat_cache: usize = take("--beat-cache").map(|s| s.parse().unwrap()).unwrap_or(6);
    // --beats-ref <lbm_run>: a second library solved in the heart's reference frame (lbm_coronary.py: the coronary
    // arteries), sampled at --beat-ref-sub³ points per blood source mapped back by the cardiac motion; its moments are
    // the cells' mixture transformed by the motion (F μ + u_wall, F Σ Fᵀ, …); it takes the sources it covers
    let beats_ref: Option<PathBuf> = take("--beats-ref").map(PathBuf::from);
    let beat_ref_sub: usize = take("--beat-ref-sub").map(|s| s.parse().unwrap()).unwrap_or(3);
    // --beat-spin off: use a library's velocities with the steady-state blood signal (its spin history belongs to
    // another protocol, e.g. the flow library for the B0 / R2* / fat scan)
    let beat_ignore_spin = take("--beat-spin").map_or(false, |v| v == "off");
    // slab excitation (hasty_mrsim::rf): windowed sinc of time-bandwidth --rf-tbw over --rf-dur (default the
    // sequence's t_rf), slab thickness --slab-thk (default the FOV's z), Bloch-simulated profile
    let rf_tbw: f64 = take("--rf-tbw").map(|s| s.parse().unwrap()).unwrap_or(6.0);
    let rf_dur: Option<f64> = take("--rf-dur").map(|s| s.parse().unwrap());
    let slab_thk_arg: Option<f64> = take("--slab-thk").map(|s| s.parse().unwrap());
    // readout Nyquist for the dev phantom's 50 cm: the yarn's |G| ≤ 23 mT/m needs ≤ 2 µs; the B0 train 3 µs
    let dwell_arg: f64 = dwell_opt.unwrap_or(if seq_kind == "flow" { 2e-6 } else { 3e-6 });
    let flag = |a: &mut Vec<String>, n: &str| a.iter().position(|x| x == n).map(|i| { a.remove(i); true }).unwrap_or(false);
    let ideal = flag(&mut args, "--ideal");
    // --plan: build the schedule, write readouts.npy / samples*.npy and plan.json with the cost estimate, no GPU work
    let plan = flag(&mut args, "--plan");
    let stat = flag(&mut args, "--static");
    let write_noisy = flag(&mut args, "--write-noisy");
    let inp = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out).unwrap();
    let ctl = hasty_recon::checkpoint::RunCtl::new(&out, 1);
    let t0 = Instant::now();
    let v = hasty_value::read(&mut std::io::BufReader::new(std::fs::File::open(&inp).unwrap())).unwrap();
    let presc = v.get("presc").unwrap();
    let center = fls(presc, "center");
    let fov = fls(presc, "fov");
    let res = fl(presc, "res");
    let iso = fls(&v, "iso");
    let period = fl(&v, "period");
    let scan_time = fl(&v, "scan_time");
    if let Some(st_) = scan_time_arg {
        assert!(st_ <= scan_time + 1e-9, "--scan-time {st_} beyond the breathing table of the input ({scan_time} s)");
    }
    let scan_time = scan_time_arg.unwrap_or(scan_time);

    // ── acquisition ──
    let mut fp = FlowSpiralParams::aorta_3t_default();
    fp.fov = [fov[0], fov[1], fov[2]];
    fp.res = [res; 3];
    fp.venc = venc;
    if let Some(vh) = venc_hi {
        fp.encoding = hasty_mrsim::sequence::Encoding::DualVenc { venc_hi: vh };
    }
    fp.scan_time = scan_time;
    fp.mean_rr = period;
    fp.rr_rel_std = rr_std;
    fp.fov_center = [center[0] - iso[0], center[1] - iso[1], center[2] - iso[2]];
    let ecg = Ecg::new(fp.scan_time + 2.0, fp.mean_rr, fp.rr_rel_std, fp.seed);
    let mut seq_meta = serde_json::json!({"kind": seq_kind});
    // the schedule a reconstruction assumes when the simulated one is played through a GIRF (samples.npy)
    let mut nominal: Option<Box<dyn Schedule>> = None;
    let acq: Box<dyn Schedule> = match seq_kind.as_str() {
        "spiral" => Box::new(Acquisition::new(fp.clone())),
        "b0" => {
            let sys = hasty_seq::system::System::prisma_xr();
            let hw = hasty_seq::pns::SafeHw::example();
            let mut spec = EchoSpec::new(b0_res, b0_k);
            let t0_ = fit_slew_to_pns(&spec, &sys, &hw, b0_pns, 6);
            spec.slew_frac = t0_.spec.slew_frac;
            spec.te_shifts = vec![0.0, sys.ceil_raster(0.5 * t0_.t_lobe)];
            let train = EchoTrain::design(&spec, &sys);
            seq_meta = serde_json::json!({"kind": "b0", "res": b0_res, "n_lobe": b0_k, "t_lobe": train.t_lobe, "te1": train.te1, "te_shifts": spec.te_shifts,
                                          "slew_frac": spec.slew_frac, "tr": train.tr, "dwell": dwell_arg, "kmax": train.kmax, "m1_lobe": train.m1_lobe,
                                          "te": spec.te_shifts.iter().map(|&s| train.te(s)).collect::<Vec<_>>(), "pns_limit": b0_pns, "pns_hw": hw.name,
                                          "directions": "golden means (Chan 2009) over i = TR·pairs + pair, pairs (d, −d)"});
            Box::new(EchoSchedule::new(train, scan_time, dwell_arg, fp.b0, ecg.clone(), fp.n_frames))
        }
        "flow" => {
            let girf = match girf_name.as_str() {
                "ideal" | "none" => hasty_seq::girf::Girf::ideal(),
                "clinical" => hasty_seq::girf::Girf::clinical_example(),
                f if f.ends_with(".json") => hasty_seq::girf::Girf::from_measurement(std::path::Path::new(f)).unwrap_or_else(|e| panic!("--girf {f}: {e}")),
                name => hasty_seq::girf::Girf::variants().into_iter().find(|v| v.0 == name).unwrap_or_else(|| panic!("--girf {name}: ideal | clinical | a Girf::variants() name | <girf.json>")).1,
            };
            let fs = FlowSchedule::load(&flow_run, scan_time, dwell_arg, fp.b0, ecg.clone(), fp.n_frames, girf, &girf_name);
            let fs = if girf_hist > 0 { fs.with_history(girf_hist) } else { fs };
            seq_meta = fs.meta.clone();
            if girf_name != "ideal" && girf_name != "none" {
                nominal = Some(Box::new(fs.with_girf(hasty_seq::girf::Girf::ideal(), "ideal")));
            }
            Box::new(fs)
        }
        other => panic!("--seq {other}: spiral | b0 | flow"),
    };
    let na = acq.n_adc();
    let tr = acq.tr();
    let n_ro = acq.n_readouts();
    println!("acquisition: {}; {} readouts × {} ADC, TR {:.2} ms, TE {:.3} ms, RR {:.3} s (σ {:.0} %)", acq.describe(), n_ro, na, tr * 1e3, acq.te() * 1e3, period, rr_std * 100.0);

    // ── coils from the body mask ──
    let bt = v.get("body").and_then(|x| x.as_tensor()).unwrap();
    let bs = bt.shape().to_vec();
    let bo = fls(&v, "body_origin_m");
    let bh = fl(&v, "body_voxel_m");
    let mut aff = [[0.0; 4]; 4];
    for a in 0..3 {
        aff[a][a] = bh;
        aff[a][3] = bo[a];
    }
    aff[3][3] = 1.0;
    let body = Volume { dims: [bs[2], bs[1], bs[0]], affine: aff, data: bt.to_vec::<u8>().unwrap() };
    let loops = body_arrays(&body, 4, 4, 0.045, center[2] - 0.17, center[2] + 0.17);
    let sens = SensitivityTable::new(&loops, &body, 2);
    let psi = noise_covariance(&loops, &body, 6, 0.1);
    let nc = loops.len();
    println!("coils: {nc} ({:.1}s)", t0.elapsed().as_secs_f64());

    // ── sources (host) ──
    let ts = v.get("tissue").unwrap();
    let tpos = tf32(ts, "pos");
    let gam = 2.0 * std::f64::consts::PI * hasty_mrsim::GAMMA_BAR * fp.b0 * 1e-6;
    let ss = |pd: f32, t1: f32| -> f32 {
        let e1 = (-tr / t1 as f64).exp();
        (pd as f64 * flip.sin() * (1.0 - e1) / (1.0 - flip.cos() * e1)) as f32
    };
    let six = FatSpectrum::six_peak();
    // the slab pulse and its profile (water); off-resonant sources see it displaced by −Δf / (γ̄ G_ss)
    // the excited slab: --slab-thk, else the prescription's slab (prep_sim --slab), else the FOV's z
    let presc_slab = presc.get("slab");
    let slab_thk = slab_thk_arg.unwrap_or_else(|| presc_slab.map_or(fov[2], |s| fl(s, "thk")));
    let slab_c: Vec<f64> = presc_slab.map_or(center.clone(), |s| fls(s, "center"));
    let pulse = hasty_mrsim::rf::SlabPulse::new(rf_dur.unwrap_or(fp.t_rf), rf_tbw, flip, slab_thk);
    let prof = pulse.profile(0.75 * slab_thk + 0.02, 3001, 0.0);
    let zshift = |ppm: f64| -> f32 { (-ppm * 1e-6 * fp.b0 / prof.g_ss) as f32 };
    println!("slab: {:.0} mm, sinc TBW {rf_tbw} over {:.2} ms (BW {:.0} Hz, G_ss {:.2} mT/m), centre flip {:.1}°, FWHM {:.1} mm, fat displaced {:.1} mm",
             slab_thk * 1e3, pulse.duration * 1e3, prof.bandwidth, prof.g_ss * 1e3, prof.alpha_at(0.0).to_degrees(), prof.fwhm() * 1e3, zshift(-3.4) * 1e3);
    let _ = &ss;
    // per source: M0 (PD), ω, R2*, the index of the input source, E1 = e^{−TR/T1} and the slab displacement (fat sources
    // split into the six peaks of the fit's spectrum: amplitude α_p, shift δ_p replacing the input's single shift); the
    // kernel forms the steady state with the local flip angle
    let mk = |d: &Value| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<usize>, Vec<f32>, Vec<f32>) {
        let (pd, t1, t2s, fld) = (tf32(d, "pd"), tf32(d, "t1"), tf32(d, "t2s"), tf32(d, "field_ppm"));
        let sh = d.get("shift_ppm").map(|_| tf32(d, "shift_ppm")).unwrap_or(vec![0.0; pd.len()]);
        let (mut m, mut om, mut r2, mut idx, mut e1, mut zo) = (vec![], vec![], vec![], vec![], vec![], vec![]);
        for i in 0..pd.len() {
            let (mi, f, ri) = (pd[i], if ideal { 0.0 } else { fld[i] as f64 }, 1.0 / t2s[i].max(1e-4));
            let ei = (-tr / t1[i].max(1e-4) as f64).exp() as f32;
            if sh[i] != 0.0 && fat_six && fat_expand {
                for (&a, &p) in six.amp.iter().zip(&six.ppm) {
                    m.push(mi * a as f32);
                    om.push((gam * (f + p)) as f32);
                    r2.push(ri);
                    idx.push(i);
                    e1.push(ei);
                    zo.push(zshift(p));
                }
            } else if sh[i] != 0.0 && fat_six {
                // one source at the water frequency; the kernel multiplies its term by c(τ) (flag in `tfat`)
                m.push(mi);
                om.push((gam * f) as f32);
                r2.push(ri);
                idx.push(i);
                e1.push(ei);
                zo.push(zshift(sh[i] as f64));
            } else {
                m.push(mi);
                om.push((gam * (f + sh[i] as f64)) as f32);
                r2.push(ri);
                idx.push(i);
                e1.push(ei);
                zo.push(zshift(sh[i] as f64));
            }
        }
        (m, om, r2, idx, e1, zo)
    };
    let expand = |x: &[f32], idx: &[usize]| -> Vec<f32> { idx.iter().flat_map(|&i| x[3 * i..3 * i + 3].iter().copied()).collect() };
    let (tm, tom, tr2, tidx, te1, tzo) = mk(ts);
    let tpos = if tidx.len() == tpos.len() / 3 { tpos } else { expand(&tpos, &tidx) };
    let nt = tm.len();
    // multi-resolution tissue sources (prep_sim --src-octree): element edge (m) and field gradient (ppm/m) per input
    // source, expanded like the other per-source arrays; absent = point sources
    let (telh, telg): (Vec<f32>, Vec<f32>) = match ts.get("size_m").map(|_| tf32(ts, "size_m")) {
        Some(hs) => {
            let g = ts.get("field_grad_ppm_m").map(|_| tf32(ts, "field_grad_ppm_m"));
            let gr = |i: usize, a: usize| -> f32 { if ideal { 0.0 } else { g.as_ref().map_or(0.0, |g| (gam * g[3 * i + a] as f64) as f32) } };
            (tidx.iter().map(|&i| hs[i]).collect(), tidx.iter().flat_map(|&i| [gr(i, 0), gr(i, 1), gr(i, 2)]).collect())
        }
        None => (vec![], vec![]),
    };
    let tel_on = !telh.is_empty();
    if tel_on {
        let mut hist = std::collections::BTreeMap::<i64, usize>::new();
        for &hh in &telh {
            *hist.entry((hh as f64 * 1e4).round() as i64).or_insert(0) += 1;
        }
        println!("tissue elements (box sources): {}", hist.iter().map(|(k, n)| format!("{:.1} mm × {n}", *k as f64 / 10.0)).collect::<Vec<_>>().join(", "));
    }
    let tfat: Vec<f32> = {
        let sh = ts.get("shift_ppm").map(|_| tf32(ts, "shift_ppm")).unwrap_or(vec![0.0; tidx.len()]);
        tidx.iter().map(|&i| if fat_six && !fat_expand && sh[i] != 0.0 { 1.0 } else { 0.0 }).collect()
    };
    let fat_tab: Vec<f32> = six.amp.iter().map(|&a| a as f32).chain(six.ppm.iter().map(|&p| (gam * p) as f32)).collect();
    let npk: i32 = if fat_six && !fat_expand { six.amp.len() as i32 } else { 0 };
    println!("fat: {} ({} fat sources with the per-sample six-peak spectrum)", fat_mode, tfat.iter().filter(|&&x| x != 0.0).count());
    let Value::List(bframes) = v.get("blood").unwrap() else { panic!("blood") };
    #[allow(clippy::type_complexity)]
    let bf: Vec<(f64, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = bframes
        .iter()
        .map(|f| {
            let (m, om, r2, idx, e1, zo) = mk(f);
            let (pos, vel) = (tf32(f, "pos"), tf32(f, "vel"));
            let (pos, vel) = if idx.len() == pos.len() / 3 { (pos, vel) } else { (expand(&pos, &idx), expand(&vel, &idx)) };
            (fl(f, "tc"), pos, vel, m, om, r2, e1, zo)
        })
        .collect();
    let nbf = bf.len();
    println!("sources: {:.2} M tissue, {} blood frames of ≈ {:.2} M ({:.1}s)", nt as f64 / 1e6, nbf, bf.iter().map(|b| b.3.len()).sum::<usize>() as f64 / nbf as f64 / 1e6, t0.elapsed().as_secs_f64());
    // beat library: per frame the rows (source → moment record, −1 without LBM coverage) and, per library beat, the
    // covered sources' moments at the frame's phase (the two library bins around it mixed in time), f16
    /// the beat library's sources: per frame the covered sources' rows and cells; a library beat's f16 moments for all
    /// frames are computed when a heartbeat first uses it and kept in a small LRU (all beats at once: ≈ 2.4 GB each
    /// at 0.5 mm, too much for 30 beats)
    struct BeatSrc {
        rows: Vec<Vec<i32>>,
        cells_of: Vec<Vec<u32>>,
        phase: Vec<(usize, usize, f64)>,
        lib: hasty_bench::beats::BeatLibrary,
        cache: Vec<(usize, std::sync::Arc<Vec<Vec<u16>>>)>,
        cache_cap: usize,
        misses: usize,
        t_moments: f64,
        seq: Vec<usize>,
        n_lib: usize,
        /// the library carries a spin history for this protocol (1) or not (0), and 1 / the blood PD's inflow gain
        spin: i32,
        pd_scale: f32,
        /// a second library solved in the heart's reference frame (the coronaries, --beats-ref): its rows follow the
        /// main library's in each frame
        rl: Option<RefSrc>,
    }
    /// per frame and covered blood source: the reference-frame cells inside the source voxel (cell, voxel fraction),
    /// the frame map at the voxel (F row-major, u_wall) and the covered fraction
    struct RefSrc {
        lib: hasty_bench::beats::BeatLibrary,
        n_lib: usize,
        phase: Vec<(usize, usize, f64)>,
        parts: Vec<Vec<Vec<(u32, f32)>>>,
        maps: Vec<Vec<[f32; 12]>>,
        cover: Vec<Vec<f32>>,
    }
    impl BeatSrc {
        fn beat(&mut self, k: usize) -> std::sync::Arc<Vec<Vec<u16>>> {
            use rayon::prelude::*;
            if let Some(i) = self.cache.iter().position(|e| e.0 == k) {
                let e = self.cache.remove(i);
                self.cache.push(e.clone());
                return e.1;
            }
            let t0 = Instant::now();
            let lib = &self.lib;
            let m: Vec<Vec<u16>> = self
                .phase
                .iter()
                .zip(&self.cells_of)
                .map(|(&(b0, b1, a), cl)| {
                    let (d0, d1) = (lib.bin(k, b0), lib.bin(k, b1));
                    cl.par_iter()
                        .flat_map_iter(|&c| {
                            // the coverage is geometric (the same in every beat); a cell missing in this beat keeps zeros
                            let m = hasty_bench::beats::BeatLibrary::moments(&d0, &d1, a, c, 0.0).unwrap_or([0f32; hasty_bench::beats::NM]);
                            m.map(hasty_cfd::motion::f32_to_f16).into_iter()
                        })
                        .collect()
                })
                .collect();
            let mut m = m;
            if let Some(rs) = &self.rl {
                let kr = k % rs.n_lib.max(1);
                for (f, mf) in m.iter_mut().enumerate() {
                    let (b0, b1, a) = rs.phase[f];
                    let (d0, d1) = (rs.lib.bin(kr, b0), rs.lib.bin(kr, b1));
                    let recs: Vec<[f32; hasty_bench::beats::NM]> = rs.parts[f]
                        .par_iter()
                        .zip(&rs.maps[f])
                        .zip(&rs.cover[f])
                        .map(|((parts, mp), &cv)| {
                            let ps: Vec<(f64, [f32; hasty_bench::beats::NM])> = parts
                                .iter()
                                .filter_map(|&(c, wv)| hasty_bench::beats::BeatLibrary::moments(&d0, &d1, a, c, 0.0).map(|r| (wv as f64 * r[0] as f64, r)))
                                .collect();
                            match hasty_bench::beats::mix(&ps, cv as f64) {
                                Some(r) => {
                                    let fm = [[mp[0] as f64, mp[1] as f64, mp[2] as f64], [mp[3] as f64, mp[4] as f64, mp[5] as f64], [mp[6] as f64, mp[7] as f64, mp[8] as f64]];
                                    hasty_bench::beats::transform(&r, &fm, &[mp[9] as f64, mp[10] as f64, mp[11] as f64])
                                }
                                None => [0f32; hasty_bench::beats::NM],
                            }
                        })
                        .collect();
                    for r in recs {
                        mf.extend(r.map(hasty_cfd::motion::f32_to_f16));
                    }
                }
            }
            let m = std::sync::Arc::new(m);
            if self.cache.len() >= self.cache_cap {
                self.cache.remove(0);
            }
            self.cache.push((k, m.clone()));
            self.misses += 1;
            self.t_moments += t0.elapsed().as_secs_f64();
            m
        }
    }
    let mut beat_src: Option<BeatSrc> = beats_run.as_ref().map(|run| {
        use rayon::prelude::*;
        let mut lib = hasty_bench::beats::BeatLibrary::open(run).unwrap_or_else(|e| panic!("--beats: {e}"));
        // the bin cache holds one beat's bins (a beat's moments read each bin once)
        lib.cache_cap = lib.bins + 2;
        let n_lib = lib.beats.len().min(beat_max);
        assert!((lib.period - period).abs() < 1e-3 * period, "beat library period {} ≠ the phantom's {period}", lib.period);
        // a spin history is protocol-specific (TR, flip, slab): it must be this scan's, or the signal is wrong
        let spin = !lib.spin.is_null() && lib.nf >= 28 && !beat_ignore_spin;
        let gain = match v.get("blood_inflow_gain") {
            Some(Value::Float(x)) => *x,
            Some(Value::Int(x)) => *x as f64,
            _ => 1.5,
        };
        if spin {
            let pr = &lib.spin["protocol"];
            let g = |k: &str| pr[k].as_f64().unwrap_or(f64::NAN);
            let gv = |k: &str| -> Vec<f64> { pr[k].as_array().map_or(vec![], |a| a.iter().filter_map(|x| x.as_f64()).collect()) };
            let (lc, ln) = (gv("center"), gv("normal"));
            let mism = [
                ((g("tr") / tr - 1.0).abs() > 0.02, format!("TR {:.2} vs {:.2} ms", g("tr") * 1e3, tr * 1e3)),
                ((g("flip_deg") - flip.to_degrees()).abs() > 0.5, format!("flip {:.1} vs {:.1}°", g("flip_deg"), flip.to_degrees())),
                ((g("thk") - slab_thk).abs() > 2e-3, format!("slab {:.0} vs {:.0} mm", g("thk") * 1e3, slab_thk * 1e3)),
                (lc.len() != 3 || (lc[2] - slab_c[2]).abs() > 2e-3, format!("slab centre z {lc:?} vs {slab_c:?}")),
                (ln.len() != 3 || ln[2] < 0.999, format!("slab normal {ln:?} vs [0, 0, 1]")),
            ];
            let bad: Vec<&String> = mism.iter().filter(|m| m.0).map(|m| &m.1).collect();
            assert!(bad.is_empty(), "--beats: the library's spin history is for another protocol: {bad:?} (rerun cardiac_lbm --spin with this scan's)");
            println!("beats: spin history of the library (TR {:.2} ms, flip {:.1}°, slab {:.0} mm) replaces the steady state; blood PD ÷ inflow gain {gain}", tr * 1e3, flip.to_degrees(), slab_thk * 1e3);
        } else {
            println!("beats: no spin history in the library (fields {}): steady-state blood signal with the inflow gain {gain}", lib.nf);
        }
        let tb0 = Instant::now();
        let at_phase = |tc: f64| {
            let s = tc.rem_euclid(period) / period * lib.bins as f64 - 0.5;
            let f = s.floor();
            ((f as i64).rem_euclid(lib.bins as i64) as usize, ((f as i64 + 1).rem_euclid(lib.bins as i64)) as usize, s - f)
        };
        let mut rows = vec![];
        let mut cells_of = vec![];
        for b in &bf {
            let (b0, b1, a) = at_phase(b.0);
            let (d0, d1) = (lib.bin(0, b0), lib.bin(0, b1));
            let n = b.3.len();
            let cells: Vec<Option<u32>> = (0..n).into_par_iter().map(|i| {
                let x = [b.1[3 * i] as f64, b.1[3 * i + 1] as f64, b.1[3 * i + 2] as f64];
                let c = lib.cell_of(&x)?;
                hasty_bench::beats::BeatLibrary::moments(&d0, &d1, a, c, beat_cover).map(|_| c)
            }).collect();
            let mut r = vec![-1i32; n];
            let mut cl = vec![];
            for (i, c) in cells.iter().enumerate() {
                if let Some(c) = c {
                    r[i] = cl.len() as i32;
                    cl.push(*c);
                }
            }
            rows.push(r);
            cells_of.push(cl);
        }
        let ncov: usize = cells_of.iter().map(|c| c.len()).sum();
        println!("beats: {} library beats of {run:?} ({} bins), {:.2} M covered sources per frame, {:.1} GB of f16 moments per beat (LRU of {beat_cache})", n_lib, lib.bins, ncov as f64 / nbf as f64 / 1e6, (ncov * hasty_bench::beats::NM * 2) as f64 / 1e9);
        let phase: Vec<(usize, usize, f64)> = bf.iter().map(|b| at_phase(b.0)).collect();
        let n_beats = ecg.r_peaks.len();
        let seq = hasty_bench::beats::beat_sequence(n_lib, n_beats, beat_seed);
        println!("beats: sources ready ({:.1}s), {n_beats} heartbeats in the scan", tb0.elapsed().as_secs_f64());
        // ── the reference-frame library (coronaries): sources whose voxel holds its cells, mapped by the cardiac motion
        let mut rows = rows;
        let rl = beats_ref.as_ref().map(|rrun| {
            let mut rlib = hasty_bench::beats::BeatLibrary::open(rrun).unwrap_or_else(|e| panic!("--beats-ref: {e}"));
            rlib.cache_cap = rlib.bins + 2;
            assert!((rlib.period - period).abs() < 1e-3 * period, "--beats-ref period {} ≠ {period}", rlib.period);
            if spin != (!rlib.spin.is_null() && rlib.nf >= 28) {
                println!("beats-ref: WARNING the reference library's spin history differs from the main library's");
            }
            // host cardiac displacement (m) [nt][3][Z][Y][X], linear in time
            let cmh = hasty_value::read(&mut std::io::BufReader::new(std::fs::File::open(v.get("cardiac_motion").and_then(|x| x.as_str()).unwrap()).unwrap())).unwrap();
            let ut_ = cmh.get("U_mm").and_then(|x| x.as_tensor()).unwrap();
            let sh_ = ut_.shape().to_vec();
            let (ntm, zd, yd, xd) = (sh_[0], sh_[2], sh_[3], sh_[4]);
            let uh: Vec<f32> = (0..ut_.numel()).map(|i| (ut_.get_f64(i).0 * 1e-3) as f32).collect();
            let co_ = fls(&cmh, "origin_m");
            let ch_ = fl(&cmh, "voxel_m");
            let u_at = |x: &[f64; 3], tc: f64| -> [f64; 3] {
                let s_ = tc.rem_euclid(period) / period * ntm as f64;
                let i0 = (s_.floor() as usize) % ntm;
                let i1 = (i0 + 1) % ntm;
                let at = s_ - s_.floor();
                let g = [0, 1, 2].map(|a| ((x[a] - co_[a]) / ch_ - 0.5).clamp(0.0, [xd, yd, zd][a] as f64 - 1.001));
                let gi = g.map(|v| v.floor() as usize);
                let gf = [0, 1, 2].map(|a| g[a] - gi[a] as f64);
                let mut u = [0.0; 3];
                for (ti, wt) in [(i0, 1.0 - at), (i1, at)] {
                    for c in 0..3 {
                        let base = (ti * 3 + c) * zd * yd * xd;
                        let mut acc = 0.0;
                        for dz in 0..2 {
                            for dy in 0..2 {
                                for dx_ in 0..2 {
                                    let w = (if dx_ == 1 { gf[0] } else { 1.0 - gf[0] }) * (if dy == 1 { gf[1] } else { 1.0 - gf[1] }) * (if dz == 1 { gf[2] } else { 1.0 - gf[2] });
                                    let idx = base + ((gi[2] + dz).min(zd - 1) * yd + (gi[1] + dy).min(yd - 1)) * xd + (gi[0] + dx_).min(xd - 1);
                                    acc += w * uh[idx] as f64;
                                }
                            }
                        }
                        u[c] += wt * acc;
                    }
                }
                u
            };
            let inv = |x: &[f64; 3], tc: f64| -> [f64; 3] {
                let mut xr = *x;
                for _ in 0..8 {
                    let u = u_at(&xr, tc);
                    xr = [x[0] - u[0], x[1] - u[1], x[2] - u[2]];
                }
                xr
            };
            let bh = fl(&v, "blood_voxel_m");
            let sub = beat_ref_sub.max(1);
            let at_ph = |tc: f64| {
                let s_ = tc.rem_euclid(period) / period * rlib.bins as f64 - 0.5;
                let f_ = s_.floor();
                ((f_ as i64).rem_euclid(rlib.bins as i64) as usize, ((f_ as i64 + 1).rem_euclid(rlib.bins as i64)) as usize, s_ - f_)
            };
            let d0r = rlib.bin(0, 0);
            let (mut parts, mut maps, mut cover) = (vec![], vec![], vec![]);
            let mut nref = 0usize;
            for (fi, b) in bf.iter().enumerate() {
                let tc = b.0;
                let n = b.3.len();
                let ent: Vec<Option<(Vec<(u32, f32)>, [f32; 12], f32)>> = (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let x = [b.1[3 * i] as f64, b.1[3 * i + 1] as f64, b.1[3 * i + 2] as f64];
                        let xr = inv(&x, tc);
                        rlib.cell_of(&xr)?; // near the vessels (active tiles) only
                        let mut cnt: std::collections::BTreeMap<u32, f32> = Default::default();
                        let mut cov = 0f32;
                        let w1 = 1.0 / (sub * sub * sub) as f32;
                        for kz in 0..sub {
                            for ky in 0..sub {
                                for kx in 0..sub {
                                    let off = [kx, ky, kz].map(|k| ((k as f64 + 0.5) / sub as f64 - 0.5) * bh);
                                    let p = [x[0] + off[0], x[1] + off[1], x[2] + off[2]];
                                    let pr = inv(&p, tc);
                                    if let Some(c) = rlib.cell_of(&pr) {
                                        let wc = d0r.get(c).map_or(0.0, |r| r[0]);
                                        if wc > 0.0 {
                                            *cnt.entry(c).or_insert(0.0) += w1;
                                            cov += w1 * wc;
                                        }
                                    }
                                }
                            }
                        }
                        if cov < beat_cover {
                            return None;
                        }
                        // F = I + ∂U/∂X at the reference point, u_wall = ∂U/∂t
                        let e = 0.5 * ch_;
                        let mut mp = [0f32; 12];
                        for a in 0..3 {
                            let mut xp = xr;
                            let mut xm = xr;
                            xp[a] += e;
                            xm[a] -= e;
                            let (up, um) = (u_at(&xp, tc), u_at(&xm, tc));
                            for c in 0..3 {
                                mp[3 * c + a] = ((if a == c { 1.0 } else { 0.0 }) + (up[c] - um[c]) / (2.0 * e)) as f32;
                            }
                        }
                        let dtm = period / ntm as f64 / 2.0;
                        let (u1, u0) = (u_at(&xr, tc + dtm), u_at(&xr, tc - dtm));
                        for c in 0..3 {
                            mp[9 + c] = ((u1[c] - u0[c]) / (2.0 * dtm)) as f32;
                        }
                        Some((cnt.into_iter().collect(), mp, cov))
                    })
                    .collect();
                let (mut pf, mut mf, mut cf) = (vec![], vec![], vec![]);
                let base = cells_of[fi].len() as i32;
                for (i, e) in ent.into_iter().enumerate() {
                    if let Some((pp, mp, cv)) = e {
                        rows[fi][i] = base + pf.len() as i32;
                        pf.push(pp);
                        mf.push(mp);
                        cf.push(cv);
                    }
                }
                nref += pf.len();
                parts.push(pf);
                maps.push(mf);
                cover.push(cf);
            }
            println!("beats-ref: {} library beats of {rrun:?} ({} bins, cells {:.2} mm), {:.0} covered sources per frame (reference frame, {sub}³ points per source)",
                     rlib.beats.len(), rlib.bins, rlib.h * 1e3, nref as f64 / nbf as f64);
            let n_lr = rlib.beats.len();
            let phase_r: Vec<(usize, usize, f64)> = bf.iter().map(|b| at_ph(b.0)).collect();
            RefSrc { lib: rlib, n_lib: n_lr, phase: phase_r, parts, maps, cover }
        });
        BeatSrc { rows, cells_of, phase, lib, cache: vec![], cache_cap: beat_cache.max(1), misses: 0, t_moments: 0.0, seq, n_lib, spin: spin as i32, pd_scale: (1.0 / gain) as f32, rl }
    });

    if plan {
        let nb_mean = bf.iter().map(|b| b.3.len()).sum::<usize>() as f64 / nbf.max(1) as f64;
        let terms = (nt as f64 + nb_mean) * (n_ro * na) as f64 * nc as f64;
        // measured direct-sum throughput (RTX 4090, this kernel): ≈ 4.6·10¹² source·sample·coil terms per second
        let est_s = terms / 4.6e12;
        let mut tab: Vec<u8> = Vec::with_capacity(n_ro * 14 * 8);
        for i in 0..n_ro {
            for x in acq.readout_row(i) {
                tab.extend_from_slice(&x.to_le_bytes());
            }
        }
        write_npy(&out.join("readouts.npy"), "<f8", &[n_ro, 14], false, &tab).unwrap();
        let write_samples = |sch: &dyn Schedule, name: &str, cols: &[usize]| {
            let mut sk: Vec<u8> = Vec::with_capacity(n_ro * na * 4 * cols.len());
            for c0 in (0..n_ro).step_by(256) {
                for r in sch.sample_params(c0..(c0 + 256).min(n_ro)).chunks(SAMPLE_FLOATS) {
                    for &q in cols {
                        sk.extend_from_slice(&r[q].to_le_bytes());
                    }
                }
            }
            write_npy(&out.join(name), "<f4", &[n_ro, na, cols.len()], false, &sk).unwrap();
        };
        match &nominal {
            Some(nom) => {
                write_samples(nom.as_ref(), "samples.npy", &[0, 1, 2, 13]);
                write_samples(acq.as_ref(), "samples_played.npy", &[0, 1, 2, 3, 4, 5, 13, 15]);
            }
            None => write_samples(acq.as_ref(), "samples.npy", &[0, 1, 2, 13]),
        }
        if seq_kind == "flow" {
            // full SAMPLE_FLOATS records (hasty-recon SimData reads them): nominal and, with a GIRF, played
            let all: Vec<usize> = (0..SAMPLE_FLOATS).collect();
            write_samples(nominal.as_deref().unwrap_or(acq.as_ref()), "sched_nominal.npy", &all);
            if nominal.is_some() {
                write_samples(acq.as_ref(), "sched_played.npy", &all);
            }
        }
        let pj = serde_json::json!({"seq": seq_meta, "n_readouts": n_ro, "n_adc": na, "coils": nc, "tissue_sources": nt, "blood_sources_mean": nb_mean,
                                    "terms": terms, "estimated_gpu_s": est_s, "scan_time": scan_time});
        std::fs::write(out.join("plan.json"), serde_json::to_string_pretty(&pj).unwrap()).unwrap();
        println!("plan: {n_ro} readouts × {na} samples × {nc} coils × {:.2} M sources = {terms:.2e} terms ≈ {:.1} min of GPU ({:.1}s) → {}", (nt as f64 + nb_mean) / 1e6, est_s / 60.0, t0.elapsed().as_secs_f64(), out.display());
        return;
    }

    // ── GPU ──
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let m = dev.load_module(&format!("simphantom_nc{nc}_v7"), SRC, &CompileSettings { options: vec![format!("-DNC={nc}")], no_cache: false }).unwrap();
    let (kcomb, kprep, kdsum) = (m.load_function("combine").unwrap(), m.load_function("prep").unwrap(), m.load_function("dsum").unwrap());
    let kbsig = m.load_function("bsig").unwrap();
    // cardiac displacement samples (m)
    let cm = hasty_value::read(&mut std::io::BufReader::new(std::fs::File::open(v.get("cardiac_motion").and_then(|x| x.as_str()).unwrap()).unwrap())).unwrap();
    let ut = cm.get("U_mm").and_then(|x| x.as_tensor()).unwrap();
    let us = ut.shape().to_vec();
    let nu = us[0];
    let cdims = [us[4], us[3], us[2]];
    let co = fls(&cm, "origin_m");
    let ch = fl(&cm, "voxel_m");
    let u_host: Vec<f32> = (0..ut.numel()).map(|i| (ut.get_f64(i).0 * 1e-3) as f32).collect();
    let d_u = st.clone_htod(&u_host).unwrap();
    drop(u_host);
    let nv_c = 3 * cdims[0] * cdims[1] * cdims[2];
    let mut d_uc = st.alloc_zeros::<f32>(nv_c).unwrap();
    let mut d_vc = st.alloc_zeros::<f32>(nv_c).unwrap();
    let c0 = [co[0] + 0.5 * ch, co[1] + 0.5 * ch, co[2] + 0.5 * ch];
    // respiratory field (sum of the bases, m at r = 1)
    let rmo = hasty_value::read(&mut std::io::BufReader::new(std::fs::File::open(v.get("resp_motion").and_then(|x| x.as_str()).unwrap()).unwrap())).unwrap();
    let rb = rmo.get("B").and_then(|x| x.as_tensor()).unwrap();
    let rs_ = rb.shape().to_vec();
    let rn = rs_[2] * rs_[3] * rs_[4];
    let rbv = rb.to_vec::<f32>().unwrap();
    let mut s_host = vec![0f32; 3 * rn];
    for k in 0..rs_[0] {
        for q in 0..3 * rn {
            s_host[q] += rbv[k * 3 * rn + q];
        }
    }
    let ro = fls(&rmo, "origin_m");
    let rh = fl(&rmo, "voxel_m");
    let d_s = st.clone_htod(&s_host).unwrap();
    // gradient nonlinearity: f(xs) and J on a 4 mm grid over the source region (scanner frame)
    let gnl = if ideal { GradientCoils::ideal() } else { GradientCoils::clinical_synthetic() };
    let (mut lo, mut hi) = ([f64::MAX; 3], [f64::MIN; 3]);
    for p in tpos.chunks(3) {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a] as f64 - iso[a] - 0.03);
            hi[a] = hi[a].max(p[a] as f64 - iso[a] + 0.03);
        }
    }
    let gh = 4e-3;
    let gd = [0, 1, 2].map(|a| ((hi[a] - lo[a]) / gh).ceil() as usize + 1);
    let gn = gd[0] * gd[1] * gd[2];
    let mut g_host = vec![0f32; 12 * gn];
    for k in 0..gd[2] {
        for j in 0..gd[1] {
            for i in 0..gd[0] {
                let xs = [lo[0] + i as f64 * gh, lo[1] + j as f64 * gh, lo[2] + k as f64 * gh];
                let f = gnl.field(&xs);
                let jm = gnl.jacobian(&xs);
                let q = i + gd[0] * (j + gd[1] * k);
                for a in 0..3 {
                    g_host[a * gn + q] = f[a] as f32;
                    for b in 0..3 {
                        g_host[(3 + 3 * a + b) * gn + q] = jm[a][b] as f32;
                    }
                }
            }
        }
    }
    let d_g = st.clone_htod(&g_host).unwrap();
    // coil sensitivities (re, im interleaved per coil)
    let sd = sens.re[0].dims;
    let sn_ = sd[0] * sd[1] * sd[2];
    let mut c_host = vec![0f32; 2 * nc * sn_];
    for c in 0..nc {
        c_host[(2 * c) * sn_..(2 * c + 1) * sn_].copy_from_slice(&sens.re[c].data);
        c_host[(2 * c + 1) * sn_..(2 * c + 2) * sn_].copy_from_slice(&sens.im[c].data);
    }
    let d_c = st.clone_htod(&c_host).unwrap();
    let sa = sens.re[0].affine;
    // sources on the GPU
    let d_tpos = st.clone_htod(&tpos).unwrap();
    let d_tfat = st.clone_htod(&tfat).unwrap();
    let d_fattab = st.clone_htod(&fat_tab).unwrap();
    let (d_tm, d_tom, d_tr2) = (st.clone_htod(&tm).unwrap(), st.clone_htod(&tom).unwrap(), st.clone_htod(&tr2).unwrap());
    let (d_te1, d_tzo) = (st.clone_htod(&te1).unwrap(), st.clone_htod(&tzo).unwrap());
    let (d_telh, d_telg) = if tel_on { (st.clone_htod(&telh).unwrap(), st.clone_htod(&telg).unwrap()) } else { (st.alloc_zeros::<f32>(1).unwrap(), st.alloc_zeros::<f32>(3).unwrap()) };
    let (el_on, el_off) = (tel_on as i32, 0i32);
    let d_bf: Vec<_> = bf
        .iter()
        .map(|b| (b.0, st.clone_htod(&b.1).unwrap(), st.clone_htod(&b.2).unwrap(), st.clone_htod(&b.3).unwrap(), st.clone_htod(&b.4).unwrap(), st.clone_htod(&b.5).unwrap(), b.3.len(), st.clone_htod(&b.6).unwrap(), st.clone_htod(&b.7).unwrap()))
        .collect();
    let d_atab = st.clone_htod(&prof.alpha).unwrap();
    let (ntab, az0, adz) = (prof.alpha.len() as i32, prof.z0 as f32, prof.dz as f32);
    let nbmax0 = d_bf.iter().map(|b| b.6).max().unwrap_or(1);
    // with a beat library every blood source becomes six sub-sources (zero amplitude where unused)
    let nbmax = if beat_src.is_some() { 6 * nbmax0 } else { nbmax0 };
    let mut d_src = st.alloc_zeros::<f32>((nt + nbmax) * SRC_FLOATS).unwrap();
    let d_bfat = st.alloc_zeros::<f32>(nbmax.max(1)).unwrap();
    let mut d_amp = st.alloc_zeros::<C32>((nt + nbmax) * nc).unwrap();
    let d_rows: Vec<CudaSlice<i32>> = beat_src.as_ref().map_or(vec![], |bs| bs.rows.iter().map(|r| st.clone_htod(r).unwrap()).collect());
    let (mut d_p6, mut d_v6, mut d_m6, mut d_o6, mut d_r6) = if beat_src.is_some() {
        (st.alloc_zeros::<f32>(3 * nbmax).unwrap(), st.alloc_zeros::<f32>(3 * nbmax).unwrap(), st.alloc_zeros::<f32>(nbmax).unwrap(), st.alloc_zeros::<f32>(nbmax).unwrap(), st.alloc_zeros::<f32>(nbmax).unwrap())
    } else {
        (st.alloc_zeros::<f32>(1).unwrap(), st.alloc_zeros::<f32>(1).unwrap(), st.alloc_zeros::<f32>(1).unwrap(), st.alloc_zeros::<f32>(1).unwrap(), st.alloc_zeros::<f32>(1).unwrap())
    };
    let (mut d_e6, mut d_z6) = if beat_src.is_some() { (st.alloc_zeros::<f32>(nbmax).unwrap(), st.alloc_zeros::<f32>(nbmax).unwrap()) } else { (st.alloc_zeros::<f32>(1).unwrap(), st.alloc_zeros::<f32>(1).unwrap()) };
    let mut d_mom = st.alloc_zeros::<u16>(1).unwrap();
    let mut mom_key: Option<(usize, usize)> = None;
    let ug = grid(&d_uc, &st, cdims, 3, c0, ch, false);
    let vg = grid(&d_vc, &st, cdims, 3, c0, ch, false);
    let sg = grid(&d_s, &st, [rs_[4], rs_[3], rs_[2]], 3, [ro[0] + 0.5 * rh, ro[1] + 0.5 * rh, ro[2] + 0.5 * rh], rh, true);
    let gg = grid(&d_g, &st, gd, 12, lo, gh, true);
    let cg = grid(&d_c, &st, sd, 2 * nc, [sa[0][3], sa[1][3], sa[2][3]], sa[0][0], false);
    println!("GPU ready ({:.1}s)", t0.elapsed().as_secs_f64());

    // breathing table
    let br = v.get("breathing").unwrap();
    let (rt, rdt, bdt) = (tf32(br, "r"), tf32(br, "rdot"), fl(br, "dt"));
    let resp = |t: f64| -> (f32, f32) {
        if stat {
            return (0.0, 0.0);
        }
        let f = (t / bdt).clamp(0.0, (rt.len() - 2) as f64);
        let i = f as usize;
        let a = (f - i as f64) as f32;
        (rt[i] * (1.0 - a) + rt[i + 1] * a, rdt[i] * (1.0 - a) + rdt[i + 1] * a)
    };

    // ── bins ──
    let bin = bin_ms * 1e-3;
    let nbins = (scan_time / bin).ceil() as usize + 1;
    let mut by_bin: Vec<Vec<usize>> = vec![vec![]; nbins];
    for i in 0..n_ro {
        by_bin[((acq.t_exc(i) / bin) as usize).min(nbins - 1)].push(i);
    }
    // k-space goes straight to disk (a 10-minute scan is ≈ 54 GB at 32 coils): data_clean.part.npy, pre-sized, every
    // finished chunk writes its readout rows at their offsets; the chunk's resume state is only a marker (its rows)
    let row_bytes = (na * nc * 8) as u64;
    let k_hdr = npy_header("<c8", &[n_ro, na, nc], false);
    let k_part = out.join("data_clean.part.npy");
    let k_need = k_hdr.len() as u64 + n_ro as u64 * row_bytes;
    let k_new = !(k_part.exists() && std::fs::metadata(&k_part).map(|m| m.len() == k_need).unwrap_or(false));
    if k_new {
        use std::io::Write;
        let mut f = std::fs::File::create(&k_part).unwrap();
        f.write_all(&k_hdr).unwrap();
        f.set_len(k_need).unwrap();
    }
    let mut kfile = std::fs::OpenOptions::new().read(true).write(true).open(&k_part).unwrap();
    let write_rows = |f: &mut std::fs::File, rows: &[usize], vals: &[num_complex::Complex32]| {
        use std::io::{Seek, SeekFrom, Write};
        let w = na * nc;
        for (k, &r) in rows.iter().enumerate() {
            let b: Vec<u8> = vals[k * w..(k + 1) * w].iter().flat_map(|c| [c.re.to_le_bytes(), c.im.to_le_bytes()].concat()).collect();
            f.seek(SeekFrom::Start(k_hdr.len() as u64 + r as u64 * row_bytes)).unwrap();
            f.write_all(&b).unwrap();
        }
        f.sync_data().unwrap();
    };
    let (fcx, fcy, fcz) = (fp.fov_center[0] as f32, fp.fov_center[1] as f32, fp.fov_center[2] as f32);
    let (isx, isy, isz) = (iso[0] as f32, iso[1] as f32, iso[2] as f32);
    // the slab profile is centred on the slab (the RF frequency offset), not necessarily on the FOV
    let (pcx, pcy, pcz) = (slab_c[0] as f32, slab_c[1] as f32, slab_c[2] as f32);

    let chunk_bins = 250;
    let timing = std::env::var("SIMPH_TIMING").is_ok();
    let mut tstage = [0f64; 5];
    let mut tick = Instant::now();
    let mut lap = |st: &std::sync::Arc<cudarc::driver::CudaStream>, k: usize, tstage: &mut [f64; 5]| {
        if timing {
            st.synchronize().unwrap();
            tstage[k] += tick.elapsed().as_secs_f64();
            tick = Instant::now();
        }
    };
    let t2 = Instant::now();
    // paced runs: measured GPU seconds per (source chunk × sample) of the direct sum (ns_ref = 0: not measured yet)
    let (mut per_chunk_s, mut ns_ref) = (1e-9f64, 0usize);
    let mut states = vec![];
    let mut b0 = 0;
    let mut terms = 0f64;
    while b0 < nbins {
        let b1 = (b0 + chunk_bins).min(nbins);
        let state = format!("sim_bins_{b0}_{b1}");
        states.push(state.clone());
        let rows: Vec<usize> = (b0..b1).flat_map(|b| by_bin[b].iter().cloned()).collect();
        if let Some(sv) = ctl.load(&state).unwrap() {
            let t = sv.as_tensor().unwrap();
            // (an empty chunk's marker has 0 elements either way: it is a marker)
            if !rows.is_empty() && t.numel() == rows.len() * na * nc {
                // an older state holds the chunk's samples: into the file
                let o: Vec<num_complex::Complex32> = t.to_vec().unwrap();
                write_rows(&mut kfile, &rows, &o);
                ctl.save(&state, &Value::Tensor(hasty_value::Tensor::from_slice(&[rows.len()], &rows.iter().map(|&r| r as f32).collect::<Vec<f32>>()).unwrap())).unwrap();
                println!("  bins {b0}..{b1}: from the saved chunk (moved to the data file)");
            } else {
                assert_eq!(t.numel(), rows.len(), "{state}: stale chunk state");
                assert!(!k_new, "{state} is done but {} was missing: rerun after deleting the state files (or --fresh)", k_part.display());
                println!("  bins {b0}..{b1}: done (in the data file)");
            }
            b0 = b1;
            continue;
        }
        let mut chunk_out = vec![num_complex::Complex32::new(0.0, 0.0); rows.len() * na * nc];
        let mut ro_off = 0;
        for b in b0..b1 {
            if by_bin[b].is_empty() {
                continue;
            }
            let tb = (b as f64 + 0.5) * bin;
            let ct = ecg.at(tb);
            let tc = if stat { 0.0 } else { ct.phase * period };
            // the heartbeat's time scale: the template cycle stretched to this RR, velocities × period / RR
            let rho = if stat { 1.0 } else { period / ct.rr };
            let (r, rdot) = resp(tb);
            // cardiac displacement / velocity grids at tc (periodic Catmull-Rom)
            let s = (tc.rem_euclid(period)) / period * nu as f64;
            let i = s.floor() as i64;
            let f = s - i as f64;
            let w = [(-f.powi(3) + 2.0 * f * f - f) / 2.0, (3.0 * f.powi(3) - 5.0 * f * f + 2.0) / 2.0, (-3.0 * f.powi(3) + 4.0 * f * f + f) / 2.0, (f.powi(3) - f * f) / 2.0];
            let dw = [(-3.0 * f * f + 4.0 * f - 1.0) / 2.0, (9.0 * f * f - 10.0 * f) / 2.0, (-9.0 * f * f + 8.0 * f + 1.0) / 2.0, (3.0 * f * f - 2.0 * f) / 2.0].map(|x| x * nu as f64 / period);
            let ks = [0, 1, 2, 3].map(|j| ((i - 1 + j as i64).rem_euclid(nu as i64)) as i32);
            let nvl = nv_c as i64;
            let (w0, w1, w2, w3) = (w[0] as f32, w[1] as f32, w[2] as f32, w[3] as f32);
            let (d0, d1, d2, d3) = (dw[0] as f32, dw[1] as f32, dw[2] as f32, dw[3] as f32);
            let (wz, dz) = if stat { (0f32, 0f32) } else { (1f32, rho as f32) };
            let (w0, w1, w2, w3, d0, d1, d2, d3) = (w0 * wz, w1 * wz, w2 * wz, w3 * wz, d0 * dz, d1 * dz, d2 * dz, d3 * dz);
            unsafe {
                st.launch_builder(&kcomb)
                    .arg(&d_u).arg(&nvl).arg(&ks[0]).arg(&ks[1]).arg(&ks[2]).arg(&ks[3])
                    .arg(&w0).arg(&w1).arg(&w2).arg(&w3).arg(&d0).arg(&d1).arg(&d2).arg(&d3)
                    .arg(&mut d_uc).arg(&mut d_vc)
                    .launch(LaunchConfig { grid_dim: ((nv_c as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                    .unwrap();
            }
            lap(&st, 0, &mut tstage);
            // tissue sources
            let ntl = nt as i64;
            let (mode0, zero, one) = (0i32, 0f32, 1f32);
            unsafe {
                st.launch_builder(&kprep)
                    .arg(&mode0).arg(&ntl).arg(&d_tpos).arg(&d_tpos).arg(&d_tm).arg(&d_tom).arg(&d_tr2).arg(&d_tfat).arg(&d_te1).arg(&d_tzo).arg(&d_telh).arg(&d_telg).arg(&el_on)
                    .arg(&ug).arg(&vg).arg(&sg).arg(&gg).arg(&cg).arg(&r).arg(&rdot).arg(&zero)
                    .arg(&isx).arg(&isy).arg(&isz).arg(&fcx).arg(&fcy).arg(&fcz).arg(&pcx).arg(&pcy).arg(&pcz).arg(&d_atab).arg(&ntab).arg(&az0).arg(&adz).arg(&one)
                    .arg(&mut d_src).arg(&mut d_amp)
                    .launch(LaunchConfig { grid_dim: ((nt as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                    .unwrap();
            }
            lap(&st, 1, &mut tstage);
            // blood sources from the nearest stored cardiac frame
            let kf = ((tc / period * nbf as f64).round() as usize) % nbf;
            let bfr = &d_bf[kf];
            let mut dl = tc - bfr.0;
            if dl > 0.5 * period {
                dl -= period;
            }
            if dl < -0.5 * period {
                dl += period;
            }
            // physical time from the frame to the bin (the template offset stretched to this heartbeat)
            let dl = if stat { 0f32 } else { (dl / rho) as f32 };
            let nbl = bfr.6 as i64;
            let mode1 = 1i32;
            let vs = rho as f32;
            let nb_used = if let Some(bs) = beat_src.as_mut() {
                // the heartbeat's library beat; its moments at this frame (uploaded when (beat, frame) changes)
                let k = bs.seq[ct.beat.min(bs.seq.len() - 1)] % bs.n_lib;
                if mom_key != Some((k, kf)) {
                    let mb = bs.beat(k);
                    let mm = &mb[kf];
                    d_mom = if mm.is_empty() { st.alloc_zeros::<u16>(1).unwrap() } else { st.clone_htod(mm).unwrap() };
                    mom_key = Some((k, kf));
                }
                unsafe {
                    st.launch_builder(&kbsig)
                        .arg(&nbl).arg(&bfr.1).arg(&bfr.2).arg(&bfr.3).arg(&bfr.4).arg(&bfr.5).arg(&bfr.7).arg(&bfr.8).arg(&d_rows[kf]).arg(&d_mom).arg(&vs).arg(&bs.spin).arg(&bs.pd_scale)
                        .arg(&mut d_p6).arg(&mut d_v6).arg(&mut d_m6).arg(&mut d_o6).arg(&mut d_r6).arg(&mut d_e6).arg(&mut d_z6)
                        .launch(LaunchConfig { grid_dim: ((bfr.6 as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                        .unwrap();
                }
                let n6 = 6 * nbl;
                let one = 1f32;
                let mut src_b = d_src.slice_mut(nt * SRC_FLOATS..);
                let mut amp_b = d_amp.slice_mut(nt * nc..);
                unsafe {
                    st.launch_builder(&kprep)
                        .arg(&mode1).arg(&n6).arg(&d_p6).arg(&d_v6).arg(&d_m6).arg(&d_o6).arg(&d_r6).arg(&d_bfat).arg(&d_e6).arg(&d_z6).arg(&d_telh).arg(&d_telg).arg(&el_off)
                        .arg(&ug).arg(&vg).arg(&sg).arg(&gg).arg(&cg).arg(&r).arg(&rdot).arg(&dl)
                        .arg(&isx).arg(&isy).arg(&isz).arg(&fcx).arg(&fcy).arg(&fcz).arg(&pcx).arg(&pcy).arg(&pcz).arg(&d_atab).arg(&ntab).arg(&az0).arg(&adz).arg(&one)
                        .arg(&mut src_b).arg(&mut amp_b)
                        .launch(LaunchConfig { grid_dim: (((6 * bfr.6) as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                        .unwrap();
                }
                6 * bfr.6
            } else {
                let mut src_b = d_src.slice_mut(nt * SRC_FLOATS..);
                let mut amp_b = d_amp.slice_mut(nt * nc..);
                unsafe {
                    st.launch_builder(&kprep)
                        .arg(&mode1).arg(&nbl).arg(&bfr.1).arg(&bfr.2).arg(&bfr.3).arg(&bfr.4).arg(&bfr.5).arg(&d_bfat).arg(&bfr.7).arg(&bfr.8).arg(&d_telh).arg(&d_telg).arg(&el_off)
                        .arg(&ug).arg(&vg).arg(&sg).arg(&gg).arg(&cg).arg(&r).arg(&rdot).arg(&dl)
                        .arg(&isx).arg(&isy).arg(&isz).arg(&fcx).arg(&fcy).arg(&fcz).arg(&pcx).arg(&pcy).arg(&pcz).arg(&d_atab).arg(&ntab).arg(&az0).arg(&adz).arg(&vs)
                        .arg(&mut src_b).arg(&mut amp_b)
                        .launch(LaunchConfig { grid_dim: ((bfr.6 as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                        .unwrap();
                }
                bfr.6
            };
            let np_ = nt + nb_used;
            lap(&st, 2, &mut tstage);
            // samples of the bin's readouts (δ = t_exc − t_bin)
            let mut smp = Vec::with_capacity(by_bin[b].len() * na * SAMPLE_FLOATS);
            for &ri in &by_bin[b] {
                let mut sp = acq.sample_params(ri..ri + 1);
                let dlt = (acq.t_exc(ri) - tb) as f32;
                for n in 0..na {
                    sp[n * SAMPLE_FLOATS + 14] = if stat { 0.0 } else { dlt };
                    if ideal {
                        for q in 9..13 {
                            sp[n * SAMPLE_FLOATS + q] = 0.0;
                        }
                    }
                }
                smp.extend(sp);
            }
            let ns = by_bin[b].len() * na;
            lap(&st, 3, &mut tstage);
            let d_smp = st.clone_htod(&smp).unwrap();
            let mut d_out = st.alloc_zeros::<f64>(2 * ns * nc).unwrap();
            // ≈ 2048 blocks fill the GPU; more source chunks only add f64 atomics on the same outputs
            let ychunks = (2048 / (ns as u64).div_ceil(128)).clamp(1, 4096);
            let chunk_u = (np_ as u64).div_ceil(ychunks).max(2048);
            let ny = (np_ as u64).div_ceil(chunk_u) as u32;
            let chunk = chunk_u as i64;
            let (ns_i, np_l) = (ns as i32, np_ as i64);
            // unpaced: one launch over all source chunks; paced (HASTY_GPU_WATTS / DUTY): slices of ≈ HASTY_GPU_SLICE s
            // of GPU time (whole source chunks, the outputs accumulate atomically either way) with a pace point after
            // each, so the duty cycle switches fast; the slice size is calibrated from the measured time per chunk
            let launch = |y0: u32, n: u32, out: &mut CudaSlice<f64>| {
                let yo = y0 as i32;
                unsafe {
                    st.launch_builder(&kdsum)
                        .arg(&d_smp).arg(&ns_i).arg(&d_src).arg(&d_amp).arg(&np_l).arg(&chunk).arg(out).arg(&d_fattab).arg(&npk).arg(&yo)
                        .launch(LaunchConfig { grid_dim: ((ns as u32).div_ceil(128), n, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 })
                        .unwrap();
                }
            };
            match hasty_cuda::pace::slice_seconds() {
                None => launch(0, ny, &mut d_out),
                Some(slice) => {
                    let mut y0 = 0u32;
                    while y0 < ny {
                        let n = if ns_ref == 0 { 1 } else { ((slice / (per_chunk_s * ns as f64).max(1e-9)) as u32).clamp(1, ny - y0) };
                        let t = Instant::now();
                        launch(y0, n, &mut d_out);
                        st.synchronize().unwrap();
                        // seconds per (source chunk × sample): the next slices aim at `slice` seconds
                        let per = t.elapsed().as_secs_f64() / (n as f64 * ns as f64);
                        per_chunk_s = if ns_ref == 0 { per } else { 0.7 * per_chunk_s + 0.3 * per };
                        ns_ref = 1;
                        y0 += n;
                        hasty_cuda::pace::pace_synced();
                    }
                }
            }
            let o = st.clone_dtoh(&d_out).unwrap();
            lap(&st, 4, &mut tstage);
            if timing && b % 50 == 0 {
                println!("    bin {b}: combine+prep tissue+blood+samples+dsum cumulative {:?} s (ns {ns}, np {np_}, grid {} × {})", tstage.map(|x| (x * 100.0).round() / 100.0), (ns as u32).div_ceil(128), ny);
            }
            for k in 0..ns * nc {
                chunk_out[ro_off * na * nc + k] = num_complex::Complex32::new(o[2 * k] as f32, o[2 * k + 1] as f32);
            }
            ro_off += by_bin[b].len();
            terms += ns as f64 * np_ as f64 * nc as f64;
            // GPU pacing point (HASTY_GPU_DUTY / HASTY_GPU_WATTS; a no-op unless configured)
            hasty_cuda::pace::pace(&st);
        }
        write_rows(&mut kfile, &rows, &chunk_out);
        drop(chunk_out);
        ctl.save(&state, &Value::Tensor(hasty_value::Tensor::from_slice(&[rows.len()], &rows.iter().map(|&r| r as f32).collect::<Vec<f32>>()).unwrap())).unwrap();
        let el = t2.elapsed().as_secs_f64();
        println!("  bins {b0}..{b1} of {nbins}: {} readouts, {:.1}s, {:.2e} terms/s", rows.len(), el, terms / el.max(1e-9));
        b0 = b1;
        if b0 < nbins && ctl.stop_requested() {
            println!("stopped after bins ..{b0} of {nbins}: rerun the same command to resume");
            std::process::exit(0);
        }
    }
    println!("simulation {:.1}s", t2.elapsed().as_secs_f64());

    // ── outputs ──
    drop(kfile);
    let k_path = out.join("data_clean.npy");
    std::fs::rename(&k_part, &k_path).unwrap();
    // streamed passes over the file: the RMS, then the noisy copy (same row order and random stream as before)
    let rows_per = (64 << 20) / row_bytes.max(1) as usize + 1;
    let for_rows = |f: &mut dyn FnMut(&[C32])| {
        use std::io::{Read, Seek, SeekFrom};
        let mut fh = std::fs::File::open(&k_path).unwrap();
        fh.seek(SeekFrom::Start(k_hdr.len() as u64)).unwrap();
        let mut buf = vec![0u8; rows_per * row_bytes as usize];
        let mut left = n_ro;
        while left > 0 {
            let n = left.min(rows_per);
            let b = &mut buf[..n * row_bytes as usize];
            fh.read_exact(b).unwrap();
            let v: Vec<C32> = b.chunks_exact(8).map(|x| C32::new(f32::from_le_bytes([x[0], x[1], x[2], x[3]]), f32::from_le_bytes([x[4], x[5], x[6], x[7]]))).collect();
            f(&v);
            left -= n;
        }
    };
    let mut ss = 0f64;
    for_rows(&mut |v: &[C32]| ss += v.iter().map(|c| (c.re as f64).powi(2) + (c.im as f64).powi(2)).sum::<f64>());
    let rms = (ss / (n_ro * na * nc) as f64).sqrt();
    let sigma = noise_rel * rms;
    if write_noisy {
        let l = hasty_phantom::scene::cholesky(&psi);
        let mut s = 0x9E3779B97F4A7C15u64;
        let mut gauss = || {
            let mut u = || {
                s ^= s >> 12;
                s ^= s << 25;
                s ^= s >> 27;
                ((s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos()
        };
        use std::io::Write;
        let mut fo = std::io::BufWriter::new(std::fs::File::create(out.join("data.npy")).unwrap());
        fo.write_all(&k_hdr).unwrap();
        for_rows(&mut |v: &[C32]| {
            let mut noisy = v.to_vec();
            for row in noisy.chunks_mut(nc) {
                let wv: Vec<Complex64> = (0..nc).map(|_| Complex64::new(gauss(), gauss()) * (sigma / 2f64.sqrt())).collect();
                for c in 0..nc {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for k in 0..=c {
                        acc += wv[k] * l[c][k];
                    }
                    row[c] = C32::new(row[c].re + acc.re as f32, row[c].im + acc.im as f32);
                }
            }
            for c in &noisy {
                fo.write_all(&c.re.to_le_bytes()).unwrap();
                fo.write_all(&c.im.to_le_bytes()).unwrap();
            }
        });
        fo.flush().unwrap();
    }
    let ro_tab: Vec<u8> = (0..n_ro).flat_map(|i| acq.readout_row(i).into_iter().flat_map(|x| x.to_le_bytes())).collect();
    write_npy(&out.join("readouts.npy"), "<f8", &[n_ro, 14], false, &ro_tab).unwrap();
    // true breathing state per readout (r, ṙ at the excitation; 0 with --static): validation of self-gating
    let rs_tab: Vec<u8> = (0..n_ro).flat_map(|i| { let (r, rd) = resp(acq.t_exc(i)); [r as f64, rd as f64].into_iter().flat_map(|x| x.to_le_bytes()) }).collect();
    write_npy(&out.join("resp.npy"), "<f8", &[n_ro, 2], false, &rs_tab).unwrap();
    // external physiology as a scanner records it (what real scans use): finger pulse oximeter (PPG) peaks = R peak +
    // pulse transit time (250 ms ± 15 ms per beat), respiratory bellows = abdominal excursion: the breathing state
    // 150 ms later (belt lag), mildly nonlinear, slow baseline drift, noise. physio.npy [readout][5]:
    // t_exc, true ECG phase, PPG phase (between PPG peaks), bellows signal, true breathing state r
    {
        let mut sd = 0x2545_F491_4F6C_DD1Du64;
        let mut u = || {
            sd ^= sd << 13;
            sd ^= sd >> 7;
            sd ^= sd << 17;
            (sd >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let ppg: Vec<f64> = ecg.r_peaks.iter().map(|&r| r + 0.25 + 0.03 * u()).collect();
        let ppg_phase = |t: f64| -> f64 {
            let i = ppg.partition_point(|&p| p <= t);
            if i == 0 || i >= ppg.len() { f64::NAN } else { (t - ppg[i - 1]) / (ppg[i] - ppg[i - 1]) }
        };
        let tab: Vec<u8> = (0..n_ro)
            .flat_map(|i| {
                let t = acq.t_exc(i);
                let rl = resp((t - 0.15).max(0.0)).0 as f64;
                let bel = rl * (1.0 + 0.1 * rl) + 0.05 * (2.0 * std::f64::consts::PI * t / 60.0).sin() + 0.01 * u();
                [t, ecg.at(t).phase, ppg_phase(t), bel, resp(t).0 as f64].into_iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()
            })
            .collect();
        write_npy(&out.join("physio.npy"), "<f8", &[n_ro, 5], false, &tab).unwrap();
    }
    if seq_kind != "spiral" {
        // nominal trajectory and sample times (what a reconstruction uses), in chunks; with a GIRF also the played one
        // (k, γM1, τ and the B0-term phase: what a reconstruction that models the chain exactly would use)
        // streamed to the file in chunks of 256 readouts: the full tables of a 10-minute scan are ≈ 15 GB, which next
        // to the simulator's own working set would exceed its memory cap
        let write_samples = |sch: &dyn Schedule, name: &str, cols: &[usize]| {
            use std::io::Write;
            let mut fo = std::io::BufWriter::with_capacity(16 << 20, std::fs::File::create(out.join(name)).unwrap());
            fo.write_all(&hasty_phantom::volume::npy_header("<f4", &[n_ro, na, cols.len()], false)).unwrap();
            let mut sk: Vec<u8> = Vec::with_capacity(256 * na * 4 * cols.len());
            for c0 in (0..n_ro).step_by(256) {
                sk.clear();
                let sp = sch.sample_params(c0..(c0 + 256).min(n_ro));
                for r in sp.chunks(SAMPLE_FLOATS) {
                    for &q in cols {
                        sk.extend_from_slice(&r[q].to_le_bytes());
                    }
                }
                fo.write_all(&sk).unwrap();
            }
            fo.flush().unwrap();
        };
        match &nominal {
            Some(nom) => {
                write_samples(nom.as_ref(), "samples.npy", &[0, 1, 2, 13]);
                write_samples(acq.as_ref(), "samples_played.npy", &[0, 1, 2, 3, 4, 5, 13, 15]);
            }
            None => write_samples(acq.as_ref(), "samples.npy", &[0, 1, 2, 13]),
        }
        if seq_kind == "flow" {
            // full SAMPLE_FLOATS records (hasty-recon SimData reads them): nominal and, with a GIRF, played
            let all: Vec<usize> = (0..SAMPLE_FLOATS).collect();
            write_samples(nominal.as_deref().unwrap_or(acq.as_ref()), "sched_nominal.npy", &all);
            if nominal.is_some() {
                write_samples(acq.as_ref(), "sched_played.npy", &all);
            }
        }
    }
    // true sensitivities and field on the reconstruction grid (axial logical frame, index i ↦ (i − n/2) res)
    let dims = [0, 1, 2].map(|a| (fov[a] / res).round() as usize);
    let nvox = dims[0] * dims[1] * dims[2];
    let mut smaps = vec![Complex64::new(0.0, 0.0); nvox * nc];
    let mut sb = vec![Complex64::new(0.0, 0.0); nc];
    for k in 0..dims[2] {
        for j in 0..dims[1] {
            for i in 0..dims[0] {
                let x = [center[0] + (i as f64 - (dims[0] / 2) as f64) * res, center[1] + (j as f64 - (dims[1] / 2) as f64) * res, center[2] + (k as f64 - (dims[2] / 2) as f64) * res];
                sens.at(&x, &mut sb);
                let q = i + dims[0] * (j + dims[1] * k);
                for c in 0..nc {
                    smaps[q + nvox * c] = sb[c];
                }
            }
        }
    }
    let c64b = |d: &[Complex64]| -> Vec<u8> { d.iter().flat_map(|c| [(c.re as f32).to_le_bytes(), (c.im as f32).to_le_bytes()].concat()).collect() };
    write_npy(&out.join("smaps_true.npy"), "<c8", &[dims[0], dims[1], dims[2], nc], true, &c64b(&smaps)).unwrap();
    let fr = tf32(&v, "field_recon");
    write_npy(&out.join("field_ppm_recon.npy"), "<f4", &dims, true, &fr.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    // reference-pose truth maps on the reconstruction grid from the input tissue sources (PD-weighted box sums):
    // water / fat PD, fat fraction, R2* (PD-weighted mean of 1/T2*); blood (Eulerian frames) not included
    {
        let tsv = v.get("tissue").unwrap();
        let (pos, pd, t2s) = (tf32(tsv, "pos"), tf32(tsv, "pd"), tf32(tsv, "t2s"));
        let sh = tsv.get("shift_ppm").map(|_| tf32(tsv, "shift_ppm")).unwrap_or(vec![0.0; pd.len()]);
        let (mut pw, mut pf, mut r2a) = (vec![0f64; nvox], vec![0f64; nvox], vec![0f64; nvox]);
        // box elements (--src-octree) spread their PD over the reconstruction voxels they cover
        let esize = tsv.get("size_m").map(|_| tf32(tsv, "size_m"));
        for i in 0..pd.len() {
            let hs = esize.as_ref().map_or(0.0, |e| e[i] as f64);
            let nsub = ((hs / res).round() as isize).max(1);
            let p = pd[i] as f64 / (nsub * nsub * nsub) as f64;
            for sz in 0..nsub {
                for sy in 0..nsub {
                    for sx in 0..nsub {
                        let sub = [sx, sy, sz];
                        let idx: Vec<isize> = (0..3)
                            .map(|a| {
                                let x = pos[3 * i + a] as f64 + if nsub > 1 { ((sub[a] as f64 + 0.5) / nsub as f64 - 0.5) * hs } else { 0.0 };
                                ((x - center[a]) / res + (dims[a] / 2) as f64).round() as isize
                            })
                            .collect();
                        if (0..3).any(|a| idx[a] < 0 || idx[a] >= dims[a] as isize) {
                            continue;
                        }
                        let q = idx[0] as usize + dims[0] * (idx[1] as usize + dims[1] * idx[2] as usize);
                        if sh[i] != 0.0 { pf[q] += p } else { pw[q] += p }
                        r2a[q] += p / (t2s[i] as f64).max(1e-4);
                    }
                }
            }
        }
        let wr = |name: &str, x: Vec<f32>| write_npy(&out.join(name), "<f4", &dims, true, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let tot: Vec<f64> = (0..nvox).map(|q| pw[q] + pf[q]).collect();
        wr("truth_r2s.npy", (0..nvox).map(|q| if tot[q] > 0.0 { (r2a[q] / tot[q]) as f32 } else { f32::NAN }).collect());
        wr("truth_ff.npy", (0..nvox).map(|q| if tot[q] > 0.0 { (pf[q] / tot[q]) as f32 } else { f32::NAN }).collect());
        wr("truth_pd_water.npy", pw.iter().map(|&x| x as f32).collect());
        wr("truth_pd_fat.npy", pf.iter().map(|&x| x as f32).collect());
    }
    let meta = serde_json::json!({
        "subject": v.get("phantom").and_then(|x| x.as_str()).unwrap_or(""), "simulator": "simphantom (moving atlas phantom)",
        "b0": fp.b0, "venc": fp.venc, "venc_hi": venc_hi, "encoding": format!("{:?}", fp.encoding), "coverage": format!("{:?}", fp.coverage),
        "fov": fp.fov, "res": fp.res, "recon_dims": dims, "n_int": fp.n_int, "fov_edge_ratio": fp.fov_edge_ratio,
        "dt": fp.dt, "dwell": fp.dwell, "t_rf": fp.t_rf, "t_spoil": fp.t_spoil, "scan_time": fp.scan_time,
        "gmax": fp.lim.gmax, "smax": fp.lim.smax, "flip_deg": flip.to_degrees(),
        "rot_presc": fp.rot_presc, "fov_center_scanner": fp.fov_center, "iso_subject": iso, "presc_center_subject": center,
        "mean_rr": fp.mean_rr, "rr_rel_std": fp.rr_rel_std, "n_frames": fp.n_frames, "seed": fp.seed,
        "tr": tr, "te": acq.te(), "n_readouts": n_ro, "n_adc": na, "n_coils": nc, "sequence": seq_meta, "schedule": acq.describe(),
        "fat_model": if !fat_six { "single peak (input shift_ppm)" } else if fat_expand { "six-peak, expanded sub-sources (hasty-qfit FatSpectrum::six_peak)" } else { "six-peak per sample in the kernel (hasty-qfit FatSpectrum::six_peak)" },
        "noise_sigma": sigma, "noise_rel": noise_rel, "psi": psi,
        "gnl": {"r0": gnl.r0, "transverse": gnl.transverse, "longitudinal": gnl.longitudinal}, "ideal": ideal, "static": stat, "bin_ms": bin_ms,
        "n_tissue_sources": nt, "n_blood_frames": nbf, "res_truth": fp.res,
        // the beat library's heartbeat → library beat sequence (the truth of each heartbeat), its spin history mode
        "beats": beat_src.as_ref().map(|bs| serde_json::json!({"run": beats_run.as_ref().map(|r| r.display().to_string()), "seed": beat_seed, "cover": beat_cover,
            "library_cycles": bs.lib.beats[..bs.n_lib].to_vec(), "sequence": bs.seq, "spin": bs.spin == 1, "pd_scale": bs.pd_scale,
            "moment_computations": bs.misses, "moment_seconds": bs.t_moments,
            "ref_run": beats_ref.as_ref().map(|r| r.display().to_string())})),
    });
    std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    for s in &states {
        ctl.clear(s);
    }
    println!("done in {:.1}s -> {}", t0.elapsed().as_secs_f64(), out.display());
}
