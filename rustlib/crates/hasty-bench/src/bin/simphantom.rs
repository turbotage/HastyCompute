//! k-space of a moving factory phantom (direct summation on the GPU), from `tools/phantom/prep_sim.py`.
//!   simphantom <sim_input.hv> <out_dir> [--bin-ms 10] [--venc 1.5] [--venc-hi v] [--noise 0.02] [--rr-std 0] [--flip 12]
//!              [--ideal] [--static] [--write-noisy] [--fat six|single] [--scan-time s] [--plan]
//!              [--seq spiral | b0 [--b0-res 2e-3] [--b0-k 24] [--b0-pns 80] [--dwell 3e-6]
//!                     | flow [--flow-run data/hasty/seq/flow_thorax] [--girf ideal|clinical|<variant>] [--dwell 2e-6]]
//! `--seq` picks the schedule (`hasty_mrsim::sequence::Schedule`): the spiral 4D-flow `Acquisition` (default) or the
//! hasty-seq B0 / R2* / fat train (`hasty_bench::seqsched::EchoSchedule`, also writes samples.npy [readout][adc][4] =
//! k (rad/m, physical) and the time since excitation). Fat sources (shift ≠ 0) are split into the six-peak spectrum
//! (hasty-qfit `FatSpectrum::six_peak`, the fit's model) unless `--fat single`. `--seq flow`: the final 4D-flow yarn TR
//! of a seqdesign run (`FlowSchedule`), optionally played through a GIRF (`--girf`: per-sample moments of the played
//! sequence, previous TR included; the B0 terms as a constant phase) — samples.npy then holds the nominal trajectory,
//! samples_played.npy [readout][adc][8] the played one (γM0 = k (rad/m), γM1 (rad·s/m), τ, B0-term phase), physical axes.
//! Flow sims also write sched_nominal.npy (and sched_played.npy) [readout][adc][16]: the full per-sample records the
//! reconstruction loader (hasty-recon `SimData`, `HASTY_TRAJ=nominal|played`) uses.
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
use hasty_phantom::volume::{write_npy, Volume};
use hasty_value::Value;
use num_complex::Complex64;

const SRC: &str = r#"
#ifndef NC
#define NC 1
#endif
#define PF 16
#define SF 20
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
                                Grid Ug, Grid Vg, Grid Sg, Grid Gg, Grid Cg, float r, float rdot, float dlt,
                                float isox, float isoy, float isoz, float fcx, float fcy, float fcz,
                                float pcx, float pcy, float pcz, float slab_half, float slab_edge, float* src, float2* amp)
{
    long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (p >= n) return;
    float x[3] = {pos[3 * p], pos[3 * p + 1], pos[3 * p + 2]}, v[3], t[3];
    if (mode == 0) {
        tri(Ug, x[0], x[1], x[2], 0, 3, t);
        tri(Vg, x[0], x[1], x[2], 0, 3, v);
        for (int c = 0; c < 3; c++) x[c] += t[c];
    } else {
        for (int c = 0; c < 3; c++) { v[c] = vel[3 * p + c]; x[c] += v[c] * dlt; }
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
    s[16] = omega[p]; s[17] = r2s[p]; s[18] = fat[p]; s[19] = 0.f;
    // axial slab (logical z = subject z)
    float dz = slab_half - fabsf(x[2] - pcz);
    float w = dz >= 0.5f * slab_edge ? 1.f : (dz <= -0.5f * slab_edge ? 0.f : 0.5f + dz / slab_edge);
    float cs[2 * NC];
    tri(Cg, x[0], x[1], x[2], 0, 2 * NC, cs);
    float mw = m[p] * w;
    for (int c = 0; c < NC; c++) amp[p * NC + c] = make_float2(mw * cs[2 * c], mw * cs[2 * c + 1]);
}

// out[i][c] += Σ_p amp[p][c] e^{−r2s τ} e^{−iφ}, sources split over blockIdx.y chunks (f64 atomics)
extern "C" __global__ void __launch_bounds__(128) dsum(const float* __restrict__ smp, int ns, const float* __restrict__ src,
                                                       const float2* __restrict__ amp, long long np, long long chunk, double* __restrict__ out,
                                                       const float* __restrict__ fatpk, int npk)
{
    __shared__ float s_src[TS * SF];
    __shared__ float2 s_amp[TS * NC];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < ns;
    const long long p0 = blockIdx.y * chunk, p1 = min(np, p0 + chunk);
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
    let flow_run = PathBuf::from(take("--flow-run").unwrap_or_else(|| "data/hasty/seq/flow_thorax".into()));
    let girf_name = take("--girf").unwrap_or_else(|| "ideal".into());
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
                name => hasty_seq::girf::Girf::variants().into_iter().find(|v| v.0 == name).unwrap_or_else(|| panic!("--girf {name}: ideal | clinical | a Girf::variants() name")).1,
            };
            let fs = FlowSchedule::load(&flow_run, scan_time, dwell_arg, fp.b0, ecg.clone(), fp.n_frames, girf, &girf_name);
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
    // per source: steady-state amplitude, ω, R2*, and the index of the input source (fat sources split into the six
    // peaks of the fit's spectrum: amplitude α_p, shift δ_p replacing the input's single shift)
    let mk = |d: &Value| -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<usize>) {
        let (pd, t1, t2s, fld) = (tf32(d, "pd"), tf32(d, "t1"), tf32(d, "t2s"), tf32(d, "field_ppm"));
        let sh = d.get("shift_ppm").map(|_| tf32(d, "shift_ppm")).unwrap_or(vec![0.0; pd.len()]);
        let (mut m, mut om, mut r2, mut idx) = (vec![], vec![], vec![], vec![]);
        for i in 0..pd.len() {
            let (mi, f, ri) = (ss(pd[i], t1[i]), if ideal { 0.0 } else { fld[i] as f64 }, 1.0 / t2s[i].max(1e-4));
            if sh[i] != 0.0 && fat_six && fat_expand {
                for (&a, &p) in six.amp.iter().zip(&six.ppm) {
                    m.push(mi * a as f32);
                    om.push((gam * (f + p)) as f32);
                    r2.push(ri);
                    idx.push(i);
                }
            } else if sh[i] != 0.0 && fat_six {
                // one source at the water frequency; the kernel multiplies its term by c(τ) (flag in `tfat`)
                m.push(mi);
                om.push((gam * f) as f32);
                r2.push(ri);
                idx.push(i);
            } else {
                m.push(mi);
                om.push((gam * (f + sh[i] as f64)) as f32);
                r2.push(ri);
                idx.push(i);
            }
        }
        (m, om, r2, idx)
    };
    let expand = |x: &[f32], idx: &[usize]| -> Vec<f32> { idx.iter().flat_map(|&i| x[3 * i..3 * i + 3].iter().copied()).collect() };
    let (tm, tom, tr2, tidx) = mk(ts);
    let tpos = if tidx.len() == tpos.len() / 3 { tpos } else { expand(&tpos, &tidx) };
    let nt = tm.len();
    let tfat: Vec<f32> = {
        let sh = ts.get("shift_ppm").map(|_| tf32(ts, "shift_ppm")).unwrap_or(vec![0.0; tidx.len()]);
        tidx.iter().map(|&i| if fat_six && !fat_expand && sh[i] != 0.0 { 1.0 } else { 0.0 }).collect()
    };
    let fat_tab: Vec<f32> = six.amp.iter().map(|&a| a as f32).chain(six.ppm.iter().map(|&p| (gam * p) as f32)).collect();
    let npk: i32 = if fat_six && !fat_expand { six.amp.len() as i32 } else { 0 };
    println!("fat: {} ({} fat sources with the per-sample six-peak spectrum)", fat_mode, tfat.iter().filter(|&&x| x != 0.0).count());
    let Value::List(bframes) = v.get("blood").unwrap() else { panic!("blood") };
    let bf: Vec<(f64, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> = bframes
        .iter()
        .map(|f| {
            let (m, om, r2, idx) = mk(f);
            let (pos, vel) = (tf32(f, "pos"), tf32(f, "vel"));
            let (pos, vel) = if idx.len() == pos.len() / 3 { (pos, vel) } else { (expand(&pos, &idx), expand(&vel, &idx)) };
            (fl(f, "tc"), pos, vel, m, om, r2)
        })
        .collect();
    let nbf = bf.len();
    println!("sources: {:.2} M tissue, {} blood frames of ≈ {:.2} M ({:.1}s)", nt as f64 / 1e6, nbf, bf.iter().map(|b| b.3.len()).sum::<usize>() as f64 / nbf as f64 / 1e6, t0.elapsed().as_secs_f64());

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
    let m = dev.load_module(&format!("simphantom_nc{nc}_v4"), SRC, &CompileSettings { options: vec![format!("-DNC={nc}")], no_cache: false }).unwrap();
    let (kcomb, kprep, kdsum) = (m.load_function("combine").unwrap(), m.load_function("prep").unwrap(), m.load_function("dsum").unwrap());
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
    let d_bf: Vec<_> = bf.iter().map(|b| (b.0, st.clone_htod(&b.1).unwrap(), st.clone_htod(&b.2).unwrap(), st.clone_htod(&b.3).unwrap(), st.clone_htod(&b.4).unwrap(), st.clone_htod(&b.5).unwrap(), b.3.len())).collect();
    let nbmax = d_bf.iter().map(|b| b.6).max().unwrap_or(1);
    let mut d_src = st.alloc_zeros::<f32>((nt + nbmax) * 20).unwrap();
    let d_bfat = st.alloc_zeros::<f32>(nbmax.max(1)).unwrap();
    let mut d_amp = st.alloc_zeros::<C32>((nt + nbmax) * nc).unwrap();
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
    let mut data = vec![C32::new(0.0, 0.0); n_ro * na * nc];
    let (fcx, fcy, fcz) = (fp.fov_center[0] as f32, fp.fov_center[1] as f32, fp.fov_center[2] as f32);
    let (isx, isy, isz) = (iso[0] as f32, iso[1] as f32, iso[2] as f32);
    let (pcx, pcy, pcz) = (center[0] as f32, center[1] as f32, center[2] as f32);
    let (slab_half, slab_edge) = ((0.5 * fov[2]) as f32, 0.002f32);
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
    let mut states = vec![];
    let mut b0 = 0;
    let mut terms = 0f64;
    while b0 < nbins {
        let b1 = (b0 + chunk_bins).min(nbins);
        let state = format!("sim_bins_{b0}_{b1}");
        states.push(state.clone());
        let rows: Vec<usize> = (b0..b1).flat_map(|b| by_bin[b].iter().cloned()).collect();
        if let Some(sv) = ctl.load(&state).unwrap() {
            let o: Vec<num_complex::Complex32> = sv.as_tensor().unwrap().to_vec().unwrap();
            assert_eq!(o.len(), rows.len() * na * nc, "{state}: stale chunk state");
            for (k, &r) in rows.iter().enumerate() {
                for q in 0..na * nc {
                    let x = o[k * na * nc + q];
                    data[r * na * nc + q] = C32::new(x.re, x.im);
                }
            }
            println!("  bins {b0}..{b1}: from the saved chunk");
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
            let tc = if stat { 0.0 } else { ecg.at(tb).phase * period };
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
            let (wz, dz) = if stat { (0f32, 0f32) } else { (1f32, 1f32) };
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
            let (mode0, zero) = (0i32, 0f32);
            unsafe {
                st.launch_builder(&kprep)
                    .arg(&mode0).arg(&ntl).arg(&d_tpos).arg(&d_tpos).arg(&d_tm).arg(&d_tom).arg(&d_tr2).arg(&d_tfat)
                    .arg(&ug).arg(&vg).arg(&sg).arg(&gg).arg(&cg).arg(&r).arg(&rdot).arg(&zero)
                    .arg(&isx).arg(&isy).arg(&isz).arg(&fcx).arg(&fcy).arg(&fcz).arg(&pcx).arg(&pcy).arg(&pcz).arg(&slab_half).arg(&slab_edge)
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
            let dl = if stat { 0f32 } else { dl as f32 };
            let nbl = bfr.6 as i64;
            let mode1 = 1i32;
            {
                let mut src_b = d_src.slice_mut(nt * 20..);
                let mut amp_b = d_amp.slice_mut(nt * nc..);
                unsafe {
                    st.launch_builder(&kprep)
                        .arg(&mode1).arg(&nbl).arg(&bfr.1).arg(&bfr.2).arg(&bfr.3).arg(&bfr.4).arg(&bfr.5).arg(&d_bfat)
                        .arg(&ug).arg(&vg).arg(&sg).arg(&gg).arg(&cg).arg(&r).arg(&rdot).arg(&dl)
                        .arg(&isx).arg(&isy).arg(&isz).arg(&fcx).arg(&fcy).arg(&fcz).arg(&pcx).arg(&pcy).arg(&pcz).arg(&slab_half).arg(&slab_edge)
                        .arg(&mut src_b).arg(&mut amp_b)
                        .launch(LaunchConfig { grid_dim: ((bfr.6 as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })
                        .unwrap();
                }
            }
            let np_ = nt + bfr.6;
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
            unsafe {
                st.launch_builder(&kdsum)
                    .arg(&d_smp).arg(&ns_i).arg(&d_src).arg(&d_amp).arg(&np_l).arg(&chunk).arg(&mut d_out).arg(&d_fattab).arg(&npk)
                    .launch(LaunchConfig { grid_dim: ((ns as u32).div_ceil(128), ny, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 })
                    .unwrap();
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
        }
        for (k, &r) in rows.iter().enumerate() {
            for q in 0..na * nc {
                let x = chunk_out[k * na * nc + q];
                data[r * na * nc + q] = C32::new(x.re, x.im);
            }
        }
        ctl.save(&state, &Value::Tensor(hasty_value::Tensor::from_slice(&[rows.len() * na * nc], &chunk_out).unwrap())).unwrap();
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
    let bytes = |d: &[C32]| -> Vec<u8> { d.iter().flat_map(|c| [c.re.to_le_bytes(), c.im.to_le_bytes()].concat()).collect() };
    write_npy(&out.join("data_clean.npy"), "<c8", &[n_ro, na, nc], false, &bytes(&data)).unwrap();
    let rms = (data.iter().map(|c| (c.re as f64).powi(2) + (c.im as f64).powi(2)).sum::<f64>() / data.len() as f64).sqrt();
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
        let mut noisy = data.clone();
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
        write_npy(&out.join("data.npy"), "<c8", &[n_ro, na, nc], false, &bytes(&noisy)).unwrap();
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
        let write_samples = |sch: &dyn Schedule, name: &str, cols: &[usize]| {
            let mut sk: Vec<u8> = Vec::with_capacity(n_ro * na * 4 * cols.len());
            for c0 in (0..n_ro).step_by(256) {
                let sp = sch.sample_params(c0..(c0 + 256).min(n_ro));
                for r in sp.chunks(SAMPLE_FLOATS) {
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
        for i in 0..pd.len() {
            let idx: Vec<isize> = (0..3).map(|a| ((pos[3 * i + a] as f64 - center[a]) / res + (dims[a] / 2) as f64).round() as isize).collect();
            if (0..3).any(|a| idx[a] < 0 || idx[a] >= dims[a] as isize) {
                continue;
            }
            let q = idx[0] as usize + dims[0] * (idx[1] as usize + dims[1] * idx[2] as usize);
            let p = pd[i] as f64;
            if sh[i] != 0.0 { pf[q] += p } else { pw[q] += p }
            r2a[q] += p / (t2s[i] as f64).max(1e-4);
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
    });
    std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta).unwrap()).unwrap();
    for s in &states {
        ctl.clear(s);
    }
    println!("done in {:.1}s -> {}", t0.elapsed().as_secs_f64(), out.display());
}
