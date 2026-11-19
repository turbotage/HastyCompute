//! Joint coil × segment channels vs today's (coil, segment) channels of the segmented NUFFT
//! (research/coil_segment_compression.md, "Implementation and measured gains").
//!
//!   segjoint_bench time --dims nx,ny,nz [--res 1e-3] [--m 8000000] [--na 3990] [--order 3] [--knots kr,rdeg,mc,cdeg]
//!       [--today C:L,...] [--joint C:R:CBJ,...] [--k 40] [--reps 2] [--grid-gb 6] [--no-gen] [--gen-nv N]
//!     forward / adjoint time (ms, median) of today's operator (C coils × L segments) and of the joint operator
//!     (R channels in batches of CBJ, Φ generated from C coil maps and K time functions per batch, mixing per sample)
//!     on random ball samples in regular readouts of na samples; static engine, or the 2-D knot engine (`--knots`).
//!     --no-gen: timing-only joint operator (Φ a fixed buffer: grids whose maps + h do not fit), and the Φ generation
//!     timed on --gen-nv voxels and scaled by voxels (kernel and CGEMM). Prints the per-stage split (Φ generation,
//!     inner engine, mixing). --chunks N: also the joint operator on N sample chunks (forward_chunks / adjoint_chunks).
//!
//!   segjoint_bench counts <scan dir> --region box|full [--weights pd|warm] [--time noise|uniform] [--tols ..]
//!       [--ro-stride N] [--op] [--rebuild] [--out file.json]
//!     ph0001 accuracy study: channel counts at equal achieved signal-weighted ratemap error for fixed nodes (count LS
//!     and energy-weighted LS), the coil-common SVD segments and the joint basis; achieved errors at the ADC times;
//!     with --op the operators against a high-L reference at the real samples (‖Δy‖/‖y‖ per readout section);
//!     with --rebuild the old basis on a perturbed ratemap (±20 Hz smooth, ±10 1/s) and on other voxel weights.
//!
//!   segjoint_bench head --dims n,n,n [--res 2e-3] [--coils 32] [--tols ..]
//!     synthetic head: ellipsoid object, 32 loops on a helmet (Biot–Savart circular-loop B1⁻ magnitude and phase,
//!     quasi-static), shimmed brain field with a frontal-sinus / temporal-bone susceptibility hot spot; counts as above.
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::CudaStream;
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_joint_basis, build_segments, JointBuildOpts, JointSegNufft, SegNufft, SegNufftOpts, SegPoints, NHI};
use num_complex::Complex64;

const GIB: f64 = (1u64 << 30) as f64;

fn arg(args: &mut Vec<String>, name: &str) -> Option<String> {
    args.iter().position(|a| a == name).map(|i| {
        args.remove(i);
        args.remove(i)
    })
}
fn flag(args: &mut Vec<String>, name: &str) -> bool {
    args.iter().position(|a| a == name).map(|i| { args.remove(i); }).is_some()
}

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

fn trim(dev: &Device, st: &Arc<CudaStream>) {
    st.synchronize().unwrap();
    unsafe {
        let mut pool = std::mem::MaybeUninit::uninit();
        if cudarc::driver::sys::cuDeviceGetDefaultMemPool(pool.as_mut_ptr(), dev.ctx.cu_device()) == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            cudarc::driver::sys::cuMemPoolTrimTo(pool.assume_init(), 0);
        }
    }
}

fn free_gib(dev: &Device) -> f64 {
    dev.ctx.mem_get_info().map(|(f, _)| f as f64 / GIB).unwrap_or(0.0)
}

/// random samples in the k ball (radius 3.1 rad/voxel), regular readouts: τ_n = TE + n · dwell
struct Samples {
    co: [Vec<f32>; 3],
    tau: Vec<f32>,
    t: Vec<f32>,
    ri: Vec<u32>,
    rv: Vec<f32>,
    dc: Vec<f32>,
    uh: Vec<f32>,
    hh: Vec<f64>,
}

impl Samples {
    /// samples j0 .. j1 (planar per-sample arrays: dc 6 × m, uh 4 × m)
    fn slice(&self, j0: usize, j1: usize) -> Samples {
        let m = self.tau.len();
        let pl = |v: &Vec<f32>, k: usize| -> Vec<f32> { if v.is_empty() { vec![] } else { (0..k).flat_map(|q| v[q * m + j0..q * m + j1].iter().copied()).collect() } };
        let sl = |v: &Vec<f32>| if v.is_empty() { vec![] } else { v[j0..j1].to_vec() };
        Samples {
            co: [sl(&self.co[0]), sl(&self.co[1]), sl(&self.co[2])],
            tau: sl(&self.tau),
            t: sl(&self.t),
            ri: if self.ri.is_empty() { vec![] } else { self.ri[j0..j1].to_vec() },
            rv: sl(&self.rv),
            dc: pl(&self.dc, 6),
            uh: pl(&self.uh, 4),
            hh: self.hh.clone(),
        }
    }
}

fn samples(m: usize, na: usize, knots: Option<(usize, u32, usize, u32)>, order: u32) -> Samples {
    let mut r = Rng(0x1234_5678_9abc_def1);
    let mut co: [Vec<f32>; 3] = Default::default();
    while co[0].len() < m {
        let p = [r.u() * 6.2, r.u() * 6.2, r.u() * 6.2];
        if p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= 3.1 * 3.1 {
            (0..3).for_each(|a| co[a].push(p[a] as f32));
        }
    }
    let tau: Vec<f32> = (0..m).map(|j| (1.6e-3 + (j % na) as f64 * 2e-6) as f32).collect();
    let (t, ri, rv) = match knots {
        Some((kr, rdeg, mc, _)) => {
            let nri = if rdeg == 0 { kr } else { kr - 1 };
            let t: Vec<f32> = (0..m).map(|_| ((r.u() + 0.5) * mc as f64) as f32).collect();
            let (ri, rv): (Vec<u32>, Vec<f32>) = (0..m).map(|_| { let x = (r.u() + 0.5) * nri as f64; ((x.floor() as u32).min(nri as u32 - 1), (x - x.floor()) as f32) }).unzip();
            (t, ri, rv)
        }
        None => (vec![], vec![], vec![]),
    };
    let dc: Vec<f32> = (0..6 * m).map(|_| (r.u() * 10.0) as f32).collect();
    let (uh, hh) = if order >= 3 { ((0..4 * m).map(|_| (r.u() * 2.0) as f32).collect(), (0..4 * NHI).map(|_| r.u() * 50.0).collect()) } else { (vec![], vec![]) };
    Samples { co, tau, t, ri, rv, dc, uh, hh }
}

fn make_engine(dev: &Device, dims: [usize; 3], res: f64, nc: usize, nseg: usize, order: u32, knots: Option<(usize, u32, usize, u32)>, grid_gb: f64) -> Result<SegNufft, Box<dyn std::error::Error>> {
    let mg = (grid_gb * GIB) as usize;
    Ok(match knots {
        None => SegNufft::new(dev, dims, [res; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg, order, frames: 1, bspline: 1, max_grid_bytes: mg })?,
        Some((kr, rdeg, mc, cdeg)) => SegNufft::new_temporal2d(dev, dims, [res; 3], nc, SegNufftOpts { sigma: 1.25, width: 10, nseg, order, frames: mc, bspline: cdeg, max_grid_bytes: mg }, kr, rdeg)?,
    })
}

fn make_points(st: &Arc<CudaStream>, e: &mut SegNufft, s: &Samples, knots: bool, order: u32) -> Result<SegPoints, Box<dyn std::error::Error>> {
    let m = s.tau.len();
    let ph0 = vec![0f32; m];
    let c = [&s.co[0][..], &s.co[1][..], &s.co[2][..]];
    Ok(match (knots, order >= 3) {
        (true, true) => e.points2d_hi(st, c, &s.tau, &ph0, &s.dc, &s.t, &s.ri, &s.rv, &s.uh, &s.hh)?,
        (true, false) => e.points2d(st, c, &s.tau, &ph0, &s.dc, &s.t, &s.ri, &s.rv)?,
        (false, true) => e.points_hi(st, c, &s.tau, &ph0, &s.dc, &s.uh, &s.hh, None)?,
        (false, false) => e.points(st, c, &s.tau, &ph0, &s.dc, None)?,
    })
}

fn median_ms(st: &Arc<CudaStream>, reps: usize, f: &mut dyn FnMut()) -> f64 {
    let mut t = vec![];
    for i in 1..=reps {
        st.synchronize().unwrap();
        let t0 = Instant::now();
        f();
        st.synchronize().unwrap();
        if i > 0 {
            t.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t[t.len() / 2]
}

fn cmd_time(mut a: Vec<String>) {
    let dims: Vec<usize> = arg(&mut a, "--dims").expect("--dims").split(',').map(|v| v.parse().unwrap()).collect();
    let dims = [dims[0], dims[1], dims[2]];
    let res: f64 = arg(&mut a, "--res").map_or(1e-3, |v| v.parse().unwrap());
    let m: usize = arg(&mut a, "--m").map_or(8_000_000, |v| v.parse().unwrap());
    let na: usize = arg(&mut a, "--na").map_or(3990, |v| v.parse().unwrap());
    let m = m / na * na;
    let order: u32 = arg(&mut a, "--order").map_or(3, |v| v.parse().unwrap());
    let knots: Option<(usize, u32, usize, u32)> = arg(&mut a, "--knots").map(|v| { let p: Vec<usize> = v.split(',').map(|x| x.parse().unwrap()).collect(); (p[0], p[1] as u32, p[2], p[3] as u32) });
    let today: Vec<(usize, usize)> = arg(&mut a, "--today").map_or(vec![], |v| v.split(',').map(|c| { let p: Vec<usize> = c.split(':').map(|x| x.parse().unwrap()).collect(); (p[0], p[1]) }).collect());
    let joint: Vec<(usize, usize, usize)> = arg(&mut a, "--joint").map_or(vec![], |v| v.split(',').map(|c| { let p: Vec<usize> = c.split(':').map(|x| x.parse().unwrap()).collect(); (p[0], p[1], p[2]) }).collect());
    let k: usize = arg(&mut a, "--k").map_or(40, |v| v.parse().unwrap());
    let reps: usize = arg(&mut a, "--reps").map_or(2, |v| v.parse().unwrap());
    let grid_gb: f64 = arg(&mut a, "--grid-gb").map_or(6.0, |v| v.parse().unwrap());
    let no_gen = flag(&mut a, "--no-gen");
    let gen_nv: usize = arg(&mut a, "--gen-nv").map_or(4_096_000, |v| v.parse().unwrap());
    // --cb-cap N: coil-batch cap of today's engines (default 16; the joint inner engine always uses its batch width)
    let cb_cap: Option<usize> = arg(&mut a, "--cb-cap").map(|v| v.parse().unwrap());
    // --chunks N: the joint operator on N sample chunks of whole readouts (forward_chunks / adjoint_chunks: g buffer of
    // one chunk), next to the whole-set operator
    let chunks: usize = arg(&mut a, "--chunks").map_or(1, |v| v.parse().unwrap());
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let nv: usize = dims.iter().product();
    let nimg = knots.map_or(1, |(kr, _, mc, _)| kr * mc);
    println!("time: dims {dims:?} ({nv} voxels, {res} m), M = {m} ({} readouts × {na}), order {order}, {}", m / na, knots.map_or("static".to_string(), |(kr, rd, mc, cd)| format!("2-D knots {kr} (deg {rd}) × {mc} (deg {cd}) = {nimg} knot images")));
    let ts = Instant::now();
    let s = samples(m, na, knots, order);
    println!("  samples built in {:.1} s", ts.elapsed().as_secs_f64());
    let mut r = Rng(77);
    for &(nc, l) in &today {
        trim(&dev, &st);
        let out: Result<(), Box<dyn std::error::Error>> = (|| {
            hasty_nufft::set_thread_cb_cap(cb_cap);
            let mut e = make_engine(&dev, dims, res, nc, l, order, knots, grid_gb)?;
            hasty_nufft::set_thread_cb_cap(None);
            let smaps: Vec<C32> = (0..nc * nv).map(|i| C32::new(1.0 / (1.0 + (i % 7) as f32), 0.1)).collect();
            e.set_maps(&st, &smaps)?;
            drop(smaps);
            let fit: Vec<usize> = (0..nv).step_by(97).collect();
            let psi = |v: usize, t: f64| 2000.0 * ((v % 113) as f64 / 113.0 - 0.5) * t;
            let (sp, at, t0, dt) = build_segments(nv, l, 1.6e-3, 9.6e-3, 256, &fit, &psi);
            let seg = e.segments(&st, &sp, &at, t0, dt)?;
            drop(sp);
            let pts = make_points(&st, &mut e, &s, knots.is_some(), order)?;
            let x: Vec<C32> = (0..nv * nimg).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
            let xd = st.clone_htod(&x)?;
            drop(x);
            let mut y = st.alloc_zeros::<C32>(nc * m)?;
            let mut xo = st.alloc_zeros::<C32>(nv * nimg)?;
            e.forward(&st, &seg, &pts, &xd, &mut y)?;
            e.adjoint(&st, &seg, &pts, &y, &mut xo)?;
            let tf = median_ms(&st, reps, &mut || e.forward(&st, &seg, &pts, &xd, &mut y).unwrap());
            let ta = median_ms(&st, reps, &mut || e.adjoint(&st, &seg, &pts, &y, &mut xo).unwrap());
            println!("  today C = {nc} × L = {l} ({} channels, coil batch {} → {} passes; tiles fwd {:?} adj {:?}{}): forward {tf:.1} ms, adjoint {ta:.1} ms, pass {:.1} ms; free {:.1} GiB", nc * l, e.cb, nc.div_ceil(e.cb), e.tiles_fwd, e.tiles_adj, e.multi_shapes(1).map_or(String::new(), |(f, a)| format!("; multi kernels fwd {f:?} adj {a:?}")), tf + ta, free_gib(&dev));
            Ok(())
        })();
        if let Err(e) = out {
            println!("  today C = {nc} × L = {l}: failed/skipped: {e}");
        }
    }
    if no_gen && !joint.is_empty() {
        for &(nc, _, cbj) in joint.iter().take(1) {
            trim(&dev, &st);
            match JointSegNufft::gen_phi_ms(&dev, &st, gen_nv, nc, k, cbj, reps) {
                Ok(t) => println!("  Φ generation, kernel (C = {nc}, K = {k}): {t:.2} ms per batch of {cbj} on {gen_nv} voxels → {:.3} ns per voxel·channel; this grid: {:.1} ms per batch", t * 1e6 / (gen_nv * cbj) as f64, t * nv as f64 / gen_nv as f64),
                Err(e) => println!("  Φ generation timing failed: {e}"),
            }
            trim(&dev, &st);
            match JointSegNufft::gen_phi_gemm_ms(&dev, &st, gen_nv, nc, k, cbj, reps) {
                Ok(t) => println!("  Φ generation, CGEMM (C = {nc}, K = {k}): {t:.2} ms per batch of {cbj} on {gen_nv} voxels → {:.3} ns per voxel·channel; this grid: {:.1} ms per batch", t * 1e6 / (gen_nv * cbj) as f64, t * nv as f64 / gen_nv as f64),
                Err(e) => println!("  Φ generation (CGEMM) timing failed: {e}"),
            }
        }
    }
    for &(nc, rr, cbj_max) in &joint {
        trim(&dev, &st);
        let out: Result<(), Box<dyn std::error::Error>> = (|| {
            let cbj = JointSegNufft::batch_for(rr, cbj_max);
            hasty_nufft::set_thread_cb_cap(Some(cbj));
            let inner = make_engine(&dev, dims, res, cbj, 1, order, knots, grid_gb.max(cbj as f64 * 8.0 * 1.3 * 1.3 * 1.3 * nv as f64 / GIB));
            hasty_nufft::set_thread_cb_cap(None);
            let inner = inner?;
            let mut op = if no_gen {
                JointSegNufft::new_timing(&dev, &st, inner, nc, k, rr, na)?
            } else {
                // random basis of the right shape (U orthonormality is irrelevant for the timing)
                let basis = hasty_nufft::JointBasis {
                    nc,
                    nv,
                    k,
                    lam: vec![1.0; rr],
                    u: (0..nc * k * rr).map(|_| Complex64::new(r.u(), r.u())).collect(),
                    ncol: rr,
                    ts: vec![],
                    stw: vec![],
                    v: vec![],
                    time_ev: vec![],
                    tab_times: vec![0.0; na],
                    vt: (0..na * k).map(|_| Complex64::new(r.u(), r.u())).collect(),
                    h: st.clone_htod(&(0..k * nv).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect::<Vec<_>>())?,
                    timing: vec![],
                };
                let smaps: Vec<C32> = (0..nc * nv).map(|i| C32::new(1.0 / (1.0 + (i % 7) as f32), 0.1)).collect();
                JointSegNufft::new(&dev, &st, inner, basis, &smaps, rr)?
            };
            let seg1 = op.segment1(&st, None)?;
            let pts = make_points(&st, &mut op.inner, &s, knots.is_some(), order)?;
            let x: Vec<C32> = (0..nv * nimg).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
            let xd = st.clone_htod(&x)?;
            drop(x);
            let mut y = st.alloc_zeros::<C32>(nc * m)?;
            let mut xo = st.alloc_zeros::<C32>(nv * nimg)?;
            op.forward(&st, &seg1, &pts, &xd, &mut y)?;
            op.adjoint(&st, &seg1, &pts, &y, &mut xo)?;
            let tf = median_ms(&st, reps, &mut || op.forward(&st, &seg1, &pts, &xd, &mut y).unwrap());
            let ta = median_ms(&st, reps, &mut || op.adjoint(&st, &seg1, &pts, &y, &mut xo).unwrap());
            op.profile = true;
            op.prof_ms = [0.0; 3];
            op.forward(&st, &seg1, &pts, &xd, &mut y)?;
            op.adjoint(&st, &seg1, &pts, &y, &mut xo)?;
            let p = op.prof_ms;
            let tot = p[0] + p[1] + p[2];
            println!(
                "  joint C = {nc}, R = {rr} ({} × {cbj}, inner coil batch {}, Φ {}, tiles fwd {:?} adj {:?}{}): forward {tf:.1} ms, adjoint {ta:.1} ms, pass {:.1} ms; profiled pass: Φ {:.1} ms ({:.1} %), inner {:.1} ms ({:.1} %), mixing {:.1} ms ({:.1} %){}; free {:.1} GiB",
                op.nb,
                op.inner.cb,
                if op.phi_gemm { "CGEMM" } else { "kernel" },
                op.inner.tiles_fwd,
                op.inner.tiles_adj,
                op.inner.multi_shapes(1).map_or(String::new(), |(f, a)| format!("; multi kernels fwd {f:?} adj {a:?}")),
                tf + ta,
                p[0],
                100.0 * p[0] / tot,
                p[1],
                100.0 * p[1] / tot,
                p[2],
                100.0 * p[2] / tot,
                if no_gen { " [timing-only: Φ not generated]" } else { "" },
                free_gib(&dev)
            );
            if chunks > 1 {
                // the same operator on sample chunks of whole readouts (g buffer of one chunk)
                drop(pts);
                let nro = m / na;
                let per = nro.div_ceil(chunks);
                let mut offs = vec![];
                let mut cps = vec![];
                for c in 0..chunks {
                    let (r0, r1) = (c * per, ((c + 1) * per).min(nro));
                    if r0 >= r1 {
                        break;
                    }
                    let sc = s.slice(r0 * na, r1 * na);
                    cps.push(make_points(&st, &mut op.inner, &sc, knots.is_some(), order)?);
                    offs.push(r0 * na);
                }
                let refs: Vec<&SegPoints> = cps.iter().collect();
                op.release_work();
                trim(&dev, &st);
                op.forward_chunks(&st, &seg1, &refs, &offs, m, &xd, &mut y)?;
                op.adjoint_chunks(&st, &seg1, &refs, &offs, m, &y, &mut xo)?;
                let tf = median_ms(&st, reps, &mut || op.forward_chunks(&st, &seg1, &refs, &offs, m, &xd, &mut y).unwrap());
                let ta = median_ms(&st, reps, &mut || op.adjoint_chunks(&st, &seg1, &refs, &offs, m, &y, &mut xo).unwrap());
                let gmax = cps.iter().map(|p| p.m).max().unwrap() * cbj * 8;
                println!("  joint C = {nc}, R = {rr}, {} sample chunks (g buffer {:.2} GiB instead of {:.2} GiB): forward {tf:.1} ms, adjoint {ta:.1} ms, pass {:.1} ms; free {:.1} GiB", cps.len(), gmax as f64 / GIB, (m * cbj * 8) as f64 / GIB, tf + ta, free_gib(&dev));
            }
            Ok(())
        })();
        if let Err(e) = out {
            println!("  joint C = {nc}, R = {rr}, cbj ≤ {cbj_max}: failed/skipped: {e}");
        }
    }
}

fn main() {
    let mut a: Vec<String> = std::env::args().skip(1).collect();
    assert!(!a.is_empty(), "segjoint_bench time|counts|head ...");
    let cmd = a.remove(0);
    match cmd.as_str() {
        "time" => cmd_time(a),
        "counts" => counts::cmd_counts(a),
        "head" => counts::cmd_head(a),
        "coilbox" => coilbox::run(a),
        o => panic!("unknown command {o}"),
    }
}

mod counts {
    use super::*;
    use hasty_nufft::{build_segments_complex, build_segments_svd_grid, nodes_weighted_error, seg_count_for, seg_time_grid, segment_model_error, weighted_spectrum};
    use hasty_recon::data::{read_npy, SimData};
    use hasty_recon::kload::LoadOpts;
    use std::path::PathBuf;

    /// One region's problem in compact voxel indices (i → grid voxel vox[i])
    pub struct Prob {
        pub nc: usize,
        /// coil maps (nc × n), compact
        pub smaps: Vec<C32>,
        /// Σ_c |S_c|²
        pub cen: Vec<f64>,
        /// signal weight (PD energy, or the warm-start estimate)
        pub w: Vec<f64>,
        /// ADC times, per-ADC time weights
        pub tau: Vec<f64>,
        pub tw_adc: Vec<f64>,
        pub label: String,
    }

    pub struct Counts {
        pub tol: f64,
        /// (L, achieved) per basis: fixed (count LS), fixed_e (energy LS), svd; joint (R, achieved)
        pub fixed: (usize, f64),
        pub fixed_e: (usize, f64),
        pub svd: (usize, f64),
        pub joint: (usize, f64),
    }

    fn achieved_seg(p: &Prob, g: &(dyn Fn(usize, f64) -> Complex64 + Sync), sp: &[C32], at: &[C32], l: usize, t0: f64, dt: f64, ev: &[usize], tix: &[usize]) -> f64 {
        let n = p.cen.len();
        let ew: Vec<f64> = ev.iter().map(|&i| p.w[i] * p.cen[i]).collect();
        let taus: Vec<f64> = tix.iter().map(|&k| p.tau[k]).collect();
        let tws: Vec<f64> = tix.iter().map(|&k| p.tw_adc[k]).collect();
        segment_model_error(sp, at, l, n, t0, dt, ev, Some(&ew), &taus, &tws, g)
    }

    /// counts at equal ACHIEVED signal-weighted error (at the ADC times, every 16th, the time weights of the problem):
    /// start from the count the predicted error gives, add channels until the achieved error is ≤ tol
    pub fn counts(dev: &Device, st: &Arc<CudaStream>, p: &Prob, g: &(dyn Fn(usize, f64) -> Complex64 + Sync), tols: &[f64], tsqr: bool, json: &mut serde_json::Value) -> (Vec<Counts>, hasty_nufft::JointBasis, (Vec<f64>, Vec<f64>)) {
        let n = p.cen.len();
        let na = p.tau.len();
        let (t0, t1) = (p.tau[0], p.tau[na - 1]);
        let (ts, tw) = seg_time_grid(&p.tau, &p.tw_adc, 288);
        let all: Vec<usize> = (0..n).collect();
        // fit sets: SVD Gram / joint on every voxel with weight; fixed nodes on every 13th (as build_segments)
        let fit: Vec<usize> = all.iter().copied().filter(|&i| p.w[i] > 0.0).collect();
        let fit13: Vec<usize> = fit.iter().copied().step_by(13).collect();
        let ew13: Vec<f64> = fit13.iter().map(|&i| p.w[i] * p.cen[i]).collect();
        let wseg: Vec<f64> = fit.iter().map(|&i| p.w[i] * p.cen[i]).collect();
        let ev: Vec<usize> = fit.iter().copied().step_by((fit.len() / 20000).max(1)).collect();
        let evj: Vec<usize> = fit.iter().copied().step_by((fit.len() / 6000).max(1)).collect();
        let tix: Vec<usize> = (0..na).step_by(16).collect();
        let t_start = Instant::now();
        println!("  [{}] {} voxels, {} fit, time grid {} points; building the joint basis on the GPU ({})", p.label, n, fit.len(), ts.len(), if tsqr { "fp32 TSQR" } else { "fp64 Gram" });
        let wj: Vec<f64> = fit.iter().map(|&i| p.w[i]).collect();
        let basis = build_joint_basis(dev, st, n, p.nc, &p.smaps, &fit, &wj, &ts, &tw, &p.tau, g, &JointBuildOpts { verbose: true, fp32_tsqr: tsqr, rmax: 1400, ..Default::default() }).unwrap();
        json["joint_build"] = serde_json::json!({"k": basis.k, "ck": basis.lam.len(), "timing": basis.timing.iter().map(|(a, b)| serde_json::json!([a, b])).collect::<Vec<_>>(), "tsqr": tsqr});
        let ev_seg = weighted_spectrum(&fit.iter().copied().step_by(3).collect::<Vec<_>>(), Some(&wseg.iter().copied().step_by(3).collect::<Vec<_>>()), &ts, &tw, g);
        println!("  [{}] coil-common weighted spectrum done ({:.0} s)", p.label, t_start.elapsed().as_secs_f64());
        let ejw: Vec<f64> = evj.iter().map(|&i| p.w[i]).collect();
        let tws: Vec<f64> = tix.iter().map(|&k| p.tw_adc[k]).collect();
        let mut out = vec![];
        for &tol in tols {
            // fixed, count LS (today): predicted scan, then achieved
            let nodes = |l: usize| -> Vec<f64> { (0..l).map(|i| if l == 1 { 0.5 * (t0 + t1) } else { t0 + (t1 - t0) * i as f64 / (l - 1) as f64 }).collect() };
            let mut lf = (2..=48).find(|&l| nodes_weighted_error(&fit13, None, Some(&ew13), &ts, &tw, &nodes(l), g) <= tol).unwrap_or(48);
            let fixed = loop {
                let (sp, at, a0, dt) = build_segments_complex(n, lf, t0, t1, na, &fit13, g);
                let e = achieved_seg(p, g, &sp, &at, lf, a0, dt, &ev, &tix);
                if e <= tol || lf >= 48 {
                    break (lf, e);
                }
                lf += 1;
            };
            let mut lfe = (2..=48).find(|&l| nodes_weighted_error(&fit13, Some(&ew13), Some(&ew13), &ts, &tw, &nodes(l), g) <= tol).unwrap_or(48);
            let fixed_e = loop {
                let (sp, at, a0, dt) = hasty_nufft::build_segments_nodes(n, &nodes(lfe), t0, t1, na, &fit13, Some(&ew13), g);
                let e = achieved_seg(p, g, &sp, &at, lfe, a0, dt, &ev, &tix);
                if e <= tol || lfe >= 48 {
                    break (lfe, e);
                }
                lfe += 1;
            };
            let mut ls = seg_count_for(&ev_seg, tol, 1, 64);
            let svd = loop {
                let (sp, at, a0, dt) = build_segments_svd_grid(n, ls, t0, t1, na, &fit.iter().copied().step_by(3).collect::<Vec<_>>(), Some(&wseg.iter().copied().step_by(3).collect::<Vec<_>>()), &ts, &tw, g);
                let e = achieved_seg(p, g, &sp, &at, ls, a0, dt, &ev, &tix);
                if e <= tol || ls >= 64 {
                    break (ls, e);
                }
                ls += 1;
            };
            let mut r = basis.count_for(tol);
            let joint = loop {
                let e = basis.error(r, &p.smaps, &evj, &ejw, &tix, &tws, g).global;
                if e <= tol || r >= basis.ncol {
                    break (r, e);
                }
                r = (r + (r / 50).max(2)).min(basis.ncol);
            };
            println!(
                "  [{}] ε = {tol:.0e}: fixed (count LS) L = {} ({:.2e}) → {} ch; fixed (energy LS) L = {} ({:.2e}) → {} ch; SVD L = {} ({:.2e}) → {} ch; joint R = {} ({:.2e}; predicted R {}); joint gain vs fixed {:.2}×, vs SVD {:.2}× ({:.0} s)",
                p.label, fixed.0, fixed.1, fixed.0 * p.nc, fixed_e.0, fixed_e.1, fixed_e.0 * p.nc, svd.0, svd.1, svd.0 * p.nc, joint.0, joint.1, basis.count_for(tol),
                (fixed.0 * p.nc) as f64 / joint.0 as f64, (svd.0 * p.nc) as f64 / joint.0 as f64, t_start.elapsed().as_secs_f64()
            );
            out.push(Counts { tol, fixed, fixed_e, svd, joint });
        }
        json["counts"] = serde_json::json!(out.iter().map(|c| serde_json::json!({"tol": c.tol, "nc": p.nc, "fixed": [c.fixed.0, c.fixed.1], "fixed_e": [c.fixed_e.0, c.fixed_e.1], "svd": [c.svd.0, c.svd.1], "joint": [c.joint.0, c.joint.1]})).collect::<Vec<_>>());
        json["svd_tail"] = serde_json::json!((1..=40).map(|l| hasty_nufft::tail_errors(&ev_seg)[l]).collect::<Vec<_>>());
        json["joint_tail"] = serde_json::json!((1..=basis.ncol.min(1400)).step_by(10).map(|r| [r as f64, basis.tail_error(r)]).collect::<Vec<_>>());
        (out, basis, (ts, tw))
    }

    fn sample_truth(sd: &SimData, grid: &hasty_recon::data::PhysicsGrid, file: &str) -> Option<Vec<f64>> {
        let p = sd_dir(sd).join(file);
        let a = read_npy(&p).ok()?.f32s();
        Some((0..sd.nv()).map(|q| { let v = sd.sample_log(&a, 0, &grid.x_log[q]); if v.is_finite() { v } else { 0.0 } }).collect())
    }
    fn sd_dir(sd: &SimData) -> PathBuf {
        sd.dir.clone()
    }

    pub fn cmd_counts(mut a: Vec<String>) {
        let dir = PathBuf::from(a.remove(0));
        let region = arg(&mut a, "--region").unwrap_or_else(|| "box".into());
        let wsel = arg(&mut a, "--weights").unwrap_or_else(|| "pd".into());
        let tsel = arg(&mut a, "--time").unwrap_or_else(|| "noise".into());
        let tols: Vec<f64> = arg(&mut a, "--tols").map_or(vec![1e-2, 1e-3, 1e-4], |v| v.split(',').map(|x| x.parse().unwrap()).collect());
        let gnl = !flag(&mut a, "--no-gnl");
        let tsqr = flag(&mut a, "--tsqr");
        let do_rebuild = flag(&mut a, "--rebuild");
        let do_op = flag(&mut a, "--op");
        let clip: Option<f64> = arg(&mut a, "--clip-hz").map(|v| v.parse().unwrap());
        let out = arg(&mut a, "--out");
        let t0 = Instant::now();
        let sd = SimData::load_lazy(&dir).unwrap();
        let w = sd.whitener();
        let grid = sd.physics_grid(gnl, &w);
        let (dims, nv, nc_all) = (sd.dims, sd.nv(), sd.nc);
        // coils: SV compression as splinerecon's default (whitened encode-0 data, ≤ 2·10⁵ samples, ≥ 99.9 %)
        let ros0 = sd.readouts_of(0, None);
        let stride = (ros0.len() * sd.na / 200_000).max(1);
        let sub: Vec<usize> = ros0.iter().step_by(stride).copied().collect();
        let s0 = sd.load_set(&sub, &LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
        let mut gr = hasty_recon::coils::CoilGram::new(nc_all);
        gr.add(&s0.y, s0.tau.len(), 1);
        let ccx = hasty_recon::coils::CoilCompression::from_gram(&gr, hasty_recon::coils::CoilRule::Energy(0.999));
        println!("coils: {}", ccx.report());
        let nc = ccx.nvc();
        let smaps_all = ccx.apply(&grid.smaps, nv);
        let na = s0.tau.len() / sub.len();
        let tau: Vec<f64> = s0.tau[..na].to_vec();
        let wb = hasty_recon::segphys::mean_coefficients(&[&s0], na);
        // ratemap: off-resonance + encode-0 mean concomitant phase, decay, folded six-peak fat (truth maps: a benchmark
        // of the operator, not a reconstruction)
        let r2 = sample_truth(&sd, &grid, "truth_r2s.npy").expect("truth_r2s.npy");
        let ff = sample_truth(&sd, &grid, "truth_ff.npy").expect("truth_ff.npy");
        let pdw = sample_truth(&sd, &grid, "truth_pd_water.npy").expect("truth_pd_water.npy");
        let pdf = sample_truth(&sd, &grid, "truth_pd_fat.npy").expect("truth_pd_fat.npy");
        let fs = hasty_qfit::waterfat::FatSpectrum::six_peak();
        let om_f: Vec<f64> = fs.ppm.iter().map(|d| 2.0 * std::f64::consts::PI * 42.577_478_51e6 * sd.b0 * d * 1e-6).collect();
        let amp_f = fs.amp.clone();
        let psi = hasty_recon::segphys::segment_phase(&grid, &wb, &tau, true);
        let tau0 = tau[0];
        let wclip = clip.map(|h| 2.0 * std::f64::consts::PI * h);
        let gfull = |q: usize, t: f64| -> Complex64 {
            let mut ph = psi(q, t);
            if let Some(wc) = wclip {
                // simulated shim: the field part clipped to ±clip (the concomitant part unchanged)
                let om = grid.omega[q];
                ph += (om.clamp(-wc, wc) - om) * t;
            }
            let z = Complex64::from_polar((-r2[q] * (t - tau0)).exp(), -ph);
            let c: Complex64 = om_f.iter().zip(&amp_f).map(|(w, a)| Complex64::from_polar(*a, -w * t)).sum();
            z * (Complex64::new(1.0 - ff[q], 0.0) + ff[q] * c)
        };
        // per-ADC time weights: noise-referenced (recon_p2np's svdn / auto:err) or uniform
        let tw_adc: Vec<f64> = if tsel == "noise" {
            let ros: Vec<usize> = (0..sd.n_readouts()).step_by(7).take(512).collect();
            let e = sd.adc_energy_whitened(&ros, &w).unwrap();
            let kap: Vec<f64> = (0..na).map(|n| (0..3).map(|ax| (s0.coords[ax][n] as f64 / sd.vox[ax]).powi(2)).sum::<f64>().sqrt()).collect();
            let sig = sd.meta["noise_sigma"].as_f64().unwrap_or(1.0);
            hasty_nufft::noise_time_weights(&e[..na], &kap, nc_all as f64 * sig * sig)
        } else {
            vec![1.0; na]
        };
        // region: the 160 mm box of research/coil_segment_compression.md (80³ at 2 mm, origin [93, 26, 12]) or the grid
        let reg: Vec<usize> = match region.as_str() {
            "box" => {
                let (o, b) = ([93usize, 26, 12], [80usize, 80, 80]);
                let mut v = vec![];
                for k in o[2]..(o[2] + b[2]).min(dims[2]) {
                    for j in o[1]..(o[1] + b[1]).min(dims[1]) {
                        for i in o[0]..(o[0] + b[0]).min(dims[0]) {
                            v.push(i + dims[0] * (j + dims[1] * k));
                        }
                    }
                }
                v
            }
            "full" => (0..nv).collect(),
            o => panic!("--region box|full, got {o}"),
        };
        // signal weights: PD energy (PD = water + fat; voxels with PD ≤ 0.01 carry none), or warm = data-derived
        // smoothed SENSE-adjoint magnitude² / Σ|S|² (splinerecon's default) on the coil-energy support
        let cen_all: Vec<f64> = (0..nv).map(|q| (0..nc).map(|c| { let z = smaps_all[c * nv + q]; (z.re as f64).powi(2) + (z.im as f64).powi(2) }).sum()).collect();
        let dev = Device::new(0).unwrap();
        let st = dev.default_stream();
        let kern = Arc::new(hasty_recon::gpu::Kernels::new(&dev).unwrap());
        let pd: Vec<f64> = (0..nv).map(|q| pdw[q] + pdf[q]).collect();
        let pdmax = pd.iter().cloned().fold(0.0, f64::max);
        let w_pd: Vec<f64> = pd.iter().map(|&v| if v > 0.01 * pdmax { v * v } else { 0.0 }).collect();
        let w_warm: Vec<f64> = {
            let ros: Vec<usize> = ros0.iter().copied().step_by((ros0.len() / 1500).max(1)).collect();
            let sw = sd.load_set(&ros, &LoadOpts { w: &w, cc: Some(&ccx.rows), keep: None, data: true }).unwrap();
            let mut op = hasty_recon::op::SenseOp::new(&dev, &st, kern.clone(), dims, [&sw.coords[0], &sw.coords[1], &sw.coords[2]], &smaps_all, nc, vec![], 1e-4).unwrap();
            let yd = st.clone_htod(&sw.y).unwrap();
            let mut xd = st.alloc_zeros::<C32>(nv).unwrap();
            op.adjoint(&st, &yd, &mut xd).unwrap();
            let mag = hasty_recon::segphys::smooth_magnitude(&st.clone_dtoh(&xd).unwrap(), dims, sd.vox, [8e-3, 8e-3, 2e-3]);
            let cmax = cen_all.iter().cloned().fold(0.0, f64::max);
            (0..nv).map(|q| if cen_all[q] > 1e-3 * cmax { (mag[q] as f64).powi(2) / cen_all[q].max(1e-30) } else { 0.0 }).collect()
        };
        trim(&dev, &st);
        let mk = |wsrc: &[f64], label: &str| -> Prob {
            let n = reg.len();
            let mut smaps = vec![C32::new(0.0, 0.0); nc * n];
            for c in 0..nc {
                for (i, &q) in reg.iter().enumerate() {
                    smaps[c * n + i] = smaps_all[c * nv + q];
                }
            }
            let wr: Vec<f64> = reg.iter().map(|&q| wsrc[q]).collect();
            let mx = wr.iter().cloned().fold(0.0, f64::max).max(1e-300);
            Prob { nc, smaps, cen: reg.iter().map(|&q| cen_all[q]).collect(), w: wr.iter().map(|v| v / mx).collect(), tau: tau.clone(), tw_adc: tw_adc.clone(), label: label.to_string() }
        };
        let greg = |i: usize, t: f64| gfull(reg[i], t);
        let mut json = serde_json::json!({"scan": dir.display().to_string(), "region": region, "weights": wsel, "time": tsel, "gnl": gnl, "nc": nc, "clip_hz": clip});
        let prob = mk(if wsel == "warm" { &w_warm } else { &w_pd }, &format!("{region}, {wsel} weights, {tsel} time weights"));
        println!("setup {:.1} s: grid {dims:?}, region {} voxels, {nc} virtual coils, {na} ADC samples τ ∈ [{:.3}, {:.3}] ms", t0.elapsed().as_secs_f64(), reg.len(), tau[0] * 1e3, tau[na - 1] * 1e3);
        let (cs, basis, _grid_t) = counts(&dev, &st, &prob, &greg, &tols, tsqr, &mut json);
        // point-image error percentiles at the counts (joint vs SVD segments; research/coil_segment_compression.md §3.4)
        {
            let fit: Vec<usize> = (0..prob.cen.len()).filter(|&i| prob.w[i] > 0.0).collect();
            let evj: Vec<usize> = fit.iter().copied().step_by((fit.len() / 6000).max(1)).collect();
            let ejw: Vec<f64> = evj.iter().map(|&i| prob.w[i]).collect();
            let tix: Vec<usize> = (0..na).step_by(16).collect();
            let tws: Vec<f64> = tix.iter().map(|&k| prob.tw_adc[k]).collect();
            let mut rows = vec![];
            for c in &cs {
                let e = basis.error(c.joint.0, &prob.smaps, &evj, &ejw, &tix, &tws, &greg);
                let mut r: Vec<(f64, f64)> = e.rho.iter().zip(&ejw).map(|(a, b)| (*a, *b)).collect();
                r.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let pct = |f: f64| r[((r.len() as f64 * f) as usize).min(r.len() - 1)].0;
                let tot: f64 = r.iter().map(|p| p.1).sum();
                let hi: f64 = r.iter().filter(|p| p.0 > 10.0 * c.tol).map(|p| p.1).sum::<f64>() / tot;
                println!("  [{}] ε = {:.0e}, joint R = {}: point-image error p50 {:.2e}, p99 {:.2e}, max {:.2e}; weight fraction with ρ > 10ε {:.2e}", prob.label, c.tol, c.joint.0, pct(0.5), pct(0.99), r.last().unwrap().0, hi);
                rows.push(serde_json::json!({"tol": c.tol, "r": c.joint.0, "p50": pct(0.5), "p99": pct(0.99), "max": r.last().unwrap().0, "frac_gt_10eps": hi}));
            }
            json["joint_rho"] = serde_json::json!(rows);
        }
        if do_rebuild {
            // (a) old basis on a ratemap update: +20 Hz smooth field, ±10 1/s decay change (smooth over the grid)
            let ext = [0, 1, 2].map(|ax| dims[ax] as f64 * sd.vox[ax]);
            let pert = |q: usize| -> (f64, f64) {
                let x = &grid.x_log[q];
                let s = (2.0 * std::f64::consts::PI * x[0] / ext[0]).sin() * (2.0 * std::f64::consts::PI * x[1] / ext[1]).cos() + 0.5 * (2.0 * std::f64::consts::PI * x[2] / ext[2]).sin();
                (2.0 * std::f64::consts::PI * 20.0 * s / 1.5, 10.0 * (2.0 * std::f64::consts::PI * x[1] / ext[1]).sin())
            };
            let gnew = |i: usize, t: f64| -> Complex64 {
                let q = reg[i];
                let (dw, dr) = pert(q);
                gfull(q, t) * Complex64::from_polar((-dr * (t - tau0)).exp(), -dw * t)
            };
            let fit: Vec<usize> = (0..prob.cen.len()).filter(|&i| prob.w[i] > 0.0).collect();
            let evj: Vec<usize> = fit.iter().copied().step_by((fit.len() / 6000).max(1)).collect();
            let ejw: Vec<f64> = evj.iter().map(|&i| prob.w[i]).collect();
            let tix: Vec<usize> = (0..na).step_by(16).collect();
            let tws: Vec<f64> = tix.iter().map(|&k| prob.tw_adc[k]).collect();
            let mut rows = vec![];
            for c in &cs {
                let e_old = basis.error(c.joint.0, &prob.smaps, &evj, &ejw, &tix, &tws, &gnew).global;
                println!("  [rebuild] ε = {:.0e}, R = {}: old basis on the updated ratemap (+20 Hz smooth, ±10 1/s): {:.2e} (was {:.2e})", c.tol, c.joint.0, e_old, c.joint.1);
                rows.push(serde_json::json!({"tol": c.tol, "r": c.joint.0, "err_old_basis_new_map": e_old, "err_own": c.joint.1}));
            }
            json["rebuild_ratemap"] = serde_json::json!(rows);
            // (b) weights: a basis built on the other weights, evaluated on PD energy and on uniform support weights
            let other = mk(if wsel == "warm" { &w_pd } else { &w_warm }, "other weights");
            let fit_o: Vec<usize> = (0..other.cen.len()).filter(|&i| other.w[i] > 0.0).collect();
            let (ts, tw) = seg_time_grid(&tau, &tw_adc, 288);
            let wj: Vec<f64> = fit_o.iter().map(|&i| other.w[i]).collect();
            let b2 = build_joint_basis(&dev, &st, other.cen.len(), nc, &other.smaps, &fit_o, &wj, &ts, &tw, &tau, &greg, &JointBuildOpts { rmax: 1400, ..Default::default() }).unwrap();
            let unif: Vec<f64> = evj.iter().map(|_| 1.0).collect();
            let mut rows = vec![];
            for c in &cs {
                let r2c = b2.count_for(c.tol);
                let e_on_this = b2.error(c.joint.0, &prob.smaps, &evj, &ejw, &tix, &tws, &greg).global;
                let e_unif = basis.error(c.joint.0, &prob.smaps, &evj, &unif, &tix, &tws, &greg).global;
                println!("  [weights] ε = {:.0e}: basis built on the {} weights at R = {}: {:.2e} on this problem's weights (own basis {:.2e}); this basis on uniform support weights {:.2e}; the other basis' own count {}", c.tol, if wsel == "warm" { "PD" } else { "warm-start" }, c.joint.0, e_on_this, c.joint.1, e_unif, r2c);
                rows.push(serde_json::json!({"tol": c.tol, "r": c.joint.0, "other_basis_on_these_weights": e_on_this, "this_basis_uniform": e_unif, "other_count": r2c}));
            }
            json["rebuild_weights"] = serde_json::json!(rows);
        }
        if do_op {
            super::opcheck::run(&dev, &st, &sd, &w, &ccx, &grid.x_log, &reg, &prob, &greg, &cs, &basis, region == "box", &mut json);
        }
        if let Some(o) = out {
            std::fs::write(&o, serde_json::to_string_pretty(&json).unwrap()).unwrap();
            println!("wrote {o}");
        }
    }

    /// synthetic head (state of the model: research note): ellipsoid object with brain / scalp, 32 circular loops on
    /// a helmet, B1⁻ of each loop = the quasi-static Biot–Savart field of a circular loop (magnitude ∝ transverse field,
    /// phase = loop azimuth + a 1/λ-like propagation phase); field: shimmed brain ±40 Hz, frontal-sinus and
    /// temporal-bone hot spots (−150 / +110 Hz Gaussians, 15–20 mm), no fat in the brain, scalp fat shell.
    pub fn cmd_head(mut a: Vec<String>) {
        let dims: Vec<usize> = arg(&mut a, "--dims").map_or(vec![96, 112, 96], |v| v.split(',').map(|x| x.parse().unwrap()).collect());
        let dims = [dims[0], dims[1], dims[2]];
        let res: f64 = arg(&mut a, "--res").map_or(2e-3, |v| v.parse().unwrap());
        let ncoil: usize = arg(&mut a, "--coils").map_or(32, |v| v.parse().unwrap());
        let tols: Vec<f64> = arg(&mut a, "--tols").map_or(vec![1e-2, 1e-3, 1e-4], |v| v.split(',').map(|x| x.parse().unwrap()).collect());
        let out = arg(&mut a, "--out");
        let nv: usize = dims.iter().product();
        let pos = |q: usize| -> [f64; 3] { [((q % dims[0]) as f64 - (dims[0] / 2) as f64) * res, (((q / dims[0]) % dims[1]) as f64 - (dims[1] / 2) as f64) * res, ((q / (dims[0] * dims[1])) as f64 - (dims[2] / 2) as f64) * res] };
        // head ellipsoid semi-axes (m): 75 (L-R) × 95 (A-P) × 85 (S-I); scalp shell 6 mm; brain inside
        let ax = [0.075, 0.095, 0.085];
        let rr = |p: &[f64; 3], s: f64| ((p[0] / (ax[0] - s)).powi(2) + (p[1] / (ax[1] - s)).powi(2) + (p[2] / (ax[2] - s)).powi(2)).sqrt();
        let pd: Vec<f64> = (0..nv).map(|q| { let p = pos(q); if rr(&p, 0.0) <= 1.0 { if rr(&p, 0.006) <= 1.0 { 0.8 } else { 1.0 } } else { 0.0 } }).collect();
        let ffv: Vec<f64> = (0..nv).map(|q| { let p = pos(q); if rr(&p, 0.0) <= 1.0 && rr(&p, 0.006) > 1.0 { 0.9 } else { 0.0 } }).collect();
        let hz: Vec<f64> = (0..nv).map(|q| {
            let p = pos(q);
            let g = |c: [f64; 3], s: f64| (-((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)) / (2.0 * s * s)).exp();
            40.0 * (p[0] / ax[0]) * (p[1] / ax[1]) + 25.0 * (p[2] / ax[2]).powi(2) - 150.0 * g([0.0, 0.07, -0.01], 0.015) + 110.0 * g([0.055, -0.01, -0.03], 0.02) + 110.0 * g([-0.055, -0.01, -0.03], 0.02)
        }).collect();
        let r2: Vec<f64> = (0..nv).map(|q| if pd[q] > 0.0 { 25.0 } else { 0.0 }).collect();
        // helmet: loops of radius 40 mm on an ellipsoid 25 mm outside the head, upper hemisphere + a band
        let mut loops: Vec<([f64; 3], [f64; 3])> = vec![];
        let ga = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        for i in 0..ncoil {
            let zc = 1.0 - 1.2 * (i as f64 + 0.5) / ncoil as f64;
            let rc = (1.0 - zc * zc).max(0.0).sqrt();
            let ph = ga * i as f64;
            let n = [rc * ph.cos(), rc * ph.sin(), zc];
            let c = [(ax[0] + 0.025) * n[0], (ax[1] + 0.025) * n[1], (ax[2] + 0.025) * n[2]];
            loops.push((c, n));
        }
        let a_loop = 0.04;
        let b1 = |c: usize, p: &[f64; 3]| -> Complex64 {
            // on-axis-like approximation of a loop's transverse field: magnitude a² / (a² + d²)^{3/2} with d the
            // distance to the loop centre, weighted by the transverse fraction of the loop normal; phase = azimuth of
            // the loop + a propagation phase 2π d / 0.26 m (3 T, tissue)
            let (lc, n) = loops[c];
            let d = [p[0] - lc[0], p[1] - lc[1], p[2] - lc[2]];
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            let ax_d = (d[0] * n[0] + d[1] * n[1] + d[2] * n[2]).abs() / d2.sqrt().max(1e-9);
            let mag = a_loop * a_loop / (a_loop * a_loop + d2).powf(1.5) * (0.4 + 0.6 * ax_d) * 1e-2;
            Complex64::from_polar(mag, n[1].atan2(n[0]) + 2.0 * std::f64::consts::PI * d2.sqrt() / 0.26)
        };
        let support: Vec<usize> = (0..nv).filter(|&q| pd[q] > 0.0).collect();
        let n = support.len();
        let mut smaps = vec![C32::new(0.0, 0.0); ncoil * n];
        for c in 0..ncoil {
            for (i, &q) in support.iter().enumerate() {
                let z = b1(c, &pos(q));
                smaps[c * n + i] = C32::new(z.re as f32, z.im as f32);
            }
        }
        // whiten-free SV compression of the maps (Gram over the support, ≥ 99.9 %)
        let mut gr = hasty_recon::coils::CoilGram::new(ncoil);
        gr.add(&smaps, n, 1);
        let ccx = hasty_recon::coils::CoilCompression::from_gram(&gr, hasty_recon::coils::CoilRule::Energy(0.999));
        println!("synthetic head {dims:?} at {res} m: {n} object voxels, {ncoil} loops → {}", ccx.report());
        let nc = ccx.nvc();
        let sm = ccx.apply(&smaps, n);
        let cen: Vec<f64> = (0..n).map(|i| (0..nc).map(|c| { let z = sm[c * n + i]; (z.re as f64).powi(2) + (z.im as f64).powi(2) }).sum()).collect();
        let na = 3990;
        let tau: Vec<f64> = (0..na).map(|k| 1.6e-3 + k as f64 * 2e-6).collect();
        let fs = hasty_qfit::waterfat::FatSpectrum::six_peak();
        let om_f: Vec<f64> = fs.ppm.iter().map(|d| 2.0 * std::f64::consts::PI * 42.577_478_51e6 * 3.0 * d * 1e-6).collect();
        let g = |i: usize, t: f64| -> Complex64 {
            let q = support[i];
            let c: Complex64 = om_f.iter().zip(&fs.amp).map(|(w, a)| Complex64::from_polar(*a, -w * t)).sum();
            Complex64::from_polar((-r2[q] * (t - 1.6e-3)).exp(), -2.0 * std::f64::consts::PI * hz[q] * t) * (Complex64::new(1.0 - ffv[q], 0.0) + ffv[q] * c)
        };
        let prob = Prob { nc, smaps: sm, cen, w: support.iter().map(|&q| pd[q] * pd[q]).collect(), tau, tw_adc: vec![1.0; na], label: format!("synthetic head {dims:?}") };
        let dev = Device::new(0).unwrap();
        let st = dev.default_stream();
        let mut json = serde_json::json!({"model": "synthetic head", "dims": dims, "res": res, "coils": ncoil, "nc": nc});
        let _ = counts(&dev, &st, &prob, &g, &tols, false, &mut json);
        if let Some(o) = out {
            std::fs::write(&o, serde_json::to_string_pretty(&json).unwrap()).unwrap();
            println!("wrote {o}");
        }
    }
}

mod opcheck {
    use super::counts::{Counts, Prob};
    use super::*;
    use hasty_recon::data::SimData;
    use hasty_recon::kload::LoadOpts;

    /// The operators at the real samples of 48 readouts of encode 0 on the region's grid (box: 80³ at 2 mm), image =
    /// PD (real, positive: the most coherent at k = 0): ‖A_basis x − A_ref x‖ / ‖A_ref x‖ overall, in the k-space
    /// centre (|k| < 70 rad/m) and the outer readout (|k| > 600 rad/m); reference = rank-40 SVD segments.
    #[allow(clippy::too_many_arguments)]
    pub fn run(dev: &Device, st: &Arc<CudaStream>, sd: &SimData, w: &[Vec<f64>], ccx: &hasty_recon::coils::CoilCompression, _xl: &[hasty_mrsim::Vec3], reg: &[usize], prob: &Prob, g: &(dyn Fn(usize, f64) -> Complex64 + Sync), cs: &[Counts], basis: &hasty_nufft::JointBasis, is_box: bool, json: &mut serde_json::Value) {
        assert!(is_box, "--op: the box region (an 80³ grid)");
        let dims = [80usize, 80, 80];
        let nv: usize = 6400 * 80;
        assert_eq!(reg.len(), nv);
        let ros0 = sd.readouts_of(0, None);
        let ros: Vec<usize> = ros0.iter().copied().step_by(ros0.len() / 48).take(48).collect();
        let set = sd.load_set(&ros, &LoadOpts { w, cc: Some(&ccx.rows), keep: None, data: false }).unwrap();
        let m = set.tau.len();
        let na = m / ros.len();
        let tau32: Vec<f32> = set.tau.iter().map(|&t| t as f32).collect();
        let kap: Vec<f64> = (0..m).map(|j| (0..3).map(|a| (set.coords[a][j] as f64 / sd.vox[a]).powi(2)).sum::<f64>().sqrt()).collect();
        let x: Vec<C32> = (0..nv).map(|i| C32::new(prob.w[i].sqrt() as f32, 0.0)).collect();
        let xd = st.clone_htod(&x).unwrap();
        let (t0, t1) = (prob.tau[0], prob.tau[na - 1]);
        let fit: Vec<usize> = (0..nv).filter(|&i| prob.w[i] > 0.0).collect();
        let wseg: Vec<f64> = fit.iter().map(|&i| prob.w[i] * prob.cen[i]).collect();
        let (ts, tw) = hasty_nufft::seg_time_grid(&prob.tau, &prob.tw_adc, 288);
        let seg_op = |l: usize, svd: bool| -> Vec<C32> {
            let mut e = SegNufft::new(dev, dims, [sd.vox[0], sd.vox[1], sd.vox[2]], prob.nc, SegNufftOpts { sigma: 1.25, width: 10, nseg: l, order: 0, frames: 1, bspline: 1, max_grid_bytes: 4 << 30 }).unwrap();
            e.set_maps(st, &prob.smaps).unwrap();
            let (sp, at, a0, dt) = if svd { hasty_nufft::build_segments_svd_grid(nv, l, t0, t1, na, &fit, Some(&wseg), &ts, &tw, g) } else { hasty_nufft::build_segments_complex(nv, l, t0, t1, na, &fit.iter().copied().step_by(13).collect::<Vec<_>>(), g) };
            let sg = e.segments(st, &sp, &at, a0, dt).unwrap();
            let p = e.points(st, [&set.coords[0], &set.coords[1], &set.coords[2]], &tau32, &vec![0f32; m], &vec![0f32; 6 * m], None).unwrap();
            let mut y = st.alloc_zeros::<C32>(prob.nc * m).unwrap();
            e.forward(st, &sg, &p, &xd, &mut y).unwrap();
            st.clone_dtoh(&y).unwrap()
        };
        let yref = seg_op(40, true);
        let rel = |y: &[C32], sel: &dyn Fn(usize) -> bool| -> f64 {
            let (mut n, mut d) = (0.0f64, 0.0f64);
            for c in 0..prob.nc {
                for j in (0..m).filter(|&j| sel(j)) {
                    let (a, b) = (y[c * m + j], yref[c * m + j]);
                    n += ((a.re - b.re) as f64).powi(2) + ((a.im - b.im) as f64).powi(2);
                    d += (b.re as f64).powi(2) + (b.im as f64).powi(2);
                }
            }
            (n / d.max(1e-300)).sqrt()
        };
        let mut rows = vec![];
        for c in cs {
            let ys = seg_op(c.svd.0, true);
            let yf = seg_op(c.fixed.0, false);
            // joint
            let cbj = JointSegNufft::batch_for(c.joint.0, 64);
            hasty_nufft::set_thread_cb_cap(Some(cbj));
            let inner = SegNufft::new(dev, dims, [sd.vox[0], sd.vox[1], sd.vox[2]], cbj, SegNufftOpts { sigma: 1.25, width: 10, nseg: 1, order: 0, frames: 1, bspline: 1, max_grid_bytes: 8 << 30 }).unwrap();
            hasty_nufft::set_thread_cb_cap(None);
            let bcopy = hasty_nufft::JointBasis { nc: basis.nc, nv: basis.nv, k: basis.k, lam: basis.lam.clone(), u: basis.u.clone(), ncol: basis.ncol, ts: basis.ts.clone(), stw: basis.stw.clone(), v: basis.v.clone(), time_ev: basis.time_ev.clone(), tab_times: basis.tab_times.clone(), vt: basis.vt.clone(), h: st.clone_dtod(&basis.h).unwrap(), timing: vec![] };
            let mut op = JointSegNufft::new(dev, st, inner, bcopy, &prob.smaps, c.joint.0).unwrap();
            let s1 = op.segment1(st, None).unwrap();
            let pj = op.inner.points(st, [&set.coords[0], &set.coords[1], &set.coords[2]], &tau32, &vec![0f32; m], &vec![0f32; 6 * m], None).unwrap();
            let mut y = st.alloc_zeros::<C32>(prob.nc * m).unwrap();
            op.forward(st, &s1, &pj, &xd, &mut y).unwrap();
            let yj = st.clone_dtoh(&y).unwrap();
            drop(op);
            let cen = |j: usize| kap[j] < 70.0;
            let out = |j: usize| kap[j] > 600.0;
            let all = |_: usize| true;
            println!("  [op] ε = {:.0e}: ‖Δy‖/‖y‖ vs rank-40 SVD at the real samples (48 readouts), all / centre / outer: fixed L {} {:.2e} / {:.2e} / {:.2e}; SVD L {} {:.2e} / {:.2e} / {:.2e}; joint R {} {:.2e} / {:.2e} / {:.2e}", c.tol, c.fixed.0, rel(&yf, &all), rel(&yf, &cen), rel(&yf, &out), c.svd.0, rel(&ys, &all), rel(&ys, &cen), rel(&ys, &out), c.joint.0, rel(&yj, &all), rel(&yj, &cen), rel(&yj, &out));
            rows.push(serde_json::json!({"tol": c.tol, "fixed": [c.fixed.0, rel(&yf, &all), rel(&yf, &cen), rel(&yf, &out)], "svd": [c.svd.0, rel(&ys, &all), rel(&ys, &cen), rel(&ys, &out)], "joint": [c.joint.0, rel(&yj, &all), rel(&yj, &cen), rel(&yj, &out)]}));
            trim(dev, st);
        }
        json["opcheck"] = serde_json::json!(rows);
    }
}

/// `coilbox <scan> --presc-box S[@C] [--margin 10] [--energy 0.999] [--ro-stride 4] [--iters 15] [--out f.json]`:
/// box-restricted coil Gram (`--coil-gram box`, research/coil_cost_reduction.md §7.1) against the global calibration
/// Gram on ph0001 encode 0: the virtual-coil counts, the box / whole-FOV energy kept, and a quality check — the same
/// static SENSE CGLS (plain model, circulant preconditioner, every `ro-stride`-th readout) with each compression,
/// compared with the 32-coil (uncompressed, whitened) reconstruction inside the box and outside, plus the plain SENSE
/// operator time per forward + adjoint.
mod coilbox {
    use super::*;
    use hasty_recon::coils::{masked_gram, masked_retention, CoilCompression, CoilGram, CoilRule, PrescBox};
    use hasty_recon::data::SimData;
    use hasty_recon::kload::LoadOpts;

    pub fn run(mut a: Vec<String>) {
        let dir = std::path::PathBuf::from(a.remove(0));
        let pb = PrescBox::parse(&arg(&mut a, "--presc-box").unwrap_or_else(|| "160".into())).unwrap();
        let margin: f64 = arg(&mut a, "--margin").map_or(10.0, |v| v.parse().unwrap()) * 1e-3;
        let energies: Vec<f64> = arg(&mut a, "--energy").map_or(vec![0.999, 0.9995], |v| v.split(',').map(|x| x.parse().unwrap()).collect());
        let ro_stride: usize = arg(&mut a, "--ro-stride").map_or(4, |v| v.parse().unwrap());
        let iters: usize = arg(&mut a, "--iters").map_or(15, |v| v.parse().unwrap());
        let out = arg(&mut a, "--out");
        let sd = SimData::load_lazy(&dir).unwrap();
        let w = sd.whitener();
        let grid = sd.physics_grid(true, &w);
        let (dims, nv, nc) = (sd.dims, sd.nv(), sd.nc);
        let ros: Vec<usize> = sd.readouts_of(0, None).into_iter().step_by(ro_stride).collect();
        let set = sd.load_set(&ros, &LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
        let m = set.tau.len();
        let dev = Device::new(0).unwrap();
        let st = dev.default_stream();
        let kern = Arc::new(hasty_recon::gpu::Kernels::new(&dev).unwrap());
        let cal_dims = dims.map(|d| 24.min(d) & !1);
        let (cal, used) = hasty_recon::espirit::calib_from_noncart(&dev, &st, &kern, [&set.coords[0], &set.coords[1], &set.coords[2]], &set.y, nc, dims, cal_dims, 30, 1e-3).unwrap();
        let ncal: usize = cal_dims.iter().product();
        let fov = [0, 1, 2].map(|ax| dims[ax] as f64 * sd.vox[ax]);
        let mk = pb.mask(cal_dims, fov, margin);
        println!("coilbox: {m} samples ({} readouts of encode 0), calibration {cal_dims:?} from {used} samples; box {:?} mm at {:?} mm + {:.0} mm: {} of {ncal} calibration voxels", ros.len(), pb.size.map(|v| v * 1e3), pb.center.map(|v| v * 1e3), margin * 1e3, mk.iter().filter(|&&v| v).count());
        let mut g_all = CoilGram::new(nc);
        g_all.add(&cal, ncal, 1);
        let g_box = masked_gram(&cal, ncal, nc, &mk);
        let box_img = pb.mask(dims, fov, 0.0);
        // the reference: all coils (identity), CGLS
        let recon = |cc: &CoilCompression| -> (Vec<C32>, f64) {
            let nvc = cc.nvc();
            let sm = cc.apply(&grid.smaps, nv);
            let y = cc.apply(&set.y, m);
            let mut op = hasty_recon::op::SenseOp::new(&dev, &st, kern.clone(), dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &sm, nvc, vec![], 1e-4).unwrap();
            let mut pre = hasty_recon::precond::CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &sm, nvc, 1e-3).unwrap();
            let yd = st.clone_htod(&y).unwrap();
            let mut x = st.alloc_zeros::<C32>(nv).unwrap();
            let mut yy = st.alloc_zeros::<C32>(nvc * m).unwrap();
            let t = median_ms(&st, 2, &mut || { op.forward(&st, &x, &mut yy).unwrap(); op.adjoint(&st, &yy, &mut x).unwrap(); });
            st.memset_zeros(&mut x).unwrap();
            hasty_recon::cg::pcgls_cb(&mut op, Some(&mut pre), &st, &yd, &mut x, 0.0, iters, &mut |_, _| {}).unwrap();
            (st.clone_dtoh(&x).unwrap(), t)
        };
        let (xr, tr) = recon(&CoilCompression::identity(nc));
        let dist = |x: &[C32], sel: &dyn Fn(usize) -> bool| -> f64 {
            let (mut n, mut d) = (0.0f64, 0.0f64);
            for q in (0..nv).filter(|&q| sel(q)) {
                n += ((x[q].re - xr[q].re) as f64).powi(2) + ((x[q].im - xr[q].im) as f64).powi(2);
                d += (xr[q].re as f64).powi(2) + (xr[q].im as f64).powi(2);
            }
            (n / d.max(1e-300)).sqrt()
        };
        println!("  reference: {nc} whitened coils, {iters} PCGLS iterations; plain SENSE forward + adjoint {tr:.0} ms");
        let mut rows = vec![];
        for &e in &energies {
            let cg = CoilCompression::from_gram(&g_all, CoilRule::Energy(e));
            let cbx = CoilCompression::from_gram(&g_box, CoilRule::Energy(e));
            let cgt = CoilCompression::from_gram(&g_all, CoilRule::Fixed(cbx.nvc()));
            for (label, cc) in [("global", &cg), ("box", &cbx), ("global, box count", &cgt)] {
                let (kin, p50, p99, mx) = masked_retention(cc, &cal, ncal, Some(&mk));
                let (kall, ..) = masked_retention(cc, &cal, ncal, None);
                let (x, t) = recon(cc);
                let (din, dout) = (dist(&x, &|q| box_img[q]), dist(&x, &|q| !box_img[q]));
                println!("  e = {e}: {label:>18}: {:>2} coils; box energy {:.3} % (voxel loss p50 {p50:.1e}, p99 {p99:.1e}, max {mx:.1e}), whole region {:.3} %; recon vs {nc}-coil: in box {din:.4}, outside {dout:.4}; SENSE fwd + adj {t:.0} ms ({:.2}× the reference)", cc.nvc(), 100.0 * kin, 100.0 * kall, tr / t);
                rows.push(serde_json::json!({"energy": e, "gram": label, "coils": cc.nvc(), "kept_box": kin, "loss_p50": p50, "loss_p99": p99, "loss_max": mx, "kept_all": kall, "dist_in_box": din, "dist_outside": dout, "sense_ms": t}));
            }
        }
        if let Some(o) = out {
            std::fs::write(&o, serde_json::to_string_pretty(&serde_json::json!({"box_mm": pb.size.map(|v| v * 1e3), "center_mm": pb.center.map(|v| v * 1e3), "margin_mm": margin * 1e3, "ro_stride": ro_stride, "iters": iters, "ref_coils": nc, "ref_sense_ms": tr, "rows": rows})).unwrap()).unwrap();
        }
    }
}
