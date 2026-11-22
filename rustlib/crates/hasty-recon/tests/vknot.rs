//! GPU: the at-knots velocity model (`VelSpec::at_knots`) through the real 2-D knot operator (`SegNufft::new_temporal2d`,
//! respiratory linear × cardiac cubic knots, one set per encode holding all knots):
//! - the objective's gradient against central finite differences (f32 pipeline);
//! - the host-resident, knot-streamed gradient / Hessian-vector product / λ_max equal the device ones;
//! - model-consistent data: P²nP (ZMSHR1 in the Gauss–Newton metric) recovers z and v from a perturbed start, with the
//!   state on the device and on the host (same iterates); with a degree-1 residual phase and the static-tissue anchor
//!   (static voxels of the synthetic truth) it also recovers the residual phase.
//!   cargo test --release -p hasty-recon --test vknot -- --nocapture
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};
use hasty_recon::gpu::Kernels;
use hasty_recon::hoststack::{run_objective_host, HostStack};
use hasty_recon::p2np::{rdot, run_objective, Objective, Precond, Stack};
use hasty_recon::setop::{SegSets, SetOp};
use hasty_recon::vmodel::cpu::Theta;
use hasty_recon::vmodel::{VelModel, VelSpec};
use hasty_recon::vobj::VelProblem;
use num_complex::Complex64 as C;

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        2.0 * self.u() - 1.0
    }
}

const DIMS: [usize; 3] = [12, 10, 8];
const KR: usize = 2;
const MC: usize = 4;

fn kenc() -> Vec<[f64; 3]> {
    // dual-VENC-like encodes (VENC 1.0 / 2.5 m/s): k = π / VENC along each axis
    let (lo, hi) = (std::f64::consts::PI / 1.0, std::f64::consts::PI / 2.5);
    let mut k = vec![[0.0; 3]];
    for a in 0..3 {
        for v in [lo, hi] {
            let mut e = [0.0; 3];
            e[a] = v;
            k.push(e);
        }
    }
    k
}

/// smooth magnitude, pulsatile velocity inside a "vessel" (static outside), small residual phase
fn truth(s: &VelSpec) -> (Theta, Vec<f32>) {
    let mut t = Theta::zeros(s);
    let [nx, ny, nz] = s.dims;
    let mut stat = vec![0f32; s.nv()];
    for r in 0..s.nv() {
        let (i, j, k) = ((r % nx) as f64, ((r / nx) % ny) as f64, (r / (nx * ny)) as f64);
        let m = (-((i - 6.0).powi(2) / 20.0 + (j - 5.0).powi(2) / 14.0 + (k - 4.0).powi(2) / 8.0)).exp() + 0.3;
        let vessel = (i - 6.0).powi(2) + (j - 5.0).powi(2) < 6.5;
        stat[r] = (!vessel) as u8 as f32;
        for kk in 0..s.kz {
            let (rr, c) = (kk / MC, kk % MC);
            t.z[kk][r] = C::from_polar(m * (1.0 + 0.1 * rr as f64 + 0.05 * c as f64), 0.3 * (i / nx as f64));
            if vessel {
                let ph = 2.0 * std::f64::consts::PI * c as f64 / MC as f64;
                t.v[kk][r] = [0.15 * ph.cos(), 0.1 * ph.sin() * (1.0 + 0.2 * rr as f64), 0.5 + 0.35 * ph.sin() + 0.05 * (k / nz as f64)];
            }
        }
    }
    for (i, a) in t.a.iter_mut().enumerate() {
        *a = 0.04 * ((i % 4) as f64 - 1.5);
    }
    (t, stat)
}

struct Problem {
    sets: SegSets,
    ys: Vec<CudaSlice<C32>>,
}

fn build(dev: &Device, kern: &Arc<Kernels>, s: &VelSpec, tr: &Theta, seed: u64) -> Problem {
    let st = dev.default_stream();
    let nv = s.nv();
    let (nc, nseg, m) = (3usize, 2usize, 6000usize);
    let mut rng = Rng(seed);
    let smaps: Vec<C32> = (0..nc * nv).map(|i| { let c = i / nv; let r = i % nv; let ph = 0.5 * c as f64 + 0.003 * r as f64; C32::new((0.7 * ph.cos()) as f32, (0.7 * ph.sin()) as f32) }).collect();
    let fit: Vec<usize> = (0..nv).step_by(3).collect();
    let psi = |v: usize, tt: f64| 300.0 * ((v % 17) as f64 / 17.0 - 0.5) * tt;
    let (sp, at, t0, dt) = build_segments(nv, nseg, 2e-3, 8e-3, 64, &fit, &psi);
    let opts = SegNufftOpts { sigma: 1.25, width: 6, nseg, order: 0, frames: MC, bspline: 3, max_grid_bytes: 1 << 28 };
    let mut eng = SegNufft::new_temporal2d(dev, DIMS, [0.004; 3], nc, opts, KR, 1).unwrap();
    eng.set_maps(&st, &smaps).unwrap();
    let seg = eng.segments(&st, &sp, &at, t0, dt).unwrap();
    let ne = s.ne();
    let pts: Vec<_> = (0..ne)
        .map(|_| {
            let c: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (std::f64::consts::PI * 0.999 * rng.n()) as f32).collect());
            let tau: Vec<f32> = (0..m).map(|_| (2e-3 + 6e-3 * rng.u()) as f32).collect();
            let phi0 = vec![0f32; m];
            let t: Vec<f32> = (0..m).map(|_| (MC as f64 * rng.u()) as f32).collect();
            let ri = vec![0u32; m];
            let rv: Vec<f32> = (0..m).map(|_| rng.u() as f32).collect();
            eng.points2d(&st, [&c[0], &c[1], &c[2]], &tau, &phi0, &[], &t, &ri, &rv).unwrap()
        })
        .collect();
    let mut sets = SegSets { eng, pts, segs: vec![seg], seg_of_set: vec![0; ne], kern: kern.clone(), surrogate: None };
    // model-consistent data y_e = A_e [x_{e,k}(θ_true)]
    let model = VelModel::new(dev, &st, s.clone()).unwrap();
    let tg = Stack { v: tr.pack(s).iter().map(|c| st.clone_htod(c).unwrap()).collect() };
    let mut ys = vec![];
    let mut xi = st.alloc_zeros::<C32>(nv).unwrap();
    for e in 0..ne {
        let mut xe = st.alloc_zeros::<C32>(s.kz * nv).unwrap();
        for k in 0..s.kz {
            model.eval(&st, &tg, e, k, &mut xi).unwrap();
            st.memcpy_dtod(&xi, &mut xe.slice_mut(k * nv..(k + 1) * nv)).unwrap();
        }
        let mut y = st.alloc_zeros::<C32>(sets.len_y(e)).unwrap();
        sets.forward(&st, e, &xe, &mut y).unwrap();
        ys.push(y);
    }
    Problem { sets, ys }
}

fn host_of(st: &Arc<cudarc::driver::CudaStream>, x: &Stack) -> HostStack {
    let lens: Vec<usize> = x.v.iter().map(|c| c.len()).collect();
    let mut h = HostStack::zeros(&lens);
    for t in 0..lens.len() {
        st.memcpy_dtoh(&x.v[t], h.chunk_mut(t)).unwrap();
    }
    h
}

fn vrms(s: &VelSpec, a: &Theta, b: &Theta, mask: Option<&[f32]>) -> f64 {
    let (mut e, mut n) = (0.0, 0usize);
    for k in 0..s.kv {
        for r in 0..s.nv() {
            if mask.is_some_and(|m| m[r] > 0.0) {
                continue;
            }
            for c in 0..3 {
                e += (a.v[k][r][c] - b.v[k][r][c]).powi(2);
            }
            n += 3;
        }
    }
    (e / n.max(1) as f64).sqrt()
}

fn rel_host(a: &HostStack, b: &HostStack) -> f64 {
    let d: f64 = a.data.iter().zip(&b.data).map(|(p, q)| ((p.re - q.re) as f64).powi(2) + ((p.im - q.im) as f64).powi(2)).sum();
    (d / HostStack::rdot(b, b).max(1e-300)).sqrt()
}

#[test]
fn knot_velocity_model_gradient_streaming_and_recovery() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let s = VelSpec::at_knots(DIMS, kenc(), KR, MC, None);
    let nv = s.nv();
    let (tr, stat) = truth(&s);
    let mut pb = build(&dev, &kern, &s, &tr, 0x51);
    let model = VelModel::new(&dev, &st, s.clone()).unwrap();
    let up = |t: &Theta| Stack { v: t.pack(&s).iter().map(|c| st.clone_htod(c).unwrap()).collect() };
    let ys = std::mem::take(&mut pb.ys);
    let mut pr = VelProblem::new(&st, kern.clone(), &mut pb.sets, ys, model, None).unwrap();
    // per-knot curvature weights (here: uneven on purpose; any positive weights keep the metric valid)
    pr.set_knot_weights(&(0..s.kz).map(|k| 0.6 + 0.1 * k as f64).collect::<Vec<_>>());
    let mut rng = Rng(0x99);
    let mut start = tr.clone();
    for v in start.v.iter_mut() {
        for x in v.iter_mut() {
            for c in 0..3 {
                x[c] += 0.12 * rng.n();
            }
        }
    }
    for z in start.z.iter_mut() {
        for x in z.iter_mut() {
            *x *= C::new(1.0 + 0.05 * rng.n(), 0.05 * rng.n());
        }
    }
    let lens = s.lens();
    // 1. gradient vs central differences
    let th = up(&start);
    let mut g = Stack::zeros_lens(&st, &lens).unwrap();
    let f0 = pr.gradient(&st, &th, &mut g).unwrap();
    let mut d = Theta::zeros(&s);
    for k in 0..s.kv {
        for r in 0..nv {
            d.v[k][r] = [rng.n(), rng.n(), rng.n()];
            d.z[k][r] = C::new(rng.n(), rng.n()) * 0.1;
        }
    }
    let h = 2e-3;
    let fp = pr.gradient(&st, &up(&start.add(h, &d)), &mut Stack::zeros_lens(&st, &lens).unwrap()).unwrap();
    let fm = pr.gradient(&st, &up(&start.add(-h, &d)), &mut Stack::zeros_lens(&st, &lens).unwrap()).unwrap();
    let fd = (fp - fm) / (2.0 * h);
    let gh = Theta::unpack(&s, &g.v.iter().map(|c| st.clone_dtoh(c).unwrap()).collect::<Vec<_>>());
    let an = gh.dot(&d);
    eprintln!("at-knots objective: f {f0:.4e}, directional derivative fd {fd:.6e} vs analytic {an:.6e} (rel {:.1e})", (fd - an).abs() / an.abs());
    assert!((fd - an).abs() < 2e-3 * an.abs(), "objective gradient: fd {fd:.6e} vs analytic {an:.6e}");
    // 2. host-resident streamed gradient = device gradient
    let thh = host_of(&st, &th);
    let mut gh2 = HostStack::zeros(&lens);
    let f1 = pr.gradient_host(&st, &thh, &mut gh2).unwrap();
    let e_g = rel_host(&gh2, &host_of(&st, &g));
    eprintln!("streamed gradient vs device: f rel {:.1e}, g rel {e_g:.1e}", (f1 / f0 - 1.0).abs());
    assert!((f1 / f0 - 1.0).abs() < 1e-5 && e_g < 1e-5, "streamed gradient differs: f {f1:e} vs {f0:e}, g rel {e_g:e}");
    // 2b. coil sampling path (p2np::CoilSampling groups): the groups {coil 0} ∪ {coils 1, 2} with weights 1 are the full
    // data term (device and streamed); a tail weight w scales the tail's part exactly
    pr.cur_groups = Some(vec![(vec![0], 1.0), (vec![1, 2], 1.0)]);
    let mut gs = Stack::zeros_lens(&st, &lens).unwrap();
    let fs = pr.gradient(&st, &th, &mut gs).unwrap();
    let mut gsh = HostStack::zeros(&lens);
    let fsh = pr.gradient_host(&st, &thh, &mut gsh).unwrap();
    let (e_s, e_sh) = (rel_host(&host_of(&st, &gs), &host_of(&st, &g)), rel_host(&gsh, &host_of(&st, &g)));
    eprintln!("coil groups (all coils, weights 1) vs full: f rel {:.1e} / {:.1e}, g rel {e_s:.1e} / {e_sh:.1e}", (fs / f0 - 1.0).abs(), (fsh / f0 - 1.0).abs());
    assert!((fs / f0 - 1.0).abs() < 1e-5 && (fsh / f0 - 1.0).abs() < 1e-5 && e_s < 1e-5 && e_sh < 1e-5, "coil-group gradient differs from the full one");
    pr.cur_groups = Some(vec![(vec![1, 2], 2.0)]);
    let mut g12 = Stack::zeros_lens(&st, &lens).unwrap();
    let f12 = pr.gradient(&st, &th, &mut g12).unwrap();
    pr.cur_groups = Some(vec![(vec![0], 1.0)]);
    let mut g0 = Stack::zeros_lens(&st, &lens).unwrap();
    let f0c = pr.gradient(&st, &th, &mut g0).unwrap();
    pr.cur_groups = None;
    // f = f_0 + f_12, so f12(w = 2) = 2 (f − f_0)
    assert!((f12 / (2.0 * (f0 - f0c)) - 1.0).abs() < 1e-4, "tail weight: f {f12:e} vs {:e}", 2.0 * (f0 - f0c));
    // 3. Hessian-vector products and λ_max, device vs streamed
    let dd = up(&d);
    let mut hd = Stack::zeros_lens(&st, &lens).unwrap();
    pr.hvp(&st, &th, &dd, &mut hd).unwrap();
    let mut hh = HostStack::zeros(&lens);
    pr.hvp_host(&st, &thh, &host_of(&st, &dd), &mut hh).unwrap();
    let e_h = rel_host(&hh, &host_of(&st, &hd));
    let sym = { let w = up(&start); let mut hw = Stack::zeros_lens(&st, &lens).unwrap(); pr.hvp(&st, &th, &w, &mut hw).unwrap(); (rdot(&kern, &st, &w, &hd).unwrap(), rdot(&kern, &st, &dd, &hw).unwrap()) };
    eprintln!("Hessian-vector product: streamed vs device rel {e_h:.1e}; symmetry {:.3e} vs {:.3e}", sym.0, sym.1);
    assert!(e_h < 1e-5, "streamed hvp differs: {e_h:e}");
    assert!((sym.0 - sym.1).abs() < 1e-4 * sym.0.abs().max(sym.1.abs()), "Hessian not symmetric");
    let lam = pr.lambda_max(&st, &th, true, 15).unwrap();
    let lam_h = pr.lambda_max_host(&st, &thh, true, 15).unwrap();
    eprintln!("λ_max(MH): device {lam:.5e}, host {lam_h:.5e}");
    assert!((lam_h / lam - 1.0).abs() < 1e-3, "λ_max differs");
    // 4. recovery from model-consistent data, state on the device and on the host
    let e0 = vrms(&s, &start, &tr, None);
    let iters = 250;
    let mut xd = up(&start);
    pr.reset();
    let mut fs_d = vec![];
    run_objective(&mut pr, &st, &mut xd, Precond::Zmshr1M, 1.0 / lam, iters, &mut |_, _| {}, &mut |_, _, f| fs_d.push(f), None).unwrap();
    let xe = Theta::unpack(&s, &xd.v.iter().map(|c| st.clone_dtoh(c).unwrap()).collect::<Vec<_>>());
    let e_dev = vrms(&s, &xe, &tr, None);
    pr.reset();
    let mut xh = host_of(&st, &up(&start));
    let mut fs_h = vec![];
    run_objective_host(&mut pr, &st, &mut xh, Precond::Zmshr1M, 1.0 / lam, iters, &mut |_, _| {}, &mut |_, _, f| fs_h.push(f), None).unwrap();
    let xhe = Theta::unpack(&s, &(0..lens.len()).map(|t| xh.chunk(t).to_vec()).collect::<Vec<_>>());
    let e_host = vrms(&s, &xhe, &tr, None);
    let zerr = |a: &Theta| { let (mut e, mut n) = (0.0, 0.0); for k in 0..s.kz { for r in 0..nv { e += (a.z[k][r] - tr.z[k][r]).norm_sqr(); n += tr.z[k][r].norm_sqr(); } } (e / n).sqrt() };
    eprintln!("recovery: velocity RMS error {e0:.4} → device {e_dev:.2e} / host {e_host:.2e} m/s; z rel error {:.2e} / {:.2e}; f {:.3e} → {:.3e} (host {:.3e}); f at it 10 device {:.6e} host {:.6e}", zerr(&xe), zerr(&xhe), fs_d[1], fs_d.last().unwrap(), fs_h.last().unwrap(), fs_d[10], fs_h[10]);
    assert!(e_dev < 0.02 * e0 && e_host < 0.02 * e0, "velocity not recovered: {e0:.4} → {e_dev:.4} / {e_host:.4}");
    assert!((fs_h[10] / fs_d[10] - 1.0).abs() < 1e-3, "host and device iterates differ");
    drop(pr);
    // 5. with a degree-1 residual phase: the anchor on the static voxels breaks its degeneracy with the time-mean
    // velocity (research/velocity_model_sr.md §A3)
    let sp = VelSpec::at_knots(DIMS, kenc(), KR, MC, Some(1));
    let (trp, _) = truth(&sp);
    let mut pbp = build(&dev, &kern, &sp, &trp, 0x52);
    let model = VelModel::new(&dev, &st, sp.clone()).unwrap();
    let ys = std::mem::take(&mut pbp.ys);
    let mut pr = VelProblem::new(&st, kern.clone(), &mut pbp.sets, ys, model, None).unwrap();
    pr.set_anchor(&st, &stat, 1.0).unwrap();
    let upp = |t: &Theta| Stack { v: t.pack(&sp).iter().map(|c| st.clone_htod(c).unwrap()).collect() };
    let mut sp0 = trp.clone();
    for (k, v) in sp0.v.iter_mut().enumerate() {
        for (r, x) in v.iter_mut().enumerate() {
            for c in 0..3 {
                x[c] = start.v[k][r][c] - tr.v[k][r][c] + trp.v[k][r][c];
            }
        }
    }
    sp0.a.iter_mut().for_each(|a| *a = 0.0);
    let thp = upp(&sp0);
    let lam = pr.lambda_max(&st, &thp, true, 15).unwrap();
    let mut xh = host_of(&st, &thp);
    pr.reset();
    run_objective_host(&mut pr, &st, &mut xh, Precond::Zmshr1M, 1.0 / lam, 250, &mut |_, _| {}, &mut |_, _, _| {}, None).unwrap();
    let lensp = sp.lens();
    let xpe = Theta::unpack(&sp, &(0..lensp.len()).map(|t| xh.chunk(t).to_vec()).collect::<Vec<_>>());
    let a_err = (xpe.a.iter().zip(&trp.a).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / trp.a.iter().map(|y| y * y).sum::<f64>()).sqrt();
    let (ev0, ev1) = (vrms(&sp, &sp0, &trp, None), vrms(&sp, &xpe, &trp, None));
    eprintln!("residual phase + anchor (host state): a rel error 1 → {a_err:.3e}, velocity RMS error {ev0:.4} → {ev1:.2e} m/s");
    assert!(a_err < 0.05, "residual phase not recovered: rel error {a_err:.3e}");
    assert!(ev1 < 0.05 * ev0, "velocity not recovered with the residual phase: {ev0:.4} → {ev1:.4}");
}

/// The at-knots gradient through a gradient sink (`Objective::gradient_host_sink`, research/sample_kernels.md §20):
/// into an f32 sink it equals the streamed `gradient_host` up to the operator's run-to-run spread (the same additions in the same order), also
/// with coil groups (several vjp passes per set) and the static-tissue anchor; into a BFP sink of b bits the
/// accumulated chunks carry ≈ √(sets) roundings of 2^−b (bound 4 · √(sets + 1) · 2^−b of ‖g‖).
#[test]
fn knot_velocity_gradient_sink() {
    use hasty_recon::bfpstate::{Buf, GradSink, StackSink};
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let s = VelSpec::at_knots(DIMS, kenc(), KR, MC, None);
    let (tr, stat) = truth(&s);
    let mut pb = build(&dev, &kern, &s, &tr, 0x53);
    let model = VelModel::new(&dev, &st, s.clone()).unwrap();
    let ys = std::mem::take(&mut pb.ys);
    let mut pr = VelProblem::new(&st, kern.clone(), &mut pb.sets, ys, model, None).unwrap();
    assert!(pr.has_gradient_sink(), "at-knots mode has a gradient sink");
    let lens = s.lens();
    let mut rng = Rng(0x7a);
    let mut start = tr.clone();
    for v in start.v.iter_mut() {
        for x in v.iter_mut() {
            for c in 0..3 {
                x[c] += 0.1 * rng.n();
            }
        }
    }
    let th = Stack { v: start.pack(&s).iter().map(|c| st.clone_htod(c).unwrap()).collect() };
    let thh = host_of(&st, &th);
    for (case, groups, anchor) in [("plain", None, false), ("coil groups", Some(vec![(vec![0], 1.0), (vec![1, 2], 1.0)]), false), ("anchor", None, true)] {
        pr.cur_groups = groups;
        if anchor {
            pr.set_anchor(&st, &stat, 0.3).unwrap();
        }
        let mut gh = HostStack::zeros(&lens);
        let f = pr.gradient_host(&st, &thh, &mut gh).unwrap();
        // the operator's run-to-run spread (atomic adds in the adjoint): the floor of every comparison
        let mut gh2 = HostStack::zeros(&lens);
        let f2 = pr.gradient_host(&st, &thh, &mut gh2).unwrap();
        let spread = rel_host(&gh2, &gh);
        let fspread = (f2 / f - 1.0).abs();
        // f32 sink: the same additions in the same order, i.e. equal up to that spread
        let mut gs = HostStack::zeros(&lens);
        let fs = pr.gradient_host_sink(&st, &thh, &mut StackSink(&mut gs)).unwrap().expect("sink path");
        let e32 = rel_host(&gs, &gh);
        let b32 = (4.0 * spread).max(1e-6);
        eprintln!("gradient sink ({case}): f32 sink f rel {:.1e} (spread {fspread:.1e}), g rel {e32:.1e} (run-to-run spread {spread:.1e}, bound {b32:.1e})", (fs / f - 1.0).abs());
        assert!((fs / f - 1.0).abs() < (4.0 * fspread).max(1e-8) && e32 < b32, "{case}: f32 sink differs (f {fs:e} vs {f:e}, g rel {e32:e})");
        // BFP sinks
        for b in [24u32, 16] {
            let mut bb = Buf::zeros(Some(hasty_bfp::Bfp::new(b)), &lens);
            let fb = pr.gradient_host_sink(&st, &thh, &mut bb).unwrap().expect("sink path");
            let mut back = HostStack::zeros(&lens);
            for t in 0..lens.len() {
                assert!(bb.get(t, back.chunk_mut(t)).unwrap());
            }
            let e = rel_host(&back, &gh);
            let bound = 4.0 * ((s.ne() + 1) as f64).sqrt() * 2f64.powi(-(b as i32)) + 4.0 * spread;
            eprintln!("gradient sink ({case}): bfp{b} g rel {e:.2e} (bound {bound:.1e}), f {:.1e} rel", (fb / f - 1.0).abs());
            assert!(e < bound && (fb / f - 1.0).abs() < (4.0 * fspread).max(1e-6), "{case}: bfp{b} sink g rel {e:e} > {bound:e}");
        }
    }
}
