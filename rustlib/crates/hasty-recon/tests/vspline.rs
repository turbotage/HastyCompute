//! GPU: the parameter-spline velocity model (`VelSpec::splines`, research/velocity_model_sr.md §F) through the real
//! 2-D knot operator (`SegNufft::new_temporal2d` with the operator frames as its cardiac knots, cubic, prefiltered),
//! respiratory groups (sets = encodes × groups, here forced to 2 knots per set so that 3 respiratory knots need 2 groups):
//! - the objective's gradient against central finite differences (f32 pipeline);
//! - host-resident (row-streamed) gradient / Hessian-vector product / λ_max equal the device ones;
//! - model-consistent data: P²nP (ZMSHR1 in the Gauss–Newton metric) recovers m, ψ and v from a perturbed start, state
//!   on the device and on the host (same iterates).
//!   cargo test --release -p hasty-recon --test vspline -- --nocapture
use std::sync::Arc;

use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts};
use hasty_recon::gpu::Kernels;
use hasty_recon::hoststack::{run_objective_host, HostStack};
use hasty_recon::p2np::{rdot, run_objective, Objective, Precond, Stack};
use hasty_recon::setop::{SegSets, SetOp};
use hasty_recon::vknot::OpFrames;
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
const KR: usize = 3;
const MC: usize = 4;
const FOP: usize = 10;

fn kenc() -> Vec<[f64; 3]> {
    let (lo, hi) = (std::f64::consts::PI / 1.0, std::f64::consts::PI / 2.5);
    let mut k = vec![[-0.5, -0.5, -0.5]];
    for a in 0..3 {
        for v in [lo, hi] {
            let mut e = [-0.5; 3];
            e[a] += v;
            k.push(e);
        }
    }
    k
}

fn raw(s: &cudarc::driver::CudaSlice<C32>, st: &Arc<cudarc::driver::CudaStream>) -> u64 {
    use cudarc::driver::DevicePtr;
    s.device_ptr(st).0
}

/// magnitude blob, a static phase ramp, pulsatile velocity in a "vessel" (static outside)
fn truth(s: &VelSpec) -> Theta {
    let mut t = Theta::zeros(s);
    let [nx, ny, nz] = s.dims;
    for q in 0..s.nv() {
        let (i, j, k) = ((q % nx) as f64, ((q / nx) % ny) as f64, (q / (nx * ny)) as f64);
        let m = (-((i - 6.0).powi(2) / 20.0 + (j - 5.0).powi(2) / 14.0 + (k - 4.0).powi(2) / 8.0)).exp() + 0.3;
        let vessel = (i - 6.0).powi(2) + (j - 5.0).powi(2) < 6.5;
        for kk in 0..s.kz {
            let (rr, c) = (kk / MC, kk % MC);
            t.z[kk][q] = C::new(m * (1.0 + 0.1 * rr as f64 + 0.05 * c as f64), 0.3 * (i / nx as f64) + 0.05 * rr as f64);
            if vessel {
                let ph = 2.0 * std::f64::consts::PI * c as f64 / MC as f64;
                t.v[kk][q] = [0.15 * ph.cos(), 0.1 * ph.sin() * (1.0 + 0.2 * rr as f64), 0.4 + 0.3 * ph.sin() + 0.05 * (k / nz as f64)];
            }
        }
    }
    t
}

fn host_of(st: &Arc<cudarc::driver::CudaStream>, x: &Stack) -> HostStack {
    let lens: Vec<usize> = x.v.iter().map(|c| c.len()).collect();
    let mut h = HostStack::zeros(&lens);
    for t in 0..lens.len() {
        st.memcpy_dtoh(&x.v[t], h.chunk_mut(t)).unwrap();
    }
    h
}

fn rel_host(a: &HostStack, b: &HostStack) -> f64 {
    let d: f64 = a.data.iter().zip(&b.data).map(|(p, q)| ((p.re - q.re) as f64).powi(2) + ((p.im - q.im) as f64).powi(2)).sum();
    (d / HostStack::rdot(b, b).max(1e-300)).sqrt()
}

fn vrms(s: &VelSpec, a: &Theta, b: &Theta) -> f64 {
    let mut e = 0.0;
    for k in 0..s.kv {
        for q in 0..s.nv() {
            for c in 0..3 {
                e += (a.v[k][q][c] - b.v[k][q][c]).powi(2);
            }
        }
    }
    (e / (s.kv * s.nv() * 3) as f64).sqrt()
}

#[test]
fn spline_velocity_model_gradient_streaming_and_recovery() {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let of = OpFrames::with_kre(KR, MC, 3, FOP, 3, 2);
    assert_eq!(of.r0, vec![0, 1]);
    let s = VelSpec::splines(DIMS, kenc(), of.clone(), None);
    let (nv, ne, ng) = (s.nv(), s.ne(), of.ng());
    let tr = truth(&s);
    // operator: 2-D knot engine with kre respiratory knots, the operator frames as cubic cardiac knots
    let (nc, nseg, m) = (3usize, 2usize, 5000usize);
    let mut rng = Rng(0x5157);
    let smaps: Vec<C32> = (0..nc * nv).map(|i| { let c = i / nv; let r = i % nv; let ph = 0.5 * c as f64 + 0.003 * r as f64; C32::new((0.7 * ph.cos()) as f32, (0.7 * ph.sin()) as f32) }).collect();
    let fit: Vec<usize> = (0..nv).step_by(3).collect();
    let psi = |v: usize, tt: f64| 300.0 * ((v % 17) as f64 / 17.0 - 0.5) * tt;
    let (sp, at, t0, dt) = build_segments(nv, nseg, 2e-3, 8e-3, 64, &fit, &psi);
    let opts = SegNufftOpts { sigma: 1.25, width: 6, nseg, order: 0, frames: FOP, bspline: 3, max_grid_bytes: 1 << 28 };
    let mut eng = SegNufft::new_temporal2d(&dev, DIMS, [0.004; 3], nc, opts, of.kre, 1).unwrap();
    eng.set_maps(&st, &smaps).unwrap();
    let seg = eng.segments(&st, &sp, &at, t0, dt).unwrap();
    let pts: Vec<_> = (0..ne * ng)
        .map(|_| {
            let c: [Vec<f32>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| (std::f64::consts::PI * 0.999 * rng.n()) as f32).collect());
            let tau: Vec<f32> = (0..m).map(|_| (2e-3 + 6e-3 * rng.u()) as f32).collect();
            let phi0 = vec![0f32; m];
            let t: Vec<f32> = (0..m).map(|_| (FOP as f64 * rng.u()) as f32).collect();
            let ri = vec![0u32; m];
            let rv: Vec<f32> = (0..m).map(|_| rng.u() as f32).collect();
            eng.points2d(&st, [&c[0], &c[1], &c[2]], &tau, &phi0, &[], &t, &ri, &rv).unwrap()
        })
        .collect();
    let mut sets = SegSets { eng, pts, segs: vec![seg], seg_of_set: vec![0; ne * ng], kern: kern.clone(), surrogate: None };
    // model-consistent data: the set images of the truth through the operator
    let model = VelModel::new(&dev, &st, s.clone()).unwrap();
    let up = |t: &Theta| Stack { v: t.pack(&s).iter().map(|c| st.clone_htod(c).unwrap()).collect() };
    let tg = up(&tr);
    let mut ys = vec![];
    for t in 0..ne * ng {
        let (e, g) = of.set_eg(t);
        let x = st.alloc_zeros::<C32>(of.set_frames() * nv).unwrap();
        for rl in 0..of.kre {
            let r = of.r0[g] + rl;
            let zr: Vec<u64> = (0..MC).map(|c| raw(&tg.v[s.chunk_z(r * MC + c)], &st)).collect();
            let vr: Vec<u64> = (0..MC).map(|c| raw(&tg.v[s.chunk_v(r * MC + c)], &st)).collect();
            model.eval_row(&st, &zr, &vr, raw(&tg.v[s.chunk_a()], &st), e, raw(&x, &st) + (rl * FOP * nv * 8) as u64).unwrap();
        }
        model.prefilter(&st, raw(&x, &st), of.kre).unwrap();
        let mut y = st.alloc_zeros::<C32>(sets.len_y(t)).unwrap();
        sets.forward(&st, t, &x, &mut y).unwrap();
        ys.push(y);
    }
    let mut pr = VelProblem::new(&st, kern.clone(), &mut sets, ys, model, None).unwrap();
    pr.set_knot_weights(&(0..s.kz).map(|k| 0.7 + 0.05 * k as f64).collect::<Vec<_>>());
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
            *x = C::new(x.re * (1.0 + 0.05 * rng.n()), x.im + 0.05 * rng.n());
        }
    }
    let lens = s.lens();
    // 1. gradient vs central differences
    let th = up(&start);
    let mut g = Stack::zeros_lens(&st, &lens).unwrap();
    let f0 = pr.gradient(&st, &th, &mut g).unwrap();
    let mut d = Theta::zeros(&s);
    for k in 0..s.kv {
        for q in 0..nv {
            d.v[k][q] = [rng.n(), rng.n(), rng.n()];
            d.z[k][q] = C::new(rng.n(), rng.n()) * 0.1;
        }
    }
    let h = 2e-3;
    let fp = pr.gradient(&st, &up(&start.add(h, &d)), &mut Stack::zeros_lens(&st, &lens).unwrap()).unwrap();
    let fm = pr.gradient(&st, &up(&start.add(-h, &d)), &mut Stack::zeros_lens(&st, &lens).unwrap()).unwrap();
    let fd = (fp - fm) / (2.0 * h);
    let gh = Theta::unpack(&s, &g.v.iter().map(|c| st.clone_dtoh(c).unwrap()).collect::<Vec<_>>());
    let an = gh.dot(&d);
    eprintln!("spline objective: f {f0:.4e}, directional derivative fd {fd:.6e} vs analytic {an:.6e} (rel {:.1e})", (fd - an).abs() / an.abs());
    assert!((fd - an).abs() < 2e-3 * an.abs(), "objective gradient: fd {fd:.6e} vs analytic {an:.6e}");
    // 2. host-resident row-streamed gradient = device gradient
    let thh = host_of(&st, &th);
    let mut gh2 = HostStack::zeros(&lens);
    let f1 = pr.gradient_host(&st, &thh, &mut gh2).unwrap();
    let e_g = rel_host(&gh2, &host_of(&st, &g));
    eprintln!("streamed gradient vs device: f rel {:.1e}, g rel {e_g:.1e}", (f1 / f0 - 1.0).abs());
    assert!((f1 / f0 - 1.0).abs() < 1e-5 && e_g < 1e-5, "streamed gradient differs: f {f1:e} vs {f0:e}, g rel {e_g:e}");
    // 3. Hessian-vector products (device vs streamed, symmetry) and λ_max
    let dd = up(&d);
    let mut hd = Stack::zeros_lens(&st, &lens).unwrap();
    pr.hvp(&st, &th, &dd, &mut hd).unwrap();
    let mut hh = HostStack::zeros(&lens);
    pr.hvp_host(&st, &thh, &host_of(&st, &dd), &mut hh).unwrap();
    let e_h = rel_host(&hh, &host_of(&st, &hd));
    let sym = { let w = up(&start); let mut hw = Stack::zeros_lens(&st, &lens).unwrap(); pr.hvp(&st, &th, &w, &mut hw).unwrap(); (rdot(&kern, &st, &w, &hd).unwrap(), rdot(&kern, &st, &dd, &hw).unwrap()) };
    eprintln!("Hessian-vector product: streamed vs device rel {e_h:.1e}; symmetry {:.4e} vs {:.4e}", sym.0, sym.1);
    assert!(e_h < 1e-5, "streamed hvp differs: {e_h:e}");
    assert!((sym.0 - sym.1).abs() < 1e-4 * sym.0.abs().max(sym.1.abs()), "Hessian not symmetric");
    // the Gauss–Newton product is the directional derivative of the gradient's data part (f32: loose)
    let lam = pr.lambda_max(&st, &th, true, 15).unwrap();
    let lam_h = pr.lambda_max_host(&st, &thh, true, 15).unwrap();
    eprintln!("λ_max(MH): device {lam:.5e}, host {lam_h:.5e}");
    assert!((lam_h / lam - 1.0).abs() < 1e-3, "λ_max differs");
    // 4. recovery from model-consistent data, state on the device and on the host
    let e0 = vrms(&s, &start, &tr);
    let iters = 250;
    let mut xd = up(&start);
    pr.reset();
    let mut fs_d = vec![];
    run_objective(&mut pr, &st, &mut xd, Precond::Zmshr1M, 1.0 / lam, iters, &mut |_, _| {}, &mut |_, _, f| fs_d.push(f), None).unwrap();
    let xe = Theta::unpack(&s, &xd.v.iter().map(|c| st.clone_dtoh(c).unwrap()).collect::<Vec<_>>());
    let e_dev = vrms(&s, &xe, &tr);
    pr.reset();
    let mut xh = host_of(&st, &up(&start));
    let mut fs_h = vec![];
    run_objective_host(&mut pr, &st, &mut xh, Precond::Zmshr1M, 1.0 / lam, iters, &mut |_, _| {}, &mut |_, _, f| fs_h.push(f), None).unwrap();
    let xhe = Theta::unpack(&s, &(0..lens.len()).map(|t| xh.chunk(t).to_vec()).collect::<Vec<_>>());
    let e_host = vrms(&s, &xhe, &tr);
    let zerr = |a: &Theta| { let (mut e, mut n) = (0.0, 0.0); for k in 0..s.kz { for q in 0..nv { e += (a.z[k][q] - tr.z[k][q]).norm_sqr(); n += tr.z[k][q].norm_sqr(); } } (e / n).sqrt() };
    eprintln!("recovery: velocity RMS error {e0:.4} → device {e_dev:.2e} / host {e_host:.2e} m/s; (m, ψ) rel error {:.2e} / {:.2e}; f {:.3e} → {:.3e} (host {:.3e}); f at it 10 device {:.6e} host {:.6e}", zerr(&xe), zerr(&xhe), fs_d[1], fs_d.last().unwrap(), fs_h.last().unwrap(), fs_d[10], fs_h[10]);
    assert!(e_dev < 0.02 * e0 && e_host < 0.02 * e0, "velocity not recovered: {e0:.4} → {e_dev:.4} / {e_host:.4}");
    assert!((fs_h[10] / fs_d[10] - 1.0).abs() < 1e-3, "host and device iterates differ");
    // 5. the robust start's complex knot CGLS (VelProblem::complex_knots) on the same sets: the normal-equation residual
    // falls, and the knots' encode-difference phase follows the truth velocity in the vessel
    let x0: Vec<C32> = (0..nv).map(|q| { let z = tr.z[0][q]; C32::new((z.re * z.im.cos()) as f32, (z.re * z.im.sin()) as f32) }).collect();
    let (ck, hist) = pr.complex_knots(&st, 1, &x0, 25, 0.0).unwrap();
    let (ck0, _) = pr.complex_knots(&st, 0, &x0, 25, 0.0).unwrap();
    eprintln!("complex knots: ‖s‖/‖s₀‖ after 1, 5, 25 iterations {:.2e} {:.2e} {:.2e}", hist[0], hist[4], hist[24]);
    assert!(hist[24] < 0.1 * hist[0], "complex knot CGLS does not converge: {hist:?}");
    let ke = s.kenc_model();
    let (mut e2, mut n2) = (0.0, 0.0);
    for k in 0..s.kz {
        for q in 0..nv {
            let v = tr.v[k][q];
            if v[2] == 0.0 {
                continue;
            }
            let (a, b) = (ck[k * nv + q], ck0[k * nv + q]);
            let want = -((ke[1][0] - ke[0][0]) * v[0] + (ke[1][1] - ke[0][1]) * v[1] + (ke[1][2] - ke[0][2]) * v[2]);
            let got = (C::new(a.re as f64, a.im as f64) * C::new(b.re as f64, -b.im as f64)).arg();
            e2 += ((got - want + std::f64::consts::PI).rem_euclid(2.0 * std::f64::consts::PI) - std::f64::consts::PI).powi(2);
            n2 += 1.0;
        }
    }
    let rms = (e2 / n2).sqrt();
    eprintln!("complex knots: encode 1 − 0 phase in the vessel vs the truth velocity's (coefficients, not values): RMS {rms:.3} rad");
    assert!(rms < 0.5, "complex knots: phase RMS {rms:.3} rad");
}
