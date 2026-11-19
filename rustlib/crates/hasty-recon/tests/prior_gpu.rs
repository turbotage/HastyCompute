//! GPU: the device priors of `knotncg` (wavgpu.rs + LlrGpu::apply_dev) against the host path on small stacks:
//! - db4 coefficients W S x and the synthesis Sᴴ Wᴴ c against `WavHuber::coef / synth` (odd sizes, coarse lines shorter
//!   than the filter), round trip;
//! - envelope value, ψ, the chunked gradient, line values / derivatives at several steps, the soft threshold;
//! - the LLR sweep on a device stack against the host stack (same blocks, same statistics);
//! - `Priors::grad_dev` against `Priors::grad`.
//!   cargo test --release -p hasty-recon --test prior_gpu -- --nocapture
use hasty_cuda::{Device, C32};
use hasty_recon::gpu::Kernels;
use hasty_recon::knotncg::{hdot, LlrBackend, LlrHuber, Priors, WavHuber};
use hasty_recon::llr::Llr;
use hasty_recon::wavgpu::WavGpu;

fn rnd(n: usize, seed: u64, amp: f32) -> Vec<C32> {
    let mut s = seed;
    let mut u = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32 - 0.5
    };
    (0..n).map(|_| C32::new(amp * u(), amp * u())).collect()
}

/// smooth images plus noise (coefficients on both sides of θ)
fn images(d: [usize; 3], k: usize, seed: u64) -> Vec<C32> {
    let nv = d[0] * d[1] * d[2];
    let noise = rnd(nv * k, seed, 0.4);
    (0..nv * k)
        .map(|i| {
            let (q, t) = (i % nv, i / nv);
            let (x, y, z) = (q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1]));
            let v = ((x as f32) * 0.3).cos() * ((y as f32) * 0.2 + t as f32 * 0.4).sin() + 0.3 * ((z as f32) * 0.5).cos();
            C32::new(v + noise[i].re, 0.5 * v + noise[i].im)
        })
        .collect()
}

fn relc(a: &[C32], b: &[C32]) -> f64 {
    let d: Vec<C32> = a.iter().zip(b).map(|(x, y)| C32::new(x.re - y.re, x.im - y.im)).collect();
    (hdot(&d, &d) / hdot(b, b).max(1e-300)).sqrt()
}

fn relf(a: &[f32], b: &[f32]) -> f64 {
    let (mut e, mut n) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        e += ((x - y) as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (e / n.max(1e-300)).sqrt()
}

fn as_f32(v: &[C32]) -> Vec<f32> {
    v.iter().flat_map(|c| [c.re, c.im]).collect()
}

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn wavelet_gpu_matches_host() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    // (dims, levels): odd sizes, coarse lines of 6, 4 and 2 (shorter than the 8-tap filter: periodic wrap)
    for (d, lv, k) in [([18usize, 12, 10], 2usize, 3usize), ([20, 9, 7], 3, 2), ([33, 17, 12], 3, 2)] {
        let nv = d[0] * d[1] * d[2];
        let x = images(d, k, 7 + lv as u64);
        let w = WavHuber::new(d, lv, 0.15, 0.7);
        let npc = WavGpu::npc(&w);
        let wg = WavGpu::new(&dev, &st, k).unwrap();
        // coefficients
        let mut ch = vec![];
        w.coef(&x, k, &mut ch);
        let xd = st.clone_htod(&x).unwrap();
        let mut cd = st.alloc_zeros::<C32>(k * npc).unwrap();
        wg.coef(&st, &w, &xd, k, &mut cd).unwrap();
        let cg = as_f32(&st.clone_dtoh(&cd).unwrap());
        let ec = relf(&cg, &ch);
        // synthesis of the host coefficients, round trip of the device ones
        let mut chc = ch.clone();
        let mut sh = vec![C32::new(0.0, 0.0); nv * k];
        w.synth(&mut chc, k, &mut sh);
        let chd: Vec<C32> = ch.chunks(2).map(|v| C32::new(v[0], v[1])).collect();
        let mut c2 = st.clone_htod(&chd).unwrap();
        let mut out = st.clone_htod(&rnd(nv * k, 3, 1.0)).unwrap();
        wg.synth_axpby(&st, &w, &mut c2, k, 1.0, &mut out, 0.0).unwrap();
        let es = relc(&st.clone_dtoh(&out).unwrap(), &sh);
        wg.synth_axpby(&st, &w, &mut cd, k, 1.0, &mut out, 0.0).unwrap();
        let er = relc(&st.clone_dtoh(&out).unwrap(), &x);
        // a·synth + b·out
        let base = rnd(nv * k, 5, 1.0);
        let mut out = st.clone_htod(&base).unwrap();
        let mut c3 = st.clone_htod(&chd).unwrap();
        wg.synth_axpby(&st, &w, &mut c3, k, 0.5, &mut out, 2.0).unwrap();
        let want: Vec<C32> = sh.iter().zip(&base).map(|(a, b)| C32::new(0.5 * a.re + 2.0 * b.re, 0.5 * a.im + 2.0 * b.im)).collect();
        let eab = relc(&st.clone_dtoh(&out).unwrap(), &want);
        println!("d {d:?} L{lv} k {k} (box {:?}): coef rel {ec:.2e}, synth rel {es:.2e}, round trip {er:.2e}, axpby {eab:.2e}", w.p);
        assert!(ec < 2e-6 && es < 2e-6 && er < 2e-6 && eab < 2e-6, "{ec} {es} {er} {eab}");
        // value, ψ
        let mut psi_h = vec![0f32; ch.len()];
        let vh = w.value_psi(&ch, Some(&mut psi_h));
        let c4 = st.clone_htod(&chd).unwrap();
        let mut psi_d = st.alloc_zeros::<C32>(k * npc).unwrap();
        let vg = wg.value_psi(&st, &w, &c4, &mut psi_d, k).unwrap();
        let vg0 = wg.value(&st, &w, &c4, k).unwrap();
        let ep = relf(&as_f32(&st.clone_dtoh(&psi_d).unwrap()), &psi_h);
        let mut c5 = st.clone_htod(&chd).unwrap();
        let vgi = wg.value_psi_inplace(&st, &w, &mut c5, k).unwrap();
        let epi = relf(&as_f32(&st.clone_dtoh(&c5).unwrap()), &psi_h);
        println!("  value host {vh:.9e} gpu {vg:.9e} / {vg0:.9e} / in place {vgi:.9e}; ψ rel {ep:.2e} (in place {epi:.2e})");
        for v in [vg, vg0, vgi] {
            assert!((v - vh).abs() <= 1e-6 * vh.abs(), "value {v} vs {vh}");
        }
        assert!(ep < 1e-6 && epi < 1e-6);
        // line along a direction at several steps (one sweep of ≤ 8 and a second group)
        let dir = rnd(nv * k, 11, 1.0);
        let mut cdh = vec![];
        w.coef(&dir, k, &mut cdh);
        let cdd = st.clone_htod(&cdh.chunks(2).map(|v| C32::new(v[0], v[1])).collect::<Vec<_>>()).unwrap();
        let steps = [0.0, 0.05, 0.1, 0.3, -0.2, 1.0, 2.5, 0.01, 0.7, 0.123];
        let lg = wg.line(&st, &w, &c4, &cdd, k, &steps).unwrap();
        for (a, (v, dv)) in steps.iter().zip(&lg) {
            let (hv, hd) = w.line(&ch, &cdh, *a);
            assert!((v - hv).abs() <= 1e-6 * hv.abs(), "line value at {a}: {v} vs {hv}");
            assert!((dv - hd).abs() <= 1e-5 * hd.abs().max(1e-3 * hv.abs()), "line derivative at {a}: {dv} vs {hd}");
        }
        println!("  line: {} steps match (value 1e-6, derivative 1e-5)", steps.len());
        // soft threshold
        let mut sh2 = ch.clone();
        w.soft(&mut sh2, 0.2);
        let mut c6 = st.clone_htod(&chd).unwrap();
        wg.soft(&st, &w, &mut c6, k, 0.2).unwrap();
        let eso = relf(&as_f32(&st.clone_dtoh(&c6).unwrap()), &sh2);
        assert!(eso < 1e-6, "soft {eso}");
        // the chunked gradient: g = a ∇ + b g with chunks of 1 and 2 knots
        let mut gh = vec![C32::new(0.0, 0.0); nv * k];
        let mut ps = psi_h.clone();
        w.synth(&mut ps, k, &mut gh);
        for kc in [1usize, 2] {
            let mut scr = st.alloc_zeros::<C32>(kc * npc).unwrap();
            let mut g = st.clone_htod(&base).unwrap();
            let v = wg.grad(&st, &w, &xd, k, &mut scr, &mut g, 1.0, 0.0).unwrap();
            let eg = relc(&st.clone_dtoh(&g).unwrap(), &gh);
            println!("  gradient (chunk {kc}): value {v:.9e}, rel {eg:.2e}");
            assert!((v - vh).abs() <= 1e-6 * vh.abs() && eg < 2e-6, "{v} vs {vh}, {eg}");
        }
    }
}

#[test]
fn llr_device_stack_matches_host() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let d = [16usize, 12, 8];
    let (k, nv) = (5usize, 16 * 12 * 8);
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let x: Vec<C32> = images(d, k, 3);
    // support with holes: blocks outside are skipped (their voxels unchanged)
    let sup: Vec<bool> = (0..nv).map(|q| (q % 16) < 12).collect();
    let mk = || {
        let mut l = Llr::new(d, 4, 1.0, sup.clone());
        l.zeta_basis = Some((0..k * k).map(|q| if q / k == q % k { 1.0 } else { 0.0 }).collect());
        l
    };
    for (thr, it) in [(None, 1u64), (Some(1.5), 3), (Some(1e9), 5)] {
        let mut lh = mk();
        lh.thr_abs = thr;
        lh.set_iteration(it);
        let mut xh = x.clone();
        let mut gh = hasty_recon::llr_gpu::LlrGpu::new(lh, &[0]).unwrap();
        gh.apply(&mut xh, k).unwrap();
        let mut ld = mk();
        ld.thr_abs = thr;
        ld.set_iteration(it);
        let mut gd = hasty_recon::llr_gpu::LlrGpu::new(ld, &[0]).unwrap();
        let mut xd = st.clone_htod(&x).unwrap();
        gd.apply_dev(&st, &mut xd, k).unwrap();
        let out = st.clone_dtoh(&xd).unwrap();
        let e = relc(&out, &xh);
        let (sa, sb) = (&gh.llr.last, &gd.llr.last);
        println!("LLR thr {thr:?}: device vs host stack rel {e:.2e}; σ {:.6e} / {:.6e}, kept {} / {}, blocks {} / {}", sa.sigma, sb.sigma, sa.n_above, sb.n_above, gh.last.blocks, gd.last.blocks);
        assert!(e < 1e-6, "{e}");
        assert_eq!(sa.n_above, sb.n_above);
        assert!((sa.s2_above - sb.s2_above).abs() <= 1e-9 * sa.s2_above.abs().max(1e-30));
    }
}

#[test]
fn priors_grad_dev_matches_host() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let d = [18usize, 12, 10];
    let (k, nv) = (4usize, 18 * 12 * 10);
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Kernels::new(&dev).unwrap();
    let x = images(d, k, 9);
    let mk = |gpu: bool| {
        let w = WavHuber::new(d, 2, 0.12, 0.6);
        let mut l = Llr::new(d, 4, 1.0, vec![true; nv]);
        l.zeta_basis = None;
        let be = if gpu { LlrBackend::Gpu(hasty_recon::llr_gpu::LlrGpu::new(l, &[0]).unwrap()) } else { LlrBackend::Cpu(l) };
        Priors::new(k, nv, Some(w), Some(LlrHuber { be, theta: 1.0, gamma: 0.6, shift: 4 }))
    };
    let mut ph = mk(false);
    let mut gh = vec![C32::new(0.0, 0.0); nv * k];
    let (vwh, vlh) = ph.grad(&x, &mut gh).unwrap();
    for chunk in [1usize, 3, 8] {
        let mut pd = mk(true);
        pd.to_gpu(&dev, &st, chunk).unwrap();
        let xd = st.clone_htod(&x).unwrap();
        let base = rnd(nv * k, 17, 1.0);
        let mut g = st.clone_htod(&base).unwrap();
        // g = 2 ∇R + 0.5 g
        let (vw, vl) = pd.grad_dev(&kern, &st, &xd, &mut g, 2.0, 0.5).unwrap();
        let want: Vec<C32> = gh.iter().zip(&base).map(|(a, b)| C32::new(2.0 * a.re + 0.5 * b.re, 2.0 * a.im + 0.5 * b.im)).collect();
        let e = relc(&st.clone_dtoh(&g).unwrap(), &want);
        println!("Priors::grad_dev (chunk {chunk}) vs grad: wavelet {vw:.8e} / {vwh:.8e}, LLR {vl:.8e} / {vlh:.8e}, gradient rel {e:.2e}; transforms {} sweeps {}", pd.n_wt, pd.n_llr);
        // the LLR sweeps differ (GPU vs CPU eigensolver): as llr_envelope_gpu_matches_cpu, 1e-4
        assert!((vw - vwh).abs() <= 1e-6 * vwh.abs() && (vl - vlh).abs() <= 1e-4 * vlh.abs(), "values");
        assert!(e < 1e-4, "{e}");
        // b = 0 must not read g (NaN there)
        let mut gn = st.clone_htod(&vec![C32::new(f32::NAN, f32::NAN); nv * k]).unwrap();
        pd.grad_dev(&kern, &st, &xd, &mut gn, 1.0, 0.0).unwrap();
        let e0 = relc(&st.clone_dtoh(&gn).unwrap(), &gh);
        assert!(e0 < 1e-4, "b = 0: {e0}");
    }
}

/// Real-size timing (p12_k32: 40 knots of [254, 144, 114], db4 L3, LLR B8): one prior gradient and one line sweep,
/// host path (CPU wavelet + LlrGpu on host stacks) vs device mode. Not a correctness test; run explicitly:
///   cargo test --release -p hasty-recon --test prior_gpu real_size_timing -- --ignored --nocapture
#[test]
#[ignore]
fn real_size_timing() {
    use std::time::Instant;
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let d = [254usize, 144, 114];
    let k = 40usize;
    let nv = d[0] * d[1] * d[2];
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Kernels::new(&dev).unwrap();
    let x = images(d, k, 5);
    let mk = || {
        let w = WavHuber::new(d, 3, 0.05, 0.5);
        let l = Llr::new(d, 8, 1.0, vec![true; nv]);
        let mut g = hasty_recon::llr_gpu::LlrGpu::new(l, &[0]).unwrap();
        g.budget = 1 << 30;
        Priors::new(k, nv, Some(w), Some(LlrHuber { be: LlrBackend::Gpu(g), theta: 1.0, gamma: 0.5, shift: 3 }))
    };
    // host path (as --prior-dev cpu): dtoh x, CPU wavelet, LLR on host stacks, htod of the gradient
    let xd = st.clone_htod(&x).unwrap();
    let mut gd = st.alloc_zeros::<C32>(nv * k).unwrap();
    let mut ph = mk();
    let mut xh = vec![C32::new(0.0, 0.0); nv * k];
    let mut gh = vec![C32::new(0.0, 0.0); nv * k];
    for rep in 0..2 {
        let t = Instant::now();
        st.memcpy_dtoh(&xd, &mut xh).unwrap();
        let t1 = Instant::now();
        ph.wav.as_ref().unwrap().coef(&xh, k, &mut vec![]);
        let tw = t1.elapsed().as_secs_f64();
        ph.grad(&xh, &mut gh).unwrap();
        st.memcpy_htod(&gh, &mut gd).unwrap();
        st.synchronize().unwrap();
        println!("host path gradient (rep {rep}): {:.2}s (one CPU coefficient stack transform alone {tw:.2}s)", t.elapsed().as_secs_f64());
    }
    let w = ph.wav.as_ref().unwrap();
    let (mut ch, mut cdh) = (vec![], vec![]);
    w.coef(&xh, k, &mut ch);
    w.coef(&gh, k, &mut cdh);
    let t = Instant::now();
    let _ = w.line(&ch, &cdh, 0.1);
    println!("host line evaluation (one step): {:.3}s", t.elapsed().as_secs_f64());
    drop((ch, cdh, xh, gh));
    // LLR alone: host stack vs device stack
    let l = ph.llr.as_mut().unwrap();
    let mut xh = st.clone_dtoh(&xd).unwrap();
    let t = Instant::now();
    l.svt(&mut xh, k, 1.0).unwrap();
    println!("LLR sweep on the host stack: {:.2}s", t.elapsed().as_secs_f64());
    drop(xh);
    let mut xt = st.clone_dtod(&xd).unwrap();
    for rep in 0..2 {
        st.memcpy_dtod(&xd, &mut xt).unwrap();
        st.synchronize().unwrap();
        let t = Instant::now();
        l.svt_dev(&st, &mut xt, k, 1.0).unwrap();
        st.synchronize().unwrap();
        let s = match &l.be { LlrBackend::Gpu(g) => g.last.clone(), _ => unreachable!() };
        println!("LLR sweep on the device stack (rep {rep}): {:.2}s ({s:?})", t.elapsed().as_secs_f64());
    }
    drop(xt);
    drop(ph);
    // device mode
    let mut pd = mk();
    pd.to_gpu(&dev, &st, 8).unwrap();
    for rep in 0..3 {
        st.synchronize().unwrap();
        let t = Instant::now();
        pd.grad_dev(&kern, &st, &xd, &mut gd, 1.0, 0.0).unwrap();
        println!("device gradient (rep {rep}): {:.3}s", t.elapsed().as_secs_f64());
    }
    let w = pd.wav.as_ref().unwrap();
    let wg = pd.gpu.as_ref().unwrap().wg.as_ref().unwrap();
    let npc = WavGpu::npc(w);
    let mut cx = st.alloc_zeros::<C32>(k * npc).unwrap();
    let mut cd = st.alloc_zeros::<C32>(k * npc).unwrap();
    for rep in 0..3 {
        st.synchronize().unwrap();
        let t = Instant::now();
        wg.coef(&st, w, &xd, k, &mut cx).unwrap();
        st.synchronize().unwrap();
        let tc = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let v = wg.value_psi(&st, w, &cx, &mut cd, k).unwrap();
        let tp = t.elapsed().as_secs_f64();
        let t = Instant::now();
        wg.synth_axpby(&st, w, &mut cd, k, 1.0, &mut gd, 1.0).unwrap();
        st.synchronize().unwrap();
        let ts = t.elapsed().as_secs_f64();
        wg.coef(&st, w, &gd, k, &mut cd).unwrap();
        st.synchronize().unwrap();
        let t = Instant::now();
        let _ = wg.line(&st, w, &cx, &cd, k, &[0.1]).unwrap();
        let t1 = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _ = wg.line(&st, w, &cx, &cd, k, &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]).unwrap();
        let t8 = t.elapsed().as_secs_f64();
        println!("device (rep {rep}): coef {:.1} ms, value+ψ {:.1} ms (value {v:.4e}), synth+axpby {:.1} ms, line sweep 1 step {:.1} ms, 8 steps {:.1} ms", tc * 1e3, tp * 1e3, ts * 1e3, t1 * 1e3, t8 * 1e3);
    }
}
