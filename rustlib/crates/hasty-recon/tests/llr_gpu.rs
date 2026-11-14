//! GPU: the GPU locally-low-rank denoiser (`llr_gpu::LlrGpu`, f64 Gram + batched eigendecomposition, f32 filter)
//! against the CPU operator (`llr::Llr`) on the same blocks and shift sequence: low-rank + noise stacks with a
//! partial support, T = 24 and T = 280, one device and the work split over two slots of device 0; exactly
//! rank-deficient blocks stay finite.
//!   cargo test --release -p hasty-recon --test llr_gpu -- --nocapture

use hasty_cuda::C32;
use hasty_recon::llr::Llr;
use hasty_recon::llr_gpu::LlrGpu;
use hasty_testkit::Rng;

fn stack(dims: [usize; 3], nt: usize, rank: usize, noise: f64, seed: u64) -> Vec<C32> {
    let nv: usize = dims.iter().product();
    let mut r = Rng::new(seed);
    let mut nrm = || {
        let (a, b) = (r.uniform().max(1e-300), r.uniform());
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    };
    let mut x = vec![C32::new(0.0, 0.0); nv * nt];
    for k in 0..rank {
        let amp = 10.0 * 0.5f64.powi(k as i32);
        let u: Vec<(f64, f64)> = (0..nv).map(|q| {
            let (i, j, l) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
            let s = ((i as f64 * 0.3 + k as f64).sin() * (j as f64 * 0.2).cos() + (l as f64 * 0.15 * (k + 1) as f64).sin()) * amp;
            (s, 0.5 * s * ((i + j) as f64 * 0.1).cos())
        }).collect();
        let v: Vec<(f64, f64)> = (0..nt).map(|_| (nrm(), nrm())).collect();
        for t in 0..nt {
            for q in 0..nv {
                let (ur, ui) = u[q];
                let (vr, vi) = v[t];
                let z = &mut x[t * nv + q];
                *z = C32::new(z.re + (ur * vr - ui * vi) as f32, z.im + (ur * vi + ui * vr) as f32);
            }
        }
    }
    for z in x.iter_mut() {
        *z = C32::new(z.re + (noise * nrm()) as f32, z.im + (noise * nrm()) as f32);
    }
    x
}

/// the tests share device 0 and one of them fills it: one at a time
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let (mut d, mut n) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        d += ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2);
        n += (y.re as f64).powi(2) + (y.im as f64).powi(2);
    }
    (d / n).sqrt()
}

#[test]
fn gpu_llr_matches_cpu() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    for (dims, nt, devices) in [([40usize, 36, 28], 24usize, vec![0usize]), ([24, 24, 16], 280, vec![0]), ([24, 24, 16], 280, vec![0, 0])] {
        let nv: usize = dims.iter().product();
        // support: an ellipsoid (blocks outside it are skipped, borders partial)
        let support: Vec<bool> = (0..nv).map(|q| {
            let (i, j, l) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
            let u = [(i as f64 / dims[0] as f64 - 0.5) / 0.45, (j as f64 / dims[1] as f64 - 0.5) / 0.4, (l as f64 / dims[2] as f64 - 0.5) / 0.45];
            u.iter().map(|v| v * v).sum::<f64>() < 1.0
        }).collect();
        let x0 = stack(dims, nt, 4, 0.3, 11);
        for kappa in [0.5, 2.0] {
            let mut cpu = Llr::new(dims, 8, kappa, support.clone());
            let mut gpu = LlrGpu::new(Llr::new(dims, 8, kappa, support.clone()), &devices).unwrap();
            // two calls each: the shift sequence must stay in step
            let (mut xc, mut xg) = (x0.clone(), x0.clone());
            for it in 0..2u64 {
                cpu.set_iteration(it);
                gpu.set_iteration(it);
                let t = std::time::Instant::now();
                cpu.apply(&mut xc, nt);
                let tc = t.elapsed().as_secs_f64();
                gpu.apply(&mut xg, nt).unwrap();
                let e = rel(&xg, &xc);
                let ch = rel(&xc, &x0);
                println!("dims {dims:?} T {nt} devices {devices:?} κ {kappa} call {it}: |gpu - cpu| / |cpu| {e:.2e} (the denoiser changes the stack by {ch:.2e}); cpu {tc:.2}s, gpu {:?}", gpu.last);
                assert!(e < 1e-4, "GPU LLR differs from the CPU operator: {e:e}");
            }
        }
    }
}

#[test]
fn gpu_llr_rank_deficient_stays_finite() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    // as llr::rank_deficient_tests: 24 images, 3 distinct
    let dims = [16usize, 16, 8];
    let nv = 16 * 16 * 8;
    let mut x = vec![C32::new(0.0, 0.0); 24 * nv];
    for t in 0..24 {
        for q in 0..nv {
            let c = if t < 8 { 0 } else { t % 3 };
            let v = ((q * 7 + c * 13) % 17) as f32 * 0.1 * (q % 5 != 0) as u8 as f32;
            x[t * nv + q] = C32::new(v * 0.8, v * 0.6 - 0.01 * c as f32);
        }
    }
    let mut xc = x.clone();
    let mut cpu = Llr::new(dims, 8, 1.0, vec![true; nv]);
    cpu.apply(&mut xc, 24);
    let mut gpu = LlrGpu::new(Llr::new(dims, 8, 1.0, vec![true; nv]), &[0]).unwrap();
    gpu.apply(&mut x, 24).unwrap();
    let bad = x.iter().filter(|c| !c.re.is_finite() || !c.im.is_finite()).count();
    assert_eq!(bad, 0, "{bad} non-finite values after the GPU LLR");
    let e = rel(&x, &xc);
    println!("rank-deficient: |gpu - cpu| / |cpu| {e:.2e}, {:?}", gpu.last);
    assert!(e < 1e-4, "{e:e}");
}

/// a batch budget far beyond the device (1 TiB) on a nearly full device: the allocations fail and the batch is halved
/// until it fits; the result is the CPU operator's as with a fitting budget
#[test]
fn gpu_llr_out_of_memory_retries_smaller_batches() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let dims = [24usize, 24, 16];
    let nt = 280;
    let nv: usize = dims.iter().product();
    let x0 = stack(dims, nt, 4, 0.3, 5);
    let mut cpu = Llr::new(dims, 8, 1.0, vec![true; nv]);
    let mut gpu = LlrGpu::new(Llr::new(dims, 8, 1.0, vec![true; nv]), &[0]).unwrap();
    gpu.budget = 1 << 40;
    let (mut xc, mut xg) = (x0.clone(), x0.clone());
    cpu.apply(&mut xc, nt);
    // the device nearly full (one batch of all 36 blocks needs ~0.3 GB): pass 1 and 2 must shrink their batches
    let dev = hasty_cuda::Device::new(0).unwrap();
    let st = dev.default_stream();
    let (free, _) = dev.ctx.mem_get_info().unwrap();
    let filler = st.alloc_zeros::<u8>(free.saturating_sub(160 << 20)).unwrap();
    st.synchronize().unwrap();
    let r = gpu.apply(&mut xg, nt);
    drop(filler);
    r.unwrap();
    let e = rel(&xg, &xc);
    println!("budget 1 TiB: |gpu - cpu| / |cpu| {e:.2e}, {:?}", gpu.last);
    assert!(e < 1e-4, "{e:e}");
}

/// rank cap (Llr::rank) and the hard rule (Llr::hard): GPU = CPU, at most R components kept
#[test]
fn gpu_llr_rank_cap_matches_cpu() {
    let _g = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let dims = [24usize, 24, 16];
    let nt = 40;
    let nv: usize = dims.iter().product();
    let x0 = stack(dims, nt, 12, 0.3, 9);
    for (kappa, rank, hard) in [(0.5, 3usize, false), (1e-3, 6, false), (1e-3, 6, true), (2.0, 40, true)] {
        let mut cpu = Llr::new(dims, 8, kappa, vec![true; nv]);
        cpu.rank = Some(rank);
        cpu.hard = hard;
        let mut l = Llr::new(dims, 8, kappa, vec![true; nv]);
        l.rank = Some(rank);
        l.hard = hard;
        let mut gpu = LlrGpu::new(l, &[0]).unwrap();
        let (mut xc, mut xg) = (x0.clone(), x0.clone());
        cpu.apply(&mut xc, nt);
        gpu.apply(&mut xg, nt).unwrap();
        let e = rel(&xg, &xc);
        println!("κ {kappa} rank cap {rank} hard {hard}: |gpu - cpu| / |cpu| {e:.2e}, change {:.2e}, k_max {}", rel(&xc, &x0), gpu.last.k_max);
        assert!(e < 1e-4, "{e:e}");
        assert!(gpu.last.k_max <= rank, "k_max {} > rank cap {rank}", gpu.last.k_max);
    }
}
