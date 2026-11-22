//! Per-GPU tune cache (segnufft/tune.rs, research/arch_targets.md item 0; env HASTY_TUNE, HASTY_TUNE_CACHE).
//! One test (the steps share the process environment), on a small 2-D knot order-3 engine (vs / ra / kl candidates):
//! 1. reuse, empty cache: the autotune times candidates and writes one file;
//! 2. a second construction with the in-process copy dropped: loaded from the file, NO timing launches, the same
//!    choices, bitwise-identical forward and adjoint output (adjoint: when the operator itself is bitwise repeatable);
//! 3. a second construction in the same process (in-process copy): no timing either;
//! 4. another shape (coil batch): a miss;
//! 5. a corrupted file: a miss (retimed, no panic) and the file rewritten readable (a missing file: step 1);
//! 6. HASTY_TUNE=retune: times again; HASTY_TUNE=off: times, and writes nothing.
//!   cargo test --release -p hasty-nufft --test seg_tune -- --nocapture
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, hi_exponents, set_thread_cb_cap, tune_cache_forget, tune_timing_stats, SegNufft, SegNufftOpts, SegPoints, SegSegments, NHI};

struct R(u64);
impl R {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

struct Case {
    st: std::sync::Arc<cudarc::driver::CudaStream>,
    op: SegNufft,
    seg: SegSegments,
    pts: SegPoints,
    x: cudarc::driver::CudaSlice<C32>,
    yd: cudarc::driver::CudaSlice<C32>,
    nimg: usize,
    m: usize,
    nc: usize,
}

/// 2-D knots (3 respiratory linear × 6 cardiac cubic), order 3, the same data for every call
fn build(nc: usize, nseg: usize, cb_cap: usize) -> Case {
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let dims = [24usize, 22, 24];
    let nv: usize = dims.iter().product();
    let (res, m, width, order) = (0.004, 30_000usize, 10usize, 3u32);
    let (mc, kr) = (6, 3);
    let mut r = R(4242);
    let mut c: [Vec<f32>; 3] = Default::default();
    while c[0].len() < m {
        let p = [r.u() * 2.0, r.u() * 2.0, r.u() * 2.0];
        if p.iter().map(|v| v * v).sum::<f64>() < 1.0 {
            for a in 0..3 {
                c[a].push((p[a] * 0.62 * std::f64::consts::PI) as f32);
            }
        }
    }
    let tau: Vec<f32> = (0..m).map(|_| (5e-3 + r.u() * 6e-3) as f32).collect();
    let phi0: Vec<f32> = (0..m).map(|_| (r.u() * 6.0) as f32).collect();
    let dcv: Vec<f32> = (0..6 * m).map(|_| (r.u() * 1.2e4) as f32).collect();
    let t: Vec<f32> = (0..m).map(|_| ((r.u() + 0.5) * mc as f64) as f32).collect();
    let nri = kr - 1;
    let ri: Vec<u32> = (0..m).map(|_| (((r.u() + 0.5) * nri as f64) as u32).min(nri as u32 - 1)).collect();
    let rv: Vec<f32> = (0..m).map(|_| (r.u() + 0.5) as f32).collect();
    let nimg = mc * kr;
    let xh: Vec<C32> = (0..nimg * nv).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let yh: Vec<C32> = (0..nc * m).map(|_| C32::new(r.u() as f32, r.u() as f32)).collect();
    let smaps: Vec<C32> = (0..nc * nv).map(|i| C32::new((1.0 + 0.3 * ((i * 7) % 11) as f64 / 11.0) as f32, (0.2 * ((i * 3) % 5) as f64 / 5.0) as f32)).collect();
    let fit: Vec<usize> = (0..nv).step_by(7).collect();
    let psi = |v: usize, tt: f64| 1500.0 * ((v % 113) as f64 / 113.0 - 0.5) * tt;
    let (sp, at, t0, dt) = build_segments(nv, nseg, 2e-3, 8e-3, 64, &fit, &psi);
    set_thread_cb_cap(Some(cb_cap));
    let opts = SegNufftOpts { sigma: 1.25, width, nseg, order, frames: mc, bspline: 3, max_grid_bytes: 1 << 30 };
    let mut op = SegNufft::new_temporal2d(&dev, dims, [res; 3], nc, opts, kr, 1).unwrap();
    set_thread_cb_cap(None);
    op.set_maps(&st, &smaps).unwrap();
    let seg = op.segments(&st, &sp, &at, t0, dt).unwrap();
    let u: Vec<f32> = (0..4 * m).map(|_| r.u() as f32).collect();
    let g: [f64; 3] = [0, 1, 2].map(|a| res * op.plan.nf[a] as f64 / (std::f64::consts::PI * width as f64));
    let ex = hi_exponents();
    let h: Vec<f64> = (0..4 * NHI).map(|i| 0.2 * r.u() / (0..3).map(|a| g[a].powi(ex[i % NHI][a] as i32)).product::<f64>()).collect();
    let pts = op.points2d_hi(&st, [&c[0][..], &c[1][..], &c[2][..]], &tau, &phi0, &dcv, &t, &ri, &rv, &u, &h).unwrap();
    let (x, yd) = (st.clone_htod(&xh).unwrap(), st.clone_htod(&yh).unwrap());
    Case { st, op, seg, pts, x, yd, nimg, m, nc }
}

impl Case {
    fn fwd(&mut self) -> Vec<C32> {
        let mut y = self.st.alloc_zeros::<C32>(self.nc * self.m).unwrap();
        self.op.forward(&self.st, &self.seg, &self.pts, &self.x, &mut y).unwrap();
        self.st.clone_dtoh(&y).unwrap()
    }
    fn adj(&mut self) -> Vec<C32> {
        let nv = self.op.nv();
        let mut xa = self.st.alloc_zeros::<C32>(self.nimg * nv).unwrap();
        self.op.adjoint(&self.st, &self.seg, &self.pts, &self.yd, &mut xa).unwrap();
        self.st.clone_dtoh(&xa).unwrap()
    }
    /// autotune; returns the candidates timed
    fn tune(&mut self) -> u64 {
        let n0 = tune_timing_stats().0;
        self.op.autotune(&self.st, &self.seg, &self.pts).unwrap();
        tune_timing_stats().0 - n0
    }
}

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(x, y)| ((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2)).sum();
    let d: f64 = b.iter().map(|y| (y.re as f64).powi(2) + (y.im as f64).powi(2)).sum();
    (n / d.max(1e-30)).sqrt()
}

fn bits(v: &[C32]) -> Vec<(u32, u32)> {
    v.iter().map(|c| (c.re.to_bits(), c.im.to_bits())).collect()
}

fn json_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                out.extend(json_files(&p));
            } else if p.extension().is_some_and(|x| x == "json") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

#[test]
fn tune_cache_reuse() {
    // a private cache directory inside the repository (rustlib/target), removed at the end
    let dir = hasty_site::target().join(format!("tune-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::env::set_var("HASTY_TUNE_CACHE", &dir);
    std::env::set_var("HASTY_TUNE", "reuse");
    // a smaller candidate set (part of the key, constant over the test): tile register blocking and hints fixed
    std::env::set_var("HASTY_SEG_FSB", "2");
    std::env::set_var("HASTY_SEG_ACB", "1");
    std::env::set_var("HASTY_SEG_LBMIN", "0");

    // 1. miss: timed and stored
    let mut a = build(10, 10, 5);
    let n1 = a.tune();
    let files = json_files(&dir);
    eprintln!("1. miss: {n1} candidates timed, files {files:?}\n   choice {}", a.op.tune_choice_desc());
    assert!(n1 > 0, "a miss times candidates");
    assert_eq!(files.len(), 1, "one tune file");
    let (ya, xa) = (a.fwd(), a.adj());
    let xa2 = a.adj();
    let adj_repeatable = bits(&xa) == bits(&xa2);
    let choice_a = a.op.tune_choice_desc();
    drop(a);

    // 2. from the file: no timing, same choice, bitwise output
    tune_cache_forget();
    let mut b = build(10, 10, 5);
    let n2 = b.tune();
    eprintln!("2. file hit: {n2} candidates timed\n   choice {}", b.op.tune_choice_desc());
    assert_eq!(n2, 0, "a hit times nothing");
    assert_eq!(b.op.tune_choice_desc(), choice_a, "same choices");
    let (yb, xb) = (b.fwd(), b.adj());
    assert!(bits(&ya) == bits(&yb), "forward output bitwise identical");
    if adj_repeatable {
        assert!(bits(&xa) == bits(&xb), "adjoint output bitwise identical");
    } else {
        let (own, d) = (rel(&xa2, &xa), rel(&xb, &xa));
        eprintln!("   adjoint not bitwise repeatable on one engine (f32 atomics order of split tiles): rel. difference {own:.2e} between two calls of the first engine, {d:.2e} between the engines");
        assert!(d <= 10.0 * own.max(1e-7), "adjoint differs beyond the engine's own repeatability");
    }
    drop(b);

    // 3. in-process copy
    let mut c = build(10, 10, 5);
    let n3 = c.tune();
    eprintln!("3. in-process hit: {n3} candidates timed");
    assert_eq!(n3, 0);
    assert!(bits(&ya) == bits(&c.fwd()));
    drop(c);

    // 4. another coil batch: a miss
    let mut d = build(10, 10, 4);
    let n4 = d.tune();
    eprintln!("4. coil batch 4: {n4} candidates timed, files {}", json_files(&dir).len());
    assert!(n4 > 0, "another shape misses");
    assert_eq!(json_files(&dir).len(), 2);
    drop(d);

    // 5. corrupted file: retimed, no panic, rewritten
    let f0 = files[0].clone();
    std::fs::write(&f0, b"{ \"format\": 1, \"key\": trunc").unwrap();
    tune_cache_forget();
    let mut e = build(10, 10, 5);
    let n5 = e.tune();
    eprintln!("5. corrupted file: {n5} candidates timed");
    assert!(n5 > 0, "a corrupted file is a miss");
    let txt = std::fs::read_to_string(&f0).unwrap();
    assert!(txt.contains("\"main\""), "the file is rewritten");

    // 6. retune times again; off neither reads nor writes
    std::env::set_var("HASTY_TUNE", "retune");
    let n6 = e.tune();
    assert!(n6 > 0, "retune times");
    std::env::set_var("HASTY_TUNE", "off");
    let before = json_files(&dir).len();
    let mt = std::fs::metadata(&f0).unwrap().modified().unwrap();
    let mut g = build(10, 10, 5);
    let n7 = g.tune();
    assert!(n7 > 0 && json_files(&dir).len() == before, "off: timed, nothing written");
    assert_eq!(std::fs::metadata(&f0).unwrap().modified().unwrap(), mt, "off: the existing file is not touched");
    eprintln!("6. retune {n6}, off {n7} candidates timed; files {before}");
    drop(g);
    drop(e);

    for k in ["HASTY_TUNE", "HASTY_TUNE_CACHE", "HASTY_SEG_FSB", "HASTY_SEG_ACB", "HASTY_SEG_LBMIN"] {
        std::env::remove_var(k);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
