//! Respiratory × cardiac basis test (research/temporal_operator.md §16): per encode, a CGLS reconstruction of the
//! knot images of a 2-D basis (`SegNufft::new_temporal2d`): respiratory bins (degree 0) or linear B-splines (degree 1)
//! in the breathing coordinate × periodic B-splines in cardiac time, sample-level weights, B0 time segments, the true
//! coil maps (strongest --coils), no regulariser (the arms differ only in the temporal basis). The knot images are
//! evaluated at given breathing states r* and the 20 cardiac frame centres and written at the truth voxels of each
//! state (frame_fit.py lumen form, FRAME_FIT_TRUTH=<truth npz>).
//!
//!   splinerecon <scan dir> --rknots K --rdeg 0|1 [--cknots 10] [--cdeg 3] [--iters 20] [--save-at 5,10,20]
//!       [--coils 8] [--nseg 10] [--width 8] [--resp truth|bellows] --eval r1:idx1.npy,r2:idx2.npy,... --out <prefix>
//!
//! Breathing coordinate: the sim's true state (resp.npy, default) or the bellows amplitude (physio.npy column 3).
//! Bins: equal-count amplitude bins (as recon_p2np --resp-bins); r* evaluates the bin whose amplitude range holds it.
//! Linear: K knots at the amplitude quantiles 0, 1/(K−1), …, 1; r* evaluates the hat weights (clamped at the ends).
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts, SegPoints, SegSegments};
use hasty_recon::cg::pcgls_cb;
use hasty_recon::data::{read_npy, SimData};
use hasty_recon::gpu::Kernels;
use hasty_recon::setop::LinOp;
use hasty_recon::Error;

struct Op2d<'a> {
    eng: &'a std::cell::RefCell<SegNufft>,
    /// (forward, adjoint) passes so far: the gradient count of the solver arms
    cnt: &'a std::cell::Cell<(usize, usize)>,
    seg: &'a SegSegments,
    pts: &'a SegPoints,
    kern: Arc<Kernels>,
    nx: usize,
    ny: usize,
}

impl LinOp for Op2d<'_> {
    fn len_x(&self) -> usize {
        self.nx
    }
    fn len_y(&self) -> usize {
        self.ny
    }
    fn kernels(&self) -> &Kernels {
        &self.kern
    }
    fn forward(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let c = self.cnt.get();
        self.cnt.set((c.0 + 1, c.1));
        Ok(self.eng.borrow_mut().forward(st, self.seg, self.pts, x, y)?)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let c = self.cnt.get();
        self.cnt.set((c.0, c.1 + 1));
        Ok(self.eng.borrow_mut().adjoint(st, self.seg, self.pts, y, x)?)
    }
}

/// The static image s through the 2-D operator: A(1 ⊗ s) = A_static s (the basis weights sum to 1 per sample), adjoint
/// = the sum of the knot images of Aᴴ r. `big` is a knot-stack scratch buffer.
struct WarmOp<'a, 'b> {
    inner: &'a mut Op2d<'b>,
    big: &'a mut CudaSlice<C32>,
    tmp: CudaSlice<C32>,
    nv: usize,
}

impl LinOp for WarmOp<'_, '_> {
    fn len_x(&self) -> usize {
        self.nv
    }
    fn len_y(&self) -> usize {
        self.inner.ny
    }
    fn kernels(&self) -> &Kernels {
        &self.inner.kern
    }
    fn forward(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let nk = self.inner.nx / self.nv;
        for k in 0..nk {
            st.memcpy_dtod(x, &mut self.big.slice_mut(k * self.nv..(k + 1) * self.nv))?;
        }
        self.inner.forward(st, self.big, y)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let nk = self.inner.nx / self.nv;
        self.inner.adjoint(st, y, self.big)?;
        st.memset_zeros(x)?;
        let one = num_complex::Complex64::new(1.0, 0.0);
        for k in 0..nk {
            st.memcpy_dtod(&self.big.slice(k * self.nv..(k + 1) * self.nv), &mut self.tmp)?;
            self.inner.kern.axpby(st, one, &self.tmp, one, x)?;
        }
        Ok(())
    }
}

/// complex64 C-order output (lumen values): `hasty_value::store` image policy (1e-4 relative rms,
/// research/io_policy.md) → `<name>.hv` (`HASTY_IO=npy`: .npy)
fn write_c8_c(path: &std::path::Path, shape: &[usize], data: &[C32]) {
    // SAFETY: C32 is two f32, repr(C)
    let b = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 8) };
    hasty_value::store::save_bytes(path, hasty_value::DType::C64, shape, hasty_value::Order::C, b, &hasty_value::Save::image()).unwrap();
}

/// centred periodic B-spline weight of knot b at time t (knot units, period m), degree 1 or 3
fn bweight(t: f64, b: usize, m: usize, deg: u32) -> f64 {
    let n = m as f64;
    let mut d = (t - b as f64).rem_euclid(n);
    if d > 0.5 * n {
        d -= n;
    }
    let a = d.abs();
    match deg {
        // degree 0 as the kernels: sample time t in [b, b + 1) belongs to knot b
        0 => ((t.rem_euclid(n)).floor() as usize % m == b) as u8 as f64,
        1 => (1.0 - a).max(0.0),
        _ if a <= 1.0 => 2.0 / 3.0 - a * a + 0.5 * a * a * a,
        _ if a <= 2.0 => (2.0 - a).powi(3) / 6.0,
        _ => 0.0,
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let kr: usize = take("--rknots").expect("--rknots K").parse().unwrap();
    let rdeg: u32 = take("--rdeg").expect("--rdeg 0|1").parse().unwrap();
    let mc: usize = take("--cknots").map_or(10, |v| v.parse().unwrap());
    let cdeg: u32 = take("--cdeg").map_or(3, |v| v.parse().unwrap());
    let iters: usize = take("--iters").map_or(20, |v| v.parse().unwrap());
    let save_at: Vec<usize> = take("--save-at").map_or(vec![iters], |s| s.split(',').map(|v| v.parse().unwrap()).collect());
    let ncoil_sel: usize = take("--coils").map_or(8, |v| v.parse().unwrap());
    let nseg: usize = take("--nseg").map_or(10, |v| v.parse().unwrap());
    let width: usize = take("--width").map_or(8, |v| v.parse().unwrap());
    let sigma: f64 = take("--sigma").map_or(1.25, |v| v.parse().unwrap());
    let resp_src = take("--resp").unwrap_or_else(|| "truth".into());
    // --phys b0 | conc (default): conc = recon_p2np's physics: per-encode mean concomitant phase in the segments and
    // the per-sample residual (φ0, k-shift, ΔC, correction order 2; hasty_recon::segphys); --gnl: GNL-aware grid
    let phys = take("--phys").unwrap_or_else(|| "conc".into());
    assert!(phys == "b0" || phys == "conc", "--phys b0|conc");
    let conc = phys == "conc";
    // --resp-keep NR:B: only the readouts of breathing bin B of NR equal-count bins (e.g. timing one bin of a layout
    // whose key count exceeds the tables' 127)
    let resp_keep: Option<(usize, usize)> = take("--resp-keep").map(|v| { let (a, b) = v.split_once(':').unwrap(); (a.parse().unwrap(), b.parse().unwrap()) });
    // --warm N: start every knot from the encode's static image (N preconditioned CGLS iterations through the 2-D
    // operator; circulant preconditioner of the plain SENSE normal operator)
    let warm: usize = take("--warm").map_or(0, |v| v.parse().unwrap());
    // --heldout N: keep 1/N of the readouts out of the fit (hash-spread, all strata) and track their residual every
    // --ho-every iterations; the iterate with the lowest held-out residual is also written (_itbest). 0: off
    let ho_n: usize = take("--heldout").map_or(0, |v| v.parse().unwrap());
    let ho_every: usize = take("--ho-every").map_or(2, |v| v.parse().unwrap());
    // --precond knot | none: Kronecker preconditioner G⁻¹ ⊗ C⁻¹ (hasty_recon::knotsolve) for the knot CG
    let precond_knot = take("--precond").map_or(false, |v| v == "knot");
    // --keep N: keep a deterministic 1/N of the readouts (hash-spread over all strata: a shorter scan), before the
    // held-out split
    let keep_n: usize = take("--keep").map_or(1, |v| v.parse().unwrap());
    // --arms pcg,ncg,ncg0,zmshr1,fista,admm0.3,admm1,admm3: solver comparison (research/convergence_strategy.md
    // §8 P1–P2): every arm from the same warm start; the prior arms continue from the PCG iterate after --prefix
    // gradients, where the prior weights and the prior-aware metric are estimated (data-derived); each arm stops at
    // --budget gradients (A + Aᴴ pairs); outputs <out>_<arm>_r*_it<gradients>_lumen and a per-iteration CSV
    let arms: Vec<String> = take("--arms").map_or(vec![], |v| v.split(',').map(|s| s.to_string()).collect());
    let prefix: usize = take("--prefix").map_or(5, |v| v.parse().unwrap());
    let budget: f64 = take("--budget").map_or(41.0, |v| v.parse().unwrap());
    // priors: γ = gamma_rel / bulk; wavelet ℓ1 weight λ_w = bulk · wthr · σ̂_w (σ̂_w: MAD of the finest diagonal
    // subband of the prefix iterate); LLR λ_l = bulk · kappa · σ̂_l (σ̂_l: the LLR σ̂ rule on the prefix iterate)
    let gamma_rel: f64 = take("--gamma-rel").map_or(0.1, |v| v.parse().unwrap());
    let wthr: f64 = take("--wthr").map_or(1.0, |v| v.parse().unwrap());
    let kappa: f64 = take("--kappa").map_or(1.0, |v| v.parse().unwrap());
    let block: usize = take("--block").map_or(8, |v| v.parse().unwrap());
    let wlevels: usize = take("--wlevels").map_or(3, |v| v.parse().unwrap());
    let admm_inner: usize = take("--admm-inner").map_or(3, |v| v.parse().unwrap());
    let power_its: usize = take("--power").map_or(8, |v| v.parse().unwrap());
    let llr_budget: f64 = take("--llr-budget-gb").map_or(1.0, |v| v.parse().unwrap());
    let priors_sel = take("--priors").unwrap_or_else(|| "wav,llr".into());
    let encodes: Option<Vec<usize>> = take("--encodes").map(|s| s.split(',').map(|v| v.parse().unwrap()).collect());
    let evals: Vec<(f64, PathBuf)> = take("--eval").expect("--eval r:idx.npy,...").split(',').map(|p| { let (r, f) = p.split_once(':').unwrap(); (r.parse().unwrap(), PathBuf::from(f)) }).collect();
    let out = take("--out").expect("--out prefix");
    // --estimate-motion NR:SCALE [--roi x0:x1,y0:y1,z0:z1] [--truth-resp resp_motion.hv]: data-driven affine
    // respiratory motion (no labels), written to <out>_affine.json, then exit (see estimate_motion)
    let est: Option<(usize, f64)> = take("--estimate-motion").map(|v| { let (a, b) = v.split_once(':').unwrap(); (a.parse().unwrap(), b.parse().unwrap()) });
    let roi: [(f64, f64); 3] = take("--roi").map_or([(0.0, 1.0); 3], |v| { let p: Vec<(f64, f64)> = v.split(',').map(|r| { let (a, b) = r.split_once(':').unwrap(); (a.parse().unwrap(), b.parse().unwrap()) }).collect(); [p[0], p[1], p[2]] });
    let truth_resp: Option<PathBuf> = take("--truth-resp").map(PathBuf::from);
    if let Some((nr, scale)) = est {
        estimate_motion(&PathBuf::from(&args[0]), nr, scale, roi, truth_resp.as_deref(), &out);
        return;
    }
    // --time-only: one forward + adjoint of encode 0 (no solver state; for bases whose CG state does not fit)
    let time_only = args.iter().position(|a| a == "--time-only").map(|i| { args.remove(i); true }).unwrap_or(false);
    let dir = PathBuf::from(&args[0]);
    let t0 = Instant::now();
    let sd = SimData::load_lazy(&dir).unwrap();
    let (dims, nv, nc_all) = (sd.dims, sd.nv(), sd.nc);
    let ne = sd.encode_m1.len();
    let nro = sd.n_readouts();
    // breathing coordinate per readout and the respiratory basis
    let amp: Vec<f64> = {
        let (file, col) = match resp_src.as_str() { "truth" => ("resp.npy", 0), "bellows" => ("physio.npy", 3), o => panic!("--resp truth|bellows, got {o}") };
        let a = read_npy(&dir.join(file)).unwrap();
        let (w, v) = (a.shape[1], a.f64s());
        assert_eq!(a.shape[0], nro);
        (0..nro).map(|i| v[i * w + col]).collect()
    };
    let nri = if rdeg == 0 { kr } else { kr - 1 };
    let (ri_ro, rv_ro, knots_desc): (Vec<u32>, Vec<f32>, String);
    let resp_weights: Box<dyn Fn(f64) -> Vec<f64>>;
    if rdeg == 0 {
        let bins = hasty_recon::selfgate::amplitude_bins(&amp, kr);
        let mut lo = vec![f64::INFINITY; kr];
        let mut hi = vec![f64::NEG_INFINITY; kr];
        for i in 0..nro {
            lo[bins[i]] = lo[bins[i]].min(amp[i]);
            hi[bins[i]] = hi[bins[i]].max(amp[i]);
        }
        knots_desc = format!("bins {:?}", (0..kr).map(|b| format!("[{:.3}, {:.3}]", lo[b], hi[b])).collect::<Vec<_>>());
        ri_ro = bins.iter().map(|&b| b as u32).collect();
        rv_ro = vec![0.0; nro];
        let (lo2, hi2) = (lo.clone(), hi.clone());
        resp_weights = Box::new(move |r: f64| {
            let b = (0..kr).find(|&b| r >= lo2[b] && r <= hi2[b]).unwrap_or_else(|| (0..kr).min_by(|&a, &b| (0.5 * (lo2[a] + hi2[a]) - r).abs().partial_cmp(&(0.5 * (lo2[b] + hi2[b]) - r).abs()).unwrap()).unwrap());
            (0..kr).map(|k| (k == b) as u8 as f64).collect()
        });
    } else {
        let mut s = amp.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q: Vec<f64> = (0..kr).map(|k| s[((k as f64 / (kr - 1) as f64) * (nro - 1) as f64).round() as usize]).collect();
        knots_desc = format!("linear knots at {:?}", q.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>());
        let locate = {
            let q = q.clone();
            move |a: f64| -> (usize, f64) {
                let i = (0..kr - 1).rev().find(|&i| a >= q[i]).unwrap_or(0);
                let v = ((a - q[i]) / (q[i + 1] - q[i]).max(1e-12)).clamp(0.0, 1.0);
                (i, v)
            }
        };
        let iv: Vec<(usize, f64)> = amp.iter().map(|&a| locate(a)).collect();
        ri_ro = iv.iter().map(|p| p.0 as u32).collect();
        rv_ro = iv.iter().map(|p| p.1 as f32).collect();
        resp_weights = Box::new(move |r: f64| {
            let (i, v) = locate(r);
            (0..kr).map(|k| if k == i { 1.0 - v } else if k == i + 1 { v } else { 0.0 }).collect()
        });
    }
    let gnl = args.iter().position(|a| a == "--gnl").map(|i| { args.remove(i); true }).unwrap_or(false);
    let w = sd.whitener();
    let grid = sd.physics_grid(gnl, &w);
    // coils: the strongest by sensitivity energy in the lumen (as framerecon)
    let lum = read_npy(&dir.join("truth_lumen_fraction.npy")).unwrap().f32s();
    let lq: Vec<usize> = (0..nv).filter(|&q| lum[q] > 0.5).collect();
    let en: Vec<f64> = (0..nc_all).map(|c| lq.iter().map(|&q| { let z = grid.smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum()).collect();
    let mut order: Vec<usize> = (0..nc_all).collect();
    order.sort_by(|&a, &b| en[b].partial_cmp(&en[a]).unwrap());
    let nc = ncoil_sel.min(nc_all);
    let keep = order[..nc].to_vec();
    let smaps: Vec<C32> = keep.iter().flat_map(|&c| grid.smaps[c * nv..(c + 1) * nv].iter().cloned()).collect();
    // evaluation states: the truth voxels of each, the respiratory weights, the cardiac weights of the 20 frame centres
    let nfo = sd.n_frames;
    let ev: Vec<(f64, Vec<usize>, Vec<f64>)> = evals.iter().map(|(r, p)| (*r, read_npy(p).unwrap().f32s().iter().map(|&v| v as usize).collect(), resp_weights(*r))).collect();
    let cw: Vec<Vec<f64>> = (0..nfo).map(|f| { let t = ((f as f64 + 0.5) / nfo as f64) * mc as f64 - 0.5; (0..mc).map(|b| bweight(t, b, mc, cdeg)).collect() }).collect();
    println!(
        "{}: grid {dims:?}, {ne} encodes, coils {nc} of {nc_all}, respiratory {} ({resp_src}): {knots_desc}; cardiac {mc} knots degree {cdeg}; {nseg} B0 segments, W {width}, {iters} CGLS iterations ({:.1}s)",
        dir.display(),
        if rdeg == 0 { format!("{kr} bins") } else { format!("{kr} linear knots") },
        t0.elapsed().as_secs_f64()
    );
    for (r, idx, g) in &ev {
        println!("  eval r* = {r}: {} voxels, respiratory weights {:?}", idx.len(), g.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    }
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    // device memory ledger (research/memory_ledger.md), planned before any heavy work: CGLS image vectors (x, s, p +
    // one spare), data vectors (y, r, q), sample tables (records 48 B + ≈ 1.25 × W³/64 tile entries of 4 B per sample),
    // coil maps and segment diagonals, the grid slot; the grid budget shrinks to what is left, else stop here
    let gib = (1u64 << 30) as f64;
    let keep_mask = hasty_recon::knotsolve::subset_mask(nro, keep_n, 0x4B45_4550);
    let m_max = (0..ne).map(|e| sd.readouts_of(e, None).iter().filter(|&&r| keep_mask[r]).count()).max().unwrap() * sd.na / resp_keep.map_or(1, |k| k.0) * if ho_n > 0 { ho_n - 1 } else { 1 } / ho_n.max(1);
    // solver image stacks: CGLS x, s, p (+ z with a preconditioner); arms: the largest arm (ZMSHR1 6, NCG / FISTA /
    // ADMM 5, PCG 4)
    let arm_stacks = arms.iter().map(|a| if a == "zmshr1" { 6.0 } else if a == "pcg" { 4.0 } else { 5.0 }).fold(0.0f64, f64::max);
    let has_prior_arms = arms.iter().any(|a| a != "pcg");
    let tiles_per_sample = 1.25 * ((width as f64 + 3.0) / 4.0).powi(3);
    let led: Vec<(&str, f64)> = vec![
        ("solver image vectors (CGLS x, s, p; arms: the largest arm; time-only 2)", if time_only { 2.0 } else if arms.is_empty() { 3.0 } else { arm_stacks - 1.0 } * (kr * mc * nv) as f64 * 8.0),
        ("CGLS data vectors (r, q: y becomes r; time-only 2)", 2.0 * (nc * m_max) as f64 * 8.0),
        ("sample tables", m_max as f64 * (48.0 + 4.0 * tiles_per_sample)),
        ("coil maps + segment diagonals", ((nc + nseg) * nv) as f64 * 8.0),
        ("knot preconditioner (CG z vector + 2 images + circulant)", if precond_knot { ((kr * mc + 2) * nv * 8) as f64 + 0.5 * gib } else { 0.0 }),
        ("prior-aware metric (3 nv f32 + plans) + LLR on GPU (batch budget + cuBLAS / cuSOLVER handles)", if has_prior_arms { hasty_recon::knotncg::PriorMetric::bytes(kr * mc, nv) as f64 + llr_budget * gib + 0.5 * gib } else { 0.0 }),
        ("FFT / kernel workspace, margin", 1.5 * gib),
    ];
    let (free, total) = cudarc::driver::result::mem_get_info().unwrap();
    let fixed: f64 = led.iter().map(|l| l.1).sum();
    let grid_cap = std::env::var("SPLINE_GRID_GB").ok().map_or(3.0, |v| v.parse::<f64>().unwrap()) * gib;
    let grid_budget = (free as f64 * 0.97 - fixed).min(grid_cap);
    println!("  device ledger ({:.1} of {:.1} GiB free):", free as f64 / gib, total as f64 / gib);
    for (k, v) in &led {
        println!("    {k}: {:.2} GiB", v / gib);
    }
    println!("    grid slot budget: {:.2} GiB", grid_budget / gib);
    if grid_budget < 0.4 * gib {
        eprintln!("splinerecon: the plan does not fit the device ({:.2} GiB fixed, {:.2} GiB free): fewer coils / knots or a host-resident solver", fixed / gib, free as f64 / gib);
        std::process::exit(2);
    }
    let opts = SegNufftOpts { sigma, width, nseg, order: if conc { 2 } else { 0 }, frames: mc, bspline: cdeg, max_grid_bytes: grid_budget as usize };
    let mut eng = SegNufft::new_temporal2d(&dev, dims, sd.vox, nc, opts, kr, rdeg).unwrap();
    eng.set_maps(&st, &smaps).unwrap();
    let eng = std::cell::RefCell::new(eng);
    let nx = kr * mc * nv;
    let ho_mask: Vec<bool> = if ho_n > 0 { hasty_recon::knotsolve::heldout_mask(nro, ho_n) } else { vec![false; nro] };
    let mut vals: Vec<Vec<Vec<C32>>> = (0..=save_at.len()).map(|_| ev.iter().map(|(_, idx, _)| vec![C32::new(0.0, 0.0); idx.len() * ne * nfo]).collect()).collect();
    let mut seg: Option<SegSegments> = None;
    let mut wbar_e: Option<Vec<[f64; 4]>> = None;
    let mut monofit: Option<hasty_recon::segphys::MonoFit> = None;
    let enc_list: Vec<usize> = encodes.unwrap_or_else(|| (0..ne).collect());
    let cnt = std::cell::Cell::new((0usize, 0usize));
    // (arm, label) → per encode the compact lumen values per evaluation state
    let mut arm_vals: std::collections::BTreeMap<(String, String), Vec<(usize, Vec<Vec<C32>>)>> = Default::default();
    let mut timing = vec![];
    for &e in &enc_list {
        let te = Instant::now();
        let mut ros = sd.readouts_of(e, None);
        ros.retain(|&r| keep_mask[r]);
        if let Some((nr, b)) = resp_keep {
            let bins = hasty_recon::selfgate::amplitude_bins(&amp, nr);
            ros.retain(|&r| bins[r] == b);
        }
        let set = sd.load_set(&ros, &hasty_recon::kload::LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
        let m = set.tau.len();
        let na = m / ros.len();
        assert_eq!(na * ros.len(), m, "fixed ADC count per readout expected");
        if conc || seg.is_none() {
            // segments over the readout window: ψ(v, τ) = ω_v τ (+ the encode's mean concomitant phase), fitted on object voxels
            let (tmin, tmax) = set.tau.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &t| (a.min(t), b.max(t)));
            let emax = (0..nv).map(|q| (0..nc).map(|c| { let z = smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum::<f64>()).fold(0.0, f64::max);
            let fit: Vec<usize> = (0..nv).filter(|&q| (0..nc).map(|c| { let z = smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum::<f64>() > 1e-3 * emax).step_by(13).collect();
            let tau_adc: Vec<f64> = set.tau[..na].to_vec();
            let wb = if conc { hasty_recon::segphys::mean_coefficients(&[&set], na) } else { vec![[0.0; 4]; na] };
            let (sp, at, s0, ds) = {
                let psi = hasty_recon::segphys::segment_phase(&grid, &wb, &tau_adc, true);
                build_segments(nv, nseg, tmin, tmax, 128, &fit, &psi)
            };
            wbar_e = Some(wb);
            if conc && monofit.is_none() {
                monofit = Some(hasty_recon::segphys::fit_monomials(&sd, &grid, &fit));
            }
            seg = Some(eng.borrow().segments(&st, &sp, &at, s0, ds).unwrap());
            println!("  segments: {nseg} over τ ∈ [{:.2}, {:.2}] ms, {} fit voxels", tmin * 1e3, tmax * 1e3, fit.len());
        }
        let seg = seg.as_ref().unwrap();
        let mut t = vec![0f32; m];
        let mut ri = vec![0u32; m];
        let mut rv = vec![0f32; m];
        for (r, &ro) in ros.iter().enumerate() {
            let card = &sd.ro.card[ro];
            for n in 0..na {
                let j = r * na + n;
                let phase = card.phase + set.tau[j] / card.rr;
                t[j] = (phase * mc as f64 - 0.5).rem_euclid(mc as f64) as f32;
                ri[j] = ri_ro[ro];
                rv[j] = rv_ro[ro];
            }
        }
        assert!(ri.iter().all(|&i| (i as usize) < nri));
        // per-sample physics: the residual concomitant terms against the encode's mean (as recon_p2np), or none
        let (coords, tau, phi0, dcv) = if conc {
            let tab = hasty_recon::segphys::seg_tables(&sd, &set, monofit.as_ref().unwrap(), true, wbar_e.as_deref());
            (tab.coords, tab.tau, tab.phi0, tab.dc)
        } else {
            (set.coords.clone(), set.tau.iter().map(|&v| v as f32).collect(), vec![0f32; m], vec![])
        };
        let mut yh = vec![C32::new(0.0, 0.0); nc * m];
        for (ci, &c) in keep.iter().enumerate() {
            yh[ci * m..(ci + 1) * m].copy_from_slice(&set.y[c * m..(c + 1) * m]);
        }
        drop(set);
        // held-out split (whole readouts): fit = the rest
        let (coords, tau, phi0, dcv, t, ri, rv, yh, ho) = {
            let hold: Vec<bool> = (0..m).map(|j| ho_mask[ros[j / na]]).collect();
            let (fi, hi): (Vec<usize>, Vec<usize>) = (0..m).partition(|&j| !hold[j]);
            let g32 = |v: &[f32], ix: &[usize]| -> Vec<f32> { ix.iter().map(|&j| v[j]).collect() };
            let gdc = |ix: &[usize]| -> Vec<f32> { if dcv.is_empty() { vec![] } else { (0..6).flat_map(|q| ix.iter().map(move |&j| q * m + j)).map(|k| dcv[k]).collect() } };
            let gy = |ix: &[usize]| -> Vec<C32> { (0..nc).flat_map(|c| ix.iter().map(move |&j| c * m + j)).map(|k| yh[k]).collect() };
            let part = |ix: &[usize]| ([g32(&coords[0], ix), g32(&coords[1], ix), g32(&coords[2], ix)], g32(&tau, ix), g32(&phi0, ix), gdc(ix), g32(&t, ix), ix.iter().map(|&j| ri[j]).collect::<Vec<u32>>(), g32(&rv, ix), gy(ix));
            if hi.is_empty() {
                (coords, tau, phi0, dcv, t, ri, rv, yh, None)
            } else {
                let f = part(&fi);
                let h = part(&hi);
                (f.0, f.1, f.2, f.3, f.4, f.5, f.6, f.7, Some(h))
            }
        };
        let m = tau.len();
        let mut pre = if warm > 0 { Some(hasty_recon::precond::CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&coords[0], &coords[1], &coords[2]], &smaps, nc, 1e-3).unwrap()) } else { None };
        let pts = eng.borrow_mut().points2d(&st, [&coords[0], &coords[1], &coords[2]], &tau, &phi0, &dcv, &t, &ri, &rv).unwrap();
        let pts_ho = ho.as_ref().map(|h| eng.borrow_mut().points2d(&st, [&h.0[0], &h.0[1], &h.0[2]], &h.1, &h.2, &h.3, &h.4, &h.5, &h.6).unwrap());
        let y_ho = ho.as_ref().map(|h| st.clone_htod(&h.7).unwrap());
        let m_ho = ho.as_ref().map_or(0, |h| h.1.len());
        drop(ho);
        // knot preconditioner: temporal Gram of the basis under the fit samples' times (every 7th sample)
        let mut kgram: Option<(Vec<f64>, Vec<f32>, Vec<f32>)> = None;
        let mut kpre = if precond_knot {
            let kk = kr * mc;
            let wts = (0..m).step_by(7).map(|j| {
                let rw: Vec<(usize, f64)> = if rdeg == 0 { vec![(ri[j] as usize, 1.0)] } else { vec![(ri[j] as usize, 1.0 - rv[j] as f64), (ri[j] as usize + 1, rv[j] as f64)] };
                let i0 = (t[j] as f64).floor() as i64;
                let cwj: Vec<(usize, f64)> = (i0 - 2..=i0 + 2).map(|b| b.rem_euclid(mc as i64) as usize).map(|b| (b, bweight(t[j] as f64, b, mc, cdeg))).filter(|p| p.1 != 0.0).collect();
                rw.iter().flat_map(|&(k, a)| cwj.iter().map(move |&(b, c)| (k * mc + b, a * c))).collect::<Vec<_>>()
            });
            let g = hasty_recon::knotsolve::temporal_gram(kk, wts);
            println!("  knot preconditioner: temporal Gram {kk}×{kk}, condition {:.1}", hasty_recon::knotsolve::gram_condition(&g, kk));
            let gi = hasty_recon::knotsolve::gram_inverse(&g, kk, 1e-3);
            let circ = hasty_recon::precond::CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&coords[0], &coords[1], &coords[2]], &smaps, nc, 1e-3).unwrap();
            if !arms.is_empty() {
                let (lam, dinv) = circ.factors(&st).unwrap();
                kgram = Some((g.clone(), lam, dinv));
            }
            Some(hasty_recon::knotsolve::KnotPrecond::new(&dev, &st, kk, nv, &gi, Some(circ)).unwrap())
        } else {
            None
        };
        drop((coords, tau, phi0, dcv, t, ri, rv));
        if timing.is_empty() && std::env::var("HASTY_SPLINE_NOTUNE").map_or(true, |v| v == "0") {
            let tt = Instant::now();
            let (kf, ka) = eng.borrow_mut().autotune(&st, seg, &pts).unwrap();
            println!("  autotune {:.1}s: tile kernels {kf:.1} / {ka:.1} ms (knot 0, one coil batch of {}; fwd {:?}, adj {:?})", tt.elapsed().as_secs_f64(), eng.borrow().cb, eng.borrow().tiles_fwd, eng.borrow().tiles_adj);
        }
        let y = st.clone_htod(&yh).unwrap();
        let yh_keep = if arms.is_empty() { drop(yh); None } else { Some(yh) };
        if time_only {
            let xr: Vec<C32> = (0..nx).map(|i| C32::new(((i * 7919) % 1000) as f32 * 1e-3, ((i * 104729) % 997) as f32 * 1e-3)).collect();
            let xd = st.clone_htod(&xr).unwrap();
            drop(xr);
            let mut yy = st.alloc_zeros::<C32>(nc * m).unwrap();
            let mut xa = st.alloc_zeros::<C32>(nx).unwrap();
            let mut tf = vec![];
            for rep in 0..3 {
                st.synchronize().unwrap();
                let a = Instant::now();
                eng.borrow_mut().forward(&st, seg, &pts, &xd, &mut yy).unwrap();
                st.synchronize().unwrap();
                let b = Instant::now();
                eng.borrow_mut().adjoint(&st, seg, &pts, &y, &mut xa).unwrap();
                st.synchronize().unwrap();
                if rep > 0 {
                    tf.push((b.duration_since(a).as_secs_f64(), b.elapsed().as_secs_f64()));
                }
            }
            let (f, a) = tf.iter().fold((0.0, 0.0), |s, t| (s.0 + t.0 / tf.len() as f64, s.1 + t.1 / tf.len() as f64));
            println!("  time-only encode {e}: forward {f:.2}s + adjoint {a:.2}s = {:.2}s per gradient pass ({} knots, {m} samples, coil batch {})", f + a, kr * mc, eng.borrow().cb);
            return;
        }
        let mut x = st.alloc_zeros::<C32>(nx).unwrap();
        if let Some(pr) = pre.as_mut() {
            let tw = Instant::now();
            let mut sx = st.alloc_zeros::<C32>(nv).unwrap();
            {
                let mut op = Op2d { eng: &eng, cnt: &cnt, seg, pts: &pts, kern: kern.clone(), nx, ny: nc * m };
                let mut wop = WarmOp { inner: &mut op, big: &mut x, tmp: st.alloc_zeros::<C32>(nv).unwrap(), nv };
                pcgls_cb(&mut wop, Some(pr), &st, &y, &mut sx, 0.0, warm, &mut |_, _| {}).unwrap();
            }
            for k in 0..kr * mc {
                st.memcpy_dtod(&sx, &mut x.slice_mut(k * nv..(k + 1) * nv)).unwrap();
            }
            st.synchronize().unwrap();
            println!("  encode {e}: warm start, {warm} preconditioned CGLS iterations on the static image ({:.0}s)", tw.elapsed().as_secs_f64());
        }
        drop(pre);
        if !arms.is_empty() {
            assert!(precond_knot, "--arms needs --precond knot");
            let (gram, lam_eff, dinv) = kgram.take().unwrap();
            let evf = |h: &[C32]| lumen_compact(&ev, &cw, kr, mc, nv, nfo, h);
            let ac = ArmsCtx { eval: &evf, dev: &dev, st: &st, kern: &kern, eng: &eng, cnt: &cnt, seg, pts: &pts, pts_ho: pts_ho.as_ref(), y_ho: y_ho.as_ref(), m_ho, nc, m, nx, nv, kk: kr * mc, dims, out: &out, e };
            let opts = ArmsOpts { arms: arms.clone(), prefix, budget, gamma_rel, wthr, kappa, block, wlevels, admm_inner, power_its, llr_budget, priors: priors_sel.clone(), save_at: save_at.clone() };
            let res = run_arms(&ac, &opts, kpre.take().unwrap(), gram, lam_eff, dinv, x, y, yh_keep.unwrap());
            for (name, snaps) in res {
                for (lab, h) in snaps {
                    arm_vals.entry((name.clone(), lab)).or_default().push((e, h));
                }
            }
            println!("  encode {e}: arms done ({:.0}s)", te.elapsed().as_secs_f64());
            timing.push(0.0);
            continue;
        }
        let tl = Instant::now();
        let mut snaps: Vec<(usize, Vec<C32>)> = vec![];
        let mut op = Op2d { eng: &eng, cnt: &cnt, seg, pts: &pts, kern: kern.clone(), nx, ny: nc * m };
        let ho_norm = y_ho.as_ref().map_or(1.0, |v| kern.dot(&st, v, v).unwrap().re.max(1e-300));
        let mut ho_pred = (m_ho > 0).then(|| st.alloc_zeros::<C32>(nc * m_ho).unwrap());
        let mut ho_hist: Vec<(usize, f64)> = vec![];
        let mut best: Option<(usize, f64, Vec<C32>)> = None;
        let hist = hasty_recon::cg::pcgls_dyn(&mut op, kpre.as_mut().map(|p| p as &mut dyn hasty_recon::cg::Precond), &st, hasty_recon::cg::Data::Owned(y), &mut x, 0.0, iters, &mut |k, xd| {
            if save_at.contains(&k) {
                snaps.push((k, st.clone_dtoh(xd).unwrap()));
            }
            if let (Some(ph), Some(yv), Some(pp)) = (pts_ho.as_ref(), y_ho.as_ref(), ho_pred.as_mut()) {
                if k % ho_every == 0 || k == iters {
                    eng.borrow_mut().forward(&st, seg, ph, xd, pp).unwrap();
                    kern.axpby(&st, num_complex::Complex64::new(-1.0, 0.0), yv, num_complex::Complex64::new(1.0, 0.0), pp).unwrap();
                    let r = kern.dot(&st, pp, pp).unwrap().re / ho_norm;
                    ho_hist.push((k, r));
                    if best.as_ref().is_none_or(|b| r < b.1) {
                        best = Some((k, r, st.clone_dtoh(xd).unwrap()));
                    }
                }
            }
        }, None)
        .unwrap()
        .0;
        if let Some((kb, rb, h)) = best.take() {
            println!("  encode {e}: held-out residual (rel. energy) {}; best at iteration {kb} ({rb:.4})", ho_hist.iter().map(|(k, r)| format!("{k}:{r:.4}")).collect::<Vec<_>>().join(" "));
            snaps.push((usize::MAX, h));
        }
        st.synchronize().unwrap();
        let tsolve = tl.elapsed().as_secs_f64();
        for (k, h) in snaps {
            let si = if k == usize::MAX { save_at.len() } else { save_at.iter().position(|&a| a == k).unwrap() };
            fill_vals(&ev, &cw, kr, mc, nv, ne, nfo, e, &h, &mut vals[si]);
        }
        let h0 = hist.first().copied().unwrap_or(1.0).max(1e-300);
        println!(
            "  encode {e}: {} readouts, {m} samples; {iters} CGLS iterations {tsolve:.1}s ({:.2}s / it); residual rel {:.3e} → {:.3e} ({:.0}s)",
            ros.len(),
            tsolve / iters as f64,
            hist.first().copied().unwrap_or(0.0) / h0,
            hist.last().copied().unwrap_or(0.0) / h0,
            te.elapsed().as_secs_f64()
        );
        timing.push(tsolve / iters as f64);
    }
    for (si, k) in save_at.iter().map(|k| k.to_string()).chain((ho_n > 0).then(|| "best".to_string())).enumerate() {
        for (s, (r, idx, _)) in ev.iter().enumerate() {
            let p = PathBuf::from(format!("{out}_r{r}_it{k}_lumen.npy"));
            write_c8_c(&p, &[idx.len(), ne, nfo], &vals[si][s]);
        }
    }
    for ((name, k), per_e) in &arm_vals {
        for (s, (r, idx, _)) in ev.iter().enumerate() {
            let mut full = vec![C32::new(0.0, 0.0); idx.len() * ne * nfo];
            for (e, slot) in per_e {
                for i in 0..idx.len() {
                    full[(i * ne + e) * nfo..(i * ne + e + 1) * nfo].copy_from_slice(&slot[s][i * nfo..(i + 1) * nfo]);
                }
            }
            let p = PathBuf::from(format!("{out}_{name}_r{r}_it{k}_lumen.npy"));
            write_c8_c(&p, &[idx.len(), ne, nfo], &full);
        }
    }
    if !arms.is_empty() {
        println!("done ({:.0}s): arms {:?}, outputs {out}_<arm>_r*_it*_lumen.hv, per-iteration {out}_arms_e*.csv", t0.elapsed().as_secs_f64(), arms);
        return;
    }
    let mean_it: f64 = timing.iter().sum::<f64>() / timing.len().max(1) as f64;
    println!("done ({:.0}s): mean {:.2}s per CGLS iteration per encode ({} knots per encode); outputs {out}_r*_it*_lumen.hv", t0.elapsed().as_secs_f64(), mean_it, kr * mc);
}


/// Data-driven affine respiratory motion, no labels or segmentation:
/// 1. encode 0, equal-count bins of the bellows amplitude (the real-scan surrogate), plain-SENSE preconditioned CGLS
///    static images on a coarser grid (all cardiac phases pooled);
/// 2. one smooth Eulerian field S shared by all bins, I_b(x) ≈ I_0(x − r_b S(x)) (`motion::fit_shared`; the support
///    mask = signal above 10 % of the 99th percentile of bin 0);
/// 3. a robust affine fit S(q) ≈ B (q − c) + t over the prescribed box (`--roi`, fractions of the FOV: the scanner's
///    prescription, not anatomy) ∩ support, weighted by magnitude and refitted with Cauchy weights on the residual
///    (IRLS, 5 rounds), so the dominant coherent motion in the box wins;
/// 4. writes A₁ (per unit bellows, physical), t₁ (m per unit bellows) relative to bin 0's surrogate, the field's affine
///    residual, and (optional `--truth-resp`, oracle check only) the phantom's displacement at the box centre.
fn estimate_motion(dir: &std::path::Path, nr: usize, scale: f64, roi: [(f64, f64); 3], truth_resp: Option<&std::path::Path>, out: &str) {
    use hasty_recon::op::SenseOp;
    use hasty_recon::precond::CirculantPrecond;
    let t0 = Instant::now();
    let mut sd = SimData::load_lazy(dir).unwrap();
    sd.set_recon_grid([scale; 3]);
    let (dims, nv, nc_all, vox) = (sd.dims, sd.nv(), sd.nc, sd.vox);
    let nro = sd.n_readouts();
    let bel: Vec<f64> = { let a = read_npy(&dir.join("physio.npy")).unwrap(); let (w, v) = (a.shape[1], a.f64s()); (0..nro).map(|i| v[i * w + 3]).collect() };
    let bins = hasty_recon::selfgate::amplitude_bins(&bel, nr);
    let w = sd.whitener();
    let grid = sd.physics_grid(false, &w);
    let en: Vec<f64> = (0..nc_all).map(|c| (0..nv).map(|q| { let z = grid.smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum()).collect();
    let mut order: Vec<usize> = (0..nc_all).collect();
    order.sort_by(|&a, &b| en[b].partial_cmp(&en[a]).unwrap());
    let keep: Vec<usize> = order[..12.min(nc_all)].to_vec();
    let nc = keep.len();
    let smaps: Vec<C32> = keep.iter().flat_map(|&c| grid.smaps[c * nv..(c + 1) * nv].iter().cloned()).collect();
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let mut ims: Vec<Vec<f32>> = vec![];
    let mut rb = vec![];
    for b in 0..nr {
        let ros: Vec<usize> = sd.readouts_of(0, None).into_iter().filter(|&r| bins[r] == b).collect();
        rb.push(ros.iter().map(|&r| bel[r]).sum::<f64>() / ros.len() as f64);
        let set = sd.load_set(&ros, &hasty_recon::kload::LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
        let m = set.tau.len();
        let mut yh = vec![C32::new(0.0, 0.0); nc * m];
        for (ci, &c) in keep.iter().enumerate() {
            yh[ci * m..(ci + 1) * m].copy_from_slice(&set.y[c * m..(c + 1) * m]);
        }
        let mut op = SenseOp::new(&dev, &st, kern.clone(), dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &smaps, nc, vec![], 1e-4).unwrap();
        let mut pre = CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &smaps, nc, 1e-3).unwrap();
        drop(set);
        let y = st.clone_htod(&yh).unwrap();
        let mut x = st.alloc_zeros::<C32>(nv).unwrap();
        pcgls_cb(&mut op, Some(&mut pre), &st, &y, &mut x, 0.0, 10, &mut |_, _| {}).unwrap();
        let h = st.clone_dtoh(&x).unwrap();
        let mag: Vec<f32> = h.iter().map(|z| (z.re * z.re + z.im * z.im).sqrt()).collect();
        let mean = mag.iter().map(|&v| v as f64).sum::<f64>() / nv as f64;
        ims.push(mag.iter().map(|&v| (v as f64 / mean) as f32).collect());
        println!("  bin {b}: {} readouts, mean bellows {:.3} ({:.0}s)", ros.len(), rb[b], t0.elapsed().as_secs_f64());
    }
    let r_rel: Vec<f64> = rb.iter().map(|r| r - rb[0]).collect();
    let mut srt: Vec<f32> = ims[0].clone();
    srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p99 = srt[(0.99 * (nv - 1) as f64) as usize];
    let mask: Vec<bool> = ims[0].iter().map(|&v| v > 0.1 * p99).collect();
    let fit = hasty_recon::motion::fit_shared(&ims, &r_rel, dims, &mask, 8.0, 2, 1e-3, 60);
    let sfield = fit.field.dense();
    println!("  shared field: relative residual {:.3} (1 = no motion model), min Jacobian {:.3}", fit.rel_resid, fit.min_jac);
    // robust affine fit of S over the box ∩ support (voxel units, positions relative to the grid centre)
    let inbox = |q: usize| -> bool {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        let f = [i as f64 / dims[0] as f64, j as f64 / dims[1] as f64, k as f64 / dims[2] as f64];
        (0..3).all(|a| f[a] >= roi[a].0 && f[a] <= roi[a].1)
    };
    let qs: Vec<usize> = (0..nv).filter(|&q| mask[q] && inbox(q)).collect();
    let pos = |q: usize| -> [f64; 3] { [(q % dims[0]) as f64 - (dims[0] / 2) as f64, ((q / dims[0]) % dims[1]) as f64 - (dims[1] / 2) as f64, (q / (dims[0] * dims[1])) as f64 - (dims[2] / 2) as f64] };
    let mut wts: Vec<f64> = qs.iter().map(|&q| ims[0][q] as f64).collect();
    let mut coef = [[0.0f64; 4]; 3];
    let mut scale_r = 1.0;
    for round in 0..5 {
        let mut ata = nalgebra::DMatrix::<f64>::zeros(4, 4);
        let mut atb = nalgebra::DMatrix::<f64>::zeros(4, 3);
        for (i, &q) in qs.iter().enumerate() {
            let p = pos(q);
            let row = [p[0], p[1], p[2], 1.0];
            for a in 0..4 {
                for b in 0..4 {
                    ata[(a, b)] += wts[i] * row[a] * row[b];
                }
                for c in 0..3 {
                    atb[(a, c)] += wts[i] * row[a] * sfield[c * nv + q];
                }
            }
        }
        let sol = ata.lu().solve(&atb).unwrap();
        for c in 0..3 {
            for a in 0..4 {
                coef[c][a] = sol[(a, c)];
            }
        }
        let res: Vec<f64> = qs.iter().map(|&q| { let p = pos(q); (0..3).map(|c| (sfield[c * nv + q] - (coef[c][0] * p[0] + coef[c][1] * p[1] + coef[c][2] * p[2] + coef[c][3])).powi(2)).sum::<f64>().sqrt() }).collect();
        let mut rs = res.clone();
        rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        scale_r = (1.4826 * rs[rs.len() / 2]).max(1e-6);
        let rms = (res.iter().zip(&wts).map(|(r, w)| w * r * r).sum::<f64>() / wts.iter().sum::<f64>()).sqrt();
        println!("  affine IRLS round {round}: weighted residual rms {:.3} vox (robust scale {:.3} vox)", rms, scale_r);
        wts = qs.iter().zip(&res).map(|(&q, &r)| ims[0][q] as f64 / (1.0 + (r / (2.0 * scale_r)).powi(2))).collect();
    }
    // physical units: S in voxels per unit bellows (relative to bin 0); A₁[a][b] = B[a][b] Δ_a / Δ_b, t₁ = t Δ
    let a1: [[f64; 3]; 3] = [0, 1, 2].map(|a| [0, 1, 2].map(|b| coef[a][b] * vox[a] / vox[b]));
    let t1: [f64; 3] = [0, 1, 2].map(|a| coef[a][3] * vox[a]);
    let rmax = r_rel.iter().cloned().fold(0.0f64, f64::max);
    let smag: f64 = qs.iter().map(|&q| (0..3).map(|c| (sfield[c * nv + q] * vox[c] * 1e3).powi(2)).sum::<f64>().sqrt()).sum::<f64>() / qs.len() as f64 * rmax;
    println!("  box ∩ support {} voxels: mean |S| × r_max = {:.2} mm; affine t₁ × r_max = {:?} mm", qs.len(), smag, t1.map(|v| (v * rmax * 1e4).round() / 10.0));
    let mut js = serde_json::json!({"surrogate": "bellows", "r_ref": rb[0], "bins": nr, "bin_bellows": rb, "scale": scale, "roi": roi, "A1": a1, "t1_m": t1,
        "field_rel_resid": fit.rel_resid, "affine_robust_scale_vox": scale_r, "voxels": qs.len(), "mean_disp_mm_at_rmax": smag});
    if let Some(tp) = truth_resp {
        // oracle check only: the phantom's Lagrangian field at the box centre for the bins' true-state difference
        let ph = read_npy(&dir.join("physio.npy")).unwrap();
        let (w, v) = (ph.shape[1], ph.f64s());
        let rt: Vec<f64> = (0..nr).map(|b| { let r: Vec<f64> = (0..nro).filter(|&i| bins[i] == b).map(|i| v[i * w + 4]).collect(); r.iter().sum::<f64>() / r.len() as f64 }).collect();
        js["oracle_true_r_per_bin"] = serde_json::json!(rt);
        js["oracle_note"] = serde_json::json!(format!("compare the predicted displacement at the box centre with the phantom field {} at Δr = {:.3}", tp.display(), rt[nr - 1] - rt[0]));
    }
    let p = PathBuf::from(format!("{out}_affine.json"));
    std::fs::write(&p, serde_json::to_string_pretty(&js).unwrap()).unwrap();
    println!("wrote {} ({:.0}s)", p.display(), t0.elapsed().as_secs_f64());
}

/// the knot images `h` of encode `e` evaluated at every evaluation state and cardiac frame centre, into `slot[state]`
#[allow(clippy::too_many_arguments)]
fn fill_vals(ev: &[(f64, Vec<usize>, Vec<f64>)], cw: &[Vec<f64>], kr: usize, mc: usize, nv: usize, ne: usize, nfo: usize, e: usize, h: &[C32], slot: &mut [Vec<C32>]) {
    for (s, (_, idx, g)) in ev.iter().enumerate() {
        for (i, &q) in idx.iter().enumerate() {
            for f in 0..nfo {
                let mut z = (0.0f64, 0.0f64);
                for kk in 0..kr {
                    if g[kk] == 0.0 {
                        continue;
                    }
                    for b in 0..mc {
                        let wgt = g[kk] * cw[f][b];
                        if wgt == 0.0 {
                            continue;
                        }
                        let v = h[(kk * mc + b) * nv + q];
                        z.0 += wgt * v.re as f64;
                        z.1 += wgt * v.im as f64;
                    }
                }
                slot[s][(i * ne + e) * nfo + f] = C32::new(z.0 as f32, z.1 as f32);
            }
        }
    }
}

/// compact lumen values of one encode: per evaluation state, [voxel][frame] (`fill_vals` for a single encode)
fn lumen_compact(ev: &[(f64, Vec<usize>, Vec<f64>)], cw: &[Vec<f64>], kr: usize, mc: usize, nv: usize, nfo: usize, h: &[C32]) -> Vec<Vec<C32>> {
    let mut slot: Vec<Vec<C32>> = ev.iter().map(|(_, idx, _)| vec![C32::new(0.0, 0.0); idx.len() * nfo]).collect();
    fill_vals(ev, cw, kr, mc, nv, 1, nfo, 0, h, &mut slot);
    slot
}

/// resident host memory of this process (GiB), for the logs
fn rss_gib() -> f64 {
    std::fs::read_to_string("/proc/self/status").ok().and_then(|t| t.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| l.split_whitespace().nth(1)?.parse::<f64>().ok())).unwrap_or(0.0) / (1u64 << 20) as f64
}

/// what the solver arms of one encode share
struct ArmsCtx<'a> {
    dev: &'a Device,
    st: &'a Arc<CudaStream>,
    kern: &'a Arc<Kernels>,
    eng: &'a std::cell::RefCell<SegNufft>,
    cnt: &'a std::cell::Cell<(usize, usize)>,
    seg: &'a SegSegments,
    pts: &'a SegPoints,
    pts_ho: Option<&'a SegPoints>,
    y_ho: Option<&'a CudaSlice<C32>>,
    m_ho: usize,
    nc: usize,
    m: usize,
    nx: usize,
    nv: usize,
    kk: usize,
    dims: [usize; 3],
    out: &'a str,
    e: usize,
    /// knot stack → compact lumen values (see [`lumen_compact`])
    eval: &'a dyn Fn(&[C32]) -> Vec<Vec<C32>>,
}

struct ArmsOpts {
    arms: Vec<String>,
    prefix: usize,
    budget: f64,
    gamma_rel: f64,
    wthr: f64,
    kappa: f64,
    block: usize,
    wlevels: usize,
    admm_inner: usize,
    power_its: usize,
    llr_budget: f64,
    priors: String,
    save_at: Vec<usize>,
}

/// held-out residual, snapshots at gradient counts and the per-iteration log of one arm
struct Tracker<'a, 'b> {
    ac: &'b ArmsCtx<'a>,
    name: String,
    offset: f64,
    budget: f64,
    save_at: Vec<usize>,
    next_save: usize,
    pred: Option<CudaSlice<C32>>,
    ho_norm: f64,
    snaps: Vec<(String, Vec<Vec<C32>>)>,
    best: Option<(f64, f64, Vec<Vec<C32>>)>,
    ho: Vec<(f64, f64)>,
    csv: String,
    t0: Instant,
}

impl<'a, 'b> Tracker<'a, 'b> {
    fn new(ac: &'b ArmsCtx<'a>, name: &str, o: &ArmsOpts, offset: f64) -> Self {
        let ho_norm = ac.y_ho.map_or(1.0, |v| ac.kern.dot(ac.st, v, v).unwrap().re.max(1e-300));
        Self { ac, name: name.into(), offset, budget: o.budget, save_at: o.save_at.clone(), next_save: 0, pred: (ac.m_ho > 0).then(|| ac.st.alloc_zeros::<C32>(ac.nc * ac.m_ho).unwrap()), ho_norm, snaps: vec![], best: None, ho: vec![], csv: String::new(), t0: Instant::now() }
    }

    fn grads(&self) -> f64 {
        let c = self.ac.cnt.get();
        (c.0 + c.1) as f64 / 2.0 + self.offset
    }

    /// after an iteration: held-out residual, snapshots, log; false = the gradient budget is spent
    fn on(&mut self, it: usize, info: Option<&hasty_recon::knotncg::Info>, x: &CudaSlice<C32>) -> bool {
        let ac = self.ac;
        let g = self.grads();
        let mut ho = f64::NAN;
        if let (Some(ph), Some(yv), Some(pp)) = (ac.pts_ho, ac.y_ho, self.pred.as_mut()) {
            ac.eng.borrow_mut().forward(ac.st, ac.seg, ph, x, pp).unwrap();
            ac.kern.axpby(ac.st, num_complex::Complex64::new(-1.0, 0.0), yv, num_complex::Complex64::new(1.0, 0.0), pp).unwrap();
            ho = ac.kern.dot(ac.st, pp, pp).unwrap().re / self.ho_norm;
            self.ho.push((g, ho));
            if self.best.as_ref().is_none_or(|b| ho < b.1) {
                self.best = Some((g, ho, (ac.eval)(&ac.st.clone_dtoh(x).unwrap())));
            }
        }
        let mut h: Option<Vec<Vec<C32>>> = None;
        while self.next_save < self.save_at.len() && g + 1e-9 >= self.save_at[self.next_save] as f64 {
            let v = h.get_or_insert_with(|| (ac.eval)(&ac.st.clone_dtoh(x).unwrap())).clone();
            self.snaps.push((self.save_at[self.next_save].to_string(), v));
            self.next_save += 1;
        }
        let nan = f64::NAN;
        let (f, pw, pl, step, beta, rs, nw, nl) = info.map_or((nan, nan, nan, nan, nan, false, 0, 0), |i| (i.f, i.pw, i.pl, i.step, i.beta, i.restart, i.n_wt, i.n_llr));
        let fp = |v: f64| if v.is_nan() { 0.0 } else { v };
        let ftot = f + fp(pw) + fp(pl);
        let secs = self.t0.elapsed().as_secs_f64();
        self.csv += &format!("{},{it},{g:.2},{f:.8e},{pw:.8e},{pl:.8e},{ftot:.8e},{ho:.6e},{step:.4e},{beta:.4e},{},{nw},{nl},{secs:.1}\n", self.name, rs as u8);
        println!("    [{} e{}] it {it:3} grads {g:5.1}: f {f:.5e} Pw {pw:.4e} Pl {pl:.4e} F {ftot:.6e} held-out {ho:.5} step {step:.3e} β {beta:.3}{} (wt {nw}, llr {nl}, {secs:.0}s)", self.name, ac.e, if rs { " R" } else { "" });
        g + 1e-9 < self.budget
    }

    /// snapshots incl. the best held-out iterate ("best")
    fn finish(mut self) -> (Vec<(String, Vec<Vec<C32>>)>, String) {
        if let Some((g, r, h)) = self.best.take() {
            println!("  [{} e{}] held-out: {}; best at {g:.1} gradients ({r:.5})", self.name, self.ac.e, self.ho.iter().map(|(g, r)| format!("{g:.0}:{r:.5}")).collect::<Vec<_>>().join(" "), );
            self.snaps.push(("best".into(), h));
        }
        (self.snaps, self.csv)
    }
}

/// The solver comparison of one encode (research/convergence_strategy.md §8 P1–P2; see `--arms`). `x` = the warm
/// start, `y` the fit data (device; consumed by the PCG arm), `yh` the same on the host (re-uploaded per arm).
#[allow(clippy::too_many_arguments)]
fn run_arms(ac: &ArmsCtx, o: &ArmsOpts, mut kpre: hasty_recon::knotsolve::KnotPrecond, gram: Vec<f64>, lam_eff: Vec<f32>, dinv: Vec<f32>, mut x: CudaSlice<C32>, y: CudaSlice<C32>, yh: Vec<C32>) -> Vec<(String, Vec<(String, Vec<Vec<C32>>)>)> {
    use hasty_recon::knotncg::{self as kn, Info, LlrBackend, LlrHuber, PriorMetric, Priors, WavHuber};
    let (st, kern, e, k, nv) = (ac.st, ac.kern, ac.e, ac.kk, ac.nv);
    let gib = (1u64 << 30) as f64;
    let mut results = vec![];
    let mut csv = String::from("arm,it,grads,f,pw,pl,F,heldout,step,beta,restart,n_wt,n_llr,secs\n");
    let mut pm = PriorMetric::new(ac.dev, st, k, ac.dims, &gram, 1e-3, lam_eff, dinv).unwrap();
    let bulk = pm.bulk();
    let (lmin, lmax) = pm.lam_g.iter().fold((f64::INFINITY, 0.0f64), |(a, b), &v| (a.min(v), b.max(v)));
    println!(
        "  arms e{e}: {} unknowns, {} fit samples → R = unknowns / samples = {:.1} (per coil); bulk curvature {bulk:.4e} (D_s {:.3e}, mean λ_G {:.3e}, λ_G range {:.1}); budget {} gradients; arms {:?}",
        ac.nx, ac.m, ac.nx as f64 / ac.m as f64, pm.d_s, pm.lam_g.iter().sum::<f64>() / k as f64, lmax / lmin, o.budget, o.arms
    );
    {
        // the data symbol's floor: the share of k-space at the metric's floor (prior-only band at high R)
        let mut c = pm.c.clone();
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (cmin, cmed, cmax) = (c[0] as f64, c[c.len() / 2] as f64, c[c.len() - 1] as f64);
        let at_floor = c.iter().filter(|&&v| (v as f64) <= 1.01 * cmin).count() as f64 / c.len() as f64;
        println!("  arms e{e}: data symbol c(k): min {cmin:.3e}, median {cmed:.3e}, max {cmax:.3e} (range {:.0}); {:.1} % of k at the floor", cmax / cmin, 100.0 * at_floor);
    }
    let mk_op = || Op2d { eng: ac.eng, cnt: ac.cnt, seg: ac.seg, pts: ac.pts, kern: kern.clone(), nx: ac.nx, ny: ac.nc * ac.m };
    // ---- PCG (the arms v2 path: CGLS with G⁻¹ ⊗ C⁻¹) and the prefix iterate of the prior arms
    let run_pcg = o.arms.iter().any(|a| a == "pcg");
    let mut x5: Option<(Vec<C32>, (usize, usize))> = None;
    {
        ac.cnt.set((0, 0));
        let mut tr = Tracker::new(ac, "pcg", o, 0.0);
        tr.on(0, None, &x);
        let its = if run_pcg { (o.budget - 1.0).ceil().max(o.prefix as f64) as usize } else { o.prefix };
        let mut op = mk_op();
        let prefix = o.prefix as f64;
        hasty_recon::cg::pcgls_dyn(&mut op, Some(&mut kpre as &mut dyn hasty_recon::cg::Precond), st, hasty_recon::cg::Data::Owned(y), &mut x, 0.0, its, &mut |k, xd| {
            let g = tr.grads();
            if x5.is_none() && g + 1e-9 >= prefix {
                x5 = Some((st.clone_dtoh(xd).unwrap(), ac.cnt.get()));
            }
            if run_pcg {
                tr.on(k, None, xd);
            }
        }, None)
        .unwrap();
        if run_pcg {
            let (snaps, c) = tr.finish();
            csv += &c;
            results.push(("pcg".to_string(), snaps));
        }
    }
    drop(kpre);
    let prior_arms: Vec<&String> = o.arms.iter().filter(|a| *a != "pcg").collect();
    if !prior_arms.is_empty() {
        let (x5h, c5) = x5.take().expect("prefix iterate");
        let gamma = o.gamma_rel / bulk;
        // ---- prior weights and the prior-aware metric from the prefix iterate (data-derived, no truth)
        let tp = Instant::now();
        let mut wav = WavHuber::new(ac.dims, o.wlevels, 0.0, gamma);
        let mut cw = vec![];
        wav.coef(&x5h, k, &mut cw);
        let sig_w = wav.sigma_finest(&cw, k);
        let lam_w = bulk * o.wthr * sig_w;
        wav.theta = lam_w * gamma;
        let fr = wav.curvature_fractions(&cw, k);
        let sw = wav.symbol(&fr);
        drop(cw);
        let mut l = hasty_recon::llr::Llr::new(ac.dims, o.block, 1.0, vec![true; nv]);
        l.zeta_basis = Some(pm.u.clone());
        let mut lg = hasty_recon::llr_gpu::LlrGpu::new(l, &[0]).unwrap();
        lg.budget = (o.llr_budget * gib) as usize;
        let mut lh = LlrHuber { be: LlrBackend::Gpu(lg), theta: 0.0, gamma, shift: 1 };
        let mut tmp = x5h.clone();
        {
            let l = lh.be.llr();
            l.thr_abs = None;
            l.set_iteration(1);
        }
        lh.be.apply(&mut tmp, k).unwrap();
        let sig_l = lh.be.llr().last.sigma;
        let lam_l = bulk * o.kappa * sig_l;
        lh.theta = lam_l * gamma;
        tmp.copy_from_slice(&x5h);
        let zs = lh.svt(&mut tmp, k, lh.theta).unwrap();
        lh.be.llr().zeta_basis = None;
        drop(tmp);
        let (use_w, use_l) = (o.priors.contains("wav"), o.priors.contains("llr"));
        let sl: Vec<f64> = if use_l { zs.zeta.iter().map(|z| z.max(0.0) / gamma).collect() } else { vec![0.0; k] };
        let swo = use_w.then_some(sw);
        println!(
            "  arms e{e}: priors {}: γ = {:.2}/bulk = {gamma:.4e}; wavelet db4 L{} σ̂_w {sig_w:.4e}, λ_w = bulk·{}·σ̂_w = {lam_w:.4e}, θ_w {:.4e}; LLR B{} σ̂_l {sig_l:.4e}, λ_l = bulk·{}·σ̂_l = {lam_l:.4e}, θ_l {:.4e} (kept {} of {} singular values at θ_l) ({:.0}s)",
            o.priors, o.gamma_rel, o.wlevels, o.wthr, wav.theta, o.block, o.kappa, lh.theta, zs.n_above, zs.n_sv, tp.elapsed().as_secs_f64()
        );
        println!("    wavelet curvature fractions (level, ε, frac): {}", fr.iter().map(|(l, e, f)| format!("{l}{}{}{}:{f:.3}", e[0], e[1], e[2])).collect::<Vec<_>>().join(" "));
        println!("    LLR discarded fraction ζ(ω) (ω in G-eigenvector order): {}", zs.zeta.iter().map(|z| format!("{z:.3}")).collect::<Vec<_>>().join(" "));
        let mut pri = Priors::new(k, nv, use_w.then_some(wav), use_l.then_some(lh));
        let set_metric = |pm: &mut PriorMetric, arm: &str| {
            pm.ident = (arm == "ncgI").then_some(bulk);
            if arm == "ncg0" || arm == "ncgI" {
                pm.set_prior(st, &vec![0.0; k], None).unwrap();
            } else if arm == "ncgcap" {
                // the data metric capped by the prior curvature: G ⊗ C + 1/γ
                pm.set_prior(st, &vec![1.0 / gamma; k], None).unwrap();
            } else if let Some(v) = arm.strip_prefix("admm") {
                let rho: f64 = v.parse::<f64>().unwrap() * bulk;
                let ns = use_w as usize + use_l as usize;
                pm.set_prior(st, &vec![ns as f64 * rho; k], None).unwrap();
            } else {
                pm.set_prior(st, &sl, swo.as_deref()).unwrap();
            }
        };
        // ---- λ_max(P∇²F_γ) for FISTA / ZMSHR1 (charged to both)
        let (mut lip, mut pow_cost) = (0.0, 0.0);
        if prior_arms.iter().any(|a| *a == "fista" || *a == "zmshr1") {
            set_metric(&mut pm, "ncg");
            st.memcpy_htod(&x5h, &mut x).unwrap();
            let c0 = ac.cnt.get();
            let mut op = mk_op();
            let (l, h) = kn::power_lmax(&mut op, kern, st, &pm, &mut pri, &x, ac.nc * ac.m, o.power_its).unwrap();
            let c1 = ac.cnt.get();
            pow_cost = ((c1.0 - c0.0) + (c1.1 - c0.1)) as f64 / 2.0;
            lip = 1.1 * l;
            println!("  arms e{e}: power iteration λ_max(P∇²F_γ) {:?} → L = 1.1 × {l:.4} ({pow_cost} gradients, charged to fista / zmshr1)", h.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
            // diagnostic (not charged): the data term alone in the same metric, λ_max(P AᴴA)
            let mut none = Priors::new(k, nv, None, None);
            let (ld, hd) = kn::power_lmax(&mut op, kern, st, &pm, &mut none, &x, ac.nc * ac.m, o.power_its).unwrap();
            println!("  arms e{e}: diagnostic λ_max(P AᴴA) (data term only, not charged) {:?} → {ld:.4}", hd.iter().map(|v| (v * 1e4).round() / 1e4).collect::<Vec<_>>());
        }
        for arm in prior_arms {
            let arm = arm.as_str();
            set_metric(&mut pm, arm);
            st.memcpy_htod(&x5h, &mut x).unwrap();
            pri.n_wt = 0;
            pri.n_llr = 0;
            pri.secs = 0.0;
            let offset = if arm == "fista" || arm == "zmshr1" { pow_cost } else { 0.0 };
            let mut tr = Tracker::new(ac, arm, o, offset);
            let mut op = mk_op();
            let ydev = st.clone_htod(&yh).unwrap();
            // the residual at the prefix iterate (carried over from PCG in a single run: not charged; the counter is
            // reset to the prefix's count after each solver's initialisation)
            let resid = |op: &mut Op2d, mut yv: CudaSlice<C32>| -> CudaSlice<C32> {
                let mut q = st.alloc_zeros::<C32>(yv.len()).unwrap();
                op.forward(st, &x, &mut q).unwrap();
                kern.axpby(st, num_complex::Complex64::new(-1.0, 0.0), &q, num_complex::Complex64::new(1.0, 0.0), &mut yv).unwrap();
                yv
            };
            let ta = Instant::now();
            // continuation stages (ncgc) stop at stage_end; the counter is reset to the prefix count only at the first
            // initialisation
            let stage_end = std::cell::Cell::new(f64::INFINITY);
            let first = std::cell::Cell::new(true);
            let mut cb = |i: &Info, xd: &CudaSlice<C32>| -> bool {
                if i.iter == 0 && first.get() {
                    ac.cnt.set(c5);
                    first.set(false);
                }
                let go = tr.on(i.iter, Some(i), xd);
                let c = ac.cnt.get();
                go && ((c.0 + c.1) as f64 / 2.0) + 1e-9 < stage_end.get()
            };
            match arm {
                "ncgc" => {
                    // γ continuation 1 → 0.1 → 0.01 per bulk, equal gradient shares, NCG restarted per stage (the
                    // metric's prior symbol is kept: outside the quadratic zone the curvature λ/|c| does not depend on γ)
                    let mut r = resid(&mut op, ydev);
                    let span = (o.budget - o.prefix as f64) / 3.0;
                    for (si, gr) in [1.0f64, 0.1, 0.01].into_iter().enumerate() {
                        let g = gr / bulk;
                        if let Some(w) = pri.wav.as_mut() {
                            w.gamma = g;
                            w.theta = lam_w * g;
                        }
                        if let Some(l) = pri.llr.as_mut() {
                            l.gamma = g;
                            l.theta = lam_l * g;
                        }
                        stage_end.set(o.prefix as f64 + span * (si + 1) as f64);
                        println!("    [ncgc e{e}] stage {si}: γ = {gr}/bulk until {:.1} gradients", stage_end.get());
                        r = kn::ncg(&mut op, kern, st, &pm, &mut pri, &mut x, r, &mut cb).unwrap();
                        let c = ac.cnt.get();
                        if (c.0 + c.1) as f64 / 2.0 + 1e-9 >= o.budget {
                            break;
                        }
                    }
                    if let Some(w) = pri.wav.as_mut() {
                        w.gamma = gamma;
                        w.theta = lam_w * gamma;
                    }
                    if let Some(l) = pri.llr.as_mut() {
                        l.gamma = gamma;
                        l.theta = lam_l * gamma;
                    }
                }
                "ncg" | "ncg0" | "ncgcap" | "ncgI" => {
                    let r = resid(&mut op, ydev);
                    kn::ncg(&mut op, kern, st, &pm, &mut pri, &mut x, r, &mut cb).unwrap();
                }
                "zmshr1" => {
                    let r = resid(&mut op, ydev);
                    kn::zmshr1(&mut op, kern, st, &pm, &mut pri, &mut x, r, 1.0 / lip, &mut cb).unwrap();
                }
                "fista" => kn::fista(&mut op, kern, st, &pm, &mut pri, &mut x, ydev, lip, &mut cb).unwrap(),
                a if a.starts_with("admm") => {
                    let rho = a[4..].parse::<f64>().unwrap() * bulk;
                    let r = resid(&mut op, ydev);
                    kn::admm(&mut op, kern, st, &pm, &mut pri, &mut x, r, rho, o.admm_inner, &mut cb).unwrap();
                }
                o => panic!("unknown arm {o} (pcg, ncg, ncg0, ncgcap, ncgI, ncgc, zmshr1, fista, admm<ρ/bulk>)"),
            }
            println!("  arms e{e}: {arm} done ({:.0}s; priors {:.0}s: {} wavelet stack transforms, {} LLR sweeps; host RSS {:.1} GiB)", ta.elapsed().as_secs_f64(), pri.secs, pri.n_wt, pri.n_llr, rss_gib());
            let (snaps, c) = tr.finish();
            csv += &c;
            results.push((arm.to_string(), snaps));
        }
    }
    std::fs::write(format!("{}_arms_e{e}.csv", ac.out), csv).unwrap();
    results
}
