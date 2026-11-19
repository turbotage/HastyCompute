//! GPU, real data: per-readout sample compression (`hasty_recon::rcompress`) against the uncompressed data term on
//! the 1 mm box scan reconstructed at 2 mm (factory_arms/ph0001/sim_flow_10min_box1mm_b; encode 0, every 4th readout,
//! samples inside the grid's band, 8 SV-compressed coils, the true coil maps, 8 B0 segments of the field map; ROI = the
//! grid box from the prescription, one design per readout; 1/6 of the readouts held out).
//! - objective (with the data constant) and gradient at a CGLS iterate: compressed = uncompressed to ≤ 1e-4;
//! - 8 CGLS iterations: the iterates and the held-out residual curve (exact: Σ ‖R A_S x − ỹ‖² + ‖y‖² − ‖ỹ‖² over the
//!   held-out readouts, relative to the original held-out energy) agree to ≤ 1e-4.
//! Skipped when the scan is not on disk. `cargo test --release -p hasty-recon --test rcompress -- --nocapture`
use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts, SegPoints, SegSegments};
use hasty_recon::cg::{pcgls_dyn, Data};
use hasty_recon::coils::{CoilCompression, CoilGram, CoilRule};
use hasty_recon::data::{SampleSet, SimData};
use hasty_recon::gpu::Kernels;
use hasty_recon::kload::LoadOpts;
use hasty_recon::rcompress::{Band, Compressor, RcDev, RcLin, RcOpts, Roi};
use hasty_recon::setop::LinOp;
use hasty_recon::Error;
use num_complex::Complex64;

struct SegLin<'a> {
    eng: &'a RefCell<SegNufft>,
    seg: &'a SegSegments,
    pts: &'a SegPoints,
    kern: Arc<Kernels>,
    nx: usize,
    ny: usize,
}

impl LinOp for SegLin<'_> {
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
        Ok(self.eng.borrow_mut().forward(st, self.seg, self.pts, x, y)?)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        Ok(self.eng.borrow_mut().adjoint(st, self.seg, self.pts, y, x)?)
    }
}

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let (n, d) = a.iter().zip(b).fold((0.0f64, 0.0f64), |s, (p, q)| (s.0 + ((p.re - q.re) as f64).powi(2) + ((p.im - q.im) as f64).powi(2), s.1 + (q.re as f64).powi(2) + (q.im as f64).powi(2)));
    (n / d.max(1e-300)).sqrt()
}

/// one part (whole readouts): records cropped to the grid band, per-readout lengths, data (nc × m)
fn part(full: &SampleSet, na: usize, ro_sel: &[usize], nc: usize) -> (SampleSet, Vec<usize>, Vec<C32>) {
    let m_all = full.tau.len();
    let mut idx = vec![];
    let mut lens = vec![];
    for &r in ro_sel {
        let b = idx.len();
        for n in 0..na {
            let j = r * na + n;
            if (0..3).all(|a| full.coords[a][j].abs() < std::f32::consts::PI * 0.999) {
                idx.push(j);
            }
        }
        if idx.len() > b {
            lens.push(idx.len() - b);
        }
    }
    let s = SampleSet {
        readouts: ro_sel.iter().map(|&r| full.readouts[r]).collect(),
        coords: [0, 1, 2].map(|a| idx.iter().map(|&j| full.coords[a][j]).collect()),
        y: vec![],
        tau: idx.iter().map(|&j| full.tau[j]).collect(),
        wq: vec![],
    };
    let y = (0..nc).flat_map(|c| idx.iter().map(move |&j| (c, j))).map(|(c, j)| full.y[c * m_all + j]).collect();
    (s, lens, y)
}

#[test]
fn compressed_cgls_matches_on_real_data() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty/phantom/factory_arms/ph0001/sim_flow_10min_box1mm_b/scan");
    if !dir.join("meta.json").exists() {
        eprintln!("skipped: {} not found", dir.display());
        return;
    }
    let mut sd = SimData::load_lazy(&dir).unwrap();
    sd.set_recon_grid([0.5; 3]);
    let (dims, vox, nv, na) = (sd.dims, sd.vox, sd.nv(), sd.na);
    let w = sd.whitener();
    let grid = sd.physics_grid(false, &w);
    let ros: Vec<usize> = sd.readouts_of(0, None).into_iter().step_by(4).collect();
    let s0 = sd.load_set(&ros[..24], &LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
    let mut g = CoilGram::new(sd.nc);
    g.add(&s0.y, s0.tau.len(), 1);
    let cc = CoilCompression::from_gram(&g, CoilRule::Fixed(8));
    let nc = cc.nvc();
    let smaps = cc.apply(&grid.smaps, nv);
    let full = sd.load_set(&ros, &LoadOpts { w: &w, cc: Some(&cc.rows), keep: None, data: true }).unwrap();
    let ho_mask = hasty_recon::knotsolve::heldout_mask(sd.n_readouts(), 6);
    let (fi, hi): (Vec<usize>, Vec<usize>) = (0..ros.len()).partition(|&r| !ho_mask[ros[r]]);
    let (fit, fit_lens, fit_y) = part(&full, na, &fi, nc);
    let (ho, ho_lens, ho_y) = part(&full, na, &hi, nc);
    drop(full);
    // compression: ROI = the grid box, band = the field map's (coil-energy weighted), decay ≤ 300 1/s
    let ce: Vec<f64> = (0..nv).map(|q| (0..nc).map(|c| { let z = smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum()).collect();
    let band = Band { r_max: 300.0, ..Band::from_ratemap(&grid.omega, None, &ce, 1e-4, 1.0, 200.0) };
    let mut comp = Compressor::new(RcOpts::new(Roi::grid_box(dims, vox), band));
    let (sc_f, yc_f) = comp.compress_set(&fit, &fit_lens, vox, &fit_y, nc);
    let (sc_h, yc_h) = comp.compress_set(&ho, &ho_lens, vox, &ho_y, nc);
    let (mf, mh) = (fit.tau.len(), ho.tau.len());
    println!("grid {dims:?} at {:.2} mm, {} fit + {} held-out readouts, {mf} + {mh} samples → {} + {} ({:.3}×), {}", vox[0] * 1e3, fit_lens.len(), ho_lens.len(), sc_f.m(), sc_h.m(), mf as f64 / sc_f.m() as f64, comp.describe());
    // RC_GPU_LOCK=<file>: flock(2) taken only for the GPU part (the setup above is CPU)
    let _lock = std::env::var("RC_GPU_LOCK").ok().map(|p| {
        let f = std::fs::OpenOptions::new().create(true).append(true).open(&p).unwrap();
        f.lock().unwrap();
        f
    });
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let one = Complex64::new(1.0, 0.0);
    let nseg = 8;
    let (tmin, tmax) = fit.tau.iter().chain(&ho.tau).fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &t| (a.min(t), b.max(t)));
    // fit voxels for the segment interpolators (few: the CPU part stays short)
    let sig: Vec<usize> = (0..nv).filter(|&q| ce[q] > 1e-3 * ce.iter().cloned().fold(0.0, f64::max)).step_by(97).collect();
    let segtab = build_segments(nv, nseg, tmin, tmax, 128, &sig, &|v, t| grid.omega[v] * t);
    // both NUFFT kernels: production (W 10, σ 1.25) and a more accurate one (W 12, σ 2: isolates the compression from
    // the NUFFT's own error, which the two sample sets sample differently)
    for (width, sigma) in [(10usize, 1.25f64), (12, 2.0)] {
        let mut eng = SegNufft::new(&dev, dims, vox, nc, SegNufftOpts { sigma, width, nseg, order: 0, frames: 1, bspline: 1, max_grid_bytes: 2 << 30 }).unwrap();
        eng.set_maps(&st, &smaps).unwrap();
        let seg = eng.segments(&st, &segtab.0, &segtab.1, segtab.2, segtab.3).unwrap();
        let mk = |eng: &mut SegNufft, s: &SampleSet, sel: Option<&[u32]>| -> SegPoints {
            let g = |v: &[f32]| -> Vec<f32> { match sel { Some(i) => i.iter().map(|&j| v[j as usize]).collect(), None => v.to_vec() } };
            let c = [g(&s.coords[0]), g(&s.coords[1]), g(&s.coords[2])];
            let tau = g(&s.tau.iter().map(|&t| t as f32).collect::<Vec<_>>());
            let z = vec![0f32; tau.len()];
            eng.points(&st, [&c[0], &c[1], &c[2]], &tau, &z, &[], None).unwrap()
        };
        let (pf, ph) = (mk(&mut eng, &fit, None), mk(&mut eng, &ho, None));
        let (pfc, phc) = (mk(&mut eng, &fit, Some(&sc_f.sel)), mk(&mut eng, &ho, Some(&sc_h.sel)));
        let rc = RcDev::new(&dev, &st, &comp, &[sc_f.clone(), sc_h.clone()]).unwrap();
        let eng = RefCell::new(eng);
        let (yf_d, yh_d, ycf_d, ych_d) = (st.clone_htod(&fit_y).unwrap(), st.clone_htod(&ho_y).unwrap(), st.clone_htod(&yc_f).unwrap(), st.clone_htod(&yc_h).unwrap());
        let ho_norm: f64 = sc_h.ynorm.iter().sum();
        let ho_const: f64 = sc_h.cnorm.iter().sum();
        let f_const: f64 = sc_f.cnorm.iter().sum();
        let iters = 8;
        // uncompressed CGLS with held-out residuals
        let run = |compressed: bool| -> (Vec<Vec<C32>>, Vec<f64>) {
            let mut xs = vec![];
            let mut hos = vec![];
            let mut x = st.alloc_zeros::<C32>(nv).unwrap();
            let mut pred = st.alloc_zeros::<C32>(nc * if compressed { sc_h.m() } else { mh }).unwrap();
            let base = SegLin { eng: &eng, seg: &seg, pts: if compressed { &pfc } else { &pf }, kern: kern.clone(), nx: nv, ny: nc * if compressed { sc_f.m() } else { mf } };
            let mut cb = |_k: usize, xd: &CudaSlice<C32>| {
                xs.push(st.clone_dtoh(xd).unwrap());
                let mut e = eng.borrow_mut();
                if compressed {
                    e.forward(&st, &seg, &phc, xd, &mut pred).unwrap();
                    rc.apply(&st, 1, &mut pred, false).unwrap();
                    kern.axpby(&st, -one, &ych_d, one, &mut pred).unwrap();
                    hos.push((kern.dot(&st, &pred, &pred).unwrap().re + ho_const) / ho_norm);
                } else {
                    e.forward(&st, &seg, &ph, xd, &mut pred).unwrap();
                    kern.axpby(&st, -one, &yh_d, one, &mut pred).unwrap();
                    hos.push(kern.dot(&st, &pred, &pred).unwrap().re / ho_norm);
                }
            };
            if compressed {
                let mut op = RcLin::new(base, &rc, 0);
                pcgls_dyn(&mut op, None, &st, Data::Borrowed(&ycf_d), &mut x, 0.0, iters, &mut cb, None).unwrap();
            } else {
                let mut op = base;
                pcgls_dyn(&mut op, None, &st, Data::Borrowed(&yf_d), &mut x, 0.0, iters, &mut cb, None).unwrap();
            }
            (xs, hos)
        };
        let (xu, hu) = run(false);
        let (xc, hc) = run(true);
        // objective and gradient at iterate 3 (uncompressed run)
        let x3 = st.clone_htod(&xu[2]).unwrap();
        let mut r = st.alloc_zeros::<C32>(nc * mf).unwrap();
        let mut gu = st.alloc_zeros::<C32>(nv).unwrap();
        eng.borrow_mut().forward(&st, &seg, &pf, &x3, &mut r).unwrap();
        kern.axpby(&st, -one, &yf_d, one, &mut r).unwrap();
        let fu = kern.dot(&st, &r, &r).unwrap().re;
        eng.borrow_mut().adjoint(&st, &seg, &pf, &r, &mut gu).unwrap();
        let mut rcm = st.alloc_zeros::<C32>(nc * sc_f.m()).unwrap();
        let mut gc = st.alloc_zeros::<C32>(nv).unwrap();
        {
            let mut op = RcLin::new(SegLin { eng: &eng, seg: &seg, pts: &pfc, kern: kern.clone(), nx: nv, ny: nc * sc_f.m() }, &rc, 0);
            op.forward(&st, &x3, &mut rcm).unwrap();
            kern.axpby(&st, -one, &ycf_d, one, &mut rcm).unwrap();
            op.adjoint(&st, &rcm, &mut gc).unwrap();
        }
        let fc = kern.dot(&st, &rcm, &rcm).unwrap().re + f_const;
        let (e_f, e_g) = ((fu - fc).abs() / fu, rel(&st.clone_dtoh(&gc).unwrap(), &st.clone_dtoh(&gu).unwrap()));
        let e_x: Vec<f64> = xu.iter().zip(&xc).map(|(a, b)| rel(b, a)).collect();
        let e_h: Vec<f64> = hu.iter().zip(&hc).map(|(a, b)| (a - b).abs() / a).collect();
        println!("W {width}, σ {sigma}: objective {e_f:.2e}, gradient {e_g:.2e}; iterates {:?}; held-out (uncompressed {:?}) rel. diff {:?}", e_x.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>(), hu.iter().map(|v| format!("{v:.5}")).collect::<Vec<_>>(), e_h.iter().map(|v| format!("{v:.1e}")).collect::<Vec<_>>());
        let tol = 1e-4;
        assert!(e_f < tol && e_g < tol, "W {width}: objective {e_f:.2e} / gradient {e_g:.2e}");
        assert!(e_x.iter().all(|&v| v < tol), "W {width}: iterates {e_x:?}");
        assert!(e_h.iter().all(|&v| v < tol), "W {width}: held-out {e_h:?}");
    }
}
