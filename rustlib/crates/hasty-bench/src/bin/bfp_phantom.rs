//! Accuracy of the BFP y ↔ z grid-FFT intermediates (research/bfp_storage.md §8, `HASTY_SEG_FFT_STORE`) on real
//! phantom data: ph0001 (`sim_flow_10min/scan`), its truth image (water + fat proton density, fat at a fixed phase,
//! smooth background phase, as research/experiments/bfp/acc.py), its true coil maps compressed to `--vcoils` virtual
//! coils by the signal-region Gram, its field map (segments over the readout window, L = `--nseg`), and the real
//! sample positions, readout times and (order 3) per-sample concomitant tables of encode `--enc` (the sim's nominal
//! trajectory, `hasty_recon::segphys` as recon_p2np). One engine; the store switched on it (f32, then each BFP width,
//! then f32 again for the run-to-run spread).
//!
//! Per case it prints the relative error against f32 of
//!   * the loaded grid slot (pad + forward grid FFT of coil batch 0): the stored buffer itself;
//!   * the forward A x at the samples;
//!   * the early gradient Aᴴ A x (residual = A x, data 0);
//!   * the near-convergence gradient Aᴴ(A x − y) with y = A_f32 x + n, n white complex Gaussian with rms(A x) / rms(n)
//!     = `--snr-k` (4.0: ph0001 at image SNR 20, research/bfp_storage.md §3.4), i.e. relative to the noise-driven
//!     gradient;
//!   * and the f32 run-to-run spread of each (atomics).
//!
//!   bfp_phantom <scan dir> [--order 0|3] [--knots2d KR:RDEG,MC:CDEG] [--readouts N] [--enc 3] [--frame F]
//!               [--vcoils 10] [--nseg 10] [--stores bfp22,bfp20,bfp24] [--snr-k 4.0] [--grid-gb 6]
//! Static (default): one (frame, encode) set; `--knots2d`: a 2-D knot engine over `--readouts` readouts of the encode
//! spread over the scan (cardiac phase from the sim's ECG, respiratory knots at quantiles of a synthetic breathing
//! amplitude of the excitation time, as opbench), the truth image at every knot.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};
use hasty_fft::Store;
use hasty_nufft::{build_segments, SegNufft, SegNufftOpts, SegPoints, SegSegments};
use hasty_recon::data::{read_npy, SimData};
use hasty_recon::segphys::{fit_monomials_hi, hi_table, mean_coefficients, seg_tables_hi, segment_phase};
use nalgebra::DMatrix;
use num_complex::Complex64;
use rayon::prelude::*;

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let (n, d) = a.par_iter().zip(b).map(|(x, y)| (((x.re - y.re) as f64).powi(2) + ((x.im - y.im) as f64).powi(2), (y.re as f64).powi(2) + (y.im as f64).powi(2))).reduce(|| (0.0, 0.0), |p, q| (p.0 + q.0, p.1 + q.1));
    (n / d.max(1e-300)).sqrt()
}

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.u().max(1e-300), self.u());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

struct Run<'a> {
    st: &'a Arc<CudaStream>,
    seg: &'a SegSegments,
    pts: &'a SegPoints,
    x: &'a CudaSlice<C32>,
    y: &'a CudaSlice<C32>,
    m: usize,
    nc: usize,
    nx: usize,
}

impl Run<'_> {
    /// (A x, Aᴴ A x, Aᴴ(A x − y))
    fn apply(&self, op: &mut SegNufft) -> (Vec<C32>, Vec<C32>, Vec<C32>) {
        let st = self.st;
        let mut yo = st.alloc_zeros::<C32>(self.nc * self.m).unwrap();
        let mut ge = st.alloc_zeros::<C32>(self.nx).unwrap();
        op.forward(st, self.seg, self.pts, self.x, &mut yo).unwrap();
        op.adjoint(st, self.seg, self.pts, &yo, &mut ge).unwrap();
        let ge_h = st.clone_dtoh(&ge).unwrap();
        let mut r = st.alloc_zeros::<C32>(self.nc * self.m).unwrap();
        op.forward_residual(st, self.seg, self.pts, self.x, self.y, &mut r).unwrap();
        op.adjoint(st, self.seg, self.pts, &r, &mut ge).unwrap();
        (st.clone_dtoh(&yo).unwrap(), ge_h, st.clone_dtoh(&ge).unwrap())
    }
}

fn main() {
    let mut a: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |k: &str| -> Option<String> { a.iter().position(|x| x == k).map(|i| { a.remove(i); a.remove(i) }) };
    let order: u32 = take("--order").map_or(0, |v| v.parse().unwrap());
    let knots2d: Option<(usize, u32, usize, u32)> = take("--knots2d").map(|v| {
        let p: Vec<usize> = v.split([',', ':']).map(|x| x.trim().parse().unwrap()).collect();
        (p[0], p[1] as u32, p[2], p[3] as u32)
    });
    let nro: usize = take("--readouts").map_or(340, |v| v.parse().unwrap());
    let enc: usize = take("--enc").map_or(3, |v| v.parse().unwrap());
    let frame: Option<usize> = take("--frame").map(|v| v.parse().unwrap());
    let nvc: usize = take("--vcoils").map_or(10, |v| v.parse().unwrap());
    let nseg: usize = take("--nseg").map_or(10, |v| v.parse().unwrap());
    let stores: Vec<Store> = take("--stores").unwrap_or_else(|| "bfp22,bfp20,bfp24".into()).split(',').map(|s| Store::parse(s).expect("--stores bfpB,..")).collect();
    let snr_k: f64 = take("--snr-k").map_or(4.0, |v| v.parse().unwrap());
    let grid_gb: f64 = take("--grid-gb").map_or(6.0, |v| v.parse().unwrap());
    let kprune: bool = take("--kprune").map_or(true, |v| v != "0");
    let basis: String = take("--basis").unwrap_or_else(|| "fixed".into());
    // research/sample_kernels.md §20: the new defaults (one-bin static order 3, bfp24 in place, phase-2 autotune
    // candidates, representative tune knot) against every switch back to the old paths, both engines autotuned
    // research/bfp_storage.md §10: the compressed-DRAM switches, each against an all-f32 engine (`--switches`
    // "fft=bfp24/grid=bfp24/maps=bfp24,diag=bfp24/…", `/` or `;` between configurations, `,` inside one)
    let switches: Option<String> = take("--switches");
    let compare_defaults = a.iter().position(|x| x == "--compare-defaults").map(|i| a.remove(i)).is_some();
    let dir = Path::new(a.first().expect("bfp_phantom <scan dir> [options]"));
    let t0 = Instant::now();
    let sd = SimData::load_traj_opts(dir, "nominal", true).expect("load the sim");
    let (dims, nv, nc_all, na) = (sd.dims, sd.nv(), sd.nc, sd.na);
    println!("ph0001 scan {}: image {dims:?} ({:.1} mm voxels), {nc_all} coils, {} readouts × {na} ADC samples, {} frames; loaded in {:.1} s", dir.display(), sd.vox[0] * 1e3, sd.n_readouts(), sd.n_frames, t0.elapsed().as_secs_f64());
    // truth image (as acc.py): water + fat at a fixed phase, smooth background phase
    let ld = |f: &str| -> Vec<f32> {
        let r = read_npy(&dir.join(f)).unwrap();
        assert_eq!(r.shape.iter().product::<usize>(), nv, "{f}: shape {:?} vs image {dims:?}", r.shape);
        r.f32s()
    };
    let (pdw, pdf) = (ld("truth_pd_water.npy"), ld("truth_pd_fat.npy"));
    let pmax = pdw.iter().zip(&pdf).map(|(a, b)| a + b).fold(0f32, f32::max);
    let img: Vec<C32> = (0..nv).into_par_iter().map(|q| {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        let c = |v: usize, n: usize| -1.0 + 2.0 * v as f64 / (n - 1) as f64;
        let (x, y, z) = (c(i, dims[0]), c(j, dims[1]), c(k, dims[2]));
        let v = (Complex64::new(pdw[q] as f64, 0.0) + Complex64::from_polar(pdf[q] as f64, 2.1)) * Complex64::from_polar(1.0, 0.8 * x + 0.5 * y - 0.3 * z + 0.6 * x * y);
        C32::new(v.re as f32, v.im as f32)
    }).collect();
    let support: Vec<usize> = (0..nv).filter(|&q| pdw[q] + pdf[q] > 0.05 * pmax).collect();
    drop((pdw, pdf));
    // physics on the nominal grid (no GNL), whitened true maps
    let psi = sd.psi.clone();
    let grid = sd.physics_grid(false, &psi);
    // virtual coils: eigenvectors of the signal-region Gram of the whitened true maps
    let gram = {
        let mut g = DMatrix::<Complex64>::zeros(nc_all, nc_all);
        for &q in &support {
            for c in 0..nc_all {
                let sc = grid.smaps[c * nv + q];
                for d in 0..nc_all {
                    let sdv = grid.smaps[d * nv + q];
                    g[(c, d)] += Complex64::new(sc.re as f64, sc.im as f64) * Complex64::new(sdv.re as f64, -sdv.im as f64);
                }
            }
        }
        g
    };
    let eig = gram.clone().symmetric_eigen();
    let mut idx: Vec<usize> = (0..nc_all).collect();
    idx.sort_by(|&p, &q| eig.eigenvalues[q].partial_cmp(&eig.eigenvalues[p]).unwrap());
    let tot: f64 = eig.eigenvalues.iter().sum();
    let kept: f64 = idx[..nvc].iter().map(|&i| eig.eigenvalues[i]).sum();
    // S_v(q) = Σ_c conj(U_cv) S_c(q)
    let smaps: Vec<C32> = (0..nvc * nv).into_par_iter().map(|i| {
        let (v, q) = (i / nv, i % nv);
        let u = eig.eigenvectors.column(idx[v]);
        let mut s = Complex64::new(0.0, 0.0);
        for c in 0..nc_all {
            let z = grid.smaps[c * nv + q];
            s += u[c].conj() * Complex64::new(z.re as f64, z.im as f64);
        }
        C32::new(s.re as f32, s.im as f32)
    }).collect();
    println!("  virtual coils {nvc} of {nc_all}: signal-region energy kept {:.3}, eigenvalue ratio last / first {:.3}; {} support voxels", kept / tot, eig.eigenvalues[idx[nvc - 1]] / eig.eigenvalues[idx[0]], support.len());
    // readouts of the encode: one (frame, encode) set, or (knots) nro spread over the scan
    let ros: Vec<usize> = match knots2d {
        None => {
            let f = frame.unwrap_or(sd.n_frames / 2);
            sd.readouts_of(enc, Some(f))
        }
        Some(_) => {
            let all = sd.readouts_of(enc, None);
            let step = (all.len() / nro).max(1);
            all.into_iter().step_by(step).take(nro).collect()
        }
    };
    let ident: Vec<Vec<f64>> = (0..nc_all).map(|i| (0..nc_all).map(|j| (i == j) as u8 as f64).collect()).collect();
    // coordinates, readout times and concomitant coefficients only (the k-space data stay on disk)
    let set = sd.load_set(&ros, &hasty_recon::kload::LoadOpts { w: &ident, cc: None, keep: None, data: false }).expect("sample set");
    let m = set.tau.len();
    // per-sample tables (order 3: concomitant ΔC and the degree-3/4 deviation of the encode's mean, as recon_p2np)
    let with_conc = order > 0;
    let wbar = if with_conc { mean_coefficients(&[&set], na) } else { vec![[0.0; 4]; na] };
    let (coords, tau, phi0, dcv, dwv, htab) = if with_conc {
        let f4 = fit_monomials_hi(&sd, &grid, &support);
        let (tab, dw) = seg_tables_hi(&sd, &set, &f4, true, Some(&wbar));
        (tab.coords, tab.tau, tab.phi0, tab.dc, dw, hi_table(&f4))
    } else {
        (set.coords.clone(), set.tau.iter().map(|&v| v as f32).collect::<Vec<f32>>(), vec![0f32; m], vec![], vec![], vec![])
    };
    let tau_adc: Vec<f64> = set.tau[..na].to_vec();
    let fitv: Vec<usize> = support.iter().copied().step_by(7).collect();
    let psi_f = segment_phase(&grid, &wbar, &tau_adc, true);
    let (sp, at, s0, ds) = if basis == "svd" {
        let g = |v: usize, t: f64| Complex64::from_polar(1.0, -psi_f(v, t));
        hasty_nufft::build_segments_svd(nv, nseg, tau_adc[0], tau_adc[na - 1], na, &fitv, na.min(512), &g)
    } else {
        build_segments(nv, nseg, tau_adc[0], tau_adc[na - 1], na, &fitv, &psi_f)
    };
    // conditioning of the segment model: the interpolator norm ‖a(τ)‖₂ over the table and the diagonals' rms
    let anorm: Vec<f64> = (0..at.len() / nseg).map(|i| (0..nseg).map(|l| { let z = at[i * nseg + l]; (z.re as f64).powi(2) + (z.im as f64).powi(2) }).sum::<f64>().sqrt()).collect();
    let drms: Vec<f64> = (0..nseg).map(|l| (support.iter().step_by(97).map(|&q| { let z = sp[l * nv + q]; (z.re as f64).powi(2) + (z.im as f64).powi(2) }).sum::<f64>() / support.iter().step_by(97).count() as f64).sqrt()).collect();
    println!("  segment basis {basis}: interpolator norm ‖a(τ)‖ min / mean / max {:.2} / {:.2} / {:.2}; diagonal rms per segment {:.2?}", anorm.iter().cloned().fold(f64::INFINITY, f64::min), anorm.iter().sum::<f64>() / anorm.len() as f64, anorm.iter().cloned().fold(0.0, f64::max), drms);
    println!("  encode {enc}: {} readouts, {m} samples; L {nseg} segments over τ ∈ [{:.2}, {:.2}] ms; order {order}", ros.len(), tau_adc[0] * 1e3, tau_adc[na - 1] * 1e3);
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let (frames, bspline) = match knots2d { Some((_, _, mc, cdeg)) => (mc, cdeg), None => (1, 1) };
    let opts = || SegNufftOpts { sigma: 1.25, width: 8, nseg, order, frames, bspline, max_grid_bytes: (grid_gb * (1u64 << 30) as f64) as usize };
    let kk = [&coords[0][..], &coords[1][..], &coords[2][..]];
    // one engine with its segments and sample tables (thread store / static-knot rule as set by the caller)
    let build = || -> (SegNufft, SegSegments, SegPoints) {
        let mut op = match knots2d {
            None => SegNufft::new(&dev, dims, sd.vox, nvc, opts()).unwrap(),
            Some((kr, rdeg, _, _)) => SegNufft::new_temporal2d(&dev, dims, sd.vox, nvc, opts(), kr, rdeg).unwrap(),
        };
        op.set_maps(&st, &smaps).unwrap();
        op.set_kprune(kprune).unwrap();
        let seg = op.segments(&st, &sp, &at, s0, ds).unwrap();
        let pts = match knots2d {
            None if order >= 3 => op.points_hi(&st, kk, &tau, &phi0, &dcv, &dwv, &htab, None).unwrap(),
            None => op.points(&st, kk, &tau, &phi0, &dcv, None).unwrap(),
            Some((kr, rdeg, mc, _)) => {
                // cardiac knot time from the sim's ECG; respiratory: a synthetic breathing amplitude of the excitation
                // time, knots at its quantiles over the readouts (as opbench)
                let amp: Vec<f64> = ros.iter().map(|&i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * sd.ro.t_exc[i] / 4.3).cos()).collect();
                let mut srt = amp.clone();
                srt.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let qf = |f: f64| srt[((f * (srt.len() - 1) as f64).round() as usize).min(srt.len() - 1)];
                let nri = if rdeg == 0 { kr } else { kr - 1 };
                let edges: Vec<f64> = (0..=nri).map(|k| qf(k as f64 / nri as f64)).collect();
                let (mut t, mut ri, mut rv) = (vec![0f32; m], vec![0u32; m], vec![0f32; m]);
                for (r, &ro) in ros.iter().enumerate() {
                    let av = amp[r];
                    let ir = (0..nri).rfind(|&k| av >= edges[k]).unwrap_or(0);
                    let v = if rdeg == 0 { 0.0 } else { ((av - edges[ir]) / (edges[ir + 1] - edges[ir]).max(1e-12)).clamp(0.0, 1.0) };
                    let card = &sd.ro.card[ro];
                    for n in 0..na {
                        let j = r * na + n;
                        t[j] = ((card.phase + set.tau[j] / card.rr) * mc as f64 - 0.5).rem_euclid(mc as f64) as f32;
                        ri[j] = ir as u32;
                        rv[j] = v as f32;
                    }
                }
                if order >= 3 {
                    op.points2d_hi(&st, kk, &tau, &phi0, &dcv, &t, &ri, &rv, &dwv, &htab).unwrap()
                } else {
                    op.points2d(&st, kk, &tau, &phi0, &dcv, &t, &ri, &rv).unwrap()
                }
            }
        };
        (op, seg, pts)
    };
    if let Some(sw) = &switches {
        compare_switches(&dev, &build, &img, nv, nvc, m, snr_k, sw, &format!("{} order {order}", match knots2d { None => "static".to_string(), Some((kr, rd, mc, cd)) => format!("2-D knots {kr}:{rd},{mc}:{cd}") }));
        println!("  total {:.1} s", t0.elapsed().as_secs_f64());
        return;
    }
    if compare_defaults {
        compare(&dev, &build, &img, nv, nvc, m, snr_k, &format!("{} order {order}", match knots2d { None => "static".to_string(), Some((kr, rd, mc, cd)) => format!("2-D knots {kr}:{rd},{mc}:{cd}") }));
        println!("  total {:.1} s", t0.elapsed().as_secs_f64());
        return;
    }
    // the engine built f32; the store switched on it
    hasty_nufft::set_thread_fft_store(Some(Store::F32));
    let (mut op, seg, pts) = build();
    hasty_nufft::set_thread_fft_store(None);
    let nimg = op.frames;
    let host_x: Vec<C32> = (0..nimg * nv).map(|i| img[i % nv]).collect();
    let x = st.clone_htod(&host_x).unwrap();
    drop(host_x);
    let nx = nimg * nv;
    println!("  engine: {nimg} image(s), coil batch {} ({} channels), {}", op.cb, op.cb * nseg, op.grid_fft_desc());
    // data y = A_f32 x + n
    let mut y0 = st.alloc_zeros::<C32>(nvc * m).unwrap();
    op.forward(&st, &seg, &pts, &x, &mut y0).unwrap();
    let ax = st.clone_dtoh(&y0).unwrap();
    let rms = (ax.par_iter().map(|z| (z.re as f64).powi(2) + (z.im as f64).powi(2)).sum::<f64>() / ax.len() as f64).sqrt();
    let sig = rms / snr_k / std::f64::consts::SQRT_2;
    let mut rng = Rng(0x5eed_5eed_1234_5678);
    let yh: Vec<C32> = ax.iter().map(|z| C32::new(z.re + (sig * rng.gauss()) as f32, z.im + (sig * rng.gauss()) as f32)).collect();
    let y = st.clone_htod(&yh).unwrap();
    // the noise alone (n = y − A x): its adjoint is the near-convergence gradient up to the forward error
    let nh: Vec<C32> = yh.iter().zip(&ax).map(|(a, b)| C32::new(a.re - b.re, a.im - b.im)).collect();
    let dn = st.clone_htod(&nh).unwrap();
    drop((ax, yh, nh));
    let adj_n = |op: &mut SegNufft| -> Vec<C32> {
        let mut g = st.alloc_zeros::<C32>(nimg * nv).unwrap();
        op.adjoint(&st, &seg, &pts, &dn, &mut g).unwrap();
        st.clone_dtoh(&g).unwrap()
    };
    let run = Run { st: &st, seg: &seg, pts: &pts, x: &x, y: &y, m, nc: nvc, nx };
    let cb = op.cb;
    // the loaded slot with the block-pruned transform (every cell defined; with k-support pruning the cells outside the
    // support are left undefined, in place they hold BFP mantissa bits)
    let slot = |op: &mut SegNufft| {
        op.set_kprune(false).unwrap();
        let s = op.debug_loaded_slot(&st, &seg, &x, 0, 0, cb, op.fuse_padcrop, true).unwrap();
        op.set_kprune(kprune).unwrap();
        s
    };
    let s_f = slot(&mut op);
    let (yf, gef, gcf) = run.apply(&mut op);
    let (yf2, gef2, gcf2) = run.apply(&mut op);
    let (nf1, nf2) = (adj_n(&mut op), adj_n(&mut op));
    // the operator's own f32 sensitivity: x and n moved by one ulp (relative 2^-24, random signs)
    {
        let mut r = Rng(31);
        let ulp = |v: &[C32], r: &mut Rng| -> Vec<C32> { v.iter().map(|z| C32::new(z.re * (1.0 + if r.u() < 0.5 { 5.96e-8 } else { -5.96e-8 }), z.im * (1.0 + if r.u() < 0.5 { 5.96e-8 } else { -5.96e-8 }))).collect() };
        let xu = st.clone_htod(&ulp(&st.clone_dtoh(&x).unwrap(), &mut r)).unwrap();
        let nu = st.clone_htod(&ulp(&st.clone_dtoh(&dn).unwrap(), &mut r)).unwrap();
        let ru = Run { st: &st, seg: &seg, pts: &pts, x: &xu, y: &y, m, nc: nvc, nx };
        let (yu, geu, gcu) = ru.apply(&mut op);
        let mut g = st.alloc_zeros::<C32>(nimg * nv).unwrap();
        op.adjoint(&st, &seg, &pts, &nu, &mut g).unwrap();
        let gu = st.clone_dtoh(&g).unwrap();
        println!("  one-ulp sensitivity (f32): x moved by one ulp moves the forward {:.1e}, Aᴴ A x {:.1e}, Aᴴ(A x − y) {:.1e}; n moved by one ulp moves Aᴴ n {:.1e}", rel(&yu, &yf), rel(&geu, &gef), rel(&gcu, &gcf), rel(&gu, &nf1));
    }
    println!("  f32 run to run: forward {:.1e}, early gradient {:.1e}, converged gradient {:.1e}, adjoint of the noise {:.1e}; |Aᴴ A x| / |Aᴴ n| = {:.1}", rel(&yf2, &yf), rel(&gef2, &gef), rel(&gcf2, &gcf), rel(&nf2, &nf1), (gef.iter().map(|z| (z.re as f64).powi(2) + (z.im as f64).powi(2)).sum::<f64>() / nf1.iter().map(|z| (z.re as f64).powi(2) + (z.im as f64).powi(2)).sum::<f64>()).sqrt());
    let case = format!("{} order {order}", match knots2d { None => "static".to_string(), Some((kr, rd, mc, cd)) => format!("2-D knots {kr}:{rd},{mc}:{cd}") });
    for &s in &stores {
        op.set_fft_store(s).unwrap();
        let b = match s { Store::Bfp(b) => b, _ => 32 };
        let q = (2.0f64).powi(-(b as i32));
        let s_b = slot(&mut op);
        let (yb, geb, gcb) = run.apply(&mut op);
        let nb = adj_n(&mut op);
        let (es, ef, ee, ec) = (rel(&s_b, &s_f), rel(&yb, &yf), rel(&geb, &gef), rel(&gcb, &gcf));
        println!("    {s}: adjoint of the noise alone {:.2e} ({:.2} · 2^-b)", rel(&nb, &nf1), rel(&nb, &nf1) / q);
        println!(
            "  {case}, {s} ({}): loaded slot {es:.2e} ({:.2} · 2^-b), forward {ef:.2e} ({:.2}), early gradient Aᴴ A x {ee:.2e} ({:.2}), converged gradient Aᴴ(Ax − y) {ec:.2e} ({:.2} · 2^-b)",
            if op.grid_fft_desc().contains("in the grid slot") { "in place" } else { "separate buffer" },
            es / q,
            ef / q,
            ee / q,
            ec / q
        );
    }
    op.set_fft_store(Store::F32).unwrap();
    let (yf3, _, _) = run.apply(&mut op);
    println!("  f32 after the BFP runs vs the first f32 forward: {:.1e}; total {:.1} s", rel(&yf3, &yf), t0.elapsed().as_secs_f64());
}

/// env switches of the old paths (research/sample_kernels.md §20): the phase-2 candidate set off, the tile / vs / ra
/// candidates timed on knot 0
const OLD_ENV: [(&str, &str); 5] = [("HASTY_SEG_VSA_STAGE", "0"), ("HASTY_SEG_LBMIN", "0"), ("HASTY_SEG_SPILLSKIP", "0"), ("HASTY_SEG_VS_SCH", "0"), ("HASTY_SEG_TUNE_KNOT", "old")];

/// outputs of one engine on the case: A x, Aᴴ n, Aᴴ A x, Aᴴ(A x − y)
type Outs = [Vec<C32>; 4];

#[allow(clippy::too_many_arguments)]
fn outs(st: &Arc<CudaStream>, op: &mut SegNufft, seg: &SegSegments, pts: &SegPoints, x: &CudaSlice<C32>, y: &CudaSlice<C32>, n: &CudaSlice<C32>, nc: usize, m: usize) -> Outs {
    let nx = x.len();
    let run = Run { st, seg, pts, x, y, m, nc, nx };
    let (ax, gn, gc) = run.apply(op);
    let mut g = st.alloc_zeros::<C32>(nx).unwrap();
    op.adjoint(st, seg, pts, n, &mut g).unwrap();
    [ax, st.clone_dtoh(&g).unwrap(), gn, gc]
}

/// The old engine (every switch back: static path, f32 store, previous candidates, knot-0 tuning) and the new defaults,
/// one after the other (one engine on the device at a time), both autotuned on the case's set; the new engine also
/// with its store switched to f32 (the kernel / path change alone). Errors relative to the old engine, next to its
/// one-ulp floor (x, n moved by one ulp) and its run-to-run spread.
#[allow(clippy::too_many_arguments)]
fn compare(dev: &Device, build: &dyn Fn() -> (SegNufft, SegSegments, SegPoints), img: &[C32], nv: usize, nvc: usize, m: usize, snr_k: f64, case: &str) {
    let st = dev.default_stream();
    let names = ["forward A x", "adjoint Aᴴ n", "early gradient Aᴴ A x", "converged gradient Aᴴ(A x − y)"];
    let ulp = |v: &[C32], r: &mut Rng| -> Vec<C32> { v.iter().map(|z| C32::new(z.re * (1.0 + if r.u() < 0.5 { 5.96e-8 } else { -5.96e-8 }), z.im * (1.0 + if r.u() < 0.5 { 5.96e-8 } else { -5.96e-8 }))).collect() };
    // ── old paths
    for (k, v) in OLD_ENV {
        std::env::set_var(k, v);
    }
    hasty_nufft::set_thread_static_knot(Some(hasty_nufft::StaticKnot::Never));
    hasty_nufft::set_thread_fft_store(Some(Store::F32));
    let t = Instant::now();
    let (mut op, seg, pts) = build();
    let (kf, ka) = op.autotune(&st, &seg, &pts).unwrap();
    hasty_nufft::set_thread_static_knot(None);
    hasty_nufft::set_thread_fft_store(None);
    for (k, _) in OLD_ENV {
        std::env::remove_var(k);
    }
    let nimg = op.frames;
    let hx: Vec<C32> = (0..nimg * nv).map(|i| img[i % nv]).collect();
    let x = st.clone_htod(&hx).unwrap();
    println!("  OLD (static path {}, store {}, previous candidates, knot-0 tuning): coil batch {}, kernels {kf:.2} / {ka:.2} ms, vs {:?}, ra {:?}, kl {}, {}; built + tuned in {:.1} s", !op.is_one_bin(), op.fft_store(), op.cb, op.vs_active(), op.ra_active(), op.kl_active(), op.grid_fft_desc(), t.elapsed().as_secs_f64());
    // data y = A_old x + n
    let mut y0 = st.alloc_zeros::<C32>(nvc * m).unwrap();
    op.forward(&st, &seg, &pts, &x, &mut y0).unwrap();
    let ax = st.clone_dtoh(&y0).unwrap();
    let rms = (ax.par_iter().map(|z| (z.re as f64).powi(2) + (z.im as f64).powi(2)).sum::<f64>() / ax.len() as f64).sqrt();
    let sig = rms / snr_k / std::f64::consts::SQRT_2;
    let mut rng = Rng(0x5eed_5eed_1234_5678);
    let nh: Vec<C32> = ax.iter().map(|_| C32::new((sig * rng.gauss()) as f32, (sig * rng.gauss()) as f32)).collect();
    let yh: Vec<C32> = ax.iter().zip(&nh).map(|(a, b)| C32::new(a.re + b.re, a.im + b.im)).collect();
    let (y, dn) = (st.clone_htod(&yh).unwrap(), st.clone_htod(&nh).unwrap());
    drop((ax, yh, y0));
    let o_old = outs(&st, &mut op, &seg, &pts, &x, &y, &dn, nvc, m);
    let o_old2 = outs(&st, &mut op, &seg, &pts, &x, &y, &dn, nvc, m);
    let mut r = Rng(31);
    let xu = st.clone_htod(&ulp(&hx, &mut r)).unwrap();
    let nu = st.clone_htod(&ulp(&nh, &mut r)).unwrap();
    let o_ux = outs(&st, &mut op, &seg, &pts, &xu, &y, &dn, nvc, m);
    let o_un = outs(&st, &mut op, &seg, &pts, &x, &y, &nu, nvc, m);
    // floor per output: x moved by one ulp (forward, gradients), n moved by one ulp (adjoint of the noise)
    let floor: [f64; 4] = [rel(&o_ux[0], &o_old[0]), rel(&o_un[1], &o_old[1]), rel(&o_ux[2], &o_old[2]), rel(&o_ux[3], &o_old[3])];
    let spread: [f64; 4] = [0, 1, 2, 3].map(|i| rel(&o_old2[i], &o_old[i]));
    drop((op, seg, pts, xu, nu));
    // ── new defaults
    let t = Instant::now();
    let (mut op, seg, pts) = build();
    let (kf, ka) = op.autotune(&st, &seg, &pts).unwrap();
    println!("  NEW (one-bin {}, store {} {}, phase-2 candidates, representative knot): coil batch {}, kernels {kf:.2} / {ka:.2} ms, vs {:?}, ra {:?}, kl {}, {}; built + tuned in {:.1} s", op.is_one_bin(), op.fft_store(), if op.grid_fft_desc().contains("in the grid slot") { "in place" } else { "separate buffer" }, op.cb, op.vs_active(), op.ra_active(), op.kl_active(), op.grid_fft_desc(), t.elapsed().as_secs_f64());
    let o_new = outs(&st, &mut op, &seg, &pts, &x, &y, &dn, nvc, m);
    let s_new = op.fft_store();
    op.set_fft_store(Store::F32).unwrap();
    let o_newf = outs(&st, &mut op, &seg, &pts, &x, &y, &dn, nvc, m);
    op.set_fft_store(s_new).unwrap();
    println!("  {case}: relative error against the OLD engine (floor = OLD with x or n moved by one ulp; spread = OLD run to run)");
    println!("    | output | floor | spread | NEW defaults | NEW / floor | NEW paths, f32 store | / floor |");
    for i in 0..4 {
        let (en, ef) = (rel(&o_new[i], &o_old[i]), rel(&o_newf[i], &o_old[i]));
        println!("    | {} | {:.2e} | {:.2e} | {en:.2e} | {:.2} | {ef:.2e} | {:.2} |", names[i], floor[i], spread[i], en / floor[i].max(1e-300), ef / floor[i].max(1e-300));
    }
}

/// thread defaults of one switch configuration "fft=bfp24,grid=bfp24,maps=bfp16,diag=bfp16" (unnamed: f32)
fn set_switches(cfg: &str) {
    let mut st = [Store::F32; 4];
    for kv in cfg.split(',').filter(|v| !v.trim().is_empty()) {
        let (k, v) = kv.split_once('=').unwrap_or_else(|| panic!("--switches: {kv}: key=value"));
        let s = Store::parse(v.trim()).unwrap_or_else(|| panic!("--switches: {v}"));
        let i = ["fft", "grid", "maps", "diag"].iter().position(|n| *n == k.trim()).unwrap_or_else(|| panic!("--switches: key {k} (fft, grid, maps, diag)"));
        st[i] = s;
    }
    hasty_nufft::set_thread_fft_store(Some(st[0]));
    hasty_nufft::set_thread_grid_store(Some(st[1]));
    hasty_nufft::set_thread_map_stores(Some((st[2], st[3])));
}

fn clear_switches() {
    hasty_nufft::set_thread_fft_store(None);
    hasty_nufft::set_thread_grid_store(None);
    hasty_nufft::set_thread_map_stores(None);
}

/// Every configuration of `sw` against an all-f32 engine (no autotune: the default kernels; one engine on the device at
/// a time). Floor = the f32 engine with x or n moved by one ulp; spread = the f32 engine run to run and a second f32
/// engine (the adjoint's atomics order differs between engines).
#[allow(clippy::too_many_arguments)]
fn compare_switches(dev: &Device, build: &dyn Fn() -> (SegNufft, SegSegments, SegPoints), img: &[C32], nv: usize, nvc: usize, m: usize, snr_k: f64, sw: &str, case: &str) {
    let st = dev.default_stream();
    let names = ["forward A x", "adjoint Aᴴ n", "early gradient Aᴴ A x", "converged gradient Aᴴ(A x − y)"];
    let ulp = |v: &[C32], r: &mut Rng| -> Vec<C32> { v.iter().map(|z| C32::new(z.re * (1.0 + if r.u() < 0.5 { 5.96e-8 } else { -5.96e-8 }), z.im * (1.0 + if r.u() < 0.5 { 5.96e-8 } else { -5.96e-8 }))).collect() };
    set_switches("");
    let (mut op, seg, pts) = build();
    clear_switches();
    let nimg = op.frames;
    let hx: Vec<C32> = (0..nimg * nv).map(|i| img[i % nv]).collect();
    let x = st.clone_htod(&hx).unwrap();
    println!("  F32 reference: coil batch {}, fft {}, grid {}, maps / diag {:?}, {}", op.cb, op.fft_store(), op.grid_store(), op.map_stores(), op.grid_fft_desc());
    let mut y0 = st.alloc_zeros::<C32>(nvc * m).unwrap();
    op.forward(&st, &seg, &pts, &x, &mut y0).unwrap();
    let ax = st.clone_dtoh(&y0).unwrap();
    let rms = (ax.par_iter().map(|z| (z.re as f64).powi(2) + (z.im as f64).powi(2)).sum::<f64>() / ax.len() as f64).sqrt();
    let sig = rms / snr_k / std::f64::consts::SQRT_2;
    let mut rng = Rng(0x5eed_5eed_1234_5678);
    let nh: Vec<C32> = ax.iter().map(|_| C32::new((sig * rng.gauss()) as f32, (sig * rng.gauss()) as f32)).collect();
    let yh: Vec<C32> = ax.iter().zip(&nh).map(|(a, b)| C32::new(a.re + b.re, a.im + b.im)).collect();
    let (y, dn) = (st.clone_htod(&yh).unwrap(), st.clone_htod(&nh).unwrap());
    drop((ax, yh, y0));
    let o_ref = outs(&st, &mut op, &seg, &pts, &x, &y, &dn, nvc, m);
    let o_ref2 = outs(&st, &mut op, &seg, &pts, &x, &y, &dn, nvc, m);
    let mut r = Rng(31);
    let xu = st.clone_htod(&ulp(&hx, &mut r)).unwrap();
    let nu = st.clone_htod(&ulp(&nh, &mut r)).unwrap();
    let o_ux = outs(&st, &mut op, &seg, &pts, &xu, &y, &dn, nvc, m);
    let o_un = outs(&st, &mut op, &seg, &pts, &x, &y, &nu, nvc, m);
    let floor: [f64; 4] = [rel(&o_ux[0], &o_ref[0]), rel(&o_un[1], &o_ref[1]), rel(&o_ux[2], &o_ref[2]), rel(&o_ux[3], &o_ref[3])];
    let mut spread: [f64; 4] = [0, 1, 2, 3].map(|i| rel(&o_ref2[i], &o_ref[i]));
    drop((op, seg, pts, xu, nu));
    // a second f32 engine
    set_switches("");
    let (mut op, seg, pts) = build();
    clear_switches();
    let o_e = outs(&st, &mut op, &seg, &pts, &x, &y, &dn, nvc, m);
    for i in 0..4 {
        spread[i] = spread[i].max(rel(&o_e[i], &o_ref[i]));
    }
    drop((op, seg, pts));
    println!("  {case}: relative error against the all-f32 engine (floor = f32 with x or n moved by one ulp; spread = f32 run to run and engine to engine)");
    println!("    | configuration | output | floor | spread | error | error / floor |");
    for cfg in sw.split([';', '/']).filter(|c| !c.trim().is_empty()) {
        set_switches(cfg);
        let t = Instant::now();
        let (mut op, seg, pts) = build();
        clear_switches();
        let o = outs(&st, &mut op, &seg, &pts, &x, &y, &dn, nvc, m);
        let active = format!("fft {}, grid {}, maps / diag {:?}", op.fft_store(), op.grid_store(), op.map_stores());
        for i in 0..4 {
            let e = rel(&o[i], &o_ref[i]);
            println!("    | {cfg} | {} | {:.2e} | {:.2e} | {e:.2e} | {:.2} |", names[i], floor[i], spread[i], e / floor[i].max(1e-300));
        }
        println!("      ({cfg}: active {active}; coil batch {}; {:.1} s)", op.cb, t.elapsed().as_secs_f64());
        drop((op, seg, pts));
    }
}
