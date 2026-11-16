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
    eng: &'a mut SegNufft,
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
        Ok(self.eng.forward(st, self.seg, self.pts, x, y)?)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        Ok(self.eng.adjoint(st, self.seg, self.pts, y, x)?)
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

fn write_c8_c(path: &std::path::Path, shape: &[usize], data: &[C32]) {
    let sh: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
    let hdr = format!("{{'descr': '<c8', 'fortran_order': False, 'shape': ({}), }}", sh.join(", "));
    let mut h2 = hdr.into_bytes();
    while (h2.len() + 11) % 64 != 0 {
        h2.push(b' ');
    }
    h2.push(b'\n');
    let mut f = vec![0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0];
    f.extend_from_slice(&(h2.len() as u16).to_le_bytes());
    f.extend_from_slice(&h2);
    for z in data {
        f.extend_from_slice(&z.re.to_le_bytes());
        f.extend_from_slice(&z.im.to_le_bytes());
    }
    std::fs::write(path, f).unwrap();
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
    let m_max = (0..ne).map(|e| sd.readouts_of(e, None).len()).max().unwrap() * sd.na / resp_keep.map_or(1, |k| k.0);
    let tiles_per_sample = 1.25 * ((width as f64 + 3.0) / 4.0).powi(3);
    let led: Vec<(&str, f64)> = vec![
        ("CGLS image vectors (x, s, p; time-only 2)", if time_only { 2.0 } else { 3.0 } * (kr * mc * nv) as f64 * 8.0),
        ("CGLS data vectors (3 × coils × samples; time-only 2)", if time_only { 2.0 } else { 3.0 } * (nc * m_max) as f64 * 8.0),
        ("sample tables", m_max as f64 * (48.0 + 4.0 * tiles_per_sample)),
        ("coil maps + segment diagonals", ((nc + nseg) * nv) as f64 * 8.0),
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
    let nx = kr * mc * nv;
    let mut vals: Vec<Vec<Vec<C32>>> = save_at.iter().map(|_| ev.iter().map(|(_, idx, _)| vec![C32::new(0.0, 0.0); idx.len() * ne * nfo]).collect()).collect();
    let mut seg: Option<SegSegments> = None;
    let mut wbar_e: Option<Vec<[f64; 4]>> = None;
    let mut monofit: Option<hasty_recon::segphys::MonoFit> = None;
    let enc_list: Vec<usize> = encodes.unwrap_or_else(|| (0..ne).collect());
    let mut timing = vec![];
    for &e in &enc_list {
        let te = Instant::now();
        let mut ros = sd.readouts_of(e, None);
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
            seg = Some(eng.segments(&st, &sp, &at, s0, ds).unwrap());
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
        let mut pre = if warm > 0 { Some(hasty_recon::precond::CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&coords[0], &coords[1], &coords[2]], &smaps, nc, 1e-3).unwrap()) } else { None };
        let pts = eng.points2d(&st, [&coords[0], &coords[1], &coords[2]], &tau, &phi0, &dcv, &t, &ri, &rv).unwrap();
        drop((coords, tau, phi0, dcv, t, ri, rv));
        if timing.is_empty() && std::env::var("HASTY_SPLINE_NOTUNE").map_or(true, |v| v == "0") {
            let tt = Instant::now();
            let (kf, ka) = eng.autotune(&st, seg, &pts).unwrap();
            println!("  autotune {:.1}s: tile kernels {kf:.1} / {ka:.1} ms (knot 0, one coil batch of {}; fwd {:?}, adj {:?})", tt.elapsed().as_secs_f64(), eng.cb, eng.tiles_fwd, eng.tiles_adj);
        }
        let y = st.clone_htod(&yh).unwrap();
        drop(yh);
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
                eng.forward(&st, seg, &pts, &xd, &mut yy).unwrap();
                st.synchronize().unwrap();
                let b = Instant::now();
                eng.adjoint(&st, seg, &pts, &y, &mut xa).unwrap();
                st.synchronize().unwrap();
                if rep > 0 {
                    tf.push((b.duration_since(a).as_secs_f64(), b.elapsed().as_secs_f64()));
                }
            }
            let (f, a) = tf.iter().fold((0.0, 0.0), |s, t| (s.0 + t.0 / tf.len() as f64, s.1 + t.1 / tf.len() as f64));
            println!("  time-only encode {e}: forward {f:.2}s + adjoint {a:.2}s = {:.2}s per gradient pass ({} knots, {m} samples, coil batch {})", f + a, kr * mc, eng.cb);
            return;
        }
        let mut x = st.alloc_zeros::<C32>(nx).unwrap();
        if let Some(pr) = pre.as_mut() {
            let tw = Instant::now();
            let mut sx = st.alloc_zeros::<C32>(nv).unwrap();
            {
                let mut op = Op2d { eng: &mut eng, seg, pts: &pts, kern: kern.clone(), nx, ny: nc * m };
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
        let tl = Instant::now();
        let mut snaps: Vec<(usize, Vec<C32>)> = vec![];
        let mut op = Op2d { eng: &mut eng, seg, pts: &pts, kern: kern.clone(), nx, ny: nc * m };
        let hist = pcgls_cb(&mut op, None, &st, &y, &mut x, 0.0, iters, &mut |k, xd| {
            if save_at.contains(&k) {
                snaps.push((k, st.clone_dtoh(xd).unwrap()));
            }
        })
        .unwrap();
        st.synchronize().unwrap();
        let tsolve = tl.elapsed().as_secs_f64();
        for (k, h) in snaps {
            let si = save_at.iter().position(|&a| a == k).unwrap();
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
                        vals[si][s][(i * ne + e) * nfo + f] = C32::new(z.0 as f32, z.1 as f32);
                    }
                }
            }
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
    for (si, &k) in save_at.iter().enumerate() {
        for (s, (r, idx, _)) in ev.iter().enumerate() {
            let p = PathBuf::from(format!("{out}_r{r}_it{k}_lumen.npy"));
            write_c8_c(&p, &[idx.len(), ne, nfo], &vals[si][s]);
        }
    }
    let mean_it: f64 = timing.iter().sum::<f64>() / timing.len().max(1) as f64;
    println!("done ({:.0}s): mean {:.2}s per CGLS iteration per encode ({} knots per encode); outputs {out}_r*_it*_lumen.npy", t0.elapsed().as_secs_f64(), mean_it, kr * mc);
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
