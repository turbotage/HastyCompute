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
    let resp_src = take("--resp").unwrap_or_else(|| "truth".into());
    let encodes: Option<Vec<usize>> = take("--encodes").map(|s| s.split(',').map(|v| v.parse().unwrap()).collect());
    let evals: Vec<(f64, PathBuf)> = take("--eval").expect("--eval r:idx.npy,...").split(',').map(|p| { let (r, f) = p.split_once(':').unwrap(); (r.parse().unwrap(), PathBuf::from(f)) }).collect();
    let out = take("--out").expect("--out prefix");
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
    let w = sd.whitener();
    let grid = sd.physics_grid(false, &w);
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
    let m_max = (0..ne).map(|e| sd.readouts_of(e, None).len()).max().unwrap() * sd.na;
    let tiles_per_sample = 1.25 * ((width as f64 + 3.0) / 4.0).powi(3);
    let led: Vec<(&str, f64)> = vec![
        ("CGLS image vectors (4 × knots × nv; time-only 2)", if time_only { 2.0 } else { 4.0 } * (kr * mc * nv) as f64 * 8.0),
        ("CGLS data vectors (3 × coils × samples; time-only 2)", if time_only { 2.0 } else { 3.0 } * (nc * m_max) as f64 * 8.0),
        ("sample tables", m_max as f64 * (48.0 + 4.0 * tiles_per_sample)),
        ("coil maps + segment diagonals", ((nc + nseg) * nv) as f64 * 8.0),
        ("FFT / kernel workspace, margin", 1.5 * gib),
    ];
    let (free, total) = cudarc::driver::result::mem_get_info().unwrap();
    let fixed: f64 = led.iter().map(|l| l.1).sum();
    let grid_budget = (free as f64 * 0.97 - fixed).min(3.0 * gib);
    println!("  device ledger ({:.1} of {:.1} GiB free):", free as f64 / gib, total as f64 / gib);
    for (k, v) in &led {
        println!("    {k}: {:.2} GiB", v / gib);
    }
    println!("    grid slot budget: {:.2} GiB", grid_budget / gib);
    if grid_budget < 0.4 * gib {
        eprintln!("splinerecon: the plan does not fit the device ({:.2} GiB fixed, {:.2} GiB free): fewer coils / knots or a host-resident solver", fixed / gib, free as f64 / gib);
        std::process::exit(2);
    }
    let opts = SegNufftOpts { sigma: 1.25, width, nseg, order: 0, frames: mc, bspline: cdeg, max_grid_bytes: grid_budget as usize };
    let mut eng = SegNufft::new_temporal2d(&dev, dims, sd.vox, nc, opts, kr, rdeg).unwrap();
    eng.set_maps(&st, &smaps).unwrap();
    let nx = kr * mc * nv;
    let mut vals: Vec<Vec<Vec<C32>>> = save_at.iter().map(|_| ev.iter().map(|(_, idx, _)| vec![C32::new(0.0, 0.0); idx.len() * ne * nfo]).collect()).collect();
    let mut seg: Option<SegSegments> = None;
    let enc_list: Vec<usize> = encodes.unwrap_or_else(|| (0..ne).collect());
    let mut timing = vec![];
    for &e in &enc_list {
        let te = Instant::now();
        let ros = sd.readouts_of(e, None);
        let set = sd.load_set(&ros, &hasty_recon::kload::LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
        let m = set.tau.len();
        let na = m / ros.len();
        assert_eq!(na * ros.len(), m, "fixed ADC count per readout expected");
        if seg.is_none() {
            // B0 segments over the readout window: ψ(v, τ) = ω_v τ, fitted on object voxels
            let (tmin, tmax) = set.tau.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &t| (a.min(t), b.max(t)));
            let emax = (0..nv).map(|q| (0..nc).map(|c| { let z = smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum::<f64>()).fold(0.0, f64::max);
            let fit: Vec<usize> = (0..nv).filter(|&q| (0..nc).map(|c| { let z = smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum::<f64>() > 1e-3 * emax).step_by(13).collect();
            let om = &grid.omega;
            let psi = |v: usize, t: f64| om[v] * t;
            let (sp, at, s0, ds) = build_segments(nv, nseg, tmin, tmax, 128, &fit, &psi);
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
        let tau: Vec<f32> = set.tau.iter().map(|&v| v as f32).collect();
        let mut yh = vec![C32::new(0.0, 0.0); nc * m];
        for (ci, &c) in keep.iter().enumerate() {
            yh[ci * m..(ci + 1) * m].copy_from_slice(&set.y[c * m..(c + 1) * m]);
        }
        let coords = set.coords.clone();
        drop(set);
        let pts = eng.points2d(&st, [&coords[0], &coords[1], &coords[2]], &tau, &vec![0f32; m], &[], &t, &ri, &rv).unwrap();
        drop((coords, tau, t, ri, rv));
        if timing.is_empty() {
            let tt = Instant::now();
            let (kf, ka) = eng.autotune(&st, seg, &pts).unwrap();
            println!("  autotune {:.1}s: tile kernels {kf:.1} / {ka:.1} ms (knot 0, one coil batch of {})", tt.elapsed().as_secs_f64(), eng.cb);
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
