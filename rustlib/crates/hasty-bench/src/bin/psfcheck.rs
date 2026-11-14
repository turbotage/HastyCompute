//! Effective point-spread function of the warm-start solver path: plain SENSE NUFFT of one encode pooled over all
//! frames (the real sampling of a simulated scan), y = A δ_q0, x_k after k PCGLS iterations with the circulant
//! preconditioner of the warm start (or none / other floors). Tells whether the time-averaged images are
//! convergence-limited (broad PSF → static tissue blended into the lumen) and whether the preconditioner does its job.
//!
//!   psfcheck <scan dir> --point i,j,k [--enc 0] [--iters 100] [--eps 1e-3,1e-5,none] [--gnl] [--stride 1]
//!            [--checkpoints 1,2,5,10,20,50,100] [--out dir]
//!
//! Per checkpoint: x(q0) (→ 1 for an exact inverse; < 1 when the row space misses part of δ), the complex sums
//! Re Σ_{|q−q0|≤r} x / Re Σ x for r = 0…12 voxels (the share of a uniform object's estimate at q0 coming from within r:
//! with blood inside radius R and static tissue outside, the velocity phase is shrunk by about this share at r = R),
//! the energy share Σ_{≤r}|x|² / Σ|x|², and the half-maximum radius of |x|. Also the radial profile of the
//! preconditioner's Fourier diagonal λ(k) (before the floor) and the share of k-space clipped by each floor.
//! `--stride s`: every s-th readout (faster, sparser). `--coils n`: the n coils with the largest |S_c(q0)| (device memory:
//! SenseOp and CG hold 5 k-space vectors of nc × samples). Writes psf_<variant>_it<k>.npy (|x|, float32) with --out.

use std::path::PathBuf;
use std::time::Instant;

use hasty_cuda::{Device, C32};
use hasty_recon::cg::pcgls_cb;
use hasty_recon::data::SimData;
use hasty_recon::gpu::Kernels;
use hasty_recon::op::SenseOp;
use hasty_recon::precond::CirculantPrecond;
use hasty_recon::setop::LinOp;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let enc: usize = take("--enc").map(|s| s.parse().unwrap()).unwrap_or(0);
    let iters: usize = take("--iters").map(|s| s.parse().unwrap()).unwrap_or(100);
    let stride: usize = take("--stride").map(|s| s.parse().unwrap()).unwrap_or(1);
    let ncoil_sel: Option<usize> = take("--coils").map(|s| s.parse().unwrap());
    let eps_list: Vec<Option<f64>> = take("--eps").unwrap_or("1e-3,1e-5,none".into()).split(',').map(|s| if s == "none" { None } else { Some(s.parse().unwrap()) }).collect();
    let cps: Vec<usize> = take("--checkpoints").unwrap_or("1,2,5,10,20,50,100".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let point: Vec<usize> = take("--point").expect("--point i,j,k").split(',').map(|s| s.parse().unwrap()).collect();
    let out = take("--out").map(PathBuf::from);
    let gnl = args.iter().position(|a| a == "--gnl").map(|i| { args.remove(i); true }).unwrap_or(false);
    let dir = PathBuf::from(&args[0]);
    let t0 = Instant::now();
    let sd = SimData::load_lazy(&dir).unwrap();
    let (dims, nv, nc_all) = (sd.dims, sd.nv(), sd.nc);
    let q0 = point[0] + dims[0] * (point[1] + dims[1] * point[2]);
    let w = sd.whitener();
    let ros: Vec<usize> = sd.readouts_of(enc, None).into_iter().step_by(stride).collect();
    let set = sd.load_set(&ros, &hasty_recon::kload::LoadOpts { w: &w, cc: None, keep: None, data: false }).unwrap();
    let m = set.coords[0].len();
    let grid = sd.physics_grid(gnl, &w);
    // coil subset: strongest sensitivities at the point (coil-major maps)
    let mut order: Vec<usize> = (0..nc_all).collect();
    order.sort_by(|&a, &b| { let z = grid.smaps[b * nv + q0]; z.re * z.re + z.im * z.im }.partial_cmp(&{ let z = grid.smaps[a * nv + q0]; z.re * z.re + z.im * z.im }).unwrap());
    let nc = ncoil_sel.unwrap_or(nc_all).min(nc_all);
    let keep = &order[..nc];
    let tot_e: f32 = (0..nc_all).map(|c| { let z = grid.smaps[c * nv + q0]; z.re * z.re + z.im * z.im }).sum();
    let kept_e: f32 = keep.iter().map(|&c| { let z = grid.smaps[c * nv + q0]; z.re * z.re + z.im * z.im }).sum();
    let smaps: Vec<C32> = keep.iter().flat_map(|&c| grid.smaps[c * nv..(c + 1) * nv].iter().cloned()).collect();
    println!("coils {nc} of {nc_all} (strongest at the point: {:.1} % of Σ|S(q0)|²)", 100.0 * kept_e / tot_e);
    println!("{}: encode {enc}, {} readouts × {} samples = {m} samples, {nc_all} coils, grid {dims:?}, point {point:?} ({:.1}s)", dir.display(), ros.len(), sd.na, t0.elapsed().as_secs_f64());
    if let Some(o) = &out {
        std::fs::create_dir_all(o).unwrap();
    }
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = std::sync::Arc::new(Kernels::new(&dev).unwrap());
    let mut op = SenseOp::new(&dev, &st, kern.clone(), dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &smaps, nc, vec![], 1e-4).unwrap();
    let mut d = vec![C32::new(0.0, 0.0); nv];
    d[q0] = C32::new(1.0, 0.0);
    let dx = st.clone_htod(&d).unwrap();
    let mut y = st.alloc_zeros::<C32>(op.len_y()).unwrap();
    op.forward(&st, &dx, &mut y).unwrap();
    // distance (voxels) of every voxel to q0
    let dist: Vec<f64> = (0..nv).map(|q| {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        let dd = [i as f64 - point[0] as f64, j as f64 - point[1] as f64, k as f64 - point[2] as f64];
        (dd[0] * dd[0] + dd[1] * dd[1] + dd[2] * dd[2]).sqrt()
    }).collect();
    let radii = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0];
    let report = |tag: &str, k: usize, h: &[C32]| {
        let tot: f64 = h.iter().map(|z| z.re as f64).sum();
        let etot: f64 = h.iter().map(|z| (z.re * z.re + z.im * z.im) as f64).sum();
        let share: Vec<String> = radii.iter().map(|&r| {
            let s: f64 = (0..nv).filter(|&q| dist[q] <= r + 1e-9).map(|q| h[q].re as f64).sum();
            format!("{:.3}", s / tot)
        }).collect();
        let eshare: Vec<String> = radii.iter().map(|&r| {
            let s: f64 = (0..nv).filter(|&q| dist[q] <= r + 1e-9).map(|q| (h[q].re * h[q].re + h[q].im * h[q].im) as f64).sum();
            format!("{:.3}", s / etot)
        }).collect();
        let a0 = (h[q0].re * h[q0].re + h[q0].im * h[q0].im).sqrt() as f64;
        // half-maximum radius: largest distance with |x| ≥ |x(q0)| / 2 among voxels within 12 voxels
        let hm = (0..nv).filter(|&q| dist[q] <= 12.0 && ((h[q].re * h[q].re + h[q].im * h[q].im).sqrt() as f64) >= 0.5 * a0).map(|q| dist[q]).fold(0.0, f64::max);
        println!("  {tag:>10} it {k:>3}: x(q0) = {:+.3}{:+.3}i, Re-sum share within r = {radii:?}: [{}], energy share [{}], half-max radius {hm:.2} vox, Re Σx {tot:.3}", h[q0].re, h[q0].im, share.join(", "), eshare.join(", "));
    };
    for eps in &eps_list {
        let tag = match eps { Some(e) => format!("eps{e:.0e}"), None => "none".into() };
        let tv = Instant::now();
        let mut pre = eps.map(|e| CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &smaps, nc, e).unwrap());
        if let (Some(p), Some(e)) = (&pre, eps) {
            // radial profile of λ (before the floor): |k| as a fraction of each axis' Nyquist edge
            let lmax = p.lambda.iter().cloned().fold(0f32, f32::max) as f64;
            let kr = |q: usize| -> f64 {
                let c = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
                (0..3).map(|a| { let f = if c[a] <= dims[a] / 2 { c[a] as f64 } else { c[a] as f64 - dims[a] as f64 } / (dims[a] as f64 / 2.0); f * f }).sum::<f64>().sqrt()
            };
            let nb = 12;
            let mut bins: Vec<Vec<f32>> = vec![vec![]; nb];
            for q in 0..nv {
                let b = ((kr(q) / 1.2) * nb as f64) as usize;
                if b < nb {
                    bins[b].push(p.lambda[q]);
                }
            }
            let prof: Vec<String> = bins.iter_mut().enumerate().map(|(b, v)| {
                if v.is_empty() { return format!("{:.1}:-", (b as f64 + 0.5) * 1.2 / nb as f64); }
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = v[v.len() / 2] as f64 / lmax;
                let clip = v.iter().filter(|&&l| (l as f64) < e * lmax).count() as f64 / v.len() as f64;
                format!("{:.1}:{med:.1e}/{:.0}%", (b as f64 + 0.5) * 1.2 / nb as f64, 100.0 * clip)
            }).collect();
            println!("  {tag}: λ(k) median / λmax and % below the floor per |k| shell (fraction of Nyquist): {}; dynamic range used {:.2e}", prof.join(" "), p.dyn_range);
        }
        let mut x = st.alloc_zeros::<C32>(nv).unwrap();
        let mut saved: Vec<(usize, Vec<C32>)> = vec![];
        let n_it = iters.min(*cps.iter().max().unwrap());
        pcgls_cb(&mut op, pre.as_mut(), &st, &y, &mut x, 0.0, n_it, &mut |k, xk| {
            if cps.contains(&k) {
                saved.push((k, st.clone_dtoh(xk).unwrap()));
            }
        })
        .unwrap();
        for (k, h) in &saved {
            report(&tag, *k, h);
            if let Some(o) = &out {
                let a: Vec<f32> = h.iter().map(|z| (z.re * z.re + z.im * z.im).sqrt()).collect();
                let bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
                let hdr = format!("{{'descr': '<f4', 'fortran_order': True, 'shape': ({}, {}, {}), }}", dims[0], dims[1], dims[2]);
                let mut h2 = hdr.into_bytes();
                while (h2.len() + 11) % 64 != 0 {
                    h2.push(b' ');
                }
                h2.push(b'\n');
                let mut f = vec![0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0];
                f.extend_from_slice(&(h2.len() as u16).to_le_bytes());
                f.extend_from_slice(&h2);
                f.extend_from_slice(&bytes);
                std::fs::write(o.join(format!("psf_{tag}_it{k}.npy")), f).unwrap();
            }
        }
        println!("  {tag}: {:.1}s", tv.elapsed().as_secs_f64());
    }
}
