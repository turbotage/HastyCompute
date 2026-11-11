//! CPU dry run of the coarse-to-fine pyramid (recon_p2np --grid-scale s --kcrop --init-from; hasty_recon::pyramid):
//! per level the grid, the grid-dependent inputs as recon_p2np builds them (coil maps, field, R2*, fat fraction,
//! lumen, through the same library calls), the k-space samples kept by the crop (from the stored trajectory) and a
//! cost model scaled from measured full-resolution numbers. No GPU, no k-space data read.
//!   pyramid_dry <sim_dir> [--scales 0.25,0.5,1] [--kcrop FRAC] [--gnl] [--every n] [--no-maps]
//!               [--t-set 1.7] [--t-grid 0.09] [--l-full 22] [--coils 31] [--sets 280]
//! --every n: every n-th readout of the trajectory for the kept fractions (default 8; readouts are rotations of one
//!   yarn shape, so a stride is exact up to the direction sampling). --t-set / --t-grid: measured seconds per set and
//!   gradient at the full level (all of it / the gridding part, the rest FFTs); --l-full: its segment count.
//! Cost model per set and gradient: FFTs ∝ L · N_os log N_os (the coil count is the same at every level), gridding ∝
//! kept samples · L (each sample spreads into the L segment channels of its coil batch). L at a cropped level: the
//! auto rule ⌈1.3·cycles + 3.5⌉ with the phase cycles of the full level scaled by the kept ADC window.

use std::path::PathBuf;

use hasty_recon::data::SimData;
use hasty_recon::kload::NpyFile;
use hasty_recon::pyramid::{maps_on_recon_grid, KCrop};
use rayon::prelude::*;

fn smooth_grid(n: usize, sigma: f64, w: usize) -> usize {
    hasty_nufft::next_smooth_even(((sigma * n as f64).ceil() as usize).max(2 * w))
}

fn pct(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let gnl = args.iter().position(|a| a == "--gnl").map(|i| { args.remove(i); true }).unwrap_or(false);
    let no_maps = args.iter().position(|a| a == "--no-maps").map(|i| { args.remove(i); true }).unwrap_or(false);
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let scales: Vec<f64> = take("--scales").unwrap_or("0.25,0.5,1".into()).split(',').map(|s| s.parse().unwrap()).collect();
    let kc = KCrop::parse(&take("--kcrop").unwrap_or_default()).unwrap();
    let every: usize = take("--every").map_or(8, |v| v.parse().unwrap());
    let t_set: f64 = take("--t-set").map_or(1.7, |v| v.parse().unwrap());
    let t_grid: f64 = take("--t-grid").map_or(0.09, |v| v.parse().unwrap());
    let l_full: usize = take("--l-full").map_or(22, |v| v.parse().unwrap());
    let coils: usize = take("--coils").map_or(31, |v| v.parse().unwrap());
    let sets: usize = take("--sets").map_or(280, |v| v.parse().unwrap());
    let dir = PathBuf::from(&args[0]);
    let t0 = std::time::Instant::now();
    let mut sd = SimData::load_lazy(&dir).unwrap();
    let (na, n_ro) = (sd.na, sd.n_readouts());
    println!("{}: native grid {:?} at {:.1} mm, {} readouts × {na} ADC samples, {} coils ({:.1}s)", dir.display(), sd.tdims, sd.res * 1e3, n_ro, sd.nc, t0.elapsed().as_secs_f64());
    let w = sd.whitener();
    // the trajectory: samples.npy (k physical rad/m + time) when present, else the per-sample records
    let (traj, stride) = match NpyFile::open(&dir.join("samples.npy")) {
        Ok(f) => (f, 4usize),
        Err(_) => (NpyFile::open(&dir.join("sched_nominal.npy")).unwrap(), hasty_mrsim::sequence::SAMPLE_FLOATS),
    };
    let rt = hasty_mrsim::transpose(&sd.rot);
    let ros: Vec<usize> = (0..n_ro).step_by(every.max(1)).collect();
    // logical k (rad/m) and τ (s) of the sampled readouts
    let t_r = std::time::Instant::now();
    let kt: Vec<(Vec<[f64; 3]>, Vec<f64>)> = ros
        .par_iter()
        .map(|&r| {
            let mut b = vec![0f32; na * stride];
            traj.read_f32(r * na * stride, &mut b).unwrap();
            let k = (0..na).map(|n| hasty_mrsim::mat_vec(&rt, &[b[n * stride] as f64, b[n * stride + 1] as f64, b[n * stride + 2] as f64])).collect();
            let tau = (0..na).map(|n| b[n * stride + if stride == 4 { 3 } else { 13 }] as f64).collect();
            (k, tau)
        })
        .collect();
    let tau = &kt[0].1;
    let t_adc = tau[na - 1] - tau[0];
    let kmax = kt.iter().flat_map(|(k, _)| k.iter().map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt())).fold(0.0f64, f64::max);
    println!("trajectory: {} of {n_ro} readouts read ({:.1}s), max |k| {kmax:.1} rad/m (π / res = {:.1}), ADC {:.3}..{:.3} ms", ros.len(), t_r.elapsed().as_secs_f64(), std::f64::consts::PI / sd.res, tau[0] * 1e3, tau[na - 1] * 1e3);
    let full_dims = sd.tdims;
    let n_os_full: Vec<usize> = (0..3).map(|a| smooth_grid(full_dims[a], 1.25, 10)).collect();
    let nos_full = n_os_full.iter().product::<usize>() as f64;
    let cyc_full = (l_full as f64 - 3.75) / 1.3;
    let t_fft_full = t_set - t_grid;
    println!("full level: oversampled grid {n_os_full:?}, measured {t_set:.2} s per set and gradient (gridding {t_grid:.2} s, FFTs {t_fft_full:.2} s), L = {l_full} (≈ {cyc_full:.1} phase cycles over {:.2} ms), {coils} coils, {sets} sets", t_adc * 1e3);
    let mut table = vec![];
    for &s in &scales {
        if (s - 1.0).abs() < 1e-12 {
            sd.dims = sd.tdims;
            sd.vox = [sd.res; 3];
        } else {
            sd.set_recon_grid([s; 3]);
        }
        let (dims, vox) = (sd.dims, sd.vox);
        let nv = sd.nv();
        let n_os: Vec<usize> = (0..3).map(|a| smooth_grid(dims[a], 1.25, 10)).collect();
        let nos = n_os.iter().product::<usize>() as f64;
        println!("\n== level {s}: grid {dims:?} ({nv} voxels), voxel {:?} mm, oversampled {n_os:?} ({:.1}× fewer points)", vox.map(|v| (v * 1e4).round() / 10.0), nos_full / nos);
        // kept samples: ball (prefix and suffix of the out-and-back yarn) and outgoing prefix only
        let crops = [KCrop { frac: kc.frac, prefix: false }, KCrop { frac: kc.frac, prefix: true }];
        let mut kept = [0usize; 2];
        let mut win = [(usize::MAX, 0usize); 2];
        for (ci, c) in crops.iter().enumerate() {
            let per: Vec<(usize, usize, usize)> = kt
                .par_iter()
                .map(|(k, _)| {
                    let coords: [Vec<f32>; 3] = [0, 1, 2].map(|a| k.iter().map(|v| (v[a] * vox[a]) as f32).collect());
                    let sel = c.select(&coords, na);
                    (sel.len(), sel.first().map_or(usize::MAX, |&j| j as usize), sel.last().map_or(0, |&j| j as usize))
                })
                .collect();
            kept[ci] = per.iter().map(|p| p.0).sum();
            win[ci] = (per.iter().map(|p| p.1).min().unwrap_or(0), per.iter().map(|p| p.2).max().unwrap_or(0));
        }
        let total = ros.len() * na;
        let frac = [kept[0] as f64 / total as f64, kept[1] as f64 / total as f64];
        for ci in 0..2 {
            let (a, b) = win[ci];
            println!("  k-space crop {}: kept {:.3} % of the samples, ADC window {a}..{b} ({:.3}..{:.3} ms, {:.2} of the readout)", crops[ci].describe(), 100.0 * frac[ci], tau[a] * 1e3, tau[b] * 1e3, (tau[b] - tau[a]) / t_adc);
        }
        // cost model
        for ci in 0..2 {
            let (a, b) = win[ci];
            let wfrac = (tau[b] - tau[a]) / t_adc;
            let l = if s == 1.0 && frac[ci] > 0.999 { l_full } else { ((1.3 * cyc_full * wfrac + 3.5).ceil() as usize).max(2).min(l_full) };
            let fft = t_fft_full * (l as f64 / l_full as f64) * (nos * nos.ln()) / (nos_full * nos_full.ln());
            let grid = t_grid * frac[ci] * (l as f64 / l_full as f64);
            let data_gib = frac[ci] * (n_ro * na) as f64 * coils as f64 * 8.0 / (1u64 << 30) as f64;
            let tab_gib = frac[ci] * (n_ro * na) as f64 * 200.0 / (1u64 << 30) as f64;
            let grid_gib = l as f64 * nos * 8.0 * coils.min(16) as f64 / (1u64 << 30) as f64;
            println!("  cost ({}): L ≈ {l}, per set and gradient {:.4} s (FFTs {:.4} + gridding {:.4}) → {:.1} s per iteration over {sets} sets ({:.1}× the full level); kept data {:.2} GiB ({coils} coils), sample tables {:.2} GiB, segment grids of a 16-coil batch {:.2} GiB",
                if crops[ci].prefix { "prefix" } else { "ball" }, fft + grid, fft, grid, sets as f64 * (fft + grid), t_set / (fft + grid), data_gib, tab_gib, grid_gib);
            table.push((s, crops[ci].prefix, dims, n_os.clone(), frac[ci], l, fft + grid, sets as f64 * (fft + grid), data_gib));
        }
        if no_maps || (s - 1.0).abs() < 1e-12 {
            continue;
        }
        // grid-dependent inputs, as recon_p2np builds them
        let tm = std::time::Instant::now();
        let grid = sd.physics_grid(gnl, &w);
        let e: Vec<f32> = (0..nv).map(|q| (0..sd.nc).map(|c| { let z = grid.smaps[c * nv + q]; z.re * z.re + z.im * z.im }).sum()).collect();
        let mut ev: Vec<f32> = e.iter().copied().filter(|&x| x > 0.0).collect();
        ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = ev.get(ev.len() / 2).copied().unwrap_or(0.0);
        let sig: Vec<usize> = (0..nv).filter(|&q| e[q] > 0.05 * med).collect();
        println!("  coil maps (truth, whitened, at the {} positions): {} × {nv}, signal voxels {} ({:.1}s)", if gnl { "GNL-true" } else { "voxel" }, sd.nc, sig.len(), tm.elapsed().as_secs_f64());
        let tw = 2.0 * std::f64::consts::PI;
        let wfac = tw * hasty_mrsim::GAMMA_BAR * sd.b0 * 1e-6;
        let mut hz_p: Vec<f64> = sig.iter().map(|&q| grid.omega[q] / tw).collect();
        let mut hz_b: Vec<f64> = sig.iter().map(|&q| wfac * sd.sample_log_box(&sd.field_ppm, 0, &grid.x_log[q]) / tw).collect();
        let (p1, p99) = (pct(&mut hz_p, 0.01), pct(&mut hz_p, 0.99));
        let (b1, b99) = (pct(&mut hz_b, 0.01), pct(&mut hz_b, 0.99));
        let (pmin, pmax) = (hz_p[0], *hz_p.last().unwrap());
        let (bmin, bmax) = (hz_b[0], *hz_b.last().unwrap());
        println!("  field (truth, field_ppm_recon on {:?}): over the signal voxels point {pmin:.0}..{pmax:.0} Hz (1–99 %: {p1:.0}..{p99:.0}), voxel box {bmin:.0}..{bmax:.0} Hz ({b1:.0}..{b99:.0}); B0-only cycles over the full readout {:.2} (point) / {:.2} (box)", sd.tdims, (pmax - pmin) * t_adc, (bmax - bmin) * t_adc);
        let r2 = hasty_recon::data::read_npy(&dir.join("truth_r2s.npy")).map(|a| a.f32s().iter().map(|v| if v.is_finite() { *v } else { 0.0 }).collect::<Vec<f32>>());
        if let Ok(r2) = r2 {
            let mut rp: Vec<f64> = sig.iter().map(|&q| sd.sample_log(&r2, 0, &grid.x_log[q])).collect();
            let mut rb: Vec<f64> = sig.iter().map(|&q| sd.sample_log_box(&r2, 0, &grid.x_log[q])).collect();
            println!("  R2* (truth): {nv} values, median over the signal voxels {:.1} (point) / {:.1} (box) 1/s", pct(&mut rp, 0.5), pct(&mut rb, 0.5));
        }
        let ffp = dir.join("truth_fat_fraction.npy");
        if ffp.exists() {
            for boxed in [false, true] {
                let tf = std::time::Instant::now();
                let (ff, res) = maps_on_recon_grid(&sd, &ffp, &grid.x_log, boxed).unwrap();
                let nmap = ff.len() / nv;
                let fat = sig.iter().filter(|&&q| ff[q] > 0.05).count();
                let mean = sig.iter().map(|&q| ff[q] as f64).sum::<f64>() / sig.len().max(1) as f64;
                println!("  fat fraction ({}): {} {} maps × {nv} voxels ({}{}), frame 0: {fat} signal voxels with ff > 5 %, mean ff {mean:.4} ({:.1}s)", ffp.file_name().unwrap().to_string_lossy(), if res { "resampled to" } else { "as stored:" }, nmap, if boxed { "voxel box" } else { "point" }, if res { "" } else { " — recon grid" }, tf.elapsed().as_secs_f64());
            }
        }
        let lp = (0..nv).filter(|&q| sd.sample_log(&sd.truth_frac, 0, &grid.x_log[q]) > 0.9).count();
        let lb = (0..nv).filter(|&q| sd.sample_log_box(&sd.truth_frac, 0, &grid.x_log[q]) > 0.9).count();
        println!("  lumen (truth fraction > 0.9): {lp} voxels (point) / {lb} (voxel box)");
    }
    println!("\nlevel | crop   | grid          | oversampled    | kept %  | L  | s / set | s / iteration | kept data GiB");
    for (s, pre, d, o, f, l, ts, ti, g) in &table {
        println!("{s:>5} | {:<6} | {:>3}×{:>3}×{:>3} | {:>3}×{:>3}×{:>3}    | {:>7.3} | {l:>2} | {ts:>7.4} | {ti:>13.1} | {g:>8.2}", if *pre { "prefix" } else { "ball" }, d[0], d[1], d[2], o[0], o[1], o[2], 100.0 * f);
    }
    println!("(full level as measured: {t_set:.2} s per set, {:.0} s per iteration)", t_set * sets as f64);
}
