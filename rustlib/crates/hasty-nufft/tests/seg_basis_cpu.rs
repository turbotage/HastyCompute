//! CPU: how many time segments does each basis need? Fixed-time diagonals with least-squares interpolators
//! (`build_segments`, current), the optimal rank-L basis (`build_segments_svd`) and the basis weighted by the k-space
//! data energy per readout time (`build_segments_svd_weighted`, time weights), measured as the separable
//! approximation error of e^{−iψ(v, τ)} over the support voxels at the interpolator table entries:
//! relative RMS over (voxels × τ), the worst relative error of one τ, and the data-weighted RMS (each τ weighted by the
//! measured data energy at that ADC time, a proxy for the data-space model error). Field: the simulator's field map of
//! factory/ph0001 (3 T, 3.56 ms readout); energy profile: data/hasty/research/seg_basis/adc_energy_sim_test.npy
//! (written from sim_test's data_clean.npy). Smooth synthetic field without data weights as a second case.
//!   cargo test --release -p hasty-nufft --test seg_basis_cpu -- --nocapture

use std::f64::consts::PI;
use std::path::Path;

use hasty_cuda::C32;
use hasty_nufft::{build_segments, build_segments_svd, build_segments_svd_weighted};
use num_complex::Complex64;

/// minimal .npy reader: little-endian f4 / f8 → f64 (memory order is irrelevant here: voxels are an unordered set)
fn npy(path: &Path) -> Option<(Vec<usize>, Vec<f64>)> {
    let b = std::fs::read(path).ok()?;
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let h = std::str::from_utf8(&b[10..10 + hl]).ok()?;
    let shape: Vec<usize> = h.split("'shape': (").nth(1)?.split(')').next()?.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let d = &b[10 + hl..];
    let v = if h.contains("'<f4'") {
        d.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect()
    } else if h.contains("'<f8'") {
        d.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
    } else {
        return None;
    };
    Some((shape, v))
}

/// (relative RMS over voxels × table entries, worst relative error of one entry, data-weighted RMS)
#[allow(clippy::too_many_arguments)]
fn model_error(segph: &[C32], atab: &[C32], nseg: usize, nv: usize, t0: f64, dt: f64, ntab: usize, vox: &[usize], g: &(dyn Fn(usize, f64) -> Complex64 + Sync), energy: Option<&[f64]>) -> (f64, f64, f64) {
    use rayon::prelude::*;
    let rows: Vec<(f64, f64, f64)> = (0..ntab)
        .into_par_iter()
        .step_by(3)
        .map(|i| {
            let t = t0 + i as f64 * dt;
            let (mut n, mut d) = (0.0, 0.0);
            for &v in vox {
                let mut z = Complex64::new(0.0, 0.0);
                for l in 0..nseg {
                    let a = atab[i * nseg + l];
                    let p = segph[l * nv + v];
                    z += Complex64::new(a.re as f64, a.im as f64) * Complex64::new(p.re as f64, p.im as f64);
                }
                let e = g(v, t);
                n += (z - e).norm_sqr();
                d += e.norm_sqr();
            }
            (n, d, energy.map_or(1.0, |e| e[i.min(e.len() - 1)]))
        })
        .collect();
    let (num, den) = rows.iter().fold((0.0, 0.0), |(a, b), r| (a + r.0, b + r.1));
    let worst = rows.iter().map(|r| (r.0 / r.1).sqrt()).fold(0.0f64, f64::max);
    let (wn, wd) = rows.iter().fold((0.0, 0.0), |(a, b), r| (a + r.2 * r.0, b + r.2 * r.1));
    ((num / den).sqrt(), worst, (wn / wd).sqrt())
}

#[allow(clippy::too_many_arguments)]
fn study(label: &str, nv: usize, support: &[usize], om: &(dyn Fn(usize) -> f64 + Sync), t_min: f64, t_max: f64, ntab: usize, ls: &[usize], energy: Option<&[f64]>) {
    let fit: Vec<usize> = support.iter().copied().step_by(3).collect();
    let probe: Vec<usize> = support.iter().copied().step_by(7).collect();
    let g = |v: usize, t: f64| Complex64::from_polar(1.0, -om(v) * t);
    let span = {
        let mut w: Vec<f64> = support.iter().map(|&v| om(v)).collect();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        (w[w.len() * 99 / 100] - w[w.len() / 100]) / (2.0 * PI)
    };
    eprintln!("{label}: {} support voxels, 1–99 % span {span:.0} Hz over {:.2} ms = {:.2} phase cycles", support.len(), (t_max - t_min) * 1e3, span * (t_max - t_min));
    eprintln!("   L | fixed times: rms / worst τ / data-weighted | SVD: rms / worst / data-weighted | SVD, data-energy τ weights: rms / worst / data-weighted");
    let dt_tab = (t_max - t_min) / (ntab - 1) as f64;
    let tw = |t: f64| -> f64 { energy.map_or(1.0, |e| e[(((t - t_min) / dt_tab).round().max(0.0) as usize).min(e.len() - 1)]) };
    for &l in ls {
        let (sp, at, t0, dt) = build_segments(nv, l, t_min, t_max, ntab, &fit, &|v, t| om(v) * t);
        let a = model_error(&sp, &at, l, nv, t0, dt, ntab, &probe, &g, energy);
        let (sp, at, t0, dt) = build_segments_svd(nv, l, t_min, t_max, ntab, &fit, 64, &g);
        let b = model_error(&sp, &at, l, nv, t0, dt, ntab, &probe, &g, energy);
        let c = if energy.is_some() {
            let (sp, at, t0, dt) = build_segments_svd_weighted(nv, l, t_min, t_max, ntab, &fit, None, Some(&tw), 96, &g);
            Some(model_error(&sp, &at, l, nv, t0, dt, ntab, &probe, &g, energy))
        } else {
            None
        };
        let f = |e: (f64, f64, f64)| format!("{:.2e} / {:.2e} / {:.2e}", e.0, e.1, e.2);
        eprintln!("  {l:>2} | {} | {} | {}", f(a), f(b), c.map_or("-".into(), f));
    }
}

#[test]
fn segments_fixed_times_vs_svd_basis() {
    // smooth synthetic field, 8 ms readout
    let dims = [40usize, 36, 24];
    let nv: usize = dims.iter().product();
    let om_s = |v: usize| {
        let (x, y, z) = ((v % dims[0]) as f64 / dims[0] as f64 - 0.5, ((v / dims[0]) % dims[1]) as f64 / dims[1] as f64 - 0.5, (v / (dims[0] * dims[1])) as f64 / dims[2] as f64 - 0.5);
        2.0 * PI * 450.0 * (0.8 * x + 0.6 * (y * y + z * z) * 4.0 - 0.3)
    };
    let all: Vec<usize> = (0..nv).collect();
    study("smooth 450 Hz", nv, &all, &om_s, 1.6e-3, 9.6e-3, 1000, &[6, 8, 10], None);
    // the simulator's field map (ppm on the reconstruction grid), factory/ph0001: 3 T, readout te .. te + n_adc · dwell
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty");
    let dir = root.join("phantom/factory/ph0001/sim_test/scan");
    let energy = npy(&root.join("research/seg_basis/adc_energy_sim_test.npy")).map(|(_, e)| e);
    if let Some((shape, ppm)) = npy(&dir.join("field_ppm_recon.npy")) {
        let nv = ppm.len();
        let f0 = 42.577e6 * 3.0;
        let om = |v: usize| 2.0 * PI * f0 * 1e-6 * ppm[v];
        let support: Vec<usize> = (0..nv).filter(|&v| ppm[v] != 0.0).collect();
        let (te, n_adc) = (2.224e-3, 1782usize);
        study(&format!("factory ph0001 field map {shape:?}"), nv, &support, &om, te, te + (n_adc - 1) as f64 * 2e-6, n_adc, &[4, 5, 6, 7, 8, 9, 10], energy.as_deref());
    } else {
        eprintln!("field map not present: skipped");
    }
}
