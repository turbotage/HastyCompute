//! Noise-referenced time-segment bases and segment counts (additions; `research/segment_basis.md`).
//!
//! The separable model g(v, τ) ≈ Σ_l a_l(τ) D_l(v) of the per-voxel time function (off-resonance, concomitant mean,
//! fat, decay) is only as good as it has to be in the data: a sample's model error matters relative to the noise σ
//! of that sample, and how much of the voxel-wise error reaches a sample depends on the image the operator sees and
//! on where the sample is in k-space. For an out-and-back readout almost all of the data energy is in the first and
//! last ≈ 50 ADC samples (the k-space centre); the middle of the readout sits at the noise level.
//!
//! * [`seg_time_grid`]: time samples for the fit that resolve a strongly peaked per-ADC-sample weight, with
//!   quadrature weights (Σ_i tw_i f(t_i) ≈ Σ_n w_n f(τ_n)) — the uniform 96-point grid of
//!   `build_segments_svd_weighted` puts one sample on the 40-sample k-space-centre peak;
//! * [`weighted_spectrum`] / [`build_segments_svd_grid`]: the weighted rank-L basis on that grid (voxel weights w_r,
//!   time weights tw_i) and the eigenvalues of its Gram: the weighted squared error of the best rank-L separable
//!   model is the tail Σ_{l ≥ L} λ_l, so the segment count for a target follows without building any segments
//!   ([`seg_count_for`]);
//! * [`snr_time_weights`]: per-ADC-sample weights 1 + SNR(n) (data energy over noise energy per sample) — the error
//!   of a sample whose signal is at the noise level only matters at the O(1) level, that of a k-space-centre sample
//!   with SNR 10⁵ at the 10⁻³ level.

use hasty_cuda::C32;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use rayon::prelude::*;

/// Time samples (s) and quadrature weights for a weighted fit over the ADC window `tau` (ascending ADC sample times)
/// with a non-negative weight `w` per ADC sample: about a third of the `ns` samples uniform over the window, the
/// rest at equal steps of the cumulative √w (dense where the weight is concentrated), merged; every ADC sample is
/// assigned to its nearest grid time and the grid time's weight is the sum of the weights of its ADC samples, so
/// Σ_i tw_i f(t_i) ≈ Σ_n w_n f(τ_n) for any f smooth on the grid spacing. Always contains both window ends.
pub fn seg_time_grid(tau: &[f64], w: &[f64], ns: usize) -> (Vec<f64>, Vec<f64>) {
    let na = tau.len();
    assert!(na >= 2 && w.len() == na, "one weight per ADC sample");
    let ns = ns.clamp(2, na);
    let nu = (ns / 3).max(2);
    let mut idx: Vec<usize> = (0..nu).map(|i| ((na - 1) as f64 * i as f64 / (nu - 1) as f64).round() as usize).collect();
    // cumulative √w (sharper peaks get proportionally fewer points than with w itself: √ balances resolution and reach)
    let mut cum = vec![0.0f64; na];
    let mut acc = 0.0;
    for n in 0..na {
        acc += w[n].max(0.0).sqrt();
        cum[n] = acc;
    }
    let nw = ns.saturating_sub(nu).max(1);
    if acc > 0.0 {
        let mut j = 0usize;
        for i in 0..nw {
            let target = acc * (i as f64 + 0.5) / nw as f64;
            while j + 1 < na && cum[j] < target {
                j += 1;
            }
            idx.push(j);
        }
    }
    idx.sort_unstable();
    idx.dedup();
    // quadrature: nearest grid sample (in ADC index) of every ADC sample
    let mut tw = vec![0.0f64; idx.len()];
    let mut g = 0usize;
    for n in 0..na {
        while g + 1 < idx.len() && (idx[g + 1] as i64 - n as i64).abs() <= (n as i64 - idx[g] as i64).abs() {
            g += 1;
        }
        tw[g] += w[n].max(0.0);
    }
    (idx.iter().map(|&i| tau[i]).collect(), tw)
}

/// Per-ADC-sample weights 1 + E(n) / (C σ²): `energy` the measured data energy per ADC sample (coil sum, mean over
/// readouts), `noise` the noise energy per sample C σ² (same units). Floor 1: the model error of a sample is never
/// weighted below its noise.
pub fn snr_time_weights(energy: &[f64], noise: f64) -> Vec<f64> {
    energy.iter().map(|&e| 1.0 + e.max(0.0) / noise.max(1e-300)).collect()
}

/// Volume-weighted mean data energy over the sampled k-space ball: the energy per sample `energy[n]` (ADC sample n)
/// weighted by the shell volume 4πκ_n² |dκ/dn| of that sample, `kap[n]` = |k| of ADC sample n (rad/m; an
/// out-and-back readout visits every shell twice, both passes count). By Parseval this is (up to the corners of the
/// k-space cube) Σ_v |x|² Σ_c |S_c|²: the energy per sample of an error image that is spatially white.
pub fn vol_energy(energy: &[f64], kap: &[f64]) -> f64 {
    let n = energy.len();
    assert!(n >= 2 && kap.len() == n);
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for i in 0..n {
        let dk = 0.5 * (kap[(i + 1).min(n - 1)] - kap[i.saturating_sub(1)]).abs();
        let dv = kap[i] * kap[i] * dk;
        num += energy[i] * dv;
        den += dv;
    }
    num / den.max(1e-300)
}

/// Per-ADC-sample weights of the noise-referenced segment fit: 1 + (E(n) + E_vol) / (C σ²) with E_vol =
/// [`vol_energy`]. The model error of a sample is (a) proportional to that sample's signal when the voxel-wise error
/// is smooth over the object (E(n): the k-space centre, SNR up to 10⁵) and (b) at least the white level E_vol ε² when
/// it is rough (field interfaces; dominates where the signal itself is low — the middle of the readout); 1: never
/// below the noise. Measured on ph0001 against the exact data-space error (research/segment_basis.md): (a) alone
/// ([`snr_time_weights`]) leaves the mid-readout error unweighted and underestimates the data error 2–4×.
pub fn noise_time_weights(energy: &[f64], kap: &[f64], noise: f64) -> Vec<f64> {
    let ev = vol_energy(energy, kap);
    energy.iter().map(|&e| 1.0 + (e.max(0.0) + ev) / noise.max(1e-300)).collect()
}

/// Weighted Gram eigen-decomposition: G̃[r, i] = √w_r g(v_r, t_i) √tw_i over the fit voxels; returns the eigenvalues
/// of G̃ᴴG̃ (descending) and the matching eigenvectors (ns × ns, columns), plus √tw.
fn weighted_gram(fit_voxels: &[usize], voxel_w: Option<&[f64]>, ts: &[f64], tw: &[f64], g: &(dyn Fn(usize, f64) -> Complex64 + Sync)) -> (Vec<f64>, DMatrix<Complex64>, Vec<f64>) {
    let ns = ts.len();
    assert_eq!(tw.len(), ns);
    if let Some(w) = voxel_w {
        assert_eq!(w.len(), fit_voxels.len(), "one weight per fit voxel");
    }
    let sw = |r: usize| voxel_w.map_or(1.0, |w| w[r].max(0.0).sqrt());
    let st: Vec<f64> = tw.iter().map(|&x| x.max(0.0).sqrt()).collect();
    let zero = || vec![Complex64::new(0.0, 0.0); ns * ns];
    let idx: Vec<usize> = (0..fit_voxels.len()).collect();
    let gram = idx
        .par_chunks(1024)
        .map(|chunk| {
            let mut acc = zero();
            let mut row = vec![Complex64::new(0.0, 0.0); ns];
            for &r in chunk {
                let (v, w) = (fit_voxels[r], sw(r));
                if w == 0.0 {
                    continue;
                }
                for (i, t) in ts.iter().enumerate() {
                    row[i] = g(v, *t) * (w * st[i]);
                }
                for i in 0..ns {
                    let ci = row[i].conj();
                    for j in i..ns {
                        acc[i * ns + j] += ci * row[j];
                    }
                }
            }
            acc
        })
        .reduce(zero, |mut a, b| {
            a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
            a
        });
    let mut c = DMatrix::<Complex64>::zeros(ns, ns);
    for i in 0..ns {
        for j in i..ns {
            c[(i, j)] = gram[i * ns + j];
            c[(j, i)] = gram[i * ns + j].conj();
        }
    }
    let eig = nalgebra::SymmetricEigen::new(c);
    let mut order: Vec<usize> = (0..ns).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap());
    let ev: Vec<f64> = order.iter().map(|&o| eig.eigenvalues[o].max(0.0)).collect();
    let vecs = DMatrix::<Complex64>::from_fn(ns, ns, |i, l| eig.eigenvectors[(i, order[l])]);
    (ev, vecs, st)
}

/// Eigenvalues (descending) of the weighted Gram of g over the fit voxels on the time grid (`ts`, `tw`, e.g. from
/// [`seg_time_grid`]): Σ_{l ≥ L} λ_l is the weighted squared error Σ_i tw_i Σ_r w_r |g − Σ_{l<L} D_l b_l|² of the
/// best rank-L separable model, Σ_l λ_l = Σ_i tw_i Σ_r w_r |g|².
pub fn weighted_spectrum(fit_voxels: &[usize], voxel_w: Option<&[f64]>, ts: &[f64], tw: &[f64], g: &(dyn Fn(usize, f64) -> Complex64 + Sync)) -> Vec<f64> {
    weighted_gram(fit_voxels, voxel_w, ts, tw, g).0
}

/// Relative weighted error √(Σ_{l ≥ L} λ_l / Σ_l λ_l) of the best rank-L model for L = 0 ..= ev.len()
pub fn tail_errors(ev: &[f64]) -> Vec<f64> {
    let tot: f64 = ev.iter().sum::<f64>().max(1e-300);
    let mut tail = vec![0.0f64; ev.len() + 1];
    for l in (0..ev.len()).rev() {
        tail[l] = tail[l + 1] + ev[l];
    }
    tail.iter().map(|t| (t / tot).sqrt()).collect()
}

/// Smallest L in [lmin, lmax] whose relative weighted error ([`tail_errors`]) is ≤ `tol` (lmax if none is).
pub fn seg_count_for(ev: &[f64], tol: f64, lmin: usize, lmax: usize) -> usize {
    let e = tail_errors(ev);
    (lmin..=lmax.min(ev.len())).find(|&l| e[l] <= tol).unwrap_or(lmax)
}

/// Weighted rank-L time segments on an explicit time grid: as `build_segments_svd_weighted` (voxel weights w_r,
/// diagonals D_l(v) = Σ_i g(v, t_i) √tw_i Ṽ_il for every voxel, least-squares interpolators refitted with the voxel
/// weights at every one of the `ntab` table entries), but with the time samples and their (quadrature × importance)
/// weights given, e.g. from [`seg_time_grid`] with [`snr_time_weights`]. Same output layout as `build_segments`.
#[allow(clippy::too_many_arguments)]
pub fn build_segments_svd_grid(
    nv: usize,
    nseg: usize,
    t_min: f64,
    t_max: f64,
    ntab: usize,
    fit_voxels: &[usize],
    voxel_w: Option<&[f64]>,
    ts: &[f64],
    tw: &[f64],
    g: &(dyn Fn(usize, f64) -> Complex64 + Sync),
) -> (Vec<C32>, Vec<C32>, f64, f64) {
    let ns = ts.len();
    assert!(nseg >= 1 && nseg <= ns, "nseg {nseg} needs at least as many time samples ({ns})");
    let (_, vecs, st) = weighted_gram(fit_voxels, voxel_w, ts, tw, g);
    let v = DMatrix::<Complex64>::from_fn(ns, nseg, |i, l| vecs[(i, l)]);
    let mut segph = vec![C32::new(0.0, 0.0); nseg * nv];
    let rows: Vec<Vec<C32>> = (0..nv)
        .into_par_iter()
        .with_min_len(512)
        .map(|vx| {
            let gv: Vec<Complex64> = ts.iter().zip(&st).map(|(&t, &w)| g(vx, t) * w).collect();
            (0..nseg)
                .map(|l| {
                    let z: Complex64 = (0..ns).map(|i| gv[i] * v[(i, l)]).sum();
                    C32::new(z.re as f32, z.im as f32)
                })
                .collect()
        })
        .collect();
    for (vx, r) in rows.iter().enumerate() {
        for l in 0..nseg {
            segph[l * nv + vx] = r[l];
        }
    }
    let sw = |r: usize| voxel_w.map_or(1.0, |w| w[r].max(0.0).sqrt());
    let b = DMatrix::<Complex64>::from_fn(fit_voxels.len(), nseg, |r, l| {
        let z = segph[l * nv + fit_voxels[r]];
        Complex64::new(z.re as f64, z.im as f64) * sw(r)
    });
    let atab = crate::segnufft::ls_interpolators_pub(&b, nseg, t_min, t_max, ntab, &|t| DVector::<Complex64>::from_iterator(fit_voxels.len(), fit_voxels.iter().enumerate().map(|(r, &vx)| g(vx, t) * sw(r))));
    (segph, atab, t_min, (t_max - t_min) / (ntab - 1) as f64)
}

/// Fixed-time segments (as `build_segments_complex`) at explicit node times `tl` (e.g. clustered at the readout ends
/// where the k-space centre is sampled) with voxel-weighted least-squares interpolators. Same output layout.
#[allow(clippy::too_many_arguments)]
pub fn build_segments_nodes(
    nv: usize,
    tl: &[f64],
    t_min: f64,
    t_max: f64,
    ntab: usize,
    fit_voxels: &[usize],
    voxel_w: Option<&[f64]>,
    g: &(dyn Fn(usize, f64) -> Complex64 + Sync),
) -> (Vec<C32>, Vec<C32>, f64, f64) {
    let nseg = tl.len();
    let mut segph = vec![C32::new(0.0, 0.0); nseg * nv];
    segph.par_chunks_mut(nv.max(1)).zip(tl.par_iter()).for_each(|(row, &t)| {
        for (v, o) in row.iter_mut().enumerate() {
            let z = g(v, t);
            *o = C32::new(z.re as f32, z.im as f32);
        }
    });
    let sw = |r: usize| voxel_w.map_or(1.0, |w| w[r].max(0.0).sqrt());
    let b = DMatrix::<Complex64>::from_fn(fit_voxels.len(), nseg, |r, l| g(fit_voxels[r], tl[l]) * sw(r));
    let atab = crate::segnufft::ls_interpolators_pub(&b, nseg, t_min, t_max, ntab, &|t| DVector::<Complex64>::from_iterator(fit_voxels.len(), fit_voxels.iter().enumerate().map(|(r, &vx)| g(vx, t) * sw(r))));
    (segph, atab, t_min, (t_max - t_min) / (ntab - 1) as f64)
}

/// Chebyshev–Lobatto node times on [t_min, t_max] (both ends included, clustered towards them)
pub fn cheb_nodes(nseg: usize, t_min: f64, t_max: f64) -> Vec<f64> {
    if nseg == 1 {
        return vec![0.5 * (t_min + t_max)];
    }
    (0..nseg).map(|l| 0.5 * (t_min + t_max) - 0.5 * (t_max - t_min) * (std::f64::consts::PI * l as f64 / (nseg - 1) as f64).cos()).collect()
}

/// Relative weighted model error of given segments: √(Σ_n w_n Σ_r w_r |g(v_r, τ_n) − Σ_l a_l(τ_n) D_l(v_r)|² /
/// Σ_n w_n Σ_r w_r |g|²) over the fit voxels at the ADC times `tau` (weights `tw` per ADC time; the interpolators
/// read from the table at the nearest entry, as the kernels do between entries to first order).
#[allow(clippy::too_many_arguments)]
pub fn segment_model_error(segph: &[C32], atab: &[C32], nseg: usize, nv: usize, t0: f64, dt: f64, fit_voxels: &[usize], voxel_w: Option<&[f64]>, tau: &[f64], tw: &[f64], g: &(dyn Fn(usize, f64) -> Complex64 + Sync)) -> f64 {
    let ntab = atab.len() / nseg.max(1);
    let sw = |r: usize| voxel_w.map_or(1.0, |w| w[r].max(0.0));
    let (num, den) = tau
        .par_iter()
        .zip(tw.par_iter())
        .map(|(&t, &w)| {
            if w <= 0.0 {
                return (0.0, 0.0);
            }
            let u = ((t - t0) / dt).clamp(0.0, (ntab - 1) as f64);
            let (i0, fr) = ((u.floor() as usize).min(ntab.saturating_sub(2)), u - (u.floor() as usize).min(ntab.saturating_sub(2)) as f64);
            let a: Vec<Complex64> = (0..nseg)
                .map(|l| {
                    let p = atab[i0 * nseg + l];
                    let q = atab[((i0 + 1).min(ntab - 1)) * nseg + l];
                    Complex64::new(p.re as f64 + fr * (q.re - p.re) as f64, p.im as f64 + fr * (q.im - p.im) as f64)
                })
                .collect();
            let (mut n, mut d) = (0.0, 0.0);
            for (r, &v) in fit_voxels.iter().enumerate() {
                let mut z = Complex64::new(0.0, 0.0);
                for l in 0..nseg {
                    let p = segph[l * nv + v];
                    z += a[l] * Complex64::new(p.re as f64, p.im as f64);
                }
                let e = g(v, t);
                n += sw(r) * (z - e).norm_sqr();
                d += sw(r) * e.norm_sqr();
            }
            (w * n, w * d)
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    (num / den.max(1e-300)).sqrt()
}

/// Relative weighted error of fixed-time segments at node times `tl` with least-squares interpolators weighted by
/// `fit_w` (None: equal, `build_segments` / `build_segments_complex`; Some: [`build_segments_nodes`]), evaluated on
/// the time grid (`ts`, `tw`) with voxel weights `err_w`: √(Σ_i tw_i Σ_r e_r |g(v_r, t_i) − Σ_l a_l(t_i) g(v_r, τ_l)|²
/// / Σ_i tw_i Σ_r e_r |g|²), the interpolators solved at each t_i as the table entries are; the counterpart of
/// [`tail_errors`] for the fixed-time basis (no table, no diagonals for the whole grid: cheap enough to scan L).
pub fn nodes_weighted_error(fit_voxels: &[usize], fit_w: Option<&[f64]>, err_w: Option<&[f64]>, ts: &[f64], tw: &[f64], tl: &[f64], g: &(dyn Fn(usize, f64) -> Complex64 + Sync)) -> f64 {
    let nseg = tl.len();
    let voxel_w = fit_w;
    let sw = |r: usize| voxel_w.map_or(1.0, |w| w[r].max(0.0).sqrt());
    let ew = |r: usize| err_w.map_or(1.0, |w| w[r].max(0.0));
    let b = DMatrix::<Complex64>::from_fn(fit_voxels.len(), nseg, |r, l| g(fit_voxels[r], tl[l]) * sw(r));
    let bh = b.adjoint();
    let mut gm = &bh * &b;
    let tr = (0..nseg).map(|l| gm[(l, l)].re).sum::<f64>() / nseg as f64;
    for l in 0..nseg {
        gm[(l, l)] += Complex64::new(1e-9 * tr, 0.0);
    }
    let chol = gm.cholesky().expect("segment Gram matrix");
    let (num, den) = ts
        .par_iter()
        .zip(tw.par_iter())
        .map(|(&t, &w)| {
            let y = DVector::<Complex64>::from_iterator(fit_voxels.len(), fit_voxels.iter().enumerate().map(|(r, &vx)| g(vx, t) * sw(r)));
            let a = chol.solve(&(&bh * &y));
            // residual per voxel without the fit weight, weighted by the error weight
            let (mut n, mut d) = (0.0f64, 0.0f64);
            for (r, &vx) in fit_voxels.iter().enumerate() {
                let gv = g(vx, t);
                let mut z = Complex64::new(0.0, 0.0);
                for l in 0..nseg {
                    z += a[l] * b[(r, l)] / sw(r).max(1e-300);
                }
                n += ew(r) * (gv - z).norm_sqr();
                d += ew(r) * gv.norm_sqr();
            }
            (w * n, w * d)
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    (num / den.max(1e-300)).sqrt()
}

/// Joint water + fat segments independent of the fat fraction (as `build_segments_fat`) at explicit node times `tl`
/// (e.g. [`cheb_nodes`]) with voxel-weighted interpolators: water diagonals P_l = e^{−iψ(·, τ_l)}, the fat spectrum at
/// the nodes c_l = c(τ_l), and a table fitted (weights `voxel_w` on both the water and the fat row of every fit voxel)
/// to {e^{−iψ(v, τ)}, c(τ) e^{−iψ(v, τ)}}. Same output as `build_segments_fat`: (P, a, c_l, t0, dt), for
/// `SegNufft::segments_fat` (per-frame fat-fraction maps) or `fold_fat_fraction`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn build_segments_fat_nodes(
    nv: usize,
    tl: &[f64],
    t_min: f64,
    t_max: f64,
    ntab: usize,
    fit_voxels: &[usize],
    voxel_w: Option<&[f64]>,
    psi: &(dyn Fn(usize, f64) -> f64 + Sync),
    c: &(dyn Fn(f64) -> Complex64 + Sync),
) -> (Vec<C32>, Vec<C32>, Vec<C32>, f64, f64) {
    let nseg = tl.len();
    let mut segph = vec![C32::new(0.0, 0.0); nseg * nv];
    segph.par_chunks_mut(nv.max(1)).zip(tl.par_iter()).for_each(|(row, &t)| {
        for (v, o) in row.iter_mut().enumerate() {
            let (s, co) = (-psi(v, t)).sin_cos();
            *o = C32::new(co as f32, s as f32);
        }
    });
    let cl: Vec<Complex64> = tl.iter().map(|&t| c(t)).collect();
    let nf = fit_voxels.len();
    let sw = |r: usize| voxel_w.map_or(1.0, |w| w[r % nf].max(0.0).sqrt());
    let b = DMatrix::<Complex64>::from_fn(2 * nf, nseg, |r, l| {
        let w = Complex64::from_polar(1.0, -psi(fit_voxels[r % nf], tl[l])) * sw(r);
        if r < nf { w } else { w * cl[l] }
    });
    let atab = crate::segnufft::ls_interpolators_pub(&b, nseg, t_min, t_max, ntab, &|t| {
        let ct = c(t);
        DVector::<Complex64>::from_iterator(2 * nf, (0..2 * nf).map(|r| {
            let w = Complex64::from_polar(1.0, -psi(fit_voxels[r % nf], t)) * sw(r);
            if r < nf { w } else { w * ct }
        }))
    });
    (segph, atab, cl.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect(), t_min, (t_max - t_min) / (ntab - 1) as f64)
}

/// Joint water + fat segments for **per-frame fat maps** with a noise-weighted SVD basis
/// (`research/readout_physics_budget.md`, T3): the weighted rank-L basis of the *union* of every fit voxel's water
/// function e^{−iψ(v, t)} (weight `w_water`) and fat function c(t) e^{−iψ(v, t)} (weight `w_fat`) on the time grid
/// (`ts`, `tw`, e.g. [`seg_time_grid`] with [`noise_time_weights`]), interpolators by least squares over the same union
/// rows, then the QDEIM cardinal form ([`crate::cardinal_form`]): the diagonals become ≈ e^{−iψ(·, τ_l)} at the node
/// times τ_l, so the per-node fat factor 1 + f (c(τ_l) − 1) of `SegNufft::segments_fat` / `fat_maps` applies (any
/// fat-fraction map, per frame). Returns (P [nseg × nv], a [ntab × nseg], c_l [nseg], t0, dt, node table entries,
/// condition number of the node matrix) for `SegNufft::segments_fat`.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn build_segments_svd_fat_union(
    nv: usize,
    nseg: usize,
    t_min: f64,
    t_max: f64,
    ntab: usize,
    fit_voxels: &[usize],
    w_water: &[f64],
    w_fat: &[f64],
    ts: &[f64],
    tw: &[f64],
    psi: &(dyn Fn(usize, f64) -> f64 + Sync),
    c: &(dyn Fn(f64) -> Complex64 + Sync),
) -> (Vec<C32>, Vec<C32>, Vec<C32>, f64, f64, Vec<usize>, f64) {
    build_segments_svd_fat_union_g(nv, nseg, t_min, t_max, ntab, fit_voxels, w_water, w_fat, ts, tw, &|v, t| Complex64::from_polar(1.0, -psi(v, t)), c)
}

/// [`build_segments_svd_fat_union`] with a general complex water function g(v, t) per voxel (e.g. with a per-voxel
/// complex rate: e^{−R2*_v (t − t0)} e^{−iψ(v, t)}); the fat function is g(v, t) c(t). With
/// g = e^{−iψ} it is that function exactly. The cardinal-form diagonals are then ≈ g(·, τ_l) at the node times
/// (decay included), so the per-node fat factor of `SegNufft::segments_fat` still applies.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn build_segments_svd_fat_union_g(
    nv: usize,
    nseg: usize,
    t_min: f64,
    t_max: f64,
    ntab: usize,
    fit_voxels: &[usize],
    w_water: &[f64],
    w_fat: &[f64],
    ts: &[f64],
    tw: &[f64],
    g: &(dyn Fn(usize, f64) -> Complex64 + Sync),
    c: &(dyn Fn(f64) -> Complex64 + Sync),
) -> (Vec<C32>, Vec<C32>, Vec<C32>, f64, f64, Vec<usize>, f64) {
    let ns = ts.len();
    let nf = fit_voxels.len();
    assert!(nseg >= 1 && nseg <= ns);
    assert_eq!(w_water.len(), nf);
    assert_eq!(w_fat.len(), nf);
    let st: Vec<f64> = tw.iter().map(|&x| x.max(0.0).sqrt()).collect();
    let cts: Vec<Complex64> = ts.iter().map(|&t| c(t)).collect();
    // union Gram (upper triangle, chunked over the fit voxels)
    let zero = || vec![Complex64::new(0.0, 0.0); ns * ns];
    let idx: Vec<usize> = (0..nf).collect();
    let gram = idx
        .par_chunks(1024)
        .map(|chunk| {
            let mut acc = zero();
            let mut row = vec![Complex64::new(0.0, 0.0); ns];
            for &r in chunk {
                let v = fit_voxels[r];
                for (wgt, fat) in [(w_water[r], false), (w_fat[r], true)] {
                    let w = wgt.max(0.0).sqrt();
                    if w == 0.0 {
                        continue;
                    }
                    for (i, &t) in ts.iter().enumerate() {
                        row[i] = g(v, t) * (w * st[i]) * if fat { cts[i] } else { Complex64::new(1.0, 0.0) };
                    }
                    for i in 0..ns {
                        let ci = row[i].conj();
                        for j in i..ns {
                            acc[i * ns + j] += ci * row[j];
                        }
                    }
                }
            }
            acc
        })
        .reduce(zero, |mut a, b| {
            a.iter_mut().zip(&b).for_each(|(x, y)| *x += y);
            a
        });
    let mut cm = DMatrix::<Complex64>::zeros(ns, ns);
    for i in 0..ns {
        for j in i..ns {
            cm[(i, j)] = gram[i * ns + j];
            cm[(j, i)] = gram[i * ns + j].conj();
        }
    }
    let eig = nalgebra::SymmetricEigen::new(cm);
    let mut order: Vec<usize> = (0..ns).collect();
    order.sort_by(|&a, &b| eig.eigenvalues[b].partial_cmp(&eig.eigenvalues[a]).unwrap());
    let vl = DMatrix::<Complex64>::from_fn(ns, nseg, |i, l| eig.eigenvectors[(i, order[l])]);
    // water coefficients of every voxel; water and fat coefficients of the fit voxels
    let coef = |v: usize, fat: bool| -> Vec<Complex64> {
        let gv: Vec<Complex64> = ts.iter().enumerate().map(|(i, &t)| g(v, t) * st[i] * if fat { cts[i] } else { Complex64::new(1.0, 0.0) }).collect();
        (0..nseg).map(|l| (0..ns).map(|i| gv[i] * vl[(i, l)]).sum()).collect()
    };
    let rows: Vec<Vec<Complex64>> = (0..nv).into_par_iter().with_min_len(512).map(|v| coef(v, false)).collect();
    let mut segph = vec![C32::new(0.0, 0.0); nseg * nv];
    for (v, r) in rows.iter().enumerate() {
        for l in 0..nseg {
            segph[l * nv + v] = C32::new(r[l].re as f32, r[l].im as f32);
        }
    }
    let fat_rows: Vec<Vec<Complex64>> = fit_voxels.par_iter().map(|&v| coef(v, true)).collect();
    let sw = |r: usize| if r < nf { w_water[r].max(0.0).sqrt() } else { w_fat[r - nf].max(0.0).sqrt() };
    let b = DMatrix::<Complex64>::from_fn(2 * nf, nseg, |r, l| (if r < nf { rows[fit_voxels[r]][l] } else { fat_rows[r - nf][l] }) * sw(r));
    let atab = crate::segnufft::ls_interpolators_pub(&b, nseg, t_min, t_max, ntab, &|t| {
        let ct = c(t);
        DVector::<Complex64>::from_iterator(2 * nf, (0..2 * nf).map(|r| {
            let z = g(fit_voxels[r % nf], t) * sw(r);
            if r < nf { z } else { z * ct }
        }))
    });
    let (segph, atab, nodes, cond) = crate::cardinal_form(&segph, &atab, nseg, nv);
    let dt = (t_max - t_min) / (ntab - 1) as f64;
    let cl: Vec<C32> = nodes.iter().map(|&e| {
        let z = c(t_min + e as f64 * dt);
        C32::new(z.re as f32, z.im as f32)
    }).collect();
    (segph, atab, cl, t_min, dt, nodes, cond)
}

/// Weighted squared error of the union rows' tail: the rank-L error of the union Gram (the rule for
/// [`build_segments_svd_fat_union`]): eigenvalues, descending.
pub fn fat_union_spectrum(fit_voxels: &[usize], w_water: &[f64], w_fat: &[f64], ts: &[f64], tw: &[f64], psi: &(dyn Fn(usize, f64) -> f64 + Sync), c: &(dyn Fn(f64) -> Complex64 + Sync)) -> Vec<f64> {
    fat_union_spectrum_g(fit_voxels, w_water, w_fat, ts, tw, &|v, t| Complex64::from_polar(1.0, -psi(v, t)), c)
}

/// [`fat_union_spectrum`] with a general complex water function g(v, t) (as [`build_segments_svd_fat_union_g`])
pub fn fat_union_spectrum_g(fit_voxels: &[usize], w_water: &[f64], w_fat: &[f64], ts: &[f64], tw: &[f64], g: &(dyn Fn(usize, f64) -> Complex64 + Sync), c: &(dyn Fn(f64) -> Complex64 + Sync)) -> Vec<f64> {
    // the union Gram = water Gram (weights w_water) + fat Gram (weights w_fat, time function times c)
    let nf = fit_voxels.len();
    let mut both: Vec<usize> = fit_voxels.to_vec();
    both.extend_from_slice(fit_voxels);
    let w: Vec<f64> = w_water.iter().chain(w_fat.iter()).copied().collect();
    let fv = both.clone();
    let g2 = move |r: usize, t: f64| -> Complex64 {
        let z = g(fv[r], t);
        if r < nf { z } else { z * c(t) }
    };
    let pseudo: Vec<usize> = (0..2 * nf).collect();
    weighted_gram(&pseudo, Some(&w), ts, tw, &g2).0
}
