//! Physics tables for the segmented physics NUFFT (`hasty_nufft::SegNufft`) from the simulator's sequence.
//!
//! Sample j sees the phase  k_j·ρ + ω(ρ) τ_j + Σ_q w_jq m_q(ρ),  m_q = concomitant monomials [z², x²+y², xz, yz] at the
//! true scanner position of the (apparent, GNL-warped) reconstruction voxel ρ, w_j their sample coefficients (whole
//! history since excitation: encoding lobes, blips and readout). Split w_j = w̄(τ_j) + Δw_j with w̄ the mean over the
//! set's readouts at the same ADC sample:
//!   * segments: ψ(ρ, τ) = ω(ρ) τ + Σ_q w̄_q(τ) m_q(ρ)  (exact, any spatial function: B0, GNL positions, FOV offset);
//!   * per sample: Σ_q Δw_jq m_q(ρ) ≈ φ0_j + b_j·ρ + ρᵀΔC_jρ from a least-squares quadratic fit of each m_q over the
//!     object support → constant phase, k-space shift (exact) and the ΔC kernel correction.

use hasty_mrsim::Vec3;

use crate::data::{PhysicsGrid, SampleSet, SimData};

/// Quadratic fits m_q(ρ) ≈ θ_q·[1, x, y, z, x², y², z², 2xy, 2xz, 2yz] (ρ logical, m).
pub struct MonoFit {
    pub theta: [[f64; 10]; 4],
    /// RMS / max residual of each fit over the fit voxels (m²)
    pub rms: [f64; 4],
    pub max: [f64; 4],
}

fn basis(p: &Vec3) -> [f64; 10] {
    [1.0, p[0], p[1], p[2], p[0] * p[0], p[1] * p[1], p[2] * p[2], 2.0 * p[0] * p[1], 2.0 * p[0] * p[2], 2.0 * p[1] * p[2]]
}

pub fn fit_monomials(sd: &SimData, grid: &PhysicsGrid, voxels: &[usize]) -> MonoFit {
    let mut ata = nalgebra::DMatrix::<f64>::zeros(10, 10);
    let mut atb = nalgebra::DMatrix::<f64>::zeros(10, 4);
    for &v in voxels {
        let b = basis(&sd.voxel_log(v));
        for i in 0..10 {
            for k in 0..10 {
                ata[(i, k)] += b[i] * b[k];
            }
            for q in 0..4 {
                atb[(i, q)] += b[i] * grid.mono[v][q];
            }
        }
    }
    let sol = ata.cholesky().expect("monomial fit").solve(&atb);
    let mut theta = [[0.0; 10]; 4];
    for q in 0..4 {
        for i in 0..10 {
            theta[q][i] = sol[(i, q)];
        }
    }
    let (mut rms, mut max) = ([0.0; 4], [0.0f64; 4]);
    for &v in voxels {
        let b = basis(&sd.voxel_log(v));
        for q in 0..4 {
            let r = grid.mono[v][q] - (0..10).map(|i| theta[q][i] * b[i]).sum::<f64>();
            rms[q] += r * r;
            max[q] = max[q].max(r.abs());
        }
    }
    rms.iter_mut().for_each(|r| *r = (*r / voxels.len() as f64).sqrt());
    MonoFit { theta, rms, max }
}

/// Per-sample inputs of `SegNufft::points` plus the set's mean concomitant coefficients per ADC sample.
pub struct SegTables {
    pub coords: [Vec<f32>; 3],
    pub tau: Vec<f32>,
    pub phi0: Vec<f32>,
    /// ΔC, [xx, yy, zz, xy, xz, yz] SoA (6 × m), rad/m²
    pub dc: Vec<f32>,
    /// acquisition time in frame units (node f at t = f), periodic over the frames
    pub t: Vec<f32>,
    /// w̄ per ADC sample and its readout times
    pub wbar: Vec<[f64; 4]>,
    pub tau_adc: Vec<f64>,
    /// largest |Δw·residual| over the fit voxels (rad): the phase the quadratic fit leaves out
    pub max_resid_phase: f64,
}

/// Tables for one sample set (`set.readouts` × ADC samples, in `SimData::sample_set` order). `wbar`: the mean
/// coefficients the segments were built with (e.g. over all frames of the encode); default: this set's own mean.
pub fn seg_tables(sd: &SimData, set: &SampleSet, fit: &MonoFit, with_conc: bool, wbar: Option<&[[f64; 4]]>) -> SegTables {
    let na = sd.na;
    let nr = set.readouts.len();
    let m = nr * na;
    let mut wbar = wbar.map(|w| w.to_vec()).unwrap_or_else(|| vec![[0.0f64; 4]; na]);
    if with_conc && wbar.iter().all(|w| w.iter().all(|&v| v == 0.0)) {
        for j in 0..m {
            for q in 0..4 {
                wbar[j % na][q] += set.wq[j][q] / nr as f64;
            }
        }
    }
    let tau_adc: Vec<f64> = (0..na).map(|n| set.tau[n]).collect();
    let mut coords = set.coords.clone();
    let mut phi0 = vec![0f32; m];
    let mut dc = vec![0f32; 6 * m];
    let mut t = vec![0f32; m];
    let nf = sd.n_frames as f64;
    let mut max_dw = [0.0f64; 4];
    for (r, &ro) in set.readouts.iter().enumerate() {
        let card = sd.ro.card[ro];
        for n in 0..na {
            let j = r * na + n;
            let phase = card.phase + set.tau[j] / card.rr;
            t[j] = (phase * nf - 0.5).rem_euclid(nf) as f32;
            if !with_conc {
                continue;
            }
            let dw: [f64; 4] = [0, 1, 2, 3].map(|q| set.wq[j][q] - wbar[n][q]);
            let th = |i: usize| (0..4).map(|q| dw[q] * fit.theta[q][i]).sum::<f64>();
            phi0[j] = th(0) as f32;
            for a in 0..3 {
                coords[a][j] += (th(1 + a) * sd.vox[a]) as f32;
            }
            for i in 0..6 {
                dc[i * m + j] = th(4 + i) as f32;
            }
            for q in 0..4 {
                max_dw[q] = max_dw[q].max(dw[q].abs());
            }
        }
    }
    let max_resid_phase = (0..4).map(|q| max_dw[q] * fit.max[q]).sum();
    SegTables { coords, tau: set.tau.iter().map(|&v| v as f32).collect(), phi0, dc, t, wbar, tau_adc, max_resid_phase }
}

/// Segment phase ψ(v, τ) = ω_v τ + Σ_q w̄_q(τ) m_q(v), w̄ (per ADC sample, times `tau_adc`, uniform) interpolated linearly.
pub fn segment_phase<'a>(grid: &'a PhysicsGrid, wbar: &'a [[f64; 4]], tau_adc: &'a [f64], with_b0: bool) -> impl Fn(usize, f64) -> f64 + Sync + 'a {
    move |v: usize, tau: f64| {
        let n = tau_adc.len();
        let u = ((tau - tau_adc[0]) / (tau_adc[n - 1] - tau_adc[0]) * (n - 1) as f64).clamp(0.0, (n - 1) as f64 - 1e-9);
        let (i, fr) = (u.floor() as usize, u - u.floor());
        let mut p = if with_b0 { grid.omega[v] * tau } else { 0.0 };
        for q in 0..4 {
            p += (wbar[i][q] + fr * (wbar[(i + 1).min(n - 1)][q] - wbar[i][q])) * grid.mono[v][q];
        }
        p
    }
}

/// Mean concomitant coefficients per ADC sample over several sample sets (e.g. all frames of one encode).
pub fn mean_coefficients(sets: &[&SampleSet], na: usize) -> Vec<[f64; 4]> {
    let mut w = vec![[0.0f64; 4]; na];
    let nr: usize = sets.iter().map(|s| s.readouts.len()).sum();
    for s in sets {
        for (j, wq) in s.wq.iter().enumerate() {
            for q in 0..4 {
                w[j % na][q] += wq[q] / nr as f64;
            }
        }
    }
    w
}

// ───────────────────────────────────────────────────────────── higher-order fit (optional, `SegNufft` order 3)

/// The degree-3/4 monomials beyond the quadratic basis, in the order `hasty_nufft::SegNufft::points_hi` expects:
/// x^a y^b z^c, degree 3 first (10), then degree 4 (15), a (x order) outer, b (y order) inner.
pub use hasty_nufft::{hi_exponents, NHI};

/// Degree-4 fits of the concomitant monomials at the true (GNL-warped) positions over the apparent logical position ρ:
///   m_q(ρ) ≈ θ_q·[1, x, y, z, x², y², z², 2xy, 2xz, 2yz] + Σ_α h_qα ρ^α   (α: the [`NHI`] monomials of degree 3, 4).
/// `quad` holds the quadratic part θ of the joint fit (so `seg_tables` maps it to φ0 / k-shift / ΔC exactly as for the
/// quadratic fit) and the residual of the whole degree-4 fit (rms / max, m²); `hi` the degree-3/4 coefficients (m^{2−n}).
pub struct MonoFitHi {
    pub quad: MonoFit,
    pub hi: [[f64; NHI]; 4],
}

impl MonoFitHi {
    /// the per-sample degree-3/4 phase of deviation `dw`: Σ_α (Σ_q dw_q h_qα) ρ^α (rad)
    pub fn hi_phase(&self, dw: &[f64; 4], p: &Vec3) -> f64 {
        let ex = hi_exponents();
        (0..NHI).map(|i| (0..4).map(|q| dw[q] * self.hi[q][i]).sum::<f64>() * p[0].powi(ex[i][0] as i32) * p[1].powi(ex[i][1] as i32) * p[2].powi(ex[i][2] as i32)).sum()
    }
}

/// [`fit_monomials`] to degree 4 (least squares over `voxels`, in coordinates scaled to the fit region for conditioning).
pub fn fit_monomials_hi(sd: &SimData, grid: &PhysicsGrid, voxels: &[usize]) -> MonoFitHi {
    const NB: usize = 10 + NHI;
    let ex = hi_exponents();
    let s = voxels.iter().map(|&v| sd.voxel_log(v).iter().fold(0.0f64, |a, &c| a.max(c.abs()))).fold(0.0f64, f64::max).max(1e-6);
    // basis in scaled coordinates u = ρ / s; coefficient in metres = scaled coefficient / s^degree
    let deg: [i32; NB] = std::array::from_fn(|i| if i == 0 { 0 } else if i < 4 { 1 } else if i < 10 { 2 } else if i < 20 { 3 } else { 4 });
    let bas = |p: &Vec3| -> [f64; NB] {
        let u = [p[0] / s, p[1] / s, p[2] / s];
        let mut b = [0.0; NB];
        b[..10].copy_from_slice(&basis(&u));
        for i in 0..NHI {
            b[10 + i] = u[0].powi(ex[i][0] as i32) * u[1].powi(ex[i][1] as i32) * u[2].powi(ex[i][2] as i32);
        }
        b
    };
    let mut ata = nalgebra::DMatrix::<f64>::zeros(NB, NB);
    let mut atb = nalgebra::DMatrix::<f64>::zeros(NB, 4);
    for &v in voxels {
        let b = bas(&sd.voxel_log(v));
        for i in 0..NB {
            for k in 0..NB {
                ata[(i, k)] += b[i] * b[k];
            }
            for q in 0..4 {
                atb[(i, q)] += b[i] * grid.mono[v][q];
            }
        }
    }
    let tr = (0..NB).map(|i| ata[(i, i)]).sum::<f64>() / NB as f64;
    for i in 0..NB {
        ata[(i, i)] += 1e-12 * tr;
    }
    let sol = ata.cholesky().expect("degree-4 monomial fit").solve(&atb);
    let mut theta = [[0.0; 10]; 4];
    let mut hi = [[0.0; NHI]; 4];
    for q in 0..4 {
        for i in 0..NB {
            let c = sol[(i, q)] / s.powi(deg[i]);
            if i < 10 {
                theta[q][i] = c;
            } else {
                hi[q][i - 10] = c;
            }
        }
    }
    let mut fit = MonoFitHi { quad: MonoFit { theta, rms: [0.0; 4], max: [0.0; 4] }, hi };
    let (mut rms, mut max) = ([0.0; 4], [0.0f64; 4]);
    for &v in voxels {
        let p = sd.voxel_log(v);
        let b = basis(&p);
        for q in 0..4 {
            let mut e = [0.0; 4];
            e[q] = 1.0;
            let r = grid.mono[v][q] - (0..10).map(|i| fit.quad.theta[q][i] * b[i]).sum::<f64>() - fit.hi_phase(&e, &p);
            rms[q] += r * r;
            max[q] = max[q].max(r.abs());
        }
    }
    rms.iter_mut().for_each(|r| *r = (*r / voxels.len().max(1) as f64).sqrt());
    fit.quad.rms = rms;
    fit.quad.max = max;
    fit
}

/// [`seg_tables`] for the degree-4 fit: the quadratic part as usual (φ0, k-shift, ΔC; `max_resid_phase` is then the
/// phase the degree-4 fit leaves out), plus the per-sample deviation Δw = w − w̄ (4 × m SoA) that weights the degree-3/4
/// coefficients `fit.hi` (`hasty_nufft::SegNufft::points_hi`). Without concomitant fields Δw = 0.
pub fn seg_tables_hi(sd: &SimData, set: &SampleSet, fit: &MonoFitHi, with_conc: bool, wbar: Option<&[[f64; 4]]>) -> (SegTables, Vec<f32>) {
    let tab = seg_tables(sd, set, &fit.quad, with_conc, wbar);
    let (na, m) = (sd.na, tab.tau.len());
    let mut dw = vec![0f32; 4 * m];
    if with_conc {
        for j in 0..m {
            for q in 0..4 {
                dw[q * m + j] = (set.wq[j][q] - tab.wbar[j % na][q]) as f32;
            }
        }
    }
    (tab, dw)
}

/// the degree-3/4 coefficient table of a fit as `SegNufft::points_hi` takes it: h_qα, q-major (4 × [`NHI`])
pub fn hi_table(fit: &MonoFitHi) -> Vec<f64> {
    fit.hi.iter().flatten().copied().collect()
}
