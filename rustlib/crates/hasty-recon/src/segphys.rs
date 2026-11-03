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
        let card = sd.acq.readouts[ro].card;
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
                coords[a][j] += (th(1 + a) * sd.res) as f32;
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
