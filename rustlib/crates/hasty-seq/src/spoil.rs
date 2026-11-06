//! Spoiling check: multi-isochromat steady state of an RF-spoiled gradient echo whose per-TR gradient moments vary
//! from TR to TR (rotated spokes, interleaved encodes).
//!
//! A voxel (cube of edge `voxel`, centre `offset` from isocentre) is filled with `n_iso` isochromats on a
//! low-discrepancy (R3) lattice. A spin at x(t) = c(t) + x_j with c(t) = offset + v (t − t_last), during TR n (starting
//! at t_n), gains the gradient phase γ[M0(τ)·(c(t_n) + x_j) + M1(τ)·v] with moments taken from the TR start. At the echo
//! M0 = 0, so the echo phase is γ M1_enc·v (uniform over the voxel). Over the whole TR it is 2π k_net·x_j (intravoxel
//! dephasing) + 2π k_net·c(t_n) + γ M1_net·v (uniform: any TR-to-TR change of k_net acts with the position, so a
//! jittering or rotating net moment scrambles the RF-spoiling phase away from isocentre). Instantaneous RF pulses with
//! quadratic phase φ_n = inc·n(n+1)/2, T1/T2 relaxation, no diffusion (conservative: diffusion only helps spoiling).
//! The signal at the echo is demodulated by φ_n and compared with the ideal spoiled (Ernst) signal.

use crate::{Vec3, GAMMA};

#[derive(Clone, Copy, Debug)]
pub struct Tissue {
    pub t1: f64,
    pub t2: f64,
}

#[derive(Clone, Debug)]
pub struct SpoilSim {
    /// flip angle, rad
    pub flip: f64,
    pub tr: f64,
    pub te: f64,
    /// quadratic RF phase increment, rad
    pub rf_inc: f64,
    /// voxel edge, m
    pub voxel: f64,
    pub n_iso: usize,
    /// voxel centre (m from isocentre); a flowing voxel's spins arrive there at the last TR
    pub offset: Vec3,
}

/// one TR's moments (physical axes, from the TR start): echo first moment, net zeroth (as k, cycles/m) and first moment
#[derive(Clone, Copy, Debug)]
pub struct TrMoments {
    pub m1_echo: Vec3,
    pub k_net: Vec3,
    pub m1_net: Vec3,
}

/// ideal spoiled-GRE echo amplitude (M0 = 1)
pub fn ernst(sim: &SpoilSim, t: &Tissue) -> f64 {
    let e1 = (-sim.tr / t.t1).exp();
    sim.flip.sin() * (1.0 - e1) / (1.0 - e1 * sim.flip.cos()) * (-sim.te / t.t2).exp()
}

/// complex echo signal (re, im) of every TR, demodulated by the RF phase
pub fn simulate(sim: &SpoilSim, tis: &Tissue, trs: &[TrMoments], v: &Vec3) -> Vec<(f64, f64)> {
    // R3 lattice (Roberts): frac(0.5 + j α_i), α = 1/φ₃^i
    let phi3 = 1.220_744_084_605_759_5f64;
    let al = [1.0 / phi3, 1.0 / (phi3 * phi3), 1.0 / (phi3 * phi3 * phi3)];
    let n = sim.n_iso;
    let pos: Vec<Vec3> = (0..n).map(|j| al.map(|a| ((0.5 + j as f64 * a).fract() - 0.5) * sim.voxel)).collect();
    let (mut mx, mut my, mut mz) = (vec![0.0f64; n], vec![0.0f64; n], vec![1.0f64; n]);
    let (e1te, e2te) = ((-sim.te / tis.t1).exp(), (-sim.te / tis.t2).exp());
    let (e1r, e2r) = ((-(sim.tr - sim.te) / tis.t1).exp(), (-(sim.tr - sim.te) / tis.t2).exp());
    let (sa, ca) = sim.flip.sin_cos();
    let tau = 2.0 * std::f64::consts::PI;
    let mut out = Vec::with_capacity(trs.len());
    let mut phase = 0.0f64;
    for (it, m) in trs.iter().enumerate() {
        phase = (phase + sim.rf_inc * it as f64) % tau;
        let (sp, cp) = phase.sin_cos();
        // position of the voxel's spins during TR n: offset + v (t_n − t_last)
        let t_n = (it as f64 - (trs.len() - 1) as f64) * sim.tr;
        let c = [0, 1, 2].map(|a| sim.offset[a] + v[a] * t_n);
        let echo = GAMMA * (m.m1_echo[0] * v[0] + m.m1_echo[1] * v[1] + m.m1_echo[2] * v[2]);
        let uni = tau * (m.k_net[0] * c[0] + m.k_net[1] * c[1] + m.k_net[2] * c[2]) + GAMMA * (m.m1_net[0] * v[0] + m.m1_net[1] * v[1] + m.m1_net[2] * v[2]) - echo;
        let (se, ce) = echo.sin_cos();
        let (mut sr, mut si) = (0.0, 0.0);
        for j in 0..n {
            // RF about the axis at angle `phase` in the transverse plane: rotate by −phase, about x by α, back
            let (x0, y0) = (mx[j] * cp + my[j] * sp, -mx[j] * sp + my[j] * cp);
            let (y1, z1) = (y0 * ca + mz[j] * sa, -y0 * sa + mz[j] * ca);
            let (x2, y2) = (x0 * cp - y1 * sp, x0 * sp + y1 * cp);
            // to the echo: relaxation and the uniform encode phase
            let (x3, y3) = ((x2 * ce - y2 * se) * e2te, (x2 * se + y2 * ce) * e2te);
            let z3 = z1 * e1te + (1.0 - e1te);
            // demodulated by the RF phase
            sr += x3 * cp + y3 * sp;
            si += -x3 * sp + y3 * cp;
            // rest of the TR: relaxation, intravoxel dephasing and the remaining uniform phase
            let p = &pos[j];
            let th = tau * (m.k_net[0] * p[0] + m.k_net[1] * p[1] + m.k_net[2] * p[2]) + uni;
            let (st, ct) = th.sin_cos();
            mx[j] = (x3 * ct - y3 * st) * e2r;
            my[j] = (x3 * st + y3 * ct) * e2r;
            mz[j] = z3 * e1r + (1.0 - e1r);
        }
        out.push((sr / n as f64, si / n as f64));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_constant_moment_with_rf_spoiling_is_ernst() {
        let sim = SpoilSim { flip: 10f64.to_radians(), tr: 6e-3, te: 2e-3, rf_inc: 117f64.to_radians(), voxel: 2e-3, n_iso: 8192, offset: [0.0; 3] };
        let t = Tissue { t1: 1.0, t2: 0.05 };
        let k = 4.0 / sim.voxel;
        let trs = vec![TrMoments { m1_echo: [0.0; 3], k_net: [0.0, 0.0, k], m1_net: [0.0; 3] }; 1500];
        let s = simulate(&sim, &t, &trs, &[0.0; 3]);
        let tail = &s[1000..];
        let mean = tail.iter().map(|(a, b)| (a * a + b * b).sqrt()).sum::<f64>() / tail.len() as f64;
        let ideal = ernst(&sim, &t);
        assert!((mean / ideal - 1.0).abs() < 0.02, "{mean} vs {ideal}");
    }
}
