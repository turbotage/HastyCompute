//! Velocity encoding: minimum-time gradient lobes for a target zeroth and first moment, encoding schemes, joint timing.
//!
//! One axis: two back-to-back trapezoids (ramps r₁, flat f₁, ramps r₁; then r₂, f₂, r₂ segments on the raster). For a
//! fixed shape the amplitudes (g₁, g₂) follow exactly from the 2 × 2 system
//!   g₁ A₁ + g₂ A₂ = M0,   g₁ B₁ + g₂ B₂ = M1   (A = area of the unit lobe, B = its first moment about the block start),
//! which is always solvable (det = A₁A₂(c₂ − c₁) > 0, c = lobe centroids). The search over shapes (segment counts)
//! returns the shortest block with |g| ≤ G_max and |g|/(r·dt) ≤ S_max; at a fixed length it returns the shape with
//! the largest margin. A pure bipolar is M0 = 0; a prephaser / rephaser merged with flow encoding is M0 ≠ 0.
//! All encodes and axes share one block length (every TR has the same timing): the longest minimum-time block, at
//! which each (encode, axis) is re-solved with the largest margin.
//! Moments are exact (piecewise-linear waveforms, `wave::Grad::moments`); M1 is given about `t_ref` (e.g. the RF
//! centre). ΔM1 between encodes does not depend on `t_ref` because M0 is the same for all encodes.

use crate::system::System;
use crate::wave::{trap_shape, Grad};
use crate::{Vec3, GAMMA};

#[derive(Clone, Copy, Debug)]
pub struct TwoLobe {
    pub r1: usize,
    pub f1: usize,
    pub r2: usize,
    pub f2: usize,
    pub g1: f64,
    pub g2: f64,
}

impl TwoLobe {
    pub fn segments(&self) -> usize {
        2 * self.r1 + self.f1 + 2 * self.r2 + self.f2
    }
    pub fn samples(&self) -> Vec<f64> {
        let mut a = trap_shape(self.r1, self.f1, self.r1, self.g1);
        let b = trap_shape(self.r2, self.f2, self.r2, self.g2);
        a.pop();
        a.extend(b);
        a
    }
    /// fraction of the binding limit (≤ 1 feasible)
    pub fn margin(&self, gmax: f64, smax: f64, dt: f64) -> f64 {
        [self.g1.abs() / gmax, self.g2.abs() / gmax, self.g1.abs() / (self.r1 as f64 * dt * smax), self.g2.abs() / (self.r2 as f64 * dt * smax)].into_iter().fold(0.0, f64::max)
    }
}

/// amplitudes for a given shape (segment counts), M1 about the block start
fn solve(r1: usize, f1: usize, r2: usize, f2: usize, m0: f64, m1: f64, dt: f64) -> TwoLobe {
    let a1 = (r1 + f1) as f64 * dt;
    let a2 = (r2 + f2) as f64 * dt;
    let t1 = (2 * r1 + f1) as f64 * dt;
    let t2 = (2 * r2 + f2) as f64 * dt;
    let (b1, b2) = (a1 * 0.5 * t1, a2 * (t1 + 0.5 * t2));
    let det = a1 * b2 - a2 * b1;
    let g1 = (m0 * b2 - a2 * m1) / det;
    let g2 = (a1 * m1 - b1 * m0) / det;
    TwoLobe { r1, f1, r2, f2, g1, g2 }
}

/// best shape with exactly `n` segments (largest margin), or None
pub fn two_lobe_fixed(m0: f64, m1: f64, n: usize, sys: &System) -> Option<TwoLobe> {
    let dt = sys.grad_raster;
    let rmax = (sys.gmax / (sys.smax * dt)).ceil() as usize + 1;
    let mut best: Option<(f64, TwoLobe)> = None;
    for r1 in 1..=rmax {
        for r2 in 1..=rmax {
            if 2 * r1 + 2 * r2 > n {
                break;
            }
            for f1 in 0..=(n - 2 * r1 - 2 * r2) {
                let f2 = n - 2 * r1 - 2 * r2 - f1;
                let s = solve(r1, f1, r2, f2, m0, m1, dt);
                let mg = s.margin(sys.gmax, sys.smax, dt);
                if mg <= 1.0 + 1e-9 && best.map_or(true, |b| mg < b.0) {
                    best = Some((mg, s));
                }
            }
        }
    }
    best.map(|b| b.1)
}

/// shortest two-lobe block for (M0, M1 about the block start)
pub fn two_lobe_min_time(m0: f64, m1: f64, sys: &System) -> TwoLobe {
    // lower bound for M0 = 0: slew-limited triangles, M1 = 2 S r³ over 4 r (any M0 ≠ 0 only adds area to move)
    let dt = sys.grad_raster;
    let r = (m1.abs() / (2.0 * sys.smax)).cbrt();
    // any M0: a waveform with area |M0| under slew S needs at least the triangle time 2 √(|M0| / S)
    let t_area = 2.0 * (m0.abs() / sys.smax).sqrt();
    let n0 = if m0 == 0.0 { ((4.0 * r / dt).floor() as usize).saturating_sub(2).max(4) } else { ((t_area / dt).floor() as usize).saturating_sub(2).max(4) };
    for n in n0..100_000 {
        if let Some(s) = two_lobe_fixed(m0, m1, n, sys) {
            return s;
        }
    }
    unreachable!("no feasible two-lobe design")
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Scheme {
    /// reference + one encode per axis
    Referenced,
    /// four-point balanced Hadamard
    Hadamard,
    /// reference + low- and high-VENC encode per axis (seven points)
    DualVenc { venc_hi: f64 },
    /// coprime dual-VENC (Beghella Bartoli, Chechin, Boccalini, Sigovan, JCMR 2025): VENC_low / VENC_high = p / (p + 1),
    /// so the two aliasing periods are coprime multiples of VENC_high / p and the velocity is recovered without
    /// aliasing up to ±p·VENC_high = ±(p + 1)·VENC_low (Chinese remainder, `venc::Crt`). `venc` is VENC_low.
    Coprime { p: u32, layout: DualLayout },
}

/// encode layouts of a dual-VENC scheme
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DualLayout {
    /// [ref, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]: shared reference (the 7-point layout of the denoiser)
    Seven,
    /// [ref, x_lo, y_lo, z_lo, ref, x_hi, y_hi, z_hi]: one reference per VENC, interleaved (the paper's 8 points)
    Eight,
    /// two balanced four-point Hadamard sets, VENC_low then VENC_high
    EightHadamard,
}

impl Scheme {
    /// (VENC_low, VENC_high) for the scheme's `venc` (single-VENC schemes: both equal)
    pub fn vencs(&self, venc: f64) -> (f64, f64) {
        match *self {
            Scheme::Referenced | Scheme::Hadamard => (venc, venc),
            Scheme::DualVenc { venc_hi } => (venc, venc_hi),
            Scheme::Coprime { p, .. } => (venc, venc * (p as f64 + 1.0) / p as f64),
        }
    }
    /// largest velocity (per encoded direction) recovered without aliasing
    pub fn unaliased(&self, venc: f64) -> f64 {
        match *self {
            Scheme::Referenced | Scheme::Hadamard => venc,
            Scheme::DualVenc { venc_hi } => venc_hi,
            Scheme::Coprime { p, .. } => (p as f64 + 1.0) * venc,
        }
    }
}

/// coprime VENCs for an extended (unaliased) velocity `v_ext` and extension factor p: (v_ext / (p + 1), v_ext / p)
pub fn coprime_vencs(v_ext: f64, p: u32) -> (f64, f64) {
    (v_ext / (p as f64 + 1.0), v_ext / p as f64)
}

/// first-moment step giving a phase difference of π at |v| = VENC
pub fn delta_m1(venc: f64) -> f64 {
    std::f64::consts::PI / (GAMMA * venc)
}

/// encode first moments (T·s²/m) per encode; `balanced` shifts each axis so that its extreme values are symmetric
/// about zero (smallest |M1| for the same differences)
pub fn scheme_m1(scheme: Scheme, venc: f64, balanced: bool) -> Vec<Vec3> {
    let d = delta_m1(venc);
    let referenced = |d: f64| vec![[0.0; 3], [d, 0.0, 0.0], [0.0, d, 0.0], [0.0, 0.0, d]];
    let hadamard = |d: f64| -> Vec<Vec3> { [[-1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]].iter().map(|h| h.map(|x| 0.5 * d * x)).collect() };
    let seven = |dl: f64, dh: f64| {
        let mut v = vec![[0.0; 3]];
        for a in 0..3 {
            for st in [dl, dh] {
                let mut e = [0.0; 3];
                e[a] = st;
                v.push(e);
            }
        }
        v
    };
    let mut m: Vec<Vec3> = match scheme {
        Scheme::Referenced => referenced(d),
        Scheme::Hadamard => hadamard(d),
        Scheme::DualVenc { venc_hi } => seven(d, delta_m1(venc_hi)),
        Scheme::Coprime { layout, .. } => {
            let dh = delta_m1(scheme.vencs(venc).1);
            match layout {
                DualLayout::Seven => seven(d, dh),
                DualLayout::Eight => [referenced(d), referenced(dh)].concat(),
                DualLayout::EightHadamard => [hadamard(d), hadamard(dh)].concat(),
            }
        }
    };
    if balanced {
        for a in 0..3 {
            let (lo, hi) = m.iter().fold((f64::MAX, f64::MIN), |(l, h), e| (l.min(e[a]), h.max(e[a])));
            let c = 0.5 * (lo + hi);
            for e in m.iter_mut() {
                e[a] -= c;
            }
        }
    }
    m
}

#[derive(Clone, Debug)]
pub struct EncodeSpec {
    /// zeroth moment every encode must have per axis (e.g. slab rephaser), T·s/m
    pub m0: Vec3,
    /// first-moment offset added to every encode (e.g. compensating the excitation / readout), T·s²/m about `t_ref`
    pub m1_offset: Vec3,
    pub scheme: Scheme,
    pub venc: f64,
    pub balanced: bool,
    /// block start and the time origin of M1 (e.g. RF centre), s
    pub t_start: f64,
    pub t_ref: f64,
}

#[derive(Clone, Debug)]
pub struct EncodeDesign {
    /// per encode: the block's gradients (physical axes), starting at `t_start`
    pub waves: Vec<Grad>,
    /// per encode: M0, M1, M2 at the block end (about `t_ref`)
    pub m0: Vec<Vec3>,
    pub m1: Vec<Vec3>,
    pub m2: Vec<Vec3>,
    pub duration: f64,
    pub shapes: Vec<[TwoLobe; 3]>,
}

pub fn design(spec: &EncodeSpec, sys: &System) -> EncodeDesign {
    // memoised: the optimiser rebuilds the same encode for every shape and derating step
    static CACHE: std::sync::OnceLock<std::sync::Mutex<std::collections::HashMap<String, EncodeDesign>>> = std::sync::OnceLock::new();
    let key = format!("{spec:?}|{sys:?}");
    let cache = CACHE.get_or_init(Default::default);
    if let Some(d) = cache.lock().unwrap().get(&key) {
        return d.clone();
    }
    let d = design_uncached(spec, sys);
    cache.lock().unwrap().insert(key, d.clone());
    d
}

fn design_uncached(spec: &EncodeSpec, sys: &System) -> EncodeDesign {
    let enc = scheme_m1(spec.scheme, spec.venc, spec.balanced);
    // M1 about the block start: ∫G (t − t_s) = M1(t_ref) − M0 (t_s − t_ref)
    let local = |e: &Vec3, a: usize| e[a] + spec.m1_offset[a] - spec.m0[a] * (spec.t_start - spec.t_ref);
    let mut n = 4;
    for e in &enc {
        for a in 0..3 {
            if spec.m0[a] != 0.0 || local(e, a) != 0.0 {
                n = n.max(two_lobe_min_time(spec.m0[a], local(e, a), sys).segments());
            }
        }
    }
    let dt = sys.grad_raster;
    let mut out = EncodeDesign { waves: vec![], m0: vec![], m1: vec![], m2: vec![], duration: n as f64 * dt, shapes: vec![] };
    for e in &enc {
        let sh: Vec<TwoLobe> = (0..3)
            .map(|a| {
                if spec.m0[a] == 0.0 && local(e, a) == 0.0 {
                    TwoLobe { r1: 1, f1: 0, r2: 1, f2: n - 4, g1: 0.0, g2: 0.0 }
                } else {
                    two_lobe_fixed(spec.m0[a], local(e, a), n, sys).expect("shared block length infeasible for an axis")
                }
            })
            .collect();
        let w = Grad::from_axes(&sh[0].samples(), &sh[1].samples(), &sh[2].samples(), dt, spec.t_start);
        let (m0, m1, m2, _) = w.moments(spec.t_ref).last();
        out.waves.push(w);
        out.m0.push(m0);
        out.m1.push(m1);
        out.m2.push(m2);
        out.shapes.push([sh[0], sh[1], sh[2]]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bipolar_exact_moments_and_limits() {
        let sys = System::prisma_xr();
        let d = delta_m1(1.5);
        let s = two_lobe_min_time(0.0, d, &sys);
        let w = Grad::from_axis(0, &s.samples(), sys.grad_raster, 0.0);
        let (m0, m1, _, _) = w.moments(0.0).last();
        assert!(m0[0].abs() < 1e-15 && (m1[0] - d).abs() / d < 1e-12, "{m0:?} {m1:?}");
        let l = w.limits();
        assert!(l.ok_axis(&sys), "{l:?}");
        // a zero-M0 bipolar's M1 does not depend on the origin
        assert!((w.moments(0.123).last().1[0] - d).abs() / d < 1e-9);
        // slew-limited triangles (G peak S r < Gmax): M1 = 2 S r³, duration 4 r, here ≈ 1.08 ms
        let r = (d / (2.0 * sys.smax)).cbrt();
        assert!(w.duration() <= 4.0 * r + 4.0 * sys.grad_raster && w.duration() >= 4.0 * r - 1e-12, "{} vs {}", w.duration(), 4.0 * r);
    }

    #[test]
    fn merged_prephaser_and_encode() {
        let sys = System::prisma_xr();
        let (m0, m1) = (-2.0e-5, 1.0e-8);
        let s = two_lobe_min_time(m0, m1, &sys);
        let w = Grad::from_axis(2, &s.samples(), sys.grad_raster, 0.0);
        let (a, b, _, _) = w.moments(0.0).last();
        assert!((a[2] - m0).abs() < 1e-15 && (b[2] - m1).abs() < 1e-15);
        assert!(w.limits().ok_axis(&sys));
    }

    #[test]
    fn schemes_have_the_right_differences() {
        let sys = System::prisma_xr();
        let cop = |layout| Scheme::Coprime { p: 2, layout };
        for (scheme, ne) in [(Scheme::Referenced, 4), (Scheme::Hadamard, 4), (Scheme::DualVenc { venc_hi: 3.0 }, 7), (cop(DualLayout::Seven), 7), (cop(DualLayout::Eight), 8), (cop(DualLayout::EightHadamard), 8)] {
            let spec = EncodeSpec { m0: [0.0, 0.0, -1.5e-5], m1_offset: [0.0; 3], scheme, venc: 1.5, balanced: true, t_start: 1.2e-3, t_ref: 0.0 };
            let d = design(&spec, &sys);
            assert_eq!(d.waves.len(), ne);
            let want = scheme_m1(scheme, 1.5, true);
            for e in 0..ne {
                for a in 0..3 {
                    assert!((d.m0[e][a] - spec.m0[a]).abs() < 1e-14);
                    let exp = want[e][a] - spec.m0[a] * (spec.t_start - spec.t_ref) + spec.m0[a] * (spec.t_start - spec.t_ref);
                    // M1 about t_ref = scheme value (the offset of M0 · (t_s − t_ref) was compensated in the design)
                    assert!((d.m1[e][a] - exp).abs() < 1e-14, "{scheme:?} e{e} a{a}: {} vs {}", d.m1[e][a], exp);
                }
                assert!(d.waves[e].limits().ok_axis(&sys));
                assert!((d.waves[e].duration() - d.duration).abs() < 1e-12);
            }
            // velocity sensitivity: ΔM1 between the reference and an encode = π / (γ VENC)
            if scheme == Scheme::Referenced {
                assert!(((d.m1[1][0] - d.m1[0][0]) - delta_m1(1.5)).abs() < 1e-14);
            }
        }
    }
}
