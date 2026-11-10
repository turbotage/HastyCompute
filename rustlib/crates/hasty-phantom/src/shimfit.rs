//! B0 shim studies on simulated field maps: which shim hardware and which objective leave the smallest off-resonance
//! *span* over the signal-carrying voxels (the span Δf·T_ro sets the reconstruction's time-segment count L, see
//! research/shimming.md).
//!
//! Bases (field per unit coefficient, Hz, at the given physical points):
//! * [`harmonic_basis`]: solid spherical harmonics (Laplace solutions, as real shim coils produce) of degree ≤ 3 about
//!   a centre: 1, x, y, z | xy, xz, yz, x²−y², 2z²−x²−y² | 7 cubic terms. Coordinates are scaled by `scale` (m) so the
//!   coefficients are Hz at that radius; limits per degree.
//! * [`loop_basis`]: the DC field B_z of receive loops driven as shim elements (integrated RF-receive / B0-shim "AC/DC"
//!   or iPRES arrays): Biot–Savart of the same loops as the coil sensitivities, Hz per ampere, a current limit.
//!
//! [`fit`]: minimise Σ w |f − B c|^p subject to |c_k| ≤ limit_k (+ ridge), p = 2 the usual least-squares shim, p > 2
//! (IRLS) pushes down the tails, i.e. the span. Box constraints by coordinate descent on the IRLS normal equations.

use hasty_mrsim::Vec3;

use crate::coils::Loop;

/// Shim terms sampled at a point set.
#[derive(Clone, Debug, Default)]
pub struct ShimBasis {
    pub names: Vec<String>,
    /// per term: field per unit coefficient (Hz) at every point
    pub cols: Vec<Vec<f64>>,
    /// per term: |coefficient| ≤ limit (f64::INFINITY = free)
    pub limit: Vec<f64>,
}

impl ShimBasis {
    pub fn len(&self) -> usize {
        self.cols.len()
    }
    pub fn is_empty(&self) -> bool {
        self.cols.is_empty()
    }
    /// both bases side by side (e.g. scanner harmonics + a loop array)
    pub fn join(mut self, other: ShimBasis) -> Self {
        self.names.extend(other.names);
        self.cols.extend(other.cols);
        self.limit.extend(other.limit);
        self
    }
}

/// Solid harmonics of degree ≤ `degree` (0..=3) about `centre`, coordinates divided by `scale` (m); `limits[d]` bounds the
/// coefficients of degree d (Hz at radius `scale`).
pub fn harmonic_basis(points: &[Vec3], centre: Vec3, scale: f64, degree: usize, limits: [f64; 4]) -> ShimBasis {
    type Term = (&'static str, usize, fn(f64, f64, f64) -> f64);
    let terms: [Term; 16] = [
        ("Z0", 0, |_, _, _| 1.0),
        ("X", 1, |x, _, _| x),
        ("Y", 1, |_, y, _| y),
        ("Z", 1, |_, _, z| z),
        ("XY", 2, |x, y, _| x * y),
        ("XZ", 2, |x, _, z| x * z),
        ("YZ", 2, |_, y, z| y * z),
        ("X2-Y2", 2, |x, y, _| x * x - y * y),
        ("Z2", 2, |x, y, z| 2.0 * z * z - x * x - y * y),
        ("Z3", 3, |x, y, z| z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y)),
        ("XZ2", 3, |x, y, z| x * (4.0 * z * z - x * x - y * y)),
        ("YZ2", 3, |x, y, z| y * (4.0 * z * z - x * x - y * y)),
        ("Z(X2-Y2)", 3, |x, y, z| z * (x * x - y * y)),
        ("XYZ", 3, |x, y, z| x * y * z),
        ("X3", 3, |x, y, _| x * (x * x - 3.0 * y * y)),
        ("Y3", 3, |x, y, _| y * (3.0 * x * x - y * y)),
    ];
    let mut b = ShimBasis::default();
    for (name, d, f) in terms.iter().filter(|t| t.1 <= degree.min(3)) {
        b.names.push(name.to_string());
        b.cols.push(points.iter().map(|p| f((p[0] - centre[0]) / scale, (p[1] - centre[1]) / scale, (p[2] - centre[2]) / scale)).collect());
        b.limit.push(limits[*d]);
    }
    b
}

/// DC shim fields of `loops` (B_z per ampere, in Hz at γ̄ = 42.577 MHz/T), each current bounded by `imax` (A).
pub fn loop_basis(points: &[Vec3], loops: &[Loop], imax: f64) -> ShimBasis {
    // Loop::b_field is per unit current in units of μ0/4π = 1e-7 T·m/A
    let hz_per = 42.577e6 * 1e-7;
    let mut b = ShimBasis::default();
    for (i, l) in loops.iter().enumerate() {
        b.names.push(format!("loop{i}"));
        b.cols.push(points.iter().map(|p| hz_per * l.b_field(p)[2]).collect());
        b.limit.push(imax);
    }
    b
}

#[derive(Clone, Copy, Debug)]
pub struct ShimOpts {
    /// norm of the residual: 2 = least squares; larger values target the span (IRLS)
    pub p: f64,
    /// Tikhonov ridge relative to the mean diagonal of the normal matrix (stabilises nearly dependent loops)
    pub ridge: f64,
    /// IRLS iterations (p > 2) and coordinate-descent sweeps per solve
    pub irls: usize,
    pub sweeps: usize,
}

impl Default for ShimOpts {
    fn default() -> Self {
        Self { p: 2.0, ridge: 1e-10, irls: 30, sweeps: 500 }
    }
}

#[derive(Clone, Debug)]
pub struct ShimResult {
    pub coef: Vec<f64>,
    /// field after shimming at every point (Hz)
    pub residual: Vec<f64>,
    /// weighted standard deviation, 1–99 % span (weighted percentiles) and peak-to-peak (w > 0), all Hz
    pub sd: f64,
    pub span: f64,
    pub p2p: f64,
    /// coefficients at their limit
    pub saturated: usize,
}

/// weighted p-quantile of `v`
fn wquantile(v: &[f64], w: &[f64], q: f64) -> f64 {
    let mut idx: Vec<usize> = (0..v.len()).filter(|&i| w[i] > 0.0).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let tot: f64 = idx.iter().map(|&i| w[i]).sum();
    let mut acc = 0.0;
    for &i in &idx {
        acc += w[i];
        if acc >= q * tot {
            return v[i];
        }
    }
    idx.last().map_or(0.0, |&i| v[i])
}

/// field statistics with weights: (weighted sd, 1–99 % span, peak-to-peak over w > 0)
pub fn field_stats(f: &[f64], w: &[f64]) -> (f64, f64, f64) {
    let tw: f64 = w.iter().sum::<f64>().max(1e-300);
    let mean = f.iter().zip(w).map(|(a, b)| a * b).sum::<f64>() / tw;
    let sd = (f.iter().zip(w).map(|(a, b)| b * (a - mean).powi(2)).sum::<f64>() / tw).sqrt();
    let span = wquantile(f, w, 0.99) - wquantile(f, w, 0.01);
    let (lo, hi) = f.iter().zip(w).filter(|(_, &b)| b > 0.0).fold((f64::MAX, f64::MIN), |(l, h), (&a, _)| (l.min(a), h.max(a)));
    (sd, span, if hi >= lo { hi - lo } else { 0.0 })
}

/// Shim `field` (Hz at the basis points) with weights `w` (e.g. a heart / vessel ROI, or the expected signal energy).
pub fn fit(field: &[f64], w: &[f64], basis: &ShimBasis, o: &ShimOpts) -> ShimResult {
    let (n, k) = (field.len(), basis.len());
    assert!(basis.cols.iter().all(|c| c.len() == n) && w.len() == n);
    let mut c = vec![0.0f64; k];
    let mut res: Vec<f64> = field.to_vec();
    let rounds = if o.p > 2.0 { o.irls.max(1) } else { 1 };
    let resid = |c: &[f64]| -> Vec<f64> { (0..n).map(|i| field[i] - (0..k).map(|p| basis.cols[p][i] * c[p]).sum::<f64>()).collect() };
    // IRLS with p continuation (2 → p by ×1.5 per round) and Newton-damped updates (Burrus): plain IRLS for large p
    // oscillates
    let mut pk = 2.0f64;
    for round in 0..rounds {
        if round > 0 {
            pk = (pk * 1.5).min(o.p);
        }
        let wr: Vec<f64> = if round == 0 {
            w.to_vec()
        } else {
            let scale = field_stats(&res, w).0.max(1e-9);
            res.iter().zip(w).map(|(r, &a)| a * ((r / scale).abs().max(1e-3)).powf(pk - 2.0)).collect()
        };
        // normal equations A c = r over the basis
        let mut a = vec![0.0f64; k * k];
        let mut rhs = vec![0.0f64; k];
        for i in 0..n {
            if wr[i] == 0.0 {
                continue;
            }
            for p in 0..k {
                let bp = basis.cols[p][i] * wr[i];
                rhs[p] += bp * field[i];
                for q in p..k {
                    a[p * k + q] += bp * basis.cols[q][i];
                }
            }
        }
        for p in 0..k {
            for q in 0..p {
                a[p * k + q] = a[q * k + p];
            }
        }
        let md = (0..k).map(|p| a[p * k + p]).sum::<f64>() / k.max(1) as f64;
        for p in 0..k {
            a[p * k + p] += o.ridge * md;
        }
        let cnew = solve_box(&a, &rhs, k, &basis.limit, &c, o.sweeps);
        // damping 1 / (p − 1) of the step (the exact Newton step of the p-norm for these weights); plain LS: full
        let q = if round == 0 { 1.0 } else { 1.0 / (pk - 1.0) };
        for p in 0..k {
            c[p] = (q * cnew[p] + (1.0 - q) * c[p]).clamp(-basis.limit[p], basis.limit[p]);
        }
        res = resid(&c);
    }
    let (sd, span, p2p) = field_stats(&res, w);
    let saturated = (0..k).filter(|&p| basis.limit[p].is_finite() && c[p].abs() >= 0.999 * basis.limit[p]).count();
    ShimResult { coef: c, residual: res, sd, span, p2p, saturated }
}

/// min ½ cᵀAc − rhsᵀc subject to |c_p| ≤ limit_p: the unconstrained solution if it is feasible, else coordinate
/// descent with clamping from the clamped solution (convex, converges)
fn solve_box(a: &[f64], rhs: &[f64], k: usize, limit: &[f64], warm: &[f64], sweeps: usize) -> Vec<f64> {
    let mut c: Vec<f64> = warm.to_vec();
    if let Some(u) = solve_dense(a, rhs, k) {
        if (0..k).all(|p| u[p].abs() <= limit[p]) {
            return u;
        }
        c = (0..k).map(|p| u[p].clamp(-limit[p], limit[p])).collect();
    }
    for _ in 0..sweeps {
        let mut change = 0.0f64;
        for p in 0..k {
            if a[p * k + p] <= 0.0 {
                continue;
            }
            let s: f64 = (0..k).filter(|&q| q != p).map(|q| a[p * k + q] * c[q]).sum();
            let v = ((rhs[p] - s) / a[p * k + p]).clamp(-limit[p], limit[p]);
            change = change.max((v - c[p]).abs() * a[p * k + p].sqrt());
            c[p] = v;
        }
        if change < 1e-9 {
            break;
        }
    }
    c
}

/// a (k × k, row-major) x = b by Gaussian elimination with partial pivoting; None if singular
fn solve_dense(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();
    for col in 0..k {
        let piv = (col..k).max_by(|&i, &j| m[i * k + col].abs().partial_cmp(&m[j * k + col].abs()).unwrap())?;
        if m[piv * k + col].abs() < 1e-300 {
            return None;
        }
        if piv != col {
            for c in 0..k {
                m.swap(col * k + c, piv * k + c);
            }
            x.swap(col, piv);
        }
        let p = m[col * k + col];
        for r in col + 1..k {
            let f = m[r * k + col] / p;
            if f != 0.0 {
                for c in col..k {
                    m[r * k + c] -= f * m[col * k + c];
                }
                x[r] -= f * x[col];
            }
        }
    }
    for col in (0..k).rev() {
        let s: f64 = (col + 1..k).map(|c| m[col * k + c] * x[c]).sum();
        x[col] = (x[col] - s) / m[col * k + col];
    }
    Some(x)
}

/// The reconstruction's segment count for a span (Hz) and readout length (s): recon_p2np's `--nseg auto` rule.
pub fn segments_for(span_hz: f64, t_ro: f64) -> usize {
    ((1.3 * span_hz * t_ro + 3.5).ceil() as usize).max(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a field made of shim terms is removed exactly; limits bind; p > 2 lowers the span of a heavy-tailed field
    #[test]
    fn harmonic_fit_exact_limits_and_span() {
        let pts: Vec<Vec3> = (0..2000).map(|i| {
            let t = i as f64 * 0.618;
            [0.08 * (t * 1.3).sin(), 0.07 * (t * 0.7).cos(), 0.05 * (t * 2.1).sin()]
        }).collect();
        let b = harmonic_basis(&pts, [0.0; 3], 0.1, 2, [f64::INFINITY; 4]);
        assert_eq!(b.len(), 9);
        let truth: Vec<f64> = (0..pts.len()).map(|i| 30.0 * b.cols[1][i] - 12.0 * b.cols[4][i] + 5.0 * b.cols[8][i] + 7.0).collect();
        let w = vec![1.0; pts.len()];
        let r = fit(&truth, &w, &b, &ShimOpts::default());
        assert!(r.sd < 1e-6, "exact shim terms left {}", r.sd);
        // a limit on the linear terms leaves part of the gradient
        let bl = harmonic_basis(&pts, [0.0; 3], 0.1, 2, [f64::INFINITY, 10.0, f64::INFINITY, f64::INFINITY]);
        let rl = fit(&truth, &w, &bl, &ShimOpts::default());
        assert!(rl.saturated >= 1 && rl.sd > 1.0);
        // heavy tails: p = 8 reduces the 1–99 % span relative to least squares
        let tail: Vec<f64> = (0..pts.len()).map(|i| truth[i] + if i % 50 == 0 { 80.0 * pts[i][0] / 0.08 } else { 0.0 } + 40.0 * pts[i][2] * pts[i][2] / 0.0025).collect();
        let r2 = fit(&tail, &w, &b, &ShimOpts::default());
        let r8 = fit(&tail, &w, &b, &ShimOpts { p: 8.0, ..Default::default() });
        assert!(r8.p2p <= r2.p2p * 1.001, "p = 8 did not lower the peak-to-peak: {} vs {}", r8.p2p, r2.p2p);
    }
}
