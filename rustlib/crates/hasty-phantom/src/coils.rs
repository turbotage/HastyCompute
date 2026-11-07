//! Receive arrays of circular loops: quasi-static Biot–Savart sensitivities (B1⁻ ∝ B_x − i B_y, B0 ∥ z; near field
//! softened over 5 mm, loops clear the curved body surface by ≥ 1.2 cm over their whole footprint) and the
//! body-loss noise covariance Ψ_ij ∝ ∫_body A_i · A_j dV (+ a small diagonal for coil/preamp noise).

use hasty_mrsim::Vec3;
use num_complex::Complex64;

use crate::volume::Volume;

#[derive(Clone, Copy, Debug)]
pub struct Loop {
    pub center: Vec3,
    /// unit normal
    pub normal: Vec3,
    pub radius: f64,
}

const NSEG: usize = 96;
/// near-field softening (m): the wire has a finite cross-section and sits in a housing, so the field at tissue
/// points is never the singular 1/r of an ideal filament; Plummer softening r² → r² + a²
const SOFT: f64 = 0.005;

fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn normalize(a: Vec3) -> Vec3 {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
}

impl Loop {
    fn segments(&self) -> Vec<(Vec3, Vec3)> {
        // orthonormal in-plane basis
        let n = normalize(self.normal);
        let t = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
        let e1 = normalize(cross(&n, &t));
        let e2 = cross(&n, &e1);
        (0..NSEG)
            .map(|s| {
                let (a0, a1) = (2.0 * std::f64::consts::PI * s as f64 / NSEG as f64, 2.0 * std::f64::consts::PI * (s + 1) as f64 / NSEG as f64);
                let p = |a: f64| [0, 1, 2].map(|q| self.center[q] + self.radius * (a.cos() * e1[q] + a.sin() * e2[q]));
                let (p0, p1) = (p(a0), p(a1));
                let mid = [0, 1, 2].map(|q| 0.5 * (p0[q] + p1[q]));
                let dl = [0, 1, 2].map(|q| p1[q] - p0[q]);
                (mid, dl)
            })
            .collect()
    }
    /// magnetic field per unit current (units of μ0/4π)
    pub fn b_field(&self, x: &Vec3) -> Vec3 {
        let mut b = [0.0; 3];
        for (m, dl) in self.segments() {
            let r = [x[0] - m[0], x[1] - m[1], x[2] - m[2]];
            let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + SOFT * SOFT;
            let c = cross(&dl, &r);
            let f = 1.0 / (r2 * r2.sqrt());
            for q in 0..3 {
                b[q] += c[q] * f;
            }
        }
        b
    }
    /// vector potential per unit current (units of μ0/4π)
    pub fn a_field(&self, x: &Vec3) -> Vec3 {
        let mut a = [0.0; 3];
        for (m, dl) in self.segments() {
            let r = ((x[0] - m[0]).powi(2) + (x[1] - m[1]).powi(2) + (x[2] - m[2]).powi(2) + SOFT * SOFT).sqrt();
            for q in 0..3 {
                a[q] += dl[q] / r;
            }
        }
        a
    }
    pub fn sensitivity(&self, x: &Vec3) -> Complex64 {
        let b = self.b_field(x);
        Complex64::new(b[0], -b[1])
    }
}

/// Anterior and posterior arrays of `n_lr × n_si` loops each, hugging the body surface over the z-range [z0, z1].
pub fn body_arrays(body: &Volume<u8>, n_lr: usize, n_si: usize, radius: f64, z0: f64, z1: f64) -> Vec<Loop> {
    let d = body.dims;
    let vs = body.voxel_size();
    // body extent in x over the z range, anterior (min y) / posterior (max y) surfaces per (x, z) column
    let col = |x: f64, z: f64| -> Option<(f64, f64)> {
        let v = body.to_voxel(&[x, 0.0, z]);
        let (i, k) = (v[0].round() as isize, v[2].round() as isize);
        if i < 0 || k < 0 || i as usize >= d[0] || k as usize >= d[2] {
            return None;
        }
        let ys: Vec<usize> = (0..d[1]).filter(|&j| body.get(i as usize, j, k as usize) != 0).collect();
        if ys.is_empty() {
            return None;
        }
        let y0 = body.to_phys([0.0, ys[0] as f64, 0.0])[1];
        let y1 = body.to_phys([0.0, *ys.last().unwrap() as f64, 0.0])[1];
        Some((y0, y1))
    };
    // lateral extent at the middle z
    let zm = 0.5 * (z0 + z1);
    let km = body.to_voxel(&[0.0, 0.0, zm])[2].round().clamp(0.0, (d[2] - 1) as f64) as usize;
    let xs: Vec<f64> = (0..d[0]).filter(|&i| (0..d[1]).any(|j| body.get(i, j, km) != 0)).map(|i| body.to_phys([i as f64, 0.0, 0.0])[0]).collect();
    let (xmin, xmax) = (xs.first().copied().unwrap_or(-0.15), xs.last().copied().unwrap_or(0.15));
    let span_x = (xmax - xmin) * 0.8;
    let xc = 0.5 * (xmin + xmax);
    let mut loops = Vec::new();
    let gap = 0.012 + vs[1];
    for side in 0..2 {
        for a in 0..n_lr {
            for b in 0..n_si {
                let x = xc + span_x * ((a as f64 + 0.5) / n_lr as f64 - 0.5);
                let z = z0 + (z1 - z0) * (b as f64 + 0.5) / n_si as f64;
                // the whole loop (not only its centre) clears the curved body surface by `gap`: the most anterior /
                // posterior surface point under the loop's footprint
                let (mut ya, mut yp) = col(x, z).unwrap_or((-0.12, 0.12));
                for s in 0..16 {
                    for f in [0.5, 1.0] {
                        let a = 2.0 * std::f64::consts::PI * s as f64 / 16.0;
                        if let Some((y0, y1)) = col(x + f * radius * a.cos(), z + f * radius * a.sin()) {
                            ya = ya.min(y0);
                            yp = yp.max(y1);
                        }
                    }
                }
                let (y, nrm) = if side == 0 { (ya - gap, [0.0, -1.0, 0.0]) } else { (yp + gap, [0.0, 1.0, 0.0]) };
                loops.push(Loop { center: [x, y, z], normal: nrm, radius });
            }
        }
    }
    loops
}

/// Ψ_ij = Σ_body A_i·A_j ΔV (body loss) + `diag_frac`·mean(diag) δ_ij, normalised to unit mean diagonal.
pub fn noise_covariance(loops: &[Loop], body: &Volume<u8>, step: usize, diag_frac: f64) -> Vec<Vec<f64>> {
    let n = loops.len();
    let mut psi = vec![vec![0.0; n]; n];
    let d = body.dims;
    for k in (0..d[2]).step_by(step) {
        for j in (0..d[1]).step_by(step) {
            for i in (0..d[0]).step_by(step) {
                if body.get(i, j, k) == 0 {
                    continue;
                }
                let x = body.to_phys([i as f64, j as f64, k as f64]);
                let a: Vec<Vec3> = loops.iter().map(|l| l.a_field(&x)).collect();
                for p in 0..n {
                    for q in p..n {
                        let v = a[p][0] * a[q][0] + a[p][1] * a[q][1] + a[p][2] * a[q][2];
                        psi[p][q] += v;
                    }
                }
            }
        }
    }
    let md = (0..n).map(|p| psi[p][p]).sum::<f64>() / n as f64;
    for p in 0..n {
        for q in 0..p {
            psi[p][q] = psi[q][p];
        }
        psi[p][p] += diag_frac * md;
    }
    let md = (0..n).map(|p| psi[p][p]).sum::<f64>() / n as f64;
    for row in psi.iter_mut() {
        for v in row.iter_mut() {
            *v /= md;
        }
    }
    psi
}

/// Coil sensitivities tabulated on a coarse grid (smooth inside the body), sampled trilinearly per point.
pub struct SensitivityTable {
    pub re: Vec<Volume<f32>>,
    pub im: Vec<Volume<f32>>,
}

impl SensitivityTable {
    pub fn new(loops: &[Loop], like: &Volume<u8>, step: usize) -> Self {
        let d = like.dims;
        let cd = [d[0].div_ceil(step) + 1, d[1].div_ceil(step) + 1, d[2].div_ceil(step) + 1];
        let mut aff = like.affine;
        for r in 0..3 {
            for c in 0..3 {
                aff[r][c] *= step as f64;
            }
        }
        let mut re: Vec<Volume<f32>> = loops.iter().map(|_| Volume::new(cd, aff)).collect();
        let mut im: Vec<Volume<f32>> = loops.iter().map(|_| Volume::new(cd, aff)).collect();
        let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        let results: Vec<Vec<(usize, Vec<Complex64>)>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..nthreads)
                .map(|t| {
                    let re0 = &re[0];
                    s.spawn(move || {
                        let mut out = Vec::new();
                        let mut q = t;
                        while q < re0.len() {
                            let (i, j, k) = (q % cd[0], (q / cd[0]) % cd[1], q / (cd[0] * cd[1]));
                            let x = re0.to_phys([i as f64, j as f64, k as f64]);
                            out.push((q, loops.iter().map(|l| l.sensitivity(&x)).collect()));
                            q += nthreads;
                        }
                        out
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for part in results {
            for (q, s) in part {
                for c in 0..loops.len() {
                    re[c].data[q] = s[c].re as f32;
                    im[c].data[q] = s[c].im as f32;
                }
            }
        }
        // normalise: mean root-sum-of-squares over the body = 1
        let (mut acc, mut cnt) = (0.0f64, 0usize);
        for q in 0..re[0].len() {
            let (i, j, k) = (q % cd[0], (q / cd[0]) % cd[1], q / (cd[0] * cd[1]));
            let (ii, jj, kk) = ((i * step).min(d[0] - 1), (j * step).min(d[1] - 1), (k * step).min(d[2] - 1));
            if like.get(ii, jj, kk) != 0 {
                acc += (0..loops.len()).map(|c| (re[c].data[q] as f64).powi(2) + (im[c].data[q] as f64).powi(2)).sum::<f64>().sqrt();
                cnt += 1;
            }
        }
        let f = (cnt.max(1) as f64 / acc.max(1e-30)) as f32;
        for c in 0..loops.len() {
            for v in re[c].data.iter_mut().chain(im[c].data.iter_mut()) {
                *v *= f;
            }
        }
        Self { re, im }
    }
    pub fn nc(&self) -> usize {
        self.re.len()
    }
    pub fn at(&self, x: &Vec3, out: &mut [Complex64]) {
        let v = self.re[0].to_voxel(x);
        for c in 0..self.nc() {
            out[c] = Complex64::new(self.re[c].sample_voxel(v, 0.0) as f64, self.im[c].sample_voxel(v, 0.0) as f64);
        }
    }
}
