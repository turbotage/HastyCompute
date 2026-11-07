//! Low-parameter respiratory motion model from breathing-resolved images: one smooth displacement field S shared by
//! all breathing states, scaled by each state's surrogate value r_b:
//!
//!   I_b(x) ≈ I_0(x − r_b S(x)),   S = cubic B-spline on a coarse control lattice (a few cm),
//!
//! fitted by least squares over all bins with a first-order smoothness penalty on the control coefficients
//! (L-BFGS, coarse-to-fine). This is the "low-parameter diffeomorphism" prior for the dynamic B0 / R2* maps: the
//! maps of any breathing state r are the reference maps pulled back along r S. Invertibility is checked by the
//! Jacobian determinant of x ↦ x − r S(x) at the largest r.
//!
//! Units: positions and displacements in voxels of the image grid (x fastest); the caller converts to metres.

/// uniform cubic B-spline basis values at fraction t ∈ [0, 1) for the four neighbouring control points
fn bspline(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [(1.0 - t).powi(3) / 6.0, (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0, (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0, t3 / 6.0]
}

/// cubic B-spline vector field on a control lattice with spacing `sp` voxels covering the image (one extra point at
/// each side); coefficients [3][nctrl] (x-fastest lattice)
#[derive(Clone, Debug)]
pub struct BSplineField {
    pub dims: [usize; 3],
    pub sp: f64,
    pub nc: [usize; 3],
    pub c: Vec<f64>,
}

impl BSplineField {
    pub fn zeros(dims: [usize; 3], sp: f64) -> Self {
        let nc = dims.map(|d| ((d as f64 - 1.0) / sp).floor() as usize + 4);
        Self { dims, sp, nc, c: vec![0.0; 3 * nc[0] * nc[1] * nc[2]] }
    }
    pub fn n_ctrl(&self) -> usize {
        self.nc[0] * self.nc[1] * self.nc[2]
    }
    /// control-point indices (flattened) and weights of voxel position p (voxel units)
    fn support(&self, p: [f64; 3]) -> ([usize; 64], [f64; 64]) {
        let mut base = [0isize; 3];
        let mut w = [[0.0; 4]; 3];
        for a in 0..3 {
            let u = p[a] / self.sp;
            let f = u.floor();
            base[a] = f as isize; // control index of the first of the 4 (offset +1 lattice margin below)
            w[a] = bspline(u - f);
        }
        let mut idx = [0usize; 64];
        let mut wt = [0.0; 64];
        let mut n = 0;
        for k in 0..4 {
            for j in 0..4 {
                for i in 0..4 {
                    let ci = [base[0] + i as isize, base[1] + j as isize, base[2] + k as isize].map(|x| x.max(0) as usize);
                    let ci = [ci[0].min(self.nc[0] - 1), ci[1].min(self.nc[1] - 1), ci[2].min(self.nc[2] - 1)];
                    idx[n] = ci[0] + self.nc[0] * (ci[1] + self.nc[1] * ci[2]);
                    wt[n] = w[0][i] * w[1][j] * w[2][k];
                    n += 1;
                }
            }
        }
        (idx, wt)
    }
    pub fn eval(&self, p: [f64; 3]) -> [f64; 3] {
        let (idx, wt) = self.support(p);
        let n = self.n_ctrl();
        let mut v = [0.0; 3];
        for q in 0..64 {
            for a in 0..3 {
                v[a] += wt[q] * self.c[a * n + idx[q]];
            }
        }
        v
    }
    /// the field on every voxel [3][nv]
    pub fn dense(&self) -> Vec<f64> {
        let nv = self.dims[0] * self.dims[1] * self.dims[2];
        let mut out = vec![0.0; 3 * nv];
        for z in 0..self.dims[2] {
            for y in 0..self.dims[1] {
                for x in 0..self.dims[0] {
                    let q = x + self.dims[0] * (y + self.dims[1] * z);
                    let v = self.eval([x as f64, y as f64, z as f64]);
                    for a in 0..3 {
                        out[a * nv + q] = v[a];
                    }
                }
            }
        }
        out
    }
    /// the same field on a lattice of spacing `sp/2` (exact refinement by evaluation + least-squares-free resampling:
    /// coefficients set so the refined spline interpolates the old one at the new control points)
    pub fn refined(&self) -> Self {
        let mut f = Self::zeros(self.dims, self.sp / 2.0);
        let n = f.n_ctrl();
        for k in 0..f.nc[2] {
            for j in 0..f.nc[1] {
                for i in 0..f.nc[0] {
                    let p = [i as f64 * f.sp, j as f64 * f.sp, k as f64 * f.sp].map(|x| x - f.sp);
                    let v = self.eval(p.map(|x| x.max(0.0)));
                    let q = i + f.nc[0] * (j + f.nc[1] * k);
                    for a in 0..3 {
                        f.c[a * n + q] = v[a];
                    }
                }
            }
        }
        f
    }
}

/// trilinear value and gradient of image `im` (dims) at p (voxel units, clamped)
fn sample(im: &[f32], d: [usize; 3], p: [f64; 3]) -> (f64, [f64; 3]) {
    let mut i0 = [0usize; 3];
    let mut fr = [0.0; 3];
    for a in 0..3 {
        let x = p[a].clamp(0.0, d[a] as f64 - 1.000001);
        i0[a] = x.floor() as usize;
        fr[a] = x - i0[a] as f64;
    }
    let at = |i: usize, j: usize, k: usize| im[(i0[0] + i) + d[0] * ((i0[1] + j) + d[1] * (i0[2] + k))] as f64;
    let c = [[[at(0, 0, 0), at(1, 0, 0)], [at(0, 1, 0), at(1, 1, 0)]], [[at(0, 0, 1), at(1, 0, 1)], [at(0, 1, 1), at(1, 1, 1)]]];
    let (fx, fy, fz) = (fr[0], fr[1], fr[2]);
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let c00 = lerp(c[0][0][0], c[0][0][1], fx);
    let c10 = lerp(c[0][1][0], c[0][1][1], fx);
    let c01 = lerp(c[1][0][0], c[1][0][1], fx);
    let c11 = lerp(c[1][1][0], c[1][1][1], fx);
    let v = lerp(lerp(c00, c10, fy), lerp(c01, c11, fy), fz);
    let gx = lerp(lerp(c[0][0][1] - c[0][0][0], c[0][1][1] - c[0][1][0], fy), lerp(c[1][0][1] - c[1][0][0], c[1][1][1] - c[1][1][0], fy), fz);
    let gy = lerp(c10 - c00, c11 - c01, fz);
    let gz = lerp(c00, c10, fy);
    let gz = lerp(c01, c11, fy) - gz;
    (v, [gx, gy, gz])
}

pub struct SharedFit {
    pub field: BSplineField,
    /// relative residual Σ_b ‖I_b − I_0∘(x − r_b S)‖² / Σ_b ‖I_b − I_0‖² (1 = no improvement over no motion)
    pub rel_resid: f64,
    /// minimum Jacobian determinant of x ↦ x − r_max S(x)
    pub min_jac: f64,
    pub history: Vec<f64>,
}

/// objective and gradient w.r.t. the coefficients; voxels where `mask` is false are ignored
fn objective(f: &BSplineField, ims: &[Vec<f32>], r: &[f64], mask: &[bool], lam: f64, grad: &mut [f64]) -> f64 {
    let d = f.dims;
    let n = f.n_ctrl();
    grad.iter_mut().for_each(|g| *g = 0.0);
    let nt = std::thread::available_parallelism().map_or(1, |x| x.get()).min(32);
    let nz = d[2];
    let parts: Vec<(f64, Vec<f64>)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..nt)
            .map(|t| {
                sc.spawn(move || {
                    let mut g = vec![0.0; 3 * n];
                    let mut e = 0.0;
                    for z in (t..nz).step_by(nt) {
                        for y in 0..d[1] {
                            for x in 0..d[0] {
                                let q = x + d[0] * (y + d[1] * z);
                                if !mask[q] {
                                    continue;
                                }
                                let p = [x as f64, y as f64, z as f64];
                                let (idx, wt) = f.support(p);
                                let mut s = [0.0; 3];
                                for k in 0..64 {
                                    for a in 0..3 {
                                        s[a] += wt[k] * f.c[a * n + idx[k]];
                                    }
                                }
                                let mut dq = [0.0; 3];
                                for b in 1..ims.len() {
                                    let pb = [p[0] - r[b] * s[0], p[1] - r[b] * s[1], p[2] - r[b] * s[2]];
                                    let (v, gr) = sample(&ims[0], d, pb);
                                    let res = v - ims[b][q] as f64;
                                    e += res * res;
                                    // ∂res/∂S = −r_b ∇I_0
                                    for a in 0..3 {
                                        dq[a] += 2.0 * res * (-r[b]) * gr[a];
                                    }
                                }
                                for k in 0..64 {
                                    for a in 0..3 {
                                        g[a * n + idx[k]] += wt[k] * dq[a];
                                    }
                                }
                            }
                        }
                    }
                    (e, g)
                })
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let mut e = 0.0;
    for (pe, pg) in parts {
        e += pe;
        for (a, b) in grad.iter_mut().zip(pg) {
            *a += b;
        }
    }
    // smoothness: λ Σ (c_i − c_j)² over lattice neighbours
    let nc = f.nc;
    for a in 0..3 {
        for k in 0..nc[2] {
            for j in 0..nc[1] {
                for i in 0..nc[0] {
                    let q = i + nc[0] * (j + nc[1] * k);
                    for (di, dj, dk) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
                        let (i2, j2, k2) = (i + di, j + dj, k + dk);
                        if i2 >= nc[0] || j2 >= nc[1] || k2 >= nc[2] {
                            continue;
                        }
                        let q2 = i2 + nc[0] * (j2 + nc[1] * k2);
                        let dlt = f.c[a * n + q] - f.c[a * n + q2];
                        e += lam * dlt * dlt;
                        grad[a * n + q] += 2.0 * lam * dlt;
                        grad[a * n + q2] -= 2.0 * lam * dlt;
                    }
                }
            }
        }
    }
    e
}

/// L-BFGS (two-loop) with backtracking line search on the coefficient vector
fn lbfgs(f: &mut BSplineField, ims: &[Vec<f32>], r: &[f64], mask: &[bool], lam: f64, iters: usize, hist: &mut Vec<f64>) {
    let m = 7;
    let nx = f.c.len();
    let mut g = vec![0.0; nx];
    let mut e = objective(f, ims, r, mask, lam, &mut g);
    let mut ss: Vec<Vec<f64>> = vec![];
    let mut ys: Vec<Vec<f64>> = vec![];
    for _ in 0..iters {
        // direction d = −H g
        let mut q = g.clone();
        let mut al = vec![0.0; ss.len()];
        for i in (0..ss.len()).rev() {
            let rho = 1.0 / dot(&ys[i], &ss[i]);
            al[i] = rho * dot(&ss[i], &q);
            for (qj, yj) in q.iter_mut().zip(&ys[i]) {
                *qj -= al[i] * yj;
            }
        }
        let gam = if let (Some(s), Some(y)) = (ss.last(), ys.last()) { dot(s, y) / dot(y, y) } else { 1.0 / norm(&g).max(1e-30) };
        q.iter_mut().for_each(|x| *x *= gam);
        for i in 0..ss.len() {
            let rho = 1.0 / dot(&ys[i], &ss[i]);
            let b = rho * dot(&ys[i], &q);
            for (qj, sj) in q.iter_mut().zip(&ss[i]) {
                *qj += (al[i] - b) * sj;
            }
        }
        let dvec: Vec<f64> = q.iter().map(|x| -x).collect();
        let slope = dot(&g, &dvec);
        if slope >= 0.0 {
            ss.clear();
            ys.clear();
            continue;
        }
        let c0 = f.c.clone();
        let mut step = 1.0;
        let mut g2 = vec![0.0; nx];
        let mut ok = false;
        for _ in 0..20 {
            for i in 0..nx {
                f.c[i] = c0[i] + step * dvec[i];
            }
            let e2 = objective(f, ims, r, mask, lam, &mut g2);
            if e2 <= e + 1e-4 * step * slope {
                let s: Vec<f64> = (0..nx).map(|i| f.c[i] - c0[i]).collect();
                let y: Vec<f64> = (0..nx).map(|i| g2[i] - g[i]).collect();
                if dot(&s, &y) > 1e-12 * norm(&s) * norm(&y) {
                    ss.push(s);
                    ys.push(y);
                    if ss.len() > m {
                        ss.remove(0);
                        ys.remove(0);
                    }
                }
                e = e2;
                g = g2.clone();
                ok = true;
                break;
            }
            step *= 0.5;
        }
        if !ok {
            f.c = c0;
            break;
        }
        hist.push(e);
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// fit the shared field: `ims[0]` the reference (end-expiration), `r[b]` the surrogate of bin b (r[0] ignored),
/// coarse-to-fine from control spacing `sp0` voxels over `levels` halvings; λ per coefficient-difference²,
/// relative to the image energy
pub fn fit_shared(ims: &[Vec<f32>], r: &[f64], dims: [usize; 3], mask: &[bool], sp0: f64, levels: usize, lam_rel: f64, iters: usize) -> SharedFit {
    let e0: f64 = (1..ims.len()).map(|b| (0..mask.len()).filter(|&q| mask[q]).map(|q| (ims[b][q] as f64 - ims[0][q] as f64).powi(2)).sum::<f64>()).sum();
    let mut f = BSplineField::zeros(dims, sp0);
    let mut hist = vec![];
    for l in 0..levels {
        if l > 0 {
            f = f.refined();
        }
        let lam = lam_rel * e0 / f.n_ctrl() as f64;
        lbfgs(&mut f, ims, r, mask, lam, iters, &mut hist);
    }
    let mut g = vec![0.0; f.c.len()];
    let lam_last = 0.0;
    let e = objective(&f, ims, r, mask, lam_last, &mut g);
    let rmax = r.iter().cloned().fold(0.0, f64::max);
    // Jacobian determinant of x − r S(x) by central differences of the dense field
    let s = f.dense();
    let nv = dims[0] * dims[1] * dims[2];
    let mut min_jac = f64::MAX;
    for z in 1..dims[2] - 1 {
        for y in 1..dims[1] - 1 {
            for x in 1..dims[0] - 1 {
                let q = x + dims[0] * (y + dims[1] * z);
                let st = [1, dims[0], dims[0] * dims[1]];
                let mut jm = [[0.0; 3]; 3];
                for a in 0..3 {
                    for b in 0..3 {
                        let dv = 0.5 * (s[a * nv + q + st[b]] - s[a * nv + q - st[b]]);
                        jm[a][b] = if a == b { 1.0 } else { 0.0 } - rmax * dv;
                    }
                }
                let det = jm[0][0] * (jm[1][1] * jm[2][2] - jm[1][2] * jm[2][1]) - jm[0][1] * (jm[1][0] * jm[2][2] - jm[1][2] * jm[2][0])
                    + jm[0][2] * (jm[1][0] * jm[2][1] - jm[1][1] * jm[2][0]);
                min_jac = min_jac.min(det);
            }
        }
    }
    SharedFit { field: f, rel_resid: e / e0.max(1e-300), min_jac, history: hist }
}

/// pull an image back along −r S: out(x) = im(x − r S(x)) (e.g. reference maps → breathing state r)
pub fn warp(im: &[f32], dims: [usize; 3], s_dense: &[f64], r: f64) -> Vec<f32> {
    let nv = dims[0] * dims[1] * dims[2];
    (0..nv)
        .map(|q| {
            let (x, y, z) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
            let p = [x as f64 - r * s_dense[q], y as f64 - r * s_dense[nv + q], z as f64 - r * s_dense[2 * nv + q]];
            sample(im, dims, p).0 as f32
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_smooth_shared_displacement() {
        // reference: blobs on a 40×40×24 grid; truth S: a smooth SI-dominant field (up to 4 voxels at r = 1)
        let d = [40usize, 40, 24];
        let nv = d[0] * d[1] * d[2];
        let blob = |p: [f64; 3]| -> f64 {
            let cs = [[14.0, 16.0, 10.0, 5.0], [26.0, 22.0, 13.0, 4.0], [20.0, 28.0, 8.0, 3.5], [18.0, 12.0, 15.0, 3.0]];
            cs.iter().map(|c| (-((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)) / (2.0 * c[3] * c[3])).exp()).sum()
        };
        let s_true = |p: [f64; 3]| -> [f64; 3] {
            let w = (-((p[0] - 20.0).powi(2) + (p[1] - 20.0).powi(2)) / 400.0).exp();
            [0.5 * w, 1.0 * w, 4.0 * w]
        };
        let rs = [0.0, 0.33, 0.66, 1.0];
        let ims: Vec<Vec<f32>> = rs
            .iter()
            .map(|&r| {
                (0..nv)
                    .map(|q| {
                        let p = [(q % d[0]) as f64, ((q / d[0]) % d[1]) as f64, (q / (d[0] * d[1])) as f64];
                        let s = s_true(p);
                        blob([p[0] - r * s[0], p[1] - r * s[1], p[2] - r * s[2]]) as f32
                    })
                    .collect()
            })
            .collect();
        let mask = vec![true; nv];
        let fit = fit_shared(&ims, &rs, d, &mask, 10.0, 2, 1e-4, 40);
        assert!(fit.rel_resid < 0.02, "relative residual {}", fit.rel_resid);
        assert!(fit.min_jac > 0.5, "Jacobian {}", fit.min_jac);
        // displacement error where the image has structure
        let sd = fit.field.dense();
        let (mut num, mut den) = (0.0, 0.0);
        for q in 0..nv {
            let p = [(q % d[0]) as f64, ((q / d[0]) % d[1]) as f64, (q / (d[0] * d[1])) as f64];
            if blob(p) < 0.2 {
                continue;
            }
            let st = s_true(p);
            for a in 0..3 {
                num += (sd[a * nv + q] - st[a]).powi(2);
                den += st[a].powi(2);
            }
        }
        let rel = (num / den).sqrt();
        assert!(rel < 0.15, "relative displacement error {rel}");
    }
}
