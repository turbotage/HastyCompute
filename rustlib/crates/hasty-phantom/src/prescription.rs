//! Slab prescription for thoracic-aorta 4D flow: sagittal-oblique "candy-cane" slab fitted to the aortic centerline.

use hasty_mrsim::{Mat3, Vec3};


#[derive(Clone, Debug)]
pub struct Prescription {
    /// logical → physical (columns: logical x, y, z = slab normal)
    pub rot: Mat3,
    /// FOV centre in subject coordinates, m
    pub center: Vec3,
    /// logical FOV, m
    pub fov: Vec3,
    pub res: Vec3,
}

/// eigen-decomposition of a symmetric 3×3 matrix (Jacobi); eigenvalues descending, eigenvectors as columns
pub fn sym_eig3(a: [[f64; 3]; 3]) -> ([f64; 3], Mat3) {
    let mut m = a;
    let mut v = hasty_mrsim::IDENTITY;
    for _ in 0..100 {
        let (mut p, mut q, mut off) = (0, 1, 0.0);
        for i in 0..3 {
            for j in i + 1..3 {
                if m[i][j].abs() > off {
                    off = m[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if off < 1e-18 {
            break;
        }
        let th = 0.5 * (2.0 * m[p][q]).atan2(m[q][q] - m[p][p]);
        let (s, c) = th.sin_cos();
        let mut r = hasty_mrsim::IDENTITY;
        r[p][p] = c;
        r[q][q] = c;
        r[p][q] = s;
        r[q][p] = -s;
        m = hasty_mrsim::mat_mul(&hasty_mrsim::mat_mul(&hasty_mrsim::transpose(&r), &m), &r);
        v = hasty_mrsim::mat_mul(&v, &r);
    }
    let mut idx = [0, 1, 2];
    idx.sort_by(|&a, &b| m[b][b].partial_cmp(&m[a][a]).unwrap());
    let vals = idx.map(|i| m[i][i]);
    let mut vecs = [[0.0; 3]; 3];
    for (c, &i) in idx.iter().enumerate() {
        for r in 0..3 {
            vecs[r][c] = v[r][i];
        }
    }
    (vals, vecs)
}

/// Fit the slab to centerline points (normally the main aortic path; supra-aortic branches are cut by the slab as in
/// practice): logical x = main axis (oriented superior), y = second axis, z = slab normal; the slab covers the points
/// ± (`radius` + `margin`), rounded to the resolution. In-plane FOV `fov_xy`.
pub fn aorta_slab(points: &[Vec3], radius: f64, res: f64, fov_xy: [f64; 2], margin: f64) -> Prescription {
    let n = points.len() as f64;
    let mut c = [0.0; 3];
    for p in points {
        for q in 0..3 {
            c[q] += p[q] / n;
        }
    }
    let mut cov = [[0.0; 3]; 3];
    for p in points {
        for i in 0..3 {
            for j in 0..3 {
                cov[i][j] += (p[i] - c[i]) * (p[j] - c[j]) / n;
            }
        }
    }
    let (_, e) = sym_eig3(cov);
    let col = |k: usize| [e[0][k], e[1][k], e[2][k]];
    let mut ex = col(0);
    if ex[2] < 0.0 {
        ex = ex.map(|v| -v);
    }
    let mut ey = col(1);
    if ey[1] < 0.0 {
        ey = ey.map(|v| -v);
    }
    let ez = [ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0]];
    let rot = [[ex[0], ey[0], ez[0]], [ex[1], ey[1], ez[1]], [ex[2], ey[2], ez[2]]];
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for p in points {
        let d = (p[0] - c[0]) * ez[0] + (p[1] - c[1]) * ez[1] + (p[2] - c[2]) * ez[2];
        lo = lo.min(d);
        hi = hi.max(d);
    }
    let mid = 0.5 * (lo + hi);
    let center = [c[0] + mid * ez[0], c[1] + mid * ez[1], c[2] + mid * ez[2]];
    let thick = ((hi - lo) + 2.0 * (radius + margin)) / res;
    let thick = (thick / 2.0).ceil() * 2.0 * res;
    Prescription { rot, center, fov: [fov_xy[0], fov_xy[1], thick], res: [res; 3] }
}
