//! The four concomitant-field spatial functions in FOV-normalised coordinates and their algebra.
//!
//! `m_1 = z^2`, `m_2 = x^2 + y^2`, `m_3 = x z`, `m_4 = y z` with `x,y,z in [-1,1]` the FOV-normalised voxel coordinates
//! (`xi_a = (idx_a - (N_a - 1)/2) * 2/N_a`, the same convention as the polynomial diagonals in `hasty-fft`).
//! A sample's phase is `Phi_j(xi) = sum_q w_{j,q} m_q(xi)` (radians; the physical constants `gamma/(2 B0)` and the gradient products live in `w`).

use num_complex::Complex64;

use crate::poly::Poly;

pub const NQ: usize = 4;

/// variables of the 3-D position algebra: xi_x, xi_y, xi_z
pub fn m_poly(q: usize) -> Poly {
    let one = Complex64::new(1.0, 0.0);
    let (x, y, z) = (Poly::var(3, 0), Poly::var(3, 1), Poly::var(3, 2));
    match q {
        0 => z.mul(&z),
        1 => {
            let mut p = x.mul(&x);
            p.add_assign(&y.mul(&y));
            p
        }
        2 => x.mul(&z),
        3 => y.mul(&z),
        _ => Poly::constant(3, one),
    }
}

/// `m_q` evaluated at a normalised point
pub fn m_eval(q: usize, x: f64, y: f64, z: f64) -> f64 {
    match q {
        0 => z * z,
        1 => x * x + y * y,
        2 => x * z,
        _ => y * z,
    }
}

/// FOV-normalised coordinate of voxel index `i` of an axis with `n` voxels
pub fn xi(i: usize, n: usize) -> f64 {
    (i as f64 - (n as f64 - 1.0) / 2.0) * 2.0 / n as f64
}
