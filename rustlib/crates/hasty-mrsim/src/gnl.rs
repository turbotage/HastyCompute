//! Gradient-coil field models.
//!
//! Gradient coil `a` produces B_z(x) = G_a · f_a(x). For ideal coils f_a(x) = x_a. Real coils add higher-order solid
//! harmonics (B_z is harmonic: ∇² f_a = 0). The encoding phase of a spin at x is γ Σ_a M0_a f_a(x), i.e. the spin is
//! encoded at the *apparent* position u(x) = f(x); a moving spin sees M1 · (J v) with J = ∂f/∂x (velocity-encoding error).

use crate::{Mat3, Vec3};

/// Spherical-harmonic gradient nonlinearity (odd orders 3 and 5, the dominant terms of symmetric cylindrical coils),
/// coefficients relative to the linear term at reference radius `r0`.
#[derive(Clone, Copy, Debug)]
pub struct GradientCoils {
    pub r0: f64,
    /// x and y coils: [c3, c5]
    pub transverse: [f64; 2],
    /// z coil: [c3, c5]
    pub longitudinal: [f64; 2],
}

impl GradientCoils {
    pub fn ideal() -> Self {
        Self { r0: 0.25, transverse: [0.0; 2], longitudinal: [0.0; 2] }
    }
    /// Synthetic but realistic clinical 3 T whole-body coil: a few % deviation at 20 cm, ~10% at 25 cm; transverse coils
    /// lose strength towards the coil ends (along z), the z coil loses strength along z.
    pub fn clinical_synthetic() -> Self {
        Self { r0: 0.25, transverse: [-0.030, -0.004], longitudinal: [-0.045, -0.004] }
    }

    /// apparent position u = f(x)
    pub fn field(&self, x: &Vec3) -> Vec3 {
        let r2 = self.r0 * self.r0;
        let r4 = r2 * r2;
        let (px, py, pz) = (x[0], x[1], x[2]);
        let rho2 = px * px + py * py;
        let z2 = pz * pz;
        let [t3, t5] = self.transverse;
        let [l3, l5] = self.longitudinal;
        // x coil: x (4z² − ρ²) and x (8z⁴ − 12 z² ρ² + ρ⁴); y analogous; z coil: z (2z² − 3ρ²) and z (8z⁴ − 40 z² ρ² + 15 ρ⁴)
        let tr3 = 4.0 * z2 - rho2;
        let tr5 = 8.0 * z2 * z2 - 12.0 * z2 * rho2 + rho2 * rho2;
        let fx = px * (1.0 + t3 * tr3 / r2 + t5 * tr5 / r4);
        let fy = py * (1.0 + t3 * tr3 / r2 + t5 * tr5 / r4);
        let fz = pz * (1.0 + l3 * (2.0 * z2 - 3.0 * rho2) / r2 + l5 * (8.0 * z2 * z2 - 40.0 * z2 * rho2 + 15.0 * rho2 * rho2) / r4);
        [fx, fy, fz]
    }

    /// Jacobian J_ab = ∂f_a/∂x_b (central differences; f is a degree-5 polynomial so h = 1e-4 m is accurate to ~1e-8)
    pub fn jacobian(&self, x: &Vec3) -> Mat3 {
        let h = 1e-4;
        let mut j = [[0.0; 3]; 3];
        for b in 0..3 {
            let mut xp = *x;
            let mut xm = *x;
            xp[b] += h;
            xm[b] -= h;
            let (fp, fm) = (self.field(&xp), self.field(&xm));
            for a in 0..3 {
                j[a][b] = (fp[a] - fm[a]) / (2.0 * h);
            }
        }
        j
    }
}
