//! MR acquisition simulation.
//!
//! * [`waveform`]: gradient waveforms on a uniform raster, trapezoids/bipolars under hardware limits, gradient moments
//!   (M0, M1, M2) and the concomitant-field integral P(t) = ∫ G Gᵀ dt.
//! * [`trajectory`]: time-optimal gradient design for an arbitrary k-space curve (Lustig et al. 2008), spiral curves.
//! * [`sequence`]: spiral 4D-flow GRE sequence → per-readout records and per-sample phase parameters.
//! * [`gnl`]: gradient-coil field models (spherical-harmonic gradient nonlinearity).
//! * [`physio`]: ECG model (RR variability) and cardiac binning.
//! * [`direct`]: GPU direct-sum truth simulator, s_c(sample) = Σ_p a_pc exp(−R2*_p τ − iφ_p(sample)), no Fourier approximations.
//!
//! Units are SI throughout: m, s, T, T/m, T/m/s, rad. Physical frame: scanner coordinates with B0 along +z, origin at the
//! gradient isocentre.

pub mod direct;
pub mod gnl;
pub mod physio;
pub mod sequence;
pub mod trajectory;
pub mod waveform;

/// ¹H gyromagnetic ratio, rad/s/T.
pub const GAMMA: f64 = 2.675_221_874_4e8;
/// ¹H gyromagnetic ratio / 2π, Hz/T.
pub const GAMMA_BAR: f64 = GAMMA / (2.0 * std::f64::consts::PI);

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Cuda(#[from] hasty_cuda::Error),
    #[error("CUDA driver error: {0:?}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("invalid argument: {0}")]
    Invalid(String),
}

/// Gradient hardware limits (Euclidean norm, so they hold for any rotation of the waveform).
#[derive(Clone, Copy, Debug)]
pub struct GradLimits {
    /// T/m
    pub gmax: f64,
    /// T/m/s
    pub smax: f64,
}

impl GradLimits {
    /// 3 T clinical system without high-performance gradients: 80 mT/m, 200 T/m/s.
    pub fn clinical_3t() -> Self {
        Self { gmax: 80e-3, smax: 200.0 }
    }
    /// Scaled copy (design margin, e.g. 0.9 for rotated readouts / safety).
    pub fn derated(self, f: f64) -> Self {
        Self { gmax: self.gmax * f, smax: self.smax * f }
    }
}

pub type Mat3 = [[f64; 3]; 3];
pub type Vec3 = [f64; 3];

pub fn mat_vec(m: &Mat3, v: &Vec3) -> Vec3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub fn mat_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut c = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            c[i][j] = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    c
}

pub fn transpose(a: &Mat3) -> Mat3 {
    let mut t = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            t[i][j] = a[j][i];
        }
    }
    t
}

pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Rotation about a unit axis by `angle` (Rodrigues).
pub fn rotation(axis: Vec3, angle: f64) -> Mat3 {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let (x, y, z) = (axis[0] / n, axis[1] / n, axis[2] / n);
    let (s, c) = angle.sin_cos();
    let t = 1.0 - c;
    [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]
}
