//! Prescribed cardiac motion on the host: the phantom's `cardiac_motion.hv` (U_mm [T, 3, Z, Y, X] float16, voxel
//! centres at origin + (i + ½) h, periodic in time with `period`), interpolated exactly as `phantom_4d.py` does
//! (periodic Catmull-Rom in time, trilinear in space with zeros outside). The kernels evaluate the same field on the
//! device (`node_motion` + `tri_zero`); the host needs it for valve / cut kinematics and the active-tile search.

use std::path::Path;

use crate::params::MotionParams;
use crate::{Error, Result};

#[derive(Clone, Debug)]
pub struct Motion {
    pub nt: usize,
    /// x, y, z
    pub dims: [usize; 3],
    pub origin: [f64; 3],
    pub h: f64,
    pub period: f64,
    /// float16 bits [T][3][Z][Y][X], mm
    pub u: Vec<u16>,
}

pub fn f16_to_f32(h: u16) -> f32 {
    let s = (h >> 15) & 1;
    let e = ((h >> 10) & 31) as i32;
    let m = (h & 1023) as f32;
    let v = if e == 0 {
        m * 2f32.powi(-24)
    } else if e == 31 {
        65504.0
    } else {
        (m + 1024.0) * 2f32.powi(e - 25)
    };
    if s == 1 {
        -v
    } else {
        v
    }
}

pub fn f32_to_f16(v: f32) -> u16 {
    let s: u16 = if v.is_sign_negative() { 0x8000 } else { 0 };
    let a = v.abs();
    if a < 6.0e-8 {
        return s;
    }
    if a >= 65504.0 {
        return s | 0x7bff;
    }
    let e = a.log2().floor() as i32;
    if e < -14 {
        // subnormal: m · 2^−24
        return s | ((a / 2f32.powi(-24)).round() as u16).min(1023);
    }
    let mut m = (a / 2f32.powi(e) - 1.0) * 1024.0;
    let mut e = e;
    m = m.round();
    if m >= 1024.0 {
        m = 0.0;
        e += 1;
    }
    s | (((e + 15) as u16) << 10) | (m as u16)
}

fn floats(v: &hasty_value::Value, key: &str) -> Result<Vec<f64>> {
    use hasty_value::Value;
    match v.get(key) {
        Some(Value::List(l)) => l
            .iter()
            .map(|x| match x {
                Value::Float(f) => Ok(*f),
                Value::Int(i) => Ok(*i as f64),
                _ => Err(Error::Value(format!("{key}: not a number"))),
            })
            .collect(),
        Some(Value::Tensor(t)) => Ok((0..t.numel()).map(|i| t.get_f64(i).0).collect()),
        Some(Value::Float(f)) => Ok(vec![*f]),
        Some(Value::Int(i)) => Ok(vec![*i as f64]),
        _ => Err(Error::Value(format!("missing {key}"))),
    }
}

impl Motion {
    pub fn load(path: &Path) -> Result<Self> {
        let v = hasty_value::read(&mut std::io::BufReader::new(std::fs::File::open(path)?)).map_err(|e| Error::Value(format!("{}: {e}", path.display())))?;
        let t = v.get("U_mm").and_then(|x| x.as_tensor()).ok_or_else(|| Error::Value("U_mm".into()))?;
        let sh = t.shape().to_vec();
        if sh.len() != 5 || sh[1] != 3 {
            return Err(Error::Value(format!("U_mm shape {sh:?}, want [T, 3, Z, Y, X]")));
        }
        let u: Vec<u16> = match t.dtype() {
            hasty_value::DType::F16 => t.bytes().chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect(),
            _ => (0..t.numel()).map(|i| f32_to_f16(t.get_f64(i).0 as f32)).collect(),
        };
        let o = floats(&v, "origin_m")?;
        Ok(Self { nt: sh[0], dims: [sh[4], sh[3], sh[2]], origin: [o[0], o[1], o[2]], h: floats(&v, "voxel_m")?[0], period: floats(&v, "period")?[0], u })
    }

    /// a motion sampled from a displacement function d(X, t) (m) on a grid (tests, analytic phantoms)
    pub fn from_fn(dims: [usize; 3], origin: [f64; 3], h: f64, nt: usize, period: f64, d: impl Fn([f64; 3], f64) -> [f64; 3]) -> Self {
        let m = dims[0] * dims[1] * dims[2];
        let mut u = vec![0u16; nt * 3 * m];
        for k in 0..nt {
            let t = k as f64 * period / nt as f64;
            for z in 0..dims[2] {
                for y in 0..dims[1] {
                    for x in 0..dims[0] {
                        let p = [origin[0] + (x as f64 + 0.5) * h, origin[1] + (y as f64 + 0.5) * h, origin[2] + (z as f64 + 0.5) * h];
                        let dd = d(p, t);
                        let q = (z * dims[1] + y) * dims[0] + x;
                        for c in 0..3 {
                            u[(k * 3 + c) * m + q] = f32_to_f16((dd[c] * 1e3) as f32);
                        }
                    }
                }
            }
        }
        Self { nt, dims, origin, h, period, u }
    }

    /// no motion (rigid case geometry)
    pub fn still(origin: [f64; 3], h: f64, period: f64) -> Self {
        Self::from_fn([2, 2, 2], origin, h, 4, period, |_, _| [0.0; 3])
    }

    pub fn voxels(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// periodic Catmull-Rom: sample indices, weights for U and for ∂U/∂t (1/s); identical to phantom_4d.cr_weights
    pub fn weights(&self, t: f64) -> ([usize; 4], [f64; 4], [f64; 4]) {
        let nt = self.nt as f64;
        let s = t.rem_euclid(self.period) / self.period * nt;
        let i = s.floor() as i64;
        let f = s - i as f64;
        let w = [(-f * f * f + 2.0 * f * f - f) / 2.0, (3.0 * f * f * f - 5.0 * f * f + 2.0) / 2.0, (-3.0 * f * f * f + 4.0 * f * f + f) / 2.0, (f * f * f - f * f) / 2.0];
        let d = [(-3.0 * f * f + 4.0 * f - 1.0) / 2.0, (9.0 * f * f - 10.0 * f) / 2.0, (-9.0 * f * f + 8.0 * f + 1.0) / 2.0, (3.0 * f * f - 2.0 * f) / 2.0].map(|x| x * nt / self.period);
        let idx = [0, 1, 2, 3].map(|j| (i - 1 + j as i64).rem_euclid(self.nt as i64) as usize);
        (idx, w, d)
    }

    pub fn params(&self, t: f64) -> MotionParams {
        let (idx, w, d) = self.weights(t);
        MotionParams { m: self.voxels() as i32, idx: idx.map(|x| x as i32), w: w.map(|x| x as f32), d: d.map(|x| x as f32) }
    }

    /// trilinear (zeros outside) of time sample k, component c, in mm
    fn sample(&self, k: usize, c: usize, x: &[f64; 3]) -> f64 {
        let [nx, ny, nz] = self.dims;
        let g = [0, 1, 2].map(|a| (x[a] - self.origin[a]) / self.h - 0.5);
        let i0 = g.map(|v| v.floor() as i64);
        let f = [g[0] - i0[0] as f64, g[1] - i0[1] as f64, g[2] - i0[2] as f64];
        let m = self.voxels();
        let mut acc = 0.0;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let (xi, yi, zi) = (i0[0] + dx, i0[1] + dy, i0[2] + dz);
                    if xi < 0 || yi < 0 || zi < 0 || xi >= nx as i64 || yi >= ny as i64 || zi >= nz as i64 {
                        continue;
                    }
                    let w = (if dx == 1 { f[0] } else { 1.0 - f[0] }) * (if dy == 1 { f[1] } else { 1.0 - f[1] }) * (if dz == 1 { f[2] } else { 1.0 - f[2] });
                    acc += w * f16_to_f32(self.u[(k * 3 + c) * m + (zi as usize * ny + yi as usize) * nx + xi as usize]) as f64;
                }
            }
        }
        acc
    }

    /// displacement U(X, t), m
    pub fn displacement(&self, x: &[f64; 3], t: f64) -> [f64; 3] {
        let (idx, w, _) = self.weights(t);
        [0, 1, 2].map(|c| (0..4).map(|j| w[j] * self.sample(idx[j], c, x)).sum::<f64>() * 1e-3)
    }

    /// material velocity ∂U/∂t at the reference point X, m/s
    pub fn velocity(&self, x: &[f64; 3], t: f64) -> [f64; 3] {
        let (idx, _, d) = self.weights(t);
        [0, 1, 2].map(|c| (0..4).map(|j| d[j] * self.sample(idx[j], c, x)).sum::<f64>() * 1e-3)
    }

    /// Φ_t(X) = X + U(X, t)
    pub fn forward(&self, x: &[f64; 3], t: f64) -> [f64; 3] {
        let u = self.displacement(x, t);
        [x[0] + u[0], x[1] + u[1], x[2] + u[2]]
    }

    /// Φ_t⁻¹(x): Newton on X + U(X, t) = x with a central-difference Jacobian (the fixed point X = x − U(X) diverges
    /// where ‖∇U‖ > 1), steps capped at one motion voxel; `iters` Newton steps
    pub fn inverse(&self, x: &[f64; 3], t: f64, iters: usize) -> [f64; 3] {
        let mut p = *x;
        let e = 0.25 * self.h;
        for _ in 0..iters {
            let u = self.displacement(&p, t);
            let r = [p[0] + u[0] - x[0], p[1] + u[1] - x[1], p[2] + u[2] - x[2]];
            let mut a = [[0.0; 3]; 3];
            for j in 0..3 {
                let mut pp = p;
                let mut pm = p;
                pp[j] += e;
                pm[j] -= e;
                let (up, um) = (self.displacement(&pp, t), self.displacement(&pm, t));
                for i in 0..3 {
                    a[i][j] = (i == j) as i32 as f64 + (up[i] - um[i]) / (2.0 * e);
                }
            }
            let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
            let mut d = r;
            if det > 0.02 {
                let inv = |i: usize, j: usize| {
                    let (i1, i2, j1, j2) = ((j + 1) % 3, (j + 2) % 3, (i + 1) % 3, (i + 2) % 3);
                    (a[i1][j1] * a[i2][j2] - a[i1][j2] * a[i2][j1]) / det
                };
                d = [0, 1, 2].map(|i| (0..3).map(|j| inv(i, j) * r[j]).sum());
            }
            let dn = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            let s = if dn > self.h { self.h / dn } else { 1.0 };
            p = [p[0] - s * d[0], p[1] - s * d[1], p[2] - s * d[2]];
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_round_trip_and_interpolation() {
        for v in [0.0f32, 1.0, -2.5, 3.14159, 1e-3, 12.75, -0.0001] {
            let r = f16_to_f32(f32_to_f16(v));
            assert!((r - v).abs() <= 1e-3 * v.abs().max(1e-4), "{v} → {r}");
        }
        // a translation sinusoid in time: displacement, velocity and the inverse
        let m = Motion::from_fn([8, 8, 8], [0.0; 3], 1e-3, 48, 1.0, |_, t| [2e-3 * (2.0 * std::f64::consts::PI * t).sin(), 0.0, 0.0]);
        let x = [4e-3, 4e-3, 4e-3];
        for t in [0.1, 0.33, 0.7] {
            let u = m.displacement(&x, t);
            let v = m.velocity(&x, t);
            let w = 2.0 * std::f64::consts::PI;
            assert!((u[0] - 2e-3 * (w * t).sin()).abs() < 2e-5, "{u:?}");
            assert!((v[0] - 2e-3 * w * (w * t).cos()).abs() < 3e-4, "{v:?}");
            let y = m.forward(&x, t);
            let xi = m.inverse(&y, t, 8);
            assert!((xi[0] - x[0]).abs() < 1e-7);
        }
    }
}
