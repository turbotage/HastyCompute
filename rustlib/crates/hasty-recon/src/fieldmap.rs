//! Estimated field maps from a separate scan (the B0 / R2* / fat train, `recon_b0`) carried to another scan's
//! reconstruction voxels by scanner position, so a flow reconstruction can use measured instead of true fields.
//!
//! A `recon_b0` run stores per respiratory bin ω (Hz), R2* (1/s), fat fraction, water and fat magnitude on its own
//! logical grid (`summary.json`: dims, res, sim_dir; geometry from the B0 sim's `meta.json`: rot_presc,
//! fov_center_scanner, gnl). `recon_b0` does not model the gradient nonlinearity, so its voxels sit at apparent
//! positions: a physical point x is found at apparent u = GNL(x) (`apparent = true`). Voxels without a fit (no
//! signal) are filled from their neighbours (onion peeling, then a few smoothing passes over the filled voxels only),
//! since every voxel with signal in the other scan (lungs, arms) needs a value.
//!
//! ```ignore
//! let mut fm = FieldMaps::load_recon_b0(run, None)?;   // None: the mean over respiratory bins
//! fm.fill(20);
//! let (om_hz, r2s, ff) = fm.sample(&x_scanner);
//! ```

use std::path::Path;

use hasty_mrsim::gnl::GradientCoils;

use crate::data::read_npy;
use crate::Error;

type Vec3 = [f64; 3];
type Mat3 = [[f64; 3]; 3];

pub struct FieldMaps {
    pub dims: [usize; 3],
    /// voxel size (m), isotropic
    pub res: f64,
    /// logical → scanner rotation and FOV centre of the scan the maps come from
    pub rot: Mat3,
    pub fov_c: Vec3,
    pub gnl: GradientCoils,
    /// the maps live at apparent (GNL-warped) positions
    pub apparent: bool,
    pub omega_hz: Vec<f32>,
    pub r2s: Vec<f32>,
    pub ff: Vec<f32>,
    /// voxels with a fit (before [`fill`](Self::fill))
    pub known: Vec<bool>,
}

fn jf(v: &serde_json::Value, k: &str) -> Result<f64, Error> {
    v[k].as_f64().ok_or_else(|| Error::Format(format!("missing number {k}")))
}

fn v3(v: &serde_json::Value) -> Result<Vec3, Error> {
    let a = v.as_array().ok_or_else(|| Error::Format("expected a 3-vector".into()))?;
    Ok([0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0)))
}

fn m3(v: &serde_json::Value) -> Result<Mat3, Error> {
    let a = v.as_array().ok_or_else(|| Error::Format("expected a 3 × 3 matrix".into()))?;
    let mut m = [[0.0; 3]; 3];
    for i in 0..3 {
        m[i] = v3(&a[i])?;
    }
    Ok(m)
}

/// the gradient-nonlinearity model of a sim's meta.json (`gnl`: r0, transverse[2], longitudinal[2])
pub fn gnl_from_meta(g: &serde_json::Value) -> Result<GradientCoils, Error> {
    Ok(GradientCoils {
        r0: jf(g, "r0")?,
        transverse: [g["transverse"][0].as_f64().unwrap_or(0.0), g["transverse"][1].as_f64().unwrap_or(0.0)],
        longitudinal: [g["longitudinal"][0].as_f64().unwrap_or(0.0), g["longitudinal"][1].as_f64().unwrap_or(0.0)],
    })
}

/// mean over the bins of a [dims, nb] Fortran array (x fastest, bin slowest), ignoring non-finite values
fn bin_mean(v: &[f32], nv: usize, bin: Option<usize>) -> Vec<f32> {
    let nb = v.len() / nv;
    match bin {
        Some(b) => v[b.min(nb - 1) * nv..(b.min(nb - 1) + 1) * nv].to_vec(),
        None => (0..nv)
            .map(|q| {
                let (s, n) = (0..nb).map(|b| v[b * nv + q]).filter(|x| x.is_finite()).fold((0f64, 0usize), |(s, n), x| (s + x as f64, n + 1));
                if n > 0 { (s / n as f64) as f32 } else { f32::NAN }
            })
            .collect(),
    }
}

impl FieldMaps {
    /// maps of a `recon_b0` run (`bin`: one respiratory bin, None: the mean over bins)
    pub fn load_recon_b0(run: &Path, bin: Option<usize>) -> Result<Self, Error> {
        let sum: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(run.join("summary.json"))?).map_err(|e| Error::Format(format!("summary.json: {e}")))?;
        let dims: [usize; 3] = [0, 1, 2].map(|a| sum["dims"][a].as_u64().unwrap_or(0) as usize);
        let res = jf(&sum, "res")?;
        let sim = Path::new(sum["sim_dir"].as_str().ok_or_else(|| Error::Format("summary.json: sim_dir".into()))?);
        let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(sim.join("meta.json"))?).map_err(|e| Error::Format(format!("meta.json: {e}")))?;
        let nv = dims[0] * dims[1] * dims[2];
        let rd = |n: &str| -> Result<Vec<f32>, Error> { Ok(bin_mean(&read_npy(&run.join(format!("{n}.npy")))?.f32s(), nv, bin)) };
        let (om, r2, ff, wm, fm) = (rd("maps_omega_hz")?, rd("maps_r2")?, rd("maps_ff")?, rd("maps_water")?, rd("maps_fat")?);
        let known: Vec<bool> = (0..nv).map(|q| om[q].is_finite() && (wm[q] + fm[q]) > 0.0).collect();
        Ok(Self { dims, res, rot: m3(&meta["rot_presc"])?, fov_c: v3(&meta["fov_center_scanner"])?, gnl: gnl_from_meta(&meta["gnl"])?, apparent: true, omega_hz: om, r2s: r2, ff, known })
    }

    /// fill the voxels without a fit: onion peeling (each pass gives the unknown voxels next to known or filled
    /// ones the mean of those neighbours) until all are set, then `smooth` Jacobi passes over the filled voxels only
    pub fn fill(&mut self, smooth: usize) {
        let d = self.dims;
        let nv = d[0] * d[1] * d[2];
        let nb = |q: usize| -> Vec<usize> {
            let (i, j, k) = (q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1]));
            let mut v = Vec::with_capacity(6);
            if i > 0 { v.push(q - 1) }
            if i + 1 < d[0] { v.push(q + 1) }
            if j > 0 { v.push(q - d[0]) }
            if j + 1 < d[1] { v.push(q + d[0]) }
            if k > 0 { v.push(q - d[0] * d[1]) }
            if k + 1 < d[2] { v.push(q + d[0] * d[1]) }
            v
        };
        if !self.known.iter().any(|&k| k) {
            return;
        }
        let mut set = self.known.clone();
        let mut maps = [std::mem::take(&mut self.omega_hz), std::mem::take(&mut self.r2s), std::mem::take(&mut self.ff)];
        for m in maps.iter_mut() {
            for q in 0..nv {
                if !set[q] || !m[q].is_finite() {
                    m[q] = 0.0;
                }
            }
        }
        loop {
            let front: Vec<(usize, [f32; 3])> = (0..nv)
                .filter(|&q| !set[q])
                .filter_map(|q| {
                    let ns: Vec<usize> = nb(q).into_iter().filter(|&n| set[n]).collect();
                    (!ns.is_empty()).then(|| (q, [0, 1, 2].map(|c| ns.iter().map(|&n| maps[c][n]).sum::<f32>() / ns.len() as f32)))
                })
                .collect();
            if front.is_empty() {
                break;
            }
            for (q, v) in front {
                set[q] = true;
                for c in 0..3 {
                    maps[c][q] = v[c];
                }
            }
        }
        let filled: Vec<usize> = (0..nv).filter(|&q| !self.known[q]).collect();
        for _ in 0..smooth {
            for m in maps.iter_mut() {
                let upd: Vec<f32> = filled.iter().map(|&q| { let ns = nb(q); ns.iter().map(|&n| m[n]).sum::<f32>() / ns.len().max(1) as f32 }).collect();
                for (&q, v) in filled.iter().zip(upd) {
                    m[q] = v;
                }
            }
        }
        let [a, b, c] = maps;
        (self.omega_hz, self.r2s, self.ff) = (a, b, c);
    }

    /// logical position in the maps' frame of a physical scanner position
    fn logical(&self, x: &Vec3) -> Vec3 {
        let u = if self.apparent { self.gnl.field(x) } else { *x };
        let r = [u[0] - self.fov_c[0], u[1] - self.fov_c[1], u[2] - self.fov_c[2]];
        // logical = Rᵀ (u − c)
        [0, 1, 2].map(|a| (0..3).map(|b| self.rot[b][a] * r[b]).sum::<f64>())
    }

    /// whether the nearest map voxel of a scanner position had a fit (inside the grid)
    pub fn known_at(&self, x: &Vec3) -> bool {
        let xl = self.logical(x);
        let d = self.dims;
        let mut q = 0usize;
        let mut st = 1usize;
        for a in 0..3 {
            let g = (xl[a] / self.res + (d[a] / 2) as f64).round();
            if g < 0.0 || g >= d[a] as f64 {
                return false;
            }
            q += g as usize * st;
            st *= d[a];
        }
        self.known[q]
    }

    /// (ω Hz, R2* 1/s, fat fraction) at a physical scanner position (trilinear, clamped at the grid edge)
    pub fn sample(&self, x: &Vec3) -> (f64, f64, f64) {
        let xl = self.logical(x);
        let d = self.dims;
        let mut i0 = [0usize; 3];
        let mut fr = [0.0; 3];
        for a in 0..3 {
            let g = (xl[a] / self.res + (d[a] / 2) as f64).clamp(0.0, (d[a] - 1) as f64 - 1e-9);
            i0[a] = g.floor() as usize;
            fr[a] = g - i0[a] as f64;
        }
        let mut acc = [0.0; 3];
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let w = (if dx == 1 { fr[0] } else { 1.0 - fr[0] }) * (if dy == 1 { fr[1] } else { 1.0 - fr[1] }) * (if dz == 1 { fr[2] } else { 1.0 - fr[2] });
                    if w == 0.0 {
                        continue;
                    }
                    let q = (i0[0] + dx).min(d[0] - 1) + d[0] * ((i0[1] + dy).min(d[1] - 1) + d[1] * (i0[2] + dz).min(d[2] - 1));
                    for (c, m) in [&self.omega_hz, &self.r2s, &self.ff].iter().enumerate() {
                        let v = m[q];
                        acc[c] += w * if v.is_finite() { v as f64 } else { 0.0 };
                    }
                }
            }
        }
        (acc[0], acc[1], acc[2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maps(dims: [usize; 3], res: f64, rot: Mat3, c: Vec3, f: &dyn Fn(Vec3) -> f64) -> FieldMaps {
        let nv = dims[0] * dims[1] * dims[2];
        let mut om = vec![0f32; nv];
        for q in 0..nv {
            let p = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
            let xl = [0, 1, 2].map(|a| (p[a] as f64 - (dims[a] / 2) as f64) * res);
            let xs = [0, 1, 2].map(|a| (0..3).map(|b| rot[a][b] * xl[b]).sum::<f64>() + c[a]);
            om[q] = f(xs) as f32;
        }
        FieldMaps { dims, res, rot, fov_c: c, gnl: GradientCoils { r0: 0.25, transverse: [0.0, 0.0], longitudinal: [0.0, 0.0] }, apparent: false, omega_hz: om.clone(), r2s: vec![20.0; nv], ff: vec![0.0; nv], known: vec![true; nv] }
    }

    #[test]
    fn samples_a_linear_field_through_a_rotated_prescription() {
        let (s, co) = (0.3f64.sin(), 0.3f64.cos());
        let rot = [[co, -s, 0.0], [s, co, 0.0], [0.0, 0.0, 1.0]];
        let f = |x: Vec3| 100.0 * x[0] - 40.0 * x[1] + 250.0 * x[2] + 3.0;
        let fm = maps([30, 24, 20], 0.004, rot, [0.01, -0.02, 0.03], &f);
        for x in [[0.01, -0.02, 0.03], [0.02, 0.0, 0.05], [-0.015, -0.03, 0.02]] {
            let (om, r2, _) = fm.sample(&x);
            assert!((om - f(x)).abs() < 1e-3, "ω {om} vs {}", f(x));
            assert!((r2 - 20.0).abs() < 1e-9);
        }
    }

    #[test]
    fn fill_reaches_every_voxel_and_keeps_the_known_ones() {
        let rot = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut fm = maps([20, 16, 12], 0.005, rot, [0.0; 3], &|x| 50.0 + 200.0 * x[0]);
        let nv = fm.omega_hz.len();
        let orig = fm.omega_hz.clone();
        // a hole (no fit) in the middle and a missing face
        for q in 0..nv {
            let p = [q % 20, (q / 20) % 16, q / 320];
            let hole = (6..14).contains(&p[0]) && (5..11).contains(&p[1]) && (3..9).contains(&p[2]);
            if hole || p[2] == 11 {
                fm.known[q] = false;
                fm.omega_hz[q] = f32::NAN;
            }
        }
        let known = fm.known.clone();
        fm.fill(30);
        assert!(fm.omega_hz.iter().all(|v| v.is_finite()));
        for q in 0..nv {
            if known[q] {
                assert_eq!(fm.omega_hz[q], orig[q]);
            }
        }
        // a linear field is (nearly) harmonic: the hole is filled close to it
        let err = (0..nv).filter(|&q| !known[q]).map(|q| (fm.omega_hz[q] - orig[q]).abs()).fold(0f32, f32::max);
        assert!(err < 2.0, "max fill error {err} Hz");
    }
}
