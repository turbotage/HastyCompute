//! Simulator output (`sim4d`) → reconstruction inputs: sequence (regenerated deterministically from the parameters),
//! whitened multi-coil data per encode/frame, per-sample phase parameters, and physics on the reconstruction grid.

use std::path::{Path, PathBuf};

use hasty_cuda::C32;
use hasty_mrsim::gnl::GradientCoils;
use hasty_mrsim::sequence::{Acquisition, Coverage, Encoding, FlowSpiralParams, SAMPLE_FLOATS};
use hasty_mrsim::{mat_vec, transpose, GradLimits, Mat3, Vec3, GAMMA_BAR};
use nalgebra::{DMatrix, SymmetricEigen};
use serde_json::Value;

use crate::Error;

pub struct NpyArray {
    pub shape: Vec<usize>,
    pub fortran: bool,
    pub descr: String,
    pub bytes: Vec<u8>,
}

pub fn read_npy(path: &Path) -> Result<NpyArray, Error> {
    let buf = std::fs::read(path)?;
    if &buf[..6] != b"\x93NUMPY" {
        return Err(Error::Format(format!("{}: not npy", path.display())));
    }
    let (hlen, hs) = if buf[6] == 1 { (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10) } else { (u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize, 12) };
    let h = std::str::from_utf8(&buf[hs..hs + hlen]).map_err(|e| Error::Format(e.to_string()))?;
    let descr = h.split("'descr':").nth(1).and_then(|s| s.split('\'').nth(1)).unwrap_or("").to_string();
    let fortran = h.contains("'fortran_order': True");
    let shp = h.split("'shape':").nth(1).and_then(|s| s.split('(').nth(1)).and_then(|s| s.split(')').next()).unwrap_or("");
    let shape = shp.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    Ok(NpyArray { shape, fortran, descr, bytes: buf[hs + hlen..].to_vec() })
}

impl NpyArray {
    pub fn f32s(&self) -> Vec<f32> {
        assert_eq!(self.descr, "<f4");
        self.bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }
    pub fn f64s(&self) -> Vec<f64> {
        assert_eq!(self.descr, "<f8");
        self.bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect()
    }
    pub fn c32s(&self) -> Vec<C32> {
        assert_eq!(self.descr, "<c8");
        self.bytes.chunks_exact(8).map(|b| C32::new(f32::from_le_bytes([b[0], b[1], b[2], b[3]]), f32::from_le_bytes([b[4], b[5], b[6], b[7]]))).collect()
    }
}

fn v3(v: &Value) -> Vec3 {
    [0, 1, 2].map(|i| v[i].as_f64().unwrap())
}
fn m3(v: &Value) -> Mat3 {
    [0, 1, 2].map(|r| [0, 1, 2].map(|c| v[r][c].as_f64().unwrap()))
}

pub struct SimData {
    pub dir: PathBuf,
    pub meta: Value,
    pub acq: Acquisition,
    /// [readout][adc][coil]
    pub data: Vec<C32>,
    pub nc: usize,
    pub na: usize,
    pub dims: [usize; 3],
    pub res: f64,
    pub psi: Vec<Vec<f64>>,
    pub gnl: GradientCoils,
    pub rot: Mat3,
    /// FOV centre in the scanner frame
    pub fov_c: Vec3,
    pub b0: f64,
    pub venc: f64,
    /// coil-major true sensitivities on the true logical grid (reference/oracle only)
    pub smaps_true: Vec<C32>,
    /// off-resonance (ppm) on the true logical grid
    pub field_ppm: Vec<f32>,
    /// truth velocity, logical components: index q + nv (a + 3 f)
    pub truth_vel: Vec<f32>,
    pub truth_frac: Vec<f32>,
    pub n_frames: usize,
}

/// Per-encode (or per-frame) sample set ready for an operator.
pub struct SampleSet {
    pub readouts: Vec<usize>,
    /// k·Δ, radians per voxel
    pub coords: [Vec<f32>; 3],
    /// whitened data, coil-major (nc × m)
    pub y: Vec<C32>,
    /// time since excitation, s
    pub tau: Vec<f64>,
    /// concomitant coefficients of [z², x²+y², xz, yz] (scanner frame), rad/m²
    pub wq: Vec<[f64; 4]>,
}

/// Physics evaluated at the *true* position of every reconstruction voxel (the image lives in apparent, GNL-warped
/// coordinates when `gnl_aware`; otherwise apparent = true).
pub struct PhysicsGrid {
    /// true logical position (m, relative to the FOV centre)
    pub x_log: Vec<Vec3>,
    pub mono: Vec<[f64; 4]>,
    /// rad/s
    pub omega: Vec<f64>,
    /// GNL Jacobian (scanner frame)
    pub jac: Vec<Mat3>,
    /// whitened sensitivities at the true positions (coil-major)
    pub smaps: Vec<C32>,
}

impl SimData {
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?).map_err(|e| Error::Format(e.to_string()))?;
        let f = |k: &str| meta[k].as_f64().unwrap();
        let u = |k: &str| meta[k].as_u64().unwrap() as usize;
        let p = FlowSpiralParams {
            b0: f("b0"),
            lim: GradLimits { gmax: f("gmax"), smax: f("smax") },
            dt: f("dt"),
            dwell: f("dwell"),
            fov: v3(&meta["fov"]),
            res: v3(&meta["res"]),
            n_int: u("n_int"),
            fov_edge_ratio: f("fov_edge_ratio"),
            venc: f("venc"),
            encoding: if meta["encoding"] == "Hadamard" { Encoding::Hadamard } else { Encoding::Referenced },
            coverage: if meta["coverage"] == "Rotated3d" { Coverage::Rotated3d } else { Coverage::StackOfSpirals },
            t_rf: f("t_rf"),
            t_spoil: f("t_spoil"),
            scan_time: f("scan_time"),
            rot_presc: m3(&meta["rot_presc"]),
            fov_center: v3(&meta["fov_center_scanner"]),
            mean_rr: f("mean_rr"),
            rr_rel_std: f("rr_rel_std"),
            n_frames: u("n_frames"),
            seed: meta["seed"].as_u64().unwrap(),
        };
        let acq = Acquisition::new(p.clone());
        let nc = u("n_coils");
        let na = u("n_adc");
        assert_eq!(acq.readouts.len(), u("n_readouts"), "regenerated schedule differs from the simulated one");
        assert_eq!(acq.n_adc(), na);
        let dims = [0, 1, 2].map(|i| meta["recon_dims"][i].as_u64().unwrap() as usize);
        let psi: Vec<Vec<f64>> = meta["psi"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()).collect();
        let g = &meta["gnl"];
        let gnl = GradientCoils { r0: g["r0"].as_f64().unwrap(), transverse: [g["transverse"][0].as_f64().unwrap(), g["transverse"][1].as_f64().unwrap()], longitudinal: [g["longitudinal"][0].as_f64().unwrap(), g["longitudinal"][1].as_f64().unwrap()] };
        // HASTY_NOISE_SIGMA: use the noise-free data and add Ψ-correlated noise with this σ per coil (complex, before
        // whitening; σ/√2 per component); 0 = noise-free
        // without data.npy (default since sims store only the clean data): noise at the simulated level
        let env_sigma = std::env::var("HASTY_NOISE_SIGMA").ok().and_then(|v| v.parse::<f64>().ok());
        let sigma = env_sigma.or_else(|| (!dir.join("data.npy").exists()).then(|| meta["noise_sigma"].as_f64().unwrap()));
        let data = match sigma {
            Some(sigma) if dir.join("data_clean.npy").exists() => {
                let mut d = read_npy(&dir.join("data_clean.npy"))?.c32s();
                let n = psi.len();
                let mut l = vec![vec![0.0f64; n]; n];
                for i in 0..n {
                    for j in 0..=i {
                        let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
                        l[i][j] = if i == j { (psi[i][i] - s).max(1e-12).sqrt() } else { (psi[i][j] - s) / l[j][j] };
                    }
                }
                let mut seed = 0x9E37_79B9_7F4A_7C15u64;
                let mut gauss = || {
                    let mut u = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64 };
                    (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos()
                };
                let mut white = vec![(0.0f64, 0.0f64); n];
                for s in d.chunks_mut(n) {
                    for w in white.iter_mut() {
                        *w = (gauss() * sigma / 2f64.sqrt(), gauss() * sigma / 2f64.sqrt());
                    }
                    for c in 0..n {
                        let (mut re, mut im) = (0.0, 0.0);
                        for k in 0..=c {
                            re += l[c][k] * white[k].0;
                            im += l[c][k] * white[k].1;
                        }
                        s[c].re += re as f32;
                        s[c].im += im as f32;
                    }
                }
                d
            }
            _ => read_npy(&dir.join("data.npy"))?.c32s(),
        };
        let smaps_true = read_npy(&dir.join("smaps_true.npy"))?.c32s();
        let field_ppm = read_npy(&dir.join("field_ppm_recon.npy"))?.f32s();
        let truth_vel = read_npy(&dir.join("truth_vel_logical.npy"))?.f32s();
        let truth_frac = read_npy(&dir.join("truth_lumen_fraction.npy"))?.f32s();
        Ok(Self {
            dir: dir.to_path_buf(),
            acq,
            data,
            nc,
            na,
            dims,
            res: p.res[0],
            psi,
            gnl,
            rot: p.rot_presc,
            fov_c: p.fov_center,
            b0: p.b0,
            venc: p.venc,
            smaps_true,
            field_ppm,
            truth_vel,
            truth_frac,
            n_frames: p.n_frames,
            meta,
        })
    }

    pub fn nv(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Ψ^{-1/2}
    pub fn whitener(&self) -> Vec<Vec<f64>> {
        let n = self.nc;
        let m = DMatrix::<f64>::from_fn(n, n, |i, j| self.psi[i][j]);
        let e = SymmetricEigen::new(m);
        let mut w = vec![vec![0.0; n]; n];
        for q in 0..n {
            let s = 1.0 / e.eigenvalues[q].max(1e-12).sqrt();
            for i in 0..n {
                for j in 0..n {
                    w[i][j] += e.eigenvectors[(i, q)] * e.eigenvectors[(j, q)] * s;
                }
            }
        }
        w
    }

    pub fn readouts_of(&self, enc: usize, frame: Option<usize>) -> Vec<usize> {
        (0..self.acq.readouts.len()).filter(|&i| self.acq.readouts[i].enc == enc && frame.map_or(true, |f| self.acq.readouts[i].frame == f)).collect()
    }

    pub fn sample_set(&self, ros: &[usize], w: &[Vec<f64>]) -> SampleSet {
        let (na, nc) = (self.na, self.nc);
        let m = ros.len() * na;
        let mut coords = [Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m)];
        let mut y = vec![C32::new(0.0, 0.0); nc * m];
        let mut tau = Vec::with_capacity(m);
        let mut wq = Vec::with_capacity(m);
        for (r, &ro) in ros.iter().enumerate() {
            let k = self.acq.k_logical(ro);
            let sp = self.acq.sample_params(ro..ro + 1);
            for n in 0..na {
                let j = r * na + n;
                for a in 0..3 {
                    coords[a].push((k[n][a] * self.res) as f32);
                }
                let s = &sp[n * SAMPLE_FLOATS..(n + 1) * SAMPLE_FLOATS];
                wq.push([s[9] as f64, s[10] as f64, s[11] as f64, s[12] as f64]);
                tau.push(s[13] as f64);
                let src = &self.data[(ro * na + n) * nc..(ro * na + n + 1) * nc];
                for c in 0..nc {
                    let (mut re, mut im) = (0.0f64, 0.0f64);
                    for c2 in 0..nc {
                        re += w[c][c2] * src[c2].re as f64;
                        im += w[c][c2] * src[c2].im as f64;
                    }
                    y[c * m + j] = C32::new(re as f32, im as f32);
                }
            }
        }
        SampleSet { readouts: ros.to_vec(), coords, y, tau, wq }
    }

    /// logical voxel centre of linear index q
    pub fn voxel_log(&self, q: usize) -> Vec3 {
        let d = self.dims;
        let (i, j, k) = (q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1]));
        [(i as f64 - (d[0] / 2) as f64) * self.res, (j as f64 - (d[1] / 2) as f64) * self.res, (k as f64 - (d[2] / 2) as f64) * self.res]
    }

    /// trilinear sample of an f32 field on the logical grid (components: `comp` stride nv)
    pub fn sample_log(&self, field: &[f32], comp: usize, x_log: &Vec3) -> f64 {
        let d = self.dims;
        let nv = self.nv();
        let mut i0 = [0usize; 3];
        let mut fr = [0.0; 3];
        for a in 0..3 {
            let g = (x_log[a] / self.res + (d[a] / 2) as f64).clamp(0.0, (d[a] - 1) as f64 - 1e-9);
            i0[a] = g.floor() as usize;
            fr[a] = g - i0[a] as f64;
        }
        let mut acc = 0.0;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let w = (if dx == 1 { fr[0] } else { 1.0 - fr[0] }) * (if dy == 1 { fr[1] } else { 1.0 - fr[1] }) * (if dz == 1 { fr[2] } else { 1.0 - fr[2] });
                    if w == 0.0 {
                        continue;
                    }
                    let (a, b, c) = ((i0[0] + dx).min(d[0] - 1), (i0[1] + dy).min(d[1] - 1), (i0[2] + dz).min(d[2] - 1));
                    acc += w * field[comp * nv + a + d[0] * (b + d[1] * c)] as f64;
                }
            }
        }
        acc
    }

    pub fn physics_grid(&self, gnl_aware: bool, w: &[Vec<f64>]) -> PhysicsGrid {
        let nv = self.nv();
        let nc = self.nc;
        let rt = transpose(&self.rot);
        let (mut x_log, mut mono, mut omega, mut jac) = (Vec::with_capacity(nv), Vec::with_capacity(nv), Vec::with_capacity(nv), Vec::with_capacity(nv));
        let mut smaps = vec![C32::new(0.0, 0.0); nc * nv];
        let wfac = 2.0 * std::f64::consts::PI * GAMMA_BAR * self.b0 * 1e-6;
        let (re_t, im_t): (Vec<f32>, Vec<f32>) = self.smaps_true.iter().map(|c| (c.re, c.im)).unzip();
        for q in 0..nv {
            let u_log = self.voxel_log(q);
            let u_s = mat_vec(&self.rot, &u_log);
            let u_s = [u_s[0] + self.fov_c[0], u_s[1] + self.fov_c[1], u_s[2] + self.fov_c[2]];
            let mut x = u_s;
            if gnl_aware {
                for _ in 0..8 {
                    let f = self.gnl.field(&x);
                    let j = self.gnl.jacobian(&x);
                    let r = [f[0] - u_s[0], f[1] - u_s[1], f[2] - u_s[2]];
                    let ji = inv3(&j);
                    let dx = mat_vec(&ji, &r);
                    for a in 0..3 {
                        x[a] -= dx[a];
                    }
                }
            }
            let xl = mat_vec(&rt, &[x[0] - self.fov_c[0], x[1] - self.fov_c[1], x[2] - self.fov_c[2]]);
            mono.push([x[2] * x[2], x[0] * x[0] + x[1] * x[1], x[0] * x[2], x[1] * x[2]]);
            omega.push(wfac * self.sample_log(&self.field_ppm, 0, &xl));
            jac.push(self.gnl.jacobian(&x));
            let s: Vec<(f64, f64)> = (0..nc).map(|c| (self.sample_log(&re_t, c, &xl), self.sample_log(&im_t, c, &xl))).collect();
            for c in 0..nc {
                let (mut re, mut im) = (0.0, 0.0);
                for c2 in 0..nc {
                    re += w[c][c2] * s[c2].0;
                    im += w[c][c2] * s[c2].1;
                }
                smaps[c * nv + q] = C32::new(re as f32, im as f32);
            }
            x_log.push(xl);
        }
        PhysicsGrid { x_log, mono, omega, jac, smaps }
    }
}

pub fn inv3(m: &Mat3) -> Mat3 {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    [
        [c(1, 1, 2, 2) / det, -c(0, 1, 2, 2) / det, c(0, 1, 1, 2) / det],
        [-c(1, 0, 2, 2) / det, c(0, 0, 2, 2) / det, -c(0, 0, 1, 2) / det],
        [c(1, 0, 2, 1) / det, -c(0, 0, 2, 1) / det, c(0, 0, 1, 1) / det],
    ]
}
