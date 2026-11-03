//! A simulation scene: subject + tissue model + fields + coils + flow + prescription → simulator sources.
//!
//! Coordinates: the subject lives in its own physical frame (LPS, m). The scanner frame is the subject frame shifted
//! by the isocentre: x_s = x − iso. Sources carry x_s (concomitant terms), u = f_GNL(x_s) − c_s (encoding position
//! relative to the FOV centre c_s = center − iso) and J-transformed velocities.
//!
//! Time dependence: tissue is static in this first version. Motion (cardiac wall motion, respiration for 5D flow) is
//! meant to enter as a map reference → physical position at (t_card, t_resp) applied in [`Scene::tissue_sources`]
//! and to the flow domain; everything downstream already takes per-time source sets.

use hasty_mrsim::direct::Source;
use hasty_mrsim::gnl::GradientCoils;
use hasty_mrsim::{mat_vec, transpose, Mat3, Vec3, GAMMA_BAR};
use num_complex::Complex64;

use crate::b0::{dipole_field, shim};
use crate::coils::{body_arrays, noise_covariance, Loop, SensitivityTable};
use crate::flow::{lagrangian_accel, FlowField};
use crate::prescription::Prescription;
use crate::subject::Subject;
use crate::tissue::{classify, susceptibility, texture, Tissue, TissueModel};
use crate::volume::Volume;

pub struct SceneParams {
    pub b0: f64,
    pub tr: f64,
    pub flip: f64,
    pub gnl: GradientCoils,
    /// isocentre offset from the default landmark (body centre in x/y, heart centroid in z), m
    pub iso_offset: Vec3,
    pub coils_lr: usize,
    pub coils_si: usize,
    pub coil_radius: f64,
    /// slab profile transition width, m
    pub slab_edge: f64,
    /// diagnostics: blood keeps its signal but does not move
    pub static_blood: bool,
    /// diagnostics: no susceptibility off-resonance (chemical shift kept)
    pub no_b0: bool,
}

impl SceneParams {
    pub fn default_3t(tr: f64) -> Self {
        Self {
            b0: 3.0,
            tr,
            flip: 12f64.to_radians(),
            gnl: GradientCoils::clinical_synthetic(),
            iso_offset: [0.0; 3],
            coils_lr: 4,
            coils_si: 4,
            coil_radius: 0.045,
            slab_edge: 0.002,
            static_blood: false,
            no_b0: false,
        }
    }
}

pub struct Scene<'a> {
    pub subject: &'a Subject,
    pub model: TissueModel,
    pub params: SceneParams,
    pub cls: Volume<u8>,
    pub air: Volume<f32>,
    pub tex: Volume<f32>,
    /// off-resonance after shim, ppm
    pub field_ppm: Volume<f32>,
    pub loops: Vec<Loop>,
    pub sens: SensitivityTable,
    pub psi: Vec<Vec<f64>>,
    pub presc: Prescription,
    /// isocentre in subject coordinates
    pub iso: Vec3,
    pub flow: &'a dyn FlowField,
}

fn smoothstep_slab(z: f64, half: f64, edge: f64) -> f64 {
    let d = half - z.abs();
    if d >= edge * 0.5 {
        1.0
    } else if d <= -edge * 0.5 {
        0.0
    } else {
        0.5 + d / edge
    }
}

impl<'a> Scene<'a> {
    pub fn new(subject: &'a Subject, model: TissueModel, params: SceneParams, presc: Prescription, flow: &'a dyn FlowField) -> Self {
        let (cls, air) = classify(subject);
        let chi = susceptibility(&model, &cls, &air);
        let tex = texture(&model, subject, &cls);
        let mut field = dipole_field(&chi, model.props[Tissue::Air as usize].chi as f32, 64, 96);
        // shim over tissue within the prescribed slab box
        let mut mask = cls.map(|_| 0u8);
        let rt = transpose(&presc.rot);
        for k in 0..cls.dims[2] {
            for j in 0..cls.dims[1] {
                for i in 0..cls.dims[0] {
                    let q = cls.idx(i, j, k);
                    let t = Tissue::from_u8(cls.data[q]);
                    if matches!(t, Tissue::Air | Tissue::Gas | Tissue::Lung) {
                        continue;
                    }
                    let x = cls.to_phys([i as f64, j as f64, k as f64]);
                    let l = mat_vec(&rt, &[x[0] - presc.center[0], x[1] - presc.center[1], x[2] - presc.center[2]]);
                    if l[0].abs() < presc.fov[0] * 0.4 && l[1].abs() < presc.fov[1] * 0.4 && l[2].abs() < presc.fov[2] * 0.5 + 0.02 {
                        mask.data[q] = 1;
                    }
                }
            }
        }
        shim(&mut field, &mask, presc.center);
        // isocentre: body centre (x, y) at the heart level, heart centroid z
        let heart = subject.label_centroid(&[51]).unwrap_or(presc.center);
        let kh = subject.body.to_voxel(&heart)[2].round().clamp(0.0, (subject.body.dims[2] - 1) as f64) as usize;
        let (mut sx, mut sy, mut n) = (0.0, 0.0, 0usize);
        for j in 0..subject.body.dims[1] {
            for i in 0..subject.body.dims[0] {
                if subject.body.get(i, j, kh) != 0 {
                    let p = subject.body.to_phys([i as f64, j as f64, kh as f64]);
                    sx += p[0];
                    sy += p[1];
                    n += 1;
                }
            }
        }
        let iso = [sx / n as f64 + params.iso_offset[0], sy / n as f64 + params.iso_offset[1], heart[2] + params.iso_offset[2]];
        let zr = (presc.center[2] - 0.17, presc.center[2] + 0.17);
        let loops = body_arrays(&subject.body, params.coils_lr, params.coils_si, params.coil_radius, zr.0, zr.1);
        let sens = SensitivityTable::new(&loops, &subject.body, 4);
        let psi = noise_covariance(&loops, &subject.body, 6, 0.1);
        Self { subject, model, params, cls, air, tex, field_ppm: field, loops, sens, psi, presc, iso, flow }
    }

    pub fn nc(&self) -> usize {
        self.loops.len()
    }

    /// FOV centre in the scanner frame
    pub fn fov_center_scanner(&self) -> Vec3 {
        [0, 1, 2].map(|q| self.presc.center[q] - self.iso[q])
    }

    /// logical coordinates relative to the FOV centre
    pub fn to_logical(&self, x: &Vec3) -> Vec3 {
        mat_vec(&transpose(&self.presc.rot), &[x[0] - self.presc.center[0], x[1] - self.presc.center[1], x[2] - self.presc.center[2]])
    }

    fn slab_weight(&self, x: &Vec3) -> f64 {
        let l = self.to_logical(x);
        smoothstep_slab(l[2], 0.5 * self.presc.fov[2], self.params.slab_edge)
    }

    fn omega(&self, x: &Vec3, shift_ppm: f64) -> f64 {
        let field = if self.params.no_b0 { 0.0 } else { self.field_ppm.sample(x, 0.0) as f64 };
        let ppm = field + shift_ppm;
        2.0 * std::f64::consts::PI * GAMMA_BAR * self.params.b0 * ppm * 1e-6
    }

    fn make_source(&self, x: &Vec3, v: Vec3, dvdt: Vec3, a: Vec3, omega: f64, r2s: f64) -> Source {
        let xs = [x[0] - self.iso[0], x[1] - self.iso[1], x[2] - self.iso[2]];
        let f = self.params.gnl.field(&xs);
        let c = self.fov_center_scanner();
        let j: Mat3 = self.params.gnl.jacobian(&xs);
        Source { u: [f[0] - c[0], f[1] - c[1], f[2] - c[2]], x: xs, vt: mat_vec(&j, &v), vtt: mat_vec(&j, &dvdt), at: mat_vec(&j, &a), omega, r2s }
    }

    fn lumen_fraction_1mm(&self, x: &Vec3) -> f64 {
        let mut n = 0;
        for dz in [-0.25e-3, 0.25e-3] {
            for dy in [-0.25e-3, 0.25e-3] {
                for dx in [-0.25e-3, 0.25e-3] {
                    if self.subject.sdf.sample(&[x[0] + dx, x[1] + dy, x[2] + dz], 1.0) < 0.0 {
                        n += 1;
                    }
                }
            }
        }
        n as f64 / 8.0
    }

    /// Static sources on the 1 mm grid within the excited slab (lumen of the flow domain excluded by its volume
    /// fraction; the vessel wall share of boundary voxels is muscle-like). Amplitudes `[src][coil]`.
    pub fn tissue_sources(&self) -> (Vec<Source>, Vec<Complex64>) {
        let d = self.cls.dims;
        let nc = self.nc();
        let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        let parts: Vec<(Vec<Source>, Vec<Complex64>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..nthreads)
                .map(|t| {
                    sc.spawn(move || {
                        let (mut src, mut amp) = (Vec::new(), Vec::new());
                        let mut s = vec![Complex64::new(0.0, 0.0); nc];
                        for k in (t..d[2]).step_by(nthreads) {
                            for j in 0..d[1] {
                                for i in 0..d[0] {
                                    let q = self.cls.idx(i, j, k);
                                    let mut tt = Tissue::from_u8(self.cls.data[q]);
                                    if matches!(tt, Tissue::Air | Tissue::Gas) {
                                        continue;
                                    }
                                    let x = self.cls.to_phys([i as f64, j as f64, k as f64]);
                                    let w_slab = self.slab_weight(&x);
                                    if w_slab <= 0.0 {
                                        continue;
                                    }
                                    let mut w = w_slab * self.tex.data[q] as f64;
                                    if tt == Tissue::Lung {
                                        w *= 1.0 - self.air.data[q] as f64;
                                    }
                                    let fl = if self.subject.sdf.sample(&x, 1.0) < 2e-3 { self.lumen_fraction_1mm(&x) } else { 0.0 };
                                    if tt == Tissue::Aorta {
                                        tt = Tissue::Muscle;
                                    }
                                    w *= 1.0 - fl;
                                    if w <= 1e-6 {
                                        continue;
                                    }
                                    let m = self.model.spgr(tt, self.params.tr, self.params.flip) * w;
                                    let p = self.model.props[tt as usize];
                                    src.push(self.make_source(&x, [0.0; 3], [0.0; 3], [0.0; 3], self.omega(&x, p.shift), 1.0 / p.t2s));
                                    self.sens.at(&x, &mut s);
                                    amp.extend(s.iter().map(|c| c * m));
                                }
                            }
                        }
                        (src, amp)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let (mut src, mut amp) = (Vec::new(), Vec::new());
        for (s, a) in parts {
            src.extend(s);
            amp.extend(a);
        }
        (src, amp)
    }

    /// Lumen points of the 0.5 mm vessel grid inside the slab (fixed Eulerian positions; rigid walls).
    pub fn blood_points(&self) -> Vec<Vec3> {
        let sdf = &self.subject.sdf;
        let d = sdf.dims;
        let mut pts = Vec::new();
        for k in 0..d[2] {
            for j in 0..d[1] {
                for i in 0..d[0] {
                    if sdf.get(i, j, k) < 0.0 {
                        let x = sdf.to_phys([i as f64, j as f64, k as f64]);
                        if self.slab_weight(&x) > 0.0 {
                            pts.push(x);
                        }
                    }
                }
            }
        }
        pts
    }

    /// Flowing-blood sources at flow time `t` (velocity, ∂v/∂t and Lagrangian acceleration from the flow field).
    pub fn blood_sources(&self, pts: &[Vec3], t: f64) -> (Vec<Source>, Vec<Complex64>) {
        let nc = self.nc();
        let vol = {
            let v = self.subject.sdf.voxel_size();
            v[0] * v[1] * v[2] / 1e-9 // relative to the 1 mm tissue voxels
        };
        let base = self.model.spgr(Tissue::Aorta, self.params.tr, self.params.flip) * vol;
        let p = self.model.props[Tissue::Aorta as usize];
        let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        let chunk = pts.len().div_ceil(nthreads).max(1);
        let parts: Vec<(Vec<Source>, Vec<Complex64>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = pts
                .chunks(chunk)
                .map(|ps| {
                    sc.spawn(move || {
                        let mut src = Vec::with_capacity(ps.len());
                        let mut amp = Vec::with_capacity(ps.len() * nc);
                        let mut s = vec![Complex64::new(0.0, 0.0); nc];
                        for x in ps {
                            let (v, dvdt, a) = if self.params.static_blood { ([0.0; 3], [0.0; 3], [0.0; 3]) } else { lagrangian_accel(self.flow, x, t, 0.5e-3) };
                            let m = base * self.slab_weight(x);
                            src.push(self.make_source(x, v, dvdt, a, self.omega(x, p.shift), 1.0 / p.t2s));
                            self.sens.at(x, &mut s);
                            amp.extend(s.iter().map(|c| c * m));
                        }
                        (src, amp)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut src = Vec::with_capacity(pts.len());
        let mut amp = Vec::with_capacity(pts.len() * nc);
        for (s, a) in parts {
            src.extend(s);
            amp.extend(a);
        }
        (src, amp)
    }
}

/// Noise covariance helper: lower Cholesky factor of Ψ (for coloured noise generation).
pub fn cholesky(psi: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = psi.len();
    let mut l = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                l[i][j] = (psi[i][i] - s).max(1e-12).sqrt();
            } else {
                l[i][j] = (psi[i][j] - s) / l[j][j];
            }
        }
    }
    l
}
