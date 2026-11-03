//! Tissue classes, MR/susceptibility properties at 3 T, steady-state spoiled-GRE amplitude, CT-derived texture.

use crate::subject::Subject;
use crate::volume::Volume;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Tissue {
    Air = 0,
    Lung,
    /// gas in airways / bowel inside the body
    Gas,
    Fat,
    Muscle,
    Bone,
    Marrow,
    Cartilage,
    Liver,
    Spleen,
    Kidney,
    Myocardium,
    /// blood outside the flow domain (heart chambers, pulmonary vessels, veins): static in the M0 model
    Blood,
    /// lumen of the aortic tree (the flow domain)
    Aorta,
}

pub const N_TISSUE: usize = 14;

impl Tissue {
    pub fn from_u8(v: u8) -> Tissue {
        assert!((v as usize) < N_TISSUE);
        unsafe { std::mem::transmute(v) }
    }
}

/// Properties of one class. `chi` in ppm (SI volume susceptibility × 1e6), `shift` chemical shift in ppm.
#[derive(Clone, Copy, Debug)]
pub struct Props {
    pub pd: f64,
    pub t1: f64,
    pub t2s: f64,
    pub chi: f64,
    pub shift: f64,
}

#[derive(Clone, Debug)]
pub struct TissueModel {
    pub props: [Props; N_TISSUE],
    /// relative amplitude modulation per standard deviation of the within-class HU (texture)
    pub texture_gain: f64,
    /// signal multiplier for blood in the flow domain (inflow enhancement)
    pub inflow_gain: f64,
}

impl TissueModel {
    /// 3 T, post-contrast blood (T1 ≈ 250 ms) as usual for aortic 4D flow; T1/T2* from the literature, rounded.
    pub fn default_3t() -> Self {
        let p = |pd, t1, t2s, chi, shift| Props { pd, t1, t2s, chi, shift };
        let props = [
            p(0.0, 1.0, 1.0, 0.36, 0.0),       // Air
            p(0.8, 1200e-3, 1.5e-3, -9.05, 0.0), // Lung (pd scaled by tissue fraction)
            p(0.0, 1.0, 1.0, 0.36, 0.0),       // Gas
            p(0.9, 380e-3, 35e-3, -8.4, -3.4),  // Fat (main methylene peak)
            p(0.70, 1420e-3, 28e-3, -9.05, 0.0), // Muscle / soft tissue
            p(0.05, 1000e-3, 1.0e-3, -11.3, 0.0), // cortical bone
            p(0.75, 600e-3, 10e-3, -9.5, 0.0),  // marrow (water part)
            p(0.6, 1000e-3, 10e-3, -9.05, 0.0), // cartilage
            p(0.70, 810e-3, 20e-3, -9.05, 0.0), // liver
            p(0.80, 1330e-3, 40e-3, -9.05, 0.0), // spleen
            p(0.85, 1200e-3, 30e-3, -9.05, 0.0), // kidney
            p(0.75, 1470e-3, 25e-3, -9.05, 0.0), // myocardium
            p(0.85, 250e-3, 45e-3, -9.05, 0.0), // blood (post contrast)
            p(0.85, 250e-3, 60e-3, -9.05, 0.0), // aortic blood (oxygenated, post contrast)
        ];
        Self { props, texture_gain: 0.08, inflow_gain: 1.5 }
    }

    /// steady-state spoiled GRE magnetisation (without T2* decay, which the simulator applies over time)
    pub fn spgr(&self, t: Tissue, tr: f64, flip: f64) -> f64 {
        let p = self.props[t as usize];
        if p.pd == 0.0 {
            return 0.0;
        }
        let e1 = (-tr / p.t1).exp();
        let s = p.pd * flip.sin() * (1.0 - e1) / (1.0 - flip.cos() * e1);
        if t == Tissue::Aorta {
            s * self.inflow_gain
        } else {
            s
        }
    }
}

fn is_bone_label(l: u8) -> bool {
    (25..=50).contains(&l) || (69..=78).contains(&l) || (91..=116).contains(&l)
}

/// Classify the 1 mm grid. Returns the class volume and the air volume fraction (lungs / gas: from HU).
pub fn classify(s: &Subject) -> (Volume<u8>, Volume<f32>) {
    let mut cls = s.labels.map(|_| 0u8);
    let mut air = s.hu.map(|_| 0f32);
    for q in 0..cls.len() {
        let hu = s.hu.data[q];
        let l = s.labels.data[q];
        let af = (-(hu as f64) / 1000.0).clamp(0.0, 1.0) as f32;
        let t = if s.body.data[q] == 0 {
            air.data[q] = 1.0;
            Tissue::Air
        } else if s.vessel.data[q] != 0 {
            Tissue::Aorta
        } else if (10..=14).contains(&l) {
            if hu < -400.0 {
                air.data[q] = af;
                Tissue::Lung
            } else if hu > 100.0 {
                Tissue::Blood
            } else {
                Tissue::Muscle
            }
        } else if hu < -400.0 {
            air.data[q] = af;
            Tissue::Gas
        } else if l == 117 {
            Tissue::Cartilage
        } else if is_bone_label(l) {
            if hu > 400.0 {
                Tissue::Bone
            } else {
                Tissue::Marrow
            }
        } else if l == 51 {
            if hu > 150.0 {
                Tissue::Blood
            } else {
                Tissue::Myocardium
            }
        } else if l == 5 {
            Tissue::Liver
        } else if l == 1 {
            Tissue::Spleen
        } else if l == 2 || l == 3 {
            Tissue::Kidney
        } else if [52, 53, 62, 63, 64, 65, 66, 67, 68].contains(&l) || hu > 150.0 {
            Tissue::Blood
        } else if (-190.0..=-30.0).contains(&hu) {
            Tissue::Fat
        } else {
            Tissue::Muscle
        };
        cls.data[q] = t as u8;
    }
    (cls, air)
}

/// Volume susceptibility (ppm) on the 1 mm grid: class value, lungs/gas as an air–tissue mixture by HU.
pub fn susceptibility(model: &TissueModel, cls: &Volume<u8>, air: &Volume<f32>) -> Volume<f32> {
    let mut chi = cls.map(|_| 0f32);
    let chi_air = model.props[Tissue::Air as usize].chi;
    for q in 0..chi.len() {
        let t = Tissue::from_u8(cls.data[q]);
        let c = match t {
            Tissue::Lung | Tissue::Gas => {
                let af = air.data[q] as f64;
                af * chi_air + (1.0 - af) * model.props[Tissue::Muscle as usize].chi
            }
            _ => model.props[t as usize].chi,
        };
        chi.data[q] = c as f32;
    }
    chi
}

/// Within-class texture factor from the CT: 1 + gain · clamp((HU − median_c)/σ_c, −2, 2).
pub fn texture(model: &TissueModel, s: &Subject, cls: &Volume<u8>) -> Volume<f32> {
    let mut stats = vec![(0.0f64, 1.0f64); N_TISSUE];
    for (c, st) in stats.iter_mut().enumerate() {
        let mut v: Vec<f32> = s.hu.data.iter().zip(&cls.data).filter(|(_, &k)| k as usize == c).map(|(&h, _)| h).step_by(7).collect();
        if v.len() > 10 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = v[v.len() / 2] as f64;
            let (q1, q3) = (v[v.len() / 4] as f64, v[3 * v.len() / 4] as f64);
            *st = (med, ((q3 - q1) / 1.349).max(1.0));
        }
    }
    let mut tex = cls.map(|_| 1f32);
    for q in 0..tex.len() {
        let (m, sd) = stats[cls.data[q] as usize];
        let z = ((s.hu.data[q] as f64 - m) / sd).clamp(-2.0, 2.0);
        tex.data[q] = (1.0 + model.texture_gain * z) as f32;
    }
    tex
}
