//! A prepared subject (output of `tools/synth/prep_subject.py`).

use std::path::{Path, PathBuf};

use hasty_mrsim::Vec3;
use serde::Deserialize;

use crate::volume::{read_npy, Volume};
use crate::Error;

#[derive(Clone, Debug, Deserialize)]
pub struct Centerline {
    pub vertices: Vec<Vec3>,
    pub edges: Vec<[usize; 2]>,
    pub radius: Vec<f64>,
}

#[derive(Deserialize)]
struct GridMeta {
    shape: [usize; 3],
    affine: [[f64; 4]; 4],
}
#[derive(Deserialize)]
struct Meta {
    case: String,
    grid_1mm: GridMeta,
    grid_vessel_05mm: GridMeta,
}

/// Anatomy on a 1 mm grid plus the vessel lumen SDF on a 0.5 mm grid, all in the physical frame (DICOM LPS, metres,
/// +z superior = B0 axis). The reference configuration: motion models map from these coordinates.
pub struct Subject {
    pub name: String,
    pub dir: PathBuf,
    pub hu: Volume<f32>,
    /// TotalSegmentator v2 `total` labels
    pub labels: Volume<u8>,
    pub body: Volume<u8>,
    /// expert aortic-vessel-tree mask (AVT), 1 mm
    pub vessel: Volume<u8>,
    /// lumen signed distance (m, negative inside), 0.5 mm
    pub sdf: Volume<f32>,
    pub centerline: Centerline,
}

impl Subject {
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let meta: Meta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)?;
        let a1 = meta.grid_1mm.affine;
        let hu = read_npy::<f32>(&dir.join("hu_1mm.npy"), a1)?;
        let labels = read_npy::<u8>(&dir.join("labels_1mm.npy"), a1)?;
        let body = read_npy::<u8>(&dir.join("body_1mm.npy"), a1)?;
        let vessel = read_npy::<u8>(&dir.join("vessel_1mm.npy"), a1)?;
        let sdf = read_npy::<f32>(&dir.join("vessel_sdf_05mm.npy"), meta.grid_vessel_05mm.affine)?;
        assert_eq!(hu.dims, meta.grid_1mm.shape);
        assert_eq!(sdf.dims, meta.grid_vessel_05mm.shape);
        let centerline: Centerline = serde_json::from_str(&std::fs::read_to_string(dir.join("centerline.json"))?)?;
        Ok(Self { name: meta.case, dir: dir.to_path_buf(), hu, labels, body, vessel, sdf, centerline })
    }

    /// centroid of a label set (physical), e.g. the heart (51)
    pub fn label_centroid(&self, labels: &[u8]) -> Option<Vec3> {
        let (mut s, mut n) = ([0.0; 3], 0usize);
        let d = self.labels.dims;
        for k in 0..d[2] {
            for j in 0..d[1] {
                for i in 0..d[0] {
                    if labels.contains(&self.labels.get(i, j, k)) {
                        s[0] += i as f64;
                        s[1] += j as f64;
                        s[2] += k as f64;
                        n += 1;
                    }
                }
            }
        }
        (n > 0).then(|| self.labels.to_phys([s[0] / n as f64, s[1] / n as f64, s[2] / n as f64]))
    }
}
