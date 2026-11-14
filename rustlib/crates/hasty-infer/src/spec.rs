//! `model.json` of a model directory (`data/hasty/models/<name>/`).
//!
//! ```json
//! { "kind": "segmentation", "recipe": "patch3d", "backend": "onnx", "files": {"main": "model.onnx"},
//!   "spacing_mm": [1,1,1], "orientation": "RAS",   (orientation "native": run on the input grid as stored, no resampling)
//!   "patch": [128,128,128], "overlap": 0.5,
//!   "norm": {"type": "zscore"}, "activation": "softmax", "in_channels": 1,
//!   "labels": {"0": "background", "1": "PVS"}, "license": "...", "source": "..." }
//! ```

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq)]
pub enum Norm {
    None,
    /// nnU-Net `ZScoreNormalization`: (x − mean) / max(std, 1e-8) over the whole array; with `nonzero_mask` only
    /// over non-zero voxels, which stay in place and the zeros outside stay 0
    ZScore { nonzero_mask: bool },
    /// nnU-Net `CTNormalization`: clip to [lo, hi], (x − mean) / max(std, 1e-8) (foreground statistics from the plans)
    Ct { lo: f32, hi: f32, mean: f32, std: f32 },
    /// clip to [p_lo, p_hi] percentiles of the positive voxels, then scale to [0, 1]
    Percentile { lo: f32, hi: f32 },
    /// x / p_hi of the positive voxels, clipped to [0, 1]
    ScaleMax { hi: f32 },
    /// Pham et al. T1 PVS nnU-Net preprocessing (clip, NLM denoise, CLAHE), see `pvsnorm`
    T1wPvs,
    /// same with the (2, 98) percentile pair of `T2wPVSNormalization`
    T2wPvs,
}

#[derive(Clone, Debug)]
pub struct ModelSpec {
    pub dir: PathBuf,
    pub name: String,
    pub kind: String,
    pub recipe: String,
    pub backend: String,
    /// role -> file name relative to `dir` (`main` for single-network recipes)
    pub files: BTreeMap<String, String>,
    pub spacing_mm: Option<[f64; 3]>,
    pub orientation: String,
    pub patch: Option<[usize; 3]>,
    pub overlap: f32,
    /// per input channel (a single entry applies to all channels)
    pub norm: Vec<Norm>,
    /// roles averaged by the recipe (nnU-Net folds); default `["main"]`
    pub ensemble: Vec<String>,
    /// crop to the bounding box of the non-zero voxels before normalisation (nnU-Net default preprocessing)
    pub crop_nonzero: bool,
    pub softmax: bool,
    pub in_channels: usize,
    pub labels: BTreeMap<u32, String>,
    pub license: String,
    pub source: String,
    pub raw: Value,
}

fn arr3<T: Copy>(j: &Value, k: &str, f: impl Fn(&Value) -> Option<T>) -> Result<Option<[T; 3]>> {
    match j.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(a)) if a.len() == 3 => {
            let v: Option<Vec<T>> = a.iter().map(&f).collect();
            let v = v.ok_or_else(|| Error::Spec(format!("{k}: bad element")))?;
            Ok(Some([v[0], v[1], v[2]]))
        }
        _ => Err(Error::Spec(format!("{k}: expected 3 numbers"))),
    }
}

impl ModelSpec {
    pub fn load(dir: &Path) -> Result<Self> {
        let txt = std::fs::read_to_string(dir.join("model.json"))?;
        let j: Value = serde_json::from_str(&txt).map_err(|e| Error::Spec(format!("{}: {e}", dir.join("model.json").display())))?;
        Self::from_json(dir, &j)
    }

    pub fn from_json(dir: &Path, j: &Value) -> Result<Self> {
        let s = |k: &str, d: &str| j.get(k).and_then(|v| v.as_str()).unwrap_or(d).to_string();
        let mut files = BTreeMap::new();
        match j.get("files") {
            Some(Value::Object(m)) => {
                for (k, v) in m {
                    files.insert(k.clone(), v.as_str().ok_or_else(|| Error::Spec(format!("files.{k} must be a string")))?.to_string());
                }
            }
            _ => return Err(Error::Spec("missing files".into())),
        }
        let parse_norm = |n: &Value| -> Result<Norm> {
            Ok(match n.get("type").and_then(|v| v.as_str()).unwrap_or("none") {
                "none" => Norm::None,
                "zscore" => Norm::ZScore { nonzero_mask: n["nonzero_mask"].as_bool().unwrap_or(false) },
                "ct" => {
                    let f = |k: &str| n[k].as_f64().map(|v| v as f32).ok_or_else(|| Error::Spec(format!("norm ct: missing {k}")));
                    Norm::Ct { lo: f("lo")?, hi: f("hi")?, mean: f("mean")?, std: f("std")? }
                }
                "percentile" => Norm::Percentile { lo: n["lo"].as_f64().unwrap_or(0.5) as f32, hi: n["hi"].as_f64().unwrap_or(99.5) as f32 },
                "t1w_pvs" => Norm::T1wPvs,
                "t2w_pvs" => Norm::T2wPvs,
                "fastsurfer_conform" => Norm::None, // handled by the slab2d_views recipe
                "scale_max" => Norm::ScaleMax { hi: n["hi"].as_f64().unwrap_or(99.0) as f32 },
                t => return Err(Error::Spec(format!("unknown norm type {t:?}"))),
            })
        };
        let norm = match j.get("norm") {
            None | Some(Value::Null) => vec![Norm::None],
            Some(Value::Array(a)) if !a.is_empty() => a.iter().map(&parse_norm).collect::<Result<Vec<_>>>()?,
            Some(n) => vec![parse_norm(n)?],
        };
        let in_channels = j.get("in_channels").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
        if norm.len() != 1 && norm.len() != in_channels {
            return Err(Error::Spec(format!("norm has {} entries for {in_channels} channels", norm.len())));
        }
        let ensemble: Vec<String> = match j.get("ensemble") {
            Some(Value::Array(a)) if !a.is_empty() => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
            _ => vec!["main".to_string()],
        };
        for r in &ensemble {
            if !files.contains_key(r) {
                return Err(Error::Spec(format!("ensemble role {r:?} has no file")));
            }
        }
        let mut labels = BTreeMap::new();
        if let Some(Value::Object(m)) = j.get("labels") {
            for (k, v) in m {
                labels.insert(k.parse::<u32>().map_err(|_| Error::Spec(format!("label id {k:?} is not an integer")))?, v.as_str().unwrap_or("").to_string());
            }
        }
        let overlap = j.get("overlap").and_then(|v| v.as_f64()).unwrap_or(0.5) as f32;
        if !(0.0..1.0).contains(&overlap) {
            return Err(Error::Spec("overlap must be in [0, 1)".into()));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            name: dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            kind: s("kind", "segmentation"),
            recipe: s("recipe", "patch3d"),
            backend: s("backend", "onnx"),
            files,
            spacing_mm: arr3(j, "spacing_mm", |v| v.as_f64())?,
            orientation: s("orientation", "RAS"),
            patch: arr3(j, "patch", |v| v.as_u64().map(|x| x as usize))?,
            overlap,
            norm,
            ensemble,
            crop_nonzero: j.get("crop_nonzero").and_then(|v| v.as_bool()).unwrap_or(false),
            softmax: j.get("activation").and_then(|v| v.as_str()).unwrap_or("softmax") == "softmax",
            in_channels,
            labels,
            license: s("license", ""),
            source: s("source", ""),
            raw: j.clone(),
        })
    }

    pub fn file(&self, role: &str) -> Result<PathBuf> {
        let f = self.files.get(role).ok_or_else(|| Error::Spec(format!("{}: no file for role {role:?}", self.name)))?;
        Ok(self.dir.join(f))
    }

    pub fn norm_for(&self, channel: usize) -> &Norm {
        self.norm.get(channel).unwrap_or(&self.norm[0])
    }

    pub fn num_classes(&self) -> usize {
        self.labels.keys().max().map(|&m| m as usize + 1).unwrap_or(0)
    }
}

/// all valid model directories below `root` (invalid ones are skipped and returned as messages)
pub fn scan(root: &Path) -> (Vec<ModelSpec>, Vec<String>) {
    let mut ok = Vec::new();
    let mut bad = Vec::new();
    if let Ok(rd) = std::fs::read_dir(root) {
        let mut dirs: Vec<_> = rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.join("model.json").is_file()).collect();
        dirs.sort();
        for d in dirs {
            match ModelSpec::load(&d) {
                Ok(m) => ok.push(m),
                Err(e) => bad.push(format!("{}: {e}", d.display())),
            }
        }
    }
    (ok, bad)
}
