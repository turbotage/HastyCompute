//! Model inference commands: `model_list`, `model_run` (segmentation models of `data/hasty/models`, see
//! `research/plan_models.md`). Geometry comes with the call (`spacing_mm`, optional `affine`), not from a file
//! format: the node editor passes the metadata of the input entry, other callers supply it themselves.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use hasty_infer::backend::Backend;
use hasty_infer::geom::{Geometry, Volume, IDENTITY};
use hasty_infer::recipe::{segment, RunOptions};
use hasty_infer::spec::{scan, ModelSpec};
use hasty_value::{Tensor, Value};
use serde_json::{json, Value as Json};

use crate::commands::Registry;
use crate::Error;

fn inv<T>(m: impl Into<String>) -> Result<T, Error> {
    Err(Error::Invalid(m.into()))
}

fn ie(e: hasty_infer::Error) -> Error {
    Error::Invalid(e.to_string())
}

/// `HASTY_MODELS`, else `<data>/models`
pub fn models_dir() -> PathBuf {
    std::env::var_os("HASTY_MODELS").map(PathBuf::from).unwrap_or_else(|| hasty_site::data().join("models"))
}

fn ort_lib() -> Option<PathBuf> {
    if std::env::var_os("HASTY_ORT").is_some() || std::env::var_os("ORT_DYLIB_PATH").is_some() {
        return None; // hasty_infer::onnx::default_lib reads them
    }
    let p = hasty_site::target().join("ort/libonnxruntime.so");
    p.exists().then_some(p)
}

struct Slot {
    key: (String, bool),
    backend: Box<dyn Backend>,
    on_cuda: bool,
}

/// One resident model; the lock also serialises GPU use of model runs.
pub struct ModelCache {
    root: PathBuf,
    slot: Mutex<Option<Slot>>,
}

impl ModelCache {
    pub fn new(root: PathBuf) -> Self {
        Self { root, slot: Mutex::new(None) }
    }

    pub fn spec(&self, name: &str) -> Result<ModelSpec, Error> {
        if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') {
            return inv(format!("bad model name {name:?}"));
        }
        ModelSpec::load(&self.root.join(name)).map_err(|e| Error::Invalid(format!("model {name:?}: {e}")))
    }

    fn open(spec: &ModelSpec, cuda: bool, backend: &str) -> Result<(Box<dyn Backend>, bool), Error> {
        match backend {
            #[cfg(feature = "onnx")]
            "onnx" => {
                let b = hasty_infer::onnx::OnnxBackend::open(spec, cuda, ort_lib().as_deref()).map_err(ie)?;
                let c = b.cuda;
                Ok((Box::new(b), c))
            }
            other => inv(format!("backend {other:?} is not available in this build (model {})", spec.name)),
        }
    }

    /// run `f` with the backend of `spec` (loaded on demand, replacing the resident model)
    pub fn with<R>(&self, spec: &ModelSpec, cuda: bool, backend: &str, f: impl FnOnce(&mut dyn Backend, bool) -> Result<R, Error>) -> Result<R, Error> {
        let mut g = self.slot.lock().unwrap_or_else(|p| p.into_inner());
        let key = (format!("{}/{backend}", spec.name), cuda);
        if g.as_ref().map(|s| s.key != key).unwrap_or(true) {
            *g = None; // free VRAM before loading the next
            let (b, on_cuda) = Self::open(spec, cuda, backend)?;
            *g = Some(Slot { key, backend: b, on_cuda });
        }
        let s = g.as_mut().unwrap();
        f(s.backend.as_mut(), s.on_cuda)
    }
}

fn geometry(o: &Json, dims: [usize; 3]) -> Result<Geometry, Error> {
    let sp = o["spacing_mm"].as_array().filter(|a| a.len() == 3).map(|a| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0)));
    let affine = match o["affine"].as_array() {
        Some(rows) if rows.len() >= 3 => {
            let mut a = IDENTITY;
            for (i, r) in rows.iter().take(3).enumerate() {
                let r = r.as_array().filter(|r| r.len() == 4).ok_or_else(|| Error::Invalid("affine: 4×4 (or 3×4) numbers".into()))?;
                for j in 0..4 {
                    a[i][j] = r[j].as_f64().ok_or_else(|| Error::Invalid("affine: numbers".into()))?;
                }
            }
            Some(a)
        }
        Some(_) => return inv("affine: 4×4 (or 3×4) numbers"),
        None => None,
    };
    let g = match (affine, sp) {
        (Some(affine), _) => Geometry { dims, affine },
        (None, Some(s)) if s.iter().all(|&v| v > 0.0) => Geometry::from_spacing(dims, s),
        _ => return inv("model_run: options.spacing_mm [sx, sy, sz] (x fastest) or options.affine required — pass the input entry's metadata"),
    };
    Ok(g)
}

pub fn register(r: &mut Registry) {
    let cache = Arc::new(ModelCache::new(models_dir()));
    let c0 = cache.clone();
    r.register("model_list", "installed models (message JSON: name, kind, recipe, backend, spacing, patch, labels, license, source, problems)", move |_, _, _| {
        let (specs, bad) = scan(&c0.root);
        let list: Vec<Json> = specs
            .iter()
            .map(|m| json!({"name": m.name, "kind": m.kind, "recipe": m.recipe, "backend": m.backend, "spacing_mm": m.spacing_mm, "orientation": m.orientation,
                "patch": m.patch, "in_channels": m.in_channels, "labels": m.labels, "license": m.license, "source": m.source}))
            .collect();
        Ok((json!({"root": c0.root, "models": list, "problems": bad}).to_string(), vec![]))
    });
    r.register(
        "model_run",
        "segmentation: inputs [one image tensor per model channel, hasty [Z,Y,X], all on the same grid]; options {model, device: cuda|cpu (default cuda), spacing_mm: [sx,sy,sz] and/or affine (4×4 voxel→RAS mm, x fastest), overlap, tta, backend, prob_classes: [ids]}. Outputs [labels u8/u16 [Z,Y,X], optional probabilities f32 [C,Z,Y,X]]; message JSON {labels (id → name), counts (id → voxels), backend, cuda, seconds}",
        move |ctx, o, inp| {
            let first = inp.first().and_then(|v| v.as_tensor()).ok_or_else(|| Error::Invalid("model_run: input 0 must be an image tensor".into()))?;
            let &[nz, ny, nx] = first.shape() else { return inv(format!("model_run: images must be [Z, Y, X], got {:?}", first.shape())) };
            let name = o["model"].as_str().ok_or_else(|| Error::Invalid("model_run: options.model missing (see model_list)".into()))?;
            let spec = cache.spec(name)?;
            if spec.kind != "segmentation" {
                return inv(format!("model {name}: kind {:?} is not segmentation", spec.kind));
            }
            if inp.len() != spec.in_channels {
                return inv(format!("model {name}: needs {} input channel(s) (one tensor each, in model order), got {}", spec.in_channels, inp.len()));
            }
            let geom = geometry(o, [nx, ny, nz])?;
            let images = inp
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let t = v.as_tensor().filter(|t| t.shape() == first.shape()).ok_or_else(|| Error::Invalid(format!("model_run: input {i} must be a tensor of shape {:?}", first.shape())))?;
                    Volume::new(geom.clone(), (0..t.numel()).map(|j| t.get_f64(j).0 as f32).collect()).map_err(ie)
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let cuda = o["device"].as_str().unwrap_or("cuda") != "cpu";
            let backend = o["backend"].as_str().unwrap_or(&spec.backend).to_string();
            let want: Vec<usize> = o["prob_classes"].as_array().map(|a| a.iter().filter_map(|x| x.as_u64().map(|x| x as usize)).collect()).unwrap_or_default();
            let opts = RunOptions { overlap: o["overlap"].as_f64().map(|x| x as f32), tta: o["tta"].as_bool().unwrap_or(false), probabilities: !want.is_empty() };
            let t0 = Instant::now();
            let mut last = 0usize;
            let bank = ctx.bank.clone();
            let label = format!("model_run {name}");
            let (seg, on_cuda) = cache.with(&spec, cuda, &backend, |b, on_cuda| {
                let seg = segment(b, &spec, &images, &opts, &mut |d, n| {
                    let pct = d * 10 / n.max(1);
                    if pct != last {
                        last = pct;
                        bank.notify("log", None, &label, &format!("{d}/{n} patches"));
                    }
                    true
                })
                .map_err(ie)?;
                Ok((seg, on_cuda))
            })?;
            let k = spec.num_classes();
            let mut counts: std::collections::BTreeMap<u16, usize> = Default::default();
            for &l in &seg.labels {
                *counts.entry(l).or_default() += 1;
            }
            let labels = if counts.keys().next_back().map(|&m| m <= 255).unwrap_or(true) {
                Tensor::from_slice(&[nz, ny, nx], &seg.labels.iter().map(|&l| l as u8).collect::<Vec<_>>())
            } else {
                Tensor::from_slice(&[nz, ny, nx], &seg.labels)
            }
            .map_err(Error::from)?;
            let mut outs = vec![Value::Tensor(labels)];
            if let Some(p) = seg.probabilities {
                let mut flat = Vec::with_capacity(want.len() * nx * ny * nz);
                for &c in &want {
                    flat.extend_from_slice(p.get(c).ok_or_else(|| Error::Invalid(format!("prob_classes: class {c} out of range (0..{k})")))?);
                }
                outs.push(Value::Tensor(Tensor::from_slice(&[want.len(), nz, ny, nx], &flat).map_err(Error::from)?));
            }
            let msg = json!({"labels": spec.labels, "counts": counts.iter().map(|(k, v)| (k.to_string(), *v)).collect::<std::collections::BTreeMap<_, _>>(), "backend": backend, "cuda": on_cuda, "seconds": t0.elapsed().as_secs_f64(), "prob_classes": want});
            Ok((msg.to_string(), outs))
        },
    );
}
