//! ONNX Runtime backend (feature `onnx`). `libonnxruntime` is dlopen'ed (`ort` `load-dynamic`), so nothing of the
//! runtime is linked or built here; use the CUDA-enabled build (`pip download onnxruntime-gpu`, unpack
//! `onnxruntime/capi/libonnxruntime.so*`) to run on the GPU.
//!
//! Library lookup: explicit path, else `HASTY_ORT`, else `ORT_DYLIB_PATH`, else `libonnxruntime.so` through the
//! dynamic loader.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ort::ep;
use ort::session::Session;
use ort::value::Tensor as OrtTensor;

use crate::backend::{Backend, Tensor};
use crate::spec::ModelSpec;
use crate::{Error, Result};

fn be<E: std::fmt::Display>(e: E) -> Error {
    Error::Backend(format!("onnx: {e}"))
}

static ENV: OnceLock<std::result::Result<ort::environment::Environment, String>> = OnceLock::new();

pub fn default_lib() -> PathBuf {
    std::env::var_os("HASTY_ORT").or_else(|| std::env::var_os("ORT_DYLIB_PATH")).map(PathBuf::from).unwrap_or_else(|| PathBuf::from("libonnxruntime.so"))
}

/// load the runtime once per process (a second call with another path is ignored)
fn env(lib: &Path) -> Result<&'static ort::environment::Environment> {
    ENV.get_or_init(|| ort::init_from(lib).map_err(|e| format!("{}: {e}", lib.display())).and_then(|b| b.build().map_err(|e| e.to_string())))
        .as_ref()
        .map_err(|e| Error::Backend(format!("onnx: {e}")))
}

pub struct OnnxBackend {
    sessions: BTreeMap<String, (Session, String)>,
    pub cuda: bool,
}

impl OnnxBackend {
    /// open every file of `spec.files`; `cuda` requests the CUDA execution provider (falls back to CPU if the
    /// provider cannot be registered, see [`OnnxBackend::cuda`] for what happened)
    pub fn open(spec: &ModelSpec, cuda: bool, lib: Option<&Path>) -> Result<Self> {
        let lib = lib.map(Path::to_path_buf).unwrap_or_else(default_lib);
        let env = env(&lib)?;
        let mut sessions = BTreeMap::new();
        let mut on_cuda = false;
        for role in spec.files.keys() {
            let mut b = None;
            if cuda {
                b = Session::builder(env).ok().and_then(|b| b.with_execution_providers([ep::CUDA::default().with_tf32(false).build().error_on_failure()]).map_err(|e| eprintln!("hasty-infer: CUDA execution provider unavailable: {e}")).ok());
                on_cuda = b.is_some();
            }
            let mut b = match b {
                Some(b) => b,
                None => Session::builder(env).map_err(be)?,
            };
            let s = b.commit_from_file(spec.file(role)?).map_err(be)?;
            let input = s.inputs().first().map(|i| i.name().to_string()).ok_or_else(|| Error::Backend(format!("onnx: {role}: model has no inputs")))?;
            sessions.insert(role.clone(), (s, input));
        }
        Ok(Self { sessions, cuda: on_cuda })
    }
}

impl Backend for OnnxBackend {
    fn name(&self) -> &str {
        "onnx"
    }

    fn forward(&mut self, role: &str, input: &Tensor) -> Result<Tensor> {
        let (s, name) = self.sessions.get_mut(role).ok_or_else(|| Error::Backend(format!("onnx: no network for role {role:?}")))?;
        let shape: Vec<i64> = input.shape.iter().map(|&d| d as i64).collect();
        let t = OrtTensor::from_array((shape, input.data.clone())).map_err(be)?;
        let out = s.run(ort::inputs![name.as_str() => t]).map_err(be)?;
        let (shape, data) = out[0].try_extract_tensor::<f32>().map_err(be)?;
        Ok(Tensor::new(shape.iter().map(|&d| d as usize).collect(), data.to_vec()))
    }
}
