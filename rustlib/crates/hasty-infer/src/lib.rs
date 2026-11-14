//! Model inference on volumes, independent of file formats and of any inference runtime.
//!
//! Layers:
//! - [`geom`]: voxel grids with an affine, resampling, conforming to a model grid.
//! - [`spec`]: `model.json` of a model directory.
//! - [`backend`]: the [`backend::Backend`] trait (tensor in, tensor out); runtimes (ONNX, TorchScript) implement it.
//! - [`recipe`]: pre/post-processing around a backend (`patch3d` tiling, ...), testable with a mock backend.
//!
//! Volume layout: x varies fastest, `dims = [nx, ny, nz]`; as a C-order tensor that is `[nz, ny, nx]`, which is also
//! the layout of hasty server tensors loaded from NIfTI. World space is RAS+ millimetres.

pub mod backend;
pub mod fastsurfer;
pub mod geom;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod pvsnorm;
pub mod recipe;
pub mod spec;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("model spec: {0}")]
    Spec(String),
    #[error("geometry: {0}")]
    Geometry(String),
    #[error("backend: {0}")]
    Backend(String),
    #[error("cancelled")]
    Cancelled,
}

pub type Result<T> = std::result::Result<T, Error>;
