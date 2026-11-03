//! Thin CUDA layer: device handle, NVRTC -> cubin compilation with an on-disk cache.
//!
//! Core numerical crates (`hasty-fft`, `hasty-nufft`) depend on this and nothing torch-related.

mod complex;
mod nvrtc;

pub use complex::{Real, C32, C64};

pub use nvrtc::{compile_cubin, CompileSettings};

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, DriverError};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("CUDA driver error: {0:?}")]
    Driver(#[from] DriverError),
    #[error("NVRTC error in kernel `{name}` (source hash {hash}):\n{log}")]
    Nvrtc {
        name: String,
        hash: String,
        log: String,
    },
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One CUDA device with its primary context and default compile arch.
#[derive(Clone)]
pub struct Device {
    pub ctx: Arc<CudaContext>,
    pub ordinal: usize,
    /// e.g. `sm_89`
    pub arch: String,
}

impl Device {
    pub fn new(ordinal: usize) -> Result<Self> {
        let ctx = CudaContext::new(ordinal)?;
        let (major, minor) = ctx.compute_capability()?;
        Ok(Self {
            ctx,
            ordinal,
            arch: format!("sm_{major}{minor}"),
        })
    }

    pub fn default_stream(&self) -> Arc<CudaStream> {
        self.ctx.default_stream()
    }

    /// Compile (or fetch from the disk cache) `src` for this device and load it.
    pub fn load_module(
        &self,
        name: &str,
        src: &str,
        settings: &CompileSettings,
    ) -> Result<Arc<CudaModule>> {
        let cubin = compile_cubin(name, src, &self.arch, settings)?;
        let module = self
            .ctx
            .load_module(cudarc::nvrtc::Ptx::from_binary(cubin))?;
        Ok(module)
    }

    pub fn load_function(
        &self,
        name: &str,
        src: &str,
        func: &str,
        settings: &CompileSettings,
    ) -> Result<CudaFunction> {
        Ok(self.load_module(name, src, settings)?.load_function(func)?)
    }
}
