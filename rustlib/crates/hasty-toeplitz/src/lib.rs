//! Toeplitz-type normal operators with spatially varying (concomitant-field) phase: kernel construction, mixing engine, expansions.
//! See `research/` for the mathematics.

mod fields;
mod kernels;
mod mix;
mod poly;
mod reference;
mod spec;

pub use fields::{m_eval, xi, NQ};
pub use kernels::{build_kernels, KernelBuildOpts, Samples};
pub use mix::MixOp;
pub use reference::{apply_dense, dense_operator, dense_series_operator, SampleSetHost, Series};
pub use spec::{
    cp, multi_indices, plain, taylor_hankel, tucker, weyl, Entry, FieldDiag, KernelKey, MixSpec,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Cuda(#[from] hasty_cuda::Error),
    #[error("CUDA driver error: {0:?}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error(transparent)]
    Fft(#[from] hasty_fft::FftError),
    #[error(transparent)]
    Nufft(#[from] hasty_nufft::NufftError),
}
