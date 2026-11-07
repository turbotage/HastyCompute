//! Reconstruction for the synthetic 4D-flow benchmark: data loading (simulator output), physics grids (GNL inverse
//! warp, off-resonance, concomitant monomials), low-rank phase models, multi-coil NUFFT encoding operators, solvers.

pub mod cg;
pub mod checkpoint;
pub mod data;
pub mod flow;
pub mod gpu;
pub mod learned;
pub mod llr;
pub mod moco;
pub mod motion;
pub mod multiterm;
pub mod op;
pub mod p2np;
pub mod plan;
pub mod vmodel;
pub mod vobj;
pub mod vinit;
pub mod precond;
pub mod segphys;
pub mod selfgate;
pub mod setop;
pub mod store;
pub mod skeleton;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Cuda(#[from] hasty_cuda::Error),
    #[error("CUDA driver error: {0:?}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error(transparent)]
    Nufft(#[from] hasty_nufft::NufftError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Fft(#[from] hasty_fft::FftError),
    #[error(transparent)]
    Toeplitz(#[from] hasty_toeplitz::Error),
    #[error("format error: {0}")]
    Format(String),
    #[error(transparent)]
    Value(#[from] hasty_value::Error),
}
