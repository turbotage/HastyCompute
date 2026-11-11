//! Reconstruction for the synthetic 4D-flow benchmark: data loading (simulator output), physics grids (GNL inverse
//! warp, off-resonance, concomitant monomials), low-rank phase models, multi-coil NUFFT encoding operators, solvers.

pub mod cg;
pub mod checkpoint;
pub mod coils;
pub mod espirit;
pub mod fieldmap;
pub mod kload;
pub mod kstage;
pub mod kstore;
pub mod data;
pub mod dfwavelet;
pub mod flow;
pub mod gpu;
pub mod hoststack;
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
pub mod wavelet;
pub mod vinit;
pub mod precond;
pub mod pyramid;
pub mod segphys;
pub mod selfgate;
pub mod sepwt;
pub mod setop;
pub mod store;
pub mod sysmem;
pub mod tlock;
pub mod airmap;
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
