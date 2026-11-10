//! Cardiac lattice-Boltzmann CFD for the flow phantom (research/flow_phantom.md Tier B, research/cfd_training_data.md §4):
//! blood flow in the heart chambers and great-vessel roots driven by the prescribed wall motion of the phantom
//! (`cardiac_motion.hv`), with kinematic valves and network-coupled inlets / outlets.
//!
//! - [`params`]: lattice constants and the `#[repr(C)]` parameter blocks shared with the C core.
//! - [`motion`]: host-side cardiac motion (f16 samples, periodic Catmull-Rom in time, trilinear in space, forward and
//!   inverse map) for the valve / cut kinematics and the active-tile search.
//! - [`case`]: the case (LBM grid, reference signed distance of the blood domain, motion, fluid, valves, cuts, run
//!   settings) from `case.json`, leaflet presets.
//! - [`backend`]: the [`backend::Backend`] trait, the sparse tile topology and the CPU backend (the C core of
//!   `kernels/lbm_core.h` compiled by `build.rs`, threads over tile-aligned node ranges).
//! - [`gpu`]: the CUDA backend (the same C core through NVRTC).
//! - [`solver`]: units, time stepping, geometry updates, valve / cut states, RCR outlets, phase-binned accumulation of
//!   the Eulerian velocity (mean and covariance), diagnostics, checkpoints.
//! - [`probe`]: debug views (fastest nodes and their surroundings, sparse field snapshots).
//!
//! Physics in `kernels/lbm_core.h` (D3Q19 regularized BGK + WALE, Bouzidi moving-wall bounce-back with refill,
//! partially saturated valve cells, equilibrium inlet / outlet slabs).

pub mod backend;
pub mod case;
pub mod gpu;
pub mod motion;
pub mod params;
pub mod probe;
pub mod solver;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("CUDA: {0}")]
    Cuda(#[from] hasty_cuda::Error),
    #[error("CUDA driver: {0:?}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("value file: {0}")]
    Value(String),
    #[error("case: {0}")]
    Case(String),
    #[error("unstable: {0}")]
    Unstable(String),
}

pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests;
