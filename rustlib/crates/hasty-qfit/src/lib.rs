//! hasty-qfit: batched nonlinear least squares for quantitative MRI — symbolic models with exact gradients and
//! Hessians, a second-order Levenberg–Marquardt solver with modified-Cholesky (GMW81 / SE99) steps, one problem per
//! CUDA thread with the whole solve in one kernel; the CPU path runs the same generated programs as a reference.
//!
//! Re-implements the ideas of the earlier Python/CuPy `pycompute` pipeline (symengine → CUDA) in Rust.
//!
//! ```text
//! let m = Model::new(&["s0", "r2"], &["te"], &[], &["s0 * exp(-r2 * te)"])?;
//! ```

pub mod chol;
pub mod expr;
pub mod fit;
pub mod gpu;
pub mod lm;
pub mod model;
pub mod program;
pub mod stats;
pub mod waterfat;

pub use fit::{fit_cpu, Data, SymProblem};
pub use lm::{lm, FitOut, HessianMode, LambdaRule, LmOptions, ModChol, Objective, Status};
pub use model::Model;
