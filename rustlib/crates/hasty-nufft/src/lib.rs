//! GPU non-uniform FFT (types 1 and 2, 1-3 D) built on `hasty-fft`. See `rustlib/PLAN.md`, track N.

mod chirp;
mod kernel;
mod plan;
mod segnufft;

pub use chirp::{ChirpFamily, ChirpNufft, ChirpTables};
pub use kernel::{gauss_legendre, next_smooth_even, EsKernel};
pub use plan::{NufftError, NufftOpts, NufftPlan, SortedPoints, SpreadMethod};
pub use segnufft::{build_segments, tile_lists, SegNufft, SegNufftOpts, SegPoints, SegSegments, TileCfg};
