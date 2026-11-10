//! GPU non-uniform FFT (types 1 and 2, 1-3 D) built on `hasty-fft`. See `rustlib/PLAN.md`, track N.

mod chirp;
mod kernel;
mod plan;
mod segnufft;

pub use chirp::{ChirpFamily, ChirpNufft, ChirpTables};
pub use kernel::{gauss_legendre, next_smooth_even, EsKernel};
pub use plan::{NufftError, NufftOpts, NufftPlan, SortedPoints, SpreadMethod};
pub use segnufft::{build_segments, build_segments_complex, build_segments_fat, build_segments_svd, build_segments_svd_weighted, fold_fat_fraction, hi_exponents, multi_plan, tile_lists, SegFatMaps, SegNufft, SegNufftOpts, SegPoints, SegSegments, TileCfg, MULTI_KMAX, NHI};
