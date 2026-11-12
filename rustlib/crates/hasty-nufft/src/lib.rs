//! GPU non-uniform FFT (types 1 and 2, 1-3 D) built on `hasty-fft`. See `rustlib/PLAN.md`, track N.

mod chirp;
mod kernel;
mod plan;
mod segbasis;
mod segnufft;
mod segwarp;

pub use chirp::{ChirpFamily, ChirpNufft, ChirpTables};
pub use kernel::{gauss_legendre, next_smooth_even, EsKernel};
pub use plan::{NufftError, NufftOpts, NufftPlan, SortedPoints, SpreadMethod};
pub use segbasis::{build_segments_fat_nodes, build_segments_svd_fat_union, build_segments_svd_fat_union_g, fat_union_spectrum, fat_union_spectrum_g, build_segments_nodes, build_segments_svd_grid, cheb_nodes, noise_time_weights, nodes_weighted_error, seg_count_for, seg_time_grid, segment_model_error, snr_time_weights, tail_errors, vol_energy, weighted_spectrum};
pub use segwarp::{cardinal_form, SegWarp, WarpStencil};
pub use segnufft::{build_segments, build_segments_complex, build_segments_fat, build_segments_svd, build_segments_svd_weighted, fold_fat_fraction, hi_exponents, multi_plan, tile_lists, SegFatMaps, SegNufft, SegNufftOpts, SegPoints, SegSegments, TileCfg, MULTI_KMAX, NHI};
