//! GPU non-uniform FFT (types 1 and 2, 1-3 D) built on `hasty-fft`. See `rustlib/PLAN.md`, track N.

mod chirp;
mod gridsize;
mod kernel;
mod plan;
mod segbasis;
mod segjoint;
mod segnufft;
// segment batches inside a coil (research/sample_kernels.md §18)
mod segsplit;
mod segwarp;

pub use chirp::{ChirpFamily, ChirpNufft, ChirpTables};
pub use gridsize::{candidates as grid_candidates, choose_grid, default_size as grid_default_size, grid_for_engine, grid_rule, set_thread_grid_rule, sigma_eff, smooth_sizes, time_grid_fft, GridChoice, GridRule, GRID_TIMING_CHANNELS};
pub use kernel::{gauss_legendre, next_smooth_even, EsKernel, KbPoly};
pub use plan::{NufftError, NufftOpts, NufftPlan, SortedPoints, SpreadMethod};
pub use segbasis::{build_segments_fat_nodes, build_segments_svd_fat_union, build_segments_svd_fat_union_g, fat_union_spectrum, fat_union_spectrum_g, build_segments_nodes, build_segments_svd_grid, cheb_nodes, noise_time_weights, nodes_weighted_error, seg_count_for, seg_time_grid, segment_model_error, snr_time_weights, tail_errors, vol_energy, weighted_spectrum};
pub use segjoint::{build_joint_basis, phi_gemm_default, JointBasis, JointBuildOpts, JointError, JointSegNufft};
pub use segsplit::{plan_batch, BatchPlan, SegSplit, SegSplitSegments};
pub use segwarp::{cardinal_form, SegWarp, WarpStencil};
pub use segnufft::{balanced_batch, fft_store_auto, fft_store_default, fuse_padcrop_default, grid_store_default, set_thread_fft_store, set_thread_grid_store, set_thread_map_stores, kprune_default, zero_tiles_default, SegSink, SegSource, seg_kernel_default, set_thread_cb_cap, set_thread_static_knot, static_knot_default, StaticKnot, set_thread_seg_kernel, SegKernel, tune_cache_forget, tune_mode, tune_timing_stats, TuneMode, build_segments, build_segments_complex, build_segments_fat, build_segments_svd, build_segments_svd_weighted, fold_fat_fraction, hi_exponents, multi_plan, multi_plan_with, seg_slot_bytes, tile_lists, SegFatMaps, SegNufft, SegNufftOpts, SegPoints, SegSegments, TileCfg, TileStats, RaCfg, RaVar, KernRes, VsCfg, VsGeom, VwCfg, MULTI_KMAX, NHI};
