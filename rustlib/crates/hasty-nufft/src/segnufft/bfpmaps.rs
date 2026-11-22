//! Block-floating-point copies of the image-side maps of the gradient (research/bfp_storage.md §10): the coil maps
//! S_c (`HASTY_SEG_MAPS_STORE`) and the segment diagonals P_l (`HASTY_SEG_DIAG_STORE`, whatever basis produced them:
//! the gradient pass only sees the maps). A copy is the dense `hasty_bfp` layout of the flat complex array (blocks of
//! 16 consecutive complex values, b mantissa words per block, one exponent byte per block in a separate array),
//! packed on the device from the f32 array by the warp codec (bit-identical to the host codec, hasty-bfp tests).
//! The fused pad / crop hooks (`fused.rs`) decode it where their staging loops load the rows (contiguous loads; the
//! same decoded values on both sides, so pad and crop stay an exact adjoint pair). Every other consumer (separate pad /
//! crop kernels, water + fat, warp, multi-image) keeps reading the f32 array, which is kept.

use super::*;

/// a dense BFP copy of a complex device array
pub(crate) struct BfpDev {
    pub bits: u32,
    pub mant: CudaSlice<u32>,
    pub ex: CudaSlice<u8>,
    /// complex values encoded
    #[allow(dead_code)]
    pub len: usize,
}

impl BfpDev {
    pub fn bytes(&self) -> usize {
        self.mant.len() * 4 + self.ex.len()
    }
}

const PACK_SRC: &str = r#"
extern "C" __global__ void bfp_pack_dense(const float* __restrict__ x, long long n, unsigned* __restrict__ mant, unsigned char* __restrict__ ex) {
  const long long w0 = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5, nw = ((long long)gridDim.x * blockDim.x) >> 5;
  const int lane = threadIdx.x & 31;
  for (long long blk = w0; blk * 32 < n; blk += nw) {
    const long long i = blk * 32 + lane;
    bfp_warp_store<MB>(i < n ? x[i] : 0.f, mant, ex, blk);
  }
}
"#;

/// storage of a map family from env `key` = f32 (default) | bfp16 … bfp24
pub(crate) fn map_store_env(key: &str) -> Store {
    match std::env::var(key).as_deref().map(str::trim) {
        Err(_) | Ok("") | Ok("f32") => Store::F32,
        Ok(v) => match Store::parse(v) {
            Some(s) => s,
            None => panic!("{key}={v}: f32 or bfp16 … bfp24"),
        },
    }
}

thread_local! {
    static MAP_STORES: std::cell::Cell<Option<(Store, Store)>> = const { std::cell::Cell::new(None) };
}

/// (coil maps, segment diagonals) storage for the engines built on this thread from now on; None: the env rule
/// (HASTY_SEG_MAPS_STORE, HASTY_SEG_DIAG_STORE). For tests and comparisons in one process.
pub fn set_thread_map_stores(s: Option<(Store, Store)>) {
    MAP_STORES.with(|c| c.set(s));
}

/// (coil maps, segment diagonals) storage default: [`set_thread_map_stores`], else env
pub(crate) fn map_stores_default() -> (Store, Store) {
    MAP_STORES.with(|c| c.get()).unwrap_or_else(|| (map_store_env("HASTY_SEG_MAPS_STORE"), map_store_env("HASTY_SEG_DIAG_STORE")))
}

/// dense BFP copy of `x` (complex) at `store` (None for f32)
pub(crate) fn pack_dense(dev: &Device, st: &Arc<CudaStream>, x: &CudaSlice<C32>, store: Store) -> Result<Option<BfpDev>, NufftError> {
    let bits = match store {
        Store::F32 => return Ok(None),
        Store::Bfp(b) => b,
    };
    let len = x.len();
    let nblk = len.div_ceil(16);
    let mut mant = st.alloc_zeros::<u32>((nblk * bits as usize).max(1))?;
    let mut ex = st.alloc_zeros::<u8>(nblk.max(1))?;
    // no --use_fast_math: the codec keeps subnormal scale factors (bit-identical to the host codec)
    let src = format!("#define MB {bits}\n{}\n{PACK_SRC}", hasty_bfp::BFP_CUH);
    let f = dev.load_function(&format!("bfp_pack_dense_b{bits}"), &src, "bfp_pack_dense", &CompileSettings::default())?;
    let n = (2 * len) as i64;
    let warps = nblk.min(1 << 16) as u32;
    let cfg = LaunchConfig { grid_dim: (warps.div_ceil(8).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
    unsafe {
        st.launch_builder(&f).arg(x).arg(&n).arg(&mut mant).arg(&mut ex).launch(cfg)?;
    }
    Ok(Some(BfpDev { bits, mant, ex, len }))
}
