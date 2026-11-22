//! Production transfer kernels (`xfer.cu`): BFP pack on the device before a device → host copy, and decode on the
//! device after a host → device copy. PCIe is ≈ 40 × slower than DRAM, so this standalone pass costs ≈ 1/40 of the
//! transfer time it saves (research/bfp_storage.md §5). The packed bytes equal the host codec's
//! ([`crate::Bfp::encode_serial`] per row) bit for bit.
//!
//! Layout: `rows` rows of `n` reals (row r at offset r · n); each row is its own block sequence of
//! [`Bfp::blocks`](crate::Bfp::blocks)(n) blocks, rows back to back in the mantissa words and the exponent bytes
//! ([`rows_words`], [`rows_ex`]). One row: a gradient chunk; several rows: the coil rows of a k-space set.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device};

use crate::Bfp;

pub const XFER_CU: &str = include_str!("xfer.cu");

/// mantissa words of `rows` rows of `n` reals
pub fn rows_words(fmt: Bfp, n: usize, rows: usize) -> usize {
    rows * fmt.mant_words(n)
}

/// exponent bytes of `rows` rows of `n` reals, padded to whole 32-bit words
pub fn rows_ex(n: usize, rows: usize) -> usize {
    (rows * Bfp::blocks(n)).div_ceil(4) * 4
}

/// The pack / unpack kernels of one width.
pub struct XferKernels {
    pub fmt: Bfp,
    pack: CudaFunction,
    unpack: CudaFunction,
    blocks: u32,
}

impl XferKernels {
    pub fn new(ctx: &Arc<CudaContext>, fmt: Bfp) -> Result<Self, hasty_cuda::Error> {
        let (major, minor) = ctx.compute_capability()?;
        let dev = Device { ctx: ctx.clone(), ordinal: ctx.ordinal(), arch: format!("sm_{major}{minor}") };
        let src = format!("{}\n{}", crate::BFP_CUH, XFER_CU);
        let set = CompileSettings { options: vec![format!("-DBFP_B={}", fmt.bits())], no_cache: false };
        let m = dev.load_module(&format!("bfp_xfer_b{}", fmt.bits()), &src, &set)?;
        let sms = ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        Ok(Self { fmt, pack: m.load_function("bfp_xfer_pack")?, unpack: m.load_function("bfp_xfer_unpack")?, blocks: sms * 8 })
    }

    fn cfg(&self, n: usize, rows: usize) -> LaunchConfig {
        let runs = rows * Bfp::blocks(n) * (32 / self.fmt.run_values());
        let need = runs.div_ceil(256).max(1) as u32;
        LaunchConfig { grid_dim: (need.min(self.blocks), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
    }

    /// Pack `rows` rows of `n` reals of `x` into `mant` / `ex`. `x` holds f32 values (or complex f32: `n` counts
    /// reals either way).
    pub fn pack<T: DeviceRepr>(&self, st: &Arc<CudaStream>, x: &CudaSlice<T>, n: usize, rows: usize, mant: &mut CudaSlice<u32>, ex: &mut CudaSlice<u8>) -> Result<(), hasty_cuda::Error> {
        assert!(x.len() * std::mem::size_of::<T>() >= 4 * n * rows && mant.len() >= rows_words(self.fmt, n, rows) && ex.len() >= rows * Bfp::blocks(n), "bfp pack: buffers too small");
        if n * rows == 0 {
            return Ok(());
        }
        let (nl, rl) = (n as i64, rows as i64);
        unsafe { st.launch_builder(&self.pack).arg(x).arg(mant).arg(ex).arg(&nl).arg(&rl).launch(self.cfg(n, rows))? };
        Ok(())
    }

    /// Decode `rows` rows of `n` reals into `out` (row r at r · n; f32 or complex f32).
    pub fn unpack<T: DeviceRepr>(&self, st: &Arc<CudaStream>, mant: &CudaSlice<u32>, ex: &CudaSlice<u8>, n: usize, rows: usize, out: &mut CudaSlice<T>) -> Result<(), hasty_cuda::Error> {
        assert!(out.len() * std::mem::size_of::<T>() >= 4 * n * rows && mant.len() >= rows_words(self.fmt, n, rows) && ex.len() >= rows * Bfp::blocks(n), "bfp unpack: buffers too small");
        if n * rows == 0 {
            return Ok(());
        }
        let (nl, rl) = (n as i64, rows as i64);
        unsafe { st.launch_builder(&self.unpack).arg(mant).arg(ex).arg(out).arg(&nl).arg(&rl).launch(self.cfg(n, rows))? };
        Ok(())
    }
}

/// Kernels per (context, width), compiled on first use and shared (NVRTC results are also disk-cached).
pub fn kernels(ctx: &Arc<CudaContext>, fmt: Bfp) -> Result<Arc<XferKernels>, hasty_cuda::Error> {
    static CACHE: Mutex<Option<HashMap<(usize, usize, u32), Arc<XferKernels>>>> = Mutex::new(None);
    let key = (Arc::as_ptr(ctx) as usize, ctx.ordinal(), fmt.bits());
    let mut g = CACHE.lock().unwrap();
    let map = g.get_or_insert_with(HashMap::new);
    if let Some(k) = map.get(&key) {
        return Ok(k.clone());
    }
    let k = Arc::new(XferKernels::new(ctx, fmt)?);
    map.insert(key, k.clone());
    Ok(k)
}
