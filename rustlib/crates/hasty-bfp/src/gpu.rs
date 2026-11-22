//! Device-side BFP kernels for tests and benchmarks (`kernels.cu` on top of `bfp.cuh`), one module per mantissa width.
//! Production kernels include [`crate::BFP_CUH`] themselves and call the codec from their own loads and stores.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device};

pub const KERNELS_CU: &str = include_str!("kernels.cu");

/// Thread layout of a pack / unpack kernel (all write the same bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// lane j = value j of a block, lane j < bits holds word j
    Warp,
    /// lane t of a half warp = complex t of a block (the strided FFT passes' 16 lines)
    HalfWarpComplex,
    /// a thread owns a run of 32 / gcd(bits, 32) values in whole words
    Run,
}

impl Layout {
    pub const ALL: [Layout; 3] = [Layout::Warp, Layout::HalfWarpComplex, Layout::Run];
    pub fn name(self) -> &'static str {
        match self {
            Layout::Warp => "warp",
            Layout::HalfWarpComplex => "hw-complex",
            Layout::Run => "run",
        }
    }
}

pub struct BfpKernels {
    pub bits: u32,
    pub pack: [CudaFunction; 3],
    pub unpack: [CudaFunction; 3],
    /// random-access decode, one complex / one real value per thread
    pub unpack_rand: [CudaFunction; 2],
    pub copy_f32: CudaFunction,
    pub copy_u32: CudaFunction,
    blocks: u32,
}

fn idx(l: Layout) -> usize {
    match l {
        Layout::Warp => 0,
        Layout::HalfWarpComplex => 1,
        Layout::Run => 2,
    }
}

impl BfpKernels {
    pub fn new(dev: &Device, bits: u32) -> Result<Self, hasty_cuda::Error> {
        assert!((8..=24).contains(&bits));
        let src = format!("{}\n{}", crate::BFP_CUH, KERNELS_CU);
        let set = CompileSettings { options: vec![format!("-DBFP_B={bits}")], no_cache: false };
        let m = dev.load_module(&format!("bfp_b{bits}"), &src, &set)?;
        let f = |n: &str| m.load_function(n).map_err(hasty_cuda::Error::from);
        let sms = dev.ctx.attribute(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as u32;
        Ok(Self {
            bits,
            pack: [f("bfp_pack_warp")?, f("bfp_pack_hw")?, f("bfp_pack_run")?],
            unpack: [f("bfp_unpack_warp")?, f("bfp_unpack_hw")?, f("bfp_unpack_run")?],
            unpack_rand: [f("bfp_unpack_rand")?, f("bfp_unpack_rand_r")?],
            copy_f32: f("copy_f32")?,
            copy_u32: f("copy_u32")?,
            blocks: sms * 8,
        })
    }

    /// grid: `blocks_per_sm` × SMs blocks of 256 threads (grid-stride loops)
    pub fn set_blocks(&mut self, blocks: u32) {
        self.blocks = blocks.max(1);
    }

    fn cfg(&self, work_threads: usize) -> LaunchConfig {
        let need = work_threads.div_ceil(256).max(1) as u32;
        LaunchConfig { grid_dim: (need.min(self.blocks), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
    }

    fn threads(&self, l: Layout, n: usize) -> usize {
        let nblk = crate::Bfp::blocks(n);
        match l {
            Layout::Warp | Layout::HalfWarpComplex => nblk.div_ceil(4) * 32,
            Layout::Run => nblk * (32 / crate::Bfp::new(self.bits).run_values()),
        }
    }

    /// pack `n` reals of `x` into `mant` / `ex` (n even for [`Layout::HalfWarpComplex`])
    pub fn pack(&self, st: &Arc<CudaStream>, l: Layout, x: &CudaSlice<f32>, n: usize, mant: &mut CudaSlice<u32>, ex: &mut CudaSlice<u8>) -> Result<(), hasty_cuda::Error> {
        assert!(x.len() >= n && mant.len() >= crate::Bfp::new(self.bits).mant_words(n) && ex.len() >= crate::Bfp::blocks(n).div_ceil(4) * 4);
        assert!(l != Layout::HalfWarpComplex || n % 2 == 0);
        let nl = n as i64;
        let cfg = self.cfg(self.threads(l, n));
        unsafe { st.launch_builder(&self.pack[idx(l)]).arg(x).arg(mant).arg(ex).arg(&nl).launch(cfg)? };
        Ok(())
    }

    pub fn unpack(&self, st: &Arc<CudaStream>, l: Layout, mant: &CudaSlice<u32>, ex: &CudaSlice<u8>, n: usize, out: &mut CudaSlice<f32>) -> Result<(), hasty_cuda::Error> {
        assert!(out.len() >= n && mant.len() >= crate::Bfp::new(self.bits).mant_words(n) && ex.len() >= crate::Bfp::blocks(n).div_ceil(4) * 4);
        assert!(l != Layout::HalfWarpComplex || n % 2 == 0);
        let nl = n as i64;
        let cfg = self.cfg(self.threads(l, n));
        unsafe { st.launch_builder(&self.unpack[idx(l)]).arg(mant).arg(ex).arg(out).arg(&nl).launch(cfg)? };
        Ok(())
    }

    /// random-access decode of `n` reals: one complex value per thread (`complex`, n even) or one real per thread
    pub fn unpack_random(&self, st: &Arc<CudaStream>, complex: bool, mant: &CudaSlice<u32>, ex: &CudaSlice<u8>, n: usize, out: &mut CudaSlice<f32>) -> Result<(), hasty_cuda::Error> {
        assert!(out.len() >= n && mant.len() >= crate::Bfp::new(self.bits).mant_words(n) && ex.len() >= crate::Bfp::blocks(n));
        assert!(!complex || n % 2 == 0);
        let nl = n as i64;
        let cfg = self.cfg(if complex { n / 2 } else { n });
        unsafe { st.launch_builder(&self.unpack_rand[if complex { 0 } else { 1 }]).arg(mant).arg(ex).arg(out).arg(&nl).launch(cfg)? };
        Ok(())
    }

    /// y = x for `n` floats (n % 4 == 0)
    pub fn copy_f32(&self, st: &Arc<CudaStream>, x: &CudaSlice<f32>, y: &mut CudaSlice<f32>, n: usize) -> Result<(), hasty_cuda::Error> {
        assert!(n % 4 == 0 && x.len() >= n && y.len() >= n);
        let n4 = (n / 4) as i64;
        let cfg = self.cfg(n / 4);
        unsafe { st.launch_builder(&self.copy_f32).arg(x).arg(y).arg(&n4).launch(cfg)? };
        Ok(())
    }

    pub fn copy_u32(&self, st: &Arc<CudaStream>, x: &CudaSlice<u32>, y: &mut CudaSlice<u32>, n: usize) -> Result<(), hasty_cuda::Error> {
        let nl = n as i64;
        let cfg = self.cfg(n);
        unsafe { st.launch_builder(&self.copy_u32).arg(x).arg(y).arg(&nl).launch(cfg)? };
        Ok(())
    }
}
