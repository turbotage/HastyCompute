//! GPU locally-low-rank denoiser: the operator of [`crate::llr::Llr`] (same blocks, shift sequence, σ̂ rule and
//! singular-value filter), with the work spread over one or more devices.
//!
//! Per block, Casorati C (R × T, R = B³ rows zero-padded, column-major; zero rows leave CᴴC unchanged):
//! - pass 1, f64 as on the CPU (an f32 Gram squares the dynamic range: measured 100 % error on the noise singular
//!   values of high-SNR blocks): G = CᴴC (cuBLAS Zgemm, strided batched), G = V Λ Vᴴ (cuSOLVER XsyevBatched);
//!   s = √max(λ, 0) per block, V as f32 kept on the host (as the CPU path keeps it);
//! - σ̂ over all blocks of all devices (the only cross-device step), threshold κσ̂;
//! - pass 2, f32: with the k largest eigenpairs (k = the most above threshold in the batch, the rest weighted 0),
//!   C ← (C V_k) diag(w) V_kᴴ, w = shrink(s)/s (two Cgemm); written back to the stack.
//!
//! Multi-GPU: the blocks are split into contiguous ranges, one per device, each driven by its own host thread.
//! Batches are sized to a per-device memory budget, since the solver keeps its own buffers on the device.

use std::sync::Arc;
use std::time::Instant;

use cudarc::cublas::{sys as cb, CudaBlas};
use cudarc::cusolver::{sys as cs, DnHandle};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, C32};
use rayon::prelude::*;

use crate::llr::Llr;
use crate::Error;

struct Slot {
    dev: Device,
    st: Arc<CudaStream>,
    blas: CudaBlas,
    dn: DnHandle,
    c2z: CudaFunction,
    top: CudaFunction,
    scale: CudaFunction,
}

#[derive(Clone, Debug, Default)]
pub struct LlrGpuStats {
    /// decomposed blocks, blocks left unchanged (eigensolver failure), the largest k of pass 2
    pub blocks: usize,
    pub skipped: usize,
    pub k_max: usize,
    /// seconds: pass 1, pass 2, total
    pub secs: [f64; 3],
}

pub struct LlrGpu {
    pub llr: Llr,
    slots: Vec<Slot>,
    /// device bytes one batch may use on each device; 0 = 60 % of that device's free memory at the call
    pub budget: usize,
    pub last: LlrGpuStats,
}

fn cfg(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

fn blas_ok(s: cb::cublasStatus_t, what: &str) -> Result<(), Error> {
    if s == cb::cublasStatus_t::CUBLAS_STATUS_SUCCESS { Ok(()) } else { Err(Error::Format(format!("{what}: {s:?}"))) }
}

fn solver_ok(s: cs::cusolverStatus_t, what: &str) -> Result<(), Error> {
    if s == cs::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS { Ok(()) } else { Err(Error::Format(format!("{what}: {s:?}"))) }
}

/// a device allocation failed (the caller may retry with a smaller batch)
fn is_oom(e: &Error) -> bool {
    matches!(e, Error::Driver(d) if d.0 == cudarc::driver::sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY)
}

/// `alloc(nbat)`, halving the batch while the device is out of memory (other users of the device, or memory the
/// solver allocated since the budget was read); nothing is written before the buffers exist, so a retry is safe
fn alloc_batch<T>(mut nbat: usize, what: &str, mut alloc: impl FnMut(usize) -> Result<T, Error>) -> Result<(usize, T), Error> {
    loop {
        match alloc(nbat) {
            Ok(b) => return Ok((nbat, b)),
            Err(e) if is_oom(&e) && nbat > 1 => {
                eprintln!("    LLR on GPU: {what}: out of device memory at a batch of {nbat} blocks, retrying with {}", nbat / 2);
                nbat /= 2;
            }
            Err(e) => return Err(e),
        }
    }
}

/// block j of a batch buffer (R × T column-major per block) ← the stack's voxels of `idx` (zero-padded rows)
fn gather(x: &[C32], nv: usize, nt: usize, r: usize, blocks: &[Vec<usize>], out: &mut [C32]) {
    out[..blocks.len() * r * nt].par_chunks_mut(r * nt).zip(blocks.par_iter()).for_each(|(o, idx)| {
        for t in 0..nt {
            let col = &mut o[r * t..r * (t + 1)];
            let xt = &x[t * nv..(t + 1) * nv];
            for (row, &q) in idx.iter().enumerate() {
                col[row] = xt[q];
            }
            col[idx.len()..].fill(C32::new(0.0, 0.0));
        }
    });
}

impl Slot {
    fn new(ordinal: usize) -> Result<Self, Error> {
        let dev = Device::new(ordinal)?;
        let st = dev.ctx.new_stream()?;
        let blas = CudaBlas::new(st.clone()).map_err(|e| Error::Format(format!("cuBLAS: {e:?}")))?;
        let dn = DnHandle::new(st.clone()).map_err(|e| Error::Format(format!("cuSOLVER: {e:?}")))?;
        let m = dev.load_module("llr_gpu", include_str!("llr_gpu.cu"), &CompileSettings::default())?;
        Ok(Self { c2z: m.load_function("c2z")?, top: m.load_function("top_cols")?, scale: m.load_function("scale_cols")?, dev, st, blas, dn })
    }

    /// wait for this slot's frees, then return the device pool's unused memory
    fn release(&self) -> Result<(), Error> {
        self.dev.ctx.bind_to_thread()?;
        self.st.synchronize()?;
        unsafe {
            let pool = cudarc::driver::result::device::get_mem_pool(self.dev.ctx.cu_device())?;
            cudarc::driver::result::mem_pool::trim_to(pool, 0)?;
        }
        Ok(())
    }

    fn budget(&self, budget: usize) -> usize {
        if budget > 0 {
            return budget;
        }
        self.dev.ctx.mem_get_info().map(|(f, _)| f / 10 * 6).unwrap_or(1 << 30)
    }

    /// XsyevBatched f64 workspace (device, host bytes) for `nb` T × T matrices
    fn eig_workspace(&self, params: cs::cusolverDnParams_t, g: &CudaSlice<f64>, w: &CudaSlice<f64>, nt: usize, nb: usize) -> Result<(usize, usize), Error> {
        let (mut dws, mut hws) = (0usize, 0usize);
        let (gp, _g0) = g.device_ptr(&self.st);
        let (wp, _g1) = w.device_ptr(&self.st);
        let s = unsafe {
            cs::cusolverDnXsyevBatched_bufferSize(self.dn.cu(), params, cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, cs::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, nt as i64, cs::cudaDataType::CUDA_C_64F, gp as *const _, nt as i64, cs::cudaDataType::CUDA_R_64F, wp as *const _, cs::cudaDataType::CUDA_C_64F, &mut dws, &mut hws, nb as i64)
        };
        solver_ok(s, "XsyevBatched_bufferSize")?;
        Ok((dws, hws))
    }

    /// Pass 1 over `blocks`: singular values per block (empty = eigensolver failure, block left unchanged) and the
    /// eigenvectors (f32, T × T column-major per block, ascending eigenvalues)
    fn pass1(&self, x: &[C32], nv: usize, nt: usize, r: usize, blocks: &[Vec<usize>], budget: usize) -> Result<(Vec<Vec<f32>>, Vec<C32>), Error> {
        self.dev.ctx.bind_to_thread()?;
        let st = &self.st;
        let n = blocks.len();
        let mut params: cs::cusolverDnParams_t = std::ptr::null_mut();
        solver_ok(unsafe { cs::cusolverDnCreateParams(&mut params) }, "cusolverDnCreateParams")?;
        // batch size from the budget: buffers per block + the eigensolver's workspace per block (queried at 64)
        let per = r * nt * (8 + 16) + nt * nt * (16 + 8) + nt * 8;
        let probe = 64.min(n.max(1));
        let ws_per = {
            let g = st.alloc_zeros::<f64>(2 * nt * nt * probe)?;
            let w = st.alloc_zeros::<f64>(nt * probe)?;
            let (d, _) = self.eig_workspace(params, &g, &w, nt, probe)?;
            d.div_ceil(probe)
        };
        let nbat0 = (self.budget(budget) / (per + ws_per)).clamp(1, n.max(1));
        let (nbat, bufs) = alloc_batch(nbat0, "pass 1", |nbat| {
            let dc = st.alloc_zeros::<C32>(nbat * r * nt)?;
            let dz = st.alloc_zeros::<f64>(2 * nbat * r * nt)?;
            let dg = st.alloc_zeros::<f64>(2 * nbat * nt * nt)?;
            let dw = st.alloc_zeros::<f64>(nbat * nt)?;
            let dv = st.alloc_zeros::<C32>(nbat * nt * nt)?;
            let dinfo = st.alloc_zeros::<i32>(nbat)?;
            let (dws, hws) = self.eig_workspace(params, &dg, &dw, nt, nbat)?;
            let dbuf = st.alloc_zeros::<u8>(dws.max(1))?;
            let hc = unsafe { self.dev.ctx.alloc_pinned_with_flags::<C32>(nbat * r * nt, 0)? };
            Ok((dc, dz, dg, dw, dv, dinfo, dws, hws, dbuf, hc))
        })?;
        let (mut dc, mut dz, mut dg, mut dw, mut dv, mut dinfo, dws, hws, mut dbuf, mut hc) = bufs;
        let mut hbuf = vec![0u8; hws.max(1)];
        let mut sv: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut vs = vec![C32::new(0.0, 0.0); n * nt * nt];
        for b0 in (0..n).step_by(nbat) {
            let bl = &blocks[b0..(b0 + nbat).min(n)];
            let nb = bl.len();
            gather(x, nv, nt, r, bl, hc.as_mut_slice()?);
            st.memcpy_htod(&hc.as_slice()?[..nb * r * nt], &mut dc.slice_mut(0..nb * r * nt))?;
            let ne = (nb * r * nt) as i64;
            unsafe { st.launch_builder(&self.c2z).arg(&dc).arg(&mut dz).arg(&ne).launch(cfg(nb * r * nt))? };
            unsafe {
                let one = cb::cuDoubleComplex { x: 1.0, y: 0.0 };
                let zero = cb::cuDoubleComplex { x: 0.0, y: 0.0 };
                let (zp, _g0) = dz.device_ptr(st);
                let (gp, _g1) = dg.device_ptr_mut(st);
                blas_ok(
                    cb::cublasZgemmStridedBatched(*self.blas.handle(), cb::cublasOperation_t::CUBLAS_OP_C, cb::cublasOperation_t::CUBLAS_OP_N, nt as i32, nt as i32, r as i32, &one, zp as *const _, r as i32, (r * nt) as i64, zp as *const _, r as i32, (r * nt) as i64, &zero, gp as *mut _, nt as i32, (nt * nt) as i64, nb as i32),
                    "Zgemm (Gram)",
                )?;
            }
            unsafe {
                let (gp, _g1) = dg.device_ptr_mut(st);
                let (wp, _g2) = dw.device_ptr_mut(st);
                let (ip, _g3) = dinfo.device_ptr_mut(st);
                let (bp, _g4) = dbuf.device_ptr_mut(st);
                solver_ok(
                    cs::cusolverDnXsyevBatched(self.dn.cu(), params, cs::cusolverEigMode_t::CUSOLVER_EIG_MODE_VECTOR, cs::cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, nt as i64, cs::cudaDataType::CUDA_C_64F, gp as *mut _, nt as i64, cs::cudaDataType::CUDA_R_64F, wp as *mut _, cs::cudaDataType::CUDA_C_64F, bp as *mut _, dws, hbuf.as_mut_ptr() as *mut _, hws, ip as *mut _, nb as i64),
                    "XsyevBatched",
                )?;
            }
            let (ti, ki, nbi) = (nt as i32, nt as i32, nb as i64);
            unsafe { st.launch_builder(&self.top).arg(&dg).arg(&mut dv).arg(&ti).arg(&ki).arg(&nbi).launch(cfg(nb * nt * nt))? };
            let info = st.clone_dtoh(&dinfo.slice(0..nb))?;
            let lam = st.clone_dtoh(&dw.slice(0..nb * nt))?;
            st.memcpy_dtoh(&dv.slice(0..nb * nt * nt), &mut vs[b0 * nt * nt..(b0 + nb) * nt * nt])?;
            st.synchronize()?;
            for j in 0..nb {
                let l = &lam[j * nt..(j + 1) * nt];
                let ok = info[j] == 0 && l.iter().all(|v| v.is_finite()) && vs[(b0 + j) * nt * nt..(b0 + j + 1) * nt * nt].iter().all(|c| c.re.is_finite() && c.im.is_finite());
                sv.push(if ok { l.iter().map(|&v| v.max(0.0).sqrt() as f32).collect() } else { vec![] });
            }
        }
        unsafe { cs::cusolverDnDestroyParams(params) };
        Ok((sv, vs))
    }

    /// Pass 2 over `blocks` (singular values `sv`, eigenvectors `vs` of pass 1), written into the stack at `xp`
    #[allow(clippy::too_many_arguments)]
    fn pass2(&self, xp: usize, xlen: usize, nv: usize, nt: usize, r: usize, blocks: &[Vec<usize>], sv: &[Vec<f32>], vs: &[C32], thr: f32, rank: Option<usize>, hard: bool, budget: usize) -> Result<usize, Error> {
        self.dev.ctx.bind_to_thread()?;
        let st = &self.st;
        let n = blocks.len();
        // SAFETY: the blocks of all devices are disjoint; x is only read here at this pass's own blocks
        let x: &mut [C32] = unsafe { std::slice::from_raw_parts_mut(xp as *mut C32, xlen) };
        // per-block thresholds (κσ̂, raised by a rank cap)
        let tbs: Vec<f32> = sv.iter().map(|s| Llr::block_threshold(rank, s, thr)).collect();
        let kb: Vec<usize> = sv.iter().zip(&tbs).map(|(s, &tb)| s.iter().filter(|&&v| Llr::weight_rule(v, tb, hard) > 0.0).count()).collect();
        let kmax = kb.iter().copied().max().unwrap_or(0);
        let mut k_used = 0;
        let per = r * nt * 8 + (nt + r) * kmax.max(1) * 8 + kmax.max(1) * 4;
        let nbat0 = (self.budget(budget) / per).clamp(1, n.max(1));
        let (nbat, bufs) = alloc_batch(nbat0, "pass 2", |nbat| {
            let dc = st.alloc_zeros::<C32>(nbat * r * nt)?;
            let dvk = st.alloc_zeros::<C32>(nbat * nt * kmax.max(1))?;
            let dp = st.alloc_zeros::<C32>(nbat * r * kmax.max(1))?;
            let dwk = st.alloc_zeros::<f32>(nbat * kmax.max(1))?;
            let hc = unsafe { self.dev.ctx.alloc_pinned_with_flags::<C32>(nbat * r * nt, 0)? };
            Ok((dc, dvk, dp, dwk, hc))
        })?;
        let (mut dc, mut dvk, mut dp, mut dwk, mut hc) = bufs;
        let mut hv = vec![C32::new(0.0, 0.0); nbat * nt * kmax.max(1)];
        let mut hw = vec![0f32; nbat * kmax.max(1)];
        for b0 in (0..n).step_by(nbat) {
            let b1 = (b0 + nbat).min(n);
            let bl = &blocks[b0..b1];
            let nb = bl.len();
            let k = (b0..b1).filter(|&j| !sv[j].is_empty()).map(|j| kb[j]).max().unwrap_or(0);
            if k == 0 {
                // every eigenpair at or below the threshold: the decomposed blocks become zero (F = 0, as on the CPU)
                for (j, idx) in bl.iter().enumerate() {
                    if !sv[b0 + j].is_empty() {
                        for t in 0..nt {
                            for &q in idx {
                                x[t * nv + q] = C32::new(0.0, 0.0);
                            }
                        }
                    }
                }
                continue;
            }
            k_used = k_used.max(k);
            gather(x, nv, nt, r, bl, hc.as_mut_slice()?);
            // the last k eigenvectors (ascending eigenvalues) and their weights; failed blocks: zero (not written back)
            for j in 0..nb {
                let (v, w) = (&mut hv[j * nt * k..(j + 1) * nt * k], &mut hw[j * k..(j + 1) * k]);
                if sv[b0 + j].is_empty() {
                    v.fill(C32::new(0.0, 0.0));
                    w.fill(0.0);
                    continue;
                }
                v.copy_from_slice(&vs[(b0 + j) * nt * nt + (nt - k) * nt..(b0 + j + 1) * nt * nt]);
                for c in 0..k {
                    w[c] = Llr::weight_rule(sv[b0 + j][nt - k + c], tbs[b0 + j], hard);
                }
            }
            st.memcpy_htod(&hc.as_slice()?[..nb * r * nt], &mut dc.slice_mut(0..nb * r * nt))?;
            st.memcpy_htod(&hv[..nb * nt * k], &mut dvk.slice_mut(0..nb * nt * k))?;
            st.memcpy_htod(&hw[..nb * k], &mut dwk.slice_mut(0..nb * k))?;
            unsafe {
                let one = cb::cuComplex { x: 1.0, y: 0.0 };
                let zero = cb::cuComplex { x: 0.0, y: 0.0 };
                let (cp, _g0) = dc.device_ptr_mut(st);
                let (vp, _g1) = dvk.device_ptr(st);
                let (pp, _g2) = dp.device_ptr_mut(st);
                // P = C V_k (R × k)
                blas_ok(
                    cb::cublasCgemmStridedBatched(*self.blas.handle(), cb::cublasOperation_t::CUBLAS_OP_N, cb::cublasOperation_t::CUBLAS_OP_N, r as i32, k as i32, nt as i32, &one, cp as *const _, r as i32, (r * nt) as i64, vp as *const _, nt as i32, (nt * k) as i64, &zero, pp as *mut _, r as i32, (r * k) as i64, nb as i32),
                    "Cgemm (C V_k)",
                )?;
                drop((_g0, _g2));
                let (ri, ki, nbi) = (r as i32, k as i32, nb as i64);
                st.launch_builder(&self.scale).arg(&mut dp).arg(&dwk).arg(&ri).arg(&ki).arg(&nbi).launch(cfg(nb * r * k))?;
                let (cp, _g0) = dc.device_ptr_mut(st);
                let (pp, _g2) = dp.device_ptr(st);
                // C ← P V_kᴴ (R × T), into C's buffer (C is consumed by the first product)
                blas_ok(
                    cb::cublasCgemmStridedBatched(*self.blas.handle(), cb::cublasOperation_t::CUBLAS_OP_N, cb::cublasOperation_t::CUBLAS_OP_C, r as i32, nt as i32, k as i32, &one, pp as *const _, r as i32, (r * k) as i64, vp as *const _, nt as i32, (nt * k) as i64, &zero, cp as *mut _, r as i32, (r * nt) as i64, nb as i32),
                    "Cgemm (P V_kᴴ)",
                )?;
            }
            st.memcpy_dtoh(&dc.slice(0..nb * r * nt), &mut hc.as_mut_slice()?[..nb * r * nt])?;
            st.synchronize()?;
            let hs = hc.as_slice()?;
            let xa = x.as_mut_ptr() as usize;
            bl.par_iter().enumerate().for_each(|(j, idx)| {
                if sv[b0 + j].is_empty() {
                    return;
                }
                let o = &hs[j * r * nt..(j + 1) * r * nt];
                let d = xa as *mut C32;
                for t in 0..nt {
                    for (row, &q) in idx.iter().enumerate() {
                        // SAFETY: blocks are disjoint
                        unsafe { *d.add(t * nv + q) = o[row + r * t] };
                    }
                }
            });
        }
        Ok(k_used)
    }
}

impl LlrGpu {
    /// `llr`: the operator (blocks, κ, support, shift seed); `devices`: CUDA ordinals sharing the work (a device may
    /// appear once; the solver's own device is the usual single entry)
    pub fn new(llr: Llr, devices: &[usize]) -> Result<Self, Error> {
        let slots = devices.iter().map(|&o| Slot::new(o)).collect::<Result<Vec<_>, _>>()?;
        if slots.is_empty() {
            return Err(Error::Format("LlrGpu: no device".into()));
        }
        Ok(Self { llr, slots, budget: 0, last: LlrGpuStats::default() })
    }

    pub fn set_iteration(&mut self, k: u64) {
        self.llr.set_iteration(k);
    }

    pub fn devices(&self) -> Vec<usize> {
        self.slots.iter().map(|s| s.dev.ordinal).collect()
    }

    /// in place on `x` laid out [t][v] (t = 0..nt), as [`Llr::apply`]
    pub fn apply(&mut self, x: &mut [C32], nt: usize) -> Result<(), Error> {
        let t0 = Instant::now();
        let nv = self.llr.dims.iter().product::<usize>();
        let r = self.llr.block.pow(3);
        let blocks = self.llr.next_blocks();
        if blocks.is_empty() {
            return Ok(());
        }
        let nd = self.slots.len();
        let chunk = blocks.len().div_ceil(nd);
        let parts: Vec<&[Vec<usize>]> = blocks.chunks(chunk).collect();
        let budget = self.budget;
        let xs: &[C32] = x;
        // pass 1 on every device
        let res: Vec<Result<(Vec<Vec<f32>>, Vec<C32>), Error>> = std::thread::scope(|sc| {
            let hs: Vec<_> = parts.iter().zip(&self.slots).map(|(bl, slot)| sc.spawn(move || slot.pass1(xs, nv, nt, r, bl, budget))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let res: Vec<(Vec<Vec<f32>>, Vec<C32>)> = res.into_iter().collect::<Result<_, _>>()?;
        let t1 = t0.elapsed().as_secs_f64();
        let skipped = res.iter().map(|(sv, _)| sv.iter().filter(|s| s.is_empty()).count()).sum();
        let Some(sigma) = Llr::sigma_hat(res.iter().flat_map(|(sv, _)| sv.iter().map(|s| s.as_slice()))) else {
            return Ok(());
        };
        let thr = (self.llr.kappa * sigma) as f32;
        let (rank, hard) = (self.llr.rank, self.llr.hard);
        // pass 2 on every device, written back in place (disjoint blocks)
        let (xp, xlen) = (x.as_mut_ptr() as usize, x.len());
        let ks: Vec<Result<usize, Error>> = std::thread::scope(|sc| {
            let hs: Vec<_> = parts.iter().zip(&self.slots).zip(&res).map(|((bl, slot), (sv, vs))| sc.spawn(move || slot.pass2(xp, xlen, nv, nt, r, bl, sv, vs, thr, rank, hard, budget))).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let k_max = ks.into_iter().collect::<Result<Vec<_>, _>>()?.into_iter().max().unwrap_or(0);
        // the batch buffers went back to the stream-ordered pool: release them to the device, so the solver (other
        // streams) gets the memory back even before its next synchronisation
        for slot in &self.slots {
            slot.release()?;
        }
        let t2 = t0.elapsed().as_secs_f64();
        self.last = LlrGpuStats { blocks: blocks.len(), skipped, k_max, secs: [t1, t2 - t1, t2] };
        Ok(())
    }
}
