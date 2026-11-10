//! CUDA backend: `kernels/lbm_core.h` + `kernels/lbm_cuda.cu` through NVRTC (hasty-cuda's cached compile), one thread
//! per node, 256-thread blocks. Buffers mirror the CPU backend (SoA populations, two copies for the pull scheme).
//!
//! Memory ≈ 240 B per node (2 × 19 f32 populations, 2 × 4 f32 macroscopic fields, inverse map, distance, material /
//! solid velocity, links, quantised wall fractions) plus the accumulator (cells per tile × 10 f32) and the f16 motion
//! samples.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device};

use crate::backend::{Backend, Fields, Topology};
use crate::params::{GeomParams, MotionParams, SpinParams, StepParams, ACC_STRIDE, Q};
use crate::Result;

pub const CORE_SRC: &str = include_str!("../kernels/lbm_core.h");
pub const CUDA_SRC: &str = include_str!("../kernels/lbm_cuda.cu");

struct Kernels {
    motion: CudaFunction,
    geom1: CudaFunction,
    geom2: CudaFunction,
    refill: CudaFunction,
    commit: CudaFunction,
    step: CudaFunction,
    accum: CudaFunction,
    diag: CudaFunction,
    scalar: CudaFunction,
    srefill: CudaFunction,
}

pub struct GpuBackend {
    pub topo: Topology,
    st: Arc<CudaStream>,
    k: Kernels,
    n: usize,
    txyz: CudaSlice<i32>,
    tnbr: CudaSlice<i32>,
    f0: CudaSlice<f32>,
    f1: CudaSlice<f32>,
    ru0: CudaSlice<f32>,
    ru1: CudaSlice<f32>,
    cur: usize,
    xref: CudaSlice<f32>,
    phi: CudaSlice<f32>,
    um: CudaSlice<f32>,
    bsol: CudaSlice<f32>,
    bcid: CudaSlice<u8>,
    pid: CudaSlice<u8>,
    rid: CudaSlice<u8>,
    rmap: CudaSlice<u8>,
    flags: CudaSlice<u8>,
    links: CudaSlice<u32>,
    qlink: CudaSlice<u8>,
    spg: CudaSlice<f32>,
    uall: CudaSlice<u16>,
    mvox: usize,
    ut: CudaSlice<f32>,
    vt: CudaSlice<f32>,
    sdf: CudaSlice<f32>,
    valves: CudaSlice<f32>,
    cuts: CudaSlice<f32>,
    acc: CudaSlice<f32>,
    diagbuf: CudaSlice<f32>,
    sc0: CudaSlice<f32>,
    sc1: CudaSlice<f32>,
    st0: CudaSlice<f32>,
    st1: CudaSlice<f32>,
    scur: usize,
    atab: CudaSlice<f32>,
}

fn cfg(n: usize) -> LaunchConfig {
    LaunchConfig { grid_dim: ((n as u32).div_ceil(256), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
}

impl GpuBackend {
    pub fn new(dev: &Device, topo: Topology, uall: &[u16], mvox: usize, sdf: &[f32], rmap: &[u8], cells: usize) -> Result<Self> {
        let st = dev.default_stream();
        let src = format!("{CORE_SRC}\n{CUDA_SRC}");
        let m = dev.load_module("hasty_cfd_lbm_v4", &src, &CompileSettings { options: vec!["--use_fast_math".into()], no_cache: false })?;
        let k = Kernels {
            motion: m.load_function("k_motion")?,
            geom1: m.load_function("k_geom1")?,
            geom2: m.load_function("k_geom2")?,
            refill: m.load_function("k_refill")?,
            commit: m.load_function("k_commit")?,
            step: m.load_function("k_step")?,
            accum: m.load_function("k_accum")?,
            diag: m.load_function("k_diag")?,
            scalar: m.load_function("k_scalar")?,
            srefill: m.load_function("k_srefill")?,
        };
        let n = topo.nodes();
        let txyz = st.clone_htod(&topo.txyz)?;
        let tnbr = st.clone_htod(&topo.tnbr)?;
        let phi = st.clone_htod(&vec![-1.0f32; n])?;
        Ok(Self {
            st: st.clone(),
            k,
            n,
            txyz,
            tnbr,
            f0: st.alloc_zeros(Q * n)?,
            f1: st.alloc_zeros(Q * n)?,
            ru0: st.alloc_zeros(4 * n)?,
            ru1: st.alloc_zeros(4 * n)?,
            cur: 0,
            xref: st.alloc_zeros(3 * n)?,
            phi,
            um: st.alloc_zeros(3 * n)?,
            bsol: st.alloc_zeros(n)?,
            bcid: st.alloc_zeros(n)?,
            pid: st.alloc_zeros(n)?,
            rid: st.alloc_zeros(n)?,
            rmap: if rmap.is_empty() { st.alloc_zeros(1)? } else { st.clone_htod(rmap)? },
            flags: st.alloc_zeros(n)?,
            links: st.alloc_zeros(n)?,
            qlink: st.alloc_zeros((Q - 1) * n)?,
            spg: st.alloc_zeros(n)?,
            uall: st.clone_htod(uall)?,
            mvox,
            ut: st.alloc_zeros(3 * mvox)?,
            vt: st.alloc_zeros(3 * mvox)?,
            sdf: st.clone_htod(sdf)?,
            valves: st.alloc_zeros(1)?,
            cuts: st.alloc_zeros(1)?,
            acc: st.alloc_zeros(topo.ntiles * cells * ACC_STRIDE)?,
            diagbuf: st.alloc_zeros(64)?,
            sc0: st.clone_htod(&vec![1.0f32; n])?,
            sc1: st.clone_htod(&vec![1.0f32; n])?,
            st0: st.alloc_zeros(n)?,
            st1: st.alloc_zeros(n)?,
            scur: 0,
            atab: st.alloc_zeros(2)?,
            topo,
        })
    }

    /// bytes on the device for a topology and output cells per tile (an estimate for planning)
    pub fn estimate_bytes(ntiles: usize, cells: usize, mvox: usize, nt_motion: usize, sdf_len: usize) -> usize {
        let n = ntiles * 512;
        n * (2 * Q * 4 + 2 * 16 + 12 + 4 + 12 + 4 + 1 + 1 + 1 + 1 + 4 + (Q - 1) + 4 + 8) + ntiles * cells * ACC_STRIDE * 4 + nt_motion * 3 * mvox * 2 + 6 * mvox * 4 + sdf_len * 5
    }
}

impl Backend for GpuBackend {
    fn pace(&self) {
        hasty_cuda::pace::pace(&self.st);
    }

    fn nodes(&self) -> usize {
        self.n
    }

    fn motion(&mut self, mp: &MotionParams) -> Result<()> {
        unsafe {
            self.st.launch_builder(&self.k.motion).arg(mp).arg(&self.uall).arg(&mut self.ut).arg(&mut self.vt).launch(cfg(self.mvox))?;
        }
        Ok(())
    }

    fn geometry(&mut self, gp: &GeomParams, valves: &[f32], cuts: &[f32], refill: bool) -> Result<()> {
        if !valves.is_empty() {
            if self.valves.len() != valves.len() {
                self.valves = self.st.alloc_zeros(valves.len())?;
            }
            self.st.memcpy_htod(valves, &mut self.valves)?;
        }
        if !cuts.is_empty() {
            if self.cuts.len() != cuts.len() {
                self.cuts = self.st.alloc_zeros(cuts.len())?;
            }
            self.st.memcpy_htod(cuts, &mut self.cuts)?;
        }
        let n = self.n as i64;
        let c = cfg(self.n);
        unsafe {
            self.st
                .launch_builder(&self.k.geom1)
                .arg(gp)
                .arg(&n)
                .arg(&self.txyz)
                .arg(&self.ut)
                .arg(&self.vt)
                .arg(&self.sdf)
                .arg(&self.valves)
                .arg(&self.cuts)
                .arg(&mut self.xref)
                .arg(&mut self.phi)
                .arg(&mut self.um)
                .arg(&mut self.bsol)
                .arg(&mut self.bcid)
                .arg(&mut self.flags)
                .arg(&mut self.spg)
                .arg(&mut self.pid)
                .arg(&self.rmap)
                .arg(&mut self.rid)
                .launch(c)?;
            self.st.launch_builder(&self.k.geom2).arg(&n).arg(&self.tnbr).arg(&self.phi).arg(&mut self.links).arg(&mut self.qlink).arg(&mut self.flags).launch(c)?;
            if refill {
                let (f, ru) = if self.cur == 0 { (&mut self.f0, &mut self.ru0) } else { (&mut self.f1, &mut self.ru1) };
                self.st.launch_builder(&self.k.refill).arg(&n).arg(&self.tnbr).arg(&self.flags).arg(&self.um).arg(f).arg(ru).launch(c)?;
                let sc = if self.scur == 0 { &mut self.sc0 } else { &mut self.sc1 };
                self.st.launch_builder(&self.k.srefill).arg(&n).arg(&self.tnbr).arg(&self.flags).arg(sc).launch(c)?;
            }
            self.st.launch_builder(&self.k.commit).arg(&n).arg(&mut self.flags).launch(c)?;
        }
        Ok(())
    }

    fn step(&mut self, sp: &StepParams) -> Result<()> {
        let n = self.n as i64;
        let (src, dst, rp, rc) = if self.cur == 0 { (&self.f0, &mut self.f1, &self.ru0, &mut self.ru1) } else { (&self.f1, &mut self.f0, &self.ru1, &mut self.ru0) };
        unsafe {
            self.st
                .launch_builder(&self.k.step)
                .arg(sp)
                .arg(&n)
                .arg(&self.tnbr)
                .arg(&self.flags)
                .arg(&self.links)
                .arg(&self.qlink)
                .arg(&self.um)
                .arg(&self.bsol)
                .arg(&self.bcid)
                .arg(&self.cuts)
                .arg(src)
                .arg(dst)
                .arg(rp)
                .arg(rc)
                .arg(&self.spg)
                .launch(cfg(self.n))?;
        }
        self.cur = 1 - self.cur;
        Ok(())
    }

    fn accumulate(&mut self, sp: &StepParams) -> Result<()> {
        let n = self.n as i64;
        let (ru, f) = if self.cur == 0 { (&self.ru0, &self.f0) } else { (&self.ru1, &self.f1) };
        unsafe {
            self.st
                .launch_builder(&self.k.accum)
                .arg(sp)
                .arg(&n)
                .arg(&self.flags)
                .arg(&self.bsol)
                .arg(ru)
                .arg(&mut self.acc)
                .arg(&self.tnbr)
                .arg(&self.phi)
                .arg(&self.spg)
                .arg(f)
                .arg(if self.scur == 0 { &self.sc0 } else { &self.sc1 })
                .launch(cfg(self.n))?;
        }
        Ok(())
    }

    fn take_acc(&mut self) -> Result<Vec<f32>> {
        let out = self.st.clone_dtoh(&self.acc)?;
        self.st.memset_zeros(&mut self.acc)?;
        Ok(out)
    }

    fn diag(&mut self, ncuts: usize, nprobes: usize, nreg: usize, light: bool) -> Result<Vec<f32>> {
        let len = 4 * ncuts + 2 * nprobes + 2 * nreg + 4 + 2 * ncuts;
        if self.diagbuf.len() < len {
            self.diagbuf = self.st.alloc_zeros(len)?;
        }
        self.st.memset_zeros(&mut self.diagbuf)?;
        let n = self.n as i64;
        let nc = ncuts as i32;
        let mode = light as i32;
        let np = nprobes as i32;
        let nr = nreg as i32;
        let ru = if self.cur == 0 { &self.ru0 } else { &self.ru1 };
        let sc = if self.scur == 0 { &self.sc0 } else { &self.sc1 };
        unsafe {
            self.st.launch_builder(&self.k.diag).arg(&n).arg(&nc).arg(&np).arg(&nr).arg(&mode).arg(&self.flags).arg(&self.bcid).arg(&self.pid).arg(&self.rid).arg(&self.cuts).arg(ru).arg(sc).arg(&mut self.diagbuf).launch(cfg(self.n))?;
        }
        let d = self.st.clone_dtoh(&self.diagbuf)?;
        Ok(d[..len].to_vec())
    }

    fn save(&mut self) -> Result<(Vec<f32>, Vec<f32>)> {
        let (f, ru) = if self.cur == 0 { (&self.f0, &self.ru0) } else { (&self.f1, &self.ru1) };
        Ok((self.st.clone_dtoh(f)?, self.st.clone_dtoh(ru)?))
    }

    fn restore(&mut self, f: &[f32], ru: &[f32]) -> Result<()> {
        let (fd, rd) = if self.cur == 0 { (&mut self.f0, &mut self.ru0) } else { (&mut self.f1, &mut self.ru1) };
        self.st.memcpy_htod(f, fd)?;
        self.st.memcpy_htod(ru, rd)?;
        Ok(())
    }

    fn fields(&mut self) -> Result<Fields> {
        let ru = if self.cur == 0 { &self.ru0 } else { &self.ru1 };
        Ok(Fields {
            flags: self.st.clone_dtoh(&self.flags)?,
            ru: self.st.clone_dtoh(ru)?,
            phi: self.st.clone_dtoh(&self.phi)?,
            um: self.st.clone_dtoh(&self.um)?,
            bsol: self.st.clone_dtoh(&self.bsol)?,
            bcid: self.st.clone_dtoh(&self.bcid)?,
        })
    }

    fn set_spin_table(&mut self, atab: &[f32]) -> Result<()> {
        let t: Vec<f32> = if atab.len() < 2 { vec![0.0; 2] } else { atab.to_vec() };
        self.atab = self.st.clone_htod(&t)?;
        Ok(())
    }

    fn scalar(&mut self, sp: &SpinParams) -> Result<()> {
        let n = self.n as i64;
        let ru = if self.cur == 0 { &self.ru0 } else { &self.ru1 };
        let (so, sn) = if self.scur == 0 { (&self.sc0, &mut self.sc1) } else { (&self.sc1, &mut self.sc0) };
        // BFECC: ŝ → st0, s_c → st1, s' → sn
        for pass in 0..3i32 {
            let mut b = self.st.launch_builder(&self.k.scalar);
            b.arg(sp).arg(&pass).arg(&n).arg(&self.txyz).arg(&self.tnbr).arg(&self.flags).arg(&self.bcid).arg(&self.cuts).arg(ru).arg(&self.atab).arg(so);
            match pass {
                0 => {
                    b.arg(so).arg(&mut self.st0);
                }
                1 => {
                    b.arg(&self.st0).arg(&mut self.st1);
                }
                _ => {
                    b.arg(&self.st1).arg(&mut *sn);
                }
            }
            unsafe {
                b.launch(cfg(self.n))?;
            }
        }
        self.scur = 1 - self.scur;
        Ok(())
    }

    fn save_scalar(&mut self) -> Result<Vec<f32>> {
        Ok(self.st.clone_dtoh(if self.scur == 0 { &self.sc0 } else { &self.sc1 })?)
    }

    fn restore_scalar(&mut self, s: &[f32]) -> Result<()> {
        let d = if self.scur == 0 { &mut self.sc0 } else { &mut self.sc1 };
        self.st.memcpy_htod(s, d)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    /// the CUDA source compiles (NVRTC only, no device needed)
    #[test]
    fn cuda_source_compiles() {
        let src = format!("{}\n{}", super::CORE_SRC, super::CUDA_SRC);
        let r = hasty_cuda::compile_cubin("hasty_cfd_lbm_check", &src, "sm_89", &hasty_cuda::CompileSettings { options: vec!["--use_fast_math".into()], no_cache: true });
        match r {
            Ok(c) => assert!(!c.is_empty()),
            Err(hasty_cuda::Error::Nvrtc { log, .. }) => panic!("NVRTC:\n{log}"),
            Err(e) => eprintln!("NVRTC not available here ({e}), skipped"),
        }
    }
}
