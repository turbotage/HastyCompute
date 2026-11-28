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
    wallmass: CudaFunction,
    wmreduce: CudaFunction,
    wmapply: CudaFunction,
    step: CudaFunction,
    accum: CudaFunction,
    diag: CudaFunction,
    scalar: CudaFunction,
    srefill: CudaFunction,
    explode: CudaFunction,
    ghostmask: CudaFunction,
    iflux: CudaFunction,
    restrict: CudaFunction,
    coalesce: CudaFunction,
}

/// buffers of the case shared by all refinement levels (one copy on the device): the motion samples and the motion
/// field at the current time (written by level 0's `motion`), the reference distance, the region and transpiration maps
struct Shared {
    uall: CudaSlice<u16>,
    ut: CudaSlice<f32>,
    vt: CudaSlice<f32>,
    sdf: CudaSlice<f32>,
    rmap: CudaSlice<u8>,
    tgrp: CudaSlice<u8>,
    tw: CudaSlice<f32>,
}

/// interface tables of a finer level (refine.rs Interface) on the device
struct Iface {
    ng: usize,
    nrim: usize,
    nhrim: usize,
    glist: CudaSlice<i32>,
    gown: CudaSlice<i32>,
    gio: CudaSlice<f32>,
    rim: CudaSlice<i32>,
    csub: CudaSlice<i32>,
    hrim: CudaSlice<i32>,
    hsub: CudaSlice<i32>,
    atile: CudaSlice<i32>,
}

pub struct GpuBackend {
    pub topo: Topology,
    st: Arc<CudaStream>,
    k: Arc<Kernels>,
    /// level 0 (owns the motion update) or a finer level
    root: bool,
    sh: Arc<Shared>,
    it: Option<Iface>,
    /// WALE scale per node × 1/32 (refinement ramp; None: none)
    wsc: Option<CudaSlice<u8>>,
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
    vplane: CudaSlice<f32>,
    nvp: i32,
    dm: CudaSlice<f32>,
    wsums: CudaSlice<f32>,
    wcorr: CudaSlice<f32>,
    wmask: u32,
    wcover: bool,
    flags: CudaSlice<u8>,
    links: CudaSlice<u32>,
    qlink: CudaSlice<u8>,
    spg: CudaSlice<f32>,
    mvox: usize,
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
        let m = dev.load_module("hasty_cfd_lbm_v5", &src, &CompileSettings { options: vec!["--use_fast_math".into()], no_cache: false })?;
        let k = Kernels {
            motion: m.load_function("k_motion")?,
            geom1: m.load_function("k_geom1")?,
            geom2: m.load_function("k_geom2")?,
            refill: m.load_function("k_refill")?,
            commit: m.load_function("k_commit")?,
            wallmass: m.load_function("k_wallmass")?,
            wmreduce: m.load_function("k_wmreduce")?,
            wmapply: m.load_function("k_wmapply")?,
            step: m.load_function("k_step")?,
            accum: m.load_function("k_accum")?,
            diag: m.load_function("k_diag")?,
            scalar: m.load_function("k_scalar")?,
            srefill: m.load_function("k_srefill")?,
            explode: m.load_function("k_explode")?,
            ghostmask: m.load_function("k_ghostmask")?,
            iflux: m.load_function("k_iflux")?,
            restrict: m.load_function("k_restrict")?,
            coalesce: m.load_function("k_coalesce")?,
        };
        let sh = Shared {
            uall: st.clone_htod(uall)?,
            ut: st.alloc_zeros(3 * mvox)?,
            vt: st.alloc_zeros(3 * mvox)?,
            sdf: st.clone_htod(sdf)?,
            rmap: if rmap.is_empty() { st.alloc_zeros(1)? } else { st.clone_htod(rmap)? },
            tgrp: st.alloc_zeros(1)?,
            tw: st.alloc_zeros(1)?,
        };
        Self::with_shared(st, Arc::new(k), Arc::new(sh), true, topo, mvox, cells)
    }

    /// node buffers of a level over shared case buffers (`cells`: output cells per tile; 0: no accumulator)
    fn with_shared(st: Arc<CudaStream>, k: Arc<Kernels>, sh: Arc<Shared>, root: bool, topo: Topology, mvox: usize, cells: usize) -> Result<Self> {
        let n = topo.nodes();
        let txyz = st.clone_htod(&topo.txyz)?;
        let tnbr = st.clone_htod(&topo.tnbr)?;
        let phi = st.clone_htod(&vec![-1.0f32; n])?;
        Ok(Self {
            st: st.clone(),
            k,
            root,
            sh,
            it: None,
            wsc: None,
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
            vplane: st.alloc_zeros(8)?,
            nvp: 0,
            dm: st.alloc_zeros(n)?,
            wsums: st.alloc_zeros(2 * crate::params::MAX_REGIONS)?,
            wcorr: st.alloc_zeros(crate::params::MAX_REGIONS)?,
            wmask: 0,
            wcover: true,
            flags: st.alloc_zeros(n)?,
            links: st.alloc_zeros(n)?,
            qlink: st.alloc_zeros((Q - 1) * n)?,
            spg: st.alloc_zeros(n)?,
            mvox,
            valves: st.alloc_zeros(1)?,
            cuts: st.alloc_zeros(1)?,
            acc: st.alloc_zeros((topo.ntiles * cells * ACC_STRIDE).max(1))?,
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
        // the motion field is shared: level 0 updates it for all levels
        if !self.root {
            return Ok(());
        }
        unsafe {
            self.st.launch_builder(&self.k.motion).arg(mp).arg(&self.sh.uall).arg(&self.sh.ut).arg(&self.sh.vt).launch(cfg(self.mvox))?;
        }
        Ok(())
    }

    fn geometry1(&mut self, gp: &GeomParams, valves: &[f32], cuts: &[f32]) -> Result<()> {
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
                .arg(&self.sh.ut)
                .arg(&self.sh.vt)
                .arg(&self.sh.sdf)
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
                .arg(&self.sh.rmap)
                .arg(&mut self.rid)
                .arg(&self.sh.tgrp)
                .arg(&self.sh.tw)
                .launch(c)?;
        }
        Ok(())
    }

    fn geometry2(&mut self, gp: &GeomParams, refill: bool) -> Result<()> {
        let n = self.n as i64;
        let c = cfg(self.n);
        unsafe {
            self.st
                .launch_builder(&self.k.geom2)
                .arg(gp)
                .arg(&n)
                .arg(&self.txyz)
                .arg(&self.tnbr)
                .arg(&self.phi)
                .arg(&self.bsol)
                .arg(&self.valves)
                .arg(&self.xref)
                .arg(&mut self.um)
                .arg(&mut self.links)
                .arg(&mut self.qlink)
                .arg(&mut self.flags)
                .launch(c)?;
            if refill {
                let (f, ru) = if self.cur == 0 { (&mut self.f0, &mut self.ru0) } else { (&mut self.f1, &mut self.ru1) };
                self.st.launch_builder(&self.k.refill).arg(&n).arg(&self.tnbr).arg(&self.flags).arg(&self.um).arg(f).arg(ru).launch(c)?;
                let sc = if self.scur == 0 { &mut self.sc0 } else { &mut self.sc1 };
                self.st.launch_builder(&self.k.srefill).arg(&n).arg(&self.tnbr).arg(&self.flags).arg(sc).launch(c)?;
            }
            if self.wmask != 0 && self.wcover && refill && gp.first == 0 {
                let ru = if self.cur == 0 { &self.ru0 } else { &self.ru1 };
                self.st.launch_builder(&self.k.wallmass).arg(&n).arg(&self.flags).arg(ru).arg(&mut self.dm).launch(c)?;
            }
            self.st.launch_builder(&self.k.commit).arg(&n).arg(&mut self.flags).launch(c)?;
        }
        Ok(())
    }

    fn step(&mut self, sp: &StepParams) -> Result<()> {
        let n = self.n as i64;
        let (src, dst, rp, rc) = if self.cur == 0 { (&self.f0, &mut self.f1, &self.ru0, &mut self.ru1) } else { (&self.f1, &mut self.f0, &self.ru1, &mut self.ru0) };
        let mut b = self.st.launch_builder(&self.k.step);
        b.arg(sp)
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
            .arg(&mut self.dm);
        match &self.wsc {
            Some(w) => b.arg(w),
            None => b.arg(&0u64),
        };
        unsafe {
            b.launch(cfg(self.n))?;
        }
        self.cur = 1 - self.cur;
        Ok(())
    }

    fn accumulate(&mut self, sp: &StepParams) -> Result<()> {
        if self.it.is_some() {
            return Err(crate::Error::Case("a finer level accumulates into level 0 (accumulate_to)".into()));
        }
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
                .arg(&self.txyz)
                .arg(&0u64)
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
        let len = 4 * ncuts + 2 * nprobes + 2 * nreg + 4 + 3 * ncuts + self.nvp as usize;
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
        let ff = if self.cur == 0 { &self.f0 } else { &self.f1 };
        unsafe {
            self.st.launch_builder(&self.k.diag).arg(&n).arg(&nc).arg(&np).arg(&nr).arg(&mode).arg(&self.flags).arg(&self.bcid).arg(&self.pid).arg(&self.rid).arg(&self.cuts).arg(ru).arg(sc).arg(&self.tnbr).arg(ff).arg(&self.nvp).arg(&self.vplane).arg(&self.xref).arg(&self.links).arg(&mut self.diagbuf).launch(cfg(self.n))?;
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
            links: self.st.clone_dtoh(&self.links)?,
            qlink: self.st.clone_dtoh(&self.qlink)?,
        })
    }

    fn set_spin_table(&mut self, atab: &[f32]) -> Result<()> {
        let t: Vec<f32> = if atab.len() < 2 { vec![0.0; 2] } else { atab.to_vec() };
        self.atab = self.st.clone_htod(&t)?;
        Ok(())
    }

    fn set_transpiration(&mut self, group: &[u8], weight: &[f32]) -> Result<()> {
        if group.len() != weight.len() || group.len() != self.sh.sdf.len() {
            return Err(crate::Error::Case(format!("transpiration: {} group / {} weight voxels, the sdf has {}", group.len(), weight.len(), self.sh.sdf.len())));
        }
        let (g, w) = (self.st.clone_htod(group)?, self.st.clone_htod(weight)?);
        let sh = Arc::get_mut(&mut self.sh).ok_or_else(|| crate::Error::Case("set_transpiration after the finer levels share the maps".into()))?;
        sh.tgrp = g;
        sh.tw = w;
        Ok(())
    }

    fn set_wall_mass(&mut self, wmask: u32, cover: bool) {
        self.wmask = wmask;
        self.wcover = cover;
    }

    fn wall_mass_peek(&mut self) -> Result<Vec<f32>> {
        let mr = crate::params::MAX_REGIONS;
        if self.wmask == 0 {
            return Ok(vec![0f32; 2 * mr]);
        }
        let n = self.n as i64;
        let keep = 1i32;
        self.st.memset_zeros(&mut self.wsums)?;
        unsafe {
            self.st.launch_builder(&self.k.wmreduce).arg(&n).arg(&self.flags).arg(&self.rid).arg(&self.wmask).arg(&keep).arg(&mut self.dm).arg(&mut self.wsums).launch(cfg(self.n))?;
        }
        Ok(self.st.clone_dtoh(&self.wsums)?)
    }

    fn wall_mass_clear(&mut self) -> Result<()> {
        self.st.memset_zeros(&mut self.dm)?;
        Ok(())
    }

    fn wall_mass_apply(&mut self, target: Option<&[f32]>) -> Result<Vec<f32>> {
        let mr = crate::params::MAX_REGIONS;
        if self.wmask == 0 {
            return Ok(vec![0f32; 2 * mr]);
        }
        let n = self.n as i64;
        let c = cfg(self.n);
        let keep = 0i32;
        self.st.memset_zeros(&mut self.wsums)?;
        unsafe {
            self.st.launch_builder(&self.k.wmreduce).arg(&n).arg(&self.flags).arg(&self.rid).arg(&self.wmask).arg(&keep).arg(&mut self.dm).arg(&mut self.wsums).launch(c)?;
        }
        let sums = self.st.clone_dtoh(&self.wsums)?;
        let corr: Vec<f32> = (0..mr).map(|r| if sums[mr + r] > 0.0 && (self.wmask >> r) & 1 == 1 { (target.map_or(0.0, |t| t[r]) - sums[r]) / sums[mr + r] } else { 0.0 }).collect();
        self.st.memcpy_htod(&corr, &mut self.wcorr)?;
        let (f, ru) = if self.cur == 0 { (&mut self.f0, &mut self.ru0) } else { (&mut self.f1, &mut self.ru1) };
        unsafe {
            self.st.launch_builder(&self.k.wmapply).arg(&n).arg(&self.flags).arg(&self.rid).arg(&self.wcorr).arg(f).arg(ru).launch(c)?;
        }
        Ok(sums)
    }

    fn set_vplanes(&mut self, vp: &[f32]) -> Result<()> {
        if !vp.is_empty() {
            self.vplane = self.st.clone_htod(vp)?;
            self.nvp = (vp.len() / 8) as i32;
        }
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

    fn new_level(&self, topo: Topology) -> Result<Self> {
        let mut b = Self::with_shared(self.st.clone(), self.k.clone(), self.sh.clone(), false, topo, self.mvox, 0)?;
        b.wmask = self.wmask;
        b.wcover = self.wcover;
        Ok(b)
    }

    fn set_roles(&mut self, roles: &[u8]) -> Result<()> {
        if roles.len() != self.n {
            return Err(crate::Error::Case(format!("roles: {} for {} nodes", roles.len(), self.n)));
        }
        let mut fl = self.st.clone_dtoh(&self.flags)?;
        for (f, r) in fl.iter_mut().zip(roles) {
            *f |= *r;
        }
        self.st.memcpy_htod(&fl, &mut self.flags)?;
        Ok(())
    }

    fn set_wale_scale(&mut self, s: &[u8]) -> Result<()> {
        if !s.is_empty() && s.len() != self.n {
            return Err(crate::Error::Case(format!("WALE scale: {} for {} nodes", s.len(), self.n)));
        }
        self.wsc = if s.is_empty() { None } else { Some(self.st.clone_htod(s)?) };
        Ok(())
    }

    fn set_interface(&mut self, it: &crate::refine::Interface) -> Result<()> {
        let up = |v: &Vec<i32>| -> Result<CudaSlice<i32>> { Ok(if v.is_empty() { self.st.alloc_zeros(1)? } else { self.st.clone_htod(v)? }) };
        self.it = Some(Iface {
            ng: it.glist.len(),
            nrim: it.rim.len(),
            nhrim: it.hrim.len(),
            glist: up(&it.glist)?,
            gown: up(&it.gown)?,
            gio: self.st.alloc_zeros(2 * it.glist.len().max(1))?,
            rim: up(&it.rim)?,
            csub: up(&it.csub)?,
            hrim: up(&it.hrim)?,
            hsub: up(&it.hsub)?,
            atile: up(&it.atile)?,
        });
        Ok(())
    }

    fn ghost_mask(&mut self, coarse: &Self) -> Result<()> {
        let it = self.it.as_ref().ok_or_else(|| crate::Error::Case("ghost_mask without interface tables".into()))?;
        if it.ng == 0 {
            return Ok(());
        }
        let ng = it.ng as i64;
        unsafe {
            self.st.launch_builder(&self.k.ghostmask).arg(&ng).arg(&it.glist).arg(&it.gown).arg(&coarse.flags).arg(&mut self.flags).launch(cfg(it.ng))?;
        }
        Ok(())
    }

    fn explode(&mut self, coarse: &Self, r: [f32; 2], interp: bool) -> Result<()> {
        let it = self.it.as_ref().ok_or_else(|| crate::Error::Case("explode without interface tables".into()))?;
        if it.ng == 0 {
            return Ok(());
        }
        let (ng, nf, nc) = (it.ng as i64, self.n as i64, coarse.n as i64);
        let (fc, ruc) = if coarse.cur == 0 { (&coarse.f0, &coarse.ru0) } else { (&coarse.f1, &coarse.ru1) };
        let scc = if coarse.scur == 0 { &coarse.sc0 } else { &coarse.sc1 };
        let sf = if self.scur == 0 { &mut self.sc0 } else { &mut self.sc1 };
        let mut b = self.st.launch_builder(&self.k.explode);
        b.arg(&ng)
                .arg(&nf)
                .arg(&nc)
                .arg(&it.glist)
                .arg(&it.gown)
                .arg(&r[0])
                .arg(&r[1])
                .arg(fc)
                .arg(ruc)
                .arg(scc);
            if interp {
                b.arg(&coarse.tnbr).arg(&coarse.flags).arg(&self.txyz);
            } else {
                b.arg(&0u64).arg(&0u64).arg(&0u64);
            }
            b.arg(&self.tnbr).arg(&self.flags).arg(&mut self.f0)
                .arg(&mut self.f1)
                .arg(&mut self.ru0)
                .arg(&mut self.ru1)
                .arg(sf);
        unsafe {
            b.launch(cfg(it.ng))?;
        }
        Ok(())
    }

    fn iflux(&mut self, first: bool) -> Result<()> {
        let it = self.it.as_mut().ok_or_else(|| crate::Error::Case("iflux without interface tables".into()))?;
        if it.ng == 0 {
            return Ok(());
        }
        let (ng, nf, fi) = (it.ng as i64, self.n as i64, first as i32);
        let src = if self.cur == 0 { &self.f0 } else { &self.f1 };
        unsafe {
            self.st.launch_builder(&self.k.iflux).arg(&ng).arg(&nf).arg(&it.glist).arg(&self.tnbr).arg(&self.flags).arg(&self.links).arg(src).arg(&fi).arg(&mut it.gio).launch(cfg(it.ng))?;
        }
        Ok(())
    }

    fn restrict_to(&self, coarse: &mut Self) -> Result<()> {
        let it = self.it.as_ref().ok_or_else(|| crate::Error::Case("restrict without interface tables".into()))?;
        if it.nhrim == 0 {
            return Ok(());
        }
        let (nh, nf, nc) = (it.nhrim as i64, self.n as i64, coarse.n as i64);
        let ruf = if self.cur == 0 { &self.ru0 } else { &self.ru1 };
        let sf = if self.scur == 0 { &self.sc0 } else { &self.sc1 };
        let sc = if coarse.scur == 0 { &mut coarse.sc0 } else { &mut coarse.sc1 };
        unsafe {
            self.st.launch_builder(&self.k.restrict).arg(&nh).arg(&nf).arg(&nc).arg(&it.hrim).arg(&it.hsub).arg(&self.flags).arg(ruf).arg(sf).arg(&mut coarse.ru0).arg(&mut coarse.ru1).arg(sc).launch(cfg(it.nhrim))?;
        }
        Ok(())
    }

    fn coalesce_to(&self, coarse: &mut Self, kfd: f32) -> Result<()> {
        let it = self.it.as_ref().ok_or_else(|| crate::Error::Case("coalesce without interface tables".into()))?;
        if it.nrim == 0 {
            return Ok(());
        }
        let (nr, nf, nc) = (it.nrim as i64, self.n as i64, coarse.n as i64);
        let ff = if self.cur == 0 { &self.f0 } else { &self.f1 };
        let (fc, ruc) = if coarse.cur == 0 { (&mut coarse.f0, &coarse.ru0) } else { (&mut coarse.f1, &coarse.ru1) };
        unsafe {
            self.st
                .launch_builder(&self.k.coalesce)
                .arg(&nr)
                .arg(&nf)
                .arg(&nc)
                .arg(&it.rim)
                .arg(&it.csub)
                .arg(&it.glist)
                .arg(&it.gio)
                .arg(&coarse.tnbr)
                .arg(&coarse.flags)
                .arg(&coarse.links)
                .arg(ruc)
                .arg(&coarse.um)
                .arg(&kfd)
                .arg(&self.flags)
                .arg(ff)
                .arg(fc)
                .launch(cfg(it.nrim))?;
        }
        Ok(())
    }

    fn accumulate_to(&self, root: &mut Self, sp: &StepParams) -> Result<()> {
        let it = self.it.as_ref().ok_or_else(|| crate::Error::Case("accumulate_to without interface tables".into()))?;
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
                .arg(&mut root.acc)
                .arg(&self.tnbr)
                .arg(&self.phi)
                .arg(&self.spg)
                .arg(f)
                .arg(if self.scur == 0 { &self.sc0 } else { &self.sc1 })
                .arg(&self.txyz)
                .arg(&it.atile)
                .launch(cfg(self.n))?;
        }
        Ok(())
    }

    fn wall_mass_sums(&mut self, keep: bool) -> Result<Vec<f32>> {
        let mr = crate::params::MAX_REGIONS;
        if self.wmask == 0 {
            return Ok(vec![0f32; 2 * mr]);
        }
        let n = self.n as i64;
        let k = keep as i32;
        self.st.memset_zeros(&mut self.wsums)?;
        unsafe {
            self.st.launch_builder(&self.k.wmreduce).arg(&n).arg(&self.flags).arg(&self.rid).arg(&self.wmask).arg(&k).arg(&mut self.dm).arg(&mut self.wsums).launch(cfg(self.n))?;
        }
        Ok(self.st.clone_dtoh(&self.wsums)?)
    }

    fn wall_mass_correct(&mut self, corr: &[f32]) -> Result<()> {
        if self.wmask == 0 {
            return Ok(());
        }
        let n = self.n as i64;
        self.st.memcpy_htod(corr, &mut self.wcorr)?;
        let (f, ru) = if self.cur == 0 { (&mut self.f0, &mut self.ru0) } else { (&mut self.f1, &mut self.ru1) };
        unsafe {
            self.st.launch_builder(&self.k.wmapply).arg(&n).arg(&self.flags).arg(&self.rid).arg(&self.wcorr).arg(f).arg(ru).launch(cfg(self.n))?;
        }
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
