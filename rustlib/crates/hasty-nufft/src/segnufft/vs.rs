//! Forward variant `vs` (design 6 of research/sample_kernels.md: value-stationary rows, lanes own channels; kernel
//! `segnufft_vs.cu`, §14 there): home-tile tables built once per sample set, the kernel launch, its candidate shapes
//! and the autotune step. A new forward next to the tile kernels (`TileCfg::kv` 0 / 1 / 2 stay as they are): the
//! engine uses it only when it is active ([`SegNufft::set_vs`], or the autotune found it faster) and the set carries
//! vs tables of the engine's geometry; every other case (no tables, host-resident tables streamed into a slot, order
//! ≥ 2, interval-major frames) runs the tile kernels.
//!
//! Env: HASTY_SEG_VS = 0 (off: no tables, never timed), 1 (on: tables built, the autotune keeps the fastest vs shape
//! even when a tile kernel is faster; active from construction), unset / auto (tables built, the autotune picks vs
//! only where it is faster). HASTY_SEG_VS_TILE = tx,ty,tz,smax (table geometry), HASTY_SEG_VS_CFG = nwc,nws,vxpad (one
//! launch shape instead of the autotune's candidates).

use super::*;

/// Table geometry of the vs forward: home tile TX × TY × TZ and work items of ≤ `smax` samples. The sample tables
/// depend on it (set it before [`SegNufft::points`]); the launch tunables ([`VsCfg`]) do not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VsGeom {
    pub tx: usize,
    pub ty: usize,
    pub tz: usize,
    pub smax: usize,
}

/// Launch tunables of the vs forward: 16-channel groups per block (`nwc`), window-position splits per block (`nws`,
/// divides ty), zero-padded x weights over TX + W − 1 instead of TX shift variants (`vxpad` 0 | 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VsCfg {
    pub nwc: usize,
    pub nws: usize,
    pub vxpad: usize,
}

/// env HASTY_SEG_VS
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum VsMode {
    Off,
    Auto,
    On,
}

pub(super) fn vs_mode() -> VsMode {
    match std::env::var("HASTY_SEG_VS").as_deref() {
        Ok("0") | Ok("off") => VsMode::Off,
        Ok("1") | Ok("on") => VsMode::On,
        Ok("auto") | Ok("") | Err(_) => VsMode::Auto,
        Ok(v) => panic!("HASTY_SEG_VS={v}: 0 | 1 | auto"),
    }
}

/// the vs tables of one sample set (device; host copies while the set is host-resident)
pub(super) struct VsTables {
    pub geom: VsGeom,
    /// entries (key << 25 | j on knot engines, j on static ones) and items [tile, first, count, zlo | zhi << 16]
    dev: Option<(CudaSlice<u32>, CudaSlice<i32>)>,
    host: Option<(Vec<u32>, Vec<i32>)>,
    len: (usize, usize),
    /// first item of knot b (static: one list), knots + 1 entries
    kofs: Vec<usize>,
}

impl VsTables {
    pub fn bytes(&self) -> usize {
        4 * (self.len.0 + self.len.1)
    }
    pub fn device_bytes(&self) -> usize {
        if self.dev.is_some() { self.bytes() } else { 0 }
    }
    pub fn to_host(&mut self, st: &Arc<CudaStream>) -> Result<(), NufftError> {
        if let Some((v, i)) = self.dev.take() {
            self.host = Some((st.clone_dtoh(&v)?, st.clone_dtoh(&i)?));
        }
        Ok(())
    }
    pub fn to_device(&mut self, st: &Arc<CudaStream>) -> Result<(), NufftError> {
        if let Some((v, i)) = self.host.take() {
            self.dev = Some((st.clone_htod(&v)?, st.clone_htod(&i)?));
        }
        Ok(())
    }
    /// items of knot b (static: b = 0)
    pub fn items_of(&self, b: usize) -> (usize, usize) {
        let b = b.min(self.kofs.len() - 2);
        (self.kofs[b], self.kofs[b + 1])
    }
}

/// the active vs forward of an engine
pub(super) struct VsActive {
    pub geom: VsGeom,
    pub cfg: VsCfg,
    pub f: CudaFunction,
    pub smem: usize,
}

/// knot offset KOFF of segnufft_knot.cu (C integer division)
fn koff(ks: usize) -> i64 {
    -((ks as i64 - 2) / 2)
}

/// shared-memory floats of one sample record (VSW in segnufft_vs.cu)
fn vsw(order: u32, w: usize, g: VsGeom, vxpad: usize) -> usize {
    let kxn = if vxpad == 1 { g.tx + w - 1 } else { w };
    let up4 = |v: usize| v.div_ceil(4) * 4;
    let vr_m = if order == 0 {
        let kxs = up4(kxn + 2);
        2 * kxs + 8 + up4(w)
    } else {
        let kxs = up4(kxn + 1);
        3 * kxs + 32 + 4 * w + 8
    };
    up4(vr_m + 2 + 5 + 1 + 3)
}

/// dynamic shared memory (bytes) of a vs launch shape (records | range table | accumulators, as the kernel)
pub(super) fn vs_smem(order: u32, w: usize, g: VsGeom, c: VsCfg) -> usize {
    let tab = ((g.tz + w - 1) * g.tx * g.ty * 4 + 15) & !15;
    g.smax * vsw(order, w, g, c.vxpad) * 4 + tab + c.nwc * 2 * (g.smax * 16 + 16) * 4
}

/// block-thread limit of a vs shape: __launch_bounds__(NT) caps registers at 64 K / NT; TX 2 order 0 needs ≈ 160,
/// TX 4 ≈ 226, order 1 more (research/sample_kernels.md §12.5)
fn vs_ntmax(order: u32, g: VsGeom) -> usize {
    if order == 0 && g.tx <= 2 { 384 } else { 256 }
}

impl SegNufft {
    /// grid channels of one coil batch and 16-channel groups
    fn vs_groups(&self, ncb: usize) -> usize {
        (ncb * self.nseg).div_ceil(16)
    }

    /// whether a geometry can run on this engine: order ≤ 1, one frame slot (static image or knot-major), W ≤ 10, the
    /// fine grid at least one window larger than the tile on every axis (one conditional wrap per index), ≤ 2^16 − 1
    /// samples per item
    pub(super) fn vs_geom_ok(&self, g: VsGeom) -> bool {
        let (w, nf) = (self.plan.ker.w, self.plan.nf);
        self.order <= 1
            && self.nf == 1
            && w <= 10
            && g.tx >= 1
            && g.ty >= 1
            && g.tz >= 1
            && g.tx <= 16
            && g.ty <= 16
            && g.tz <= 64
            && g.tx * g.ty <= 256
            && (1..65536).contains(&g.smax)
            && nf[0] >= g.tx + w - 1
            && nf[1] >= g.ty + 9
            && nf[2] >= g.tz + w - 1
    }

    /// whether a launch shape is valid for this engine's batch: whole coils per block (one owner per (sample, coil):
    /// one block row for all channels, or 16 · nwc a multiple of NSEG), thread limit, shared memory, nws | ty
    pub(super) fn vs_cfg_ok(&self, g: VsGeom, c: VsCfg) -> bool {
        let gr = self.vs_groups(self.cb);
        let rows = gr.div_ceil(c.nwc.max(1));
        c.nwc >= 1
            && c.nwc <= gr
            && c.nws >= 1
            && g.ty % c.nws == 0
            && c.vxpad <= 1
            && !(c.vxpad == 1 && g.tx == 1)
            && (rows == 1 || (16 * c.nwc) % self.nseg == 0)
            && 32 * c.nwc * c.nws <= vs_ntmax(self.order, g)
            && vs_smem(self.order, self.plan.ker.w, g, c) <= self.smax
    }

    /// default geometry (env HASTY_SEG_VS_TILE = tx,ty,tz,smax): order 0 4 × 4 × 16, items of 64 samples with ≥ 64
    /// channels per batch else 32 (the prototype's tuned shapes, §12.6); order 1 2 × 4 × 16 × 32 (§14)
    pub(super) fn vs_default_geom(&self) -> VsGeom {
        if let Ok(v) = std::env::var("HASTY_SEG_VS_TILE") {
            let p: Vec<usize> = v.split(',').map(|x| x.trim().parse().expect("HASTY_SEG_VS_TILE = tx,ty,tz,smax")).collect();
            assert!(p.len() == 4, "HASTY_SEG_VS_TILE = tx,ty,tz,smax");
            return VsGeom { tx: p[0], ty: p[1], tz: p[2], smax: p[3] };
        }
        let nch = self.cb * self.nseg;
        if self.order == 0 { VsGeom { tx: 4, ty: 4, tz: 16, smax: if nch >= 64 { 64 } else { 32 } } } else { VsGeom { tx: 2, ty: 4, tz: 16, smax: 32 } }
    }

    /// launch shapes the autotune times (env HASTY_SEG_VS_CFG = nwc,nws,vxpad: that one): channel groups per block
    /// 1–4 and the largest valid, window splits 1 and 2, vxpad 0 / 1
    pub(super) fn vs_candidates(&self, g: VsGeom) -> Vec<VsCfg> {
        if let Ok(v) = std::env::var("HASTY_SEG_VS_CFG") {
            let p: Vec<usize> = v.split(',').map(|x| x.trim().parse().expect("HASTY_SEG_VS_CFG = nwc,nws,vxpad")).collect();
            let c = VsCfg { nwc: p[0], nws: *p.get(1).unwrap_or(&1), vxpad: *p.get(2).unwrap_or(&0) };
            assert!(self.vs_cfg_ok(g, c), "HASTY_SEG_VS_CFG={v}: not a valid vs shape for {g:?} on this engine");
            return vec![c];
        }
        let gr = self.vs_groups(self.cb);
        let mut nwcs: Vec<usize> = (1..=gr.min(4)).collect();
        if let Some(n) = (1..=gr).rev().find(|&n| self.vs_cfg_ok(g, VsCfg { nwc: n, nws: 1, vxpad: 0 })) {
            nwcs.push(n);
        }
        nwcs.sort_unstable();
        nwcs.dedup();
        let mut v = Vec::new();
        for &nwc in &nwcs {
            for nws in [1usize, 2] {
                for vxpad in [0usize, 1] {
                    let c = VsCfg { nwc, nws, vxpad };
                    if self.vs_cfg_ok(g, c) {
                        v.push(c);
                    }
                }
            }
        }
        v
    }

    /// the default launch shape of a geometry: the largest valid channel group, no splits, no padding
    pub(super) fn vs_default_cfg(&self, g: VsGeom) -> Option<VsCfg> {
        let gr = self.vs_groups(self.cb);
        (1..=gr).rev().map(|nwc| VsCfg { nwc, nws: 1, vxpad: 0 }).find(|&c| self.vs_cfg_ok(g, c))
    }

    /// compile (disk-cached) the vs forward of a shape
    pub(super) fn vs_compile(&self, g: VsGeom, c: VsCfg) -> Result<VsActive, NufftError> {
        let (kpre, ktag) = kernel_prefix(&self.plan);
        let src = format!(
            "{kpre}#define W {}\n#define BETA {:.17e}\n#define NSEG {}\n#define CORR {}\n#define NCB {}\n#define TX {}\n#define TY {}\n#define TZ {}\n#define SMAX {}\n#define NWC {}\n#define NWS {}\n#define VXPAD {}\n#define KS {}\n#define RDEG {}\n{}",
            self.plan.ker.w,
            self.plan.ker.beta,
            self.nseg,
            self.order,
            self.cb,
            g.tx,
            g.ty,
            g.tz,
            g.smax,
            c.nwc,
            c.nws,
            c.vxpad,
            self.ks,
            self.rdeg,
            include_str!("../segnufft_vs.cu")
        );
        let name = format!("segnufft_vs{ktag}_w{}_l{}_o{}_c{}_k{}r{}_t{}x{}x{}_s{}_g{}_{}_{}", self.plan.ker.w, self.nseg, self.order, self.cb, self.ks, self.rdeg, g.tx, g.ty, g.tz, g.smax, c.nwc, c.nws, c.vxpad);
        let module = self.dev.load_module(&name, &src, &CompileSettings::default())?;
        let smem = vs_smem(self.order, self.plan.ker.w, g, c);
        let f = tile_fn(&module, "seg_forward_vs", smem)?;
        Ok(VsActive { geom: g, cfg: c, f, smem })
    }

    /// engine construction: the default geometry when vs is a candidate (HASTY_SEG_VS unset / auto / 1) and fits this
    /// engine; with HASTY_SEG_VS=1 the default launch shape (or HASTY_SEG_VS_CFG) is active from the start
    pub(super) fn vs_init(&mut self) -> Result<(), NufftError> {
        let mode = vs_mode();
        if mode == VsMode::Off {
            return Ok(());
        }
        let g = self.vs_default_geom();
        let Some(c) = self.vs_default_cfg(g).filter(|_| self.vs_geom_ok(g)) else { return Ok(()) };
        self.vs_geom = Some(g);
        if mode == VsMode::On {
            let c = if std::env::var("HASTY_SEG_VS_CFG").is_ok() { self.vs_candidates(g)[0] } else { c };
            self.vs = Some(self.vs_compile(g, c)?);
        }
        Ok(())
    }

    /// Geometry of the vs tables built by the next [`points`](Self::points) / [`points2d`](Self::points2d) calls
    /// (None: no tables; sets built before keep theirs). Err if the engine cannot run it. Clears an active vs forward
    /// of another geometry.
    pub fn set_vs_geom(&mut self, g: Option<VsGeom>) -> Result<(), NufftError> {
        if let Some(g) = g {
            if !self.vs_geom_ok(g) || self.vs_default_cfg(g).is_none() {
                return Err(NufftError::Invalid(format!("vs geometry {g:?}: not valid on this engine (order {}, frame slots {}, grid {:?}, W {})", self.order, self.nf, self.plan.nf, self.plan.ker.w)));
            }
        }
        if self.vs.as_ref().is_some_and(|a| Some(a.geom) != g) {
            self.vs = None;
        }
        self.vs_geom = g;
        Ok(())
    }

    /// the geometry of the vs tables built from now on (None: no tables)
    pub fn vs_geom(&self) -> Option<VsGeom> {
        self.vs_geom
    }

    /// Use the vs forward with launch shape `c` on the current geometry (sets with vs tables of that geometry; others
    /// run the tile kernels), or `None` for the tile kernels everywhere.
    pub fn set_vs(&mut self, c: Option<VsCfg>) -> Result<(), NufftError> {
        let Some(c) = c else {
            self.vs = None;
            return Ok(());
        };
        let g = self.vs_geom.ok_or_else(|| NufftError::Invalid("set_vs: no vs geometry on this engine (set_vs_geom, HASTY_SEG_VS)".into()))?;
        if !self.vs_cfg_ok(g, c) {
            return Err(NufftError::Invalid(format!("set_vs: {c:?} is not a valid launch shape for {g:?}")));
        }
        self.vs = Some(self.vs_compile(g, c)?);
        Ok(())
    }

    /// the active vs forward (geometry, launch shape); None: the tile kernels
    pub fn vs_active(&self) -> Option<(VsGeom, VsCfg)> {
        self.vs.as_ref().map(|a| (a.geom, a.cfg))
    }

    /// kernel times (ms) of the last autotune's forward choice: (best tile kernel, best vs shape or NaN)
    pub fn vs_tune_times(&self) -> Option<(f32, f32)> {
        self.vs_tune.map(|t| (t.0, t.1))
    }

    /// the last autotune's fastest vs launch shape (whether or not it was kept)
    pub fn vs_tune_cfg(&self) -> Option<VsCfg> {
        self.vs_tune.and_then(|t| t.2)
    }

    /// Build the vs tables of a set (None when the engine has no vs geometry). `sc`: fine-grid coordinates SoA 3 × m;
    /// `keys`: knot engines, the interval key of every sample (1-D: interval; 2-D: ri · mc + ic).
    pub(super) fn vs_tables(&self, st: &Arc<CudaStream>, sc: &[f32], m: usize, keys: Option<&[u32]>) -> Result<Option<VsTables>, NufftError> {
        let Some(g) = self.vs_geom else { return Ok(None) };
        if !self.vs_geom_ok(g) {
            return Ok(None);
        }
        let w = self.plan.ker.w;
        // knots of an interval key
        let (nint, ks, kr, rdeg) = (self.mc, self.ks, self.kr, self.rdeg);
        let nknots = if self.ks > 0 { self.frames } else { 1 };
        let knots_of = |key: u32, out: &mut Vec<usize>| {
            out.clear();
            if ks == 0 {
                out.push(0);
                return;
            }
            let (ri, ic) = if kr > 0 { (key as usize / nint, key as usize % nint) } else { (0, key as usize) };
            let resp: &[usize] = &if kr == 0 { [0, usize::MAX] } else if rdeg == 0 { [ri, usize::MAX] } else { [ri, ri + 1] };
            for &k in resp.iter().filter(|&&k| k != usize::MAX) {
                for o in 0..ks {
                    let bc = (ic as i64 + koff(ks) + o as i64).rem_euclid(nint as i64) as usize;
                    let b = k * nint + bc;
                    if !out.contains(&b) {
                        out.push(b);
                    }
                }
            }
        };
        let (vl, items, kofs) = build_vs_tables(sc, m, self.plan.nf, w, g, keys, nknots, &knots_of);
        let len = (vl.len(), items.len());
        // device buffers hold at least one element (empty tables)
        let dv = if vl.is_empty() { st.clone_htod(&[0u32])? } else { st.clone_htod(&vl)? };
        let di = if items.is_empty() { st.clone_htod(&[0i32; 4])? } else { st.clone_htod(&items)? };
        Ok(Some(VsTables { geom: g, dev: Some((dv, di)), host: None, len, kofs }))
    }

    /// vs gather of knot b (static: b = 0) for coils c0..c0+ncb: y[c0.., :] += β_b A(slot)
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_vs(&self, st: &Arc<CudaStream>, a: &VsActive, seg: &SegSegments, pts: &SegPoints, t: &VsTables, b: usize, g: Grids, y: &mut CudaSlice<C32>, c0: usize, ncb: usize) -> Result<(), NufftError> {
        let (k0, k1) = t.items_of(b);
        if k1 == k0 {
            return Ok(());
        }
        let (vl, items) = t.dev.as_ref().expect("vs tables on the device");
        let items = items.slice(4 * k0..4 * k1);
        let (t0, dt, ntab, m_i, c0_i, ncb_i, d) = (seg.tab.1, seg.tab.2, seg.tab.0 as i32, pts.m as i32, c0 as i32, ncb as i32, self.plan.dims());
        let (b_i, nint, nri) = (b as i32, pts.knot.max(1) as i32, pts.nri as i32);
        let itab = seg.itab(b);
        let rows = self.vs_groups(ncb).div_ceil(a.cfg.nwc);
        let (yp, _gy) = y.device_ptr_mut(st);
        unsafe {
            st.launch_builder(&a.f)
                .arg(&pts.rec)
                .arg(vl)
                .arg(&items)
                .arg(&itab)
                .arg(&ntab)
                .arg(&t0)
                .arg(&dt)
                .arg(&g)
                .arg(&b_i)
                .arg(&nint)
                .arg(&nri)
                .arg(&yp)
                .arg(&m_i)
                .arg(&c0_i)
                .arg(&ncb_i)
                .arg(&d)
                .launch(LaunchConfig { grid_dim: ((k1 - k0) as u32, rows as u32, 1), block_dim: ((32 * a.cfg.nwc * a.cfg.nws) as u32, 1, 1), shared_mem_bytes: a.smem as u32 })?;
        }
        Ok(())
    }

    /// Forward gather of interval / knot i for coils c0..c0+ncb into y (rows c0 .. c0 + ncb, accumulating): the vs
    /// forward when it is active and the set has device vs tables of its geometry, else the tile kernel.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gather(&self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, i: usize, g: Grids, y: &mut CudaSlice<C32>, c0: usize, ncb: usize) -> Result<(), NufftError> {
        if let (Some(a), Some(t)) = (self.vs.as_ref(), pts.vs.as_ref()) {
            if t.geom == a.geom && t.dev.is_some() {
                return self.launch_vs(st, a, seg, pts, t, i, g, y, c0, ncb);
            }
        }
        if self.ks > 0 {
            self.launch_fwd_k(st, &self.f.fwd, &self.tiles_fwd, seg, pts, i, g, y, c0, ncb)
        } else {
            self.launch_fwd(st, &self.f.fwd, &self.tiles_fwd, seg, pts, i, g, y, c0, ncb)
        }
    }

    /// Autotune step of the vs forward on `pts` (device tables of the engine's geometry): time every candidate shape
    /// on interval / knot `i` (as the tile kernels were timed) and return the fastest (ms, shape); None without tables.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn autotune_vs(&self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, i: usize, g: Grids, ys: &mut CudaSlice<C32>, ncb: usize) -> Result<Option<(f32, VsActive)>, NufftError> {
        let Some(t) = pts.vs.as_ref().filter(|t| t.dev.is_some() && Some(t.geom) == self.vs_geom) else { return Ok(None) };
        let geom = t.geom;
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let ev = (self.dev.ctx.new_event(flags)?, self.dev.ctx.new_event(flags)?);
        let mut best: Option<(f32, VsActive)> = None;
        for c in self.vs_candidates(geom) {
            let a = self.vs_compile(geom, c)?;
            self.launch_vs(st, &a, seg, pts, t, i, g, ys, 0, ncb)?;
            let mut ms = f32::INFINITY;
            for _ in 0..3 {
                ev.0.record(st)?;
                self.launch_vs(st, &a, seg, pts, t, i, g, ys, 0, ncb)?;
                ev.1.record(st)?;
                ev.1.synchronize()?;
                ms = ms.min(ev.0.elapsed_ms(&ev.1)?);
            }
            if std::env::var("HASTY_SEG_VS_LOG").is_ok_and(|v| v != "0") {
                eprintln!("    [vs] {geom:?} {c:?}: {ms:.3} ms (smem {} B, regs {})", a.smem, a.f.num_regs().unwrap_or(-1));
            }
            if best.as_ref().is_none_or(|b| ms < b.0) {
                best = Some((ms, a));
            }
        }
        Ok(best)
    }
}

/// Host build of the vs tables (research/sample_kernels.md §9, §12, §14): per knot (static: one list) the samples of
/// its window sorted by (home tile, oz, oy, ox, j), cut into items of ≤ smax samples of one tile, re-sorted inside an
/// item by (class oy · tx + ox, oz, j). Returns (entries, items [tile, first, count, zlo | zhi << 16], knot offsets).
#[allow(clippy::too_many_arguments)]
pub(super) fn build_vs_tables(sc: &[f32], m: usize, nf: [usize; 3], w: usize, g: VsGeom, keys: Option<&[u32]>, nknots: usize, knots_of: &(dyn Fn(u32, &mut Vec<usize>) + Sync)) -> (Vec<u32>, Vec<i32>, Vec<usize>) {
    use rayon::prelude::*;
    let td = [g.tx, g.ty, g.tz];
    let nt = [0, 1, 2].map(|a| nf[a].div_ceil(td[a]));
    // per sample: home tile and tile-relative corner (oz << 16 | oy << 8 | ox), the kernel's f32 arithmetic
    let ts: Vec<(u32, u32)> = (0..m)
        .into_par_iter()
        .map(|j| {
            let c = [0, 1, 2].map(|a| ((sc[a * m + j] - w as f32 * 0.5).ceil() as i64).rem_euclid(nf[a] as i64) as usize);
            let tile = ((c[2] / td[2] * nt[1] + c[1] / td[1]) * nt[0] + c[0] / td[0]) as u32;
            let sub = (((c[2] % td[2]) << 16) | ((c[1] % td[1]) << 8) | (c[0] % td[0])) as u32;
            (tile, sub)
        })
        .collect();
    // samples per knot
    let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nknots];
    match keys {
        None => lists[0] = (0..m as u32).collect(),
        Some(k) => {
            let mut kb = Vec::new();
            for j in 0..m {
                knots_of(k[j], &mut kb);
                for &b in &kb {
                    lists[b].push(j as u32);
                }
            }
        }
    }
    let mut vl: Vec<u32> = Vec::with_capacity(lists.iter().map(|l| l.len()).sum());
    let mut items: Vec<i32> = Vec::new();
    let mut kofs = vec![0usize];
    for mut l in lists {
        l.par_sort_unstable_by_key(|&j| (ts[j as usize].0, ts[j as usize].1, j));
        // items: runs of one tile, ≤ smax samples
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        let mut i = 0;
        while i < l.len() {
            let t = ts[l[i] as usize].0;
            let mut e = i;
            while e < l.len() && e - i < g.smax && ts[l[e] as usize].0 == t {
                e += 1;
            }
            ranges.push((i, e));
            i = e;
        }
        // in-item class order (disjoint slices, in parallel)
        {
            let mut parts: Vec<&mut [u32]> = Vec::with_capacity(ranges.len());
            let mut rest: &mut [u32] = &mut l;
            for &(a, b) in &ranges {
                let (x, r) = std::mem::take(&mut rest).split_at_mut(b - a);
                parts.push(x);
                rest = r;
            }
            parts.into_par_iter().for_each(|p| {
                p.sort_unstable_by_key(|&j| {
                    let s = ts[j as usize].1;
                    (((s >> 8) & 255) * g.tx as u32 + (s & 255), s >> 16, j)
                })
            });
        }
        let base = vl.len();
        for &(a, b) in &ranges {
            let (zlo, zhi) = l[a..b].iter().fold((u32::MAX, 0u32), |(lo, hi), &j| { let z = ts[j as usize].1 >> 16; (lo.min(z), hi.max(z)) });
            items.extend_from_slice(&[ts[l[a] as usize].0 as i32, (base + a) as i32, (b - a) as i32, (zlo | (zhi << 16)) as i32]);
        }
        match keys {
            None => vl.extend_from_slice(&l),
            Some(k) => vl.extend(l.iter().map(|&j| (k[j as usize] << 25) | j)),
        }
        kofs.push(items.len() / 4);
    }
    (vl, items, kofs)
}
