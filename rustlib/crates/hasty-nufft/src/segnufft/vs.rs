//! Forward variant `vs` (design 6 of research/sample_kernels.md: value-stationary rows, lanes own channels; kernel
//! `segnufft_vs.cu`, §14 there): home-tile tables built once per sample set, the kernel launch, its candidate shapes
//! and the autotune step. A new forward next to the tile kernels (`TileCfg::kv` 0 / 1 / 2 stay as they are): the
//! engine uses it only when it is active ([`SegNufft::set_vs`], or the autotune found it faster) and the set carries
//! vs tables of the engine's geometry; every other case (no tables, host-resident tables streamed into a slot,
//! interval-major frames, static order 3) runs the tile kernels.
//!
//! Orders 2 / 3 (`vt`, research/sample_kernels.md §13 / §16): the same tables and launch shapes with the per-(sample,
//! plane) shared complex-weight table body (vxpad 0 only); order 3 on knot-major engines (static order 3 lives in the
//! multi-image kernels), with the degree-3/4 coefficients of `points_hi` / `points2d_hi` as trailing arguments.
//!
//! Env: HASTY_SEG_VS = 0 (off: no tables, never timed), 1 (on: tables built, the autotune keeps the fastest vs shape
//! even when a tile kernel is faster; active from construction), unset / auto (tables built, the autotune picks vs
//! only where it is faster). HASTY_SEG_VT = 0 | 1 | auto: the same for engines of order ≥ 2 only (unset: follow
//! HASTY_SEG_VS). HASTY_SEG_VS_TILE = tx,ty,tz,smax (table geometry), HASTY_SEG_VS_CFG = nwc,nws,vxpad (one launch
//! shape instead of the autotune's candidates).
//!
//! Phase 2 (research/sample_kernels.md §19): `sch` < smax stages an item in sub-chunks of ≤ sch samples (smaller shared
//! memory: more blocks per SM on the 100 KB parts; bitwise the same result, same tables), and every candidate is spill
//! screened on the running device. HASTY_SEG_VS_SCH = auto (default: sub-chunk candidates where they add blocks per
//! SM) | 0 (none) | <n> (that one); HASTY_SEG_VS_CFG = nwc,nws,vxpad[,sch].
//! HASTY_SEG_VT_MMA = 0 (default: FMA contraction) | bf16 | fp16 | auto: the vt forward (orders ≥ 2) with the contraction
//! on tensor cores (`VTC` in segnufft_vs.cu, research/sample_kernels.md §21; sub-chunks ≤ 32, `nws` = K-chunk splits
//! 1 | 2, sm_80+); auto lets the autotune time the fp16 variant next to the FMA shapes (measured slower on the 4090).
//! HASTY_SEG_VT_WS = 0 | 1 | auto (default): the warp-specialised vt forward (`VW` in segnufft_vs.cu,
//! research/sample_kernels.md §22: build warps write each (sample, plane) weight table once per block, copy warps stage
//! the plane window in shared memory by cp.async, consumer warps contract from shared memory; named-barrier hand-over, sm_80+);
//! auto adds its shapes to the autotune (kept only where faster), 1 times only them, all = 1 plus the `VW 2` shapes
//! (shared build with the vt register window), 0 never. HASTY_SEG_VW_CFG =
//! nwc,ncw,nbw,nst,sch: one vw shape (16-channel slices, consumer warps, build warps, window stages, sub-chunk).

use super::*;
use super::ra::{kern_res, spill_skip, KernRes};

/// Table geometry of the vs forward: home tile TX × TY × TZ and work items of ≤ `smax` samples. The sample tables
/// depend on it (set it before [`SegNufft::points`]); the launch tunables ([`VsCfg`]) do not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VsGeom {
    pub tx: usize,
    pub ty: usize,
    pub tz: usize,
    pub smax: usize,
}

/// Launch tunables of the vs forward: 16-channel groups per block (`nwc`), window-position splits per block (`nws`,
/// divides ty), zero-padded x weights over TX + W − 1 instead of TX shift variants (`vxpad` 0 | 1).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VsCfg {
    pub nwc: usize,
    pub nws: usize,
    pub vxpad: usize,
}

/// Launch tunables of the warp-specialised vt forward (`VW`, research/sample_kernels.md §22): consumer warps `ncw`,
/// weight-build warps `nbw`, shared window stages `nst`, weight-table ring slots `nwt` (≥ the sub-chunk); plus one
/// window-copy warps `ncp`. The 16-channel slices per block are [`VsCfg::nwc`] (nws 1, vxpad 0). `ncw` 0 selects
/// `VW 2`: the vt kernel's own channel-group warps (register window) read the tables that `nbw` build warps write into a
/// ring of `nwt` tables (no window staging; `nst`, `ncp` unused).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct VwCfg {
    pub ncw: usize,
    pub nbw: usize,
    pub nst: usize,
    pub nwt: usize,
    pub ncp: usize,
}

impl VwCfg {
    /// threads per block (`VW 2`: plus the `nwc` channel-group warps, see [`VsActive::threads`])
    pub fn threads(&self) -> usize {
        32 * (self.ncw + self.nbw + self.ncp)
    }
    /// `VW 2` (shared build, register window)
    pub fn shared_build(&self) -> bool {
        self.ncw == 0
    }
}

/// env HASTY_SEG_VT_WS: 0 | off, 1 | on (only vw shapes), auto | 2 (default: vw shapes next to the FMA vt)
pub(super) fn vt_ws_env() -> usize {
    match std::env::var("HASTY_SEG_VT_WS").as_deref() {
        Ok("0") | Ok("off") => 0,
        Ok("1") | Ok("on") | Ok("all") => 1,
        Ok("2") | Ok("auto") | Ok("") | Err(_) => 2,
        Ok(v) => panic!("HASTY_SEG_VT_WS={v}: 0 | 1 | all | auto"),
    }
}

/// consumer / build / window-copy warps the autotune times for the vw forward
pub(super) const VW_ROLES: [(usize, usize, usize); 3] = [(10, 3, 3), (8, 2, 2), (5, 1, 2)];

/// dynamic shared memory (bytes) of a vw shape (VW_SMEM of segnufft_vs.cu; VW 2: the vt layout with the ring of tables
/// and the slot map instead of the per-warp tables)
pub(super) fn vw_smem(order: u32, w: usize, g: VsGeom, c: VsCfg, sch: usize, v: VwCfg) -> usize {
    if v.shared_build() {
        let tab = ((g.tz + w - 1) * g.tx * g.ty * 4 + 15) & !15;
        return sch * vsw(order, w, g, 0) * 4 + tab + c.nwc * 2 * (sch * 16 + 16) * 4 + v.nwt * 800 + 2 * sch;
    }
    let nplm = g.tz + w - 1;
    let vnp = (g.tx + w - 1) * (g.ty + 9);
    let ent = sch * vsw(order, w, g, 0) * 4;
    let poff = ent + g.smax * 4 + 2 * nplm * 4;
    let acc = (poff + vnp.max(g.smax) * 4 + 15) & !15;
    acc + c.nwc * 2 * (sch * 16 + 16) * 4 + v.nwt * 800 + v.nst * vnp * 16 * 8
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

/// the mode of an engine of correction order `order`: HASTY_SEG_VT (orders ≥ 2) if set, else HASTY_SEG_VS
pub(super) fn vs_mode_order(order: u32) -> VsMode {
    if order >= 2 {
        match std::env::var("HASTY_SEG_VT").as_deref() {
            Ok("0") | Ok("off") => return VsMode::Off,
            Ok("1") | Ok("on") => return VsMode::On,
            Ok("auto") => return VsMode::Auto,
            Ok("") | Err(_) => {}
            Ok(v) => panic!("HASTY_SEG_VT={v}: 0 | 1 | auto"),
        }
    }
    vs_mode()
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

/// the active vs forward of an engine (`sch`: samples staged per sub-chunk, = geom.smax for one pass)
pub(super) struct VsActive {
    pub geom: VsGeom,
    pub cfg: VsCfg,
    pub sch: usize,
    pub f: CudaFunction,
    pub smem: usize,
    pub res: KernRes,
    /// tensor-core contraction (VTC, research/sample_kernels.md §21): 1 bf16 hi / lo, 2 fp16 hi / lo (scaled); 0: the FMA
    /// vt / vs kernel
    pub mma: usize,
    /// the warp-specialised vt (VW; research/sample_kernels.md §22); None: the kernel of `mma`
    pub ws: Option<VwCfg>,
}

impl VsActive {
    /// threads per block
    pub fn threads(&self) -> usize {
        self.ws.map_or(32 * self.cfg.nwc * self.cfg.nws, |v| if v.shared_build() { 32 * (self.cfg.nwc + v.nbw) } else { v.threads() })
    }
}

/// env HASTY_SEG_VT_MMA: 0 | off (default: the FMA vt kernel), bf16 | 1 (tensor-core vt, bf16 hi / lo), fp16 | 2
/// (tensor-core vt, scaled fp16 hi / lo), auto | 3 (the autotune times the FMA vt and the fp16 tensor-core vt)
pub(super) fn vt_mma_env() -> usize {
    match std::env::var("HASTY_SEG_VT_MMA").as_deref() {
        Ok("0") | Ok("off") | Ok("") | Err(_) => 0,
        Ok("1") | Ok("bf16") => 1,
        Ok("2") | Ok("fp16") | Ok("on") => 2,
        Ok("3") | Ok("auto") => 3,
        Ok(v) => panic!("HASTY_SEG_VT_MMA={v}: 0 | bf16 | fp16 | auto"),
    }
}

/// sub-chunk sizes the tensor-core vt is timed at (accumulators: sch / 8 n8 tiles × 16 registers)
pub(super) const VTC_SCH: [usize; 3] = [8, 16, 24];

/// knot offset KOFF of segnufft_knot.cu (C integer division)
fn koff(ks: usize) -> i64 {
    -((ks as i64 - 2) / 2)
}

/// shared-memory floats of one sample record (VSW in segnufft_vs.cu)
fn vsw(order: u32, w: usize, g: VsGeom, vxpad: usize) -> usize {
    if order >= 2 {
        // vt record: 160 + e/d/sc 24 (+ hc 28 at order 3), M 4, F 5, rounded to 4
        return if order >= 3 { 224 } else { 196 };
    }
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
#[allow(dead_code)]
pub(super) fn vs_smem(order: u32, w: usize, g: VsGeom, c: VsCfg) -> usize {
    vs_smem_v(order, w, g, c, g.smax)
}

/// as [`vs_smem`] with items staged in sub-chunks of `sch` ≤ smax samples (VSCH)
pub(super) fn vs_smem_v(order: u32, w: usize, g: VsGeom, c: VsCfg, sch: usize) -> usize {
    vs_smem_m(order, w, g, c, sch, 0)
}

/// as [`vs_smem_v`]; `mma` 1: the tensor-core vt (records | entries | plane ranges | point offsets | two bf16 weight
/// tables of NKC · 8 + 4 uint4 per sample; the accumulators alias the tables)
pub(super) fn vs_smem_m(order: u32, w: usize, g: VsGeom, c: VsCfg, sch: usize, mma: usize) -> usize {
    if mma >= 1 {
        let np = (g.tx + w - 1) * (g.ty + w - 1);
        let nkc = np.div_ceil(16);
        let head = ((sch * vsw(order, w, g, c.vxpad) + (g.smax + g.tz + w - 1).div_ceil(2) * 2 + nkc * 16) * 4 + 15) & !15;
        let bw = 2 * sch.div_ceil(8) * 8 * (nkc * 8 + 4) * 16;
        return head + bw.max(c.nwc * 2 * (sch * 16 + 16) * 4);
    }
    let tab = ((g.tz + w - 1) * g.tx * g.ty * 4 + 15) & !15;
    // vt (order ≥ 2): one double-buffered 10 × 10 complex weight table (1600 B) per warp after the accumulators
    let vt = if order >= 2 { c.nwc * c.nws * 1600 } else { 0 };
    sch * vsw(order, w, g, c.vxpad) * 4 + tab + c.nwc * 2 * (sch * 16 + 16) * 4 + vt
}

/// env HASTY_SEG_VS_SCH: None = auto, Some(0) = no sub-chunk candidates, Some(n) = only that sub-chunk
fn vs_sch_env() -> Option<usize> {
    match std::env::var("HASTY_SEG_VS_SCH").as_deref() {
        Ok("auto") | Ok("") | Err(_) => None,
        Ok("off") => Some(0),
        Ok(v) => Some(v.trim().parse().unwrap_or_else(|_| panic!("HASTY_SEG_VS_SCH={v}: auto | 0 | <samples>"))),
    }
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

    /// whether a geometry can run on this engine: order ≤ 2 (≤ 3 on knot-major engines; static order 3 lives in the
    /// multi-image kernels), one frame slot (static image or knot-major), W ≤ 10, the fine grid at least one window
    /// larger than the tile on every axis (one conditional wrap per index), ≤ 2^16 − 1 samples per item
    pub(super) fn vs_geom_ok(&self, g: VsGeom) -> bool {
        let (w, nf) = (self.plan.ker.w, self.plan.nf);
        (self.order <= 2 || (self.ks > 0 && self.order <= 3))
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
        self.vs_cfg_ok_v(g, c, g.smax)
    }

    /// as [`vs_cfg_ok`](Self::vs_cfg_ok) with sub-chunks of `sch` (1 ≤ sch ≤ smax) samples
    pub(super) fn vs_cfg_ok_v(&self, g: VsGeom, c: VsCfg, sch: usize) -> bool {
        self.vs_cfg_ok_m(g, c, sch, self.vs_mma())
    }

    /// the contraction of the non-autotuned vs paths: 1 (tensor core) with HASTY_SEG_VT_MMA=1 at orders ≥ 2
    pub(super) fn vs_mma(&self) -> usize {
        if (1..=2).contains(&self.vt_mma) && self.order >= 2 { self.vt_mma } else { 0 }
    }

    /// Select the vt contraction (0 FMA, 1 tensor-core bf16, 2 tensor-core fp16, 3 the autotune times FMA and fp16;
    /// research/sample_kernels.md §21). Clears an active vs forward of another contraction (set it again with
    /// [`set_vs_v`](Self::set_vs_v)).
    pub fn set_vt_mma(&mut self, m: usize) {
        assert!(m <= 3, "set_vt_mma: 0 | 1 | 2 | 3");
        self.vt_mma = m;
        if self.vs.as_ref().is_some_and(|a| a.mma != self.vs_mma()) {
            self.vs = None;
        }
    }

    /// whether the tensor-core vt can run this shape: orders ≥ 2, sm_80+, no x padding, ≤ 2 chunk splits (nws),
    /// sub-chunks ≤ 32 (accumulators in registers), items ≤ 64, f32 grid slots
    fn vtc_ok(&self, g: VsGeom, c: VsCfg, sch: usize) -> bool {
        let sm: u32 = self.dev.arch.trim_start_matches("sm_").trim_end_matches(['a', 'f']).parse().unwrap_or(0);
        self.order >= 2 && sm >= 80 && c.vxpad == 0 && c.nws <= 2 && sch <= 32 && g.smax <= 64 && self.gfft.grid_bits() == 0
    }

    /// as [`vs_cfg_ok_v`](Self::vs_cfg_ok_v) for contraction `mma`
    pub(super) fn vs_cfg_ok_m(&self, g: VsGeom, c: VsCfg, sch: usize, mma: usize) -> bool {
        if mma >= 1 && !self.vtc_ok(g, c, sch) {
            return false;
        }
        let gr = self.vs_groups(self.cb);
        let rows = gr.div_ceil(c.nwc.max(1));
        c.nwc >= 1
            && c.nwc <= gr
            && c.nws >= 1
            && g.ty % c.nws == 0
            && c.vxpad <= 1
            && !(c.vxpad == 1 && g.tx == 1)
            && !(c.vxpad == 1 && self.order >= 2)
            && (rows == 1 || (16 * c.nwc) % self.nseg == 0)
            && 32 * c.nwc * c.nws <= vs_ntmax(self.order, g)
            && (1..=g.smax).contains(&sch)
            && vs_smem_m(self.order, self.plan.ker.w, g, c, sch, mma) <= self.smax
    }

    /// HASTY_SEG_VS_CFG = nwc,nws,vxpad[,sch] (sch 0 or absent: smax)
    fn vs_env_cfg(g: VsGeom) -> Option<(VsCfg, usize)> {
        let v = std::env::var("HASTY_SEG_VS_CFG").ok()?;
        let p: Vec<usize> = v.split(',').map(|x| x.trim().parse().expect("HASTY_SEG_VS_CFG = nwc,nws,vxpad[,sch]")).collect();
        let c = VsCfg { nwc: p[0], nws: *p.get(1).unwrap_or(&1), vxpad: *p.get(2).unwrap_or(&0) };
        let sch = match p.get(3) {
            Some(&n) if n > 0 => n,
            _ => g.smax,
        };
        Some((c, sch))
    }

    /// default geometry (env HASTY_SEG_VS_TILE = tx,ty,tz,smax): order 0 4 × 4 × 16, items of 64 samples with ≥ 64
    /// channels per batch else 32 (the prototype's tuned shapes, §12.6); order 1 2 × 4 × 16 × 32 (§14); orders 2 / 3
    /// (vt) 4 × 4 × 16 × 48 (§13.5)
    pub(super) fn vs_default_geom(&self) -> VsGeom {
        if let Ok(v) = std::env::var("HASTY_SEG_VS_TILE") {
            let p: Vec<usize> = v.split(',').map(|x| x.trim().parse().expect("HASTY_SEG_VS_TILE = tx,ty,tz,smax")).collect();
            assert!(p.len() == 4, "HASTY_SEG_VS_TILE = tx,ty,tz,smax");
            return VsGeom { tx: p[0], ty: p[1], tz: p[2], smax: p[3] };
        }
        let nch = self.cb * self.nseg;
        match self.order {
            0 => VsGeom { tx: 4, ty: 4, tz: 16, smax: if nch >= 64 { 64 } else { 32 } },
            1 => VsGeom { tx: 2, ty: 4, tz: 16, smax: 32 },
            _ => VsGeom { tx: 4, ty: 4, tz: 16, smax: 48 },
        }
    }

    /// launch shapes the autotune times (env HASTY_SEG_VS_CFG = nwc,nws,vxpad: that one): channel groups per block
    /// 1–4 and the largest valid, window splits 1 and 2, vxpad 0 / 1
    pub(super) fn vs_candidates(&self, g: VsGeom) -> Vec<VsCfg> {
        if let Some((c, _)) = Self::vs_env_cfg(g) {
            assert!(self.vs_cfg_ok(g, c), "HASTY_SEG_VS_CFG: {c:?} is not a valid vs shape for {g:?} on this engine");
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
                for vxpad in if self.order >= 2 { 0..1usize } else { 0..2 } {
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
        // validity as the FMA kernel (the tensor-core vt has no one-pass shape: its sub-chunks are ≤ 32)
        (1..=gr).rev().map(|nwc| VsCfg { nwc, nws: 1, vxpad: 0 }).find(|&c| self.vs_cfg_ok_m(g, c, g.smax, 0))
    }

    /// compile (disk-cached) the vs forward of a shape
    pub(super) fn vs_compile(&self, g: VsGeom, c: VsCfg) -> Result<VsActive, NufftError> {
        self.vs_compile_v(g, c, g.smax)
    }

    /// compile (disk-cached) the vs forward of a shape with sub-chunks of `sch` samples; its resources on this device
    pub(super) fn vs_compile_v(&self, g: VsGeom, c: VsCfg, sch: usize) -> Result<VsActive, NufftError> {
        self.vs_compile_m(g, c, sch, self.vs_mma())
    }

    /// as [`vs_compile_v`](Self::vs_compile_v) for contraction `mma` (1: `#define VTC 1`, module suffix `_mma`)
    pub(super) fn vs_compile_m(&self, g: VsGeom, c: VsCfg, sch: usize, mma: usize) -> Result<VsActive, NufftError> {
        let (kpre, ktag) = kernel_prefix(&self.plan);
        let (mut vdef, mut vtag) = if sch < g.smax { (format!("#define VSCH {sch}\n"), format!("_q{sch}")) } else { (String::new(), String::new()) };
        if mma >= 1 {
            vdef.push_str(&format!("#define VTC {mma}\n"));
            vtag.push_str(&format!("_mma{mma}"));
        }
        let src = format!(
            "{kpre}{vdef}#define W {}\n#define BETA {:.17e}\n#define NSEG {}\n#define CORR {}\n#define NCB {}\n#define TX {}\n#define TY {}\n#define TZ {}\n#define SMAX {}\n#define NWC {}\n#define NWS {}\n#define VXPAD {}\n#define KS {}\n#define RDEG {}\n{}",
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
        // BFP grid slots (HASTY_SEG_GRID_STORE): the window loads decode (`gload_bfp`)
        let src = format!("{}{src}", super::gbfp_prefix(self.gfft.grid_bits()));
        let name = format!("segnufft_vs{ktag}_w{}_l{}_o{}_c{}_k{}r{}_t{}x{}x{}_s{}_g{}_{}_{}{vtag}", self.plan.ker.w, self.nseg, self.order, self.cb, self.ks, self.rdeg, g.tx, g.ty, g.tz, g.smax, c.nwc, c.nws, c.vxpad);
        let module = self.dev.load_module(&name, &src, &CompileSettings::default())?;
        let smem = vs_smem_m(self.order, self.plan.ker.w, g, c, sch, mma);
        let f = tile_fn(&module, "seg_forward_vs", smem)?;
        let res = kern_res(&self.dev, &f, (32 * c.nwc * c.nws) as u32, smem, &name, &src, "seg_forward_vs");
        Ok(VsActive { geom: g, cfg: c, sch, f, smem, res, mma, ws: None })
    }

    /// whether the warp-specialised vt (VW) can run a shape: orders ≥ 2, sm_80+ (cp.async), nws 1, vxpad 0,
    /// f32 grid slots, whole coils per block row, ring ≥ sub-chunk, ≤ 1024 threads, shared memory
    pub(super) fn vw_ok(&self, g: VsGeom, c: VsCfg, sch: usize, v: VwCfg) -> bool {
        let sm: u32 = self.dev.arch.trim_start_matches("sm_").trim_end_matches(['a', 'f']).parse().unwrap_or(0);
        let gr = self.vs_groups(self.cb);
        let rows = gr.div_ceil(c.nwc.max(1));
        if v.shared_build() {
            return self.order >= 2
                && self.gfft.grid_bits() == 0
                && c.nws == 1
                && c.vxpad == 0
                && c.nwc >= 1
                && c.nwc <= gr
                && (rows == 1 || (16 * c.nwc) % self.nseg == 0)
                && v.nbw >= 1
                && v.nwt >= sch
                && v.nwt <= 255
                && 32 * (c.nwc + v.nbw) <= 1024
                && (1..=g.smax).contains(&sch)
                && vw_smem(self.order, self.plan.ker.w, g, c, sch, v) <= self.smax;
        }
        self.order >= 2
            && sm >= 80
            && self.gfft.grid_bits() == 0
            && c.nws == 1
            && c.vxpad == 0
            && c.nwc >= 1
            && c.nwc <= gr
            && (rows == 1 || (16 * c.nwc) % self.nseg == 0)
            && v.ncw >= 1
            && v.nbw >= 1
            && (1..=5).contains(&v.nst)
            && v.ncp >= 1
            && v.nwt >= sch
            && v.threads() <= 1024
            && (1..=g.smax).contains(&sch)
            && vw_smem(self.order, self.plan.ker.w, g, c, sch, v) <= self.smax
    }

    /// compile (disk-cached) the warp-specialised vt forward of a shape (`#define VW 1`)
    pub(super) fn vs_compile_w(&self, g: VsGeom, c: VsCfg, sch: usize, v: VwCfg) -> Result<VsActive, NufftError> {
        let (kpre, ktag) = kernel_prefix(&self.plan);
        let mut vdef = if sch < g.smax { format!("#define VSCH {sch}\n") } else { String::new() };
        if v.shared_build() {
            // register cap for 2 blocks per SM (the vt window needs ≈ 200 without the build), at most 255
            let mr = (65536 / (2 * 32 * (c.nwc + v.nbw)) / 8 * 8).min(255);
            vdef.push_str(&format!("#define VW 2\n#define VWB {}\n#define VWT {}\n#define VWMR {mr}\n", v.nbw, v.nwt));
        } else {
            vdef.push_str(&format!("#define VW 1\n#define VWC {}\n#define VWB {}\n#define VWS {}\n#define VWT {}\n#define VWP {}\n", v.ncw, v.nbw, v.nst, v.nwt, v.ncp));
        }
        let src = format!(
            "{kpre}{vdef}#define W {}\n#define BETA {:.17e}\n#define NSEG {}\n#define CORR {}\n#define NCB {}\n#define TX {}\n#define TY {}\n#define TZ {}\n#define SMAX {}\n#define NWC {}\n#define NWS 1\n#define VXPAD 0\n#define KS {}\n#define RDEG {}\n{}",
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
            self.ks,
            self.rdeg,
            include_str!("../segnufft_vs.cu")
        );
        let name = format!(
            "segnufft_vw{ktag}_w{}_l{}_o{}_c{}_k{}r{}_t{}x{}x{}_s{}_g{}_q{sch}_r{}b{}s{}t{}p{}",
            self.plan.ker.w, self.nseg, self.order, self.cb, self.ks, self.rdeg, g.tx, g.ty, g.tz, g.smax, c.nwc, v.ncw, v.nbw, v.nst, v.nwt, v.ncp
        );
        let module = self.dev.load_module(&name, &src, &CompileSettings::default())?;
        let smem = vw_smem(self.order, self.plan.ker.w, g, c, sch, v);
        let f = tile_fn(&module, "seg_forward_vs", smem)?;
        let nt = if v.shared_build() { 32 * (c.nwc + v.nbw) } else { v.threads() };
        let res = kern_res(&self.dev, &f, nt as u32, smem, &name, &src, "seg_forward_vs");
        Ok(VsActive { geom: g, cfg: VsCfg { nws: 1, vxpad: 0, ..c }, sch, f, smem, res, mma: 0, ws: Some(v) })
    }

    /// Use the warp-specialised vt forward (research/sample_kernels.md §22) with channel slices `c.nwc`, sub-chunk `sch`
    /// (0: smax) and roles `v` on the current geometry.
    pub fn set_vs_w(&mut self, c: VsCfg, sch: usize, v: VwCfg) -> Result<(), NufftError> {
        let g = self.vs_geom.ok_or_else(|| NufftError::Invalid("set_vs_w: no vs geometry on this engine".into()))?;
        let sch = if sch == 0 { g.smax } else { sch };
        if !self.vw_ok(g, c, sch, v) {
            return Err(NufftError::Invalid(format!("set_vs_w: {c:?} {v:?} (sub-chunk {sch}) is not a valid vw shape for {g:?}")));
        }
        self.vs = Some(self.vs_compile_w(g, c, sch, v)?);
        Ok(())
    }

    /// the active vs forward's vw roles (None: not the warp-specialised kernel)
    pub fn vs_active_w(&self) -> Option<VwCfg> {
        self.vs.as_ref().and_then(|a| a.ws)
    }

    /// Select whether the autotune times the warp-specialised vt (0 never, 1 only it, 2 next to the FMA vt)
    pub fn set_vt_ws(&mut self, m: usize) {
        assert!(m <= 2, "set_vt_ws: 0 | 1 | 2");
        self.vt_ws = m;
    }

    /// HASTY_SEG_VW_CFG = nwc,ncw,nbw,nst,sch[,ncp] (sch 0: smax; ncp default 1; ncw 0: VW 2): one vw shape for the autotune
    pub(super) fn vw_env_cfg(g: VsGeom) -> Option<(VsCfg, usize, VwCfg)> {
        let v = std::env::var("HASTY_SEG_VW_CFG").ok()?;
        let p: Vec<usize> = v.split(',').map(|x| x.trim().parse().expect("HASTY_SEG_VW_CFG = nwc,ncw,nbw,nst,sch[,ncp]")).collect();
        assert!(p.len() == 5 || p.len() == 6, "HASTY_SEG_VW_CFG = nwc,ncw,nbw,nst,sch[,ncp]");
        let sch = if p[4] == 0 { g.smax } else { p[4] };
        let ncp = if p[1] == 0 { 0 } else { *p.get(5).unwrap_or(&1) };
        Some((VsCfg { nwc: p[0], nws: 1, vxpad: 0 }, sch, VwCfg { ncw: p[1], nbw: p[2], nst: p[3], nwt: sch, ncp }))
    }

    /// vw shapes the autotune times: all channel slices in one block row (else the largest valid), roles VW_ROLES,
    /// 2 window stages, the largest of the sub-chunks smax / 32 / 24 / 16 that fits; ring = sub-chunk
    pub(super) fn vw_candidates(&self, g: VsGeom) -> Vec<(VsCfg, usize, VwCfg)> {
        if let Some(e) = Self::vw_env_cfg(g) {
            assert!(self.vw_ok(g, e.0, e.1, e.2), "HASTY_SEG_VW_CFG: {e:?} is not a valid vw shape for {g:?}");
            return vec![e];
        }
        let gr = self.vs_groups(self.cb);
        let mut v = Vec::new();
        let mut schs: Vec<usize> = [g.smax, 32, 24, 16].iter().map(|&q| q.min(g.smax)).collect();
        schs.dedup();
        let fits = |c: VsCfg, ncw: usize, nbw: usize, ncp: usize| schs.iter().copied().find(|&q| self.vw_ok(g, c, q, VwCfg { ncw, nbw, nst: 2, nwt: q, ncp }));
        let Some(nwc) = (1..=gr).rev().find(|&n| fits(VsCfg { nwc: n, nws: 1, vxpad: 0 }, 5, 1, 2).is_some()) else { return v };
        let c = VsCfg { nwc, nws: 1, vxpad: 0 };
        for &(ncw, nbw, ncp) in VW_ROLES.iter() {
            if let Some(sch) = fits(c, ncw, nbw, ncp) {
                v.push((c, sch, VwCfg { ncw, nbw, nst: 2, nwt: sch, ncp }));
            }
        }
        // VW 2 (shared build, register window): 1 / 2 build warps, sub-chunks 24 / 20 / 16; only with HASTY_SEG_VT_WS=all
        // (measured 1.4–2 × slower than vt on the 4090: the register window then needs ≤ 168 registers and spills, or
        // runs one block per SM; research/sample_kernels.md §22)
        let all = std::env::var("HASTY_SEG_VT_WS").as_deref() == Ok("all");
        for nbw in if all { vec![1usize, 2] } else { vec![] } {
            for sch in [24usize, 20, 16] {
                let sch = sch.min(g.smax);
                let w = VwCfg { ncw: 0, nbw, nst: 0, nwt: sch, ncp: 0 };
                if self.vw_ok(g, c, sch, w) && !v.contains(&(c, sch, w)) {
                    v.push((c, sch, w));
                }
            }
        }
        v
    }

    /// Resources of the vs forward of a launch shape and sub-chunk on the current geometry and this device (compiles
    /// it; disk-cached). None: no geometry or invalid.
    pub fn vs_resources(&self, c: VsCfg, sch: usize) -> Result<Option<KernRes>, NufftError> {
        let Some(g) = self.vs_geom else { return Ok(None) };
        if !self.vs_cfg_ok_v(g, c, sch) {
            return Ok(None);
        }
        Ok(Some(self.vs_compile_v(g, c, sch)?.res))
    }

    /// Sub-chunk sizes the autotune adds for a compiled one-pass shape: per block count above the one-pass shape's
    /// that shared memory allows (up to the register-limited count), the largest sub-chunk (≥ 8) reaching it.
    pub(super) fn vs_sch_plan(&self, a: &VsActive) -> Vec<usize> {
        let g = a.geom;
        match vs_sch_env() {
            Some(0) => return vec![],
            Some(n) => return if n < g.smax && self.vs_cfg_ok_v(g, a.cfg, n) { vec![n] } else { vec![] },
            None => {}
        }
        let nt = (32 * a.cfg.nwc * a.cfg.nws) as u32;
        let attr = cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
        if a.f.set_attribute(attr, self.smax as i32).is_err() {
            return vec![];
        }
        let occ = |sch: usize| a.f.occupancy_max_active_blocks_per_multiprocessor(nt, vs_smem_v(self.order, self.plan.ker.w, g, a.cfg, sch), None).unwrap_or(0);
        let breg = a.f.occupancy_max_active_blocks_per_multiprocessor(nt, 0, None).unwrap_or(0);
        let mut v: Vec<usize> = Vec::new();
        let mut have = occ(g.smax);
        for sch in (8..g.smax).rev() {
            if have >= breg {
                break;
            }
            let b = occ(sch);
            if b > have && self.vs_cfg_ok_v(g, a.cfg, sch) {
                v.push(sch);
                have = b;
            }
        }
        // restore the launch limit of the one-pass shape
        let _ = a.f.set_attribute(attr, a.smem as i32);
        v
    }

    /// engine construction: the default geometry when vs is a candidate (HASTY_SEG_VS unset / auto / 1) and fits this
    /// engine; with HASTY_SEG_VS=1 the default launch shape (or HASTY_SEG_VS_CFG) is active from the start
    pub(super) fn vs_init(&mut self) -> Result<(), NufftError> {
        let mode = vs_mode_order(self.order);
        if mode == VsMode::Off {
            return Ok(());
        }
        let g = self.vs_default_geom();
        let Some(c) = self.vs_default_cfg(g).filter(|_| self.vs_geom_ok(g)) else { return Ok(()) };
        self.vs_geom = Some(g);
        if mode == VsMode::On {
            // tensor-core vt (HASTY_SEG_VT_MMA bf16 | fp16) without HASTY_SEG_VS_CFG: 2 chunk splits, sub-chunks of 8
            let dflt = if self.vs_mma() >= 1 { (VsCfg { nws: 2, ..c }, 8.min(g.smax)) } else { (c, g.smax) };
            if let Some((c, sch, v)) = Self::vw_env_cfg(g).filter(|_| self.order >= 2) {
                // HASTY_SEG_VW_CFG: the warp-specialised vt from the start (research/sample_kernels.md §22)
                assert!(self.vw_ok(g, c, sch, v), "HASTY_SEG_VW_CFG: {c:?} {v:?} sub-chunk {sch} is not a valid vw shape for {g:?}");
                self.vs = Some(self.vs_compile_w(g, c, sch, v)?);
                return Ok(());
            }
            let (c, sch) = Self::vs_env_cfg(g).unwrap_or(dflt);
            assert!(self.vs_cfg_ok_v(g, c, sch), "HASTY_SEG_VS_CFG: {c:?} sub-chunk {sch} is not a valid vs shape for {g:?}");
            self.vs = Some(self.vs_compile_v(g, c, sch)?);
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
        self.set_vs_v(c, 0)
    }

    /// [`set_vs`](Self::set_vs) with items staged in sub-chunks of `sch` samples (0: the geometry's smax, one pass)
    pub fn set_vs_v(&mut self, c: Option<VsCfg>, sch: usize) -> Result<(), NufftError> {
        let Some(c) = c else {
            self.vs = None;
            return Ok(());
        };
        let g = self.vs_geom.ok_or_else(|| NufftError::Invalid("set_vs: no vs geometry on this engine (set_vs_geom, HASTY_SEG_VS)".into()))?;
        let sch = if sch == 0 { g.smax } else { sch };
        if !self.vs_cfg_ok_v(g, c, sch) {
            return Err(NufftError::Invalid(format!("set_vs: {c:?} (sub-chunk {sch}) is not a valid launch shape for {g:?}")));
        }
        self.vs = Some(self.vs_compile_v(g, c, sch)?);
        Ok(())
    }

    /// the active vs forward (geometry, launch shape); None: the tile kernels
    pub fn vs_active(&self) -> Option<(VsGeom, VsCfg)> {
        self.vs.as_ref().map(|a| (a.geom, a.cfg))
    }

    /// the active vs forward with its sub-chunk and resources on this device
    pub fn vs_active_v(&self) -> Option<(VsGeom, VsCfg, usize, KernRes)> {
        self.vs.as_ref().map(|a| (a.geom, a.cfg, a.sch, a.res))
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
        // order 3: trailing degree-3/4 table pointers (0 without `points_hi` / `points2d_hi` data: the order-2 operator)
        let (uw, ht) = self.hi_ptrs(st, pts);
        let (yp, _gy) = y.device_ptr_mut(st);
        unsafe {
            let mut lb = st.launch_builder(&a.f);
            lb.arg(&pts.rec)
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
                .arg(&d);
            if self.order >= 3 {
                lb.arg(&uw).arg(&ht);
            }
            lb.launch(LaunchConfig { grid_dim: ((k1 - k0) as u32, rows as u32, 1), block_dim: (a.threads() as u32, 1, 1), shared_mem_bytes: a.smem as u32 })?;
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
        let log = std::env::var("HASTY_SEG_VS_LOG").is_ok_and(|v| v != "0");
        let skip = spill_skip();
        let mut time = |a: &VsActive| -> Result<f32, NufftError> {
            let _note = super::tune::Noted(std::time::Instant::now());
            self.launch_vs(st, a, seg, pts, t, i, g, &mut *ys, 0, ncb)?;
            let mut ms = f32::INFINITY;
            for _ in 0..3 {
                ev.0.record(st)?;
                self.launch_vs(st, a, seg, pts, t, i, g, &mut *ys, 0, ncb)?;
                ev.1.record(st)?;
                ev.1.synchronize()?;
                ms = ms.min(ev.0.elapsed_ms(&ev.1)?);
            }
            Ok(ms)
        };
        let show = |a: &VsActive, what: &str| {
            if log {
                eprintln!("    [vs] {geom:?} {:?} sch {}: {what} (smem {} B, regs {}, local {} B, {} warps / SM)", a.cfg, a.sch, a.smem, a.res.regs, a.res.local, a.res.warps);
            }
        };
        let mut timed: Vec<(f32, VsActive)> = Vec::new();
        let mut spilled: Vec<VsActive> = Vec::new();
        let env = Self::vs_env_cfg(geom);
        if let Some((c, sch)) = env.filter(|e| e.1 < geom.smax) {
            assert!(self.vs_cfg_ok_v(geom, c, sch), "HASTY_SEG_VS_CFG: {c:?} sub-chunk {sch} is not a valid vs shape for {geom:?}");
            let a = self.vs_compile_v(geom, c, sch)?;
            let ms = time(&a)?;
            show(&a, &format!("{ms:.3} ms"));
            return Ok(Some((ms, a)));
        }
        let fma = (!(1..=2).contains(&self.vt_mma) || self.order < 2) && !(self.vt_ws == 1 && self.order >= 2);
        for c in if fma { self.vs_candidates(geom) } else { Vec::new() } {
            let a = self.vs_compile_m(geom, c, geom.smax, 0)?;
            if skip && a.res.spill > 0 {
                show(&a, "skipped: spills on this device");
                spilled.push(a);
                continue;
            }
            let ms = time(&a)?;
            show(&a, &format!("{ms:.3} ms"));
            timed.push((ms, a));
        }
        if env.is_none() && fma {
            // sub-chunk staging for the two fastest one-pass shapes
            timed.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut extra: Vec<(VsCfg, usize)> = Vec::new();
            for (_, a) in timed.iter().take(2) {
                for sch in self.vs_sch_plan(a) {
                    extra.push((a.cfg, sch));
                }
            }
            for (c, sch) in extra {
                let a = self.vs_compile_m(geom, c, sch, 0)?;
                if skip && a.res.spill > 0 {
                    show(&a, "skipped: spills on this device");
                    continue;
                }
                let ms = time(&a)?;
                show(&a, &format!("{ms:.3} ms"));
                timed.push((ms, a));
            }
        }
        let mk = if self.vt_mma == 3 { 2 } else { self.vt_mma };
        if env.is_none() && mk >= 1 && self.order >= 2 {
            // tensor-core vt (research/sample_kernels.md §21): 16-channel groups min(4, all) and the largest valid,
            // 1 / 2 chunk splits, sub-chunks VTC_SCH
            let gr = self.vs_groups(self.cb);
            let mut nwcs = vec![gr.min(4)];
            if let Some(n) = (1..=gr).rev().find(|&n| VTC_SCH.iter().any(|&q| self.vs_cfg_ok_m(geom, VsCfg { nwc: n, nws: 1, vxpad: 0 }, q, mk))) {
                nwcs.push(n);
            }
            nwcs.sort_unstable();
            nwcs.dedup();
            for &nwc in &nwcs {
                for nws in [1usize, 2] {
                    for &sch in VTC_SCH.iter() {
                        let c = VsCfg { nwc, nws, vxpad: 0 };
                        if sch > geom.smax || !self.vs_cfg_ok_m(geom, c, sch, mk) {
                            continue;
                        }
                        let a = self.vs_compile_m(geom, c, sch, mk)?;
                        if skip && a.res.spill > 0 {
                            show(&a, "mma: skipped: spills on this device");
                            continue;
                        }
                        let ms = time(&a)?;
                        show(&a, &format!("mma {ms:.3} ms"));
                        timed.push((ms, a));
                    }
                }
            }
        }
        if env.is_none() && self.vt_ws >= 1 && self.order >= 2 {
            // warp-specialised vt (research/sample_kernels.md §22)
            for (c, sch, v) in self.vw_candidates(geom) {
                let a = self.vs_compile_w(geom, c, sch, v)?;
                if skip && a.res.spill > 0 {
                    show(&a, &format!("vw {v:?}: skipped: spills on this device"));
                    continue;
                }
                let ms = time(&a)?;
                show(&a, &format!("vw {v:?} {ms:.3} ms"));
                timed.push((ms, a));
            }
        }
        if timed.is_empty() {
            for a in spilled {
                let ms = time(&a)?;
                show(&a, &format!("{ms:.3} ms (spills)"));
                timed.push((ms, a));
            }
        }
        Ok(timed.into_iter().min_by(|a, b| a.0.total_cmp(&b.0)))
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
