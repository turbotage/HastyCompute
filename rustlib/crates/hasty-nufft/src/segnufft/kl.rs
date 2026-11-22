//! Knot-path layout (research/sample_kernels.md §17): what a knot launch of the knot-major operator does with the
//! tiles of the fine grid, on top of the engine's adjoint (ra or the tile kernel; nothing of those is changed).
//! * tile-list statistics of a sample set (`SegPoints::tile_stats`, opbench);
//! * skip: no blocks for tiles without samples, `seg_zero_empty` for those of them that the pruned FFT reads;
//! * kl kernel (segnufft_kl_adj.cu): ra on whole tiles (no split atomics / zeroing), windows by binary search.
//! The layout that stages every sample once for all knots it touches was counted and measured and not built (§17.2:
//! at a fixed grid budget it needs more sample visits, not fewer).

use super::*;
use super::ra::{kern_res, lbmin_set, spill_skip, stage_mode, KernRes, RaVar, StageMode};

/// Statistics of the TT³ tile lists of a sample set ([`SegPoints::tile_stats`]).
#[derive(Clone, Debug, Default)]
pub struct TileStats {
    /// tiles of the fine grid, work items, items of tiles with no sample at all (zero-count items: stored as zeros by
    /// the adjoint), split items
    pub tiles: usize,
    pub items: usize,
    pub zero_items: usize,
    pub split_items: usize,
    /// list entries (sample × tile pairs) over all items, and per nonempty tile
    pub entries: usize,
    pub entries_per_tile: f64,
    /// knot engines: per knot window, the list entries and the tiles whose window is empty although the tile has
    /// samples (min / mean / max over knots)
    pub win_entries: (usize, f64, usize),
    pub win_empty_tiles: (usize, f64, usize),
}

impl SegPoints {
    /// Tile-list statistics (device → host copy of the items and lists; diagnostics only). `nint` / `ks` / `kr`/
    /// `rdeg` describe the knot windows (engine fields `mc`, `ks`, `kr`, `rdeg`); ks 0: no window statistics.
    pub fn tile_stats(&self, st: &Arc<CudaStream>, nf: [usize; 3], tt: usize, ks: usize, kr: usize, rdeg: i32) -> Result<TileStats, NufftError> {
        let iv = self.ivals[0];
        let items: Vec<i32> = st.clone_dtoh(&self.items.slice(4 * iv.first..4 * (iv.first + iv.count)))?;
        let tl: Vec<u32> = st.clone_dtoh(&self.tl)?;
        let ntiles = nf.iter().map(|n| n.div_ceil(tt)).product::<usize>();
        let mut s = TileStats { tiles: ntiles, items: iv.count, ..Default::default() };
        let mut nonempty = 0usize;
        for it in items.chunks(4) {
            if it[2] == 0 {
                s.zero_items += 1;
            }
            if it[3] != 0 {
                s.split_items += 1;
            }
            if it[2] > 0 && it[3] != 1 {
                nonempty += 1;
            }
            s.entries += it[2] as usize;
        }
        s.entries_per_tile = s.entries as f64 / nonempty.max(1) as f64;
        if ks == 0 || self.knot == 0 {
            return Ok(s);
        }
        // knot windows: keys (interval, or ri · nint + ic) of every entry, per tile
        let nint = self.knot as usize;
        let koff = -(((ks as i64) - 2) / 2);
        let nknots = if kr > 0 { kr * nint } else { nint };
        let in_win = |key: usize, b: usize| -> bool {
            let (k, bc) = if kr > 0 { (b / nint, b % nint) } else { (0, b) };
            let (ri, ic) = if kr > 0 { (key / nint, key % nint) } else { (0, key) };
            let o = ((bc as i64 - koff - ic as i64).rem_euclid(nint as i64)) as usize;
            let card = ks >= nint || o < ks;
            let resp = kr == 0 || if rdeg == 0 { ri == k } else { ri == k || ri + 1 == k };
            card && resp
        };
        let shift = 25;
        // per tile: key histogram (tiles with split items: summed over their items)
        let mut tile_keys: std::collections::HashMap<i32, Vec<u32>> = std::collections::HashMap::new();
        let nkeys = if kr > 0 { self.nri as usize * nint } else { nint };
        for it in items.chunks(4) {
            if it[2] == 0 {
                continue;
            }
            let h = tile_keys.entry(it[0]).or_insert_with(|| vec![0u32; nkeys]);
            for &e in &tl[it[1] as usize..(it[1] + it[2]) as usize] {
                h[(e >> shift) as usize] += 1;
            }
        }
        let (mut we, mut wt) = (Vec::with_capacity(nknots), Vec::with_capacity(nknots));
        let hs: Vec<&Vec<u32>> = tile_keys.values().collect();
        for b in 0..nknots {
            let win: Vec<usize> = (0..nkeys).filter(|&key| in_win(key, b)).collect();
            let (mut e, mut t) = (0usize, 0usize);
            for h in &hs {
                let n: u32 = win.iter().map(|&k| h[k]).sum();
                e += n as usize;
                t += (n == 0) as usize;
            }
            we.push(e);
            wt.push(t);
        }
        let mm = |v: &[usize]| (*v.iter().min().unwrap(), v.iter().sum::<usize>() as f64 / v.len() as f64, *v.iter().max().unwrap());
        s.win_entries = mm(&we);
        s.win_empty_tiles = mm(&wt);
        Ok(s)
    }
}

// ---------------------------------------------------------------- knot-path layout: switches and state
//
// HASTY_SEG_KL = 0 | 1 | auto (default auto). The knot-path adjoint
//   * launches no blocks for the tiles that have no sample at all (zero-count items, ≈ 35 % of the 2 mm thorax tiles:
//     outside the trajectory's k-space ball). The k-support pruned grid FFT never reads their cells (every column's z
//     range is the hull of its nonempty tiles); the zero-count tiles it does read (inside a column hull of the
//     engine's support union, e.g. tiles only another set's samples touch) are zeroed by `seg_zero_empty` first;
//   * optionally runs the whole-tile kl kernel (one block per tile, the window read in place): every tile stored by
//     one block, no split-tile zeroing or atomics.
// The §23 contract stays: tiles with samples are stored (split tiles: zeroed + atomics, or one owner with the kl
// kernel). 1: skip + kl kernel (default shape, or HASTY_SEG_KL_CFG = nxg,cb,chk) whenever the engine can run them
// (knot-major, pruned FFT); auto: the autotune keeps, per engine, the fastest of {all items, skip, skip + best kl
// shape} on knot 0; 0: never (the previous path). HASTY_SEG_KL_LOG = 1: per-shape times.

/// env HASTY_SEG_KL
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum KlMode {
    Off,
    Auto,
    On,
}

pub(super) fn kl_mode() -> KlMode {
    match std::env::var("HASTY_SEG_KL").as_deref() {
        Ok("0") | Ok("off") => KlMode::Off,
        Ok("1") | Ok("on") => KlMode::On,
        Ok("auto") | Ok("") | Err(_) => KlMode::Auto,
        Ok(v) => panic!("HASTY_SEG_KL={v}: 0 | 1 | auto"),
    }
}

/// the active kl adjoint kernel (segnufft_kl_adj.cu) of a launch shape and staging variant (`RaVar::inplace` is
/// ignored: kl always reads its windows in place)
pub(super) struct KlKernel {
    pub cfg: RaCfg,
    pub var: RaVar,
    pub f: CudaFunction,
    pub smem: usize,
    pub res: KernRes,
}

#[derive(Default)]
pub(super) struct KlState {
    /// skip the zero-count items in the knot adjoint
    pub skip: bool,
    /// the whole-tile kl adjoint (implies the skip); None: the engine's adjoint (ra or tile kernel)
    pub kernel: Option<KlKernel>,
    /// `seg_tile_count` and the per-tile totals of the set being scattered
    count_f: Option<CudaFunction>,
    tcount: Option<CudaSlice<i32>>,
    /// `seg_zero_empty` of this engine's channel count
    zero: Option<CudaFunction>,
    /// per tile column the kz hull (h, t) in cells of the support union with its fingerprint
    hull: Option<(u64, CudaSlice<i32>)>,
    /// the last autotune's knot-0 adjoint times (ms, split-tile zeroing included): (engine adjoint over all items, the
    /// same skipping empty tiles, the best kl kernel or NaN) and that kernel's shape
    pub tune: Option<(f32, f32, f32, Option<RaCfg>)>,
}

impl SegNufft {
    /// whether the engine can run the knot-path skip: knot-major (one slot), k-support pruned grid FFT, plain path
    pub(super) fn kl_ok(&self) -> bool {
        self.ks > 0 && self.nf == 1 && self.kprune && self.gfft.is_pruned() && !self.wf && self.warp.is_none() && self.tiles_adj.tt == 4
    }

    /// engine construction: with HASTY_SEG_KL=1 the skip and the kl kernel of the default shape (or HASTY_SEG_KL_CFG)
    pub(super) fn kl_init(&mut self) -> Result<(), NufftError> {
        self.kl.skip = kl_mode() == KlMode::On;
        if self.kl.skip && self.ks > 0 && self.nf == 1 && self.tiles_adj.tt == 4 {
            if let Some((c, v)) = Self::kl_env_cfg() {
                assert!(self.kl_cfg_ok_v(c, v), "HASTY_SEG_KL_CFG: {c:?} {v:?} is not a valid kl shape on this engine");
                self.kl.kernel = Some(self.kl_compile_v(c, v)?);
            } else if let Some(c) = self.kl_default_cfg() {
                self.kl.kernel = Some(self.kl_compile(c)?);
            }
        }
        Ok(())
    }

    /// the default kl shape (the ra autotune's usual best on the knot path, research/sample_kernels.md §16): four x
    /// groups, the fewest block rows, 16-sample chunks
    pub(super) fn kl_default_cfg(&self) -> Option<RaCfg> {
        let nchv = self.cb * self.nseg;
        (1..=64).map(|rows| RaCfg { nxg: 4, cb: nchv.div_ceil(8 * rows), chk: 16 }).find(|&c| self.kl_cfg_ok(c))
    }

    /// whether the next knot adjoint runs the skip path (the kl kernel implies it)
    pub(super) fn kl_skip_active(&self) -> bool {
        (self.kl.skip || self.kl.kernel.is_some()) && self.kl_ok()
    }

    /// whether the next knot adjoint runs the whole-tile kl kernel (no split-tile zeroing)
    pub(super) fn kl_kernel_active(&self) -> bool {
        self.kl.kernel.is_some() && self.kl_ok()
    }

    /// Turn the knot-path skip of empty tiles on / off for the engine's adjoint (it runs only where
    /// [`kl_ok`](Self::kl_ok): knot-major, pruned FFT). `set_kl(false)` also drops the kl kernel: the previous path.
    pub fn set_kl(&mut self, on: bool) {
        self.kl.skip = on;
        if !on {
            self.kl.kernel = None;
        }
    }

    /// Use the whole-tile kl adjoint with launch shape `c` (implies the skip), or None for the engine's adjoint (the
    /// skip stays as set). Err if the engine cannot run it or the shape is invalid.
    pub fn set_kl_kernel(&mut self, c: Option<RaCfg>) -> Result<(), NufftError> {
        self.set_kl_kernel_v(c, RaVar::BASE)
    }

    /// [`set_kl_kernel`](Self::set_kl_kernel) with a staging variant (stages, lbmin; `inplace` ignored)
    pub fn set_kl_kernel_v(&mut self, c: Option<RaCfg>, v: RaVar) -> Result<(), NufftError> {
        let Some(c) = c else {
            self.kl.kernel = None;
            return Ok(());
        };
        if !self.kl_ok() || self.order > 3 {
            return Err(NufftError::Invalid("set_kl_kernel: knot-major engines with the pruned grid FFT only".into()));
        }
        if !self.kl_cfg_ok_v(c, v) {
            return Err(NufftError::Invalid(format!("set_kl_kernel: {c:?} {v:?} is not a valid launch shape on this engine")));
        }
        self.kl.kernel = Some(self.kl_compile_v(c, v)?);
        Ok(())
    }

    /// the active kl kernel with its staging variant and resources on this device
    pub fn kl_kernel_v(&self) -> Option<(RaCfg, RaVar, KernRes)> {
        self.kl.kernel.as_ref().filter(|_| self.kl_ok()).map(|k| (k.cfg, k.var, k.res))
    }

    /// Resources of the kl kernel of a shape and staging variant on this device (compiles it; disk-cached). None:
    /// invalid here.
    pub fn kl_resources(&self, c: RaCfg, v: RaVar) -> Result<Option<KernRes>, NufftError> {
        if self.ks == 0 || !self.kl_cfg_ok_v(c, v) {
            return Ok(None);
        }
        Ok(Some(self.kl_compile_v(c, v)?.res))
    }

    /// whether the knot-path skip is on (and applicable)
    pub fn kl_active(&self) -> bool {
        self.kl_skip_active()
    }

    /// the active kl kernel shape (None: the engine's adjoint, with or without the skip)
    pub fn kl_kernel(&self) -> Option<RaCfg> {
        self.kl.kernel.as_ref().filter(|_| self.kl_ok()).map(|k| k.cfg)
    }

    /// the last autotune's knot-0 adjoint times (ms, split-tile zeroing included): the engine adjoint (ra or tile
    /// kernel) over all items, the same skipping empty tiles, the best kl kernel (NaN: none ran)
    pub fn kl_tune_times(&self) -> Option<(f32, f32, f32)> {
        self.kl.tune.map(|t| (t.0, t.1, t.2))
    }

    /// the last autotune's fastest kl kernel shape (whether or not it was kept)
    pub fn kl_tune_cfg(&self) -> Option<RaCfg> {
        self.kl.tune.and_then(|t| t.3)
    }

    fn kl_fp(&self) -> u64 {
        // FNV-1a over the support union's bit words
        let mut h: u64 = 0xcbf29ce484222325;
        for &w in &self.kcols.zt {
            h = (h ^ w).wrapping_mul(0x100000001b3);
        }
        h ^ (self.kcols.zt.len() as u64)
    }

    /// the zero / count module and the device hull table of the current support union (rebuilt when it grows)
    fn kl_prepare(&mut self, st: &Arc<CudaStream>) -> Result<(), NufftError> {
        if self.kl.zero.is_none() {
            let nch = self.cb * self.nseg;
            let src = format!("#define TT {}\n#define NCH {}\n{}", self.tiles_adj.tt, nch, include_str!("../segnufft_kl.cu"));
            let module = self.dev.load_module(&format!("segnufft_kl_t{}_n{}", self.tiles_adj.tt, nch), &src, &CompileSettings::default())?;
            self.kl.zero = Some(module.load_function("seg_zero_empty")?);
            self.kl.count_f = Some(module.load_function("seg_tile_count")?);
        }
        let fp = self.kl_fp();
        if self.kl.hull.as_ref().is_none_or(|h| h.0 != fp) {
            let (tt, nf, k) = (self.tiles_adj.tt, self.plan.nf, &self.kcols);
            let nt = nf.map(|n| n.div_ceil(tt));
            let mut hv = vec![0i32; 2 * nt[0] * nt[1]];
            if !k.zt.is_empty() {
                for c in 0..nt[0] * nt[1] {
                    let w = &k.zt[c * k.wz..(c + 1) * k.wz];
                    if w.iter().all(|&x| x == 0) {
                        continue;
                    }
                    // KCols::support: the wrapped hull of the column's nonempty z tiles, in cells
                    let (h, t) = KSupport::hull(nf[2], |z| (w[z / tt / 64] >> ((z / tt) % 64)) & 1 == 1);
                    hv[2 * c] = h as i32;
                    hv[2 * c + 1] = t as i32;
                }
            }
            self.kl.hull = Some((fp, st.clone_htod(&hv)?));
        }
        Ok(())
    }

    /// Before a knot adjoint's knot loop: the per-tile entry totals of `pts` for the kl kernel (when it is active).
    pub(super) fn kl_begin(&mut self, st: &Arc<CudaStream>, pts: &SegPoints) -> Result<(), NufftError> {
        if !self.kl_kernel_active() {
            return Ok(());
        }
        self.kl_prepare(st)?;
        let iv = pts.ivals[0];
        let ntiles = self.plan.nf.iter().map(|n| n.div_ceil(self.tiles_adj.tt)).product::<usize>();
        if self.kl.tcount.as_ref().is_none_or(|t| t.len() < ntiles) {
            self.kl.tcount = Some(st.alloc_zeros::<i32>(ntiles)?);
        }
        let tc = self.kl.tcount.as_mut().unwrap();
        st.memset_zeros(&mut tc.slice_mut(0..ntiles))?;
        if iv.nnz > 0 {
            let items = pts.items.slice(4 * iv.first..4 * (iv.first + iv.nnz));
            let n_i = iv.nnz as i32;
            unsafe { st.launch_builder(self.kl.count_f.as_ref().unwrap()).arg(&items).arg(&n_i).arg(&*tc).launch(LaunchConfig { grid_dim: (iv.nnz.div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })? };
        }
        Ok(())
    }

    /// in store mode: zero the zero-count tiles of `pts` that the pruned FFT reads
    fn kl_zero_empty(&mut self, st: &Arc<CudaStream>, pts: &SegPoints, g: Grids) -> Result<(), NufftError> {
        let iv = pts.ivals[0];
        // HASTY_SEG_KL_NOZERO=1: debug control only (wrong results where another set's support reads empty tiles)
        if iv.count == iv.nnz || std::env::var("HASTY_SEG_KL_NOZERO").is_ok_and(|v| v == "1") {
            return Ok(());
        }
        self.kl_prepare(st)?;
        let items = pts.items.slice(4 * (iv.first + iv.nnz)..4 * (iv.first + iv.count));
        let (f, hull) = (self.kl.zero.as_ref().unwrap(), &self.kl.hull.as_ref().unwrap().1);
        let d = self.plan.dims();
        let gp = g.p[0];
        unsafe { st.launch_builder(f).arg(&items).arg(hull).arg(&gp).arg(&d).launch(LaunchConfig { grid_dim: ((iv.count - iv.nnz) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })? };
        Ok(())
    }

    /// Knot-path adjoint scatter of knot b (segnufft/kl.rs): in store mode zero the zero-count tiles the FFT reads, then
    /// the kl kernel (whole tiles; `kl_begin` ran for `pts`) or the engine's adjoint (ra or the tile kernel) over the
    /// items with samples only.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn scatter_knot_kl(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, b: usize, g: Grids, addmask: i32, y: &CudaSlice<C32>, c0: usize, ncb: usize) -> Result<(), NufftError> {
        if addmask & 1 == 0 {
            self.kl_zero_empty(st, pts, g)?;
        }
        if self.kl_kernel_active() {
            let k = self.kl.kernel.as_ref().unwrap();
            return self.launch_kl(st, k, seg, pts, b, g, addmask, y, c0, ncb);
        }
        self.scatter_n(st, seg, pts, b, g, addmask, y, c0, ncb, pts.ivals[0].nnz)
    }

    /// the kl kernel of knot b over the items with samples (whole tiles) for coils c0..c0+ncb
    #[allow(clippy::too_many_arguments)]
    fn launch_kl(&self, st: &Arc<CudaStream>, k: &KlKernel, seg: &SegSegments, pts: &SegPoints, b: usize, g: Grids, addmask: i32, y: &CudaSlice<C32>, c0: usize, ncb: usize) -> Result<(), NufftError> {
        let iv = pts.ivals[0];
        if iv.nnz == 0 {
            return Ok(());
        }
        let (t0, dt, ntab, m_i, c0_i, ncb_i, d) = (seg.tab.1, seg.tab.2, seg.tab.0 as i32, pts.m as i32, c0 as i32, ncb as i32, self.plan.dims());
        let (b_i, nint, nri) = (b as i32, pts.knot as i32, pts.nri as i32);
        let itab = seg.itab(b);
        let items = pts.items.slice(4 * iv.first..4 * (iv.first + iv.nnz));
        let (uw, ht) = self.hi_ptrs(st, pts);
        let tc = self.kl.tcount.as_ref().expect("kl_begin before the kl kernel");
        let rows = (ncb * self.nseg).div_ceil(k.cfg.cpw());
        unsafe {
            let mut lb = st.launch_builder(&k.f);
            lb.arg(&pts.rec)
                .arg(&pts.tl)
                .arg(&items)
                .arg(tc)
                .arg(&itab)
                .arg(&ntab)
                .arg(&t0)
                .arg(&dt)
                .arg(y)
                .arg(&m_i)
                .arg(&c0_i)
                .arg(&ncb_i)
                .arg(&g)
                .arg(&b_i)
                .arg(&nint)
                .arg(&nri)
                .arg(&addmask)
                .arg(&d);
            if self.order >= 3 {
                lb.arg(&uw).arg(&ht);
            }
            lb.launch(LaunchConfig { grid_dim: (iv.nnz as u32, rows as u32, 1), block_dim: (128, 1, 1), shared_mem_bytes: k.smem as u32 })?;
        }
        Ok(())
    }

    /// whether a kl launch shape is valid on this engine (as ra: ≤ 16 accumulators per row, shared memory)
    pub(super) fn kl_cfg_ok(&self, c: RaCfg) -> bool {
        self.kl_cfg_ok_v(c, RaVar::BASE)
    }

    pub(super) fn kl_cfg_ok_v(&self, c: RaCfg, v: RaVar) -> bool {
        let nchv = self.cb * self.nseg;
        matches!(c.nxg, 1 | 2 | 4)
            && c.cb >= 1
            && 4 / c.nxg * c.cb <= 16
            && (1..=64).contains(&c.chk)
            && nchv.div_ceil(c.cpw()) <= 65535
            && matches!(v.stages, 1 | 2)
            && v.lbmin <= 8
            && kl_smem_v(self.order, self.nseg, self.cb, c, v) + 256 <= self.smax
    }

    /// HASTY_SEG_KL_CFG = nxg,cb,chk[,stages,lbmin]
    pub(super) fn kl_env_cfg() -> Option<(RaCfg, RaVar)> {
        let v = std::env::var("HASTY_SEG_KL_CFG").ok()?;
        let p: Vec<usize> = v.split(',').map(|x| x.trim().parse().expect("HASTY_SEG_KL_CFG = nxg,cb,chk[,stages,lbmin]")).collect();
        let c = RaCfg { nxg: p[0], cb: *p.get(1).unwrap_or(&1), chk: *p.get(2).unwrap_or(&16) };
        Some((c, RaVar { inplace: false, stages: *p.get(3).unwrap_or(&1), lbmin: *p.get(4).unwrap_or(&0) }))
    }

    /// Launch shapes the autotune times (env HASTY_SEG_KL_CFG = nxg,cb,chk: that one): as ra, per x grouping the
    /// channel blocks that cover the batch in 1–4 block rows, chunks of 16 and 32.
    pub fn kl_candidates(&self) -> Vec<RaCfg> {
        if let Some((c, _)) = Self::kl_env_cfg() {
            assert!(self.kl_cfg_ok(c), "HASTY_SEG_KL_CFG: {c:?} is not a valid kl shape on this engine");
            return vec![c];
        }
        let nchv = self.cb.min(self.nc) * self.nseg;
        let mut v = Vec::new();
        for nxg in [1usize, 2, 4] {
            let ncl = 32 / nxg;
            let mut cbs: Vec<usize> = (1..=4).map(|rows| nchv.div_ceil(ncl * rows)).filter(|&cb| cb >= 1 && 4 / nxg * cb <= 16).collect();
            cbs.dedup();
            for cb in cbs {
                for chk in [16usize, 32] {
                    let c = RaCfg { nxg, cb, chk };
                    if self.kl_cfg_ok(c) && !v.contains(&c) {
                        v.push(c);
                    }
                }
            }
        }
        v
    }

    /// compile (disk-cached) the kl kernel of a shape
    pub(super) fn kl_compile(&self, c: RaCfg) -> Result<KlKernel, NufftError> {
        self.kl_compile_v(c, RaVar::BASE)
    }

    /// compile (disk-cached) the kl kernel of a shape and staging variant; its resources on this device
    pub(super) fn kl_compile_v(&self, c: RaCfg, v: RaVar) -> Result<KlKernel, NufftError> {
        let v = RaVar { inplace: false, ..v };
        let (kpre, ktag) = kernel_prefix(&self.plan);
        let vdef = if v.is_base() { String::new() } else { format!("#define RA_STAGES {}\n#define LBMIN {}\n", v.stages, v.lbmin) };
        let src = format!(
            "{kpre}{vdef}#define W {}\n#define BETA {:.17e}\n#define NSEG {}\n#define CORR {}\n#define NCB {}\n#define NXG {}\n#define CB {}\n#define CHK {}\n#define KS {}\n#define RDEG {}\n{}",
            self.plan.ker.w,
            self.plan.ker.beta,
            self.nseg,
            self.order,
            self.cb,
            c.nxg,
            c.cb,
            c.chk,
            self.ks,
            self.rdeg,
            include_str!("../segnufft_kl_adj.cu")
        );
        let name = format!("segnufft_kladj{ktag}_w{}_l{}_o{}_c{}_k{}r{}_x{}_b{}_h{}{}", self.plan.ker.w, self.nseg, self.order, self.cb, self.ks, self.rdeg, c.nxg, c.cb, c.chk, v.tag());
        let module = self.dev.load_module(&name, &src, &CompileSettings::default())?;
        let smem = kl_smem_v(self.order, self.nseg, self.cb, c, v);
        let f = tile_fn(&module, "seg_adjoint_kl", smem)?;
        let res = kern_res(&self.dev, &f, 128, smem, &name, &src, "seg_adjoint_kl");
        Ok(KlKernel { cfg: c, var: v, f, smem, res })
    }

    /// Staging variants the kl autotune adds for one (nxg, cb): chunks 8, 12, 16 and the largest chunk ≤ 32 that keeps
    /// the register-limited block count, each with the register prefetch and (HASTY_SEG_VSA_STAGE ≠ inplace) the
    /// cp.async double buffer (kl always reads in place).
    pub(super) fn kl_variant_plan(&self, nxg: usize, cb: usize) -> Result<Vec<(RaCfg, RaVar)>, NufftError> {
        let mode = stage_mode();
        if mode == StageMode::Off {
            return Ok(vec![]);
        }
        let stages: &[usize] = if mode >= StageMode::CpAsync { &[1, 2] } else { &[1] };
        let mut chks = vec![8usize, 12, 16];
        let probe = RaCfg { nxg, cb, chk: 16 };
        if self.kl_cfg_ok(probe) {
            let k = self.kl_compile(probe)?;
            if let Some(f) = super::ra::chunk_fit(&k.f, self.smax, |chk| kl_smem(self.order, self.nseg, self.cb, RaCfg { chk, ..probe })) {
                chks.push(f);
            }
        }
        chks.sort_unstable();
        chks.dedup();
        let mut v = Vec::new();
        for &chk in &chks {
            for &st in stages {
                let (c, var) = (RaCfg { nxg, cb, chk }, RaVar { inplace: false, stages: st, lbmin: 0 });
                // chunk 16 with the register prefetch is the base candidate itself
                if !(chk == 16 && st == 1) && self.kl_cfg_ok_v(c, var) {
                    v.push((c, var));
                }
            }
        }
        Ok(v)
    }

    /// the active knot adjoint (ra when active and applicable, else the tile kernel) over the first `n` items
    #[allow(clippy::too_many_arguments)]
    fn scatter_n(&self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, b: usize, g: Grids, addmask: i32, y: &CudaSlice<C32>, c0: usize, ncb: usize, n: usize) -> Result<(), NufftError> {
        if n == 0 {
            return Ok(());
        }
        let (t0, dt, ntab, m_i, c0_i, ncb_i, d) = (seg.tab.1, seg.tab.2, seg.tab.0 as i32, pts.m as i32, c0 as i32, ncb as i32, self.plan.dims());
        let (b_i, nint, nri) = (b as i32, pts.knot as i32, pts.nri as i32);
        let itab = seg.itab(b);
        let iv = pts.ivals[0];
        let items = pts.items.slice(4 * iv.first..4 * (iv.first + n));
        let (uw, ht) = self.hi_ptrs(st, pts);
        let (f, grid, block, smem) = if let Some(a) = self.ra.as_ref().filter(|_| self.ra_ok()) {
            let rows = (ncb * self.nseg).div_ceil(a.cfg.cpw());
            (&a.f, (n as u32, rows as u32, 1), (128u32, 1, 1), a.smem as u32)
        } else {
            let tc = &self.tiles_adj;
            (&self.f.adj, (n as u32, ncb.div_ceil(tc.cpba) as u32, 1), (tc.threads(self.shape()).1 as u32, 1, 1), tc.smem(self.shape())[0] as u32)
        };
        unsafe {
            let mut lb = st.launch_builder(f);
            lb.arg(&pts.rec)
                .arg(&pts.tl)
                .arg(&items)
                .arg(&itab)
                .arg(&ntab)
                .arg(&t0)
                .arg(&dt)
                .arg(y)
                .arg(&m_i)
                .arg(&c0_i)
                .arg(&ncb_i)
                .arg(&g)
                .arg(&b_i)
                .arg(&nint)
                .arg(&nri)
                .arg(&addmask)
                .arg(&d);
            if self.order >= 3 {
                lb.arg(&uw).arg(&ht);
            }
            lb.launch(LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: smem })?;
        }
        Ok(())
    }

    /// Autotune step of the knot-path layout on a representative knot (1-D: knot 0; 2-D: the first cardiac knot of
    /// the middle respiratory knot) (store mode, as the other adjoint candidates; split-tile
    /// zeroing included where the path needs it): the engine's chosen adjoint over all items, the same skipping the
    /// empty tiles, and every kl kernel shape. Keeps the fastest (HASTY_SEG_KL=1: the kl kernel's best shape even when
    /// slower; 0: never timed). Returns the kept time, None when not timed.
    pub(super) fn autotune_kl(&mut self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, g: Grids, ys: &CudaSlice<C32>, ncb: usize) -> Result<Option<f32>, NufftError> {
        if kl_mode() == KlMode::Off || pts.knot == 0 {
            return Ok(None);
        }
        let keep_kernel = self.kl.kernel.take();
        if !self.kl_ok() {
            self.kl.kernel = keep_kernel;
            return Ok(None);
        }
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let ev = (self.dev.ctx.new_event(flags)?, self.dev.ctx.new_event(flags)?);
        let split = pts.ivals[0].split && self.zero_tiles;
        // a representative knot: 2-D, an interior respiratory knot (knot 0's window holds one respiratory interval, half
        // the samples of an interior knot: on the 2 mm thorax it picked the kl kernel, slower over all knots, §17)
        let kb = self.tune_knot(true);
        let log = std::env::var("HASTY_SEG_KL_LOG").is_ok_and(|v| v != "0");
        // 0: all items, 1: skip, 2: kl kernel
        let time = |e: &mut Self, path: usize| -> Result<f32, NufftError> {
            let _note = super::tune::Noted(std::time::Instant::now());
            let mut best = f32::INFINITY;
            for rep in 0..4 {
                ev.0.record(st)?;
                match path {
                    0 => {
                        if split {
                            e.zero_split(st, pts, 0, 0)?;
                        }
                        e.scatter_n(st, seg, pts, kb, g, 0, ys, 0, ncb, pts.ivals[0].count)?;
                    }
                    1 => {
                        if split {
                            e.zero_split(st, pts, 0, 0)?;
                        }
                        e.kl_zero_empty(st, pts, g)?;
                        e.scatter_n(st, seg, pts, kb, g, 0, ys, 0, ncb, pts.ivals[0].nnz)?;
                    }
                    _ => {
                        e.kl_zero_empty(st, pts, g)?;
                        let k = e.kl.kernel.as_ref().unwrap();
                        e.launch_kl(st, k, seg, pts, kb, g, 0, ys, 0, ncb)?;
                    }
                }
                ev.1.record(st)?;
                ev.1.synchronize()?;
                if rep > 0 {
                    best = best.min(ev.0.elapsed_ms(&ev.1)?);
                }
            }
            Ok(best)
        };
        let t_all = time(self, 0)?;
        let t_skip = time(self, 1)?;
        let mut best_k: Option<(f32, KlKernel)> = None;
        if self.order <= 3 {
            self.kl_prepare(st)?;
            let skip = spill_skip();
            let show = |k: &KlKernel, what: &str| {
                if log {
                    eprintln!("    [kl] {:?} {:?}: {what} (smem {} B, regs {}, local {} B, spill {} B, {} warps / SM)", k.cfg, k.var, k.smem, k.res.regs, k.res.local, k.res.spill, k.res.warps);
                }
            };
            // times of the kernels run: (ms, shape, variant, warps); spill-screened kernels kept for the fallback
            let mut timed: Vec<(f32, RaCfg, RaVar, u32)> = Vec::new();
            let mut spilled: Vec<KlKernel> = Vec::new();
            let mut begun = false;
            let mut run = |e: &mut Self, k: KlKernel, force: bool, timed: &mut Vec<(f32, RaCfg, RaVar, u32)>, spilled: &mut Vec<KlKernel>, best_k: &mut Option<(f32, KlKernel)>| -> Result<(), NufftError> {
                if skip && !force && k.res.spill > 0 {
                    show(&k, "skipped: spills on this device");
                    spilled.push(k);
                    return Ok(());
                }
                e.kl.kernel = Some(k);
                if !begun {
                    // the per-tile totals of this set (as kl_begin, with the kernel about to be timed)
                    e.kl_begin(st, pts)?;
                    begun = true;
                }
                let t = time(e, 2)?;
                let k = e.kl.kernel.take().unwrap();
                show(&k, &format!("{t:.3} ms"));
                timed.push((t, k.cfg, k.var, k.res.warps));
                if best_k.as_ref().is_none_or(|b| t < b.0) {
                    *best_k = Some((t, k));
                }
                Ok(())
            };
            let env = Self::kl_env_cfg();
            if let Some((c, v)) = env.filter(|(_, v)| !v.is_base()) {
                // HASTY_SEG_KL_CFG with a staging variant: that one only
                assert!(self.kl_cfg_ok_v(c, v), "HASTY_SEG_KL_CFG: {c:?} {v:?} is not a valid kl shape on this engine");
                let k = self.kl_compile_v(c, v)?;
                run(self, k, true, &mut timed, &mut spilled, &mut best_k)?;
            } else {
                for c in self.kl_candidates() {
                    let k = self.kl_compile(c)?;
                    run(self, k, false, &mut timed, &mut spilled, &mut best_k)?;
                }
            }
            if env.is_none() {
                // phase B: staging variants of the two fastest (nxg, cb), then launch-bounds hints for the fastest three
                timed.sort_by(|a, b| a.0.total_cmp(&b.0));
                let mut pairs: Vec<(usize, usize)> = Vec::new();
                for c in timed.iter().map(|t| t.1).chain(spilled.iter().map(|k| k.cfg)) {
                    if !pairs.contains(&(c.nxg, c.cb)) && pairs.len() < 2 {
                        pairs.push((c.nxg, c.cb));
                    }
                }
                for (nxg, cb) in pairs {
                    for (c, v) in self.kl_variant_plan(nxg, cb)? {
                        let k = self.kl_compile_v(c, v)?;
                        run(self, k, false, &mut timed, &mut spilled, &mut best_k)?;
                    }
                }
                timed.sort_by(|a, b| a.0.total_cmp(&b.0));
                let top: Vec<(RaCfg, RaVar, u32)> = timed.iter().filter(|t| t.2.lbmin == 0).take(3).map(|t| (t.1, t.2, t.3)).collect();
                for (c, v, w) in top {
                    for m in lbmin_set() {
                        let vm = RaVar { lbmin: m, ..v };
                        if !self.kl_cfg_ok_v(c, vm) {
                            continue;
                        }
                        let k = self.kl_compile_v(c, vm)?;
                        if k.res.spill > 0 || k.res.warps < w {
                            show(&k, &format!("hint not kept: spills or fewer warps than unhinted ({w})"));
                            continue;
                        }
                        run(self, k, false, &mut timed, &mut spilled, &mut best_k)?;
                    }
                }
            }
            if timed.is_empty() {
                // every candidate spills on this device: time them all
                for k in std::mem::take(&mut spilled) {
                    run(self, k, true, &mut timed, &mut spilled, &mut best_k)?;
                }
            }
        }
        let t_k = best_k.as_ref().map_or(f32::NAN, |b| b.0);
        self.kl.tune = Some((t_all, t_skip, t_k, best_k.as_ref().map(|b| b.1.cfg)));
        let on = kl_mode() == KlMode::On;
        let (t, skip, kernel) = match best_k {
            Some((tk, k)) if on || (tk < t_skip && tk < t_all) => (tk, true, Some(k)),
            _ if on || t_skip < t_all => (t_skip, true, None),
            _ => (t_all, false, None),
        };
        let _ = keep_kernel;
        self.kl.skip = skip;
        self.kl.kernel = kernel;
        Ok(Some(t))
    }
}

/// dynamic shared memory (bytes) of a kl kernel shape (segnufft_kl_adj.cu OFF_* layout; the output park reuses it)
pub(super) fn kl_smem(order: u32, nseg: usize, ncb: usize, c: RaCfg) -> usize {
    kl_smem_v(order, nseg, ncb, c, RaVar::BASE)
}

/// as [`kl_smem`] for a staging variant (RA_STAGES 2: two buffers of entries, uw, records and coil values)
pub(super) fn kl_smem_v(order: u32, nseg: usize, ncb: usize, c: RaCfg, v: RaVar) -> usize {
    let sg = v.stages;
    let a16 = |x: usize| (x + 15) & !15;
    let nka = match order {
        0 => 1,
        1 => 3,
        _ => 5,
    };
    let (ncl, cpw) = (32 / c.nxg, c.cpw());
    let cbp = c.cb.div_ceil(2) * 2;
    let ncbrm = ((cpw + nseg - 1) / nseg + 1).min(ncb);
    let chk = c.chk;
    let se = 0;
    let ht = a16(se + sg * 4 * chk);
    let uw = a16(ht + if order >= 3 { 16 * 25 } else { 0 });
    let rec = a16(uw + if order >= 3 { sg * 16 * chk } else { 0 });
    let ry = a16(rec + sg * chk * 48);
    let ra = a16(ry + sg * chk * ncbrm * 8);
    let ax = a16(ra + chk * nseg * 8);
    let bx = a16(ax + chk * 3 * nka * 4 * 4);
    let e = a16(bx + chk * 4);
    let h = a16(e + if order >= 2 { chk * 15 * 4 } else { 0 });
    let z = a16(h + if order >= 3 { chk * 25 * 4 } else { 0 });
    let w = a16(z + chk * (cbp / 2) * ncl * 16);
    let stage = w + if order >= 1 { chk * 16 * 4 * 8 } else { 0 };
    let park = 4 * 4 * (cpw | 1) * 8;
    stage.max(park)
}

