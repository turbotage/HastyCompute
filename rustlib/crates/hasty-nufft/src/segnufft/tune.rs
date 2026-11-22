//! Per-GPU cache of the autotune's choices (research/arch_targets.md, item 0).
//!
//! [`SegNufft::autotune`] / the multi-image autotune time every kernel candidate on the given set (≈ 2–5 min per
//! engine on the 4090 at the velocity shape). The kernels come from the NVRTC disk cache, the CHOICES did not: this
//! module stores every choice an autotune makes (forward / adjoint TileCfg; the vs / vt / vw forward with geometry,
//! shape, sub-chunk, contraction and warp-specialised variant; the ra adjoint shape + staging variant + launch-bounds
//! hint; the knot-path layout and its kl kernel; the multi-image K-image shapes) with the measured kernel times, as one
//! small JSON file per key under
//!
//!   `hasty_site::tune_cache()/<gpu>-cc<maj><min>-drv<driver>-nvrtc<maj>.<min>/<key>.json`
//!
//! (default `rustlib/target/hasty-tune/…`, next to the kernel cache `rustlib/target/hasty-kernels`).
//!
//! Key = FNV-64 of a canonical description ([`key_desc`]) of everything that changes the choice:
//! - the engine shape (segments, order, coils, coil batch, frame slots, knot support / structure, rdeg, gridding kernel
//!   W / β / kind, fine grid, image grid, opt-in shared memory, start tile edge / item length, vs geometry, vt / vw
//!   modes, zero / prune / fused / fft-store / grid-store / map / diagonal storage, warp / water-fat paths);
//! - every `HASTY_SEG_*` environment switch (they change the candidate set or the kernels; only the `*_LOG` / `KINFO`
//!   logging switches are left out) and `HASTY_NVRTC_LINEINFO`;
//! - a hash of the kernel sources and of the tuning code ([`source_hash`]): an edited kernel or candidate list misses;
//! - a coarse bucket of the tuned set's table statistics ([`TabStats`]): half-octave buckets of the sample count, of
//!   the nonempty tiles and of the samples per nonempty tile, eighths of the split-item fraction, plus the knot layout
//!   and the vs items of the tuned knot. Any trajectory whose density (samples per tile) or extent (nonempty tiles)
//!   differs by ≥ √2 lands in another bucket; within a bucket the candidates' times scale together (the kernels are
//!   per-tile work loops), so the order of the candidates, i.e. the choice, does not change.
//!
//! Switch `HASTY_TUNE` = `reuse` (default: load a matching choice, else time and store) | `retune` (time again and
//! overwrite) | `off` (never read or write: exactly the previous behaviour). One log line per autotune says hit / miss
//! / retune / stale with the key. A loaded choice is validated against the engine (shapes fit the shared memory, the
//! variant is valid here, the tables match); otherwise the engine retunes. Files are written to a temporary name and
//! renamed (concurrent writers: last complete file wins, readers never see a partial one); an unreadable file is a miss.
//! Within one process an in-memory copy serves repeated constructions with the same key without touching the disk.

use super::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

/// format of the JSON files (bump when the record layout changes)
const FORMAT: u32 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TuneMode {
    Reuse,
    Retune,
    Off,
}

/// env HASTY_TUNE = reuse (default) | retune | off
pub fn tune_mode() -> TuneMode {
    match std::env::var("HASTY_TUNE").as_deref().map(str::trim) {
        Err(_) | Ok("") | Ok("reuse") | Ok("1") => TuneMode::Reuse,
        Ok("retune") => TuneMode::Retune,
        Ok("off") | Ok("0") => TuneMode::Off,
        Ok(v) => panic!("HASTY_TUNE={v}: reuse | retune | off"),
    }
}

// ---- timing-launch counters (instrumentation: where setup time goes; the tests check "no timing on a hit") ----

static TIMED: AtomicU64 = AtomicU64::new(0);
static TIMED_NS: AtomicU64 = AtomicU64::new(0);

/// one timed candidate (its warm-up + repetitions) that started at `t0`
pub(super) fn note_timed(t0: std::time::Instant) {
    TIMED.fetch_add(1, Relaxed);
    TIMED_NS.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
}

/// guard: counts one timed candidate when dropped (also on an early error return)
pub(super) struct Noted(pub std::time::Instant);
impl Drop for Noted {
    fn drop(&mut self) {
        note_timed(self.0);
    }
}

/// (candidates timed, seconds spent in their timing launches) since process start, over all engines
pub fn tune_timing_stats() -> (u64, f64) {
    (TIMED.load(Relaxed), TIMED_NS.load(Relaxed) as f64 * 1e-9)
}

// ---- records ----

/// NaN-carrying kernel time (JSON has no NaN): None = NaN
fn nn(x: f32) -> Option<f32> {
    x.is_finite().then_some(x)
}
fn un(x: Option<f32>) -> f32 {
    x.unwrap_or(f32::NAN)
}

/// the state an autotune step leaves: unchanged (the step did not run: no tables, mode off, engine cannot run it), the
/// variant off (the tile kernels kept), or a variant on
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) enum Sel<T> {
    Keep,
    Off,
    On(T),
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct VsRec {
    pub geom: VsGeom,
    pub cfg: VsCfg,
    pub sch: usize,
    pub mma: usize,
    pub ws: Option<VwCfg>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct RaRec {
    pub cfg: RaCfg,
    pub var: RaVar,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct KlRec {
    pub skip: bool,
    pub kernel: Option<RaRec>,
    /// (engine adjoint over all items, skipping empty tiles, best kl kernel) ms and that kernel's shape
    pub t_all: Option<f32>,
    pub t_skip: Option<f32>,
    pub t_kl: Option<f32>,
    pub kl_cfg: Option<RaCfg>,
}

/// everything [`SegNufft::autotune`] decides on a single-image / knot-major engine
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct MainRec {
    pub fwd: TileCfg,
    pub adj: TileCfg,
    /// returned (forward, adjoint) kernel ms
    pub tf: Option<f32>,
    pub ta: Option<f32>,
    pub vs: Sel<VsRec>,
    /// (tile forward, vs forward) ms and the vs shape timed (None: the step did not run)
    pub vs_tune: Option<(Option<f32>, Option<f32>, Option<VsCfg>)>,
    pub ra: Sel<RaRec>,
    pub ra_tune: Option<(Option<f32>, Option<f32>, Option<RaCfg>)>,
    /// None: the knot-path step did not run (state unchanged)
    pub kl: Option<KlRec>,
}

/// the K-image shapes of the multi-image autotune
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct MultiRec {
    pub tf: TileCfg,
    pub ta: TileCfg,
    pub t_f: Option<f32>,
    pub t_a: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct TuneFile {
    pub format: u32,
    pub key: String,
    /// the canonical description hashed into `key` (checked on load: a hash collision is a miss)
    pub desc: String,
    pub gpu: String,
    /// seconds the timing took and candidates timed when this file was written
    pub tune_s: f64,
    pub candidates: u64,
    /// unix seconds of the write
    pub written: u64,
    pub main: Option<MainRec>,
    pub multi: Option<MultiRec>,
}

// ---- key ----

fn fnv(h: u64, b: &[u8]) -> u64 {
    b.iter().fold(h, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3))
}
const FNV0: u64 = 0xcbf2_9ce4_8422_2325;

/// hash of the kernel sources and the tuning code compiled into this binary
pub(super) fn source_hash() -> u64 {
    const SRC: &[&str] = &[
        include_str!("../segnufft.cu"),
        include_str!("../segnufft_knot.cu"),
        include_str!("../segnufft_multi.cu"),
        include_str!("../segnufft_vs.cu"),
        include_str!("../segnufft_ra.cu"),
        include_str!("../segnufft_kl.cu"),
        include_str!("../segnufft_kl_adj.cu"),
        include_str!("../kernel.rs"),
        include_str!("../segnufft.rs"),
        include_str!("vs.rs"),
        include_str!("ra.rs"),
        include_str!("kl.rs"),
        include_str!("tune.rs"),
        hasty_bfp::BFP_CUH,
    ];
    SRC.iter().fold(FNV0, |h, s| fnv(h, s.as_bytes()))
}

/// the HASTY_SEG_* switches (sorted; logging switches left out) and HASTY_NVRTC_LINEINFO
pub(super) fn env_desc() -> String {
    let mut v: Vec<(String, String)> = std::env::vars()
        .filter(|(k, _)| (k.starts_with("HASTY_SEG_") && !k.ends_with("_LOG") && k != "HASTY_SEG_KINFO") || k == "HASTY_NVRTC_LINEINFO")
        .collect();
    v.sort();
    v.iter().map(|(k, x)| format!("{k}={x}")).collect::<Vec<_>>().join(",")
}

/// `<gpu name>-cc<maj><min>-drv<driver>-nvrtc<maj>.<min>` (driver: /proc/driver/nvidia/version, else the CUDA driver API
/// version)
pub(super) fn gpu_tag(dev: &Device) -> String {
    static MEMO: Mutex<Option<HashMap<usize, String>>> = Mutex::new(None);
    let memo = MEMO.lock().unwrap().as_ref().and_then(|m| m.get(&dev.ordinal).cloned());
    if let Some(t) = memo {
        return t;
    }
    let name: String = dev.ctx.name().unwrap_or_else(|_| "gpu".into()).chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    let name = name.split('-').filter(|s| !s.is_empty()).collect::<Vec<_>>().join("-");
    let cc = dev.arch.trim_start_matches("sm_");
    let drv = std::fs::read_to_string("/proc/driver/nvidia/version")
        .ok()
        .and_then(|s| s.split_whitespace().find(|w| w.split('.').count() >= 2 && w.split('.').all(|p| !p.is_empty() && p.chars().all(|c| c.is_ascii_digit()))).map(str::to_owned))
        .unwrap_or_else(|| {
            let mut v = 0i32;
            unsafe {
                let _ = cudarc::driver::sys::cuDriverGetVersion(&mut v);
            }
            format!("api{v}")
        });
    let (a, b) = hasty_cuda::nvrtc_version();
    let t = format!("{name}-cc{cc}-drv{drv}-nvrtc{a}.{b}");
    MEMO.lock().unwrap().get_or_insert_with(HashMap::new).insert(dev.ordinal, t.clone());
    t
}

/// floor(2 log2 x): half-octave bucket (−1 for 0)
fn hb(x: f64) -> i64 {
    if x <= 0.0 { -1 } else { (2.0 * x.log2()).floor() as i64 }
}

/// Coarse statistics of the tuned set's tile lists (one device → host copy of its work items, ≈ 16 B per item):
/// buckets of the sample count, nonempty tiles, entries (sample-tile incidences) per nonempty tile, split fraction.
#[derive(Clone, Copy, Debug)]
pub(super) struct TabStats {
    pub m: usize,
    pub tiles: usize,
    pub entries: usize,
    pub split_frac: f64,
}

impl TabStats {
    pub fn of(st: &Arc<CudaStream>, pts: &SegPoints) -> Result<Self, NufftError> {
        let (a, b) = pts.ivals.iter().fold((usize::MAX, 0usize), |(a, b), iv| (a.min(iv.first), b.max(iv.first + iv.count)));
        let items: Vec<i32> = if b > a { st.clone_dtoh(&pts.items.slice(4 * a..4 * b))? } else { Vec::new() };
        let mut tiles = std::collections::HashSet::new();
        let (mut entries, mut nz, mut split) = (0usize, 0usize, 0usize);
        for it in items.chunks_exact(4) {
            if it[2] > 0 {
                tiles.insert(it[0]);
                entries += it[2] as usize;
                nz += 1;
                if it[3] != 0 {
                    split += 1;
                }
            }
        }
        Ok(Self { m: pts.m, tiles: tiles.len(), entries, split_frac: split as f64 / nz.max(1) as f64 })
    }
    pub fn desc(&self) -> String {
        format!(
            "m~{} tiles~{} spt~{} split~{}",
            hb(self.m as f64),
            hb(self.tiles as f64),
            hb(self.entries as f64 / self.tiles.max(1) as f64),
            (8.0 * self.split_frac).floor() as i64
        )
    }
}

/// canonical description of the engine state that changes an autotune's choice (see the module docs); `what`: the
/// autotune ("main", or the multi-image one with its plan)
pub(super) fn key_desc(e: &SegNufft, pts: &SegPoints, stats: &TabStats, what: &str) -> String {
    let p = &e.plan;
    let vs = pts.vs.as_ref().map(|t| {
        let tk = e.tune_knot(false);
        let (k0, k1) = t.items_of(if e.ks > 0 { tk } else { 0 });
        format!("{:?} dev{} items~{}", t.geom, t.device_bytes() > 0, hb((k1 - k0) as f64))
    });
    format!(
        "{what} | gpu {} | src {:016x} | nseg {} order {} nc {} cb {} nf {} frames {} ks {} kr {} mc {} rdeg {} | W {} beta {:.9e} kern {:?} grid {:?} nb {:?} | smax {} tiles0 {}/{} | vs_geom {:?} vt_mma {} vt_ws {} | zero {} kprune {} pruned {} fuse {} fftstore {:?} gridstore {:?} gbfp {} maps {:?} diag {:?} wf {} warp {} acc {}/{} one_bin {} | pts knot {} nri {} ivals {} vs {:?} | {} | env {}",
        gpu_tag(&e.dev),
        source_hash(),
        e.nseg,
        e.order,
        e.nc,
        e.cb,
        e.nf,
        e.frames,
        e.ks,
        e.kr,
        e.mc,
        e.rdeg,
        p.ker.w,
        p.ker.beta,
        e.kernel_kind(),
        p.nf,
        { let d = p.dims(); [d.nb0, d.nb1, d.nb2] },
        e.smax,
        e.tiles_fwd.tt,
        e.tiles_fwd.max_item,
        e.vs_geom,
        e.vt_mma,
        e.vt_ws,
        e.zero_tiles,
        e.kprune,
        e.gfft.is_pruned(),
        e.fuse_padcrop,
        e.gfft.store,
        e.gfft.gstore,
        e.gfft.grid_bits(),
        e.maps_store,
        e.diag_store,
        e.wf,
        e.warp.is_some(),
        e.adjoint_accumulate,
        e.forward_accumulate,
        e.is_one_bin(),
        pts.knot,
        pts.nri,
        pts.ivals.len(),
        vs,
        stats.desc(),
        env_desc()
    )
}

// ---- store ----

static MEM: Mutex<Option<HashMap<String, TuneFile>>> = Mutex::new(None);

/// one autotune's cache lookup / store
pub(super) struct Lookup {
    pub mode: TuneMode,
    pub key: String,
    desc: String,
    gpu: String,
    path: PathBuf,
    /// what the lookup found (Reuse: the record, if any and readable)
    pub found: Option<TuneFile>,
    /// why nothing was found (for the log line)
    note: String,
    t0: std::time::Instant,
    timed0: u64,
}

impl Lookup {
    /// None with HASTY_TUNE=off
    pub fn begin(e: &SegNufft, st: &Arc<CudaStream>, pts: &SegPoints, what: &str) -> Result<Option<Self>, NufftError> {
        let mode = tune_mode();
        if mode == TuneMode::Off {
            return Ok(None);
        }
        let stats = TabStats::of(st, pts)?;
        let desc = key_desc(e, pts, &stats, what);
        let key = format!("{:016x}", fnv(FNV0, desc.as_bytes()));
        let gpu = gpu_tag(&e.dev);
        let path = hasty_site::tune_cache().join(&gpu).join(format!("{key}.json"));
        let mut l = Lookup { mode, key, desc, gpu, path, found: None, note: String::new(), t0: std::time::Instant::now(), timed0: tune_timing_stats().0 };
        if mode == TuneMode::Reuse {
            // the guard is dropped before the file branch, which locks MEM again (edition 2021 keeps an if-let
            // scrutinee's temporaries alive into the else branch)
            let mem = MEM.lock().unwrap().as_ref().and_then(|m| m.get(&l.key).cloned());
            if let Some(f) = mem {
                l.note = "in-process".into();
                l.found = Some(f);
            } else {
                match std::fs::read(&l.path) {
                    Err(_) => l.note = "no file".into(),
                    Ok(b) => match serde_json::from_slice::<TuneFile>(&b) {
                        Ok(f) if f.format == FORMAT && f.key == l.key && f.desc == l.desc => {
                            l.note = "file".into();
                            MEM.lock().unwrap().get_or_insert_with(HashMap::new).insert(l.key.clone(), f.clone());
                            l.found = Some(f);
                        }
                        Ok(_) => l.note = "file of another format / key".into(),
                        Err(err) => l.note = format!("unreadable file ({err})"),
                    },
                }
            }
        }
        Ok(Some(l))
    }

    pub fn log_hit(&self, what: &str) {
        eprintln!("    [tune] hit {what} {} ({}, {}): choice loaded, no timing", self.key, self.note, self.path.display());
    }

    /// a loaded record that does not fit this engine: retime
    pub fn stale(&mut self, why: &str) {
        self.note = format!("stale: {why}");
        self.found = None;
    }

    /// after the timing: store the record (temp file + rename) and the in-process copy, log the miss / retune
    pub fn store(self, what: &str, main: Option<MainRec>, multi: Option<MultiRec>) {
        let tune_s = self.t0.elapsed().as_secs_f64();
        let candidates = tune_timing_stats().0 - self.timed0;
        let written = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
        let f = TuneFile { format: FORMAT, key: self.key.clone(), desc: self.desc.clone(), gpu: self.gpu.clone(), tune_s, candidates, written, main, multi };
        let label = if self.mode == TuneMode::Retune { "retune".to_string() } else { format!("miss ({})", self.note) };
        let saved = match serde_json::to_vec_pretty(&f) {
            Ok(bytes) => write_atomic(&self.path, &bytes),
            Err(e) => Err(std::io::Error::other(e)),
        };
        MEM.lock().unwrap().get_or_insert_with(HashMap::new).insert(self.key.clone(), f);
        match saved {
            Ok(()) => eprintln!("    [tune] {label} {what} {}: timed {candidates} candidates in {tune_s:.1} s, saved {}", self.key, self.path.display()),
            Err(e) => eprintln!("    [tune] {label} {what} {}: timed {candidates} candidates in {tune_s:.1} s, NOT saved ({e}: {})", self.key, self.path.display()),
        }
    }
}

/// write-then-rename (concurrent processes / threads never see a partial file; the last complete write wins)
fn write_atomic(path: &std::path::Path, bytes: &[u8]) -> std::io::Result<()> {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    let tmp = path.with_extension(format!("tmp{}-{}", std::process::id(), SEQ.fetch_add(1, Relaxed)));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&tmp);
    })
}

/// drop the in-process copies (tests: force the next lookup to the disk)
pub fn tune_cache_forget() {
    *MEM.lock().unwrap() = None;
}

// ---- record / apply ----

impl SegNufft {
    /// the record of what the autotune just chose (`t`: its returned times; `vs_ran` / `ra_ran`: whether those steps
    /// timed candidates, `kl_ran`: whether the knot-path step ran)
    pub(super) fn tune_record_main(&self, t: (f32, f32), vs_ran: bool, ra_ran: bool, kl_ran: bool) -> MainRec {
        let vs = if !vs_ran {
            Sel::Keep
        } else {
            self.vs.as_ref().map_or(Sel::Off, |a| Sel::On(VsRec { geom: a.geom, cfg: a.cfg, sch: a.sch, mma: a.mma, ws: a.ws }))
        };
        let ra = if !ra_ran { Sel::Keep } else { self.ra.as_ref().map_or(Sel::Off, |a| Sel::On(RaRec { cfg: a.cfg, var: a.var })) };
        let kl = kl_ran.then(|| {
            let tu = self.kl.tune.unwrap_or((f32::NAN, f32::NAN, f32::NAN, None));
            KlRec { skip: self.kl.skip, kernel: self.kl.kernel.as_ref().map(|k| RaRec { cfg: k.cfg, var: k.var }), t_all: nn(tu.0), t_skip: nn(tu.1), t_kl: nn(tu.2), kl_cfg: tu.3 }
        });
        MainRec {
            fwd: self.tiles_fwd,
            adj: self.tiles_adj,
            tf: nn(t.0),
            ta: nn(t.1),
            vs,
            vs_tune: self.vs_tune.map(|(a, b, c)| (nn(a), nn(b), c)),
            ra,
            ra_tune: self.ra_tune.map(|(a, b, c)| (nn(a), nn(b), c)),
            kl,
        }
    }

    /// why a record cannot be applied to this engine / set (None: it can)
    pub(super) fn tune_check_main(&self, r: &MainRec, pts: &SegPoints) -> Option<String> {
        let sh = self.shape();
        let t0 = self.tiles_fwd;
        if r.fwd.tt != t0.tt || r.adj.tt != t0.tt || r.fwd.max_item != t0.max_item || r.adj.max_item != t0.max_item {
            return Some("tile edge / item length".into());
        }
        if !r.fwd.fits(sh, self.smax) || !r.adj.fits(sh, self.smax) {
            return Some("tile shapes do not fit".into());
        }
        if let Sel::On(v) = &r.vs {
            if self.vs_geom != Some(v.geom) || !pts.vs.as_ref().is_some_and(|t| t.geom == v.geom && t.device_bytes() > 0) {
                return Some("vs geometry / tables".into());
            }
            let ok = match v.ws {
                Some(w) => self.vw_ok(v.geom, v.cfg, v.sch, w),
                None => self.vs_cfg_ok_m(v.geom, v.cfg, v.sch, v.mma),
            };
            if !ok {
                return Some(format!("vs shape {:?} not valid", v));
            }
        }
        if let Sel::On(a) = &r.ra {
            // the ra validity depends on the adjoint item length (checked above) and the engine
            if !self.ra_ok() || !self.ra_cfg_ok_v(a.cfg, a.var) {
                return Some(format!("ra shape {a:?} not valid"));
            }
        }
        if let Some(k) = &r.kl {
            if let Some(kk) = &k.kernel {
                if !self.kl_ok() || !self.kl_cfg_ok_v(kk.cfg, kk.var) {
                    return Some(format!("kl shape {kk:?} not valid"));
                }
            }
        }
        None
    }

    /// apply a checked record: the state the timed autotune would have left; returns its kernel times
    pub(super) fn tune_apply_main(&mut self, r: &MainRec) -> Result<(f32, f32), NufftError> {
        self.set_tiles(r.fwd, r.adj)?;
        match &r.ra {
            Sel::Keep => {}
            Sel::Off => self.ra = None,
            Sel::On(a) => self.ra = Some(self.ra_compile_v(a.cfg, a.var)?),
        }
        if let Some(t) = r.ra_tune {
            self.ra_tune = Some((un(t.0), un(t.1), t.2));
        }
        match &r.vs {
            Sel::Keep => {}
            Sel::Off => self.vs = None,
            Sel::On(v) => {
                self.vs = Some(match v.ws {
                    Some(w) => self.vs_compile_w(v.geom, v.cfg, v.sch, w)?,
                    None => self.vs_compile_m(v.geom, v.cfg, v.sch, v.mma)?,
                })
            }
        }
        if let Some(t) = r.vs_tune {
            self.vs_tune = Some((un(t.0), un(t.1), t.2));
        }
        if let Some(k) = &r.kl {
            self.kl.skip = k.skip;
            self.kl.kernel = match &k.kernel {
                Some(kk) => Some(self.kl_compile_v(kk.cfg, kk.var)?),
                None => None,
            };
            self.kl.tune = Some((un(k.t_all), un(k.t_skip), un(k.t_kl), k.kl_cfg));
        }
        Ok((un(r.tf), un(r.ta)))
    }
}

impl SegNufft {
    /// every choice an autotune decides, in words (tests compare two engines; logs)
    pub fn tune_choice_desc(&self) -> String {
        format!(
            "fwd {:?} adj {:?} | vs {:?} | ra {:?} | kl skip {} kernel {:?}",
            self.tiles_fwd,
            self.tiles_adj,
            self.vs.as_ref().map(|a| (a.geom, a.cfg, a.sch, a.mma, a.ws)),
            self.ra.as_ref().map(|a| (a.cfg, a.var)),
            self.kl.skip,
            self.kl.kernel.as_ref().map(|k| (k.cfg, k.var))
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(c: usize) -> TileCfg {
        TileCfg { tt: 4, cpba: c, cpbf: 5, ppf: 4, chunk: 48, max_item: 1024, acb: 1, fsb: 2, fcb: 1, kv: 1 }
    }

    fn file(tag: &str) -> TuneFile {
        let main = MainRec {
            fwd: tile(1),
            adj: tile(5),
            tf: Some(43.6),
            ta: Some(f32::NAN).and_then(nn),
            vs: Sel::On(VsRec { geom: VsGeom { tx: 4, ty: 4, tz: 16, smax: 48 }, cfg: VsCfg { nwc: 4, nws: 1, vxpad: 0 }, sch: 29, mma: 0, ws: None }),
            vs_tune: Some((Some(73.2), None, None)),
            ra: Sel::Keep,
            ra_tune: None,
            kl: Some(KlRec { skip: true, kernel: Some(RaRec { cfg: RaCfg { nxg: 4, cb: 7, chk: 12 }, var: RaVar { inplace: true, stages: 2, lbmin: 0 } }), t_all: Some(1.0), t_skip: None, t_kl: Some(0.5), kl_cfg: None }),
        };
        TuneFile { format: FORMAT, key: tag.into(), desc: format!("desc {tag}"), gpu: "test".into(), tune_s: 1.5, candidates: 7, written: 0, main: Some(main), multi: None }
    }

    /// JSON round trip of every record kind (NaN times as null, the three Sel states), CPU only
    #[test]
    fn record_round_trip() {
        let f = file("a");
        let s = serde_json::to_string_pretty(&f).unwrap();
        let g: TuneFile = serde_json::from_str(&s).unwrap();
        assert_eq!(g.main, f.main);
        assert!(un(g.main.as_ref().unwrap().ta).is_nan());
        let mut h = f.clone();
        h.main.as_mut().unwrap().ra = Sel::Off;
        h.multi = Some(MultiRec { tf: tile(2), ta: tile(3), t_f: Some(1.0), t_a: None });
        let g: TuneFile = serde_json::from_str(&serde_json::to_string(&h).unwrap()).unwrap();
        assert_eq!(g.main, h.main);
        assert_eq!(g.multi, h.multi);
        // corrupted / truncated files do not parse (the lookup turns that into a miss)
        assert!(serde_json::from_str::<TuneFile>(&s[..s.len() / 2]).is_err());
        assert!(serde_json::from_str::<TuneFile>("garbage").is_err());
    }

    /// the bucket boundaries: a density change of √2 always changes the bucket
    #[test]
    fn half_octave_buckets() {
        for x in [1.0f64, 3.0, 37.0, 3861.6, 1e6] {
            assert_eq!(hb(x * 2f64.sqrt() * 1.0001), hb(x) + 1, "x {x}");
            assert!(hb(x * 1.2) <= hb(x) + 1);
        }
        assert_eq!(hb(0.0), -1);
    }

    /// concurrent writers to one path (threads here; processes behave alike: rename is atomic on one file system):
    /// every read sees one complete file, never a partial one
    #[test]
    fn concurrent_writers() {
        let dir = hasty_site::target().join(format!("tune-test-unit-{}", std::process::id()));
        let path = dir.join("k.json");
        let writers: Vec<_> = (0..8)
            .map(|w| {
                let p = path.clone();
                std::thread::spawn(move || {
                    for i in 0..40 {
                        // files of different lengths, so a torn write would not parse
                        let mut f = file(&format!("w{w}-{i}"));
                        f.desc = "x".repeat(100 + 37 * w + i);
                        write_atomic(&p, &serde_json::to_vec_pretty(&f).unwrap()).unwrap();
                    }
                })
            })
            .collect();
        let reader = {
            let p = path.clone();
            std::thread::spawn(move || {
                let mut n = 0;
                for _ in 0..400 {
                    if let Ok(b) = std::fs::read(&p) {
                        let f: TuneFile = serde_json::from_slice(&b).expect("partial tune file seen");
                        assert!(f.key.starts_with('w'));
                        n += 1;
                    }
                }
                n
            })
        };
        for w in writers {
            w.join().unwrap();
        }
        let n = reader.join().unwrap();
        let left: Vec<_> = std::fs::read_dir(&dir).unwrap().filter_map(|e| e.ok()).map(|e| e.file_name()).collect();
        assert_eq!(left.len(), 1, "temporary files left: {left:?}");
        let _ = std::fs::remove_dir_all(&dir);
        eprintln!("concurrent writers: {n} complete reads");
    }

    #[test]
    fn mode_parse() {
        // HASTY_TUNE is not set in the test environment unless a caller sets it
        if std::env::var("HASTY_TUNE").is_err() {
            assert_eq!(tune_mode(), TuneMode::Reuse);
        }
    }
}
