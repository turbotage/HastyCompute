//! Adjoint variant `ra` (design 7 of research/sample_kernels.md: row-blocked output-tile adjoint; kernel
//! `segnufft_ra.cu`, §15 / §16 there): launch shapes, compilation, the scatter dispatch and the autotune step. A new
//! adjoint next to the tile kernels (`TileCfg::kv` 0 / 1 / 2 stay as they are), on the engine's own TT = 4 tile lists,
//! items and split flags: no extra tables. The engine uses it for every single-image scatter (adjoint, group, source;
//! static and knot-major, 1-D and 2-D knots) when it is active ([`SegNufft::set_ra`], or the autotune found it faster)
//! and the engine can run it: one frame slot (static image or knot-major), TT = 4, correction order ≤ 2 on static
//! engines (order 3 there lives in the multi-image kernels) and ≤ 3 on knot-major engines. Interval-major frames, the
//! multi-image / fat paths and other tile edges keep their kernels.
//!
//! Output contract (HASTY_SEG_ZERO=tiles, research/temporal_operator.md §23): unsplit tiles are stored (single owner;
//! bit 0 of addmask: read-modify-write), split tiles add atomically onto their zeroed cells, as the tile kernels.
//!
//! Env: HASTY_SEG_VSA = 0 (off: never timed), 1 (on: the autotune keeps the fastest ra shape even when a tile kernel is
//! faster; active from construction with the default shape), unset / auto (the autotune picks ra only where it is
//! faster). HASTY_SEG_VSA_CFG = nxg,cb,chk (one launch shape instead of the autotune's candidates), HASTY_SEG_VSA_LOG = 1
//! (per-candidate times).
//!
//! Phase 2 (research/sample_kernels.md §19, research/arch_targets.md §4 items 2 / 3): a staging variant [`RaVar`] next
//! to the launch shape — the item list read in place (no LISTMAX buffer), smaller chunks, cp.async double-buffered
//! staging, a per-shape `__launch_bounds__` minimum-blocks hint — and spill screening of every candidate on the running
//! device (`CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` against the family's known stack frame). The previous candidates stay
//! and are timed first; the variants are timed for the fastest two (nxg, cb) of them. Env:
//! HASTY_SEG_VSA_STAGE = 0 (only the previous candidates) | inplace (+ in-place lists, smaller chunks) | cpasync (+ the
//! cp.async double buffer) | bal (+ balanced staging, stages 3, §20) | dyn (+ the fused pipeline, stages 4) | auto
//! (default = bal); HASTY_SEG_LBMIN = auto (default: hints 3 / 4 where they do not spill
//! and keep the warps) | 0 (none) | 3 | 4 (only that hint); HASTY_SEG_SPILLSKIP = 1 (default) | 0;
//! HASTY_SEG_VSA_CFG = nxg,cb,chk[,inplace,stages,lbmin] (stages 5 / 6: warp-specialised, §22); HASTY_SEG_VSA_WS = 0 (default) | auto | 4 | 2.

use super::*;

/// Launch shape of the ra adjoint: x groups per warp (`nxg` 1 | 2 | 4: a lane holds 4 / nxg points of each tile row),
/// channels per lane (`cb`; a block row covers 32 / nxg · cb channels) and samples staged per chunk (`chk`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RaCfg {
    pub nxg: usize,
    pub cb: usize,
    pub chk: usize,
}

impl RaCfg {
    /// channels of one block row
    pub fn cpw(&self) -> usize {
        32 / self.nxg * self.cb
    }
}

/// Staging variant of the ra / kl adjoint (phase 2; [`RaVar::BASE`] = the kernel as first integrated): `inplace` reads
/// the item list in place (ra only; kl always does), `stages` 1 = register prefetch, 2 = cp.async double buffer (needs
/// `inplace` in ra), 3 = cp.async + dynamic warp-unit staging, 4 = 3 + the MAC of chunk k − 1 fused with staging of
/// chunk k (two barriers per chunk; ra only, research/sample_kernels.md §20), 5 / 6 = warp-specialised cp.async
/// pipeline with 4 / 2 producer warps after the 4 MAC warps (RA_WS, research/sample_kernels.md §22; ra only, in place),
/// `lbmin` > 0 = `__launch_bounds__(threads, lbmin)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RaVar {
    pub inplace: bool,
    pub stages: usize,
    pub lbmin: usize,
}

impl RaVar {
    pub const BASE: RaVar = RaVar { inplace: false, stages: 1, lbmin: 0 };
    pub fn is_base(&self) -> bool {
        *self == Self::BASE
    }
    /// producer warps of the warp-specialised variant (stages 5: 4, stages 6: 2; else 0)
    pub fn ws(&self) -> usize {
        match self.stages {
            5 => 4,
            6 => 2,
            _ => 0,
        }
    }
    /// threads per block (ra: 4 MAC warps + producers)
    pub fn threads(&self) -> usize {
        32 * (TT_RA + self.ws())
    }
    /// NVRTC defines (empty for the base variant: its source is the previous kernel's)
    pub(super) fn defines(&self) -> String {
        if self.is_base() {
            return String::new();
        }
        if self.ws() > 0 {
            return format!("#define RA_INPLACE 1\n#define RA_STAGES 2\n#define RA_WS {}\n#define LBMIN {}\n", self.ws(), self.lbmin);
        }
        format!("#define RA_INPLACE {}\n#define RA_STAGES {}\n#define LBMIN {}\n", self.inplace as u32, self.stages, self.lbmin)
    }
    /// module-name suffix (empty for the base variant)
    pub(super) fn tag(&self) -> String {
        if self.is_base() {
            String::new()
        } else {
            format!("_v{}{}{}", self.inplace as u32, self.stages, self.lbmin)
        }
    }
}

impl Default for RaVar {
    fn default() -> Self {
        Self::BASE
    }
}

/// env HASTY_SEG_VSA_STAGE: which staging variants the autotune adds (ra and kl)
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum StageMode {
    Off,
    InPlace,
    CpAsync,
    /// + the ra balanced-staging variant (stages 3; research/sample_kernels.md §20)
    Bal,
    /// + the fused pipeline (stages 4)
    Dyn,
}

pub(super) fn stage_mode() -> StageMode {
    match std::env::var("HASTY_SEG_VSA_STAGE").as_deref() {
        Ok("0") | Ok("off") => StageMode::Off,
        Ok("inplace") => StageMode::InPlace,
        Ok("cpasync") => StageMode::CpAsync,
        Ok("bal") | Ok("auto") | Ok("") | Err(_) => StageMode::Bal,
        Ok("dyn") => StageMode::Dyn,
        Ok(v) => panic!("HASTY_SEG_VSA_STAGE={v}: 0 | inplace | cpasync | bal | dyn | auto"),
    }
}

/// env HASTY_SEG_VSA_WS = 0 (default: never; measured 1.8 × slower at 50 channels, research/sample_kernels.md §22) |
/// auto (the warp-specialised stages 5 / 6 join the cp.async candidates where the stage mode allows the cp.async double
/// buffer) | 4 | 2 (only that producer count)
pub(super) fn ws_stages() -> Vec<usize> {
    match std::env::var("HASTY_SEG_VSA_WS").as_deref() {
        Ok("auto") => vec![5, 6],
        Ok("0") | Ok("off") | Ok("") | Err(_) => vec![],
        Ok("4") => vec![5],
        Ok("2") => vec![6],
        Ok(v) => panic!("HASTY_SEG_VSA_WS={v}: auto | 0 | 4 | 2"),
    }
}

/// env HASTY_SEG_LBMIN: the launch-bounds hints the autotune adds (ra and kl)
pub(super) fn lbmin_set() -> Vec<usize> {
    match std::env::var("HASTY_SEG_LBMIN").as_deref() {
        Ok("auto") | Ok("") | Err(_) => vec![3, 4],
        Ok("0") | Ok("off") => vec![],
        Ok(v) => vec![v.trim().parse().unwrap_or_else(|_| panic!("HASTY_SEG_LBMIN={v}: auto | 0 | <min blocks>"))],
    }
}

/// env HASTY_SEG_SPILLSKIP (default 1): skip candidates that spill on the running device
pub(super) fn spill_skip() -> bool {
    !matches!(std::env::var("HASTY_SEG_SPILLSKIP").as_deref(), Ok("0") | Ok("off"))
}

/// Resources of a compiled kernel on the running device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernRes {
    /// registers per thread, local memory per thread (bytes: the stack frame, spills included; driver attributes)
    pub regs: i32,
    pub local: i32,
    /// spill stores (bytes) in the ptxas report of the same source and arch (0 when the kernel has no local memory)
    pub spill: i32,
    /// resident blocks and warps per SM at this launch (driver occupancy calculator)
    pub blocks: u32,
    pub warps: u32,
}

/// Resources of kernel `f` (entry `entry` of module `name`, compiled from `src` for this device). Registers, local
/// memory and occupancy come from the driver (microseconds). A local frame alone does not mean spills (every ra / kl /
/// tile kernel carries the 28-byte array of the `sincosf` slow path, research/sample_kernels.md §19), so kernels with
/// local memory get their spill bytes from [`ptxas_spill`] (one NVRTC compile per source, arch and NVRTC version, cached
/// on disk next to the cubins).
pub(super) fn kern_res(dev: &Device, f: &CudaFunction, threads: u32, smem: usize, name: &str, src: &str, entry: &str) -> KernRes {
    let regs = f.num_regs().unwrap_or(-1);
    let local = f.local_size_bytes().unwrap_or(0);
    let blocks = f.occupancy_max_active_blocks_per_multiprocessor(threads, smem, None).unwrap_or(0);
    let spill = if local > 0 { ptxas_spill(&dev.arch, name, src, entry) } else { 0 };
    KernRes { regs, local, spill, blocks, warps: blocks * threads.div_ceil(32) }
}

fn fnv64(h: u64, b: &[u8]) -> u64 {
    b.iter().fold(h, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3))
}

/// Spill-store bytes of entry `entry` in the ptxas report (`--ptxas-options=-v`) of `src` compiled for `arch` with the
/// engine's options (HASTY_NVRTC_LINEINFO=1 adds -lineinfo, as hasty-cuda). Memoised per process and cached as
/// `<kernel cache>/ptxinfo/<name>-<hash>.spill`; −1 if the report could not be obtained (then treated as no spill).
pub(super) fn ptxas_spill(arch: &str, name: &str, src: &str, entry: &str) -> i32 {
    use cudarc::nvrtc::sys;
    use std::ffi::{c_char, CStr, CString};
    static MEMO: std::sync::Mutex<Option<std::collections::HashMap<u64, i32>>> = std::sync::Mutex::new(None);
    let lineinfo = std::env::var("HASTY_NVRTC_LINEINFO").is_ok_and(|v| v == "1");
    let (mut va, mut vb) = (0, 0);
    unsafe {
        let _ = sys::nvrtcVersion(&mut va, &mut vb);
    }
    let mut h = fnv64(0xcbf29ce484222325, src.as_bytes());
    h = fnv64(h, arch.as_bytes());
    h = fnv64(h, entry.as_bytes());
    h = fnv64(h, format!("nvrtc{va}.{vb} lineinfo{lineinfo}").as_bytes());
    if let Some(v) = MEMO.lock().unwrap().as_ref().and_then(|m| m.get(&h).copied()) {
        return v;
    }
    let file = hasty_site::kernel_cache().join("ptxinfo").join(format!("{name}-{h:016x}.spill"));
    let v = std::fs::read_to_string(&file).ok().and_then(|s| s.trim().parse::<i32>().ok()).unwrap_or_else(|| {
        let opts: Vec<CString> = [format!("--gpu-architecture={arch}"), "--std=c++17".into(), "--ptxas-options=-v".into(), "--no-cache".into()]
            .into_iter()
            .chain(lineinfo.then(|| "-lineinfo".to_string()))
            .map(|o| CString::new(o).unwrap())
            .collect();
        let ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
        let (csrc, cname) = (CString::new(src).unwrap(), CString::new(format!("{name}.cu")).unwrap());
        let log = unsafe {
            let mut prog: sys::nvrtcProgram = std::ptr::null_mut();
            if sys::nvrtcCreateProgram(&mut prog, csrc.as_ptr(), cname.as_ptr(), 0, std::ptr::null(), std::ptr::null()) != sys::nvrtcResult::NVRTC_SUCCESS {
                return -1;
            }
            let r = sys::nvrtcCompileProgram(prog, ptrs.len() as i32, ptrs.as_ptr());
            let mut n = 0usize;
            let _ = sys::nvrtcGetProgramLogSize(prog, &mut n);
            let mut buf = vec![0 as c_char; n.max(1)];
            let _ = sys::nvrtcGetProgramLog(prog, buf.as_mut_ptr());
            let _ = sys::nvrtcDestroyProgram(&mut prog);
            if r != sys::nvrtcResult::NVRTC_SUCCESS {
                return -1;
            }
            CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
        };
        // ptxas -v: "Compiling entry function 'E' for 'sm_89'" / "Function properties for E" then
        // "N bytes stack frame, S bytes spill stores, L bytes spill loads"
        let mut cur = false;
        let mut spill = -1;
        for line in log.lines() {
            if line.contains("Function properties for ") || line.contains("Compiling entry function '") {
                cur = line.contains(entry);
                continue;
            }
            if cur && line.contains("bytes spill stores") {
                let i = line.find("bytes spill stores").unwrap();
                spill = line[..i].split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).last().and_then(|s| s.parse().ok()).unwrap_or(-1);
                break;
            }
        }
        if spill >= 0 {
            let _ = std::fs::create_dir_all(file.parent().unwrap());
            let tmp = file.with_extension(format!("tmp{}", std::process::id()));
            if std::fs::write(&tmp, format!("{spill}\n")).is_ok() {
                let _ = std::fs::rename(&tmp, &file);
            }
        }
        spill
    });
    MEMO.lock().unwrap().get_or_insert_with(Default::default).insert(h, v);
    v
}

/// env HASTY_SEG_VSA
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RaMode {
    Off,
    Auto,
    On,
}

pub(super) fn ra_mode() -> RaMode {
    match std::env::var("HASTY_SEG_VSA").as_deref() {
        Ok("0") | Ok("off") => RaMode::Off,
        Ok("1") | Ok("on") => RaMode::On,
        Ok("auto") | Ok("") | Err(_) => RaMode::Auto,
        Ok(v) => panic!("HASTY_SEG_VSA={v}: 0 | 1 | auto"),
    }
}

/// the active ra adjoint of an engine
pub(super) struct RaActive {
    pub cfg: RaCfg,
    pub var: RaVar,
    pub f: CudaFunction,
    pub smem: usize,
    pub res: KernRes,
}

const TT_RA: usize = 4;
/// static shared memory of the kernel (2-D knot window bounds) and a margin
const RA_STATIC: usize = 256;

fn a16(x: usize) -> usize {
    (x + 15) & !15
}

/// dynamic shared memory (bytes) of a shape (the kernel's OFF_* layout; the output park reuses it)
#[allow(dead_code)]
pub(super) fn ra_smem(order: u32, nseg: usize, ncb: usize, listmax: usize, c: RaCfg) -> usize {
    ra_smem_v(order, nseg, ncb, listmax, c, RaVar::BASE)
}

/// dynamic shared memory (bytes) of a shape and staging variant (RA_INPLACE: entries, htab, uw and the raw inputs
/// instead of the list copy; RA_STAGES 2: two buffers of the raw inputs)
pub(super) fn ra_smem_v(order: u32, nseg: usize, ncb: usize, listmax: usize, c: RaCfg, v: RaVar) -> usize {
    let nka = match order {
        0 => 1,
        1 => 3,
        _ => 5,
    };
    let (ncl, cpw) = (32 / c.nxg, c.cpw());
    let cbp = c.cb.div_ceil(2) * 2;
    let ncbrm = ((cpw + nseg - 1) / nseg + 1).min(ncb);
    let chk = c.chk;
    let ra = if v.inplace {
        let st = v.stages.min(2);
        let ht = a16(st * 4 * chk);
        let uw = a16(ht + if order >= 3 { 16 * 25 } else { 0 });
        let rec = a16(uw + if order >= 3 { st * 16 * chk } else { 0 });
        let ry = a16(rec + st * chk * 48);
        a16(ry + st * chk * ncbrm * 8)
    } else {
        let rec = a16(4 * listmax);
        let ry = a16(rec + chk * 48);
        a16(ry + chk * ncbrm * 8)
    };
    let ax = a16(ra + chk * nseg * 8);
    let bx = a16(ax + chk * 3 * nka * TT_RA * 4);
    let e = a16(bx + chk * 4);
    let h = a16(e + if order >= 2 { chk * 15 * 4 } else { 0 });
    let z = a16(h + if order >= 3 { chk * 25 * 4 } else { 0 });
    let w = a16(z + chk * (cbp / 2) * ncl * 16);
    let mut stage = w + if order >= 1 { chk * 16 * TT_RA * 8 } else { 0 };
    if v.stages == 4 && order == 0 {
        // fused pipeline at order 0: second box and kernel-value buffers (orders ≥ 1: boxes in registers)
        stage = a16(a16(stage) + chk * 4) + chk * 3 * nka * TT_RA * 4;
    }
    if v.ws() > 0 {
        // warp-specialised: second MAC-input buffers (box, z, weights; order 0: kernel values)
        let bb = a16(stage);
        let zb = a16(bb + chk * 4);
        let wb = a16(zb + chk * (cbp / 2) * ncl * 16);
        let axb = a16(wb + if order >= 1 { chk * 16 * TT_RA * 8 } else { 0 });
        stage = axb + if order == 0 { chk * 3 * nka * TT_RA * 4 } else { 0 };
    }
    let park = TT_RA * TT_RA * (cpw | 1) * 8;
    stage.max(park)
}

impl SegNufft {
    /// whether the engine can run the ra adjoint: one frame slot (static image or knot-major engine), TT = 4 tile
    /// lists, order ≤ 2 on static engines (order 3: multi-image kernels), ≤ 3 on knot-major engines
    pub(super) fn ra_ok(&self) -> bool {
        self.nf == 1 && self.tiles_adj.tt == TT_RA && (self.order <= 2 || (self.ks > 0 && self.order <= 3))
    }

    /// whether a launch shape is valid on this engine (registers: 4 / nxg · cb ≤ 16 complex accumulators per row; shared
    /// memory; ≤ 65535 block rows)
    pub(super) fn ra_cfg_ok(&self, c: RaCfg) -> bool {
        self.ra_cfg_ok_v(c, RaVar::BASE)
    }

    /// as [`ra_cfg_ok`](Self::ra_cfg_ok) for a staging variant (stages 1 | 2, stages 2 only in place, hint ≤ 16 blocks)
    pub(super) fn ra_cfg_ok_v(&self, c: RaCfg, v: RaVar) -> bool {
        let nchv = self.cb * self.nseg;
        matches!(c.nxg, 1 | 2 | 4)
            && c.cb >= 1
            && TT_RA / c.nxg * c.cb <= 16
            && (1..=64).contains(&c.chk)
            && nchv.div_ceil(c.cpw()) <= 65535
            && matches!(v.stages, 1..=6)
            && (v.stages == 1 || v.inplace)
            && v.lbmin <= 8
            && ra_smem_v(self.order, self.nseg, self.cb, self.tiles_adj.max_item, c, v) + RA_STATIC <= self.smax
    }

    /// Launch shapes the autotune times (env HASTY_SEG_VSA_CFG = nxg,cb,chk: that one): per x grouping, the channel
    /// blocks that cover the batch's channels in 1–4 block rows, chunks of 16 and 32 samples.
    pub fn ra_candidates(&self) -> Vec<RaCfg> {
        if let Some((c, _)) = Self::ra_env_cfg() {
            assert!(self.ra_cfg_ok(c), "HASTY_SEG_VSA_CFG: {c:?} is not a valid ra shape on this engine");
            return vec![c];
        }
        let nchv = self.cb.min(self.nc) * self.nseg;
        let mut v = Vec::new();
        for nxg in [1usize, 2, 4] {
            let ncl = 32 / nxg;
            let mut cbs: Vec<usize> = (1..=4).map(|rows| nchv.div_ceil(ncl * rows)).filter(|&cb| cb >= 1 && TT_RA / nxg * cb <= 16).collect();
            cbs.dedup();
            for cb in cbs {
                for chk in [16usize, 32] {
                    let c = RaCfg { nxg, cb, chk };
                    if self.ra_cfg_ok(c) && !v.contains(&c) {
                        v.push(c);
                    }
                }
            }
        }
        v
    }

    /// HASTY_SEG_VSA_CFG = nxg,cb,chk[,inplace,stages,lbmin]
    pub(super) fn ra_env_cfg() -> Option<(RaCfg, RaVar)> {
        let v = std::env::var("HASTY_SEG_VSA_CFG").ok()?;
        let p: Vec<usize> = v.split(',').map(|x| x.trim().parse().expect("HASTY_SEG_VSA_CFG = nxg,cb,chk[,inplace,stages,lbmin]")).collect();
        let c = RaCfg { nxg: p[0], cb: *p.get(1).unwrap_or(&1), chk: *p.get(2).unwrap_or(&16) };
        let var = RaVar { inplace: *p.get(3).unwrap_or(&0) != 0, stages: *p.get(4).unwrap_or(&1), lbmin: *p.get(5).unwrap_or(&0) };
        Some((c, var))
    }

    /// the default launch shape (the prototype's best at 10 × 10, research/sample_kernels.md §15.5): two x groups, the
    /// fewest block rows with ≤ 8 channels per lane, 16-sample chunks
    pub(super) fn ra_default_cfg(&self) -> Option<RaCfg> {
        let nchv = self.cb * self.nseg;
        (1..=64).map(|rows| RaCfg { nxg: 2, cb: nchv.div_ceil(16 * rows), chk: 16 }).find(|&c| self.ra_cfg_ok(c))
    }

    /// compile (disk-cached) the ra adjoint of a shape
    pub(super) fn ra_compile(&self, c: RaCfg) -> Result<RaActive, NufftError> {
        self.ra_compile_v(c, RaVar::BASE)
    }

    /// compile (disk-cached) the ra adjoint of a shape and staging variant; its resources on this device
    pub(super) fn ra_compile_v(&self, c: RaCfg, v: RaVar) -> Result<RaActive, NufftError> {
        let (kpre, ktag) = kernel_prefix(&self.plan);
        let mi = self.tiles_adj.max_item;
        // research switch HASTY_SEG_RA_DEFS = NAME=V,...: extra defines (e.g. RA_GRAB=1, RA_ZSPL=4 for stages 3 / 4);
        // unset: the source and module name are unchanged
        let xdefs = std::env::var("HASTY_SEG_RA_DEFS").unwrap_or_default();
        let xdef: String = xdefs.split(',').filter(|d| !d.is_empty()).map(|d| format!("#define {}\n", d.replacen('=', " ", 1))).collect();
        let xtag = if xdef.is_empty() { String::new() } else { format!("_d{:016x}", fnv64(0xcbf2_9ce4_8422_2325, xdefs.as_bytes())) };
        let src = format!(
            "{xdef}{kpre}{}#define W {}\n#define BETA {:.17e}\n#define NSEG {}\n#define CORR {}\n#define NCB {}\n#define LISTMAX {}\n#define NXG {}\n#define CB {}\n#define CHK {}\n#define KS {}\n#define RDEG {}\n{}",
            v.defines(),
            self.plan.ker.w,
            self.plan.ker.beta,
            self.nseg,
            self.order,
            self.cb,
            mi,
            c.nxg,
            c.cb,
            c.chk,
            self.ks,
            self.rdeg,
            include_str!("../segnufft_ra.cu")
        );
        let name = format!("segnufft_ra{ktag}_w{}_l{}_o{}_c{}_k{}r{}_m{}_x{}_b{}_h{}{}", self.plan.ker.w, self.nseg, self.order, self.cb, self.ks, self.rdeg, mi, c.nxg, c.cb, c.chk, v.tag()) + &xtag;
        let module = self.dev.load_module(&name, &src, &CompileSettings::default())?;
        let smem = ra_smem_v(self.order, self.nseg, self.cb, mi, c, v);
        let f = tile_fn(&module, "seg_adjoint_ra", smem)?;
        let res = kern_res(&self.dev, &f, v.threads() as u32, smem, &name, &src, "seg_adjoint_ra");
        Ok(RaActive { cfg: c, var: v, f, smem, res })
    }

    /// Resources of the ra adjoint of a shape and staging variant on this device (compiles it; disk-cached): registers,
    /// local memory, spill bytes beyond the spill-free frame, resident blocks / warps per SM. None: invalid here.
    pub fn ra_resources(&self, c: RaCfg, v: RaVar) -> Result<Option<KernRes>, NufftError> {
        if !self.ra_ok() || !self.ra_cfg_ok_v(c, v) {
            return Ok(None);
        }
        Ok(Some(self.ra_compile_v(c, v)?.res))
    }

    /// engine construction: with HASTY_SEG_VSA=1 the default launch shape (or HASTY_SEG_VSA_CFG) is active from the start
    pub(super) fn ra_init(&mut self) -> Result<(), NufftError> {
        if ra_mode() != RaMode::On || !self.ra_ok() {
            return Ok(());
        }
        if let Some((c, v)) = Self::ra_env_cfg() {
            assert!(self.ra_cfg_ok_v(c, v), "HASTY_SEG_VSA_CFG: {c:?} {v:?} is not a valid ra shape on this engine");
            self.ra = Some(self.ra_compile_v(c, v)?);
        } else if let Some(c) = self.ra_default_cfg() {
            self.ra = Some(self.ra_compile(c)?);
        }
        Ok(())
    }

    /// Use the ra adjoint with launch shape `c` for every single-image scatter this engine can run it on, or `None` for
    /// the tile kernels. Err if the engine cannot run ra or the shape is invalid.
    pub fn set_ra(&mut self, c: Option<RaCfg>) -> Result<(), NufftError> {
        self.set_ra_v(c, RaVar::BASE)
    }

    /// [`set_ra`](Self::set_ra) with a staging variant ([`RaVar`])
    pub fn set_ra_v(&mut self, c: Option<RaCfg>, v: RaVar) -> Result<(), NufftError> {
        let Some(c) = c else {
            self.ra = None;
            return Ok(());
        };
        if !self.ra_ok() {
            return Err(NufftError::Invalid(format!("set_ra: not available on this engine (frame slots {}, tile edge {}, order {}, knot support {})", self.nf, self.tiles_adj.tt, self.order, self.ks)));
        }
        if !self.ra_cfg_ok_v(c, v) {
            return Err(NufftError::Invalid(format!("set_ra: {c:?} {v:?} is not a valid launch shape on this engine")));
        }
        self.ra = Some(self.ra_compile_v(c, v)?);
        Ok(())
    }

    /// the active ra launch shape (None: the tile kernels)
    pub fn ra_active(&self) -> Option<RaCfg> {
        self.ra.as_ref().map(|a| a.cfg)
    }

    /// the active ra launch shape with its staging variant and resources on this device
    pub fn ra_active_v(&self) -> Option<(RaCfg, RaVar, KernRes)> {
        self.ra.as_ref().map(|a| (a.cfg, a.var, a.res))
    }

    /// kernel times (ms) of the last autotune's adjoint choice: (best tile kernel, best ra shape or NaN)
    pub fn ra_tune_times(&self) -> Option<(f32, f32)> {
        self.ra_tune.map(|t| (t.0, t.1))
    }

    /// the last autotune's fastest ra launch shape (whether or not it was kept)
    pub fn ra_tune_cfg(&self) -> Option<RaCfg> {
        self.ra_tune.and_then(|t| t.2)
    }

    /// ra scatter of interval / knot i (static: the interval's items; knot: knot i's window of the one item list) for
    /// coils c0..c0+ncb into the slot (bit 0 of `addmask`: add instead of store)
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_ra(&self, st: &Arc<CudaStream>, a: &RaActive, seg: &SegSegments, pts: &SegPoints, i: usize, g: Grids, addmask: i32, y: &CudaSlice<C32>, c0: usize, ncb: usize) -> Result<(), NufftError> {
        let iv = if self.ks > 0 { pts.ivals[0] } else { pts.ivals[i] };
        if iv.count == 0 {
            return Ok(());
        }
        let (t0, dt, ntab, m_i, c0_i, ncb_i, d) = (seg.tab.1, seg.tab.2, seg.tab.0 as i32, pts.m as i32, c0 as i32, ncb as i32, self.plan.dims());
        let (b_i, nint, nri) = (if self.ks > 0 { i as i32 } else { 0 }, pts.knot.max(1) as i32, pts.nri as i32);
        let itab = seg.itab(if self.ks > 0 { i } else { 0 });
        let items = pts.items.slice(4 * iv.first..4 * (iv.first + iv.count));
        let (uw, ht) = self.hi_ptrs(st, pts);
        let rows = (ncb * self.nseg).div_ceil(a.cfg.cpw());
        unsafe {
            let mut lb = st.launch_builder(&a.f);
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
            lb.launch(LaunchConfig { grid_dim: (iv.count as u32, rows as u32, 1), block_dim: (a.var.threads() as u32, 1, 1), shared_mem_bytes: a.smem as u32 })?;
        }
        Ok(())
    }

    /// Adjoint scatter of interval / knot i for coils c0..c0+ncb into the grid slots (bit o of `addmask`: add instead of
    /// store): the ra adjoint when it is active and the engine can run it, else the tile kernel.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn scatter(&self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, i: usize, g: Grids, addmask: i32, y: &CudaSlice<C32>, c0: usize, ncb: usize) -> Result<(), NufftError> {
        if let Some(a) = self.ra.as_ref().filter(|_| self.ra_ok()) {
            return self.launch_ra(st, a, seg, pts, i, g, addmask, y, c0, ncb);
        }
        if self.ks > 0 {
            self.launch_adj_k(st, &self.f.adj, &self.tiles_adj, seg, pts, i, g, addmask, y, c0, ncb)
        } else {
            self.launch_adj(st, &self.f.adj, &self.tiles_adj, seg, pts, i, g, addmask, y, c0, ncb)
        }
    }

    /// Staging variants the autotune adds for one (nxg, cb) (ra): in place with chunks 8, 12, 16 and the largest chunk
    /// ≤ 32 that keeps the register-limited block count on this device (shared memory then does not cut blocks), each
    /// with the register prefetch and (HASTY_SEG_VSA_STAGE ≠ inplace) the cp.async double buffer.
    pub(super) fn ra_variant_plan(&self, nxg: usize, cb: usize) -> Result<Vec<(RaCfg, RaVar)>, NufftError> {
        let mode = stage_mode();
        if mode == StageMode::Off {
            return Ok(vec![]);
        }
        let mut stages: Vec<usize> = if mode >= StageMode::Dyn {
            vec![1, 2, 3, 4]
        } else if mode >= StageMode::Bal {
            vec![1, 2, 3]
        } else if mode >= StageMode::CpAsync {
            vec![1, 2]
        } else {
            vec![1]
        };
        if mode >= StageMode::CpAsync {
            stages.extend(ws_stages());
        }
        let mut chks = vec![8usize, 12, 16];
        let probe = (RaCfg { nxg, cb, chk: 16 }, RaVar { inplace: true, stages: 1, lbmin: 0 });
        if self.ra_cfg_ok_v(probe.0, probe.1) {
            let a = self.ra_compile_v(probe.0, probe.1)?;
            let fit = chunk_fit(&a.f, self.smax, |chk| ra_smem_v(self.order, self.nseg, self.cb, self.tiles_adj.max_item, RaCfg { chk, ..probe.0 }, probe.1));
            if let Some(f) = fit {
                chks.push(f);
            }
        }
        chks.sort_unstable();
        chks.dedup();
        let mut v = Vec::new();
        for &chk in &chks {
            for &st in &stages {
                let (c, var) = (RaCfg { nxg, cb, chk }, RaVar { inplace: true, stages: st, lbmin: 0 });
                if self.ra_cfg_ok_v(c, var) {
                    v.push((c, var));
                }
            }
        }
        Ok(v)
    }

    /// Autotune step of the ra adjoint: time every candidate shape on interval / knot `i` (as the tile adjoint was
    /// timed: store mode into the slot) and return the fastest (ms, shape); None when the engine cannot run ra.
    /// Phase A: the previous candidates (base variant). Phase B: for the fastest two (nxg, cb) of phase A, the staging
    /// variants of [`ra_variant_plan`](Self::ra_variant_plan); then the launch-bounds hints for the fastest three so
    /// far, kept only without spills and with at least the warps of the unhinted kernel. Candidates that spill on this
    /// device are skipped (HASTY_SEG_SPILLSKIP=0: timed anyway); if every candidate spills, all are timed.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn autotune_ra(&self, st: &Arc<CudaStream>, seg: &SegSegments, pts: &SegPoints, i: usize, g: Grids, ys: &CudaSlice<C32>, ncb: usize) -> Result<Option<(f32, RaActive)>, NufftError> {
        if !self.ra_ok() {
            return Ok(None);
        }
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let ev = (self.dev.ctx.new_event(flags)?, self.dev.ctx.new_event(flags)?);
        let log = std::env::var("HASTY_SEG_VSA_LOG").is_ok_and(|v| v != "0");
        let skip = spill_skip();
        let time = |a: &RaActive| -> Result<f32, NufftError> {
            let _note = super::tune::Noted(std::time::Instant::now());
            self.launch_ra(st, a, seg, pts, i, g, 0, ys, 0, ncb)?;
            let mut ms = f32::INFINITY;
            for _ in 0..3 {
                ev.0.record(st)?;
                self.launch_ra(st, a, seg, pts, i, g, 0, ys, 0, ncb)?;
                ev.1.record(st)?;
                ev.1.synchronize()?;
                ms = ms.min(ev.0.elapsed_ms(&ev.1)?);
            }
            Ok(ms)
        };
        let show = |a: &RaActive, what: &str| {
            if log {
                eprintln!("    [ra] {:?} {:?}: {what} (smem {} B, regs {}, local {} B, spill {} B, {} warps / SM)", a.cfg, a.var, a.smem, a.res.regs, a.res.local, a.res.spill, a.res.warps);
            }
        };
        let mut timed: Vec<(f32, RaActive)> = Vec::new();
        let mut spilled: Vec<RaActive> = Vec::new();
        let run = |a: RaActive, timed: &mut Vec<(f32, RaActive)>, spilled: &mut Vec<RaActive>| -> Result<(), NufftError> {
            if skip && a.res.spill > 0 {
                show(&a, "skipped: spills on this device");
                spilled.push(a);
                return Ok(());
            }
            let ms = time(&a)?;
            show(&a, &format!("{ms:.3} ms"));
            timed.push((ms, a));
            Ok(())
        };
        let env = Self::ra_env_cfg();
        if let Some((c, v)) = env.filter(|(_, v)| !v.is_base()) {
            // HASTY_SEG_VSA_CFG with a staging variant: that one only
            assert!(self.ra_cfg_ok_v(c, v), "HASTY_SEG_VSA_CFG: {c:?} {v:?} is not a valid ra shape on this engine");
            let a = self.ra_compile_v(c, v)?;
            let ms = time(&a)?;
            show(&a, &format!("{ms:.3} ms"));
            return Ok(Some((ms, a)));
        }
        for c in self.ra_candidates() {
            run(self.ra_compile(c)?, &mut timed, &mut spilled)?;
        }
        if env.is_none() {
            // phase B: staging variants of the two fastest (nxg, cb) (spilling bases count after the timed ones)
            timed.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut pairs: Vec<(usize, usize)> = Vec::new();
            for c in timed.iter().map(|t| t.1.cfg).chain(spilled.iter().map(|a| a.cfg)) {
                if !pairs.contains(&(c.nxg, c.cb)) && pairs.len() < 2 {
                    pairs.push((c.nxg, c.cb));
                }
            }
            for (nxg, cb) in pairs {
                for (c, v) in self.ra_variant_plan(nxg, cb)? {
                    run(self.ra_compile_v(c, v)?, &mut timed, &mut spilled)?;
                }
            }
            // launch-bounds hints for the three fastest unhinted candidates
            timed.sort_by(|a, b| a.0.total_cmp(&b.0));
            let top: Vec<(RaCfg, RaVar, u32)> = timed.iter().filter(|t| t.1.var.lbmin == 0).take(3).map(|t| (t.1.cfg, t.1.var, t.1.res.warps)).collect();
            for (c, v, w) in top {
                for m in lbmin_set() {
                    let vm = RaVar { lbmin: m, ..v };
                    if !self.ra_cfg_ok_v(c, vm) {
                        continue;
                    }
                    let a = self.ra_compile_v(c, vm)?;
                    if a.res.spill > 0 || a.res.warps < w {
                        show(&a, &format!("hint not kept: spills or fewer warps than unhinted ({w})"));
                        continue;
                    }
                    run(a, &mut timed, &mut spilled)?;
                }
            }
        }
        if timed.is_empty() {
            // every candidate spills on this device: time them all
            for a in spilled {
                let ms = time(&a)?;
                show(&a, &format!("{ms:.3} ms (spills)"));
                timed.push((ms, a));
            }
        }
        Ok(timed.into_iter().min_by(|a, b| a.0.total_cmp(&b.0)))
    }
}

/// The largest chunk ≤ 32 (≥ 8) whose dynamic shared memory (`smem(chk)`, bytes) keeps the register-limited block
/// count of kernel `f` (128 threads) on this device (driver occupancy calculator; static shared memory counted by the
/// driver): the chunk at which shared memory stops costing blocks. None if even 8 does. Raises `f`'s dynamic
/// shared-memory limit to `smax` (use a probe compile).
pub(super) fn chunk_fit(f: &CudaFunction, smax: usize, smem: impl Fn(usize) -> usize) -> Option<usize> {
    let attr = cudarc::driver::sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
    let lim = smax.saturating_sub(RA_STATIC);
    f.set_attribute(attr, lim as i32).ok()?;
    let breg = f.occupancy_max_active_blocks_per_multiprocessor(128, 0, None).ok()?;
    (8..=32usize).rev().find(|&chk| smem(chk) <= lim && f.occupancy_max_active_blocks_per_multiprocessor(128, smem(chk), None).is_ok_and(|b| b >= breg && b > 0))
}
