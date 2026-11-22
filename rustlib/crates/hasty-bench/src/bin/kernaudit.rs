//! kernaudit: offline compile audit of the production segmented-NUFFT and FFT kernels for several GPU
//! architectures. No GPU and no CUDA context are used: NVRTC is a library, and `--ptxas-options=-v` makes it
//! report registers, stack frame, spills, static shared memory and barriers per entry function.
//!
//! For every (case, kernel family, autotune candidate, architecture) the tool builds the kernel source exactly as
//! the engine does (the same `#define` prefix + the same .cu file), compiles it to a cubin for that architecture,
//! parses the ptxas report, adds the dynamic shared memory the engine would request for that shape, and computes
//! the theoretical occupancy (resident blocks and warps per SM, and what limits them) from the architecture table
//! below. Configs that spill, use a stack frame, exceed the per-block shared-memory opt-in or the per-SM limits, or
//! fall below a warp target are flagged.
//!
//! Mirrors (copied, not linked, because the engine keeps them private) as of 2026-11-20:
//! - `segnufft.rs`: `compile` (tile kernels), `TileCfg::{fit_abc, fits, smem, threads}`, the autotune's candidate loops;
//! - `segnufft/vs.rs`: `vs_compile`, `vsw`, `vs_smem`, `vs_ntmax`, `vs_cfg_ok`, `vs_candidates`, `vs_default_geom`;
//! - `segnufft/ra.rs`: `ra_compile`, `ra_smem`, `ra_cfg_ok`, `ra_candidates`;
//! - `segnufft/kl.rs`: `kl_compile`, `kl_smem`, `kl_cfg_ok`, `kl_candidates`;
//! - hasty-fft `plan.rs`: `choose_shape` (the strided axis-pass shape), `AxisPlan::from_spec_impl`'s smem rule.
//! If those change, re-check the mirrors (the tool prints the define prefix it uses with `--show-src`).
//! Phase 2 (2026-11-20, research/sample_kernels.md §19): families `ra2` / `kl2` (staging variants RA_INPLACE,
//! RA_STAGES, smaller chunks, LBMIN: mirrors `ra_smem_v` / `kl_smem_v` of segnufft/ra.rs / kl.rs) and `vs2` (vs / vt
//! sub-chunk staging VSCH: mirror of `vs_smem_v`).
//!
//! Output: `<out>/results.jsonl` (one row per compiled (kernel, arch)), `<out>/summary.md` (tables + flags).
//!
//! Usage:
//!   kernaudit [--out DIR] [--arch sm_80,sm_89,sm_90,sm_100,sm_120] [--cases a,b] [--families tile,vs,ra,kl,fft,whatif,ra2,kl2,vs2]
//!             [--jobs 4] [--warp-target 8] [--cuobjdump] [--show-src] [--list]
//!
//! research/arch_targets.md §5 describes the results; research/experiments/arch_audit/run.sh reruns it.

use std::{
    collections::{BTreeMap, HashMap},
    ffi::{c_char, CStr, CString},
    fs,
    io::Write,
    path::PathBuf,
    ptr,
    sync::Mutex,
};

use cudarc::nvrtc::sys;
use hasty_fft::{factorize, generate_axis_source, AxisCfg};
use hasty_nufft::EsKernel;

const SEG_CU: &str = include_str!("../../../hasty-nufft/src/segnufft.cu");
const KNOT_CU: &str = include_str!("../../../hasty-nufft/src/segnufft_knot.cu");
const VS_CU: &str = include_str!("../../../hasty-nufft/src/segnufft_vs.cu");
const RA_CU: &str = include_str!("../../../hasty-nufft/src/segnufft_ra.cu");
const KLADJ_CU: &str = include_str!("../../../hasty-nufft/src/segnufft_kl_adj.cu");
const NHI: usize = 25;

// ------------------------------------------------------------------------------------------------------------------
// Architecture table (CUDA Programming Guide, compute-capability tables; research/arch_targets.md §1)

#[derive(Clone, Copy, Debug)]
struct Arch {
    name: &'static str,
    /// GPUs of this compute capability we may run on
    gpus: &'static str,
    max_threads_sm: usize,
    max_blocks_sm: usize,
    max_warps_sm: usize,
    regs_sm: usize,
    /// max shared memory per SM (largest carveout) and per block with opt-in
    smem_sm: usize,
    smem_optin: usize,
}

const RESERVED_SMEM_PER_BLOCK: usize = 1024;

const ARCHS: &[Arch] = &[
    Arch { name: "sm_80", gpus: "A100 40/80 GB", max_threads_sm: 2048, max_blocks_sm: 32, max_warps_sm: 64, regs_sm: 65536, smem_sm: 164 * 1024, smem_optin: 163 * 1024 },
    Arch { name: "sm_89", gpus: "RTX 4090, L40S", max_threads_sm: 1536, max_blocks_sm: 24, max_warps_sm: 48, regs_sm: 65536, smem_sm: 100 * 1024, smem_optin: 99 * 1024 },
    Arch { name: "sm_90", gpus: "H100 SXM/PCIe, H200, GH200", max_threads_sm: 2048, max_blocks_sm: 32, max_warps_sm: 64, regs_sm: 65536, smem_sm: 228 * 1024, smem_optin: 227 * 1024 },
    Arch { name: "sm_100", gpus: "B200 (not a target; reference)", max_threads_sm: 2048, max_blocks_sm: 32, max_warps_sm: 64, regs_sm: 65536, smem_sm: 228 * 1024, smem_optin: 227 * 1024 },
    // CC 12.0: the Programming Guide lists 24 blocks / SM, the Blackwell tuning guide 32; 24 is used (never binding here)
    Arch { name: "sm_120", gpus: "RTX 5090", max_threads_sm: 1536, max_blocks_sm: 24, max_warps_sm: 48, regs_sm: 65536, smem_sm: 100 * 1024, smem_optin: 99 * 1024 },
];

#[derive(Clone, Debug, Default)]
struct Occ {
    blocks: usize,
    warps: usize,
    limiter: String,
}

/// Theoretical occupancy (the occupancy-calculator rules): registers allocated per warp in units of 256 and warps per
/// SM floored to the allocation granularity of 4; shared memory per block = static + dynamic + 1 KB reserved, against
/// the largest carveout; thread, warp and block limits. Validated against the 4090 numbers of
/// research/sample_kernels.md (§11.3, §12.7, §15.6).
fn occupancy(a: &Arch, regs: usize, threads: usize, smem: usize) -> Occ {
    let wb = threads.div_ceil(32).max(1);
    let rpw = (regs.max(1) * 32).div_ceil(256) * 256;
    let warps_regs = (a.regs_sm / rpw) / 4 * 4;
    let b_regs = if wb * rpw > a.regs_sm { 0 } else { warps_regs / wb };
    let b_smem = if smem == 0 { usize::MAX } else { a.smem_sm / (smem + RESERVED_SMEM_PER_BLOCK) };
    let b_thr = a.max_threads_sm / threads.max(1);
    let b_warps = a.max_warps_sm / wb;
    let b = b_regs.min(b_smem).min(b_thr).min(b_warps).min(a.max_blocks_sm);
    let mut lim = Vec::new();
    if b == b_regs {
        lim.push("regs");
    }
    if b == b_smem {
        lim.push("smem");
    }
    if b == b_thr.min(b_warps) {
        lim.push("threads");
    }
    if b == a.max_blocks_sm {
        lim.push("blocks");
    }
    Occ { blocks: b, warps: b * wb, limiter: lim.join("+") }
}

// ------------------------------------------------------------------------------------------------------------------
// NVRTC with the ptxas report

#[derive(Clone, Debug, Default)]
struct EntryRes {
    regs: usize,
    stack: usize,
    spill_st: usize,
    spill_ld: usize,
    smem_static: usize,
    barriers: usize,
}

fn nvrtc_compile(name: &str, src: &str, arch: &str, extra: &[&str]) -> Result<(Vec<u8>, String), String> {
    let csrc = CString::new(src).map_err(|_| "NUL in source".to_string())?;
    let cname = CString::new(format!("{name}.cu")).unwrap();
    let mut opts: Vec<CString> = vec![CString::new(format!("--gpu-architecture={arch}")).unwrap(), CString::new("--std=c++17").unwrap(), CString::new("--ptxas-options=-v").unwrap(), CString::new("--no-cache").unwrap()];
    // --no-cache: NVRTC 13 caches cubins itself; a cache hit returns no ptxas report (and would write to disk)
    for o in extra {
        opts.push(CString::new(*o).unwrap());
    }
    let opt_ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
    unsafe {
        let mut prog: sys::nvrtcProgram = ptr::null_mut();
        let r = sys::nvrtcCreateProgram(&mut prog, csrc.as_ptr(), cname.as_ptr(), 0, ptr::null(), ptr::null());
        if r != sys::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("nvrtcCreateProgram: {r:?}"));
        }
        let r = sys::nvrtcCompileProgram(prog, opt_ptrs.len() as i32, opt_ptrs.as_ptr());
        let mut log_size = 0usize;
        let _ = sys::nvrtcGetProgramLogSize(prog, &mut log_size);
        let mut log = vec![0 as c_char; log_size.max(1)];
        let _ = sys::nvrtcGetProgramLog(prog, log.as_mut_ptr());
        let log_str = CStr::from_ptr(log.as_ptr()).to_string_lossy().into_owned();
        if r != sys::nvrtcResult::NVRTC_SUCCESS {
            let _ = sys::nvrtcDestroyProgram(&mut prog);
            return Err(format!("{r:?}\n{log_str}"));
        }
        let mut size = 0usize;
        let r = sys::nvrtcGetCUBINSize(prog, &mut size);
        if r != sys::nvrtcResult::NVRTC_SUCCESS || size == 0 {
            let _ = sys::nvrtcDestroyProgram(&mut prog);
            return Err(format!("nvrtcGetCUBINSize: {r:?} (is `{arch}` a real arch?)"));
        }
        let mut cubin = vec![0u8; size];
        let _ = sys::nvrtcGetCUBIN(prog, cubin.as_mut_ptr() as *mut c_char);
        let _ = sys::nvrtcDestroyProgram(&mut prog);
        Ok((cubin, log_str))
    }
}

fn nvrtc_version() -> String {
    let (mut a, mut b) = (0, 0);
    unsafe {
        let _ = sys::nvrtcVersion(&mut a, &mut b);
    }
    format!("{a}.{b}")
}

/// numbers before a keyword in a ptxas line ("40 bytes spill stores" → 40)
fn num_before(line: &str, key: &str) -> Option<usize> {
    let i = line.find(key)?;
    line[..i].split(|c: char| !c.is_ascii_digit()).filter(|s| !s.is_empty()).last()?.parse().ok()
}

/// parse the ptxas -v report: per entry function its properties
fn parse_ptxas(log: &str) -> HashMap<String, EntryRes> {
    let mut out: HashMap<String, EntryRes> = HashMap::new();
    let mut cur: Option<String> = None;
    for line in log.lines() {
        if let Some(i) = line.find("Function properties for ") {
            cur = Some(line[i + 24..].trim().to_string());
            out.entry(cur.clone().unwrap()).or_default();
            continue;
        }
        if let Some(i) = line.find("Compiling entry function '") {
            let rest = &line[i + 26..];
            cur = rest.find('\'').map(|j| rest[..j].to_string());
            if let Some(c) = &cur {
                out.entry(c.clone()).or_default();
            }
            continue;
        }
        let Some(c) = &cur else { continue };
        let e = out.get_mut(c).unwrap();
        if line.contains("bytes stack frame") {
            e.stack = num_before(line, "bytes stack frame").unwrap_or(0);
            e.spill_st = num_before(line, "bytes spill stores").unwrap_or(0);
            e.spill_ld = num_before(line, "bytes spill loads").unwrap_or(0);
        }
        if line.contains("Used ") && line.contains("registers") {
            e.regs = num_before(line, "registers").unwrap_or(0);
            e.barriers = num_before(line, "barriers").unwrap_or(0);
            e.smem_static = num_before(line, "bytes smem").unwrap_or(0);
        }
    }
    out
}

// ------------------------------------------------------------------------------------------------------------------
// Cases (engine shapes the reconstruction and the benchmarks run)

#[derive(Clone, Copy, Debug)]
struct Case {
    name: &'static str,
    what: &'static str,
    nseg: usize,
    order: u32,
    /// coils per batch (grid channels = cb · nseg)
    cb: usize,
    /// knot support (0 = static image) and respiratory degree (−1 = 1-D)
    ks: usize,
    rdeg: i32,
    w: usize,
    sigma: f64,
    max_item: usize,
}

const CASES: &[Case] = &[
    Case { name: "vel2d_o3_cb5", what: "velocity recon: 2-D knots (resp linear x cardiac cubic), order 3, L 10, coil batch 5 (50 ch)", nseg: 10, order: 3, cb: 5, ks: 4, rdeg: 1, w: 10, sigma: 1.25, max_item: 1024 },
    Case { name: "vel2d_o3_cb10", what: "as vel2d_o3_cb5 with coil batch 10 (100 ch: the big-VRAM batch)", nseg: 10, order: 3, cb: 10, ks: 4, rdeg: 1, w: 10, sigma: 1.25, max_item: 1024 },
    Case { name: "knot1d_o2_cb5", what: "1-D cubic knots, order 2, L 10, coil batch 5", nseg: 10, order: 2, cb: 5, ks: 4, rdeg: -1, w: 10, sigma: 1.25, max_item: 1024 },
    Case { name: "static_o2_cb5", what: "static image, order 2, L 10, coil batch 5", nseg: 10, order: 2, cb: 5, ks: 0, rdeg: -1, w: 10, sigma: 1.25, max_item: 1024 },
    Case { name: "static_o0_10x10", what: "static image, order 0, 10 coils x L 10 in one batch (prototype 10 x 10)", nseg: 10, order: 0, cb: 10, ks: 0, rdeg: -1, w: 10, sigma: 1.25, max_item: 1024 },
    Case { name: "static_o0_48x1", what: "static image, order 0, 48 coils x L 1 in one batch (ra nxg 2 cb 3 spill case)", nseg: 1, order: 0, cb: 48, ks: 0, rdeg: -1, w: 10, sigma: 1.25, max_item: 1024 },
];

// ------------------------------------------------------------------------------------------------------------------
// Jobs

#[derive(Clone, Debug)]
struct Job {
    arch: Arch,
    case: String,
    family: &'static str,
    cfg: String,
    /// module name (as the engine's cache name)
    module: String,
    src: String,
    /// entries to report (function name, threads per block, dynamic smem bytes, role)
    entries: Vec<(String, usize, usize, &'static str)>,
    extra_opts: Vec<&'static str>,
    note: String,
}

fn beta(c: &Case) -> f64 {
    EsKernel::with_width(c.w, c.sigma).beta
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

// ---- tile kernels (segnufft.cu [+ segnufft_knot.cu]) ----

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct TileCfg {
    tt: usize,
    cpba: usize,
    cpbf: usize,
    ppf: usize,
    chunk: usize,
    max_item: usize,
    acb: usize,
    fsb: usize,
    fcb: usize,
    kv: usize,
}

fn pcvb(nch: usize) -> usize {
    let mut v = 32;
    while v > 4 && v * (nch | 1) > 2048 {
        v /= 2;
    }
    v
}

fn adj_store_smem(tt3: usize, cpb: usize, nseg: usize) -> usize {
    8 * tt3 * ((cpb * nseg) | 1)
}

impl TileCfg {
    fn smem(&self, c: &Case, nf: usize) -> [usize; 2] {
        let (tt3, ch, nseg) = (self.tt.pow(3), self.chunk, c.nseg);
        let nk = [1usize, 3, 5][c.order.min(2) as usize];
        let up = |b: usize, a: usize| b.div_ceil(a) * a;
        let st = 48 * ch + 4 * 3 * nk * self.tt * ch + 4 * 15 * if c.order >= 2 { ch } else { 1 } + 4 * ch;
        let wb = if self.kv >= 1 && c.order == 0 { 4 } else { 8 };
        let st = up(up(st, 8) + 8 * nf * nseg * ch + wb * (tt3 + self.ppf * nf) * ch, 16);
        let base = up(st + 4 * self.max_item, 16);
        let base = base + if c.ks > 0 && c.order >= 3 { up(4 * NHI * ch, 16) } else { 0 };
        let gn = nf * self.cpbf * nseg;
        let gts = gn + (2 + 4 - gn % 4) % 4;
        let adj = if self.kv >= 1 { up(base + 8 * ch * self.cpba, 16) + 8 * ch * self.cpba * nseg } else { base + 8 * ch * self.cpba };
        [adj.max(adj_store_smem(tt3, self.cpba, nseg)), base + 8 * tt3 * gts]
    }
    fn threads(&self, nf: usize) -> (usize, usize) {
        (self.chunk * self.cpbf * self.ppf * nf / (self.fsb * self.fcb), self.tt.pow(3) * nf * self.cpba / self.acb)
    }
    fn fits(&self, c: &Case, nf: usize, smax: usize) -> bool {
        let fthreads = self.chunk * self.cpbf * self.ppf * nf;
        let fb = self.fsb * self.fcb;
        (self.kv == 0 || (c.ks > 0 && self.tt == 4 && self.kv <= 2))
            && (self.kv != 2 || self.fcb == 1)
            && self.fcb >= 1
            && self.smem(c, nf).iter().all(|&b| b <= smax)
            && self.chunk % self.fsb == 0
            && self.cpbf % self.fcb == 0
            && fthreads % fb == 0
            && (fthreads / fb) % 32 == 0
            && self.cpba % self.acb == 0
            && self.tt.pow(3) * nf * self.cpba / self.acb <= 1024
    }
    #[allow(clippy::too_many_arguments)]
    fn fit_abc(c: &Case, nf: usize, tt: usize, cpba: usize, cpbf: usize, ppf: usize, acb: usize, fsb: usize, fcb: usize, kv: usize, smax: usize) -> Option<Self> {
        let lanes = ppf * nf;
        if !ppf.is_power_of_two() || lanes > 32 || cpba % acb != 0 || cpbf % fcb != 0 || tt.pow(3) * nf * cpba / acb > 1024 {
            return None;
        }
        let mut ch = (if fsb > 1 { 64usize } else { 32 }).min(512 * fsb * fcb / (cpbf * lanes));
        while ch > 0 {
            let t = Self { tt, cpba, cpbf, ppf, chunk: ch, max_item: c.max_item, acb, fsb, fcb, kv };
            if t.fits(c, nf, smax) {
                return Some(t);
            }
            ch -= 1;
        }
        None
    }
    fn tag(&self) -> String {
        format!("tt{} cpba{} cpbf{} ppf{} chunk{} acb{} fsb{} fcb{} kv{}", self.tt, self.cpba, self.cpbf, self.ppf, self.chunk, self.acb, self.fsb, self.fcb, self.kv)
    }
}

fn tile_src(c: &Case, nf: usize, tc: &TileCfg) -> (String, String) {
    let src = format!(
        "#define W {}\n#define BETA {:.17e}\n#define NSEG {}\n#define CORR {}\n#define NF {}\n#define NCB {}\n#define TT {}\n#define CPBA {}\n#define CPBF {}\n#define PPF {}\n#define CHUNK {}\n#define LISTMAX {}\n#define PCVB {}\n#define ACB {}\n#define FSB {}\n#define FCB {}\n#define KV {}\n{}",
        c.w,
        beta(c),
        c.nseg,
        c.order,
        nf,
        c.cb,
        tc.tt,
        tc.cpba,
        tc.cpbf,
        tc.ppf,
        tc.chunk,
        tc.max_item,
        pcvb(c.cb * c.nseg),
        tc.acb,
        tc.fsb,
        tc.fcb,
        tc.kv,
        if c.ks > 0 { format!("#define KS {}\n#define RDEG {}\n{}\n{}", c.ks, c.rdeg, SEG_CU, KNOT_CU) } else { SEG_CU.to_string() }
    );
    let name = format!(
        "segnufft{}_w{}_l{}_o{}_f{}_c{}_t{}_{}_{}_{}_{}_{}_a{}_s{}_v{}{}",
        if c.ks > 0 { format!("_k{}{}", c.ks, if c.rdeg >= 0 { format!("r{}", c.rdeg) } else { String::new() }) } else { String::new() },
        c.w,
        c.nseg,
        c.order,
        nf,
        c.cb,
        tc.tt,
        tc.cpba,
        tc.cpbf,
        tc.ppf,
        tc.chunk,
        tc.max_item,
        tc.acb,
        tc.fsb,
        tc.kv,
        if tc.fcb > 1 { format!("_b{}", tc.fcb) } else { String::new() }
    );
    (src, name)
}

/// the tile autotune's forward and adjoint candidates (segnufft.rs `autotune`), on the device's opt-in limit
fn tile_jobs(c: &Case, a: &Arch, jobs: &mut Vec<Job>) {
    if c.ks == 0 && c.order >= 3 {
        return; // static order 3 runs the multi-image kernels (not audited)
    }
    let nf = 1;
    let cb = c.cb;
    let smax = a.smem_optin.max(48 * 1024);
    let mut groups: Vec<usize> = [1usize, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96].into_iter().filter(|&g| g <= cb && (cb <= 16 || g >= 4)).collect();
    groups.push(cb);
    groups.sort_unstable();
    groups.dedup();
    let few = c.nseg <= 2;
    let kvs: Vec<usize> = if c.ks > 0 { vec![0, 1] } else { vec![0] };
    let fsbs = [1usize, 2, 4];
    let fcbs: Vec<usize> = if few { vec![1, 2, 4, 8] } else { vec![1] };
    let blk: Vec<(usize, usize)> = fsbs.iter().map(|&f| (f, 1)).chain(fcbs.iter().filter(|&&b| b > 1).map(|&b| (1, b))).collect();
    let acbs: Vec<usize> = if few { vec![1, 2, 4, 8] } else { vec![1, 2, 4] };
    let (fname, aname) = if c.ks > 0 { ("seg_forward_k", "seg_adjoint_k") } else { ("seg_forward_t", "seg_adjoint_t") };
    let mut seen = std::collections::HashSet::new();
    for &cpbf in &groups {
        for &kv in &kvs {
            for ppf in [1usize, 2, 4] {
                for &(fsb, fcb) in &blk {
                    if !(fcb == 1 || ppf <= 2) {
                        continue;
                    }
                    let Some(tc) = TileCfg::fit_abc(c, nf, 4, 1, cpbf, ppf, 1, fsb, fcb, kv, smax) else { continue };
                    if tc.chunk < 8 || tc.threads(nf).0 > 512 || !seen.insert(tc) {
                        continue;
                    }
                    let (src, name) = tile_src(c, nf, &tc);
                    jobs.push(Job { arch: *a, case: c.name.into(), family: "tile_fwd", cfg: tc.tag(), module: name, src, entries: vec![(fname.into(), tc.threads(nf).0, tc.smem(c, nf)[1], "fwd")], extra_opts: vec![], note: String::new() });
                }
            }
        }
    }
    // adjoint: the autotune holds the forward's best (cpbf, ppf, fsb, fcb); here the default lanes (cpbf 1, ppf 2)
    for &cpba in &groups {
        for &acb in &acbs {
            for &kv in &kvs {
                let Some(tc) = TileCfg::fit_abc(c, nf, 4, cpba, 1, 2, acb, 1, 1, kv, smax) else { continue };
                if tc.threads(nf).1 > 1024 || !seen.insert(tc) {
                    continue;
                }
                let (src, name) = tile_src(c, nf, &tc);
                jobs.push(Job { arch: *a, case: c.name.into(), family: "tile_adj", cfg: tc.tag(), module: name, src, entries: vec![(aname.into(), tc.threads(nf).1, tc.smem(c, nf)[0], "adj")], extra_opts: vec![], note: String::new() });
            }
        }
    }
}

// ---- vs / vt forward (segnufft_vs.cu) ----

#[derive(Clone, Copy, Debug)]
struct VsGeom {
    tx: usize,
    ty: usize,
    tz: usize,
    smax: usize,
}
#[derive(Clone, Copy, Debug)]
struct VsCfg {
    nwc: usize,
    nws: usize,
    vxpad: usize,
}

fn vsw(order: u32, w: usize, g: VsGeom, vxpad: usize) -> usize {
    if order >= 2 {
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

fn vs_smem(order: u32, w: usize, g: VsGeom, c: VsCfg) -> usize {
    let tab = ((g.tz + w - 1) * g.tx * g.ty * 4 + 15) & !15;
    let vt = if order >= 2 { c.nwc * c.nws * 1600 } else { 0 };
    g.smax * vsw(order, w, g, c.vxpad) * 4 + tab + c.nwc * 2 * (g.smax * 16 + 16) * 4 + vt
}

fn vs_ntmax(order: u32, g: VsGeom) -> usize {
    if order == 0 && g.tx <= 2 { 384 } else { 256 }
}

fn vs_ok(c: &Case) -> bool {
    (c.order <= 2 || (c.ks > 0 && c.order <= 3)) && c.w <= 10
}

fn vs_cfg_ok(c: &Case, g: VsGeom, v: VsCfg, smax: usize) -> bool {
    let gr = (c.cb * c.nseg).div_ceil(16);
    let rows = gr.div_ceil(v.nwc.max(1));
    v.nwc >= 1
        && v.nwc <= gr
        && v.nws >= 1
        && g.ty % v.nws == 0
        && v.vxpad <= 1
        && !(v.vxpad == 1 && g.tx == 1)
        && !(v.vxpad == 1 && c.order >= 2)
        && (rows == 1 || (16 * v.nwc) % c.nseg == 0)
        && 32 * v.nwc * v.nws <= vs_ntmax(c.order, g)
        && vs_smem(c.order, c.w, g, v) <= smax
}

fn vs_default_geom(c: &Case) -> VsGeom {
    let nch = c.cb * c.nseg;
    match c.order {
        0 => VsGeom { tx: 4, ty: 4, tz: 16, smax: if nch >= 64 { 64 } else { 32 } },
        1 => VsGeom { tx: 2, ty: 4, tz: 16, smax: 32 },
        _ => VsGeom { tx: 4, ty: 4, tz: 16, smax: 48 },
    }
}

fn vs_candidates(c: &Case, g: VsGeom, smax: usize) -> Vec<VsCfg> {
    let gr = (c.cb * c.nseg).div_ceil(16);
    let mut nwcs: Vec<usize> = (1..=gr.min(4)).collect();
    if let Some(n) = (1..=gr).rev().find(|&n| vs_cfg_ok(c, g, VsCfg { nwc: n, nws: 1, vxpad: 0 }, smax)) {
        nwcs.push(n);
    }
    nwcs.sort_unstable();
    nwcs.dedup();
    let mut v = Vec::new();
    for &nwc in &nwcs {
        for nws in [1usize, 2] {
            for vxpad in if c.order >= 2 { 0..1usize } else { 0..2 } {
                let x = VsCfg { nwc, nws, vxpad };
                if vs_cfg_ok(c, g, x, smax) {
                    v.push(x);
                }
            }
        }
    }
    v
}

fn vs_job(c: &Case, a: &Arch, g: VsGeom, v: VsCfg, family: &'static str, note: &str) -> Job {
    let src = format!(
        "#define W {}\n#define BETA {:.17e}\n#define NSEG {}\n#define CORR {}\n#define NCB {}\n#define TX {}\n#define TY {}\n#define TZ {}\n#define SMAX {}\n#define NWC {}\n#define NWS {}\n#define VXPAD {}\n#define KS {}\n#define RDEG {}\n{}",
        c.w, beta(c), c.nseg, c.order, c.cb, g.tx, g.ty, g.tz, g.smax, v.nwc, v.nws, v.vxpad, c.ks, c.rdeg, VS_CU
    );
    let name = format!("segnufft_vs_w{}_l{}_o{}_c{}_k{}r{}_t{}x{}x{}_s{}_g{}_{}_{}", c.w, c.nseg, c.order, c.cb, c.ks, c.rdeg, g.tx, g.ty, g.tz, g.smax, v.nwc, v.nws, v.vxpad);
    let cfg = format!("{}x{}x{} smax{} nwc{} nws{} vxpad{}", g.tx, g.ty, g.tz, g.smax, v.nwc, v.nws, v.vxpad);
    Job { arch: *a, case: c.name.into(), family, cfg, module: name, src, entries: vec![("seg_forward_vs".into(), 32 * v.nwc * v.nws, vs_smem(c.order, c.w, g, v), "fwd")], extra_opts: vec![], note: note.into() }
}

fn vs_jobs(c: &Case, a: &Arch, jobs: &mut Vec<Job>) {
    if !vs_ok(c) {
        return;
    }
    let smax = a.smem_optin;
    let mut geoms = vec![vs_default_geom(c)];
    // geometries of the prototype / tests (research/sample_kernels.md §12.6, §13.5, §14.2)
    for g in [VsGeom { tx: 2, ty: 4, tz: 16, smax: 32 }, VsGeom { tx: 4, ty: 4, tz: 16, smax: 64 }, VsGeom { tx: 2, ty: 4, tz: 8, smax: 48 }, VsGeom { tx: 2, ty: 4, tz: 8, smax: 32 }] {
        if !geoms.iter().any(|h| h.tx == g.tx && h.ty == g.ty && h.tz == g.tz && h.smax == g.smax) {
            geoms.push(g);
        }
    }
    let fam = if c.order >= 2 { "vt_fwd" } else { "vs_fwd" };
    for g in geoms {
        for v in vs_candidates(c, g, smax) {
            jobs.push(vs_job(c, a, g, v, fam, ""));
        }
    }
}

// ---- ra / kl adjoint (segnufft_ra.cu, segnufft_kl_adj.cu) ----

#[derive(Clone, Copy, Debug)]
struct RaCfg {
    nxg: usize,
    cb: usize,
    chk: usize,
}
impl RaCfg {
    fn cpw(&self) -> usize {
        32 / self.nxg * self.cb
    }
}
const TT_RA: usize = 4;
const RA_STATIC: usize = 256;
fn a16(x: usize) -> usize {
    (x + 15) & !15
}

fn ra_smem(order: u32, nseg: usize, ncb: usize, listmax: usize, c: RaCfg) -> usize {
    let nka = match order {
        0 => 1,
        1 => 3,
        _ => 5,
    };
    let (ncl, cpw) = (32 / c.nxg, c.cpw());
    let cbp = c.cb.div_ceil(2) * 2;
    let ncbrm = ((cpw + nseg - 1) / nseg + 1).min(ncb);
    let chk = c.chk;
    let rec = a16(4 * listmax);
    let ry = a16(rec + chk * 48);
    let ra = a16(ry + chk * ncbrm * 8);
    let ax = a16(ra + chk * nseg * 8);
    let bx = a16(ax + chk * 3 * nka * TT_RA * 4);
    let e = a16(bx + chk * 4);
    let h = a16(e + if order >= 2 { chk * 15 * 4 } else { 0 });
    let z = a16(h + if order >= 3 { chk * 25 * 4 } else { 0 });
    let w = a16(z + chk * (cbp / 2) * ncl * 16);
    let stage = w + if order >= 1 { chk * 16 * TT_RA * 8 } else { 0 };
    let park = TT_RA * TT_RA * (cpw | 1) * 8;
    stage.max(park)
}

fn kl_smem(order: u32, nseg: usize, ncb: usize, c: RaCfg) -> usize {
    let nka = match order {
        0 => 1,
        1 => 3,
        _ => 5,
    };
    let (ncl, cpw) = (32 / c.nxg, c.cpw());
    let cbp = c.cb.div_ceil(2) * 2;
    let ncbrm = ((cpw + nseg - 1) / nseg + 1).min(ncb);
    let chk = c.chk;
    let ht = a16(4 * chk);
    let uw = a16(ht + if order >= 3 { 16 * 25 } else { 0 });
    let rec = a16(uw + if order >= 3 { 16 * chk } else { 0 });
    let ry = a16(rec + chk * 48);
    let ra = a16(ry + chk * ncbrm * 8);
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

fn ra_ok(c: &Case) -> bool {
    c.order <= 2 || (c.ks > 0 && c.order <= 3)
}

fn ra_cfg_ok(c: &Case, r: RaCfg, listmax: usize, smax: usize) -> bool {
    let nchv = c.cb * c.nseg;
    matches!(r.nxg, 1 | 2 | 4) && r.cb >= 1 && TT_RA / r.nxg * r.cb <= 16 && (1..=64).contains(&r.chk) && nchv.div_ceil(r.cpw()) <= 65535 && ra_smem(c.order, c.nseg, c.cb, listmax, r) + RA_STATIC <= smax
}

fn kl_cfg_ok(c: &Case, r: RaCfg, smax: usize) -> bool {
    let nchv = c.cb * c.nseg;
    matches!(r.nxg, 1 | 2 | 4) && r.cb >= 1 && 4 / r.nxg * r.cb <= 16 && (1..=64).contains(&r.chk) && nchv.div_ceil(r.cpw()) <= 65535 && kl_smem(c.order, c.nseg, c.cb, r) + 256 <= smax
}

/// ra / kl candidates (identical rule): per x grouping the channel blocks covering the batch in 1–4 block rows, chunks 16 / 32
fn ra_candidates(c: &Case) -> Vec<RaCfg> {
    let nchv = c.cb * c.nseg;
    let mut v: Vec<RaCfg> = Vec::new();
    for nxg in [1usize, 2, 4] {
        let ncl = 32 / nxg;
        let mut cbs: Vec<usize> = (1..=4).map(|rows| nchv.div_ceil(ncl * rows)).filter(|&cb| cb >= 1 && TT_RA / nxg * cb <= 16).collect();
        cbs.dedup();
        for cb in cbs {
            for chk in [16usize, 32] {
                let r = RaCfg { nxg, cb, chk };
                if !v.iter().any(|x| x.nxg == r.nxg && x.cb == r.cb && x.chk == r.chk) {
                    v.push(r);
                }
            }
        }
    }
    v
}

fn ra_job(c: &Case, a: &Arch, r: RaCfg, listmax: usize, family: &'static str, note: &str) -> Job {
    let src = format!(
        "#define W {}\n#define BETA {:.17e}\n#define NSEG {}\n#define CORR {}\n#define NCB {}\n#define LISTMAX {}\n#define NXG {}\n#define CB {}\n#define CHK {}\n#define KS {}\n#define RDEG {}\n{}",
        c.w, beta(c), c.nseg, c.order, c.cb, listmax, r.nxg, r.cb, r.chk, c.ks, c.rdeg, RA_CU
    );
    let name = format!("segnufft_ra_w{}_l{}_o{}_c{}_k{}r{}_m{}_x{}_b{}_h{}", c.w, c.nseg, c.order, c.cb, c.ks, c.rdeg, listmax, r.nxg, r.cb, r.chk);
    let cfg = format!("nxg{} cb{} chk{} listmax{}", r.nxg, r.cb, r.chk, listmax);
    Job { arch: *a, case: c.name.into(), family, cfg, module: name, src, entries: vec![("seg_adjoint_ra".into(), 32 * TT_RA, ra_smem(c.order, c.nseg, c.cb, listmax, r), "adj")], extra_opts: vec![], note: note.into() }
}

fn ra_jobs(c: &Case, a: &Arch, jobs: &mut Vec<Job>) {
    if !ra_ok(c) {
        return;
    }
    for r in ra_candidates(c) {
        if ra_cfg_ok(c, r, c.max_item, a.smem_optin) {
            jobs.push(ra_job(c, a, r, c.max_item, "ra_adj", ""));
        }
    }
}

fn kl_jobs(c: &Case, a: &Arch, jobs: &mut Vec<Job>) {
    if c.ks == 0 || !ra_ok(c) {
        return;
    }
    for r in ra_candidates(c) {
        if !kl_cfg_ok(c, r, a.smem_optin) {
            continue;
        }
        let src = format!(
            "#define W {}\n#define BETA {:.17e}\n#define NSEG {}\n#define CORR {}\n#define NCB {}\n#define NXG {}\n#define CB {}\n#define CHK {}\n#define KS {}\n#define RDEG {}\n{}",
            c.w, beta(c), c.nseg, c.order, c.cb, r.nxg, r.cb, r.chk, c.ks, c.rdeg, KLADJ_CU
        );
        let name = format!("segnufft_kladj_w{}_l{}_o{}_c{}_k{}r{}_x{}_b{}_h{}", c.w, c.nseg, c.order, c.cb, c.ks, c.rdeg, r.nxg, r.cb, r.chk);
        let cfg = format!("nxg{} cb{} chk{}", r.nxg, r.cb, r.chk);
        jobs.push(Job { arch: *a, case: c.name.into(), family: "kl_adj", cfg, module: name, src, entries: vec![("seg_adjoint_kl".into(), 128, kl_smem(c.order, c.nseg, c.cb, r), "adj")], extra_opts: vec![], note: String::new() });
    }
}

/// shapes the phase-2 list proposes for large-shared-memory parts (sm_80 / sm_90): larger vs/vt items, longer ra lists.
/// Same kernels and defines; only the tunables change. Registered as family `whatif` with the idea in `note`.
fn whatif_jobs(c: &Case, a: &Arch, jobs: &mut Vec<Job>) {
    if vs_ok(c) {
        let g0 = vs_default_geom(c);
        for smax in [96usize, 128, 192, 256] {
            let g = VsGeom { smax, ..g0 };
            for v in vs_candidates(c, g, a.smem_optin).into_iter().filter(|v| v.nws == 1 && v.vxpad == 0) {
                jobs.push(vs_job(c, a, g, v, "whatif", "vs/vt larger items (SMAX) for big shared memory"));
            }
        }
    }
    if ra_ok(c) {
        for lm in [2048usize, 4096] {
            for r in ra_candidates(c).into_iter().filter(|r| r.chk == 16) {
                if ra_cfg_ok(c, r, lm, a.smem_optin) {
                    jobs.push(ra_job(c, a, r, lm, "whatif", "ra longer item lists (fewer split tiles, fewer atomics)"));
                }
            }
            // deeper chunks (§15.8 item 2): CHK 48 / 64
        }
        for chk in [48usize, 64] {
            for r in ra_candidates(c).into_iter().filter(|r| r.chk == 16) {
                let r = RaCfg { chk, ..r };
                if ra_cfg_ok(c, r, c.max_item, a.smem_optin) {
                    jobs.push(ra_job(c, a, r, c.max_item, "whatif", "ra deeper staged chunks (fewer barriers per sample)"));
                }
            }
        }
    }
}

// ---- phase 2: staging variants of ra / kl, sub-chunk staging of vs / vt ----

#[derive(Clone, Copy, Debug)]
struct RaVar {
    inplace: bool,
    stages: usize,
    lbmin: usize,
}

fn ra_smem_v(order: u32, nseg: usize, ncb: usize, listmax: usize, c: RaCfg, v: RaVar) -> usize {
    if !v.inplace {
        return ra_smem(order, nseg, ncb, listmax, c);
    }
    // segnufft/ra.rs ra_smem_v: entries [st] | htab | uw [st] | records [st] | y [st] | (as ra_smem from w_l on)
    let nka = match order {
        0 => 1,
        1 => 3,
        _ => 5,
    };
    let (ncl, cpw) = (32 / c.nxg, c.cpw());
    let cbp = c.cb.div_ceil(2) * 2;
    let ncbrm = ((cpw + nseg - 1) / nseg + 1).min(ncb);
    let (chk, st) = (c.chk, v.stages);
    let ht = a16(st * 4 * chk);
    let uw = a16(ht + if order >= 3 { 16 * 25 } else { 0 });
    let rec = a16(uw + if order >= 3 { st * 16 * chk } else { 0 });
    let ry = a16(rec + st * chk * 48);
    let ra = a16(ry + st * chk * ncbrm * 8);
    let ax = a16(ra + chk * nseg * 8);
    let bx = a16(ax + chk * 3 * nka * TT_RA * 4);
    let e = a16(bx + chk * 4);
    let h = a16(e + if order >= 2 { chk * 15 * 4 } else { 0 });
    let z = a16(h + if order >= 3 { chk * 25 * 4 } else { 0 });
    let w = a16(z + chk * (cbp / 2) * ncl * 16);
    let stage = w + if order >= 1 { chk * 16 * TT_RA * 8 } else { 0 };
    let park = TT_RA * TT_RA * (cpw | 1) * 8;
    stage.max(park)
}

fn kl_smem_v(order: u32, nseg: usize, ncb: usize, c: RaCfg, v: RaVar) -> usize {
    // segnufft/kl.rs kl_smem_v: the per-chunk raw inputs (entries, uw, records, y) times the stage count
    let nka = match order {
        0 => 1,
        1 => 3,
        _ => 5,
    };
    let (ncl, cpw) = (32 / c.nxg, c.cpw());
    let cbp = c.cb.div_ceil(2) * 2;
    let ncbrm = ((cpw + nseg - 1) / nseg + 1).min(ncb);
    let (chk, sg) = (c.chk, v.stages);
    let ht = a16(sg * 4 * chk);
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

fn ra_var_defs(v: RaVar, kl: bool) -> String {
    if kl {
        format!("#define RA_STAGES {}\n#define LBMIN {}\n", v.stages, v.lbmin)
    } else {
        format!("#define RA_INPLACE {}\n#define RA_STAGES {}\n#define LBMIN {}\n", v.inplace as u32, v.stages, v.lbmin)
    }
}

/// (nxg, cb) shapes audited for the staging variants: one block row at nxg 4 and nxg 2, two block rows at nxg 4 (the
/// shapes the 4090 autotune picks for every case of research/sample_kernels.md §16–§17), each if a ra candidate
fn p2_pairs(c: &Case) -> Vec<(usize, usize)> {
    let nch = c.cb * c.nseg;
    let cands = ra_candidates(c);
    let mut v = Vec::new();
    for (nxg, rows) in [(4usize, 1usize), (2, 1), (4, 2)] {
        let cb = nch.div_ceil(32 / nxg * rows);
        if cands.iter().any(|r| r.nxg == nxg && r.cb == cb) && !v.contains(&(nxg, cb)) {
            v.push((nxg, cb));
        }
    }
    v
}

/// staging variants audited per shape: chunks 8 / 12 / 16 / 24 (the autotune's 8, 12, 16 and its device fit, ≤ 24 on
/// sm_89 / sm_120 and larger on sm_80 / sm_90) × stages 1 / 2 × hint 0 / 4, and hint 3 at (12, cp.async) and (16, 1)
fn p2_vars(inplace: bool) -> Vec<(usize, RaVar)> {
    let mut v = Vec::new();
    for chk in [8usize, 12, 16, 24] {
        for stages in [1usize, 2] {
            for lbmin in [0usize, 4] {
                v.push((chk, RaVar { inplace, stages, lbmin }));
            }
        }
    }
    v.push((12, RaVar { inplace, stages: 2, lbmin: 3 }));
    v.push((16, RaVar { inplace, stages: 1, lbmin: 3 }));
    v
}

/// ra staging variants: every (nxg, cb) of the candidates × chunks × {in place + register prefetch, in place +
/// cp.async} × LBMIN {0, 3, 4}; plus the base kernel's LBMIN 3 / 4 at chunk 16
fn ra2_jobs(c: &Case, a: &Arch, jobs: &mut Vec<Job>) {
    if !ra_ok(c) {
        return;
    }
    for &(nxg, cb) in &p2_pairs(c) {
        let mut vs = p2_vars(true);
        for lbmin in [3usize, 4] {
            vs.push((16, RaVar { inplace: false, stages: 1, lbmin }));
        }
        for (chk, v) in vs {
            let r = RaCfg { nxg, cb, chk };
            let smem = ra_smem_v(c.order, c.nseg, c.cb, c.max_item, r, v);
            if smem + RA_STATIC > a.smem_optin {
                continue;
            }
            let src = format!(
                "{}#define W {}\n#define BETA {:.17e}\n#define NSEG {}\n#define CORR {}\n#define NCB {}\n#define LISTMAX {}\n#define NXG {}\n#define CB {}\n#define CHK {}\n#define KS {}\n#define RDEG {}\n{}",
                ra_var_defs(v, false), c.w, beta(c), c.nseg, c.order, c.cb, c.max_item, r.nxg, r.cb, r.chk, c.ks, c.rdeg, RA_CU
            );
            let name = format!("segnufft_ra_w{}_l{}_o{}_c{}_k{}r{}_m{}_x{}_b{}_h{}_v{}{}{}", c.w, c.nseg, c.order, c.cb, c.ks, c.rdeg, c.max_item, r.nxg, r.cb, r.chk, v.inplace as u32, v.stages, v.lbmin);
            let cfg = format!("nxg{} cb{} chk{} inplace{} stages{} lbmin{}", r.nxg, r.cb, r.chk, v.inplace as u32, v.stages, v.lbmin);
            jobs.push(Job { arch: *a, case: c.name.into(), family: "ra2_adj", cfg, module: name, src, entries: vec![("seg_adjoint_ra".into(), 32 * TT_RA, smem, "adj")], extra_opts: vec![], note: String::new() });
        }
    }
}

/// kl staging variants: every (nxg, cb) × chunks × stages {1, 2} × LBMIN {0, 3, 4}
fn kl2_jobs(c: &Case, a: &Arch, jobs: &mut Vec<Job>) {
    if c.ks == 0 || !ra_ok(c) {
        return;
    }
    for &(nxg, cb) in &p2_pairs(c) {
        for (chk, v) in p2_vars(false) {
            {
                {
                    let r = RaCfg { nxg, cb, chk };
                    let smem = kl_smem_v(c.order, c.nseg, c.cb, r, v);
                    if smem + 256 > a.smem_optin {
                        continue;
                    }
                    let src = format!(
                        "{}#define W {}\n#define BETA {:.17e}\n#define NSEG {}\n#define CORR {}\n#define NCB {}\n#define NXG {}\n#define CB {}\n#define CHK {}\n#define KS {}\n#define RDEG {}\n{}",
                        ra_var_defs(v, true), c.w, beta(c), c.nseg, c.order, c.cb, r.nxg, r.cb, r.chk, c.ks, c.rdeg, KLADJ_CU
                    );
                    let name = format!("segnufft_kladj_w{}_l{}_o{}_c{}_k{}r{}_x{}_b{}_h{}_v0{}{}", c.w, c.nseg, c.order, c.cb, c.ks, c.rdeg, r.nxg, r.cb, r.chk, v.stages, v.lbmin);
                    let cfg = format!("nxg{} cb{} chk{} stages{} lbmin{}", r.nxg, r.cb, r.chk, v.stages, v.lbmin);
                    jobs.push(Job { arch: *a, case: c.name.into(), family: "kl2_adj", cfg, module: name, src, entries: vec![("seg_adjoint_kl".into(), 128, smem, "adj")], extra_opts: vec![], note: String::new() });
                }
            }
        }
    }
}

fn vs_smem_v(order: u32, w: usize, g: VsGeom, c: VsCfg, sch: usize) -> usize {
    let tab = ((g.tz + w - 1) * g.tx * g.ty * 4 + 15) & !15;
    let vt = if order >= 2 { c.nwc * c.nws * 1600 } else { 0 };
    sch * vsw(order, w, g, c.vxpad) * 4 + tab + c.nwc * 2 * (sch * 16 + 16) * 4 + vt
}

/// vs / vt sub-chunk staging: the default geometry's candidates (and the test geometries) × VSCH ∈ {8, 12, 16, 24, 32}
/// below smax
fn vs2_jobs(c: &Case, a: &Arch, jobs: &mut Vec<Job>) {
    if !vs_ok(c) {
        return;
    }
    let geoms = vec![vs_default_geom(c)];
    let fam = if c.order >= 2 { "vt2_fwd" } else { "vs2_fwd" };
    for g in geoms {
        // the one-pass candidates on the largest shared memory (sub-chunks only shrink it)
        for v in vs_candidates(c, g, 227 * 1024) {
            for sch in [12usize, 16, 24, 32].into_iter().filter(|&q| q < g.smax) {
                let smem = vs_smem_v(c.order, c.w, g, v, sch);
                if smem > a.smem_optin {
                    continue;
                }
                let src = format!(
                    "#define VSCH {sch}\n#define W {}\n#define BETA {:.17e}\n#define NSEG {}\n#define CORR {}\n#define NCB {}\n#define TX {}\n#define TY {}\n#define TZ {}\n#define SMAX {}\n#define NWC {}\n#define NWS {}\n#define VXPAD {}\n#define KS {}\n#define RDEG {}\n{}",
                    c.w, beta(c), c.nseg, c.order, c.cb, g.tx, g.ty, g.tz, g.smax, v.nwc, v.nws, v.vxpad, c.ks, c.rdeg, VS_CU
                );
                let name = format!("segnufft_vs_w{}_l{}_o{}_c{}_k{}r{}_t{}x{}x{}_s{}_g{}_{}_{}_q{sch}", c.w, c.nseg, c.order, c.cb, c.ks, c.rdeg, g.tx, g.ty, g.tz, g.smax, v.nwc, v.nws, v.vxpad);
                let cfg = format!("{}x{}x{} smax{} nwc{} nws{} vxpad{} sch{sch}", g.tx, g.ty, g.tz, g.smax, v.nwc, v.nws, v.vxpad);
                jobs.push(Job { arch: *a, case: c.name.into(), family: fam, cfg, module: name, src, entries: vec![("seg_forward_vs".into(), 32 * v.nwc * v.nws, smem, "fwd")], extra_opts: vec![], note: String::new() });
            }
        }
    }
}

// ---- FFT axis passes (hasty-fft codegen, plain loads / stores; the fused pad / crop hooks are not audited) ----

fn choose_shape(n: usize, radices: &[usize], contiguous: bool, tile_cnt: usize, real_bytes: usize, smem_limit: usize) -> (usize, usize) {
    let rmax = *radices.iter().max().unwrap();
    let tpl = (n / rmax).max(1);
    let passes = radices.len();
    let inplace = radices.iter().all(|r| rmax % r == 0) && tpl * rmax == n;
    let per_line = if passes > 1 { (if inplace { 1 } else { 2 }) * n * 2 * real_bytes } else { 0 };
    let mut lpb = if contiguous { (256 / tpl).max(1) } else { 16 };
    if !contiguous {
        lpb = lpb.min(tile_cnt.next_power_of_two().max(1));
    }
    while per_line > 0 && lpb > 1 && lpb * per_line > smem_limit {
        lpb /= 2;
    }
    while lpb * tpl > 1024 && lpb > 1 {
        lpb /= 2;
    }
    (lpb, tpl)
}

fn fft_jobs(a: &Arch, jobs: &mut Vec<Job>) {
    // (grid label, axis length, image length on that axis): grid 288 x 288 x 140 (image 225 x 225 x 112, flow_thorax),
    // grid 800 x 480 x 480 (image ≈ 640 x 384 x 384, torso)
    let axes = [("288x288x140 x/y", 288usize, 225usize), ("288x288x140 z", 140, 112), ("800x480x480 x", 800, 640), ("800x480x480 y/z", 480, 384)];
    for (label, n, nin) in axes {
        let Some(radices) = factorize(n) else { continue };
        for (dir, pruned) in [("fwd", false), ("fwd", true), ("inv", true)] {
            let (lpb, tpl) = choose_shape(n, &radices, false, 50, 4, a.smem_optin);
            let mut cfg = AxisCfg::simple(n, radices.clone(), false, lpb, tpl, "float", dir == "fwd", 1.0);
            if dir == "fwd" {
                cfg.n_in = nin;
            } else {
                cfg.n_out = nin;
            }
            if pruned {
                cfg.arange = true;
                if dir == "fwd" {
                    cfg.lrange_out = true;
                } else {
                    cfg.lrange_in = true;
                }
            }
            let smem = cfg.smem_bytes(4);
            if smem > a.smem_optin {
                continue;
            }
            let src = generate_axis_source(&cfg);
            let tag = format!("{label} n{n} {dir}{} lpb{lpb} tpl{tpl} radices{:?}", if pruned { " pruned" } else { "" }, radices);
            jobs.push(Job { arch: *a, case: "fft".into(), family: "fft_axis", cfg: tag.clone(), module: format!("fft_axis_n{n}_{dir}{}", if pruned { "_ar" } else { "" }), src, entries: vec![("fft_axis".into(), lpb * tpl, smem, "fft")], extra_opts: vec!["--use_fast_math"], note: String::new() });
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Row {
    arch: &'static str,
    case: String,
    family: &'static str,
    cfg: String,
    entry: String,
    role: &'static str,
    res: EntryRes,
    threads: usize,
    smem_dyn: usize,
    occ: Occ,
    flags: Vec<String>,
    note: String,
    error: Option<String>,
    secs: f64,
}

fn json_str(s: &str) -> String {
    let mut o = String::from("\"");
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

impl Row {
    fn json(&self) -> String {
        format!(
            "{{\"arch\":{},\"case\":{},\"family\":{},\"cfg\":{},\"entry\":{},\"role\":{},\"regs\":{},\"stack\":{},\"spill_st\":{},\"spill_ld\":{},\"smem_static\":{},\"barriers\":{},\"threads\":{},\"smem_dyn\":{},\"blocks_sm\":{},\"warps_sm\":{},\"limiter\":{},\"flags\":[{}],\"note\":{},\"error\":{},\"secs\":{:.2}}}",
            json_str(self.arch),
            json_str(&self.case),
            json_str(self.family),
            json_str(&self.cfg),
            json_str(&self.entry),
            json_str(self.role),
            self.res.regs,
            self.res.stack,
            self.res.spill_st,
            self.res.spill_ld,
            self.res.smem_static,
            self.res.barriers,
            self.threads,
            self.smem_dyn,
            self.occ.blocks,
            self.occ.warps,
            json_str(&self.occ.limiter),
            self.flags.iter().map(|f| json_str(f)).collect::<Vec<_>>().join(","),
            json_str(&self.note),
            self.error.as_deref().map_or("null".to_string(), json_str),
            self.secs
        )
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let has = |k: &str| args.iter().any(|a| a == k);
    if has("--help") || has("-h") {
        println!("kernaudit [--out DIR] [--arch sm_80,sm_89,sm_90,sm_100,sm_120] [--cases a,b] [--families tile,vs,ra,kl,fft,whatif,ra2,kl2,vs2] [--jobs 4] [--warp-target 8] [--cuobjdump] [--opt \"nvrtc options\"] [--show-src] [--list]");
        println!("cases:");
        for c in CASES {
            println!("  {:<18} {}", c.name, c.what);
        }
        return;
    }
    let out = PathBuf::from(get("--out").unwrap_or_else(|| format!("{}/../research/experiments/arch_audit/out", env!("CARGO_MANIFEST_DIR").trim_end_matches("/crates/hasty-bench"))));
    let arch_sel: Vec<String> = get("--arch").unwrap_or("sm_80,sm_89,sm_90,sm_100,sm_120".into()).split(',').map(String::from).collect();
    let case_sel: Option<Vec<String>> = get("--cases").map(|s| s.split(',').map(String::from).collect());
    let fam_sel: Vec<String> = get("--families").unwrap_or("tile,vs,ra,kl,fft,whatif".into()).split(',').map(String::from).collect();
    let njobs: usize = get("--jobs").and_then(|s| s.parse().ok()).unwrap_or(4);
    let warp_target: usize = get("--warp-target").and_then(|s| s.parse().ok()).unwrap_or(8);
    let use_cuobjdump = has("--cuobjdump");
    // --opt "a b": extra NVRTC options for every job (experiments, e.g. "--maxrregcount=168"); recorded in each row's note
    let extra_opt: Vec<&'static str> = get("--opt").map(|s| s.split_whitespace().map(|x| &*Box::leak(x.to_string().into_boxed_str())).collect()).unwrap_or_default();

    let archs: Vec<Arch> = ARCHS.iter().filter(|a| arch_sel.iter().any(|s| s == a.name)).copied().collect();
    let cases: Vec<Case> = CASES.iter().filter(|c| case_sel.as_ref().is_none_or(|s| s.iter().any(|x| x == c.name))).copied().collect();
    let mut jobs = Vec::new();
    for a in &archs {
        for c in &cases {
            if fam_sel.iter().any(|f| f == "tile") {
                tile_jobs(c, a, &mut jobs);
            }
            if fam_sel.iter().any(|f| f == "vs") {
                vs_jobs(c, a, &mut jobs);
            }
            if fam_sel.iter().any(|f| f == "ra") {
                ra_jobs(c, a, &mut jobs);
            }
            if fam_sel.iter().any(|f| f == "kl") {
                kl_jobs(c, a, &mut jobs);
            }
            if fam_sel.iter().any(|f| f == "whatif") {
                whatif_jobs(c, a, &mut jobs);
            }
            if fam_sel.iter().any(|f| f == "ra2") {
                ra2_jobs(c, a, &mut jobs);
            }
            if fam_sel.iter().any(|f| f == "kl2") {
                kl2_jobs(c, a, &mut jobs);
            }
            if fam_sel.iter().any(|f| f == "vs2") {
                vs2_jobs(c, a, &mut jobs);
            }
        }
        if fam_sel.iter().any(|f| f == "fft") {
            fft_jobs(a, &mut jobs);
        }
    }
    // --subst "from=>to": what-if source edit applied to every job's source (e.g. a launch-bounds minimum), recorded in the note
    let subst: Option<(String, String)> = get("--subst").map(|s| {
        let (a, b) = s.split_once("=>").expect("--subst from=>to");
        (a.to_string(), b.to_string())
    });
    for j in jobs.iter_mut() {
        if let Some((a, b)) = &subst {
            if j.src.contains(a.as_str()) {
                j.src = j.src.replace(a.as_str(), b);
                j.note = format!("{}{}subst: {a} => {b}", j.note, if j.note.is_empty() { "" } else { "; " });
            }
        }
        j.extra_opts.extend(extra_opt.iter().copied());
        if !extra_opt.is_empty() {
            j.note = format!("{}{}opts: {}", j.note, if j.note.is_empty() { "" } else { "; " }, extra_opt.join(" "));
        }
    }
    // --grep a,b: keep only jobs whose cfg contains every substring (quick checks of a few shapes)
    if let Some(gs) = get("--grep") {
        let pats: Vec<String> = gs.split(',').map(String::from).collect();
        jobs.retain(|j| pats.iter().all(|p| j.cfg.contains(p.as_str())));
    }
    if has("--list") {
        for j in &jobs {
            println!("{} {} {} {}", j.arch.name, j.case, j.family, j.cfg);
        }
        println!("{} jobs", jobs.len());
        return;
    }
    if has("--show-src") {
        if let Some(j) = jobs.first() {
            println!("{}", &j.src[..j.src.find("\n//").unwrap_or(j.src.len().min(2000))]);
        }
        return;
    }
    fs::create_dir_all(&out).expect("create --out");
    eprintln!("kernaudit: NVRTC {}, {} jobs on {} threads, out {}", nvrtc_version(), jobs.len(), njobs, out.display());
    let rows: Mutex<Vec<Row>> = Mutex::new(Vec::new());
    let done = std::sync::atomic::AtomicUsize::new(0);
    let tmpdir = std::env::temp_dir().join(format!("kernaudit-{}", std::process::id()));
    if use_cuobjdump {
        let _ = fs::create_dir_all(&tmpdir);
    }
    let pool = rayon::ThreadPoolBuilder::new().num_threads(njobs).build().unwrap();
    let total = jobs.len();
    pool.install(|| {
        use rayon::prelude::*;
        jobs.par_iter().for_each(|j| {
            let t0 = std::time::Instant::now();
            let r = nvrtc_compile(&j.module, &j.src, j.arch.name, &j.extra_opts);
            let secs = t0.elapsed().as_secs_f64();
            let mut out_rows = Vec::new();
            match r {
                Err(e) => {
                    for (entry, threads, smem, role) in &j.entries {
                        out_rows.push(Row { arch: j.arch.name, case: j.case.clone(), family: j.family, cfg: j.cfg.clone(), entry: entry.clone(), role, res: EntryRes::default(), threads: *threads, smem_dyn: *smem, occ: Occ::default(), flags: vec!["COMPILE_ERROR".into()], note: j.note.clone(), error: Some(e.chars().take(4000).collect()), secs });
                    }
                }
                Ok((cubin, log)) => {
                    let parsed = parse_ptxas(&log);
                    let objdump = if use_cuobjdump {
                        let f = tmpdir.join(format!("{}-{}.cubin", j.module, j.arch.name));
                        let _ = fs::write(&f, &cubin);
                        let o = std::process::Command::new("/usr/local/cuda/bin/cuobjdump").arg("--dump-resource-usage").arg(&f).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).into_owned());
                        let _ = fs::remove_file(&f);
                        o
                    } else {
                        None
                    };
                    for (entry, threads, smem, role) in &j.entries {
                        let res = parsed.get(entry).cloned().unwrap_or_default();
                        let total_smem = res.smem_static + smem;
                        let occ = occupancy(&j.arch, res.regs, *threads, total_smem);
                        let mut flags = Vec::new();
                        if res.spill_st > 0 || res.spill_ld > 0 {
                            flags.push(format!("SPILL st{} ld{}", res.spill_st, res.spill_ld));
                        }
                        if res.stack > 0 {
                            flags.push(format!("STACK {}", res.stack));
                        }
                        if total_smem > j.arch.smem_optin {
                            flags.push("SMEM>OPTIN".into());
                        }
                        if occ.blocks == 0 {
                            flags.push("NO_FIT".into());
                        } else if occ.warps < warp_target {
                            flags.push(format!("LOW_OCC {}w", occ.warps));
                        }
                        if !parsed.contains_key(entry) {
                            flags.push("ENTRY_NOT_IN_LOG".into());
                        }
                        let mut note = j.note.clone();
                        if let Some(o) = &objdump {
                            // cross-check: the cuobjdump line of the entry ("Function <name>:\n REG:.. STACK:.. SHARED:.. LOCAL:..")
                            if let Some(i) = o.find(&format!("Function {entry}:")) {
                                let l = o[i..].lines().nth(1).unwrap_or("").trim().to_string();
                                note = if note.is_empty() { format!("cuobjdump: {l}") } else { format!("{note}; cuobjdump: {l}") };
                            }
                        }
                        out_rows.push(Row { arch: j.arch.name, case: j.case.clone(), family: j.family, cfg: j.cfg.clone(), entry: entry.clone(), role, res, threads: *threads, smem_dyn: *smem, occ, flags, note, error: None, secs });
                    }
                }
            }
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if n % 25 == 0 || n == total {
                eprintln!("  {n}/{total}");
            }
            rows.lock().unwrap().extend(out_rows);
        });
    });
    let _ = fs::remove_dir_all(&tmpdir);
    let mut rows = rows.into_inner().unwrap();
    rows.sort_by(|a, b| (a.case.as_str(), a.family, a.cfg.as_str(), a.arch).cmp(&(b.case.as_str(), b.family, b.cfg.as_str(), b.arch)));
    let mut f = fs::File::create(out.join("results.jsonl")).unwrap();
    for r in &rows {
        writeln!(f, "{}", r.json()).unwrap();
    }
    write_summary(&out, &rows, &archs, &cases, warp_target);
    eprintln!("kernaudit: wrote {} rows to {}", rows.len(), out.display());
}

fn write_summary(out: &std::path::Path, rows: &[Row], archs: &[Arch], cases: &[Case], warp_target: usize) {
    let mut s = String::new();
    s.push_str(&format!("# kernaudit summary (NVRTC {})\n\n", nvrtc_version()));
    s.push_str("Theoretical occupancy from ptxas registers, static + dynamic shared memory (+1 KB reserved per block) and the per-SM limits of each architecture. No GPU was used.\n\n");
    s.push_str("Architectures:\n\n| arch | GPUs | threads / SM | warps / SM | blocks / SM | regs / SM | smem / SM | smem opt-in / block |\n|---|---|---|---|---|---|---|---|\n");
    for a in archs {
        s.push_str(&format!("| {} | {} | {} | {} | {} | {} | {} KB | {} KB |\n", a.name, a.gpus, a.max_threads_sm, a.max_warps_sm, a.max_blocks_sm, a.regs_sm, a.smem_sm / 1024, a.smem_optin / 1024));
    }
    s.push_str("\nCases:\n\n");
    for c in cases {
        s.push_str(&format!("- `{}`: {}\n", c.name, c.what));
    }
    // per (case, family, cfg): one line with regs / spills / warps per arch
    s.push_str("\n## Per configuration\n\nCell = registers / warps per SM (limiter); `S` = spills, `K` = stack frame, `X` = does not fit, `-` = not a candidate on that arch (shared memory).\n\n");
    let mut groups: BTreeMap<(String, &str), BTreeMap<String, Vec<&Row>>> = BTreeMap::new();
    for r in rows {
        groups.entry((r.case.clone(), r.family)).or_default().entry(r.cfg.clone()).or_default().push(r);
    }
    for ((case, fam), cfgs) in &groups {
        s.push_str(&format!("### {case} / {fam}\n\n| cfg | threads | smem dyn (B) |"));
        for a in archs {
            s.push_str(&format!(" {} |", a.name));
        }
        s.push_str("\n|---|---|---|");
        for _ in archs {
            s.push_str("---|");
        }
        s.push('\n');
        for (cfg, rs) in cfgs {
            let r0 = rs[0];
            s.push_str(&format!("| {} | {} | {} |", cfg, r0.threads, r0.smem_dyn));
            for a in archs {
                match rs.iter().find(|r| r.arch == a.name) {
                    None => s.push_str(" - |"),
                    Some(r) if r.error.is_some() => s.push_str(" ERR |"),
                    Some(r) => {
                        let mut m = String::new();
                        if r.res.spill_st > 0 || r.res.spill_ld > 0 {
                            m.push_str(&format!(" S{}", r.res.spill_st));
                        }
                        if r.res.stack > 0 {
                            m.push_str(&format!(" K{}", r.res.stack));
                        }
                        if r.occ.blocks == 0 {
                            m.push_str(" X");
                        }
                        s.push_str(&format!(" {}/{} ({}){} |", r.res.regs, r.occ.warps, r.occ.limiter, m));
                    }
                }
            }
            s.push('\n');
        }
        s.push('\n');
    }
    s.push_str(&format!("## Flags (spills, stack, no fit, below {warp_target} warps / SM, compile errors)\n\n| arch | case | family | cfg | regs | flags |\n|---|---|---|---|---|---|\n"));
    for r in rows.iter().filter(|r| !r.flags.is_empty()) {
        s.push_str(&format!("| {} | {} | {} | {} | {} | {} |\n", r.arch, r.case, r.family, r.cfg, r.res.regs, r.flags.join(", ")));
    }
    let errs: Vec<&Row> = rows.iter().filter(|r| r.error.is_some()).collect();
    if !errs.is_empty() {
        s.push_str("\n## Compile errors (first lines)\n\n");
        for r in errs {
            s.push_str(&format!("- {} {} {} {}: `{}`\n", r.arch, r.case, r.family, r.cfg, r.error.as_ref().unwrap().lines().take(3).collect::<Vec<_>>().join(" | ")));
        }
    }
    fs::write(out.join("summary.md"), s).unwrap();
}
