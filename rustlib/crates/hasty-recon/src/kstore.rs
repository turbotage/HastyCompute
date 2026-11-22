//! Per-set k-space store: the whitened, coil-compressed data of every sample set as coil-major rows (`nvc × m_t`),
//! held in one of several representations so the same reconstruction runs on a workstation (little VRAM, RAM shared
//! with simulations) and on a cluster node (many GPUs, large RAM / VRAM):
//! - `ram`: dense C32 in RAM, zero decode cost (`KStore::dense` borrows a set without copying);
//! - `lz4`, `zstd`, `zstdN`: compressed in RAM, blocks of `block` samples per coil with the float components
//!   byte-shuffled (the `hasty_value` block codec), decoded in parallel on read;
//! - `disk:` / `disk=PATH:` prefix: the same blocks (or raw ones) in a file read by positional reads, with a JSON
//!   index next to it (`PATH.json`) so later runs or other processes can reopen it ([`KStore::open`]);
//! - `+qREL` suffix: noise-relative quantisation before compression, values stored as integer multiples of
//!   REL·σ (σ = noise std per real component of the whitened data). Lossy by a known amount: the rounding adds
//!   noise of variance REL²/12 · σ² (REL 0.25 → +0.52 % noise variance, +0.26 % noise std), and Gaussian noise in
//!   steps of σ/4 needs ≈ 4–6 bits per component instead of 32.
//! - `bfp` / `bfpN` / `bfp+qREL` (RAM only): block floating point (`hasty_bfp`, 16 complex values per block along
//!   the samples of one coil, each coil row its own block sequence). `bfp`: the width per set from the noise rule
//!   of research/bfp_storage.md §3.6, the smallest b whose step in the set's peak block is ≤ REL · σ (REL 0.25
//!   default: b ≥ log2(peak / σ) + 4); `bfpN`: N bits. No entropy coding (zstd + q packs the host copy 3–4 × tighter),
//!   but no host decode either: `kstage::Stager` uploads the packed words as they are and decodes on the device,
//!   so the PCIe bytes and the staging CPU time both fall (4.06 B per complex at bfp16 instead of 8).
//!
//! The device keeps whatever fits (resident `CudaSlice`s, as before); the rest is staged per set from a store by
//! `kstage::Stager`. Reads take `&self` and are lock-free (positional file reads, `OnceLock` sets): several GPU
//! workers can stage from one store concurrently, and a coil subset reads only its coils' blocks (stochastic coil
//! sampling, coil-parallel multi-GPU splits).

use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use hasty_cuda::C32;
use hasty_value::codec::{block_raw_len, compress_block, decompress_block};
use hasty_value::{Codec, Options};
use rayon::prelude::*;
use serde_json::json;

use crate::Error;

/// Default samples per coil and block (256 KiB of C32): large enough for the codecs, small enough for parallel
/// decode of one set and cheap coil subsets.
pub const DEFAULT_BLOCK: usize = 32768;

/// Width rule of a BFP k-space representation
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BfpRule {
    /// the smallest b with the peak block's step ≤ q σ (σ: noise std per real component of the stored data)
    Sigma { q: f64 },
    /// a fixed width
    Bits(u32),
}

impl BfpRule {
    /// the width for a set whose largest |component| is `peak`, at noise std `sigma` per component
    pub fn bits(&self, peak: f32, sigma: Option<f64>) -> Result<hasty_bfp::Bfp, Error> {
        match *self {
            BfpRule::Bits(b) => Ok(hasty_bfp::Bfp::new(b)),
            BfpRule::Sigma { q } => {
                let s = sigma.filter(|s| *s > 0.0).ok_or_else(|| Error::Format("k-space BFP: the σ rule needs the noise level (noise-free data: give a width, bfpN)".into()))?;
                // an all-zero or tiny set: 8 bits already meet the rule
                let b = hasty_bfp::Bfp::bits_for_step(peak, q * s).ok_or_else(|| Error::Format(format!("k-space BFP: peak / σ = {:.3e} needs more than 24 bits for steps ≤ {q} σ", peak as f64 / s)))?;
                Ok(hasty_bfp::Bfp::new(b))
            }
        }
    }
    pub fn describe(&self) -> String {
        match self {
            BfpRule::Sigma { q } => format!("BFP, width per set from the noise rule (peak block step ≤ {q} σ)"),
            BfpRule::Bits(b) => format!("BFP{b}"),
        }
    }
}

/// Representation of a [`KStore`] (see the module docs for the string form).
#[derive(Clone, Debug, PartialEq)]
pub struct StoreSpec {
    /// None: RAM; Some(path): a file (index at `path.json`)
    pub file: Option<PathBuf>,
    /// block codec (Codec::None in RAM without quantisation = dense)
    pub codec: Codec,
    /// quantisation step relative to the noise std per real component (None: lossless)
    pub quant_rel: Option<f64>,
    /// samples per coil and block
    pub block: usize,
    /// block floating point instead of the codec (RAM only); `quant_rel` is then None
    pub bfp: Option<BfpRule>,
}

impl StoreSpec {
    pub fn dense() -> Self {
        Self { file: None, codec: Codec::None, quant_rel: None, block: DEFAULT_BLOCK, bfp: None }
    }

    /// `[disk[=PATH]:]CODEC[+qREL]` with CODEC ram | raw | lz4 | zstd | zstdN (N = level). `disk` without a path
    /// uses `default_file`.
    pub fn parse(s: &str, default_file: &Path) -> Result<Self, Error> {
        let bad = |m: &str| Error::Format(format!("k-space store '{s}': {m} (expected [disk[=PATH]:]ram|raw|lz4|zstd|zstdN[+qREL])"));
        let (file, rest) = match s.split_once(':') {
            Some((d, r)) if d == "disk" => (Some(default_file.to_path_buf()), r),
            Some((d, r)) if d.starts_with("disk=") => (Some(PathBuf::from(&d[5..])), r),
            _ if s == "disk" => (Some(default_file.to_path_buf()), "raw"),
            _ => (None, s),
        };
        let mut parts = rest.split('+');
        let head = parts.next().unwrap_or("");
        if let Some(w) = head.strip_prefix("bfp") {
            if file.is_some() {
                return Err(bad("bfp is a RAM representation (the disk tiers keep zstd + q)"));
            }
            let mut rule = if w.is_empty() {
                BfpRule::Sigma { q: 0.25 }
            } else {
                let b: u32 = w.parse().map_err(|_| bad("bfp width"))?;
                if !(8..=24).contains(&b) {
                    return Err(bad("bfp width must be in 8..=24"));
                }
                BfpRule::Bits(b)
            };
            for p in parts {
                match (p.strip_prefix('q'), &mut rule) {
                    (Some(v), BfpRule::Sigma { q }) => {
                        let r: f64 = if v.is_empty() { 0.25 } else { v.parse().map_err(|_| bad("bfp step bound"))? };
                        if !(r > 0.0 && r <= 4.0) {
                            return Err(bad("bfp step bound must be in (0, 4]"));
                        }
                        *q = r;
                    }
                    _ => return Err(bad("suffix (bfp takes +qREL only without a fixed width)")),
                }
            }
            return Ok(Self { file: None, codec: Codec::None, quant_rel: None, block: DEFAULT_BLOCK, bfp: Some(rule) });
        }
        let codec = match head {
            "ram" | "raw" | "none" | "dense" => Codec::None,
            "lz4" => Codec::Lz4,
            "zstd" => Codec::Zstd(3),
            z if z.starts_with("zstd") => Codec::Zstd(z[4..].parse().map_err(|_| bad("zstd level"))?),
            _ => return Err(bad("codec")),
        };
        let mut quant_rel = None;
        for p in parts {
            match p.strip_prefix('q') {
                Some(v) => {
                    let r: f64 = if v.is_empty() { 0.25 } else { v.parse().map_err(|_| bad("quantisation step"))? };
                    if !(r > 0.0 && r <= 4.0) {
                        return Err(bad("quantisation step must be in (0, 4]"));
                    }
                    quant_rel = Some(r);
                }
                None => return Err(bad("suffix")),
            }
        }
        Ok(Self { file, codec, quant_rel, block: DEFAULT_BLOCK, bfp: None })
    }

    /// dense in RAM (no blocks)
    pub fn is_dense(&self) -> bool {
        self.file.is_none() && self.codec == Codec::None && self.quant_rel.is_none() && self.bfp.is_none()
    }

    pub fn describe(&self) -> String {
        if let Some(r) = &self.bfp {
            return format!("{} in RAM", r.describe());
        }
        let c = match self.codec {
            Codec::None => "raw".to_string(),
            Codec::Lz4 => "lz4".into(),
            Codec::Zstd(l) => format!("zstd{l}"),
        };
        let q = self.quant_rel.map_or(String::new(), |r| format!(" + quantised in steps of {r}σ (lossy: +{:.2} % noise variance)", 100.0 * r * r / 12.0));
        match &self.file {
            None if self.is_dense() => "dense in RAM".into(),
            None => format!("{c} blocks in RAM{q}"),
            Some(p) => format!("{c} blocks in {}{q}", p.display()),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StoreStats {
    /// C32 bytes of all sets
    pub raw_bytes: usize,
    /// bytes as stored (blocks with headers, or dense)
    pub stored_bytes: usize,
    /// bytes held in RAM (dense data or compressed arenas; a file store holds only its index)
    pub ram_bytes: usize,
    /// quantised values clipped to the i32 range
    pub saturated: usize,
}

impl StoreStats {
    pub fn ratio(&self) -> f64 {
        self.raw_bytes as f64 / self.stored_bytes.max(1) as f64
    }
}

enum SetBody {
    Dense(Vec<C32>),
    /// RAM blocks: arena + (offset, length) per block, index c · nb + b
    Ram { arena: Vec<u8>, idx: Vec<(u64, u32)> },
    /// file blocks: (absolute offset, length)
    File { idx: Vec<(u64, u32)> },
    /// BFP: coil row c (2 m reals) at mant[c · rw ..] / ex[c · rb ..] (`hasty_bfp::xfer` row layout)
    Bfp { fmt: hasty_bfp::Bfp, mant: Vec<u32>, ex: Vec<u8> },
}

/// See the module docs.
pub struct KStore {
    spec: StoreSpec,
    nvc: usize,
    lens: Vec<usize>,
    /// absolute quantisation step per real component (0: lossless)
    step: f64,
    /// noise std per real component of the stored data (the pipeline's whitened-noise level), if known
    sigma: Option<f64>,
    sets: Vec<OnceLock<SetBody>>,
    file: Option<File>,
    next: Mutex<u64>,
    stored: AtomicUsize,
    saturated: AtomicUsize,
    tag: String,
    /// sample selection applied on read ([`KStore::with_selection`]); None: the stored sets
    view: Option<View>,
}

/// kept sample indices per set (ascending) and their counts
struct View {
    sel: Vec<Vec<u32>>,
    lens: Vec<usize>,
}

fn as_bytes(x: &[C32]) -> &[u8] {
    // SAFETY: Complex<f32> is repr(C) of two f32, no padding
    unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, std::mem::size_of_val(x)) }
}

fn as_bytes_mut(x: &mut [C32]) -> &mut [u8] {
    // SAFETY: as above; every bit pattern is a valid f32
    unsafe { std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut u8, std::mem::size_of_val(x)) }
}

#[inline]
fn zigzag(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

#[inline]
fn unzigzag(u: u32) -> i32 {
    ((u >> 1) as i32) ^ -((u & 1) as i32)
}

impl KStore {
    /// An empty store for sets of `lens[t]` samples per coil and `nvc` coils; fill with [`put`](Self::put), then
    /// [`finish`](Self::finish). `sigma` = noise std per real component of the stored data (needed for `+q`).
    /// `tag` identifies the content (e.g. sim, whitening, compression, noise) for reuse checks of file stores.
    pub fn create(spec: StoreSpec, nvc: usize, lens: Vec<usize>, sigma: Option<f64>, tag: &str) -> Result<Self, Error> {
        let step = match spec.quant_rel {
            None => 0.0,
            Some(r) => match sigma {
                Some(s) if s > 0.0 => r * s,
                _ => return Err(Error::Format("k-space store: quantisation needs the noise level (noise-free data: use a lossless store)".into())),
            },
        };
        let file = match &spec.file {
            Some(p) => {
                if let Some(d) = p.parent() {
                    std::fs::create_dir_all(d)?;
                }
                let _ = std::fs::remove_file(p.with_extension(ext_json(p)));
                Some(OpenOptions::new().create(true).truncate(true).read(true).write(true).open(p)?)
            }
            None => None,
        };
        if matches!(spec.bfp, Some(BfpRule::Sigma { .. })) && !sigma.is_some_and(|s| s > 0.0) {
            return Err(Error::Format("k-space store: bfp (σ rule) needs the noise level (noise-free data: bfpN)".into()));
        }
        let n = lens.len();
        Ok(Self {
            spec,
            nvc,
            lens,
            step,
            sigma,
            sets: (0..n).map(|_| OnceLock::new()).collect(),
            file,
            next: Mutex::new(0),
            stored: AtomicUsize::new(0),
            saturated: AtomicUsize::new(0),
            tag: tag.to_string(),
            view: None,
        })
    }

    /// Reopen a finished file store (`path` and its `path.json` index).
    pub fn open(path: &Path) -> Result<Self, Error> {
        let j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path.with_extension(ext_json(path)))?).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
        let fmt = |m: &str| Error::Format(format!("{}: index {m}", path.display()));
        if j["format"] != "hasty-kstore-1" {
            return Err(fmt("format"));
        }
        let u = |v: &serde_json::Value| v.as_u64().ok_or_else(|| fmt("number")).map(|x| x as usize);
        let nvc = u(&j["nvc"])?;
        let lens: Vec<usize> = j["lens"].as_array().ok_or_else(|| fmt("lens"))?.iter().map(u).collect::<Result<_, _>>()?;
        let codec = match j["codec"].as_str().unwrap_or("") {
            "raw" => Codec::None,
            "lz4" => Codec::Lz4,
            z if z.starts_with("zstd") => Codec::Zstd(z[4..].parse().unwrap_or(3)),
            _ => return Err(fmt("codec")),
        };
        let spec = StoreSpec { file: Some(path.to_path_buf()), codec, quant_rel: j["quant_rel"].as_f64(), block: u(&j["block"])?, bfp: None };
        let sets: Vec<OnceLock<SetBody>> = j["index"].as_array().ok_or_else(|| fmt("index"))?.iter().map(|s| {
            let a = s.as_array().unwrap();
            let idx = a.chunks(2).map(|p| (p[0].as_u64().unwrap(), p[1].as_u64().unwrap() as u32)).collect();
            let l = OnceLock::new();
            let _ = l.set(SetBody::File { idx });
            l
        }).collect();
        if sets.len() != lens.len() {
            return Err(fmt("set count"));
        }
        Ok(Self {
            spec,
            nvc,
            lens,
            step: j["step"].as_f64().unwrap_or(0.0),
            sigma: j["sigma"].as_f64(),
            sets,
            file: Some(File::open(path)?),
            next: Mutex::new(u(&j["file_bytes"])? as u64),
            stored: AtomicUsize::new(u(&j["file_bytes"])?),
            saturated: AtomicUsize::new(u(&j["saturated"]).unwrap_or(0)),
            tag: j["tag"].as_str().unwrap_or("").to_string(),
            view: None,
        })
    }

    pub fn spec(&self) -> &StoreSpec {
        &self.spec
    }
    pub fn tag(&self) -> &str {
        &self.tag
    }
    pub fn n_sets(&self) -> usize {
        self.lens.len()
    }
    pub fn nvc(&self) -> usize {
        self.nvc
    }
    /// samples per coil of set t (as read: the selected samples of a [`with_selection`](Self::with_selection) view)
    pub fn len(&self, t: usize) -> usize {
        self.lens()[t]
    }
    pub fn lens(&self) -> &[usize] {
        match &self.view {
            Some(v) => &v.lens,
            None => &self.lens,
        }
    }
    /// samples per coil of set t as stored (before a selection)
    pub fn stored_len(&self, t: usize) -> usize {
        self.lens[t]
    }
    /// C32 elements of the largest set (all coils, as read)
    pub fn max_set(&self) -> usize {
        self.lens().iter().max().copied().unwrap_or(0) * self.nvc
    }

    /// A view of the store restricted to the samples `sel[t]` of every set (ascending indices into the stored set,
    /// e.g. a k-space crop to a coarse grid's band, `pyramid::KCrop`): [`len`](Self::len), [`lens`](Self::lens),
    /// [`max_set`](Self::max_set), [`read`](Self::read) and [`read_vec`](Self::read_vec) see the selected sets, so
    /// `Stager` and the solvers work unchanged. The stored data are untouched (a file store is not rewritten): blocks
    /// are decoded as before and filtered, blocks without a kept sample are skipped.
    pub fn with_selection(mut self, sel: Vec<Vec<u32>>) -> Result<Self, Error> {
        if sel.len() != self.lens.len() {
            return Err(Error::Format(format!("k-space store selection: {} sets for a store of {}", sel.len(), self.lens.len())));
        }
        for (t, s) in sel.iter().enumerate() {
            if !s.windows(2).all(|w| w[0] < w[1]) || s.last().is_some_and(|&j| j as usize >= self.lens[t]) {
                return Err(Error::Format(format!("k-space store selection: set {t} not ascending within its {} samples", self.lens[t])));
            }
        }
        let lens = sel.iter().map(|s| s.len()).collect();
        self.view = Some(View { sel, lens });
        Ok(self)
    }

    /// a selection is applied on read
    pub fn is_view(&self) -> bool {
        self.view.is_some()
    }
    /// absolute quantisation step (0: lossless)
    pub fn step(&self) -> f64 {
        self.step
    }
    /// noise std per real component the store was created with
    pub fn sigma(&self) -> Option<f64> {
        self.sigma
    }
    /// BFP width of set t (None: not a BFP set, or not stored)
    pub fn bfp_fmt(&self, t: usize) -> Option<hasty_bfp::Bfp> {
        match self.sets[t].get() {
            Some(SetBody::Bfp { fmt, .. }) => Some(*fmt),
            _ => None,
        }
    }

    /// Copy the packed coil rows `coils` (None: all) of BFP set t into `mant` / `ex` (`hasty_bfp::xfer` row layout,
    /// `coils.len()` rows of 2 · len(t) reals); returns the width. Not for selection views.
    pub fn read_packed(&self, t: usize, coils: Option<&[usize]>, mant: &mut [u32], ex: &mut [u8]) -> Result<hasty_bfp::Bfp, Error> {
        if self.view.is_some() {
            return Err(Error::Format("k-space store: packed reads of a selection view".into()));
        }
        let Some(SetBody::Bfp { fmt, mant: sm, ex: se }) = self.sets[t].get() else {
            return Err(Error::Format(format!("k-space store: set {t} is not stored as BFP")));
        };
        let m = self.lens[t];
        let (rw, rb) = (fmt.mant_words(2 * m), hasty_bfp::Bfp::blocks(2 * m));
        let all: Vec<usize>;
        let coils = match coils {
            Some(c) => c,
            None => {
                all = (0..self.nvc).collect();
                &all
            }
        };
        if mant.len() < coils.len() * rw || ex.len() < coils.len() * rb {
            return Err(Error::Format(format!("k-space store packed read: set {t}: buffers too small")));
        }
        for (j, &c) in coils.iter().enumerate() {
            mant[j * rw..(j + 1) * rw].copy_from_slice(&sm[c * rw..(c + 1) * rw]);
            ex[j * rb..(j + 1) * rb].copy_from_slice(&se[c * rb..(c + 1) * rb]);
        }
        Ok(*fmt)
    }

    /// BFP sets: (width histogram (bits, sets), worst block step / σ, predicted rms error / σ = √(mean step² / 12) / σ
    /// over all blocks), None without BFP sets
    pub fn bfp_summary(&self) -> Option<(Vec<(u32, usize)>, f64, f64)> {
        let mut hist: std::collections::BTreeMap<u32, usize> = Default::default();
        let (mut worst, mut s2, mut nblk) = (0.0f64, 0.0f64, 0usize);
        for s in self.sets.iter().filter_map(|s| s.get()) {
            if let SetBody::Bfp { fmt, ex, mant } = s {
                *hist.entry(fmt.bits()).or_default() += 1;
                let nb = mant.len() / fmt.bits() as usize;
                for &e in &ex[..nb] {
                    if e != hasty_bfp::ENAN {
                        let st = fmt.step(e) as f64;
                        worst = worst.max(st);
                        s2 += st * st;
                    }
                }
                nblk += nb;
            }
        }
        if hist.is_empty() {
            return None;
        }
        let sg = self.sigma.unwrap_or(f64::NAN);
        Some((hist.into_iter().collect(), worst / sg, (s2 / nblk.max(1) as f64 / 12.0).sqrt() / sg))
    }
    pub fn is_filled(&self, t: usize) -> bool {
        self.sets[t].get().is_some()
    }

    fn nb(&self, t: usize) -> usize {
        self.lens[t].div_ceil(self.spec.block.max(1))
    }

    fn options(&self) -> Options {
        Options { codec: self.spec.codec, shuffle: true, ..Options::default() }
    }

    fn encode(&self, x: &[C32]) -> Vec<u8> {
        if self.step > 0.0 {
            let inv = 1.0 / self.step;
            let mut sat = 0usize;
            let q: Vec<u32> = x
                .iter()
                .flat_map(|z| [z.re, z.im])
                .map(|v| {
                    let r = (v as f64 * inv).round();
                    if r.abs() > i32::MAX as f64 {
                        sat += 1;
                    }
                    zigzag(r.clamp(i32::MIN as f64, i32::MAX as f64) as i32)
                })
                .collect();
            if sat > 0 {
                self.saturated.fetch_add(sat, Ordering::Relaxed);
            }
            // SAFETY: u32 slice viewed as bytes
            let b = unsafe { std::slice::from_raw_parts(q.as_ptr() as *const u8, q.len() * 4) };
            compress_block(b, 4, false, &self.options())
        } else {
            compress_block(as_bytes(x), 4, false, &self.options())
        }
    }

    fn decode(&self, blk: &[u8], out: &mut [C32]) -> Result<(), Error> {
        if self.step > 0.0 {
            let mut q = vec![0u32; 2 * out.len()];
            // SAFETY: u32 slice viewed as bytes
            let b = unsafe { std::slice::from_raw_parts_mut(q.as_mut_ptr() as *mut u8, q.len() * 4) };
            decompress_block(blk, b)?;
            let s = self.step as f32;
            for (o, p) in out.iter_mut().zip(q.chunks_exact(2)) {
                *o = C32::new(unzigzag(p[0]) as f32 * s, unzigzag(p[1]) as f32 * s);
            }
            Ok(())
        } else {
            Ok(decompress_block(blk, as_bytes_mut(out))?)
        }
    }

    /// Store set t from its coil-major data (`nvc × len(t)`). Thread-safe; each set once.
    pub fn put(&self, t: usize, y: &[C32]) -> Result<(), Error> {
        self.put_owned(t, y.to_vec())
    }

    /// [`put`](Self::put) taking ownership (a dense store keeps the vector without copying).
    pub fn put_owned(&self, t: usize, y: Vec<C32>) -> Result<(), Error> {
        let m = self.lens[t];
        if y.len() != self.nvc * m {
            return Err(Error::Format(format!("k-space store: set {t} has {} values, expected {} × {m}", y.len(), self.nvc)));
        }
        let body = if let Some(rule) = &self.spec.bfp {
            let peak = y.par_iter().map(|z| z.re.abs().max(z.im.abs())).reduce(|| 0.0f32, f32::max);
            let fmt = rule.bits(peak, self.sigma)?;
            let (rw, rb) = (fmt.mant_words(2 * m), hasty_bfp::Bfp::blocks(2 * m));
            let mut mant = vec![0u32; self.nvc * rw];
            let mut ex = vec![0u8; (self.nvc * rb).div_ceil(4) * 4];
            if m > 0 {
                mant.par_chunks_mut(rw).zip(ex[..self.nvc * rb].par_chunks_mut(rb)).enumerate().for_each(|(c, (mw, eb))| {
                    fmt.encode_serial(hasty_bfp::as_reals(&y[c * m..(c + 1) * m]), mw, eb);
                });
            }
            self.stored.fetch_add(mant.len() * 4 + ex.len(), Ordering::Relaxed);
            SetBody::Bfp { fmt, mant, ex }
        } else if self.spec.is_dense() {
            self.stored.fetch_add(y.len() * 8, Ordering::Relaxed);
            SetBody::Dense(y)
        } else {
            let (bs, nb) = (self.spec.block, self.nb(t));
            let blocks: Vec<Vec<u8>> = (0..self.nvc * nb)
                .into_par_iter()
                .map(|i| {
                    let (c, b) = (i / nb, i % nb);
                    let lo = b * bs;
                    self.encode(&y[c * m + lo..c * m + (lo + bs).min(m)])
                })
                .collect();
            drop(y);
            let total: usize = blocks.iter().map(|b| b.len()).sum();
            self.stored.fetch_add(total, Ordering::Relaxed);
            let mut idx = Vec::with_capacity(blocks.len());
            match &self.file {
                None => {
                    let mut arena = Vec::with_capacity(total);
                    for b in &blocks {
                        idx.push((arena.len() as u64, b.len() as u32));
                        arena.extend_from_slice(b);
                    }
                    SetBody::Ram { arena, idx }
                }
                Some(f) => {
                    let base = {
                        let mut n = self.next.lock().unwrap();
                        let b = *n;
                        *n += total as u64;
                        b
                    };
                    let mut buf = Vec::with_capacity(total);
                    for b in &blocks {
                        idx.push((base + buf.len() as u64, b.len() as u32));
                        buf.extend_from_slice(b);
                    }
                    f.write_all_at(&buf, base)?;
                    SetBody::File { idx }
                }
            }
        };
        self.sets[t].set(body).map_err(|_| Error::Format(format!("k-space store: set {t} stored twice")))
    }

    /// borrow set t of a dense store (coil-major), None otherwise
    pub fn dense(&self, t: usize) -> Option<&[C32]> {
        match self.sets[t].get() {
            Some(SetBody::Dense(v)) => Some(v),
            _ => None,
        }
    }

    /// Read coil rows `coils` (None: all, in order) of set t into `out` (coil-major, `coils.len() × len(t)`).
    /// Blocks are decoded in parallel.
    pub fn read(&self, t: usize, coils: Option<&[usize]>, out: &mut [C32]) -> Result<(), Error> {
        let m = self.lens[t];
        let all: Vec<usize>;
        let coils = match coils {
            Some(c) => c,
            None => {
                all = (0..self.nvc).collect();
                &all
            }
        };
        if let Some(v) = &self.view {
            return self.read_selected(t, coils, &v.sel[t], out);
        }
        if out.len() != coils.len() * m {
            return Err(Error::Format(format!("k-space store read: set {t}: buffer {} for {} coils × {m}", out.len(), coils.len())));
        }
        if m == 0 {
            return Ok(());
        }
        let body = self.sets[t].get().ok_or_else(|| Error::Format(format!("k-space store: set {t} not stored")))?;
        let (bs, nb) = (self.spec.block, self.nb(t));
        match body {
            SetBody::Dense(v) => {
                out.par_chunks_mut(m).zip(coils.par_iter()).for_each(|(row, &c)| row.copy_from_slice(&v[c * m..(c + 1) * m]));
                Ok(())
            }
            SetBody::Ram { arena, idx } => out.par_chunks_mut(m).zip(coils.par_iter()).try_for_each(|(row, &c)| {
                row.par_chunks_mut(bs).enumerate().try_for_each(|(b, blk)| {
                    let (o, l) = idx[c * nb + b];
                    self.decode(&arena[o as usize..o as usize + l as usize], blk)
                })
            }),
            SetBody::Bfp { fmt, mant, ex } => {
                let (rw, rb) = (fmt.mant_words(2 * m), hasty_bfp::Bfp::blocks(2 * m));
                out.par_chunks_mut(m).zip(coils.par_iter()).for_each(|(row, &c)| fmt.decode_serial(&mant[c * rw..(c + 1) * rw], &ex[c * rb..(c + 1) * rb], hasty_bfp::as_reals_mut(row)));
                Ok(())
            }
            SetBody::File { idx } => {
                let f = self.file.as_ref().unwrap();
                out.par_chunks_mut(m).zip(coils.par_iter()).try_for_each(|(row, &c)| {
                    row.par_chunks_mut(bs).enumerate().try_for_each(|(b, blk)| {
                        let (o, l) = idx[c * nb + b];
                        let mut buf = vec![0u8; l as usize];
                        f.read_exact_at(&mut buf, o)?;
                        debug_assert_eq!(block_raw_len(&buf).ok(), Some(blk.len() * 8));
                        self.decode(&buf, blk)
                    })
                })
            }
        }
    }

    /// [`read`](Self::read) of the samples `sel` (ascending) of set t: per coil row the blocks holding kept samples,
    /// decoded in parallel and gathered
    fn read_selected(&self, t: usize, coils: &[usize], sel: &[u32], out: &mut [C32]) -> Result<(), Error> {
        let (m, ms) = (self.lens[t], sel.len());
        if out.len() != coils.len() * ms {
            return Err(Error::Format(format!("k-space store read: set {t}: buffer {} for {} coils × {ms} selected", out.len(), coils.len())));
        }
        if ms == 0 {
            return Ok(());
        }
        let body = self.sets[t].get().ok_or_else(|| Error::Format(format!("k-space store: set {t} not stored")))?;
        if let SetBody::Bfp { fmt, mant, ex } = body {
            let (rw, rb) = (fmt.mant_words(2 * m), hasty_bfp::Bfp::blocks(2 * m));
            out.par_chunks_mut(ms).zip(coils.par_iter()).for_each(|(row, &c)| {
                let mut full = vec![C32::new(0.0, 0.0); m];
                fmt.decode_serial(&mant[c * rw..(c + 1) * rw], &ex[c * rb..(c + 1) * rb], hasty_bfp::as_reals_mut(&mut full));
                for (o, &j) in row.iter_mut().zip(sel) {
                    *o = full[j as usize];
                }
            });
            return Ok(());
        }
        if let SetBody::Dense(v) = body {
            out.par_chunks_mut(ms).zip(coils.par_iter()).for_each(|(row, &c)| {
                for (o, &j) in row.iter_mut().zip(sel) {
                    *o = v[c * m + j as usize];
                }
            });
            return Ok(());
        }
        let (bs, nb) = (self.spec.block, self.nb(t));
        // kept samples of block b: sel[bounds[b]..bounds[b + 1]]
        let bounds: Vec<usize> = (0..=nb).map(|b| sel.partition_point(|&j| (j as usize) < b * bs)).collect();
        out.par_chunks_mut(ms).zip(coils.par_iter()).try_for_each(|(row, &c)| {
            let mut parts = Vec::with_capacity(nb);
            let mut rest = row;
            for b in 0..nb {
                let (p, r) = rest.split_at_mut(bounds[b + 1] - bounds[b]);
                parts.push((b, p));
                rest = r;
            }
            parts.into_par_iter().filter(|(_, p)| !p.is_empty()).try_for_each(|(b, p)| -> Result<(), Error> {
                let lo = b * bs;
                let mut blk = vec![C32::new(0.0, 0.0); (lo + bs).min(m) - lo];
                match body {
                    SetBody::Ram { arena, idx } => {
                        let (o, l) = idx[c * nb + b];
                        self.decode(&arena[o as usize..o as usize + l as usize], &mut blk)?;
                    }
                    SetBody::File { idx } => {
                        let (o, l) = idx[c * nb + b];
                        let mut buf = vec![0u8; l as usize];
                        self.file.as_ref().unwrap().read_exact_at(&mut buf, o)?;
                        self.decode(&buf, &mut blk)?;
                    }
                    SetBody::Dense(_) | SetBody::Bfp { .. } => unreachable!(),
                }
                for (x, &j) in p.iter_mut().zip(&sel[bounds[b]..bounds[b + 1]]) {
                    *x = blk[j as usize - lo];
                }
                Ok(())
            })
        })
    }

    /// set t, all coils, as a new vector
    pub fn read_vec(&self, t: usize) -> Result<Vec<C32>, Error> {
        let mut v = vec![C32::new(0.0, 0.0); self.nvc * self.len(t)];
        self.read(t, None, &mut v)?;
        Ok(v)
    }

    pub fn stats(&self) -> StoreStats {
        let raw = self.lens.iter().sum::<usize>() * self.nvc * 8;
        let ram = self.sets.iter().filter_map(|s| s.get()).map(|b| match b {
            SetBody::Dense(v) => v.len() * 8,
            SetBody::Ram { arena, idx } => arena.len() + idx.len() * 12,
            SetBody::File { idx } => idx.len() * 12,
            SetBody::Bfp { mant, ex, .. } => mant.len() * 4 + ex.len(),
        }).sum();
        StoreStats { raw_bytes: raw, stored_bytes: self.stored.load(Ordering::Relaxed), ram_bytes: ram, saturated: self.saturated.load(Ordering::Relaxed) }
    }

    pub fn report(&self) -> String {
        let s = self.stats();
        let gb = |b: usize| b as f64 / (1u64 << 30) as f64;
        format!(
            "{}: {} sets × {} coils, {:.2} GiB raw → {:.2} GiB stored (×{:.2}), {:.2} GiB in RAM{}",
            self.spec.describe(),
            self.n_sets(),
            self.nvc,
            gb(s.raw_bytes),
            gb(s.stored_bytes),
            s.ratio(),
            gb(s.ram_bytes),
            if s.saturated > 0 { format!(", {} values clipped", s.saturated) } else { String::new() }
        ) + &match self.bfp_summary() {
            Some((h, worst, rms)) => format!("; BFP widths {}, worst block step {worst:.3} σ, predicted rms error {rms:.2e} σ", h.iter().map(|(b, n)| format!("{b} bits × {n}")).collect::<Vec<_>>().join(", ")),
            None => String::new(),
        }
    }

    /// Write the index of a file store (`path.json`; all sets must be stored) so it can be reopened. No-op in RAM.
    pub fn finish(&self) -> Result<(), Error> {
        let (Some(p), Some(f)) = (&self.spec.file, &self.file) else { return Ok(()) };
        f.sync_data()?;
        let mut index = Vec::with_capacity(self.n_sets());
        for (t, s) in self.sets.iter().enumerate() {
            match s.get() {
                Some(SetBody::File { idx }) => index.push(idx.iter().flat_map(|&(o, l)| [json!(o), json!(l)]).collect::<Vec<_>>()),
                _ => return Err(Error::Format(format!("k-space store finish: set {t} not stored"))),
            }
        }
        let codec = match self.spec.codec {
            Codec::None => "raw".to_string(),
            Codec::Lz4 => "lz4".into(),
            Codec::Zstd(l) => format!("zstd{l}"),
        };
        let j = json!({
            "format": "hasty-kstore-1",
            "tag": self.tag,
            "nvc": self.nvc,
            "lens": self.lens,
            "block": self.spec.block,
            "codec": codec,
            "quant_rel": self.spec.quant_rel,
            "step": self.step,
            "sigma": self.sigma,
            "file_bytes": *self.next.lock().unwrap(),
            "saturated": self.saturated.load(Ordering::Relaxed),
            "index": index,
        });
        std::fs::write(p.with_extension(ext_json(p)), serde_json::to_string(&j).unwrap())?;
        Ok(())
    }
}

/// `x.kst` → `x.kst.json`
fn ext_json(p: &Path) -> String {
    match p.extension() {
        Some(e) => format!("{}.json", e.to_string_lossy()),
        None => "json".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data(nvc: usize, m: usize, seed: u64) -> Vec<C32> {
        let mut s = seed | 1;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        // noise-like values plus a few large "k-space centre" samples
        (0..nvc * m).map(|i| C32::new(u() as f32 + if i % m < 3 { 500.0 } else { 0.0 }, u() as f32)).collect()
    }

    #[test]
    fn spec_parse() {
        let d = Path::new("/x/default.kst");
        assert!(StoreSpec::parse("ram", d).unwrap().is_dense());
        assert_eq!(StoreSpec::parse("zstd9", d).unwrap().codec, Codec::Zstd(9));
        let s = StoreSpec::parse("disk:lz4+q0.5", d).unwrap();
        assert_eq!((s.file.as_deref(), s.codec, s.quant_rel), (Some(d), Codec::Lz4, Some(0.5)));
        let s = StoreSpec::parse("disk=/tmp/a.kst:zstd", d).unwrap();
        assert_eq!(s.file.as_deref(), Some(Path::new("/tmp/a.kst")));
        assert!(StoreSpec::parse("gzip", d).is_err());
        assert_eq!(StoreSpec::parse("bfp", d).unwrap().bfp, Some(BfpRule::Sigma { q: 0.25 }));
        assert_eq!(StoreSpec::parse("bfp+q0.125", d).unwrap().bfp, Some(BfpRule::Sigma { q: 0.125 }));
        assert_eq!(StoreSpec::parse("bfp18", d).unwrap().bfp, Some(BfpRule::Bits(18)));
        assert!(!StoreSpec::parse("bfp", d).unwrap().is_dense());
        assert!(StoreSpec::parse("disk:bfp", d).is_err() && StoreSpec::parse("bfp30", d).is_err() && StoreSpec::parse("bfp16+q0.5", d).is_err());
        assert!(StoreSpec::parse("lz4+q9", d).is_err());
    }

    #[test]
    fn roundtrip_all_representations() {
        // scratch inside the repository (target/), removed afterwards
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp").join(format!("hasty_kstore_test_{}", std::process::id()));
        let (nvc, lens) = (5, vec![70001usize, 0, 1234, 32768]);
        let sets: Vec<Vec<C32>> = lens.iter().enumerate().map(|(t, &m)| data(nvc, m, 7 + t as u64)).collect();
        for spec in ["ram", "raw", "lz4", "zstd", "zstd7", "disk:raw", "disk:lz4", "lz4+q0.25", "disk:zstd+q0.5", "bfp16", "bfp", "bfp+q0.5", "bfp9"] {
            let mut sp = StoreSpec::parse(spec, &dir.join(format!("{}.kst", spec.replace([':', '+'], "_")))).unwrap();
            sp.block = 10000;
            let sigma = 0.2887; // std of U(−0.5, 0.5)
            let ks = KStore::create(sp.clone(), nvc, lens.clone(), Some(sigma), "test").unwrap();
            // filled out of order, concurrently
            (0..lens.len()).into_par_iter().rev().for_each(|t| ks.put(t, &sets[t]).unwrap());
            ks.finish().unwrap();
            let check = |ks: &KStore| {
                for t in 0..lens.len() {
                    let v = ks.read_vec(t).unwrap();
                    let tol = if sp.quant_rel.is_some() {
                        0.5 * ks.step() as f32 + 1e-4
                    } else if sp.bfp.is_some() {
                        // half the worst block step (bfp_summary reports it over σ)
                        let (_, worst, _) = ks.bfp_summary().unwrap();
                        (0.5 * worst * sigma) as f32
                    } else {
                        0.0
                    };
                    let err = v.iter().zip(&sets[t]).map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs())).fold(0.0f32, f32::max);
                    assert!(err <= tol, "{spec} set {t}: error {err} > {tol}");
                    // coil subset
                    if lens[t] > 0 {
                        let cs = [3usize, 0];
                        let mut o = vec![C32::new(0.0, 0.0); 2 * lens[t]];
                        ks.read(t, Some(&cs), &mut o).unwrap();
                        let m = lens[t];
                        assert_eq!(&o[..m], &v[3 * m..4 * m], "{spec}: subset");
                        assert_eq!(&o[m..], &v[..m], "{spec}: subset");
                    }
                }
            };
            check(&ks);
            if sp.file.is_some() {
                let re = KStore::open(sp.file.as_ref().unwrap()).unwrap();
                assert_eq!(re.tag(), "test");
                check(&re);
            }
            println!("{spec:>16}: {}", ks.report());
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn selection_view_reads_the_kept_samples() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp").join(format!("hasty_kstore_view_{}", std::process::id()));
        let (nvc, lens) = (3, vec![25000usize, 0, 999, 40000]);
        let sets: Vec<Vec<C32>> = lens.iter().enumerate().map(|(t, &m)| data(nvc, m, 31 + t as u64)).collect();
        // prefix + suffix of every "readout" of 1000 samples, one set empty, one set whole blocks skipped
        let sel: Vec<Vec<u32>> = lens.iter().enumerate().map(|(t, &m)| (0..m as u32).filter(|&j| match t { 3 => j < 7000 && (j % 1000 < 40 || j % 1000 > 980), _ => j % 1000 < 40 || j % 1000 > 980 }).collect()).collect();
        for spec in ["ram", "lz4", "disk:zstd", "zstd+q0.25", "bfp", "bfp12"] {
            let mut sp = StoreSpec::parse(spec, &dir.join(format!("{}.kst", spec.replace([':', '+'], "_")))).unwrap();
            sp.block = 4096;
            let ks = KStore::create(sp.clone(), nvc, lens.clone(), Some(0.2887), "view").unwrap();
            (0..lens.len()).into_par_iter().for_each(|t| ks.put(t, &sets[t]).unwrap());
            ks.finish().unwrap();
            let full: Vec<Vec<C32>> = (0..lens.len()).map(|t| ks.read_vec(t).unwrap()).collect();
            let ks = if sp.file.is_some() { KStore::open(sp.file.as_ref().unwrap()).unwrap() } else { ks };
            let v = ks.with_selection(sel.clone()).unwrap();
            assert!(v.is_view());
            assert_eq!(v.max_set(), nvc * sel.iter().map(|s| s.len()).max().unwrap());
            for t in 0..lens.len() {
                let (m, ms) = (lens[t], sel[t].len());
                assert_eq!((v.len(t), v.stored_len(t)), (ms, m));
                let got = v.read_vec(t).unwrap();
                for c in 0..nvc {
                    for (i, &j) in sel[t].iter().enumerate() {
                        assert_eq!(got[c * ms + i], full[t][c * m + j as usize], "{spec} set {t} coil {c} sample {j}");
                    }
                }
                if ms > 0 {
                    let mut o = vec![C32::new(0.0, 0.0); ms];
                    v.read(t, Some(&[2]), &mut o).unwrap();
                    assert_eq!(&o[..], &got[2 * ms..3 * ms], "{spec}: coil subset");
                }
            }
        }
        let ks = KStore::create(StoreSpec::dense(), 1, vec![10], None, "").unwrap();
        assert!(ks.with_selection(vec![vec![3, 2]]).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bfp_sigma_rule_bounds_every_block_step() {
        // k-space-like rows: a decaying centre peak up to 4000 σ on noise σ; the σ rule's width keeps every block's
        // step ≤ q σ, the error per value ≤ half a step, and the rms error at the predicted √(step²/12)
        let (nvc, m) = (6usize, 20000usize);
        let sigma = 1.0f64;
        let mut s = 0x1234_5678u64;
        let mut g = || {
            // Box–Muller from xorshift
            let mut u = || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
            };
            (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos()
        };
        let y: Vec<C32> = (0..nvc * m).map(|i| {
            let j = (i % m) as f64;
            let a = 4000.0 * (-j / 300.0).exp() / (1.0 + (i / m) as f64);
            C32::new((a + sigma * g()) as f32, (0.3 * a + sigma * g()) as f32)
        }).collect();
        for q in [0.25f64, 0.5] {
            let sp = StoreSpec::parse(&format!("bfp+q{q}"), Path::new("/unused")).unwrap();
            let ks = KStore::create(sp, nvc, vec![m], Some(sigma), "").unwrap();
            ks.put(0, &y).unwrap();
            let (h, worst, rms_pred) = ks.bfp_summary().unwrap();
            let b = h[0].0;
            let formula = (4000.0f64 / sigma).log2() + 2.0 + (1.0 / q).log2();
            assert!(worst <= q + 1e-12, "q {q}: worst step {worst} σ");
            assert!((b as f64) <= formula.ceil() && (b as f64) >= formula.floor() - 1.0, "q {q}: width {b} vs rule {formula:.2}");
            let d = ks.read_vec(0).unwrap();
            let (mut se, mut mx) = (0.0f64, 0.0f64);
            for (a, b) in d.iter().zip(&y) {
                for e in [(a.re - b.re) as f64, (a.im - b.im) as f64] {
                    se += e * e;
                    mx = mx.max(e.abs());
                }
            }
            let rms = (se / (2 * y.len()) as f64).sqrt() / sigma;
            println!("q {q}: {b} bits, worst step {worst:.3} σ, max error {mx:.3} σ, rms {rms:.2e} σ (predicted {rms_pred:.2e})");
            assert!(mx <= 0.5 * worst * sigma + 1e-9);
            assert!((rms / rms_pred - 1.0).abs() < 0.15, "rms {rms:.3e} vs predicted {rms_pred:.3e}");
        }
        // noise-free data: the σ rule refuses, a fixed width works
        assert!(KStore::create(StoreSpec::parse("bfp", Path::new("/u")).unwrap(), 1, vec![4], None, "").is_err());
        assert!(KStore::create(StoreSpec::parse("bfp16", Path::new("/u")).unwrap(), 1, vec![4], None, "").is_ok());
    }

    #[test]
    fn quantisation_noise_matches_rel2_over_12() {
        let n = 200_000;
        let x = data(1, n, 99);
        let sigma = 0.2887;
        let mut sp = StoreSpec::parse("lz4+q0.5", Path::new("/unused")).unwrap();
        sp.block = 4096;
        let ks = KStore::create(sp, 1, vec![n], Some(sigma), "").unwrap();
        ks.put(0, &x).unwrap();
        let y = ks.read_vec(0).unwrap();
        let var = x.iter().zip(&y).map(|(a, b)| ((a.re - b.re) as f64).powi(2) + ((a.im - b.im) as f64).powi(2)).sum::<f64>() / (2 * n) as f64;
        let expect = (0.5 * sigma).powi(2) / 12.0;
        assert!((var / expect - 1.0).abs() < 0.03, "added variance {var:.3e} vs {expect:.3e}");
    }
}
