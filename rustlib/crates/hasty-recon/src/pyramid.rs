//! Coarse-to-fine spatial pyramid over grids of one FOV (`SimData::set_recon_grid`): the k-space samples a coarse
//! grid can represent ([`KCrop`]), and band-limited resampling of images between the levels ([`resample_fourier`],
//! [`InitImages`]).
//!
//! k-space crop. A grid of voxel Δ_a represents |k_a Δ_a| ≤ π (its Nyquist box); samples outside alias on the
//! oversampled NUFFT grid, so they are dropped. Kept: the ellipsoid Σ_a (k_a Δ_a)² ≤ (π · frac)² (a ball in the
//! grid's normalised coordinates, intersected with the box when frac > 1). A ball rather than the box because
//! (i) the trajectory's density is radially symmetric (3D centre-out yarn, innermost shell ≈ 400× the edge density),
//! so the box corners would add sparse, direction-dependent coverage and an anisotropic point-spread function;
//! (ii) along a readout |k| first grows monotonically, so the ball keeps one contiguous readout prefix (and, for an
//! out-and-back readout, a suffix) — the same ADC window for every readout. `prefix` keeps only the outgoing part
//! up to the first sample outside the ball: the kept ADC window then ends early, and the segment count of the
//! B0 / concomitant model (∝ the phase spread over the window) drops with it, at the price of the returning
//! half of the central samples.
//!
//! Resampling. Same FOV, voxel = FOV / dims: the Fourier series coefficients of a grid image, X_m = Σ_i x_i
//! e^{−2πi m (i − ⌊d/2⌋)/d} (m ∈ [−⌊d/2⌋, d − ⌊d/2⌋)), are the NUFFT model's data at the DFT frequencies k = 2π m / FOV
//! (`hasty-nufft` convention: unnormalised Σ_v x_v e^{−i k·u_v}, u_v = (i − ⌊d/2⌋) Δ). [`resample_fourier`] keeps
//! them (zero-padding / cropping in k-space, per axis, with the origin phase of odd sizes and a symmetric split /
//! fold of the Nyquist bin), so the resampled image predicts the same data on the common band: its values are the
//! band-limited interpolant times the voxel-volume ratio Π d_src / d_dst (exact for band-limited inputs; up then
//! down is the identity).

use std::f64::consts::PI;
use std::path::{Path, PathBuf};

use hasty_cuda::C32;
use hasty_mrsim::Vec3;
use num_complex::Complex64;
use rayon::prelude::*;
use rustfft::FftPlanner;

use crate::data::{read_npy, SampleSet, SimData};
use crate::kload::NpyFile;
use crate::Error;

// ───────────────────────────────────────────────────────────── k-space crop

/// See the module docs. String form `[FRAC][:prefix]` (FRAC default 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KCrop {
    /// ball radius in units of the grid's Nyquist limit π / Δ_a
    pub frac: f64,
    /// keep only each readout's outgoing part up to its first sample outside the ball
    pub prefix: bool,
}

impl Default for KCrop {
    fn default() -> Self {
        Self { frac: 1.0, prefix: false }
    }
}

impl KCrop {
    /// `""`, `FRAC`, `FRAC:prefix`, `prefix`
    pub fn parse(s: &str) -> Result<Self, Error> {
        let bad = || Error::Format(format!("--kcrop '{s}': [FRAC][:prefix], FRAC in (0, √3]"));
        let mut k = Self::default();
        for p in s.split(':').filter(|p| !p.is_empty()) {
            if p == "prefix" {
                k.prefix = true;
            } else {
                k.frac = p.parse().map_err(|_| bad())?;
            }
        }
        if !(k.frac > 0.0 && k.frac <= 3f64.sqrt() + 1e-12) {
            return Err(bad());
        }
        Ok(k)
    }

    /// is a sample at normalised coordinates c = k·Δ (rad / voxel) kept
    #[inline]
    pub fn keep(&self, c: &[f32; 3]) -> bool {
        let r2: f64 = c.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let lim = PI * self.frac;
        r2 <= lim * lim && c.iter().all(|&v| (v as f64).abs() <= PI)
    }

    /// Kept sample indices (ascending) of a set whose samples are readouts of `na` consecutive samples in ADC order
    /// (`SimData::sample_set` / `load_set` order), from the coordinates k·Δ.
    pub fn select(&self, coords: &[Vec<f32>; 3], na: usize) -> Vec<u32> {
        let m = coords[0].len();
        assert!(m < u32::MAX as usize, "set too large for u32 sample indices");
        let mut out = Vec::new();
        for r0 in (0..m).step_by(na.max(1)) {
            for j in r0..(r0 + na).min(m) {
                if self.keep(&[coords[0][j], coords[1][j], coords[2][j]]) {
                    out.push(j as u32);
                } else if self.prefix {
                    break;
                }
            }
        }
        out
    }

    pub fn describe(&self) -> String {
        format!("ball |k·Δ| ≤ {:.3}π{}", self.frac, if self.prefix { ", outgoing readout prefix only" } else { "" })
    }
}

/// `v[sel]`
pub fn gather<T: Copy>(v: &[T], sel: &[u32]) -> Vec<T> {
    sel.iter().map(|&j| v[j as usize]).collect()
}

/// rows of length `m` (coil-major data, SoA tables): every row restricted to `sel`
pub fn gather_rows<T: Copy + Send + Sync>(v: &[T], m: usize, sel: &[u32]) -> Vec<T> {
    if m == 0 {
        return vec![];
    }
    assert_eq!(v.len() % m, 0, "gather_rows: {} values are not rows of {m}", v.len());
    v.par_chunks(m).flat_map_iter(|row| sel.iter().map(move |&j| row[j as usize])).collect()
}

/// The set restricted to the samples `sel` (coordinates, τ, concomitant coefficients and — when present — the
/// coil-major data). `readouts` are kept (they index the readouts the samples come from: sample j of the full set
/// belongs to readout `readouts[j / na]`).
pub fn crop_set(set: &SampleSet, sel: &[u32]) -> SampleSet {
    let m = set.coords[0].len();
    SampleSet {
        readouts: set.readouts.clone(),
        coords: [0, 1, 2].map(|a| gather(&set.coords[a], sel)),
        y: if set.y.is_empty() { vec![] } else { gather_rows(&set.y, m, sel) },
        tau: if set.tau.len() == m { gather(&set.tau, sel) } else { vec![] },
        wq: if set.wq.len() == m { gather(&set.wq, sel) } else { vec![] },
    }
}

/// Kept samples per ADC index over all sets and the ADC window that holds them.
#[derive(Clone, Debug)]
pub struct AdcStats {
    pub kept: usize,
    pub total: usize,
    /// kept samples at ADC index n, all sets
    pub per_adc: Vec<usize>,
    /// readouts (all sets)
    pub readouts: usize,
    /// first / last ADC index with a kept sample
    pub window: (usize, usize),
}

impl AdcStats {
    /// `sel[t]`: kept indices of set t, which has `lens[t]` samples (readouts of `na`)
    pub fn new(sel: &[Vec<u32>], lens: &[usize], na: usize) -> Self {
        let mut per_adc = vec![0usize; na];
        for s in sel {
            for &j in s {
                per_adc[j as usize % na] += 1;
            }
        }
        let lo = per_adc.iter().position(|&c| c > 0).unwrap_or(0);
        let hi = per_adc.iter().rposition(|&c| c > 0).unwrap_or(0);
        Self { kept: sel.iter().map(|s| s.len()).sum(), total: lens.iter().sum(), per_adc, readouts: lens.iter().map(|&m| m / na.max(1)).sum(), window: (lo, hi) }
    }

    pub fn fraction(&self) -> f64 {
        self.kept as f64 / self.total.max(1) as f64
    }

    /// fraction of the readouts that keep ADC sample n
    pub fn keep_fraction(&self) -> Vec<f64> {
        self.per_adc.iter().map(|&c| c as f64 / self.readouts.max(1) as f64).collect()
    }
}

// ───────────────────────────────────────────────────────────── band-limited resampling

/// Fourier coefficients of one line of length `n` (unnormalised FFT, bin k = m mod n) → the line of length `n2`
/// (FFT bins), keeping X_m on the common band: origin phase e^{2πi m ⌊n/2⌋/n} / e^{−2πi m ⌊n2/2⌋/n2}; an even source
/// Nyquist bin is split half / half onto ±n/2 when upsampling, the target Nyquist bin of an even n2 < n takes the sum
/// of ±n2/2 (the adjoint of the split, so up then down is the identity).
fn map_line(src: &[Complex64], n2: usize) -> Vec<Complex64> {
    let n = src.len();
    let (h, h2) = ((n / 2) as i64, (n2 / 2) as i64);
    let (lo2, hi2) = (-h2, n2 as i64 - h2);
    let mut out = vec![Complex64::new(0.0, 0.0); n2];
    let ph_src = |m: i64| Complex64::from_polar(1.0, 2.0 * PI * m as f64 * h as f64 / n as f64);
    let ph_dst = |m: i64| Complex64::from_polar(1.0, -2.0 * PI * m as f64 * h2 as f64 / n2 as f64);
    let mut place = |m_src: i64, m_dst: i64, v: Complex64| {
        out[m_dst.rem_euclid(n2 as i64) as usize] += v * ph_src(m_src) * ph_dst(m_dst);
    };
    for m in -h..(n as i64 - h) {
        let c = src[m.rem_euclid(n as i64) as usize];
        if n % 2 == 0 && m == -h && n2 > n {
            // source Nyquist: ±n/2 both exist on the larger grid
            place(m, m, c * 0.5);
            place(m, -m, c * 0.5);
        } else if (lo2..hi2).contains(&m) {
            place(m, m, c);
        } else if n2 % 2 == 0 && n2 < n && m == h2 {
            // +n2/2 folds onto the target Nyquist −n2/2
            place(m, -m, c);
        }
    }
    out
}

/// Resample one complex image (x fastest) from grid `src` to grid `dst` of the same FOV, keeping its Fourier
/// coefficients (see the module docs: values = band-limited interpolant × Π src_a / dst_a). Separable, one axis at a
/// time, lines in parallel; f64 arithmetic.
pub fn resample_fourier(x: &[C32], src: [usize; 3], dst: [usize; 3]) -> Vec<C32> {
    assert_eq!(x.len(), src.iter().product::<usize>(), "resample_fourier: image size");
    let mut cur: Vec<Complex64> = x.iter().map(|c| Complex64::new(c.re as f64, c.im as f64)).collect();
    let mut dims = src;
    let mut planner = FftPlanner::<f64>::new();
    for a in 0..3 {
        let (n, n2) = (dims[a], dst[a]);
        if n == n2 {
            continue;
        }
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n2);
        let mut nd = dims;
        nd[a] = n2;
        let stride = [1, dims[0], dims[0] * dims[1]][a];
        let stride2 = [1, nd[0], nd[0] * nd[1]][a];
        // lines: all index combinations of the other two axes
        let others: Vec<usize> = (0..3).filter(|&b| b != a).collect();
        let (no0, no1) = (dims[others[0]], dims[others[1]]);
        let st_src = |b: usize| [1, dims[0], dims[0] * dims[1]][b];
        let st_dst = |b: usize| [1, nd[0], nd[0] * nd[1]][b];
        let lines: Vec<(usize, Vec<Complex64>)> = (0..no0 * no1)
            .into_par_iter()
            .map(|l| {
                let (i0, i1) = (l % no0, l / no0);
                let base = i0 * st_src(others[0]) + i1 * st_src(others[1]);
                let mut buf: Vec<Complex64> = (0..n).map(|k| cur[base + k * stride]).collect();
                fwd.process(&mut buf);
                let mut o = map_line(&buf, n2);
                inv.process(&mut o);
                let s = 1.0 / n2 as f64;
                o.iter_mut().for_each(|v| *v *= s);
                (i0 * st_dst(others[0]) + i1 * st_dst(others[1]), o)
            })
            .collect();
        let mut next = vec![Complex64::new(0.0, 0.0); nd.iter().product()];
        for (base, o) in lines {
            for (k, v) in o.into_iter().enumerate() {
                next[base + k * stride2] = v;
            }
        }
        cur = next;
        dims = nd;
    }
    cur.into_iter().map(|c| C32::new(c.re as f32, c.im as f32)).collect()
}

// ───────────────────────────────────────────────────────────── initial images from a previous level

/// The images of a previous level's output: `recon_<tag>.npy` (complex64, shape (X, Y, Z, E, F), x fastest =
/// Fortran order, as `recon_p2np` writes it) or `warm.npy` / `warm_full.npy` (X, Y, Z, E). Read image by image
/// (positional reads) and resampled to the current grid by [`resample_fourier`].
pub struct InitImages {
    pub path: PathBuf,
    pub dims: [usize; 3],
    pub ne: usize,
    pub nf: usize,
    file: Option<NpyFile>,
    /// C-order input (numpy default), held in memory, reordered to x fastest
    mem: Option<Vec<C32>>,
}

impl InitImages {
    /// A file, or a run directory: its newest `recon_*.npy`, else `warm_full.npy`, else `warm.npy`.
    pub fn open(path: &Path) -> Result<Self, Error> {
        let file = if path.is_dir() { Self::pick(path)? } else { path.to_path_buf() };
        let f = NpyFile::open(&file)?;
        if f.descr != "<c8" {
            return Err(Error::Format(format!("--init-from {}: complex64 expected, got {}", file.display(), f.descr)));
        }
        let s = f.shape.clone();
        if s.len() < 3 || s.len() > 5 {
            return Err(Error::Format(format!("--init-from {}: shape {s:?}, expected (X, Y, Z[, E[, F]])", file.display())));
        }
        let dims = [s[0], s[1], s[2]];
        let (ne, nf) = (s.get(3).copied().unwrap_or(1), s.get(4).copied().unwrap_or(1));
        let (file_h, mem) = if f.fortran || s.iter().filter(|&&d| d > 1).count() <= 1 {
            (Some(f), None)
        } else {
            // C order: element (x, y, z, e, f) at f + F (e + E (z + Z (y + Y x))); read whole, reorder
            let a = read_npy(&file)?.c32s();
            let nv = dims[0] * dims[1] * dims[2];
            let mut v = vec![C32::new(0.0, 0.0); a.len()];
            for fr in 0..nf {
                for e in 0..ne {
                    for q in 0..nv {
                        let (x, y, z) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
                        v[q + nv * (e + ne * fr)] = a[fr + nf * (e + ne * (z + dims[2] * (y + dims[1] * x)))];
                    }
                }
            }
            (None, Some(v))
        };
        Ok(Self { path: file, dims, ne, nf, file: file_h, mem })
    }

    fn pick(dir: &Path) -> Result<PathBuf, Error> {
        let mut recon: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir)?
            .flatten()
            .filter(|e| { let n = e.file_name().to_string_lossy().into_owned(); n.starts_with("recon_") && n.ends_with(".npy") })
            .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
            .collect();
        recon.sort();
        if let Some((_, p)) = recon.pop() {
            return Ok(p);
        }
        for n in ["warm_full.npy", "warm.npy"] {
            if dir.join(n).exists() {
                return Ok(dir.join(n));
            }
        }
        Err(Error::Format(format!("--init-from {}: no recon_*.npy, warm_full.npy or warm.npy", dir.display())))
    }

    pub fn nv(&self) -> usize {
        self.dims.iter().product()
    }

    /// source frame used for frame `fr` of a run with `nf` frames (`nf_card` cardiac frames per respiratory bin):
    /// equal counts, one frame (broadcast), or cardiac frames broadcast over the bins
    pub fn frame_for(&self, fr: usize, nf: usize, nf_card: usize) -> Result<usize, Error> {
        if self.nf == nf {
            Ok(fr)
        } else if self.nf == 1 {
            Ok(0)
        } else if self.nf == nf_card {
            Ok(fr % nf_card)
        } else {
            Err(Error::Format(format!("--init-from {}: {} frames for a run with {nf} frames (equal, 1, or the {nf_card} cardiac frames)", self.path.display(), self.nf)))
        }
    }

    /// image (frame f, encode e) on the source grid
    pub fn image_src(&self, f: usize, e: usize) -> Result<Vec<C32>, Error> {
        let nv = self.nv();
        let off = nv * (e + self.ne * f);
        match (&self.file, &self.mem) {
            (_, Some(m)) => Ok(m[off..off + nv].to_vec()),
            (Some(file), None) => {
                let mut v = vec![C32::new(0.0, 0.0); nv];
                file.read_c32(off, &mut v)?;
                Ok(v)
            }
            _ => unreachable!(),
        }
    }

    /// image (frame f, encode e) resampled to `dims`
    pub fn image(&self, f: usize, e: usize, dims: [usize; 3]) -> Result<Vec<C32>, Error> {
        Ok(resample_fourier(&self.image_src(f, e)?, self.dims, dims))
    }

    /// time average of encode e over the source frames, resampled to `dims`
    pub fn mean_image(&self, e: usize, dims: [usize; 3]) -> Result<Vec<C32>, Error> {
        let nv = self.nv();
        let mut acc = vec![Complex64::new(0.0, 0.0); nv];
        for f in 0..self.nf {
            for (a, v) in acc.iter_mut().zip(self.image_src(f, e)?) {
                *a += Complex64::new(v.re as f64, v.im as f64);
            }
        }
        let s = 1.0 / self.nf as f64;
        let m: Vec<C32> = acc.iter().map(|c| C32::new((c.re * s) as f32, (c.im * s) as f32)).collect();
        Ok(resample_fourier(&m, self.dims, dims))
    }
}

// ───────────────────────────────────────────────────────────── truth / input maps on the reconstruction grid

/// Maps on the native (truth) grid (`n` maps of `tdims` voxels, map-major) sampled at the reconstruction voxels'
/// true positions `x_log`: trilinear at the position, or (`boxed`) the mean over the reconstruction voxel
/// ([`SimData::sample_log_box`]).
pub fn native_maps_to_grid(sd: &SimData, maps: &[f32], x_log: &[Vec3], boxed: bool) -> Vec<f32> {
    let tnv = sd.tdims.iter().product::<usize>();
    assert_eq!(maps.len() % tnv, 0, "native maps: {} values for {tnv} voxels", maps.len());
    let n = maps.len() / tnv;
    let nv = x_log.len();
    let mut out = vec![0f32; n * nv];
    out.par_chunks_mut(nv).enumerate().for_each(|(c, o)| {
        for (q, v) in o.iter_mut().enumerate() {
            *v = if boxed { sd.sample_log_box(maps, c, &x_log[q]) } else { sd.sample_log(maps, c, &x_log[q]) } as f32;
        }
    });
    out
}

/// A float32 map file (fat fraction etc.: shape (X, Y, Z[, F]) or flat) on the reconstruction grid: as stored when
/// its spatial shape (or length) is the reconstruction grid's, resampled from the native grid when it is the native
/// grid's (any number of maps), an error otherwise. Returns the values (map-major) and whether it was resampled.
pub fn maps_on_recon_grid(sd: &SimData, path: &Path, x_log: &[Vec3], boxed: bool) -> Result<(Vec<f32>, bool), Error> {
    let a = read_npy(path)?;
    let v = a.f32s();
    let (nv, tnv) = (sd.nv(), sd.tdims.iter().product::<usize>());
    let spatial = (a.shape.len() >= 3).then(|| [a.shape[0], a.shape[1], a.shape[2]]);
    if a.shape.len() >= 3 && !a.fortran && a.shape.iter().filter(|&&d| d > 1).count() > 1 {
        return Err(Error::Format(format!("{}: C-order map; x-fastest (Fortran order) expected", path.display())));
    }
    let on_recon = match spatial { Some(s) => s == sd.dims, None => v.len() % nv == 0 };
    if on_recon || sd.native_grid() {
        return Ok((v, false));
    }
    let on_native = match spatial { Some(s) => s == sd.tdims, None => v.len() % tnv == 0 };
    if on_native {
        return Ok((native_maps_to_grid(sd, &v, x_log, boxed), true));
    }
    Err(Error::Format(format!("{}: shape {:?} is neither the reconstruction grid {:?} nor the native grid {:?}", path.display(), a.shape, sd.dims, sd.tdims)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }
    }

    #[test]
    fn kcrop_ball_and_prefix() {
        // two readouts of an out-and-back profile along random directions: |k·Δ| = 4 u (1 − u) · 1.2π, u = n / (na − 1)
        let na = 101;
        let mut u = rng(3);
        let mut c: [Vec<f32>; 3] = [vec![], vec![], vec![]];
        let mut dirs = vec![];
        for _ in 0..2 {
            let d = [u(), u(), u()];
            let nd = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            dirs.push(d.map(|x| x / nd));
        }
        for d in &dirs {
            for n in 0..na {
                let t = n as f64 / (na - 1) as f64;
                let r = 4.0 * t * (1.0 - t) * 1.2 * PI;
                for a in 0..3 {
                    c[a].push((r * d[a]) as f32);
                }
            }
        }
        let k = KCrop::parse("0.5").unwrap();
        let sel = k.select(&c, na);
        // kept iff r ≤ π/2 (and inside the box, implied)
        let want: Vec<u32> = (0..2 * na).filter(|&j| { let t = (j % na) as f64 / (na - 1) as f64; 4.0 * t * (1.0 - t) * 1.2 <= 0.5 + 1e-6 }).map(|j| j as u32).collect();
        assert_eq!(sel, want);
        // a prefix and a suffix of each readout, same ADC window for both readouts
        assert!(sel.windows(2).all(|w| w[0] < w[1]));
        let st = AdcStats::new(&[sel.clone()], &[2 * na], na);
        assert_eq!((st.window.0, st.window.1), (0, na - 1));
        assert!(st.per_adc.iter().all(|&n| n == 0 || n == 2));
        let kp = KCrop::parse("0.5:prefix").unwrap();
        let sp = kp.select(&c, na);
        let pre: Vec<u32> = want.iter().copied().filter(|&j| (j as usize % na) < na / 2).collect();
        assert_eq!(sp, pre);
        let st = AdcStats::new(&[sp], &[2 * na], na);
        assert_eq!(st.window.0, 0);
        assert!(st.window.1 < na / 2);
        // frac > 1: the box still applies
        let kb = KCrop::parse("1.7").unwrap();
        assert!(!kb.keep(&[3.2, 0.0, 0.0]) && kb.keep(&[3.1, 0.0, 0.0]) && kb.keep(&[3.0, 3.0, 3.0]) && !kb.keep(&[3.1, 3.1, 3.1]));
        assert!(KCrop::parse("0").is_err() && KCrop::parse("x").is_err());
        assert_eq!(KCrop::parse("prefix").unwrap(), KCrop { frac: 1.0, prefix: true });
    }

    #[test]
    fn crop_set_gathers_consistently() {
        let (na, nr, nc) = (7, 3, 2);
        let m = na * nr;
        let set = SampleSet {
            readouts: vec![4, 9, 11],
            coords: [0, 1, 2].map(|a| (0..m).map(|j| (j * 10 + a) as f32).collect()),
            y: (0..nc * m).map(|i| C32::new(i as f32, -(i as f32))).collect(),
            tau: (0..m).map(|j| j as f64).collect(),
            wq: (0..m).map(|j| [j as f64; 4]).collect(),
        };
        let sel: Vec<u32> = vec![0, 1, 7, 8, 9, 20];
        let s = crop_set(&set, &sel);
        assert_eq!(s.coords[1], sel.iter().map(|&j| (j * 10 + 1) as f32).collect::<Vec<_>>());
        assert_eq!(s.tau, sel.iter().map(|&j| j as f64).collect::<Vec<_>>());
        for c in 0..nc {
            for (i, &j) in sel.iter().enumerate() {
                assert_eq!(s.y[c * sel.len() + i], set.y[c * m + j as usize]);
            }
        }
        assert_eq!(s.readouts, set.readouts);
    }

    /// f(p) = Σ_m a_m e^{2πi m p} on the grid points p_i = (i − ⌊n/2⌋)/n of each axis
    fn bandlimited(dims: [usize; 3], modes: &[([i64; 3], Complex64)], scale: f64) -> Vec<C32> {
        let nv = dims.iter().product::<usize>();
        (0..nv)
            .map(|q| {
                let idx = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
                let p: [f64; 3] = std::array::from_fn(|a| (idx[a] as f64 - (dims[a] / 2) as f64) / dims[a] as f64);
                let v: Complex64 = modes.iter().map(|(m, c)| c * Complex64::from_polar(1.0, 2.0 * PI * (0..3).map(|a| m[a] as f64 * p[a]).sum::<f64>())).sum();
                C32::new((v.re * scale) as f32, (v.im * scale) as f32)
            })
            .collect()
    }

    #[test]
    fn resample_exact_for_bandlimited_inputs() {
        // even / odd sizes in both directions; modes strictly inside both bands
        for (src, dst) in [([8, 6, 5], [16, 12, 10]), ([16, 12, 10], [8, 6, 5]), ([9, 7, 6], [12, 15, 4]), ([10, 10, 9], [10, 13, 9])] {
            let mut u = rng(11);
            let lim: [i64; 3] = std::array::from_fn(|a| ((src[a].min(dst[a]) as i64) - 1) / 2);
            let modes: Vec<([i64; 3], Complex64)> = (0..12).map(|_| {
                let m: [i64; 3] = std::array::from_fn(|a| ((u() + 0.5) * (2 * lim[a] + 1) as f64).floor() as i64 - lim[a]);
                (m, Complex64::new(u(), u()))
            }).collect();
            let x = bandlimited(src, &modes, 1.0);
            let got = resample_fourier(&x, src, dst);
            let vol = src.iter().product::<usize>() as f64 / dst.iter().product::<usize>() as f64;
            let want = bandlimited(dst, &modes, vol);
            let (mut e, mut n) = (0f64, 0f64);
            for (a, b) in got.iter().zip(&want) {
                e += ((a.re - b.re) as f64).powi(2) + ((a.im - b.im) as f64).powi(2);
                n += (b.re as f64).powi(2) + (b.im as f64).powi(2);
            }
            let rel = (e / n).sqrt();
            println!("band-limited {src:?} → {dst:?}: relative error {rel:.2e}");
            assert!(rel < 2e-6, "{src:?} → {dst:?}: relative error {rel:.2e}");
        }
    }

    #[test]
    fn resample_up_then_down_is_identity() {
        // arbitrary (not band-limited) images, Nyquist bins included
        for (src, dst) in [([8, 6, 4], [16, 9, 10]), ([7, 6, 5], [14, 12, 15]), ([12, 10, 8], [13, 10, 9])] {
            let mut u = rng(5);
            let x: Vec<C32> = (0..src.iter().product::<usize>()).map(|_| C32::new(u() as f32, u() as f32)).collect();
            let back = resample_fourier(&resample_fourier(&x, src, dst), dst, src);
            let err = x.iter().zip(&back).map(|(a, b)| (a.re - b.re).abs().max((a.im - b.im).abs())).fold(0f32, f32::max);
            println!("round trip {src:?} → {dst:?} → back: max error {err:.2e}");
            assert!(err < 1e-5, "{src:?} → {dst:?} → back: max error {err:.2e}");
        }
    }

    #[test]
    fn resample_keeps_nufft_data_on_the_common_band() {
        // y(k) = Σ_v x_v e^{−i k·u_v}, u_v = (i − ⌊n/2⌋) Δ, Δ = FOV / n: equal for the coarse image and its resampled
        // version at the coarse grid's DFT frequencies k = 2π m / FOV (the hasty-nufft convention)
        let (src, dst) = ([6usize, 5, 4], [12usize, 10, 9]);
        let fov = [0.3f64, 0.25, 0.2];
        let mut u = rng(9);
        let x: Vec<C32> = (0..src.iter().product::<usize>()).map(|_| C32::new(u() as f32, u() as f32)).collect();
        let y = resample_fourier(&x, src, dst);
        let data = |img: &[C32], d: [usize; 3], k: [f64; 3]| -> Complex64 {
            img.iter().enumerate().map(|(q, v)| {
                let idx = [q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1])];
                let ph: f64 = (0..3).map(|a| k[a] * (idx[a] as f64 - (d[a] / 2) as f64) * fov[a] / d[a] as f64).sum();
                Complex64::new(v.re as f64, v.im as f64) * Complex64::from_polar(1.0, -ph)
            }).sum()
        };
        for m in [[0i64, 0, 0], [1, -2, 1], [-2, 2, -1], [2, 1, 1]] {
            let k: [f64; 3] = std::array::from_fn(|a| 2.0 * PI * m[a] as f64 / fov[a]);
            let (a, b) = (data(&x, src, k), data(&y, dst, k));
            assert!((a - b).norm() < 1e-5 * a.norm().max(1.0), "m {m:?}: {a} vs {b}");
        }
    }

    fn write_npy_c8(path: &Path, shape: &[usize], fortran: bool, v: &[C32]) {
        let hdr = format!("{{'descr': '<c8', 'fortran_order': {}, 'shape': ({}), }}", if fortran { "True" } else { "False" }, shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "));
        let mut h = hdr.into_bytes();
        while (10 + h.len() + 1) % 64 != 0 {
            h.push(b' ');
        }
        h.push(b'\n');
        let mut out = vec![0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0];
        out.extend_from_slice(&(h.len() as u16).to_le_bytes());
        out.extend_from_slice(&h);
        for c in v {
            out.extend_from_slice(&c.re.to_le_bytes());
            out.extend_from_slice(&c.im.to_le_bytes());
        }
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn init_images_read_frames_encodes_and_directories() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp").join(format!("hasty_pyramid_init_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (d, ne, nf) = ([4usize, 3, 2], 2usize, 3usize);
        let nv = 24;
        // value encodes (q, e, f)
        let val = |q: usize, e: usize, f: usize| C32::new(q as f32 + 100.0 * e as f32, f as f32);
        let fort: Vec<C32> = (0..nf).flat_map(|f| (0..ne).flat_map(move |e| (0..nv).map(move |q| val(q, e, f)))).collect();
        write_npy_c8(&dir.join("recon_a.npy"), &[d[0], d[1], d[2], ne, nf], true, &fort);
        // the same array in C order
        let mut corder = vec![C32::new(0.0, 0.0); fort.len()];
        for f in 0..nf {
            for e in 0..ne {
                for q in 0..nv {
                    let (x, y, z) = (q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1]));
                    corder[f + nf * (e + ne * (z + d[2] * (y + d[1] * x)))] = val(q, e, f);
                }
            }
        }
        write_npy_c8(&dir.join("c_order.npy"), &[d[0], d[1], d[2], ne, nf], false, &corder);
        for p in [dir.join("recon_a.npy"), dir.join("c_order.npy")] {
            let ini = InitImages::open(&p).unwrap();
            assert_eq!((ini.dims, ini.ne, ini.nf), (d, ne, nf));
            let im = ini.image_src(2, 1).unwrap();
            assert!((0..nv).all(|q| im[q] == val(q, 1, 2)), "{}", p.display());
            // same grid: resampling is the identity; mean over frames
            let mean = ini.mean_image(1, d).unwrap();
            assert!((0..nv).all(|q| (mean[q].re - val(q, 1, 0).re).abs() < 1e-4 && (mean[q].im - 1.0).abs() < 1e-5));
            assert_eq!(ini.frame_for(5, 6, 3).unwrap(), 2);
            assert!(ini.frame_for(1, 4, 4).is_err());
        }
        // a run directory: the newest recon_*.npy; else warm_full / warm
        assert_eq!(InitImages::open(&dir).unwrap().path.file_name().unwrap(), "recon_a.npy");
        let d2 = dir.join("w");
        std::fs::create_dir_all(&d2).unwrap();
        write_npy_c8(&d2.join("warm.npy"), &[d[0], d[1], d[2], ne], true, &fort[..ne * nv]);
        let w = InitImages::open(&d2).unwrap();
        assert_eq!((w.ne, w.nf, w.path.file_name().unwrap().to_str().unwrap()), (ne, 1, "warm.npy"));
        assert_eq!(w.frame_for(7, 40, 20).unwrap(), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
