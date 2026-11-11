//! Streaming loader for simulator output: k-space data and per-sample records are read readout by readout from the
//! `.npy` files (positional reads, any order, any thread) instead of being loaded whole, then whitened, optionally
//! filtered (calibration region) and coil-compressed on the way in. With [`SimData::load_lazy`] a 10-minute scan
//! (54 GB of 32-coil data) never exists uncompressed in RAM: each set arrives at its final size (`nvc` coils) and
//! goes into a [`KStore`](crate::kstore::KStore) of the chosen representation.
//!
//! Noise. Sims store noise-free data; the legacy loader adds Ψ-correlated noise to the raw coils with one sequential
//! generator. Here it is added after whitening, where Ψ-correlated noise is white: W L n (W = Ψ^{-1/2}, L Lᵀ = Ψ) is
//! an orthogonal map of white noise, so iid complex Gaussian noise of σ/√2 per component has exactly the legacy
//! distribution. The generator is counter-based (a function of the seed, the sample's index in the scan and the
//! coil), so every sample gets the same noise however the scan is split, ordered or parallelised, and the
//! calibration region and the compressed data see the same noise (as in a real scan). The realisation differs from
//! the legacy loader's.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use hasty_cuda::C32;
use num_complex::Complex64;
use rayon::prelude::*;

use crate::data::{SampleSet, SimData};
use crate::Error;

/// A `.npy` file read by positional reads (C order, little-endian).
pub struct NpyFile {
    pub path: PathBuf,
    file: File,
    pub shape: Vec<usize>,
    pub descr: String,
    /// byte offset of element 0
    pub offset: u64,
    pub fortran: bool,
}

impl NpyFile {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let file = File::open(path)?;
        let mut pre = [0u8; 12];
        file.read_exact_at(&mut pre, 0)?;
        if &pre[..6] != b"\x93NUMPY" {
            return Err(Error::Format(format!("{}: not npy", path.display())));
        }
        let (hlen, hs) = if pre[6] == 1 { (u16::from_le_bytes([pre[8], pre[9]]) as usize, 10) } else { (u32::from_le_bytes([pre[8], pre[9], pre[10], pre[11]]) as usize, 12) };
        let mut hb = vec![0u8; hlen];
        file.read_exact_at(&mut hb, hs as u64)?;
        let h = String::from_utf8_lossy(&hb).to_string();
        let descr = h.split("'descr':").nth(1).and_then(|s| s.split('\'').nth(1)).unwrap_or("").to_string();
        let fortran = h.contains("'fortran_order': True");
        let shp = h.split("'shape':").nth(1).and_then(|s| s.split('(').nth(1)).and_then(|s| s.split(')').next()).unwrap_or("");
        let shape = shp.split(',').filter_map(|t| t.trim().parse().ok()).collect();
        Ok(Self { path: path.to_path_buf(), file, shape, descr, offset: (hs + hlen) as u64, fortran })
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    fn elem(&self) -> usize {
        match self.descr.as_str() {
            "<c8" | "<f8" => 8,
            "<f4" => 4,
            "<c16" => 16,
            d => panic!("{}: dtype {d}", self.path.display()),
        }
    }

    fn read_raw(&self, start: usize, out: &mut [u8]) -> Result<(), Error> {
        let e = self.elem();
        if start * e + out.len() > self.numel() * e {
            return Err(Error::Format(format!("{}: read past the end", self.path.display())));
        }
        self.file.read_exact_at(out, self.offset + (start * e) as u64)?;
        Ok(())
    }

    /// complex64 elements [start, start + out.len())
    pub fn read_c32(&self, start: usize, out: &mut [C32]) -> Result<(), Error> {
        assert_eq!(self.descr, "<c8", "{}", self.path.display());
        // SAFETY: Complex<f32> is two f32, repr(C); little-endian host
        let b = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 8) };
        self.read_raw(start, b)
    }

    /// float32 elements [start, start + out.len())
    pub fn read_f32(&self, start: usize, out: &mut [f32]) -> Result<(), Error> {
        assert_eq!(self.descr, "<f4", "{}", self.path.display());
        let b = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
        self.read_raw(start, b)
    }
}

/// The data of a lazily loaded sim: the k-space file and, for schedule-based sims, the per-sample records.
pub struct LazySource {
    /// [readout][adc][coil] complex64
    pub data: NpyFile,
    /// noise std per complex coil added on load (σ/√2 per component, white after whitening); None: none
    pub sigma: Option<f64>,
    pub seed: u64,
    /// [readout][adc][SAMPLE_FLOATS] (schedule-based sims)
    pub sched: Option<NpyFile>,
}

#[inline]
fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `out[c]` = iid standard complex Gaussian pairs (unit variance per component) of scan sample `s`: a function of
/// (seed, s) only (Box–Muller on a counter-seeded SplitMix64 stream).
pub fn noise_pairs(seed: u64, s: u64, out: &mut [(f64, f64)]) {
    let mut st = seed ^ s.wrapping_mul(0xD1B5_4A32_D192_ED03);
    let _ = splitmix(&mut st);
    for o in out.iter_mut() {
        let u1 = ((splitmix(&mut st) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let u2 = (splitmix(&mut st) >> 11) as f64 / (1u64 << 53) as f64;
        let r = (-2.0 * u1.ln()).sqrt();
        let (sn, cs) = (2.0 * std::f64::consts::PI * u2).sin_cos();
        *o = (r * cs, r * sn);
    }
}

/// What [`SimData::load_set`] produces.
pub struct LoadOpts<'a> {
    /// whitener Ψ^{-1/2} (nc × nc)
    pub w: &'a [Vec<f64>],
    /// coil compression rows (nvc × nc, applied after whitening); None: all whitened coils
    pub cc: Option<&'a [Vec<Complex64>]>,
    /// keep only samples whose coordinates (k·Δ, rad / voxel) pass
    pub keep: Option<&'a (dyn Fn(&[f32; 3]) -> bool + Sync)>,
    /// load the data (false: coordinates, τ and concomitant coefficients only)
    pub data: bool,
}

impl SimData {
    /// [`load`](Self::load) without the k-space data and per-sample records in RAM (read on demand by
    /// [`load_set`](Self::load_set) and the sample accessors; same noise level, counter-based noise, see the module
    /// docs). Schedule-based sims use the nominal trajectory unless `HASTY_TRAJ=played`.
    pub fn load_lazy(dir: &Path) -> Result<Self, Error> {
        let traj = std::env::var("HASTY_TRAJ").unwrap_or_else(|_| "nominal".into());
        Self::load_traj_opts(dir, &traj, true)
    }

    /// data are read on demand
    pub fn is_lazy(&self) -> bool {
        self.lazy.is_some()
    }

    /// raw (unwhitened) coil data of readout `ro`, [adc][coil]; in memory (noise already added) or from the file
    /// (noise added in [`load_set`](Self::load_set))
    pub fn readout_raw(&self, ro: usize, out: &mut [C32]) -> Result<(), Error> {
        let n = self.na * self.nc;
        assert_eq!(out.len(), n);
        match &self.lazy {
            None => {
                out.copy_from_slice(&self.data[ro * n..(ro + 1) * n]);
                Ok(())
            }
            Some(l) => l.data.read_c32(ro * n, out),
        }
    }

    /// Whitened data energy per ADC sample index: the mean over `ros` of Σ_c |(W y)_c|² (W = `w`, Ψ^{-1/2}), not
    /// normalised — in the units of the noise σ² per whitened coil, for noise-referenced segment weights
    /// (`hasty_nufft::snr_time_weights`). Raw data as [`readout_raw`](Self::readout_raw) returns them (lazy files:
    /// before the noise is added).
    pub fn adc_energy_whitened(&self, ros: &[usize], w: &[Vec<f64>]) -> Result<Vec<f64>, Error> {
        use rayon::prelude::*;
        let (na, nc) = (self.na, self.nc);
        let parts: Vec<Vec<f64>> = ros
            .par_chunks(16)
            .map(|chunk| -> Result<Vec<f64>, Error> {
                let mut e = vec![0.0f64; na];
                let mut buf = vec![C32::new(0.0, 0.0); na * nc];
                for &ro in chunk {
                    self.readout_raw(ro, &mut buf)?;
                    for n in 0..na {
                        let src = &buf[n * nc..(n + 1) * nc];
                        for c in 0..nc {
                            let (mut re, mut im) = (0.0f64, 0.0f64);
                            for c2 in 0..nc {
                                re += w[c][c2] * src[c2].re as f64;
                                im += w[c][c2] * src[c2].im as f64;
                            }
                            e[n] += re * re + im * im;
                        }
                    }
                }
                Ok(e)
            })
            .collect::<Result<_, _>>()?;
        let mut e = vec![0.0f64; na];
        for p in &parts {
            e.iter_mut().zip(p).for_each(|(a, b)| *a += b);
        }
        let k = ros.len().max(1) as f64;
        Ok(e.into_iter().map(|x| x / k).collect())
    }

    /// Data energy per ADC sample index, Σ over `ros` and coils of |y|², normalised to a maximum of 1 (raw data, memory
    /// or file). For a centre-out readout it falls steeply with the readout time: the weight of each readout time in the
    /// data-norm model error (`hasty_nufft::build_segments_svd_weighted`, time weights).
    pub fn adc_energy(&self, ros: &[usize]) -> Result<Vec<f64>, Error> {
        use rayon::prelude::*;
        let (na, nc) = (self.na, self.nc);
        let parts: Vec<Vec<f64>> = ros
            .par_chunks(16)
            .map(|chunk| -> Result<Vec<f64>, Error> {
                let mut e = vec![0.0f64; na];
                let mut buf = vec![C32::new(0.0, 0.0); na * nc];
                for &ro in chunk {
                    self.readout_raw(ro, &mut buf)?;
                    for n in 0..na {
                        e[n] += buf[n * nc..(n + 1) * nc].iter().map(|z| (z.re as f64).powi(2) + (z.im as f64).powi(2)).sum::<f64>();
                    }
                }
                Ok(e)
            })
            .collect::<Result<_, _>>()?;
        let mut e = vec![0.0f64; na];
        for p in &parts {
            e.iter_mut().zip(p).for_each(|(a, b)| *a += b);
        }
        let m = e.iter().cloned().fold(0.0f64, f64::max).max(1e-300);
        Ok(e.into_iter().map(|x| x / m).collect())
    }

    /// The sample set of readouts `ros` (in this order), like [`sample_set`](Self::sample_set) but parallel over
    /// readouts, from memory or the file, with optional coil compression (y is `nvc × m` coil-major) and sample
    /// filter. Without `o.data` y is empty.
    pub fn load_set(&self, ros: &[usize], o: &LoadOpts) -> Result<SampleSet, Error> {
        let (na, nc) = (self.na, self.nc);
        let nout = o.cc.map_or(nc, |c| c.len());
        let wf: Vec<f64> = o.w.iter().flat_map(|r| r.iter().copied()).collect();
        assert_eq!(wf.len(), nc * nc);
        let uf: Vec<Complex64> = o.cc.map(|c| c.iter().flat_map(|r| r.iter().copied()).collect()).unwrap_or_default();
        let (noise, seed) = match &self.lazy {
            Some(l) => (l.sigma.map(|s| s / 2f64.sqrt()), l.seed),
            None => (None, 0),
        };
        struct Part {
            coords: [Vec<f32>; 3],
            tau: Vec<f64>,
            wq: Vec<[f64; 4]>,
            y: Vec<C32>,
        }
        let parts: Vec<Part> = ros
            .par_iter()
            .map(|&ro| -> Result<Part, Error> {
                let (k, sp) = self.readout_samples(ro);
                let mut p = Part { coords: [vec![], vec![], vec![]], tau: vec![], wq: vec![], y: vec![] };
                let mut sel = Vec::with_capacity(na);
                for n in 0..na {
                    let c = [0, 1, 2].map(|a| (k[n][a] * self.vox[a]) as f32);
                    if o.keep.is_some_and(|f| !f(&c)) {
                        continue;
                    }
                    sel.push(n);
                    for a in 0..3 {
                        p.coords[a].push(c[a]);
                    }
                    let s = &sp[n * hasty_mrsim::sequence::SAMPLE_FLOATS..(n + 1) * hasty_mrsim::sequence::SAMPLE_FLOATS];
                    p.wq.push([s[9] as f64, s[10] as f64, s[11] as f64, s[12] as f64]);
                    p.tau.push(s[13] as f64);
                }
                if !o.data || sel.is_empty() {
                    return Ok(p);
                }
                let mut raw = vec![C32::new(0.0, 0.0); na * nc];
                self.readout_raw(ro, &mut raw)?;
                p.y = vec![C32::new(0.0, 0.0); sel.len() * nout];
                let (mut z, mut g) = (vec![Complex64::new(0.0, 0.0); nc], vec![(0.0, 0.0); nc]);
                for (j, &n) in sel.iter().enumerate() {
                    let x = &raw[n * nc..(n + 1) * nc];
                    for c in 0..nc {
                        let (mut re, mut im) = (0.0f64, 0.0f64);
                        let wr = &wf[c * nc..(c + 1) * nc];
                        for c2 in 0..nc {
                            re += wr[c2] * x[c2].re as f64;
                            im += wr[c2] * x[c2].im as f64;
                        }
                        z[c] = Complex64::new(re, im);
                    }
                    if let Some(sg) = noise {
                        noise_pairs(seed, (ro * na + n) as u64, &mut g);
                        for c in 0..nc {
                            z[c] += Complex64::new(sg * g[c].0, sg * g[c].1);
                        }
                    }
                    // known spatially constant phase of the sample (GIRF B0 terms on the played trajectory): removed
                    let s15 = sp[n * hasty_mrsim::sequence::SAMPLE_FLOATS + 15] as f64;
                    if s15 != 0.0 {
                        let r = Complex64::from_polar(1.0, s15);
                        for v in z.iter_mut() {
                            *v *= r;
                        }
                    }
                    let out = &mut p.y[j * nout..(j + 1) * nout];
                    if uf.is_empty() {
                        for c in 0..nc {
                            out[c] = C32::new(z[c].re as f32, z[c].im as f32);
                        }
                    } else {
                        for v in 0..nout {
                            let u = &uf[v * nc..(v + 1) * nc];
                            let (mut re, mut im) = (0.0f64, 0.0f64);
                            for c in 0..nc {
                                re += u[c].re * z[c].re - u[c].im * z[c].im;
                                im += u[c].re * z[c].im + u[c].im * z[c].re;
                            }
                            out[v] = C32::new(re as f32, im as f32);
                        }
                    }
                }
                Ok(p)
            })
            .collect::<Result<_, _>>()?;
        let m: usize = parts.iter().map(|p| p.tau.len()).sum();
        let mut set = SampleSet { readouts: ros.to_vec(), coords: [Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m)], y: vec![], tau: Vec::with_capacity(m), wq: Vec::with_capacity(m) };
        for p in &parts {
            for a in 0..3 {
                set.coords[a].extend_from_slice(&p.coords[a]);
            }
            set.tau.extend_from_slice(&p.tau);
            set.wq.extend_from_slice(&p.wq);
        }
        if o.data {
            let offs: Vec<usize> = parts.iter().scan(0, |o, p| { let r = *o; *o += p.tau.len(); Some(r) }).collect();
            set.y = vec![C32::new(0.0, 0.0); nout * m];
            set.y.par_chunks_mut(m.max(1)).enumerate().for_each(|(c, row)| {
                for (p, &off) in parts.iter().zip(&offs) {
                    for j in 0..p.tau.len() {
                        row[off + j] = p.y[j * nout + c];
                    }
                }
            });
        }
        Ok(set)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npy_positional_reads_and_noise_counter() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp").join(format!("hasty_kload_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // a "sample records" file [readout][adc][16] f32 and a data file [readout][adc][coil] c8
        let (nro, na, nf, nc) = (5usize, 7usize, 16usize, 3usize);
        let rec: Vec<f32> = (0..nro * na * nf).map(|i| i as f32 * 0.5).collect();
        hasty_value::npy::write(&dir.join("rec.npy"), &hasty_value::Tensor::from_slice(&[nro, na, nf], &rec).unwrap()).unwrap();
        let dat: Vec<num_complex::Complex32> = (0..nro * na * nc).map(|i| num_complex::Complex32::new(i as f32, -(i as f32))).collect();
        hasty_value::npy::write(&dir.join("dat.npy"), &hasty_value::Tensor::from_slice(&[nro, na, nc], &dat).unwrap()).unwrap();
        let f = NpyFile::open(&dir.join("rec.npy")).unwrap();
        assert_eq!((f.shape.clone(), f.numel()), (vec![nro, na, nf], nro * na * nf));
        for ro in [0, 3, 4] {
            let mut b = vec![0f32; na * nf];
            f.read_f32(ro * na * nf, &mut b).unwrap();
            assert_eq!(&b[..], &rec[ro * na * nf..(ro + 1) * na * nf]);
        }
        let mut past = vec![0f32; 2];
        assert!(f.read_f32(nro * na * nf - 1, &mut past).is_err());
        let d = NpyFile::open(&dir.join("dat.npy")).unwrap();
        let mut b = vec![C32::new(0.0, 0.0); na * nc];
        d.read_c32(2 * na * nc, &mut b).unwrap();
        assert!(b.iter().zip(&dat[2 * na * nc..3 * na * nc]).all(|(x, y)| x.re == y.re && x.im == y.im));
        // the noise of a sample depends only on (seed, sample): unit variance, independent of the call order
        let (mut a, mut c) = (vec![(0.0, 0.0); 4], vec![(0.0, 0.0); 4]);
        noise_pairs(7, 123, &mut a);
        noise_pairs(7, 124, &mut c);
        let mut a2 = vec![(0.0, 0.0); 4];
        noise_pairs(7, 123, &mut a2);
        assert_eq!(a, a2);
        assert_ne!(a, c);
        let n = 200_000u64;
        let (mut s2, mut m) = (0.0, 0.0);
        let mut g = vec![(0.0, 0.0); 2];
        for s in 0..n {
            noise_pairs(1, s, &mut g);
            for (x, y) in &g {
                s2 += x * x + y * y;
                m += x + y;
            }
        }
        let k = (4 * n) as f64;
        assert!((s2 / k - 1.0).abs() < 0.01 && (m / k).abs() < 0.01, "var {} mean {}", s2 / k, m / k);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
