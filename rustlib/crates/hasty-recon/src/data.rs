//! Simulator output (`sim4d`, `simphantom`) → reconstruction inputs: sequence (the spiral regenerated deterministically
//! from the parameters, or a schedule-based sim's stored readouts and per-sample records), whitened multi-coil data per
//! encode/frame, per-sample phase parameters, and physics on the reconstruction grid.

use std::path::{Path, PathBuf};

use hasty_cuda::C32;
use hasty_mrsim::gnl::GradientCoils;
use hasty_mrsim::physio::{CardiacTime, Ecg};
use hasty_mrsim::sequence::{Acquisition, Coverage, Encoding, FlowSpiralParams, SAMPLE_FLOATS};
use hasty_mrsim::{mat_vec, transpose, GradLimits, Mat3, Vec3, GAMMA, GAMMA_BAR};
use nalgebra::{DMatrix, SymmetricEigen};
use serde_json::Value;

use crate::Error;

pub struct NpyArray {
    pub shape: Vec<usize>,
    pub fortran: bool,
    pub descr: String,
    pub bytes: Vec<u8>,
}

pub fn read_npy(path: &Path) -> Result<NpyArray, Error> {
    let buf = std::fs::read(path)?;
    if &buf[..6] != b"\x93NUMPY" {
        return Err(Error::Format(format!("{}: not npy", path.display())));
    }
    let (hlen, hs) = if buf[6] == 1 { (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10) } else { (u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize, 12) };
    let h = std::str::from_utf8(&buf[hs..hs + hlen]).map_err(|e| Error::Format(e.to_string()))?;
    let descr = h.split("'descr':").nth(1).and_then(|s| s.split('\'').nth(1)).unwrap_or("").to_string();
    let fortran = h.contains("'fortran_order': True");
    let shp = h.split("'shape':").nth(1).and_then(|s| s.split('(').nth(1)).and_then(|s| s.split(')').next()).unwrap_or("");
    let shape = shp.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    Ok(NpyArray { shape, fortran, descr, bytes: buf[hs + hlen..].to_vec() })
}

impl NpyArray {
    pub fn f32s(&self) -> Vec<f32> {
        assert_eq!(self.descr, "<f4");
        self.bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }
    pub fn f64s(&self) -> Vec<f64> {
        assert_eq!(self.descr, "<f8");
        self.bytes.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap())).collect()
    }
    pub fn c32s(&self) -> Vec<C32> {
        assert_eq!(self.descr, "<c8");
        self.bytes.chunks_exact(8).map(|b| C32::new(f32::from_le_bytes([b[0], b[1], b[2], b[3]]), f32::from_le_bytes([b[4], b[5], b[6], b[7]]))).collect()
    }
}

fn v3(v: &Value) -> Vec3 {
    [0, 1, 2].map(|i| v[i].as_f64().unwrap())
}
fn m3(v: &Value) -> Mat3 {
    [0, 1, 2].map(|r| [0, 1, 2].map(|c| v[r][c].as_f64().unwrap()))
}

/// per-readout schedule data every reconstruction needs (from the spiral `Acquisition` or a sim's readouts.npy)
pub struct ReadoutInfo {
    pub enc: Vec<usize>,
    pub frame: Vec<usize>,
    pub card: Vec<CardiacTime>,
    pub t_exc: Vec<f64>,
}

/// per-sample records of a schedule-based sim (`simphantom --seq flow`): [readout][adc][SAMPLE_FLOATS], physical axes,
/// the nominal trajectory (`sched_nominal.npy`) or the one played through the simulated GIRF (`sched_played.npy`)
pub struct SchedSamples {
    pub rec: Vec<f32>,
    pub traj: String,
}

pub struct SimData {
    pub dir: PathBuf,
    pub meta: Value,
    /// the spiral acquisition (`sequence.kind` spiral); None for schedule-based sims ([`SimData::spiral`])
    pub acq: Option<Acquisition>,
    pub ro: ReadoutInfo,
    pub sched: Option<SchedSamples>,
    /// velocity-encoding first moments per encode at the echo, logical frame (T·s²/m)
    pub encode_m1: Vec<Vec3>,
    /// [readout][adc][coil]
    pub data: Vec<C32>,
    pub nc: usize,
    pub na: usize,
    /// reconstruction grid (default: the native grid; [`SimData::set_recon_grid`] refines it)
    pub dims: [usize; 3],
    /// reconstruction voxel size per axis (m)
    pub vox: [f64; 3],
    /// native (simulation / truth) grid and its isotropic voxel size: truth fields live here
    pub tdims: [usize; 3],
    pub res: f64,
    pub psi: Vec<Vec<f64>>,
    pub gnl: GradientCoils,
    pub rot: Mat3,
    /// FOV centre in the scanner frame
    pub fov_c: Vec3,
    pub b0: f64,
    pub venc: f64,
    /// coil-major true sensitivities on the true logical grid (reference/oracle only)
    pub smaps_true: Vec<C32>,
    /// off-resonance (ppm) on the true logical grid
    pub field_ppm: Vec<f32>,
    /// truth velocity, logical components: index q + nv (a + 3 f)
    pub truth_vel: Vec<f32>,
    pub truth_frac: Vec<f32>,
    pub n_frames: usize,
    /// data and per-sample records read on demand ([`SimData::load_lazy`], `kload`); None: in memory (`data`)
    pub lazy: Option<std::sync::Arc<crate::kload::LazySource>>,
    /// ADC samples per readout in the sim's files (`n_adc`); `na` ≤ this when a reconstruction window is active
    pub na_file: usize,
    /// file ADC indices of the reconstruction samples (ascending; `meta.sequence.recon_adc`, e.g. v2 without its
    /// rewinder navigator); None: every ADC sample (na == na_file). Every accessor (sample sets, raw data, energies)
    /// presents readouts of `na` samples; the others are reachable through [`SimData::adc_navigator`] and the `_full`
    /// accessors. `HASTY_RECON_WINDOW=all` disables the window.
    pub adc_keep: Option<Vec<usize>>,
    /// per-sample records recomputed through a model chain (`HASTY_TRAJ=girf:<file|variant>`, [`crate::played`])
    pub model_traj: Option<std::sync::Arc<crate::played::ModelTraj>>,
}

/// Per-encode (or per-frame) sample set ready for an operator.
pub struct SampleSet {
    pub readouts: Vec<usize>,
    /// k·Δ, radians per voxel
    pub coords: [Vec<f32>; 3],
    /// whitened data, coil-major (nc × m)
    pub y: Vec<C32>,
    /// time since excitation, s
    pub tau: Vec<f64>,
    /// concomitant coefficients of [z², x²+y², xz, yz] (scanner frame), rad/m²
    pub wq: Vec<[f64; 4]>,
}

/// Physics evaluated at the *true* position of every reconstruction voxel (the image lives in apparent, GNL-warped
/// coordinates when `gnl_aware`; otherwise apparent = true).
pub struct PhysicsGrid {
    /// true logical position (m, relative to the FOV centre)
    pub x_log: Vec<Vec3>,
    pub mono: Vec<[f64; 4]>,
    /// rad/s
    pub omega: Vec<f64>,
    /// GNL Jacobian (scanner frame)
    pub jac: Vec<Mat3>,
    /// whitened sensitivities at the true positions (coil-major)
    pub smaps: Vec<C32>,
}

impl SimData {
    /// load a sim; schedule-based sims use the nominal trajectory unless `HASTY_TRAJ=played` (the GIRF-played one,
    /// i.e. a reconstruction that models the chain exactly) or `HASTY_TRAJ=girf:<girf.json|variant>` (the records
    /// recomputed through a model chain, e.g. an estimated GIRF, with `HASTY_GIRF_HIST` previous TRs, default 10;
    /// [`crate::played`])
    pub fn load(dir: &Path) -> Result<Self, Error> {
        let traj = std::env::var("HASTY_TRAJ").unwrap_or_else(|_| "nominal".into());
        Self::load_traj(dir, &traj)
    }

    /// the spiral acquisition (panics for schedule-based sims: spiral-only tools)
    pub fn spiral(&self) -> &Acquisition {
        self.acq.as_ref().expect("this tool needs a spiral sim (sequence.kind = spiral)")
    }

    pub fn load_traj(dir: &Path, traj: &str) -> Result<Self, Error> {
        Self::load_traj_opts(dir, traj, false)
    }

    /// `lazy`: leave the k-space data and the per-sample records on disk (see `kload`)
    pub fn load_traj_opts(dir: &Path, traj: &str, lazy: bool) -> Result<Self, Error> {
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?).map_err(|e| Error::Format(e.to_string()))?;
        let f = |k: &str| meta[k].as_f64().unwrap();
        let u = |k: &str| meta[k].as_u64().unwrap() as usize;
        let kind = meta["sequence"]["kind"].as_str().unwrap_or("spiral").to_string();
        let p = FlowSpiralParams {
            b0: f("b0"),
            lim: GradLimits { gmax: f("gmax"), smax: f("smax") },
            dt: f("dt"),
            dwell: f("dwell"),
            fov: v3(&meta["fov"]),
            res: v3(&meta["res"]),
            n_int: u("n_int"),
            fov_edge_ratio: f("fov_edge_ratio"),
            venc: f("venc"),
            encoding: match meta["encoding"].as_str().unwrap_or("Referenced") {
                "Hadamard" => Encoding::Hadamard,
                e if e.starts_with("DualVenc") => Encoding::DualVenc { venc_hi: meta["venc_hi"].as_f64().expect("DualVenc sim without venc_hi") },
                _ => Encoding::Referenced,
            },
            coverage: if meta["coverage"] == "Rotated3d" { Coverage::Rotated3d } else { Coverage::StackOfSpirals },
            t_rf: f("t_rf"),
            t_spoil: f("t_spoil"),
            scan_time: f("scan_time"),
            rot_presc: m3(&meta["rot_presc"]),
            fov_center: v3(&meta["fov_center_scanner"]),
            mean_rr: f("mean_rr"),
            rr_rel_std: f("rr_rel_std"),
            n_frames: u("n_frames"),
            seed: meta["seed"].as_u64().unwrap(),
        };
        let nc = u("n_coils");
        let na = u("n_adc");
        let na_file = na;
        let n_ro = u("n_readouts");
        let mut lazy_sched: Option<crate::kload::NpyFile> = None;
        let mut model_traj: Option<std::sync::Arc<crate::played::ModelTraj>> = None;
        let (acq, ro, sched, encode_m1, venc) = match kind.as_str() {
            "spiral" => {
                let acq = Acquisition::new(p.clone());
                assert_eq!(acq.readouts.len(), n_ro, "regenerated schedule differs from the simulated one");
                assert_eq!(acq.n_adc(), na);
                let ro = ReadoutInfo {
                    enc: acq.readouts.iter().map(|r| r.enc).collect(),
                    frame: acq.readouts.iter().map(|r| r.frame).collect(),
                    card: acq.readouts.iter().map(|r| r.card).collect(),
                    t_exc: acq.readouts.iter().map(|r| r.t_exc).collect(),
                };
                let m1 = acq.params.encode_m1();
                (Some(acq), ro, None, m1, p.venc)
            }
            "flow" => {
                let tab = read_npy(&dir.join("readouts.npy"))?.f64s();
                assert_eq!(tab.len(), n_ro * 14, "readouts.npy");
                let ecg = Ecg::new(p.scan_time + 2.0, p.mean_rr, p.rr_rel_std, p.seed);
                let t_exc: Vec<f64> = (0..n_ro).map(|i| tab[14 * i]).collect();
                let ro = ReadoutInfo {
                    enc: (0..n_ro).map(|i| tab[14 * i + 1] as usize).collect(),
                    frame: (0..n_ro).map(|i| tab[14 * i + 3] as usize).collect(),
                    card: t_exc.iter().map(|&t| ecg.at(t)).collect(),
                    t_exc,
                };
                let name = match traj {
                    "nominal" => "sched_nominal.npy",
                    "played" => "sched_played.npy",
                    g if g.starts_with("girf:") => "",
                    other => return Err(Error::Format(format!("trajectory {other}: nominal | played | girf:<girf.json|variant>"))),
                };
                if let Some(spec) = traj.strip_prefix("girf:") {
                    let hist: usize = std::env::var("HASTY_GIRF_HIST").ok().and_then(|v| v.parse().ok()).unwrap_or(10);
                    let girf = crate::played::ModelTraj::girf_of(spec)?;
                    let mt = crate::played::ModelTraj::new(&meta, &tab, &girf, spec, hist)?;
                    if mt.times.len() != na {
                        return Err(Error::Format(format!("model trajectory: {} ADC samples for the sim's {na}", mt.times.len())));
                    }
                    model_traj = Some(std::sync::Arc::new(mt));
                }
                let (rec, sched_file) = if model_traj.is_some() {
                    (vec![], None)
                } else if lazy {
                    let f = crate::kload::NpyFile::open(&dir.join(name))?;
                    assert_eq!(f.numel(), n_ro * na * SAMPLE_FLOATS, "{name}");
                    (vec![], Some(f))
                } else {
                    let rec = read_npy(&dir.join(name))?.f32s();
                    assert_eq!(rec.len(), n_ro * na * SAMPLE_FLOATS, "{name}");
                    (rec, None)
                };
                // first ADC sample of every readout (k, M1)
                let first: Vec<[f32; 6]> = match &sched_file {
                    None if model_traj.is_some() => {
                        use rayon::prelude::*;
                        let mt = model_traj.as_ref().unwrap();
                        (0..n_ro).into_par_iter().map(|i| {
                            let seq: Vec<(usize, hasty_seq::Mat3)> = (i.saturating_sub(mt.hist)..=i).map(|j| (mt.enc[j], mt.rot[j])).collect();
                            let m = mt.stream.moments(&seq, mt.design.t_rf, &mt.times[..1])[0];
                            let r = hasty_mrsim::sequence::sample_record(&m.m0, &m.m1, &m.m2, &m.p, mt.b0, mt.times[0] - mt.design.t_rf);
                            std::array::from_fn(|j| r[j])
                        }).collect()
                    }
                    None => (0..n_ro).map(|i| std::array::from_fn(|j| rec[i * na * SAMPLE_FLOATS + j])).collect(),
                    Some(f) => {
                        use rayon::prelude::*;
                        (0..n_ro).into_par_iter().map(|i| { let mut b = [0f32; 6]; f.read_f32(i * na * SAMPLE_FLOATS, &mut b).map(|_| b) }).collect::<Result<_, _>>()?
                    }
                };
                lazy_sched = sched_file;
                // encode M1 at the echo (first ADC sample), mean over the encode's readouts, physical → logical
                let ne = ro.enc.iter().max().map_or(0, |e| e + 1);
                let rt = transpose(&p.rot_presc);
                let mut m1 = vec![[0.0f64; 3]; ne];
                let mut cnt = vec![0usize; ne];
                for i in 0..n_ro {
                    let r0 = &first[i];
                    for a in 0..3 {
                        m1[ro.enc[i]][a] += r0[3 + a] as f64 / GAMMA;
                    }
                    cnt[ro.enc[i]] += 1;
                }
                let m1: Vec<Vec3> = m1.iter().zip(&cnt).map(|(m, &c)| mat_vec(&rt, &m.map(|x| x / c.max(1) as f64))).collect();
                let venc = meta["sequence"]["encoding"]["venc_lo"].as_f64().unwrap_or(p.venc);
                (None, ro, Some(SchedSamples { rec, traj: traj.into() }), m1, venc)
            }
            other => return Err(Error::Format(format!("sequence kind {other}: SimData reads spiral and flow sims (b0: hasty-bench b0data)"))),
        };
        let dims = [0, 1, 2].map(|i| meta["recon_dims"][i].as_u64().unwrap() as usize);
        let psi: Vec<Vec<f64>> = meta["psi"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()).collect();
        let g = &meta["gnl"];
        let gnl = GradientCoils { r0: g["r0"].as_f64().unwrap(), transverse: [g["transverse"][0].as_f64().unwrap(), g["transverse"][1].as_f64().unwrap()], longitudinal: [g["longitudinal"][0].as_f64().unwrap(), g["longitudinal"][1].as_f64().unwrap()] };
        // HASTY_NOISE_SIGMA: use the noise-free data and add Ψ-correlated noise with this σ per coil (complex, before
        // whitening; σ/√2 per component); 0 = noise-free
        // without data.npy (default since sims store only the clean data): noise at the simulated level
        let env_sigma = std::env::var("HASTY_NOISE_SIGMA").ok().and_then(|v| v.parse::<f64>().ok());
        let sigma = env_sigma.or_else(|| (!dir.join("data.npy").exists()).then(|| meta["noise_sigma"].as_f64().unwrap()));
        let mut lazy_src = None;
        let data = match sigma {
            _ if lazy => {
                // noise (if any) is added on load, after whitening (kload)
                let (file, sg) = match sigma {
                    Some(sg) if dir.join("data_clean.npy").exists() => ("data_clean.npy", Some(sg).filter(|&v| v > 0.0)),
                    _ => ("data.npy", None),
                };
                let f = crate::kload::NpyFile::open(&dir.join(file))?;
                assert_eq!(f.numel(), n_ro * na * nc, "{file}");
                lazy_src = Some(std::sync::Arc::new(crate::kload::LazySource { data: f, sigma: sg, seed: 0x9E37_79B9_7F4A_7C15, sched: lazy_sched.take() }));
                vec![]
            }
            Some(sigma) if dir.join("data_clean.npy").exists() => {
                let mut d = read_npy(&dir.join("data_clean.npy"))?.c32s();
                let n = psi.len();
                let mut l = vec![vec![0.0f64; n]; n];
                for i in 0..n {
                    for j in 0..=i {
                        let s: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
                        l[i][j] = if i == j { (psi[i][i] - s).max(1e-12).sqrt() } else { (psi[i][j] - s) / l[j][j] };
                    }
                }
                let mut seed = 0x9E37_79B9_7F4A_7C15u64;
                let mut gauss = || {
                    let mut u = || { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; ((seed >> 11) as f64 + 0.5) / (1u64 << 53) as f64 };
                    (-2.0 * u().ln()).sqrt() * (2.0 * std::f64::consts::PI * u()).cos()
                };
                let mut white = vec![(0.0f64, 0.0f64); n];
                for s in d.chunks_mut(n) {
                    for w in white.iter_mut() {
                        *w = (gauss() * sigma / 2f64.sqrt(), gauss() * sigma / 2f64.sqrt());
                    }
                    for c in 0..n {
                        let (mut re, mut im) = (0.0, 0.0);
                        for k in 0..=c {
                            re += l[c][k] * white[k].0;
                            im += l[c][k] * white[k].1;
                        }
                        s[c].re += re as f32;
                        s[c].im += im as f32;
                    }
                }
                d
            }
            _ => read_npy(&dir.join("data.npy"))?.c32s(),
        };
        // reconstruction window of the design (simphantom meta.sequence.recon_adc, half-open ADC-index ranges); the
        // remaining ADC samples (e.g. the v2 rewinder) are navigator data
        let adc_keep: Option<Vec<usize>> = match std::env::var("HASTY_RECON_WINDOW").as_deref() {
            Ok("all") => None,
            _ => recon_window(&meta, na_file)?,
        };
        let na = adc_keep.as_ref().map_or(na_file, |k| k.len());
        let smaps_true = read_npy(&dir.join("smaps_true.npy"))?.c32s();
        let field_ppm = read_npy(&dir.join("field_ppm_recon.npy"))?.f32s();
        let truth_vel = read_npy(&dir.join("truth_vel_logical.npy"))?.f32s();
        let truth_frac = read_npy(&dir.join("truth_lumen_fraction.npy"))?.f32s();
        Ok(Self {
            dir: dir.to_path_buf(),
            acq,
            ro,
            sched,
            encode_m1,
            data,
            nc,
            na,
            dims,
            vox: [p.res[0]; 3],
            tdims: dims,
            res: p.res[0],
            psi,
            gnl,
            rot: p.rot_presc,
            fov_c: p.fov_center,
            b0: p.b0,
            venc,
            smaps_true,
            field_ppm,
            truth_vel,
            truth_frac,
            n_frames: p.n_frames,
            meta,
            lazy: lazy_src,
            na_file,
            adc_keep,
            model_traj,
        })
    }

    /// file ADC index of reconstruction sample n
    #[inline]
    pub fn adc_file_index(&self, n: usize) -> usize {
        self.adc_keep.as_ref().map_or(n, |k| k[n])
    }

    /// file ADC indices outside the reconstruction window (navigator samples; empty without a window)
    pub fn adc_navigator(&self) -> Vec<usize> {
        match &self.adc_keep {
            None => vec![],
            Some(k) => {
                let mut keep = vec![false; self.na_file];
                k.iter().for_each(|&i| keep[i] = true);
                (0..self.na_file).filter(|&i| !keep[i]).collect()
            }
        }
    }

    /// the SAMPLE_FLOATS records of readout `ro` at every ADC sample of the file (window ignored), physical axes
    pub fn readout_records_full(&self, ro: usize) -> Vec<f32> {
        let (na, nf) = (self.na_file, SAMPLE_FLOATS);
        if let Some(mt) = &self.model_traj {
            return mt.records(ro);
        }
        match &self.sched {
            None => self.spiral().sample_params(ro..ro + 1),
            Some(sc) if sc.rec.is_empty() => {
                let f = self.lazy.as_ref().and_then(|l| l.sched.as_ref()).expect("lazy sim without its sample records");
                let mut r = vec![0f32; na * nf];
                f.read_f32(ro * na * nf, &mut r).expect("sample records");
                r
            }
            Some(sc) => sc.rec[ro * na * nf..(ro + 1) * na * nf].to_vec(),
        }
    }

    /// voxels of the reconstruction grid
    pub fn nv(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Reconstruct on a finer (or coarser) grid: per-axis scale s_a, dims_a = the smallest 2·3·5·7-smooth size ≥
    /// s_a·n_a (cheap FFTs), voxel size = native FOV / dims_a (the FOV is kept exactly). Truth fields stay on the
    /// native grid and are sampled at the reconstruction voxels' true positions.
    pub fn set_recon_grid(&mut self, scale: [f64; 3]) {
        for a in 0..3 {
            let target = (scale[a] * self.tdims[a] as f64).ceil().max(1.0) as usize;
            let n = (target..).find(|&n| smooth7(n)).unwrap();
            self.dims[a] = n;
            self.vox[a] = self.tdims[a] as f64 * self.res / n as f64;
        }
    }

    /// the reconstruction grid equals the native one
    pub fn native_grid(&self) -> bool {
        self.dims == self.tdims
    }

    /// Ψ^{-1/2}
    pub fn whitener(&self) -> Vec<Vec<f64>> {
        let n = self.nc;
        let m = DMatrix::<f64>::from_fn(n, n, |i, j| self.psi[i][j]);
        let e = SymmetricEigen::new(m);
        let mut w = vec![vec![0.0; n]; n];
        for q in 0..n {
            let s = 1.0 / e.eigenvalues[q].max(1e-12).sqrt();
            for i in 0..n {
                for j in 0..n {
                    w[i][j] += e.eigenvectors[(i, q)] * e.eigenvectors[(j, q)] * s;
                }
            }
        }
        w
    }

    pub fn n_readouts(&self) -> usize {
        self.ro.enc.len()
    }

    pub fn readouts_of(&self, enc: usize, frame: Option<usize>) -> Vec<usize> {
        (0..self.n_readouts()).filter(|&i| self.ro.enc[i] == enc && frame.map_or(true, |f| self.ro.frame[i] == f)).collect()
    }

    /// k (rad/m, logical frame) and the SAMPLE_FLOATS record of readout `ro`
    pub(crate) fn readout_samples(&self, ro: usize) -> (Vec<Vec3>, Vec<f32>) {
        if self.adc_keep.is_some() || self.model_traj.is_some() {
            // reconstruction window / model chain: the records of the kept samples
            let full = self.readout_records_full(ro);
            let nf = SAMPLE_FLOATS;
            let rec: Vec<f32> = match &self.adc_keep {
                Some(k) => k.iter().flat_map(|&i| full[i * nf..(i + 1) * nf].iter().copied()).collect(),
                None => full,
            };
            let rt = transpose(&self.rot);
            let k = (0..self.na).map(|n| mat_vec(&rt, &[rec[n * nf] as f64, rec[n * nf + 1] as f64, rec[n * nf + 2] as f64])).collect();
            return (k, rec);
        }
        match &self.sched {
            None => {
                let a = self.spiral();
                (a.k_logical(ro), a.sample_params(ro..ro + 1))
            }
            Some(sc) => {
                let (na, nf) = (self.na, SAMPLE_FLOATS);
                let rec = if sc.rec.is_empty() {
                    let f = self.lazy.as_ref().and_then(|l| l.sched.as_ref()).expect("lazy sim without its sample records");
                    let mut r = vec![0f32; na * nf];
                    f.read_f32(ro * na * nf, &mut r).expect("sample records");
                    r
                } else {
                    sc.rec[ro * na * nf..(ro + 1) * na * nf].to_vec()
                };
                let rt = transpose(&self.rot);
                let k = (0..na).map(|n| mat_vec(&rt, &[rec[n * nf] as f64, rec[n * nf + 1] as f64, rec[n * nf + 2] as f64])).collect();
                (k, rec)
            }
        }
    }

    pub fn sample_set(&self, ros: &[usize], w: &[Vec<f64>]) -> SampleSet {
        let (na, nc) = (self.na, self.nc);
        let m = ros.len() * na;
        let mut coords = [Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m)];
        let mut y = vec![C32::new(0.0, 0.0); nc * m];
        let mut tau = Vec::with_capacity(m);
        let mut wq = Vec::with_capacity(m);
        for (r, &ro) in ros.iter().enumerate() {
            let (k, sp) = self.readout_samples(ro);
            for n in 0..na {
                let j = r * na + n;
                for a in 0..3 {
                    coords[a].push((k[n][a] * self.vox[a]) as f32);
                }
                let s = &sp[n * SAMPLE_FLOATS..(n + 1) * SAMPLE_FLOATS];
                wq.push([s[9] as f64, s[10] as f64, s[11] as f64, s[12] as f64]);
                tau.push(s[13] as f64);
                let nfile = self.adc_file_index(n);
                let src = &self.data[(ro * self.na_file + nfile) * nc..(ro * self.na_file + nfile + 1) * nc];
                // a known spatially constant phase of the sample (the GIRF's B0 terms on the played trajectory; 0
                // otherwise): the signal carries e^{−iφ0}, removed here
                let (ps, pc) = (s[15] as f64).sin_cos();
                for c in 0..nc {
                    let (mut re, mut im) = (0.0f64, 0.0f64);
                    for c2 in 0..nc {
                        re += w[c][c2] * src[c2].re as f64;
                        im += w[c][c2] * src[c2].im as f64;
                    }
                    y[c * m + j] = C32::new((re * pc - im * ps) as f32, (re * ps + im * pc) as f32);
                }
            }
        }
        SampleSet { readouts: ros.to_vec(), coords, y, tau, wq }
    }

    /// logical centre of reconstruction voxel q
    pub fn voxel_log(&self, q: usize) -> Vec3 {
        let d = self.dims;
        let (i, j, k) = (q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1]));
        [(i as f64 - (d[0] / 2) as f64) * self.vox[0], (j as f64 - (d[1] / 2) as f64) * self.vox[1], (k as f64 - (d[2] / 2) as f64) * self.vox[2]]
    }

    /// trilinear sample of an f32 field on the native (truth) logical grid (components: `comp` stride native nv)
    pub fn sample_log(&self, field: &[f32], comp: usize, x_log: &Vec3) -> f64 {
        let d = self.tdims;
        let nv = d[0] * d[1] * d[2];
        let mut i0 = [0usize; 3];
        let mut fr = [0.0; 3];
        for a in 0..3 {
            let g = (x_log[a] / self.res + (d[a] / 2) as f64).clamp(0.0, (d[a] - 1) as f64 - 1e-9);
            i0[a] = g.floor() as usize;
            fr[a] = g - i0[a] as f64;
        }
        let mut acc = 0.0;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let w = (if dx == 1 { fr[0] } else { 1.0 - fr[0] }) * (if dy == 1 { fr[1] } else { 1.0 - fr[1] }) * (if dz == 1 { fr[2] } else { 1.0 - fr[2] });
                    if w == 0.0 {
                        continue;
                    }
                    let (a, b, c) = ((i0[0] + dx).min(d[0] - 1), (i0[1] + dy).min(d[1] - 1), (i0[2] + dz).min(d[2] - 1));
                    acc += w * field[comp * nv + a + d[0] * (b + d[1] * c)] as f64;
                }
            }
        }
        acc
    }

    /// Mean of an f32 field on the native grid over the reconstruction voxel centred at the true position `x_log`:
    /// trilinear samples at the centres of ⌈vox_a / res⌉ sub-cells per axis (a box filter of the reconstruction
    /// voxel; [`sample_log`](Self::sample_log) itself when the reconstruction voxel is not larger than the native
    /// one). The voxel's extent is taken in logical axes, i.e. the GNL Jacobian across one voxel is neglected.
    pub fn sample_log_box(&self, field: &[f32], comp: usize, x_log: &Vec3) -> f64 {
        let n: [usize; 3] = std::array::from_fn(|a| ((self.vox[a] / self.res) - 1e-9).ceil().max(1.0) as usize);
        if n == [1, 1, 1] {
            return self.sample_log(field, comp, x_log);
        }
        let mut acc = 0.0;
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let o = [i, j, k];
                    let p: Vec3 = std::array::from_fn(|a| x_log[a] + ((o[a] as f64 + 0.5) / n[a] as f64 - 0.5) * self.vox[a]);
                    acc += self.sample_log(field, comp, &p);
                }
            }
        }
        acc / (n[0] * n[1] * n[2]) as f64
    }

    pub fn physics_grid(&self, gnl_aware: bool, w: &[Vec<f64>]) -> PhysicsGrid {
        let nv = self.nv();
        let nc = self.nc;
        let rt = transpose(&self.rot);
        let (mut x_log, mut mono, mut omega, mut jac) = (Vec::with_capacity(nv), Vec::with_capacity(nv), Vec::with_capacity(nv), Vec::with_capacity(nv));
        let mut smaps = vec![C32::new(0.0, 0.0); nc * nv];
        let wfac = 2.0 * std::f64::consts::PI * GAMMA_BAR * self.b0 * 1e-6;
        let (re_t, im_t): (Vec<f32>, Vec<f32>) = self.smaps_true.iter().map(|c| (c.re, c.im)).unzip();
        for q in 0..nv {
            let u_log = self.voxel_log(q);
            let u_s = mat_vec(&self.rot, &u_log);
            let u_s = [u_s[0] + self.fov_c[0], u_s[1] + self.fov_c[1], u_s[2] + self.fov_c[2]];
            let mut x = u_s;
            if gnl_aware {
                for _ in 0..8 {
                    let f = self.gnl.field(&x);
                    let j = self.gnl.jacobian(&x);
                    let r = [f[0] - u_s[0], f[1] - u_s[1], f[2] - u_s[2]];
                    let ji = inv3(&j);
                    let dx = mat_vec(&ji, &r);
                    for a in 0..3 {
                        x[a] -= dx[a];
                    }
                }
            }
            let xl = mat_vec(&rt, &[x[0] - self.fov_c[0], x[1] - self.fov_c[1], x[2] - self.fov_c[2]]);
            mono.push([x[2] * x[2], x[0] * x[0] + x[1] * x[1], x[0] * x[2], x[1] * x[2]]);
            omega.push(wfac * self.sample_log(&self.field_ppm, 0, &xl));
            jac.push(self.gnl.jacobian(&x));
            let s: Vec<(f64, f64)> = (0..nc).map(|c| (self.sample_log(&re_t, c, &xl), self.sample_log(&im_t, c, &xl))).collect();
            for c in 0..nc {
                let (mut re, mut im) = (0.0, 0.0);
                for c2 in 0..nc {
                    re += w[c][c2] * s[c2].0;
                    im += w[c][c2] * s[c2].1;
                }
                smaps[c * nv + q] = C32::new(re as f32, im as f32);
            }
            x_log.push(xl);
        }
        PhysicsGrid { x_log, mono, omega, jac, smaps }
    }
}

/// the reconstruction window of a sim (`meta.sequence.recon_adc`: half-open ADC-index ranges, simphantom) as the
/// kept file ADC indices; None when absent or covering every sample
pub fn recon_window(meta: &Value, na_file: usize) -> Result<Option<Vec<usize>>, Error> {
    let Some(r) = meta["sequence"]["recon_adc"].as_array() else { return Ok(None) };
    let mut keep = vec![];
    for q in r {
        let (lo, hi) = (q[0].as_u64().unwrap_or(0) as usize, q[1].as_u64().unwrap_or(0) as usize);
        if lo > hi || hi > na_file || keep.last().is_some_and(|&l| lo <= l) {
            return Err(Error::Format(format!("meta.sequence.recon_adc {r:?}: ranges must be ascending, disjoint and inside the {na_file} ADC samples")));
        }
        keep.extend(lo..hi);
    }
    Ok((keep.len() < na_file).then_some(keep))
}

/// only prime factors 2, 3, 5, 7
pub fn smooth7(mut n: usize) -> bool {
    for p in [2, 3, 5, 7] {
        while n % p == 0 && n > 1 {
            n /= p;
        }
    }
    n == 1
}

pub fn inv3(m: &Mat3) -> Mat3 {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    [
        [c(1, 1, 2, 2) / det, -c(0, 1, 2, 2) / det, c(0, 1, 1, 2) / det],
        [-c(1, 0, 2, 2) / det, c(0, 0, 2, 2) / det, -c(0, 0, 1, 2) / det],
        [c(1, 0, 2, 1) / det, -c(0, 0, 2, 1) / det, c(0, 0, 1, 1) / det],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy(path: &Path, descr: &str, shape: &[usize], bytes: &[u8]) {
        let hdr = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({}{}), }}", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "), if shape.len() == 1 { "," } else { "" });
        let mut h = hdr.into_bytes();
        while (10 + h.len() + 1) % 64 != 0 {
            h.push(b' ');
        }
        h.push(b'\n');
        let mut out = vec![0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0];
        out.extend_from_slice(&(h.len() as u16).to_le_bytes());
        out.extend_from_slice(&h);
        out.extend_from_slice(bytes);
        std::fs::write(path, out).unwrap();
    }
    fn f4(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn schedule_based_flow_sim_loads() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp/simdata_flow");
        std::fs::create_dir_all(&dir).unwrap();
        let (n_ro, na, nc, d) = (4usize, 3usize, 2usize, 4usize);
        let nv = d * d * d;
        // logical → scanner: 90° about z (logical x = scanner y)
        let rp = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let meta = serde_json::json!({
            "b0": 2.89, "gmax": 0.08, "smax": 200.0, "dt": 1e-5, "dwell": 2e-6, "fov": [0.2, 0.2, 0.2], "res": [0.05, 0.05, 0.05],
            "n_int": 1, "fov_edge_ratio": 1.0, "venc": 1.5, "encoding": "Referenced", "coverage": "Rotated3d", "t_rf": 1.5e-4,
            "t_spoil": 1e-3, "scan_time": 0.04, "rot_presc": rp, "fov_center_scanner": [0.0, 0.0, 0.0], "mean_rr": 1.0,
            "rr_rel_std": 0.0, "n_frames": 2, "seed": 1, "n_coils": nc, "n_adc": na, "n_readouts": n_ro, "recon_dims": [d, d, d],
            "psi": [[1.0, 0.0], [0.0, 1.0]], "gnl": {"r0": 0.25, "transverse": [0.0, 0.0], "longitudinal": [0.0, 0.0]},
            "noise_sigma": 0.0, "sequence": {"kind": "flow", "encoding": {"venc_lo": 1.0}},
        });
        std::fs::write(dir.join("meta.json"), serde_json::to_string(&meta).unwrap()).unwrap();
        let tab: Vec<u8> = (0..n_ro).flat_map(|i| {
            let mut r = [0.0f64; 14];
            r[0] = i as f64 * 0.01;
            r[1] = (i % 2) as f64;
            r[3] = (i / 2) as f64;
            r.into_iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()
        }).collect();
        npy(&dir.join("readouts.npy"), "<f8", &[n_ro, 14], &tab);
        let (m1z, phi) = (1e-8f64, 0.3f32);
        let mut rec = vec![0f32; n_ro * na * SAMPLE_FLOATS];
        for i in 0..n_ro {
            for n in 0..na {
                let r = &mut rec[(i * na + n) * SAMPLE_FLOATS..(i * na + n + 1) * SAMPLE_FLOATS];
                r[0] = 10.0 * (n + 1) as f32;
                r[1] = 20.0 * (n + 1) as f32;
                r[2] = 30.0 * (n + 1) as f32;
                r[5] = (if i % 2 == 0 { 1.0 } else { -1.0 } * GAMMA * m1z) as f32;
                r[13] = 1e-3 + n as f32 * 1e-5;
                r[15] = phi;
            }
        }
        npy(&dir.join("sched_nominal.npy"), "<f4", &[n_ro, na, SAMPLE_FLOATS], &f4(&rec));
        // data: 1 · e^{−iφ0} on every coil
        let data: Vec<f32> = (0..n_ro * na * nc).flat_map(|_| [phi.cos(), -phi.sin()]).collect();
        npy(&dir.join("data.npy"), "<c8", &[n_ro, na, nc], &f4(&data));
        npy(&dir.join("smaps_true.npy"), "<c8", &[nc, d, d, d], &f4(&vec![0.0; 2 * nc * nv]));
        npy(&dir.join("field_ppm_recon.npy"), "<f4", &[d, d, d], &f4(&vec![0.0; nv]));
        npy(&dir.join("truth_vel_logical.npy"), "<f4", &[6 * nv], &f4(&vec![0.0; 6 * nv]));
        npy(&dir.join("truth_lumen_fraction.npy"), "<f4", &[nv], &f4(&vec![0.0; nv]));

        let sd = SimData::load_traj(&dir, "nominal").unwrap();
        assert!(sd.acq.is_none() && sd.n_readouts() == n_ro && sd.venc == 1.0);
        assert_eq!(sd.readouts_of(1, None), vec![1, 3]);
        assert_eq!(sd.readouts_of(0, Some(1)), vec![2]);
        assert!((sd.encode_m1[0][2] - m1z).abs() < 1e-14 && (sd.encode_m1[1][2] + m1z).abs() < 1e-14, "{:?}", sd.encode_m1);
        let w = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let set = sd.sample_set(&[1], &w);
        for n in 0..na {
            // logical = Rpᵀ physical: logical x = scanner y, logical y = −scanner x
            let want = [20.0 * (n + 1) as f64, -10.0 * (n + 1) as f64, 30.0 * (n + 1) as f64].map(|k| k * 0.05);
            for a in 0..3 {
                assert!((set.coords[a][n] as f64 - want[a]).abs() < 1e-5, "sample {n} axis {a}: {} vs {}", set.coords[a][n], want[a]);
            }
            for c in 0..nc {
                let y = set.y[c * na + n];
                assert!((y.re - 1.0).abs() < 1e-6 && y.im.abs() < 1e-6, "B0-term phase not removed: {y:?}");
            }
        }
        assert!((set.tau[1] - (1e-3 + 1e-5)).abs() < 1e-9);
        assert!(SimData::load_traj(&dir, "played").is_err());
        assert_eq!((sd.na, sd.na_file), (na, na));
        assert!(sd.adc_keep.is_none() && sd.adc_navigator().is_empty());

        // a reconstruction window (simphantom meta.sequence.recon_adc): ADC samples 0 and 1 kept, 2 = navigator
        let mut meta_w = meta.clone();
        meta_w["sequence"]["recon_adc"] = serde_json::json!([[0, 2]]);
        std::fs::write(dir.join("meta.json"), serde_json::to_string(&meta_w).unwrap()).unwrap();
        // data: sample n carries the value n + 1 (navigator sample = 3)
        let data: Vec<f32> = (0..n_ro * na * nc).flat_map(|i| { let n = (i / nc) % na; let z = num_complex::Complex32::from_polar(1.0 + n as f32, -phi); [z.re, z.im] }).collect();
        npy(&dir.join("data.npy"), "<c8", &[n_ro, na, nc], &f4(&data));
        let sw = SimData::load_traj(&dir, "nominal").unwrap();
        assert_eq!((sw.na, sw.na_file), (2, na));
        assert_eq!(sw.adc_keep.as_deref(), Some(&[0usize, 1][..]));
        assert_eq!(sw.adc_navigator(), vec![2]);
        let set = sw.sample_set(&[1, 3], &w);
        assert_eq!(set.tau.len(), 4);
        for (j, n) in [0usize, 1, 0, 1].into_iter().enumerate() {
            assert!((set.tau[j] - (1e-3 + n as f64 * 1e-5)).abs() < 1e-9, "tau {j}");
            assert!((set.coords[2][j] as f64 - 30.0 * (n + 1) as f64 * 0.05).abs() < 1e-5);
            for c in 0..nc {
                let y = set.y[c * 4 + j];
                assert!((y.re - (1.0 + n as f32)).abs() < 1e-5 && y.im.abs() < 1e-5, "sample {j}: {y:?}");
            }
        }
        // raw data of the window and of the whole ADC (the navigator stays reachable)
        let mut raw = vec![C32::new(0.0, 0.0); 2 * nc];
        sw.readout_raw(0, &mut raw).unwrap();
        assert!(((raw[nc].re.powi(2) + raw[nc].im.powi(2)).sqrt() - 2.0).abs() < 1e-5);
        let mut full = vec![C32::new(0.0, 0.0); na * nc];
        sw.readout_raw_full(0, &mut full).unwrap();
        assert!(((full[2 * nc].re.powi(2) + full[2 * nc].im.powi(2)).sqrt() - 3.0).abs() < 1e-5);
        assert_eq!(sw.readout_records_full(0).len(), na * SAMPLE_FLOATS);
        // the streamed loader (kload) sees the same window, noise counters by file index
        let sl = SimData::load_traj_opts(&dir, "nominal", true).unwrap();
        assert_eq!(sl.na, 2);
        let ls = sl.load_set(&[1, 3], &crate::kload::LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
        assert_eq!(ls.tau.len(), 4);
        for j in 0..4 {
            for c in 0..nc {
                assert!((ls.y[c * 4 + j].re - set.y[c * 4 + j].re).abs() + (ls.y[c * 4 + j].im - set.y[c * 4 + j].im).abs() < 1e-5, "kload vs sample_set {j}");
            }
        }
        // HASTY_RECON_WINDOW=all is read by the loader: the window can be disabled (checked through recon_window)
        assert_eq!(recon_window(&meta_w, na).unwrap(), Some(vec![0, 1]));
        let mut meta_all = meta.clone();
        meta_all["sequence"]["recon_adc"] = serde_json::json!([[0, na]]);
        assert_eq!(recon_window(&meta_all, na).unwrap(), None);
        meta_all["sequence"]["recon_adc"] = serde_json::json!([[0, na + 1]]);
        assert!(recon_window(&meta_all, na).is_err());
        std::fs::write(dir.join("meta.json"), serde_json::to_string(&meta).unwrap()).unwrap();
    }

    /// end-to-end on a real simulator output (`HASTY_TEST_SIM=<scan dir>` of `simphantom --seq flow --flow-run
    /// flow_thorax_v2 --girf <chain> --girf-hist N`): the reconstruction window from the meta, the navigator reachable,
    /// and `girf:<chain>` with the same history reproducing the simulator's sched_played.npy; `cargo test -- --ignored`
    #[test]
    #[ignore]
    fn real_sim_window_and_model_trajectory() {
        let Ok(dir) = std::env::var("HASTY_TEST_SIM") else { return };
        let dir = Path::new(&dir);
        let meta: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
        let chain = meta["sequence"]["girf"].as_str().unwrap().to_string();
        let hist = meta["sequence"]["girf_history"].as_u64().unwrap_or(1);
        std::env::set_var("HASTY_GIRF_HIST", hist.to_string());
        let pl = SimData::load_traj_opts(dir, "played", true).unwrap();
        let md = SimData::load_traj_opts(dir, &format!("girf:{chain}"), true).unwrap();
        eprintln!("{}: n_adc {} → window {} samples, navigator {}", dir.display(), pl.na_file, pl.na, pl.adc_navigator().len());
        assert!(pl.na < pl.na_file && pl.na + pl.adc_navigator().len() == pl.na_file);
        assert_eq!((md.na, md.na_file), (pl.na, pl.na_file));
        let mut worst = 0.0f32;
        for ro in [0usize, 1, pl.n_readouts() / 2, pl.n_readouts() - 1] {
            let (a, b) = (pl.readout_records_full(ro), md.readout_records_full(ro));
            worst = worst.max(a.iter().zip(&b).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max));
        }
        eprintln!("model trajectory girf:{chain} (history {hist}) vs sched_played.npy: max |Δ| {worst:.2e}");
        assert!(worst == 0.0, "the reconstruction's model records differ from the simulator's: {worst}");
        let w: Vec<Vec<f64>> = pl.whitener();
        let set = pl.load_set(&[0, 1], &crate::kload::LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
        assert_eq!(set.tau.len(), 2 * pl.na);
        assert!(set.tau[pl.na - 1] < set.tau[pl.na - 2] + 1e-5 + 3.0 * (set.tau[1] - set.tau[0]));
    }
}

/// The time a group of velocity encodes' velocity estimate refers to, relative to the time origin of `t` (s).
///
/// Encode e (first moment m1_e) is played at t_e. With u(t) = u0 + u'·t the phase differences to encode 0 are
/// (m1_e − m1_0)·u0 + (m1_e t_e − m1_0 t_0)·u' (× γ), so the least-squares velocity is u0 + T u' with
/// T = pinv(ΔM) D, rows ΔM_e = m1_e − m1_0, D_e = m1_e t_e − m1_0 t_0. Returns the mean of T's diagonal over the
/// encoded axes (the off-diagonal is acceleration cross-talk between axes, a property of the encode order). When encode
/// 0 carries a first moment (e.g. the referenced scheme with m1_0 = −venc-moment on every axis), this differs from the
/// group's middle readout: stamping a group of encodes into one cardiac frame by its middle readout shifts the
/// velocities in time.
pub fn velocity_time(m1: &[Vec3], t: &[f64]) -> f64 {
    assert_eq!(m1.len(), t.len());
    let n = m1.len();
    if n < 2 {
        return t.first().copied().unwrap_or(0.0);
    }
    let dm = DMatrix::from_fn(n - 1, 3, |e, a| m1[e + 1][a] - m1[0][a]);
    let d = DMatrix::from_fn(n - 1, 3, |e, a| m1[e + 1][a] * t[e + 1] - m1[0][a] * t[0]);
    let a = dm.transpose() * &dm;
    let scale = a.trace().max(f64::MIN_POSITIVE);
    let axes: Vec<usize> = (0..3).filter(|&i| a[(i, i)] > 1e-9 * scale).collect();
    if axes.is_empty() {
        return t.iter().sum::<f64>() / n as f64;
    }
    // restricted to the encoded axes (a 1-D or 2-D encoding leaves the others out)
    let k = axes.len();
    let dmr = DMatrix::from_fn(n - 1, k, |e, j| dm[(e, axes[j])]);
    let dr = DMatrix::from_fn(n - 1, k, |e, j| d[(e, axes[j])]);
    let tm = (dmr.transpose() * &dmr).try_inverse().expect("velocity_time: encodes do not span their axes") * (dmr.transpose() * dr);
    tm.trace() / k as f64
}

#[cfg(test)]
mod velocity_time_tests {
    use super::velocity_time;

    #[test]
    fn velocity_time_cases() {
        let tr = 0.01;
        // reference without first moment: velocity of each axis at its own encode; mean over the 3 axes
        let m1 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let t: Vec<f64> = (0..4).map(|e| e as f64 * tr).collect();
        assert!((velocity_time(&m1, &t) - 2.0 * tr).abs() < 1e-12);
        // balanced two-sided pair per axis (one axis): midpoint
        let m1 = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!((velocity_time(&m1, &[0.0, tr]) - 0.5 * tr).abs() < 1e-12);
        // the 7-point referenced scheme of the flow_thorax sequence (moments in venc units): 1.346 TR after encode 0
        let (p, q) = (1.0, 1.0 / 3.0);
        let m1 = [[-p, -p, -p], [p, -p, -p], [q, -p, -p], [-p, p, -p], [-p, q, -p], [-p, -p, p], [-p, -p, q]];
        let t: Vec<f64> = (0..7).map(|e| e as f64 * tr).collect();
        assert!((velocity_time(&m1, &t) / tr - 1.346_153_846).abs() < 1e-6, "{}", velocity_time(&m1, &t) / tr);
    }
}
