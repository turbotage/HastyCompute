//! Bloch ensemble simulator for the flow sequence: a small parcel of spins (isochromats) moves along given paths while
//! the designed TRs are played (hard-pulse RF with RF-spoiling phase, the physical gradient waveforms of every encode and
//! rotation, T1 / T2 relaxation, off-resonance, concomitant Maxwell fields). Everything is integrated per raster step by
//! the exact rotation about the effective field (Rodrigues), relaxation split off per step. No diffusion.
//!
//! Used by the server command `bloch_parcel` (plotting_website/bloch): which plays a parcel from a relaxed state
//! (inflow from the vena cava) or from a pre-rolled steady state, and records the ensemble at chosen raster steps.
//!
//! Conventions: physical axes, SI, positions in m from the isocentre. Precession is left-handed (dM/dt = γ M × B, a
//! positive field turns M_xy clockwise) unless `left_handed` is off, which flips the transverse rotation sense to the
//! one the recon's signal model uses (phase +γ M0·x). The receiver reference follows the RF phase (demodulated echo).

use std::path::Path;

use rayon::prelude::*;

use crate::traj::{acquisition_order, rotations, RotSet};
use crate::{mat_vec, Mat3, Vec3, GAMMA};

// ───────────────────────────────────────────────────────────── npy (f8 / f4, C order)

/// shape and values of a little-endian float `.npy` file
pub fn read_npy_f64(path: &Path) -> Result<(Vec<usize>, Vec<f64>), String> {
    let b = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(format!("{}: not an npy file", path.display()));
    }
    let (hl, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let h = std::str::from_utf8(&b[off..off + hl]).map_err(|e| e.to_string())?;
    if h.contains("'fortran_order': True") {
        return Err(format!("{}: Fortran order", path.display()));
    }
    let f8 = h.contains("<f8");
    if !f8 && !h.contains("<f4") {
        return Err(format!("{}: only <f8 / <f4 supported ({h})", path.display()));
    }
    let s0 = h.find("'shape': (").ok_or("npy: no shape")? + 10;
    let s1 = h[s0..].find(')').ok_or("npy: bad shape")? + s0;
    let shape: Vec<usize> = h[s0..s1].split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().map_err(|_| "npy: bad shape")).collect::<Result<_, _>>()?;
    let n: usize = shape.iter().product();
    let d = &b[off + hl..];
    let data: Vec<f64> = if f8 {
        if d.len() < 8 * n {
            return Err("npy: short data".into());
        }
        (0..n).map(|i| f64::from_le_bytes(d[8 * i..8 * i + 8].try_into().unwrap())).collect()
    } else {
        if d.len() < 4 * n {
            return Err("npy: short data".into());
        }
        (0..n).map(|i| f32::from_le_bytes(d[4 * i..4 * i + 4].try_into().unwrap()) as f64).collect()
    };
    Ok((shape, data))
}

// ───────────────────────────────────────────────────────────── the designed TR

/// one designed flow TR (a `seqdesign` run directory: `final.json`, `summary.json`, `tr_grad.npy`, `spoke_grad.npy`):
/// the encode and tail blocks on physical axes per encode, the readout the base spoke rotated per rotation
#[derive(Clone, Debug)]
pub struct FlowDesign {
    pub name: String,
    /// gradient raster, s
    pub dt: f64,
    /// RF centre after the TR start, s
    pub t_rf: f64,
    /// hard-pulse duration, s
    pub rf_dur: f64,
    /// raster samples of the TR (steps; `n_tr + 1` corner samples per waveform)
    pub n_tr: usize,
    pub n_enc: usize,
    /// raster index of the readout start (the echo: k = 0) and of the last ADC sample
    pub i_ro: usize,
    pub i_adc: usize,
    /// raster index where the encode block / the tail block starts
    pub i_enc: usize,
    pub i_tail: usize,
    /// [encode][n_tr + 1] physical TR (the encode and tail parts are used)
    pub tr_grad: Vec<Vec<Vec3>>,
    /// base readout, corner samples 0..=i_adc
    pub base: Vec<Vec3>,
    /// B1 envelope per raster step of the RF window (`rf_steps`), normalised to Σ env · dt = 1 (flip · env = ω1, rad/s):
    /// a hard pulse is constant 1 / rf_dur
    pub rf_env: Vec<f64>,
    /// slab selection: axis, slice-select gradient (T/m) and thickness (m); None = non-selective hard pulse
    pub slab: Option<SlabGeom>,
}

#[derive(Clone, Copy, Debug)]
pub struct SlabGeom {
    pub axis: usize,
    pub g_ss: f64,
    pub thickness: f64,
    pub tbw: f64,
}

/// slab-selective excitation replacing the design's hard pulse: windowed-sinc (Hamming) B1 over `dur` with the
/// slice-select gradient `bandwidth / (γ̄ · thickness)` along `axis`, a rephasing lobe of half its area on that axis
/// laid onto the start of the encode block (the encode block itself is unchanged; it starts `gap` after the pulse)
#[derive(Clone, Debug)]
pub struct SlabSpec {
    pub thickness: f64,
    pub tbw: f64,
    pub dur: f64,
    pub axis: usize,
    /// dead time before the pulse and gap between the pulse and the encode block, s
    pub dead: f64,
    pub gap: f64,
    /// limits of the rephasing lobe
    pub gmax: f64,
    pub smax: f64,
}

impl SlabSpec {
    /// the simulator's defaults (`simphantom`: TBW 6, 0.8 ms) along z
    pub fn thorax(thickness: f64) -> Self {
        Self { thickness, tbw: 6.0, dur: 0.8e-3, axis: 2, dead: 100e-6, gap: 30e-6, gmax: 0.04, smax: 100.0 }
    }
}

/// windowed-sinc (Hamming) envelope at the centres of `n` steps of `dt`, Σ env · dt = 1
fn sinc_env(n: usize, tbw: f64, dt: f64) -> Vec<f64> {
    let mut env: Vec<f64> = (0..n)
        .map(|k| {
            let x = (k as f64 + 0.5) / n as f64 * 2.0 - 1.0;
            let t = x * tbw / 2.0;
            let sinc = if t.abs() < 1e-12 { 1.0 } else { (std::f64::consts::PI * t).sin() / (std::f64::consts::PI * t) };
            sinc * (0.54 + 0.46 * (std::f64::consts::PI * x).cos())
        })
        .collect();
    let area: f64 = env.iter().sum::<f64>() * dt;
    env.iter_mut().for_each(|v| *v /= area);
    env
}

/// linear interpolation of corner samples to `k` times as many steps
fn upsample(v: &[Vec3], k: usize) -> Vec<Vec3> {
    if v.len() < 2 {
        return v.to_vec();
    }
    let mut o = Vec::with_capacity((v.len() - 1) * k + 1);
    for w in v.windows(2) {
        for j in 0..k {
            let f = j as f64 / k as f64;
            o.push([0, 1, 2].map(|a| w[0][a] + f * (w[1][a] - w[0][a])));
        }
    }
    o.push(*v.last().unwrap());
    o
}

impl FlowDesign {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let rj = |f: &str| -> Result<serde_json::Value, String> {
            serde_json::from_str(&std::fs::read_to_string(dir.join(f)).map_err(|e| format!("{}: {e}", dir.join(f).display()))?).map_err(|e| format!("{f}: {e}"))
        };
        let (fin, summ) = (rj("final.json")?, rj("summary.json")?);
        let b = &summ["best"];
        let u = |k: &str| b[k].as_u64().ok_or(format!("summary.json best.{k}")).map(|x| x as usize);
        let (i_ro, i_adc, n_tr, i_enc, i_tail) = (u("i_ro")?, u("i_adc")?, u("n_tr")?, u("i_enc")?, u("i_tail")?);
        let n_enc = fin["n_enc"].as_u64().ok_or("final.json n_enc")? as usize;
        let dt = fin["dt"].as_f64().ok_or("final.json dt")?;
        let t_rf = fin["t_rf"].as_f64().ok_or("final.json t_rf")?;
        // the RF end after the TR start: the navigator rules' value, else a symmetric hard pulse after 100 µs dead time
        let rf_end = b["nav_adc"]["rf_end"].as_f64().unwrap_or(2.0 * t_rf - 100e-6);
        let rf_dur = 2.0 * (rf_end - t_rf);
        if rf_dur <= 0.0 {
            return Err(format!("bad RF timing: t_rf {t_rf}, rf_end {rf_end}"));
        }
        let (sh, tg) = read_npy_f64(&dir.join("tr_grad.npy"))?;
        if sh.len() != 3 || sh[0] != n_enc || sh[2] != 3 || sh[1] < n_tr {
            return Err(format!("tr_grad.npy shape {sh:?} does not fit n_enc {n_enc}, n_tr {n_tr}"));
        }
        let ntr1 = sh[1];
        let tr_grad = (0..n_enc).map(|e| (0..ntr1).map(|i| [tg[(e * ntr1 + i) * 3], tg[(e * ntr1 + i) * 3 + 1], tg[(e * ntr1 + i) * 3 + 2]]).collect()).collect();
        let (sh, sg) = read_npy_f64(&dir.join("spoke_grad.npy"))?;
        if sh.len() != 2 || sh[0] <= i_adc || sh[1] != 3 {
            return Err(format!("spoke_grad.npy shape {sh:?} shorter than the ADC window {i_adc}"));
        }
        let base = (0..=i_adc).map(|i| [sg[3 * i], sg[3 * i + 1], sg[3 * i + 2]]).collect();
        let name = dir.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let nrf = ((rf_dur / dt).round() as usize).max(1);
        Ok(Self { name, dt, t_rf, rf_dur, n_tr, n_enc, i_ro, i_adc, i_enc, i_tail, tr_grad, base, rf_env: vec![1.0 / rf_dur; nrf], slab: None })
    }

    /// TR length, s
    pub fn tr(&self) -> f64 {
        self.n_tr as f64 * self.dt
    }
    /// echo time (k = 0 at the readout start) after the RF centre, s
    pub fn te(&self) -> f64 {
        self.i_ro as f64 * self.dt - self.t_rf
    }
    /// corner samples (n_tr + 1) of encode `e` with the readout rotated by `r`
    pub fn tr_wave(&self, e: usize, r: &Mat3) -> Vec<Vec3> {
        let mut g: Vec<Vec3> = self.tr_grad[e][..=self.n_tr.min(self.tr_grad[e].len() - 1)].to_vec();
        g.resize(self.n_tr + 1, [0.0; 3]);
        for (i, v) in self.base.iter().enumerate() {
            g[self.i_ro + i] = mat_vec(r, v);
        }
        g
    }
    /// the design with a slab-selective pulse instead of the hard pulse (see [`SlabSpec`]): the TR is longer by the
    /// difference between the new RF block and the old one, TE (from the new RF centre) changes with it
    pub fn with_slab(&self, sp: &SlabSpec) -> FlowDesign {
        let dt = self.dt;
        let (n_dead, n_rf, n_gap) = ((sp.dead / dt).round() as usize, ((sp.dur / dt).round() as usize).max(2), (sp.gap / dt).round() as usize);
        let p = n_dead + n_rf + n_gap;
        let g_ss = sp.tbw / sp.dur / (crate::GAMMA_BAR * sp.thickness);
        let env = sinc_env(n_rf, sp.tbw, dt);
        // rephasing lobe: cancels the moment from the pulse centre to the end of the gradient (plateau + half-step ramps)
        let reph = crate::wave::trapezoid(-0.5 * g_ss * (n_rf as f64 + 1.0) * dt, sp.gmax, sp.smax, dt);
        let delta = p as isize - self.i_enc as isize;
        let mk = |old: &Vec<Vec3>| -> Vec<Vec3> {
            let tail = &old[self.i_enc.min(old.len())..=self.n_tr.min(old.len() - 1)];
            let mut g = vec![[0.0; 3]; p + tail.len()];
            for v in g.iter_mut().take(n_dead + n_rf + 1).skip(n_dead) {
                v[sp.axis] = g_ss;
            }
            g[p..].copy_from_slice(tail);
            for (i, r) in reph.iter().enumerate() {
                if p + i < g.len() {
                    g[p + i][sp.axis] += r;
                }
            }
            g
        };
        let sh = |x: usize| (x as isize + delta) as usize;
        FlowDesign {
            name: format!("{} + slab", self.name),
            dt,
            t_rf: (n_dead as f64 + 0.5 * n_rf as f64) * dt,
            rf_dur: n_rf as f64 * dt,
            n_tr: sh(self.n_tr),
            n_enc: self.n_enc,
            i_ro: sh(self.i_ro),
            i_adc: self.i_adc,
            i_enc: p,
            i_tail: sh(self.i_tail),
            tr_grad: self.tr_grad.iter().map(mk).collect(),
            base: self.base.clone(),
            rf_env: env,
            slab: Some(SlabGeom { axis: sp.axis, g_ss, thickness: sp.thickness, tbw: sp.tbw }),
        }
    }

    /// the same design on a raster `k` times finer (gradients linearly interpolated between the corner samples, so the
    /// waveform is the same piecewise-linear one; a slab pulse's envelope is re-sampled, a hard pulse stays constant).
    /// Do this after `with_slab`, so that the gradient ramps keep the length they were designed with.
    pub fn refine(&self, k: usize) -> FlowDesign {
        if k <= 1 {
            return self.clone();
        }
        let dt = self.dt / k as f64;
        let n_rf = self.rf_env.len() * k;
        let rf_env = match self.slab {
            Some(g) => sinc_env(n_rf, g.tbw, dt),
            None => vec![1.0 / self.rf_dur; n_rf],
        };
        FlowDesign {
            name: self.name.clone(),
            dt,
            t_rf: self.t_rf,
            rf_dur: self.rf_dur,
            n_tr: self.n_tr * k,
            n_enc: self.n_enc,
            i_ro: self.i_ro * k,
            i_adc: self.i_adc * k,
            i_enc: self.i_enc * k,
            i_tail: self.i_tail * k,
            tr_grad: self.tr_grad.iter().map(|g| upsample(g, k)).collect(),
            base: upsample(&self.base, k),
            rf_env,
            slab: self.slab,
        }
    }

    /// raster step of the readout (from `i_ro`) where |k| is largest: the readout goes out to k_max until there and
    /// returns afterwards (the ADC window covers both)
    pub fn i_kmax(&self) -> usize {
        let (mut k, mut best, mut at) = ([0.0f64; 3], 0.0f64, 0usize);
        for i in 1..self.base.len() {
            for a in 0..3 {
                k[a] += 0.5 * (self.base[i - 1][a] + self.base[i][a]) * self.dt;
            }
            let n = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
            if n > best {
                best = n;
                at = i;
            }
        }
        at
    }

    /// raster steps [a, b) holding the pulse
    pub fn rf_steps(&self) -> (usize, usize) {
        let rf_end = self.t_rf + 0.5 * self.rf_dur;
        let b = (rf_end / self.dt).round() as usize;
        let a = b - self.rf_env.len();
        (a, b)
    }
}

// ───────────────────────────────────────────────────────────── simulation

#[derive(Clone, Copy, Debug)]
pub struct Tissue {
    pub t1: f64,
    pub t2: f64,
}

#[derive(Clone, Debug)]
pub struct BlochCfg {
    /// flip angle, rad (nominal; × `b1_scale`)
    pub flip: f64,
    /// quadratic RF-spoiling phase increment, rad (φ_n = inc · n(n+1)/2, n counting from the first pre-roll TR)
    pub rf_inc: f64,
    pub b1_scale: f64,
    /// slab centre along the slab axis, m from the isocentre (transmit frequency offset of a slab design)
    pub slab_center: f64,
    pub tissue: Tissue,
    /// static field, T (concomitant fields)
    pub b0: f64,
    /// constant off-resonance, Hz
    pub df: f64,
    pub concomitant: bool,
    pub left_handed: bool,
    /// TRs played with the parcel flowing (output) and TRs played before with the spins at rest at their start
    /// positions (steady state); the pre-roll continues the encode / rotation / RF-phase sequence
    pub n_tr: usize,
    pub preroll: usize,
    /// rotations of the scan (super-Fibonacci, acquisition order); the TR counter wraps around them
    pub n_rot: usize,
    /// initial M_z / M0 (1 = fully relaxed)
    pub mz0: f64,
    /// output samples per TR (evenly spaced raster steps plus the marks: RF end, echo, ADC end, tail start, TR end)
    pub per_tr: usize,
    /// TR range [a, b) recorded every `stride` raster steps instead
    pub detail: Option<(usize, usize, usize)>,
    /// the mean magnetisation is recorded every this many raster steps (all TRs)
    pub mean_stride: usize,
}

impl Default for BlochCfg {
    fn default() -> Self {
        Self {
            flip: 15f64.to_radians(),
            rf_inc: 117f64.to_radians(),
            b1_scale: 1.0,
            slab_center: 0.0,
            tissue: Tissue { t1: 1.7, t2: 0.2 },
            b0: 3.0,
            df: 0.0,
            concomitant: true,
            left_handed: true,
            n_tr: 200,
            preroll: 0,
            n_rot: 6000,
            mz0: 1.0,
            per_tr: 32,
            detail: None,
            mean_stride: 10,
        }
    }
}

/// what the run recorded
#[derive(Clone, Debug, Default)]
pub struct BlochOut {
    /// output sample k: TR index (0 = the first parcel TR) and raster step within it (state after that many steps)
    pub tr: Vec<u32>,
    pub step: Vec<u32>,
    /// [n_out][N][3] magnetisation (Mx, My, Mz) / M0, in the lab-frame rotating with the RF carrier
    pub m: Vec<f32>,
    /// [n_out][N][3] positions, m
    pub pos: Vec<f32>,
    /// [n_mean][3] ensemble mean at steps s·mean_stride of every TR (n_mean = n_tr · ceil(n_steps / stride))
    pub mean: Vec<f32>,
    /// [n_mean] mean over the spins of the per-spin |M⊥| (no dephasing loss: separates spread from saturation)
    pub mean_inc: Vec<f32>,
    /// per TR at the echo step: ⟨|M⊥|⟩, ⟨Mz⟩, ⟨|M|⟩ over the spins
    pub echo_inc: Vec<[f64; 3]>,
    /// per TR the ensemble mean demodulated echo (re, im) at the echo step, and the RF phase of the TR
    pub echo: Vec<[f64; 2]>,
    pub rf_phase: Vec<f64>,
    /// per TR: encode index and rotation index of the played TR
    pub encode: Vec<u32>,
    pub rot: Vec<u32>,
    /// gradient waveform actually played during the parcel TRs, every `g_stride` steps: [n_tr][n_g][3] T/m
    pub g: Vec<f32>,
    pub g_stride: usize,
    pub n_spins: usize,
}

/// ideal RF-spoiled steady-state echo magnitude (Ernst, relaxation to the echo included)
pub fn ernst(flip: f64, tr: f64, te_after_rf: f64, t: &Tissue) -> f64 {
    let e1 = (-tr / t.t1).exp();
    flip.sin() * (1.0 - e1) / (1.0 - e1 * flip.cos()) * (-te_after_rf / t.t2).exp()
}

/// output steps of one TR (sorted, unique) from `per_tr` and the marks
fn out_steps(d: &FlowDesign, per_tr: usize, stride: Option<usize>) -> Vec<usize> {
    let n = d.n_tr;
    let mut s: Vec<usize> = match stride {
        Some(k) => (0..=n).step_by(k.max(1)).collect(),
        None => (0..per_tr.max(2)).map(|i| (i * n) / (per_tr.max(2) - 1)).collect(),
    };
    let (_, rf_b) = d.rf_steps();
    s.extend([rf_b, d.i_ro, d.i_ro + d.i_adc, d.i_tail, n]);
    s.retain(|&x| x <= n);
    s.sort_unstable();
    s.dedup();
    s
}

fn rf_phase(inc: f64, g: usize) -> f64 {
    let tau = std::f64::consts::TAU;
    (inc * ((g as u64 * (g as u64 + 1) / 2) as f64)).rem_euclid(tau)
}

/// Play the sequence for the ensemble. `pos(j, t)` is the position (m from the isocentre) of spin j at time t (s,
/// 0 = the start of the first parcel TR; the pre-roll holds each spin at `pos(j, 0)`). `progress(k, n)` is called per
/// TR (k of n including the pre-roll); returning false stops the run (None).
pub fn simulate(d: &FlowDesign, cfg: &BlochCfg, n_spins: usize, pos: &(dyn Fn(usize, f64) -> Vec3 + Sync), progress: &mut dyn FnMut(usize, usize) -> bool) -> Option<BlochOut> {
    let n_steps = d.n_tr;
    let dt = d.dt;
    let tau = std::f64::consts::TAU;
    let (rf_a, rf_b) = d.rf_steps();
    let flip_eff = cfg.flip * cfg.b1_scale;
    // transmit frequency offset of a slab centred away from the isocentre: B1 rotates at γ G_ss z_c in the carrier frame
    let w_tx = d.slab.map_or(0.0, |g| GAMMA * g.g_ss * cfg.slab_center);
    let rf_c = d.t_rf / dt - 0.5; // step index (continuous) of the pulse centre
    let hand = if cfg.left_handed { -1.0 } else { 1.0 };
    let (e1, e2) = ((-dt / cfg.tissue.t1).exp(), (-dt / cfg.tissue.t2).exp());
    let wdf = tau * cfg.df;
    let n_rot = cfg.n_rot.max(1);
    let set = rotations(RotSet::SuperFibonacci, n_rot);
    let order = acquisition_order(RotSet::SuperFibonacci, n_rot);
    let rot_of = |j: usize| &set[order[j % n_rot]];

    let sel_default = out_steps(d, cfg.per_tr, None);
    let sel_detail = cfg.detail.map(|(_, _, k)| out_steps(d, cfg.per_tr, Some(k)));
    let mstride = cfg.mean_stride.max(1);
    let n_mpt = n_steps.div_ceil(mstride);
    let g_stride = ((10e-6 / dt).round() as usize).max(1);
    let n_g = n_steps / g_stride + 1;

    let mut out = BlochOut { n_spins, g_stride, ..Default::default() };
    out.mean = vec![0.0; cfg.n_tr * n_mpt * 3];
    out.mean_inc = vec![0.0; cfg.n_tr * n_mpt];
    out.g = vec![0.0; cfg.n_tr * n_g * 3];
    let mut st: Vec<[f64; 3]> = vec![[0.0, 0.0, cfg.mz0]; n_spins];
    let p0: Vec<Vec3> = (0..n_spins).map(|j| pos(j, 0.0)).collect();
    let chunk = n_spins.div_ceil(rayon::current_num_threads() * 2).max(4);
    let total = cfg.preroll + cfg.n_tr;

    for gi in 0..total {
        let live = gi >= cfg.preroll;
        let n = gi.saturating_sub(cfg.preroll);
        let (e, j) = (gi % d.n_enc, gi / d.n_enc);
        let w = d.tr_wave(e, rot_of(j));
        let phi = rf_phase(cfg.rf_inc, gi);
        let (cphi, sphi) = (phi.cos(), phi.sin());
        let t_tr = n as f64 * d.tr();
        let sel: &[usize] = if live && cfg.detail.is_some_and(|(a, b, _)| n >= a && n < b) { sel_detail.as_ref().unwrap() } else { &sel_default };
        if live {
            out.encode.push(e as u32);
            out.rot.push((j % n_rot) as u32);
            out.rf_phase.push(phi);
            for s in 0..n_g {
                let v = w[(s * g_stride).min(n_steps)];
                for a in 0..3 {
                    out.g[(n * n_g + s) * 3 + a] = v[a] as f32;
                }
            }
        }
        struct Chunk {
            s0: usize,
            mean: Vec<f64>,
            echo: [f64; 2],
            einc: [f64; 3],
            minc: Vec<f64>,
            m: Vec<f32>,
            p: Vec<f32>,
        }
        let ws = &w;
        let res: Vec<Chunk> = st
            .par_chunks_mut(chunk)
            .enumerate()
            .map(|(ci, ms)| {
                let s0 = ci * chunk;
                let mut c = Chunk { s0, mean: vec![0.0; if live { n_mpt * 3 } else { 0 }], echo: [0.0; 2], einc: [0.0; 3], minc: vec![0.0; if live { n_mpt } else { 0 }], m: Vec::new(), p: Vec::new() };
                // output samples: by spin within the chunk per sample (sample-major like the global layout)
                let nloc = ms.len();
                let mut rec_m = vec![0f32; if live { sel.len() * nloc * 3 } else { 0 }];
                let mut rec_p = vec![0f32; if live { sel.len() * nloc * 3 } else { 0 }];
                for (jj, m) in ms.iter_mut().enumerate() {
                    let js = s0 + jj;
                    let mut si = 0usize; // next output sample
                    let mut mm = *m;
                    for i in 0..=n_steps {
                        let tpos = if live { t_tr + i as f64 * dt } else { 0.0 };
                        // record the state after i steps
                        if live {
                            if si < sel.len() && sel[si] == i {
                                let p = pos(js, tpos);
                                let o = (si * nloc + jj) * 3;
                                for a in 0..3 {
                                    rec_m[o + a] = mm[a] as f32;
                                    rec_p[o + a] = p[a] as f32;
                                }
                                si += 1;
                            }
                            if i % mstride == 0 && i / mstride < n_mpt {
                                let o = (i / mstride) * 3;
                                c.mean[o] += mm[0];
                                c.mean[o + 1] += mm[1];
                                c.mean[o + 2] += mm[2];
                                c.minc[i / mstride] += mm[0].hypot(mm[1]);
                            }
                            if i == d.i_ro {
                                // demodulated by the RF phase
                                c.echo[0] += mm[0] * cphi + mm[1] * sphi;
                                c.echo[1] += -mm[0] * sphi + mm[1] * cphi;
                                c.einc[0] += mm[0].hypot(mm[1]);
                                c.einc[1] += mm[2];
                                c.einc[2] += (mm[0] * mm[0] + mm[1] * mm[1] + mm[2] * mm[2]).sqrt();
                            }
                        }
                        if i == n_steps {
                            break;
                        }
                        // step i: gradient average (linear between the corner samples), position at the step centre
                        let g = [0, 1, 2].map(|a| 0.5 * (ws[i][a] + ws[i + 1][a]));
                        let x = if live { pos(js, t_tr + (i as f64 + 0.5) * dt) } else { p0[js] };
                        let mut bz = g[0] * x[0] + g[1] * x[1] + g[2] * x[2];
                        if cfg.concomitant && cfg.b0 > 0.0 {
                            let (r2, gp2) = (x[0] * x[0] + x[1] * x[1], g[0] * g[0] + g[1] * g[1]);
                            bz += g[2] * g[2] / (8.0 * cfg.b0) * r2 + gp2 / (2.0 * cfg.b0) * x[2] * x[2] - g[2] / (2.0 * cfg.b0) * (g[0] * x[0] + g[1] * x[1]) * x[2];
                        }
                        let wz = GAMMA * bz + wdf;
                        if i >= rf_a && i < rf_b {
                            // exact rotation about (w1 cos φ, w1 sin φ, wz); sense by `hand`
                            let w1 = flip_eff * d.rf_env[i - rf_a];
                            let ang = hand * w_tx * (i as f64 - rf_c) * dt;
                            let (sa, ca) = ang.sin_cos();
                            let wv = [w1 * (cphi * ca - sphi * sa), w1 * (sphi * ca + cphi * sa), wz];
                            let wn = (wv[0] * wv[0] + wv[1] * wv[1] + wv[2] * wv[2]).sqrt();
                            let th = wn * dt;
                            if th > 1e-15 {
                                let k = [wv[0] / wn, wv[1] / wn, wv[2] / wn];
                                let (cs, sn) = (th.cos(), hand * th.sin());
                                let kxm = [k[1] * mm[2] - k[2] * mm[1], k[2] * mm[0] - k[0] * mm[2], k[0] * mm[1] - k[1] * mm[0]];
                                let kd = k[0] * mm[0] + k[1] * mm[1] + k[2] * mm[2];
                                mm = [0, 1, 2].map(|a| mm[a] * cs + kxm[a] * sn + k[a] * kd * (1.0 - cs));
                            }
                        } else {
                            let (s, c0) = (hand * wz * dt).sin_cos();
                            mm = [mm[0] * c0 - mm[1] * s, mm[0] * s + mm[1] * c0, mm[2]];
                        }
                        mm = [mm[0] * e2, mm[1] * e2, mm[2] * e1 + (1.0 - e1)];
                    }
                    *m = mm;
                }
                c.m = rec_m;
                c.p = rec_p;
                c
            })
            .collect();
        if live {
            let base = out.tr.len();
            out.tr.extend(std::iter::repeat_n(n as u32, sel.len()));
            out.step.extend(sel.iter().map(|&s| s as u32));
            out.m.resize((base + sel.len()) * n_spins * 3, 0.0);
            out.pos.resize((base + sel.len()) * n_spins * 3, 0.0);
            let mut echo = [0.0f64; 2];
            let mut einc = [0.0f64; 3];
            for c in &res {
                let ns = c.m.len() / (sel.len() * 3).max(1);
                for k in 0..sel.len() {
                    let dst = ((base + k) * n_spins + c.s0) * 3;
                    out.m[dst..dst + ns * 3].copy_from_slice(&c.m[k * ns * 3..(k + 1) * ns * 3]);
                    out.pos[dst..dst + ns * 3].copy_from_slice(&c.p[k * ns * 3..(k + 1) * ns * 3]);
                }
                for q in 0..n_mpt * 3 {
                    out.mean[n * n_mpt * 3 + q] += (c.mean[q] / n_spins as f64) as f32;
                }
                for q in 0..n_mpt {
                    out.mean_inc[n * n_mpt + q] += (c.minc[q] / n_spins as f64) as f32;
                }
                echo[0] += c.echo[0];
                echo[1] += c.echo[1];
                for a in 0..3 {
                    einc[a] += c.einc[a];
                }
            }
            out.echo_inc.push(einc.map(|x| x / n_spins as f64));
            out.echo.push([echo[0] / n_spins as f64, echo[1] / n_spins as f64]);
        }
        if !progress(gi + 1, total) {
            return None;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// a synthetic design: hard pulse at 100–200 µs, then a constant spoiler of `cyc` cycles/m along z after the echo
    fn toy(spoil_cycles_per_m: f64) -> FlowDesign {
        let (dt, n_tr) = (10e-6, 600usize);
        let (i_ro, i_adc) = (50usize, 100usize);
        // spoiler: 200 samples at constant G over the last 400 µs... area = G·t = k/γ̄
        let t_sp = 2e-3;
        let ns = (t_sp / dt) as usize;
        let gz = spoil_cycles_per_m / (crate::GAMMA_BAR * t_sp);
        let mut g = vec![[0.0; 3]; n_tr + 1];
        for v in g.iter_mut().skip(n_tr - ns) {
            v[2] = gz;
        }
        FlowDesign { name: "toy".into(), dt, t_rf: 150e-6, rf_dur: 100e-6, n_tr, n_enc: 1, i_ro, i_adc, i_enc: 20, i_tail: n_tr - ns, tr_grad: vec![g], base: vec![[0.0; 3]; i_adc + 1], rf_env: vec![1.0 / 100e-6; 10], slab: None }
    }

    fn ball(n: usize, r: f64) -> Vec<Vec3> {
        let phi3 = 1.220_744_084_605_759_5f64;
        let al = [1.0 / phi3, 1.0 / (phi3 * phi3), 1.0 / (phi3 * phi3 * phi3)];
        (0..n * 2)
            .map(|j| al.map(|a| ((0.5 + j as f64 * a).fract() - 0.5) * 2.0 * r))
            .filter(|p| p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= r * r)
            .take(n)
            .collect()
    }

    #[test]
    fn hard_pulse_flip_and_phase() {
        let d = toy(0.0);
        let cfg = BlochCfg { flip: 90f64.to_radians(), rf_inc: 0.0, n_tr: 1, per_tr: 4, concomitant: false, tissue: Tissue { t1: 1e9, t2: 1e9 }, ..Default::default() };
        let o = simulate(&d, &cfg, 1, &|_, _| [0.0; 3], &mut |_, _| true).unwrap();
        let k = o.step.iter().position(|&s| s as usize == d.rf_steps().1).unwrap();
        let m = &o.m[k * 3..k * 3 + 3];
        // B1 along +x, left-handed: M → +y; precisely |Mxy| = 1, Mz = 0
        assert!((m[0] * m[0] + m[1] * m[1]).sqrt() > 0.9999 && m[2].abs() < 1e-6, "{m:?}");
        assert!(m[1] > 0.999, "{m:?}");
    }

    #[test]
    fn rf_spoiled_steady_state_is_ernst() {
        let d = toy(4.0 / 2e-3); // 4 cycles over 2 mm
        let cfg = BlochCfg { flip: 15f64.to_radians(), n_tr: 100, preroll: 900, per_tr: 3, concomitant: false, tissue: Tissue { t1: 1.0, t2: 0.03 }, b0: 3.0, ..Default::default() };
        let ps = ball(400, 3e-3);
        let o = simulate(&d, &cfg, ps.len(), &|j, _| ps[j], &mut |_, _| true).unwrap();
        let mean: f64 = o.echo[50..].iter().map(|e| (e[0] * e[0] + e[1] * e[1]).sqrt()).sum::<f64>() / 50.0;
        let ideal = ernst(cfg.flip, d.tr(), d.te(), &cfg.tissue);
        assert!((mean / ideal - 1.0).abs() < 0.05, "{mean} vs {ideal}");
    }

    #[test]
    fn relaxed_start_first_echo_is_sin_flip() {
        let d = toy(0.0);
        let cfg = BlochCfg { flip: 15f64.to_radians(), n_tr: 1, per_tr: 3, concomitant: false, tissue: Tissue { t1: 1e9, t2: 1e9 }, ..Default::default() };
        let o = simulate(&d, &cfg, 1, &|_, _| [0.0; 3], &mut |_, _| true).unwrap();
        let e = o.echo[0];
        assert!(((e[0] * e[0] + e[1] * e[1]).sqrt() - 15f64.to_radians().sin()).abs() < 1e-6);
    }

    /// the real v2 design: a static parcel pre-rolled to steady state gives the Ernst echo (blood, 15°); a relaxed parcel
    /// starts at sin α and approaches it
    #[test]
    fn v2_design_static_parcel_steady_state() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty/seq/flow_thorax_v2");
        if !dir.join("tr_grad.npy").exists() {
            return;
        }
        let d = FlowDesign::load(&dir).unwrap();
        assert_eq!(d.n_enc, 7);
        let ps = ball(200, 3e-3);
        let mut cfg = BlochCfg { n_tr: 60, preroll: 700, per_tr: 3, rf_inc: 117f64.to_radians(), ..Default::default() };
        let t = std::time::Instant::now();
        let o = simulate(&d, &cfg, ps.len(), &|j, _| ps[j], &mut |_, _| true).unwrap();
        let mean: f64 = o.echo[20..].iter().map(|e| (e[0] * e[0] + e[1] * e[1]).sqrt()).sum::<f64>() / 40.0;
        let ideal = ernst(cfg.flip, d.tr(), d.te(), &cfg.tissue);
        eprintln!("v2 steady: {mean:.5} vs Ernst {ideal:.5} ({:.2} s)", t.elapsed().as_secs_f64());
        assert!((mean / ideal - 1.0).abs() < 0.1, "{mean} vs {ideal}");
        cfg.preroll = 0;
        let o = simulate(&d, &cfg, ps.len(), &|j, _| ps[j], &mut |_, _| true).unwrap();
        let e0 = (o.echo[0][0].powi(2) + o.echo[0][1].powi(2)).sqrt();
        assert!((e0 / cfg.flip.sin() - 1.0).abs() < 0.02, "{e0}");
    }

    /// slab pulse on the real v2 (1 mm) design: centre flip, thickness and a displaced slab, relaxed static spins along z
    #[test]
    fn slab_profile_thickness_and_centre() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty/seq/flow_thorax_v2_1mm");
        if !dir.join("tr_grad.npy").exists() {
            return;
        }
        let base = FlowDesign::load(&dir).unwrap();
        let d = base.with_slab(&SlabSpec::thorax(0.175));
        assert_eq!(d.n_tr, base.n_tr + d.i_enc - base.i_enc);
        let g = d.slab.unwrap();
        assert!((g.g_ss * 1e3 - 0.99).abs() < 0.1, "G_ss {}", g.g_ss);
        let zs: Vec<f64> = (-20..=20).map(|k| k as f64 * 0.01).collect();
        for centre in [0.0, 0.03] {
            let cfg = BlochCfg { flip: 15f64.to_radians(), n_tr: 1, per_tr: 2, concomitant: false, tissue: Tissue { t1: 1e9, t2: 1e9 }, slab_center: centre, ..Default::default() };
            let o = simulate(&d, &cfg, zs.len(), &|j, _| [0.0, 0.0, zs[j] + 0.0], &mut |_, _| true).unwrap();
            // the sample right after the RF block and rephaser: step i_enc + 20 (rephaser ≤ 200 µs)
            let k = o.step.iter().position(|&s| s as usize == d.i_ro).unwrap();
            let flip_at = |j: usize| {
                let m = &o.m[(k * zs.len() + j) * 3..(k * zs.len() + j) * 3 + 3];
                (m[0] * m[0] + m[1] * m[1]).sqrt().atan2(m[2]).to_degrees()
            };
            let jc = zs.iter().position(|&z| (z - centre).abs() < 1e-9).unwrap_or(20);
            let c = flip_at(jc);
            let half = zs.iter().enumerate().filter(|&(j, _)| flip_at(j) > 7.5).map(|(_, &z)| z).collect::<Vec<_>>();
            let (lo, hi) = (half.first().copied().unwrap(), half.last().copied().unwrap());
            eprintln!("slab centre {centre}: α(centre) {c:.2}°, half-max range {lo:.2} … {hi:.2} m, α at ±0.2 m {:.3}° / {:.3}°", flip_at(0), flip_at(40));
            assert!((c - 15.0).abs() < 1.0, "{c}");
            assert!(((hi - lo) - 0.175).abs() < 0.03 && ((hi + lo) / 2.0 - centre).abs() < 0.015, "{lo} {hi}");
            assert!(flip_at(0) < 1.0 && flip_at(40) < 1.0);
        }
    }

    /// 1 µs raster (refine 10) gives the same physics as the 10 µs raster: slab profile and steady-state echo
    #[test]
    fn refined_raster_agrees() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty/seq/flow_thorax_v2_1mm");
        if !dir.join("tr_grad.npy").exists() {
            return;
        }
        let coarse = FlowDesign::load(&dir).unwrap().with_slab(&SlabSpec::thorax(0.175));
        let fine = coarse.refine(10);
        assert_eq!(fine.n_tr, coarse.n_tr * 10);
        // the v2 1 mm readout reaches k_max (recon window 9.48 ms) after 948 raster steps, then returns
        assert_eq!(coarse.i_kmax(), 948);
        assert_eq!(fine.i_kmax(), 9480);
        assert!((fine.dt * 10.0 - coarse.dt).abs() < 1e-15 && (fine.tr() - coarse.tr()).abs() < 1e-12 && (fine.te() - coarse.te()).abs() < 1e-12);
        assert_eq!(fine.rf_steps(), (coarse.rf_steps().0 * 10, coarse.rf_steps().1 * 10));
        let area = |d: &FlowDesign| d.rf_env.iter().sum::<f64>() * d.dt;
        assert!((area(&fine) - 1.0).abs() < 1e-9);
        let ps = ball(120, 3e-3);
        let cfg = BlochCfg { n_tr: 40, preroll: 150, per_tr: 3, tissue: Tissue { t1: 0.388, t2: 0.111 }, ..Default::default() };
        let echo = |d: &FlowDesign| {
            let o = simulate(d, &cfg, ps.len(), &|j, _| ps[j], &mut |_, _| true).unwrap();
            o.echo[10..].iter().map(|e| (e[0] * e[0] + e[1] * e[1]).sqrt()).sum::<f64>() / 30.0
        };
        let (ec, ef) = (echo(&coarse), echo(&fine));
        eprintln!("steady echo: 10 µs {ec:.5}, 1 µs {ef:.5}");
        assert!((ef / ec - 1.0).abs() < 0.03, "{ec} {ef}");
        // slab profile on the fine raster
        let zs: Vec<f64> = (-20..=20).map(|k| k as f64 * 0.01).collect();
        let cfg = BlochCfg { n_tr: 1, per_tr: 2, concomitant: false, tissue: Tissue { t1: 1e9, t2: 1e9 }, ..Default::default() };
        let o = simulate(&fine, &cfg, zs.len(), &|j, _| [0.0, 0.0, zs[j]], &mut |_, _| true).unwrap();
        let k = o.step.iter().position(|&s| s as usize == fine.i_ro).unwrap();
        let a = |j: usize| {
            let m = &o.m[(k * zs.len() + j) * 3..(k * zs.len() + j) * 3 + 3];
            (m[0] * m[0] + m[1] * m[1]).sqrt().atan2(m[2]).to_degrees()
        };
        assert!((a(20) - 15.0).abs() < 1.0 && a(0) < 1.0 && a(40) < 1.0, "{} {} {}", a(20), a(0), a(40));
    }

    /// TR-to-TR ripple of the steady-state echo of a static parcel vs spin count and spoiler area, and the diffusion
    /// weighting (b) of one TR (diagnostic, run with --nocapture)
    #[test]
    fn echo_ripple_print() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty/seq/flow_thorax_v2_1mm");
        if !dir.join("tr_grad.npy").exists() {
            return;
        }
        let base = FlowDesign::load(&dir).unwrap().with_slab(&SlabSpec::thorax(0.175));
        let stats = |d: &FlowDesign, n: usize, r: f64, n_tr: usize| {
            let ps = ball(n, r);
            let cfg = BlochCfg { n_tr, preroll: 300, per_tr: 3, tissue: Tissue { t1: 0.388, t2: 0.111 }, concomitant: std::env::var("NOCONC").is_err(), ..Default::default() };
            let o = simulate(d, &cfg, ps.len(), &|j, _| ps[j], &mut |_, _| true).unwrap();
            let e: Vec<f64> = o.echo.iter().map(|e| (e[0] * e[0] + e[1] * e[1]).sqrt()).collect();
            let mean = e.iter().sum::<f64>() / e.len() as f64;
            let sd = (e.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / e.len() as f64).sqrt();
            let (mn, mx) = e.iter().fold((1e9f64, -1e9f64), |a, &x| (a.0.min(x), a.1.max(x)));
            let inc = o.echo_inc.iter().map(|x| x[0]).sum::<f64>() / e.len() as f64;
            (mean, sd, mn, mx, inc)
        };
        let ideal = ernst(15f64.to_radians(), base.tr(), base.te(), &Tissue { t1: 0.388, t2: 0.111 });
        eprintln!("Ernst {ideal:.5}");
        for (n, r) in [(200, 1e-3), (1000, 1e-3), (8000, 1e-3), (8000, 0.5e-3), (8000, 2e-3), (8000, 3e-3)] {
            let (m, s, a, b, inc) = stats(&base, n, r, 100);
            eprintln!("N {n:5} r {:.1} mm: echo mean {m:.5} sd {s:.5} ({:.1} %) range {a:.4}…{b:.4} per-spin {inc:.4}", r * 1e3, 100.0 * s / m);
        }
        // spoiler area scaled (tail block gradients × f)
        for f in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0] {
            let mut d = base.clone();
            for w in d.tr_grad.iter_mut() {
                for g in w.iter_mut().skip(d.i_tail) {
                    for c in g.iter_mut() {
                        *c *= f;
                    }
                }
            }
            let (m, s, a, b, inc) = stats(&d, 8000, 1e-3, 100);
            eprintln!("spoiler ×{f}: echo mean {m:.5} sd {s:.5} ({:.1} %) range {a:.4}…{b:.4} per-spin {inc:.4}", 100.0 * s / m);
        }
        // diffusion weighting of one TR for the transverse coherence between two RF pulses: b = ∫ (2π k)² dt, k = γ̄ ∫ G
        let (ra, rb) = base.rf_steps();
        let t0 = (ra + rb) / 2;
        let ident: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        for e in 0..base.n_enc {
            let w = base.tr_wave(e, &ident);
            let (mut k, mut b, mut kmax) = ([0.0f64; 3], 0.0f64, 0.0f64);
            for i in t0..base.n_tr {
                for a in 0..3 {
                    k[a] += crate::GAMMA_BAR * 0.5 * (w[i][a] + w[i + 1][a]) * base.dt;
                }
                let kk = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
                kmax = kmax.max(kk);
                b += (2.0 * std::f64::consts::PI * kk).powi(2) * base.dt;
            }
            let kn = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt();
            eprintln!("encode {e}: b(one TR) {:.3} s/mm², net k at the next RF {:.3} cycles/mm (max {:.3}); exp(-bD) D=3e-3 mm²/s: {:.4}", b * 1e-6, kn * 1e-3, kmax * 1e-3, (-b * 1e-6 * 3e-3).exp());
        }
    }

    /// prints the slab flip-angle profile at 1 µs (diagnostic, run with --nocapture)
    #[test]
    fn slab_profile_print() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty/seq/flow_thorax_v2_1mm");
        if !dir.join("tr_grad.npy").exists() {
            return;
        }
        let d = FlowDesign::load(&dir).unwrap().with_slab(&SlabSpec::thorax(0.175)).refine(10);
        let zs: Vec<f64> = (0..=30).map(|k| k as f64 * 0.01).collect();
        let cfg = BlochCfg { n_tr: 1, per_tr: 2, concomitant: false, tissue: Tissue { t1: 1e9, t2: 1e9 }, ..Default::default() };
        let o = simulate(&d, &cfg, zs.len(), &|j, _| [0.0, 0.0, zs[j]], &mut |_, _| true).unwrap();
        let k = o.step.iter().position(|&s| s as usize == d.rf_steps().1).unwrap();
        for (j, z) in zs.iter().enumerate() {
            let m = &o.m[(k * zs.len() + j) * 3..(k * zs.len() + j) * 3 + 3];
            eprintln!("z {:.0} mm: flip {:.3}°", z * 1e3, (m[0] * m[0] + m[1] * m[1]).sqrt().atan2(m[2]).to_degrees());
        }
    }
}
