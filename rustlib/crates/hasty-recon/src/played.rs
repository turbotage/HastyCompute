//! Per-sample records of a schedule-based flow sim recomputed through a *model* gradient chain (an estimated GIRF), on
//! the fly: what a reconstruction that models the played gradients uses instead of the simulator's own
//! `sched_played.npy` (which holds the *true* chain). Selected with `HASTY_TRAJ=girf:<girf.json>` (a `hasty-girf-1`
//! table file, hasty_seq `Girf::from_measurement`) or `girf:<variant>` (a `Girf::variants()` name / `clinical`, for
//! tests); `HASTY_GIRF_HIST` previous TRs are carried (default 10, research/girf_sensitivity.md).
//!
//! The sequence comes from the sim's seqdesign run (`meta.sequence.run`: tr_grad.npy, spoke_grad.npy, summary.json,
//! final.json), the per-readout encode and rotation from readouts.npy, so the records equal `FlowSchedule::sample_params`
//! of the simulator for the same chain (hasty-bench test `model_records_match_the_schedule`). Each readout costs
//! ≈ 1 ms on one core (hasty_seq `PlayedStream`: LTI superposition of precomputed block responses), nothing is
//! written to disk.

use std::path::{Path, PathBuf};

use hasty_mrsim::sequence::{sample_record, SAMPLE_FLOATS};
use hasty_seq::girf::{Girf, PlayedStream};
use hasty_seq::{Mat3, Vec3};
use serde_json::Value;

use crate::data::read_npy;
use crate::Error;

/// The flow TR of a seqdesign run (`--spoil flowcomp`): encode / tail blocks on physical axes per encode, base readout.
#[derive(Clone)]
pub struct FlowDesign {
    pub run: PathBuf,
    pub dt: f64,
    pub t_rf: f64,
    pub i_ro: usize,
    pub i_adc: usize,
    pub n_tr: usize,
    /// [encode][n_tr + 1] physical TR of the design's first rotation (its readout samples are replaced)
    pub tr_grad: Vec<Vec<Vec3>>,
    /// base readout, samples 0 ..= i_adc
    pub base: Vec<Vec3>,
    /// raster ranges of the base readout holding reconstruction samples (`best.recon`; default the whole ADC)
    pub recon: Vec<[usize; 2]>,
}

impl FlowDesign {
    /// `run`: absolute, or relative to the repository root (as the simulator records it in `meta.sequence.run`)
    pub fn load(run: &Path) -> Result<Self, Error> {
        let run = if run.is_absolute() { run.to_path_buf() } else { hasty_site::repo_path(run) };
        let js = |f: &str| -> Result<Value, Error> { serde_json::from_str(&std::fs::read_to_string(run.join(f))?).map_err(|e| Error::Format(format!("{}: {e}", run.join(f).display()))) };
        let summ = js("summary.json")?;
        let fin = js("final.json")?;
        let b = &summ["best"];
        if b["spoil_mode"].as_str() != Some("flowcomp") {
            return Err(Error::Format(format!("{}: the model trajectory needs the per-encode physical tail (--spoil flowcomp)", run.display())));
        }
        let u = |k: &str| b[k].as_u64().map(|v| v as usize).ok_or_else(|| Error::Format(format!("summary.json: best.{k}")));
        let (i_ro, i_adc, n_tr) = (u("i_ro")?, u("i_adc")?, u("n_tr")?);
        let n_enc = fin["n_enc"].as_u64().ok_or_else(|| Error::Format("final.json: n_enc".into()))? as usize;
        let dt = fin["dt"].as_f64().ok_or_else(|| Error::Format("final.json: dt".into()))?;
        let t_rf = fin["t_rf"].as_f64().ok_or_else(|| Error::Format("final.json: t_rf".into()))?;
        let tg = read_npy(&run.join("tr_grad.npy"))?.f64s();
        let ntr1 = tg.len() / (3 * n_enc);
        if ntr1 < n_tr {
            return Err(Error::Format("tr_grad.npy shorter than the TR".into()));
        }
        let tr_grad: Vec<Vec<Vec3>> = (0..n_enc)
            .map(|e| {
                let mut g: Vec<Vec3> = (0..ntr1).map(|i| [tg[(e * ntr1 + i) * 3], tg[(e * ntr1 + i) * 3 + 1], tg[(e * ntr1 + i) * 3 + 2]]).collect();
                g.resize(n_tr + 1, [0.0; 3]);
                g.truncate(n_tr + 1);
                g
            })
            .collect();
        let sg = read_npy(&run.join("spoke_grad.npy"))?.f64s();
        let base: Vec<Vec3> = (0..=i_adc).map(|i| [sg[3 * i], sg[3 * i + 1], sg[3 * i + 2]]).collect();
        let recon = b["recon"].as_array().map(|a| a.iter().map(|r| [r[0].as_u64().unwrap_or(0) as usize, r[1].as_u64().unwrap_or(i_adc as u64) as usize]).collect()).unwrap_or_else(|| vec![[0, i_adc]]);
        Ok(Self { run, dt, t_rf, i_ro, i_adc, n_tr, tr_grad, base, recon })
    }

    pub fn n_enc(&self) -> usize {
        self.tr_grad.len()
    }

    /// TR types with their readout samples zeroed (the rotation-independent parts, `PlayedStream::new`)
    pub fn fixed_parts(&self) -> Vec<Vec<Vec3>> {
        self.tr_grad
            .iter()
            .map(|g| {
                let mut g = g.clone();
                for v in g[self.i_ro..self.i_ro + self.base.len()].iter_mut() {
                    *v = [0.0; 3];
                }
                g
            })
            .collect()
    }

    /// the played-trajectory engine for chain `girf` with `hist` previous TRs
    pub fn stream(&self, girf: &Girf, hist: usize) -> PlayedStream {
        PlayedStream::new(girf, &self.fixed_parts(), &self.base, self.i_ro, self.n_tr, self.dt, hist.max(1))
    }
}

/// Per-readout records of a sim through a model chain (see the module docs).
pub struct ModelTraj {
    pub design: FlowDesign,
    pub stream: PlayedStream,
    pub hist: usize,
    /// encode and readout rotation (physical) of every readout, acquisition order (readout i = TR i)
    pub enc: Vec<usize>,
    pub rot: Vec<Mat3>,
    /// ADC sample times from the TR start
    pub times: Vec<f64>,
    pub b0: f64,
    /// what the chain is (file path or variant name)
    pub name: String,
}

impl ModelTraj {
    /// `spec`: a `hasty-girf-1` file path, `clinical`, `ideal` or a `Girf::variants()` name
    pub fn girf_of(spec: &str) -> Result<Girf, Error> {
        match spec {
            "ideal" | "none" => Ok(Girf::ideal()),
            "clinical" => Ok(Girf::clinical_example()),
            p if p.ends_with(".json") => Girf::from_measurement(Path::new(p)).map_err(Error::Format),
            name => Girf::variants().into_iter().find(|v| v.0 == name).map(|v| v.1).ok_or_else(|| Error::Format(format!("GIRF {name}: a hasty-girf-1 .json file, ideal, clinical or a Girf::variants() name"))),
        }
    }

    /// from a sim's meta (sequence.run, dwell, b0, n_adc) and its readouts.npy table ([n_ro][14]: t_exc, encode, ·, frame,
    /// phase, rotation row-major)
    pub fn new(meta: &Value, tab: &[f64], girf: &Girf, name: &str, hist: usize) -> Result<Self, Error> {
        let run = meta["sequence"]["run"].as_str().ok_or_else(|| Error::Format("meta.sequence.run missing (the model trajectory needs the seqdesign run)".into()))?;
        let design = FlowDesign::load(Path::new(run))?;
        let dwell = meta["sequence"]["dwell"].as_f64().or_else(|| meta["dwell"].as_f64()).ok_or_else(|| Error::Format("meta: dwell".into()))?;
        let na = meta["n_adc"].as_u64().ok_or_else(|| Error::Format("meta: n_adc".into()))? as usize;
        let b0 = meta["b0"].as_f64().ok_or_else(|| Error::Format("meta: b0".into()))?;
        let n_ro = tab.len() / 14;
        let enc: Vec<usize> = (0..n_ro).map(|i| tab[14 * i + 1] as usize).collect();
        if enc.iter().any(|&e| e >= design.n_enc()) {
            return Err(Error::Format("readouts.npy: encode index beyond the design's encodes".into()));
        }
        let rot: Vec<Mat3> = (0..n_ro).map(|i| [0, 1, 2].map(|a| [0, 1, 2].map(|b| tab[14 * i + 5 + 3 * a + b]))).collect();
        let times: Vec<f64> = (0..na).map(|n| design.i_ro as f64 * design.dt + n as f64 * dwell).collect();
        let stream = design.stream(girf, hist);
        Ok(Self { design, stream, hist: hist.max(1), enc, rot, times, b0, name: name.to_string() })
    }

    /// the SAMPLE_FLOATS records of readout `ro` at every ADC sample (as the simulator's sched_*.npy rows): played γM0,
    /// γM1, ½γM2, concomitant coefficients from the played ∫GGᵀ, τ, and the B0-term phase (slot 15)
    pub fn records(&self, ro: usize) -> Vec<f32> {
        let seq: Vec<(usize, Mat3)> = (ro.saturating_sub(self.hist)..=ro).map(|j| (self.enc[j], self.rot[j])).collect();
        let pl = self.stream.moments(&seq, self.design.t_rf, &self.times);
        let mut out = Vec::with_capacity(self.times.len() * SAMPLE_FLOATS);
        for (m, &t) in pl.iter().zip(&self.times) {
            let mut r = sample_record(&m.m0, &m.m1, &m.m2, &m.p, self.b0, t - self.design.t_rf);
            r[15] = m.phi_b0 as f32;
            out.extend_from_slice(&r);
        }
        out
    }
}
