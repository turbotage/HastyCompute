//! Measured GIRF file (output of the GIRF QA estimator, research/experiments/girf_qa) → [`Girf`] at a thermal state,
//! and the resonance list for the sequence designer.
//!
//! Format `hasty-girf-measured` v1 (JSON):
//! ```text
//! { "format": "hasty-girf-measured", "version": 1, "scanner": "...", "date": "...",
//!   "thermal": { "model": "power-2c", "tau_s": [τ1, τ2], "w": w, "p_flow_T2m2": P, "x_ref": x_tables },
//!   "axes": [ { "f": [Hz...], "re": [...], "im": [...],                 // continuous self term, resonances divided out
//!               "resonances": [ { "f0": Hz, "zeta": ζ, "gain": g, "df0_dx": Hz, "dgain_dx": 1, "sd_f0": Hz } ] }, ×3 ],
//!   "cross": [ { "out": o, "in": i, "hp": [[k, τ], ...] } ],            // Σ k (iωτ)/(1 + iωτ)
//!   "b0":    [ { "f": [...], "re": [...], "im": [...] } | { "hp": [[k, τ], ...] } | null, ×3 ],   // T per T/m
//!   "spatial": [ { "in": a, "basis": "xy", "hp": [[k, τ], ...] } ],     // 2nd / 3rd order (not used by Girf)
//!   "receiver": { "delay_ns_vs_4us": { "2": ns, "8": ns, "16": ns }, "coil_offset_ns": ns },
//!   "nonlinearity": { "beta": [βx, βy, βz], "g_ref": 0.07 } }
//! ```
//! The thermal state x is the normalised power-history state of the coil (0 = cold, 1 = the flow scan's steady
//! state); resonance f0 / gain move linearly with x (`df0_dx`, `dgain_dx`), the background tables are at `x_ref`.
//! A `Girf::from_measurement(path, x)` wrapper in girf.rs is a one-liner over [`from_measurement`].

use num_complex::Complex64;
use serde_json::Value;

use crate::girf::{Girf, Response};

#[derive(Clone, Debug)]
pub struct MeasuredResonance {
    pub axis: usize,
    pub f0: f64,
    pub zeta: f64,
    pub gain: f64,
    pub df0_dx: f64,
    pub dgain_dx: f64,
    pub sd_f0: f64,
}

#[derive(Clone, Debug)]
pub struct MeasuredGirf {
    pub scanner: String,
    pub axes: [Response; 3],
    pub resonances: Vec<MeasuredResonance>,
    pub cross: Vec<(usize, usize, Response)>,
    pub b0: [Option<Response>; 3],
    pub x_ref: f64,
    pub tau_s: Vec<f64>,
    pub w: f64,
    pub p_flow: f64,
    pub rx_delay_ns: Vec<(f64, f64)>,
    pub beta: [f64; 3],
}

fn arr(v: &Value, k: &str) -> Result<Vec<f64>, String> {
    v[k].as_array().ok_or(format!("missing array {k}"))?.iter().map(|x| x.as_f64().ok_or(format!("{k}: not a number"))).collect()
}

fn table(v: &Value) -> Result<Response, String> {
    let (f, re, im) = (arr(v, "f")?, arr(v, "re")?, arr(v, "im")?);
    if f.len() != re.len() || f.len() != im.len() || f.windows(2).any(|w| w[1] <= w[0]) {
        return Err("table: f / re / im lengths differ or f not ascending".into());
    }
    Ok(Response::Table { f, h: re.iter().zip(&im).map(|(a, b)| Complex64::new(*a, *b)).collect() })
}

/// Σ high-pass terms as a list of responses (Girf adds cross entries; for single-response slots use `hp_table`)
fn hp_terms(v: &Value) -> Result<Vec<Response>, String> {
    v["hp"].as_array().ok_or("missing hp")?.iter().map(|t| {
        let k = t[0].as_f64().ok_or("hp k")?;
        let tau = t[1].as_f64().ok_or("hp tau")?;
        Ok(Response::HighPass { tau, k })
    }).collect()
}

/// a sum of high-pass terms sampled into a table (0 … 5 MHz, log-spaced above 20 kHz)
fn hp_table(terms: &[Response]) -> Response {
    let mut f: Vec<f64> = (0..=4000).map(|i| i as f64 * 5.0).collect();
    let n = 400;
    for i in 1..=n {
        f.push(20e3 * (5e6f64 / 20e3).powf(i as f64 / n as f64));
    }
    let h = f.iter().map(|&x| terms.iter().map(|r| r.at(x)).sum()).collect();
    Response::Table { f, h }
}

impl MeasuredGirf {
    pub fn parse(text: &str) -> Result<Self, String> {
        let v: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if v["format"].as_str() != Some("hasty-girf-measured") || v["version"].as_i64() != Some(1) {
            return Err("not a hasty-girf-measured v1 file".into());
        }
        let ax = v["axes"].as_array().ok_or("missing axes")?;
        if ax.len() != 3 {
            return Err("need 3 axes".into());
        }
        let mut resonances = vec![];
        let mut axes = vec![];
        for (a, x) in ax.iter().enumerate() {
            axes.push(table(x)?);
            for r in x["resonances"].as_array().cloned().unwrap_or_default() {
                let g = |k: &str, d: f64| r[k].as_f64().unwrap_or(d);
                resonances.push(MeasuredResonance { axis: a, f0: g("f0", f64::NAN), zeta: g("zeta", f64::NAN), gain: g("gain", 0.0), df0_dx: g("df0_dx", 0.0), dgain_dx: g("dgain_dx", 0.0), sd_f0: g("sd_f0", f64::NAN) });
            }
        }
        if resonances.iter().any(|r| !(r.f0 > 0.0) || !(r.zeta > 0.0)) {
            return Err("resonance with missing f0 / zeta".into());
        }
        let mut cross = vec![];
        for c in v["cross"].as_array().cloned().unwrap_or_default() {
            let (o, i) = (c["out"].as_u64().ok_or("cross out")? as usize, c["in"].as_u64().ok_or("cross in")? as usize);
            for r in hp_terms(&c)? {
                cross.push((o, i, r));
            }
        }
        let mut b0: [Option<Response>; 3] = [None, None, None];
        if let Some(bs) = v["b0"].as_array() {
            for (a, b) in bs.iter().enumerate().take(3) {
                b0[a] = if b.is_null() { None } else if b.get("f").is_some() { Some(table(b)?) } else { Some(hp_table(&hp_terms(b)?)) };
            }
        }
        let th = &v["thermal"];
        let rx: Vec<(f64, f64)> = v["receiver"]["delay_ns_vs_4us"].as_object().map(|o| o.iter().filter_map(|(k, x)| Some((k.parse::<f64>().ok()?, x.as_f64()?))).collect()).unwrap_or_default();
        let beta = v["nonlinearity"]["beta"].as_array().map(|b| [0, 1, 2].map(|i| b.get(i).and_then(|x| x.as_f64()).unwrap_or(0.0))).unwrap_or([0.0; 3]);
        Ok(Self {
            scanner: v["scanner"].as_str().unwrap_or("").into(),
            axes: [axes[0].clone(), axes[1].clone(), axes[2].clone()],
            resonances,
            cross,
            b0,
            x_ref: th["x_ref"].as_f64().unwrap_or(0.0),
            tau_s: th["tau_s"].as_array().map(|a| a.iter().filter_map(|x| x.as_f64()).collect()).unwrap_or_default(),
            w: th["w"].as_f64().unwrap_or(1.0),
            p_flow: th["p_flow_T2m2"].as_f64().unwrap_or(f64::NAN),
            rx_delay_ns: rx,
            beta,
        })
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        Self::parse(&std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?)
    }

    /// resonances at thermal state x (axis, f0, ζ, gain)
    pub fn resonance_list(&self, x: f64) -> Vec<(usize, f64, f64, f64)> {
        self.resonances.iter().map(|r| (r.axis, r.f0 + r.df0_dx * (x - self.x_ref), r.zeta, r.gain + r.dgain_dx * (x - self.x_ref))).collect()
    }

    /// the chain at thermal state x: background tables × Π (1 + g BP) of the resonances at x, cross and B0 terms
    pub fn girf(&self, x: f64) -> Girf {
        let res = self.resonance_list(x);
        let axis = [0, 1, 2].map(|a| {
            let mut r = self.axes[a].clone();
            for &(ra, f0, zeta, gain) in &res {
                if ra == a {
                    r = Response::Resonant { base: Box::new(r), f0, zeta, gain };
                }
            }
            r
        });
        Girf { axis, cross: self.cross.clone(), b0: self.b0.clone(), tail: 0.15 }
    }

    /// normalised thermal state of a power history (1-s bins of Σ_a G_a², T²/m²) at its end: w x₁ + (1 − w) x₂,
    /// x_k = (P ∗ e^{−t/τ_k}/τ_k) / P_flow
    pub fn state_of(&self, power: &[f64]) -> f64 {
        if self.tau_s.len() < 2 || !(self.p_flow > 0.0) {
            return self.x_ref;
        }
        let mut xs = [0.0; 2];
        for (k, x) in xs.iter_mut().enumerate() {
            let a = (-1.0 / self.tau_s[k]).exp();
            for &p in power {
                *x = *x * a + p * (1.0 - a);
            }
            *x /= self.p_flow;
        }
        self.w * xs[0] + (1.0 - self.w) * xs[1]
    }
}

/// measured chain at thermal state x (see [`MeasuredGirf::girf`])
pub fn from_measurement(path: &std::path::Path, x: f64) -> Result<Girf, String> {
    Ok(MeasuredGirf::load(path)?.girf(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEMO: &str = r#"{"format": "hasty-girf-measured", "version": 1, "scanner": "test",
      "thermal": {"model": "power-2c", "tau_s": [200.0, 1800.0], "w": 0.5, "p_flow_T2m2": 3.6e-4, "x_ref": 0.8},
      "axes": [{"f": [0, 1000, 2000, 1e6], "re": [1, 0.999, 0.998, 0.5], "im": [0, -0.01, -0.02, 0],
                "resonances": [{"f0": 700.0, "zeta": 0.005, "gain": 0.04, "df0_dx": -3.5, "dgain_dx": 0.001, "sd_f0": 0.02}]},
               {"f": [0, 1e6], "re": [1, 1], "im": [0, 0], "resonances": []},
               {"f": [0, 1e6], "re": [1, 1], "im": [0, 0]}],
      "cross": [{"out": 1, "in": 0, "hp": [[3e-4, 1e-3], [-1e-4, 2e-4]]}],
      "b0": [{"hp": [[2e-5, 2e-3]]}, null, {"f": [0, 1e6], "re": [0, 0], "im": [0, 0]}],
      "receiver": {"delay_ns_vs_4us": {"2": -3000.0, "8": 6000.0}},
      "nonlinearity": {"beta": [1e-4, 2e-4, 0]}}"#;

    #[test]
    fn parse_and_evaluate() {
        let m = MeasuredGirf::parse(DEMO).unwrap();
        assert_eq!(m.resonances.len(), 1);
        let g0 = m.girf(0.8);
        let g1 = m.girf(1.8);
        // the resonance peak sits at f0(x): 700 Hz at x_ref, 696.5 Hz one unit warmer
        let peak = |g: &Girf| (6900..7100).map(|i| i as f64 * 0.1).max_by(|a, b| g.axis[0].at(*a).norm().partial_cmp(&g.axis[0].at(*b).norm()).unwrap()).unwrap();
        assert!((peak(&g0) - 700.0).abs() < 0.3, "{}", peak(&g0));
        assert!((peak(&g1) - 696.5).abs() < 0.3, "{}", peak(&g1));
        // background table between nodes, with the resonance factor far from f0
        let h = g0.axis[0].at(1500.0);
        let bpf = |f: f64| { let (w, w0) = (2.0 * std::f64::consts::PI * f, 2.0 * std::f64::consts::PI * 700.0); Complex64::new(0.0, 2.0 * 0.005 * w0 * w) / Complex64::new(w0 * w0 - w * w, 2.0 * 0.005 * w0 * w) };
        let expect = Complex64::new(0.9985, -0.015) * (1.0 + 0.04 * bpf(1500.0));
        assert!((h - expect).norm() < 1e-9, "{h} {expect}");
        assert_eq!(g0.cross.len(), 2);
        assert!(g0.b0[0].is_some() && g0.b0[1].is_none() && g0.b0[2].is_some());
        let b = g0.b0[0].as_ref().unwrap().at(500.0);
        let hp = Response::HighPass { tau: 2e-3, k: 2e-5 }.at(500.0);
        assert!((b - hp).norm() < 1e-3 * hp.norm(), "{b} {hp}");
        assert_eq!(m.beta, [1e-4, 2e-4, 0.0]);
        // thermal state: a long constant flow-scan load → 1
        let p = vec![3.6e-4; 20000];
        assert!((m.state_of(&p) - 1.0).abs() < 1e-3);
        // the chain applies
        let w = crate::wave::Grad::from_axis(0, &crate::wave::trap_shape(25, 100, 25, 0.02), 4e-6, 0.0);
        let (out, b0) = g0.apply(&w);
        assert!(out.len() > w.len() && b0.iter().any(|x| *x != 0.0));
        assert!(MeasuredGirf::parse("{}").is_err());
    }

    /// a file written by research/experiments/girf_qa/girfmeas_io.py (its "check" block holds the Python chain's H at
    /// a few frequencies): run with HASTY_GIRF_MEASURED=<file>, skipped otherwise
    #[test]
    fn python_written_file_matches() {
        let Ok(p) = std::env::var("HASTY_GIRF_MEASURED") else { return };
        let txt = std::fs::read_to_string(&p).unwrap();
        let v: Value = serde_json::from_str(&txt).unwrap();
        let m = MeasuredGirf::parse(&txt).unwrap();
        let chk = &v["check"];
        let x = chk["x"].as_f64().unwrap();
        let g = m.girf(x);
        let fs: Vec<f64> = chk["f"].as_array().unwrap().iter().map(|f| f.as_f64().unwrap()).collect();
        let mut worst: f64 = 0.0;
        for a in 0..3 {
            for (i, &f) in fs.iter().enumerate() {
                let e = &chk["axes"][a][i];
                let want = Complex64::new(e[0].as_f64().unwrap(), e[1].as_f64().unwrap());
                worst = worst.max((g.axis[a].at(f) - want).norm());
            }
            let e = &chk["b0_500"][a];
            let want = Complex64::new(e[0].as_f64().unwrap(), e[1].as_f64().unwrap());
            let got = g.b0[a].as_ref().unwrap().at(500.0);
            assert!((got - want).norm() <= 1e-9 * want.norm().max(1e-12), "b0 axis {a}: {got} vs {want}");
        }
        eprintln!("{p}: {} resonances, max |ΔH| vs the Python chain {worst:.2e}", m.resonances.len());
        assert!(worst < 1e-9, "{worst}");
    }
}
