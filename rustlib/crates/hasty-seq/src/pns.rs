//! Peripheral nerve stimulation: the SAFE model (Hebrank & Gebhardt, ISMRM 2000 #2007), as implemented by
//! Szczepankiewicz & Witzel (safe_pns_prediction, BSD-3-Clause; also in pypulseq).
//!
//! Per physical axis, with s = dG/dt (T/m/s) on the raster and RC low-passes LP_τ (α = dt / (τ + dt)):
//!   stim = (a₁ |LP_τ₁(s)| + a₂ LP_τ₂(|s|) + a₃ |LP_τ₃(s)|) / stim_limit · g_scale · 100   [% of the threshold]
//! and the combined stimulation is the root-sum-square over the axes; PNS is acceptable while it stays below 100 %
//! (the scanner's normal mode usually caps lower, e.g. 80 %).
//! Hardware parameters come from the gradient system's `.asc` file (`GradPatSup.Phys.PNS.flGSWD…`, g_scale from
//! `asGPAParameters[0].sGCParameters.flGScaleFactor…`). The built-in `example()` is pypulseq's illustrative set, not a
//! real scanner.

use crate::wave::Grad;
use crate::Vec3;

#[derive(Clone, Debug)]
pub struct SafeAxis {
    /// time constants, ms
    pub tau: [f64; 3],
    /// weights (a₁ + a₂ + a₃ = 1)
    pub a: [f64; 3],
    /// T/m/s
    pub stim_limit: f64,
    pub stim_thresh: f64,
    pub g_scale: f64,
}

#[derive(Clone, Debug)]
pub struct SafeHw {
    pub name: String,
    pub axes: [SafeAxis; 3],
}

impl SafeHw {
    /// pypulseq's `safe_example_hw()` (illustrative values, NOT a real gradient system)
    pub fn example() -> Self {
        let ax = |tau: [f64; 3], a: [f64; 3], lim: f64, thr: f64, gs: f64| SafeAxis { tau, a, stim_limit: lim, stim_thresh: thr, g_scale: gs };
        Self {
            name: "MP_GPA_EXAMPLE".into(),
            axes: [
                ax([0.20, 0.03, 3.00], [0.40, 0.10, 0.50], 30.0, 24.0, 0.35),
                ax([1.50, 2.50, 0.15], [0.55, 0.15, 0.30], 15.0, 12.0, 0.31),
                ax([2.00, 0.12, 1.00], [0.42, 0.40, 0.18], 25.0, 20.0, 0.25),
            ],
        }
    }

    /// parse a Siemens `.asc` gradient-system file (PNS model; `cardiac` = the cardiac stimulation model instead)
    pub fn from_asc(text: &str, cardiac: bool) -> Result<Self, String> {
        let block = if cardiac { "GradPatSup.Phys.CarNS." } else { "GradPatSup.Phys.PNS." };
        let mut kv = std::collections::HashMap::new();
        let mut name = "unknown".to_string();
        for line in text.lines() {
            let l = line.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            let Some((k, v)) = l.split_once('=') else { continue };
            let (k, v) = (k.trim(), v.split(['#']).next().unwrap_or("").split("//").next().unwrap_or("").trim());
            if k.starts_with("asCOMP") && k.ends_with(".tName") {
                name = v.trim_matches('"').to_string();
            }
            let key = if let Some(r) = k.strip_prefix(block) {
                r.to_string()
            } else if k.starts_with("GradPatSup.") {
                continue;
            } else {
                k.rsplit('.').next().unwrap_or(k).to_string()
            };
            if let Ok(x) = v.parse::<f64>() {
                kv.entry(key).or_insert(x);
            }
        }
        let get = |k: &str| kv.get(k).copied().ok_or_else(|| format!("missing {k}"));
        let axis = |c: char| -> Result<SafeAxis, String> {
            let tau = [get(&format!("flGSWDTau{c}[0]"))?, get(&format!("flGSWDTau{c}[1]"))?, get(&format!("flGSWDTau{c}[2]"))?];
            let a = [get(&format!("flGSWDA{c}[0]"))?, get(&format!("flGSWDA{c}[1]"))?, get(&format!("flGSWDA{c}[2]"))?];
            let gs = get(&format!("flGScaleFactor{c}")).unwrap_or(1.0 / std::f64::consts::PI);
            Ok(SafeAxis { tau, a, stim_limit: get(&format!("flGSWDStimulationLimit{c}"))?, stim_thresh: get(&format!("flGSWDStimulationThreshold{c}"))?, g_scale: gs })
        };
        let hw = Self { name, axes: [axis('X')?, axis('Y')?, axis('Z')?] };
        for (i, ax) in hw.axes.iter().enumerate() {
            if (ax.a.iter().sum::<f64>() - 1.0).abs() > 1e-3 {
                return Err(format!("axis {i}: a1 + a2 + a3 ≠ 1"));
            }
        }
        Ok(hw)
    }

    pub fn longest_tau_ms(&self) -> f64 {
        self.axes.iter().flat_map(|a| a.tau).fold(0.0, f64::max)
    }
}

/// PNS of a waveform: per axis and combined, % of the threshold, one value per raster segment
#[derive(Clone, Debug)]
pub struct Pns {
    pub dt: f64,
    /// samples of zero padding before the waveform's first segment
    pub pad: usize,
    pub axis: Vec<Vec3>,
    pub norm: Vec<f64>,
    pub max: f64,
    pub argmax: usize,
}

fn lowpass(x: &[f64], tau_ms: f64, dt_ms: f64) -> Vec<f64> {
    let alpha = dt_ms / (tau_ms + dt_ms);
    let mut y = Vec::with_capacity(x.len());
    let mut p = 0.0;
    for &v in x {
        p = alpha * v + (1.0 - alpha) * p;
        y.push(p);
    }
    y
}

/// SAFE model on the physical-axis waveform `g`, with SAFE's zero padding (longest τ before, 4 × after)
pub fn pns(g: &Grad, hw: &SafeHw) -> Pns {
    let dt = g.dt;
    let zpt = hw.longest_tau_ms() * 4.0 / 1000.0;
    let (pad1, pad2) = ((zpt / 4.0 / dt).round() as usize, (zpt / dt).round() as usize);
    let n = g.len() + pad1 + pad2;
    let gw = |i: usize, a: usize| if i >= pad1 && i - pad1 < g.len() { g.g[i - pad1][a] } else { 0.0 };
    let dt_ms = dt * 1e3;
    let mut axis = vec![[0.0; 3]; n - 1];
    for a in 0..3 {
        let s: Vec<f64> = (0..n - 1).map(|i| (gw(i + 1, a) - gw(i, a)) / dt).collect();
        let sa: Vec<f64> = s.iter().map(|x| x.abs()).collect();
        let h = &hw.axes[a];
        let (l1, l2, l3) = (lowpass(&s, h.tau[0], dt_ms), lowpass(&sa, h.tau[1], dt_ms), lowpass(&s, h.tau[2], dt_ms));
        for i in 0..n - 1 {
            axis[i][a] = (h.a[0] * l1[i].abs() + h.a[1] * l2[i] + h.a[2] * l3[i].abs()) / h.stim_limit * h.g_scale * 100.0;
        }
    }
    let norm: Vec<f64> = axis.iter().map(crate::norm).collect();
    let (argmax, max) = norm.iter().enumerate().fold((0, 0.0), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
    Pns { dt, pad: pad1, axis, norm, max, argmax }
}

/// steady state of a repeated block (one TR): PNS of `repeats` back-to-back copies, maximum over the last copy
pub fn pns_periodic(tr: &Grad, hw: &SafeHw, repeats: usize) -> f64 {
    let mut g = tr.clone();
    for _ in 1..repeats {
        g.g.extend_from_slice(&tr.g[1..]);
    }
    let p = pns(&g, hw);
    let start = p.pad + (repeats - 1) * (tr.len() - 1);
    p.norm[start..(start + tr.len() - 1).min(p.norm.len())].iter().cloned().fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wave::trap_shape;

    #[test]
    fn long_constant_slew_converges_to_the_static_value() {
        // a 10 ms ramp at constant slew s on x: all low-passes converge to s, so stim → s · g_scale / limit · 100
        let dt = 10e-6;
        let s = 40.0;
        let n = 1000;
        let ramp: Vec<f64> = (0..=n).map(|i| s * i as f64 * dt).collect();
        let w = Grad::from_axis(0, &ramp, dt, 0.0);
        let hw = SafeHw::example();
        let p = pns(&w, &hw);
        let x = &hw.axes[0];
        let expect = s * x.g_scale / x.stim_limit * 100.0;
        let at_end = p.axis[p.pad + n - 1][0];
        assert!((at_end - expect).abs() / expect < 0.05, "{at_end} vs {expect}"); // τ₃ = 3 ms not fully settled after 10 ms
        // (the waveform then drops to zero in one raster step: exclude that artificial step)
        let ramp_max = p.norm[..p.pad + n].iter().cloned().fold(0.0, f64::max);
        assert!(ramp_max <= expect * 1.001, "{ramp_max} vs {expect}");
    }

    #[test]
    fn asc_parsing() {
        let text = "### ASCCONV BEGIN ###\nasCOMP[0].tName = \"TEST_GPA\"\n\
GradPatSup.Phys.PNS.flGSWDTauX[0] = 0.2\nGradPatSup.Phys.PNS.flGSWDTauX[1] = 0.03\nGradPatSup.Phys.PNS.flGSWDTauX[2] = 3\n\
GradPatSup.Phys.PNS.flGSWDAX[0] = 0.4\nGradPatSup.Phys.PNS.flGSWDAX[1] = 0.1\nGradPatSup.Phys.PNS.flGSWDAX[2] = 0.5\n\
GradPatSup.Phys.PNS.flGSWDStimulationLimitX = 30\nGradPatSup.Phys.PNS.flGSWDStimulationThresholdX = 24\n\
GradPatSup.Phys.PNS.flGSWDTauY[0] = 1.5\nGradPatSup.Phys.PNS.flGSWDTauY[1] = 2.5\nGradPatSup.Phys.PNS.flGSWDTauY[2] = 0.15\n\
GradPatSup.Phys.PNS.flGSWDAY[0] = 0.55\nGradPatSup.Phys.PNS.flGSWDAY[1] = 0.15\nGradPatSup.Phys.PNS.flGSWDAY[2] = 0.3\n\
GradPatSup.Phys.PNS.flGSWDStimulationLimitY = 15\nGradPatSup.Phys.PNS.flGSWDStimulationThresholdY = 12\n\
GradPatSup.Phys.PNS.flGSWDTauZ[0] = 2\nGradPatSup.Phys.PNS.flGSWDTauZ[1] = 0.12\nGradPatSup.Phys.PNS.flGSWDTauZ[2] = 1\n\
GradPatSup.Phys.PNS.flGSWDAZ[0] = 0.42\nGradPatSup.Phys.PNS.flGSWDAZ[1] = 0.4\nGradPatSup.Phys.PNS.flGSWDAZ[2] = 0.18\n\
GradPatSup.Phys.PNS.flGSWDStimulationLimitZ = 25\nGradPatSup.Phys.PNS.flGSWDStimulationThresholdZ = 20\n\
asGPAParameters[0].sGCParameters.flGScaleFactorX = 0.35\nasGPAParameters[0].sGCParameters.flGScaleFactorY = 0.31\n\
asGPAParameters[0].sGCParameters.flGScaleFactorZ = 0.25\n### ASCCONV END ###\n";
        let hw = SafeHw::from_asc(text, false).unwrap();
        let ex = SafeHw::example();
        assert_eq!(hw.name, "TEST_GPA");
        for a in 0..3 {
            assert_eq!(hw.axes[a].tau, ex.axes[a].tau);
            assert_eq!(hw.axes[a].a, ex.axes[a].a);
            assert_eq!(hw.axes[a].g_scale, ex.axes[a].g_scale);
        }
        let w = Grad::from_axis(1, &trap_shape(20, 30, 20, 0.04), 10e-6, 0.0);
        assert!((pns(&w, &hw).max - pns(&w, &ex).max).abs() < 1e-12);
    }
}
