//! Echo design for B0 / R2* / fat mapping (WP3): short out-and-back multi-echo trains (hasty-seq `multiecho`) vs an
//! idealised long-readout acquisition like the 4D-flow TR, by the Cramér–Rao bound per unit scan time.
//!
//!     echodesign [--out data/hasty/seq/studies/echodesign] [--pns 80]
//!
//! Model: voxel-wise water–fat, six-peak fat, s(t) = (ρ_w + ρ_f c(t)) e^{−(R2* + iω)t} (hasty-qfit `waterfat`).
//! Noise: echo e's image gets the ADC time T_e it owns in every TR; over a scan of T seconds its noise variance is
//! σ₀² TR n_shift / (T T_e) (σ₀ = 1 per √s of ADC), and the SPGR signal at the Ernst angle S = √((1 − E1)/(1 + E1))
//! scales every amplitude. The Fisher information per second is therefore Σ_e w_e J_eᵀJ_e with w_e = S² T_e / (TR n_shift),
//! and the reported numbers are standard deviations for 1 s of scanning (σ ∝ 1/√T). Only ratios between designs at the
//! same resolution are meaningful (absolute SNR depends on coil, voxel and σ₀).
//!
//! The long readout is *idealised*: all of its ADC time is credited to its two k = 0 crossings (start and end of the
//! out-and-back), as if the k(t)–time coupling inside the readout cost nothing. Its real per-voxel information is
//! lower, since high k is only ever sampled mid-readout, and the off-resonance phase inside it (fat: 2π·434 Hz × 4 ms
//! ≈ 11 rad) has to be resolved by the encoding model.
//!
//! Swap margin: noise-free profile likelihood (amplitudes eliminated, fine (ω, R2*) grid); d² = R(second minimum) −
//! R(true) per second of scanning. A two-hypothesis decision errs with probability Φ(−√(d² T)/2), so T₁₀₋₃ = 38.2/d²
//! seconds gives a swap / alias probability of 1e-3 (relative, as above).
//!
//! Writes <out>/echodesign.json; figure: tools/seq/plot_echodesign.py <out>.

use std::path::PathBuf;

use hasty_qfit::stats::crlb;
use hasty_qfit::waterfat::{FatSpectrum, Grid, Profile, Species, WaterFat};
use hasty_seq::multiecho::{fit_slew_to_pns, golden_dirs, train_pns, EchoSpec, EchoTrain, LobeShape};
use hasty_seq::pns::SafeHw;
use hasty_seq::system::System;
use hasty_seq::GAMMA;
use serde_json::json;

struct Tissue {
    name: &'static str,
    pd: f64,
    t1: f64,
    r2: f64,
    ff: f64,
}

const TISSUES: [Tissue; 6] = [
    Tissue { name: "blood", pd: 0.85, t1: 1.90, r2: 20.0, ff: 0.0 },
    Tissue { name: "myocardium", pd: 0.75, t1: 1.50, r2: 35.0, ff: 0.0 },
    Tissue { name: "liver", pd: 0.70, t1: 0.80, r2: 45.0, ff: 0.05 },
    Tissue { name: "fat", pd: 0.90, t1: 0.38, r2: 60.0, ff: 0.92 },
    Tissue { name: "iron liver", pd: 0.70, t1: 0.60, r2: 300.0, ff: 0.05 },
    Tissue { name: "lung", pd: 0.15, t1: 1.30, r2: 700.0, ff: 0.0 },
];

/// one acquisition, reduced to what the voxel model sees: echo times (all TE shifts), ADC time per echo, TR
struct Acq {
    name: String,
    kind: &'static str,
    res: f64,
    te: Vec<f64>,
    t_adc: Vec<f64>,
    n_shift: usize,
    tr: f64,
    info: serde_json::Value,
}

fn ernst(tr: f64, t1: f64) -> f64 {
    let e1 = (-tr / t1).exp();
    ((1.0 - e1) / (1.0 + e1)).sqrt()
}

/// FF = |f| / (|w| + |f|) and its gradient w.r.t. [wr, wi, fr, fi] (at |f| = 0 along the water direction)
fn ff_grad(p: &[f64]) -> (f64, [f64; 4]) {
    let (w, f) = ((p[0] * p[0] + p[1] * p[1]).sqrt(), (p[2] * p[2] + p[3] * p[3]).sqrt());
    let s = w + f;
    let wh = [p[0] / w, p[1] / w];
    let fh = if f > 0.0 { [p[2] / f, p[3] / f] } else { wh };
    let (dw, df) = (-f / (s * s), w / (s * s));
    (f / s, [dw * wh[0], dw * wh[1], df * fh[0], df * fh[1]])
}

struct Eval {
    sd_om_hz: f64,
    sd_r2: f64,
    sd_ff: f64,
    d2: f64,
}

fn evaluate(acq: &Acq, keep: &[bool], t: &Tissue, b0: f64, om_hz: f64) -> Eval {
    let idx: Vec<usize> = (0..acq.te.len()).filter(|&i| keep[i]).collect();
    let te: Vec<f64> = idx.iter().map(|&i| acq.te[i]).collect();
    let s2 = ernst(acq.tr, t.t1).powi(2);
    let w: Vec<f64> = idx.iter().map(|&i| s2 * acq.t_adc[i] / (acq.tr * acq.n_shift as f64)).collect();
    let wf = WaterFat::new(Species::WaterFat, te.clone(), b0, FatSpectrum::six_peak());
    let om = 2.0 * std::f64::consts::PI * om_hz;
    let p = [t.pd * (1.0 - t.ff), 0.0, t.pd * t.ff, 0.0, t.r2, om];
    let nd = te.len();
    let np = 6;
    let Some(c) = crlb(&wf.model, &p, &wf.vars(), nd, &[], Some(&w), 1.0) else {
        return Eval { sd_om_hz: f64::NAN, sd_r2: f64::NAN, sd_ff: f64::NAN, d2: f64::NAN };
    };
    let (_, g) = ff_grad(&p);
    let mut v_ff = 0.0;
    for a in 0..4 {
        for b in 0..4 {
            v_ff += g[a] * c[a * np + b] * g[b];
        }
    }
    // noise-free profile on a fine grid: second-best local minimum over ω
    let mut r2n: Vec<f64> = (0..20).map(|i| 5.0 * i as f64).collect();
    r2n.extend((0..57).map(|i| 100.0 + 25.0 * i as f64));
    let om_max = 2.0 * std::f64::consts::PI * 700.0;
    let n_om = 351;
    let grid = Grid { om: (0..n_om).map(|i| -om_max + 2.0 * om_max * i as f64 / (n_om - 1) as f64).collect(), r2: r2n };
    let prof = Profile::new(&wf, &grid, Some(&w));
    let sig = wf.signal(&p);
    let y: Vec<f64> = sig.iter().map(|z| z.re).chain(sig.iter().map(|z| z.im)).collect();
    let cand = prof.candidates_cpu(&y, 1, 4);
    let mut r: Vec<f64> = cand.resid.iter().cloned().filter(|x| x.is_finite()).collect();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let d2 = if r.len() >= 2 { r[1] - r[0] } else { f64::INFINITY };
    Eval { sd_om_hz: c[5 * np + 5].sqrt() / (2.0 * std::f64::consts::PI), sd_r2: c[4 * np + 4].sqrt(), sd_ff: v_ff.sqrt(), d2 }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let opt = |n: &str| args.iter().position(|a| a == n).map(|i| args[i + 1].clone());
    let out = PathBuf::from(opt("--out").unwrap_or("data/hasty/seq/studies/echodesign".into()));
    std::fs::create_dir_all(&out).unwrap();
    let pns_lim: f64 = opt("--pns").map_or(80.0, |s| s.parse().unwrap());
    let sys = System::prisma_xr();
    let hw = SafeHw::example();
    let b0 = sys.b0;

    let mut acqs: Vec<Acq> = Vec::new();
    for &res in &[2e-3, 3e-3] {
        for &k in &[4usize, 8, 12, 16, 24, 32] {
            for shape in [LobeShape::Trapezoid, LobeShape::Sine] {
                for n_shift in [1usize, 2] {
                    if shape == LobeShape::Sine && n_shift == 2 {
                        continue;
                    }
                    let mut spec = EchoSpec::new(res, k);
                    spec.shape = shape;
                    let tr0 = fit_slew_to_pns(&spec, &sys, &hw, pns_lim, 6);
                    if n_shift == 2 {
                        spec.slew_frac = tr0.spec.slew_frac;
                        spec.te_shifts = vec![0.0, sys.ceil_raster(0.5 * tr0.t_lobe)];
                    }
                    let tr = if n_shift == 2 { EchoTrain::design(&spec, &sys) } else { tr0 };
                    let (p_real, p_axis) = train_pns(&tr, &sys, &hw, 6);
                    let mut te = Vec::new();
                    let mut ta = Vec::new();
                    for &s in &tr.spec.te_shifts {
                        te.extend(tr.te(s));
                        ta.extend(tr.adc_time());
                    }
                    let lim = tr.readout(&golden_dirs(0, tr.dirs_per_tr()), 0.0).limits();
                    let fat_hz = GAMMA / (2.0 * std::f64::consts::PI) * b0 * 3.4e-6;
                    acqs.push(Acq {
                        name: format!("short {:.0} mm K={k} {:?}{}", res * 1e3, shape, if n_shift == 2 { " +½ shift" } else { "" }),
                        kind: "short",
                        res,
                        te,
                        t_adc: ta,
                        n_shift,
                        tr: tr.tr,
                        info: json!({"K": k, "shape": format!("{shape:?}"), "n_shift": n_shift, "t_lobe_ms": tr.t_lobe * 1e3, "te1_ms": tr.te1 * 1e3,
                                     "tr_ms": tr.tr * 1e3, "slew_frac": tr.spec.slew_frac, "g_peak_mT_m": lim.g_vec * 1e3, "pns_real": p_real, "pns_axis": p_axis,
                                     "flow_phase_odd_rad_per_m_s": GAMMA * tr.m1_lobe, "fat_phase_at_kmax_rad": 2.0 * std::f64::consts::PI * fat_hz * 0.5 * tr.t_lobe,
                                     "adc_duty": tr.adc_time().iter().sum::<f64>() / tr.tr}),
                    });
                }
            }
        }
        // idealised long readouts (4D-flow-like out-and-back, T_ro 4 / 8 ms), 4 TE shifts over the fat period
        for &t_ro in &[4e-3, 8e-3] {
            let te1 = sys.ceil_raster(50e-6 + sys.rf_ringdown.max(sys.adc_dead));
            let shifts: Vec<f64> = (0..4).map(|s| s as f64 * 0.6e-3).collect();
            let tr = sys.ceil_raster(sys.rf_dead + 50e-6 + te1 + 1.8e-3 + t_ro + 1.0e-3);
            let mut te = Vec::new();
            let mut ta = Vec::new();
            for s in &shifts {
                te.extend([te1 + s, te1 + s + t_ro]);
                ta.extend([0.5 * t_ro, 0.5 * t_ro]);
            }
            let fat_hz = GAMMA / (2.0 * std::f64::consts::PI) * b0 * 3.4e-6;
            acqs.push(Acq {
                name: format!("long {:.0} mm T_ro={:.0} ms (idealised)", res * 1e3, t_ro * 1e3),
                kind: "long",
                res,
                te,
                t_adc: ta,
                n_shift: shifts.len(),
                tr,
                info: json!({"t_ro_ms": t_ro * 1e3, "tr_ms": tr * 1e3, "shifts_ms": shifts.iter().map(|s| s * 1e3).collect::<Vec<_>>(),
                             "fat_phase_at_mid_readout_rad": 2.0 * std::f64::consts::PI * fat_hz * 0.5 * t_ro, "adc_duty": t_ro / tr}),
            });
        }
    }

    // evaluate every design × echo selection × tissue
    struct Row {
        name: String,
        res: f64,
        sel: &'static str,
        tr: f64,
        info: serde_json::Value,
        n_echo: usize,
        ev: Vec<Eval>,
    }
    let mut table: Vec<Row> = Vec::new();
    for a in &acqs {
        for sel in ["all", "even"] {
            if sel == "even" && a.kind != "short" {
                continue;
            }
            let ne = a.te.len() / a.n_shift;
            let keep: Vec<bool> = (0..a.te.len()).map(|i| sel == "all" || (i % ne) % 2 == 0).collect();
            let ev = TISSUES.iter().map(|t| evaluate(a, &keep, t, b0, 50.0)).collect();
            table.push(Row { name: a.name.clone(), res: a.res, sel, tr: a.tr, info: a.info.clone(), n_echo: keep.iter().filter(|x| **x).count(), ev });
        }
    }
    // scan-time factors relative to the reference design at the same resolution: (sd / sd_ref)² for ω, R2*, FF and
    // d²_ref / d² for the swap margin = how much longer this design must scan for the same precision / reliability
    let refname = |res: f64| format!("short {:.0} mm K=16 Trapezoid", res * 1e3);
    let mut rows = Vec::new();
    for res in [2e-3, 3e-3] {
        let r0 = table.iter().find(|r| r.name == refname(res) && r.sel == "all").expect("reference design");
        println!("\n== {:.0} mm; reference {} [all]: per 1 s of scan (σ₀ = 1 per √s ADC) ==", res * 1e3, refname(res));
        for (t, e) in TISSUES.iter().zip(&r0.ev) {
            println!("   {:<11} sd ω {:.3e} Hz, sd R2* {:.3e} 1/s, sd FF {:.3e}, d² {:.3e}", t.name, e.sd_om_hz, e.sd_r2, e.sd_ff, e.d2);
        }
        println!("{:<46} {:>6} {:>6} {:>5} {:>5} {:>3} | scan-time factor vs reference: ω / R2* / FF / swap", "design", "TR ms", "Tlobe", "slew", "PNS%", "nE");
        println!("{:<74} | {}", "", TISSUES.iter().map(|t| format!("{:>27}", t.name)).collect::<String>());
        for r in table.iter().filter(|r| r.res == res) {
            let f = |k: &str| r.info.get(k).and_then(|v| v.as_f64()).unwrap_or(f64::NAN);
            let mut line = String::new();
            let mut per = Vec::new();
            for ((t, e), e0) in TISSUES.iter().zip(&r.ev).zip(&r0.ev) {
                let tf = [(e.sd_om_hz / e0.sd_om_hz).powi(2), (e.sd_r2 / e0.sd_r2).powi(2), (e.sd_ff / e0.sd_ff).powi(2), e0.d2 / e.d2];
                let fmt = |x: f64| if x < 1e3 { format!("{x:.2}") } else { format!("{x:.0e}") };
                line += &format!(" {:>6}/{:>5}/{:>5}/{:>6}", fmt(tf[0]), fmt(tf[1]), fmt(tf[2]), fmt(tf[3]));
                per.push(json!({"tissue": t.name, "sd_om_hz": e.sd_om_hz, "sd_r2": e.sd_r2, "sd_ff": e.sd_ff, "d2": e.d2,
                                "time_factor": {"om": tf[0], "r2": tf[1], "ff": tf[2], "swap": tf[3]}}));
            }
            println!("{:<46} {:>6.2} {:>6.3} {:>5.2} {:>5.1} {:>3} |{line}", format!("{} [{}]", r.name, r.sel), r.tr * 1e3, f("t_lobe_ms"), f("slew_frac"), f("pns_real"), r.n_echo);
            rows.push(json!({"name": r.name, "res_mm": r.res * 1e3, "echoes": r.sel, "tr_ms": r.tr * 1e3, "n_echo": r.n_echo, "info": r.info, "tissues": per,
                             "reference": refname(res)}));
        }
    }
    let summary = json!({"b0": b0, "pns_limit": pns_lim, "pns_model": hw.name, "omega_hz": 50.0,
                         "tissues": TISSUES.iter().map(|t| json!({"name": t.name, "pd": t.pd, "t1": t.t1, "r2": t.r2, "ff": t.ff})).collect::<Vec<_>>(),
                         "rows": rows});
    std::fs::write(out.join("echodesign.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("wrote {}", out.join("echodesign.json").display());
}
