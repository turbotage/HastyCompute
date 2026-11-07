//! The chosen B0 / R2* / fat mapping sequence (WP3): short out-and-back lobe pairs, ½-lobe TE shift on alternate TRs,
//! slew fitted to the PNS limit (hasty-seq `multiecho`). Writes the design and the arrays of the final-design figures.
//!
//!     b0seq <out> [--res 2e-3] [--k 24] [--shape trap|sine] [--pns 80] [--n-tr 4]
//!
//! <out>/final.json (+ design numbers), seq_grad.npy [n, 3] / seq_part.npy [n] / seq_pns.npy [n−1, 4] (n-tr consecutive
//! TRs, golden-means pair directions, shifts alternating), tr_grad.npy / tr_moments.npy [2, n, 3 | 6] (one TR per TE
//! shift; moments from the RF centre: k cycles/m, M1 T·s²/m), k_spokes.npy [2K, lobe, 3] (every lobe of the first two
//! TRs), girf.npy [n, 7] (t, nominal xyz, played xyz through `Girf::clinical_example`), echoes.json.
//! Figures: tools/seq/plot_final.py <out>.

use std::path::{Path, PathBuf};

use hasty_phantom::volume::write_npy;
use hasty_seq::girf::Girf;
use hasty_seq::multiecho::{fit_slew_to_pns, golden_dirs, train_pns, EchoSpec, EchoTrain, LobeShape};
use hasty_seq::pns::{pns, SafeHw};
use hasty_seq::system::System;
use hasty_seq::wave::Grad;
use hasty_seq::{Vec3, GAMMA};
use serde_json::json;

fn f8(path: &Path, shape: &[usize], v: &[f64]) {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_npy(path, "<f8", shape, false, &b).unwrap();
}
fn flat3(v: &[Vec3]) -> Vec<f64> {
    v.iter().flat_map(|p| p.iter().copied()).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(args.get(1).expect("usage: b0seq <out> [options]"));
    let opt = |n: &str| args.iter().position(|a| a == n).map(|i| args[i + 1].clone());
    let res: f64 = opt("--res").map_or(2e-3, |s| s.parse().unwrap());
    let k: usize = opt("--k").map_or(24, |s| s.parse().unwrap());
    let pns_lim: f64 = opt("--pns").map_or(80.0, |s| s.parse().unwrap());
    let n_tr: usize = opt("--n-tr").map_or(4, |s| s.parse().unwrap());
    let shape = match opt("--shape").as_deref() {
        Some("sine") => LobeShape::Sine,
        _ => LobeShape::Trapezoid,
    };
    std::fs::create_dir_all(&out).unwrap();
    let sys = System::prisma_xr();
    let hw = SafeHw::example();

    let mut spec = EchoSpec::new(res, k);
    spec.shape = shape;
    let t0 = fit_slew_to_pns(&spec, &sys, &hw, pns_lim, 6);
    spec.slew_frac = t0.spec.slew_frac;
    spec.te_shifts = vec![0.0, sys.ceil_raster(0.5 * t0.t_lobe)];
    let tr = EchoTrain::design(&spec, &sys);
    let (p_real, p_axis) = train_pns(&tr, &sys, &hw, 6);
    let dpt = tr.dirs_per_tr();
    let t_rf = sys.rf_dead + 0.5 * spec.rf_dur;

    // n_tr consecutive TRs
    let mut g: Vec<Vec3> = Vec::new();
    let mut part: Vec<f64> = Vec::new();
    let mut tr_g = Vec::new();
    let mut tr_m = Vec::new();
    for t in 0..n_tr {
        let s = spec.te_shifts[t % 2];
        let dirs = golden_dirs(t * dpt, dpt);
        let w = tr.tr_waveform(&dirs, s, &sys);
        let i0 = ((t_rf + tr.te1 + s) / tr.dt).round() as usize;
        let i1 = i0 + spec.n_lobe * (tr.lobe.len() - 1);
        let i2 = i1 + tr.spoiler.len() - 1;
        let codes: Vec<f64> = (0..w.len()).map(|i| if i < i0 { 0.0 } else if i < i1 { 2.0 } else if i < i2 { 3.0 } else { 0.0 }).collect();
        if t < 2 {
            let m = w.moments(t_rf);
            let kk = m.k();
            tr_g.extend(flat3(&w.g));
            for i in 0..kk.len() {
                tr_m.extend_from_slice(&kk[i]);
                tr_m.extend_from_slice(&m.m1[i]);
            }
        }
        let skip = if g.is_empty() { 0 } else { 1 };
        g.extend_from_slice(&w.g[skip..]);
        part.extend_from_slice(&codes[skip..]);
    }
    let n_one = (tr.tr / tr.dt).round() as usize + 1;
    let sq = Grad { dt: tr.dt, t0: 0.0, g };
    f8(&out.join("seq_grad.npy"), &[sq.len(), 3], &flat3(&sq.g));
    f8(&out.join("seq_part.npy"), &[part.len()], &part);
    let pr = pns(&sq, &hw);
    let mut a = vec![];
    for (ax, nn) in pr.axis.iter().zip(&pr.norm).skip(pr.pad).take(sq.len() - 1) {
        a.extend_from_slice(ax);
        a.push(*nn);
    }
    f8(&out.join("seq_pns.npy"), &[a.len() / 4, 4], &a);
    f8(&out.join("tr_grad.npy"), &[2, n_one, 3], &tr_g);
    f8(&out.join("tr_moments.npy"), &[2, n_one, 6], &tr_m);

    // every lobe of the first two TRs in k
    let nl = tr.lobe.len();
    let mut ks = vec![];
    for t in 0..2 {
        let ro = tr.readout(&golden_dirs(t * dpt, dpt), 0.0);
        let kk = ro.moments(0.0).k();
        for j in 0..spec.n_lobe {
            ks.extend(flat3(&kk[j * (nl - 1)..=(j + 1) * (nl - 1)]));
        }
    }
    f8(&out.join("k_spokes.npy"), &[2 * spec.n_lobe, nl, 3], &ks);

    // GIRF of the first TR
    let w0 = tr.tr_waveform(&golden_dirs(0, dpt), 0.0, &sys);
    let (ga, _) = Girf::clinical_example().apply(&w0);
    let mut gi = vec![];
    for i in 0..w0.len() {
        gi.push(i as f64 * w0.dt);
        gi.extend_from_slice(&w0.g[i]);
        gi.extend_from_slice(&ga.g[i + Girf::PRE]);
    }
    f8(&out.join("girf.npy"), &[w0.len(), 7], &gi);

    let lim = sq.limits();
    let m1 = tr.m1_at_echoes(&golden_dirs(0, dpt));
    let echoes = json!({
        "te_ms": spec.te_shifts.iter().map(|&s| tr.te(s).iter().map(|t| t * 1e3).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "adc_ms": tr.adc_time().iter().map(|t| t * 1e3).collect::<Vec<_>>(),
        "m1_norm": m1.iter().map(hasty_seq::norm).collect::<Vec<_>>(),
        "flow_phase_rad_per_m_s": m1.iter().map(|v| GAMMA * hasty_seq::norm(v)).collect::<Vec<_>>(),
    });
    std::fs::write(out.join("echoes.json"), serde_json::to_string_pretty(&echoes).unwrap()).unwrap();
    let fin = json!({"kind": "b0_multiecho", "dt": tr.dt, "tr": tr.tr, "te": tr.te(0.0), "te_shift": spec.te_shifts, "t_rf": t_rf, "kmax": tr.kmax, "res": res,
                     "gmax": sys.gmax, "smax": sys.smax, "slew_factors": {"readout": spec.slew_frac, "spoiler": spec.slew_frac}, "pns_limit": pns_lim, "pns_hw": hw.name,
                     "pns_train": p_real, "pns_train_one_axis": p_axis, "pns_window_max": pr.max, "g_peak": lim.g_vec, "s_peak": lim.s_vec,
                     "n_lobe": spec.n_lobe, "t_lobe": tr.t_lobe, "te1": tr.te1, "m1_lobe": tr.m1_lobe, "shape": format!("{shape:?}"), "polarity": "Pairs",
                     "n_tr_window": n_tr, "variants": ["TE shift 0", format!("TE shift {:.2} ms", spec.te_shifts[1] * 1e3)],
                     "parts": ["RF / idle", "-", "readout lobes (ADC)", "spoiler"], "girf": "clinical_example (illustrative)",
                     "title": format!("B0/R2*/fat train: {} {:?} lobes ({} pairs), {:.1} mm, lobe {:.2} ms, TR {:.2} ms", spec.n_lobe, shape, spec.n_lobe / 2, res * 1e3, tr.t_lobe * 1e3, tr.tr * 1e3)});
    std::fs::write(out.join("final.json"), serde_json::to_string_pretty(&fin).unwrap()).unwrap();
    println!(
        "b0seq: {} lobes of {:.3} ms ({:?}), slew {:.2} × S_max (PNS {:.1} % real directions, worst one-axis {:?}), peak |G| {:.1} mT/m, TE₁ {:.3} ms, TR {:.3} ms, shift {:.3} ms, γM1 odd {:.2} rad/(m/s) → {}",
        spec.n_lobe,
        tr.t_lobe * 1e3,
        shape,
        spec.slew_frac,
        p_real,
        p_axis.map(|x| (x * 10.0).round() / 10.0),
        lim.g_vec * 1e3,
        tr.te1 * 1e3,
        tr.tr * 1e3,
        spec.te_shifts[1] * 1e3,
        GAMMA * tr.m1_lobe,
        out.display()
    );
}
