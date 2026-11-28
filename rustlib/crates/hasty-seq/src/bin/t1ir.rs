//! EPG study of inversion-prepared (or unprepared) cyclic flip/TR trains over a wide T1 range with a gadolinium model
//! (research/t1_ivim_dti_sequences.md §2.3f). Cases: T1 grid of one "series" (CSF or tissue uptake); T2 follows from the
//! concentration: C = (1/T1 − 1/T1₀)/r1, 1/T2 = 1/T2₀ + r2 C.
//!
//! usage: t1ir <sig|fisher> <ir|none> <segs> <series: csf|gm|wm> [te_ms] [r1] [r2]
//!   segs = "flip_deg:tr_ms:n,..." played after the inversion (a hard 180° · B1 pulse, crusher, 15 ms gap) every period
//!   sig    : CSV of the |signal| of the last period (for fitting the ideal model in python)
//!   fisher : σ_R1 (1/s, per unit SNR per TR, scan 480 s, weight = ADC time / 9.5 ms, ADC = min(15 ms, TR − 1.5 ms)) for the
//!            ideal model (ρ,T1,B1), the EPG with T2 known (ρ,T1,B1) and the EPG with T2 free (ρ,T1,B1,T2)
use hasty_seq::epg::{run_train, spoil_phases, Epg, Pulse, TrainOpts};
use hasty_seq::spoil::Tissue;

const GAP: f64 = 15e-3;
type Segs = Vec<(f64, f64, usize)>; // flip rad, tr s, n

fn parse_segs(s: &str) -> Segs {
    s.split(',')
        .map(|b| {
            let p: Vec<f64> = b.split(':').map(|x| x.parse().unwrap()).collect();
            (p[0].to_radians(), p[1] * 1e-3, p[2] as usize)
        })
        .collect()
}

/// |signal| of the last period (weights not applied)
fn period_sig(ir: bool, segs: &Segs, t1: f64, t2: f64, b1: f64, te: f64, kmax: usize, inc: f64, ideal: bool) -> Vec<f64> {
    let tis = Tissue { t1, t2 };
    let mut e = Epg::new(kmax);
    let mut n = 0usize;
    let mut out = Vec::new();
    let ncyc = 9;
    for c in 0..ncyc {
        if ir {
            e.rf(std::f64::consts::PI * b1, 0.0);
            e.crush(true);
            e.relax(GAP, &tis);
        }
        if c == ncyc - 1 {
            out.clear();
        }
        for &(flip, tr, cnt) in segs {
            let ph = spoil_phases(inc, n, cnt);
            let pulses: Vec<Pulse> = ph.iter().map(|&p| Pulse { flip, tr, phase: p }).collect();
            let s = run_train(&mut e, &pulses, &tis, &TrainOpts { b1, te, units: std::env::var("UNITS").ok().and_then(|v| v.parse().ok()).unwrap_or(1), ideal });
            n += cnt;
            if c == ncyc - 1 {
                out.extend(s.iter().map(|v| v.re)); // signed (phase-sensitive) signal
            }
        }
    }
    out
}

fn weights(segs: &Segs) -> Vec<f64> {
    let mut w = Vec::new();
    for &(_, tr, cnt) in segs {
        let wt = (15.0f64).min(tr * 1e3 - 1.5) / 9.5;
        w.extend(std::iter::repeat(wt).take(cnt));
    }
    w
}

fn period_dur(ir: bool, segs: &Segs) -> f64 {
    segs.iter().map(|s| s.1 * s.2 as f64).sum::<f64>() + if ir { GAP } else { 0.0 }
}

#[allow(clippy::too_many_arguments)]
fn sigma_r1(ir: bool, segs: &Segs, t1: f64, t2: f64, b1: f64, te: f64, inc: f64, ideal: bool, free: &[usize]) -> Option<f64> {
    let kmax = 150;
    let p0 = [1.0, t1, b1, t2];
    let w = weights(segs);
    let f = |p: &[f64; 4]| -> Vec<f64> {
        period_sig(ir, segs, p[1], p[3], p[2], te, kmax, inc, ideal).iter().zip(&w).map(|(v, w)| v * p[0] * w.sqrt()).collect()
    };
    let mut d: Vec<Vec<f64>> = Vec::new();
    for &i in free {
        let h = 1e-4;
        let (mut pp, mut pm) = (p0, p0);
        pp[i] *= 1.0 + h;
        pm[i] *= 1.0 - h;
        let (sp, sm) = (f(&pp), f(&pm));
        d.push(sp.iter().zip(&sm).map(|(a, b)| (a - b) / (2.0 * h)).collect());
    }
    let m = free.len();
    let reps = 480.0 / period_dur(ir, segs);
    let mut j = vec![vec![0.0; m]; m];
    for a in 0..m {
        for b in 0..m {
            j[a][b] = d[a].iter().zip(&d[b]).map(|(x, y)| x * y).sum::<f64>() * reps;
        }
    }
    let sc: Vec<f64> = (0..m).map(|a| j[a][a].sqrt().max(1e-300)).collect();
    let mut aug = vec![vec![0.0; 2 * m]; m];
    for a in 0..m {
        for b in 0..m {
            aug[a][b] = j[a][b] / (sc[a] * sc[b]);
        }
        aug[a][m + a] = 1.0;
    }
    for c in 0..m {
        let piv = (c..m).max_by(|&x, &y| aug[x][c].abs().partial_cmp(&aug[y][c].abs()).unwrap()).unwrap();
        if aug[piv][c].abs() < 1e-9 {
            return None;
        }
        aug.swap(c, piv);
        let pv = aug[c][c];
        for k in 0..2 * m {
            aug[c][k] /= pv;
        }
        for r in 0..m {
            if r != c {
                let fct = aug[r][c];
                for k in 0..2 * m {
                    let t = aug[c][k];
                    aug[r][k] -= fct * t;
                }
            }
        }
    }
    let it = free.iter().position(|&i| i == 1).unwrap();
    Some((aug[it][m + it] / (sc[it] * sc[it])).sqrt() / t1)
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a[1].as_str();
    let ir = a[2] == "ir";
    let segs = parse_segs(&a[3]);
    let series = a[4].as_str();
    let te = a.get(5).map(|s| s.parse::<f64>().unwrap()).unwrap_or(2.0) * 1e-3;
    let r1: f64 = a.get(6).map(|s| s.parse().unwrap()).unwrap_or(4.0);
    let r2: f64 = a.get(7).map(|s| s.parse().unwrap()).unwrap_or(5.0);
    let (t10, t20) = match series {
        "csf" => (4.3, 1.6),
        "gm" => (1.33, 0.095),
        "wm" => (0.83, 0.07),
        _ => panic!("series"),
    };
    let grid = [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.2, 3.0, 3.8, 4.3];
    let cases: Vec<(f64, f64)> = grid
        .iter()
        .filter(|&&t| t <= t10 + 1e-9)
        .map(|&t| {
            let c = ((1.0 / t - 1.0 / t10) / r1).max(0.0); // mM
            (t, 1.0 / (1.0 / t20 + r2 * c))
        })
        .collect();
    let incs_s = std::env::var("INCS").unwrap_or("117,150".into());
    let incs_n: Vec<(String, f64)> = incs_s.split(',').map(|x| (x.to_string(), x.parse::<f64>().unwrap())).collect();
    let incs: Vec<(&str, f64)> = incs_n.iter().map(|(n, v)| (n.as_str(), *v)).collect();
    match mode {
        "sig" => {
            println!("series,T1,T2,b1,inc,idx,flip_deg,tr_ms,val");
            let meta: Vec<(f64, f64)> = segs.iter().flat_map(|&(f, tr, n)| std::iter::repeat((f.to_degrees(), tr * 1e3)).take(n)).collect();
            for &(t1, t2) in &cases {
                for b1 in [1.0, 0.8] {
                    for &(iname, inc) in &incs {
                        let s = period_sig(ir, &segs, t1, t2, b1, te, 150, inc.to_radians(), false);
                        for (i, v) in s.iter().enumerate() {
                            println!("{series},{t1},{t2:.4},{b1},{iname},{i},{:.2},{:.1},{v:.9e}", meta[i].0, meta[i].1);
                        }
                    }
                }
            }
        }
        "eval" => {
            // stdin lines "T1,T2,B1" -> signed signal of the last period (first of INCS), one row per line
            use std::io::BufRead;
            let inc = incs[0].1.to_radians();
            for line in std::io::stdin().lock().lines() {
                let l = line.unwrap();
                if l.trim().is_empty() {
                    continue;
                }
                let v: Vec<f64> = l.split(',').map(|x| x.trim().parse().unwrap()).collect();
                let sg = period_sig(ir, &segs, v[0], v[1], v[2], te, 150, inc, false);
                println!("{}", sg.iter().map(|x| format!("{x:.9e}")).collect::<Vec<_>>().join(","));
            }
        }
        "dict" => {
            // EPG dictionary over (T1, T2, B1): rows "T1,T2,B1,s_0,...,s_{N-1}" (signed, phase-sensitive, rho = 1), one increment (first of INCS)
            let lst = |k: &str, d: &str| -> Vec<f64> { std::env::var(k).unwrap_or(d.into()).split(',').map(|x| x.parse().unwrap()).collect() };
            let (n1, lo, hi) = (lst("DT1", "24,0.3,4.5"), 0.0, 0.0);
            let _ = (lo, hi);
            let t1g: Vec<f64> = (0..n1[0] as usize).map(|i| (n1[1].ln() + (n1[2].ln() - n1[1].ln()) * i as f64 / (n1[0] - 1.0)).exp()).collect();
            let t2g = lst("DT2", "0.03,0.06,0.1,0.2,0.4,0.8,1.6,2.4");
            let b1g = lst("DB1", "0.7,0.8,0.9,1.0,1.1,1.2");
            let inc = incs[0].1.to_radians();
            for &t1 in &t1g {
                for &t2 in &t2g {
                    if t2 > 0.9 * t1 {
                        continue;
                    }
                    for &b1 in &b1g {
                        let sg = period_sig(ir, &segs, t1, t2, b1, te, 150, inc, false);
                        let row: Vec<String> = sg.iter().map(|v| format!("{v:.7e}")).collect();
                        println!("{t1:.5},{t2:.5},{b1:.3},{}", row.join(","));
                    }
                }
            }
        }
        "fisher" => {
            println!("series,T1,T2,b1,inc,model,free,sigma_R1");
            for &(t1, t2) in &cases {
                for b1 in [1.0, 0.8] {
                    for &(iname, inc) in &incs {
                        for (model, ideal, free) in [("ideal", true, vec![0usize, 1, 2]), ("epg", false, vec![0, 1, 2]), ("epg", false, vec![0, 1, 2, 3])] {
                            let k = sigma_r1(ir, &segs, t1, t2, b1, te, inc.to_radians(), ideal, &free);
                            println!("{series},{t1},{t2:.4},{b1},{iname},{model},{},{}", free.len(), k.map(|v| format!("{v:.5}")).unwrap_or("NONID".into()));
                        }
                    }
                }
            }
        }
        _ => panic!("mode"),
    }
}
