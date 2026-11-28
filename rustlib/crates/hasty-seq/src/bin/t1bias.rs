//! EPG bias study for no-prep flip × TR schedules (research/t1_ivim_dti_sequences.md §2.3d): prints the steady-state
//! echo signal (CSV) of a cyclic train of (flip, TR) blocks under RF spoiling, for several tissues, B1 scales and RF
//! phase increments. `python research/experiments/t1_crlb/t1bias.py` fits the ideal-spoiling model to it.
//!
//! usage: t1bias <schedule> [te_ms] [kmax]   schedule = "flip:tr_ms:seconds,flip:tr_ms:seconds,..." (flip in deg)
use hasty_seq::epg::{run_train, spoil_phases, Epg, Pulse, TrainOpts};
use hasty_seq::spoil::Tissue;

/// |signal| of the last cycle: `ncyc` cycles of `blocks`, increments cycling through `incl` per cycle
fn sig(blocks: &[(f64, f64, usize)], t1: f64, t2: f64, b1: f64, te: f64, kmax: usize, incl: &[f64], ideal: bool, ncyc: usize) -> Vec<f64> {
    let tis = Tissue { t1, t2 };
    let mut e = Epg::new(kmax);
    let mut n = 0usize;
    let mut out = Vec::new();
    for c in 0..ncyc {
        let inc = incl[c % incl.len()].to_radians();
        if c == ncyc - 1 {
            out.clear();
        }
        for &(flip, tr, cnt) in blocks {
            let ph = spoil_phases(inc, n, cnt);
            let pulses: Vec<Pulse> = ph.iter().map(|&p| Pulse { flip, tr, phase: p }).collect();
            let s = run_train(&mut e, &pulses, &tis, &TrainOpts { b1, te, units: 1, ideal });
            n += cnt;
            if c == ncyc - 1 {
                out.extend(s.iter().map(|v| v.norm()));
            }
        }
    }
    out
}

/// relative-parameter Fisher κ_T1 = sqrt((J⁻¹)_T1T1) for parameters `idx` of (rho, T1, B1, T2) (σ = 1 per pulse, scan of
/// `t_scan` seconds repeating the cycle); None if J is (numerically) singular
fn kappa(blocks: &[(f64, f64, usize)], t1: f64, t2: f64, b1: f64, te: f64, kmax: usize, incl: &[f64], ideal: bool, free: &[usize], t_scan: f64) -> Option<f64> {
    let p0 = [1.0, t1, b1, t2];
    let ncyc = 14;
    let f = |p: &[f64; 4]| -> Vec<f64> { sig(blocks, p[1], p[3], p[2], te, kmax, incl, ideal, ncyc).iter().map(|v| v * p[0]).collect() };
    let dur: f64 = blocks.iter().map(|b| b.1 * b.2 as f64).sum();
    let mut d: Vec<Vec<f64>> = Vec::new();
    for &i in free {
        let h = 1e-4;
        let (mut pp, mut pm) = (p0, p0);
        pp[i] *= 1.0 + h;
        pm[i] *= 1.0 - h;
        let (sp, sm) = (f(&pp), f(&pm));
        d.push(sp.iter().zip(&sm).map(|(a, b)| (a - b) / (2.0 * h)).collect()); // d s / d ln p
    }
    let m = free.len();
    let mut j = vec![vec![0.0; m]; m];
    for a in 0..m {
        for b in 0..m {
            j[a][b] = d[a].iter().zip(&d[b]).map(|(x, y)| x * y).sum::<f64>() * t_scan / dur;
        }
    }
    // normalise diagonal, invert by Gauss-Jordan with pivoting; singular → None
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
    Some((aug[it][m + it] / (sc[it] * sc[it])).sqrt())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.get(4).map(|s| s == "fisher").unwrap_or(false) {
        return fisher(&a);
    }
    let sched = a.get(1).expect("schedule");
    let te = a.get(2).map(|s| s.parse::<f64>().unwrap()).unwrap_or(2.0) * 1e-3;
    let kmax: usize = a.get(3).map(|s| s.parse().unwrap()).unwrap_or(150);
    // (flip rad, tr s, count)
    let blocks: Vec<(f64, f64, usize)> = sched
        .split(',')
        .map(|b| {
            let p: Vec<f64> = b.split(':').map(|x| x.parse().unwrap()).collect();
            (p[0].to_radians(), p[1] * 1e-3, (p[2] / (p[1] * 1e-3)).round() as usize)
        })
        .collect();
    let tissues = [("fat", 0.29, 0.07), ("WM", 0.83, 0.07), ("GM", 1.33, 0.095), ("blood", 1.75, 0.15), ("CSF", 4.2, 1.5)];
    let incs: [(&str, Vec<f64>); 4] = [("117", vec![117.0]), ("50", vec![50.0]), ("150", vec![150.0]), ("alt117/50", vec![117.0, 50.0])];
    let cyc: usize = blocks.iter().map(|b| b.2).sum();
    let ncyc = 14;
    println!("tissue,b1,inc,block,idx,flip_deg,tr_ms,re,im");
    for (tn, t1, t2) in tissues {
        let tis = Tissue { t1, t2 };
        for b1 in [1.0, 0.8] {
            for (iname, incl) in &incs {
                // one increment per cycle-block-pair alternates: alt uses incl[cycle index % len]
                let mut e = Epg::new(kmax);
                let mut n = 0usize;
                for c in 0..ncyc {
                    let inc = incl[c % incl.len()].to_radians();
                    for (bi, &(flip, tr, cnt)) in blocks.iter().enumerate() {
                        let ph = spoil_phases(inc, n, cnt);
                        let pulses: Vec<Pulse> = ph.iter().map(|&p| Pulse { flip, tr, phase: p }).collect();
                        let s = run_train(&mut e, &pulses, &tis, &TrainOpts { b1, te, units: 1, ideal: false });
                        n += cnt;
                        if c >= ncyc - 2 {
                            for (i, v) in s.iter().enumerate() {
                                println!("{tn},{b1},{iname},{},{},{:.2},{:.1},{:.9e},{:.9e}", bi + (c - (ncyc - 2)) * blocks.len(), i, flip.to_degrees(), tr * 1e3, v.re, v.im);
                            }
                        }
                    }
                }
                let _ = cyc;
            }
        }
    }
}

fn fisher(a: &[String]) {
    let sched = &a[1];
    let te = a[2].parse::<f64>().unwrap() * 1e-3;
    let kmax: usize = a[3].parse().unwrap();
    let blocks: Vec<(f64, f64, usize)> = sched
        .split(',')
        .map(|b| {
            let p: Vec<f64> = b.split(':').map(|x| x.parse().unwrap()).collect();
            (p[0].to_radians(), p[1] * 1e-3, (p[2] / (p[1] * 1e-3)).round() as usize)
        })
        .collect();
    let tissues = [("fat", 0.29, 0.07), ("WM", 0.83, 0.07), ("GM", 1.33, 0.095), ("blood", 1.75, 0.15), ("CSF", 4.2, 1.5)];
    let incs: [(&str, Vec<f64>); 3] = [("117", vec![117.0]), ("150", vec![150.0]), ("alt117/150", vec![117.0, 150.0])];
    println!("tissue,b1,inc,model,free,kappa");
    for (tn, t1, t2) in tissues {
        for b1 in [1.0, 0.8] {
            for (iname, incl) in &incs {
                // ideal Ernst model (rho,T1,B1) ; EPG with T2 known (rho,T1,B1) ; EPG with T2 free (rho,T1,B1,T2)
                for (model, ideal, free) in [("ernst", true, vec![0usize, 1, 2]), ("epg", false, vec![0, 1, 2]), ("epg", false, vec![0, 1, 2, 3])] {
                    let k = kappa(&blocks, t1, t2, b1, te, kmax, incl, ideal, &free, 480.0);
                    println!("{tn},{b1},{iname},{model},{},{}", free.len(), k.map(|v| format!("{v:.5}")).unwrap_or("NONID".into()));
                }
            }
        }
    }
}
