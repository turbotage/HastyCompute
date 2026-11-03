//! Concomitant-phase structure of the simulated spiral 4D-flow sequence (research tool, see
//! research/concomitant_kspace_representation.md).
//!   conc_study <sim_dir> [n_readouts]
//! Per sample n: φ_n(x) = Σ_q w_nq m_q(x) in scanner coordinates; with x = c + Rρ (ρ logical, relative to the FOV
//! centre): φ = φ0_n + b_n·ρ + ρᵀ C_n ρ. Reports (1) the size of each part over the FOV, (2) how well C_n (6 entries)
//! and b_n can be written as functions of the sample's k-space position (polynomials in k, separate per encode).
use hasty_mrsim::sequence::SAMPLE_FLOATS;
use hasty_mrsim::{mat_mul, transpose, Mat3};
use hasty_recon::data::SimData;
use nalgebra::{DMatrix, DVector};

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let nro: usize = std::env::args().nth(2).map(|s| s.parse().unwrap()).unwrap_or(400);
    let sd = SimData::load(&dir).unwrap();
    let r = sd.rot;
    let c = sd.fov_c;
    let half = [0, 1, 2].map(|a| sd.meta["fov"][a].as_f64().unwrap() / 2.0);
    let na = sd.na;
    let step = (sd.acq.readouts.len() / nro).max(1);
    // rows: (encode, k (rad/m logical), C (6, rad/m²), b (3, rad/m), φ0)
    let mut rows: Vec<(usize, [f64; 3], [f64; 6], [f64; 3], f64, f64)> = Vec::new();
    for ro in (0..sd.acq.readouts.len()).step_by(step) {
        let sp = sd.acq.sample_params(ro..ro + 1);
        let enc = sd.acq.readouts[ro].enc;
        for n in (0..na).step_by(4) {
            let s = &sp[n * SAMPLE_FLOATS..(n + 1) * SAMPLE_FLOATS];
            let w = [s[9] as f64, s[10] as f64, s[11] as f64, s[12] as f64];
            // φ = w0 z² + w1 (x²+y²) + w2 xz + w3 yz = xᵀ Cp x
            let cp: Mat3 = [[w[1], 0.0, 0.5 * w[2]], [0.0, w[1], 0.5 * w[3]], [0.5 * w[2], 0.5 * w[3], w[0]]];
            let cl = mat_mul(&mat_mul(&transpose(&r), &cp), &r);
            let cpc = hasty_mrsim::mat_vec(&cp, &c);
            let b = hasty_mrsim::mat_vec(&transpose(&r), &cpc).map(|v| 2.0 * v);
            let phi0: f64 = (0..3).map(|a| c[a] * cpc[a]).sum();
            let m = sd.acq.moments_logical(ro, n);
            let k = m.m0.map(|v| hasty_mrsim::GAMMA * v);
            rows.push((enc, k, [cl[0][0], cl[1][1], cl[2][2], cl[0][1], cl[0][2], cl[1][2]], b, phi0, s[13] as f64));
        }
    }
    // (1) sizes over the FOV
    let mut q_max: f64 = 0.0;
    let mut l_max: f64 = 0.0;
    let mut p0_max: f64 = 0.0;
    for (_, _, cc, b, p0, _) in &rows {
        let quad = cc[0].abs() * half[0] * half[0] + cc[1].abs() * half[1] * half[1] + cc[2].abs() * half[2] * half[2] + 2.0 * (cc[3].abs() * half[0] * half[1] + cc[4].abs() * half[0] * half[2] + cc[5].abs() * half[1] * half[2]);
        q_max = q_max.max(quad);
        l_max = l_max.max((0..3).map(|a| b[a].abs() * half[a]).sum::<f64>());
        p0_max = p0_max.max(p0.abs());
    }
    println!("{} samples (every 4th ADC of {} readouts). FOV half-extent {:?} m", rows.len(), rows.len() / (na / 4), half);
    println!("max |constant| {:.2} rad, max |linear b·ρ| over FOV {:.2} rad, max |quadratic ρᵀCρ| over FOV {:.2} rad", p0_max, l_max, q_max);
    let kspace_shift: f64 = rows.iter().map(|r| (r.3[0] * r.3[0] + r.3[1] * r.3[1] + r.3[2] * r.3[2]).sqrt()).fold(0.0, f64::max);
    println!("max k-space shift |b| = {:.1} rad/m (= {:.2} k-cells of the 2 mm grid's 1/FOV)", kspace_shift, kspace_shift / (2.0 * std::f64::consts::PI / sd.meta["fov"][0].as_f64().unwrap()));
    // (2) representability: fit each C entry by polynomials in (kx, ky, kz) of total degree D, separately per encode,
    // error expressed as phase at the FOV corner (rad); also fit by readout time τ alone.
    let kmax = rows.iter().map(|r| r.1.iter().map(|v| v.abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
    for deg in [0usize, 2, 4, 6] {
        let mut monos: Vec<[usize; 3]> = Vec::new();
        for a in 0..=deg {
            for b in 0..=deg - a {
                for cc in 0..=deg - a - b {
                    monos.push([a, b, cc]);
                }
            }
        }
        let mut worst: f64 = 0.0;
        let mut rms_num = 0.0;
        let mut rms_den = 0.0;
        for enc in 0..4 {
            let sub: Vec<_> = rows.iter().filter(|r| r.0 == enc).collect();
            let a = DMatrix::<f64>::from_fn(sub.len(), monos.len(), |i, j| {
                let k = sub[i].1;
                (0..3).map(|d| (k[d] / kmax).powi(monos[j][d] as i32)).product()
            });
            let svd = a.clone().svd(true, true);
            for e in 0..6 {
                let y = DVector::<f64>::from_iterator(sub.len(), sub.iter().map(|r| r.2[e]));
                let x = svd.solve(&y, 1e-12).unwrap();
                let res = &a * x - &y;
                let scale = match e { 0 => half[0] * half[0], 1 => half[1] * half[1], 2 => half[2] * half[2], 3 => 2.0 * half[0] * half[1], 4 => 2.0 * half[0] * half[2], _ => 2.0 * half[1] * half[2] };
                worst = worst.max(res.amax() * scale);
                rms_num += res.norm_squared() * scale * scale;
                rms_den += y.norm_squared() * scale * scale;
            }
        }
        println!("C_n ≈ poly(k) of degree {deg} ({} terms, per encode): worst residual phase at FOV corner {:.3} rad, relative rms {:.2e}", monos.len(), worst, (rms_num / rms_den).sqrt());
    }
    // fit by readout time τ alone (per encode, polynomial degree 6)
    let mut worst: f64 = 0.0;
    for enc in 0..4 {
        let sub: Vec<_> = rows.iter().filter(|r| r.0 == enc).collect();
        let tmax = sub.iter().map(|r| r.5).fold(0.0, f64::max);
        let a = DMatrix::<f64>::from_fn(sub.len(), 7, |i, j| (sub[i].5 / tmax).powi(j as i32));
        let svd = a.clone().svd(true, true);
        for e in 0..6 {
            let y = DVector::<f64>::from_iterator(sub.len(), sub.iter().map(|r| r.2[e]));
            let x = svd.solve(&y, 1e-12).unwrap();
            let scale = match e { 0 => half[0] * half[0], 1 => half[1] * half[1], 2 => half[2] * half[2], 3 => 2.0 * half[0] * half[1], 4 => 2.0 * half[0] * half[2], _ => 2.0 * half[1] * half[2] };
            worst = worst.max((&a * x - &y).amax() * scale);
        }
    }
    println!("C_n ≈ poly(readout time) degree 6 per encode: worst residual phase at FOV corner {:.3} rad", worst);

    // ranks of the joint phase ψ(n, u) for the reference encode
    let nv = sd.nv();
    let rss: Vec<f32> = (0..nv).map(|q| (0..sd.nc).map(|cc| { let z = sd.smaps_true[cc * nv + q]; z.re * z.re + z.im * z.im }).sum::<f32>()).collect();
    let mut srt = rss.clone();
    srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let thr = srt[srt.len() / 2];
    let body: Vec<usize> = (0..nv).filter(|&q| rss[q] > thr && sd.field_ppm[q] != 0.0).collect();
    let wfac = 2.0 * std::f64::consts::PI * hasty_mrsim::GAMMA_BAR * sd.b0 * 1e-6;
    let rho: Vec<[f64; 3]> = (0..nv).map(|q| sd.voxel_log(q)).collect();
    let samp: Vec<&(usize, [f64; 3], [f64; 6], [f64; 3], f64, f64)> = rows.iter().filter(|r| r.0 == 0).collect();
    let quad = |j: usize, u: usize| -> f64 {
        let cc = samp[j].2;
        let p = rho[u];
        cc[0] * p[0] * p[0] + cc[1] * p[1] * p[1] + cc[2] * p[2] * p[2] + 2.0 * (cc[3] * p[0] * p[1] + cc[4] * p[0] * p[2] + cc[5] * p[1] * p[2])
    };
    let lin = |j: usize, u: usize| -> f64 { (0..3).map(|a| samp[j].3[a] * rho[u][a]).sum::<f64>() };
    let b0 = |j: usize, u: usize| -> f64 { wfac * sd.field_ppm[u] as f64 * samp[j].5 };
    if std::env::var("B0_SCALING").is_ok() {
        // B0-only rank vs readout length: stretch the readout-time axis about TE (phase variation over the readout ∝ T_ro)
        let te = samp.iter().map(|r| r.5).fold(f64::MAX, f64::min);
        for stretch in [1.0, 2.0, 3.0, 4.0] {
            let (fp, sm) = (&sd.field_ppm, &samp);
            let f = move |j: usize, u: usize| wfac * fp[u] as f64 * (te + stretch * (sm[j].5 - te));
            let mut ls = Vec::new();
            for tol in [1e-2, 2e-3, 1e-3] {
                let sk = hasty_recon::skeleton::build(&f, samp.len(), &body, 1200, tol, 120);
                ls.push(format!("L({tol:.0e}) = {}", sk.rows.len()));
            }
            println!("readout {:.1} ms (×{stretch}): B0-only {}", 4.42 * stretch, ls.join(", "));
        }
        return;
    }
    for deg in [0usize] {
        banded(&sd, &samp, &body, &rho, wfac, deg);
    }
    return;
    #[allow(unreachable_code)]
    println!("rank of exp(-iψ) over {} samples × {} body voxels (reference encode):", samp.len(), body.len());
    for tol in [1e-2, 2e-3] {
        let cases: Vec<(&str, Box<dyn Fn(usize, usize) -> f64 + Sync>)> = vec![
            ("B0 only", Box::new(move |j, u| b0(j, u))),
            ("quadratic concomitant (linear in trajectory)", Box::new(move |j, u| quad(j, u))),
            ("quadratic + linear concomitant", Box::new(move |j, u| quad(j, u) + lin(j, u))),
            ("B0 + quadratic concomitant", Box::new(move |j, u| b0(j, u) + quad(j, u))),
            ("B0 + quadratic + linear", Box::new(move |j, u| b0(j, u) + quad(j, u) + lin(j, u))),
        ];
        for (name, f) in &cases {
            let sk = hasty_recon::skeleton::build(f.as_ref(), samp.len(), &body, 1200, tol, 80);
            println!("  tol {tol:.0e}: {name:<46} L = {:>2} (held-out {:.1e})", sk.rows.len(), sk.heldout_err);
        }
    }
}

/// Low-rank (time-segmentation) rank needed for B0, quadratic concomitant (constant/linear removed) and both, at two
/// tolerances, via the skeleton (CUR) builder on body voxels.
#[allow(dead_code)]
pub fn ranks() {}

type Row = (usize, [f64; 3], [f64; 6], [f64; 3], f64, f64);

/// Time segmentation banded in readout time τ, segment diagonals carrying B0 and the τ-mean quadratic concomitant phase
/// D_ℓ = exp(−i[ω τ_ℓ + ρᵀ C̄(τ_ℓ) ρ]); per-sample least-squares interpolators over the L segments, optionally with the
/// first-order correction (1 − i ρᵀ ΔC_n ρ), ΔC_n = C_n − C̄(τ_n), ψ = ω τ + ρᵀCρ (linear part in the trajectory).
fn banded(sd: &SimData, samp: &[&Row], body: &[usize], rho: &[[f64; 3]], wfac: f64, bdeg: usize) {
    use num_complex::Complex64 as C;
    // B0 split: ω = p(ρ) + ω_res with p a polynomial of degree bdeg over the body; constant → per-sample phase, linear →
    // time-dependent trajectory shift (both exact, dropped), quadratic → Q added to C_n as Q·τ_n
    let mono: Vec<[usize; 3]> = (0..=bdeg).flat_map(|d| (0..=d).flat_map(move |a| (0..=d - a).map(move |b| [a, b, d - a - b]))).collect();
    let bvox: Vec<usize> = body.iter().copied().step_by((body.len() / 40000).max(1)).collect();
    let am = DMatrix::<f64>::from_fn(bvox.len(), mono.len(), |i, j| (0..3).map(|d| rho[bvox[i]][d].powi(mono[j][d] as i32)).product());
    let om = DVector::<f64>::from_iterator(bvox.len(), bvox.iter().map(|&u| wfac * sd.field_ppm[u] as f64));
    let pc = am.clone().svd(true, true).solve(&om, 1e-12).unwrap();
    let poly = |u: usize| -> f64 { (0..mono.len()).map(|j| pc[j] * (0..3).map(|d| rho[u][d].powi(mono[j][d] as i32)).product::<f64>()).sum() };
    let mut q = [0.0f64; 6];
    for (j, m) in mono.iter().enumerate() {
        if m.iter().sum::<usize>() == 2 {
            match m { [2, 0, 0] => q[0] = pc[j], [0, 2, 0] => q[1] = pc[j], [0, 0, 2] => q[2] = pc[j], [1, 1, 0] => q[3] = pc[j] / 2.0, [1, 0, 1] => q[4] = pc[j] / 2.0, _ => q[5] = pc[j] / 2.0 }
        }
    }
    let omres: Vec<f64> = (0..rho.len()).map(|u| wfac * sd.field_ppm[u] as f64 - if bdeg > 0 { poly(u) } else { 0.0 }).collect();
    let span = { let mut v: Vec<f64> = body.iter().map(|&u| omres[u]).collect(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); (v[v.len() / 100], v[99 * v.len() / 100]) };
    println!("B0 polynomial degree {bdeg} removed: residual off-resonance 1–99% {:.0}..{:.0} Hz", span.0 / (2.0 * std::f64::consts::PI), span.1 / (2.0 * std::f64::consts::PI));
    let rows2: Vec<Row> = samp.iter().map(|r| { let mut r2 = **r; if bdeg >= 2 { for e in 0..6 { r2.2[e] += q[e] * r.5; } } r2 }).collect();
    let samp: Vec<&Row> = rows2.iter().collect();
    let sd_field = |u: usize| omres[u];
    let quadf = |cc: &[f64; 6], p: &[f64; 3]| cc[0] * p[0] * p[0] + cc[1] * p[1] * p[1] + cc[2] * p[2] * p[2] + 2.0 * (cc[3] * p[0] * p[1] + cc[4] * p[0] * p[2] + cc[5] * p[1] * p[2]);
    // C̄(τ): degree-6 polynomial fit per entry over this encode's samples
    let tmax = samp.iter().map(|r| r.5).fold(0.0, f64::max);
    let tmin = samp.iter().map(|r| r.5).fold(f64::MAX, f64::min);
    let a = DMatrix::<f64>::from_fn(samp.len(), 7, |i, j| ((samp[i].5 - tmin) / (tmax - tmin)).powi(j as i32));
    let svd = a.clone().svd(true, true);
    let coefs: Vec<DVector<f64>> = (0..6).map(|e| svd.solve(&DVector::from_iterator(samp.len(), samp.iter().map(|r| r.2[e])), 1e-12).unwrap()).collect();
    let cbar = |t: f64| -> [f64; 6] { let x = (t - tmin) / (tmax - tmin); [0, 1, 2, 3, 4, 5].map(|e| (0..7).map(|j| coefs[e][j] * x.powi(j as i32)).sum()) };
    let vox: Vec<usize> = body.iter().copied().step_by((body.len() / 6000).max(1)).collect();
    let smp: Vec<&Row> = samp.iter().copied().step_by((samp.len() / 600).max(1)).collect();
    println!("banded segmentation (B0 + τ-mean concomitant in D_ℓ), {} samples × {} voxels:", smp.len(), vox.len());
    for l in [6usize, 8, 10, 12, 16] {
        let tl: Vec<f64> = (0..l).map(|i| tmin + (tmax - tmin) * (i as f64 + 0.5) / l as f64).collect();
        let basis = DMatrix::<C>::from_fn(vox.len(), l, |v, i| {
            let u = vox[v];
            C::from_polar(1.0, -(sd_field(u) * tl[i] + quadf(&cbar(tl[i]), &rho[u])))
        });
        for (corr, nnz) in [(false, l), (true, l), (true, 2), (true, 3), (true, 4)] {
            if nnz > l { continue; }
            let (mut num, mut den) = (0.0, 0.0);
            for r in &smp {
                let t = r.5;
                // nearest `nnz` segments in τ (banded interpolators)
                let mut order: Vec<usize> = (0..l).collect();
                order.sort_by(|&a, &b| (tl[a] - t).abs().partial_cmp(&(tl[b] - t).abs()).unwrap());
                let keep: Vec<usize> = order[..nnz].to_vec();
                let target: Vec<C> = vox.iter().map(|&u| C::from_polar(1.0, -(sd_field(u) * t + quadf(&r.2, &rho[u])))).collect();
                let dc = { let cb = cbar(t); [0, 1, 2, 3, 4, 5].map(|e| r.2[e] - cb[e]) };
                let base = DMatrix::<C>::from_fn(vox.len(), keep.len(), |v, i| basis[(v, keep[i])]);
                let b = if corr {
                    let mut m = base.clone();
                    for v in 0..vox.len() {
                        let f = C::new(1.0, -quadf(&dc, &rho[vox[v]]));
                        for i in 0..keep.len() {
                            m[(v, i)] *= f;
                        }
                    }
                    m
                } else {
                    base
                };
                let y = DVector::<C>::from_vec(target.clone());
                let x = b.clone().svd(true, true).solve(&y, 1e-10).unwrap();
                let res = &b * x - &y;
                num += res.norm_squared();
                den += y.norm_squared();
            }
            println!("  L = {l:>2} {} {}: relative error {:.2e}", if corr { "+ first-order ΔC" } else { "                " }, if nnz == l { "dense ".to_string() } else { format!("{nnz}-sparse") }, (num / den).sqrt());
        }
    }
}
