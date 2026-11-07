//! Self-gating surrogates from repeated k = 0 samples (one complex value per coil and TR, e.g. the FID at the first
//! echo of a center-out train): temporal band-pass per channel (difference of zero-phase Gaussian smoothers), PCA over
//! the channels, the leading component as the surrogate.
//!
//! Sign convention for breathing: the waveform dwells near end-expiration (cos^{2n}-like), so its distribution has a
//! long tail toward inspiration; the sign is chosen so that the skewness is positive, and the surrogate is scaled to
//! r ≈ 0 at end-expiration (2nd percentile) and ≈ 1 at end-inspiration (98th).

use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64;

/// zero-phase Gaussian smoothing with standard deviation `sigma` samples (edges: renormalised truncated kernel)
pub fn gauss_smooth(x: &[f64], sigma: f64) -> Vec<f64> {
    if sigma <= 0.0 {
        return x.to_vec();
    }
    let h = (4.0 * sigma).ceil() as isize;
    let w: Vec<f64> = (-h..=h).map(|i| (-0.5 * (i as f64 / sigma).powi(2)).exp()).collect();
    let n = x.len() as isize;
    (0..n)
        .map(|i| {
            let (mut s, mut ws) = (0.0, 0.0);
            for (j, wj) in (-h..=h).zip(&w) {
                let k = i + j;
                if k >= 0 && k < n {
                    s += wj * x[k as usize];
                    ws += wj;
                }
            }
            s / ws
        })
        .collect()
}

pub struct Surrogate {
    /// normalised surrogate per TR (0 end-expiration, 1 end-inspiration for breathing)
    pub r: Vec<f64>,
    /// leading principal component before normalisation (sign fixed)
    pub raw: Vec<f64>,
    /// fraction of the band-passed channel variance in the leading component
    pub explained: f64,
}

/// surrogate from `z` [n_tr][nc] sampled every `dt` seconds; band (s): keep periods between `t_min` and `t_max`
/// (breathing: 1.5 s … 15 s)
pub fn surrogate(z: &[Complex64], n_tr: usize, nc: usize, dt: f64, t_min: f64, t_max: f64) -> Surrogate {
    let nch = 2 * nc;
    // channels: re / im of each coil, band-passed: smooth(σ ≈ t_min/(2π)) − smooth(σ ≈ t_max/(2π))
    let (s_lo, s_hi) = (t_min / (2.0 * std::f64::consts::PI * dt), t_max / (2.0 * std::f64::consts::PI * dt));
    let ch: Vec<Vec<f64>> = (0..nch)
        .map(|c| {
            let x: Vec<f64> = (0..n_tr).map(|i| if c % 2 == 0 { z[i * nc + c / 2].re } else { z[i * nc + c / 2].im }).collect();
            let a = gauss_smooth(&x, s_lo);
            let b = gauss_smooth(&x, s_hi);
            a.iter().zip(&b).map(|(p, q)| p - q).collect()
        })
        .collect();
    // per-channel scale to unit variance (coils of very different gain), then PCA
    let sd: Vec<f64> = ch.iter().map(|v| (v.iter().map(|x| x * x).sum::<f64>() / n_tr as f64).sqrt().max(1e-30)).collect();
    let mut cov = DMatrix::<f64>::zeros(nch, nch);
    for a in 0..nch {
        for b in a..nch {
            let s: f64 = (0..n_tr).map(|i| ch[a][i] * ch[b][i]).sum::<f64>() / (n_tr as f64 * sd[a] * sd[b]);
            cov[(a, b)] = s;
            cov[(b, a)] = s;
        }
    }
    let e = SymmetricEigen::new(cov);
    let (imax, lmax) = e.eigenvalues.iter().enumerate().fold((0, f64::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b });
    let tot: f64 = e.eigenvalues.iter().sum();
    let mut pc: Vec<f64> = (0..n_tr).map(|i| (0..nch).map(|a| e.eigenvectors[(a, imax)] * ch[a][i] / sd[a]).sum()).collect();
    let mean = pc.iter().sum::<f64>() / n_tr as f64;
    let var = pc.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n_tr as f64;
    let skew = pc.iter().map(|x| (x - mean).powi(3)).sum::<f64>() / n_tr as f64 / var.powf(1.5).max(1e-300);
    if skew < 0.0 {
        pc.iter_mut().for_each(|x| *x = -*x);
    }
    let mut srt = pc.clone();
    srt.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| srt[((n_tr - 1) as f64 * f) as usize];
    let (p2, p98) = (q(0.02), q(0.98));
    let r = pc.iter().map(|x| (x - p2) / (p98 - p2).max(1e-30)).collect();
    Surrogate { r, raw: pc, explained: lmax / tot }
}

/// equal-count amplitude bins: bin index per TR (0 = lowest surrogate = end-expiration)
pub fn amplitude_bins(r: &[f64], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..r.len()).collect();
    idx.sort_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap());
    let mut bin = vec![0; r.len()];
    for (rank, &i) in idx.iter().enumerate() {
        bin[i] = (rank * n / r.len()).min(n - 1);
    }
    bin
}

/// Pearson correlation
pub fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma).powi(2);
        sbb += (y - mb).powi(2);
    }
    sab / (saa * sbb).sqrt().max(1e-300)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_skewed_breathing_waveform_from_mixed_coils() {
        // r(t) = cos^4(π t / T) style (dwells at 0), 8 coils with random complex gains, drift and noise
        let (n, nc, dt) = (2500usize, 8usize, 0.024);
        let mut s = 12345u64;
        let mut u = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let truth: Vec<f64> = (0..n).map(|i| (std::f64::consts::PI * i as f64 * dt / 4.0).sin().powi(4)).collect();
        let gains: Vec<Complex64> = (0..nc).map(|_| Complex64::new(u(), u())).collect();
        let base: Vec<Complex64> = (0..nc).map(|_| Complex64::new(1.0 + u(), u())).collect();
        let z: Vec<Complex64> = (0..n)
            .flat_map(|i| {
                let t = i as f64 * dt;
                let tr = truth[i];
                (0..nc).map(move |c| (c, t, tr)).collect::<Vec<_>>()
            })
            .map(|(c, t, tr)| base[c] * (1.0 + 0.002 * t) + gains[c] * 0.1 * tr)
            .collect();
        let z: Vec<Complex64> = z.into_iter().map(|v| v + Complex64::new(0.002 * u(), 0.002 * u())).collect();
        let sg = surrogate(&z, n, nc, dt, 1.5, 15.0);
        let c = corr(&sg.r, &truth);
        assert!(c > 0.97, "correlation {c}");
        let b = amplitude_bins(&sg.r, 4);
        let mean_t = |k: usize| -> f64 {
            let v: Vec<f64> = (0..n).filter(|&i| b[i] == k).map(|i| truth[i]).collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        assert!(mean_t(0) < mean_t(3), "bins not ordered expiration → inspiration");
    }
}
