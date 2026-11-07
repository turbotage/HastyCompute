//! External physiology recorded with a scan (`simphantom` physio.npy: t, true ECG phase, PPG phase, bellows, true r).

/// scale to [0, 1] by the 2nd / 98th percentiles (0 ≈ end-expiration for a bellows / breathing signal)
pub fn normalise(x: &[f64]) -> Vec<f64> {
    let mut s: Vec<f64> = x.iter().cloned().filter(|v| v.is_finite()).collect();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| s[((s.len() - 1) as f64 * f) as usize];
    let (lo, hi) = (q(0.02), q(0.98));
    x.iter().map(|v| (v - lo) / (hi - lo).max(1e-30)).collect()
}

/// group index per readout: equal-count quantile levels of `r` (0 = lowest)
pub fn levels(r: &[f64], n: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..r.len()).collect();
    idx.sort_by(|&a, &b| r[a].partial_cmp(&r[b]).unwrap());
    let mut g = vec![0; r.len()];
    for (rank, &i) in idx.iter().enumerate() {
        g[i] = (rank * n / r.len()).min(n - 1);
    }
    g
}
