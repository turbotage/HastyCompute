//! Physiology clocks. Cardiac: R-peak times with beat-to-beat variability. Respiration is reserved for 5D flow
//! (`resp_phase` returns `None` until a respiratory model is added).

/// ECG model: RR intervals ~ mean · (1 + σ·ξ), ξ uniform-ish (sum of uniforms), deterministic from `seed`.
#[derive(Clone, Debug)]
pub struct Ecg {
    pub r_peaks: Vec<f64>,
}

/// Where in the cardiac cycle an instant falls.
#[derive(Clone, Copy, Debug)]
pub struct CardiacTime {
    pub beat: usize,
    /// time since the preceding R peak, s
    pub since_r: f64,
    /// length of the current RR interval, s
    pub rr: f64,
    /// since_r / rr ∈ [0, 1)
    pub phase: f64,
}

impl Ecg {
    pub fn new(duration: f64, mean_rr: f64, rel_std: f64, seed: u64) -> Self {
        let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407) | 1;
        let mut u = || {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            (s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut r = vec![-mean_rr * u()];
        while *r.last().unwrap() < duration {
            let xi = (u() + u() + u() - 1.5) * 2.0; // ~ zero mean, unit-ish variance
            r.push(r.last().unwrap() + mean_rr * (1.0 + rel_std * xi).max(0.5));
        }
        Self { r_peaks: r }
    }

    pub fn at(&self, t: f64) -> CardiacTime {
        let i = match self.r_peaks.binary_search_by(|p| p.partial_cmp(&t).unwrap()) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };
        let i = i.min(self.r_peaks.len() - 2);
        let rr = self.r_peaks[i + 1] - self.r_peaks[i];
        let since_r = t - self.r_peaks[i];
        CardiacTime { beat: i, since_r, rr, phase: (since_r / rr).clamp(0.0, 1.0 - 1e-12) }
    }

    pub fn resp_phase(&self, _t: f64) -> Option<f64> {
        None
    }
}
