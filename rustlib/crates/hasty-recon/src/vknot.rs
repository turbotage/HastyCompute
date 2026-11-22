//! Respiratory × cardiac knot basis of the at-knots velocity model (`VelSpec::at_knots`, research/velocity_model_sr.md,
//! research/temporal_operator.md §16–§21) and its memory ledger (research/memory_ledger.md).
//!
//! - [`KnotBasis`]: kr respiratory knots (degree 0 = equal-count amplitude bins, degree 1 = linear hats at the amplitude
//!   quantiles 0, 1/(kr−1), …, 1) × mc periodic cardiac knots (B-spline degree 0 | 1 | 3, knot b at cardiac time
//!   (b + ½)/mc of the cycle), knot k = r · mc + c — the layout and conventions of `SegNufft::new_temporal2d` /
//!   `points2d` (the same rules as `splinerecon`). The breathing coordinate is whatever amplitude the caller passes (the
//!   bellows by default in the drivers; the simulator's true state only as an explicit oracle).
//! - per-sample knot coordinates (cardiac t in knot units, respiratory interval + fraction), weights of any (breathing
//!   amplitude, cardiac phase), the diagonal of the temporal Gram (per-knot curvature of the metric), and the complex
//!   interpolation the operator applies between knots.
//! - [`plan`]: where the state, the k-space data and the sample tables live (device / host), checked against the
//!   device and host budgets before the solver allocates anything; a plan that fits neither is refused with the table.

/// centred periodic B-spline weight of cardiac knot `b` at time `t` (knot units, period `m`), degree 0, 1 or 3 — the
/// kernels' convention (degree 0: t ∈ [b, b + 1) belongs to knot b)
pub fn cardiac_weight(t: f64, b: usize, m: usize, deg: u32) -> f64 {
    let n = m as f64;
    let mut d = (t - b as f64).rem_euclid(n);
    if d > 0.5 * n {
        d -= n;
    }
    let a = d.abs();
    match deg {
        0 => ((t.rem_euclid(n)).floor() as usize % m == b) as u8 as f64,
        1 => (1.0 - a).max(0.0),
        3 if a <= 1.0 => 2.0 / 3.0 - a * a + 0.5 * a * a * a,
        3 if a <= 2.0 => (2.0 - a).powi(3) / 6.0,
        3 => 0.0,
        _ => panic!("cardiac B-spline degree {deg} (0, 1 or 3)"),
    }
}

/// The 2-D knot basis. `rknots`: degree 1 — the knot amplitudes (quantiles); degree 0 — the bins' [lo, hi] amplitude
/// ranges (2 values per bin).
#[derive(Clone, Debug)]
pub struct KnotBasis {
    pub kr: usize,
    pub rdeg: u32,
    pub mc: usize,
    pub cdeg: u32,
    pub rknots: Vec<f64>,
}

impl KnotBasis {
    /// `RK:RDEG,CK:CDEG`, e.g. "6:1,16:3" → (kr, rdeg, mc, cdeg)
    pub fn parse_spec(s: &str) -> Result<(usize, u32, usize, u32), String> {
        let bad = || format!("--knots2d RK:RDEG,CK:CDEG (e.g. 6:1,16:3), got {s}");
        let (r, c) = s.split_once(',').ok_or_else(bad)?;
        let p = |x: &str| -> Result<(usize, u32), String> {
            let (a, b) = x.split_once(':').ok_or_else(bad)?;
            Ok((a.trim().parse().map_err(|_| bad())?, b.trim().parse().map_err(|_| bad())?))
        };
        let ((kr, rdeg), (mc, cdeg)) = (p(r)?, p(c)?);
        if rdeg > 1 || !(cdeg == 0 || cdeg == 1 || cdeg == 3) || kr == 0 || (rdeg == 1 && kr < 2) || mc < 2 || mc < cdeg as usize + 1 {
            return Err(format!("--knots2d {s}: respiratory degree 0|1 (≥ 2 knots for 1), cardiac degree 0|1|3 with ≥ degree + 1 knots"));
        }
        Ok((kr, rdeg, mc, cdeg))
    }

    /// The basis over the breathing amplitudes `amp` of the readouts used (bins: equal counts; linear: quantiles)
    pub fn from_amplitude(amp: &[f64], kr: usize, rdeg: u32, mc: usize, cdeg: u32) -> Self {
        assert!(!amp.is_empty(), "knot basis: no readouts");
        let rknots = if rdeg == 0 {
            let bins = crate::selfgate::amplitude_bins(amp, kr);
            let (mut lo, mut hi) = (vec![f64::INFINITY; kr], vec![f64::NEG_INFINITY; kr]);
            for (i, &b) in bins.iter().enumerate() {
                lo[b] = lo[b].min(amp[i]);
                hi[b] = hi[b].max(amp[i]);
            }
            (0..kr).flat_map(|b| [lo[b], hi[b]]).collect()
        } else {
            let mut s = amp.to_vec();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = s.len();
            (0..kr).map(|k| s[((k as f64 / (kr - 1) as f64) * (n - 1) as f64).round() as usize]).collect()
        };
        Self { kr, rdeg, mc, cdeg, rknots }
    }

    pub fn nk(&self) -> usize {
        self.kr * self.mc
    }

    /// respiratory intervals of the tables (bins: kr; linear: kr − 1)
    pub fn nri(&self) -> usize {
        if self.rdeg == 0 { self.kr } else { self.kr - 1 }
    }

    /// (interval, fraction) of a breathing amplitude (linear: clamped to the end knots; bins: the bin holding it, else
    /// the nearest bin centre)
    pub fn locate(&self, a: f64) -> (u32, f32) {
        if self.rdeg == 0 {
            let b = (0..self.kr).find(|&b| a >= self.rknots[2 * b] && a <= self.rknots[2 * b + 1]).unwrap_or_else(|| {
                let c = |b: usize| 0.5 * (self.rknots[2 * b] + self.rknots[2 * b + 1]);
                (0..self.kr).min_by(|&x, &y| (c(x) - a).abs().partial_cmp(&(c(y) - a).abs()).unwrap()).unwrap()
            });
            return (b as u32, 0.0);
        }
        let q = &self.rknots;
        let i = (0..self.kr - 1).rev().find(|&i| a >= q[i]).unwrap_or(0);
        let v = ((a - q[i]) / (q[i + 1] - q[i]).max(1e-12)).clamp(0.0, 1.0);
        (i as u32, v as f32)
    }

    /// per readout (interval, fraction); bins exactly as `selfgate::amplitude_bins` over `amp` (ties by rank)
    pub fn assign(&self, amp: &[f64]) -> Vec<(u32, f32)> {
        if self.rdeg == 0 {
            return crate::selfgate::amplitude_bins(amp, self.kr).into_iter().map(|b| (b as u32, 0.0)).collect();
        }
        amp.iter().map(|&a| self.locate(a)).collect()
    }

    /// respiratory weights of the kr knots at (interval, fraction)
    pub fn resp_weights_at(&self, ri: u32, rv: f32) -> Vec<f64> {
        let mut w = vec![0.0; self.kr];
        if self.rdeg == 0 {
            w[ri as usize] = 1.0;
        } else {
            w[ri as usize] = 1.0 - rv as f64;
            w[ri as usize + 1] += rv as f64;
        }
        w
    }

    /// respiratory weights at a breathing amplitude
    pub fn resp_weights(&self, a: f64) -> Vec<f64> {
        let (i, v) = self.locate(a);
        self.resp_weights_at(i, v)
    }

    /// cardiac knot coordinate (knot units, [0, mc)) of a cardiac phase φ ∈ [0, 1): t = φ · mc − ½
    pub fn cardiac_t(&self, phase: f64) -> f64 {
        (phase * self.mc as f64 - 0.5).rem_euclid(self.mc as f64)
    }

    /// the mc cardiac weights at knot coordinate t
    pub fn cardiac_weights(&self, t: f64) -> Vec<f64> {
        (0..self.mc).map(|b| cardiac_weight(t, b, self.mc, self.cdeg)).collect()
    }

    /// weights of the K knots at (breathing amplitude, cardiac phase), k = r · mc + c
    pub fn weights(&self, amp: f64, phase: f64) -> Vec<f64> {
        let (wr, wc) = (self.resp_weights(amp), self.cardiac_weights(self.cardiac_t(phase)));
        (0..self.nk()).map(|k| wr[k / self.mc] * wc[k % self.mc]).collect()
    }

    /// sparse knot weights of one sample (cardiac t in knot units, respiratory interval / fraction)
    pub fn sample_weights(&self, t: f64, ri: u32, rv: f32) -> Vec<(usize, f64)> {
        let wr = self.resp_weights_at(ri, rv);
        let i0 = t.floor() as i64;
        let cw: Vec<(usize, f64)> = (i0 - 2..=i0 + 2).map(|b| b.rem_euclid(self.mc as i64) as usize).map(|b| (b, cardiac_weight(t, b, self.mc, self.cdeg))).filter(|p| p.1 != 0.0).collect();
        // distinct cardiac knots only (small mc wraps the window onto itself)
        let mut cw2: Vec<(usize, f64)> = vec![];
        for (b, w) in cw {
            if !cw2.iter().any(|p| p.0 == b) {
                cw2.push((b, w));
            }
        }
        wr.iter().enumerate().filter(|p| *p.1 != 0.0).flat_map(|(r, &a)| cw2.iter().map(move |&(b, c)| (r * self.mc + b, a * c))).collect()
    }

    /// knot coordinates of every sample: cardiac t = (cardiac frame coordinate t_f of `segphys::SegTables::t`, frame
    /// units of `nf` frames) converted to knot units, and per readout (interval, fraction) expanded over the readout's
    /// samples (`readout_of[j]` = the sample's index into `per_readout`)
    pub fn sample_coords(&self, t_frames: &[f32], nf: usize, per_readout: &[(u32, f32)], readout_of: &dyn Fn(usize) -> usize) -> (Vec<f32>, Vec<u32>, Vec<f32>) {
        let m = t_frames.len();
        let (mc, nff) = (self.mc as f64, nf as f64);
        let t: Vec<f32> = t_frames.iter().map(|&tf| ((tf as f64 + 0.5) * mc / nff - 0.5).rem_euclid(mc) as f32).collect();
        let (mut ri, mut rv) = (vec![0u32; m], vec![0f32; m]);
        for j in 0..m {
            let (a, b) = per_readout[readout_of(j)];
            ri[j] = a;
            rv[j] = b;
        }
        (t, ri, rv)
    }

    /// human-readable description of the respiratory knots
    pub fn describe(&self) -> String {
        if self.rdeg == 0 {
            format!("{} bins {:?} × {} cardiac knots degree {}", self.kr, (0..self.kr).map(|b| format!("[{:.3}, {:.3}]", self.rknots[2 * b], self.rknots[2 * b + 1])).collect::<Vec<_>>(), self.mc, self.cdeg)
        } else {
            format!("{} linear knots at {:?} × {} cardiac knots degree {}", self.kr, self.rknots.iter().map(|v| (v * 1000.0).round() / 1000.0).collect::<Vec<_>>(), self.mc, self.cdeg)
        }
    }
}

/// diagonal of the temporal Gram G_kk = mean_j w_k(j)² over the samples (`weights` per sample) and its condition
/// number (full Gram, `knotsolve::temporal_gram`)
pub fn knot_gram(k: usize, weights: impl Iterator<Item = Vec<(usize, f64)>>) -> (Vec<f64>, f64) {
    let g = crate::knotsolve::temporal_gram(k, weights);
    let d = (0..k).map(|i| g[i * k + i]).collect();
    (d, crate::knotsolve::gram_condition(&g, k))
}

// ─── operator frames of the parameter-spline model (research/velocity_model_sr.md §F) ────────────────────────────

/// Operator frames of the parameter-spline velocity model: the parameters (m, ψ, v) live in the kr × mc respiratory ×
/// cardiac basis; the operator sees, per respiratory knot r, `fop` cardiac frames of the model (frame f at cardiac phase
/// (f + ½)/fop) and interpolates them periodically in cardiac time (degree 3: the cubic spline through the frame values,
/// coefficients = the (1/6, 2/3, 1/6)-circulant inverse of the values; degree 1: linear between them). Between
/// respiratory knots the operator mixes the rows complex-linearly: the model is defined that way (a breathing state
/// between two knots is the partial-volume mixture of the two knot states; frames_caveat.json: that predicts the truth
/// as well as parameter-linear interpolation, 1.42e-2 vs 1.41e-2).
///
/// The 2-D knot tables hold < 128 (respiratory interval × cardiac knot) keys, so the respiratory knots are split into
/// groups of `kre` consecutive knots (kre − 1 intervals, (kre − 1) · fop < 128): one set per (encode e, group g),
/// t = e · ng + g, holding the kre · fop frames of its knots (row-major: local knot rl, cardiac frame f); a sample of
/// global interval i goes to the group holding i. The last group is shifted back to end at knot kr − 1 (it may start
/// on a knot of the previous group: that row carries no samples there).
#[derive(Clone, Debug, PartialEq)]
pub struct OpFrames {
    pub kr: usize,
    pub mc: usize,
    /// cardiac degree of the parameter basis (1 or 3)
    pub cdeg: u32,
    /// operator cardiac frames per respiratory knot and the operator's cardiac degree (1 or 3)
    pub fop: usize,
    pub odeg: u32,
    /// respiratory knots per set (the engine's kr)
    pub kre: usize,
    /// first respiratory knot of every group
    pub r0: Vec<usize>,
}

impl OpFrames {
    /// `--vop frames:F[:DEG]` (default degree 3)
    pub fn parse_spec(s: &str) -> Result<(usize, u32), String> {
        let bad = || format!("--vop frames:F[:DEG] | knots, got {s}");
        let v: Vec<&str> = s.split(':').collect();
        if v[0] != "frames" || v.len() < 2 || v.len() > 3 {
            return Err(bad());
        }
        let f: usize = v[1].parse().map_err(|_| bad())?;
        let d: u32 = v.get(2).map_or(Ok(3), |x| x.parse().map_err(|_| bad()))?;
        if f < 4 || !(d == 1 || d == 3) {
            return Err(format!("--vop frames:F[:DEG]: F ≥ 4, degree 1 or 3 (got {s})"));
        }
        Ok((f, d))
    }

    /// groups for a parameter basis kr (linear, ≥ 2) × mc (degree cdeg) and fop operator frames of degree odeg
    pub fn new(kr: usize, mc: usize, cdeg: u32, fop: usize, odeg: u32) -> Self {
        assert!(fop < 128, "operator frames: fop < 128 (2-D table keys)");
        Self::with_kre(kr, mc, cdeg, fop, odeg, (1 + 127 / fop).min(kr))
    }

    /// as [`new`](Self::new) with kre respiratory knots per set (2 ≤ kre ≤ kr, (kre − 1) · fop < 128)
    pub fn with_kre(kr: usize, mc: usize, cdeg: u32, fop: usize, odeg: u32, kre: usize) -> Self {
        assert!(kr >= 2, "operator frames: ≥ 2 linear respiratory knots");
        assert!(kre >= 2 && kre <= kr && (kre - 1) * fop < 128, "operator frames: 2 ≤ kre ≤ kr with (kre − 1) · fop < 128");
        let (nri, nie) = (kr - 1, kre - 1);
        let ng = nri.div_ceil(nie);
        let r0 = (0..ng).map(|g| (g * nie).min(kr - kre)).collect();
        Self { kr, mc, cdeg, fop, odeg, kre, r0 }
    }

    pub fn ng(&self) -> usize {
        self.r0.len()
    }

    pub fn k(&self) -> usize {
        self.kr * self.mc
    }

    /// frames per set (kre · fop)
    pub fn set_frames(&self) -> usize {
        self.kre * self.fop
    }

    /// operator images per encode (all groups)
    pub fn images_per_encode(&self) -> usize {
        self.ng() * self.set_frames()
    }

    /// (encode, group) of set t = e · ng + g
    pub fn set_eg(&self, t: usize) -> (usize, usize) {
        (t / self.ng(), t % self.ng())
    }

    /// group of a global respiratory interval and the interval local to it
    pub fn group_of(&self, ri: u32) -> (usize, u32) {
        let nie = self.kre - 1;
        let g = (ri as usize / nie).min(self.ng() - 1);
        (g, ri - self.r0[g] as u32)
    }

    /// cardiac parameter weights (knot c, w ≠ 0) of operator frame f
    pub fn frame_weights(&self, f: usize) -> Vec<(usize, f64)> {
        let t = ((f as f64 + 0.5) * self.mc as f64 / self.fop as f64 - 0.5).rem_euclid(self.mc as f64);
        let mut w: Vec<(usize, f64)> = vec![];
        let i0 = t.floor() as i64;
        for b in i0 - 2..=i0 + 2 {
            let b = b.rem_euclid(self.mc as i64) as usize;
            let x = cardiac_weight(t, b, self.mc, self.cdeg);
            if x != 0.0 && !w.iter().any(|p| p.0 == b) {
                w.push((b, x));
            }
        }
        w
    }

    /// periodic cubic-spline interpolation filter C[n] (n = 0..fop): coefficients c_f = Σ_n C[n] x_{f−n} interpolate the
    /// frame values x (cubic B-spline (1/6, 2/3, 1/6) at the frames); symmetric, so the adjoint is the same filter.
    /// Degree 1: the identity.
    pub fn prefilter(&self) -> Vec<f64> {
        let f = self.fop;
        if self.odeg == 1 {
            return (0..f).map(|n| (n == 0) as u8 as f64).collect();
        }
        (0..f)
            .map(|n| {
                (0..f)
                    .map(|k| {
                        let w = 2.0 * std::f64::consts::PI * k as f64 / f as f64;
                        (w * n as f64).cos() / (2.0 / 3.0 + w.cos() / 3.0)
                    })
                    .sum::<f64>()
                    / f as f64
            })
            .collect()
    }

    /// the operator's cardiac coordinate (frame units of fop) of a cardiac phase φ ∈ [0, 1)
    pub fn op_t(&self, phase: f64) -> f64 {
        (phase * self.fop as f64 - 0.5).rem_euclid(self.fop as f64)
    }

    /// the model signal at an arbitrary (respiratory weights over the kr knots, cardiac phase) from the per-knot frame
    /// evaluation `row(r, phase)` (the respiratory rows mixed complex-linearly, as the operator does)
    pub fn mix_rows(&self, wr: &[f64], phase: f64, row: &mut dyn FnMut(usize, f64) -> num_complex::Complex64) -> num_complex::Complex64 {
        let mut x = num_complex::Complex64::new(0.0, 0.0);
        for (r, &w) in wr.iter().enumerate() {
            if w != 0.0 {
                x += w * row(r, phase);
            }
        }
        x
    }

    pub fn describe(&self) -> String {
        format!(
            "{} cardiac operator frames per respiratory knot (cardiac {} between frames), {} groups of {} respiratory knots ({} operator images per encode, ×{:.2} the {} knots)",
            self.fop,
            if self.odeg == 3 { "cubic-spline interpolation" } else { "linear" },
            self.ng(),
            self.kre,
            self.images_per_encode(),
            self.images_per_encode() as f64 / self.k() as f64,
            self.k()
        )
    }
}

/// cardiac parameter weights of all `fop` frames and, per cardiac knot c, the frames it touches with their weights
/// (padded with −1 / 0 to the longest list): the tables of the GPU frame kernels (`vmodel::VelModel`)
pub fn frame_tables(of: &OpFrames) -> (Vec<i32>, Vec<f32>, usize, Vec<i32>, Vec<f32>, usize, Vec<i32>) {
    let fw: Vec<Vec<(usize, f64)>> = (0..of.fop).map(|f| of.frame_weights(f)).collect();
    let nn = fw.iter().map(|w| w.len()).max().unwrap_or(1);
    let (mut fi, mut fv) = (vec![-1i32; of.fop * nn], vec![0f32; of.fop * nn]);
    let mut kf: Vec<Vec<(usize, f64)>> = vec![vec![]; of.mc];
    for (f, w) in fw.iter().enumerate() {
        for (j, &(c, x)) in w.iter().enumerate() {
            fi[f * nn + j] = c as i32;
            fv[f * nn + j] = x as f32;
            kf[c].push((f, x));
        }
    }
    let nkf = kf.iter().map(|l| l.len()).max().unwrap_or(1);
    let (mut ki, mut kv) = (vec![-1i32; of.mc * nkf], vec![0f32; of.mc * nkf]);
    for (c, l) in kf.iter().enumerate() {
        for (j, &(f, x)) in l.iter().enumerate() {
            ki[c * nkf + j] = f as i32;
            kv[c * nkf + j] = x as f32;
        }
    }
    // the knot that accumulates frame f's residual-phase term (its first node)
    let prim: Vec<i32> = fw.iter().map(|w| w[0].0 as i32).collect();
    (fi, fv, nn, ki, kv, nkf, prim)
}

// ─── memory ledger ───────────────────────────────────────────────────────────────────────────────────────────────

/// Sizes the knot-mode ledger is built from (bytes unless noted)
#[derive(Clone, Debug)]
pub struct LedgerIn {
    pub nv: usize,
    pub nk: usize,
    /// images in one set image (0: nk; spline mode: kre · fop operator frames)
    pub img_frames: usize,
    /// device bytes of the streamed state's slots (0: six node chunks; spline mode: the two row slots)
    pub slots: usize,
    /// parameter stack (all chunks)
    pub stack: usize,
    /// largest chunk (complex elements × 8)
    pub chunk_max: usize,
    /// sets (one per encode)
    pub nsets: usize,
    /// k-space data bytes of all sets / of the largest set, and whether they can be resident (false: staged from the
    /// store / pinned already)
    pub data_all: usize,
    pub data_max: usize,
    pub data_movable: bool,
    /// sample-table bytes of all sets / of the largest set (0: not placeable)
    pub tab_all: usize,
    pub tab_max: usize,
    /// device bytes resident now that are not placeable (operator: coil maps, segments, grids, FFT plans, …), i.e.
    /// total − free minus the resident data and tables
    pub fixed: usize,
    /// device budget (bytes usable in total)
    pub dev_budget: usize,
    /// host bytes still available (cgroup / MemAvailable) and the pinned host already holding data / tables now
    pub host_avail: usize,
    /// ZMSHR1 (secant pair)
    pub quasi: bool,
    /// extra device bytes of the prior on the device (0: host priors)
    pub prior_dev: usize,
    /// pinned (h2d) and pageable transfer rates, bytes / s
    pub h2d: f64,
    pub pageable: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LedgerPlan {
    /// parameter stacks in host memory, θ streamed knot by knot (`run_objective_host`)
    pub state_on_host: bool,
    /// device state: the secant pair / rank-one direction in pinned host memory
    pub prev_on_host: bool,
    pub data_on_host: bool,
    pub tables_on_host: bool,
    pub device_peak: usize,
    pub host_peak: usize,
    /// predicted host↔device traffic per gradient (s)
    pub transfer_s: f64,
    /// (entry, device bytes, host bytes)
    pub entries: Vec<(String, usize, usize)>,
}

const GIB: f64 = 1073741824.0;

impl LedgerPlan {
    pub fn describe(&self) -> String {
        let mut s = format!(
            "knot-mode memory plan: state {}, data {}, sample tables {}; device peak {:.2} GiB, host {:.2} GiB, predicted transfers {:.1} s / gradient\n",
            if self.state_on_host { "host (θ streamed knot by knot)" } else if self.prev_on_host { "device (secant pair in pinned host memory)" } else { "device" },
            if self.data_on_host { "host (staged per set)" } else { "device" },
            if self.tables_on_host { "host (streamed per set)" } else { "device" },
            self.device_peak as f64 / GIB,
            self.host_peak as f64 / GIB,
            self.transfer_s
        );
        for (k, d, h) in &self.entries {
            s += &format!("    {k:<62} device {:>7.2} GiB  host {:>7.2} GiB\n", *d as f64 / GIB, *h as f64 / GIB);
        }
        s
    }
}

/// Choose the placement with the least predicted transfer time that fits both budgets; Err(ledger) when none does.
/// Device needs per gradient: one encode's set image, then its adjoint (K nv; the image is freed first), the residual of the largest set, the
/// data (all sets resident, or one staged set), the tables (resident, or two slots of the largest set), the fixed
/// operator, the model scratch, and the state (device: x, g, step + secant pair (+ w) or the λ_max power stacks;
/// host: six node slots). Host needs: host state 4 stacks (x, g, x_prev, g_prev; the λ_max power iteration's two
/// extra stacks come before the solver's — Lanczos (`--lam-method`, default): three extra host stacks with host state,
/// one (q_{j−1}) with device state, plus K stacks only when the warm start is on, which recon_p2np's `--lam-warm auto`
/// allows only within min(HASTY_LAM_WARM_GB, ¼ of the available memory)), device state 1 stack (the prior's copy) +
/// the pinned pair and w.
pub fn plan(n: &LedgerIn, force_state: Option<bool>, force_data: Option<bool>, force_tables: Option<bool>) -> Result<LedgerPlan, String> {
    plan_sink(n, false, force_state, force_data, force_tables)
}

/// [`plan`] for an objective with (`has_sink`) or without a gradient sink path (`Objective::has_gradient_sink`: the
/// velocity objective in at-knots mode has one, so a BFP host state stores g and g_prev at their width; without it they
/// stay f32)
pub fn plan_sink(n: &LedgerIn, has_sink: bool, force_state: Option<bool>, force_data: Option<bool>, force_tables: Option<bool>) -> Result<LedgerPlan, String> {
    let store = crate::bfpstate::StateStore::from_env()?;
    let kst = crate::kstage::stage_from_env().map_err(|e| e.to_string())?;
    plan_with(n, store.effective(has_sink), kst, force_state, force_data, force_tables)
}

/// [`plan`] with the host state's storage (`HASTY_STATE_STORE`, [`crate::bfpstate::StateStore`]) and the k-space
/// staging tier (`HASTY_KSTAGE`, [`crate::kstage::stage_from_env`]) given explicitly (`store` as the run uses it:
/// [`crate::bfpstate::StateStore::effective`] of the objective's gradient sink). Sizes from `hasty_bfp::Bfp`
/// bytes: BFP host state = x (f32) + g, g_prev, secant step in their widths (+ one more set-sized f32 temporary of the
/// metric steps); packed staging = the f32 set the device decodes into plus its packed upload, PCIe at the BFP size;
/// resident packed = every set at the BFP size on the device plus one decoded set, no PCIe per gradient. With both
/// None the plan is the f32 one, entry for entry.
pub fn plan_with(n: &LedgerIn, store: crate::bfpstate::StateStore, kst: Option<(crate::kstore::BfpRule, bool)>, force_state: Option<bool>, force_data: Option<bool>, force_tables: Option<bool>) -> Result<LedgerPlan, String> {
    let ratio = |f: Option<hasty_bfp::Bfp>| f.map_or(1.0, |b| b.bytes_per_complex() / 8.0);
    let sb = |f: Option<hasty_bfp::Bfp>| (n.stack as f64 * ratio(f)).ceil() as usize;
    let kr = kst.as_ref().map(|(r, _)| crate::kstage::planning_ratio(r));
    let kres = kst.as_ref().is_some_and(|(_, r)| *r);
    let img = if n.img_frames > 0 { n.img_frames } else { n.nk } * n.nv * 8;
    let (data_all, data_max, tab_all, tab_max) = (n.data_all, n.data_max, n.tab_all, n.tab_max);
    let scratch = 2 * n.nv * 8 + 16 * n.nv * 4 + if n.slots > 0 { n.slots } else { 6 * n.chunk_max };
    let margin = (1.0 * GIB) as usize;
    let mut best: Option<LedgerPlan> = None;
    let mut rejected: Vec<String> = vec![];
    for state_host in [false, true] {
        if force_state.is_some_and(|f| f != state_host) {
            continue;
        }
        for data_host in [false, true] {
            if force_data.is_some_and(|f| f != data_host) || (!data_host && !n.data_movable) {
                continue;
            }
            for tables_host in [false, true] {
                if force_tables.is_some_and(|f| f != tables_host) || (tables_host && tab_all == 0) {
                    continue;
                }
                for prev_host in [false, true] {
                    if (state_host || !n.quasi) && prev_host {
                        continue;
                    }
                    let mut e: Vec<(String, usize, usize)> = vec![];
                    e.push(("operator (coil maps, segments, fine grids, FFT plans), resident".into(), n.fixed, 0));
                    // the set image is freed after the residual, before its adjoint is allocated: one knot stack
                    e.push((format!("set image / its adjoint ({} images × {} voxels, one set)", if n.img_frames > 0 { n.img_frames } else { n.nk }, n.nv), img, 0));
                    e.push(("residual of the largest set".into(), data_max, 0));
                    match (data_host, kr) {
                        (true, Some(r)) if kres => e.push((format!("k-space data: all sets packed on the device (BFP, {:.3} of f32) + one decoded set", r), (data_all as f64 * r).ceil() as usize + data_max, 0)),
                        (true, Some(r)) => e.push((format!("k-space data: one set staged packed (BFP, {:.3} of f32), decoded on the device", r), data_max + (data_max as f64 * r).ceil() as usize, 0)),
                        _ => e.push((if data_host { "k-space data: one set staged".into() } else { "k-space data, resident".into() }, if data_host { data_max } else { data_all }, 0)),
                    }
                    e.push((if tables_host { "sample tables: two streaming slots".into() } else { "sample tables, resident".into() }, if tables_host { 2 * tab_max } else { tab_all }, 0));
                    e.push(("model scratch (node images, accumulators, metric weights)".into(), scratch, if state_host { n.nk * n.nv * 4 } else { n.nk * n.nv * 4 }));
                    if n.prior_dev > 0 {
                        e.push(("prior on the device".into(), n.prior_dev, 0));
                    }
                    if state_host && !store.is_f32() {
                        let h = n.stack + sb(store.grad) + if n.quasi { sb(store.grad) + sb(store.pair) + n.chunk_max } else { 0 };
                        e.push((format!("parameter state ({}; host)", store.describe()), 0, h));
                    } else if state_host {
                        e.push(("parameter state: x, g, x_prev, g_prev (host)".into(), 0, 4 * n.stack));
                    } else {
                        let dev_st = 3 + if n.quasi && !prev_host { 3 } else { 0 };
                        e.push((format!("parameter state: {dev_st} stacks on the device (x, g, step{})", if n.quasi && !prev_host { ", x_prev, g_prev, w" } else { "" }), dev_st * n.stack, n.stack + if prev_host { 3 * n.stack } else { 0 }));
                    }
                    e.push(("margin (allocator, kernels)".into(), margin, (2.0 * GIB) as usize));
                    let dev: usize = e.iter().map(|x| x.1).sum();
                    let host: usize = e.iter().map(|x| x.2).sum();
                    // traffic per gradient (s): staged data and streamed tables over pinned memory; host state θ up 2E
                    // times and the gradient up / down E times over pageable memory (E = sets)
                    let ns = n.nsets.max(1) as f64;
                    let mut tr = 0.0;
                    if data_host && !kres {
                        tr += data_all as f64 * kr.unwrap_or(1.0) / n.h2d;
                    }
                    if tables_host {
                        tr += tab_all as f64 / n.h2d;
                    }
                    if state_host {
                        // θ up twice and the gradient down per set: the gradient download at the BFP size when packed
                        tr += (3.0 + ratio(store.grad)) * ns * n.stack as f64 / n.pageable;
                    }
                    if prev_host {
                        tr += 4.0 * n.stack as f64 / n.h2d;
                    }
                    let p = LedgerPlan { state_on_host: state_host, prev_on_host: prev_host, data_on_host: data_host, tables_on_host: tables_host, device_peak: dev, host_peak: host, transfer_s: tr, entries: e };
                    if dev > n.dev_budget || host > n.host_avail {
                        rejected.push(format!("state {}, data {}, tables {}{}: device {:.2} / {:.2} GiB, host {:.2} / {:.2} GiB", if state_host { "host" } else { "device" }, if data_host { "host" } else { "device" }, if tables_host { "host" } else { "device" }, if prev_host { ", pair host" } else { "" }, dev as f64 / GIB, n.dev_budget as f64 / GIB, host as f64 / GIB, n.host_avail as f64 / GIB));
                        if best.is_none() && state_host && data_host && (tables_host || tab_all == 0) {
                            // keep the leanest plan for the refusal message
                            rejected.push(p.describe());
                        }
                        continue;
                    }
                    if best.as_ref().is_none_or(|b| p.transfer_s < b.transfer_s) {
                        best = Some(p);
                    }
                }
            }
        }
    }
    best.ok_or_else(|| format!("knot mode: no placement fits (device budget {:.2} GiB, host {:.2} GiB available):\n  {}", n.dev_budget as f64 / GIB, n.host_avail as f64 / GIB, rejected.join("\n  ")))
}

/// Size the knot engine's grid slot from the ledger (research/memory_ledger.md §3: the slot as a ledger entry, sized
/// before the engine is built): `n.fixed` holds the operator's device bytes WITHOUT the grid slot; `slot_of(cb)` is
/// the slot of a coil batch of `cb` coils (the BFP intermediates included). The largest batch (≤ `cb_cap` coils) whose
/// slot fits next to the rest of the plan bounds the choice; among the batches up to it the rule of the engine's
/// planner (`hasty_nufft::plan_batch`, research/sample_kernels.md §18.7 / §18.9): the fewest transformed channels
/// (every coil batch transforms a full slot, so a ragged last batch costs FFT time for nothing), then the fewest
/// passes. The placement at that slot is [`plan_with`]'s (least transfer). `safety` (e.g. 0.05) is kept free of the
/// device budget for the estimate's error. Returns (slot bytes, coil batch, the plan at that slot); Err(the refusal at
/// one coil) when not even one coil fits.
#[allow(clippy::too_many_arguments)]
pub fn slot_budget_with(n: &LedgerIn, store: crate::bfpstate::StateStore, kst: Option<(crate::kstore::BfpRule, bool)>, force_state: Option<bool>, force_data: Option<bool>, force_tables: Option<bool>, nc: usize, cb_cap: usize, safety: f64, slot_of: &dyn Fn(usize) -> usize) -> Result<(usize, usize, LedgerPlan), String> {
    let nc = nc.max(1);
    let cap = cb_cap.clamp(1, nc);
    let mut m = n.clone();
    m.dev_budget = (n.dev_budget as f64 * (1.0 - safety.clamp(0.0, 0.5))) as usize;
    let fits = |cb: usize| -> Result<LedgerPlan, String> {
        let mut q = m.clone();
        q.fixed = n.fixed + slot_of(cb);
        plan_with(&q, store, kst, force_state, force_data, force_tables)
    };
    // the largest batch that fits (a smaller slot always fits where a larger one does)
    let mut cb_max = 0;
    let mut last = String::new();
    for cb in (1..=cap).rev() {
        match fits(cb) {
            Ok(_) => {
                cb_max = cb;
                break;
            }
            Err(e) => last = e,
        }
    }
    if cb_max == 0 {
        return Err(last);
    }
    let cb = (1..=cb_max).min_by_key(|&c| (nc.div_ceil(c) * c, nc.div_ceil(c), std::cmp::Reverse(c))).unwrap();
    // the engine balances its batches at the same pass count
    let cb = nc.div_ceil(nc.div_ceil(cb));
    let pl = fits(cb)?;
    Ok((slot_of(cb), cb, pl))
}

/// [`slot_budget_with`] with the host-state store and k-space staging of the env (as [`plan`]); `has_sink`: whether
/// the objective hands its gradient over set by set (`Objective::has_gradient_sink`; the BFP state then applies to g)
pub fn slot_budget(n: &LedgerIn, has_sink: bool, force_state: Option<bool>, force_data: Option<bool>, force_tables: Option<bool>, nc: usize, cb_cap: usize, safety: f64, slot_of: &dyn Fn(usize) -> usize) -> Result<(usize, usize, LedgerPlan), String> {
    let store = crate::bfpstate::StateStore::from_env()?;
    let kst = crate::kstage::stage_from_env().map_err(|e| e.to_string())?;
    slot_budget_with(n, store.effective(has_sink), kst, force_state, force_data, force_tables, nc, cb_cap, safety, slot_of)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_weights_partition_and_layout() {
        let amp: Vec<f64> = (0..1000).map(|i| ((i * 37) % 1000) as f64 / 1000.0).collect();
        for (kr, rdeg, mc, cdeg) in [(6, 1, 16, 3), (4, 0, 10, 3), (2, 1, 4, 1), (3, 0, 5, 0)] {
            let b = KnotBasis::from_amplitude(&amp, kr, rdeg, mc, cdeg);
            for &(a, ph) in &[(0.0, 0.0), (0.33, 0.51), (0.999, 0.97), (0.5, 0.25)] {
                let w = b.weights(a, ph);
                assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-12, "partition of unity ({kr},{rdeg},{mc},{cdeg})");
            }
            // sample weights agree with the dense weights
            for &(a, ph) in &[(0.2, 0.1), (0.71, 0.83)] {
                let (ri, rv) = b.locate(a);
                let t = b.cardiac_t(ph);
                let sw = b.sample_weights(t, ri, rv);
                let dw = b.weights(a, ph);
                let mut dense = vec![0.0; b.nk()];
                for (k, w) in sw {
                    dense[k] += w;
                }
                for k in 0..b.nk() {
                    assert!((dense[k] - dw[k]).abs() < 1e-6, "sample weights ({kr},{rdeg},{mc},{cdeg}) knot {k}");
                }
            }
        }
        // cardiac knot b sits at phase (b + ½)/mc: cubic weight there 2/3, linear 1
        let b = KnotBasis::from_amplitude(&amp, 6, 1, 16, 3);
        assert!((b.cardiac_weights(b.cardiac_t(5.5 / 16.0))[5] - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(KnotBasis::parse_spec("6:1,16:3").unwrap(), (6, 1, 16, 3));
        assert!(KnotBasis::parse_spec("6:2,16:3").is_err());
    }

    #[test]
    fn op_frames_groups_prefilter_weights() {
        // 6 knots × 48 frames: 2 intervals per group (2 · 48 < 128), groups 0–2, 2–4, 3–5
        let of = OpFrames::new(6, 16, 3, 48, 3);
        assert_eq!((of.kre, of.r0.clone()), (3, vec![0, 2, 3]));
        assert_eq!(of.images_per_encode(), 9 * 48);
        for (ri, want) in [(0u32, (0usize, 0u32)), (1, (0, 1)), (2, (1, 0)), (3, (1, 1)), (4, (2, 1))] {
            assert_eq!(of.group_of(ri), want, "interval {ri}");
        }
        let of2 = OpFrames::new(6, 16, 3, 24, 3);
        assert_eq!((of2.kre, of2.ng()), (6, 1));
        assert_eq!(OpFrames::new(6, 16, 3, 96, 3).r0, vec![0, 1, 2, 3, 4]);
        // the prefilter inverts the cubic B-spline at the frames: Σ_b β(f − b) c_b = x_f
        let c = of.prefilter();
        let x: Vec<f64> = (0..48).map(|f| ((f as f64) * 0.37).sin() + 0.2 * (f as f64 * 1.3).cos()).collect();
        let coef: Vec<f64> = (0..48).map(|f| (0..48).map(|n| c[n] * x[(f + 48 - n) % 48]).sum()).collect();
        for f in 0..48 {
            let back = (2.0 / 3.0) * coef[f] + (coef[(f + 1) % 48] + coef[(f + 47) % 48]) / 6.0;
            assert!((back - x[f]).abs() < 1e-12, "prefilter frame {f}");
        }
        assert!((0..48).all(|n| (c[n] - c[(48 - n) % 48]).abs() < 1e-14), "prefilter symmetric");
        // frame weights: partition of unity, ≤ 4 cardiac knots; the tables list every (frame, knot) once both ways
        for f in 0..48 {
            let w = of.frame_weights(f);
            assert!(w.len() <= 4 && (w.iter().map(|p| p.1).sum::<f64>() - 1.0).abs() < 1e-12);
        }
        let (fi, _, nn, ki, _, nkf, prim) = frame_tables(&of);
        let a = fi.iter().filter(|&&v| v >= 0).count();
        let b = ki.iter().filter(|&&v| v >= 0).count();
        assert_eq!(a, b);
        assert!(nn <= 4 && nkf <= 13 && prim.len() == 48);
        assert_eq!(OpFrames::parse_spec("frames:48").unwrap(), (48, 3));
        assert!(OpFrames::parse_spec("frames:48:2").is_err());
    }

    #[test]
    fn ledger_bfp_state_and_kspace_1mm_box() {
        // the 1 mm box of research/memory_ledger.md (R2): 280 sets (40 frames × 7 encodes) of 8.53 M voxels (streamed
        // once per gradient: one θ set in this ledger's transfer model), one
        // stack 17.8 GiB, k-space 7.69 GiB (largest set 1/280 × 1.3), tables 9.30 GiB, operator ≈ 13.4 GiB resident
        // (maps 1.50, diagonals 5.25, fat 0.68, grids 6.0), RTX 4090 (23 GiB usable), host: 80 G cgroup − 16.6 GiB setup
        let gib = 1usize << 30;
        let g = |x: f64| (x * gib as f64) as usize;
        let n = LedgerIn { nv: 8_530_000, nk: 1, img_frames: 1, slots: 0, stack: g(17.8), chunk_max: g(17.8 / 280.0), nsets: 1, data_all: g(7.69), data_max: g(7.69 / 280.0 * 1.3), data_movable: true, tab_all: g(9.30), tab_max: g(9.30 / 280.0 * 1.3), fixed: g(13.4), dev_budget: 23 * gib, host_avail: g(80.0 * 0.931 - 16.6), quasi: true, prior_dev: 0, h2d: 22e9, pageable: 10e9 };
        use crate::bfpstate::StateStore;
        use crate::kstore::BfpRule;
        let f32p = plan_with(&n, StateStore::F32, None, Some(true), None, None);
        println!("f32 host state: {}", match &f32p { Ok(p) => p.describe(), Err(e) => e.clone() });
        assert!(f32p.is_err(), "4 × 17.8 GiB must not fit the host next to the setup");
        let b16 = StateStore::parse("bfp16").unwrap();
        // an objective without a gradient sink (the velocity model): only the secant record shrinks
        let ns = plan_with(&n, b16.effective(false), None, Some(true), None, None);
        println!("bfp16 record only (no gradient sink): {}", match &ns { Ok(p) => p.describe(), Err(e) => e.clone() });
        let p = plan_with(&n, b16, None, Some(true), None, None).unwrap();
        println!("bfp16 host state:\n{}", p.describe());
        let st = p.entries.iter().find(|e| e.0.starts_with("parameter state")).unwrap().2 as f64 / gib as f64;
        assert!((st - 45.0).abs() < 0.5, "state {st:.2} GiB");
        let pk = plan_with(&n, b16, Some((BfpRule::Sigma { q: 0.25 }, false)), Some(true), Some(true), None).unwrap();
        let pr = plan_with(&n, b16, Some((BfpRule::Sigma { q: 0.25 }, true)), Some(true), Some(true), None).unwrap();
        println!("+ packed k-space staging:\n{}+ resident packed k-space:\n{}", pk.describe(), pr.describe());
        assert!(pk.transfer_s < p.transfer_s && pr.transfer_s < pk.transfer_s);
        // f32 plans are unchanged by the new arguments
        let m = LedgerIn { stack: gib, host_avail: 60 * gib, ..n.clone() };
        assert_eq!(plan_with(&m, StateStore::F32, None, None, None, None).unwrap(), plan(&m, None, None, None).unwrap());
    }

    #[test]
    fn ledger_slot_budget_fewest_passes() {
        use crate::bfpstate::StateStore;
        let gib = 1usize << 30;
        // the knot operator without its slot (3 GiB), small state: the slot takes what the plan leaves
        let n = LedgerIn { nv: 1_000_000, nk: 96, img_frames: 0, slots: 0, stack: gib / 2, chunk_max: 12_000_000, nsets: 7, data_all: 2 * gib, data_max: gib / 3, data_movable: true, tab_all: 2 * gib, tab_max: gib / 3, fixed: 3 * gib, dev_budget: 23 * gib, host_avail: 60 * gib, quasi: true, prior_dev: 0, h2d: 20e9, pageable: 8e9 };
        let slot = |cb: usize| cb * gib;
        // 10 coils of 1 GiB: all 10 fit next to the rest (one pass)
        let (s, cb, p) = slot_budget_with(&n, StateStore::F32, None, None, None, None, 10, 16, 0.05, &slot).unwrap();
        assert_eq!((cb, s), (10, 10 * gib), "{}", p.describe());
        assert!(p.device_peak <= (23.0 * 0.95 * gib as f64) as usize);
        // 2 GiB per coil: 5 coils (2 passes), the largest equal batch that fits
        let (_, cb, _) = slot_budget_with(&n, StateStore::F32, None, None, None, None, 10, 16, 0.05, &|c| 2 * c * gib).unwrap();
        assert_eq!(cb, 5);
        // 5 GiB per coil: at most 3 coils fit; 3 would be ragged (3, 3, 3, 1: 12 transformed channels), 2 coils (5
        // passes, 10 channels) is the planner's choice
        let (_, cb, _) = slot_budget_with(&n, StateStore::F32, None, None, None, None, 10, 16, 0.05, &|c| 5 * c * gib).unwrap();
        assert_eq!(cb, 2);
        // the coil cap
        let (_, cb, _) = slot_budget_with(&n, StateStore::F32, None, None, None, None, 32, 16, 0.05, &|c| c * gib / 8).unwrap();
        assert_eq!(cb, 16);
        // not even one coil fits: Err
        assert!(slot_budget_with(&n, StateStore::F32, None, None, None, None, 10, 16, 0.05, &|c| 40 * c * gib).is_err());
    }

    #[test]
    fn ledger_prefers_device_and_refuses_clearly() {
        let gib = 1usize << 30;
        let mut n = LedgerIn { nv: 4_170_000, nk: 96, img_frames: 0, slots: 0, stack: 8 * gib, chunk_max: 50_000_000, nsets: 7, data_all: 7 * gib, data_max: gib, data_movable: true, tab_all: 7 * gib, tab_max: gib, fixed: 5 * gib, dev_budget: 23 * gib, host_avail: 60 * gib, quasi: true, prior_dev: 0, h2d: 20e9, pageable: 10e9 };
        let p = plan(&n, None, None, None).unwrap();
        assert!(p.state_on_host, "8 GiB stacks do not fit 23 GiB next to the operator: {}", p.describe());
        n.stack = gib / 4;
        n.nk = 8;
        n.fixed = 3 * gib;
        let p = plan(&n, None, None, None).unwrap();
        assert!(!p.state_on_host && !p.data_on_host && !p.tables_on_host, "{}", p.describe());
        n.host_avail = gib;
        n.stack = 8 * gib;
        n.nk = 96;
        let e = plan(&n, None, None, None).unwrap_err();
        assert!(e.contains("no placement fits"), "{e}");
    }
}
