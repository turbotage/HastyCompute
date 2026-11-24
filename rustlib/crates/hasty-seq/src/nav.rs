//! Navigator ADC windows of the flow TR (research/motion_navigators.md, research/sequence_design.md §7–9): besides the
//! readout ADC, the ADC also runs during the velocity-encode block and during the tail (ramp-down / rewinder / spoiler
//! block) of every TR. These samples are navigator / auxiliary data only, never reconstructed: every reconstruction
//! sample lies inside kmax, the encode lobes reach ≈ 2.3 kmax and the spoiler 4 kmax.
//!
//! Timing rules (Pulseq convention, as [`crate::pulseq::Sequence::check`]):
//! - every ADC event has the dead time `dead` before its first sample and after its last one, inside its own block, and
//!   two consecutive ADC events (encode | readout | tail) are ≥ 2·dead apart (the readout ADC starts at the echo, i.e.
//!   at the encode block's end, and ends at or before the tail block's start);
//! - an ADC event starts no earlier than `rf_ringdown` after the RF end (and after the encode block start);
//! - the tail window ends `dead` before the TR end (the next TR's RF block carries its own RF dead time);
//! - samples lie on the dwell grid counted from the TR start, like the readout (readout sample n at i_ro·dt + n·dwell):
//!   sample n of an event at t0 + n·dwell, its interval [t, t + dwell) inside the window;
//! - a window with more than `max_samples` samples is split into ADC events 2·dead apart (gap rounded up to the dwell).
//!
//! GE / pge2 ([`AdcRules::ge_pge2`]; HarmonizedMRI design rules, notes in [`crate::pulseq`] and
//! research/girf_qa_sequence.md §1): dwell a multiple of 2 µs, ADC dead 40 µs, RF ring-down 54 µs (60 µs used). The
//! maximum number of samples per ADC event is not documented [CHECK at the scanner]; 8192 is assumed (the Siemens cap),
//! which no navigator window reaches. pge2's 117 µs per segment instance (one TR = encode | rotated readout | tail
//! segments) lengthens the TR; it is not part of the design here.
//!
//! k positions of the navigator samples: γ̄ M0 of the physical TR waveform at the sample times, integrated from the
//! excitation (exact for piecewise-linear gradients, [`crate::wave::Grad::moments_at`]), as for the readout.

use crate::tr::{Part, Tr};
use crate::{Mat3, Vec3, GAMMA_BAR};

/// ADC timing rules for the navigator windows
#[derive(Clone, Debug, PartialEq)]
pub struct AdcRules {
    /// dwell, s (the readout's)
    pub dwell: f64,
    /// dead time before the first and after the last sample of every ADC event, s
    pub dead: f64,
    /// RF ring-down: earliest ADC start after the RF end, s
    pub rf_ringdown: f64,
    /// samples per ADC event (None: no cap)
    pub max_samples: Option<usize>,
}

impl AdcRules {
    /// GE via pge2 (the likely scanner): ADC dead 40 µs, RF ring-down 60 µs, ≤ 8192 samples per event [CHECK]
    pub fn ge_pge2(dwell: f64) -> Self {
        Self { dwell, dead: 40e-6, rf_ringdown: 60e-6, max_samples: Some(8192) }
    }
    /// the design system's own numbers ([`crate::system::System`]: Prisma 10 µs dead, 30 µs ring-down), no cap
    pub fn from_system(sys: &crate::system::System, dwell: f64) -> Self {
        Self { dwell, dead: sys.adc_dead, rf_ringdown: sys.rf_ringdown, max_samples: None }
    }
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({"dwell": self.dwell, "dead": self.dead, "rf_ringdown": self.rf_ringdown, "max_samples": self.max_samples})
    }
    pub fn from_json(v: &serde_json::Value) -> Option<Self> {
        Some(Self {
            dwell: v["dwell"].as_f64()?,
            dead: v["dead"].as_f64()?,
            rf_ringdown: v["rf_ringdown"].as_f64()?,
            max_samples: v["max_samples"].as_u64().map(|n| n as usize),
        })
    }
}

/// one navigator ADC window (one or more ADC events) inside a gradient block of the TR
#[derive(Clone, Debug, PartialEq)]
pub struct AdcWindow {
    /// [`Part::Encode`] or [`Part::Tail`]
    pub part: Part,
    /// the gradient block it lies in, TR raster indices [a, b] (time a·dt … b·dt from the TR start)
    pub block: [usize; 2],
    pub dwell: f64,
    /// ADC events: time of the first sample from the TR start (s), number of samples
    pub events: Vec<(f64, usize)>,
}

impl AdcWindow {
    /// "encode" | "tail"
    pub fn name(&self) -> &'static str {
        match self.part {
            Part::Encode => "encode",
            Part::Tail => "tail",
            Part::Readout => "readout",
            Part::Rf => "rf",
        }
    }
    pub fn n(&self) -> usize {
        self.events.iter().map(|e| e.1).sum()
    }
    /// sample times from the TR start, events in order
    pub fn times(&self) -> Vec<f64> {
        self.events.iter().flat_map(|&(t0, n)| (0..n).map(move |i| t0 + i as f64 * self.dwell)).collect()
    }
    /// [first sample index, samples] of every event within the window's sample list
    pub fn event_ranges(&self) -> Vec<[usize; 2]> {
        let mut a = 0;
        self.events
            .iter()
            .map(|e| {
                let r = [a, e.1];
                a += e.1;
                r
            })
            .collect()
    }
    /// description for summary.json / meta.json; `t_exc` = the RF centre (s from the TR start)
    pub fn to_json(&self, t_exc: f64) -> serde_json::Value {
        let t = self.times();
        serde_json::json!({
            "name": self.name(), "block_raster": self.block, "dwell": self.dwell, "n": self.n(),
            "events": self.events.iter().map(|e| serde_json::json!({"t0_tr": e.0, "t0_exc": e.0 - t_exc, "n": e.1})).collect::<Vec<_>>(),
            "event_ranges": self.event_ranges(),
            "t_first_exc": t.first().map(|x| x - t_exc), "t_last_exc": t.last().map(|x| x - t_exc),
        })
    }
}

/// the longest run of samples on the dwell grid (from the TR start) whose intervals lie in [lo, hi] (s), split into
/// events of ≤ max_samples, 2·dead apart
fn place(lo: f64, hi: f64, r: &AdcRules) -> Vec<(f64, usize)> {
    let d = r.dwell;
    let mut k = (lo / d - 1e-9).ceil() as i64;
    let k_end = (hi / d + 1e-9).floor() as i64; // sample intervals end at ≤ k_end·d
    let gap = (2.0 * r.dead / d - 1e-9).ceil() as i64;
    let mut ev = vec![];
    while k < k_end {
        let n = ((k_end - k) as usize).min(r.max_samples.unwrap_or(usize::MAX));
        ev.push((k as f64 * d, n));
        k += n as i64 + gap;
    }
    ev
}

/// navigator windows of a TR given its timing (all s or TR raster indices, from the TR start): RF end `rf_end`, encode
/// block [i_enc, i_ro], readout ADC [i_ro·dt, ro_end) (its first sample at the echo), tail block [i_tail, n_tr]
#[allow(clippy::too_many_arguments)]
pub fn nav_windows(dt: f64, rf_end: f64, i_enc: usize, i_ro: usize, ro_end: f64, i_tail: usize, n_tr: usize, r: &AdcRules) -> Vec<AdcWindow> {
    let enc_lo = (i_enc as f64 * dt).max(rf_end + r.rf_ringdown) + r.dead;
    let enc_hi = i_ro as f64 * dt - 2.0 * r.dead;
    let tail_lo = (ro_end + 2.0 * r.dead).max(i_tail as f64 * dt + r.dead);
    let tail_hi = n_tr as f64 * dt - r.dead;
    let mut w = vec![];
    for (part, block, lo, hi) in [(Part::Encode, [i_enc, i_ro], enc_lo, enc_hi), (Part::Tail, [i_tail, n_tr], tail_lo, tail_hi)] {
        let events = place(lo, hi, r);
        if !events.is_empty() {
            w.push(AdcWindow { part, block, dwell: r.dwell, events });
        }
    }
    w
}

impl Tr {
    /// end of the readout ADC (s from the TR start): floor(T_adc / dwell) samples from the echo, as the simulator
    /// schedule (hasty-bench `FlowSchedule`) places them
    pub fn readout_adc_end(&self, dwell: f64) -> f64 {
        let n = ((self.spoke.i_adc as f64 * self.dt) / dwell).floor();
        self.i_ro as f64 * self.dt + n * dwell
    }
    /// navigator ADC windows (encode block, tail block) of this TR under `rules`
    pub fn nav_windows(&self, rules: &AdcRules) -> Vec<AdcWindow> {
        nav_windows(self.dt, self.rf_end, self.i_enc, self.i_ro, self.readout_adc_end(rules.dwell), self.i_tail, self.n, rules)
    }
    /// k (cycles/m, physical axes) of encode `e`, rotation `r` at the window's samples: γ̄ M0 from the excitation
    pub fn nav_k(&self, e: usize, r: &Mat3, w: &AdcWindow) -> Vec<Vec3> {
        self.physical(e, r).moments_at(self.t_rf, &w.times()).iter().map(|m| m.0.map(|x| GAMMA_BAR * x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::{DualLayout, Scheme};
    use crate::system::System;
    use crate::tr::{EncodeShape, Spoil, TrSpec};
    use crate::traj::{rotations, RotSet, Spoke, TopOpts, Yarn};

    fn flow_tr() -> (Tr, Vec<Mat3>) {
        let sys = System::prisma_xr();
        let y = Yarn { kmax: 250.0, fov: 0.32, n_design: 2200.0, edge_density: 1.0, kg0: 0.0, wobble: 0.0, wobble_freq: 2.0 };
        let mut sp = Spoke::design(&y.curve_return(60_000, 0.0), &System { smax: sys.smax * 0.45, ..sys.clone() }, &TopOpts { v_end: Some(0.0), ..TopOpts::default() });
        sp.close_to(&[0.0; 3], sys.gmax, sys.smax * 0.45);
        let spoil = [0.0, 0.0, 2.0 / (crate::GAMMA_BAR * 2e-3)];
        let spec = TrSpec { sys: sys.clone(), enc_slew: 0.3, tail_slew: 0.5, rf_dur: 100e-6, scheme: Scheme::Coprime { p: 2, layout: DualLayout::Seven }, venc: 1.0, balanced: true, spoil: Spoil::FlowComp(spoil), encode_shape: EncodeShape::TwoLobe, tail_shape: EncodeShape::TwoLobe };
        let rots = rotations(RotSet::SuperFibonacci, 16);
        (Tr::build(&spec, sp, &rots), rots)
    }

    fn on_grid(t: f64, d: f64) -> bool {
        (t / d - (t / d).round()).abs() < 1e-6
    }

    #[test]
    fn windows_inside_their_blocks_and_timing_rules() {
        let (tr, _) = flow_tr();
        for rules in [AdcRules::ge_pge2(2e-6), AdcRules::from_system(&System::prisma_xr(), 2e-6), AdcRules::ge_pge2(4e-6)] {
            let w = tr.nav_windows(&rules);
            assert_eq!(w.len(), 2);
            assert_eq!((w[0].part, w[1].part), (Part::Encode, Part::Tail));
            let (d, dead) = (rules.dwell, rules.dead);
            let ro0 = tr.i_ro as f64 * tr.dt;
            let ro1 = tr.readout_adc_end(d);
            for win in &w {
                let (a, b) = (win.block[0] as f64 * tr.dt, win.block[1] as f64 * tr.dt);
                for &(t0, n) in &win.events {
                    let t1 = t0 + n as f64 * d;
                    // inside the gradient block, dead time inside it too, on the dwell grid
                    assert!(t0 - dead >= a - 1e-12 && t1 + dead <= b + 1e-12, "{} event [{t0}, {t1}) outside [{a}, {b}]", win.name());
                    assert!(on_grid(t0, d));
                    assert!(n <= rules.max_samples.unwrap_or(usize::MAX));
                }
                // maximal: one more sample at either end would break a rule
                let (t0, _) = win.events[0];
                let (tl, nl) = *win.events.last().unwrap();
                let t_end = tl + nl as f64 * d;
                match win.part {
                    Part::Encode => {
                        assert!(t0 - dead >= tr.rf_end + rules.rf_ringdown - 1e-12, "encode ADC starts in the RF ring-down");
                        assert!(t0 - d - dead < (tr.i_enc as f64 * tr.dt).max(tr.rf_end + rules.rf_ringdown) - 1e-12);
                        assert!(t_end + 2.0 * dead <= ro0 + 1e-12, "encode ADC too close to the readout ADC");
                        assert!(t_end + d + 2.0 * dead > ro0 + 1e-12);
                    }
                    Part::Tail => {
                        assert!(t0 - 2.0 * dead >= ro1 - 1e-12, "tail ADC too close to the readout ADC");
                        assert!(t0 - d < (ro1 + 2.0 * dead).max(tr.i_tail as f64 * tr.dt + dead) - 1e-12);
                        assert!(t_end + dead <= tr.tr() + 1e-12);
                        assert!(t_end + d + dead > tr.tr() + 1e-12);
                    }
                    _ => unreachable!(),
                }
            }
            println!("rules {rules:?}: encode {} samples ({:.3}–{:.3} ms after RF), tail {} ({:.3}–{:.3} ms)", w[0].n(), (w[0].times()[0] - tr.t_rf) * 1e3, (w[0].times().last().unwrap() - tr.t_rf) * 1e3, w[1].n(), (w[1].times()[0] - tr.t_rf) * 1e3, (w[1].times().last().unwrap() - tr.t_rf) * 1e3);
        }
    }

    #[test]
    fn split_into_events_with_gaps() {
        let (tr, _) = flow_tr();
        let one = tr.nav_windows(&AdcRules::ge_pge2(2e-6));
        let rules = AdcRules { max_samples: Some(100), ..AdcRules::ge_pge2(2e-6) };
        let w = tr.nav_windows(&rules);
        for (win, whole) in w.iter().zip(&one) {
            assert!(win.events.len() > 1);
            assert_eq!(win.events[0].0, whole.events[0].0);
            for p in win.events.windows(2) {
                let end = p[0].0 + p[0].1 as f64 * rules.dwell;
                assert!(p[1].0 - end >= 2.0 * rules.dead - 1e-12, "ADC events closer than 2 dead times");
                assert!(p[1].0 - end < 2.0 * rules.dead + rules.dwell);
            }
            let (tl, nl) = *win.events.last().unwrap();
            assert!(tl + nl as f64 * rules.dwell <= whole.events[0].0 + whole.n() as f64 * rules.dwell + 1e-12);
            assert!(win.n() < whole.n());
            let er = win.event_ranges();
            assert_eq!(er.last().map(|r| r[0] + r[1]), Some(win.n()));
        }
    }

    #[test]
    fn nav_k_from_the_gradient_moments() {
        let (tr, rots) = flow_tr();
        let w = tr.nav_windows(&AdcRules::ge_pge2(2e-6));
        // independent reference: γ̄ ∫ G by the trapezoid rule on a 0.5 µs grid through the raster points (exact: G is
        // piecewise linear on the 10 µs raster)
        let kref = |g: &crate::wave::Grad, t: f64| -> Vec3 {
            let h = 0.5e-6;
            let n = (t / h).floor() as usize;
            let mut k = [0.0; 3];
            let mut add = |t0: f64, t1: f64| {
                let (a, b) = (g.at(t0), g.at(t1));
                for q in 0..3 {
                    k[q] += 0.5 * (a[q] + b[q]) * (t1 - t0) * GAMMA_BAR;
                }
            };
            for i in 0..n {
                add(i as f64 * h, (i + 1) as f64 * h);
            }
            add(n as f64 * h, t);
            k
        };
        let kmax = 250.0;
        for e in [0, 1, 3, 6] {
            let g = tr.physical(e, &rots[0]);
            for win in &w {
                let k = tr.nav_k(e, &rots[0], win);
                let t = win.times();
                for i in (0..t.len()).step_by(37) {
                    let kr = kref(&g, t[i]);
                    let err = crate::norm(&crate::sub(&k[i], &kr));
                    assert!(err < 1e-6 * kmax, "encode {e} {} sample {i}: {:?} vs {kr:?}", win.name(), k[i]);
                }
                // the encode block is on physical axes (same k for every rotation); so is the FlowComp tail once the
                // readout is closed at k = 0
                let k2 = tr.nav_k(e, &rots[7], win);
                let dk = k.iter().zip(&k2).map(|(a, b)| crate::norm(&crate::sub(a, b))).fold(0.0, f64::max);
                assert!(dk < 1e-6 * kmax, "{} k depends on the rotation: {dk}", win.name());
            }
            // the encode lobes leave kmax (navigator only); the tail ends near the spoil moment (2 cycles / 2 mm = 4 kmax)
            let kt = tr.nav_k(e, &rots[0], &w[1]);
            let kend = crate::norm(kt.last().unwrap());
            assert!(kend > 3.5 * kmax && kend < 4.1 * kmax, "tail end |k| {kend}");
        }
        // the encodes differ in the encode window (different M1 → different lobes)
        let k0 = tr.nav_k(0, &rots[0], &w[0]);
        let k1 = tr.nav_k(1, &rots[0], &w[0]);
        assert!(k0.iter().zip(&k1).map(|(a, b)| crate::norm(&crate::sub(a, b))).fold(0.0, f64::max) > 1.0);
    }

    /// the chosen design's timing (data/hasty/seq/flow_thorax summary.json: i_enc 23, i_ro 175, i_adc 798, i_tail 973,
    /// n 1129 at 10 µs; RF 100–200 µs, centre 150 µs) under the GE rules at 2 µs
    #[test]
    fn flow_thorax_counts() {
        let rules = AdcRules::ge_pge2(2e-6);
        let ro_end = 175.0 * 10e-6 + (798.0f64 * 10e-6 / 2e-6).floor() * 2e-6;
        let w = nav_windows(10e-6, 200e-6, 23, 175, ro_end, 973, 1129, &rules);
        assert_eq!(w[0].n(), 685);
        assert_eq!(w[1].n(), 720);
        let (te, tt) = (w[0].times(), w[1].times());
        assert!((te[0] - 300e-6).abs() < 1e-12 && (te.last().unwrap() - 1668e-6).abs() < 1e-12);
        assert!((tt[0] - 9810e-6).abs() < 1e-12 && (tt.last().unwrap() - 11248e-6).abs() < 1e-12);
    }
}
