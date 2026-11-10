//! `hasty-seq` designs as simulator schedules (`hasty_mrsim::sequence::Schedule`): per-sample moments of the played
//! (nominal, or through a GIRF) physical waveform at the ADC times, exact for piecewise-linear gradients
//! (`Grad::moments_at`).

use std::path::Path;

use hasty_mrsim::physio::Ecg;
use hasty_mrsim::sequence::{sample_record, Schedule, SAMPLE_FLOATS};
use hasty_seq::girf::{played_moments, Girf};
use hasty_seq::multiecho::{golden_dirs, EchoTrain};
use hasty_seq::traj::{acquisition_order, rotations, RotSet};
use hasty_seq::wave::Grad;
use hasty_seq::{mat_vec, Mat3, Vec3};

/// the B0 / R2* / fat train: one readout record per TR (the whole train is one ADC window), TE shifts cycled over TRs,
/// golden-means pair directions in TR order; time origin of the moments = RF centre (hard pulse, no gradient during RF)
pub struct EchoSchedule {
    pub train: EchoTrain,
    pub n_tr: usize,
    pub dwell: f64,
    pub n_adc: usize,
    pub b0: f64,
    pub ecg: Ecg,
    pub n_frames: usize,
}

impl EchoSchedule {
    pub fn new(train: EchoTrain, scan_time: f64, dwell: f64, b0: f64, ecg: Ecg, n_frames: usize) -> Self {
        let n_tr = (scan_time / train.tr).floor() as usize;
        let n_adc = (train.spec.n_lobe as f64 * train.t_lobe / dwell).floor() as usize + 1;
        Self { train, n_tr, dwell, n_adc, b0, ecg, n_frames }
    }
    pub fn shift(&self, i: usize) -> f64 {
        let s = &self.train.spec.te_shifts;
        s[i % s.len()]
    }
    pub fn dirs(&self, i: usize) -> Vec<Vec3> {
        let d = self.train.dirs_per_tr();
        golden_dirs(i * d, d)
    }
    /// ADC sample times of TR i after the RF centre
    pub fn adc_times(&self, i: usize) -> Vec<f64> {
        let t0 = self.train.te1 + self.shift(i);
        (0..self.n_adc).map(|n| t0 + n as f64 * self.dwell).collect()
    }
}

impl Schedule for EchoSchedule {
    fn n_readouts(&self) -> usize {
        self.n_tr
    }
    fn n_adc(&self) -> usize {
        self.n_adc
    }
    fn tr(&self) -> f64 {
        self.train.tr
    }
    fn te(&self) -> f64 {
        self.train.te1
    }
    fn t_exc(&self, i: usize) -> f64 {
        i as f64 * self.train.tr
    }
    fn readout_row(&self, i: usize) -> [f64; 14] {
        let card = self.ecg.at(self.t_exc(i));
        let frame = ((card.phase * self.n_frames as f64) as usize).min(self.n_frames.max(1) - 1);
        let nsh = self.train.spec.te_shifts.len();
        let mut v = [0.0; 14];
        v[..5].copy_from_slice(&[self.t_exc(i), (i % nsh) as f64, 0.0, frame as f64, card.phase]);
        v[5] = 1.0;
        v[9] = 1.0;
        v[13] = 1.0;
        v
    }
    fn sample_params(&self, range: std::ops::Range<usize>) -> Vec<f32> {
        let mut out = Vec::with_capacity(range.len() * self.n_adc * SAMPLE_FLOATS);
        for i in range {
            let s = self.shift(i);
            let g = self.train.readout(&self.dirs(i), self.train.te1 + s);
            let times = self.adc_times(i);
            for (m, &t) in g.moments_at(0.0, &times).iter().zip(&times) {
                out.extend_from_slice(&sample_record(&m.0, &m.1, &m.2, &m.3, self.b0, t));
            }
        }
        out
    }
    fn describe(&self) -> String {
        let t = &self.train;
        format!(
            "B0/R2*/fat train: {} {:?} lobes of {:.3} ms ({:?}), res {:.1} mm, TE₁ {:.3} ms, shifts {:?} ms, TR {:.2} ms, dwell {:.1} µs",
            t.spec.n_lobe,
            t.spec.shape,
            t.t_lobe * 1e3,
            t.spec.polarity,
            t.spec.res * 1e3,
            t.te1 * 1e3,
            t.spec.te_shifts.iter().map(|x| x * 1e3).collect::<Vec<_>>(),
            t.tr * 1e3,
            self.dwell * 1e6
        )
    }
}

/// the final 4D-flow TR of a `seqdesign` run (out-and-back yarn, per-encode flow-compensated tail): readout i is
/// rotation j = i / n_enc, encode e = i % n_enc (all encodes of a rotation back to back); the rotations are a
/// super-Fibonacci set with as many members as the scan has rotations, in acquisition order (every contiguous run — a
/// cardiac / respiratory bin — spread over SO(3)). The encode and tail blocks are on physical axes (the run's
/// `tr_grad.npy`), the readout is R_j times the base spoke (`spoke_grad.npy`). Moments per sample through `girf`
/// (`Girf::ideal()`: the nominal waveform), integrated from the excitation, the previous TR included (its tails).
pub struct FlowSchedule {
    pub dt: f64,
    pub t_rf: f64,
    pub i_ro: usize,
    pub i_adc: usize,
    /// raster samples of one TR (the next TR starts at sample n_tr)
    pub n_tr: usize,
    pub n_enc: usize,
    /// [encode][n_tr + 1] physical TR of the run's first rotation (encode and tail parts used)
    pub tr_grad: Vec<Vec<Vec3>>,
    /// base readout, samples 0..=i_adc
    pub base: Vec<Vec3>,
    pub rots: Vec<Mat3>,
    pub n_readouts: usize,
    pub dwell: f64,
    pub n_adc: usize,
    pub b0: f64,
    pub ecg: Ecg,
    pub n_frames: usize,
    pub girf: Girf,
    pub girf_name: String,
    pub meta: serde_json::Value,
}

impl FlowSchedule {
    pub fn load(run: &Path, scan_time: f64, dwell: f64, b0: f64, ecg: Ecg, n_frames: usize, girf: Girf, girf_name: &str) -> Self {
        let rd = |f: &str| hasty_recon::data::read_npy(&run.join(f)).unwrap_or_else(|e| panic!("{}: {e}", run.join(f).display()));
        let summ: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(run.join("summary.json")).unwrap()).unwrap();
        let fin: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(run.join("final.json")).unwrap()).unwrap();
        let b = &summ["best"];
        assert_eq!(b["spoil_mode"].as_str(), Some("flowcomp"), "FlowSchedule needs the per-encode physical tail (--spoil flowcomp)");
        let u = |k: &str| b[k].as_u64().unwrap() as usize;
        let (i_ro, i_adc, n_tr) = (u("i_ro"), u("i_adc"), u("n_tr"));
        let n_enc = fin["n_enc"].as_u64().unwrap() as usize;
        let dt = fin["dt"].as_f64().unwrap();
        let t_rf = fin["t_rf"].as_f64().unwrap();
        let tg = rd("tr_grad.npy").f64s();
        let ntr1 = tg.len() / (3 * n_enc);
        assert!(ntr1 >= n_tr, "tr_grad.npy shorter than the TR");
        let tr_grad: Vec<Vec<Vec3>> = (0..n_enc).map(|e| (0..ntr1).map(|i| [tg[(e * ntr1 + i) * 3], tg[(e * ntr1 + i) * 3 + 1], tg[(e * ntr1 + i) * 3 + 2]]).collect()).collect();
        let sg = rd("spoke_grad.npy").f64s();
        let base: Vec<Vec3> = (0..=i_adc).map(|i| [sg[3 * i], sg[3 * i + 1], sg[3 * i + 2]]).collect();
        let mut fs = Self::assemble(dt, t_rf, i_ro, i_adc, n_tr, tr_grad, base, scan_time, dwell, b0, ecg, n_frames, girf, girf_name);
        fs.meta["run"] = serde_json::json!(run);
        fs.meta["encoding"] = summ["encoding"].clone();
        fs.meta["encode"] = summ.get("encode").cloned().unwrap_or(serde_json::json!({"shape": "two-lobe"}));
        fs.meta["res"] = fin["res"].clone();
        fs.meta["kmax"] = fin["kmax"].clone();
        fs
    }
    /// from a designed TR in memory (needs `Spoil::FlowComp`: encode and tail on physical axes)
    pub fn from_tr(tr: &hasty_seq::tr::Tr, scan_time: f64, dwell: f64, b0: f64, ecg: Ecg, n_frames: usize, girf: Girf, girf_name: &str) -> Self {
        assert!(tr.enc_tail.is_some(), "FlowSchedule needs the per-encode physical tail (Spoil::FlowComp)");
        let tr_grad: Vec<Vec<Vec3>> = (0..tr.n_enc()).map(|e| tr.physical(e, &hasty_seq::IDENTITY).g).collect();
        let base = tr.spoke.grad.g.clone();
        Self::assemble(tr.dt, tr.t_rf, tr.i_ro, tr.spoke.i_adc, tr.n, tr_grad, base, scan_time, dwell, b0, ecg, n_frames, girf, girf_name)
    }
    #[allow(clippy::too_many_arguments)]
    fn assemble(dt: f64, t_rf: f64, i_ro: usize, i_adc: usize, n_tr: usize, tr_grad: Vec<Vec<Vec3>>, base: Vec<Vec3>, scan_time: f64, dwell: f64, b0: f64, ecg: Ecg, n_frames: usize, girf: Girf, girf_name: &str) -> Self {
        let n_enc = tr_grad.len();
        let tr = n_tr as f64 * dt;
        let n_readouts = ((scan_time / tr).floor() as usize / n_enc) * n_enc;
        let n_rot = (n_readouts / n_enc).max(1);
        let set = rotations(RotSet::SuperFibonacci, n_rot);
        let rots: Vec<Mat3> = acquisition_order(RotSet::SuperFibonacci, n_rot).into_iter().map(|i| set[i]).collect();
        let t_adc = i_adc as f64 * dt;
        let n_adc = (t_adc / dwell).floor() as usize;
        let meta = serde_json::json!({
            "kind": "flow", "tr": tr, "te": i_ro as f64 * dt - t_rf, "t_adc": t_adc, "dwell": dwell, "n_enc": n_enc,
            "n_rot": n_rot, "rotations": "super-Fibonacci (n_rot), acquisition_order; readout i = rotation i / n_enc, encode i % n_enc",
            "girf": girf_name,
        });
        Self { dt, t_rf, i_ro, i_adc, n_tr, n_enc, tr_grad, base, rots, n_readouts, dwell, n_adc, b0, ecg, n_frames, girf, girf_name: girf_name.into(), meta }
    }
    /// physical TR samples 0..=n_tr of encode e, rotation r
    pub fn tr_wave(&self, e: usize, r: &Mat3) -> Vec<Vec3> {
        let mut g: Vec<Vec3> = self.tr_grad[e][..=self.n_tr.min(self.tr_grad[e].len() - 1)].to_vec();
        g.resize(self.n_tr + 1, [0.0; 3]);
        for (i, v) in self.base.iter().enumerate() {
            g[self.i_ro + i] = mat_vec(r, v);
        }
        g
    }
    /// ADC sample times after the TR start (the first at the echo)
    pub fn adc_times(&self) -> Vec<f64> {
        (0..self.n_adc).map(|n| self.i_ro as f64 * self.dt + n as f64 * self.dwell).collect()
    }
    /// the same schedule with another chain (e.g. the nominal one for what a reconstruction assumes)
    pub fn with_girf(&self, girf: Girf, name: &str) -> Self {
        let mut meta = self.meta.clone();
        meta["girf"] = serde_json::json!(name);
        Self { tr_grad: self.tr_grad.clone(), base: self.base.clone(), rots: self.rots.clone(), ecg: self.ecg.clone(), girf, girf_name: name.into(), meta, ..*self }
    }
}

impl Schedule for FlowSchedule {
    fn n_readouts(&self) -> usize {
        self.n_readouts
    }
    fn n_adc(&self) -> usize {
        self.n_adc
    }
    fn tr(&self) -> f64 {
        self.n_tr as f64 * self.dt
    }
    fn te(&self) -> f64 {
        self.i_ro as f64 * self.dt - self.t_rf
    }
    fn t_exc(&self, i: usize) -> f64 {
        i as f64 * self.tr() + self.t_rf
    }
    fn readout_row(&self, i: usize) -> [f64; 14] {
        let card = self.ecg.at(self.t_exc(i));
        let frame = ((card.phase * self.n_frames as f64) as usize).min(self.n_frames.max(1) - 1);
        let r = &self.rots[i / self.n_enc];
        let mut v = [0.0; 14];
        v[..5].copy_from_slice(&[self.t_exc(i), (i % self.n_enc) as f64, 0.0, frame as f64, card.phase]);
        for a in 0..3 {
            for b in 0..3 {
                v[5 + 3 * a + b] = r[a][b];
            }
        }
        v
    }
    fn sample_params(&self, range: std::ops::Range<usize>) -> Vec<f32> {
        let times = self.adc_times();
        let ideal = matches!(self.girf_name.as_str(), "ideal" | "none" | "");
        let per: Vec<Vec<f32>> = hasty_seq::par_map(&range.collect::<Vec<_>>(), |&i| {
            let (j, e) = (i / self.n_enc, i % self.n_enc);
            let this = Grad { dt: self.dt, t0: 0.0, g: self.tr_wave(e, &self.rots[j]) };
            let prev: Vec<Vec3> = if ideal || i == 0 {
                vec![]
            } else {
                let (jp, ep) = ((i - 1) / self.n_enc, (i - 1) % self.n_enc);
                let mut p = self.tr_wave(ep, &self.rots[jp]);
                p.truncate(self.n_tr);
                p
            };
            let pl = played_moments(&self.girf, &prev, &this, self.t_rf, &times);
            let mut out = Vec::with_capacity(self.n_adc * SAMPLE_FLOATS);
            for (m, &t) in pl.iter().zip(&times) {
                let mut r = sample_record(&m.m0, &m.m1, &m.m2, &m.p, self.b0, t - self.t_rf);
                // the GIRF's B0 terms: a spatially constant phase since the excitation (slot 15, added by the kernel)
                r[15] = m.phi_b0 as f32;
                out.extend_from_slice(&r);
            }
            out
        });
        per.concat()
    }
    fn describe(&self) -> String {
        format!(
            "4D-flow yarn TR ({}): {} encodes, TR {:.3} ms, TE {:.3} ms, T_adc {:.2} ms, dwell {:.1} µs, {} rotations, GIRF {}",
            self.meta["run"].as_str().unwrap_or("?"),
            self.n_enc,
            self.tr() * 1e3,
            self.te() * 1e3,
            self.i_adc as f64 * self.dt * 1e3,
            self.dwell * 1e6,
            self.rots.len(),
            self.girf_name
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hasty_seq::multiecho::EchoSpec;
    use hasty_seq::system::System;

    fn flow_tr() -> hasty_seq::tr::Tr {
        use hasty_seq::encode::{DualLayout, Scheme};
        use hasty_seq::tr::{EncodeShape, Spoil, Tr, TrSpec};
        use hasty_seq::traj::{Spoke, TopOpts, Yarn};
        let sys = System::prisma_xr();
        let y = Yarn { kmax: 150.0, fov: 0.3, n_design: 800.0, edge_density: 1.0, kg0: 0.0, wobble: 0.0, wobble_freq: 2.0 };
        let sp = Spoke::design(&y.curve_return(30_000, 0.0), &System { smax: sys.smax * 0.45, ..sys.clone() }, &TopOpts { v_end: Some(0.0), ..TopOpts::default() });
        let spoil = [0.0, 0.0, 2.0 / (hasty_seq::GAMMA_BAR * 3e-3)];
        let spec = TrSpec { sys, enc_slew: 0.3, tail_slew: 0.5, rf_dur: 100e-6, scheme: Scheme::Coprime { p: 2, layout: DualLayout::Seven }, venc: 1.0, balanced: true, spoil: Spoil::FlowComp(spoil), encode_shape: EncodeShape::TwoLobe, tail_shape: EncodeShape::TwoLobe };
        Tr::build(&spec, sp, &rotations(RotSet::SuperFibonacci, 16))
    }

    #[test]
    fn flow_schedule_moments_and_played_chain() {
        let tr = flow_tr();
        let ecg = Ecg::new(2.0, 1.0, 0.0, 1);
        let fs = FlowSchedule::from_tr(&tr, 0.5, 4e-6, 2.89, ecg.clone(), 20, Girf::ideal(), "ideal");
        assert_eq!(fs.n_readouts % fs.n_enc, 0);
        let gam = 2.6752218744e8f64;
        let p = fs.sample_params(0..2 * fs.n_enc);
        let base_k = Grad { dt: fs.dt, t0: 0.0, g: fs.base.clone() }.moments_at(0.0, &fs.adc_times().iter().map(|t| t - fs.i_ro as f64 * fs.dt).collect::<Vec<_>>());
        for i in 0..2 * fs.n_enc {
            let (j, e) = (i / fs.n_enc, i % fs.n_enc);
            let r = &p[i * fs.n_adc * SAMPLE_FLOATS..(i + 1) * fs.n_adc * SAMPLE_FLOATS];
            // k = 0 at the echo, τ = TE there, and the encoding M1 of the design
            assert!(r[0].abs() + r[1].abs() + r[2].abs() < 1e-3, "readout {i}: k at the echo {:?}", &r[..3]);
            assert!((r[13] as f64 - fs.te()).abs() < 1e-9);
            for a in 0..3 {
                let want = gam * tr.encode.m1[e][a];
                assert!((r[3 + a] as f64 - want).abs() < 1e-4 * want.abs().max(1.0), "readout {i} axis {a}: γM1 {} vs {want}", r[3 + a]);
            }
            // the readout is R_j times the base spoke
            for n in (0..fs.n_adc).step_by(97) {
                let kb = mat_vec(&fs.rots[j], &base_k[n].0).map(|v| gam * v);
                for a in 0..3 {
                    assert!((r[n * SAMPLE_FLOATS + a] as f64 - kb[a]).abs() < 1e-3 * (1.0 + kb[a].abs()), "readout {i} sample {n}");
                }
            }
        }
        // through a GIRF: small k errors, no B0 phase without B0 terms; the B0-eddy chain adds a phase
        let vars = Girf::variants();
        let get = |n: &str| vars.iter().find(|v| v.0 == n).unwrap().1.clone();
        let fd = fs.with_girf(get("delays"), "delays");
        let q = fd.sample_params(fs.n_enc..fs.n_enc + 1);
        let dk = (0..fs.n_adc).map(|n| (0..3).map(|a| ((q[n * SAMPLE_FLOATS + a] - p[(fs.n_enc * fs.n_adc + n) * SAMPLE_FLOATS + a]) as f64 / (2.0 * std::f64::consts::PI)).powi(2)).sum::<f64>().sqrt()).fold(0.0, f64::max);
        // a delay d shifts k by ≈ γ̄ |G| d (3.1 µs on the slowest axis)
        let gmax = fs.base.iter().map(|g| hasty_seq::norm(g)).fold(0.0, f64::max);
        let bound = 1.2 * hasty_seq::GAMMA_BAR * gmax * 3.1e-6;
        assert!(dk > 0.1 * bound && dk < bound, "delay chain k error {dk} cycles/m (bound {bound})");
        assert!(q.chunks(SAMPLE_FLOATS).all(|r| r[15] == 0.0));
        let fb = fs.with_girf(get("b0eddy"), "b0eddy");
        let qb = fb.sample_params(fs.n_enc + 1..fs.n_enc + 2);
        assert!(qb.chunks(SAMPLE_FLOATS).any(|r| r[15].abs() > 1e-3), "B0-eddy phase missing");
    }

    #[test]
    fn echo_schedule_k_returns_to_zero_at_the_crossings() {
        let sys = System::prisma_xr();
        let mut spec = EchoSpec::new(3e-3, 8);
        spec.te_shifts = vec![0.0, 0.4e-3];
        let tr = EchoTrain::design(&spec, &sys);
        let sch = EchoSchedule::new(tr.clone(), 0.2, 1e-6, 2.89, Ecg::new(1.0, 1.0, 0.0, 1), 20);
        for i in 0..2 {
            let p = sch.sample_params(i..i + 1);
            let times = sch.adc_times(i);
            let te = tr.te(sch.shift(i));
            for (e, &t_e) in te.iter().enumerate() {
                let n = ((t_e - times[0]) / sch.dwell).round() as usize;
                if n >= sch.n_adc {
                    continue;
                }
                let k = &p[n * SAMPLE_FLOATS..n * SAMPLE_FLOATS + 3];
                let kn = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt() / (2.0 * std::f32::consts::PI);
                assert!(kn < 0.05, "TR {i} echo {e}: |k| {kn} cycles/m");
                assert!((p[n * SAMPLE_FLOATS + 13] as f64 - t_e).abs() < 1e-9);
            }
            // turn-around of lobe 0 at kmax
            let n = ((te[0] + 0.5 * tr.t_lobe - times[0]) / sch.dwell).round() as usize;
            let k = &p[n * SAMPLE_FLOATS..n * SAMPLE_FLOATS + 3];
            let kn = (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]).sqrt() as f64 / (2.0 * std::f64::consts::PI);
            assert!((kn - tr.kmax).abs() < 1e-3 * tr.kmax, "{kn} vs {}", tr.kmax);
        }
    }
}
