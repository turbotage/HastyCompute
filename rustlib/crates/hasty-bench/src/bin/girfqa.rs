//! GIRF QA measurement sequence for GE / pge2 (research/girf_qa_sequence.md): writes one Pulseq file per protocol
//! part, the frame table, the test inputs, the gradient-power timeline (thermal covariate), the v2-validation layout,
//! PNS checks and a protocol summary. CPU, seconds to a minute.
//!
//!     girfqa --out data/hasty/seq/girf_qa/protocol [--plan full|core] [--flow data/hasty/seq/flow_thorax_v2]
//!            [--vrot data/hasty/seq/girf_qa/inputs] [--targets res.json] [--only p2] [--no-seq] [--changeover 20]
//!
//! Parts (pass 1): c_cal (calibration prescan), s_cold (survey, self + B0 terms, 200 ms windows), x_3d (phase-encoded
//! thin slices: cross terms, B0 and 2nd / 3rd-order spatial terms), l_lin (amplitude linearity), h1_flow (heating
//! with the v2 flow TR train), s_warm (survey at the flow-scan load, heating between sweeps), v_v2 (direct
//! measurement of the played v2 readouts with 10 previous TRs), h2_hot (1.5× load), s_hot, d_cool (cool-down
//! tracking). Pass 2 (`--targets`, from the pass-1 estimate): h3_flow + r_target (narrow chirps and tone bursts at the
//! detected resonances, at the flow-scan load).
//! For every part also `<part>_check.seq` (one instance of every segment, rotations baked in) with its raster waveform
//! and ADC times as .npy, for the pypulseq round trip (research/experiments/girf_qa/pulseq_roundtrip.py).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use hasty_phantom::volume::write_npy;
use hasty_recon::data::read_npy;
use hasty_seq::girfqa::*;
use hasty_seq::pns::{pns, pns_periodic};
use hasty_seq::pnsge::{pns_ge, pns_ge_periodic};
use hasty_seq::pulseq::{block_pulse, trap_area, AdcEv, Block, GradEv, Label, Sequence};
use hasty_seq::traj::{acquisition_order, rotations, RotSet};
use hasty_seq::wave::Grad;
use hasty_seq::{Mat3, Vec3, GAMMA};
use serde_json::json;

const NCOL: usize = 26;

fn arg(args: &[String], k: &str) -> Option<String> {
    args.iter().position(|a| a == k).and_then(|i| args.get(i + 1).cloned())
}

fn wr_f64(path: &Path, shape: &[usize], v: &[f64]) {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_npy(path, "<f8", shape, false, &b).unwrap();
}

/// per-axis ∫G² dt of a block (physical; rotation applied to the corner samples)
fn block_energy(b: &Block, dt: f64) -> Vec3 {
    let pl = |g: &[f64]| -> f64 { g.windows(2).map(|w| dt / 3.0 * (w[0] * w[0] + w[0] * w[1] + w[1] * w[1])).sum() };
    match b.rotation {
        None => [0, 1, 2].map(|a| match &b.g[a] {
            None => 0.0,
            Some(GradEv::Trap { amp, rise, flat, fall, .. }) => amp * amp * (flat + (rise + fall) / 3.0),
            Some(GradEv::Pl { g, .. }) | Some(GradEv::Arb { g, .. }) => pl(g),
        }),
        Some(q) => {
            let m = hasty_seq::quat_to_mat(q);
            let n = b.g.iter().flatten().map(|g| if let GradEv::Pl { g, .. } = g { g.len() } else { 0 }).max().unwrap_or(0);
            let at = |a: usize, i: usize| -> f64 { if let Some(GradEv::Pl { g, .. }) = &b.g[a] { g.get(i).copied().unwrap_or(0.0) } else { 0.0 } };
            [0, 1, 2].map(|o| {
                let s: Vec<f64> = (0..n).map(|i| (0..3).map(|l| m[o][l] * at(l, i)).sum()).collect();
                pl(&s)
            })
        }
    }
}

struct Part {
    name: String,
    purpose: String,
    coil: String,
    seq: Sequence,
    frames: Vec<[f64; NCOL]>,
    /// absolute start of the part (s)
    t0: f64,
    /// as-played clock (s from the part start) incl. segment gaps
    clock: f64,
    n_adc_events: usize,
    n_adc_samples: usize,
    /// (time, per-axis energy) per block on the as-played clock
    energy: Vec<(f64, Vec3)>,
    first_of_trid: BTreeMap<i64, (usize, usize)>,
    /// one instance of every segment type (rotations baked in) for the pypulseq round trip
    check: Sequence,
    capture: bool,
}

impl Part {
    fn new(name: &str, purpose: &str, coil: &str, cfg: &QaConfig) -> Self {
        let mut check = Sequence::new(cfg.sys.clone());
        check.expand_rotations = true;
        Self { name: name.into(), purpose: purpose.into(), coil: coil.into(), seq: Sequence::new(cfg.sys.clone()), frames: vec![], t0: 0.0, clock: 0.0, n_adc_events: 0, n_adc_samples: 0, energy: vec![], first_of_trid: BTreeMap::new(), check, capture: false }
    }
    /// append blocks; returns the as-played start time of the first block (after any segment gap)
    fn push(&mut self, blocks: Vec<Block>, gap: f64) -> f64 {
        let mut t_first = None;
        for b in blocks {
            if let Some(l) = b.labels.iter().find(|l| l.name == "TRID") {
                if self.seq.n_blocks() > 0 {
                    self.clock += gap;
                }
                let bi = self.seq.n_blocks();
                let new = !self.first_of_trid.contains_key(&l.value);
                if new {
                    self.first_of_trid.insert(l.value, (bi, bi));
                }
                self.capture = new && self.check.n_blocks() < 3000;
            }
            if self.capture {
                let empty = b.rf.is_none() && b.adc.is_none() && b.g.iter().all(|g| g.is_none());
                if empty && b.duration.unwrap_or(0.0) > 2e-3 {
                    self.check.add(Block { duration: Some(1e-3), labels: b.labels.clone(), ..Default::default() });
                } else {
                    self.check.add(b.clone());
                }
            }
            if t_first.is_none() {
                t_first = Some(self.clock);
            }
            if let Some(a) = &b.adc {
                self.n_adc_events += 1;
                self.n_adc_samples += a.num;
            }
            let e = block_energy(&b, self.seq.sys.grad_raster);
            let d = self.seq.add(b);
            self.energy.push((self.clock + 0.5 * d, e));
            self.clock += d;
        }
        t_first.unwrap_or(self.clock)
    }
    fn duration(&self) -> f64 {
        self.clock
    }
}

#[allow(clippy::too_many_arguments)]
fn rec(part: usize, idx: usize, f: &FrameSpec, t_start: f64, tm: &FrameTiming, adc_ev0: usize, t_hist0: f64, n_hist: usize, dwell: f64) -> [f64; NCOL] {
    let pe = f.pe.unwrap_or([f64::NAN, f64::NAN]);
    let n: usize = tm.nadc.iter().sum();
    [
        part as f64, idx as f64, f.kind as f64, f.axis as f64, f.slice as f64, f.pos, f.eta, f.sigma, f.level, f.input as f64, pe[0], pe[1], f.avg as f64,
        t_start, t_start + tm.t_exc, tm.t_in0, tm.adc0.first().copied().unwrap_or(f64::NAN), n as f64, f.readout as f64, f.thk, adc_ev0 as f64, tm.nadc.len() as f64, f.t_adc, t_hist0, n_hist as f64, dwell,
    ]
}

struct Ctx {
    cfg: QaConfig,
    inputs: Vec<Input>,
    ftr: FlowTr,
    rots: Vec<Mat3>,
    rot_next: usize,
    heater: Vec<Vec3>,
    p_flow: f64,
    p_heater: f64,
    next_trid: i64,
    trids: BTreeMap<String, i64>,
}

impl Ctx {
    fn trid(&mut self, key: &str) -> i64 {
        if let Some(&t) = self.trids.get(key) {
            return t;
        }
        let t = self.next_trid;
        self.next_trid += 1;
        self.trids.insert(key.into(), t);
        t
    }
    /// heating for `dur` seconds at load `load` × P_flow(GE): flow TRs, plus heater TRs when load > 1
    fn heat(&mut self, part: &mut Part, dur: f64, load: f64) {
        let gap = self.cfg.seg_gap;
        let t_end = part.clock + dur;
        let trid0 = self.trid("flowtr");
        self.trids.insert("flowtr_last".into(), trid0);
        // reserve TRIDs trid0 .. trid0 + 14 for the 15 flow segments
        self.next_trid = self.next_trid.max(trid0 + 15);
        let th = self.trid("heater");
        let t_flow = (self.ftr.enc[0].len() + self.ftr.readout.len() + self.ftr.tails[0].len() - 3) as f64 * self.ftr.dt + 3.0 * gap;
        let t_h = (self.heater.len() - 1) as f64 * self.cfg.sys.grad_raster + gap;
        // fraction of heater TRs (in time) for the target mean power
        let fh = if load > 1.0 { ((load - 1.0) * self.p_flow / (self.p_heater - self.p_flow)).clamp(0.0, 1.0) } else { 0.0 };
        let mut acc = 0.0;
        let mut i = 0usize;
        while part.clock < t_end {
            acc += fh * t_flow / t_h;
            if acc >= 1.0 {
                acc -= 1.0;
                let mut b = Block { duration: Some((self.heater.len() - 1) as f64 * self.cfg.sys.grad_raster), labels: vec![Label::set("TRID", th)], ..Default::default() };
                for a in 0..3 {
                    b.g[a] = Some(GradEv::Pl { g: self.heater.iter().map(|v| v[a]).collect(), delay: 0.0 });
                }
                part.push(vec![b], gap);
            }
            let e = i % self.ftr.enc.len();
            let r = self.rots[self.rot_next % self.rots.len()];
            if e == self.ftr.enc.len() - 1 {
                self.rot_next += 1;
            }
            part.push(self.ftr.tr_blocks(e, &r, 1.0, trid0), gap);
            i += 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn survey(ctx: &mut Ctx, part: &mut Part, pidx: usize, kind: u8, inputs: &[(usize, f64)], units: &[(usize, usize)], groups: usize, t_adc: f64, pe: Option<[f64; 2]>, heat: Option<(f64, f64)>, avg0: usize, key: &str) {
    let cfg = ctx.cfg.clone();
    for g in 0..groups {
        let eta = if (g + avg0 / 2) % 2 == 0 { 1.0 } else { -1.0 };
        for &(iu, level) in inputs {
            let trids: Vec<i64> = (0..3).map(|a| ctx.trid(&format!("{key}_{}_{a}_{}", ctx.inputs[iu].name.split('@').next().unwrap(), if pe.is_some() { "pe" } else { "" }))).collect();
            let order = sweep_order(units.len());
            for (m, chunk) in order.chunks(units.len()).enumerate() {
                for &(u, sigma) in chunk {
                    let (a, s) = units[u];
                    let f = FrameSpec { kind, trid: trids[a], axis: a, slice: s, pos: cfg.slices[s], eta, sigma, level, input: iu, pe, thk: cfg.thk, t_adc, avg: avg0 + 2 * g + m / 2, readout: -1 };
                    let (bl, tm) = frame(&cfg, &f, &ctx.inputs[iu].g);
                    let ev0 = part.n_adc_events;
                    let t = part.push(bl, cfg.seg_gap);
                    let idx = part.frames.len();
                    part.frames.push(rec(pidx, idx, &f, part.t0 + t, &tm, ev0, f64::NAN, 0, cfg.dwell));
                }
                if let Some((dur_per_sweep, load)) = heat {
                    ctx.heat(part, dur_per_sweep, load);
                }
            }
        }
    }
}

fn pe_survey(ctx: &mut Ctx, part: &mut Part, pidx: usize, inputs: &[usize], groups: usize, t_adc: f64) {
    let units = slice_units(&ctx.cfg, &[0, 1, 2, 3]);
    for (q, k) in pe_steps(&ctx.cfg).iter().enumerate() {
        let ins: Vec<(usize, f64)> = inputs.iter().map(|&i| (i, 1.0)).collect();
        let n0 = part.frames.len();
        survey(ctx, part, pidx, 2, &ins, &units, groups, t_adc, Some(*k), None, 0, "x");
        for r in &mut part.frames[n0..] {
            r[12] += 100.0 * q as f64; // avg column carries the PE index × 100 for X frames
        }
    }
}

/// calibration prescan (projections, 3D dual-echo B0, receiver delay per dwell, T1 / T2, slice positions, thin-slice
/// reference images, noise)
fn calibration(ctx: &mut Ctx, part: &mut Part, pidx: usize) {
    let cfg = ctx.cfg.clone();
    let s = cfg.sys.clone();
    let gap = cfg.seg_gap;
    let r = s.grad_raster;
    let ceil = |t: f64| (t / r - 1e-9).ceil() * r;
    let mut idx = 0usize;
    let mut push_frame = |part: &mut Part, kind: u8, axis: usize, slice: usize, sigma: f64, level: f64, bl: Vec<Block>, t_exc: f64, adc0: f64, n: usize| {
        let ev0 = part.n_adc_events;
        let t = part.push(bl, gap);
        let mut row = [f64::NAN; NCOL];
        row[0] = pidx as f64;
        row[1] = idx as f64;
        row[2] = kind as f64;
        row[3] = axis as f64;
        row[4] = slice as f64;
        row[7] = sigma;
        row[8] = level;
        row[13] = part.t0 + t;
        row[14] = part.t0 + t + t_exc;
        row[16] = adc0;
        row[17] = n as f64;
        row[20] = ev0 as f64;
        part.frames.push(row);
        idx += 1;
    };
    // noise (no RF), 16 × 4096 samples
    let tn = ctx.trid("c_noise");
    for _ in 0..16 {
        let b = Block { adc: Some(AdcEv { num: 4096, dwell: cfg.dwell, delay: s.adc_dead, freq: 0.0, phase: 0.0 }), labels: vec![Label::set("TRID", tn)], ..Default::default() };
        let d = Block::delay(0.05);
        push_frame(part, 18, 0, 0, 0.0, 0.0, vec![b, d], 0.0, s.adc_dead + 0.5 * cfg.dwell, 4096);
    }
    // 1. projections ±x ±y ±z: hard 20°, readout FOV 0.4 m, 256 samples, TR 100 ms, 8 averages
    let n_ro = 256usize;
    let fov = 0.4;
    let g_ro = 1.0 / (fov * s.gamma * cfg.dwell);
    for avg in 0..8 {
        for a in 0..3 {
            for &sg in &[1.0, -1.0] {
                let tr = ctx.trid(&format!("c_proj_{a}"));
                let rf = block_pulse(&s, 20f64.to_radians(), 200e-6, s.rf_dead, 'e');
                let b1 = Block { rf: Some(rf), labels: vec![Label::set("TRID", tr)], ..Default::default() };
                let rise = ceil(g_ro / (0.2 * s.smax));
                let flat = ceil(n_ro as f64 * cfg.dwell + 2.0 * s.adc_dead);
                let pre = trap_area(&s, g_ro * (0.5 * flat + 0.5 * rise), 30e-3, 0.2 * s.smax, 0.0).scaled(-sg);
                let mut b2 = Block::default();
                b2.g[a] = Some(pre);
                let mut b3 = Block { adc: Some(AdcEv { num: n_ro, dwell: cfg.dwell, delay: ((rise + s.adc_dead) * 1e6).ceil() * 1e-6, freq: 0.0, phase: 0.0 }), ..Default::default() };
                b3.g[a] = Some(GradEv::Trap { amp: sg * g_ro, rise, flat, fall: rise, delay: 0.0 });
                let d = Block::delay(0.08);
                push_frame(part, 11, a, avg, sg, 1.0, vec![b1, b2, b3, d], s.rf_dead + 100e-6, f64::NAN, n_ro);
            }
        }
    }
    // 2. 3D dual-echo GRE for B0 (5 mm, FOV 240 mm, 48 × 48 phase encodes, TE ≈ 3 / 8 ms, RF spoiled)
    let n3 = 48usize;
    let fov3 = 0.24;
    let g3 = 1.0 / (fov3 * s.gamma * 8e-6);
    let tb = ctx.trid("c_b0map");
    let dk = 1.0 / (fov3 * s.gamma); // T·s/m per PE step
    for iy in 0..n3 {
        for iz in 0..n3 {
            let ky = (iy as f64 - n3 as f64 / 2.0) * dk;
            let kz = (iz as f64 - n3 as f64 / 2.0) * dk;
            let k = iy * n3 + iz;
            let mut rf = block_pulse(&s, 5f64.to_radians(), 100e-6, s.rf_dead, 'e');
            rf.phase = (117f64.to_radians() * (k * (k + 1) / 2) as f64) % (2.0 * std::f64::consts::PI);
            let b1 = Block { rf: Some(rf), labels: vec![Label::set("TRID", tb)], ..Default::default() };
            let pmax = trap_area(&s, n3 as f64 / 2.0 * dk, 20e-3, 0.2 * s.smax, 0.0);
            let rise = ceil(g3 / (0.2 * s.smax));
            let flat = ceil(n3 as f64 * 8e-6 + 2.0 * s.adc_dead);
            let mut b2 = Block::default();
            b2.g[0] = Some(trap_area(&s, -g3 * (0.5 * flat + 0.5 * rise), 30e-3, 0.2 * s.smax, 0.0));
            b2.g[1] = Some(pmax.scaled((ky / (n3 as f64 / 2.0 * dk)).max(-1.0).min(1.0) + if ky == 0.0 { 1e-6 } else { 0.0 }));
            b2.g[2] = Some(pmax.scaled((kz / (n3 as f64 / 2.0 * dk)).max(-1.0).min(1.0) + if kz == 0.0 { 1e-6 } else { 0.0 }));
            let mk_ro = || {
                let mut b = Block { adc: Some(AdcEv { num: n3, dwell: 8e-6, delay: ((rise + s.adc_dead) * 1e6).ceil() * 1e-6, freq: 0.0, phase: rf_phase(k) }), ..Default::default() };
                b.g[0] = Some(GradEv::Trap { amp: g3, rise, flat, fall: rise, delay: 0.0 });
                b
            };
            let mut fly = Block::default();
            fly.g[0] = Some(trap_area(&s, -g3 * (flat + rise), 30e-3, 0.2 * s.smax, 0.0));
            let gap4 = Block::delay(4e-3);
            let mut sp = Block::default();
            sp.g[0] = Some(trap_area(&s, 4.0 / (s.gamma * 5e-3), 30e-3, 0.2 * s.smax, 0.0));
            for a in 1..3 {
                let p = pmax.scaled(-((if a == 1 { ky } else { kz }) / (n3 as f64 / 2.0 * dk)) + 1e-6);
                sp.g[a] = Some(p);
            }
            push_frame(part, 12, iy, iz, 0.0, 0.0, vec![b1, b2, mk_ro(), fly, gap4, mk_ro(), sp], s.rf_dead + 50e-6, f64::NAN, 2 * n3);
        }
    }
    // 3. receiver delay per dwell: hard 30°, ADC 20 ms at dwell d with frequency offset ±20 kHz, 4 averages, TR 0.5 s
    for avg in 0..4 {
        for &dw in &[2e-6f64, 4e-6, 8e-6, 16e-6] {
            for &df in &[20e3f64, -20e3] {
                let t = ctx.trid("c_rxdelay");
                let n = (20e-3 / dw).round() as usize;
                let rf = block_pulse(&s, 30f64.to_radians(), 100e-6, s.rf_dead, 'e');
                let b1 = Block { rf: Some(rf), labels: vec![Label::set("TRID", t)], ..Default::default() };
                let b2 = Block { adc: Some(AdcEv { num: n, dwell: dw, delay: s.adc_dead, freq: df, phase: 0.0 }), ..Default::default() };
                let d = Block::delay(0.45);
                let b1d = s.rf_dead + 100e-6 + s.rf_ringdown;
                let b1d = (b1d / s.block_raster).ceil() * s.block_raster;
                push_frame(part, 13, (dw * 1e6).round() as usize, avg, df.signum(), df.abs(), vec![b1, b2, d], s.rf_dead + 50e-6, b1d + s.adc_dead + 0.5 * dw - (s.rf_dead + 50e-6), n);
            }
        }
    }
    // 4. T1: inversion recovery, hard 180° 2 ms, TI list, 90° FID 20 ms, 3 s recovery
    for &ti in &[0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0] {
        let t = ctx.trid("c_t1");
        let inv = block_pulse(&s, std::f64::consts::PI, 2e-3, s.rf_dead, 'i');
        let b0 = Block { rf: Some(inv), labels: vec![Label::set("TRID", t)], ..Default::default() };
        let mut sp = Block::default();
        for a in 0..3 {
            sp.g[a] = Some(trap_area(&s, 4.0 / (s.gamma * 5e-3), 30e-3, 0.2 * s.smax, 0.0));
        }
        let w = Block::delay(ceil(ti - 2.2e-3 - 1.5e-3).max(r));
        let rf = block_pulse(&s, 0.5 * std::f64::consts::PI, 200e-6, s.rf_dead, 'e');
        let b1 = Block { rf: Some(rf), ..Default::default() };
        let b2 = Block { adc: Some(AdcEv { num: 5000, dwell: 4e-6, delay: s.adc_dead, freq: 0.0, phase: 0.0 }), ..Default::default() };
        let d = Block::delay(3.0);
        push_frame(part, 14, 0, 0, 0.0, ti, vec![b0, sp, w, b1, b2, d], f64::NAN, f64::NAN, 5000);
    }
    // 5. T2: CPMG 90° + 32 × 180° (ESP 10 ms), 1 ms ADC on every echo, 4 averages, 4 s recovery
    for _ in 0..4 {
        let t = ctx.trid("c_t2");
        let ex = block_pulse(&s, 0.5 * std::f64::consts::PI, 200e-6, s.rf_dead, 'e');
        let mut bl = vec![Block { rf: Some(ex), duration: Some(ceil(s.rf_dead + 200e-6 + 5e-3 - 100e-6 - 300e-6)), labels: vec![Label::set("TRID", t)], ..Default::default() }];
        for _ in 0..32 {
            let mut rf = block_pulse(&s, std::f64::consts::PI, 400e-6, s.rf_dead, 'r');
            rf.phase = 0.5 * std::f64::consts::PI;
            bl.push(Block { rf: Some(rf), duration: Some(ceil(s.rf_dead + 400e-6 + s.rf_ringdown + 1e-3)), ..Default::default() });
            let d180 = ceil(s.rf_dead + 400e-6 + s.rf_ringdown + 1e-3);
            let adc_delay = ((s.rf_dead + 200e-6 + 5e-3 - d180 - 0.5e-3) * 1e6).round() * 1e-6;
            bl.push(Block { adc: Some(AdcEv { num: 250, dwell: 4e-6, delay: adc_delay, freq: 0.0, phase: 0.0 }), duration: Some(ceil(10e-3 - d180)), ..Default::default() });
        }
        bl.push(Block::delay(4.0));
        push_frame(part, 15, 0, 0, 0.0, 0.0, bl, f64::NAN, f64::NAN, 32 * 250);
    }
    // 6. slice positions: each thin slice, readout along its normal (FOV 0.2 m, 1024 samples), both readout and
    //    slice-select polarities, 2 averages, slices interleaved
    let units = slice_units(&cfg, &[0, 1, 2, 3]);
    let g_sp = 1.0 / (0.2 * s.gamma * cfg.dwell);
    for _avg in 0..2 {
        for &eta in &[1.0, -1.0] {
            for &sg in &[1.0, -1.0] {
                for &(a, sl) in &units {
                    let t = ctx.trid(&format!("c_slicepos_{a}"));
                    let (mut bl, t_exc, _) = excitation_blocks(&cfg, a, cfg.slices[sl], eta, cfg.thk, None, t);
                    let rise = ceil(g_sp / (0.2 * s.smax));
                    let flat = ceil(1024.0 * cfg.dwell + 2.0 * s.adc_dead);
                    let mut b2 = Block::default();
                    b2.g[a] = Some(trap_area(&s, g_sp * (0.5 * flat + 0.5 * rise), 30e-3, 0.2 * s.smax, 0.0).scaled(-sg));
                    let mut b3 = Block { adc: Some(AdcEv { num: 1024, dwell: cfg.dwell, delay: ((rise + s.adc_dead) * 1e6).ceil() * 1e-6, freq: 0.0, phase: 0.0 }), ..Default::default() };
                    b3.g[a] = Some(GradEv::Trap { amp: sg * g_sp, rise, flat, fall: rise, delay: 0.0 });
                    let mut sp = Block::default();
                    for q in 0..3 {
                        sp.g[q] = Some(trap_area(&s, 4.0 / (s.gamma * cfg.thk), 30e-3, 0.2 * s.smax, 0.0));
                    }
                    let d = Block::delay(0.06);
                    bl.extend([b2, b3, sp, d]);
                    push_frame(part, 16, a, sl, sg, eta, bl, t_exc, f64::NAN, 1024);
                }
            }
        }
    }
    // 7. thin-slice reference images (ρ, receive weighting, in-plane B0) for the phase-encoded stage: 2D dual-echo
    //    GRE of each slice, 64 × 64 over 256 mm, readout on the first in-plane axis, PE on the second, 30°
    let fov2 = 0.256;
    let n2 = 64usize;
    let g2 = 1.0 / (fov2 * s.gamma * 8e-6);
    let dk2 = 1.0 / (fov2 * s.gamma);
    for ip in 0..n2 {
        for &(a, sl) in &units {
            let others: Vec<usize> = (0..3).filter(|&b| b != a).collect();
            let t = ctx.trid(&format!("c_sliceimg_{a}"));
            let mut c2 = cfg.clone();
            c2.flip = 30f64.to_radians();
            let (mut bl, t_exc, _) = excitation_blocks(&c2, a, cfg.slices[sl], 1.0, cfg.thk, None, t);
            let pmax = trap_area(&s, n2 as f64 / 2.0 * dk2, 20e-3, 0.2 * s.smax, 0.0);
            let kp = (ip as f64 - n2 as f64 / 2.0) / (n2 as f64 / 2.0);
            let rise = ceil(g2 / (0.2 * s.smax));
            let flat = ceil(n2 as f64 * 8e-6 + 2.0 * s.adc_dead);
            let mut b2 = Block::default();
            b2.g[others[0]] = Some(trap_area(&s, -g2 * (0.5 * flat + 0.5 * rise), 30e-3, 0.2 * s.smax, 0.0));
            b2.g[others[1]] = Some(pmax.scaled(if kp == 0.0 { 1e-6 } else { kp }));
            let mk = || {
                let mut b = Block { adc: Some(AdcEv { num: n2, dwell: 8e-6, delay: ((rise + s.adc_dead) * 1e6).ceil() * 1e-6, freq: 0.0, phase: 0.0 }), ..Default::default() };
                b.g[others[0]] = Some(GradEv::Trap { amp: g2, rise, flat, fall: rise, delay: 0.0 });
                b
            };
            let mut fly = Block::default();
            fly.g[others[0]] = Some(trap_area(&s, -g2 * (flat + rise), 30e-3, 0.2 * s.smax, 0.0));
            let gap5 = Block::delay(5e-3);
            let mut sp = Block::default();
            for q in 0..3 {
                sp.g[q] = Some(trap_area(&s, 4.0 / (s.gamma * cfg.thk), 30e-3, 0.2 * s.smax, 0.0));
            }
            bl.extend([b2, mk(), fly, gap5, mk(), sp, Block::delay(10e-3)]);
            push_frame(part, 17, a, sl, 0.0, ip as f64, bl, t_exc, f64::NAN, 2 * n2);
        }
    }
}

fn rf_phase(k: usize) -> f64 {
    (117f64.to_radians() * (k * (k + 1) / 2) as f64) % (2.0 * std::f64::consts::PI)
}

/// v2 validation: for every listed readout, axis and slice, 10 previous TRs (as in the scan) then the thin slice and
/// the measured TR's axis-a components
fn v_block(ctx: &mut Ctx, part: &mut Part, pidx: usize, vr: &[Mat3], venc: &[usize], nhist: usize, groups: usize, vin: &mut Vec<Vec<Vec<f64>>>) {
    let cfg = ctx.cfg.clone();
    let nr = venc.len() / (nhist + 1);
    let trid0 = ctx.trid("flowtr");
    ctx.next_trid = ctx.next_trid.max(trid0 + 15);
    let tv: Vec<i64> = (0..3).map(|a| ctx.trid(&format!("v_{a}"))).collect();
    // units (readout, axis, slice), interleaved: axis fastest, then slice, then readout
    let mut units = vec![];
    for r in 0..nr {
        for s in 0..cfg.slices.len() {
            for a in 0..3 {
                units.push((r, a, s));
            }
        }
    }
    vin.clear();
    for r in 0..nr {
        let mut per_a = vec![];
        for a in 0..3 {
            let e = venc[r * (nhist + 1) + nhist];
            let f = FrameSpec { kind: 4, trid: tv[a], axis: a, slice: 0, pos: 0.0, eta: 1.0, sigma: 1.0, level: 1.0, input: 0, pe: None, thk: cfg.thk_v, t_adc: 0.0, avg: 0, readout: r as i64 };
            let (_, _, gin) = v_frame(&cfg, &f, &ctx.ftr, e, &vr[r * (nhist + 1) + nhist]);
            per_a.push(gin);
        }
        vin.push(per_a);
    }
    for g in 0..groups {
        let eta = if g % 2 == 0 { 1.0 } else { -1.0 };
        for (m, chunk) in sweep_order(units.len()).chunks(units.len()).enumerate() {
            for &(u, sigma) in chunk {
                let (r, a, s) = units[u];
                // history
                let t_h0 = part.clock;
                for j in 0..nhist {
                    let k = r * (nhist + 1) + j;
                    part.push(ctx.ftr.tr_blocks(venc[k], &vr[k], sigma, trid0), cfg.seg_gap);
                }
                let t_h0 = t_h0 + cfg.seg_gap;
                let e = venc[r * (nhist + 1) + nhist];
                let f = FrameSpec { kind: 4, trid: tv[a], axis: a, slice: s, pos: cfg.slices[s], eta, sigma, level: 1.0, input: 0, pe: None, thk: cfg.thk_v, t_adc: 0.0, avg: 2 * g + m / 2, readout: r as i64 };
                let (bl, tm, _) = v_frame(&cfg, &f, &ctx.ftr, e, &vr[r * (nhist + 1) + nhist]);
                let ev0 = part.n_adc_events;
                let t = part.push(bl, cfg.seg_gap);
                let idx = part.frames.len();
                part.frames.push(rec(pidx, idx, &f, part.t0 + t, &tm, ev0, part.t0 + t_h0, nhist, cfg.dwell));
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let out = PathBuf::from(arg(&args, "--out").unwrap_or("data/hasty/seq/girf_qa/protocol".into()));
    let plan = arg(&args, "--plan").unwrap_or("full".into());
    let flow = PathBuf::from(arg(&args, "--flow").unwrap_or("data/hasty/seq/flow_thorax_v2".into()));
    let vrot = PathBuf::from(arg(&args, "--vrot").unwrap_or("data/hasty/seq/girf_qa/inputs".into()));
    let targets = arg(&args, "--targets");
    let only = arg(&args, "--only");
    let no_seq = args.iter().any(|a| a == "--no-seq");
    let changeover: f64 = arg(&args, "--changeover").map(|s| s.parse().unwrap()).unwrap_or(20.0);
    std::fs::create_dir_all(&out).unwrap();
    let cfg = QaConfig::default();
    let t_start = std::time::Instant::now();

    // ── flow TR on the GE raster
    let summ: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(flow.join("summary.json")).unwrap()).unwrap();
    let b = &summ["best"];
    let dt0 = summ["problem"]["dt"].as_f64().unwrap();
    let (i_ro, i_tail, i_adc, te) = (b["i_ro"].as_u64().unwrap() as usize, b["i_tail"].as_u64().unwrap() as usize, b["i_adc"].as_u64().unwrap() as usize, b["te"].as_f64().unwrap());
    let tg = read_npy(&flow.join("tr_grad.npy")).unwrap();
    let (ne, np1) = (tg.shape[0], tg.shape[1]);
    let tv = tg.f64s();
    let tr_grad: Vec<Vec<Vec3>> = (0..ne).map(|e| (0..np1).map(|i| [0, 1, 2].map(|q| tv[(e * np1 + i) * 3 + q])).collect()).collect();
    let ftr = FlowTr::from_design(&tr_grad, dt0, i_ro, i_tail, i_adc, te, cfg.sys.grad_raster);
    let gapn = (cfg.seg_gap / cfg.sys.grad_raster).round() as usize; // 117 µs → 29 samples (116 µs on the 4 µs raster)
    let n_rot = 6079;
    let rset = rotations(RotSet::SuperFibonacci, n_rot);
    let ord = acquisition_order(RotSet::SuperFibonacci, n_rot);
    let rots: Vec<Mat3> = ord.iter().map(|&i| rset[i]).collect();
    // as-played flow TR power (GE: segments + gaps), mean Σ G²
    let mut e_flow = 0.0;
    let mut t_flow = 0.0;
    for e in 0..ne {
        let w = ftr.tr_played(e, &rots[e], 1.0, gapn);
        e_flow += mean_g2(&w) * w.len() as f64;
        t_flow += w.len() as f64;
    }
    let p_flow = e_flow / t_flow;
    let heater = heater_tr(&cfg, 35e-3, 40.0, 14.1e-3);
    let p_heater = mean_g2(&heater) * heater.len() as f64 / (heater.len() as f64 + gapn as f64);
    println!("girfqa: flow TR on {:.0} µs: encode {:.3} ms, readout {:.3} ms, tail {:.3} ms (+3 × {:.0} µs gaps); P_flow {:.3e} T²/m² (G_rms {:.1} mT/m), heater {:.3e}", cfg.sys.grad_raster * 1e6, ftr.t_ro * 1e3, (ftr.readout.len() - 1) as f64 * ftr.dt * 1e3, (ftr.tails[0].len() - 1) as f64 * ftr.dt * 1e3, cfg.seg_gap * 1e6, p_flow, p_flow.sqrt() * 1e3, p_heater);

    // ── inputs
    let mut inputs: Vec<Input> = vec![];
    let mut s_frac = BTreeMap::new();
    for (i, s) in survey_inputs(&cfg, cfg.thk) {
        s_frac.insert(i.name.clone(), s);
        inputs.push(i);
    }
    let i_chirp = 0usize;
    let i_tri100 = inputs.iter().position(|i| i.name == "tri100").unwrap();
    let i_tri240 = inputs.iter().position(|i| i.name == "tri240").unwrap();
    // pass-2 targeted inputs
    let mut tgt_list = vec![];
    if let Some(tp) = &targets {
        let tj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(tp).unwrap()).unwrap();
        for t in tj["targets"].as_array().unwrap() {
            let (a, f0, z) = (t["axis"].as_u64().unwrap() as usize, t["f0"].as_f64().unwrap(), t["zeta"].as_f64().unwrap_or(0.01).clamp(0.002, 0.05));
            let (g1, s1) = pns_fit(&cfg, 0.5, |s| narrow_chirp(&cfg, f0, z, 30e-3, s, cfg.thk, 1e-3));
            let i1 = inputs.len();
            inputs.push(Input { name: format!("nchirp@{f0:.2}"), g: g1 });
            let (g2, s2) = pns_fit(&cfg, 0.5, |s| tone_burst(&cfg, f0, 20e-3, s, cfg.thk, 1e-3));
            let i2 = inputs.len();
            inputs.push(Input { name: format!("tone@{f0:.2}"), g: g2 });
            s_frac.insert(format!("nchirp@{f0:.2}"), s1);
            s_frac.insert(format!("tone@{f0:.2}"), s2);
            tgt_list.push((a, f0, z, i1, i2));
        }
    }
    let n_in_max = inputs.iter().map(|i| i.g.len()).max().unwrap();
    let mut inarr = vec![0.0; inputs.len() * n_in_max];
    for (k, i) in inputs.iter().enumerate() {
        inarr[k * n_in_max..k * n_in_max + i.g.len()].copy_from_slice(&i.g);
    }
    wr_f64(&out.join("inputs.npy"), &[inputs.len(), n_in_max], &inarr);
    // flow TR arrays (GE raster) for the simulator
    let flat3 = |w: &[Vec<Vec3>]| -> Vec<f64> { w.iter().flat_map(|v| v.iter().flat_map(|x| x.iter().copied())).collect() };
    wr_f64(&out.join("flowtr_enc.npy"), &[ne, ftr.enc[0].len(), 3], &flat3(&ftr.enc));
    wr_f64(&out.join("flowtr_ro.npy"), &[ftr.readout.len(), 3], &flat3(&[ftr.readout.clone()]));
    wr_f64(&out.join("flowtr_tail.npy"), &[ne, ftr.tails[0].len(), 3], &flat3(&ftr.tails));

    // ── v2 validation layout: the girf_sensitivity budget readouts with 10 previous TRs (exported by prep_inputs.py)
    let (vr, venc, nhist) = match (read_npy(&vrot.join("vrot_R.npy")), read_npy(&vrot.join("vrot_enc.npy"))) {
        (Ok(r), Ok(e)) => {
            let rv = r.f64s();
            let ev: Vec<usize> = if e.descr.contains("i8") { e.bytes.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as usize).collect() } else { e.f64s().iter().map(|x| *x as usize).collect() };
            let nh = r.shape[1] - 1;
            let m: Vec<Mat3> = (0..r.shape[0] * r.shape[1]).map(|k| [0, 1, 2].map(|i| [0, 1, 2].map(|j| rv[k * 9 + i * 3 + j]))).collect();
            (m, ev, nh)
        }
        _ => {
            eprintln!("girfqa: no {}/vrot_*.npy: v2 validation uses 8 readouts of the heating rotation set", vrot.display());
            let nh = 10;
            let mut m = vec![];
            let mut e = vec![];
            for r in 0..8 {
                for j in 0..=nh {
                    let i = r * 700 + j;
                    m.push(rots[(i / 7) % rots.len()]);
                    e.push(i % 7);
                }
            }
            (m, e, nh)
        }
    };
    let nvr = venc.len() / (nhist + 1);

    let mut ctx = Ctx { cfg: cfg.clone(), inputs: inputs.clone(), ftr: ftr.clone(), rots: rots.clone(), rot_next: 0, heater: heater.clone(), p_flow, p_heater, next_trid: 1, trids: BTreeMap::new() };
    let full = plan == "full";
    let units12 = slice_units(&cfg, &[0, 1, 2, 3]);
    let units_outer = slice_units(&cfg, &[0, 3]);
    let tl = cfg.t_adc_long;
    let ts = cfg.t_adc_short;
    let mut parts: Vec<Part> = vec![];
    let mut clock = 0.0;
    let mut vin = vec![];
    let add = |mut p: Part, clock: &mut f64, parts: &mut Vec<Part>| {
        p.t0 = *clock;
        *clock += p.duration() + changeover;
        parts.push(p);
    };
    let zero_ref = (i_chirp, 0.0); // level 0 → tiny amplitude (pge2 rule): the zero-input reference
    let sweep_in = |ins: &[usize]| -> Vec<(usize, f64)> { ins.iter().map(|&i| (i, 1.0)).collect() };
    let want = |n: &str| only.as_deref().map_or(true, |o| (o == "p2" && (n.starts_with("h3") || n.starts_with("r_"))) || o.split(',').any(|x| x == n));
    let p1_list: Vec<&str> = vec!["c_cal", "t_rx", "s_cold", "x_3d", "l_lin", "h1_flow", "s_warm", "v_v2", "h2_hot", "s_hot", "d_cool"];
    for name in p1_list {
        if !want(name) {
            continue;
        }
        let pidx = parts.len();
        let mut p;
        match name {
            "c_cal" => {
                p = Part::new(name, "calibration: noise, projections (centre/radius), 3D dual-echo B0 (shim check), receiver delay per dwell (±20 kHz FIDs), T1 (IR), T2 (CPMG), slice positions (readout along the slice normal, ± readout and slice polarity), thin-slice reference images (ρ, receive weighting, in-plane B0)", "array (images) + body (rx delay repeat)", &cfg);
                p.t0 = clock;
                calibration(&mut ctx, &mut p, pidx);
            }
            "t_rx" => {
                p = Part::new(name, "receiver timing per dwell: thin slices, tri60 + tri100, ADC dwell 2 / 4 / 8 / 16 µs (10 ms windows, 100 ms frames); the GIRF delay fitted per dwell gives δ_rx(d) − δ_rx(4 µs); run once with the body coil and once with the array", "body + array (run twice)", &cfg);
                p.t0 = clock;
                let cfg0 = ctx.cfg.clone();
                for &dw in &[4e-6f64, 2e-6, 8e-6, 16e-6] {
                    ctx.cfg.dwell = dw;
                    ctx.cfg.frame_pad = 0.09;
                    survey(&mut ctx, &mut p, pidx, 7, &[(1, 1.0), (i_tri100, 1.0)], &units12, 2, 10e-3, None, None, 0, &format!("rx{}", (dw * 1e6).round() as i64));
                }
                ctx.cfg = cfg0;
            }
            "s_cold" => {
                p = Part::new(name, "survey, cold: self + B0 terms, 200 ms windows (narrow resonances), chirp + 4 triangles + zero reference, ± test polarity, ± slice select", "body", &cfg);
                p.t0 = clock;
                let mut ins = sweep_in(&(0..5).collect::<Vec<_>>());
                ins.push(zero_ref);
                survey(&mut ctx, &mut p, pidx, 1, &ins, &units12, if full { 12 } else { 6 }, tl, None, None, 0, "s");
            }
            "x_3d" => {
                p = Part::new(name, "phase-encoded thin slices (5 × 5 over 200 mm): cross terms, B0 term, 2nd / 3rd-order spatial terms; chirp + tri100, 100 ms windows", "array", &cfg);
                p.t0 = clock;
                pe_survey(&mut ctx, &mut p, pidx, &[i_chirp, i_tri100], if full { 2 } else { 1 }, ts);
            }
            "l_lin" => {
                if !full {
                    continue;
                }
                p = Part::new(name, "amplitude linearity: tri100 / tri240 at 0.25, 0.5, 0.75, 1.0, both polarities, 100 ms windows", "body", &cfg);
                p.t0 = clock;
                let mut ins = vec![];
                for &l in &[0.25, 0.5, 0.75, 1.0] {
                    ins.push((i_tri100, l));
                    ins.push((i_tri240, l));
                }
                survey(&mut ctx, &mut p, pidx, 3, &ins, &units12, 3, ts, None, None, 0, "l");
            }
            "h1_flow" => {
                p = Part::new(name, "heating with the v2 flow TR train (pge2 segments: encode / rotated readout / tail), to the flow-scan thermal state", "-", &cfg);
                p.t0 = clock;
                ctx.heat(&mut p, if full { 720.0 } else { 600.0 }, 1.0);
            }
            "s_warm" => {
                p = Part::new(name, "survey at the flow-scan load: chirp, tri100, tri240, zero reference; 8 s of heating (flow TRs + heater TRs at 1.3 × so that the mean power = the flow scan's) after every 12-frame sweep", "body", &cfg);
                p.t0 = clock;
                let ins = vec![(i_chirp, 1.0), (i_tri100, 1.0), (i_tri240, 1.0), zero_ref];
                survey(&mut ctx, &mut p, pidx, 1, &ins, &units12, if full { 4 } else { 3 }, tl, None, Some((8.0, 1.3)), 100, "s");
            }
            "v_v2" => {
                p = Part::new(name, "direct measurement of the played v2 readouts (axis-a components) after 10 previous TRs of the scan, 1.0 mm slices, at the flow-scan load", "body", &cfg);
                p.t0 = clock;
                v_block(&mut ctx, &mut p, pidx, &vr, &venc, nhist, 1, &mut vin);
            }
            "h2_hot" => {
                if !full {
                    continue;
                }
                p = Part::new(name, "heating at 1.5 × the flow-scan power (flow TRs + low-slew heater TRs)", "-", &cfg);
                p.t0 = clock;
                ctx.heat(&mut p, 480.0, 1.5);
            }
            "s_hot" => {
                if !full {
                    continue;
                }
                p = Part::new(name, "survey at 1.5 × load: chirp + tri100, heating between sweeps", "body", &cfg);
                p.t0 = clock;
                survey(&mut ctx, &mut p, pidx, 1, &[(i_chirp, 1.0), (i_tri100, 1.0)], &units12, 2, tl, None, Some((8.0, 1.97)), 200, "s");
            }
            "d_cool" => {
                p = Part::new(name, "cool-down tracking: chirp on the outer slices every 2 min (24 frames), idle in between", "body", &cfg);
                p.t0 = clock;
                let reps = if full { 12 } else { 6 };
                for k in 0..reps {
                    survey(&mut ctx, &mut p, pidx, 5, &[(i_chirp, 1.0)], &units_outer, 1, tl, None, None, 300 + 2 * k, "d");
                    let t_set_end = p.clock;
                    let tr = ctx.trid("idle");
                    let mut first = true;
                    while p.clock < t_set_end + 120.0 - 5.5 {
                        let mut b = Block::delay(1.0);
                        if first {
                            b.labels.push(Label::set("TRID", tr));
                            first = false;
                        }
                        p.push(vec![b], cfg.seg_gap);
                    }
                }
            }
            _ => unreachable!(),
        }
        add(p, &mut clock, &mut parts);
    }
    if !tgt_list.is_empty() {
        let pidx = parts.len();
        let mut p = Part::new("h3_flow", "pass 2: heating to the flow-scan state", "-", &cfg);
        p.t0 = clock;
        ctx.heat(&mut p, 480.0, 1.0);
        add(p, &mut clock, &mut parts);
        let pidx2 = parts.len();
        let mut p = Part::new("r_target", "pass 2: targeted resonance inputs (narrow chirp ±6 half-widths, 30 ms; tone burst 20 ms) on the resonance's axis, 200 ms ring-down, flow-load heating between sweeps", "body", &cfg);
        p.t0 = clock;
        for (k, &(a, _f0, _z, i1, i2)) in tgt_list.iter().enumerate() {
            let units: Vec<(usize, usize)> = (0..cfg.slices.len()).map(|s| (a, s)).collect();
            survey(&mut ctx, &mut p, pidx2, 6, &[(i1, 1.0), (i2, 1.0)], &units, 3, tl, None, Some((2.0, 1.38)), 1000 * (k + 1), &format!("r{k}"));
        }
        let _ = pidx;
        add(p, &mut clock, &mut parts);
    }

    // ── outputs: frames, power timeline, sequences, check files
    let mut frames: Vec<f64> = vec![];
    let mut nfr = 0;
    for p in &parts {
        for r in &p.frames {
            frames.extend_from_slice(r);
            nfr += 1;
        }
    }
    wr_f64(&out.join("frames.npy"), &[nfr, NCOL], &frames);
    let total = clock;
    let nb = total.ceil() as usize + 1;
    let mut pw = vec![0.0; nb * 4];
    for i in 0..nb {
        pw[i * 4] = i as f64 + 0.5;
    }
    for p in &parts {
        for &(t, e) in &p.energy {
            let i = ((p.t0 + t).floor() as usize).min(nb - 1);
            for a in 0..3 {
                pw[i * 4 + 1 + a] += e[a];
            }
        }
    }
    wr_f64(&out.join("power.npy"), &[nb, 4], &pw);
    if !vin.is_empty() {
        let nmax = vin.iter().flat_map(|v| v.iter().map(|g| g.len())).max().unwrap();
        let mut arr = vec![0.0; nvr * 3 * nmax];
        for r in 0..nvr {
            for a in 0..3 {
                let g = &vin[r][a];
                arr[(r * 3 + a) * nmax..(r * 3 + a) * nmax + g.len()].copy_from_slice(g);
            }
        }
        wr_f64(&out.join("vframe_in.npy"), &[nvr, 3, nmax], &arr);
        let rr: Vec<f64> = vr.iter().flat_map(|m| m.iter().flat_map(|r| r.iter().copied())).collect();
        wr_f64(&out.join("vframe_R.npy"), &[nvr, nhist + 1, 3, 3], &rr);
        wr_f64(&out.join("vframe_enc.npy"), &[nvr, nhist + 1], &venc.iter().map(|&e| e as f64).collect::<Vec<_>>());
    }

    // PNS per segment type (one representative instance at full level, both models) and of the flow TR train
    let mut pns_tab = serde_json::Map::new();
    for (k, i) in inputs.iter().enumerate() {
        let v: Vec<(f64, f64)> = (0..3).map(|a| pns_pair(&cfg, &i.g, a)).collect();
        pns_tab.insert(format!("input {k} {}", i.name), json!({"safe_example_xyz": v.iter().map(|x| x.0).collect::<Vec<_>>(), "ge_hrmw_xyz": v.iter().map(|x| x.1).collect::<Vec<_>>(), "slew_fraction": s_frac.get(&i.name)}));
    }
    let mut worst_flow: (f64, f64) = (0.0, 0.0);
    for j in 0..30 {
        let e = j % ne;
        let w = ftr.tr_played(e, &rots[j * 37 % rots.len()], 1.0, gapn);
        let g = Grad { dt: ftr.dt, t0: 0.0, g: w };
        worst_flow.0 = worst_flow.0.max(pns_periodic(&g, &cfg.safe, 4));
        worst_flow.1 = worst_flow.1.max(pns_ge_periodic(&g, &cfg.ge, 4));
    }
    pns_tab.insert("flow TR train (pge2 segments + gaps, 30 rotations)".into(), json!({"safe_example": worst_flow.0, "ge_hrmw": worst_flow.1}));
    let hg = Grad { dt: cfg.sys.grad_raster, t0: 0.0, g: heater.clone() };
    pns_tab.insert("heater TR".into(), json!({"safe_example": pns_periodic(&hg, &cfg.safe, 4), "ge_hrmw": pns_ge_periodic(&hg, &cfg.ge, 4)}));
    // frame overhead: spoiler + rephaser + slice select (worst axis)
    let f0 = FrameSpec { kind: 1, trid: 1, axis: 1, slice: 0, pos: cfg.slices[0], eta: 1.0, sigma: 1.0, level: 1.0, input: 0, pe: Some([pe_kmax(&cfg), pe_kmax(&cfg)]), thk: cfg.thk, t_adc: 0.01, avg: 0, readout: -1 };
    let (bl, _) = frame(&cfg, &f0, &vec![0.0; 3]);
    let mut sq = Sequence::new(cfg.sys.clone());
    for b in bl {
        sq.add(b);
    }
    let fg = sq.grad_raster();
    pns_tab.insert("frame overhead (slice select, rephaser + max PE, spoiler)".into(), json!({"safe_example": pns(&fg, &cfg.safe).max, "ge_hrmw": pns_ge(&fg, &cfg.ge, 1000).2}));

    let mut ptab = vec![];
    let mut checks = serde_json::Map::new();
    for p in &parts {
        let n_frames = p.frames.len();
        let bytes_body = p.n_adc_samples as f64 * 8.0 * 2.0; // complex64 × 2 channels [CHECK body-coil channel count]
        let bytes_array = p.n_adc_samples as f64 * 8.0 * 48.0;
        ptab.push(json!({"name": p.name, "purpose": p.purpose, "coil": p.coil, "t0_s": p.t0, "duration_s": p.duration(), "n_blocks": p.seq.n_blocks(), "n_frames": n_frames,
            "n_adc_events": p.n_adc_events, "n_adc_samples": p.n_adc_samples, "data_GB_body2ch": bytes_body / 1e9, "data_GB_array48ch": bytes_array / 1e9,
            "n_segments_types": p.first_of_trid.len(), "mean_power_T2m2": p.energy.iter().map(|x| x.1.iter().sum::<f64>()).sum::<f64>() / p.duration().max(1e-9) }));
        if !no_seq {
            let errs = p.seq.check();
            if !errs.is_empty() {
                eprintln!("{}: {} check problems, first: {}", p.name, errs.len(), errs[0]);
            }
            let txt = p.seq.write(&out.join(format!("{}.seq", p.name))).unwrap();
            let cs = &p.check;
            let ctxt = cs.write(&out.join(format!("{}_check.seq", p.name))).unwrap();
            let g = cs.grad_raster();
            wr_f64(&out.join(format!("{}_check_grad.npy", p.name)), &[g.len(), 3], &g.g.iter().flat_map(|v| v.iter().copied()).collect::<Vec<_>>());
            let at: Vec<f64> = cs.adc_times().into_iter().flatten().collect();
            wr_f64(&out.join(format!("{}_check_adc.npy", p.name)), &[at.len()], &at);
            // PNS over one instance of every segment type of the part (both models)
            let ps = pns(&g, &cfg.safe).max;
            let pg = pns_ge(&g, &cfg.ge, 2000).2;
            checks.insert(p.name.clone(), json!({"seq_bytes": txt.len(), "check_blocks": cs.n_blocks(), "check_bytes": ctxt.len(), "check_problems": errs.len(), "pns_safe_example_max": ps, "pns_ge_hrmw_max": pg, "n_shapes": p.seq.n_shapes()}));
        }
        println!("  {:<9} {:>7.1} s  {:>7} blocks {:>6} frames {:>6} ADC events  {}", p.name, p.duration(), p.seq.n_blocks(), n_frames, p.n_adc_events, p.coil);
    }
    let cols = ["part", "frame", "kind", "axis", "slice", "pos_m", "eta", "sigma", "level", "input", "pe_u_radm", "pe_v_radm", "avg", "t_start_s", "t_exc_s", "t_in0_rel_exc_s", "adc0_rel_exc_s", "n_adc", "readout", "thk_m", "adc_event0", "n_adc_events", "t_adc_s", "t_hist0_s", "n_hist", "dwell_s"];
    let summary = json!({
        "generator": "hasty-bench girfqa", "plan": plan, "total_s": total, "changeover_s": changeover,
        "frame_columns": cols, "kinds": {"7": "receiver timing per dwell", "1": "survey S (cold / warm / hot)", "2": "phase-encoded X", "3": "linearity L", "4": "v2 validation V", "5": "cool-down D", "6": "targeted R", "11": "projection", "12": "3D B0 map", "13": "rx delay", "14": "T1 IR", "15": "T2 CPMG", "16": "slice position", "17": "slice image", "18": "noise"},
        "inputs": inputs.iter().map(|i| json!({"name": i.name, "n": i.g.len(), "slew_fraction": s_frac.get(&i.name)})).collect::<Vec<_>>(),
        "targets": tgt_list.iter().map(|t| json!({"axis": t.0, "f0": t.1, "zeta": t.2, "inputs": [t.3, t.4]})).collect::<Vec<_>>(),
        "config": {"grad_raster": cfg.sys.grad_raster, "rf_raster": cfg.sys.rf_raster, "adc_raster": cfg.sys.adc_raster, "dwell": cfg.dwell, "adc_chunk": cfg.adc_chunk, "gamma_pulseq": cfg.sys.gamma, "gamma_rad": GAMMA,
            "slices_m": cfg.slices, "thk": cfg.thk, "thk_v": cfg.thk_v, "tbw": cfg.tbw, "t_rf": cfg.t_rf, "g_ss": cfg.g_ss(cfg.thk), "g_ss_v": cfg.g_ss(cfg.thk_v), "flip_deg": cfg.flip.to_degrees(),
            "t_adc_long": tl, "t_adc_short": ts, "pe_n": cfg.pe_n, "pe_fov": cfg.pe_fov, "seg_gap": cfg.seg_gap, "seg_gap_samples": gapn, "g_cap": cfg.g_cap, "klim": cfg.klim(cfg.thk), "pns_limit": cfg.pns_limit, "ge_coil": cfg.ge.name},
        "flow_tr": {"dt": ftr.dt, "t_ro": ftr.t_ro, "n_enc": ftr.enc[0].len(), "n_ro": ftr.readout.len(), "n_tail": ftr.tails[0].len(), "te": te, "t_adc": ftr.t_adc, "p_flow_T2m2": p_flow, "p_heater_T2m2": p_heater, "n_hist": nhist, "n_v_readouts": nvr},
        "parts": ptab, "pns": pns_tab, "checks": checks, "trids": ctx.trids, "elapsed_s": t_start.elapsed().as_secs_f64(),
    });
    std::fs::write(out.join("protocol.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    println!("girfqa: {} parts, {:.1} min total (incl. {changeover} s changeovers), {} frames → {} ({:.1} s)", parts.len(), total / 60.0, nfr, out.display(), t_start.elapsed().as_secs_f64());
}
