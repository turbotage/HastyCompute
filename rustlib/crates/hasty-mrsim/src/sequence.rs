//! Spiral 4D-flow GRE sequence: per-TR waveform = velocity-encoding bipolars (+ kz blip) followed by a rotated spiral-out
//! readout. Produces the readout schedule (timing, encode, rotation, cardiac time) and, per ADC sample, the gradient moments
//! in the physical frame, from which the truth simulator and the reconstruction model are both derived.

use crate::physio::{CardiacTime, Ecg};
use crate::trajectory::{spiral_readout, Readout};
use crate::waveform::{bipolar, trapezoid, MomentSample, Moments, Waveform};
use crate::{mat_mul, rotation, GradLimits, Mat3, Vec3, GAMMA, GAMMA_BAR, IDENTITY};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Encoding {
    /// reference (M1 = 0) + one axis per encode with ΔM1 = π/(γ VENC)
    Referenced,
    /// balanced four-point (Hadamard): M1_e = ΔM1/2 · h_e
    Hadamard,
    /// seven-point dual-VENC [ref, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]: per axis one encode at `venc` (the sensitive,
    /// low VENC) and one at `venc_hi`; the layout of the learned denoiser (rustlib/tools/denoiser)
    DualVenc { venc_hi: f64 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Coverage {
    /// in-plane spiral interleaves (golden-angle rotation about z) × kz partitions (golden-ratio ordering)
    StackOfSpirals,
    /// the spiral plane is rotated in 3D (golden-means directions + golden-angle in-plane rotation)
    Rotated3d,
}

#[derive(Clone, Debug)]
pub struct FlowSpiralParams {
    /// T
    pub b0: f64,
    pub lim: GradLimits,
    /// gradient design raster, s
    pub dt: f64,
    /// ADC dwell, s
    pub dwell: f64,
    /// logical FOV (x, y in-plane; z slab), m
    pub fov: Vec3,
    /// logical resolution, m
    pub res: Vec3,
    /// spiral interleaves that fully sample one kz plane (stack) / the in-plane spiral (3D)
    pub n_int: usize,
    /// variable density: FOV at kmax relative to the centre (1 = uniform)
    pub fov_edge_ratio: f64,
    /// m/s
    pub venc: f64,
    pub encoding: Encoding,
    pub coverage: Coverage,
    /// time from the RF centre until encoding gradients may start (RF half-width + slab rephaser), s
    pub t_rf: f64,
    /// spoiler + dead time after the readout ramp-down, s
    pub t_spoil: f64,
    /// total acquisition time, s
    pub scan_time: f64,
    /// logical → physical rotation of the prescription
    pub rot_presc: Mat3,
    /// FOV centre in the physical frame (isocentre = origin), m
    pub fov_center: Vec3,
    /// cardiac model
    pub mean_rr: f64,
    pub rr_rel_std: f64,
    pub n_frames: usize,
    pub seed: u64,
}

impl FlowSpiralParams {
    /// Thoracic-aorta 4D flow on a clinical 3 T system (80 mT/m, 200 T/m/s): 2 mm isotropic, VENC 1.5 m/s,
    /// sagittal-oblique slab, stack-of-spirals.
    pub fn aorta_3t_default() -> Self {
        Self {
            b0: 3.0,
            lim: GradLimits::clinical_3t().derated(0.95),
            dt: 4e-6,
            dwell: 2e-6,
            fov: [0.36, 0.27, 0.08],
            res: [0.002, 0.002, 0.002],
            n_int: 16,
            fov_edge_ratio: 0.7,
            venc: 1.5,
            encoding: Encoding::Referenced,
            coverage: Coverage::StackOfSpirals,
            t_rf: 0.8e-3,
            t_spoil: 1.0e-3,
            scan_time: 60.0,
            rot_presc: IDENTITY,
            fov_center: [0.0; 3],
            mean_rr: 0.9,
            rr_rel_std: 0.05,
            n_frames: 20,
            seed: 1,
        }
    }
    pub fn n_partitions(&self) -> usize {
        match self.coverage {
            Coverage::StackOfSpirals => (self.fov[2] / self.res[2]).round() as usize,
            Coverage::Rotated3d => 1,
        }
    }
    /// first-moment step that gives a phase difference of π at |v| = VENC
    pub fn delta_m1(&self) -> f64 {
        std::f64::consts::PI / (GAMMA * self.venc)
    }
    /// logical M1 vector of each encode
    pub fn encode_m1(&self) -> Vec<Vec3> {
        let d = self.delta_m1();
        match self.encoding {
            Encoding::Referenced => vec![[0.0; 3], [d, 0.0, 0.0], [0.0, d, 0.0], [0.0, 0.0, d]],
            Encoding::Hadamard => [[-1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [1.0, -1.0, 1.0], [-1.0, 1.0, 1.0]].iter().map(|h| [0, 1, 2].map(|a| 0.5 * d * h[a])).collect(),
            Encoding::DualVenc { venc_hi } => {
                let dh = std::f64::consts::PI / (GAMMA * venc_hi);
                let mut m = vec![[0.0; 3]];
                for a in 0..3 {
                    for step in [d, dh] {
                        let mut v = [0.0; 3];
                        v[a] = step;
                        m.push(v);
                    }
                }
                m
            }
        }
    }
}

/// One TR.
#[derive(Clone, Copy, Debug)]
pub struct ReadoutRec {
    /// absolute excitation time, s
    pub t_exc: f64,
    pub enc: usize,
    pub part: usize,
    /// logical rotation of the base readout
    pub rot: Mat3,
    pub card: CardiacTime,
    pub frame: usize,
}

/// Per-ADC-sample parameters for the direct-sum simulator (f32, 16 per sample):
/// k = γM0 (rad/m), m1 = γM1 (rad·s/m), m2h = γM2/2 (rad·s²/m), w = concomitant coefficients of [z², x²+y², xz, yz]
/// (rad/m²), tau = time since excitation (s), delta = excitation time minus the flow-state time (s, set by the driver),
/// phi0 = a spatially constant phase (rad; the B0 terms of a played GIRF chain, 0 otherwise).
pub const SAMPLE_FLOATS: usize = 16;

#[derive(Clone, Debug)]
pub struct Acquisition {
    pub params: FlowSpiralParams,
    pub readout: Readout,
    /// readout start relative to excitation (= TE of the spiral-out), s
    pub t_ro: f64,
    pub tr: f64,
    /// base readout moments, waveform placed at t_ro (time origin = excitation)
    pub ro_mom: Moments,
    /// ADC sample times relative to excitation, s
    pub adc_times: Vec<f64>,
    /// pre-readout moments per [encode][partition], logical frame
    pub pre: Vec<Vec<MomentSample>>,
    /// kz of each partition, cycles/m
    pub kz: Vec<f64>,
    pub readouts: Vec<ReadoutRec>,
    pub ecg: Ecg,
}

fn frac(x: f64) -> f64 {
    x - x.floor()
}

/// rotation taking +z to the unit vector n, followed by an in-plane rotation ψ about n
fn align_z(n: Vec3, psi: f64) -> Mat3 {
    let z = [0.0, 0.0, 1.0];
    let c = n[2];
    let axis = [z[1] * n[2] - z[2] * n[1], z[2] * n[0] - z[0] * n[2], z[0] * n[1] - z[1] * n[0]];
    let s = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let r_align = if s < 1e-12 { if c > 0.0 { IDENTITY } else { rotation([1.0, 0.0, 0.0], std::f64::consts::PI) } } else { rotation(axis, s.atan2(c)) };
    mat_mul(&r_align, &rotation(z, psi))
}

impl Acquisition {
    pub fn new(p: FlowSpiralParams) -> Self {
        let kmax = 0.5 / p.res[0].min(p.res[1]);
        let readout = spiral_readout(kmax, p.n_int, p.fov[0].max(p.fov[1]), p.fov[0].max(p.fov[1]) * p.fov_edge_ratio, &p.lim, p.dt);
        let nz = p.n_partitions();
        let kz: Vec<f64> = (0..nz).map(|j| (j as f64 - (nz / 2) as f64) / p.fov[2]).collect();
        let enc_m1 = p.encode_m1();
        // pre-readout waveforms: per-axis bipolars (limits shared between simultaneously active axes), then the kz blip
        // pre-readout waveforms: per-axis bipolars (limits shared between simultaneously active axes), then the kz blip at
        // the same time for every encode (its M1 then cancels in the encode differences)
        let bips: Vec<Vec<Vec<f64>>> = enc_m1
            .iter()
            .map(|m1| {
                let active = m1.iter().filter(|&&v| v != 0.0).count().max(1);
                let l = p.lim.derated(1.0 / (active as f64).sqrt());
                (0..3).map(|a| bipolar(m1[a], &l, p.dt)).collect()
            })
            .collect();
        let tb = bips.iter().flatten().map(|b| b.len()).max().unwrap_or(0);
        let mut pre_w: Vec<Vec<Waveform>> = Vec::new();
        for bip in &bips {
            let mut row = Vec::new();
            for &kzj in &kz {
                let blip = trapezoid(kzj / GAMMA_BAR, &p.lim, p.dt);
                let mut z = bip[2].clone();
                z.resize(tb, 0.0);
                z.extend_from_slice(&blip);
                let mut w = Waveform::from_axes([&bip[0], &bip[1], &z], p.dt, p.t_rf);
                if w.g.is_empty() {
                    w.g.push([0.0; 3]);
                }
                row.push(w);
            }
            pre_w.push(row);
        }
        let t_pre_end = pre_w.iter().flatten().map(|w| w.t_end()).fold(p.t_rf, f64::max);
        let t_ro = (t_pre_end / p.dt).ceil() * p.dt;
        let pre: Vec<Vec<MomentSample>> = pre_w.iter().map(|row| row.iter().map(|w| w.moments().last()).collect()).collect();
        let mut wro = readout.wave.clone();
        wro.t0 = t_ro;
        let ro_mom = wro.moments();
        let nadc = ((readout.adc.1 - readout.adc.0) / p.dwell).floor() as usize;
        let adc_times: Vec<f64> = (0..nadc).map(|n| t_ro + readout.adc.0 + n as f64 * p.dwell).collect();
        let tr = t_ro + readout.wave.duration() + p.t_spoil;
        // schedule: shots of n_enc consecutive TRs
        let ecg = Ecg::new(p.scan_time + 2.0, p.mean_rr, p.rr_rel_std, p.seed);
        let nenc = enc_m1.len();
        let nshots = (p.scan_time / (tr * nenc as f64)).floor() as usize;
        let golden_angle = std::f64::consts::PI * (3.0 - 5f64.sqrt());
        let (g1, g2) = (0.4656, 0.6823); // 2D golden means
        let mut readouts = Vec::with_capacity(nshots * nenc);
        // per-partition visit counters: the in-plane angle must advance per visit of *that* partition. (Using the global
        // shot index for both the golden-ratio partition order and the golden angle aliases them: shots landing in one
        // partition are Fibonacci-spaced and the golden angle times a Fibonacci number is almost a full turn.)
        let mut visits = vec![0usize; nz];
        for s in 0..nshots {
            let (part, rot) = match p.coverage {
                Coverage::StackOfSpirals => {
                    let part = (frac(s as f64 * 0.618_033_988_749_895) * nz as f64) as usize;
                    let v = visits[part];
                    visits[part] += 1;
                    let ang = v as f64 * golden_angle + part as f64 * 2.0 * std::f64::consts::PI * 0.754_877_666_246_692_7;
                    (part, rotation([0.0, 0.0, 1.0], ang))
                }
                Coverage::Rotated3d => {
                    let cz = 1.0 - 2.0 * frac(s as f64 * g1);
                    let ph = 2.0 * std::f64::consts::PI * frac(s as f64 * g2);
                    let st = (1.0 - cz * cz).max(0.0).sqrt();
                    (0, align_z([st * ph.cos(), st * ph.sin(), cz], s as f64 * golden_angle))
                }
            };
            for e in 0..nenc {
                let t_exc = (s * nenc + e) as f64 * tr;
                let card = ecg.at(t_exc);
                let frame = ((card.phase * p.n_frames as f64) as usize).min(p.n_frames - 1);
                readouts.push(ReadoutRec { t_exc, enc: e, part, rot, card, frame });
            }
        }
        Self { params: p, readout, t_ro, tr, ro_mom, adc_times, pre, kz, readouts, ecg }
    }

    pub fn n_adc(&self) -> usize {
        self.adc_times.len()
    }
    pub fn n_samples(&self) -> usize {
        self.readouts.len() * self.n_adc()
    }

    /// moments (logical frame) at ADC sample `n` of readout `i`
    pub fn moments_logical(&self, i: usize, n: usize) -> MomentSample {
        let r = &self.readouts[i];
        let ro = self.ro_mom.at(self.adc_times[n]).rotate(&r.rot);
        self.pre[r.enc][r.part].add(&ro)
    }
    pub fn moments_physical(&self, i: usize, n: usize) -> MomentSample {
        self.moments_logical(i, n).rotate(&self.params.rot_presc)
    }

    /// direct-sum parameters for readouts `range` (row-major [readout][adc sample][16])
    pub fn sample_params(&self, range: std::ops::Range<usize>) -> Vec<f32> {
        let na = self.n_adc();
        let mut out = Vec::with_capacity(range.len() * na * SAMPLE_FLOATS);
        let wc = GAMMA / (2.0 * self.params.b0);
        for i in range {
            for n in 0..na {
                let m = self.moments_physical(i, n);
                let p = &m.p; // xx yy zz xy xz yz
                let vals = [
                    GAMMA * m.m0[0],
                    GAMMA * m.m0[1],
                    GAMMA * m.m0[2],
                    GAMMA * m.m1[0],
                    GAMMA * m.m1[1],
                    GAMMA * m.m1[2],
                    0.5 * GAMMA * m.m2[0],
                    0.5 * GAMMA * m.m2[1],
                    0.5 * GAMMA * m.m2[2],
                    wc * (p[0] + p[1]),
                    wc * p[2] / 4.0,
                    -wc * p[4],
                    -wc * p[5],
                    self.adc_times[n],
                    0.0,
                    0.0,
                ];
                out.extend(vals.iter().map(|&v| v as f32));
            }
        }
        out
    }

    /// nominal k-space of readout i in the logical frame, rad/m (what a reconstruction uses as trajectory)
    pub fn k_logical(&self, i: usize) -> Vec<Vec3> {
        (0..self.n_adc()).map(|n| self.moments_logical(i, n).m0.map(|v| GAMMA * v)).collect()
    }
}

/// What the direct-sum simulator needs from any sequence: the readout schedule (one record per excitation, a fixed
/// number of ADC samples each) and the per-sample parameters in the [`SAMPLE_FLOATS`] layout. Implemented by the spiral
/// [`Acquisition`] here and by the `hasty-seq` designs (hasty-bench `seqsched`).
pub trait Schedule: Sync {
    fn n_readouts(&self) -> usize;
    fn n_adc(&self) -> usize;
    fn tr(&self) -> f64;
    /// first echo time after the excitation, s
    fn te(&self) -> f64;
    fn t_exc(&self, i: usize) -> f64;
    /// row of readouts.npy: t_exc, enc, part, frame, cardiac phase, rotation (9, row-major)
    fn readout_row(&self, i: usize) -> [f64; 14];
    /// [readout][adc sample][SAMPLE_FLOATS] for readouts `range` (delta = 0; the driver sets it)
    fn sample_params(&self, range: std::ops::Range<usize>) -> Vec<f32>;
    /// one-line description for logs
    fn describe(&self) -> String;
}

impl Schedule for Acquisition {
    fn n_readouts(&self) -> usize {
        self.readouts.len()
    }
    fn n_adc(&self) -> usize {
        self.adc_times.len()
    }
    fn tr(&self) -> f64 {
        self.tr
    }
    fn te(&self) -> f64 {
        self.t_ro
    }
    fn t_exc(&self, i: usize) -> f64 {
        self.readouts[i].t_exc
    }
    fn readout_row(&self, i: usize) -> [f64; 14] {
        let r = &self.readouts[i];
        let mut v = [0.0; 14];
        v[..5].copy_from_slice(&[r.t_exc, r.enc as f64, r.part as f64, r.frame as f64, r.card.phase]);
        for a in 0..3 {
            for b in 0..3 {
                v[5 + 3 * a + b] = r.rot[a][b];
            }
        }
        v
    }
    fn sample_params(&self, range: std::ops::Range<usize>) -> Vec<f32> {
        Acquisition::sample_params(self, range)
    }
    fn describe(&self) -> String {
        format!("spiral 4D flow ({:?}, {:?}), {} encodes", self.params.encoding, self.params.coverage, self.params.encode_m1().len())
    }
}

/// the [`SAMPLE_FLOATS`] record of one sample from physical-frame moments (time origin = excitation): M0, M1, M2 (T·s^n/m)
/// and P = ∫G Gᵀ (xx, yy, zz, xy, xz, yz), field b0 (T), time since excitation tau (s)
pub fn sample_record(m0: &Vec3, m1: &Vec3, m2: &Vec3, p: &[f64; 6], b0: f64, tau: f64) -> [f32; SAMPLE_FLOATS] {
    let wc = GAMMA / (2.0 * b0);
    let v = [
        GAMMA * m0[0],
        GAMMA * m0[1],
        GAMMA * m0[2],
        GAMMA * m1[0],
        GAMMA * m1[1],
        GAMMA * m1[2],
        0.5 * GAMMA * m2[0],
        0.5 * GAMMA * m2[1],
        0.5 * GAMMA * m2[2],
        wc * (p[0] + p[1]),
        wc * p[2] / 4.0,
        -wc * p[4],
        -wc * p[5],
        tau,
        0.0,
        0.0,
    ];
    v.map(|x| x as f32)
}
