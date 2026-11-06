//! One TR of the flow sequence and its physical waveforms: [RF][velocity encode (physical axes)][R·readout (base
//! spoke, rotated)][R·tail]. The readout starts at k = 0 (center-out), i.e. at the echo, so the encode's first moment is
//! the velocity encoding at TE. The tail (ramp-down + rewinder + spoiler in one) takes the TR's total zeroth moment to
//! the same *physical* spoil moment every TR (what RF spoiling needs): in the base frame its target is Rᵀ·spoil, so it
//! differs per rotation; the TR length is set by the longest tail over the rotation set. Only the ADC part is a pure
//! rotation of the base spoke.
//!
//! PNS of the played sequence: the SAFE model on the concatenated physical waveform (all encodes of each rotation back to
//! back, rotations in acquisition order), so tails of previous TRs with other rotations are included.

use crate::encode::{design, EncodeDesign, EncodeSpec, Scheme};
use crate::pns::{pns, SafeHw};
use crate::system::System;
use crate::traj::{tail, Spoke};
use crate::wave::Grad;
use crate::{mat_vec, norm, transpose, Mat3, Vec3};

#[derive(Clone, Debug)]
pub struct TrSpec {
    /// per-axis hardware limits (encode); the tail is designed with vector limits gmax, smax·tail_slew
    pub sys: System,
    /// slew factors of the encode and the tail (PNS derating)
    pub enc_slew: f64,
    pub tail_slew: f64,
    /// hard-pulse duration, s
    pub rf_dur: f64,
    pub scheme: Scheme,
    pub venc: f64,
    pub balanced: bool,
    pub spoil: Spoil,
}

/// part of the TR a raster segment belongs to
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Rf,
    Encode,
    Readout,
    Tail,
}

/// the TR's total zeroth moment (gradient spoiling)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Spoil {
    /// the same moment on physical axes every TR (T·s/m), what RF-spoiling theory assumes; needs a per-rotation tail
    Physical(Vec3),
    /// no tail: the readout's end point is the moment (magnitude fixed, direction rotating with the spoke), so the
    /// whole TR after the encode is trajectory
    Endpoint,
    /// the trajectory continues out along its end direction to |M0| = this (T·s/m): rotates with the spoke, the tail
    /// is the same in the base frame for every rotation (a rotation of one base waveform)
    Radial(f64),
    /// constant physical moment (T·s/m) *and* the same total first moment for every encode: one block per encode after
    /// the readout carries the spoil moment and rewinds that encode's velocity-encoding M1 (moving spins then see the
    /// same per-TR phase in every TR, which keeps RF spoiling intact for flow). Needs a readout that ends at rest at
    /// k ≈ 0 with ≈ 0 net M1 (the out-and-back yarn), so the block does not depend on the rotation.
    FlowComp(Vec3),
}

#[derive(Clone, Debug)]
pub struct Tr {
    pub dt: f64,
    /// RF centre (s from the TR start), and raster indices: encode start, readout start (echo), tail start, TR end
    pub t_rf: f64,
    pub i_enc: usize,
    pub i_ro: usize,
    pub i_tail: usize,
    pub n: usize,
    pub encode: EncodeDesign,
    pub spoke: Spoke,
    pub spoil: Spoil,
    pub tail_gmax: f64,
    pub tail_smax: f64,
    /// longest tail over the rotation set the TR was built for (raster segments)
    pub n_tail: usize,
    /// per-encode tail blocks on physical axes (`Spoil::FlowComp`)
    pub enc_tail: Option<EncodeDesign>,
}

fn append(g: &mut Vec<Vec3>, w: &[Vec3]) {
    assert!(norm(g.last().unwrap()) == 0.0 && norm(&w[0]) < 1e-15, "blocks must join at G = 0");
    g.extend_from_slice(&w[1..]);
}

impl Tr {
    /// the TR for `spoke`, long enough for the tails of every rotation in `rots`
    pub fn build(spec: &TrSpec, spoke: Spoke, rots: &[Mat3]) -> Tr {
        let sys = &spec.sys;
        let dt = sys.grad_raster;
        let t_rf = sys.rf_dead + 0.5 * spec.rf_dur;
        let i_enc = (sys.ceil_raster(sys.rf_dead + spec.rf_dur + sys.rf_ringdown) / dt).round() as usize;
        let enc_sys = System { smax: sys.smax * spec.enc_slew, ..sys.clone() };
        let es = EncodeSpec { m0: [0.0; 3], m1_offset: [0.0; 3], scheme: spec.scheme, venc: spec.venc, balanced: spec.balanced, t_start: i_enc as f64 * dt, t_ref: t_rf };
        let encode = design(&es, &enc_sys);
        let i_ro = i_enc + encode.waves[0].len() - 1;
        let i_tail = i_ro + spoke.i_adc;
        let mut tr = Tr { dt, t_rf, i_enc, i_ro, i_tail, n: 0, encode, spoke, spoil: spec.spoil, tail_gmax: sys.gmax, tail_smax: sys.smax * spec.tail_slew, n_tail: 0, enc_tail: None };
        if let Spoil::FlowComp(sp) = spec.spoil {
            assert!(norm(&tr.spoke.g_end()) == 0.0, "FlowComp needs a readout ending at rest");
            // total TR M1 of encode e = M1_enc,e + M1_readout + tail M1; the tail targets −M1_enc,e + offset, so every
            // encode ends with M1_readout + offset. The offset per axis is chosen as a single lobe's natural M1
            // (area · (block start + T/2 − t_rf)), iterated on the block length T, shortest kept
            let tail_sys = System { smax: sys.smax * spec.tail_slew, ..sys.clone() };
            let t_s = i_tail as f64 * dt;
            let mk = |half: f64| {
                let off = [0, 1, 2].map(|a| sp[a] * (t_s - t_rf + half));
                design(&EncodeSpec { m0: sp, m1_offset: off, scheme: spec.scheme, venc: -spec.venc, balanced: spec.balanced, t_start: t_s, t_ref: t_rf }, &tail_sys)
            };
            let mut best = mk(0.0);
            let mut half = 0.5 * best.duration;
            for _ in 0..4 {
                let d = mk(half);
                half = 0.5 * d.duration;
                if d.duration < best.duration {
                    best = d;
                }
            }
            tr.n_tail = best.waves[0].len() - 1;
            tr.enc_tail = Some(best);
        } else {
            tr.n_tail = crate::par_map(rots, |r| tr.tail(r).len() - 1).into_iter().max().unwrap_or(0);
        }
        tr.n = i_tail + tr.n_tail;
        tr
    }
    /// tail of rotation `r` in the base frame (sample 0 = readout end gradient)
    pub fn tail(&self, r: &Mat3) -> Grad {
        if let Some(et) = &self.enc_tail {
            // encode 0's block, expressed in the base frame of rotation r
            let rt = transpose(r);
            return Grad { dt: self.dt, t0: 0.0, g: et.waves[0].g.iter().map(|v| mat_vec(&rt, v)).collect() };
        }
        let target = match self.spoil {
            Spoil::FlowComp(_) => unreachable!(),
            Spoil::Physical(s) => mat_vec(&transpose(r), &s),
            Spoil::Endpoint => self.spoke.m0_end,
            Spoil::Radial(a) => {
                let m = self.spoke.m0_end;
                crate::scale(&m, a / norm(&m))
            }
        };
        tail(&self.spoke.g_end(), &self.spoke.m0_end, &target, self.tail_gmax, self.tail_smax, self.dt)
    }
    pub fn te(&self) -> f64 {
        self.i_ro as f64 * self.dt - self.t_rf
    }
    pub fn tr(&self) -> f64 {
        self.n as f64 * self.dt
    }
    pub fn n_enc(&self) -> usize {
        self.encode.waves.len()
    }
    pub fn part(&self, i: usize) -> Part {
        if i < self.i_enc {
            Part::Rf
        } else if i < self.i_ro {
            Part::Encode
        } else if i < self.i_tail {
            Part::Readout
        } else {
            Part::Tail
        }
    }
    /// samples 0..n of the TR for every encode (sample n, the zero start of the next TR, omitted), rotation `r`
    pub fn physical_all(&self, r: &Mat3) -> Vec<Vec<Vec3>> {
        if let Some(et) = &self.enc_tail {
            let sp: Vec<Vec3> = self.spoke.grad.g.iter().map(|v| mat_vec(r, v)).collect();
            return (0..self.n_enc())
                .map(|e| {
                    let mut g = vec![[0.0; 3]; self.i_enc + 1];
                    append(&mut g, &self.encode.waves[e].g);
                    append(&mut g, &sp);
                    append(&mut g, &et.waves[e].g);
                    g.resize(self.n, [0.0; 3]);
                    g
                })
                .collect();
        }
        let t = self.tail(r);
        assert!(t.len() - 1 <= self.n_tail, "tail of this rotation longer than the TR was built for");
        let mut sp = self.spoke.grad.g.clone();
        sp.extend_from_slice(&t.g[1..]);
        let sp: Vec<Vec3> = sp.iter().map(|v| mat_vec(r, v)).collect();
        (0..self.n_enc())
            .map(|e| {
                let mut g = vec![[0.0; 3]; self.i_enc + 1];
                append(&mut g, &self.encode.waves[e].g);
                append(&mut g, &sp);
                g.resize(self.n, [0.0; 3]);
                g
            })
            .collect()
    }
    pub fn physical(&self, e: usize, r: &Mat3) -> Grad {
        let mut g = self.physical_all(r).swap_remove(e);
        g.push([0.0; 3]);
        Grad { dt: self.dt, t0: 0.0, g }
    }
}

/// worst PNS of the played sequence over `rots` (acquisition order, every encode per rotation)
#[derive(Clone, Debug)]
pub struct SeqPns {
    pub max: f64,
    /// maximum within each TR part: RF, encode, readout, tail
    pub part_max: [f64; 4],
    /// rotation (position in `rots`), encode, raster index within the TR, and the TR part
    pub rot: usize,
    pub enc: usize,
    pub i: usize,
    pub part: Part,
}

/// the physical waveform of rotations `rots` (all encodes each), ending with the zero start of the next TR
pub fn sequence(tr: &Tr, rots: &[Mat3]) -> Grad {
    let mut g = Vec::with_capacity(rots.len() * tr.n_enc() * tr.n + 1);
    for r in rots {
        for b in tr.physical_all(r) {
            g.extend(b);
        }
    }
    g.push([0.0; 3]);
    Grad { dt: tr.dt, t0: 0.0, g }
}

pub fn pns_sequence(tr: &Tr, rots: &[Mat3], hw: &SafeHw) -> SeqPns {
    // chunks of rotations, each preceded by enough previous rotations to cover the SAFE memory (≈ 4 τ_max); chunks run
    // on `crate::threads()` threads
    let ne = tr.n_enc();
    let hist = (((4.0e-3 * hw.longest_tau_ms()) / (tr.tr() * ne as f64)).ceil() as usize).max(1);
    let chunk = 32;
    let starts: Vec<usize> = (0..rots.len()).step_by(chunk).collect();
    let parts = crate::par_map(&starts, |&start| {
        let h0 = start.saturating_sub(hist);
        let end = (start + chunk).min(rots.len());
        let p = pns(&sequence(tr, &rots[h0..end]), hw);
        let skip = (start - h0) * ne * tr.n;
        let stop = (end - h0) * ne * tr.n;
        let mut best = SeqPns { max: 0.0, part_max: [0.0; 4], rot: 0, enc: 0, i: 0, part: Part::Rf };
        for j in skip..stop {
            let v = p.norm[p.pad + j];
            let (tri, i) = (j / tr.n, j % tr.n);
            let part = tr.part(i);
            let k = part as usize;
            if v > best.part_max[k] {
                best.part_max[k] = v;
            }
            if v > best.max {
                best = SeqPns { max: v, part_max: best.part_max, rot: h0 + tri / ne, enc: tri % ne, i, part };
            }
        }
        best
    });
    let mut best = SeqPns { max: 0.0, part_max: [0.0; 4], rot: 0, enc: 0, i: 0, part: Part::Rf };
    let mut pm = [0.0f64; 4];
    for p in parts {
        for k in 0..4 {
            pm[k] = pm[k].max(p.part_max[k]);
        }
        if p.max > best.max {
            best = p;
        }
    }
    best.part_max = pm;
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traj::{rotations, RotSet, TopOpts, Yarn};

    #[test]
    fn tr_assembly_moments() {
        let sys = System::prisma_xr();
        let y = Yarn { kmax: 200.0, fov: 0.3, n_design: 6000.0, edge_density: 1.0, kg0: 0.0, wobble: 0.5, wobble_freq: 3.0 };
        let sp = Spoke::design(&y.curve(100_000), &sys, &TopOpts::default());
        let spoil = [0.0, 0.0, 2.0 / (crate::GAMMA_BAR * 0.0025)];
        let spec = TrSpec { sys: sys.clone(), enc_slew: 1.0, tail_slew: 1.0, rf_dur: 100e-6, scheme: Scheme::Hadamard, venc: 1.5, balanced: true, spoil: Spoil::Physical(spoil) };
        let rots = rotations(RotSet::SuperFibonacci, 64);
        let tr = Tr::build(&spec, sp, &rots);
        let d = crate::encode::delta_m1(1.5);
        let m1s: Vec<Vec3> = (0..tr.n_enc()).map(|e| tr.physical(e, &rots[3]).moments(tr.t_rf).m1[tr.i_ro]).collect();
        for r in rots.iter().step_by(7) {
            for e in 0..tr.n_enc() {
                let w = tr.physical(e, r);
                assert_eq!(w.len(), tr.n + 1);
                assert!(w.limits().ok_axis(&sys), "{:?}", w.limits());
                let m = w.moments(tr.t_rf);
                // k = 0 at the echo, and the TR's total moment is the physical spoil moment for every rotation
                assert!(norm(&m.m0[tr.i_ro]) < 1e-12);
                let mend = m.last().0;
                assert!(norm(&crate::sub(&mend, &spoil)) < 1e-12, "{mend:?}");
            }
        }
        // Hadamard: encodes 0 → 1 differ by (ΔM1, ΔM1, 0)
        let s01 = [0, 1, 2].map(|a| m1s[1][a] - m1s[0][a]);
        assert!((s01[0] - d).abs() / d < 1e-9 && (s01[1] - d).abs() / d < 1e-9 && s01[2].abs() < 1e-15, "{s01:?}");
        let p = pns_sequence(&tr, &rots[..20], &SafeHw::example());
        assert!(p.max > 0.0 && p.max.is_finite());
    }
}

#[cfg(test)]
mod flowcomp_tests {
    use super::*;
    use crate::encode::DualLayout;
    use crate::traj::{rotations, RotSet, TopOpts, Yarn};

    #[test]
    fn flowcomp_tail_constant_m0_and_equal_m1() {
        let sys = System::prisma_xr();
        let y = Yarn { kmax: 250.0, fov: 0.32, n_design: 2200.0, edge_density: 1.0, kg0: 0.0, wobble: 0.0, wobble_freq: 2.0 };
        let sp = Spoke::design(&y.curve_return(60_000, 0.0), &System { smax: sys.smax * 0.45, ..sys.clone() }, &TopOpts { v_end: Some(0.0), ..TopOpts::default() });
        let spoil = [0.0, 0.0, 2.0 / (crate::GAMMA_BAR * 2e-3)];
        let spec = TrSpec { sys: sys.clone(), enc_slew: 0.3, tail_slew: 0.5, rf_dur: 100e-6, scheme: Scheme::Coprime { p: 2, layout: DualLayout::Seven }, venc: 1.0, balanced: true, spoil: Spoil::FlowComp(spoil) };
        let rots = rotations(RotSet::SuperFibonacci, 32);
        let tr = Tr::build(&spec, sp, &rots);
        let l = tr.enc_tail.as_ref().unwrap().waves[0].limits();
        assert!(l.ok_axis(&System { smax: sys.smax * 0.5, ..sys.clone() }), "{l:?}");
        let mut spread: f64 = 0.0;
        let mut m1_scale: f64 = 0.0;
        for r in &rots {
            let ends: Vec<_> = (0..tr.n_enc()).map(|e| tr.physical(e, r).moments(tr.t_rf).last()).collect();
            for (m0, m1, _, _) in &ends {
                // total M0 = the physical spoil moment (readout returns to k = 0)
                assert!(norm(&crate::sub(m0, &spoil)) < 1e-3 * norm(&spoil), "{m0:?}");
                spread = spread.max(norm(&crate::sub(m1, &ends[0].1)));
            }
            m1_scale = m1_scale.max(norm(&tr.encode.m1[1]));
        }
        // equal total M1 across encodes: spread tiny compared with the encode first moments
        assert!(spread < 1e-6 * m1_scale.max(1e-12) + 1e-14, "spread {spread} vs encode M1 {m1_scale}");
        println!("flowcomp tail {:.3} ms, M1 spread {spread:.2e} (encode M1 {m1_scale:.2e})", tr.n_tail as f64 * tr.dt * 1e3);
    }
}
