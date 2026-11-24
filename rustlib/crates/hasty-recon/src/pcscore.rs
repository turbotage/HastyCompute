//! Per-frame velocity score against the simulator's phase-contrast signal truth on ANY reconstruction grid
//! (research/temporal_pyramid.md §13). Scoring only: no method reads this truth.
//!
//! Truth: `truth_pc_frames.npz` of the sim (bias_truth.py --signal-frames: the ideal per-cardiac-frame PC signal
//! C_{e,f} = Σ w s e^{−i k_e·u} of the LBM-covered native voxels, spin history and intravoxel velocity spread
//! included), unpacked by research/experiments/temporal_pyramid/pc_native.py into `pc_idx.npy` (flat Fortran native
//! indices, f8), `pc_c.npy` ((n, E, F) c8 C-order) and `pc_covmin.npy` (minimum LBM coverage over the frames, f4).
//!
//! Resampling: the recon voxel's truth is the complex mean of C over the voxel's box centred at its true position
//! (the `SimData::sample_log_box` sub-positions, trilinear on the native grid), i.e. the signal-weighted partial-volume
//! average the voxel measures. A voxel is SCORED only when every native voxel entering that mean is in the truth with
//! coverage ≥ COVMIN in every frame (blood throughout the cycle; the truth has no tissue term). Its truth velocity
//! is the reconstruction's own decoder applied to the truth signal (the same aliasing / decoding rules on both sides).
use hasty_mrsim::Vec3;
use num_complex::Complex64;

use crate::data::{read_npy, SimData};
use crate::flow::FlowDecoder;
use crate::Error;

/// the native truth (sparse over the LBM-covered voxels)
pub struct PcNative {
    pub ne: usize,
    pub nf: usize,
    /// native flat voxel → row of `c` (u32::MAX: no truth)
    lookup: Vec<u32>,
    /// (n, E, F) row-major
    c: Vec<Complex64>,
    covmin: Vec<f32>,
}

impl PcNative {
    pub fn open(dir: &std::path::Path, sd: &SimData) -> Result<Self, Error> {
        let idx: Vec<usize> = read_npy(&dir.join("pc_idx.npy"))?.f64s().iter().map(|&v| v as usize).collect();
        let ca = read_npy(&dir.join("pc_c.npy"))?;
        assert!(!ca.fortran && ca.shape.len() == 3 && ca.shape[0] == idx.len(), "pc_c.npy: (n, E, F) C-order c8");
        let (ne, nf) = (ca.shape[1], ca.shape[2]);
        let c: Vec<Complex64> = ca.c32s().iter().map(|z| Complex64::new(z.re as f64, z.im as f64)).collect();
        let covmin = read_npy(&dir.join("pc_covmin.npy"))?.f32s();
        let d = sd.tdims;
        let mut lookup = vec![u32::MAX; d[0] * d[1] * d[2]];
        for (i, &q) in idx.iter().enumerate() {
            lookup[q] = i as u32;
        }
        Ok(PcNative { ne, nf, lookup, c, covmin })
    }

    /// trilinear corners (native row, weight) at the logical position x; `None` if a corner with weight > 0 has no
    /// truth row or coverage below covmin (strict), else the rows (missing corners dropped when !strict)
    fn corners(&self, sd: &SimData, x: &Vec3, covmin: f64, strict: bool) -> Option<Vec<(usize, f64)>> {
        let d = sd.tdims;
        let mut i0 = [0usize; 3];
        let mut fr = [0.0; 3];
        for a in 0..3 {
            let g = x[a] / sd.res + (d[a] / 2) as f64;
            if g < 0.0 || g > (d[a] - 1) as f64 {
                return if strict { None } else { Some(vec![]) };
            }
            let g = g.min((d[a] - 1) as f64 - 1e-9);
            i0[a] = g.floor() as usize;
            fr[a] = g - i0[a] as f64;
        }
        let mut out = Vec::with_capacity(8);
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let w = (if dx == 1 { fr[0] } else { 1.0 - fr[0] }) * (if dy == 1 { fr[1] } else { 1.0 - fr[1] }) * (if dz == 1 { fr[2] } else { 1.0 - fr[2] });
                    if w <= 1e-12 {
                        continue;
                    }
                    let q = (i0[0] + dx) + d[0] * ((i0[1] + dy) + d[1] * (i0[2] + dz));
                    let r = self.lookup[q];
                    if r == u32::MAX || (self.covmin[r as usize] as f64) < covmin {
                        if strict {
                            return None;
                        }
                        if r == u32::MAX {
                            continue;
                        }
                    }
                    out.push((r as usize, w));
                }
            }
        }
        Some(out)
    }

    /// box mean of C over the recon voxel at x ((E, F) row-major); strict: None unless fully covered
    pub fn box_mean(&self, sd: &SimData, x: &Vec3, covmin: f64, strict: bool) -> Option<Vec<Complex64>> {
        let n: [usize; 3] = std::array::from_fn(|a| ((sd.vox[a] / sd.res) - 1e-9).ceil().max(1.0) as usize);
        let m = self.ne * self.nf;
        let mut acc = vec![Complex64::new(0.0, 0.0); m];
        let mut any = false;
        for k in 0..n[2] {
            for j in 0..n[1] {
                for i in 0..n[0] {
                    let o = [i, j, k];
                    let p: Vec3 = std::array::from_fn(|a| x[a] + ((o[a] as f64 + 0.5) / n[a] as f64 - 0.5) * sd.vox[a]);
                    for (r, w) in self.corners(sd, &p, covmin, strict)? {
                        any = true;
                        for (a, z) in acc.iter_mut().zip(&self.c[r * m..(r + 1) * m]) {
                            *a += z * w;
                        }
                    }
                }
            }
        }
        if !any {
            return if strict { None } else { Some(acc) };
        }
        let s = 1.0 / (n[0] * n[1] * n[2]) as f64;
        acc.iter_mut().for_each(|a| *a *= s);
        Some(acc)
    }
}

/// the truth on the reconstruction grid at the scored voxels
pub struct PcTruth {
    pub vox: Vec<usize>,
    pub ne: usize,
    pub nf: usize,
    /// (n, E, F) row-major
    pub c: Vec<Complex64>,
    /// decoded truth velocity [i * nf + f]
    pub v: Vec<Vec3>,
    pub venc: f64,
    pub covmin: f64,
}

/// one score of a reconstruction (or an oracle) against the truth
#[derive(Clone, Debug, Default)]
pub struct PcScore {
    pub rmse: f64,
    pub rel: f64,
    pub mean_rmse: f64,
    pub detail_rmse: f64,
    pub truth_rms: f64,
    pub truth_detail_rms: f64,
    pub median: f64,
    pub p90: f64,
    /// fraction of voxel-frames with |v − v★| > `venc` (the sim's VENC, meta venc: the low VENC of dual-VENC)
    pub wrap: f64,
    pub venc: f64,
    /// RMS of the wrapped error of arg(x_e x̄_0) against arg(C_e C̄_0), encodes 1.., rad
    pub phase_rms: f64,
    pub per_frame: Vec<f64>,
}

impl PcScore {
    pub fn line(&self) -> String {
        format!(
            "RMSE {:.3} m/s (rel {:.3}; time mean {:.3}, detail {:.3} of truth {:.3}), |Δv| median {:.3} p90 {:.3}, > {:.2} m/s {:.1} %, phase {:.3} rad",
            self.rmse, self.rel, self.mean_rmse, self.detail_rmse, self.truth_detail_rms, self.median, self.p90, self.venc, 100.0 * self.wrap, self.phase_rms
        )
    }
}

impl PcTruth {
    /// scored voxels and their truth; `x_log` = the recon voxels' true positions
    pub fn build(nat: &PcNative, sd: &SimData, x_log: &[Vec3], dec: &FlowDecoder, covmin: f64) -> Self {
        use rayon::prelude::*;
        let (ne, nf) = (nat.ne, nat.nf);
        assert_eq!(ne, dec.n_enc(), "pc truth: {} encodes, the decoder {}", ne, dec.n_enc());
        let got: Vec<(usize, Vec<Complex64>)> = (0..x_log.len()).into_par_iter().filter_map(|q| nat.box_mean(sd, &x_log[q], covmin, true).map(|c| (q, c))).collect();
        let vox: Vec<usize> = got.iter().map(|g| g.0).collect();
        let mut c = Vec::with_capacity(got.len() * ne * nf);
        for (_, g) in &got {
            c.extend_from_slice(g);
        }
        let mut v = Vec::with_capacity(vox.len() * nf);
        for i in 0..vox.len() {
            for f in 0..nf {
                let z: Vec<Complex64> = (0..ne).map(|e| c[(i * ne + e) * nf + f]).collect();
                v.push(dec.velocity(&z));
            }
        }
        PcTruth { vox, ne, nf, c, v, venc: sd.venc, covmin }
    }

    /// nested coarse frames (`--card-frames K`): the complex mean of the truth over each group of nf / K consecutive
    /// cardiac frames (what a coarse frame pooling those readouts measures, equal weights), re-decoded
    pub fn coarsen(self, k: usize, dec: &FlowDecoder) -> Self {
        if k == self.nf {
            return self;
        }
        assert!(k > 0 && self.nf % k == 0, "pc truth: {} frames do not nest into {k}", self.nf);
        let (ne, nf, m) = (self.ne, self.nf, self.nf / k);
        let n = self.n();
        let mut c = vec![Complex64::new(0.0, 0.0); n * ne * k];
        for ie in 0..n * ne {
            for f in 0..nf {
                c[ie * k + f / m] += self.c[ie * nf + f] / m as f64;
            }
        }
        let v = (0..n * k).map(|ik| dec.velocity(&(0..ne).map(|e| c[((ik / k) * ne + e) * k + ik % k]).collect::<Vec<_>>())).collect();
        PcTruth { vox: self.vox, ne, nf: k, c, v, venc: self.venc, covmin: self.covmin }
    }

    pub fn n(&self) -> usize {
        self.vox.len()
    }

    /// one truth per respiratory bin (each at the bin's true breathing state, same cardiac frames) stacked into the
    /// operator's frame order f = b · nf_bin + c, on the voxels that are blood throughout the cycle in EVERY bin (the
    /// intersection: a fixed voxel set over all frames)
    pub fn stack_bins(bins: Vec<PcTruth>) -> Self {
        assert!(!bins.is_empty(), "pc truth: no bins");
        let (ne, nfb, venc, covmin) = (bins[0].ne, bins[0].nf, bins[0].venc, bins[0].covmin);
        for b in &bins {
            assert!(b.ne == ne && b.nf == nfb && b.venc == venc, "pc truth bins: encodes / frames / venc differ");
        }
        let nb = bins.len();
        let pos: Vec<std::collections::HashMap<usize, usize>> = bins.iter().map(|b| b.vox.iter().enumerate().map(|(i, &q)| (q, i)).collect()).collect();
        let vox: Vec<usize> = bins[0].vox.iter().copied().filter(|q| pos.iter().all(|m| m.contains_key(q))).collect();
        let nf = nb * nfb;
        let mut c = vec![Complex64::new(0.0, 0.0); vox.len() * ne * nf];
        let mut v = vec![[0.0f64; 3]; vox.len() * nf];
        for (i, q) in vox.iter().enumerate() {
            for (b, t) in bins.iter().enumerate() {
                let j = pos[b][q];
                for f in 0..nfb {
                    for e in 0..ne {
                        c[(i * ne + e) * nf + b * nfb + f] = t.c_at(j, e, f);
                    }
                    v[i * nf + b * nfb + f] = t.v[j * nfb + f];
                }
            }
        }
        PcTruth { vox, ne, nf, c, v, venc, covmin }
    }

    pub fn c_at(&self, i: usize, e: usize, f: usize) -> Complex64 {
        self.c[(i * self.ne + e) * self.nf + f]
    }

    /// score `nfr` reconstruction frames (frame fr ↔ truth cardiac frame fr % nf): vel(fr, i) = the recon's velocity
    /// at scored voxel i (logical, GNL-corrected as the recon reports it), img(fr, e, i) its image value
    pub fn score(&self, nfr: usize, vel: &(dyn Fn(usize, usize) -> Vec3 + Sync), img: &(dyn Fn(usize, usize, usize) -> Complex64 + Sync)) -> PcScore {
        use rayon::prelude::*;
        let n = self.n();
        let nfr = nfr.max(1);
        let ne = self.ne;
        // per voxel: (Σ|Δ|², Σ mean², Σ detail², Σ t², Σ t-detail², Σ phase², per-frame Σ|Δ|², |Δ| list)
        let per: Vec<(f64, f64, f64, f64, f64, f64, Vec<f64>, Vec<f64>)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let vs: Vec<Vec3> = (0..nfr).map(|fr| vel(fr, i)).collect();
                let ts: Vec<Vec3> = (0..nfr).map(|fr| self.v[i * self.nf + fr % self.nf]).collect();
                let (mut e2, mut em, mut ed, mut t2, mut td, mut ph) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                let mut pf = vec![0.0; nfr];
                let mut ab = Vec::with_capacity(nfr);
                for a in 0..3 {
                    let vm = vs.iter().map(|v| v[a]).sum::<f64>() / nfr as f64;
                    let tm = ts.iter().map(|v| v[a]).sum::<f64>() / nfr as f64;
                    em += (vm - tm).powi(2);
                    for fr in 0..nfr {
                        ed += ((vs[fr][a] - vm) - (ts[fr][a] - tm)).powi(2);
                        td += (ts[fr][a] - tm).powi(2);
                        t2 += ts[fr][a].powi(2);
                    }
                }
                for fr in 0..nfr {
                    let d2: f64 = (0..3).map(|a| (vs[fr][a] - ts[fr][a]).powi(2)).sum();
                    e2 += d2;
                    pf[fr] = d2;
                    ab.push(d2.sqrt());
                    let x0 = img(fr, 0, i);
                    let c0 = self.c_at(i, 0, fr % self.nf);
                    for e in 1..ne {
                        let d = img(fr, e, i) * x0.conj() * (self.c_at(i, e, fr % self.nf) * c0.conj()).conj();
                        ph += if d.norm() > 0.0 { d.arg().powi(2) } else { (self.c_at(i, e, fr % self.nf) * c0.conj()).arg().powi(2) };
                    }
                }
                (e2, em, ed, t2, td, ph, pf, ab)
            })
            .collect();
        let nn = (n * nfr).max(1) as f64;
        let s = |k: usize| per.iter().map(|p| [p.0, p.1, p.2, p.3, p.4, p.5][k]).sum::<f64>();
        let mut ab: Vec<f64> = per.iter().flat_map(|p| p.7.iter().copied()).collect();
        ab.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pct = |p: f64| ab.get(((ab.len() as f64 - 1.0) * p).round() as usize).copied().unwrap_or(f64::NAN);
        let wrap = ab.iter().filter(|&&d| d > self.venc).count() as f64 / ab.len().max(1) as f64;
        let per_frame: Vec<f64> = (0..nfr).map(|fr| (per.iter().map(|p| p.6[fr]).sum::<f64>() / n.max(1) as f64).sqrt()).collect();
        let (e2, t2) = (s(0), s(3));
        PcScore {
            rmse: (e2 / nn).sqrt(),
            rel: (e2 / t2.max(1e-300)).sqrt(),
            mean_rmse: (s(1) / n.max(1) as f64).sqrt(),
            detail_rmse: (s(2) / nn).sqrt(),
            truth_rms: (t2 / nn).sqrt(),
            truth_detail_rms: (s(4) / nn).sqrt(),
            median: pct(0.5),
            p90: pct(0.9),
            wrap,
            venc: self.venc,
            phase_rms: (s(5) / (nn * (ne - 1).max(1) as f64)).sqrt(),
            per_frame,
        }
    }

    /// the oracle checks of the score (printed once at setup): the truth itself, its complex time average, zero
    pub fn oracles(&self, dec: &FlowDecoder, nfr: usize) -> Vec<(String, PcScore)> {
        let ne = self.ne;
        let nf = self.nf;
        let mean_c: Vec<Complex64> = (0..self.n() * ne).map(|ie| (0..nf).map(|f| self.c[ie * nf + f]).sum::<Complex64>() / nf as f64).collect();
        let mean_v: Vec<Vec3> = (0..self.n()).map(|i| dec.velocity(&(0..ne).map(|e| mean_c[i * ne + e]).collect::<Vec<_>>())).collect();
        let zero_v = dec.velocity(&vec![Complex64::new(0.0, 0.0); ne]);
        vec![
            ("truth frames".into(), self.score(nfr, &|fr, i| self.v[i * nf + fr % nf], &|fr, e, i| self.c_at(i, e, fr % nf))),
            ("time-averaged truth".into(), self.score(nfr, &|_, i| mean_v[i], &|_, e, i| mean_c[i * ne + e])),
            ("zero image".into(), self.score(nfr, &|_, _| zero_v, &|_, _, _| Complex64::new(0.0, 0.0))),
        ]
    }
}

/// inverse-crime object on the recon grid (research/experiments/recon_diag/synth_pd.py, resampled): the static
/// PD_water + PD_fat (real, box mean at the voxel's true position) in every encode, plus the box-mean blood signal of
/// the truth (missing native voxels contribute 0) with encode 0's phase removed per frame. Returns the static image
/// and the blood voxels [(q, (E, F) row-major)].
pub fn synth_object(nat: &PcNative, sd: &SimData, x_log: &[Vec3], scan: &std::path::Path) -> Result<(Vec<f32>, Vec<(usize, Vec<Complex64>)>), Error> {
    use rayon::prelude::*;
    let pw = read_npy(&scan.join("truth_pd_water.npy"))?.f32s();
    let pf = read_npy(&scan.join("truth_pd_fat.npy"))?.f32s();
    let pd: Vec<f32> = pw.iter().zip(&pf).map(|(a, b)| if (a + b).is_finite() { a + b } else { 0.0 }).collect();
    let stat: Vec<f32> = (0..x_log.len()).into_par_iter().map(|q| sd.sample_log_box(&pd, 0, &x_log[q]) as f32).collect();
    let (ne, nf) = (nat.ne, nat.nf);
    let blood: Vec<(usize, Vec<Complex64>)> = (0..x_log.len())
        .into_par_iter()
        .filter_map(|q| {
            let c = nat.box_mean(sd, &x_log[q], 0.0, false)?;
            if c.iter().all(|z| z.norm() == 0.0) {
                return None;
            }
            let mut o = vec![Complex64::new(0.0, 0.0); ne * nf];
            for f in 0..nf {
                let c0 = c[f];
                let ph = if c0.norm() > 0.0 { c0.conj() / c0.norm() } else { Complex64::new(1.0, 0.0) };
                for e in 0..ne {
                    o[e * nf + f] = c[e * nf + f] * ph;
                }
            }
            Some((q, o))
        })
        .collect();
    Ok((stat, blood))
}
