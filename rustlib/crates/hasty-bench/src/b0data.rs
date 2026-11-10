//! Loading a simulated B0-train scan (`simphantom --seq b0`) for reconstruction: whitening (Ψ^{-1/2}), coil
//! compression from the Gram matrix of the whitened maps (orthonormal rows: the noise stays white), Σ|S|² = 1 map
//! normalisation, echo windows (each sample to its nearest k = 0 crossing; window-constant time TE for now), the
//! mean concomitant coefficients per window, an optional coarser grid (samples beyond its band dropped) and the
//! reference-pose truth maps. Shared by `recon_b0` (per-echo images) and `mbq_b0` (model-based).

use std::path::Path;

use hasty_cuda::C32;
use hasty_mrsim::physio::Ecg;
use hasty_mrsim::sequence::{Schedule, SAMPLE_FLOATS};
use hasty_recon::data::read_npy;
use hasty_seq::multiecho::{EchoSpec, EchoTrain};
use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64;

use crate::seqsched::EchoSchedule;

pub struct B0Data {
    pub dims: [usize; 3],
    /// voxel size (m)
    pub res: f64,
    pub nv: usize,
    pub nvc: usize,
    pub b0: f64,
    pub n_shift: usize,
    pub n_echo: usize,
    /// readout groups (motion states; 1 = all readouts together); set t = group · n_win + window
    pub n_group: usize,
    pub n_win: usize,
    /// per window t = s · n_echo + e: coordinates k·Δ (rad/voxel), whitened compressed data (coil-major), TE
    pub coords: Vec<[Vec<f32>; 3]>,
    pub ys: Vec<Vec<C32>>,
    pub te: Vec<f64>,
    /// per window: time offset of each sample from the window's TE, δ = τ − TE (s)
    pub dts: Vec<Vec<f32>>,
    /// half window (s): |δ| ≤ t_lobe / 2
    pub half_window: f64,
    /// mean concomitant coefficients of [z², x²+y², xz, yz] (rad/m²) over the window's low-|k| samples
    pub wbar: Vec<[f64; 4]>,
    /// normalised compressed maps (nvc × nv)
    pub smaps: Vec<C32>,
    /// coil energy of the un-normalised maps (support / receive bias)
    pub sos: Vec<f64>,
    pub support: Vec<bool>,
    /// truth on this grid (NaN where not available): ω (Hz), R2* (1/s), fat fraction, water / fat PD
    pub truth_om_hz: Vec<f32>,
    pub truth_r2: Vec<f32>,
    pub truth_ff: Vec<f32>,
    pub meta: serde_json::Value,
}

/// g×-coarser grid (x fastest): centred triangle weights around the fine index g·j
pub fn down(x: &[Complex64], d: [usize; 3], g: usize) -> Vec<Complex64> {
    if g == 1 {
        return x.to_vec();
    }
    let mut cur = x.to_vec();
    let mut cd = d;
    for a in 0..3 {
        let nd = cd[a] / g;
        let mut od = cd;
        od[a] = nd;
        let mut o = vec![Complex64::new(0.0, 0.0); od[0] * od[1] * od[2]];
        let st = [1, cd[0], cd[0] * cd[1]];
        let sto = [1, od[0], od[0] * od[1]];
        for k in 0..od[2] {
            for j in 0..od[1] {
                for i in 0..od[0] {
                    let idx = [i, j, k];
                    let mut base = 0;
                    for b in 0..3 {
                        if b != a {
                            base += idx[b] * st[b];
                        }
                    }
                    let (mut acc, mut ws) = (Complex64::new(0.0, 0.0), 0.0);
                    for off in -(g as isize - 1)..=(g as isize - 1) {
                        let f = (g * idx[a]) as isize + off;
                        if f < 0 || f >= cd[a] as isize {
                            continue;
                        }
                        let w = 1.0 - off.unsigned_abs() as f64 / g as f64;
                        acc += cur[base + f as usize * st[a]] * w;
                        ws += w;
                    }
                    o[i * sto[0] + j * sto[1] + k * sto[2]] = acc / ws;
                }
            }
        }
        cur = o;
        cd = od;
    }
    cur
}

fn down_f32(x: &[f32], d: [usize; 3], g: usize) -> Vec<f32> {
    let c: Vec<Complex64> = x.iter().map(|&v| Complex64::new(if v.is_finite() { v as f64 } else { f64::NAN }, 0.0)).collect();
    down(&c, d, g).iter().map(|c| c.re as f32).collect()
}

pub struct LoadOpts {
    /// virtual coils (used when `coil_energy` is None)
    pub nvc: usize,
    /// Some(e): the fewest virtual coils holding a fraction e of the whitened map energy (`hasty_recon::coils`)
    pub coil_energy: Option<f64>,
    pub grid_scale: usize,
    pub support_rel: f64,
    pub clean: bool,
    /// optional group index per readout (TR)
    pub group: Option<Vec<usize>>,
}

pub fn load(dir: &Path, o: &LoadOpts) -> B0Data {
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let sq = &meta["sequence"];
    assert_eq!(sq["kind"], "b0", "not a B0-train simulation");
    let gs = o.grid_scale;
    let dims_f: [usize; 3] = [0, 1, 2].map(|a| meta["recon_dims"][a].as_u64().unwrap() as usize);
    let res_f = meta["res"][0].as_f64().unwrap();
    let nv_f = dims_f[0] * dims_f[1] * dims_f[2];
    assert!(dims_f.iter().all(|d| d % gs == 0), "grid scale {gs} must divide {dims_f:?}");
    let dims = dims_f.map(|d| d / gs);
    let res = res_f * gs as f64;
    let nv = dims[0] * dims[1] * dims[2];
    let (n_ro, na, nc) = (meta["n_readouts"].as_u64().unwrap() as usize, meta["n_adc"].as_u64().unwrap() as usize, meta["n_coils"].as_u64().unwrap() as usize);
    let b0 = meta["b0"].as_f64().unwrap();
    let te_s: Vec<Vec<f64>> = sq["te"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()).collect();
    let t_lobe = sq["t_lobe"].as_f64().unwrap();
    let (n_shift, n_echo) = (te_s.len(), te_s[0].len());
    let n_win = n_shift * n_echo;
    let n_group = o.group.as_ref().map_or(1, |g| g.iter().max().unwrap() + 1);
    let nset = n_group * n_win;
    // schedule (concomitant coefficients per sample)
    let sys = hasty_seq::system::System::prisma_xr();
    let mut spec = EchoSpec::new(sq["res"].as_f64().unwrap(), sq["n_lobe"].as_u64().unwrap() as usize);
    spec.slew_frac = sq["slew_frac"].as_f64().unwrap();
    spec.te_shifts = sq["te_shifts"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    let train = EchoTrain::design(&spec, &sys);
    let sch = EchoSchedule::new(train, meta["scan_time"].as_f64().unwrap(), sq["dwell"].as_f64().unwrap(), b0, Ecg::new(1.0, 1.0, 0.0, 1), 1);
    assert_eq!((sch.n_readouts(), sch.n_adc()), (n_ro, na), "schedule mismatch");
    let samples = read_npy(&dir.join("samples.npy")).unwrap().f32s();
    let ro = read_npy(&dir.join("readouts.npy")).unwrap().f64s();
    let dname = if !o.clean && dir.join("data.npy").exists() { "data.npy" } else { "data_clean.npy" };
    let data = read_npy(&dir.join(dname)).unwrap().c32s();
    // whitening
    let psi = DMatrix::<f64>::from_fn(nc, nc, |i, j| meta["psi"][i][j].as_f64().unwrap());
    let e = SymmetricEigen::new(psi);
    let mut wh = DMatrix::<f64>::zeros(nc, nc);
    for q in 0..nc {
        let s = 1.0 / e.eigenvalues[q].max(1e-12).sqrt();
        for i in 0..nc {
            for j in 0..nc {
                wh[(i, j)] += e.eigenvectors[(i, q)] * e.eigenvectors[(j, q)] * s;
            }
        }
    }
    let smaps0 = read_npy(&dir.join("smaps_true.npy")).unwrap().c32s();
    let mut sw = vec![Complex64::new(0.0, 0.0); nc * nv_f];
    for q in 0..nv_f {
        for c in 0..nc {
            let mut acc = Complex64::new(0.0, 0.0);
            for c2 in 0..nc {
                let s = smaps0[c2 * nv_f + q];
                acc += wh[(c, c2)] * Complex64::new(s.re as f64, s.im as f64);
            }
            sw[c * nv_f + q] = acc;
        }
    }
    let rule = match o.coil_energy { Some(e) => hasty_recon::coils::CoilRule::Energy(e), None => hasty_recon::coils::CoilRule::Fixed(o.nvc) };
    let mut gram = hasty_recon::coils::CoilGram::new(nc);
    gram.add_c64(&sw, nv_f, 7);
    let ccx = hasty_recon::coils::CoilCompression::from_gram(&gram, rule);
    println!("coil compression (whitened maps): {}", ccx.report());
    let nvc = ccx.nvc();
    let comp: Vec<Vec<Complex64>> = ccx.rows.clone();
    let mut smaps = Vec::with_capacity(nvc * nv);
    for v in 0..nvc {
        let mut f = vec![Complex64::new(0.0, 0.0); nv_f];
        for q in 0..nv_f {
            for c in 0..nc {
                f[q] += comp[v][c] * sw[c * nv_f + q];
            }
        }
        smaps.extend(down(&f, dims_f, gs).into_iter().map(|c| C32::new(c.re as f32, c.im as f32)));
    }
    drop(sw);
    let mut sos = vec![0f64; nv];
    for v in 0..nvc {
        for q in 0..nv {
            let s = smaps[v * nv + q];
            sos[q] += (s.re as f64).powi(2) + (s.im as f64).powi(2);
        }
    }
    let smed = {
        let mut v: Vec<f64> = sos.iter().cloned().filter(|&x| x > 0.0).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let support: Vec<bool> = sos.iter().map(|&s| s > o.support_rel * smed).collect();
    let smx = sos.iter().cloned().fold(0.0, f64::max);
    for q in 0..nv {
        let sc = (1.0 / (sos[q] + 1e-6 * smx).sqrt()) as f32;
        for v in 0..nvc {
            smaps[v * nv + q].re *= sc;
            smaps[v * nv + q].im *= sc;
        }
    }
    let aw: Vec<Vec<Complex64>> = (0..nvc).map(|v| (0..nc).map(|c| (0..nc).map(|c2| comp[v][c2] * wh[(c2, c)]).sum()).collect()).collect();
    // windows
    let mut set_of = vec![u32::MAX; n_ro * na];
    let mut ms = vec![0usize; nset];
    for i in 0..n_ro {
        let s = ro[i * 14 + 1] as usize;
        for n in 0..na {
            let r = &samples[(i * na + n) * 4..(i * na + n) * 4 + 4];
            if r[..3].iter().any(|&k| (k as f64 * res).abs() > std::f64::consts::PI) {
                continue;
            }
            let ee = ((((r[3] as f64) - te_s[s][0]) / t_lobe).round().max(0.0) as usize).min(n_echo - 1);
            let t = o.group.as_ref().map_or(0, |g| g[i]) * n_win + s * n_echo + ee;
            set_of[i * na + n] = t as u32;
            ms[t] += 1;
        }
    }
    let mut coords: Vec<[Vec<f32>; 3]> = ms.iter().map(|&m| [Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m)]).collect();
    let mut ys: Vec<Vec<C32>> = ms.iter().map(|&m| vec![C32::new(0.0, 0.0); nvc * m]).collect();
    let mut dts: Vec<Vec<f32>> = ms.iter().map(|&m| Vec::with_capacity(m)).collect();
    let mut fill = vec![0usize; nset];
    let kmax_rad = 2.0 * std::f64::consts::PI * sq["kmax"].as_f64().unwrap();
    let mut wsum = vec![[0f64; 4]; nset];
    let mut wcnt = vec![0f64; nset];
    for c0 in (0..n_ro).step_by(64) {
        let c1 = (c0 + 64).min(n_ro);
        let sp = sch.sample_params(c0..c1);
        for i in c0..c1 {
            for n in 0..na {
                let j = i * na + n;
                if set_of[j] == u32::MAX {
                    continue;
                }
                let t = set_of[j] as usize;
                let r = &samples[j * 4..j * 4 + 4];
                for a in 0..3 {
                    coords[t][a].push(r[a] * res as f32);
                }
                dts[t].push((r[3] as f64 - te_s[(t % n_win) / n_echo][t % n_echo]) as f32);
                let (m, col) = (ms[t], fill[t]);
                let src = &data[j * nc..(j + 1) * nc];
                for v in 0..nvc {
                    let mut acc = Complex64::new(0.0, 0.0);
                    for c in 0..nc {
                        acc += aw[v][c] * Complex64::new(src[c].re as f64, src[c].im as f64);
                    }
                    ys[t][v * m + col] = C32::new(acc.re as f32, acc.im as f32);
                }
                fill[t] += 1;
                let kr = ((r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) as f64).sqrt();
                if kr < 0.2 * kmax_rad {
                    let s = &sp[((i - c0) * na + n) * SAMPLE_FLOATS..];
                    for q in 0..4 {
                        wsum[t][q] += s[9 + q] as f64;
                    }
                    wcnt[t] += 1.0;
                }
            }
        }
    }
    let wbar = (0..nset).map(|t| wsum[t].map(|x| x / wcnt[t].max(1.0))).collect();
    let te = (0..nset).map(|t| te_s[(t % n_win) / n_echo][t % n_echo]).collect();
    // truth
    let gbar = hasty_mrsim::GAMMA_BAR;
    let rd = |name: &str| -> Vec<f32> { if dir.join(name).exists() { down_f32(&read_npy(&dir.join(name)).unwrap().f32s(), dims_f, gs) } else { vec![f32::NAN; nv] } };
    let truth_om_hz = rd("field_ppm_recon.npy").iter().map(|&p| (gbar * b0 * p as f64 * 1e-6) as f32).collect();
    B0Data {
        dims,
        res,
        nv,
        nvc,
        b0,
        n_shift,
        n_echo,
        n_group,
        n_win,
        coords,
        ys,
        te,
        dts,
        half_window: 0.5 * t_lobe,
        wbar,
        smaps,
        sos,
        support,
        truth_om_hz,
        truth_r2: rd("truth_r2s.npy"),
        truth_ff: rd("truth_ff.npy"),
        meta,
    }
}

impl B0Data {
    /// position of voxel q relative to the isocentre (m)
    pub fn pos(&self, q: usize) -> [f64; 3] {
        let d = self.dims;
        let (i, j, k) = (q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1]));
        [(i as f64 - (d[0] / 2) as f64) * self.res, (j as f64 - (d[1] / 2) as f64) * self.res, (k as f64 - (d[2] / 2) as f64) * self.res]
    }
    /// concomitant phase of window t at voxel q (rad; the data carry e^{−i·this})
    pub fn conc(&self, t: usize, q: usize) -> f64 {
        let x = self.pos(q);
        let m = [x[2] * x[2], x[0] * x[0] + x[1] * x[1], x[0] * x[2], x[1] * x[2]];
        (0..4).map(|a| self.wbar[t][a] * m[a]).sum()
    }
    pub fn nset(&self) -> usize {
        self.coords.len()
    }
}
