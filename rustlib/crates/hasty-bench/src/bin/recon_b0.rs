//! B0 / R2* / fat maps from a simulated multi-echo train scan (`simphantom --seq b0`), stage 1 of WP3:
//! echo images by P²nP with an LLR prior across echo times, then the voxel-wise water–fat fit (hasty-qfit, GPU).
//!
//!     recon_b0 <sim_dir> <out_dir> [--vcoils 12] [--iters 40] [--kappa 0.3] [--block 8] [--every 5] [--clean]
//!              [--no-conc] [--support 0.05] [--resp-bins 1] [--resp gate|bellows|truth] [--grid-scale 1] [--moco <motionfit run>]
//!              [--precond zmshr1m|circ|zmshr1|none] [--eps 1e-3]
//!
//! Support (LLR blocks): coil energy above `support` × its median (the whole body; true maps peak strongly at the
//! coils). Fitted voxels: support ∩ first-echo magnitude > 5 % of its 99th percentile. Preconditioning (no density
//! compensation): per-set circulant M from the set's sample density × coil energy (`CirculantPrecond`, floor `eps`),
//! with ZMSHR1 in its metric by default.
//!
//! Motion-compensated (`--moco`, with the same --resp-bins / --grid-scale as the motionfit run): one reference-pose
//! image per (shift, crossing), every bin's data through its warp x(q − r_b S(q)) (hasty-recon `moco`); the maps are
//! fitted once in the reference pose and the dynamic maps per bin are their pull-backs.
//!
//! Free breathing (`--resp-bins N`): every TR's FID at the first crossing (k ≈ 0, all coils) gives a self-gating
//! surrogate (hasty-recon `selfgate`; `--resp truth` uses the simulator's r(t) instead, an oracle); TRs are binned by
//! amplitude (equal counts, bin 0 = end-expiration) and the sets become (bin, shift, crossing) with one LLR prior over
//! all of them, so each bin gets its own B0 / R2* / fat maps (dynamic maps). `--grid-scale g` reconstructs on a g×
//! coarser grid (coordinates scaled; maps / truth resampled with a centred triangle).
//!
//! - Data: whitened with Ψ^{-1/2} (no density compensation), coil-compressed to `vcoils` virtual coils from the Gram
//!   matrix of the whitened maps (orthonormal rows: the noise stays white); true sensitivities (oracle) for now,
//!   normalised to Σ|S|² = 1.
//! - Echo sets: every sample belongs to its nearest k = 0 crossing (TE shift s, crossing e): the time-symmetric echo of
//!   the train; set time TE_{s,e}. The samples' time offsets inside a set (|τ − TE| ≤ T_lobe/2) are ignored here
//!   (stage 2, the model-based reconstruction, uses the exact τ).
//! - Concomitant field of the B0 scan itself: its phase w(t)·m(x) (m = [z², x²+y², xz, yz] about the isocentre) grows
//!   through the train like an off-resonance. It is removed from the echo images with the set's mean w at low |k|
//!   (rebuilt from the schedule), so the map is the field B0, not B0 + the scan's own Maxwell terms (`--no-conc` keeps
//!   them, for comparison).
//! - Fit: water–fat, six-peak fat, one R2*, profile-likelihood starts + multi-start LM on the GPU (f32), weights =
//!   relative inverse noise variance of each echo image (∝ its sample count).
//! Resumable (STOP file / Ctrl-C; rerun continues). Writes echoes.npy (selected echo images), maps_*.npy, truth_*.npy,
//! summary.json, and figures (tools/b0/plot_b0recon.py).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use hasty_bench::seqsched::EchoSchedule;
use hasty_cuda::{Device, C32};
use hasty_mrsim::physio::Ecg;
use hasty_mrsim::sequence::{Schedule, SAMPLE_FLOATS};
use hasty_phantom::volume::write_npy;
use hasty_qfit::waterfat::gpu::WaterFatGpu;
use hasty_qfit::waterfat::{FatSpectrum, Species, WaterFat, WfConfig};
use hasty_recon::checkpoint::{Outcome, RunCtl};
use hasty_recon::data::read_npy;
use hasty_recon::gpu::Kernels;
use hasty_recon::llr::Llr;
use hasty_recon::op::SenseOp;
use hasty_recon::p2np::{P2np, Precond, Stack};
use hasty_recon::setop::SenseSets;
use hasty_seq::multiecho::{EchoSpec, EchoTrain};
use nalgebra::{DMatrix, SymmetricEigen};
use num_complex::Complex64;

fn f32npy(path: &std::path::Path, dims: &[usize], v: &[f32]) {
    write_npy(path, "<f4", dims, true, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

/// g×-coarser grid (x fastest): centred triangle weights 1 − |o|/g, o ∈ (−g, g), around the fine index g·j (the coarse
/// voxel j sits at (j − n/2) g Δ = (g j − n/2) Δ on the fine grid)
fn down(x: &[Complex64], d: [usize; 3], g: usize) -> Vec<Complex64> {
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

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |n: &str| -> Option<String> { args.iter().position(|a| a == n).map(|i| { args.remove(i); args.remove(i) }) };
    let nvc: usize = take("--vcoils").map_or(12, |s| s.parse().unwrap());
    let iters: usize = take("--iters").map_or(40, |s| s.parse().unwrap());
    let kappa: f64 = take("--kappa").map_or(0.3, |s| s.parse().unwrap());
    let block: usize = take("--block").map_or(8, |s| s.parse().unwrap());
    let every: usize = take("--every").map_or(5, |s| s.parse().unwrap());
    let sup_thr: f64 = take("--support").map_or(0.05, |s| s.parse().unwrap());
    let nb: usize = take("--resp-bins").map_or(1, |s| s.parse().unwrap());
    let resp_mode = take("--resp").unwrap_or_else(|| "gate".into());
    let gs: usize = take("--grid-scale").map_or(1, |s| s.parse().unwrap());
    let moco_dir: Option<PathBuf> = take("--moco").map(PathBuf::from);
    let pre_name = take("--precond").unwrap_or_else(|| "zmshr1m".into());
    let eps: f64 = take("--eps").map_or(1e-3, |s| s.parse().unwrap());
    let flag = |a: &mut Vec<String>, n: &str| a.iter().position(|x| x == n).map(|i| { a.remove(i); true }).unwrap_or(false);
    let clean = flag(&mut args, "--clean");
    let no_conc = flag(&mut args, "--no-conc");
    let dir = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out).unwrap();
    let t0 = Instant::now();
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json")).unwrap()).unwrap();
    let sq = &meta["sequence"];
    assert_eq!(sq["kind"], "b0", "not a B0-train simulation");
    let dims_f: [usize; 3] = [0, 1, 2].map(|a| meta["recon_dims"][a].as_u64().unwrap() as usize);
    let res_f = meta["res"][0].as_f64().unwrap();
    let nv_f = dims_f[0] * dims_f[1] * dims_f[2];
    assert!(dims_f.iter().all(|d| d % gs == 0), "--grid-scale {gs} must divide the grid {dims_f:?}");
    let dims = dims_f.map(|d| d / gs);
    let res = res_f * gs as f64;
    let nv = dims[0] * dims[1] * dims[2];
    let (n_ro, na, nc) = (meta["n_readouts"].as_u64().unwrap() as usize, meta["n_adc"].as_u64().unwrap() as usize, meta["n_coils"].as_u64().unwrap() as usize);
    let b0 = meta["b0"].as_f64().unwrap();
    let te: Vec<Vec<f64>> = sq["te"].as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()).collect();
    let t_lobe = sq["t_lobe"].as_f64().unwrap();
    let n_echo = te[0].len();
    let n_shift = te.len();
    let nset = nb * n_shift * n_echo;

    // ── schedule rebuilt from the metadata (concomitant coefficients per sample), checked against samples.npy ──
    let sys = hasty_seq::system::System::prisma_xr();
    let mut spec = EchoSpec::new(sq["res"].as_f64().unwrap(), sq["n_lobe"].as_u64().unwrap() as usize);
    spec.slew_frac = sq["slew_frac"].as_f64().unwrap();
    spec.te_shifts = sq["te_shifts"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect();
    let train = EchoTrain::design(&spec, &sys);
    let sch = EchoSchedule::new(train, meta["scan_time"].as_f64().unwrap(), sq["dwell"].as_f64().unwrap(), b0, Ecg::new(1.0, 1.0, 0.0, 1), 1);
    assert_eq!((sch.n_readouts(), sch.n_adc()), (n_ro, na), "schedule mismatch");
    let samples = read_npy(&dir.join("samples.npy")).unwrap().f32s();
    {
        let sp = sch.sample_params(n_ro / 2..n_ro / 2 + 1);
        let o = (n_ro / 2) * na * 4;
        let err = (0..na).map(|n| (0..3).map(|q| (sp[n * SAMPLE_FLOATS + q] - samples[o + 4 * n + q]).abs()).fold(0f32, f32::max)).fold(0f32, f32::max);
        assert!(err < 1e-2, "rebuilt schedule differs from samples.npy: {err} rad/m");
    }
    let ro = read_npy(&dir.join("readouts.npy")).unwrap().f64s();

    // ── data: whiten (Ψ^{-1/2}) and compress ──
    let dname = if !clean && dir.join("data.npy").exists() { "data.npy" } else { "data_clean.npy" };
    let data = read_npy(&dir.join(dname)).unwrap().c32s();
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
    // whitened maps, Gram → compression matrix (rows = top eigenvectors^H)
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
    let mut g = DMatrix::<Complex64>::zeros(nc, nc);
    for q in (0..nv_f).step_by(7) {
        for a in 0..nc {
            for b in 0..nc {
                g[(a, b)] += sw[a * nv_f + q] * sw[b * nv_f + q].conj();
            }
        }
    }
    let ge = SymmetricEigen::new(g);
    let mut order: Vec<usize> = (0..nc).collect();
    order.sort_by(|&a, &b| ge.eigenvalues[b].partial_cmp(&ge.eigenvalues[a]).unwrap());
    let tot: f64 = ge.eigenvalues.iter().sum();
    let kept: f64 = order[..nvc].iter().map(|&i| ge.eigenvalues[i]).sum();
    let comp: Vec<Vec<Complex64>> = order[..nvc].iter().map(|&i| (0..nc).map(|c| ge.eigenvectors[(c, i)].conj()).collect()).collect();
    let mut smaps_f = vec![Complex64::new(0.0, 0.0); nvc * nv_f];
    for v in 0..nvc {
        for q in 0..nv_f {
            let mut acc = Complex64::new(0.0, 0.0);
            for c in 0..nc {
                acc += comp[v][c] * sw[c * nv_f + q];
            }
            smaps_f[v * nv_f + q] = acc;
        }
    }
    let mut smaps: Vec<C32> = (0..nvc).flat_map(|v| down(&smaps_f[v * nv_f..(v + 1) * nv_f], dims_f, gs).into_iter().map(|c| C32::new(c.re as f32, c.im as f32))).collect();
    drop(smaps_f);
    let mut sos = vec![0f64; nv];
    for v in 0..nvc {
        for q in 0..nv {
            { let s = smaps[v * nv + q]; sos[q] += (s.re as f64).powi(2) + (s.im as f64).powi(2); }
        }
    }
    let smed = {
        let mut v: Vec<f64> = sos.iter().cloned().filter(|&x| x > 0.0).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let support: Vec<bool> = sos.iter().map(|&s| s > sup_thr * smed).collect();
    // ESPIRiT-style normalisation Σ_c |S_c|² = 1 (as recon_p2np): raw Biot–Savart maps blow up next to the loops and
    // dominate the normal operator; the receive bias moves into the image magnitude (ω, R2*, fat fraction unaffected)
    let smx = sos.iter().cloned().fold(0.0, f64::max);
    for q in 0..nv {
        let sc = (1.0 / (sos[q] + 1e-6 * smx).sqrt()) as f32;
        for v in 0..nvc {
            smaps[v * nv + q].re *= sc;
            smaps[v * nv + q].im *= sc;
        }
    }
    // combined whitening + compression: y_v = Σ_c (A W)_{vc} y_c
    let aw: Vec<Vec<Complex64>> = (0..nvc).map(|v| (0..nc).map(|c| (0..nc).map(|c2| comp[v][c2] * wh[(c2, c)]).sum()).collect()).collect();
    println!("{n_ro} TRs × {na} samples × {nc} coils ({dname}); {nvc} virtual coils keep {:.2} % of the map energy ({:.0}s)", 100.0 * kept / tot, t0.elapsed().as_secs_f64());

    // ── respiratory surrogate (FID at the first crossing) and bins ──
    let kmax_rad = 2.0 * std::f64::consts::PI * sq["kmax"].as_f64().unwrap();
    let mut zfid = vec![Complex64::new(0.0, 0.0); n_ro * nc];
    for i in 0..n_ro {
        let mut cnt = 0.0f64;
        for n in 0..na {
            let r = &samples[(i * na + n) * 4..(i * na + n) * 4 + 4];
            if (r[3] as f64) > te[ro[i * 14 + 1] as usize][0] + 0.5 * t_lobe {
                break;
            }
            if (((r[0] * r[0] + r[1] * r[1] + r[2] * r[2]) as f64).sqrt()) < 0.05 * kmax_rad {
                for c in 0..nc {
                    let d = data[(i * na + n) * nc + c];
                    zfid[i * nc + c] += Complex64::new(d.re as f64, d.im as f64);
                }
                cnt += 1.0;
            }
        }
        for c in 0..nc {
            zfid[i * nc + c] /= cnt.max(1.0);
        }
    }
    let truth_r: Option<Vec<f64>> = dir.join("resp.npy").exists().then(|| read_npy(&dir.join("resp.npy")).unwrap().f64s().chunks(2).map(|c| c[0]).collect());
    let tr_s = meta["tr"].as_f64().unwrap();
    let sg = hasty_recon::selfgate::surrogate(&zfid, n_ro, nc, tr_s, 1.5, 15.0);
    let gate_corr = truth_r.as_ref().map(|t| hasty_recon::selfgate::corr(&sg.r, t));
    let r_used: Vec<f64> = match resp_mode.as_str() {
        "truth" => truth_r.clone().expect("--resp truth needs resp.npy"),
        // respiratory bellows as recorded (physio.npy column 3), scaled to [0, 1] by its 2nd / 98th percentiles
        "bellows" => {
            let ph = read_npy(&dir.join("physio.npy")).expect("--resp bellows needs physio.npy").f64s();
            let b: Vec<f64> = ph.chunks(5).map(|c| c[3]).collect();
            hasty_bench::physio::normalise(&b)
        }
        _ => sg.r.clone(),
    };
    let used_corr = truth_r.as_ref().map(|t| hasty_recon::selfgate::corr(&r_used, t));
    let bin = if nb > 1 { hasty_recon::selfgate::amplitude_bins(&r_used, nb) } else { vec![0; n_ro] };
    let bin_truth_r: Vec<f64> = (0..nb)
        .map(|b| {
            let v: Vec<f64> = (0..n_ro).filter(|&i| bin[i] == b).map(|i| truth_r.as_ref().map_or(f64::NAN, |t| t[i])).collect();
            v.iter().sum::<f64>() / v.len().max(1) as f64
        })
        .collect();
    println!("self-gating: leading component {:.0} % of the band-passed FID variance; correlation with the true breathing {}; bins ({resp_mode}) mean true r {:?}",
             100.0 * sg.explained, gate_corr.map_or("n/a".into(), |c| format!("{c:.3}")), bin_truth_r.iter().map(|x| (x * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    let mut sgt: Vec<u8> = Vec::new();
    for i in 0..n_ro {
        for x in [i as f64 * tr_s, r_used[i], truth_r.as_ref().map_or(f64::NAN, |t| t[i]), bin[i] as f64] {
            sgt.extend_from_slice(&x.to_le_bytes());
        }
    }
    write_npy(&out.join("selfgate.npy"), "<f8", &[n_ro, 4], false, &sgt).unwrap();

    // ── echo sets ──
    let mut set_of = vec![0u32; n_ro * na];
    let mut ms = vec![0usize; nset];
    for i in 0..n_ro {
        let s = ro[i * 14 + 1] as usize;
        for n in 0..na {
            let tau = samples[(i * na + n) * 4 + 3] as f64;
            // samples beyond the (possibly coarser) grid's band |k_a Δ| ≤ π would wrap in the NUFFT: dropped
            let kq = &samples[(i * na + n) * 4..(i * na + n) * 4 + 3];
            if kq.iter().any(|&k| (k as f64 * res).abs() > std::f64::consts::PI) {
                set_of[i * na + n] = u32::MAX;
                continue;
            }
            let e = (((tau - te[s][0]) / t_lobe).round().max(0.0) as usize).min(n_echo - 1);
            let t = (bin[i] * n_shift + s) * n_echo + e;
            set_of[i * na + n] = t as u32;
            ms[t] += 1;
        }
    }
    let mut coords: Vec<[Vec<f32>; 3]> = ms.iter().map(|&m| [Vec::with_capacity(m), Vec::with_capacity(m), Vec::with_capacity(m)]).collect();
    let mut ys: Vec<Vec<C32>> = ms.iter().map(|&m| vec![C32::new(0.0, 0.0); nvc * m]).collect();
    let mut fill = vec![0usize; nset];
    // mean concomitant coefficients per set over its low-|k| samples
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
                let m = ms[t];
                let col = fill[t];
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
    drop(data);
    let wbar: Vec<[f64; 4]> = (0..nset).map(|t| wsum[t].map(|x| x / wcnt[t].max(1.0))).collect();
    let set_te: Vec<f64> = (0..nset).map(|t| te[(t / n_echo) % n_shift][t % n_echo]).collect();
    println!("{nset} echo sets ({nb} breathing bins × {n_shift} TE shifts × {n_echo} crossings), {}–{} samples each ({:.0}s)", ms.iter().min().unwrap(), ms.iter().max().unwrap(), t0.elapsed().as_secs_f64());

    // ── P²nP with LLR across the echo images ──
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let mut op = SenseOp::new(&dev, &st, kern.clone(), dims, [&coords[0][0], &coords[0][1], &coords[0][2]], &smaps, nvc, vec![], 1e-4).unwrap();
    // motion-compensated: reference sets (shift, crossing) with one block per bin
    let nd = n_shift * n_echo;
    let (ms_ref, moco_r): (Vec<Vec<usize>>, Vec<f64>) = match &moco_dir {
        Some(md) => {
            let mj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(md.join("motion.json")).unwrap()).unwrap();
            let rr: Vec<f64> = mj["per_bin"].as_array().unwrap().iter().map(|b| b["r_rel"].as_f64().unwrap()).collect();
            assert_eq!(rr.len(), nb, "--moco needs the motion run's number of bins ({})", rr.len());
            assert_eq!(mj["dims"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect::<Vec<_>>(), dims.to_vec(), "--moco grid mismatch");
            ((0..nd).map(|d| (0..nb).map(|b| ms[b * nd + d]).collect()).collect(), rr)
        }
        None => (vec![], vec![]),
    };
    let need_circ = pre_name == "circ" || pre_name == "zmshr1m";
    let circ: Vec<Vec<hasty_recon::precond::CirculantPrecond>> = if need_circ {
        (0..if moco_dir.is_some() { nd } else { nset })
            .map(|t| {
                let cat: [Vec<f32>; 3] = if moco_dir.is_some() {
                    [0, 1, 2].map(|a| (0..nb).flat_map(|b| coords[b * nd + t][a].iter().copied()).collect())
                } else {
                    [0, 1, 2].map(|a| coords[t][a].clone())
                };
                vec![hasty_recon::precond::CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&cat[0], &cat[1], &cat[2]], &smaps, nvc, eps).unwrap()]
            })
            .collect()
    } else {
        vec![]
    };
    if need_circ {
        println!("circulant preconditioners: {} (eps {eps}) ({:.0}s)", circ.len(), t0.elapsed().as_secs_f64());
    }
    let mut sets_plain;
    let mut sets_moco;
    let nset_x = if moco_dir.is_some() { nd } else { nset };
    let dys: Vec<cudarc::driver::CudaSlice<C32>>;
    let sop: &mut dyn hasty_recon::setop::SetOp = if let Some(md) = &moco_dir {
        let smm = read_npy(&md.join("motion_S_mm.npy")).unwrap().f32s();
        let disp: Vec<cudarc::driver::CudaSlice<f32>> = moco_r.iter().map(|&r| st.clone_htod(&smm.iter().map(|x| (r * *x as f64 / (res * 1e3)) as f32).collect::<Vec<f32>>()).unwrap()).collect();
        let mut rc: Vec<Vec<[cudarc::driver::CudaSlice<f32>; 3]>> = Vec::with_capacity(nd);
        let mut ry: Vec<cudarc::driver::CudaSlice<C32>> = Vec::with_capacity(nd);
        for d in 0..nd {
            rc.push((0..nb).map(|b| [0, 1, 2].map(|a| st.clone_htod(&coords[b * nd + d][a]).unwrap())).collect());
            let cat: Vec<C32> = (0..nb).flat_map(|b| ys[b * nd + d].iter().copied()).collect();
            ry.push(st.clone_htod(&cat).unwrap());
        }
        drop(coords);
        drop(ys);
        dys = ry;
        sets_moco = hasty_recon::moco::MocoSets::new(&st, &dev, &mut op, rc, ms_ref.clone(), disp).unwrap();
        &mut sets_moco
    } else {
        let dcoords: Vec<[cudarc::driver::CudaSlice<f32>; 3]> = coords.iter().map(|c| [0, 1, 2].map(|a| st.clone_htod(&c[a]).unwrap())).collect();
        drop(coords);
        dys = ys.iter().map(|y| st.clone_htod(y).unwrap()).collect();
        drop(ys);
        sets_plain = SenseSets { op: &mut op, coords: dcoords, ms: ms.clone(), maps: vec![], map_of_set: vec![] };
        &mut sets_plain
    };
    let mut p = P2np::new(sop, dys.iter().collect(), circ, &st).unwrap();
    let lmax = p.step_norm(&st, false, 12).unwrap();
    let lmax_c = if need_circ { p.step_norm(&st, true, 12).unwrap() } else { 1.0 };
    let (precond, alpha) = match pre_name.as_str() {
        "none" => (Precond::None, 1.0 / lmax),
        "circ" => (Precond::Circulant, 1.0 / lmax_c),
        "zmshr1" => (Precond::Zmshr1, 1.0 / lmax),
        _ => (Precond::Zmshr1M, 1.0 / lmax_c),
    };
    println!("λ_max(N) = {lmax:.3e}, λ_max(M N) = {lmax_c:.3e}, preconditioner {pre_name} ({:.0}s)", t0.elapsed().as_secs_f64());
    let ctl = RunCtl::new(&out, every);
    let tag = format!("p2np_{pre_name}_k{kappa}{}", if moco_dir.is_some() { "_moco" } else { "" });
    let mut x = Stack::zeros(&st, nset_x, nv).unwrap();
    let mut llr = Llr::new(dims, block, kappa, support.clone());
    let mut denoise = |k: usize, h: &mut Vec<C32>| {
        if kappa > 0.0 {
            llr.set_iteration(k as u64);
            llr.apply(h, nset_x);
        }
    };
    let tr = Instant::now();
    let mut report = |k: usize, _x: &Stack, f: f64| {
        let k = k.wrapping_add(1);
        if k % every == 0 || k == iters {
            println!("  iter {k:3}: f {f:.5e} ({:.1}s)", tr.elapsed().as_secs_f64());
        }
    };
    if let Outcome::Stopped(k) = p.run_from(&st, &mut x, precond, alpha, iters, &mut denoise, &mut report, Some((&ctl, &tag))).unwrap() {
        println!("stopped before iteration {k}; rerun to continue");
        return;
    }
    let imgs: Vec<Vec<C32>> = x.v.iter().map(|v| st.clone_dtoh(v).unwrap()).collect();
    println!("echo images done ({:.0}s)", t0.elapsed().as_secs_f64());

    // ── concomitant correction and the voxel fit ──
    let pos = |q: usize| -> [f64; 3] {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        [(i as f64 - (dims[0] / 2) as f64) * res, (j as f64 - (dims[1] / 2) as f64) * res, (k as f64 - (dims[2] / 2) as f64) * res]
    };
    let e0mag: Vec<f32> = (0..nv).map(|q| (imgs[0][q].re.powi(2) + imgs[0][q].im.powi(2)).sqrt()).collect();
    let e0_99 = {
        let mut v: Vec<f32> = (0..nv).filter(|&q| support[q]).map(|q| e0mag[q]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(0.99 * (v.len() - 1) as f64) as usize]
    };
    let vox: Vec<usize> = (0..nv).filter(|&q| support[q] && e0mag[q] > 0.05 * e0_99).collect();
    let nvx = vox.len();
    let nb_fit = if moco_dir.is_some() { 1 } else { nb };
    let nprob = nb_fit * nvx;
    // concomitant coefficients of the fitted images (moco: the bins' mean)
    let wfit: Vec<[f64; 4]> = if moco_dir.is_some() {
        (0..nd).map(|d| { let c: f64 = (0..nb).map(|b| wcnt[b * nd + d]).sum(); [0, 1, 2, 3].map(|a| (0..nb).map(|b| wsum[b * nd + d][a]).sum::<f64>() / c.max(1.0)) }).collect()
    } else {
        wbar.clone()
    };
    let mut y = vec![0.0f64; 2 * nd * nprob];
    for b in 0..nb_fit {
        for (vi, &q) in vox.iter().enumerate() {
            let x = pos(q);
            let mono = [x[2] * x[2], x[0] * x[0] + x[1] * x[1], x[0] * x[2], x[1] * x[2]];
            let pi = b * nvx + vi;
            for d in 0..nd {
                let t = b * nd + d;
                let ph = if no_conc { 0.0 } else { (0..4).map(|a| wfit[t][a] * mono[a]).sum::<f64>() };
                let z = Complex64::new(imgs[t][q].re as f64, imgs[t][q].im as f64) * Complex64::from_polar(1.0, ph);
                y[d * nprob + pi] = z.re;
                y[(nd + d) * nprob + pi] = z.im;
            }
        }
    }
    let wf = WaterFat::new(Species::WaterFat, set_te[..nd].to_vec(), b0, FatSpectrum::six_peak());
    let cfg = WfConfig { r2_max: 1000.0, grid_r2: vec![0.0, 15.0, 30.0, 60.0, 120.0, 250.0, 500.0, 1000.0], ..Default::default() };
    let gpu = WaterFatGpu::<f32>::new(&dev, &wf, &cfg).unwrap();
    let fit = gpu.fit(&y, nprob).unwrap();
    println!("voxel fits: {nvx} voxels × {nb_fit} bins ({:.0}s)", t0.elapsed().as_secs_f64());
    let par = |j: usize, pi: usize| fit.params[j * nprob + pi];
    // maps [bin][voxel] (bin-major), bin 0 = end-expiration
    let mut om_hz = vec![f32::NAN; nb_fit * nv];
    let mut r2 = vec![f32::NAN; nb_fit * nv];
    let mut ff = vec![f32::NAN; nb_fit * nv];
    let mut wm = vec![0f32; nb_fit * nv];
    let mut fm = vec![0f32; nb_fit * nv];
    for b in 0..nb_fit {
        for (vi, &q) in vox.iter().enumerate() {
            let pi = b * nvx + vi;
            let (w, f) = ((par(0, pi).powi(2) + par(1, pi).powi(2)).sqrt(), (par(2, pi).powi(2) + par(3, pi).powi(2)).sqrt());
            let o = b * nv + q;
            om_hz[o] = (par(5, pi) / (2.0 * std::f64::consts::PI)) as f32;
            r2[o] = par(4, pi) as f32;
            ff[o] = (f / (w + f).max(1e-30)) as f32;
            wm[o] = w as f32;
            fm[o] = f as f32;
        }
    }
    // moco: dynamic maps = the reference maps pulled back along r_b S (bin 0 = reference pose)
    if let Some(md) = &moco_dir {
        let smm = read_npy(&md.join("motion_S_mm.npy")).unwrap().f32s();
        let sv: Vec<f64> = smm.iter().map(|x| *x as f64 / (res * 1e3)).collect();
        for m in [&mut om_hz, &mut r2, &mut ff, &mut wm, &mut fm] {
            let reference = m.clone();
            let mut all = Vec::with_capacity(nb * nv);
            for &r in &moco_r {
                all.extend(hasty_recon::motion::warp(&reference, dims, &sv, r));
            }
            *m = all;
        }
    }
    // truth: field (ppm) on the recon grid → Hz
    let fld = read_npy(&dir.join("field_ppm_recon.npy")).unwrap().f32s();
    let gbar = hasty_mrsim::GAMMA_BAR;
    let fldc: Vec<Complex64> = fld.iter().map(|&p| Complex64::new(gbar * b0 * p as f64 * 1e-6, 0.0)).collect();
    let om_true: Vec<f32> = down(&fldc, dims_f, gs).iter().map(|c| c.re as f32).collect();
    // error in tissue with signal (water or fat magnitude above 10 % of the 99th percentile)
    let mut mag: Vec<f32> = vox.iter().map(|&q| wm[q] + fm[q]).collect();
    mag.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m99 = mag[(0.99 * (mag.len() - 1) as f64) as usize];
    let sel: Vec<usize> = vox.iter().copied().filter(|&q| wm[q] + fm[q] > 0.1 * m99).collect();
    let mut d: Vec<f64> = sel.iter().map(|&q| (om_hz[q] - om_true[q]) as f64).collect();
    let mean = d.iter().sum::<f64>() / d.len() as f64;
    let rmse = (d.iter().map(|x| x * x).sum::<f64>() / d.len() as f64).sqrt();
    d.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = d[d.len() / 2];
    let p95 = d.iter().map(|x| x.abs()).fold(Vec::new(), |mut v, x| { v.push(x); v });
    let mut p95 = p95;
    p95.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (a95, a99) = (p95[(0.95 * (p95.len() - 1) as f64) as usize], p95[(0.99 * (p95.len() - 1) as f64) as usize]);
    let swaps = sel.iter().filter(|&&q| (om_hz[q] - om_true[q]).abs() > 150.0).count();
    println!("ω − truth over {} signal voxels: mean {mean:.2} Hz, median {med:.2}, RMSE {rmse:.2}, |·| p95 {a95:.2} / p99 {a99:.2}; |Δ| > 150 Hz (swap / alias): {swaps}", sel.len());

    let dims_b = [dims[0], dims[1], dims[2], nb];
    for (name, v) in [("maps_omega_hz", &om_hz), ("maps_r2", &r2), ("maps_ff", &ff), ("maps_water", &wm), ("maps_fat", &fm)] {
        f32npy(&out.join(format!("{name}.npy")), &dims_b, v);
    }
    f32npy(&out.join("truth_omega_hz.npy"), &dims, &om_true);
    f32npy(&out.join("support.npy"), &dims, &support.iter().map(|&b| b as u8 as f32).collect::<Vec<_>>());
    // echo images of shift 0 (magnitude and phase, conc-corrected), [nv][n_echo] fortran
    let mut eimg = vec![0f32; 2 * nv * n_echo];
    for e in 0..n_echo {
        for q in 0..nv {
            let x = pos(q);
            let mono = [x[2] * x[2], x[0] * x[0] + x[1] * x[1], x[0] * x[2], x[1] * x[2]];
            let ph = if no_conc { 0.0 } else { (0..4).map(|a| wbar[e][a] * mono[a]).sum::<f64>() };
            let z = Complex64::new(imgs[e][q].re as f64, imgs[e][q].im as f64) * Complex64::from_polar(1.0, ph);
            eimg[q + nv * (2 * e)] = z.norm() as f32;
            eimg[q + nv * (2 * e + 1)] = z.arg() as f32;
        }
    }
    f32npy(&out.join("echoes_mag_phase.npy"), &[dims[0], dims[1], dims[2], 2 * n_echo], &eimg);
    // concomitant pseudo off-resonance at the slab corners (slope of w̄·m over TE, shift 0)
    let corner = [0.5 * dims[0] as f64 * res, 0.0, 0.5 * dims[2] as f64 * res];
    let mono_c = [corner[2] * corner[2], corner[0] * corner[0] + corner[1] * corner[1], corner[0] * corner[2], corner[1] * corner[2]];
    let phc: Vec<f64> = (0..n_echo).map(|e| (0..4).map(|a| wbar[e][a] * mono_c[a]).sum()).collect();
    let slope_hz = (phc[n_echo - 1] - phc[0]) / (set_te[n_echo - 1] - set_te[0]) / (2.0 * std::f64::consts::PI);
    println!("the scan's own concomitant field at the FOV edge (x = {:.2} m, z = {:.2} m): {slope_hz:.2} Hz equivalent", corner[0], corner[2]);
    let summary = serde_json::json!({
        "sim_dir": dir, "data": dname, "vcoils": nvc, "map_energy_kept": kept / tot, "iters": iters, "kappa": kappa, "block": block, "lambda_max": lmax,
        "n_sets": nset, "te": set_te, "set_samples": ms, "conc_corrected": !no_conc, "wbar": wbar, "conc_edge_hz": slope_hz,
        "omega_err_hz": {"n": sel.len(), "mean": mean, "median": med, "rmse": rmse, "abs_p95": a95, "abs_p99": a99, "gt150": swaps},
        "fit": {"species": "WaterFat", "fat": "six_peak", "r2_max": cfg.r2_max, "om_max_hz": cfg.om_max / (2.0 * std::f64::consts::PI), "n_start": cfg.n_start},
        "dims": dims, "res": res, "grid_scale": gs, "b0": b0, "seconds": t0.elapsed().as_secs_f64(),
        "moco": moco_dir, "resp": {"bins": nb, "mode": resp_mode, "gate_explained": sg.explained, "gate_corr_truth": gate_corr, "used_corr_truth": used_corr, "bin_mean_true_r": bin_truth_r},
    });
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    let tools = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/b0/plot_b0recon.py");
    let py = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
    match std::process::Command::new(&py).arg("-I").arg(&tools).arg(&out).status() {
        Ok(s) if s.success() => println!("figures: {}", out.join("figs").display()),
        r => println!("figure script failed ({r:?}); run: {} -I {} {}", py.display(), tools.display(), out.display()),
    }
    println!("done in {:.0}s → {}", t0.elapsed().as_secs_f64(), out.display());
}
