//! Model-based B0 / R2* / fat from a simulated multi-echo train scan, stage S1 of research/b0_r2s_model_based.md:
//! the unknowns are tissue properties shared by all echoes, not echo images.
//!
//!     mbq_b0 <sim_dir> <out_dir> --init <recon_b0 run> [--grid-scale 1] [--vcoils 12] [--knot-mm 20] [--iters 150]
//!            [--rho-iters 40] [--lam-rho 2e-3] [--lam-r2 2e-3] [--every 10] [--clean] [--terms 1]
//!
//! `--moco <motionfit run> [--resp bellows|truth] [--levels 16]` (S4, respiratory): every readout's breathing state
//! r(t) (the recorded bellows, scaled to [0, 1]; or the simulator's truth) is quantised into `levels` equal-count
//! groups; group g sees the reference-pose images through x(q − (r̄_g − r̄_ref) S(q)) with S the motion run's shared
//! field (hasty-recon `moco::MocoTermSets`), so all readouts inform one map set in the reference pose.
//! `--precond none|jacobi|circ` (default jacobi): L-BFGS initial inverse Hessian per block (ρ, R2*, spline), each
//! scaled by its own secant γ: the Gauss–Newton diagonal (sample counts × |∂x/∂θ|² summed over windows, + the prior
//! curvature); `circ` replaces the ρ blocks by D^{-1/2} (m M) D^{-1/2} with M the circulant approximation of
//! (EᴴE)⁻¹ over all samples (hasty-recon `CirculantPrecond`) — the k-space density part the diagonal cannot see.
//! `--terms L` (S2): exact sample times inside each echo window: the window's image is the model at L nodes
//! TE + δ_l, combined per sample with band-limited interpolators a_l(δ_n) (hasty-recon `multiterm`, design band
//! |f| ≤ 1050 Hz, R2* ≤ 500 s⁻¹); L = 1 is the window-constant model.
//!
//! Model (window-constant times, S1): echo window t (TE shift s, crossing e) sees the image
//!   x_t(q) = (ρ_w(q) + ρ_f(q) c(TE_t)) e^{−(R2*(q) + iω(q)) TE_t} e^{−i w̄_t·m(q)},   ω = B c  (cubic B-splines),
//! data term ½ Σ_t ‖E_t x_t − y_t‖² (whitened, compressed, Σ|S|² = 1 maps; E_t the window's SENSE-NUFFT), priors
//! Huber-TV on (ρ_w, ρ_f) jointly and on R2*, R2* = s · softplus(u) ≥ 0 (a reparametrisation: same MLE inside).
//! Optimiser: L-BFGS in scaled variables (amplitudes / their median, R2* / 100 s⁻¹, spline coefficients / 2π·50 Hz);
//! phase 1 solves for ρ with R2*, ω from the initialiser fixed (the linear part, VarPro-style warm start), phase 2
//! all jointly. Gradient: Σ_t (∂x_t/∂θ)ᴴ E_tᴴ r_t, the image-domain chain rule is voxel-wise (host, threaded).
//! Initialiser: R2* and ω of a per-echo `recon_b0` run (any grid scale; resampled), ρ from phase 1.
//! Resumable (STOP file / Ctrl-C: the scaled variables and the iteration are saved; L-BFGS memory restarts).
//! Writes maps_*.npy, truth_*.npy, metrics.csv, summary.json; figures tools/b0/plot_mbq.py <out>.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use hasty_bench::b0data::{load, LoadOpts};
use hasty_cuda::{Device, C32};
use hasty_phantom::volume::write_npy;
use hasty_qfit::waterfat::FatSpectrum;
use hasty_recon::checkpoint::RunCtl;
use hasty_recon::data::read_npy;
use hasty_recon::gpu::Kernels;
use hasty_recon::op::SenseOp;
use hasty_recon::p2np::{P2np, Stack};
use num_complex::Complex64 as C;

/// separable cubic B-spline basis for a scalar field on the grid (knot spacing h voxels)
struct Spline {
    dims: [usize; 3],
    nb: [usize; 3],
    b: [Vec<f64>; 3],
}

fn bsp(t: f64) -> [f64; 4] {
    let (t2, t3) = (t * t, t * t * t);
    [(1.0 - t).powi(3) / 6.0, (3.0 * t3 - 6.0 * t2 + 4.0) / 6.0, (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) / 6.0, t3 / 6.0]
}

impl Spline {
    fn new(dims: [usize; 3], h: f64) -> Self {
        let nb = dims.map(|n| ((n as f64 - 1.0) / h).floor() as usize + 4);
        let b = [0, 1, 2].map(|a| {
            let mut m = vec![0.0; dims[a] * nb[a]];
            for x in 0..dims[a] {
                let u = x as f64 / h;
                let f = u.floor();
                let w = bsp(u - f);
                for (o, wv) in w.iter().enumerate() {
                    let j = (f as usize + o).min(nb[a] - 1);
                    m[x * nb[a] + j] += wv;
                }
            }
            m
        });
        Self { dims, nb, b }
    }
    fn n(&self) -> usize {
        self.nb[0] * self.nb[1] * self.nb[2]
    }
    /// ω = B c  (c: [nb_x, nb_y, nb_z] x fastest → grid x fastest)
    fn apply(&self, c: &[f64]) -> Vec<f64> {
        let (d, nb) = (self.dims, self.nb);
        // contract z, then y, then x
        let mut t1 = vec![0.0; nb[0] * nb[1] * d[2]];
        for z in 0..d[2] {
            for k in 0..nb[2] {
                let w = self.b[2][z * nb[2] + k];
                if w == 0.0 {
                    continue;
                }
                for ij in 0..nb[0] * nb[1] {
                    t1[ij + nb[0] * nb[1] * z] += w * c[ij + nb[0] * nb[1] * k];
                }
            }
        }
        let mut t2 = vec![0.0; nb[0] * d[1] * d[2]];
        for z in 0..d[2] {
            for y in 0..d[1] {
                for j in 0..nb[1] {
                    let w = self.b[1][y * nb[1] + j];
                    if w == 0.0 {
                        continue;
                    }
                    for i in 0..nb[0] {
                        t2[i + nb[0] * (y + d[1] * z)] += w * t1[i + nb[0] * (j + nb[1] * z)];
                    }
                }
            }
        }
        let mut out = vec![0.0; d[0] * d[1] * d[2]];
        for yz in 0..d[1] * d[2] {
            for x in 0..d[0] {
                let mut s = 0.0;
                for i in 0..nb[0] {
                    s += self.b[0][x * nb[0] + i] * t2[i + nb[0] * yz];
                }
                out[x + d[0] * yz] = s;
            }
        }
        out
    }
    /// Bᵀ g
    fn adjoint(&self, g: &[f64]) -> Vec<f64> {
        let (d, nb) = (self.dims, self.nb);
        let mut t2 = vec![0.0; nb[0] * d[1] * d[2]];
        for yz in 0..d[1] * d[2] {
            for x in 0..d[0] {
                let v = g[x + d[0] * yz];
                if v == 0.0 {
                    continue;
                }
                for i in 0..nb[0] {
                    t2[i + nb[0] * yz] += self.b[0][x * nb[0] + i] * v;
                }
            }
        }
        let mut t1 = vec![0.0; nb[0] * nb[1] * d[2]];
        for z in 0..d[2] {
            for y in 0..d[1] {
                for j in 0..nb[1] {
                    let w = self.b[1][y * nb[1] + j];
                    if w == 0.0 {
                        continue;
                    }
                    for i in 0..nb[0] {
                        t1[i + nb[0] * (j + nb[1] * z)] += w * t2[i + nb[0] * (y + d[1] * z)];
                    }
                }
            }
        }
        let mut c = vec![0.0; self.n()];
        for z in 0..d[2] {
            for k in 0..nb[2] {
                let w = self.b[2][z * nb[2] + k];
                if w == 0.0 {
                    continue;
                }
                for ij in 0..nb[0] * nb[1] {
                    c[ij + nb[0] * nb[1] * k] += w * t1[ij + nb[0] * nb[1] * z];
                }
            }
        }
        c
    }
}

fn softplus(u: f64) -> (f64, f64) {
    // value and derivative
    if u > 30.0 {
        (u, 1.0)
    } else {
        let e = u.exp();
        ((1.0 + e).ln(), e / (1.0 + e))
    }
}
fn softplus_inv(y: f64) -> f64 {
    let y = y.max(1e-6);
    if y > 30.0 { y } else { (y.exp() - 1.0).ln() }
}

/// trilinear resample of a map on a grid with `gs_src` / `gs_dst` voxel ratio (same FOV, centred convention)
fn resample(src: &[f32], dsrc: [usize; 3], ddst: [usize; 3]) -> Vec<f32> {
    let r = [0, 1, 2].map(|a| dsrc[a] as f64 / ddst[a] as f64);
    let nv = ddst[0] * ddst[1] * ddst[2];
    (0..nv)
        .map(|q| {
            let idx = [q % ddst[0], (q / ddst[0]) % ddst[1], q / (ddst[0] * ddst[1])];
            // position in source index units: (i − n/2) Δd = (j − m/2) Δs → j = (i − n/2) r + m/2
            let p = [0, 1, 2].map(|a| ((idx[a] as f64 - (ddst[a] / 2) as f64) * r[a] + (dsrc[a] / 2) as f64).clamp(0.0, dsrc[a] as f64 - 1.0001));
            let i0 = p.map(|x| x.floor() as usize);
            let f = [0, 1, 2].map(|a| p[a] - i0[a] as f64);
            let (mut s, mut ws) = (0.0, 0.0);
            for c in 0..8 {
                let o = [c & 1, (c >> 1) & 1, c >> 2];
                let w: f64 = (0..3).map(|a| if o[a] == 1 { f[a] } else { 1.0 - f[a] }).product();
                let v = src[(i0[0] + o[0]) + dsrc[0] * ((i0[1] + o[1]) + dsrc[1] * (i0[2] + o[2]))];
                if v.is_finite() {
                    s += w * v as f64;
                    ws += w;
                }
            }
            if ws > 0.0 { (s / ws) as f32 } else { f32::NAN }
        })
        .collect()
}

fn par_chunks<F: Fn(usize, usize) + Sync>(n: usize, f: F) {
    let nt = std::thread::available_parallelism().map_or(1, |x| x.get()).min(32);
    let ch = n.div_ceil(nt);
    std::thread::scope(|sc| {
        for t in 0..nt {
            let (a, b) = (t * ch, ((t + 1) * ch).min(n));
            if a < b {
                let f = &f;
                sc.spawn(move || f(a, b));
            }
        }
    });
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |n: &str| -> Option<String> { args.iter().position(|a| a == n).map(|i| { args.remove(i); args.remove(i) }) };
    let gs: usize = take("--grid-scale").map_or(1, |s| s.parse().unwrap());
    let nvc: usize = take("--vcoils").map_or(12, |s| s.parse().unwrap());
    let knot_mm: f64 = take("--knot-mm").map_or(20.0, |s| s.parse().unwrap());
    let iters: usize = take("--iters").map_or(150, |s| s.parse().unwrap());
    let rho_iters: usize = take("--rho-iters").map_or(40, |s| s.parse().unwrap());
    let lam_rho: f64 = take("--lam-rho").map_or(2e-3, |s| s.parse().unwrap());
    let lam_r2: f64 = take("--lam-r2").map_or(2e-3, |s| s.parse().unwrap());
    let every: usize = take("--every").map_or(10, |s| s.parse().unwrap());
    let nterm: usize = take("--terms").map_or(1, |s| s.parse().unwrap());
    let moco: Option<PathBuf> = take("--moco").map(PathBuf::from);
    let resp_mode = take("--resp").unwrap_or_else(|| "bellows".into());
    let nlev: usize = take("--levels").map_or(16, |s| s.parse().unwrap());
    let pre_name = take("--precond").unwrap_or_else(|| "jacobi".into());
    let init = PathBuf::from(take("--init").expect("--init <recon_b0 run> (R2*, ω initialiser)"));
    let clean = args.iter().position(|a| a == "--clean").map(|i| { args.remove(i); true }).unwrap_or(false);
    let dir = PathBuf::from(&args[0]);
    let out = PathBuf::from(&args[1]);
    std::fs::create_dir_all(&out).unwrap();
    let t0 = Instant::now();
    // breathing state per readout → groups (moco)
    let (group, r_tr): (Option<Vec<usize>>, Vec<f64>) = match &moco {
        Some(_) => {
            let r: Vec<f64> = match resp_mode.as_str() {
                "truth" => read_npy(&dir.join("resp.npy")).unwrap().f64s().chunks(2).map(|c| c[0]).collect(),
                _ => hasty_bench::physio::normalise(&read_npy(&dir.join("physio.npy")).unwrap().f64s().chunks(5).map(|c| c[3]).collect::<Vec<_>>()),
            };
            (Some(hasty_bench::physio::levels(&r, nlev)), r)
        }
        None => (None, vec![]),
    };
    let d = load(&dir, &LoadOpts { nvc, grid_scale: gs, support_rel: 0.05, clean, group: group.clone() });
    // reference sets (windows): the image unknowns; all groups' data enter each
    let (dims, nv, nset) = (d.dims, d.nv, d.n_win);
    println!("{nset} windows, grid {dims:?} at {:.1} mm, {} virtual coils ({:.0}s)", d.res * 1e3, nvc, t0.elapsed().as_secs_f64());

    // ── initialiser: R2*, ω (and magnitude for the support / scale) from a recon_b0 run ──
    let isumm: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(init.join("summary.json")).unwrap()).unwrap();
    let idims: [usize; 3] = [0, 1, 2].map(|a| isumm["dims"][a].as_u64().unwrap() as usize);
    let inv = idims[0] * idims[1] * idims[2];
    let rd = |n: &str| -> Vec<f32> { let v = read_npy(&init.join(n)).unwrap().f32s(); resample(&v[..inv], idims, dims) };
    let (om0, r20, w0, f0) = (rd("maps_omega_hz.npy"), rd("maps_r2.npy"), rd("maps_water.npy"), rd("maps_fat.npy"));
    let mag0: Vec<f64> = (0..nv).map(|q| (w0[q] + f0[q]) as f64).collect();
    let vox: Vec<usize> = (0..nv).filter(|&q| d.support[q]).collect();
    let nx = vox.len();
    let pos_of: Vec<i64> = {
        let mut p = vec![-1i64; nv];
        for (i, &q) in vox.iter().enumerate() {
            p[q] = i as i64;
        }
        p
    };
    let s_rho = {
        let mut m: Vec<f64> = vox.iter().map(|&q| mag0[q]).filter(|x| x.is_finite() && *x > 0.0).collect();
        m.sort_by(|a, b| a.partial_cmp(b).unwrap());
        m[m.len() * 3 / 4].max(1e-12)
    };
    let (s_r2, s_c) = (100.0, 2.0 * std::f64::consts::PI * 50.0);
    let spl = Spline::new(dims, knot_mm * 1e-3 / d.res);
    let spl2 = { let mut s2 = Spline::new(dims, knot_mm * 1e-3 / d.res); for a in 0..3 { s2.b[a].iter_mut().for_each(|x| *x *= *x); } s2 };
    let nb = spl.n();
    // spline fit of the initial ω (weighted LS by CG on Bᵀ W B c = Bᵀ W ω, W = signal magnitude in reliable voxels)
    let wfit: Vec<f64> = (0..nv).map(|q| if d.support[q] && om0[q].is_finite() && mag0[q] > 0.2 * s_rho { mag0[q] } else { 0.0 }).collect();
    let omv: Vec<f64> = (0..nv).map(|q| if wfit[q] > 0.0 { 2.0 * std::f64::consts::PI * om0[q] as f64 } else { 0.0 }).collect();
    let normal = |c: &[f64]| -> Vec<f64> {
        let w = spl.apply(c);
        let ww: Vec<f64> = (0..nv).map(|q| wfit[q] * w[q]).collect();
        let mut r = spl.adjoint(&ww);
        let tr = 1e-6 * wfit.iter().sum::<f64>() / nb as f64;
        for (ri, ci) in r.iter_mut().zip(c) {
            *ri += tr * ci;
        }
        r
    };
    let rhs = spl.adjoint(&(0..nv).map(|q| wfit[q] * omv[q]).collect::<Vec<_>>());
    let mut c0 = vec![0.0; nb];
    {
        let mut r = rhs.clone();
        let mut p = r.clone();
        let mut rr: f64 = r.iter().map(|x| x * x).sum();
        for _ in 0..200 {
            let ap = normal(&p);
            let a = rr / p.iter().zip(&ap).map(|(x, y)| x * y).sum::<f64>();
            for i in 0..nb {
                c0[i] += a * p[i];
                r[i] -= a * ap[i];
            }
            let rn: f64 = r.iter().map(|x| x * x).sum();
            if rn < 1e-20 * rhs.iter().map(|x| x * x).sum::<f64>() {
                break;
            }
            for i in 0..nb {
                p[i] = r[i] + rn / rr * p[i];
            }
            rr = rn;
        }
    }
    println!("init from {}: {nx} support voxels, ρ scale {s_rho:.3e}, {nb} spline coefficients ({knot_mm} mm knots) ({:.0}s)", init.display(), t0.elapsed().as_secs_f64());

    // variables (scaled): [ρw re | ρw im | ρf re | ρf im | u (R2*) | c]
    let nvar = 5 * nx + nb;
    let ctl = RunCtl::new(&out, every);
    let state_path = out.join("state_v.npy");
    let mut v = vec![0.0f64; nvar];
    let mut it0 = 0usize;
    if state_path.exists() {
        v = read_npy(&state_path).unwrap().f64s();
        it0 = serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(out.join("state.json")).unwrap()).unwrap()["iter"].as_u64().unwrap() as usize;
        println!("resuming at iteration {it0}");
    } else {
        for (i, &q) in vox.iter().enumerate() {
            let r2 = if r20[q].is_finite() { r20[q] as f64 } else { 30.0 };
            v[4 * nx + i] = softplus_inv(r2.clamp(1.0, 900.0) / s_r2);
        }
        for j in 0..nb {
            v[5 * nx + j] = c0[j] / s_c;
        }
    }

    // ── GPU operator ──
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let mut op = SenseOp::new(&dev, &st, kern.clone(), dims, [&d.coords[0][0], &d.coords[0][1], &d.coords[0][2]], &d.smaps, nvc, vec![], 1e-4).unwrap();
    let ng = d.n_group;
    let dcoords: Vec<[cudarc::driver::CudaSlice<f32>; 3]> = d.coords.iter().map(|c| [0, 1, 2].map(|a| st.clone_htod(&c[a]).unwrap())).collect();
    // data per reference set: the groups' blocks back to back
    let dys: Vec<cudarc::driver::CudaSlice<C32>> = (0..nset).map(|w| st.clone_htod(&(0..ng).flat_map(|g| d.ys[g * nset + w].iter().copied()).collect::<Vec<_>>()).unwrap()).collect();
    let ms: Vec<usize> = d.coords.iter().map(|c| c[0].len()).collect();
    let bi = hasty_recon::multiterm::BandInterp::new(d.half_window, nterm, 1050.0, 500.0, 401);
    let wts: Vec<Vec<cudarc::driver::CudaSlice<C32>>> = (0..ng * nset)
        .map(|t| {
            let a: Vec<Vec<C>> = d.dts[t].iter().map(|&dl| bi.at(dl as f64)).collect();
            (0..nterm).map(|l| st.clone_htod(&a.iter().map(|w| C32::new(w[l].re as f32, w[l].im as f32)).collect::<Vec<_>>()).unwrap()).collect()
        })
        .collect();
    println!("time model: {nterm} node(s) per window at δ = {:?} µs, worst-case interpolation error {:.2e}", bi.nodes.iter().map(|x| (x * 1e7).round() / 10.0).collect::<Vec<_>>(), bi.max_err);
    let mut sets_plain;
    let mut sets_moco;
    let mut moco_info = serde_json::Value::Null;
    let sop: &mut dyn hasty_recon::setop::SetOp = if let Some(md) = &moco {
        let mj: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(md.join("motion.json")).unwrap()).unwrap();
        assert_eq!(mj["dims"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect::<Vec<_>>(), dims.to_vec(), "--moco grid mismatch");
        let r_ref = mj["r_gate"][0].as_f64().unwrap();
        let smm = read_npy(&md.join("motion_S_mm.npy")).unwrap().f32s();
        let grp = group.as_ref().unwrap();
        let rbar: Vec<f64> = (0..ng).map(|g| { let v: Vec<f64> = (0..r_tr.len()).filter(|&i| grp[i] == g).map(|i| r_tr[i]).collect(); v.iter().sum::<f64>() / v.len() as f64 }).collect();
        let disp: Vec<cudarc::driver::CudaSlice<f32>> = rbar.iter().map(|&r| st.clone_htod(&smm.iter().map(|x| ((r - r_ref) * *x as f64 / (d.res * 1e3)) as f32).collect::<Vec<f32>>()).unwrap()).collect();
        println!("motion: {ng} breathing levels ({resp_mode}), r̄ − r_ref {:?}, field from {}", rbar.iter().map(|r| ((r - r_ref) * 100.0).round() / 100.0).collect::<Vec<_>>(), md.display());
        moco_info = serde_json::json!({"dir": md, "resp": resp_mode, "levels": ng, "r_ref": r_ref, "r_bar": rbar});
        let mut dc = dcoords.into_iter();
        let coords_b: Vec<Vec<[cudarc::driver::CudaSlice<f32>; 3]>> = (0..ng).map(|_| (0..nset).map(|_| dc.next().unwrap()).collect()).collect();
        let ms_b: Vec<Vec<usize>> = (0..ng).map(|g| ms[g * nset..(g + 1) * nset].to_vec()).collect();
        let mut wt = wts.into_iter();
        let w_b: Vec<Vec<Vec<cudarc::driver::CudaSlice<C32>>>> = (0..ng).map(|_| (0..nset).map(|_| wt.next().unwrap()).collect()).collect();
        sets_moco = hasty_recon::moco::MocoTermSets::new(&st, &dev, &mut op, coords_b, ms_b, w_b, disp).unwrap();
        &mut sets_moco
    } else {
        sets_plain = hasty_recon::multiterm::MultiTermSets::new(&st, &mut op, dcoords, ms, wts).unwrap();
        &mut sets_plain
    };
    let mut p2 = P2np::new(sop, dys.iter().collect(), vec![], &st).unwrap();
    let mut xs = Stack::zeros(&st, nset, nterm * nv).unwrap();
    let mut gs_ = Stack::zeros(&st, nset, nterm * nv).unwrap();
    let fat = FatSpectrum::six_peak();
    // node times and fat basis per (window, node): index t·L + l
    let tnode: Vec<f64> = (0..nset).flat_map(|t| bi.nodes.iter().map(move |&dl| (t, dl))).map(|(t, dl)| d.te[t] + dl).collect();
    let cf: Vec<C> = tnode.iter().map(|&t| fat.basis(d.b0, t)).collect();
    // per-window concomitant phasors at the support voxels
    let conc: Vec<Vec<C>> = (0..nset).map(|t| vox.iter().map(|&q| C::from_polar(1.0, -d.conc(t, q))).collect()).collect();

    // neighbours for TV (support voxels only), +x, +y, +z
    let nbr: Vec<[i64; 3]> = vox
        .iter()
        .map(|&q| {
            let idx = [q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1])];
            let st_ = [1, dims[0], dims[0] * dims[1]];
            [0, 1, 2].map(|a| if idx[a] + 1 < dims[a] { pos_of[q + st_[a]] } else { -1 })
        })
        .collect();
    let eps_tv = 0.02;
    // λ in units of the data term: λ_eff = λ · ½‖y‖² / n_vox (a prior term of O(1) per voxel weighs like λ of the
    // zero-image misfit per voxel)
    let y2: f64 = d.ys.iter().flat_map(|y| y.iter()).map(|c| (c.re as f64).powi(2) + (c.im as f64).powi(2)).sum::<f64>() * 0.5;
    let sigma = d.meta["noise_sigma"].as_f64().unwrap_or(0.0);
    let n_meas: usize = d.coords.iter().map(|c| c[0].len()).sum::<usize>() * nvc;
    let floor = if clean { 0.0 } else { 0.5 * sigma * sigma * n_meas as f64 };
    println!("data term at the noise floor (½ σ² N, whitened): {floor:.3e}; zero-image misfit ½‖y‖² {y2:.3e}");
    let lam_rho = lam_rho * y2 / nx as f64;
    let lam_r2 = lam_r2 * y2 / nx as f64;

    // per reference window: total samples over groups (weights of the Gauss–Newton diagonal)
    let mtot: Vec<f64> = (0..nset).map(|w| (0..d.n_group).map(|g| d.coords[g * nset + w][0].len()).sum::<usize>() as f64).collect();
    let m_all: f64 = mtot.iter().sum();
    let mut circ = if pre_name == "circ" {
        let cat: [Vec<f32>; 3] = [0, 1, 2].map(|a| d.coords.iter().flat_map(|c| c[a].iter().copied()).collect());
        Some(hasty_recon::precond::CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&cat[0], &cat[1], &cat[2]], &d.smaps, nvc, 1e-3).unwrap())
    } else {
        None
    };
    // Gauss–Newton diagonal of the current iterate, refreshed by every evaluation
    let diag = std::rc::Rc::new(std::cell::RefCell::new(vec![1.0f64; nvar]));
    let diag_e = diag.clone();
    // objective: phase 1 (`rho_only`) keeps u and c fixed
    let mut evals = 0usize;
    let mut eval = |v: &[f64], grad: &mut [f64], rho_only: bool| -> (f64, f64) {
        evals += 1;
        let om = spl.apply(&v[5 * nx..].iter().map(|x| x * s_c).collect::<Vec<_>>());
        let mut r2 = vec![0.0; nx];
        let mut dr2 = vec![0.0; nx];
        for i in 0..nx {
            let (s, ds) = softplus(v[4 * nx + i]);
            r2[i] = s_r2 * s;
            dr2[i] = s_r2 * ds;
        }
        // images per window
        let mut host: Vec<Vec<C32>> = vec![vec![C32::new(0.0, 0.0); nterm * nv]; nset];
        let mut ee: Vec<Vec<C>> = vec![vec![C::new(0.0, 0.0); nterm * nx]; nset];
        std::thread::scope(|sc| {
            for (t, (h, e_t)) in host.iter_mut().zip(ee.iter_mut()).enumerate() {
                let (om, r2, conc, cf, tnode, vox, v) = (&om, &r2, &conc, &cf, &tnode, &vox, v);
                sc.spawn(move || {
                    for l in 0..nterm {
                        let (te, c_tl) = (tnode[t * nterm + l], cf[t * nterm + l]);
                        for i in 0..nx {
                            let q = vox[i];
                            let e = (-C::new(r2[i], om[q]) * te).exp() * conc[t][i];
                            let rho = C::new(v[i], v[nx + i]) + C::new(v[2 * nx + i], v[3 * nx + i]) * c_tl;
                            let x = rho * e * s_rho;
                            h[l * nv + q] = C32::new(x.re as f32, x.im as f32);
                            e_t[l * nx + i] = e;
                        }
                    }
                });
            }
        });
        for t in 0..nset {
            st.memcpy_htod(&host[t], &mut xs.v[t]).unwrap();
        }
        let f_data = p2.gradient(&st, &xs, &mut gs_).unwrap();
        let g_img: Vec<Vec<C32>> = gs_.v.iter().map(|g| st.clone_dtoh(g).unwrap()).collect();
        // chain rule (gradient wrt conj ρ → real parts: ∂f/∂Re ρ = 2 Re(conj(...)) convention: f real, g = Eᴴr
        // ∂f/∂θ = Re Σ_t conj(g_t) ∂x_t/∂θ)
        grad.iter_mut().for_each(|x| *x = 0.0);
        let mut gom = vec![0.0; nv];
        let mut dg = vec![0.0; nvar];
        let mut hom = vec![0.0; nv];
        {
            let grad_ptr = grad.as_mut_ptr() as usize;
            let gom_ptr = gom.as_mut_ptr() as usize;
            let dg_ptr = dg.as_mut_ptr() as usize;
            let hom_ptr = hom.as_mut_ptr() as usize;
            par_chunks(nx, |a, b| {
                let grad = unsafe { std::slice::from_raw_parts_mut(grad_ptr as *mut f64, nvar) };
                let gom = unsafe { std::slice::from_raw_parts_mut(gom_ptr as *mut f64, nv) };
                let dg = unsafe { std::slice::from_raw_parts_mut(dg_ptr as *mut f64, nvar) };
                let hom = unsafe { std::slice::from_raw_parts_mut(hom_ptr as *mut f64, nv) };
                for i in a..b {
                    let q = vox[i];
                    let rho_w = C::new(v[i], v[nx + i]);
                    let rho_f = C::new(v[2 * nx + i], v[3 * nx + i]);
                    let (mut gw, mut gf, mut gr2, mut go) = (C::new(0.0, 0.0), C::new(0.0, 0.0), 0.0, 0.0);
                    let (mut hw, mut hf, mut hx) = (0.0, 0.0, 0.0);
                    for t in 0..nset {
                        for l in 0..nterm {
                            let g = g_img[t][l * nv + q];
                            let g = C::new(g.re as f64, g.im as f64);
                            let e = ee[t][l * nx + i];
                            let (c_tl, te) = (cf[t * nterm + l], tnode[t * nterm + l]);
                            let gc = g.conj() * e * s_rho;
                            // x = s(ρw + ρf c) e: ∂x/∂Re ρw = s e, ∂x/∂Im ρw = i s e
                            gw += gc; // Re(gc) → Re ρw, Re(i gc) = −Im(gc) → Im ρw
                            gf += gc * c_tl;
                            let x = (rho_w + rho_f * c_tl) * e * s_rho;
                            let gx = g.conj() * x * (-te);
                            gr2 += gx.re;
                            go += (gx * C::new(0.0, 1.0)).re;
                            let wt = mtot[t] / nterm as f64;
                            let e2 = e.norm_sqr() * s_rho * s_rho;
                            hw += wt * e2;
                            hf += wt * e2 * c_tl.norm_sqr();
                            hx += wt * x.norm_sqr() * te * te;
                        }
                    }
                    dg[i] = hw;
                    dg[nx + i] = hw;
                    dg[2 * nx + i] = hf;
                    dg[3 * nx + i] = hf;
                    dg[4 * nx + i] = hx * dr2[i] * dr2[i];
                    hom[q] = hx;
                    grad[i] = gw.re;
                    grad[nx + i] = -gw.im;
                    grad[2 * nx + i] = gf.re;
                    grad[3 * nx + i] = -gf.im;
                    if !rho_only {
                        grad[4 * nx + i] = gr2 * dr2[i];
                        gom[q] = go;
                    }
                }
            });
        }
        if !rho_only {
            let gc = spl.adjoint(&gom);
            for j in 0..nb {
                grad[5 * nx + j] = gc[j] * s_c;
            }
        }
        {
            let hc = spl2.adjoint(&hom);
            let cr = 12.0 / eps_tv;
            for i in 0..nx {
                for ch in 0..4 {
                    dg[ch * nx + i] += lam_rho * cr;
                }
                dg[4 * nx + i] += lam_r2 * cr * (dr2[i] / s_r2).powi(2);
            }
            for j in 0..nb {
                dg[5 * nx + j] = hc[j] * s_c * s_c;
            }
            // floor each block at 1 % of its median: coefficients / voxels with (almost) no data curvature would
            // otherwise get enormous steps (spline coefficients at the support edge, R2* in empty voxels)
            for (a, b) in [(0, 4 * nx), (4 * nx, 5 * nx), (5 * nx, nvar)] {
                let mut srt: Vec<f64> = dg[a..b].iter().cloned().filter(|x| *x > 0.0).collect();
                if srt.is_empty() {
                    continue;
                }
                srt.sort_by(|x, y| x.partial_cmp(y).unwrap());
                let fl = 1e-2 * srt[srt.len() / 2];
                for x in dg[a..b].iter_mut() {
                    *x = x.max(fl);
                }
            }
            *diag_e.borrow_mut() = dg;
        }
        // priors: Huber-TV, joint over the four ρ channels (scaled units) and on R2* (/s_r2)
        let mut f_prior = 0.0;
        for i in 0..nx {
            for a in 0..3 {
                let j = nbr[i][a];
                if j < 0 {
                    continue;
                }
                let j = j as usize;
                let dv = [0, 1, 2, 3].map(|ch| v[ch * nx + j] - v[ch * nx + i]);
                let n2 = dv.iter().map(|x| x * x).sum::<f64>();
                let s = (n2 + eps_tv * eps_tv).sqrt();
                f_prior += lam_rho * (s - eps_tv);
                for ch in 0..4 {
                    let gdv = lam_rho * dv[ch] / s;
                    grad[ch * nx + j] += gdv;
                    grad[ch * nx + i] -= gdv;
                }
                if !rho_only {
                    let (a2, b2) = (r2[j] / s_r2, r2[i] / s_r2);
                    let dd = a2 - b2;
                    let s2 = (dd * dd + eps_tv * eps_tv).sqrt();
                    f_prior += lam_r2 * (s2 - eps_tv);
                    let gd = lam_r2 * dd / s2;
                    grad[4 * nx + j] += gd * dr2[j] / s_r2;
                    grad[4 * nx + i] -= gd * dr2[i] / s_r2;
                }
            }
        }
        if rho_only {
            for x in grad[4 * nx..].iter_mut() {
                *x = 0.0;
            }
        }
        (f_data + f_prior, f_data)
    };

    // finite-difference check of the analytic gradient, block by block (ρ, R2* parameter, spline coefficients)
    if std::env::var("MBQ_GRADCHECK").is_ok() {
        let mut g = vec![0.0; nvar];
        let mut vv = v.clone();
        for (i, x) in vv[..4 * nx].iter_mut().enumerate() {
            *x = 0.3 * ((i as f64 * 0.37).sin());
        }
        eval(&vv, &mut g, false);
        for (name, a, b) in [("rho", 0, 4 * nx), ("r2", 4 * nx, 5 * nx), ("spline", 5 * nx, nvar)] {
            let dir: Vec<f64> = (0..nvar).map(|i| if i >= a && i < b { (i as f64 * 1.618).sin() + 0.1 } else { 0.0 }).collect();
            let gd: f64 = dir.iter().zip(&g).map(|(x, y)| x * y).sum();
            for h in [1e-2, 1e-3] {
                let vp: Vec<f64> = vv.iter().zip(&dir).map(|(x, d)| x + h * d).collect();
                let vm: Vec<f64> = vv.iter().zip(&dir).map(|(x, d)| x - h * d).collect();
                let mut gt = vec![0.0; nvar];
                let (fp, _) = eval(&vp, &mut gt, false);
                let (fm, _) = eval(&vm, &mut gt, false);
                println!("gradcheck {name}: h {h:.0e}: FD {:.6e} vs analytic {gd:.6e} (rel {:.2e})", (fp - fm) / (2.0 * h), ((fp - fm) / (2.0 * h) - gd) / gd.abs().max(1e-30));
            }
        }
        return;
    }
    // λ in units of the data term: rescale by the initial data misfit per voxel
    // (the priors above use λ directly on scaled variables; report both terms)
    let mut csv = std::fs::OpenOptions::new().create(true).append(true).open(out.join("metrics.csv")).unwrap();
    use std::io::Write;
    if it0 == 0 {
        writeln!(csv, "phase,iter,f,f_data,seconds").unwrap();
    }
    let tr = Instant::now();
    let mut run = |v: &mut Vec<f64>, k0: usize, kmax: usize, rho_only: bool, csv: &mut std::fs::File| -> Option<usize> {
        // L-BFGS (two-loop, m = 8, Armijo backtracking)
        let m = 8;
        let n = v.len();
        let mut g = vec![0.0; n];
        let (mut f, mut fd) = eval(v, &mut g, rho_only);
        let mut ss: Vec<Vec<f64>> = vec![];
        let mut yy: Vec<Vec<f64>> = vec![];
        for k in k0..kmax {
            if ctl.stop_requested() {
                return Some(k);
            }
            let mut q = g.clone();
            let mut al = vec![0.0; ss.len()];
            for i in (0..ss.len()).rev() {
                let rho = 1.0 / dotp(&yy[i], &ss[i]);
                al[i] = rho * dotp(&ss[i], &q);
                for (qj, yj) in q.iter_mut().zip(&yy[i]) {
                    *qj -= al[i] * yj;
                }
            }
            if pre_name == "none" {
                let gam = match (ss.last(), yy.last()) {
                    (Some(s), Some(y)) => dotp(s, y) / dotp(y, y),
                    _ => 1e-2 / (dotp(&g, &g).sqrt() / (n as f64).sqrt()).max(1e-30),
                };
                q.iter_mut().for_each(|x| *x *= gam);
            } else {
                // blocks ρ (0..4nx), R2* (4nx..5nx), spline (5nx..): H0 = γ_k P_k
                let dgv = diag.borrow().clone();
                let blocks = [(0, 4 * nx), (4 * nx, 5 * nx), (5 * nx, n)];
                let mut apply = |z: &mut [f64]| {
                    for (bi, &(a, b)) in blocks.iter().enumerate() {
                        if bi == 0 {
                            if let Some(cp) = circ.as_mut() {
                                for ch in 0..2 {
                                    let o = 2 * ch * nx;
                                    let mut img = vec![C32::new(0.0, 0.0); nv];
                                    for i in 0..nx {
                                        let sc = 1.0 / dgv[o + i].max(1e-300).sqrt();
                                        img[vox[i]] = C32::new((z[o + i] * sc) as f32, (z[o + nx + i] * sc) as f32);
                                    }
                                    let di = st.clone_htod(&img).unwrap();
                                    let mut dz = st.alloc_zeros::<C32>(nv).unwrap();
                                    cp.apply(&st, &di, &mut dz).unwrap();
                                    let h = st.clone_dtoh(&dz).unwrap();
                                    for i in 0..nx {
                                        let sc = m_all / dgv[o + i].max(1e-300).sqrt();
                                        z[o + i] = h[vox[i]].re as f64 * sc;
                                        z[o + nx + i] = h[vox[i]].im as f64 * sc;
                                    }
                                }
                                continue;
                            }
                        }
                        for i in a..b {
                            z[i] /= dgv[i].max(1e-300);
                        }
                    }
                };
                let mut gams = [1.0f64; 3];
                if let (Some(s), Some(y)) = (ss.last(), yy.last()) {
                    let mut py = y.clone();
                    apply(&mut py);
                    for (bi, &(a, b)) in blocks.iter().enumerate() {
                        let sy: f64 = (a..b).map(|i| s[i] * y[i]).sum();
                        let ypy: f64 = (a..b).map(|i| y[i] * py[i]).sum();
                        // the per-block secant scale; P_k ≈ the inverse Gauss–Newton diagonal, so γ ≈ 1 is expected —
                        // clamped so that a block whose gradient barely changed cannot blow up the step
                        if sy > 0.0 && ypy > 0.0 {
                            gams[bi] = (sy / ypy).clamp(0.02, 50.0);
                        }
                    }
                }
                apply(&mut q);
                for (bi, &(a, b)) in blocks.iter().enumerate() {
                    for x in q[a..b].iter_mut() {
                        *x *= gams[bi];
                    }
                }
            }
            for i in 0..ss.len() {
                let rho = 1.0 / dotp(&yy[i], &ss[i]);
                let b = rho * dotp(&yy[i], &q);
                for (qj, sj) in q.iter_mut().zip(&ss[i]) {
                    *qj += (al[i] - b) * sj;
                }
            }
            let dir: Vec<f64> = q.iter().map(|x| -x).collect();
            let slope = dotp(&g, &dir);
            if !(slope < 0.0) {
                ss.clear();
                yy.clear();
                continue;
            }
            let v0 = v.clone();
            let mut step = 1.0;
            let mut g2 = vec![0.0; n];
            let mut ok = false;
            for _ in 0..20 {
                for i in 0..n {
                    v[i] = v0[i] + step * dir[i];
                }
                let (f2, fd2) = eval(v, &mut g2, rho_only);
                if f2 <= f + 1e-4 * step * slope {
                    let s: Vec<f64> = (0..n).map(|i| v[i] - v0[i]).collect();
                    let y: Vec<f64> = (0..n).map(|i| g2[i] - g[i]).collect();
                    if dotp(&s, &y) > 1e-12 * dotp(&s, &s).sqrt() * dotp(&y, &y).sqrt() {
                        ss.push(s);
                        yy.push(y);
                        if ss.len() > m {
                            ss.remove(0);
                            yy.remove(0);
                        }
                    }
                    f = f2;
                    fd = fd2;
                    g = g2.clone();
                    ok = true;
                    break;
                }
                step *= 0.5;
            }
            if !ok {
                v.copy_from_slice(&v0);
                println!("  line search failed at {k}; restarting the memory");
                ss.clear();
                yy.clear();
                continue;
            }
            let ph = if rho_only { "rho" } else { "joint" };
            writeln!(csv, "{ph},{},{f:.8e},{fd:.8e},{:.1}", k + 1, tr.elapsed().as_secs_f64()).unwrap();
            if (k + 1) % every == 0 || k + 1 == kmax {
                println!("  {ph} {:4}: f {f:.5e} (data {fd:.5e}) ({:.0}s)", k + 1, tr.elapsed().as_secs_f64());
                save(v, k + 1, &state_path, &out);
            }
        }
        None
    };
    // phase 1 (ρ only) then phase 2 (joint); the iteration counter runs over both
    if it0 < rho_iters {
        if let Some(k) = run(&mut v, it0, rho_iters, true, &mut csv) {
            save(&v, k, &state_path, &out);
            println!("stopped at {k}; rerun to continue");
            return;
        }
        it0 = rho_iters;
    }
    if let Some(k) = run(&mut v, it0, rho_iters + iters, false, &mut csv) {
        save(&v, k, &state_path, &out);
        println!("stopped at {k}; rerun to continue");
        return;
    }
    println!("{evals} objective evaluations ({:.0}s)", t0.elapsed().as_secs_f64());

    // ── maps and errors ──
    let om = spl.apply(&v[5 * nx..].iter().map(|x| x * s_c).collect::<Vec<_>>());
    let mut om_hz = vec![f32::NAN; nv];
    let mut r2m = vec![f32::NAN; nv];
    let mut ffm = vec![f32::NAN; nv];
    let mut wm = vec![0f32; nv];
    let mut fm = vec![0f32; nv];
    for (i, &q) in vox.iter().enumerate() {
        let w = (v[i].powi(2) + v[nx + i].powi(2)).sqrt() * s_rho;
        let f = (v[2 * nx + i].powi(2) + v[3 * nx + i].powi(2)).sqrt() * s_rho;
        om_hz[q] = (om[q] / (2.0 * std::f64::consts::PI)) as f32;
        r2m[q] = (s_r2 * softplus(v[4 * nx + i]).0) as f32;
        ffm[q] = (f / (w + f).max(1e-30)) as f32;
        wm[q] = w as f32;
        fm[q] = f as f32;
    }
    let mut mag: Vec<f32> = vox.iter().map(|&q| wm[q] + fm[q]).collect();
    mag.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m99 = mag[(0.99 * (mag.len() - 1) as f64) as usize];
    let sel: Vec<usize> = vox.iter().copied().filter(|&q| wm[q] + fm[q] > 0.1 * m99).collect();
    let stat = |est: &[f32], tru: &[f32], sel: &[usize]| -> serde_json::Value {
        let mut dv: Vec<f64> = sel.iter().filter(|&&q| tru[q].is_finite() && est[q].is_finite()).map(|&q| (est[q] - tru[q]) as f64).collect();
        if dv.is_empty() {
            return serde_json::Value::Null;
        }
        let n = dv.len() as f64;
        let mean = dv.iter().sum::<f64>() / n;
        let rmse = (dv.iter().map(|x| x * x).sum::<f64>() / n).sqrt();
        dv.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = dv[dv.len() / 2];
        let mut ab: Vec<f64> = dv.iter().map(|x| x.abs()).collect();
        ab.sort_by(|a, b| a.partial_cmp(b).unwrap());
        serde_json::json!({"n": dv.len(), "mean": mean, "median": med, "rmse": rmse, "abs_p50": ab[ab.len() / 2], "abs_p95": ab[(0.95 * (ab.len() - 1) as f64) as usize]})
    };
    let e_om = stat(&om_hz, &d.truth_om_hz, &sel);
    let e_r2 = stat(&r2m, &d.truth_r2, &sel);
    let e_ff = stat(&ffm, &d.truth_ff, &sel);
    println!("errors over {} signal voxels:\n  ω (Hz) {e_om}\n  R2* (1/s) {e_r2}\n  FF {e_ff}", sel.len());
    let wr = |name: &str, x: &[f32]| write_npy(&out.join(name), "<f4", &dims, true, &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    wr("maps_omega_hz.npy", &om_hz);
    wr("maps_r2.npy", &r2m);
    wr("maps_ff.npy", &ffm);
    wr("maps_water.npy", &wm);
    wr("maps_fat.npy", &fm);
    wr("truth_omega_hz.npy", &d.truth_om_hz);
    wr("truth_r2.npy", &d.truth_r2);
    wr("truth_ff.npy", &d.truth_ff);
    wr("support.npy", &d.support.iter().map(|&b| b as u8 as f32).collect::<Vec<_>>());
    let summary = serde_json::json!({"sim_dir": dir, "init": init, "dims": dims, "res": d.res, "vcoils": nvc, "knot_mm": knot_mm, "n_spline": nb, "iters": iters, "rho_iters": rho_iters, "terms": nterm, "precond": pre_name, "moco": moco_info, "interp_err": bi.max_err, "noise_floor": floor,
                                     "lam_rho": lam_rho, "lam_r2": lam_r2, "s_rho": s_rho, "n_vox": nx, "n_sets": nset, "evals": evals, "seconds": t0.elapsed().as_secs_f64(),
                                     "errors": {"omega_hz": e_om, "r2": e_r2, "ff": e_ff, "n_signal": sel.len()}});
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&summary).unwrap()).unwrap();
    let tools = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/b0/plot_mbq.py");
    let py = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.venv/bin/python");
    match std::process::Command::new(&py).arg("-I").arg(&tools).arg(&out).status() {
        Ok(s) if s.success() => println!("figures: {}", out.join("figs").display()),
        r => println!("figure script failed ({r:?})"),
    }
    println!("done in {:.0}s → {}", t0.elapsed().as_secs_f64(), out.display());
}

fn dotp(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn save(v: &[f64], k: usize, path: &std::path::Path, out: &std::path::Path) {
    write_npy(path, "<f8", &[v.len()], false, &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
    std::fs::write(out.join("state.json"), serde_json::json!({"iter": k}).to_string()).unwrap();
}
