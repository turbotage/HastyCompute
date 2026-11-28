//! `bloch_parcel`: Bloch simulation of a small parcel of blood (a few hundred isochromats) carried along pathlines of a
//! 4D flow value while a designed flow sequence is played (`hasty_seq::bloch`; viewer `plotting_website/bloch`).
//! `bloch_designs` lists the sequence designs (`seqdesign` run directories) that can be played.

use std::path::PathBuf;

use hasty_seq::bloch::{ernst, simulate, BlochCfg, FlowDesign, SlabSpec, Tissue};
use hasty_value::{Tensor, Value};
use serde_json::{json, Value as Json};

use crate::commands::Registry;
use crate::flow::Flow;
use crate::Error;

fn inv<T>(m: impl Into<String>) -> Result<T, Error> {
    Err(Error::Invalid(m.into()))
}

/// sequence designs directory: env HASTY_SEQ_DIR, else `<data>/seq` (hasty_site::data)
fn seq_dir() -> PathBuf {
    std::env::var_os("HASTY_SEQ_DIR").filter(|v| !v.is_empty()).map(PathBuf::from).unwrap_or_else(|| hasty_site::data().join("seq"))
}

fn is_design(d: &std::path::Path) -> bool {
    ["final.json", "summary.json", "tr_grad.npy", "spoke_grad.npy"].iter().all(|f| d.join(f).is_file())
}

fn arr3(o: &Json, k: &str) -> Option<[f64; 3]> {
    let a = o[k].as_array()?;
    (a.len() == 3).then(|| [0, 1, 2].map(|i| a[i].as_f64().unwrap_or(0.0)))
}

/// `n` points of an R3 low-discrepancy lattice in a ball of radius `r`
fn ball(n: usize, r: f64, shift: f64) -> Vec<[f64; 3]> {
    let phi3 = 1.220_744_084_605_759_5f64;
    let al = [1.0 / phi3, 1.0 / (phi3 * phi3), 1.0 / (phi3 * phi3 * phi3)];
    let mut out = Vec::with_capacity(n);
    let mut j = 0usize;
    while out.len() < n && j < 100 * n + 100 {
        let p = al.map(|a| ((0.5 + shift + j as f64 * a).fract() - 0.5) * 2.0 * r);
        if p[0] * p[0] + p[1] * p[1] + p[2] * p[2] <= r * r {
            out.push(p);
        }
        j += 1;
    }
    out
}

/// piecewise-linear path of one spin, physical metres
struct Paths {
    dtp: f64,
    n: usize,
    /// [spin][step]
    p: Vec<Vec<[f32; 3]>>,
}

impl Paths {
    fn at(&self, j: usize, t: f64) -> [f64; 3] {
        let p = &self.p[j];
        if t <= 0.0 {
            return p[0].map(|x| x as f64);
        }
        let f = t / self.dtp;
        let i = f as usize;
        if i + 1 >= self.n {
            return p[self.n - 1].map(|x| x as f64);
        }
        let w = f - i as f64;
        [0, 1, 2].map(|a| p[i][a] as f64 * (1.0 - w) + p[i + 1][a] as f64 * w)
    }
}

pub fn register(r: &mut Registry) {
    r.register("bloch_designs", "sequence designs the Bloch parcel simulator can play (message JSON: dir, designs [{name, tr, te, n_enc, n_tr_steps}])", |_, _, _| {
        let dir = seq_dir();
        let mut v = vec![];
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                if is_design(&e.path()) {
                    if let Ok(d) = FlowDesign::load(&e.path()) {
                        v.push(json!({"name": d.name, "tr": d.tr(), "te": d.te(), "n_enc": d.n_enc, "n_tr_steps": d.n_tr, "dt": d.dt}));
                    }
                }
            }
        }
        v.sort_by(|a, b| a["name"].as_str().cmp(&b["name"].as_str()));
        Ok((json!({"dir": dir.to_string_lossy(), "designs": v}).to_string(), vec![]))
    });
    r.register(
        "bloch_parcel",
        "Bloch simulation of a parcel of spins carried along pathlines of a flow value (input 0, optional: without it the parcel moves with `velocity`) while a designed flow sequence is played. \
         options {design (name under data/hasty/seq, default flow_thorax_v2_1mm), rf: slab (windowed-sinc slab pulse with slice-select gradient and rephaser replacing the design's hard pulse, default) | hard, slab_thk_mm (175), slab_center_mm (0), slab_axis (z), tbw (6), rf_dur_us (800), n_tr (300), start: relaxed (M_z = 1 at the first TR, e.g. entering from the vena cava) | steady (spins first rest at their start positions for `preroll` TRs, default 600), preroll, \
         n_spins (200), radius_mm (3), center [x,y,z] voxel coordinates (plug mode: metres from the isocentre), t0 (release frame), advect: pathline (each spin on its own pathline) | rigid (parcel shape kept), substeps (16 per frame), velocity [vx,vy,vz] m/s (plug mode), \
         iso [x,y,z] voxel coordinates of the isocentre (default grid centre), flip_deg (15), rf_inc_deg (117), t1 (1.7), t2 (0.2), b0 (3), df_hz (0), b1_scale (1), concomitant (true), left_handed (true), mz0 (1), n_rot (scan_s / TR / encodes, scan_s 600), \
         per_tr (32 recorded steps per TR), detail [tr_from, tr_to, stride] (every `stride` simulation steps in these TRs), dt_us (simulation step, default 1: the design's 10 µs raster refined, gradients interpolated linearly, slab envelope re-sampled), mean_stride (steps, default 100 µs), seed}; \
         message JSON (timing, recorded samples out_tr / out_step, echo per TR, ...), outputs [m f32 [n_out, N, 3], pos f32 [n_out, N, 3] (m from the isocentre), mean f32 [n_mean, 3], g f32 [n_tr, n_g, 3] (T/m), mean_inc f32 [n_tr·n_mean_per_tr] (per-spin mean |M⊥|)]",
        |ctx, o, inp| {
            let name = o["design"].as_str().unwrap_or("flow_thorax_v2_1mm");
            if name.is_empty() || name.contains('/') || name.contains('\\') || name.starts_with('.') {
                return inv("bloch_parcel: design must be a plain directory name");
            }
            let dpath = seq_dir().join(name);
            if !is_design(&dpath) {
                return inv(format!("bloch_parcel: no design {name:?} in {}", seq_dir().display()));
            }
            let d0 = FlowDesign::load(&dpath).map_err(Error::Invalid)?;
            let f = |k: &str, dflt: f64| o[k].as_f64().unwrap_or(dflt);
            // RF: the simulator's slab pulse (windowed sinc, TBW 6, 0.8 ms, along z) or the design's non-selective hard pulse
            let slab_spec = match o["rf"].as_str().unwrap_or("slab") {
                "slab" => {
                    let mut sp = SlabSpec::thorax(f("slab_thk_mm", 175.0) * 1e-3);
                    sp.tbw = f("tbw", 6.0);
                    sp.dur = f("rf_dur_us", 800.0) * 1e-6;
                    sp.axis = match o["slab_axis"].as_str().unwrap_or("z") {
                        "x" => 0,
                        "y" => 1,
                        _ => 2,
                    };
                    Some(sp)
                }
                "hard" => None,
                r => return inv(format!("bloch_parcel: rf must be slab | hard, got {r:?}")),
            };
            let d = match &slab_spec {
                Some(sp) => d0.with_slab(sp),
                None => d0,
            };
            // simulation raster: the design's gradient raster (10 µs) refined to dt_us (default 1 µs)
            let dt_us = o["dt_us"].as_f64().unwrap_or(1.0);
            if !(0.1..=100.0).contains(&dt_us) {
                return inv("bloch_parcel: dt_us must be within 0.1 … 100");
            }
            let refine = ((d.dt / (dt_us * 1e-6)).round() as usize).clamp(1, 100);
            let d = d.refine(refine);
            let steady = match o["start"].as_str().unwrap_or("relaxed") {
                "relaxed" => false,
                "steady" => true,
                s => return inv(format!("bloch_parcel: start must be relaxed | steady, got {s:?}")),
            };
            let cfg = BlochCfg {
                flip: f("flip_deg", 15.0).to_radians(),
                rf_inc: f("rf_inc_deg", 117.0).to_radians(),
                b1_scale: f("b1_scale", 1.0),
                slab_center: f("slab_center_mm", 0.0) * 1e-3,
                tissue: Tissue { t1: f("t1", 1.7), t2: f("t2", 0.2) },
                b0: f("b0", 3.0),
                df: f("df_hz", 0.0),
                concomitant: o["concomitant"].as_bool().unwrap_or(true),
                left_handed: o["left_handed"].as_bool().unwrap_or(true),
                n_tr: o["n_tr"].as_u64().unwrap_or(300).clamp(1, 20_000) as usize,
                preroll: o["preroll"].as_u64().map(|x| x as usize).unwrap_or(if steady { 600 } else { 0 }).min(20_000),
                n_rot: o["n_rot"].as_u64().map(|x| x as usize).unwrap_or(((f("scan_s", 600.0) / d.tr()).floor() as usize / d.n_enc).max(1)).max(1),
                mz0: f("mz0", 1.0),
                per_tr: o["per_tr"].as_u64().unwrap_or(32).clamp(2, 2000) as usize,
                detail: o["detail"].as_array().filter(|a| a.len() == 3).map(|a| (a[0].as_u64().unwrap_or(0) as usize, a[1].as_u64().unwrap_or(0) as usize, a[2].as_u64().unwrap_or(5).max(1) as usize)),
                mean_stride: o["mean_stride"].as_u64().map_or(((100e-6 / d.dt).round() as usize).max(1), |x| x.max(1) as usize),
            };
            if cfg.tissue.t1 <= 0.0 || cfg.tissue.t2 <= 0.0 {
                return inv("bloch_parcel: t1 and t2 must be positive");
            }
            let n = o["n_spins"].as_u64().unwrap_or(200).clamp(1, 5000) as usize;
            let radius = f("radius_mm", 3.0) * 1e-3;
            let offs = ball(n, radius, (o["seed"].as_u64().unwrap_or(1) % 997) as f64 / 997.0);
            let dur = cfg.n_tr as f64 * d.tr();
            ctx.progress(0.0, "paths");

            // paths in metres from the isocentre
            let flow = inp.first().map(|v| Flow::from_value(v)).transpose()?;
            let (paths, iso, spacing, dims, exited) = if let Some(fl) = &flow {
                let c = arr3(o, "center").ok_or_else(|| Error::Invalid("bloch_parcel: options.center [x,y,z] voxel coordinates".into()))?;
                let iso = arr3(o, "iso").unwrap_or_else(|| fl.dims.map(|n| (n as f64 - 1.0) / 2.0));
                let sp = fl.spacing;
                let t0 = f("t0", 0.0);
                let sub = f("substeps", 16.0);
                let rigid = o["advect"].as_str() == Some("rigid");
                let seeds: Vec<[f64; 3]> = if rigid { vec![c] } else { offs.iter().map(|d| [0, 1, 2].map(|a| c[a] + d[a] / sp[a])).collect() };
                let (hs, vp) = fl.parcel_paths(t0, dur + d.tr(), &seeds, sub);
                let phys = |v: [f64; 3]| [0, 1, 2].map(|a| (v[a] - iso[a]) * sp[a]);
                let nst = vp[0].0.len();
                let p: Vec<Vec<[f32; 3]>> = if rigid {
                    (0..n).map(|j| vp[0].0.iter().map(|&v| { let q = phys(v); [0, 1, 2].map(|a| (q[a] + offs[j][a]) as f32) }).collect()).collect()
                } else {
                    vp.iter().map(|(l, _)| l.iter().map(|&v| phys(v).map(|x| x as f32)).collect()).collect()
                };
                let exited = vp.iter().filter(|(_, e)| e.is_some()).count();
                (Paths { dtp: hs, n: nst, p }, Some(iso), Some(sp), Some(fl.dims), exited)
            } else {
                let c = arr3(o, "center").unwrap_or([0.0; 3]);
                let v = arr3(o, "velocity").unwrap_or([0.0; 3]);
                let p = (0..n).map(|j| (0..2).map(|k| [0, 1, 2].map(|a| (c[a] + offs[j][a] + v[a] * dur * k as f64) as f32)).collect()).collect();
                (Paths { dtp: dur + d.tr(), n: 2, p }, None, None, None, 0)
            };
            let steady_rate = ctx.cancel.clone();
            let t_start = std::time::Instant::now();
            let out = simulate(&d, &cfg, n, &|j, t| paths.at(j, t), &mut |k, nn| {
                if k % 8 == 0 {
                    ctx.progress(0.05 + 0.95 * k as f64 / nn as f64, &format!("TR {k}/{nn}"));
                }
                !steady_rate.load(std::sync::atomic::Ordering::Relaxed)
            })
            .ok_or_else(|| Error::Invalid("bloch_parcel: cancelled".into()))?;
            let n_out = out.tr.len();
            // flip-angle profile of the slab pulse along its axis (resting relaxed spins, no relaxation, 0 … ±300 mm around the centre)
            let (prof_z, prof_flip): (Vec<f64>, Vec<f64>) = match d.slab {
                Some(g) => {
                    let zs: Vec<f64> = (-150..=150).map(|i| cfg.slab_center + i as f64 * 2e-3).collect();
                    let pc = BlochCfg { n_tr: 1, preroll: 0, per_tr: 2, detail: None, concomitant: false, tissue: hasty_seq::bloch::Tissue { t1: 1e9, t2: 1e9 }, mz0: 1.0, df: 0.0, ..cfg.clone() };
                    let ax = g.axis;
                    match simulate(&d, &pc, zs.len(), &|j, _| { let mut v = [0.0; 3]; v[ax] = zs[j]; v }, &mut |_, _| true) {
                        Some(po) => {
                            let k = po.step.iter().position(|&st| st as usize == d.rf_steps().1).unwrap_or(0);
                            let fl: Vec<f64> = (0..zs.len()).map(|j| { let m = &po.m[(k * zs.len() + j) * 3..(k * zs.len() + j) * 3 + 3]; (m[0] as f64).hypot(m[1] as f64).atan2(m[2] as f64).to_degrees() }).collect();
                            (zs, fl)
                        }
                        None => (vec![], vec![]),
                    }
                }
                None => (vec![], vec![]),
            };
            let rf_mx = d.rf_env.iter().fold(1e-30f64, |a, &b| a.max(b.abs()));
            let rf_env_n: Vec<f64> = d.rf_env.iter().map(|v| v / rf_mx).collect();
            let n_mpt = d.n_tr.div_ceil(cfg.mean_stride);
            let n_g = d.n_tr / out.g_stride + 1;
            let (a, b) = d.rf_steps();
            let ideal = ernst(cfg.flip * cfg.b1_scale, d.tr(), d.te(), &cfg.tissue);
            let mut meta = json!({
                "design": d.name, "i_peak": d.i_ro + d.i_kmax(), "rf_env": rf_env_n, "slab_profile": {"z": prof_z, "flip_deg": prof_flip}, "dt": d.dt, "dt_us": d.dt * 1e6, "tr": d.tr(), "te": d.te(), "n_enc": d.n_enc, "n_steps": d.n_tr,
                "i_ro": d.i_ro, "i_adc": d.i_adc, "i_enc": d.i_enc, "i_tail": d.i_tail, "rf_steps": [a, b], "t_rf": d.t_rf, "rf_dur": d.rf_dur,
            });
            let more = json!({
                "n_tr": cfg.n_tr, "preroll": cfg.preroll, "n_spins": n, "n_out": n_out, "n_mean_per_tr": n_mpt, "mean_stride": cfg.mean_stride,
                "g_stride": out.g_stride, "n_g": n_g,
                "flip_deg": cfg.flip.to_degrees(), "rf_inc_deg": cfg.rf_inc.to_degrees(), "t1": cfg.tissue.t1, "t2": cfg.tissue.t2, "b0": cfg.b0, "df_hz": cfg.df, "b1_scale": cfg.b1_scale,
                "concomitant": cfg.concomitant, "left_handed": cfg.left_handed, "start": if steady { "steady" } else { "relaxed" },
                "ernst": ideal, "mode": if flow.is_some() { "pathline" } else { "plug" },
                "slab": d.slab.map(|g| json!({"axis": g.axis, "g_ss": g.g_ss, "thickness": g.thickness, "tbw": g.tbw, "center": cfg.slab_center})), "n_rot": cfg.n_rot,
                "iso": iso, "spacing": spacing, "dims": dims, "spins_left_volume": exited,
                "out_tr": out.tr, "out_step": out.step, "echo": out.echo, "echo_inc": out.echo_inc, "rf_phase": out.rf_phase, "encode": out.encode, "rot": out.rot,
                "seconds": t_start.elapsed().as_secs_f64(),
            });
            meta.as_object_mut().unwrap().extend(more.as_object().unwrap().clone());
            let outs = vec![
                Value::Tensor(Tensor::from_slice(&[n_out, n, 3], &out.m)?),
                Value::Tensor(Tensor::from_slice(&[n_out, n, 3], &out.pos)?),
                Value::Tensor(Tensor::from_slice(&[cfg.n_tr * n_mpt, 3], &out.mean)?),
                Value::Tensor(Tensor::from_slice(&[cfg.n_tr, n_g, 3], &out.g)?),
                Value::Tensor(Tensor::from_slice(&[cfg.n_tr * n_mpt], &out.mean_inc)?),
            ];
            Ok((meta.to_string(), outs))
        },
    );
}
