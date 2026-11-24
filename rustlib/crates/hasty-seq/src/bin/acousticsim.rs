//! Acoustic noise simulator: Pulseq `.seq` → what it sounds like (research/acoustic_sim.md).
//!
//!     acousticsim --seq file.seq --out DIR [options]
//!
//! The sequence is split into TRs (at the RF blocks, or `--tr-ms`). A short signature of every TR's sound (octave
//! band levels + dB(A), from the TR repeated periodically) is computed over the whole sequence; events are the
//! start, the loudest stretch and the points where the sound changes (median signature of `--median` TRs after vs
//! before, > `--thr-db`). Around each event `--ntr` TRs are simulated and written as mono WAV (48 kHz) + arrays +
//! summary; figures: tools/seq/plot_acoustic.py DIR.
//!
//! Which part is simulated:
//!   (default)            automatic events (at most `--max-events`)
//!   --at 12.5,60         windows centred on these sequence times (s); only these
//!   --tr-start 0,5000    windows starting at these TR indices; only these
//!   --ntr N              TRs per window (24)
//!   --overview           with --at / --tr-start: still compute the whole-sequence signatures (37 s per 50k TRs)
//!   --rf-per-tr K        RF pulses per TR (2 for a spin echo: excitation + refocusing; default 1)
//!   --lead S             lead-in before a window (s, 0.15) so the ringing already present is right
//! Model: `--model file.json` (see --dump-model), else the toy 3 T model (uncalibrated: levels indicative only).
//!   --b0 T               field (scales the force)
//!   --reverb off|T[,dB]  reverberation: RT60 (s) and level re direct sound (dB); default 0.4,-12
//!   --protection         apply the hearing-protection attenuation table (what the patient hears)
//!   --girf file.json     play the gradients through a measured GIRF first (`--girf-x` thermal state, 0)
//! WAV level: all files share one gain (loudest peak at 0.9 full scale) so loudness compares between windows;
//!   --each               normalise every file on its own
//!   --fs-db X            absolute: full scale = a sine of X dB SPL (rms), no normalisation (clipping is clamped);
//!                        the same X for several runs makes their loudness comparable. --also-norm: also write <name>_norm.wav
//! Writes <out>/summary.json, signatures.npy (n_tr × 8: octave bands 125 Hz…8 kHz, dB(A)), tr_t.npy (TR start, s),
//! and per window w<i>_<kind>_tr<start>.{wav, p.npy (Pa, 48 kHz), pa.npy (A-weighted), g.npy (T/m, raster)}.

use std::path::PathBuf;

use hasty_seq::acoustic::*;
use hasty_seq::girfmeas;
use hasty_seq::pulseq::SeqFile;
use hasty_seq::wave::Grad;
use rustfft::FftPlanner;
use serde_json::json;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let opt = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let flag = |k: &str| args.iter().any(|a| a == k);
    if let Err(e) = run(&opt, &flag) {
        eprintln!("acousticsim: {e}");
        std::process::exit(1);
    }
}

fn num<T: std::str::FromStr>(s: Option<String>, d: T, name: &str) -> Result<T, String> {
    match s {
        None => Ok(d),
        Some(v) => v.parse().map_err(|_| format!("{name}: cannot parse '{v}'")),
    }
}

fn list<T: std::str::FromStr>(s: Option<String>, name: &str) -> Result<Vec<T>, String> {
    s.map(|v| v.split(',').map(|x| x.trim().parse().map_err(|_| format!("{name}: cannot parse '{x}'"))).collect()).unwrap_or(Ok(vec![]))
}

const AX: [&str; 3] = ["x", "y", "z"];

struct Window {
    kind: String,
    start: usize,
    note: String,
}

fn run(opt: &dyn Fn(&str) -> Option<String>, flag: &dyn Fn(&str) -> bool) -> Result<(), String> {
    let mut model = match opt("--model") {
        Some(p) => AcousticModel::from_json(&std::fs::read_to_string(&p).map_err(|e| format!("{p}: {e}"))?)?,
        None => AcousticModel::toy_3t(),
    };
    if let Some(b) = opt("--b0") {
        model.b0 = b.parse().map_err(|_| "--b0")?;
    }
    model.use_protection = flag("--protection");
    // reverberation: --reverb off | RT60[,level dB re direct]; default 0.4 s, -12 dB (the noise outlasts the gradients
    // by 300 – 500 ms, Ravicz et al. 2000); a reverb given in the model file is used when --reverb is absent
    match opt("--reverb").as_deref() {
        Some("off") => model.reverb = None,
        Some(v) => {
            let p: Vec<f64> = v.split(',').map(|x| x.trim().parse().map_err(|_| format!("--reverb: cannot parse '{x}'"))).collect::<Result<_, _>>()?;
            model.reverb = Some((p[0], p.get(1).copied().unwrap_or(-12.0)));
        }
        None if model.reverb.is_none() => model.reverb = Some((0.4, -12.0)),
        None => {}
    }
    let out = PathBuf::from(opt("--out").ok_or("--out DIR required")?);
    std::fs::create_dir_all(&out).map_err(|e| e.to_string())?;
    if flag("--dump-model") {
        let p = out.join("model_toy_3t.json");
        std::fs::write(&p, serde_json::to_string_pretty(&model.to_json()).unwrap()).map_err(|e| e.to_string())?;
        println!("wrote {}", p.display());
        return Ok(());
    }
    let seq_path = opt("--seq").ok_or("--seq FILE required")?;
    let ntr: usize = num(opt("--ntr"), 24, "--ntr")?;
    let lead: f64 = num(opt("--lead"), 0.15, "--lead")?;
    let w_med: usize = num(opt("--median"), 16, "--median")?;
    let thr: f64 = num(opt("--thr-db"), 3.0, "--thr-db")?;
    let max_ev: usize = num(opt("--max-events"), 6, "--max-events")?;
    let at: Vec<f64> = list(opt("--at"), "--at")?;
    let tr_start: Vec<usize> = list(opt("--tr-start"), "--tr-start")?;
    let tr_ms: Option<f64> = opt("--tr-ms").map(|v| v.parse().map_err(|_| "--tr-ms")).transpose()?;

    let text = std::fs::read_to_string(&seq_path).map_err(|e| format!("{seq_path}: {e}"))?;
    let mut sf = SeqFile::parse(&text)?;
    drop(text);
    let trs = tr_split(&sf, tr_ms, num(opt("--rf-per-tr"), 1, "--rf-per-tr")?)?;
    let nt = trs.len();
    let total = trs.last().map_or(0.0, |t| t.t1);
    println!("{seq_path}: {} blocks, {nt} TRs, {total:.2} s; model: {}{}", sf.blocks.len(), model.name, if model.use_protection { " + hearing protection" } else { "" });
    let girf = match opt("--girf") {
        Some(p) => Some(girfmeas::from_measurement(std::path::Path::new(&p), num(opt("--girf-x"), 0.0, "--girf-x")?)?),
        None => None,
    };
    let played = |g: Grad| -> Grad {
        match &girf {
            None => g,
            Some(gf) => {
                let (a, _) = gf.apply(&g);
                let pre = hasty_seq::girf::Girf::PRE;
                Grad { dt: g.dt, t0: g.t0, g: a.g[pre..pre + g.len()].to_vec() }
            }
        }
    };

    // ── signatures of every TR (the whole sequence, serial: SeqFile is swapped in place)
    // (skipped when the windows are given by hand, unless --overview asks for the whole-sequence figure)
    let manual = !at.is_empty() || !tr_start.is_empty();
    let mut sig: Vec<[f64; NSIG]> = Vec::with_capacity(nt);
    if !manual || flag("--overview") {
        let mut planner = FftPlanner::<f64>::new();
        for t in &trs {
            let g = played(grad_range(&mut sf, t.b0, t.b1, t.t0));
            sig.push(tr_signature(&model, &g, &mut planner));
        }
        let tr_t: Vec<f32> = trs.iter().map(|t| t.t0 as f32).collect();
        let flat: Vec<f32> = sig.iter().flat_map(|r| r.iter().map(|v| *v as f32)).collect();
        write_npy_f32(&out.join("signatures.npy"), &[nt, NSIG], &flat).map_err(|e| e.to_string())?;
        write_npy_f32(&out.join("tr_t.npy"), &[nt], &tr_t).map_err(|e| e.to_string())?;
    }

    // ── events and windows
    let events = if manual { vec![] } else { detect_events(&sig, w_med, thr, max_ev) };
    let mut wins: Vec<Window> = vec![];
    if at.is_empty() && tr_start.is_empty() {
        for e in &events {
            let start = if e.kind == "start" { 0 } else { e.tr.saturating_sub(ntr / 2) };
            let band = if e.col < 7 { format!("{} Hz band", BANDS[e.col]) } else { "dB(A)".into() };
            wins.push(Window {
                kind: e.kind.into(),
                start,
                note: if e.kind == "change" { format!("{band} changes {:.1} dB (dB(A) {:.1} → {:.1})", e.score, e.la_before, e.la_after) } else { format!("dB(A) {:.1}", e.la_after) },
            });
        }
    }
    for t in &at {
        let k = trs.iter().position(|x| x.t1 > *t).unwrap_or(nt - 1);
        wins.push(Window { kind: "at".into(), start: k.saturating_sub(ntr / 2), note: format!("centred on t = {t} s") });
    }
    for &k in &tr_start {
        wins.push(Window { kind: "tr".into(), start: k, note: format!("from TR {k}") });
    }
    for w in &mut wins {
        w.start = w.start.min(nt.saturating_sub(ntr));
    }
    if wins.is_empty() {
        return Err("no window to simulate".into());
    }

    // ── simulate
    struct Done {
        w: Window,
        sim: Sim,
        i0: usize,
        i1: usize,
        g: Vec<[f64; 3]>,
        g_dt: f64,
        summary: serde_json::Value,
    }
    let mut done: Vec<Done> = vec![];
    for (wi, w) in wins.into_iter().enumerate() {
        let (s, e) = (w.start, (w.start + ntr).min(nt));
        let mut s0 = s;
        while s0 > 0 && trs[s].t0 - trs[s0].t0 < lead {
            s0 -= 1;
        }
        let g = played(grad_range(&mut sf, trs[s0].b0, trs[e - 1].b1, trs[s0].t0));
        let sim = simulate(&model, &g, 0.2)?;
        let (t_a, t_b) = (trs[s].t0, trs[e - 1].t1);
        let i0 = (((t_a - sim.t0) * sim.fs).round() as usize).min(sim.p.len());
        let i1 = (((t_b - sim.t0) * sim.fs).round() as usize).min(sim.p.len());
        let pw = &sim.pa[i0..i1];
        let per_tr: Vec<f64> = (s..e)
            .map(|k| {
                let a = (((trs[k].t0 - t_a) * sim.fs) as usize).min(pw.len());
                let b = (((trs[k].t1 - t_a) * sim.fs) as usize).min(pw.len());
                db_spl(mean_sq(&pw[a..b.max(a)]))
            })
            .collect();
        let durs: Vec<f64> = (s..e).map(|k| trs[k].t1 - trs[k].t0).collect();
        let (dmin, dmax) = (durs.iter().cloned().fold(f64::INFINITY, f64::min), durs.iter().cloned().fold(0.0, f64::max));
        let tr_period = (dmax - dmin < 0.01 * dmax).then_some(0.5 * (dmin + dmax));
        let lns = lines(&model, &sim, i0, i1, tr_period, 12);
        let tob = third_octave(&sim.p[i0..i1], sim.fs);
        let peak = sim.p[i0..i1].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let name = format!("w{wi:02}_{}_tr{s}", w.kind);
        let gi0 = ((t_a - g.t0) / g.dt).round() as usize;
        let gi1 = (((t_b - g.t0) / g.dt).round() as usize + 1).min(g.len());
        let summary = json!({
            "name": name, "kind": w.kind, "note": w.note, "tr_first": s, "tr_last": e - 1, "t_start": t_a, "t_end": t_b,
            "tr_period": tr_period, "lead_s": trs[s].t0 - trs[s0].t0, "fs": sim.fs, "g_dt": g.dt,
            "laeq": db_spl(mean_sq(pw)), "lafmax": l_fast_max(pw, sim.fs), "lpeak_db": db_spl(peak * peak),
            // IEC 60601-2-33: hearing protection required above 99 dB(A) rms or 140 dB unweighted peak; ISMRM/SMRT
            // guideline: protection should bring the level at the eardrum below 85 dB(A) (levels as simulated: with
            // --protection they are at the eardrum, otherwise at the ear without protection)
            "limits": {"iec_99_dba_exceeded": db_spl(mean_sq(pw)) > 99.0, "iec_140_peak_exceeded": db_spl(peak * peak) > 140.0, "smrt_85_dba_exceeded": db_spl(mean_sq(pw)) > 85.0, "with_protection": model.use_protection},
            "leq_unweighted": db_spl(mean_sq(&sim.p[i0..i1])),
            "per_tr_laeq": per_tr, "tr_t0": (s..e).map(|k| trs[k].t0 - t_a).collect::<Vec<_>>(),
            "third_octave": tob.iter().map(|(f, l)| json!([f, l])).collect::<Vec<_>>(),
            "lines": lns.iter().map(|l| json!({
                "f": l.f, "db_spl": l.db_spl, "db_a": l.db_a, "axis": AX[l.axis],
                "resonance": {"axis": AX[l.res.0], "f0": l.res.1, "q": l.res.2, "offset_half_bw": l.res.3},
                "tr_harmonic": l.harm,
            })).collect::<Vec<_>>(),
        });
        let lim = if db_spl(mean_sq(pw)) > 99.0 { "  > IEC 99 dB(A)" } else if db_spl(mean_sq(pw)) > 85.0 { "  > SMRT 85 dB(A)" } else { "" };
        println!(
            "  [{wi}] {:8} TR {s}..{} t {t_a:.3}–{t_b:.3} s  LAeq {:.1} dB(A)  LAFmax {:.1}  Lpeak {:.1} dB{lim}   {}",
            w.kind,
            e - 1,
            summary["laeq"].as_f64().unwrap(),
            summary["lafmax"].as_f64().unwrap(),
            summary["lpeak_db"].as_f64().unwrap(),
            w.note
        );
        for l in lns.iter().take(4) {
            println!(
                "        {:7.1} Hz  {:5.1} dB(A)  axis {}  nearest resonance {} {:.0} Hz ({:+.1} half-bw){}",
                l.f,
                l.db_a,
                AX[l.axis],
                AX[l.res.0],
                l.res.1,
                l.res.3,
                l.harm.map_or(String::new(), |h| format!("  = {h:.2} × TR rate"))
            );
        }
        done.push(Done { w: Window { kind: w.kind, start: s, note: w.note }, g: g.g[gi0.min(gi1)..gi1].to_vec(), g_dt: g.dt, sim, i0, i1, summary });
    }

    // ── WAV gain and files
    let peak_all = done.iter().map(|d| d.sim.p[d.i0..d.i1].iter().fold(0.0f64, |m, v| m.max(v.abs()))).fold(0.0, f64::max);
    let abs_fs = opt("--fs-db").map(|v| v.parse::<f64>().map_err(|_| "--fs-db")).transpose()?;
    let common = if peak_all > 0.0 { 0.9 / peak_all } else { 1.0 };
    let mut summaries = vec![];
    for d in &done {
        let name = d.summary["name"].as_str().unwrap().to_string();
        let seg = &d.sim.p[d.i0..d.i1];
        let own = seg.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let gain = match abs_fs {
            Some(db) => 1.0 / (P_REF * 10f64.powf(db / 20.0) * std::f64::consts::SQRT_2),
            None if flag("--each") && own > 0.0 => 0.9 / own,
            None => common,
        };
        write_wav(&out.join(format!("{name}.wav")), seg, d.sim.fs, gain).map_err(|e| e.to_string())?;
        if abs_fs.is_some() && flag("--also-norm") && own > 0.0 {
            write_wav(&out.join(format!("{name}_norm.wav")), seg, d.sim.fs, 0.9 / own).map_err(|e| e.to_string())?;
        }
        let f32s = |v: &[f64]| v.iter().map(|x| *x as f32).collect::<Vec<f32>>();
        write_npy_f32(&out.join(format!("{name}_p.npy")), &[seg.len()], &f32s(seg)).map_err(|e| e.to_string())?;
        let pa = &d.sim.pa[d.i0..d.i1];
        write_npy_f32(&out.join(format!("{name}_pa.npy")), &[pa.len()], &f32s(pa)).map_err(|e| e.to_string())?;
        let gf: Vec<f32> = d.g.iter().flat_map(|v| v.iter().map(|x| *x as f32)).collect();
        write_npy_f32(&out.join(format!("{name}_g.npy")), &[d.g.len(), 3], &gf).map_err(|e| e.to_string())?;
        let mut s = d.summary.clone();
        s["wav"] = json!(format!("{name}.wav"));
        s["wav_gain_pa_to_fs"] = json!(gain);
        s["wav_full_scale_peak_db_spl"] = json!(db_spl(1.0 / (gain * gain)));
        let _ = (&d.w.kind, d.w.start, &d.w.note, d.g_dt);
        summaries.push(s);
    }
    let sumj = json!({
        "seq": seq_path, "n_tr": nt, "duration_s": total, "model": model.to_json(), "protection_applied": model.use_protection,
        "calibrated": false,
        "events": events.iter().map(|e| json!({"tr": e.tr, "t": trs[e.tr].t0, "kind": e.kind, "score_db": e.score, "la_before": e.la_before, "la_after": e.la_after, "col": e.col})).collect::<Vec<_>>(),
        "bands_hz": BANDS, "windows": summaries,
        "wav_gain": if abs_fs.is_some() { "absolute" } else if flag("--each") { "per file" } else { "common" },
    });
    std::fs::write(out.join("summary.json"), serde_json::to_string_pretty(&sumj).unwrap()).map_err(|e| e.to_string())?;
    println!("wrote {} windows to {} (figures: tools/seq/plot_acoustic.py {})", done.len(), out.display(), out.display());
    Ok(())
}
