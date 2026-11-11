//! Air map of a saved image volume (hasty_recon::airmap), offline: for validation on saved iterates and for seeding a
//! reconstruction (`recon_p2np --air-init`) from another one, e.g. a converged coarse pyramid level.
//!   airmap <in.npy> --out <air_map.npy> [--vox MM[,MM,MM]] [--bins N] [--tau T] [--release R] [--blur MM[,MM..]]
//!          [--ball MM] [--margin MM] [--ramp MM] [--no-slices] [--prior <npy> [--seeded]] [--iteration K]
//! in.npy (Fortran order, x fastest): complex images (X,Y,Z[,E[,F]]) (encodes E, frames F; --bins N splits the
//! frames into N respiratory bins of F/N consecutive frames, as recon_p2np orders them), or an f32 reference
//! magnitude (X,Y,Z). --vox: voxel size (default 2 mm). --prior: a previous map / mask / seed (as `--air-init`);
//! --seeded: the prior has not been applied to the input yet (a seed): applied first, as recon_p2np does.
//! Writes the map (X,Y,Z,4) f32 = [w, candidate, protected, R_s / L] and its .json sidecar.

use std::path::PathBuf;

use hasty_recon::airmap::{self, AirOpts};
use hasty_recon::tlock::FrameLayout;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut flag = |name: &str| args.iter().position(|a| a == name).map(|i| { args.remove(i); true }).unwrap_or(false);
    let no_slices = flag("--no-slices");
    let seeded = flag("--seeded");
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let out = PathBuf::from(take("--out").expect("--out <air_map.npy>"));
    let vox: [f64; 3] = take("--vox").map_or([2.0; 3], |s| {
        let v: Vec<f64> = s.split(',').map(|x| x.parse().expect("--vox MM[,MM,MM]")).collect();
        if v.len() == 1 { [v[0]; 3] } else { [v[0], v[1], v[2]] }
    });
    let bins: usize = take("--bins").map_or(1, |s| s.parse().expect("--bins N"));
    let mut o = AirOpts { slices: !no_slices, ..Default::default() };
    if let Some(v) = take("--tau") { o.tau = v.parse().expect("--tau T"); }
    if let Some(v) = take("--release") { o.release = v.parse().expect("--release R"); }
    if let Some(v) = take("--blur") { o.blur_mm = v.split(',').map(|x| x.parse().expect("--blur MM[,MM..]")).collect(); }
    if let Some(v) = take("--ball") { o.ball_mm = v.parse().expect("--ball MM"); }
    if let Some(v) = take("--margin") { o.margin_mm = v.parse().expect("--margin MM"); }
    if let Some(v) = take("--ramp") { o.ramp_mm = v.parse().expect("--ramp MM"); }
    let prior_path = take("--prior").map(PathBuf::from);
    let iteration: usize = take("--iteration").map_or(0, |s| s.parse().expect("--iteration K"));
    let input = PathBuf::from(args.first().expect("airmap <in.npy> --out <air_map.npy> [options]"));
    let t = std::time::Instant::now();
    let a = hasty_recon::data::read_npy(&input).unwrap();
    assert!(a.fortran || a.shape[..3].iter().filter(|&&n| n > 1).count() <= 1, "{}: need Fortran order (x fastest)", input.display());
    let d = [a.shape[0], a.shape[1], a.shape[2]];
    let prior = prior_path.map(|p| {
        let (m, desc) = airmap::load_seed(&p, d, vox, &o).unwrap();
        println!("prior: {desc}");
        m
    });
    let map = match a.descr.as_str() {
        "<c8" => {
            let mut h = a.c32s();
            let ne = a.shape.get(3).copied().unwrap_or(1);
            let nf: usize = a.shape.get(4..).map_or(1, |s| s.iter().product());
            assert!(nf.is_multiple_of(bins), "{nf} frames do not split into {bins} bins");
            let lay = FrameLayout::new(d, vox.map(|v| v * 1e-3), nf / bins, bins, ne, false);
            println!("{}: {:?}, {} encodes × {} frames ({} bins) [{:.1}s read]", input.display(), d, ne, nf, bins, t.elapsed().as_secs_f64());
            if let (true, Some(p)) = (seeded, prior.as_ref()) {
                airmap::apply(&mut h, &lay, p);
            }
            airmap::estimate(&h, &lay, &o, prior.as_ref(), iteration)
        }
        "<f4" => {
            let mut r: Vec<f32> = a.f32s().iter().map(|x| x.abs()).collect();
            if let (true, Some(p)) = (seeded, prior.as_ref()) {
                r.iter_mut().zip(&p.w).for_each(|(x, w)| *x *= 1.0 - w);
            }
            assert_eq!(r.len(), d.iter().product::<usize>(), "an f32 input is one reference magnitude volume (X,Y,Z)");
            airmap::estimate_from_reference(&r, d, vox, &o, prior.as_ref(), iteration)
        }
        x => panic!("{}: dtype {x}", input.display()),
    };
    let s = &map.stats;
    let nv = map.nv();
    println!(
        "air map ({:.1}s): level {:.4e}, low {} ({:.1} %), seeds {}, candidates {}, zeroed (w ≥ ½) {} ({:.1} % of the volume), added {}, released {}, protected {}",
        t.elapsed().as_secs_f64(), map.level, s.low, 100.0 * s.low as f64 / nv as f64, s.seeds, s.candidates, s.marked, 100.0 * s.marked as f64 / nv as f64, s.added, s.released, s.protected
    );
    let bytes: Vec<u8> = map.stacked().into_iter().flat_map(|x| x.to_le_bytes()).collect();
    hasty_phantom::volume::write_npy(&out, "<f4", &[d[0], d[1], d[2], 4], true, &bytes).unwrap();
    std::fs::write(out.with_extension("json"), serde_json::to_string_pretty(&airmap::sidecar(&map, &o, d, vox)).unwrap()).unwrap();
    println!("wrote {}", out.display());
}
