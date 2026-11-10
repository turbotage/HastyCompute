//! Shim study on a phantom's unshimmed field (research/shimming.md): which shim hardware and objective leave the
//! smallest off-resonance span over the heart and over the signal-carrying voxels, and the reconstruction's segment
//! count that span implies.
//!
//!   shimstudy <dir from tools/phantom/shim_field.py> [--b0 3] [--t-ro 3.56e-3] [--imax 2] [--sub 2] [--out <json>]
//!
//! Hardware: solid harmonics of degree ≤ 1 / 2 / 3 (unconstrained; degree 1 also stands for the gradient offsets), the
//! 32 receive loops of the simulated array as DC shim elements (integrated receive / shim, "AC/DC"; |I| ≤ imax).
//! Objectives: prep_sim's shim box (uniform), the heart ROI (uniform), all signal voxels weighted by PD; least squares
//! (p = 2) or a span-targeting p = 8. Reported per configuration: SD / 1–99 % span / peak-to-peak over the heart ROI and
//! over the signal voxels (PD-weighted), the segment count L of the signal span (recon_p2np's --nseg auto rule), and the
//! same after the reconstruction absorbs the remaining linear part exactly (per-sample k-shift).

use std::path::{Path, PathBuf};

use hasty_phantom::coils::body_arrays;
use hasty_phantom::shimfit::{field_stats, fit, harmonic_basis, loop_basis, segments_for, ShimBasis, ShimOpts};
use hasty_phantom::volume::Volume;

/// C-order [z][y][x] npy → x-fastest data and dims [nx, ny, nz]
fn npy_zyx<T: Copy>(path: &Path, conv: impl Fn(&[u8]) -> T, es: usize) -> ([usize; 3], Vec<T>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (hl, hs) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let h = std::str::from_utf8(&b[hs..hs + hl]).unwrap();
    assert!(!h.contains("'fortran_order': True"), "{}: C order expected", path.display());
    let shape: Vec<usize> = h.split("'shape': (").nth(1).unwrap().split(')').next().unwrap().split(',').filter_map(|x| x.trim().parse().ok()).collect();
    let n: usize = shape.iter().product();
    let d = &b[hs + hl..];
    ([shape[2], shape[1], shape[0]], (0..n).map(|q| conv(&d[q * es..])).collect())
}

fn f32s(path: &Path) -> ([usize; 3], Vec<f32>) {
    npy_zyx(path, |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]), 4)
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |k: &str| -> Option<String> { args.iter().position(|a| a == k).map(|i| { args.remove(i); args.remove(i) }) };
    let b0: f64 = take("--b0").map_or(3.0, |v| v.parse().unwrap());
    let t_ro: f64 = take("--t-ro").map_or(3.56e-3, |v| v.parse().unwrap());
    let imax: f64 = take("--imax").map_or(2.0, |v| v.parse().unwrap());
    let out: Option<PathBuf> = take("--out").map(PathBuf::from);
    // every sub-th voxel per axis (the field is smooth on the shim scale; keeps the fits fast)
    let sub: usize = take("--sub").map_or(2, |v| v.parse().unwrap());
    let dir = PathBuf::from(args.first().expect("usage: shimstudy <dir> [--b0 3] [--t-ro s] [--imax A] [--out json]"));
    let geom: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("geom.json")).unwrap()).unwrap();
    let v3 = |k: &str| -> [f64; 3] { let a = geom[k].as_array().unwrap(); [a[0].as_f64().unwrap(), a[1].as_f64().unwrap(), a[2].as_f64().unwrap()] };
    let (org, vox, fovc) = (v3("origin_m"), geom["voxel_m"].as_f64().unwrap(), v3("fov_center_m"));
    let hz_per_ppm = 42.577e6 * b0 * 1e-6;
    let (dims, field) = f32s(&dir.join("field_ppm.npy"));
    let (_, signal) = f32s(&dir.join("signal.npy"));
    let (_, heart) = f32s(&dir.join("heart.npy"));
    let (_, boxm) = f32s(&dir.join("box.npy"));
    // points: every voxel that carries signal or lies in the heart ROI or the shim box
    let sel: Vec<usize> = (0..field.len())
        .filter(|&q| (q % dims[0]) % sub == 0 && ((q / dims[0]) % dims[1]) % sub == 0 && (q / (dims[0] * dims[1])) % sub == 0)
        .filter(|&q| signal[q] > 0.0 || heart[q] > 0.0 || boxm[q] > 0.0)
        .collect();
    let pos: Vec<[f64; 3]> = sel.iter().map(|&q| {
        let (i, j, k) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        [org[0] + i as f64 * vox, org[1] + j as f64 * vox, org[2] + k as f64 * vox]
    }).collect();
    let f_hz: Vec<f64> = sel.iter().map(|&q| field[q] as f64 * hz_per_ppm).collect();
    let w_sig: Vec<f64> = sel.iter().map(|&q| signal[q] as f64).collect();
    let w_heart: Vec<f64> = sel.iter().map(|&q| heart[q] as f64).collect();
    let w_box: Vec<f64> = sel.iter().map(|&q| boxm[q] as f64).collect();
    println!("{} points ({} heart ROI, {} signal) at {:.0} mm (every {sub}th), B0 {b0} T, readout {:.2} ms", sel.len(), w_heart.iter().filter(|&&w| w > 0.0).count(), w_sig.iter().filter(|&&w| w > 0.0).count(), vox * 1e3 * sub as f64, t_ro * 1e3);
    // the simulated receive array (as simphantom: 4 × 4 anterior + posterior loops of r = 45 mm over centre z ± 17 cm)
    let (bdims, body) = npy_zyx(&dir.join("body_coils.npy"), |b| b[0], 1);
    let bo = v3("body_coils_origin_m");
    let bh = geom["body_coils_voxel_m"].as_f64().unwrap();
    let mut aff = [[0.0; 4]; 4];
    for a in 0..3 {
        aff[a][a] = bh;
        aff[a][3] = bo[a];
    }
    aff[3][3] = 1.0;
    let bodyv = Volume { dims: bdims, affine: aff, data: body };
    let loops = body_arrays(&bodyv, 4, 4, 0.045, fovc[2] - 0.17, fovc[2] + 0.17);
    println!("receive array: {} loops, DC shim limit ±{imax} A", loops.len());
    let inf = [f64::INFINITY; 4];
    let h1 = || harmonic_basis(&pos, fovc, 0.1, 1, inf);
    let h2 = || harmonic_basis(&pos, fovc, 0.1, 2, inf);
    let h3 = || harmonic_basis(&pos, fovc, 0.1, 3, inf);
    let lp = || loop_basis(&pos, &loops, imax);
    let none = ShimBasis::default();
    let p2 = ShimOpts::default();
    let p8 = ShimOpts { p: 8.0, ..Default::default() };
    type Cfg<'a> = (&'a str, ShimBasis, &'a [f64], ShimOpts);
    let cfgs: Vec<Cfg> = vec![
        ("no shim", none, &w_sig, p2),
        ("SH≤1, box", h1(), &w_box, p2),
        ("SH≤2, box (prep_sim-like)", h2(), &w_box, p2),
        ("SH≤2, heart", h2(), &w_heart, p2),
        ("SH≤2, signal", h2(), &w_sig, p2),
        ("SH≤2, heart, p=8", h2(), &w_heart, p8),
        ("SH≤3, heart", h3(), &w_heart, p2),
        ("SH≤3, heart, p=8", h3(), &w_heart, p8),
        ("SH≤2 + loops, heart", h2().join(lp()), &w_heart, p2),
        ("SH≤2 + loops, heart, p=8", h2().join(lp()), &w_heart, p8),
        ("SH≤2 + loops, signal", h2().join(lp()), &w_sig, p2),
        ("SH≤3 + loops, heart, p=8", h3().join(lp()), &w_heart, p8),
    ];
    let lin = harmonic_basis(&pos, fovc, 0.1, 1, inf);
    let mut rows = vec![];
    println!("{:<28} | heart: SD / span / p2p (Hz) | signal (PD-w): SD / span / p2p (Hz) | L | L, linear absorbed | at limit", "configuration");
    for (name, basis, w, o) in cfgs {
        let r = if basis.is_empty() { fit(&f_hz, w, &harmonic_basis(&pos, fovc, 0.1, 0, [0.0; 4]), &p2) } else { fit(&f_hz, w, &basis, &o) };
        let res = &r.residual;
        let (hs, hsp, hp) = field_stats(res, &w_heart);
        let (ss, ssp, sp) = field_stats(res, &w_sig);
        // the reconstruction removes a linear field exactly (k-shift per sample): what segments still have to cover
        let rl = fit(res, &w_sig, &lin, &p2);
        let (_, lsp, _) = field_stats(&rl.residual, &w_sig);
        let (l, ll) = (segments_for(ssp, t_ro), segments_for(lsp, t_ro));
        println!("{name:<28} | {hs:>5.1} / {hsp:>5.0} / {hp:>5.0} | {ss:>5.1} / {ssp:>5.0} / {sp:>5.0} | {l:>2} | {ll:>2} | {}", r.saturated);
        rows.push(serde_json::json!({"config": name, "heart": {"sd": hs, "span": hsp, "p2p": hp}, "signal": {"sd": ss, "span": ssp, "p2p": sp}, "L": l, "L_linear_absorbed": ll, "saturated": r.saturated, "coef": r.coef}));
    }
    if let Some(p) = out {
        std::fs::write(&p, serde_json::to_string_pretty(&serde_json::json!({"dir": dir, "b0": b0, "t_ro": t_ro, "imax": imax, "rows": rows})).unwrap()).unwrap();
        println!("wrote {}", p.display());
    }
}
