//! The Rust adapter + TorchScript backend must reproduce Python's `denoise_volume` on the exported reference volume
//! (`python -I tools/denoiser/export.py --random` writes data/hasty/denoiser/test_export).
use std::path::PathBuf;

use hasty_denoise::NE;
use hasty_denoise_torch::{Format, TorchDenoiser};
use num_complex::Complex32 as C32;

fn read_c8(p: &std::path::Path) -> Vec<C32> {
    let b = std::fs::read(p).unwrap();
    b.chunks_exact(8).map(|c| C32::new(f32::from_le_bytes(c[0..4].try_into().unwrap()), f32::from_le_bytes(c[4..8].try_into().unwrap()))).collect()
}

#[test]
fn torchscript_matches_python_reference() {
    check(Format::TorchScript, "model.ts");
}

#[test]
fn aoti_matches_python_reference() {
    check(Format::Aoti, "model_aoti.pt2");
}

fn check(format: Format, file: &str) {
    let dir = std::env::var_os("HASTY_DENOISER_EXPORT").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty/denoiser/test_export"));
    if !dir.join(file).exists() {
        eprintln!("skipped: no {file} at {} (run tools/denoiser/export.py --random [--aoti])", dir.display());
        return;
    }
    let r: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("ref/ref.json")).unwrap()).unwrap();
    let dims: Vec<usize> = r["dims"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let frames = r["frames"].as_u64().unwrap() as usize;
    let kv: Vec<f32> = r["kv"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let h = r["h"].as_f64().unwrap() as f32;
    let mut x = read_c8(&dir.join("ref/volume_in.c8"));
    let want = read_c8(&dir.join("ref/volume_out.c8"));
    let d = TorchDenoiser::load(&dir, format, false).unwrap();
    let kv: [f32; NE] = kv.try_into().unwrap();
    let dims = [dims[0], dims[1], dims[2]];
    let input = x.clone();
    d.denoise(&mut x, dims, frames, kv, h, 1).unwrap();
    let num: f32 = x.iter().zip(&want).map(|(a, b)| (a - b).norm_sqr()).sum();
    let den: f32 = want.iter().map(|b| b.norm_sqr()).sum();
    let changed: f32 = input.iter().zip(&want).map(|(a, b)| (a - b).norm_sqr()).sum::<f32>() / den;
    let rel = (num / den).sqrt();
    eprintln!("{format:?}: Rust (axes reversed, slots reordered, padded) vs Python: rel. error {rel:.2e}; the model changes the input by {:.2e}", changed.sqrt());
    assert!(changed.sqrt() > 1e-3, "reference map is (nearly) the identity — test would be vacuous");
    assert!(rel < 1e-4);

    // exp(−i kv v) convention: conjugate-relative input -> conjugate-relative output of the same map
    let mut flipped = input.clone();
    let nv = dims.iter().product::<usize>();
    let conv = |s: &mut Vec<C32>| {
        for t in 0..frames {
            for q in 0..nv {
                let r = s[t * NE * nv + q];
                let a = r.norm_sqr();
                let u2 = if a > 0.0 { r * r / a } else { C32::new(1.0, 0.0) };
                for e in 1..NE {
                    let z = &mut s[(t * NE + e) * nv + q];
                    *z = z.conj() * u2;
                }
            }
        }
    };
    conv(&mut flipped);
    d.denoise(&mut flipped, dims, frames, kv, h, -1).unwrap();
    conv(&mut flipped);
    let num: f32 = flipped.iter().zip(&want).map(|(a, b)| (a - b).norm_sqr()).sum();
    eprintln!("phase_sign −1 path: rel. error {:.2e}", (num / den).sqrt());
    assert!((num / den).sqrt() < 1e-4);
}

#[test]
fn tiled_matches_whole_volume() {
    // tiles of 61 voxels with a 28-voxel margin along an axis of 70 (padded 73): two tiles, kept centres stitched
    let dir = std::env::var_os("HASTY_DENOISER_EXPORT").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../data/hasty/denoiser/test_export"));
    if !dir.join("model.ts").exists() {
        eprintln!("skipped: no model.ts at {}", dir.display());
        return;
    }
    let d = TorchDenoiser::load(&dir, Format::TorchScript, false).unwrap();
    let (dims, frames) = ([70usize, 13, 9], 6usize);
    let n = NE * frames * dims.iter().product::<usize>();
    let x: Vec<C32> = (0..n).map(|i| C32::new((((i * 7919) % 2003) as f32 / 1000.0 - 1.0) + 1.5 * ((i % 7 == 0) as u8 as f32), ((i * 104729) % 1999) as f32 / 1000.0 - 1.0)).collect();
    let kv = [0.0, 3.14159, 1.25664, 3.14159, 1.25664, 3.14159, 1.25664];
    let mut whole = x.clone();
    d.denoise(&mut whole, dims, frames, kv, 2e-3, -1).unwrap();
    let mut t = x.clone();
    let den: f32 = whole.iter().map(|b| b.norm_sqr()).sum();
    let changed: f32 = whole.iter().zip(&x).map(|(a, b)| (a - b).norm_sqr()).sum::<f32>() / den;
    assert!(changed.sqrt() > 1e-3);
    // spatial tiles, time windows (6 frames, rf_time 3: windows keep 1 and 2 frames... down to the minimum), both
    let frames_long = 10usize;
    let n2 = NE * frames_long * dims.iter().product::<usize>();
    let x2: Vec<C32> = (0..n2).map(|i| x[i % n]).collect();
    let mut whole2 = x2.clone();
    d.denoise(&mut whole2, dims, frames_long, kv, 2e-3, -1).unwrap();
    for (tile, keep, xs, w) in [(Some(61), None, &x, &whole), (None, Some(2), &x2, &whole2), (Some(61), Some(3), &x2, &whole2)] {
        let mut t = xs.clone();
        d.denoise_tiled(&mut t, dims, if xs.len() == n { frames } else { frames_long }, kv, 2e-3, -1, tile, keep).unwrap();
        let num: f32 = w.iter().zip(&t).map(|(a, b)| (a - b).norm_sqr()).sum();
        let den: f32 = w.iter().map(|b| b.norm_sqr()).sum();
        eprintln!("tiles {tile:?}, kept frames {keep:?} vs whole volume {dims:?}: rel. error {:.2e} (model changes the input by {:.2e})", (num / den).sqrt(), changed.sqrt());
        assert!((num / den).sqrt() < 1e-5);
    }
}
