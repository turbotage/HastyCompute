//! Parity of the Rust T1wPVS normalisation with the Python reference (tools/models/pvs_t1_ref_norm.py).
//! Needs `HASTY_PVS_REF` = directory with img.npy, bbox.json, norm_classic.npy; skipped otherwise.
use hasty_infer::pvsnorm::*;
use std::path::{Path, PathBuf};

/// minimal .npy reader: little-endian f32, C-order
pub fn read_npy(p: &Path) -> (Vec<usize>, Vec<f32>) {
    let b = std::fs::read(p).unwrap();
    assert_eq!(&b[..6], b"\x93NUMPY");
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let hdr = std::str::from_utf8(&b[10..10 + hl]).unwrap();
    assert!(hdr.contains("'<f4'") && hdr.contains("False"), "{hdr}");
    let sh = hdr.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
    let shape: Vec<usize> = sh.split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect();
    let data = b[10 + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    (shape, data)
}

fn stats(a: &[f32], b: &[f32]) -> (f64, f32) {
    let mean = a.iter().zip(b).map(|(x, y)| (x - y).abs() as f64).sum::<f64>() / a.len() as f64;
    (mean, a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max))
}

#[test]
fn matches_python_classic() {
    let Some(dir) = std::env::var_os("HASTY_PVS_REF").map(PathBuf::from) else { return eprintln!("skipped") };
    let (shape, img) = read_npy(&dir.join("img.npy"));
    let bb: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("bbox.json")).unwrap()).unwrap();
    let g = |k: &str, i: usize| bb[k][i].as_u64().unwrap() as usize;
    let (lo, hi) = ([g("lo", 0), g("lo", 1), g("lo", 2)], [g("hi", 0), g("hi", 1), g("hi", 2)]);
    let cs = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
    let mut crop = Vec::with_capacity(cs[0] * cs[1] * cs[2]);
    for z in lo[0]..hi[0] {
        for y in lo[1]..hi[1] {
            let r = (z * shape[1] + y) * shape[2];
            crop.extend_from_slice(&img[r + lo[2]..r + hi[2]]);
        }
    }
    let (rs, reference) = read_npy(&dir.join("norm_classic.npy"));
    assert_eq!(rs, cs);

    // stage-wise checks make mismatches attributable
    let t = std::time::Instant::now();
    let out = if std::env::var("HASTY_PVS_CONTRAST").as_deref() == Ok("t2") { t2w_pvs_norm(&crop, cs) } else { t1w_pvs_norm(&crop, cs) };
    eprintln!("rust norm {:.1}s", t.elapsed().as_secs_f64());
    let (mean, max) = stats(&out, &reference);
    eprintln!("final: mean |diff| {mean:.5}, max {max:.4}");
    assert!(mean < 2e-3, "mean abs diff {mean}");
}
