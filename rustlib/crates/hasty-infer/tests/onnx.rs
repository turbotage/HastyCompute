//! Needs libonnxruntime (HASTY_ORT) and the model made by tools/models/make_test_onnx.py in
//! `$HASTY_ORT_TEST_MODEL` (a model directory); skipped otherwise.
#![cfg(feature = "onnx")]

use hasty_infer::backend::{Backend, Tensor};
use hasty_infer::geom::{Geometry, Volume};
use hasty_infer::onnx::OnnxBackend;
use hasty_infer::recipe::{segment_patch3d, RunOptions};
use hasty_infer::spec::ModelSpec;
use std::path::PathBuf;

fn model() -> Option<ModelSpec> {
    let d = PathBuf::from(std::env::var_os("HASTY_ORT_TEST_MODEL")?);
    std::env::var_os("HASTY_ORT")?;
    Some(ModelSpec::load(&d).unwrap())
}

#[test]
fn forward_matches_formula() {
    let Some(spec) = model() else { return eprintln!("skipped") };
    let mut b = OnnxBackend::open(&spec, false, None).unwrap();
    let x: Vec<f32> = (0..2 * 3 * 4).map(|i| i as f32 / 24.0).collect();
    let y = b.forward("main", &Tensor::new(vec![1, 1, 2, 3, 4], x.clone())).unwrap();
    assert_eq!(y.shape, vec![1, 2, 2, 3, 4]);
    for i in 0..x.len() {
        assert_eq!(y.data[i], 0.0);
        assert!((y.data[x.len() + i] - (20.0 * x[i] - 10.0)).abs() < 1e-4);
    }
}

#[test]
fn segment_end_to_end_cpu_and_cuda() {
    let Some(spec) = model() else { return eprintln!("skipped") };
    let g = Geometry::from_spacing([30, 25, 20], [1.0; 3]);
    let mut d = vec![0.0f32; g.numel()];
    for z in 5..15 {
        for y in 6..18 {
            for x in 8..22 {
                d[(z * 25 + y) * 30 + x] = 1.0;
            }
        }
    }
    let img = Volume::new(g, d.clone()).unwrap();
    for cuda in [false, true] {
        let mut b = OnnxBackend::open(&spec, cuda, None).unwrap();
        eprintln!("cuda requested {cuda}, active {}", b.cuda);
        let seg = segment_patch3d(&mut b, &spec, &[img.clone()], &RunOptions { tta: true, ..Default::default() }, &mut |_, _| true).unwrap();
        let bad = seg.labels.iter().zip(&d).filter(|(l, x)| (**l == 1) != (**x > 0.5)).count();
        assert_eq!(bad, 0, "cuda={cuda}");
    }
}

mod pvs {
    //! Parity of the exported Pham T1 PVS nnU-Net: needs HASTY_ORT, HASTY_PVS_MODEL (export dir with model.json,
    //! model.onnx, ref_x.npy, ref_y.npy) and HASTY_PVS_REF (testdata/ref from pvs_t1_ref_norm.py +
    //! pvs_ref_predict.py: img.npy, bbox.json, norm_classic.npy, logits_classic_{plain,mirror}.npy).
    use super::*;
    use hasty_infer::recipe::{tile_probabilities, Tile};
    use std::path::Path;

    fn read_npy(p: &Path) -> (Vec<usize>, Vec<f32>) {
        let b = std::fs::read(p).unwrap();
        let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
        let hdr = std::str::from_utf8(&b[10..10 + hl]).unwrap();
        assert!(hdr.contains("'<f4'") && hdr.contains("False"), "{hdr}");
        let sh = hdr.split("'shape': (").nth(1).unwrap().split(')').next().unwrap();
        let shape = sh.split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect();
        (shape, b[10 + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    fn env() -> Option<(ModelSpec, PathBuf, PathBuf)> {
        std::env::var_os("HASTY_ORT")?;
        let m = PathBuf::from(std::env::var_os("HASTY_PVS_MODEL")?);
        let r = PathBuf::from(std::env::var_os("HASTY_PVS_REF")?);
        Some((ModelSpec::load(&m).unwrap(), m, r))
    }

    fn softmax(l: &[f32], k: usize) -> Vec<f32> {
        let nv = l.len() / k;
        let mut o = l.to_vec();
        for v in 0..nv {
            let m = (0..k).map(|c| l[c * nv + v]).fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = (0..k).map(|c| (l[c * nv + v] - m).exp()).sum();
            for c in 0..k {
                o[c * nv + v] = (l[c * nv + v] - m).exp() / s;
            }
        }
        o
    }

    #[test]
    fn network_matches_torch() {
        let Some((spec, m, _)) = env() else { return eprintln!("skipped") };
        let (_, x) = read_npy(&m.join("ref_x.npy"));
        let (ys, y) = read_npy(&m.join("ref_y.npy"));
        let mut b = OnnxBackend::open(&spec, true, None).unwrap();
        let out = b.forward("main", &Tensor::new(vec![1, 1, 160, 128, 112], x)).unwrap();
        assert_eq!(out.shape, ys);
        let max = out.data.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let scale = y.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        eprintln!("onnx vs torch: max |diff| {max:.2e} (logit scale {scale:.1}), cuda {}", b.cuda);
        assert!(max < 2e-2 * scale.max(1.0));
    }

    #[test]
    fn tiling_matches_nnunet() {
        let Some((spec, _, r)) = env() else { return eprintln!("skipped") };
        let (shape, norm) = read_npy(&r.join("norm_classic.npy"));
        let vol = Volume::new(Geometry::from_spacing([shape[2], shape[1], shape[0]], [1.0; 3]), norm).unwrap();
        let mut b = OnnxBackend::open(&spec, true, None).unwrap();
        let k = spec.num_classes();
        for (tta, name) in [(false, "plain"), (true, "mirror")] {
            let (_, lg) = read_npy(&r.join(format!("logits_classic_{name}.npy")));
            let want = softmax(&lg, k);
            let tile = Tile { roles: &spec.ensemble, k, patch: spec.patch.unwrap(), overlap: spec.overlap, tta, softmax: true };
            let p = tile_probabilities(&mut b, &[vol.data.clone()], vol.geom.dims, &tile, &mut |_, _| true).unwrap();
            let max = p.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let mean = p.iter().zip(&want).map(|(a, b)| (a - b).abs() as f64).sum::<f64>() / p.len() as f64;
            eprintln!("{name}: prob diff mean {mean:.2e} max {max:.2e}");
            assert!(mean < 1e-6 && max < 0.05, "{name}");
        }
    }

    #[test]
    fn full_pipeline_labels() {
        let Some((spec, _, r)) = env() else { return eprintln!("skipped") };
        let (shape, img) = read_npy(&r.join("img.npy"));
        let vol = Volume::new(Geometry::from_spacing([shape[2], shape[1], shape[0]], [2.0; 3]), img).unwrap();
        let mut b = OnnxBackend::open(&spec, true, None).unwrap();
        let t = std::time::Instant::now();
        let seg = segment_patch3d(&mut b, &spec, &[vol.clone()], &RunOptions { probabilities: true, ..Default::default() }, &mut |_, _| true).unwrap();
        eprintln!("segment: {:.1}s, labels {:?}", t.elapsed().as_secs_f64(), (0..3).map(|c| seg.labels.iter().filter(|&&l| l == c).count()).collect::<Vec<_>>());
        // compare the foreground probability inside the nnU-Net crop with the Python reference (Rust normalisation vs
        // skimage/dipy differs in a few CLAHE bins, so this bounds the effect of that on the network output)
        let (_, lg) = read_npy(&r.join("logits_classic_plain.npy"));
        let want = softmax(&lg, 3);
        let bb: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(r.join("bbox.json")).unwrap()).unwrap();
        let g = |k: &str, i: usize| bb[k][i].as_u64().unwrap() as usize;
        let (lo, hi) = ([g("lo", 0), g("lo", 1), g("lo", 2)], [g("hi", 0), g("hi", 1), g("hi", 2)]);
        let probs = seg.probabilities.unwrap();
        let (mut sum, mut n, mut maxd) = (0.0f64, 0usize, 0.0f32);
        let cs = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
        for z in 0..cs[0] {
            for y in 0..cs[1] {
                for x in 0..cs[2] {
                    let full = ((z + lo[0]) * shape[1] + y + lo[1]) * shape[2] + x + lo[2];
                    let cr = (z * cs[1] + y) * cs[2] + x;
                    let d = ((1.0 - probs[0][full]) - (1.0 - want[cr])).abs();
                    sum += d as f64;
                    maxd = maxd.max(d);
                    n += 1;
                }
            }
        }
        eprintln!("foreground prob vs python-normalised reference: mean {:.2e}, max {maxd:.3}", sum / n as f64);
        assert!(sum / (n as f64) < 1e-3);
        assert_eq!(seg.labels.len(), vol.data.len());
    }
}

mod parity {
    //! Any exported model: `HASTY_MODEL_PARITY=<dir with model.json, ref_x.npy, ref_y.npy>` + HASTY_ORT. Checks the
    //! first network of the model against the torch output stored by export_nnunet.py.
    use super::*;

    #[test]
    fn onnx_matches_torch_fixture() {
        let (Some(dir), Some(_)) = (std::env::var_os("HASTY_MODEL_PARITY"), std::env::var_os("HASTY_ORT")) else { return eprintln!("skipped") };
        let dir = PathBuf::from(dir);
        let spec = ModelSpec::load(&dir).unwrap();
        let rd = |n: &str| {
            let b = std::fs::read(dir.join(n)).unwrap();
            let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
            let hdr = std::str::from_utf8(&b[10..10 + hl]).unwrap().to_string();
            let sh: Vec<usize> = hdr.split("'shape': (").nth(1).unwrap().split(')').next().unwrap().split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect();
            let d: Vec<f32> = b[10 + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            (sh, d)
        };
        // one fixture per role: ref_x.npy (first fold) or ref_<role>_x.npy
        let mut b = OnnxBackend::open(&spec, std::env::var_os("HASTY_CPU").is_none(), None).unwrap();
        let mut checked = 0;
        for role in spec.ensemble.clone() {
            let (fx, fy) = if dir.join(format!("ref_{role}_x.npy")).exists() { (format!("ref_{role}_x.npy"), format!("ref_{role}_y.npy")) } else if checked == 0 { ("ref_x.npy".into(), "ref_y.npy".into()) } else { continue };
            let (xs, x) = rd(&fx);
            let (ys, y) = rd(&fy);
            let out = b.forward(&role, &Tensor::new(xs, x)).unwrap();
            assert_eq!(out.shape, ys, "{role}");
            let max = out.data.iter().zip(&y).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let scale = y.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            eprintln!("{}/{role}: onnx vs torch max |diff| {max:.2e}, logit scale {scale:.1}", spec.name);
            assert!(max < 2e-2 * scale.max(1.0), "{role}");
            checked += 1;
        }
        assert!(checked > 0);
    }
}

mod bench {
    //! `HASTY_BENCH_MODEL=<dir>` role `axial` of the FastSurfer export: forward time per slice vs batch size.
    use super::*;

    #[test]
    fn fastsurfer_forward_times() {
        let (Some(dir), Some(_)) = (std::env::var_os("HASTY_BENCH_MODEL"), std::env::var_os("HASTY_ORT")) else { return eprintln!("skipped") };
        let spec = ModelSpec::load(&PathBuf::from(dir)).unwrap();
        for cuda in [true, false] {
            let mut b = OnnxBackend::open(&spec, cuda, None).unwrap();
            for n in [1usize, 4, 16] {
                let x = Tensor::new(vec![n, 7, 256, 256], vec![0.3; n * 7 * 65536]);
                b.forward("axial", &x).unwrap(); // warm-up (cuDNN autotune)
                let t = std::time::Instant::now();
                let reps = if cuda { 4 } else { 1 };
                for _ in 0..reps {
                    b.forward("axial", &x).unwrap();
                }
                let dt = t.elapsed().as_secs_f64() / reps as f64;
                eprintln!("cuda={cuda} (active {}) batch {n}: {:.1} ms per call, {:.1} ms per slice", b.cuda, dt * 1e3, dt * 1e3 / n as f64);
            }
        }
    }
}
