//! `model_list` / `model_run` through the registry. Needs HASTY_MODELS (dir containing the model "pointwise" made by
//! tools/models/make_test_onnx.py) and HASTY_ORT (libonnxruntime); skipped otherwise.
use std::sync::Arc;

use hasty_server::commands::Ctx;
use hasty_server::{Bank, Registry};
use hasty_value::{Tensor, Value};

#[test]
fn list_and_run() {
    if std::env::var_os("HASTY_MODELS").is_none() || std::env::var_os("HASTY_ORT").is_none() {
        return eprintln!("skipped");
    }
    if !std::path::Path::new(&std::env::var_os("HASTY_MODELS").unwrap()).join("pointwise/model.json").exists() {
        return eprintln!("skipped (no pointwise model in HASTY_MODELS)");
    }
    let reg = Registry::base();
    let ctx = Ctx { bank: Arc::new(Bank::new()) };
    let (msg, _) = reg.run("model_list", &ctx, "", &[]).unwrap();
    let j: serde_json::Value = serde_json::from_str(&msg).unwrap();
    assert!(j["models"].as_array().unwrap().iter().any(|m| m["name"] == "pointwise"), "{msg}");

    let (nx, ny, nz) = (24usize, 20usize, 18usize);
    let mut d = vec![0.0f32; nx * ny * nz];
    for z in 4..14 {
        for y in 5..15 {
            for x in 6..20 {
                d[(z * ny + y) * nx + x] = 1.0;
            }
        }
    }
    let img = Arc::new(Value::Tensor(Tensor::from_slice(&[nz, ny, nx], &d).unwrap()));
    let (m, outs) = reg.run("model_run", &ctx, r#"{"model":"pointwise","device":"cpu","spacing_mm":[1,1,1],"prob_classes":[1]}"#, &[img.clone()]).unwrap();
    let lab = outs[0].as_tensor().unwrap();
    assert_eq!(lab.shape(), &[nz, ny, nx]);
    let l: &[u8] = lab.as_slice().unwrap();
    assert!(l.iter().zip(&d).all(|(a, b)| (*a == 1) == (*b > 0.5)), "{m}");
    assert_eq!(outs[1].as_tensor().unwrap().shape(), &[1, nz, ny, nx]);

    // errors
    assert!(reg.run("model_run", &ctx, r#"{"model":"pointwise"}"#, &[img.clone()]).is_err()); // no geometry
    assert!(reg.run("model_run", &ctx, r#"{"model":"../x","spacing_mm":[1,1,1]}"#, &[img]).is_err());
}

/// Real model through the server path. Needs HASTY_MODELS containing pvs_t1_wmbg, HASTY_ORT, and HASTY_PVS_REF/img.npy
/// (written by tools/models/pvs_t1_ref_norm.py); skipped otherwise.
#[test]
fn pvs_t1_native_grid() {
    let (Some(_), Some(_), Some(r)) = (std::env::var_os("HASTY_MODELS"), std::env::var_os("HASTY_ORT"), std::env::var_os("HASTY_PVS_REF")) else { return eprintln!("skipped") };
    let b = std::fs::read(std::path::Path::new(&r).join("img.npy")).unwrap();
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let hdr = std::str::from_utf8(&b[10..10 + hl]).unwrap();
    let sh: Vec<usize> = hdr.split("'shape': (").nth(1).unwrap().split(')').next().unwrap().split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect();
    let d: Vec<f32> = b[10 + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let reg = Registry::base();
    let ctx = Ctx { bank: Arc::new(Bank::new()) };
    let (msg, _) = reg.run("model_list", &ctx, "", &[]).unwrap();
    assert!(msg.contains("pvs_t1_wmbg"), "{msg}");
    let img = Arc::new(Value::Tensor(Tensor::from_slice(&sh, &d).unwrap()));
    let (m, outs) = reg.run("model_run", &ctx, r#"{"model":"pvs_t1_wmbg","spacing_mm":[2,2,2]}"#, &[img]).unwrap();
    let j: serde_json::Value = serde_json::from_str(&m).unwrap();
    eprintln!("{m}");
    assert_eq!(outs[0].as_tensor().unwrap().shape(), &sh[..]);
    assert_eq!(j["counts"].as_object().unwrap().values().map(|c| c.as_u64().unwrap()).sum::<u64>() as usize, d.len());
}

/// TotalSegmentator MR (3 mm) smoke test on a brain T1: the brain class (50) must have a plausible adult brain volume (1–2 L). Needs the
/// model in HASTY_MODELS (totalseg_mr_3mm), HASTY_ORT and HASTY_PVS_REF/img.npy (RAS, 2 mm); skipped otherwise.
#[test]
fn totalseg_mr_brain_smoke() {
    let (Some(_), Some(_), Some(r)) = (std::env::var_os("HASTY_MODELS"), std::env::var_os("HASTY_ORT"), std::env::var_os("HASTY_PVS_REF")) else { return eprintln!("skipped") };
    if !std::path::Path::new(&std::env::var_os("HASTY_MODELS").unwrap()).join("totalseg_mr_3mm/model.json").exists() {
        return eprintln!("skipped (model not installed)");
    }
    let b = std::fs::read(std::path::Path::new(&r).join("img.npy")).unwrap();
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let hdr = std::str::from_utf8(&b[10..10 + hl]).unwrap();
    let sh: Vec<usize> = hdr.split("'shape': (").nth(1).unwrap().split(')').next().unwrap().split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect();
    let d: Vec<f32> = b[10 + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let reg = Registry::base();
    let ctx = Ctx { bank: Arc::new(Bank::new()) };
    let img = Arc::new(Value::Tensor(Tensor::from_slice(&sh, &d).unwrap()));
    let (m, outs) = reg.run("model_run", &ctx, r#"{"model":"totalseg_mr_3mm","spacing_mm":[2,2,2]}"#, &[img]).unwrap();
    let j: serde_json::Value = serde_json::from_str(&m).unwrap();
    let counts = |id: usize| j["counts"][id.to_string()].as_u64().unwrap_or(0);
    let nonzero = d.iter().filter(|&&v| v != 0.0).count() as u64;
    eprintln!("brain {} / nonzero {} ; counts {} ; {}s", counts(50), nonzero, j["counts"], j["seconds"]);
    assert_eq!(outs[0].as_tensor().unwrap().shape(), &sh[..]);
    let litres = counts(50) as f64 * 8e-6; // 2 mm voxels = 8 mm³
    assert!((1.0..2.0).contains(&litres), "brain volume {litres} L");
}

/// FastSurfer (1 mm VINN) on a brain T1: plausible structure volumes. Needs fastsurfer_vinn_1mm in HASTY_MODELS,
/// HASTY_ORT and HASTY_PVS_REF/img.npy (RAS, 2 mm); skipped otherwise.
#[test]
fn fastsurfer_brain_smoke() {
    let (Some(m), Some(_), Some(r)) = (std::env::var_os("HASTY_MODELS"), std::env::var_os("HASTY_ORT"), std::env::var_os("HASTY_PVS_REF")) else { return eprintln!("skipped") };
    if !std::path::Path::new(&m).join("fastsurfer_vinn_1mm/model.json").exists() {
        return eprintln!("skipped (model not installed)");
    }
    let b = std::fs::read(std::path::Path::new(&r).join("img.npy")).unwrap();
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let hdr = std::str::from_utf8(&b[10..10 + hl]).unwrap();
    let sh: Vec<usize> = hdr.split("'shape': (").nth(1).unwrap().split(')').next().unwrap().split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect();
    let d: Vec<f32> = b[10 + hl..].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let reg = Registry::base();
    let ctx = Ctx { bank: Arc::new(Bank::new()) };
    let img = Arc::new(Value::Tensor(Tensor::from_slice(&sh, &d).unwrap()));
    let (msg, outs) = reg.run("model_run", &ctx, r#"{"model":"fastsurfer_vinn_1mm","spacing_mm":[2,2,2]}"#, &[img]).unwrap();
    let j: serde_json::Value = serde_json::from_str(&msg).unwrap();
    let ml = |id: usize| j["counts"][id.to_string()].as_u64().unwrap_or(0) as f64 * 8e-3; // 2 mm voxels = 8 µL... = 8e-3 mL
    eprintln!("{}s; WM l/r {:.0}/{:.0} mL, hippocampus l/r {:.1}/{:.1}, thalamus l/r {:.1}/{:.1}, lat. ventricle l/r {:.1}/{:.1}", j["seconds"], ml(2), ml(41), ml(17), ml(53), ml(10), ml(49), ml(4), ml(43));
    assert_eq!(outs[0].as_tensor().unwrap().shape(), &sh[..]);
    for (name, v, lo, hi) in [("WM left", ml(2), 200.0, 350.0), ("WM right", ml(41), 200.0, 350.0), ("hippocampus left", ml(17), 2.0, 6.0), ("thalamus right", ml(49), 4.0, 10.0)] {
        assert!(v > lo && v < hi, "{name} volume {v} mL");
    }
}

fn read_npy<T: Copy>(p: &std::path::Path, f: impl Fn(&[u8]) -> T, w: usize) -> (Vec<usize>, Vec<T>) {
    let b = std::fs::read(p).unwrap();
    let hl = u16::from_le_bytes([b[8], b[9]]) as usize;
    let hdr = std::str::from_utf8(&b[10..10 + hl]).unwrap();
    let sh: Vec<usize> = hdr.split("'shape': (").nth(1).unwrap().split(')').next().unwrap().split(',').filter(|t| !t.trim().is_empty()).map(|t| t.trim().parse().unwrap()).collect();
    (sh, b[10 + hl..].chunks_exact(w).map(f).collect())
}

/// Pham T2 PVS model on HCP-Aging T2w (OpenNeuro ds005595, CC0) against that dataset's PVS mask (white matter PVS only, refined
/// labels). Needs pvs_t2_wmbg in HASTY_MODELS, HASTY_ORT and HASTY_HCP (dir with img<sub>.npy / mask<sub>.npy, subject list
/// in HASTY_HCP_SUBS, default "001"); skipped otherwise. Prints Dice; asserts only a sanity floor.
#[test]
fn pvs_t2_hcp_dice() {
    let (Some(m), Some(_), Some(r)) = (std::env::var_os("HASTY_MODELS"), std::env::var_os("HASTY_ORT"), std::env::var_os("HASTY_HCP")) else { return eprintln!("skipped") };
    if !std::path::Path::new(&m).join(format!("{}/model.json", std::env::var("HASTY_HCP_MODEL").unwrap_or("pvs_t2_wmbg".into()))).exists() {
        return eprintln!("skipped (model not installed)");
    }
    let reg = Registry::base();
    let ctx = Ctx { bank: Arc::new(Bank::new()) };
    let subs = std::env::var("HASTY_HCP_SUBS").unwrap_or("001".into());
    for s in subs.split(',') {
        let (sh, d) = read_npy(&std::path::Path::new(&r).join(format!("img{s}.npy")), |c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]), 4);
        let (_, gt) = read_npy(&std::path::Path::new(&r).join(format!("mask{s}.npy")), |c| c[0], 1);
        let img = Arc::new(Value::Tensor(Tensor::from_slice(&sh, &d).unwrap()));
        let (msg, outs) = reg.run("model_run", &ctx, &format!(r#"{{"model":"{}","device":"{}","spacing_mm":[0.8,0.8,0.8]}}"#, std::env::var("HASTY_HCP_MODEL").unwrap_or("pvs_t2_wmbg".into()), std::env::var("HASTY_DEVICE").unwrap_or("cpu".into())), &[img]).unwrap();
        let lab = outs[0].as_tensor().unwrap();
        let l: &[u8] = lab.as_slice().unwrap();
        let (mut tp_wm, mut p_wm, mut p_any, mut tp_any, mut g) = (0u64, 0u64, 0u64, 0u64, 0u64);
        for (a, b) in l.iter().zip(&gt) {
            let (pw, pa, gg) = (*a == 1, *a != 0, *b != 0);
            p_wm += pw as u64;
            p_any += pa as u64;
            g += gg as u64;
            tp_wm += (pw && gg) as u64;
            tp_any += (pa && gg) as u64;
        }
        let dice = |tp: u64, p: u64| 2.0 * tp as f64 / (p + g) as f64;
        eprintln!("sub-{s}: gt {g} voxels; pred WM {p_wm} (Dice {:.3}), WM+BG {p_any} (Dice {:.3}); {msg}", dice(tp_wm, p_wm), dice(tp_any, p_any));
        assert!(dice(tp_any, p_any) > 0.3, "sub-{s} Dice too low");
    }
}

/// Run models on a real volume and keep the labels. HASTY_CASE = dir with img.npy and geom.json (tools/models/nii_to_npy.py),
/// HASTY_CASE_MODELS = comma list of installed models; writes <case>/out_<model>.npy (u8, [z,y,x]) and prints counts. Skipped otherwise.
#[test]
fn case_models() {
    let (Some(_), Some(_), Some(r), Some(ms)) = (std::env::var_os("HASTY_MODELS"), std::env::var_os("HASTY_ORT"), std::env::var_os("HASTY_CASE"), std::env::var("HASTY_CASE_MODELS").ok()) else { return eprintln!("skipped") };
    let dir = std::path::PathBuf::from(r);
    let (sh, d) = read_npy(&dir.join("img.npy"), |c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]), 4);
    let geom = std::fs::read_to_string(dir.join("geom.json")).unwrap();
    let reg = Registry::base();
    let ctx = Ctx { bank: Arc::new(Bank::new()) };
    let img = Arc::new(Value::Tensor(Tensor::from_slice(&sh, &d).unwrap()));
    for m in ms.split(',') {
        let mut o: serde_json::Value = serde_json::from_str(&geom).unwrap();
        o["model"] = m.into();
        let (msg, outs) = reg.run("model_run", &ctx, &o.to_string(), &[img.clone()]).unwrap();
        let j: serde_json::Value = serde_json::from_str(&msg).unwrap();
        eprintln!("{m}: {}s cuda {} counts {}", j["seconds"], j["cuda"], j["counts"]);
        let l = outs[0].as_tensor().unwrap();
        let v: Vec<u8> = match l.as_slice::<u8>() {
            Ok(s) => s.to_vec(),
            Err(_) => l.as_slice::<u16>().unwrap().iter().map(|&x| x.min(255) as u8).collect(),
        };
        let mut f = b"\x93NUMPY\x01\x00".to_vec();
        let hdr = format!("{{'descr': '|u1', 'fortran_order': False, 'shape': ({}, {}, {}), }}", sh[0], sh[1], sh[2]);
        let pad = (64 - (10 + hdr.len() + 1) % 64) % 64;
        let hdr = format!("{hdr}{}\n", " ".repeat(pad));
        f.extend((hdr.len() as u16).to_le_bytes());
        f.extend(hdr.as_bytes());
        f.extend(&v);
        std::fs::write(dir.join(format!("out_{m}.npy")), f).unwrap();
    }
}
