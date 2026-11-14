use hasty_infer::backend::{Mock, Tensor};
use hasty_infer::geom::{conform_geometry, covering_geometry, rescaled_geometry, resample, Geometry, Interp, Volume, IDENTITY};
use hasty_infer::recipe::{segment_patch3d, starts, tile_probabilities, RunOptions, Tile};
use hasty_infer::spec::ModelSpec;
use serde_json::json;
use std::path::Path;

fn spec(patch: [usize; 3], orientation: &str, spacing: Option<[f64; 3]>) -> ModelSpec {
    let mut j = json!({"recipe": "patch3d", "files": {"main": "x"}, "patch": patch, "overlap": 0.5, "orientation": orientation,
        "norm": {"type": "none"}, "labels": {"0": "bg", "1": "fg"}});
    if let Some(s) = spacing {
        j["spacing_mm"] = json!(s);
    }
    ModelSpec::from_json(Path::new("m"), &j).unwrap()
}

/// threshold net: logit1 - logit0 = x - 0.5, as 2 channels
fn threshold() -> Mock<impl FnMut(&str, &Tensor) -> Tensor + Send> {
    Mock(|_, t: &Tensor| {
        let n = t.data.len();
        let mut d = vec![0.0f32; 2 * n];
        for i in 0..n {
            d[n + i] = (t.data[i] - 0.5) * 20.0;
        }
        let mut s = t.shape.clone();
        s[1] = 2;
        Tensor::new(s, d)
    })
}

fn blob(g: &Geometry) -> Volume {
    let [nx, ny, nz] = g.dims;
    let mut d = vec![0.0f32; g.numel()];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                if (4..9).contains(&x) && (3..8).contains(&y) && (2..10).contains(&z) {
                    d[(z * ny + y) * nx + x] = 1.0;
                }
            }
        }
    }
    Volume::new(g.clone(), d).unwrap()
}

fn tp(b: &mut dyn hasty_infer::backend::Backend, v: &Volume, patch: [usize; 3], tta: bool, go: bool) -> hasty_infer::Result<Vec<f32>> {
    let roles = vec!["main".to_string()];
    let tile = Tile { roles: &roles, k: 2, patch, overlap: 0.5, tta, softmax: true };
    tile_probabilities(b, &[v.data.clone()], v.geom.dims, &tile, &mut |_, _| go)
}

#[test]
fn starts_cover() {
    assert_eq!(starts(10, 16, 0.5), vec![0]);
    // nnU-Net: ceil((100-32)/16)+1 = 6 evenly spaced starts 0..68
    assert_eq!(starts(100, 32, 0.5), vec![0, 14, 27, 41, 54, 68]);
    assert_eq!(starts(32, 32, 0.5), vec![0]);
}

#[test]
fn tiled_equals_whole_for_pointwise_net() {
    let g = Geometry::from_spacing([23, 19, 17], [1.0; 3]);
    let v = blob(&g);
    let mut small = threshold();
    let p = tp(&mut small, &v, [8, 8, 8], false, true).unwrap();
    let mut big = threshold();
    let q = tp(&mut big, &v, [32, 32, 32], false, true).unwrap();
    let m = p.iter().zip(&q).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(m < 1e-5, "{m}");
}

#[test]
fn tta_pointwise_identity() {
    let g = Geometry::from_spacing([13, 11, 9], [1.0; 3]);
    let v = blob(&g);
    let a = tp(&mut threshold(), &v, [8, 8, 8], false, true).unwrap();
    let b = tp(&mut threshold(), &v, [8, 8, 8], true, true).unwrap();
    assert!(a.iter().zip(&b).all(|(x, y)| (x - y).abs() < 1e-5));
}

#[test]
fn cancel() {
    let g = Geometry::from_spacing([13, 11, 9], [1.0; 3]);
    let r = tp(&mut threshold(), &blob(&g), [8, 8, 8], false, false);
    assert!(matches!(r, Err(hasty_infer::Error::Cancelled)));
}

#[test]
fn flipped_oblique_input_roundtrip() {
    // input stored LIA with 2 mm anisotropic voxels and an offset; model wants RAS 1.5 mm
    let mut a = IDENTITY;
    a[0][0] = -2.0;
    a[1][2] = 2.5; // voxel k -> A
    a[2][1] = -2.0; // voxel j -> I
    a[0][3] = 30.0;
    a[1][3] = -40.0;
    a[2][3] = 10.0;
    let g = Geometry { dims: [20, 14, 12], affine: a };
    assert_eq!(g.orientation(), "LIA");
    let img = blob(&g);
    let sp = spec([16, 16, 16], "RAS", Some([1.5; 3]));
    let seg = segment_patch3d(&mut threshold(), &sp, &[img.clone()], &RunOptions::default(), &mut |_, _| true).unwrap();
    let agree = seg.labels.iter().zip(&img.data).filter(|(l, x)| (**l == 1) == (**x > 0.5)).count();
    let frac = agree as f64 / img.data.len() as f64;
    assert!(frac > 0.97, "label agreement {frac}");
    assert!(seg.labels.iter().any(|&l| l == 1));
}

#[test]
fn conform_inverse() {
    let mut a = IDENTITY;
    a[0][0] = -1.0;
    a[1][1] = 1.0;
    a[2][2] = 1.0;
    a[0][3] = 100.0;
    let g = Geometry { dims: [30, 20, 10], affine: a };
    let c = conform_geometry(&g, 1.0, [40, 40, 40], "LIA").unwrap();
    assert_eq!(c.orientation(), "LIA");
    let v = blob(&g);
    let fwd = resample(&v, &c, Interp::Linear, 0.0).unwrap();
    let back = resample(&fwd, &g, Interp::Linear, 0.0).unwrap();
    let err = v.data.iter().zip(&back.data).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    assert!(err < 1e-4, "{err}");
    let cov = covering_geometry(&g, [1.0; 3], "RAS").unwrap();
    assert!(cov.dims.iter().all(|&d| d >= 10));
}

#[test]
fn two_channels_and_ensemble() {
    // net: logit1 - logit0 = 10 * (ch0 - ch1) ; two roles that differ by a sign flip cancel in the ensemble
    let g = Geometry::from_spacing([21, 17, 15], [1.0; 3]);
    let a = blob(&g);
    let b = Volume::zeros(g.clone());
    let mut sp = spec([8, 8, 8], "native", None);
    sp.in_channels = 2;
    let mk = |sign: f32| {
        Mock(move |_: &str, t: &Tensor| {
            let n = t.shape[2] * t.shape[3] * t.shape[4];
            let mut d = vec![0.0f32; 2 * n];
            for i in 0..n {
                d[n + i] = sign * 10.0 * (t.data[i] - t.data[n + i]);
            }
            Tensor::new(vec![1, 2, t.shape[2], t.shape[3], t.shape[4]], d)
        })
    };
    // role-dependent behaviour through a single closure
    let mut be = Mock(move |role: &str, t: &Tensor| {
        let mut m = mk(if role == "neg" { -1.0 } else { 1.0 });
        hasty_infer::backend::Backend::forward(&mut m, role, t).unwrap()
    });
    let seg = segment_patch3d(&mut be, &sp, &[a.clone(), b.clone()], &RunOptions::default(), &mut |_, _| true).unwrap();
    let bad = seg.labels.iter().zip(&a.data).filter(|(l, x)| (**l == 1) != (**x > 0.5)).count();
    assert_eq!(bad, 0);
    sp.ensemble = vec!["main".into(), "neg".into()];
    sp.files.insert("neg".into(), "x".into());
    let seg = segment_patch3d(&mut be, &sp, &[a, b], &RunOptions::default(), &mut |_, _| true).unwrap();
    assert!(seg.labels.iter().all(|&l| l == 0), "opposite roles must cancel");
    // wrong channel count
    assert!(segment_patch3d(&mut be, &sp, &[Volume::zeros(g)], &RunOptions::default(), &mut |_, _| true).is_err());
}

#[test]
fn native_with_spacing_resamples_and_returns_to_input_grid() {
    // 2 mm stored grid, model wants 1 mm on the same axes
    let g = Geometry::from_spacing([12, 10, 8], [2.0; 3]);
    let r = rescaled_geometry(&g, [1.0; 3]);
    assert_eq!(r.dims, [24, 20, 16]);
    let img = blob(&g);
    let sp = spec([16, 16, 16], "native", Some([1.0; 3]));
    let seg = segment_patch3d(&mut threshold(), &sp, &[img.clone()], &RunOptions::default(), &mut |_, _| true).unwrap();
    let bad = seg.labels.iter().zip(&img.data).filter(|(l, x)| (**l == 1) != (**x > 0.5)).count();
    assert!(bad as f64 / img.data.len() as f64 > -1.0 && bad < 40, "{bad} of {}", img.data.len());
    assert_eq!(seg.labels.len(), img.data.len());
}

#[test]
fn crop_nonzero_pastes_back() {
    let g = Geometry::from_spacing([30, 24, 20], [1.0; 3]);
    let img = blob(&g);
    let mut sp = spec([8, 8, 8], "native", None);
    sp.crop_nonzero = true;
    let seg = segment_patch3d(&mut threshold(), &sp, &[img.clone()], &RunOptions { probabilities: true, ..Default::default() }, &mut |_, _| true).unwrap();
    assert!(seg.labels.iter().zip(&img.data).all(|(l, x)| (*l == 1) == (*x > 0.5)));
    let p = seg.probabilities.unwrap();
    assert_eq!(p[0].len(), img.data.len());
    assert_eq!(p[0][0], 1.0); // outside the crop: class 0 with probability 1
}

#[test]
fn normalisation_schemes() {
    use hasty_infer::recipe::normalize;
    use hasty_infer::spec::Norm;
    let mut d: Vec<f32> = vec![0.0, 0.0, 2.0, 4.0, 6.0, 8.0];
    normalize(&mut d, &Norm::ZScore { nonzero_mask: true }, [1, 1, 6]);
    assert_eq!(&d[..2], &[0.0, 0.0]);
    assert!((d[2] + 1.3416).abs() < 1e-3 && (d[5] - 1.3416).abs() < 1e-3);
    let mut d: Vec<f32> = vec![-5.0, 0.0, 10.0, 100.0];
    normalize(&mut d, &Norm::Ct { lo: 0.0, hi: 50.0, mean: 10.0, std: 10.0 }, [1, 1, 4]);
    assert_eq!(d, vec![-1.0, -1.0, 0.0, 4.0]);
}
