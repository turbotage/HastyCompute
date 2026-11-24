//! G3 surrogate (research/precond_estimator.md §7): the device pieces equal their host references.
//! - `chan_symbol_dev` (build_kernels with complex weights, Fejér fold) = the direct Fejér sum `chan_symbol_host`, up to
//!   the NUFFT sign convention (the test reports which: mirrored or not; the binary picks it by the operator check);
//! - `SurDev::apply` = `SurHost::apply` for the same symbols, conventions and ridge;
//! - `SurDev::solve` (inner PCG and Chebyshev) = `inner_pcg_host` / `cheb_host`.

use std::sync::Arc;

use hasty_cuda::{Device, C32};
use hasty_nufft::SegHostTables;
use hasty_recon::ppre::{rdot, Fft3};
use hasty_recon::surrogate::{cheb_host, chan_symbol_dev, chan_symbol_host, compress, inner_pcg_host, pair_weights, Inner, SurConv, SurDev, SurHost};

fn rel(a: &[C32], b: &[C32]) -> f64 {
    let d: Vec<C32> = a.iter().zip(b).map(|(x, y)| C32::new(x.re - y.re, x.im - y.im)).collect();
    rdot(&d, &d).sqrt() / rdot(b, b).sqrt().max(1e-300)
}

#[test]
fn surrogate_device_matches_host() {
    let dims = [16usize, 12, 10];
    let nv: usize = dims.iter().product();
    let dev = Device::new(0).expect("CUDA device");
    let st = dev.default_stream();
    let kern = Arc::new(hasty_recon::gpu::Kernels::new(&dev).unwrap());
    // 2 coils, 3 fixed-time segments of a field, hat interpolators
    let smaps: Vec<Vec<C32>> = (0..2).map(|c| (0..nv).map(|v| C32::new(0.6 + 0.3 * ((v + 7 * c) as f32 * 0.013).sin(), 0.2 * ((v * (c + 1)) as f32 * 0.007).cos())).collect()).collect();
    let om: Vec<f64> = (0..nv).map(|v| 200.0 * ((v % 16) as f64 - 7.5) + 50.0 * ((v / 16) % 12) as f64).collect();
    let (t0, t1, l, ntab) = (1e-3, 9e-3, 3usize, 81usize);
    let tl: Vec<f64> = (0..l).map(|i| t0 + (t1 - t0) * i as f64 / (l - 1) as f64).collect();
    let (tlr, omr) = (&tl, &om);
    let segph: Vec<C32> = (0..l).flat_map(|i| omr.iter().map(move |&w| { let p = -w * tlr[i]; C32::new(p.cos() as f32, p.sin() as f32) }).collect::<Vec<_>>()).collect();
    let dt = (t1 - t0) / (ntab - 1) as f64;
    let mut atab = vec![C32::new(0.0, 0.0); ntab * l];
    for i in 0..ntab {
        let u = (i as f64 * dt) / (tl[1] - tl[0]);
        let j = (u.floor() as usize).min(l - 2);
        let fr = (u - j as f64) as f32;
        atab[i * l + j] = C32::new(1.0 - fr, 0.0);
        atab[i * l + j + 1] = C32::new(fr, 0.0);
    }
    let tabs = SegHostTables { segph, atab, nseg: l, ntab, t0, dt };
    let m = 3000;
    let mut coords = [vec![0f32; m], vec![0f32; m], vec![0f32; m]];
    let mut taus = vec![0f64; m];
    for i in 0..m {
        let s = i as f64 / m as f64;
        let r = 2.9 * (1.0 - (2.0 * s - 1.0).abs());
        let a = 61.0 * s;
        coords[0][i] = (r * a.cos()) as f32;
        coords[1][i] = (r * a.sin()) as f32;
        coords[2][i] = (1.3 * (17.0 * s).sin()) as f32;
        taus[i] = t0 + (t1 - t0) * s;
    }
    let c = [&coords[0][..], &coords[1][..], &coords[2][..]];
    let dw: Vec<f32> = (0..nv).map(|v| smaps.iter().map(|s| s[v].re * s[v].re + s[v].im * s[v].im).sum()).collect();
    let (maps, nmix, loss) = compress(&tabs, nv, 2, &dw, &taus);
    eprintln!("compression to 2 of 3: energy not kept {loss:.2e}");
    let w = pair_weights(&tabs, &nmix, 2, &taus);
    // device symbol vs the direct Fejér sum, both conventions
    let mut lam_dev = vec![];
    for (p, wp) in w.iter().enumerate() {
        let d = chan_symbol_dev(&dev, &st, dims, c, wp).unwrap();
        let (h0, h1) = (chan_symbol_host(dims, c, wp, false), chan_symbol_host(dims, c, wp, true));
        let (e0, e1) = (rel(&d, &h0), rel(&d, &h1));
        eprintln!("pair {p}: device vs host Fejér {e0:.2e} (as is), {e1:.2e} (mirrored k)");
        assert!(e0.min(e1) < 1e-3, "device Chan symbol matches neither convention: {e0:e} / {e1:e}");
        lam_dev.push(d);
    }
    // device apply = host apply for the same symbols and every convention
    let smap_refs: Vec<&[C32]> = smaps.iter().map(|v| &v[..]).collect();
    let mut sd = SurDev::new(&dev, &st, kern.clone(), dims, &maps, lam_dev.clone(), &smap_refs, 0.05).unwrap();
    let x: Vec<C32> = hasty_recon::ppre::phases(nv, 5);
    let dx = st.clone_htod(&x).unwrap();
    let mut dy = st.alloc_zeros::<C32>(nv).unwrap();
    for conv in [SurConv::default(), SurConv { mirror: true, transpose: false, conj_maps: true, scale: 1.7 }, SurConv { mirror: false, transpose: true, conj_maps: false, scale: 0.5 }] {
        sd.set_conv(&st, conv).unwrap();
        sd.apply(&st, &dx, &mut dy).unwrap();
        let got = st.clone_dtoh(&dy).unwrap();
        let sh = SurHost { fft: Arc::new(Fft3::new(dims)), ls: 2, maps: maps.clone(), lam: lam_dev.clone(), smaps: smaps.clone(), mu: 0.05, conv };
        let mut want = vec![C32::new(0.0, 0.0); nv];
        sh.apply(&x, &mut want);
        assert!(rel(&got, &want) < 1e-4, "device vs host surrogate ({conv:?}): {:e}", rel(&got, &want));
    }
    // inner solvers: device = host (identity metric)
    sd.set_conv(&st, SurConv::default()).unwrap();
    let sh = SurHost { fft: Arc::new(Fft3::new(dims)), ls: 2, maps: maps.clone(), lam: lam_dev.clone(), smaps: smaps.clone(), mu: 0.05, conv: SurConv::default() };
    let mut dz = st.alloc_zeros::<C32>(nv).unwrap();
    sd.solve(&st, None, Inner::Pcg(6), &dx, &mut dz).unwrap();
    let got = st.clone_dtoh(&dz).unwrap();
    let mut want = vec![C32::new(0.0, 0.0); nv];
    inner_pcg_host(&mut |a, b| sh.apply(a, b), &mut |a, b| b.copy_from_slice(a), &x, &mut want, 6);
    assert!(rel(&got, &want) < 1e-3, "inner PCG device vs host: {:e}", rel(&got, &want));
    let (lo, hi) = sd.bounds(&st, None, 12, 30.0).unwrap();
    eprintln!("Chebyshev interval [{lo:.3e}, {hi:.3e}]");
    sd.solve(&st, None, Inner::Cheb(8), &dx, &mut dz).unwrap();
    let got = st.clone_dtoh(&dz).unwrap();
    cheb_host(&mut |a, b| sh.apply(a, b), &mut |a, b| b.copy_from_slice(a), &x, &mut want, 8, lo, hi);
    assert!(rel(&got, &want) < 1e-3, "Chebyshev device vs host: {:e}", rel(&got, &want));
}
