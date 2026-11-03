use hasty_cuda::{Device, C32};
use hasty_nufft::{ChirpFamily, ChirpNufft, NufftOpts};
use hasty_testkit::{rel_l2, Rng, C64 as RC64};

fn xi(i: usize, n: usize) -> f64 {
    (i as f64 - (n as f64 - 1.0) / 2.0) * 2.0 / n as f64
}
fn mq(q: usize, x: f64, y: f64, z: f64) -> f64 {
    [z * z, x * x + y * y, x * z, y * z][q]
}

/// y_j = sum_i f_i exp(-i k_j.(i - N/2)) exp(-i Phi_j(xi_i)),  i = (ix, iy, iz), N = (Nx, Ny, Nz)
fn direct(modes: [usize; 3], k: &[Vec<f64>; 3], w: &[Vec<f64>; 4], f: &[RC64]) -> Vec<RC64> {
    let m = k[0].len();
    (0..m)
        .map(|j| {
            let mut acc = RC64::new(0.0, 0.0);
            for iz in 0..modes[2] {
                for iy in 0..modes[1] {
                    for ix in 0..modes[0] {
                        let (x, y, z) = (xi(ix, modes[0]), xi(iy, modes[1]), xi(iz, modes[2]));
                        let phi: f64 = (0..4).map(|q| w[q][j] * mq(q, x, y, z)).sum();
                        let lin = k[0][j] * (ix as f64 - (modes[0] / 2) as f64)
                            + k[1][j] * (iy as f64 - (modes[1] / 2) as f64)
                            + k[2][j] * (iz as f64 - (modes[2] / 2) as f64);
                        acc += f[(iz * modes[1] + iy) * modes[0] + ix]
                            * RC64::from_polar(1.0, -(lin + phi));
                    }
                }
            }
            acc
        })
        .collect()
}

fn adjoint_direct(
    modes: [usize; 3],
    k: &[Vec<f64>; 3],
    w: &[Vec<f64>; 4],
    c: &[RC64],
) -> Vec<RC64> {
    let nm: usize = modes.iter().product();
    let mut out = vec![RC64::new(0.0, 0.0); nm];
    for j in 0..c.len() {
        for iz in 0..modes[2] {
            for iy in 0..modes[1] {
                for ix in 0..modes[0] {
                    let (x, y, z) = (xi(ix, modes[0]), xi(iy, modes[1]), xi(iz, modes[2]));
                    let phi: f64 = (0..4).map(|q| w[q][j] * mq(q, x, y, z)).sum();
                    let lin = k[0][j] * (ix as f64 - (modes[0] / 2) as f64)
                        + k[1][j] * (iy as f64 - (modes[1] / 2) as f64)
                        + k[2][j] * (iz as f64 - (modes[2] / 2) as f64);
                    out[(iz * modes[1] + iy) * modes[0] + ix] +=
                        c[j] * RC64::from_polar(1.0, lin + phi);
                }
            }
        }
    }
    out
}

#[test]
fn chirp_nufft_matches_direct_sums() {
    let dev = Device::new(0).unwrap();
    let stream = dev.default_stream();
    let modes = [14usize, 12, 10];
    let m = 300;
    let mut r = Rng::new(21);
    let k: [Vec<f64>; 3] = [0, 1, 2].map(|_| (0..m).map(|_| r.uniform() * 2.9).collect());
    let phi_box = 1.5;
    let w: [Vec<f64>; 4] =
        [0, 1, 2, 3].map(|_| (0..m).map(|_| phi_box * r.uniform() * 0.5).collect());
    let nm: usize = modes.iter().product();
    let f = r.complex_vec(nm);
    let c = r.complex_vec(m);
    let fam = ChirpFamily {
        w_box: [0.75; 4],
        s_half: 4,
        pod_tol: 3e-4,
        l_max: 120,
        nfov: 13,
        ridge: 1e-7,
        n_train: 400,
        train: None,
    };
    let mut op = ChirpNufft::new(
        &dev,
        modes,
        NufftOpts {
            tol: 1e-5,
            sigma: 2.0,
            ntransf: 1,
            ..Default::default()
        },
        fam,
    )
    .unwrap();
    eprintln!(
        "L = {}, held-out row representation error (max, mean) = {:?}",
        op.tables.l, op.tables.heldout_err
    );
    let up = |v: &Vec<f64>| {
        stream
            .clone_htod(&v.iter().map(|&x| x as f32).collect::<Vec<f32>>())
            .unwrap()
    };
    let dk: Vec<_> = k.iter().map(up).collect();
    let dw: Vec<_> = w.iter().map(up).collect();
    op.set_points(
        &stream,
        &[&dk[0], &dk[1], &dk[2]],
        [&dw[0], &dw[1], &dw[2], &dw[3]],
        m,
    )
    .unwrap();
    let to32 = |v: &[RC64]| {
        v.iter()
            .map(|c| C32::new(c.re as f32, c.im as f32))
            .collect::<Vec<_>>()
    };
    let from32 = |v: &[C32]| {
        v.iter()
            .map(|c| RC64::new(c.re as f64, c.im as f64))
            .collect::<Vec<_>>()
    };
    let df = stream.clone_htod(&to32(&f)).unwrap();
    let mut dc = stream.alloc_zeros::<C32>(m).unwrap();
    op.forward(&stream, &df, &mut dc).unwrap();
    let got = from32(&stream.clone_dtoh(&dc).unwrap());
    let want = direct(modes, &k, &w, &f);
    let e_fwd = rel_l2(&got, &want);
    // plain NUFFT (no chirp) as the scale of the effect we are correcting
    let plain_want = direct(
        modes,
        &k,
        &[vec![0.0; m], vec![0.0; m], vec![0.0; m], vec![0.0; m]],
        &f,
    );
    let effect = rel_l2(&plain_want, &want);

    let dcin = stream.clone_htod(&to32(&c)).unwrap();
    let mut dfo = stream.alloc_zeros::<C32>(nm).unwrap();
    op.adjoint(&stream, &dcin, &mut dfo).unwrap();
    let got_a = from32(&stream.clone_dtoh(&dfo).unwrap());
    let want_a = adjoint_direct(modes, &k, &w, &c);
    let e_adj = rel_l2(&got_a, &want_a);
    eprintln!("forward err {e_fwd:e}, adjoint err {e_adj:e}  (chirp effect itself: {effect:e})");
    assert!(
        e_fwd < 0.03 * effect && e_adj < 0.03 * effect,
        "fwd {e_fwd:e} adj {e_adj:e} effect {effect:e}"
    );

    // precomputed fp16 stencils + tiled gather/scatter must reproduce the on-the-fly result to fp16 rounding
    op.precompute_stencils(&stream).unwrap();
    let mut dc2 = stream.alloc_zeros::<C32>(m).unwrap();
    op.forward(&stream, &df, &mut dc2).unwrap();
    let got2 = from32(&stream.clone_dtoh(&dc2).unwrap());
    let e_stored = rel_l2(&got2, &got);
    let mut dfo2 = stream.alloc_zeros::<C32>(nm).unwrap();
    op.adjoint(&stream, &dcin, &mut dfo2).unwrap();
    let got_a2 = from32(&stream.clone_dtoh(&dfo2).unwrap());
    let e_stored_a = rel_l2(&got_a2, &got_a);
    eprintln!("stored-stencil path vs on-the-fly: forward {e_stored:e}, adjoint {e_stored_a:e}");
    assert!(e_stored < 2e-3 && e_stored_a < 2e-3, "stored path mismatch {e_stored:e} {e_stored_a:e}");
}
