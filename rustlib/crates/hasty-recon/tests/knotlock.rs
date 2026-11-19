//! Per-axis knot lock (hasty_recon::knotlock, research/static_lock.md §7): metric / projections, prox properties,
//! classification on synthetic knot stacks, detection power, equivalence with tlock on frame stacks, velocity model.
//! CPU only, small sizes.
use hasty_cuda::C32;
use hasty_recon::flow::FlowDecoder;
use hasty_recon::knotlock::{self, Coupling, KnotBasis, KnotLockOpts, KnotMap, NoiseModel};
use hasty_recon::tlock::{self, FrameLayout, LockOpts};
use num_complex::Complex;

type Cf = Complex<f32>;

struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn g(&mut self) -> f64 {
        (-2.0 * self.u().ln()).sqrt() * (std::f64::consts::TAU * self.u()).cos()
    }
}

/// periodic cubic B-spline weight of knot b at cardiac knot coordinate t ∈ [0, mc) (splinerecon convention)
fn cubic(t: f64, b: usize, mc: usize) -> f64 {
    let n = mc as f64;
    let mut d = (t - b as f64).rem_euclid(n);
    if d > 0.5 * n {
        d -= n;
    }
    let a = d.abs();
    if a <= 1.0 {
        2.0 / 3.0 - a * a + 0.5 * a * a * a
    } else if a <= 2.0 {
        (2.0 - a).powi(3) / 6.0
    } else {
        0.0
    }
}

/// 6 linear respiratory knots at the amplitude quantiles × 16 periodic cubic cardiac knots, exhale-heavy breathing
/// (r = ((1 − cos θ)/2)², θ uniform), cardiac phase independent (or coupled by `couple`): (full Gram, knots)
fn target_gram(kr: usize, mc: usize, couple: f64, n: usize) -> Vec<f64> {
    let mut rng = Rng(0x5151_2323);
    let samp: Vec<(f64, f64)> = (0..n)
        .map(|_| {
            let th = std::f64::consts::TAU * rng.u();
            let r = ((1.0 - th.cos()) / 2.0).powi(2);
            // coupling: the cardiac phase density depends on the breathing state (u^(1 + 4 couple r))
            let c = rng.u().powf(1.0 + 4.0 * couple * r) * mc as f64;
            (r, c)
        })
        .collect();
    let mut rs: Vec<f64> = samp.iter().map(|s| s.0).collect();
    rs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q: Vec<f64> = (0..kr).map(|i| rs[((rs.len() - 1) as f64 * i as f64 / (kr - 1) as f64) as usize]).collect();
    let w = samp.iter().map(|&(r, t)| {
        let i = (0..kr - 1).rev().find(|&i| r >= q[i]).unwrap_or(0);
        let v = ((r - q[i]) / (q[i + 1] - q[i]).max(1e-12)).clamp(0.0, 1.0);
        let rw = [(i, 1.0 - v), (i + 1, v)];
        let i0 = t.floor() as i64;
        let cw: Vec<(usize, f64)> = (i0 - 2..=i0 + 2).map(|b| b.rem_euclid(mc as i64) as usize).map(|b| (b, cubic(t, b, mc))).filter(|p| p.1 != 0.0).collect();
        rw.iter().flat_map(|&(k, a)| cw.iter().map(move |&(b, c)| (k * mc + b, a * c))).collect::<Vec<_>>()
    });
    hasty_recon::knotsolve::temporal_gram(kr * mc, w)
}

fn chol_inv_t(g: &[f64], n: usize) -> nalgebra::DMatrix<f64> {
    // L⁻ᵀ with G = L Lᵀ: x = L⁻ᵀ w has covariance G⁻¹
    let m = nalgebra::DMatrix::<f64>::from_fn(n, n, |i, j| g[i * n + j]);
    let l = m.cholesky().expect("Gram not PD").l();
    l.try_inverse().unwrap().transpose()
}

/// coefficient noise per voxel with covariance s² G_s⁻¹ (white = false) or s² λ I, from white fields w (K × nv)
fn temporal_noise(basis: &KnotBasis, w: &[Vec<f64>], white: bool) -> Vec<Vec<f64>> {
    let (kr, mc, k) = (basis.kr, basis.mc, basis.k());
    let nv = w[0].len();
    let (ar, ac) = (chol_inv_t(&basis.g_r, kr), chol_inv_t(&basis.g_c, mc));
    let lam = (k as f64 / ((0..kr).map(|i| basis.g_r[i * kr + i]).sum::<f64>() * (0..mc).map(|i| basis.g_c[i * mc + i]).sum::<f64>())).sqrt();
    let mut out = vec![vec![0.0; nv]; k];
    for q in 0..nv {
        for a in 0..kr {
            for b in 0..mc {
                // N = A_r W A_cᵀ (row-major κ = k · mc + b)
                let mut s = 0.0;
                if white {
                    s = lam * w[a * mc + b][q];
                } else {
                    for i in 0..kr {
                        for j in 0..mc {
                            s += ar[(a, i)] * w[i * mc + j][q] * ac[(b, j)];
                        }
                    }
                }
                out[a * mc + b][q] = s;
            }
        }
    }
    out
}

fn blur3(v: &[f64], d: [usize; 3], s: f64) -> Vec<f64> {
    let r = (3.0 * s).ceil() as isize;
    let k: Vec<f64> = (-r..=r).map(|i| (-0.5 * (i as f64 / s).powi(2)).exp()).collect();
    let mut x = v.to_vec();
    for a in 0..3 {
        let st = [1, d[0], d[0] * d[1]][a];
        let src = x.clone();
        for q in 0..x.len() {
            let i = (q / st) % d[a];
            let mut acc = 0.0;
            for (t, &g) in k.iter().enumerate() {
                let j = (i as isize + t as isize - r).rem_euclid(d[a] as isize) as usize;
                acc += g * src[q - i * st + j * st];
            }
            x[q] = acc;
        }
    }
    let m = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    x.iter().map(|v| v / m.sqrt()).collect()
}

fn lin6x16() -> KnotBasis {
    KnotBasis::from_gram(&target_gram(6, 16, 0.0, 400_000), 6, 16, true, NoiseModel::Auto)
}

#[test]
fn metric_shares_and_noise_dimensions() {
    let b = lin6x16();
    let n = &b.norms_t(0.0);
    println!("shares resp {:?}", b.om_r.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>());
    println!("shares card min {:.4} max {:.4}; nonsep {:.2e}, metric bound {:.4}", b.om_c.iter().cloned().fold(1.0, f64::min), b.om_c.iter().cloned().fold(0.0, f64::max), b.nonsep, b.metric_bound);
    println!("LS: {n:?}");
    println!("White: {:?}", b.norms_white);
    let (kr, mc) = (6.0, 16.0);
    let close = |a: f64, e: f64| (a - e).abs() < 1e-6 * e.max(1.0);
    assert!(close(n.k_eff, kr * mc) && close(n.n0, 1.0) && close(n.n_pc, kr), "LS dims: {n:?}");
    assert!(close(n.n_card, kr * (mc - 1.0)) && close(n.n_rm, kr - 1.0) && close(n.n_co, mc - 1.0) && close(n.n_int, (kr - 1.0) * (mc - 1.0)));
    // quantile knots: the shares are not uniform (they depend on the amplitude density inside each interval)
    let dev = b.om_r.iter().map(|w| (w - 1.0 / 6.0).abs()).fold(0.0, f64::max);
    assert!(dev > 0.01, "respiratory shares {:?}", b.om_r);
    // operator norm in G of the plain coefficient average over breathing, (1 ω_Eᵀ) ⊗ I with ω_E = 1/kr:
    // ‖G_r^{1/2} 1‖ ‖G_r^{-1/2} ω_E‖ = √(ω_Eᵀ G_r⁻¹ ω_E) ≥ 1, = 1 only for uniform shares (Cauchy–Schwarz)
    let gr = nalgebra::DMatrix::<f64>::from_fn(6, 6, |i, j| b.g_r[i * 6 + j]);
    let we = nalgebra::DVector::<f64>::from_element(6, 1.0 / 6.0);
    let norm_e = (we.transpose() * gr.try_inverse().unwrap() * &we)[(0, 0)].sqrt();
    println!("plain-average breathing lock: G-operator norm {norm_e:.4} (share-weighted projection: 1)");
    assert!(norm_e > 1.0 + 1e-4);
    assert!(b.nonsep < 0.02 && b.metric_bound < 1.1, "independent sampling should be nearly separable: {} {}", b.nonsep, b.metric_bound);
    // G_s⁻¹ for the knot preconditioner
    let (gs, gi) = (b.gs(), b.gs_inverse(1e-12));
    let k = b.k();
    let mut err = 0f64;
    for i in 0..k {
        for j in 0..k {
            let v: f64 = (0..k).map(|l| gi[i * k + l] * gs[l * k + j]).sum();
            err = err.max((v - (i == j) as u8 as f64).abs());
        }
    }
    assert!(err < 1e-6, "gs_inverse: {err}");
    // breathing-locked heart rate (strong coupling): non-separability visible in the diagnostics
    let bc = KnotBasis::from_gram(&target_gram(6, 16, 0.5, 200_000), 6, 16, true, NoiseModel::Ls);
    println!("coupled: nonsep {:.3}, metric bound {:.3}", bc.nonsep, bc.metric_bound);
    assert!(bc.nonsep > 3.0 * b.nonsep);
}

/// random knot stack of nv voxels (K images)
fn rand_stack(rng: &mut Rng, k: usize, nv: usize) -> Vec<Vec<Cf>> {
    (0..k).map(|_| (0..nv).map(|_| Cf::new(rng.g() as f32, rng.g() as f32)).collect()).collect()
}

fn rand_map(rng: &mut Rng, nv: usize, binary: bool) -> KnotMap {
    let w = |rng: &mut Rng| -> Vec<f32> { (0..nv).map(|_| if binary { (rng.u() > 0.5) as u8 as f32 } else { rng.u() as f32 }).collect() };
    KnotMap { w_card: w(rng), w_resp: w(rng), w_rx: w(rng), sigma_raw: (0..nv).map(|_| (0.05 + 0.2 * rng.u()) as f32).collect(), sigma_v: (0..nv).map(|_| (0.05 + 0.2 * rng.u()) as f32).collect(), ..Default::default() }
}

fn lock_z(basis: &KnotBasis, map: &KnotMap, opts: &KnotLockOpts, x: &[Vec<Cf>]) -> Vec<Vec<Cf>> {
    let mut y: Vec<Vec<C32>> = x.iter().map(|s| s.iter().map(|c| C32::new(c.re, c.im)).collect()).collect();
    let mut refs: Vec<&mut [C32]> = y.iter_mut().map(|s| s.as_mut_slice()).collect();
    knotlock::apply_z(&mut refs, basis, map, opts);
    y.iter().map(|s| s.iter().map(|c| Cf::new(c.re, c.im)).collect()).collect()
}

fn sub(a: &[Vec<Cf>], b: &[Vec<Cf>]) -> Vec<Vec<Cf>> {
    a.iter().zip(b).map(|(x, y)| x.iter().zip(y).map(|(p, q)| p - q).collect()).collect()
}

fn gdot(basis: &KnotBasis, a: &[Vec<Cf>], b: &[Vec<Cf>]) -> f64 {
    let ar: Vec<&[Cf]> = a.iter().map(|s| s.as_slice()).collect();
    let br: Vec<&[Cf]> = b.iter().map(|s| s.as_slice()).collect();
    knotlock::gs_dot(basis, &ar, &br)
}

fn all_opts() -> Vec<(String, KnotLockOpts)> {
    let mut v = vec![];
    for group in [false, true] {
        for coupling in [Coupling::Nested, Coupling::Additive] {
            for mu in [0.7, 20.0] {
                v.push((format!("group {group} {coupling:?} μ {mu}"), KnotLockOpts { base: LockOpts { mu, group, ..Default::default() }, coupling }));
            }
        }
    }
    v
}

#[test]
fn prox_is_firmly_nonexpansive_in_the_gs_metric() {
    let mut rng = Rng(0xABCDEF);
    for (name, basis) in [("lin6×cubic16", lin6x16()), ("bins 3 × 8", KnotBasis::frames(3, 8))] {
        let (k, nv) = (basis.k(), 300);
        for binary in [false, true] {
            let map = rand_map(&mut rng, nv, binary);
            for (oname, opts) in all_opts() {
                let mut worst = f64::NEG_INFINITY;
                for _ in 0..4 {
                    let x = rand_stack(&mut rng, k, nv);
                    // y near x in part of the trials (thresholds active on one, not the other)
                    let mut y = rand_stack(&mut rng, k, nv);
                    if rng.u() < 0.5 {
                        y = x.iter().zip(&y).map(|(a, b)| a.iter().zip(b).map(|(p, q)| p + q * 0.05).collect()).collect();
                    }
                    let (dx, dy) = (lock_z(&basis, &map, &opts, &x), lock_z(&basis, &map, &opts, &y));
                    let (dd, xy) = (sub(&dx, &dy), sub(&x, &y));
                    let (lhs, rhs) = (gdot(&basis, &dd, &dd), gdot(&basis, &dd, &xy));
                    worst = worst.max((lhs - rhs) / gdot(&basis, &xy, &xy));
                }
                assert!(worst < 1e-5, "{name} {oname} binary {binary}: ‖Dx−Dy‖² − ⟨Dx−Dy, x−y⟩ = {worst:.2e} (rel)");
            }
        }
        println!("{name}: firm non-expansiveness holds in G_s for all modes");
    }
}

#[test]
fn fixed_points_and_group_threshold() {
    let basis = lin6x16();
    let (kr, mc, k) = (basis.kr, basis.mc, basis.k());
    let nv = 4;
    let map = KnotMap { w_card: vec![1.0; nv], w_resp: vec![1.0; nv], w_rx: vec![1.0; nv], sigma_raw: vec![0.1; nv], sigma_v: vec![0.1; nv], ..Default::default() };
    let mut rng = Rng(77);
    // voxel 0: exactly static; 1: static over the cardiac cycle (a_k); 2: static over breathing (d_b); 3: dynamic
    let a: Vec<Cf> = (0..kr).map(|_| Cf::new(rng.g() as f32, rng.g() as f32)).collect();
    let d: Vec<Cf> = (0..mc).map(|_| Cf::new(rng.g() as f32, rng.g() as f32)).collect();
    let x: Vec<Vec<Cf>> = (0..k).map(|kk| vec![Cf::new(0.7, -0.2), a[kk / mc], d[kk % mc], Cf::new(rng.g() as f32, rng.g() as f32)]).collect();
    for (oname, opts) in all_opts() {
        let y = lock_z(&basis, &map, &opts, &x);
        let err0 = (0..k).map(|kk| (y[kk][0] - x[kk][0]).norm()).fold(0f32, f32::max);
        assert!(err0 < 1e-6, "{oname}: static voxel changed by {err0}");
    }
    // per-axis fixed points: w_c = 1 alone keeps voxel 1, w_r = w_x = 1 alone keeps voxel 2
    for (oname, opts) in all_opts() {
        let mc_only = KnotMap { w_resp: vec![0.0; nv], w_rx: vec![0.0; nv], ..map.clone() };
        let y = lock_z(&basis, &mc_only, &opts, &x);
        let e1 = (0..k).map(|kk| (y[kk][1] - x[kk][1]).norm()).fold(0f32, f32::max);
        assert!(e1 < 1e-5, "{oname}: cardiac-static voxel changed by {e1}");
        if opts.coupling == Coupling::Additive {
            let mr_only = KnotMap { w_card: vec![0.0; nv], ..map.clone() };
            let y = lock_z(&basis, &mr_only, &opts, &x);
            let e2 = (0..k).map(|kk| (y[kk][2] - x[kk][2]).norm()).fold(0f32, f32::max);
            assert!(e2 < 1e-5, "{oname}: breathing-static voxel changed by {e2}");
            // and a hard additive respiratory lock makes the dynamic voxel constant along breathing
            if opts.base.mu > 10.0 && !opts.base.group {
                let y3 = lock_z(&basis, &mr_only, &KnotLockOpts { base: LockOpts { mu: 1e7, ..opts.base.clone() }, ..opts.clone() }, &x);
                let spread = (0..mc).map(|b| (1..kr).map(|kk| (y3[kk * mc + b][3] - y3[b][3]).norm()).fold(0f32, f32::max)).fold(0f32, f32::max);
                assert!(spread < 1e-4, "additive hard respiratory lock left breathing variation {spread}");
            }
        }
    }
    // group: a voxel moving with the heart only keeps its motion minus the threshold; the mean is unchanged
    let s = 0.02f32;
    let mut xm = vec![vec![Cf::new(1.0, 0.0); 1]; k];
    for kk in 0..k {
        xm[kk][0] += d[kk % mc] - d.iter().zip(&basis.om_c).map(|(v, &w)| v * w as f32).sum::<Cf>();
    }
    let mapg = KnotMap { w_card: vec![1.0], w_resp: vec![0.0], w_rx: vec![0.0], sigma_raw: vec![s], ..Default::default() };
    let zero: Vec<Vec<Cf>> = (0..k).map(|_| vec![Cf::new(1.0, 0.0)]).collect();
    let dev = |y: &[Vec<Cf>]| gdot(&basis, &sub(y, &zero), &sub(y, &zero)).sqrt();
    let before = dev(&xm);
    for coupling in [Coupling::Nested, Coupling::Additive] {
        let mu = 1.5;
        let y = lock_z(&basis, &mapg, &KnotLockOpts { base: LockOpts { mu, group: true, ..Default::default() }, coupling }, &xm);
        let n = basis.norms_t(0.0);
        let tau = mu * s as f64 * if coupling == Coupling::Nested { n.n_card } else { n.n_co }.sqrt();
        let after = dev(&y);
        println!("group {coupling:?}: cardiac deviation {before:.4} → {after:.4}, τ {tau:.4}");
        assert!(((before - after) - tau).abs() < 1e-4 * before.max(1.0), "group lock: {before} − {after} ≠ τ {tau}");
        let mean: Cf = (0..k).map(|kk| y[kk][0] * (basis.om_r[kk / mc] * basis.om_c[kk % mc]) as f32).sum();
        assert!((mean - Cf::new(1.0, 0.0)).norm() < 1e-5, "mean changed: {mean}");
    }
}

/// knot stack (kr × mc × ne, FrameLayout f = k · mc + b, encode-minor) of a synthetic phantom; returns (h, regions)
struct Phantom {
    h: Vec<C32>,
    lay: FrameLayout,
    dec: FlowDecoder,
    d: [usize; 3],
}

fn knot_phantom(basis: &KnotBasis, ne: usize, sig_t: f64, seed: u64, white_noise: bool) -> Phantom {
    let d = [24usize, 24, 24];
    let nv: usize = d.iter().product();
    let (kr, mc, k) = (basis.kr, basis.mc, basis.k());
    let m1: Vec<[f64; 3]> = {
        let kk = std::f64::consts::PI / 1.0 / hasty_mrsim::GAMMA;
        [[0.0; 3], [kk, 0.0, 0.0], [0.0, kk, 0.0], [0.0, 0.0, kk]][..ne].to_vec()
    };
    let dec = FlowDecoder::new(&m1);
    let lay = FrameLayout::new(d, [2e-3; 3], mc, kr, ne, false);
    let inb = |p: [usize; 3], lo: [usize; 3], hi: [usize; 3]| (0..3).all(|a| p[a] >= lo[a] && p[a] < hi[a]);
    let mut rng = Rng(seed);
    let s = sig_t / (k as f64).sqrt();
    let mut h = vec![C32::new(0.0, 0.0); nv * k * ne];
    for e in 0..ne {
        // white + aliasing-like (spatially correlated, FOV-dependent amplitude) noise with the temporal model
        let mut wre: Vec<Vec<f64>> = (0..k).map(|_| (0..nv).map(|_| rng.g()).collect()).collect();
        let mut wim: Vec<Vec<f64>> = (0..k).map(|_| (0..nv).map(|_| rng.g()).collect()).collect();
        for kk in 0..k {
            let (a, b) = (blur3(&(0..nv).map(|_| rng.g()).collect::<Vec<_>>(), d, 1.2), blur3(&(0..nv).map(|_| rng.g()).collect::<Vec<_>>(), d, 1.2));
            for q in 0..nv {
                let amp = 0.5 + (q % d[0]) as f64 / d[0] as f64;
                wre[kk][q] = (wre[kk][q] * 0.6 + a[q] * 0.8) * amp;
                wim[kk][q] = (wim[kk][q] * 0.6 + b[q] * 0.8) * amp;
            }
        }
        let (nre, nim) = (temporal_noise(basis, &wre, white_noise), temporal_noise(basis, &wim, white_noise));
        for q in 0..nv {
            let p = [q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1])];
            for kk in 0..kr {
                for b in 0..mc {
                    let ph = std::f64::consts::TAU * b as f64 / mc as f64;
                    let r = kk as f64 / (kr - 1) as f64;
                    let mut m = 1.0f64;
                    let mut v = [0.0f64; 3];
                    // A: moves with the heart only (same cardiac curve in every breathing state)
                    if inb(p, [3, 3, 3], [9, 9, 9]) {
                        m = 1.0 + 0.4 * ph.sin() * ((p[0] as f64 - 2.5) / 6.0);
                    }
                    // B: moves with breathing only
                    if inb(p, [14, 3, 3], [20, 9, 9]) {
                        m = 1.0 + 0.5 * r;
                    }
                    // C: both (heart displaced by breathing: cardiac amplitude and phase depend on r)
                    if inb(p, [3, 14, 3], [9, 20, 9]) {
                        m = 1.0 + 0.3 * r + 0.4 * (ph + 1.5 * r).sin() * (0.5 + r);
                    }
                    // D: cardiac motion whose amplitude changes with breathing, breathing mean unchanged
                    if inb(p, [14, 14, 3], [20, 20, 9]) {
                        m = 1.0 + 0.4 * ph.sin() * (0.3 + 1.4 * r);
                    }
                    // vessel with steady flow along z
                    if inb(p, [10, 10, 0], [13, 13, 24]) && ne > 1 {
                        v = [0.0, 0.0, 0.4];
                    }
                    for e2 in [e] {
                        let z = Complex::<f64>::from_polar(m, 0.3 + dec.phase(e2, &v));
                        let kk2 = kk * mc + b;
                        h[lay.img(kk, b, e2) * nv + q] = C32::new((z.re + s * nre[kk2][q]) as f32, (z.im + s * nim[kk2][q]) as f32);
                    }
                }
            }
        }
    }
    Phantom { h, lay, dec, d }
}

#[test]
fn map_classifies_static_cardiac_resp_dynamic_voxels() {
    let basis = lin6x16();
    let ph = knot_phantom(&basis, 4, 0.02, 0x1234, false);
    let d = ph.d;
    let nv: usize = d.iter().product();
    let opts = KnotLockOpts { base: LockOpts { blur_mm: 2.0, dilate_mm: 2.0, ..Default::default() }, coupling: Coupling::Additive };
    let t = std::time::Instant::now();
    let map = knotlock::estimate(&ph.h, &ph.lay, &basis, &opts, Some(&ph.dec), 0);
    println!("map {:.2}s; s² blur {:.2e}, raw {:.2e}; static card {:.3} resp {:.3} full {:.3}; undetermined {:.3}; model check LS {:.3} White {:.3}", t.elapsed().as_secs_f64(), map.sigma2_blur, map.sigma2_raw, map.static_card, map.static_resp, map.static_full, map.undetermined, map.model_check[0], map.model_check[1]);
    let at = |p: [usize; 3]| p[0] + d[0] * (p[1] + d[1] * p[2]);
    let inb = |q: usize, lo: [usize; 3], hi: [usize; 3]| {
        let p = [q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1])];
        (0..3).all(|a| p[a] >= lo[a] && p[a] < hi[a])
    };
    let bg: Vec<usize> = (0..nv).filter(|&q| !inb(q, [1, 1, 1], [22, 22, 11]) && !inb(q, [8, 8, 0], [15, 15, 24])).collect();
    let mean = |w: &[f32], ids: &[usize]| ids.iter().map(|&q| w[q] as f64).sum::<f64>() / ids.len() as f64;
    let med_s = { let mut v: Vec<f64> = bg.iter().map(|&q| map.s_card[q] as f64).collect(); v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    println!("background: w_card {:.3} w_resp {:.3} w_rx {:.3}; median s_card {med_s:.3}", mean(&map.w_card, &bg), mean(&map.w_resp, &bg), mean(&map.w_rx, &bg));
    assert!(mean(&map.w_card, &bg) > 0.97 && mean(&map.w_resp, &bg) > 0.9 && mean(&map.w_rx, &bg) > 0.9, "background not locked");
    assert!((0.8..1.25).contains(&med_s), "static score not calibrated: {med_s}");
    assert!((map.model_check[0] - 1.0).abs() < 0.15, "LS model check {:?}", map.model_check);
    let show = |n: &str, q: usize| println!("  {n}: s_card {:.2} s_resp {:.2} s_int {:.2} s_flow {:.2} → w_card {:.2} w_resp {:.2} w_rx {:.2}", map.s_card[q], map.s_resp[q], map.s_int[q], map.s_flow[q], map.w_card[q], map.w_resp[q], map.w_rx[q]);
    let (a, b, c, dd, ves) = (at([7, 6, 6]), at([17, 6, 6]), at([6, 17, 6]), at([17, 17, 6]), at([11, 11, 12]));
    for (n, q) in [("A heart only", a), ("B breathing only", b), ("C both", c), ("D cardiac × breathing", dd), ("vessel", ves)] {
        show(n, q);
    }
    assert!(map.w_card[a] == 0.0 && map.w_resp[a] > 0.99 && map.w_rx[a] > 0.99, "A: heart-only voxel should be locked over breathing only");
    assert!(map.w_card[b] > 0.99 && map.w_resp[b] == 0.0, "B: breathing-only voxel should be locked over the cardiac cycle only");
    assert!(map.w_card[c] == 0.0 && map.w_resp[c] == 0.0 && map.w_rx[c] == 0.0, "C: dynamic voxel locked");
    assert!(map.w_card[dd] == 0.0 && map.w_rx[dd] == 0.0, "D: breathing-dependent cardiac curve locked along breathing");
    assert!(map.w_card[ves] == 0.0 && map.w_resp[ves] == 0.0, "steady flow locked");
    // the lock: A keeps its cardiac curve, loses breathing noise; background frozen with a hard lock
    let mut x = ph.h.clone();
    knotlock::apply(&mut x, &ph.lay, &basis, &map, &KnotLockOpts { base: LockOpts { mu: 1e6, ..opts.base.clone() }, ..opts.clone() });
    let stat = *bg.iter().find(|&&q| map.w_card[q] == 1.0 && map.w_resp[q] == 1.0 && map.w_rx[q] == 1.0).unwrap();
    let (mc, kr) = (basis.mc, basis.kr);
    let val = |x: &[C32], kk: usize, b: usize, q: usize| { let z = x[ph.lay.img(kk, b, 1) * nv + q]; Cf::new(z.re, z.im) };
    let spread = |x: &[C32], q: usize| (0..kr).flat_map(|kk| (0..mc).map(move |b| (kk, b))).map(|(kk, b)| (val(x, kk, b, q) - val(x, 0, 0, q)).norm()).fold(0f32, f32::max);
    assert!(spread(&x, stat) < 1e-4, "static voxel not frozen: {}", spread(&x, stat));
    let resp_spread = |x: &[C32], q: usize| (0..mc).map(|b| (1..kr).map(|kk| (val(x, kk, b, q) - val(x, 0, b, q)).norm()).fold(0f32, f32::max)).fold(0f32, f32::max);
    println!("A: breathing spread {:.4} → {:.2e}; cardiac range kept {:.3} → {:.3}", resp_spread(&ph.h, a), resp_spread(&x, a), (0..mc).map(|b| val(&ph.h, 2, b, a).norm()).fold(0f32, f32::max), (0..mc).map(|b| val(&x, 2, b, a).norm()).fold(0f32, f32::max));
    assert!(resp_spread(&x, a) < 1e-4);
}

#[test]
fn detection_power_and_noise_model() {
    // relative cardiac variation (std / mean) in strips along x < 24: 0.05, 0.1, 0.2, 0.3 (6 voxels each); static
    // for x ≥ 24 (the majority, as in a real FOV). Noise σ_t such that the smallest detectable variation is 0.06
    // (conf = 1): variation ≥ detect_ok must not be locked. Coefficient noise LS-coloured or white (Auto must
    // calibrate both); heavy noise: tissue undetermined → not locked, air (x ≥ 48) locked.
    let basis = lin6x16();
    let (kr, mc, k) = (basis.kr, basis.mc, basis.k());
    let d = [60usize, 12, 12];
    let nv: usize = d.iter().product();
    let lay = FrameLayout::new(d, [2e-3; 3], mc, kr, 1, false);
    let rhos = [0.05, 0.1, 0.2, 0.3];
    for (sig_t, white) in [(0.02, false), (0.02, true), (0.5, false)] {
        let mut rng = Rng(99);
        let w: Vec<Vec<f64>> = (0..2 * k).map(|_| (0..nv).map(|_| rng.g()).collect()).collect();
        let nre = temporal_noise(&basis, &w[..k], white);
        let nim = temporal_noise(&basis, &w[k..], white);
        let s = sig_t / (k as f64).sqrt();
        let mut h = vec![C32::new(0.0, 0.0); nv * k];
        for q in 0..nv {
            let x = q % d[0];
            let air = sig_t > 0.1 && x >= 48;
            let rho = if x < 24 { rhos[x / 6] } else { 0.0 };
            for kk in 0..kr {
                for b in 0..mc {
                    let ph = std::f64::consts::TAU * b as f64 / mc as f64;
                    // std of A sin over the cycle = A / √2
                    let m = if air { 0.0 } else { 1.0 + rho * 2f64.sqrt() * ph.sin() };
                    h[lay.img(kk, b, 0) * nv + q] = C32::new((m + s * nre[kk * mc + b][q]) as f32, (s * nim[kk * mc + b][q]) as f32);
                }
            }
        }
        let opts = KnotLockOpts { base: LockOpts { blur_mm: 2.0, dilate_mm: 0.0, ..Default::default() }, ..Default::default() };
        let map = knotlock::estimate(&h, &lay, &basis, &opts, None, 0);
        let region = |lo: usize, hi: usize| -> (f64, f64) {
            let ids: Vec<usize> = (0..nv).filter(|&q| (lo..hi).contains(&(q % d[0]))).collect();
            (ids.iter().map(|&q| map.w_card[q] as f64).sum::<f64>() / ids.len() as f64, ids.iter().map(|&q| map.s_card[q] as f64).sum::<f64>() / ids.len() as f64)
        };
        let ws: Vec<(f64, f64)> = (0..4).map(|i| region(6 * i + 2, 6 * i + 4)).collect();
        let st = region(30, 44);
        println!(
            "σ_t {sig_t} {}: model check LS {:.3} White {:.3}, noise mix t {:.3e} / raw {:.3e}; undetermined {:.3}; static (w, s) {:.2} {:.2}; strips (ρ: w_card, s_card) {}",
            if white { "white coeffs" } else { "LS coeffs" },
            map.model_check[0],
            map.model_check[1],
            map.noise_mix[0],
            map.noise_mix[1],
            map.undetermined,
            st.0,
            st.1,
            rhos.iter().zip(&ws).map(|(r, (w, s))| format!("{r}: {w:.2} {s:.2}")).collect::<Vec<_>>().join(", ")
        );
        if sig_t < 0.1 {
            assert!(st.0 > 0.95 && (0.85..1.2).contains(&st.1), "static region: w {} s {}", st.0, st.1);
            assert!(ws[1].0 < 0.05 && ws[2].0 == 0.0 && ws[3].0 == 0.0, "variation ≥ detect_ok locked: {ws:?}");
            let i = white as usize;
            assert!((map.model_check[i] - 1.0).abs() < 0.15, "model check {:?}", map.model_check);
        } else {
            let ids: Vec<usize> = (0..nv).filter(|&q| q % d[0] >= 50).collect();
            let air = ids.iter().map(|&q| map.w_card[q] as f64).sum::<f64>() / ids.len() as f64;
            assert!(st.0 < 0.05 && air > 0.9, "heavy noise: tissue {}, air {air}", st.0);
        }
    }
}

/// tlock's unit-test phantom: 24³, 2 bins × 10 frames × 4 encodes
fn tlock_phantom() -> (Vec<C32>, FrameLayout, FlowDecoder) {
    let d = [24usize, 24, 24];
    let nv = d.iter().product::<usize>();
    let (nc, nr, ne) = (10usize, 2usize, 4usize);
    let k = std::f64::consts::PI / 1.0 / hasty_mrsim::GAMMA;
    let dec = FlowDecoder::new(&[[0.0; 3], [k, 0.0, 0.0], [0.0, k, 0.0], [0.0, 0.0, k]]);
    let lay = FrameLayout::new(d, [2e-3; 3], nc, nr, ne, false);
    let inbox = |q: usize, lo: [usize; 3], hi: [usize; 3]| {
        let p = [q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1])];
        (0..3).all(|a| p[a] >= lo[a] && p[a] < hi[a])
    };
    let mut rng = Rng(0x1234_5678);
    let mut h = vec![C32::new(0.0, 0.0); nv * nc * nr * ne];
    for r in 0..nr {
        for c in 0..nc {
            let ph = std::f64::consts::TAU * c as f64 / nc as f64;
            for q in 0..nv {
                let p = [q % d[0], (q / d[0]) % d[1], q / (d[0] * d[1])];
                let mut m = 1.0f64;
                let mut v = [0.0f64; 3];
                if inbox(q, [4, 4, 4], [10, 10, 10]) {
                    m = 1.0 + 0.6 * ph.sin() * ((p[0] as f64 - 3.5) / 6.0);
                }
                if inbox(q, [14, 4, 4], [20, 10, 10]) {
                    m = 1.0 + 0.5 * r as f64;
                }
                if inbox(q, [10, 16, 0], [13, 19, 24]) {
                    v = [0.0, 0.0, 0.4];
                }
                for e in 0..ne {
                    let z = Complex::<f64>::from_polar(m, 0.3 + dec.phase(e, &v));
                    h[lay.img(r, c, e) * nv + q] = C32::new(z.re as f32 + 0.02 * rng.g() as f32, z.im as f32 + 0.02 * rng.g() as f32);
                }
            }
        }
    }
    (h, lay, dec)
}

#[test]
fn frames_basis_reproduces_tlock() {
    let (h, lay, dec) = tlock_phantom();
    let nv = lay.nv();
    let lo = LockOpts { blur_mm: 2.0, dilate_mm: 2.0, ..Default::default() };
    let tm = tlock::estimate(&h, &lay, &lo, Some(&dec), 0);
    let basis = KnotBasis::frames(lay.nr, lay.nc);
    let k = basis.k() as f32;
    let ko = KnotLockOpts { base: lo.clone(), coupling: Coupling::Nested };
    let km = knotlock::estimate(&h, &lay, &basis, &ko, Some(&dec), 0);
    let maxd = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs() / (1.0 + x.abs())).fold(0f32, f32::max);
    let sr_scaled: Vec<f32> = tm.sigma_raw.iter().map(|v| v / k.sqrt()).collect();
    let diffs = [maxd(&tm.s_card, &km.s_card), maxd(&tm.s_resp, &km.s_resp), maxd(&tm.s_flow, &km.s_flow), maxd(&tm.w_card, &km.w_card), maxd(&tm.w_resp, &km.w_resp), maxd(&sr_scaled, &km.sigma_raw)];
    println!("map vs tlock (s_card, s_resp, s_flow, w_card, w_resp, σ_raw/√K): max rel diff {diffs:?}; σ² ratio {:.6}", tm.sigma2_blur / (km.sigma2_blur * k as f64));
    // tlock forms variances as Σx² − |Σx|²/n in f32-accumulated sums: for static voxels (V ≈ σ²/F ≪ |x|²) that
    // cancels to a few % (s_resp: 2 bins); the error is absolute in V, so it vanishes where the score matters
    let rel_hi = |a: &[f32], b: &[f32]| a.iter().zip(b).filter(|(x, _)| **x > 3.0).map(|(x, y)| (x - y).abs() / x).fold(0f32, f32::max);
    let hi = [rel_hi(&tm.s_card, &km.s_card), rel_hi(&tm.s_resp, &km.s_resp)];
    println!("  where s > 3: max rel diff s_card {:.2e}, s_resp {:.2e}", hi[0], hi[1]);
    assert!(hi.iter().all(|&v| v < 5e-3) && diffs[0] < 5e-3 && diffs[1] < 0.15 && diffs[2] < 1e-4 && diffs[5] < 1e-4, "map differs from tlock: {diffs:?}");
    assert!(diffs[3] < 0.02 && diffs[4] < 0.02, "weights differ: {diffs:?}");
    assert!(km.w_rx.iter().all(|&v| v == 0.0));
    // the lock with tlock's map
    let map = KnotMap { w_card: tm.w_card.clone(), w_resp: tm.w_resp.clone(), w_rx: vec![0.0; nv], sigma_raw: sr_scaled, ..Default::default() };
    for (mu, group) in [(20.0, false), (1e6, false), (3.0, true), (0.5, true)] {
        let lo2 = LockOpts { mu, group, ..lo.clone() };
        let mut a = h.clone();
        tlock::apply(&mut a, &lay, &tm, &lo2);
        for coupling in [Coupling::Nested, Coupling::Additive] {
            if group && coupling == Coupling::Additive {
                continue; // additive groups split the cardiac piece: a different (still valid) prox
            }
            let mut b = h.clone();
            knotlock::apply(&mut b, &lay, &basis, &map, &KnotLockOpts { base: lo2.clone(), coupling });
            let e = a.iter().zip(&b).map(|(x, y)| ((x.re - y.re).powi(2) + (x.im - y.im).powi(2)).sqrt()).fold(0f32, f32::max);
            println!("lock μ {mu} group {group} {coupling:?}: max |tlock − knotlock| {e:.2e}");
            assert!(e < 1e-5, "lock differs from tlock: {e}");
        }
    }
}

#[test]
fn velocity_model_lock() {
    let basis = lin6x16();
    let (kr, mc, k) = (basis.kr, basis.mc, basis.k());
    let d = [24usize, 16, 16];
    let nv: usize = d.iter().product();
    let kk = std::f64::consts::PI / 1.0 / hasty_mrsim::GAMMA;
    let dec = FlowDecoder::new(&[[0.0; 3], [kk, 0.0, 0.0], [0.0, kk, 0.0], [0.0, 0.0, kk]]);
    let mut rng = Rng(4242);
    let sz = 0.02 / (k as f64).sqrt();
    let sv = 0.02;
    // x < 8: static tissue, 8 ≤ x < 16: pulsatile flow (v_z(b), heart only), x ≥ 16: steady flow v_z = 0.4
    let mut z: Vec<Vec<C32>> = vec![vec![C32::new(0.0, 0.0); nv]; k];
    let mut v: Vec<Vec<f32>> = vec![vec![0.0; 3 * nv]; k];
    for kk2 in 0..k {
        let b = kk2 % mc;
        let ph = std::f64::consts::TAU * b as f64 / mc as f64;
        for q in 0..nv {
            let x = q % d[0];
            z[kk2][q] = C32::new((1.0 + sz * rng.g()) as f32, (sz * rng.g()) as f32);
            let vz = if x < 8 { 0.0 } else if x < 16 { 0.3 + 0.25 * ph.sin() } else { 0.4 };
            for c in 0..3 {
                v[kk2][c * nv + q] = (if c == 2 { vz } else { 0.0 } + sv * rng.g()) as f32;
            }
        }
    }
    let zr: Vec<&[C32]> = z.iter().map(|s| s.as_slice()).collect();
    let vr: Vec<&[f32]> = v.iter().map(|s| s.as_slice()).collect();
    let opts = KnotLockOpts { base: LockOpts { blur_mm: 2.0, dilate_mm: 0.0, ..Default::default() }, ..Default::default() };
    let t = std::time::Instant::now();
    let map = knotlock::estimate_vel(&zr, &vr, d, [2e-3; 3], &basis, &opts, &dec, 0);
    let region = |w: &[f32], lo: usize, hi: usize| { let ids: Vec<usize> = (0..nv).filter(|&q| (lo..hi).contains(&(q % d[0]))).collect(); ids.iter().map(|&q| w[q] as f64).sum::<f64>() / ids.len() as f64 };
    let med = |w: &[f32]| { let mut s = w.to_vec(); s.sort_by(|a, b| a.partial_cmp(b).unwrap()); s[s.len() / 2] };
    // injected v noise per coefficient sv, LS-normalised per dof: compare the order of magnitude
    println!("vel map {:.2}s: static w_card {:.3} w_resp {:.3} | pulsatile {:.3} {:.3} | steady {:.3} {:.3}; σ_v median {:.3e}, σ_z {:.3e} (per dof); noise mix {:.3?}", t.elapsed().as_secs_f64(), region(&map.w_card, 0, 6), region(&map.w_resp, 0, 6), region(&map.w_card, 9, 15), region(&map.w_resp, 9, 15), region(&map.w_card, 17, 23), region(&map.w_resp, 17, 23), med(&map.sigma_v), med(&map.sigma_raw), map.noise_mix);
    assert!(region(&map.w_card, 0, 6) > 0.95 && region(&map.w_resp, 0, 6) > 0.95);
    assert!(region(&map.w_card, 9, 15) < 0.02, "pulsatile flow locked over the cardiac cycle");
    // flowing voxels stay unlocked on both axes (tlock's s_v rule), even where the flow does not vary with breathing
    assert!(region(&map.w_resp, 9, 15) < 0.02, "flow locked over breathing");
    assert!(region(&map.w_card, 17, 23) < 0.02 && region(&map.w_resp, 17, 23) < 0.02, "steady flow locked");
    // the v lock: mean velocity kept everywhere; static region becomes constant; pulsatile flow keeps its curve
    // lock test map: the pulsatile region locked over breathing (as a non-flowing heart-only voxel would be)
    let mut lmap = map.clone();
    for q in 0..nv {
        if (8..16).contains(&(q % d[0])) {
            lmap.w_resp[q] = 1.0;
            lmap.w_rx[q] = 1.0;
        }
    }
    let mean_v = |v: &[Vec<f32>], q: usize| -> f64 { (0..k).map(|kk2| v[kk2][2 * nv + q] as f64 * basis.om_r[kk2 / mc] * basis.om_c[kk2 % mc]).sum() };
    for group in [false, true] {
        let mut v2 = v.clone();
        let o = KnotLockOpts { base: LockOpts { mu: if group { 2.0 } else { 1e6 }, group, ..opts.base.clone() }, ..opts.clone() };
        {
            let mut refs: Vec<&mut [f32]> = v2.iter_mut().map(|s| s.as_mut_slice()).collect();
            knotlock::apply_v(&mut refs, &basis, &lmap, &o);
        }
        let dm = (0..nv).map(|q| (mean_v(&v2, q) - mean_v(&v, q)).abs()).fold(0f64, f64::max);
        let q_s = 3 + d[0] * (8 + d[1] * 8);
        let q_p = 12 + d[0] * (8 + d[1] * 8);
        let spread = |v: &[Vec<f32>], q: usize| (0..k).map(|kk2| (v[kk2][2 * nv + q] - v[0][2 * nv + q]).abs()).fold(0f32, f32::max);
        let resp_spread = |v: &[Vec<f32>], q: usize| (0..mc).map(|b| (1..kr).map(|kk2| (v[kk2 * mc + b][2 * nv + q] - v[b][2 * nv + q]).abs()).fold(0f32, f32::max)).fold(0f32, f32::max);
        let card_range = |v: &[Vec<f32>], q: usize| { let s: Vec<f32> = (0..mc).map(|b| v[2 * mc + b][2 * nv + q]).collect(); s.iter().cloned().fold(f32::MIN, f32::max) - s.iter().cloned().fold(f32::MAX, f32::min) };
        println!("v lock group {group}: Δ mean v max {dm:.2e}; static spread {:.3} → {:.2e}; pulsatile breathing spread {:.3} → {:.3e}, cardiac range {:.3} → {:.3}", spread(&v, q_s), spread(&v2, q_s), resp_spread(&v, q_p), resp_spread(&v2, q_p), card_range(&v, q_p), card_range(&v2, q_p));
        assert!(dm < 1e-5, "mean velocity changed by {dm}");
        assert!(spread(&v2, q_s) < 1e-3 * spread(&v, q_s).max(1e-3) || group && spread(&v2, q_s) < 1e-6, "static velocity not constant");
        assert!(card_range(&v2, q_p) > 0.4, "pulsatile curve lost");
        assert!(resp_spread(&v2, q_p) < 0.1 * resp_spread(&v, q_p));
    }
    // z lock (complex, z's own noise level)
    let mut z2 = z.clone();
    let mut refs: Vec<&mut [C32]> = z2.iter_mut().map(|s| s.as_mut_slice()).collect();
    knotlock::apply_z(&mut refs, &basis, &map, &KnotLockOpts { base: LockOpts { mu: 2.0, group: true, ..opts.base.clone() }, ..opts.clone() });
    let q_s = 3 + d[0] * (8 + d[1] * 8);
    let e = (0..k).map(|kk2| ((z2[kk2][q_s].re - z2[0][q_s].re).powi(2) + (z2[kk2][q_s].im - z2[0][q_s].im).powi(2)).sqrt()).fold(0f32, f32::max);
    assert!(e < 1e-6, "static z not constant under the group lock: {e}");
}

#[test]
fn lock_and_map_timing() {
    // 64³ × 96 knots × 1 encode (25 M complex, 200 MB): per-voxel cost for the extrapolation to the 2 mm target
    let basis = lin6x16();
    let (kr, mc, k) = (basis.kr, basis.mc, basis.k());
    let d = [64usize, 64, 64];
    let nv: usize = d.iter().product();
    let lay = FrameLayout::new(d, [2e-3; 3], mc, kr, 1, false);
    let mut rng = Rng(5);
    let h: Vec<C32> = (0..nv * k).map(|_| C32::new(1.0 + 0.01 * rng.g() as f32, 0.01 * rng.g() as f32)).collect();
    let opts = KnotLockOpts { base: LockOpts { blur_mm: 3.0, ..Default::default() }, ..Default::default() };
    let t = std::time::Instant::now();
    let map = knotlock::estimate(&h, &lay, &basis, &opts, None, 0);
    let tm = t.elapsed().as_secs_f64();
    let full = KnotMap { w_card: vec![1.0; nv], w_resp: vec![1.0; nv], w_rx: vec![1.0; nv], ..map.clone() };
    let mut x = h.clone();
    let mut tq = f64::INFINITY;
    let mut tg = f64::INFINITY;
    for _ in 0..3 {
        let t = std::time::Instant::now();
        knotlock::apply(&mut x, &lay, &basis, &full, &opts);
        tq = tq.min(t.elapsed().as_secs_f64());
        let t = std::time::Instant::now();
        knotlock::apply(&mut x, &lay, &basis, &full, &KnotLockOpts { base: LockOpts { group: true, mu: 1.5, ..opts.base.clone() }, ..opts.clone() });
        tg = tg.min(t.elapsed().as_secs_f64());
    }
    let scale = 4.17e6 * 7.0 / nv as f64;
    println!("64³ × 96 knots: map {tm:.2}s, lock quadratic {tq:.3}s, group {tg:.3}s (threads {}); 2 mm target (4.17 M voxels × 96 knots × 7 encodes): map ≈ {:.0}s, lock ≈ {:.1}s quadratic / {:.1}s group", rayon::current_num_threads(), tm * scale, tq * scale, tg * scale);
}
