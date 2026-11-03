use hasty_cuda::{Device, C32, C64};
use hasty_mrsim::direct::{direct_sum_cpu, pack_sources, DirectSim, Source};
use hasty_mrsim::gnl::GradientCoils;
use hasty_mrsim::sequence::{Acquisition, FlowSpiralParams, SAMPLE_FLOATS};
use hasty_mrsim::trajectory::spiral_readout;
use hasty_mrsim::waveform::{bipolar, trapezoid, Waveform};
use hasty_mrsim::{GradLimits, GAMMA, GAMMA_BAR};
use num_complex::Complex64;

#[test]
fn trapezoid_and_bipolar_hit_targets_within_limits() {
    let lim = GradLimits::clinical_3t();
    let dt = 4e-6;
    for &area in &[1e-6, 3e-5, -2e-4] {
        let t = trapezoid(area, &lim, dt);
        let w = Waveform::from_axes([&t, &[], &[]], dt, 0.0);
        let m = w.moments().last();
        assert!((m.m0[0] - area).abs() < 1e-9 * area.abs().max(1e-6), "area {area} got {}", m.m0[0]);
        assert!(w.max_amplitude() <= lim.gmax * (1.0 + 1e-9) && w.max_slew() <= lim.smax * (1.0 + 1e-6));
    }
    let venc = 1.5;
    let m1 = std::f64::consts::PI / (GAMMA * venc);
    let b = bipolar(m1, &lim, dt);
    let w = Waveform::from_axes([&b, &[], &[]], dt, 1e-3);
    let m = w.moments().last();
    assert!(m.m0[0].abs() < 1e-12, "bipolar M0 {}", m.m0[0]);
    assert!((m.m1[0] - m1).abs() < 1e-6 * m1, "bipolar M1 {} vs {}", m.m1[0], m1);
    assert!(w.max_amplitude() <= lim.gmax * (1.0 + 1e-9) && w.max_slew() <= lim.smax * (1.0 + 1e-6));
    eprintln!("VENC 1.5 m/s bipolar: {:.3} ms", w.duration() * 1e3);
}

#[test]
fn exact_moments_at_arbitrary_times() {
    let dt = 1e-5;
    let g: Vec<f64> = (0..200).map(|i| (i as f64 * 0.07).sin() * 0.02).collect();
    let w = Waveform::from_axes([&g, &g, &[]], dt, 2e-3);
    let m = w.moments();
    // compare with a 64x finer raster of the same piecewise-linear waveform
    let fine: Vec<f64> = (0..(199 * 64 + 1)).map(|j| {
        let x = j as f64 / 64.0;
        let i = (x.floor() as usize).min(198);
        let f = x - i as f64;
        g[i] * (1.0 - f) + g[i + 1] * f
    }).collect();
    let wf = Waveform::from_axes([&fine, &[], &[]], dt / 64.0, 2e-3);
    let mf = wf.moments();
    for &j in &[37usize, 1000, 5555, 12000] {
        let t = mf.t[j];
        let a = m.at(t);
        assert!((a.m0[0] - mf.m0[j][0]).abs() < 1e-14, "m0 {} vs {}", a.m0[0], mf.m0[j][0]);
        assert!((a.m1[0] - mf.m1[j][0]).abs() < 1e-16);
        assert!((a.m2[0] - mf.m2[j][0]).abs() < 1e-18);
        assert!((a.p[0] - mf.p[j][0]).abs() < 1e-14);
    }
}

#[test]
fn spiral_design_respects_limits_and_reaches_kmax() {
    let lim = GradLimits::clinical_3t().derated(0.95);
    let kmax = 0.5 / 0.002;
    let ro = spiral_readout(kmax, 16, 0.36, 0.36 * 0.7, &lim, 4e-6);
    let w = &ro.wave;
    let m = w.moments();
    let kend = m.at(ro.adc.1).m0;
    let kr = (kend[0].hypot(kend[1])) * GAMMA_BAR;
    eprintln!("spiral: ADC {:.2} ms, total {:.2} ms, |k_end| = {:.1} (kmax {:.1}), Gmax {:.1} mT/m, Smax {:.1} T/m/s",
        (ro.adc.1 - ro.adc.0) * 1e3, w.duration() * 1e3, kr, kmax, w.max_amplitude() * 1e3, w.max_slew());
    assert!((kr - kmax).abs() < 0.01 * kmax);
    assert!(w.max_amplitude() <= lim.gmax * 1.001);
    assert!(w.max_slew() <= lim.smax * 1.05);
}

#[test]
fn gnl_field_is_harmonic_and_ideal_at_centre() {
    let c = GradientCoils::clinical_synthetic();
    let x = [0.11, -0.07, 0.15];
    let h = 1e-3;
    for a in 0..3 {
        let mut lap = 0.0;
        for b in 0..3 {
            let (mut xp, mut xm) = (x, x);
            xp[b] += h;
            xm[b] -= h;
            lap += (c.field(&xp)[a] - 2.0 * c.field(&x)[a] + c.field(&xm)[a]) / (h * h);
        }
        // finite-difference truncation of the degree-5 terms is ~1e-5; a non-harmonic cubic would give ~0.3
        assert!(lap.abs() < 1e-4, "laplacian of f_{a} = {lap}");
    }
    let j0 = c.jacobian(&[0.0; 3]);
    for a in 0..3 {
        for b in 0..3 {
            assert!((j0[a][b] - if a == b { 1.0 } else { 0.0 }).abs() < 1e-7);
        }
    }
    let f = c.field(&[0.0, 0.0, 0.2]);
    eprintln!("GNL z-displacement at z = 20 cm: {:.1} mm; x at x = 20 cm: {:.1} mm", (f[2] - 0.2) * 1e3, (c.field(&[0.2, 0.0, 0.0])[0] - 0.2) * 1e3);
}

#[test]
fn acquisition_schedule_and_encoding() {
    let mut p = FlowSpiralParams::aorta_3t_default();
    p.scan_time = 5.0;
    let acq = Acquisition::new(p.clone());
    eprintln!("TE {:.2} ms, TR {:.2} ms, ADC samples {}, readouts {}, partitions {}", acq.t_ro * 1e3, acq.tr * 1e3, acq.n_adc(), acq.readouts.len(), acq.kz.len());
    // velocity encoding: γ (M1_e − M1_ref) = π / VENC along the encoded axis (at the start of the readout)
    for j in [0, acq.kz.len() / 2, acq.kz.len() - 1] {
    let m_ref = acq.pre[0][j];
    for e in 1..4 {
        let d = acq.pre[e][j].m1[e - 1] - m_ref.m1[e - 1];
        assert!((GAMMA * d - std::f64::consts::PI / p.venc).abs() < 1e-6 * std::f64::consts::PI / p.venc, "encode {e}: {}", GAMMA * d);
    }
    }
    // kz of each partition reached by the blip
    for (j, &kz) in acq.kz.iter().enumerate() {
        assert!((acq.pre[0][j].m0[2] * GAMMA_BAR - kz).abs() < 1e-6 * (1.0 + kz.abs()));
    }
    let sp = acq.sample_params(0..3);
    assert_eq!(sp.len(), 3 * acq.n_adc() * SAMPLE_FLOATS);
    assert!(sp.iter().all(|v| v.is_finite()));
}

#[test]
fn gpu_direct_sum_matches_f64_cpu() {
    let dev = Device::new(0).unwrap();
    let stream = dev.default_stream();
    let mut p = FlowSpiralParams::aorta_3t_default();
    p.scan_time = 0.2;
    let acq = Acquisition::new(p);
    let samples = acq.sample_params(0..2);
    let ns = samples.len() / SAMPLE_FLOATS;
    // subsample to keep the CPU reference cheap
    let pick: Vec<usize> = (0..ns).step_by(37).collect();
    let smp: Vec<f32> = pick.iter().flat_map(|&i| samples[i * SAMPLE_FLOATS..(i + 1) * SAMPLE_FLOATS].to_vec()).collect();
    let mut st = 12345u64;
    let mut rnd = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        (st >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    let coils = GradientCoils::clinical_synthetic();
    let np = 3000;
    let nc = 4;
    let sources: Vec<Source> = (0..np)
        .map(|_| {
            let x = [rnd() * 0.3, rnd() * 0.25, rnd() * 0.3];
            let u = coils.field(&x);
            Source { u, x, vt: [rnd(), rnd(), rnd()], vtt: [rnd() * 5.0, 0.0, 0.0], at: [rnd() * 20.0, rnd() * 20.0, 0.0], omega: rnd() * 2000.0, r2s: 30.0 + 20.0 * rnd() }
        })
        .collect();
    let amp: Vec<Complex64> = (0..np * nc).map(|_| Complex64::new(rnd(), rnd())).collect();
    let want = direct_sum_cpu(&smp, &sources, &amp, nc);
    let sim = DirectSim::new(&dev, nc).unwrap();
    let d_s = stream.clone_htod(&smp).unwrap();
    let d_p = stream.clone_htod(&pack_sources(&sources)).unwrap();
    let d_a = stream.clone_htod(&amp.iter().map(|c| C32::new(c.re as f32, c.im as f32)).collect::<Vec<_>>()).unwrap();
    let mut d_o = stream.alloc_zeros::<C64>(pick.len() * nc).unwrap();
    sim.accumulate(&stream, &d_s, pick.len(), &d_p, &d_a, np, &mut d_o).unwrap();
    let got = stream.clone_dtoh(&d_o).unwrap();
    let (mut num, mut den) = (0.0, 0.0);
    for (g, w) in got.iter().zip(&want) {
        num += (g.re - w.re).powi(2) + (g.im - w.im).powi(2);
        den += w.norm_sqr();
    }
    let rel = (num / den).sqrt();
    eprintln!("GPU direct sum vs f64 CPU: relative error {rel:.2e} over {} samples x {np} sources x {nc} coils", pick.len());
    assert!(rel < 1e-4, "relative error {rel}");
}
