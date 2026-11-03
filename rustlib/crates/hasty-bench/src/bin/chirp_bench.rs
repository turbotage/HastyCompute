//! Chirp-aware NUFFT normal operator  A^H A  (forward then adjoint, f32): ms per coil.
//!   chirp_bench [--sigma s] [--tol t] [--shalf S] N:K:C ...        radial-spoke trajectory, small concomitant coefficients
use cudarc::driver::sys::CUevent_flags;
use hasty_cuda::{Device, C32};
use hasty_nufft::{ChirpFamily, ChirpNufft, NufftOpts};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        let mut s = self.0;
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        self.0 = s;
        (s.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).map(|i| {
            args.remove(i);
            args.remove(i)
        })
    };
    let sigma: f64 = take("--sigma").map(|s| s.parse().unwrap()).unwrap_or(2.0);
    let tol: f64 = take("--tol").map(|s| s.parse().unwrap()).unwrap_or(1e-4);
    let shalf: usize = take("--shalf").map(|s| s.parse().unwrap()).unwrap_or(4);
    let wbox: f64 = take("--wbox").map(|s| s.parse().unwrap()).unwrap_or(0.5);
    let seqtrain = args.iter().position(|a| a == "--seqtrain").map(|i| { args.remove(i); true }).unwrap_or(false);
    let stored = args.iter().position(|a| a == "--stored").map(|i| { args.remove(i); true }).unwrap_or(false);
    let dev = Device::new(0).unwrap();
    let stream = dev.default_stream();
    let ev = (
        dev.ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap(),
        dev.ctx
            .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .unwrap(),
    );
    println!("chirp-aware NUFFT normal operator (A^H A per coil), sigma={sigma} tol={tol} S={shalf} w_box={wbox} stored_stencils={stored} sequence_trained_tables={seqtrain}");
    println!(
        "{:<18} | L  stencil taps | forward ms/coil | adjoint ms/coil | normal-op ms/coil",
        "N:K:C"
    );
    for spec in &args {
        let p: Vec<usize> = spec.split(':').map(|x| x.parse().unwrap()).collect();
        let (n, k, c) = (p[0], p[1], p[2]);
        // radial spokes
        let mut r = Rng(5);
        let (t, s) = (512usize, k.div_ceil(512));
        let (mut x, mut y, mut z) = (vec![0f32; k], vec![0f32; k], vec![0f32; k]);
        let mut w: [Vec<f32>; 4] = [vec![0f32; k], vec![0f32; k], vec![0f32; k], vec![0f32; k]];
        for sp in 0..s {
            let zc = r.next();
            let ph = (r.next() + 1.0) * std::f64::consts::PI;
            let st = (1.0 - zc * zc).sqrt();
            let nrm = [st * ph.cos(), st * ph.sin(), zc];
            for tt in 0..t {
                let j = sp * t + tt;
                if j >= k {
                    break;
                }
                let u = (tt as f64 + 0.5) / t as f64;
                let rr = std::f64::consts::PI * 0.99 * u;
                x[j] = (rr * nrm[0]) as f32;
                y[j] = (rr * nrm[1]) as f32;
                z[j] = (rr * nrm[2]) as f32;
                let g = wbox * u * u * u;
                w[0][j] = (g * (nrm[0] * nrm[0] + nrm[1] * nrm[1])) as f32;
                w[1][j] = (g * nrm[2] * nrm[2] / 4.0) as f32;
                w[2][j] = (-g * nrm[0] * nrm[2]) as f32;
                w[3][j] = (-g * nrm[1] * nrm[2]) as f32;
            }
        }
        let seq_train: Option<Vec<[f64; 4]>> = if seqtrain {
            let mut rr = Rng(11);
            Some((0..600).map(|_| { let j = (((rr.next() * 0.5 + 0.5) * k as f64) as usize).min(k - 1); [w[0][j] as f64, w[1][j] as f64, w[2][j] as f64, w[3][j] as f64] }).collect())
        } else {
            None
        };
        let fam = ChirpFamily {
            w_box: [wbox; 4],
            s_half: shalf,
            pod_tol: 1e-3,
            l_max: 80,
            nfov: 13,
            ridge: 1e-7,
            n_train: 400,
            train: seq_train,
        };
        let t0 = std::time::Instant::now();
        let mut op = ChirpNufft::new(
            &dev,
            [n, n, n],
            NufftOpts {
                tol,
                sigma,
                ntransf: c,
                ..Default::default()
            },
            fam,
        )
        .unwrap();
        let build = t0.elapsed().as_secs_f32();
        let dx = stream.clone_htod(&x).unwrap();
        let dy = stream.clone_htod(&y).unwrap();
        let dz = stream.clone_htod(&z).unwrap();
        let dw: Vec<_> = w.iter().map(|v| stream.clone_htod(v).unwrap()).collect();
        op.set_points(
            &stream,
            &[&dx, &dy, &dz],
            [&dw[0], &dw[1], &dw[2], &dw[3]],
            k,
        )
        .unwrap();
        let t_pre = if stored {
            stream.synchronize().unwrap();
            let t0 = std::time::Instant::now();
            op.precompute_stencils(&stream).unwrap();
            stream.synchronize().unwrap();
            t0.elapsed().as_secs_f32()
        } else {
            0.0
        };
        let nm = n * n * n;
        let f: Vec<C32> = (0..nm * c)
            .map(|i| C32::new((i % 17) as f32 * 0.01, (i % 5) as f32 * 0.02))
            .collect();
        let df = stream.clone_htod(&f).unwrap();
        let mut dc = stream.alloc_zeros::<C32>(k * c).unwrap();
        let mut dfo = stream.alloc_zeros::<C32>(nm * c).unwrap();
        let (mut tf, mut ta) = (Vec::new(), Vec::new());
        for i in 0..6 {
            ev.0.record(&stream).unwrap();
            op.forward(&stream, &df, &mut dc).unwrap();
            ev.1.record(&stream).unwrap();
            ev.1.synchronize().unwrap();
            let a = ev.0.elapsed_ms(&ev.1).unwrap();
            ev.0.record(&stream).unwrap();
            op.adjoint(&stream, &dc, &mut dfo).unwrap();
            ev.1.record(&stream).unwrap();
            ev.1.synchronize().unwrap();
            let b = ev.0.elapsed_ms(&ev.1).unwrap();
            if i >= 2 {
                tf.push(a / c as f32);
                ta.push(b / c as f32);
            }
        }
        tf.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ta.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (a, b) = (tf[tf.len() / 2], ta[ta.len() / 2]);
        let jw = op.plan.ker.w + 2 * shalf;
        println!(
            "{:<18} | {:>2} {:>7} | {:>15.3} | {:>15.3} | {:>17.3}   (setup {:.1}s{}, row-repr err max {:.1e})",
            spec,
            op.tables.l,
            jw * jw * jw,
            a,
            b,
            a + b,
            build,
            if stored { format!(", stencil precompute {:.2}s, {:.1} GB", t_pre, (k * jw * jw * jw * 4) as f64 * 1e-9) } else { String::new() },
            op.tables.heldout_err.0
        );
    }
}
