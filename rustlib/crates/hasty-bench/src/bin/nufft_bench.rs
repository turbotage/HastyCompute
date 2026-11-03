//! hasty-nufft 3-D benchmark; same point generators as bench/cpp/nufft_baseline.cu.
//!   nufft_bench [--tol t] [--ntransf n] [--method auto|gm|sm] kind:M:N ...      kind = uniform | radial

use cudarc::driver::sys::CUevent_flags;
use hasty_cuda::{Device, C32};
use hasty_nufft::{NufftOpts, NufftPlan, SpreadMethod};

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

fn gen(kind: &str, m: usize) -> [Vec<f32>; 3] {
    let mut r = Rng(12345);
    let pi = std::f64::consts::PI;
    let (mut x, mut y, mut z) = (vec![0f32; m], vec![0f32; m], vec![0f32; m]);
    if kind == "uniform" {
        for j in 0..m {
            x[j] = (r.next() * pi * 0.999) as f32;
            y[j] = (r.next() * pi * 0.999) as f32;
            z[j] = (r.next() * pi * 0.999) as f32;
        }
    } else {
        let t = 512usize;
        let s = m.div_ceil(t);
        for sp in 0..s {
            let zc = r.next();
            let ph = (r.next() + 1.0) * pi;
            let st = (1.0 - zc * zc).sqrt();
            let (nx, ny, nz) = (st * ph.cos(), st * ph.sin(), zc);
            for tt in 0..t {
                let j = sp * t + tt;
                if j >= m {
                    break;
                }
                let rr = pi * 0.999 * (tt as f64 + 0.5) / t as f64;
                x[j] = (rr * nx) as f32;
                y[j] = (rr * ny) as f32;
                z[j] = (rr * nz) as f32;
            }
        }
    }
    [x, y, z]
}

fn median(v: &mut Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).map(|i| {
            args.remove(i);
            args.remove(i)
        })
    };
    let tol: f64 = take("--tol").map(|s| s.parse().unwrap()).unwrap_or(1e-6);
    let ntransf: usize = take("--ntransf").map(|s| s.parse().unwrap()).unwrap_or(1);
    let method = match take("--method").as_deref() {
        Some("gm") => SpreadMethod::Gm,
        Some("sm") => SpreadMethod::Sm,
        _ => SpreadMethod::Auto,
    };
    let sigma: f64 = take("--sigma").map(|s| s.parse().unwrap()).unwrap_or(2.0);
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
    println!("hasty-nufft f32 tol={tol} sigma={sigma} ntransf={ntransf} spread={method:?}");
    println!(
        "{:<22} | setpts ms | type1 ms/transf | type2 ms/transf | width nf",
        "kind:M:N"
    );
    for spec in &args {
        let p: Vec<&str> = spec.split(':').collect();
        let (kind, m, n): (&str, usize, usize) =
            (p[0], p[1].parse().unwrap(), p[2].parse().unwrap());
        let pts = gen(kind, m);
        let dpts: Vec<_> = pts.iter().map(|v| stream.clone_htod(v).unwrap()).collect();
        let refs: Vec<&_> = dpts.iter().collect();
        let nm = n * n * n;
        let mut host = Rng(99);
        let c: Vec<C32> = (0..m * ntransf)
            .map(|_| C32::new(host.next() as f32 * 0.5, host.next() as f32 * 0.5))
            .collect();
        let f: Vec<C32> = (0..nm * ntransf)
            .map(|_| C32::new(host.next() as f32 * 0.5, host.next() as f32 * 0.5))
            .collect();
        let dc = stream.clone_htod(&c).unwrap();
        let df = stream.clone_htod(&f).unwrap();
        let mut dcout = stream.alloc_zeros::<C32>(m * ntransf).unwrap();
        let mut dfout = stream.alloc_zeros::<C32>(nm * ntransf).unwrap();
        let mut plan = NufftPlan::<f32>::new(
            &dev,
            &[n, n, n],
            NufftOpts {
                tol,
                sigma,
                spread: method,
                ntransf,
                ..Default::default()
            },
        )
        .unwrap();
        stream.synchronize().unwrap();
        ev.0.record(&stream).unwrap();
        plan.set_points(&stream, &refs, m).unwrap();
        ev.1.record(&stream).unwrap();
        ev.1.synchronize().unwrap();
        let t_set = ev.0.elapsed_ms(&ev.1).unwrap();
        let mut t1 = Vec::new();
        let mut t2 = Vec::new();
        for i in 0..8 {
            ev.0.record(&stream).unwrap();
            plan.type1(&stream, &dc, &mut dfout, 1).unwrap();
            ev.1.record(&stream).unwrap();
            ev.1.synchronize().unwrap();
            let a = ev.0.elapsed_ms(&ev.1).unwrap();
            ev.0.record(&stream).unwrap();
            plan.type2(&stream, &df, &mut dcout, -1).unwrap();
            ev.1.record(&stream).unwrap();
            ev.1.synchronize().unwrap();
            let b = ev.0.elapsed_ms(&ev.1).unwrap();
            if i >= 3 {
                t1.push(a / ntransf as f32);
                t2.push(b / ntransf as f32);
            }
        }
        println!(
            "{:<22} | {:>9.2} | {:>15.3} | {:>15.3} | w={} nf={}",
            spec,
            t_set,
            median(&mut t1),
            median(&mut t2),
            plan.ker.w,
            plan.nf[0]
        );
    }
}
