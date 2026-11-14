//! LLR at a given size on the GPU (and optionally the CPU): synthetic stack (smooth rank-4 signal + noise) on an
//! ellipsoidal support, timing of one call per pass.
//!   llrbench [--dims 254,144,114] [--t 280] [--kappa 1] [--devices 0[,1]] [--cpu]

use hasty_cuda::C32;
use hasty_recon::llr::Llr;
use hasty_recon::llr_gpu::LlrGpu;
use rayon::prelude::*;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |n: &str| a.iter().position(|s| s == n).and_then(|i| a.get(i + 1)).cloned();
    let dims: Vec<usize> = get("--dims").unwrap_or("254,144,114".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let dims = [dims[0], dims[1], dims[2]];
    let nt: usize = get("--t").map_or(280, |v| v.parse().unwrap());
    let kappa: f64 = get("--kappa").map_or(1.0, |v| v.parse().unwrap());
    let devs: Vec<usize> = get("--devices").unwrap_or("0".into()).split(',').map(|v| v.parse().unwrap()).collect();
    let nv: usize = dims.iter().product();
    let support: Vec<bool> = (0..nv).map(|q| {
        let (i, j, l) = (q % dims[0], (q / dims[0]) % dims[1], q / (dims[0] * dims[1]));
        let u = [(i as f64 / dims[0] as f64 - 0.5) / 0.48, (j as f64 / dims[1] as f64 - 0.5) / 0.48, (l as f64 / dims[2] as f64 - 0.5) / 0.48];
        u.iter().map(|v| v * v).sum::<f64>() < 1.0
    }).collect();
    let t0 = std::time::Instant::now();
    let mut x = vec![C32::new(0.0, 0.0); nv * nt];
    x.par_chunks_mut(nv).enumerate().for_each(|(t, xt)| {
        let mut h = (t as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut u = || { h ^= h << 13; h ^= h >> 7; h ^= h << 17; (h >> 11) as f32 / (1u64 << 53) as f32 - 0.5 };
        let c: Vec<f32> = (0..8).map(|k| ((t * (k + 3)) as f32 * 0.37).sin()).collect();
        for (q, z) in xt.iter_mut().enumerate() {
            let (i, j, l) = ((q % dims[0]) as f32, ((q / dims[0]) % dims[1]) as f32, (q / (dims[0] * dims[1])) as f32);
            let s = 10.0 * (1.0 + 0.3 * c[0]) + 4.0 * c[1] * (i * 0.05).sin() + 2.0 * c[2] * (j * 0.07).cos() + c[3] * (l * 0.09).sin();
            *z = C32::new(s + u(), 0.4 * s * c[4] + u());
        }
    });
    println!("stack {dims:?} × {nt} ({:.1} GiB) built in {:.1}s", (nv * nt * 8) as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64());
    let mut g = LlrGpu::new(Llr::new(dims, 8, kappa, support.clone()), &devs).unwrap();
    for it in 0..2 {
        g.set_iteration(it);
        let t = std::time::Instant::now();
        g.apply(&mut x, nt).unwrap();
        println!("GPU {devs:?} call {it}: {:.1}s ({:?})", t.elapsed().as_secs_f64(), g.last);
    }
    if a.iter().any(|s| s == "--cpu") {
        let mut c = Llr::new(dims, 8, kappa, support);
        let t = std::time::Instant::now();
        c.apply(&mut x, nt);
        println!("CPU: {:.1}s", t.elapsed().as_secs_f64());
    }
}
