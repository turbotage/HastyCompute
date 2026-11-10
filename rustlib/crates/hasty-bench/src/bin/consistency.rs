//! Simulator vs reconstruction operator on the same samples: point sources at voxel centres (ideal physics), direct sum
//! vs SenseOp::forward with the matching image.
//!   consistency <sim_dir>
use std::sync::Arc;

use hasty_cuda::{Device, C32, C64};
use hasty_mrsim::direct::{pack_sources, DirectSim, Source};
use hasty_mrsim::sequence::SAMPLE_FLOATS;
use hasty_mrsim::mat_vec;
use hasty_recon::data::SimData;
use hasty_recon::gpu::Kernels;
use hasty_recon::op::SenseOp;
use num_complex::Complex64;

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let sd = SimData::load(&dir).unwrap();
    let nc = 2usize;
    let nv = sd.nv();
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let ros: Vec<usize> = sd.readouts_of(0, None).into_iter().take(40).collect();
    let ident: Vec<Vec<f64>> = (0..sd.nc).map(|i| (0..sd.nc).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    let set = sd.sample_set(&ros, &ident);
    let m = set.tau.len();
    // sample params with physics zeroed except encoding
    let mut smp = Vec::with_capacity(m * SAMPLE_FLOATS);
    for &r in &ros {
        let mut s = sd.spiral().sample_params(r..r + 1);
        for n in 0..sd.na {
            for q in 9..16 {
                s[n * SAMPLE_FLOATS + q] = 0.0;
            }
        }
        smp.extend(s);
    }
    // point sources at 30 voxel centres
    let mut st_rng = 7u64;
    let mut pick = || { st_rng = st_rng.wrapping_mul(6364136223846793005).wrapping_add(1); ((st_rng >> 33) as usize) % nv };
    let vox: Vec<usize> = (0..30).map(|_| pick()).collect();
    let mut srcs = Vec::new();
    let mut amp = Vec::new();
    let mut img = vec![C32::new(0.0, 0.0); nv];
    for (i, &q) in vox.iter().enumerate() {
        let lg = sd.voxel_log(q);
        let p = mat_vec(&sd.rot, &lg); // physical, relative to the FOV centre
        srcs.push(Source { u: p, x: [p[0] + sd.fov_c[0], p[1] + sd.fov_c[1], p[2] + sd.fov_c[2]], ..Default::default() });
        let a = Complex64::new(1.0 + i as f64 * 0.1, 0.3);
        amp.push(a);
        amp.push(a * Complex64::new(0.0, 1.0));
        img[q] = C32::new(a.re as f32, a.im as f32);
    }
    let sim = DirectSim::new(&dev, nc).unwrap();
    let d_s = st.clone_htod(&smp).unwrap();
    let d_p = st.clone_htod(&pack_sources(&srcs)).unwrap();
    let d_a = st.clone_htod(&amp.iter().map(|c| C32::new(c.re as f32, c.im as f32)).collect::<Vec<_>>()).unwrap();
    let mut d_o = st.alloc_zeros::<C64>(m * nc).unwrap();
    sim.accumulate(&st, &d_s, m, &d_p, &d_a, srcs.len(), &mut d_o).unwrap();
    let want = st.clone_dtoh(&d_o).unwrap();
    // operator: coil 0 map = 1, coil 1 map = i
    let mut smaps = vec![C32::new(1.0, 0.0); nc * nv];
    for q in 0..nv {
        smaps[nv + q] = C32::new(0.0, 1.0);
    }
    let kern = Arc::new(Kernels::new(&dev).unwrap());
    let mut op = SenseOp::new(&dev, &st, kern, sd.dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &smaps, nc, vec![], 1e-6).unwrap();
    let d_x = st.clone_htod(&img).unwrap();
    let mut d_y = st.alloc_zeros::<C32>(nc * m).unwrap();
    op.forward(&st, &d_x, &mut d_y).unwrap();
    let got = st.clone_dtoh(&d_y).unwrap();
    let (mut num, mut den, mut num_conj) = (0.0, 0.0, 0.0);
    for j in 0..m {
        for c in 0..nc {
            let w = Complex64::new(want[j * nc + c].re, want[j * nc + c].im);
            let g = Complex64::new(got[c * m + j].re as f64, got[c * m + j].im as f64);
            num += (g - w).norm_sqr();
            num_conj += (g.conj() - w).norm_sqr();
            den += w.norm_sqr();
        }
    }
    println!("simulator vs operator: rel err {:.3e} (vs conj: {:.3e}) over {m} samples", (num / den).sqrt(), (num_conj / den).sqrt());
    for j in [0usize, 500, 1500, m / 2] {
        println!("  sample {j}: sim {:?}  op {:?}", want[j * nc], got[j]);
    }
}
