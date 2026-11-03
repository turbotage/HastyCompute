//! Per-coil adjoint (gridding without density compensation) of the reference encode: RSS image and the coil-map
//! combined image, to check coil-map consistency between simulator and reconstruction.
//!   coilcheck <sim_dir>
use hasty_cuda::{Device, C32};
use hasty_nufft::{NufftOpts, NufftPlan};
use hasty_recon::data::SimData;
use num_complex::Complex64;

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    let sd = SimData::load(&dir).unwrap();
    let w = sd.whitener();
    let set = sd.sample_set(&sd.readouts_of(0, None), &w);
    let m = set.tau.len();
    let nc = sd.nc;
    let nv = sd.nv();
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let mut p = NufftPlan::<f32>::new(&dev, &sd.dims, NufftOpts { tol: 1e-4, ntransf: nc, ..Default::default() }).unwrap();
    let dc: Vec<_> = set.coords.iter().map(|c| st.clone_htod(c).unwrap()).collect();
    p.set_points(&st, &[&dc[0], &dc[1], &dc[2]], m).unwrap();
    let y = st.clone_htod(&set.y).unwrap();
    let mut img = st.alloc_zeros::<C32>(nc * nv).unwrap();
    p.type1(&st, &y, &mut img, 1).unwrap();
    let h = st.clone_dtoh(&img).unwrap();
    let grid = sd.physics_grid(false, &w);
    let mut rss = vec![0f32; nv];
    let mut comb = vec![0f32; nv];
    let mut comb_conj = vec![0f32; nv];
    for q in 0..nv {
        let (mut r, mut a, mut b, mut s2) = (0.0f64, Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), 0.0f64);
        for c in 0..nc {
            let z = Complex64::new(h[c * nv + q].re as f64, h[c * nv + q].im as f64);
            let s = Complex64::new(grid.smaps[c * nv + q].re as f64, grid.smaps[c * nv + q].im as f64);
            r += z.norm_sqr();
            a += s.conj() * z;
            b += s * z;
            s2 += s.norm_sqr();
        }
        rss[q] = r.sqrt() as f32;
        comb[q] = (a.norm() / s2.sqrt().max(1e-3)) as f32;
        comb_conj[q] = (b.norm() / s2.sqrt().max(1e-3)) as f32;
    }
    for (name, v) in [("adj_rss", &rss), ("adj_comb", &comb), ("adj_comb_conjmaps", &comb_conj)] {
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        hasty_phantom::volume::write_npy(&dir.join(format!("{name}.npy")), "<f4", &sd.dims, true, &bytes).unwrap();
    }
    println!("wrote adj_rss / adj_comb / adj_comb_conjmaps");
}
