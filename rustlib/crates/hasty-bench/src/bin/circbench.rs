//! Cost of the circulant metric (CirculantPrecond) on a real grid: build time, z = M r and z = M⁻¹ r with the arrays on
//! the device and host-resident (recon_p2np's --circ-share encode path: two N-grid arrays uploaded per application).
//!
//!   circbench <scan dir> [--enc 0] [--reps 20] [--gnl]
//!
//! The circulant is built from the pooled samples of one encode over all frames (as recon_p2np does per encode).

use std::path::PathBuf;
use std::time::Instant;

use hasty_cuda::{Device, C32};
use hasty_recon::data::SimData;
use hasty_recon::gpu::Kernels;
use hasty_recon::precond::{CircShared, CirculantPrecond};

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let enc: usize = take("--enc").map(|s| s.parse().unwrap()).unwrap_or(0);
    let reps: usize = take("--reps").map(|s| s.parse().unwrap()).unwrap_or(20);
    let gnl = args.iter().position(|a| a == "--gnl").map(|i| { args.remove(i); true }).unwrap_or(false);
    let dir = PathBuf::from(&args[0]);
    let sd = SimData::load_lazy(&dir).unwrap();
    let (dims, nv, nc) = (sd.dims, sd.nv(), sd.nc);
    let w = sd.whitener();
    let ros = sd.readouts_of(enc, None);
    let set = sd.load_set(&ros, &hasty_recon::kload::LoadOpts { w: &w, cc: None, keep: None, data: false }).unwrap();
    let grid = sd.physics_grid(gnl, &w);
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = std::sync::Arc::new(Kernels::new(&dev).unwrap());
    let t0 = Instant::now();
    let mut pc = CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&set.coords[0], &set.coords[1], &set.coords[2]], &grid.smaps, nc, 1e-3).unwrap();
    st.synchronize().unwrap();
    println!("grid {dims:?} ({nv} voxels), encode {enc}: {} readouts, {} samples, {nc} coils; build {:.2}s", ros.len(), set.coords[0].len(), t0.elapsed().as_secs_f64());
    let r: Vec<C32> = (0..nv).map(|q| C32::new(((q * 7919) % 1000) as f32 * 1e-3, ((q * 104729) % 1000) as f32 * 1e-3)).collect();
    let dr = st.clone_htod(&r).unwrap();
    let mut z = st.alloc_zeros::<C32>(nv).unwrap();
    let mut time = |name: &str, pc: &mut CirculantPrecond, inv: bool| {
        for _ in 0..3 {
            if inv { pc.apply_inv(&st, &dr, &mut z).unwrap() } else { pc.apply(&st, &dr, &mut z).unwrap() }
        }
        st.synchronize().unwrap();
        let t = Instant::now();
        for _ in 0..reps {
            if inv { pc.apply_inv(&st, &dr, &mut z).unwrap() } else { pc.apply(&st, &dr, &mut z).unwrap() }
        }
        st.synchronize().unwrap();
        println!("  {name}: {:.2} ms per application", t.elapsed().as_secs_f64() * 1e3 / reps as f64);
    };
    time("device M r", &mut pc, false);
    time("device M⁻¹ r", &mut pc, true);
    let ws = CircShared::new(&dev, &st, dims).unwrap();
    let mut ph = pc.to_host(&st, &ws).unwrap();
    time("host-resident M r", &mut ph, false);
    time("host-resident M⁻¹ r", &mut ph, true);
}
