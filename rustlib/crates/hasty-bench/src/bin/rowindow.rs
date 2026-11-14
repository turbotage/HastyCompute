//! Time-averaged plain-model images per encode from readout-time windows of a simulated scan (diagnostic for the
//! velocity shrink). Every encode pooled over all frames (as the warm start of recon_p2np), plain SENSE NUFFT, PCGLS
//! with the circulant preconditioner, only the samples whose time after the echo lies in a window [a, b) ms.
//! The yarn readout goes out and back (|k| reaches Nyquist half-way, returns to k = 0 at the end), so [0, T/2) and
//! [T/2, T) both cover k-space to Nyquist: the first at short effective TE, the second late (B0 / fat phase, flow
//! displacement during the readout). Compare their velocity phases (warm_fit.py) to locate the shrink.
//!
//!   rowindow <scan dir> --out dir [--windows 0-1.5,0-4,4-8,0-8] [--iters 30] [--coils 12] [--stride 2] [--gnl]
//!            [--enc 0,1,...]
//!
//! Writes win_<a>-<b>.npy (complex64, Fortran (nx, ny, nz, ne), the warm.npy layout). `--coils n`: the n coils with the
//! most sensitivity energy inside the lumen (truth_lumen_fraction > 0.5); `--stride s`: every s-th readout.

use std::path::PathBuf;
use std::time::Instant;

use hasty_cuda::{Device, C32};
use hasty_recon::cg::pcgls_cb;
use hasty_recon::data::{read_npy, SimData};
use hasty_recon::gpu::Kernels;
use hasty_recon::op::SenseOp;
use hasty_recon::precond::CirculantPrecond;

fn write_c8(path: &std::path::Path, shape: &[usize], data: &[C32]) {
    let sh: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
    let hdr = format!("{{'descr': '<c8', 'fortran_order': True, 'shape': ({}), }}", sh.join(", "));
    let mut h2 = hdr.into_bytes();
    while (h2.len() + 11) % 64 != 0 {
        h2.push(b' ');
    }
    h2.push(b'\n');
    let mut f = vec![0x93u8, b'N', b'U', b'M', b'P', b'Y', 1, 0];
    f.extend_from_slice(&(h2.len() as u16).to_le_bytes());
    f.extend_from_slice(&h2);
    for z in data {
        f.extend_from_slice(&z.re.to_le_bytes());
        f.extend_from_slice(&z.im.to_le_bytes());
    }
    std::fs::write(path, f).unwrap();
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let iters: usize = take("--iters").map(|s| s.parse().unwrap()).unwrap_or(30);
    let stride: usize = take("--stride").map(|s| s.parse().unwrap()).unwrap_or(2);
    let ncoil_sel: usize = take("--coils").map(|s| s.parse().unwrap()).unwrap_or(12);
    let windows: Vec<(f64, f64)> = take("--windows")
        .unwrap_or("0-1.5,0-4,4-8,0-8".into())
        .split(',')
        .map(|w| {
            let (a, b) = w.split_once('-').expect("window a-b (ms)");
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .collect();
    let out = PathBuf::from(take("--out").expect("--out dir"));
    let enc_sel: Option<Vec<usize>> = take("--enc").map(|s| s.split(',').map(|v| v.parse().unwrap()).collect());
    let gnl = args.iter().position(|a| a == "--gnl").map(|i| { args.remove(i); true }).unwrap_or(false);
    let dir = PathBuf::from(&args[0]);
    std::fs::create_dir_all(&out).unwrap();
    let t0 = Instant::now();
    let sd = SimData::load_lazy(&dir).unwrap();
    let (dims, nv, nc_all) = (sd.dims, sd.nv(), sd.nc);
    let ne = sd.encode_m1.len();
    let encs = enc_sel.unwrap_or((0..ne).collect());
    let w = sd.whitener();
    let grid = sd.physics_grid(gnl, &w);
    // coils: most sensitivity energy over the lumen
    let lum = read_npy(&dir.join("truth_lumen_fraction.npy")).unwrap().f32s();
    let lq: Vec<usize> = (0..nv).filter(|&q| lum[q] > 0.5).collect();
    let en: Vec<f64> = (0..nc_all).map(|c| lq.iter().map(|&q| { let z = grid.smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum()).collect();
    let mut order: Vec<usize> = (0..nc_all).collect();
    order.sort_by(|&a, &b| en[b].partial_cmp(&en[a]).unwrap());
    let nc = ncoil_sel.min(nc_all);
    let keep = order[..nc].to_vec();
    let smaps: Vec<C32> = keep.iter().flat_map(|&c| grid.smaps[c * nv..(c + 1) * nv].iter().cloned()).collect();
    println!(
        "{}: {ne} encodes, grid {dims:?}, coils {nc} of {nc_all} ({:.1} % of the lumen sensitivity energy), stride {stride}, windows {windows:?} ms after the echo, {iters} PCGLS iterations ({:.1}s)",
        dir.display(),
        100.0 * keep.iter().map(|&c| en[c]).sum::<f64>() / en.iter().sum::<f64>(),
        t0.elapsed().as_secs_f64()
    );
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = std::sync::Arc::new(Kernels::new(&dev).unwrap());
    let mut imgs: Vec<Vec<C32>> = vec![vec![C32::new(0.0, 0.0); nv * encs.len()]; windows.len()];
    for (ei, &e) in encs.iter().enumerate() {
        let te = Instant::now();
        let ros: Vec<usize> = sd.readouts_of(e, None).into_iter().step_by(stride).collect();
        let set = sd.load_set(&ros, &hasty_recon::kload::LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
        let m = set.tau.len();
        let tau0 = set.tau.iter().cloned().fold(f64::INFINITY, f64::min);
        for (wi, &(a, b)) in windows.iter().enumerate() {
            let idx: Vec<usize> = (0..m).filter(|&j| { let t = (set.tau[j] - tau0) * 1e3; t >= a && t < b }).collect();
            let ms = idx.len();
            let coords: [Vec<f32>; 3] = [0, 1, 2].map(|ax| idx.iter().map(|&j| set.coords[ax][j]).collect());
            let mut yh = vec![C32::new(0.0, 0.0); nc * ms];
            for (ci, &c) in keep.iter().enumerate() {
                for (jj, &j) in idx.iter().enumerate() {
                    yh[ci * ms + jj] = set.y[c * m + j];
                }
            }
            let mut op = SenseOp::new(&dev, &st, kern.clone(), dims, [&coords[0], &coords[1], &coords[2]], &smaps, nc, vec![], 1e-4).unwrap();
            let mut pre = CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&coords[0], &coords[1], &coords[2]], &smaps, nc, 1e-3).unwrap();
            let y = st.clone_htod(&yh).unwrap();
            drop(yh);
            let mut x = st.alloc_zeros::<C32>(nv).unwrap();
            pcgls_cb(&mut op, Some(&mut pre), &st, &y, &mut x, 0.0, iters, &mut |_, _| {}).unwrap();
            let h = st.clone_dtoh(&x).unwrap();
            imgs[wi][ei * nv..(ei + 1) * nv].copy_from_slice(&h);
            println!("  encode {e}, window {a}-{b} ms: {ms} of {m} samples ({:.1}s)", te.elapsed().as_secs_f64());
        }
    }
    for (wi, &(a, b)) in windows.iter().enumerate() {
        write_c8(&out.join(format!("win_{a}-{b}.npy")), &[dims[0], dims[1], dims[2], encs.len()], &imgs[wi]);
    }
    println!("done ({:.1}s)", t0.elapsed().as_secs_f64());
}
