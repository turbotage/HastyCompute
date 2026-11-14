//! Per-frame plain reconstruction of a simulated scan, as a PC-VIPR-style practitioner would do it (reality check for
//! the per-frame P²nP results): every cardiac frame × encode reconstructed on its own from that frame's readouts
//! (respiratory states pooled), plain SENSE NUFFT, a few PCGLS iterations with the circulant preconditioner (no
//! regulariser; early stopping), optionally only the samples in a readout-time window and with every rotation's
//! encodes binned into one frame (`--bin-group`, the frame of the readout nearest the group's velocity time, as recon_p2np; `--bin-group-mid`: the
//! middle readout's), and optionally gated to
//! one respiratory bin (`--resp-bin b/NR`: bellows amplitude, equal-count bins as recon_p2np --resp-bins NR; a
//! practitioner gates thoracic 4D flow, the anatomy moves ~10 mm with breathing).
//!
//!   framerecon <scan dir> --idx lumen_idx.npy --out vals.npy [--iters 10] [--coils 12] [--window a-b] [--bin-group]
//!       [--resp-bin b/NR] [--share K]
//!
//! `--save-at n1,n2,..`: also write the lumen values at these PCGLS iterations (`<out>_it{n}.npy`, n ≤ --iters) and
//! print the residual history of frame 0 (convergence of the per-frame normal equations).
//! `--synthetic warm.npy,synth.npy`: inverse crime. Per (frame, encode) the data are synthesised with the
//! reconstruction's own plain operator from x★ = the warm image of the encode (warm_full.npy, Fortran (X, Y, Z, E)) with
//! the lumen voxels of --idx replaced by synth.npy (n, E, F) (research/experiments/recon_diag/synth_lumen.py: the true
//! per-frame phase differences); no noise. Same sampling, same reconstruction: the per-frame problem with consistent data.
//! `--share K`: view sharing as PC-VIPR-style reconstructions do, frame f uses the readouts of the cardiac frames
//! f-K..f+K (cyclic): more data per frame, temporal blur.
//!
//! `--idx`: int64 flat (Fortran) voxel indices (frame_fit.py's truth voxels, from truth_pc_frames.npz). Writes only
//! those voxels: complex64 (n, ne, nf) C-order, the frame_fit.py `*_lumen.npy` form (no full 4.7 GB stack on disk).

use std::path::PathBuf;
use std::time::Instant;

use hasty_cuda::{Device, C32};
use hasty_recon::cg::pcgls_cb;
use hasty_recon::data::{read_npy, SimData};
use hasty_recon::gpu::Kernels;
use hasty_recon::op::SenseOp;
use hasty_recon::precond::CirculantPrecond;

fn write_c8_c(path: &std::path::Path, shape: &[usize], data: &[C32]) {
    let sh: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
    let hdr = format!("{{'descr': '<c8', 'fortran_order': False, 'shape': ({}), }}", sh.join(", "));
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
    let bin_group = args.iter().position(|a| a == "--bin-group").map(|i| { args.remove(i); true }).unwrap_or(false);
    let bin_group_mid = args.iter().position(|a| a == "--bin-group-mid").map(|i| { args.remove(i); true }).unwrap_or(false);
    let mut take = |name: &str| -> Option<String> { args.iter().position(|a| a == name).map(|i| { args.remove(i); args.remove(i) }) };
    let iters: usize = take("--iters").map(|s| s.parse().unwrap()).unwrap_or(10);
    let ncoil_sel: usize = take("--coils").map(|s| s.parse().unwrap()).unwrap_or(12);
    let window: Option<(f64, f64)> = take("--window").map(|w| {
        let (a, b) = w.split_once('-').expect("--window a-b (ms after the echo)");
        (a.parse().unwrap(), b.parse().unwrap())
    });
    let resp_bin: Option<(usize, usize)> = take("--resp-bin").map(|v| {
        let (b, n) = v.split_once('/').expect("--resp-bin b/NR");
        (b.parse().unwrap(), n.parse().unwrap())
    });
    let share: usize = take("--share").map(|s| s.parse().unwrap()).unwrap_or(0);
    let save_at: Vec<usize> = take("--save-at").map(|s| s.split(',').map(|v| v.parse().unwrap()).collect()).unwrap_or_default();
    let synthetic: Option<(PathBuf, PathBuf)> = take("--synthetic").map(|v| {
        let (a, b) = v.split_once(',').expect("--synthetic warm.npy,synth.npy");
        (PathBuf::from(a), PathBuf::from(b))
    });
    let out = PathBuf::from(take("--out").expect("--out vals.npy"));
    let idx_path = PathBuf::from(take("--idx").expect("--idx lumen_idx.npy"));
    let dir = PathBuf::from(&args[0]);
    let t0 = Instant::now();
    let mut sd = SimData::load_lazy(&dir).unwrap();
    let (dims, nv, nc_all, nf) = (sd.dims, sd.nv(), sd.nc, sd.n_frames);
    let ne = sd.encode_m1.len();
    if let Some((b, nr)) = resp_bin {
        // physio.npy columns: t, ECG phase, PPG phase, bellows, true r (simphantom); readouts of other bins dropped
        let ph = read_npy(&dir.join("physio.npy")).unwrap();
        let (n_ro, w) = (sd.n_readouts(), ph.shape[1]);
        assert_eq!(ph.shape[0], n_ro);
        let v = ph.f64s();
        let amp: Vec<f64> = (0..n_ro).map(|i| v[i * w + 3]).collect();
        let bins = hasty_recon::selfgate::amplitude_bins(&amp, nr);
        let (mut s, mut n) = (0.0, 0usize);
        for i in 0..n_ro {
            if bins[i] == b {
                s += amp[i];
                n += 1;
            } else {
                sd.ro.frame[i] = usize::MAX;
            }
        }
        println!("--resp-bin {b}/{nr}: {n} of {n_ro} readouts, mean bellows amplitude {:.3}", s / n.max(1) as f64);
    }
    if bin_group {
        let mut moved = 0usize;
        for g0 in (0..sd.n_readouts()).step_by(ne) {
            let g1 = (g0 + ne).min(sd.n_readouts());
            assert!((g0..g1).all(|i| sd.ro.enc[i] == i - g0), "--bin-group: readouts not grouped as encode 0..{ne}");
            let fm = sd.ro.frame[bin_group_readout(&sd, g0, g1, bin_group_mid)];
            for i in g0..g1 {
                moved += (sd.ro.frame[i] != fm) as usize;
                sd.ro.frame[i] = fm;
            }
        }
        println!("--bin-group: {moved} of {} readouts moved", sd.n_readouts());
    }
    let idx: Vec<usize> = read_npy(&idx_path).unwrap().f32s().iter().map(|&v| v as usize).collect();
    let synth: Option<(Vec<C32>, Vec<C32>)> = synthetic.as_ref().map(|(wp, sp)| {
        let w = read_npy(wp).unwrap();
        assert!(w.descr == "<c8" && w.fortran && w.shape == vec![dims[0], dims[1], dims[2], ne], "--synthetic: warm (X, Y, Z, E) Fortran c8");
        let s = read_npy(sp).unwrap();
        assert!(s.descr == "<c8" && !s.fortran && s.shape == vec![idx.len(), ne, nf], "--synthetic: synth (n, E, F) C-order c8");
        let c8 = |b: &[u8]| -> Vec<C32> { b.chunks_exact(8).map(|c| C32::new(f32::from_le_bytes(c[0..4].try_into().unwrap()), f32::from_le_bytes(c[4..8].try_into().unwrap()))).collect() };
        (c8(&w.bytes), c8(&s.bytes))
    });
    let w = sd.whitener();
    let grid = sd.physics_grid(false, &w);
    let lum = read_npy(&dir.join("truth_lumen_fraction.npy")).unwrap().f32s();
    let lq: Vec<usize> = (0..nv).filter(|&q| lum[q] > 0.5).collect();
    let en: Vec<f64> = (0..nc_all).map(|c| lq.iter().map(|&q| { let z = grid.smaps[c * nv + q]; (z.re * z.re + z.im * z.im) as f64 }).sum()).collect();
    let mut order: Vec<usize> = (0..nc_all).collect();
    order.sort_by(|&a, &b| en[b].partial_cmp(&en[a]).unwrap());
    let nc = ncoil_sel.min(nc_all);
    let keep = order[..nc].to_vec();
    let smaps: Vec<C32> = keep.iter().flat_map(|&c| grid.smaps[c * nv..(c + 1) * nv].iter().cloned()).collect();
    println!(
        "{}: {nf} cardiac frames × {ne} encodes, grid {dims:?}, coils {nc} of {nc_all} ({:.1} % of the lumen sensitivity energy), window {window:?} ms, {iters} PCGLS it + circulant, {} output voxels ({:.1}s)",
        dir.display(),
        100.0 * keep.iter().map(|&c| en[c]).sum::<f64>() / en.iter().sum::<f64>(),
        idx.len(),
        t0.elapsed().as_secs_f64()
    );
    let dev = Device::new(0).unwrap();
    let st = dev.default_stream();
    let kern = std::sync::Arc::new(Kernels::new(&dev).unwrap());
    let n = idx.len();
    let mut vals = vec![C32::new(0.0, 0.0); n * ne * nf];
    let mut vals_at: Vec<Vec<C32>> = save_at.iter().map(|_| vec![C32::new(0.0, 0.0); n * ne * nf]).collect();
    for f in 0..nf {
        let tf = Instant::now();
        let mut counts = vec![];
        for e in 0..ne {
            let ros: Vec<usize> = (0..=2 * share).flat_map(|d| sd.readouts_of(e, Some((f + nf + d - share) % nf))).collect();
            let set = sd.load_set(&ros, &hasty_recon::kload::LoadOpts { w: &w, cc: None, keep: None, data: true }).unwrap();
            let m = set.tau.len();
            let sel: Vec<usize> = match window {
                Some((a, b)) => {
                    let tau0 = set.tau.iter().cloned().fold(f64::INFINITY, f64::min);
                    (0..m).filter(|&j| { let t = (set.tau[j] - tau0) * 1e3; t >= a && t < b }).collect()
                }
                None => (0..m).collect(),
            };
            let ms = sel.len();
            let coords: [Vec<f32>; 3] = [0, 1, 2].map(|ax| sel.iter().map(|&j| set.coords[ax][j]).collect());
            let mut yh = vec![C32::new(0.0, 0.0); nc * ms];
            for (ci, &c) in keep.iter().enumerate() {
                for (jj, &j) in sel.iter().enumerate() {
                    yh[ci * ms + jj] = set.y[c * m + j];
                }
            }
            drop(set);
            let mut op = SenseOp::new(&dev, &st, kern.clone(), dims, [&coords[0], &coords[1], &coords[2]], &smaps, nc, vec![], 1e-4).unwrap();
            let mut pre = CirculantPrecond::new(&dev, &st, kern.clone(), dims, [&coords[0], &coords[1], &coords[2]], &smaps, nc, 1e-3).unwrap();
            let y = match &synth {
                None => st.clone_htod(&yh).unwrap(),
                Some((wv, sv)) => {
                    // x★: the warm image of encode e with the lumen values of frame f; y = A x★ (plain operator)
                    let mut xs: Vec<C32> = wv[e * nv..(e + 1) * nv].to_vec();
                    for (i, &q) in idx.iter().enumerate() {
                        xs[q] = sv[(i * ne + e) * nf + f];
                    }
                    let xd = st.clone_htod(&xs).unwrap();
                    let mut yd = st.alloc_zeros::<C32>(nc * ms).unwrap();
                    op.forward(&st, &xd, &mut yd).unwrap();
                    yd
                }
            };
            drop(yh);
            let mut x = st.alloc_zeros::<C32>(nv).unwrap();
            let mut snaps: Vec<(usize, Vec<C32>)> = vec![];
            let hist = pcgls_cb(&mut op, Some(&mut pre), &st, &y, &mut x, 0.0, iters, &mut |k, xd| {
                if save_at.contains(&k) {
                    let h = st.clone_dtoh(xd).unwrap();
                    snaps.push((k, idx.iter().map(|&q| h[q]).collect()));
                }
            })
            .unwrap();
            for (k, v) in snaps {
                let si = save_at.iter().position(|&a| a == k).unwrap();
                for (i, z) in v.into_iter().enumerate() {
                    vals_at[si][(i * ne + e) * nf + f] = z;
                }
            }
            if f == 0 && e == 0 && !save_at.is_empty() {
                let h0 = hist.first().copied().unwrap_or(1.0).max(1e-300);
                let pick: Vec<String> = hist.iter().enumerate().filter(|(k, _)| *k < 5 || (k + 1) % 5 == 0).map(|(k, v)| format!("{}:{:.3e}", k + 1, v / h0)).collect();
                println!("  frame 0 encode 0 residual history (relative): {}", pick.join(" "));
            }
            let h = st.clone_dtoh(&x).unwrap();
            for (i, &q) in idx.iter().enumerate() {
                vals[(i * ne + e) * nf + f] = h[q];
            }
            counts.push(ros.len());
        }
        println!("  frame {f}: readouts per encode {counts:?} ({:.1}s)", tf.elapsed().as_secs_f64());
    }
    write_c8_c(&out, &[n, ne, nf], &vals);
    for (si, &k) in save_at.iter().enumerate() {
        let stem = out.file_stem().unwrap().to_string_lossy().to_string();
        let p = out.with_file_name(format!("{stem}_it{k}.npy"));
        write_c8_c(&p, &[n, ne, nf], &vals_at[si]);
        println!("wrote {}", p.display());
    }
    println!("done ({:.1}s): {}", t0.elapsed().as_secs_f64(), out.display());
}

/// `--bin-group`: the readout whose frame a group of encodes (readouts g0..g1, encode 0..) gets: the middle one
/// (`mid`), or the one nearest the time the group's velocity estimate refers to ([`hasty_recon::data::velocity_time`])
fn bin_group_readout(sd: &hasty_recon::data::SimData, g0: usize, g1: usize, mid: bool) -> usize {
    if mid || g1 - g0 < 2 {
        return g0 + (g1 - g0) / 2;
    }
    let t: Vec<f64> = (g0..g1).map(|i| sd.ro.t_exc[i] - sd.ro.t_exc[g0]).collect();
    let m1: Vec<_> = (g0..g1).map(|i| sd.encode_m1[sd.ro.enc[i]]).collect();
    let tv = hasty_recon::data::velocity_time(&m1, &t);
    (g0..g1).min_by(|&a, &b| (t[a - g0] - tv).abs().partial_cmp(&(t[b - g0] - tv).abs()).unwrap()).unwrap()
}
