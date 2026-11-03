//! Off-resonance from susceptibility: ΔB_z/B0 = F⁻¹[(1/3 − k_z²/|k|²) F χ] (Lorentz-corrected dipole convolution, B0 ∥ z),
//! followed by a simulated 2nd-order spherical-harmonic shim over a box.

use std::sync::Arc;

use num_complex::Complex32;
use rustfft::{Fft, FftPlanner};

use crate::volume::Volume;

fn good_size(n: usize) -> usize {
    (n..).find(|&m| {
        let mut x = m;
        for p in [2, 3, 5] {
            while x % p == 0 {
                x /= p;
            }
        }
        x == 1
    })
    .unwrap()
}

/// In-place 3-D FFT of an x-fastest complex volume, lines distributed over threads.
fn fft3(data: &mut [Complex32], dims: [usize; 3], inverse: bool) {
    let mut planner = FftPlanner::<f32>::new();
    let nthreads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    for axis in 0..3 {
        let n = dims[axis];
        let fft: Arc<dyn Fft<f32>> = if inverse { planner.plan_fft_inverse(n) } else { planner.plan_fft_forward(n) };
        let stride = match axis {
            0 => 1,
            1 => dims[0],
            _ => dims[0] * dims[1],
        };
        let nlines = dims[0] * dims[1] * dims[2] / n;
        // line l -> base offset
        let base = |l: usize| -> usize {
            match axis {
                0 => l * dims[0],
                1 => (l % dims[0]) + (l / dims[0]) * dims[0] * dims[1],
                _ => l,
            }
        };
        let ptr = data.as_mut_ptr() as usize;
        std::thread::scope(|s| {
            for t in 0..nthreads {
                let fft = fft.clone();
                s.spawn(move || {
                    let mut line = vec![Complex32::new(0.0, 0.0); n];
                    let mut scratch = vec![Complex32::new(0.0, 0.0); fft.get_inplace_scratch_len()];
                    let d = ptr as *mut Complex32;
                    let mut l = t;
                    while l < nlines {
                        let b = base(l);
                        // SAFETY: lines are disjoint across threads
                        unsafe {
                            for i in 0..n {
                                line[i] = *d.add(b + i * stride);
                            }
                            fft.process_with_scratch(&mut line, &mut scratch);
                            for i in 0..n {
                                *d.add(b + i * stride) = line[i];
                            }
                        }
                        l += nthreads;
                    }
                });
            }
        });
    }
}

/// Field perturbation in ppm of B0 for an axis-aligned volume (affine must be diagonal, z = B0 axis). The volume is
/// zero-padded with `pad_xy` voxels of `chi_outside` laterally and extended by edge replication over `pad_z` voxels along
/// z (the body continues above and below the crop).
pub fn dipole_field(chi: &Volume<f32>, chi_outside: f32, pad_xy: usize, pad_z: usize) -> Volume<f32> {
    let a = &chi.affine;
    assert!(a[0][1] == 0.0 && a[0][2] == 0.0 && a[1][0] == 0.0 && a[1][2] == 0.0 && a[2][0] == 0.0 && a[2][1] == 0.0, "axis-aligned volume required");
    let d = chi.dims;
    let pd = [good_size(d[0] + 2 * pad_xy), good_size(d[1] + 2 * pad_xy), good_size(d[2] + 2 * pad_z)];
    let off = [(pd[0] - d[0]) / 2, (pd[1] - d[1]) / 2, (pd[2] - d[2]) / 2];
    let n = pd[0] * pd[1] * pd[2];
    let mut buf = vec![Complex32::new(0.0, 0.0); n];
    for k in 0..pd[2] {
        let ks = (k as isize - off[2] as isize).clamp(0, d[2] as isize - 1) as usize;
        for j in 0..pd[1] {
            for i in 0..pd[0] {
                let (ii, jj) = (i as isize - off[0] as isize, j as isize - off[1] as isize);
                let v = if ii >= 0 && jj >= 0 && (ii as usize) < d[0] && (jj as usize) < d[1] { chi.get(ii as usize, jj as usize, ks) } else { chi_outside };
                buf[i + pd[0] * (j + pd[1] * k)] = Complex32::new(v - chi_outside, 0.0);
            }
        }
    }
    fft3(&mut buf, pd, false);
    let vs = chi.voxel_size();
    for k in 0..pd[2] {
        let kz = (if k <= pd[2] / 2 { k as f64 } else { k as f64 - pd[2] as f64 }) / (pd[2] as f64 * vs[2]);
        for j in 0..pd[1] {
            let ky = (if j <= pd[1] / 2 { j as f64 } else { j as f64 - pd[1] as f64 }) / (pd[1] as f64 * vs[1]);
            for i in 0..pd[0] {
                let kx = (if i <= pd[0] / 2 { i as f64 } else { i as f64 - pd[0] as f64 }) / (pd[0] as f64 * vs[0]);
                let k2 = kx * kx + ky * ky + kz * kz;
                let dk = if k2 > 0.0 { 1.0 / 3.0 - kz * kz / k2 } else { 0.0 };
                buf[i + pd[0] * (j + pd[1] * k)] *= (dk / n as f64) as f32;
            }
        }
    }
    fft3(&mut buf, pd, true);
    let mut out = chi.map(|_| 0f32);
    for k in 0..d[2] {
        for j in 0..d[1] {
            for i in 0..d[0] {
                out.data[i + d[0] * (j + d[1] * k)] = buf[(i + off[0]) + pd[0] * ((j + off[1]) + pd[1] * (k + off[2]))].re;
            }
        }
    }
    out
}

/// Spherical-harmonic shim basis up to 2nd order at a physical point relative to `c`.
fn sh2(x: &[f64; 3], c: &[f64; 3]) -> [f64; 9] {
    let (px, py, pz) = (x[0] - c[0], x[1] - c[1], x[2] - c[2]);
    [1.0, px, py, pz, px * py, px * pz, py * pz, px * px - py * py, 2.0 * pz * pz - px * px - py * py]
}

/// Fit and subtract 0th–2nd order spherical harmonics over voxels where `mask` is set (least squares). Returns the
/// shim coefficients.
pub fn shim(field: &mut Volume<f32>, mask: &Volume<u8>, centre: [f64; 3]) -> [f64; 9] {
    let mut ata = [[0.0f64; 9]; 9];
    let mut atb = [0.0f64; 9];
    let d = field.dims;
    for k in 0..d[2] {
        for j in 0..d[1] {
            for i in 0..d[0] {
                let q = field.idx(i, j, k);
                if mask.data[q] == 0 {
                    continue;
                }
                let b = sh2(&field.to_phys([i as f64, j as f64, k as f64]), &centre);
                for r in 0..9 {
                    atb[r] += b[r] * field.data[q] as f64;
                    for c in 0..9 {
                        ata[r][c] += b[r] * b[c];
                    }
                }
            }
        }
    }
    // Gauss-Jordan with partial pivoting
    let mut m = ata;
    let mut x = atb;
    for col in 0..9 {
        let piv = (col..9).max_by(|&a, &b| m[a][col].abs().partial_cmp(&m[b][col].abs()).unwrap()).unwrap();
        m.swap(col, piv);
        x.swap(col, piv);
        let p = m[col][col];
        for c in 0..9 {
            m[col][c] /= p;
        }
        x[col] /= p;
        for r in 0..9 {
            if r != col {
                let f = m[r][col];
                for c in 0..9 {
                    m[r][c] -= f * m[col][c];
                }
                x[r] -= f * x[col];
            }
        }
    }
    for k in 0..d[2] {
        for j in 0..d[1] {
            for i in 0..d[0] {
                let q = field.idx(i, j, k);
                let b = sh2(&field.to_phys([i as f64, j as f64, k as f64]), &centre);
                let s: f64 = (0..9).map(|r| b[r] * x[r]).sum();
                field.data[q] -= s as f32;
            }
        }
    }
    x
}
