//! Toeplitz kernel construction: `K_k(p) = FFT_{2N}[ circ( sum_j wt_j w_j^mu extra_j (D/(N/2))^gamma e^{+i k_j . D} ) ](p)`.
//!
//! The kernel-domain sum is a type-1 NUFFT with `2N` modes per axis; the output is requested in FFT order so that it is the circulant
//! embedding directly (entry `D = -N` is never used by an `N`-point convolution and is left as is).

use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, Real};
use hasty_fft::{Direction, FftPlan, Norm};
use hasty_nufft::{NufftOpts, NufftPlan};

use crate::spec::KernelKey;
use crate::Error;

/// Non-uniform samples on the device. `coords` in radians per voxel (x fastest), `wt` real weights, `w` the four concomitant coefficients.
pub struct Samples<'a, R: Real> {
    pub m: usize,
    pub coords: [&'a CudaSlice<R>; 3],
    pub wt: &'a CudaSlice<R>,
    pub w: [&'a CudaSlice<R>; 4],
    /// complex per-sample weight sets referenced by `KernelKey::extra`
    pub extra: &'a [CudaSlice<R::Complex>],
}

pub struct KernelBuildOpts {
    pub tol: f64,
    pub sigma: f64,
}

impl Default for KernelBuildOpts {
    fn default() -> Self {
        Self {
            tol: 1e-6,
            sigma: 2.0,
        }
    }
}

/// Build every kernel of `keys` for an image of `dims = [Zn, Yn, Xn]` (padded to `2*dims`). Returns one `(2N)^3` complex array per key.
pub fn build_kernels<R: Real>(
    dev: &Device,
    stream: &Arc<CudaStream>,
    dims: [usize; 3],
    samples: &Samples<R>,
    keys: &[KernelKey],
    opts: &KernelBuildOpts,
) -> Result<Vec<CudaSlice<R::Complex>>, Error> {
    let pad = [2 * dims[0], 2 * dims[1], 2 * dims[2]];
    let npad: usize = pad.iter().product();
    let src = format!(
        "#define REAL {}\n{}",
        R::CTYPE,
        include_str!("toeplitz_kernels.cu")
    );
    let module = dev.load_module(
        &format!("toeplitz_kernels_{}", R::CTYPE),
        &src,
        &CompileSettings::default(),
    )?;
    let f_w = module.load_function("make_weights")?;
    let f_d = module.load_function("delta_mul")?;
    let mut plan = NufftPlan::<R>::new(
        dev,
        &[pad[2], pad[1], pad[0]],
        NufftOpts {
            tol: opts.tol,
            sigma: opts.sigma,
            ntransf: 1,
            fft_order: true,
            ..Default::default()
        },
    )?;
    plan.set_points(stream, &samples.coords, samples.m)?;
    let fft = FftPlan::<R>::new(dev, &pad, Direction::Forward, Norm::None)?;

    // group kernels by their sample-weight set (mu, extra); several gamma share one type-1 transform
    let mut groups: BTreeMap<([u8; 4], Option<usize>), Vec<usize>> = BTreeMap::new();
    for (i, k) in keys.iter().enumerate() {
        groups.entry((k.mu, k.extra)).or_default().push(i);
    }
    let mut out: Vec<Option<CudaSlice<R::Complex>>> = (0..keys.len()).map(|_| None).collect();
    let mut cw = stream.alloc_zeros::<R::Complex>(samples.m)?;
    let mut h = stream.alloc_zeros::<R::Complex>(npad)?;
    let m_i = samples.m as i32;
    for ((mu, extra), ids) in groups {
        let (has_extra, dummy) = match extra {
            Some(e) => (1i32, e),
            None => (0, 0),
        };
        let mu_i: [i32; 4] = [mu[0] as i32, mu[1] as i32, mu[2] as i32, mu[3] as i32];
        unsafe {
            let mut b = stream.launch_builder(&f_w);
            b.arg(samples.wt)
                .arg(samples.w[0])
                .arg(samples.w[1])
                .arg(samples.w[2])
                .arg(samples.w[3]);
            b.arg(&mu_i[0]).arg(&mu_i[1]).arg(&mu_i[2]).arg(&mu_i[3]);
            if has_extra == 1 {
                b.arg(&samples.extra[dummy]);
            } else {
                b.arg(samples.wt); // unused
            }
            b.arg(&has_extra).arg(&mut cw).arg(&m_i);
            b.launch(LaunchConfig::for_num_elems(samples.m as u32))?;
        }
        plan.type1(stream, &cw, &mut h, 1)?;
        for id in ids {
            let g = keys[id].gamma;
            let mut kbuf = stream.alloc_zeros::<R::Complex>(npad)?;
            let (p0, p1, p2) = (pad[2] as i32, pad[1] as i32, pad[0] as i32);
            let (g0, g1, g2) = (g[0] as i32, g[1] as i32, g[2] as i32);
            unsafe {
                stream
                    .launch_builder(&f_d)
                    .arg(&h)
                    .arg(&mut kbuf)
                    .arg(&p0)
                    .arg(&p1)
                    .arg(&p2)
                    .arg(&g0)
                    .arg(&g1)
                    .arg(&g2)
                    .launch(LaunchConfig::for_num_elems(npad as u32))?;
            }
            fft.execute_inplace(stream, &mut kbuf)?;
            out[id] = Some(kbuf);
        }
    }
    Ok(out.into_iter().map(|o| o.unwrap()).collect())
}
