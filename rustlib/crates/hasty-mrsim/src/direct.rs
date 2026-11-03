//! GPU direct-sum truth simulator.
//!
//! s_c(i) = Σ_p a_pc · exp(−R2*_p τ_i) · exp(−i φ_p(i)),
//! φ = k·u + m1·(ṽ + δ ṽ_t) + m2h·ã + w·m(x) + ω τ,
//! with sample parameters from [`crate::sequence::Acquisition::sample_params`] and source parameters from [`pack_sources`].
//! No Fourier-domain approximation: cost is O(samples · sources · coils), f32 terms, f64 accumulation.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, LaunchConfig, PushKernelArg};
use hasty_cuda::{CompileSettings, Device, C32, C64};

use crate::sequence::SAMPLE_FLOATS;
use crate::{Error, Vec3};

pub const SOURCE_FLOATS: usize = 20;

/// One point source (an isochromat cloud of fixed size, e.g. a fine-grid voxel).
#[derive(Clone, Copy, Debug, Default)]
pub struct Source {
    /// apparent (GNL-warped) position minus the FOV centre, m
    pub u: Vec3,
    /// true physical position (isocentre frame), m — used by the concomitant terms
    pub x: Vec3,
    /// J v: velocity seen by the encoding gradients, m/s
    pub vt: Vec3,
    /// J ∂v/∂t, m/s² (first-order correction for the offset between excitation time and flow-state time)
    pub vtt: Vec3,
    /// J a (Lagrangian acceleration), m/s²
    pub at: Vec3,
    /// off-resonance, rad/s
    pub omega: f64,
    /// 1/s
    pub r2s: f64,
}

pub fn monomials(x: &Vec3) -> [f64; 4] {
    [x[2] * x[2], x[0] * x[0] + x[1] * x[1], x[0] * x[2], x[1] * x[2]]
}

pub fn pack_sources(src: &[Source]) -> Vec<f32> {
    let mut v = Vec::with_capacity(src.len() * SOURCE_FLOATS);
    for s in src {
        let m = monomials(&s.x);
        let f = [
            s.u[0], s.u[1], s.u[2], m[0], m[1], m[2], m[3], s.vt[0], s.vt[1], s.vt[2], s.vtt[0], s.vtt[1], s.vtt[2], s.at[0], s.at[1], s.at[2], s.omega, s.r2s, 0.0, 0.0,
        ];
        v.extend(f.iter().map(|&x| x as f32));
    }
    v
}

pub struct DirectSim {
    func: CudaFunction,
    pub nc: usize,
    /// work per launch (samples × sources), keeps single launches well below a second
    pub max_terms: usize,
}

impl DirectSim {
    pub fn new(dev: &Device, nc: usize) -> Result<Self, Error> {
        let settings = CompileSettings { options: vec![format!("-DNC={nc}")], no_cache: false };
        let func = dev.load_function(&format!("mrsim_direct_nc{nc}"), include_str!("direct_sum.cu"), "direct_sum", &settings)?;
        Ok(Self { func, nc, max_terms: 1 << 37 })
    }

    /// `out[i][c] += Σ_p …` for `ns` samples (16 floats each) and `np` sources (20 floats + `nc` complex amplitudes each).
    pub fn accumulate(
        &self,
        stream: &Arc<CudaStream>,
        samples: &CudaSlice<f32>,
        ns: usize,
        sources: &CudaSlice<f32>,
        amp: &CudaSlice<C32>,
        np: usize,
        out: &mut CudaSlice<C64>,
    ) -> Result<(), Error> {
        let s = samples.slice(..);
        let mut o = out.slice_mut(..);
        self.accumulate_view(stream, &s, ns, sources, amp, np, &mut o)
    }

    pub fn accumulate_view(
        &self,
        stream: &Arc<CudaStream>,
        samples: &CudaView<f32>,
        ns: usize,
        sources: &CudaSlice<f32>,
        amp: &CudaSlice<C32>,
        np: usize,
        out: &mut CudaViewMut<C64>,
    ) -> Result<(), Error> {
        assert!(samples.len() >= ns * SAMPLE_FLOATS && sources.len() >= np * SOURCE_FLOATS && amp.len() >= np * self.nc && out.len() >= ns * self.nc);
        if ns == 0 || np == 0 {
            return Ok(());
        }
        let s_chunk = ns.min(1 << 20);
        let p_chunk = (self.max_terms / s_chunk).max(64).min(np);
        let mut s0 = 0;
        while s0 < ns {
            let sn = s_chunk.min(ns - s0);
            let mut p0 = 0;
            while p0 < np {
                let pn = p_chunk.min(np - p0);
                let smp = samples.slice(s0 * SAMPLE_FLOATS..(s0 + sn) * SAMPLE_FLOATS);
                let src = sources.slice(p0 * SOURCE_FLOATS..(p0 + pn) * SOURCE_FLOATS);
                let a = amp.slice(p0 * self.nc..(p0 + pn) * self.nc);
                let mut o = out.slice_mut(s0 * self.nc..(s0 + sn) * self.nc);
                let (sn_i, pn_i) = (sn as i32, pn as i32);
                unsafe {
                    stream
                        .launch_builder(&self.func)
                        .arg(&smp)
                        .arg(&sn_i)
                        .arg(&src)
                        .arg(&a)
                        .arg(&pn_i)
                        .arg(&mut o)
                        .launch(LaunchConfig { grid_dim: ((sn as u32).div_ceil(128), 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 })?;
                }
                p0 += pn;
            }
            s0 += sn;
        }
        Ok(())
    }
}

/// f64 CPU reference of the same sum (tests / spot checks of the GPU result on random sample subsets).
pub fn direct_sum_cpu(samples: &[f32], sources: &[Source], amp: &[num_complex::Complex64], nc: usize) -> Vec<num_complex::Complex64> {
    let ns = samples.len() / SAMPLE_FLOATS;
    let mut out = vec![num_complex::Complex64::new(0.0, 0.0); ns * nc];
    for i in 0..ns {
        let p: Vec<f64> = samples[i * SAMPLE_FLOATS..(i + 1) * SAMPLE_FLOATS].iter().map(|&v| v as f64).collect();
        for (j, s) in sources.iter().enumerate() {
            let m = monomials(&s.x);
            let mut ph = p[0] * s.u[0] + p[1] * s.u[1] + p[2] * s.u[2];
            for a in 0..3 {
                ph += p[3 + a] * (s.vt[a] + p[14] * s.vtt[a]) + p[6 + a] * s.at[a];
            }
            for q in 0..4 {
                ph += p[9 + q] * m[q];
            }
            ph += s.omega * p[13];
            let e = num_complex::Complex64::from_polar((-s.r2s * p[13]).exp(), -ph);
            for c in 0..nc {
                out[i * nc + c] += e * amp[j * nc + c];
            }
        }
    }
    out
}
