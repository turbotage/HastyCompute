//! Fine-grid sizes chosen by measured FFT time (research/temporal_operator.md §20; stencil_kernel_optimality.md
//! proposal 7). The kernel width W is fixed, so the gridding cost hardly depends on the grid size; the FFT does, through
//! the volume and the efficiency of each length (radix passes, shared memory per line). A larger σ at the same W only
//! lowers the aliasing error, so any size ≥ the smooth size of the requested σ is at least as accurate: the choice
//! is made among those by time alone.
//!
//! `HASTY_NUFFT_GRID=auto` (SegNufft engines; or [`set_thread_grid_rule`]) times the block-pruned channel-interleaved
//! transform ([`hasty_fft::PrunedChanFft3`], the engine's layout) for a few candidate lengths per axis, one axis at a
//! time around the default, and keeps the fastest per axis (separable model), accepting the combination only if a
//! direct timing confirms it beats the default. Unset or `fixed`: the default sizes; `n0,n1,n2`: explicit sizes.
//! Decisions are cached per process. With a chosen grid the engine sets the window shape β for the effective
//! σ_eff = min_a nf_a / N_a ≥ σ ([`sigma_eff`]) instead of the requested σ.

use std::collections::HashMap;
use std::sync::Mutex;

use hasty_cuda::{Device, C32};
use hasty_fft::PrunedChanFft3;

use crate::kernel::next_smooth_even;
use crate::plan::NufftError;

/// Even 2^a 3^b 5^c 7^d lengths in [lo, hi].
pub fn smooth_sizes(lo: usize, hi: usize) -> Vec<usize> {
    let mut v = vec![];
    let mut n = next_smooth_even(lo);
    while n <= hi {
        v.push(n);
        n = next_smooth_even(n + 1);
    }
    v
}

/// default fine-grid length of an axis: the smallest even 7-smooth n ≥ σ N (and ≥ 2W), as `NufftPlan`
pub fn default_size(n: usize, sigma: f64, w: usize) -> usize {
    next_smooth_even(((sigma * n as f64).ceil() as usize).max(2 * w))
}

/// Median time (ms) of one forward + one inverse block-pruned transform on a grid `nf` (x fastest) for an image `dims`
/// with `nch` interleaved channels (`buf` must hold ≥ nf0 nf1 nf2 nch elements). Errors if the plan is declined.
pub fn time_grid_fft(dev: &Device, nf: [usize; 3], dims: [usize; 3], nch: usize, buf: &mut cudarc::driver::CudaSlice<C32>, reps: usize) -> Result<f64, NufftError> {
    let keep = [2, 1, 0].map(|a| (dims[a] - dims[a] / 2, dims[a] / 2));
    let p = PrunedChanFft3::<f32>::new(dev, [nf[2], nf[1], nf[0]], nch, keep, 1.0).map_err(|e| NufftError::Invalid(format!("grid {nf:?}: {e}")))?;
    let need = p.len();
    let st = dev.default_stream();
    let mut view = buf.slice_mut(0..need);
    let fl = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
    let ev = (dev.ctx.new_event(fl)?, dev.ctx.new_event(fl)?);
    let mut ts = vec![];
    for i in 0..reps + 1 {
        ev.0.record(&st)?;
        p.forward(&st, &mut view)?;
        p.inverse(&st, &mut view)?;
        ev.1.record(&st)?;
        ev.1.synchronize()?;
        if i > 0 {
            ts.push(ev.0.elapsed_ms(&ev.1)? as f64);
        }
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(ts[ts.len() / 2])
}

/// Outcome of [`choose_grid`]: the sizes and the measurements behind them.
#[derive(Clone, Debug)]
pub struct GridChoice {
    pub nf: [usize; 3],
    pub default: [usize; 3],
    /// per axis: (length, ms of forward + inverse with the other axes at their defaults)
    pub per_axis: [Vec<(usize, f64)>; 3],
    /// (default, chosen) directly timed
    pub t_default: f64,
    pub t_chosen: f64,
}

static CACHE: Mutex<Option<HashMap<(usize, [usize; 3], u64, usize, usize), [usize; 3]>>> = Mutex::new(None);

/// Candidate lengths per axis: smooth sizes from the default up to `grow` × the default (at most `kmax`), so σ never
/// drops below the requested one.
pub fn candidates(n: usize, sigma: f64, w: usize, grow: f64, kmax: usize) -> Vec<usize> {
    let d = default_size(n, sigma, w);
    let mut v = smooth_sizes(d, (d as f64 * grow).floor() as usize);
    v.truncate(kmax);
    v
}

/// Measure and choose the fine-grid sizes for an image `dims` at oversampling ≥ `sigma`, width `w`, with `nch`
/// channels per transform (module docs). Needs a scratch buffer of the largest timed grid × nch (allocated here,
/// freed on return; ≈ 2.5 GB at 1 mm with 16 channels, before the engine allocates its slots).
pub fn choose_grid(dev: &Device, dims: [usize; 3], sigma: f64, w: usize, nch: usize, grow: f64, kmax: usize) -> Result<GridChoice, NufftError> {
    let def = dims.map(|n| default_size(n, sigma, w));
    let cand = [0, 1, 2].map(|a| candidates(dims[a], sigma, w, grow, kmax));
    // scratch for the one-axis-at-a-time timings (the chosen combination re-allocates only if it is larger)
    let vol = |g: [usize; 3]| g.iter().product::<usize>();
    let big: usize = (0..3).map(|a| { let mut g = def; g[a] = *cand[a].last().unwrap_or(&def[a]); vol(g) }).max().unwrap().max(vol(def));
    let st = dev.default_stream();
    let mut buf = st.alloc_zeros::<C32>(big * nch)?;
    let reps = 3;
    let t_default = time_grid_fft(dev, def, dims, nch, &mut buf, reps)?;
    let mut per_axis: [Vec<(usize, f64)>; 3] = Default::default();
    let mut nf = def;
    for a in 0..3 {
        let mut best = (def[a], t_default);
        for &n in &cand[a] {
            let t = if n == def[a] {
                t_default
            } else {
                let mut g = def;
                g[a] = n;
                match time_grid_fft(dev, g, dims, nch, &mut buf, reps) {
                    Ok(t) => t,
                    Err(_) => continue,
                }
            };
            per_axis[a].push((n, t));
            if t < best.1 {
                best = (n, t);
            }
        }
        nf[a] = best.0;
    }
    if vol(nf) > big {
        drop(buf);
        buf = st.alloc_zeros::<C32>(vol(nf) * nch)?;
    }
    let t_chosen = if nf == def { t_default } else { time_grid_fft(dev, nf, dims, nch, &mut buf, reps)? };
    if t_chosen >= t_default {
        nf = def;
    }
    Ok(GridChoice { nf, default: def, per_axis, t_default, t_chosen })
}

/// Fine-grid size rule of new SegNufft engines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GridRule {
    /// the smallest even 7-smooth n ≥ σ N per axis (the plan's rule)
    Fixed,
    /// [`choose_grid`]: the fastest measured sizes with σ_a ≥ σ (candidates up to 1.15 × the default, ≤ 4 per axis)
    Auto,
    /// explicit sizes (x fastest; each ≥ the fixed rule's size, even, 7-smooth)
    Explicit([usize; 3]),
}

thread_local! {
    static GRID_RULE: std::cell::Cell<Option<GridRule>> = const { std::cell::Cell::new(None) };
}

/// Grid rule of the engines built on this thread from now on (None: back to env HASTY_NUFFT_GRID).
pub fn set_thread_grid_rule(r: Option<GridRule>) {
    GRID_RULE.with(|c| c.set(r));
}

/// [`set_thread_grid_rule`], else env HASTY_NUFFT_GRID = fixed (default) | auto | "n0,n1,n2"
pub fn grid_rule() -> Result<GridRule, NufftError> {
    if let Some(r) = GRID_RULE.with(|c| c.get()) {
        return Ok(r);
    }
    match std::env::var("HASTY_NUFFT_GRID").as_deref() {
        Err(_) | Ok("fixed") | Ok("") => Ok(GridRule::Fixed),
        Ok("auto") => Ok(GridRule::Auto),
        Ok(v) => {
            let n: Vec<usize> = v.split(',').filter_map(|t| t.trim().parse().ok()).collect();
            if n.len() == 3 {
                Ok(GridRule::Explicit([n[0], n[1], n[2]]))
            } else {
                Err(NufftError::Invalid(format!("HASTY_NUFFT_GRID={v}: fixed, auto or n0,n1,n2")))
            }
        }
    }
}

/// channels of the timed transforms: fixed, so that every engine of a process on the same image gets the same grid
/// (sample sets are grid-specific and may be shared between engines)
pub const GRID_TIMING_CHANNELS: usize = 16;

/// Grid sizes for a SegNufft engine ([`grid_rule`]): None = the plan's fixed rule. Auto decisions are cached per
/// (device, image, σ, W) and process; HASTY_NUFFT_GRID_LOG=1 prints them.
pub fn grid_for_engine(dev: &Device, dims: [usize; 3], sigma: f64, w: usize) -> Result<Option<[usize; 3]>, NufftError> {
    match grid_rule()? {
        GridRule::Fixed => Ok(None),
        GridRule::Explicit(nf) => Ok(Some(nf)),
        GridRule::Auto => {
            let nch = GRID_TIMING_CHANNELS;
            let key = (dev.ordinal, dims, sigma.to_bits(), w, nch);
            if let Some(nf) = CACHE.lock().unwrap().get_or_insert_with(HashMap::new).get(&key) {
                return Ok(Some(*nf));
            }
            let c = choose_grid(dev, dims, sigma, w, nch, 1.15, 4)?;
            if std::env::var("HASTY_NUFFT_GRID_LOG").is_ok_and(|v| v != "0") {
                eprintln!("    [grid] {dims:?} σ ≥ {sigma}, {nch} channels: default {:?} {:.2} ms → {:?} {:.2} ms (per axis {:?})", c.default, c.t_default, c.nf, c.t_chosen, c.per_axis);
            }
            CACHE.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, c.nf);
            Ok(Some(c.nf))
        }
    }
}

/// effective oversampling of a grid: min_a nf_a / N_a (the kernel shape β is set for it; every axis has at least it)
pub fn sigma_eff(nf: [usize; 3], dims: [usize; 3]) -> f64 {
    (0..3).map(|a| nf[a] as f64 / dims[a] as f64).fold(f64::INFINITY, f64::min)
}
