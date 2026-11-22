//! Segment batches inside a coil (research/sample_kernels.md §18).
//!
//! The segmented operator is a sum over its L time segments,
//!
//!   y_c = e^{−iφ0} Σ_l a_l(τ) F[S_c P_l x](k) (+ the per-sample ΔC / order-3 terms, linear in each segment's grid),
//!
//! so it splits exactly into G groups of segments: A = Σ_g A_g, Aᴴ = Σ_g A_gᴴ, where A_g is the same operator with only
//! the diagonals P_l and interpolators a_l of group g. [`SegSplit`] runs a [`SegNufft`] built with `lb` segments (the
//! group size) once per group: forward calls after the first add into y (`SegNufft::forward_accumulate`), adjoint calls
//! after the first add into x (`SegNufft::adjoint_accumulate`). A grid slot then holds cb × lb channels instead of
//! cb × L, which is what makes a grid of ≈ 2 GB per channel (512³ image, 640³ grid) fit at all: one coil of L = 10
//! segments would need 21 GB of slot.
//!
//! Cost: the image side (pad, FFT, crop per channel) is unchanged apart from one extra read of x and the coil map per
//! extra group; the sample side runs once per (coil batch, group) pass instead of once per coil batch. When L is not a
//! multiple of the group size the last group is padded with zero segments (zero diagonals and interpolators): those
//! channels cost FFT time and contribute nothing, so [`plan_batch`] balances the groups.
//!
//! The plain engine is the special case of one group ([`SegSplit::from_engine`]): every call is then the engine's own.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::{Device, C32};

use crate::plan::NufftError;
use crate::segnufft::{balanced_batch, SegNufft, SegNufftOpts, SegPoints, SegSegments};

/// A channel batch: `cb` coils × `lb` segments per grid slot, `groups` segment groups of `lb` (the last one padded when
/// L is not a multiple), `passes` = coil batches × groups (each a pass over the samples), `bytes` of grid slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BatchPlan {
    pub cb: usize,
    pub lb: usize,
    pub groups: usize,
    pub passes: usize,
    pub channels: usize,
    pub bytes: usize,
}

impl BatchPlan {
    /// the plan for exactly `cb` coils × `lb` segments (lb balanced over the groups it implies)
    pub fn new(nc: usize, nseg: usize, channel_bytes: usize, cb: usize, lb: usize) -> Self {
        let lb = lb.clamp(1, nseg);
        let groups = nseg.div_ceil(lb);
        let lb = nseg.div_ceil(groups);
        let cb = balanced_batch(nc, cb.clamp(1, nc.max(1)));
        Self { cb, lb, groups, passes: nc.div_ceil(cb) * groups, channels: cb * lb, bytes: cb * lb * channel_bytes }
    }
}

/// The channel batch for a grid slot of `budget` bytes (one channel = `channel_bytes`, the engine's fine grid × 8 B;
/// `max_channels` caps it, e.g. from a measured optimum). Whole coils of all L segments while one coil fits (cb ≤
/// `max_cb`), else one coil in segment groups. Among the batches that fit it takes the fewest transformed channels —
/// every coil batch transforms the full slot, so a ragged last batch (10 coils in batches of 3: 3, 3, 3, 1) or a
/// zero-padded segment group costs FFT time for nothing (measured at 320³: 3 coils per batch 772 ms, 2 coils 714 ms,
/// research/sample_kernels.md §18) — then the fewest passes over the samples. None if not even one channel fits.
pub fn plan_batch(nc: usize, nseg: usize, channel_bytes: usize, budget: usize, max_channels: Option<usize>, max_cb: usize) -> Option<BatchPlan> {
    let cap = (budget / channel_bytes.max(1)).min(max_channels.unwrap_or(usize::MAX));
    if cap == 0 || nc == 0 || nseg == 0 {
        return None;
    }
    // transformed channels of a plan: batches × slot channels (padding included)
    let cost = |p: &BatchPlan| (nc.div_ceil(p.cb) * p.groups * p.channels, p.passes);
    let cands: Vec<BatchPlan> = if cap >= nseg {
        (1..=(cap / nseg).min(nc).min(max_cb.max(1))).map(|cb| BatchPlan::new(nc, nseg, channel_bytes, cb, nseg)).collect()
    } else {
        (1..=cap).map(|lb| BatchPlan::new(nc, nseg, channel_bytes, 1, lb)).collect()
    };
    cands.into_iter().filter(|p| p.channels <= cap).min_by_key(cost)
}

/// Segment groups of one encode: `groups[g]` holds the diagonals / interpolators of segments g·lb .. (g+1)·lb (zero
/// padded at the end)
pub struct SegSplitSegments {
    pub groups: Vec<SegSegments>,
}

/// The segmented operator over segment groups (module docs). `eng` is built with `lb` segments; its coil batch, tile
/// shapes, variants and FFT plans are its own and are shared by every group.
pub struct SegSplit {
    pub eng: SegNufft,
    /// segments of the operator (L) and per group
    pub nseg: usize,
    pub lb: usize,
    pub groups: usize,
}

impl SegSplit {
    /// one group: the engine's own operator
    pub fn from_engine(eng: SegNufft) -> Self {
        let nseg = eng.nseg;
        Self { eng, nseg, lb: nseg, groups: 1 }
    }

    /// `opts.nseg` = L segments in groups of (at most) `lb`; the engine's coil batch comes from `opts.max_grid_bytes`
    /// over cb × lb channels as usual. `knot_major` as [`SegNufft::new_temporal`].
    pub fn new(dev: &Device, dims: [usize; 3], res: [f64; 3], nc: usize, opts: SegNufftOpts, lb: usize, knot_major: bool) -> Result<Self, NufftError> {
        let nseg = opts.nseg;
        let groups = nseg.div_ceil(lb.clamp(1, nseg));
        let lb = nseg.div_ceil(groups);
        let eng = SegNufft::new_temporal(dev, dims, res, nc, SegNufftOpts { nseg: lb, ..opts }, knot_major)?;
        Ok(Self { eng, nseg, lb, groups })
    }

    /// The planner's batch ([`plan_batch`], whole coils while one fits, else segment groups of one coil) for a grid-slot
    /// budget of `budget` bytes (`opts.max_grid_bytes` is replaced by the plan's slot), on the engine's default fine
    /// grid (σ rule; with HASTY_NUFFT_GRID set the engine may choose another grid and the budget is then approximate).
    /// Errors if not even one channel fits.
    pub fn planned(dev: &Device, dims: [usize; 3], res: [f64; 3], nc: usize, opts: SegNufftOpts, budget: usize, max_channels: Option<usize>, knot_major: bool) -> Result<Self, NufftError> {
        let ge: usize = dims.iter().map(|&n| crate::gridsize::default_size(n, opts.sigma, opts.width)).product();
        let ch = ge * std::mem::size_of::<C32>();
        let p = plan_batch(nc, opts.nseg, ch, budget, max_channels, 16).ok_or_else(|| NufftError::Invalid(format!("grid budget {budget} B holds no channel of {ch} B")))?;
        // the slot of the plan plus the BFP y ↔ z intermediates of the default store (a dense buffer when the plane's
        // lines are not whole 16-line tiles)
        let slot = crate::seg_slot_bytes(dims, opts.sigma, opts.width, p.lb, p.cb, 1).max(p.cb * p.lb * ch) + (1 << 20);
        let o = SegNufftOpts { max_grid_bytes: slot, ..opts };
        if p.groups == 1 {
            let mut e = Self::from_engine(SegNufft::new_temporal(dev, dims, res, nc, o, knot_major)?);
            e.lb = e.nseg;
            return Ok(e);
        }
        Self::new(dev, dims, res, nc, o, p.lb, knot_major)
    }

    /// as [`SegNufft::segments`] with all L segments (`segph` L × nv, `atab` [ntab][L]): one device table per group
    pub fn segments(&self, st: &Arc<CudaStream>, segph: &[C32], atab: &[C32], t0: f64, dt: f64) -> Result<SegSplitSegments, NufftError> {
        let (nv, l, lb) = (self.eng.nv(), self.nseg, self.lb);
        assert_eq!(segph.len(), l * nv, "segments: {} diagonal values for {l} segments × {nv} voxels", segph.len());
        assert_eq!(atab.len() % l, 0);
        if self.groups == 1 {
            return Ok(SegSplitSegments { groups: vec![self.eng.segments(st, segph, atab, t0, dt)?] });
        }
        let ntab = atab.len() / l;
        let zero = C32::new(0.0, 0.0);
        let mut groups = Vec::with_capacity(self.groups);
        for g in 0..self.groups {
            let (l0, l1) = (g * lb, ((g + 1) * lb).min(l));
            let mut ph = vec![zero; lb * nv];
            ph[..(l1 - l0) * nv].copy_from_slice(&segph[l0 * nv..l1 * nv]);
            let mut at = vec![zero; ntab * lb];
            for i in 0..ntab {
                at[i * lb..i * lb + (l1 - l0)].copy_from_slice(&atab[i * l + l0..i * l + l1]);
            }
            groups.push(self.eng.segments(st, &ph, &at, t0, dt)?);
        }
        Ok(SegSplitSegments { groups })
    }

    /// y = A x (all groups summed)
    pub fn forward(&mut self, st: &Arc<CudaStream>, seg: &SegSplitSegments, pts: &SegPoints, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        let keep = self.eng.forward_accumulate;
        let mut r = Ok(());
        for (g, s) in seg.groups.iter().enumerate() {
            self.eng.forward_accumulate = keep || g > 0;
            r = self.eng.forward(st, s, pts, x, y);
            if r.is_err() {
                break;
            }
        }
        self.eng.forward_accumulate = keep;
        r
    }

    /// r = A x − data (the first group forms the residual, the others add)
    pub fn forward_residual(&mut self, st: &Arc<CudaStream>, seg: &SegSplitSegments, pts: &SegPoints, x: &CudaSlice<C32>, data: &CudaSlice<C32>, r: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        let keep = self.eng.forward_accumulate;
        let mut res = Ok(());
        for (g, s) in seg.groups.iter().enumerate() {
            res = if g == 0 {
                self.eng.forward_accumulate = false;
                self.eng.forward_residual(st, s, pts, x, data, r)
            } else {
                self.eng.forward_accumulate = true;
                self.eng.forward(st, s, pts, x, r)
            };
            if res.is_err() {
                break;
            }
        }
        self.eng.forward_accumulate = keep;
        res
    }

    /// x = Aᴴ y (all groups summed; with `eng.adjoint_accumulate` x += Aᴴ y)
    pub fn adjoint(&mut self, st: &Arc<CudaStream>, seg: &SegSplitSegments, pts: &SegPoints, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), NufftError> {
        let keep = self.eng.adjoint_accumulate;
        let mut r = Ok(());
        for (g, s) in seg.groups.iter().enumerate() {
            self.eng.adjoint_accumulate = keep || g > 0;
            r = self.eng.adjoint(st, s, pts, y, x);
            if r.is_err() {
                break;
            }
        }
        self.eng.adjoint_accumulate = keep;
        r
    }

    /// the engine's autotune on the first group (every group has the same shape)
    pub fn autotune(&mut self, st: &Arc<CudaStream>, seg: &SegSplitSegments, pts: &SegPoints) -> Result<(f32, f32), NufftError> {
        self.eng.autotune(st, &seg.groups[0], pts)
    }

    /// grid-FFT transforms of one call: the engine's count (per group pass)
    pub fn plan(&self) -> BatchPlan {
        let ch = self.eng.plan.grid_elems() * std::mem::size_of::<C32>();
        BatchPlan { cb: self.eng.cb, lb: self.lb, groups: self.groups, passes: self.eng.nc.div_ceil(self.eng.cb) * self.groups, channels: self.eng.cb * self.lb, bytes: self.eng.cb * self.lb * ch }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_batch_rules() {
        let ch = 1usize << 30;
        // whole coils while one fits: 100 channels in 110 GB → 10 coils of 10
        let p = plan_batch(10, 10, ch, 110 * ch, None, 16).unwrap();
        assert_eq!((p.cb, p.lb, p.groups, p.passes), (10, 10, 1, 1));
        // 33 channels: 3 coils would run batches 3, 3, 3, 1 (120 transformed channels): 2 coils (100, 5 passes)
        let p = plan_batch(10, 10, ch, 33 * ch, None, 16).unwrap();
        assert_eq!((p.cb, p.lb, p.passes), (2, 10, 5));
        // one coil does not fit: 4 channels → groups of 4 would pad to 12; groups of 2 transform exactly 10
        let p = plan_batch(10, 10, ch, 4 * ch, None, 16).unwrap();
        assert_eq!((p.cb, p.lb, p.groups, p.passes), (1, 2, 5, 50));
        // 5 channels → 2 groups of 5 (exact)
        let p = plan_batch(1, 10, ch, 5 * ch + 1, None, 16).unwrap();
        assert_eq!((p.cb, p.lb, p.groups, p.passes), (1, 5, 2, 2));
        // 16 segments, 9 channels → 2 groups of 8
        let p = plan_batch(1, 16, ch, 9 * ch, None, 16).unwrap();
        assert_eq!((p.lb, p.groups), (8, 2));
        // a channel cap below the memory
        let p = plan_batch(10, 10, ch, 110 * ch, Some(50), 16).unwrap();
        assert_eq!((p.cb, p.lb, p.passes), (5, 10, 2));
        assert!(plan_batch(10, 10, ch, ch - 1, None, 16).is_none());
    }
}
