//! Pruned in-place 3-D FFT of channel-interleaved data `[n2][n1][n0][nch]` (channels fastest, e.g. the coil × segment
//! grids of a multi-channel NUFFT) whose image occupies a *wrapped* block per axis: indices `[0, hi_a) ∪ [n_a − lo_a, n_a)`
//! (the zero-padded NUFFT layout with the image centred on index 0). Everything outside is zero on the way in
//! (forward) or not needed on the way out (inverse), so
//!
//! forward  x-pass: lines (y, z) in the block only, reading the block's x entries; y-pass: planes z in the block,
//!          reading the block's y entries; z-pass: all lines, reading the block's z entries — nothing outside the block
//!          is ever read, so the buffer needs no zeroing before the image is written;
//! inverse  z-pass: all lines, writing the block's z entries; y-pass: planes z in the block, writing the block's y
//!          entries; x-pass: lines (y, z) in the block, writing the block's x entries — only the block is valid after.
//!
//! Fraction of the unpruned work per direction ≈ (h1 h2 + h2 + 1) / 3 · (1 + h) / 2 with h_a the block fraction of axis
//! a: 0.73 at oversampling 1.25, 0.44 at 2. Results equal the full transform on the block (tested to rounding).

use std::sync::Arc;

use cudarc::driver::{CudaStream, DevicePtrMut};
use hasty_cuda::{Device, Real};

use crate::codegen::KernelFormat;
use crate::plan::{AxisPlan, AxisSpec, Direction, FftError, LaunchPtrs, LineAddr, LineSplit};

pub struct PrunedChanFft3<R: Real> {
    /// `[n2, n1, n0]`
    pub n: [usize; 3],
    pub nch: usize,
    /// per axis `[2, 1, 0]`: (head, tail) of the wrapped block
    pub keep: [(usize, usize); 3],
    fwd: [AxisPlan<R>; 3],
    inv: [AxisPlan<R>; 3],
}

impl<R: Real> PrunedChanFft3<R> {
    /// `n = [n2, n1, n0]`, `keep[a] = (hi, lo)` for the same axes: the block is `[0, hi) ∪ [n − lo, n)`. `scale_inv`
    /// multiplies the inverse (1 = unnormalised, as the forward).
    pub fn new(dev: &Device, n: [usize; 3], nch: usize, keep: [(usize, usize); 3], scale_inv: f64) -> Result<Self, FftError> {
        for a in 0..3 {
            let (h, l) = keep[a];
            if h == 0 || h + l > n[a] {
                return Err(FftError::Invalid(format!("pruned block ({h}, {l}) does not fit axis length {}", n[a])));
            }
        }
        let [n2, n1, n0] = n;
        let cnt = |a: usize| keep[a].0 + keep[a].1;
        let gap = |a: usize| n[a] - cnt(a);
        let (sx, sy, sz) = (nch, n0 * nch, n1 * n0 * nch);
        // tiles run over `la` (stride 1: channels, or channels × x × … where contiguous); the line count `lb` is a
        // grid dimension (≤ 65535)
        let addr = |la_cnt: usize, lb_cnt: usize, in_sn: usize, lb: usize| LineAddr { la_cnt, lb_cnt, in_la: 1, in_lb: lb, in_sn, out_la: 1, out_lb: lb, out_sn: in_sn };
        let spec = |len: usize, dir: Direction, scale: f64, ad: LineAddr, (n_in, n_in_tail): (usize, usize), (n_out, n_out_tail): (usize, usize)| AxisSpec {
            n: len,
            n_in,
            n_out,
            n_in_tail,
            n_out_tail,
            dir,
            scale,
            conv: false,
            pre: vec![],
            post: vec![],
            accumulate: false,
            contiguous: false,
            addr: ad,
            kfmt: KernelFormat::Complex,
            poly_pre: false,
            poly_post: false,
            vox: [1, 1, 1],
        };
        // lines of the x pass: channels × (y, z) in the block; of the y pass: (channels, x) × z in the block; of the
        // z pass: (channels, x, y)
        let split_x = LineSplit { lb1_cnt: cnt(1), lb1_hi: keep[1].0, lb1_gap: gap(1), lb2_hi: keep[0].0, lb2_gap: gap(0), in_lb2: sz, out_lb2: sz };
        let split_y = LineSplit { lb1_cnt: cnt(0), lb1_hi: keep[0].0, lb1_gap: gap(0), lb2_hi: usize::MAX / 2, lb2_gap: 0, in_lb2: 0, out_lb2: 0 };
        if cnt(1) * cnt(0) > 65535 {
            return Err(FftError::Invalid(format!("pruned x pass: {} lines exceed the grid limit", cnt(1) * cnt(0))));
        }
        let full = |a: usize| (n[a], 0usize);
        use Direction::*;
        let mk = |sp: AxisSpec, split: Option<LineSplit>| -> Result<AxisPlan<R>, FftError> {
            let mut p = AxisPlan::from_spec(dev, &sp)?;
            if let Some(s) = split {
                p.split = s;
            }
            Ok(p)
        };
        let fwd = [
            mk(spec(n0, Forward, 1.0, addr(nch, cnt(1) * cnt(0), sx, sy), keep[2], full(2)), Some(split_x))?,
            mk(spec(n1, Forward, 1.0, addr(n0 * nch, cnt(0), sy, sz), keep[1], full(1)), Some(split_y))?,
            mk(spec(n2, Forward, 1.0, addr(n1 * n0 * nch, 1, sz, 0), keep[0], full(0)), None)?,
        ];
        let inv = [
            mk(spec(n2, Inverse, 1.0, addr(n1 * n0 * nch, 1, sz, 0), full(0), keep[0]), None)?,
            mk(spec(n1, Inverse, 1.0, addr(n0 * nch, cnt(0), sy, sz), full(1), keep[1]), Some(split_y))?,
            mk(spec(n0, Inverse, scale_inv, addr(nch, cnt(1) * cnt(0), sx, sy), full(2), keep[2]), Some(split_x))?,
        ];
        Ok(Self { n, nch, keep, fwd, inv })
    }

    pub fn len(&self) -> usize {
        self.n.iter().product::<usize>() * self.nch
    }

    fn run<B: DevicePtrMut<R::Complex>>(plans: &[AxisPlan<R>; 3], stream: &Arc<CudaStream>, buf: &mut B, need: usize) -> Result<(), FftError> {
        if buf.len() < need {
            return Err(FftError::Invalid(format!("buffer of {} elements, the pruned transform needs {need}", buf.len())));
        }
        let (p, _g) = buf.device_ptr_mut(stream);
        for ax in plans {
            ax.launch(stream, &LaunchPtrs { inp: p, out: p, ..Default::default() })?;
        }
        Ok(())
    }

    /// in place: block data in, full spectrum out (entries outside the block are never read)
    pub fn forward<B: DevicePtrMut<R::Complex>>(&self, stream: &Arc<CudaStream>, buf: &mut B) -> Result<(), FftError> {
        Self::run(&self.fwd, stream, buf, self.len())
    }

    /// in place: full spectrum in, the block of the inverse transform out (the rest is left undefined)
    pub fn inverse<B: DevicePtrMut<R::Complex>>(&self, stream: &Arc<CudaStream>, buf: &mut B) -> Result<(), FftError> {
        Self::run(&self.inv, stream, buf, self.len())
    }
}
