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
//!
//! **k-support pruning** ([`PrunedChanFft3::new_ksupport`], [`KSupport`]): a NUFFT gathers from / scatters into only
//! the k-space cells its samples' kernel supports meet. With that set known (per kx row of the z pass, and its
//! projections on the three axes), the forward also skips the y-pass lines (kx) and z-pass columns (kx, ky) that meet
//! no sample and writes only the needed kx range, the ky range of each kx line and the kz range of each column (the
//! chord of a ball); the inverse skips the same lines and reads only the same ranges (the rest of a scatter grid is
//! zero). Exact on the support: the forward spectrum is correct on every
//! cell of the support (elsewhere undefined), the inverse equals the full one whenever the input is zero outside it.

use std::sync::Arc;

use cudarc::driver::{CudaStream, DevicePtrMut};
use hasty_cuda::{Device, Real};

use crate::codegen::KernelFormat;
use crate::plan::{AxisPlan, AxisSpec, Direction, FftError, LaunchPtrs, LineAddr, LineSplit};

/// The k-space cells a transform must produce (forward) or may read (inverse), as wrapped index ranges
/// `[0, head) ∪ [n − tail, n)` (the FFT output layout, k = 0 at index 0).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KSupport {
    /// per axis `[x, y, z]` = `[n0, n1, n2]` (x fastest): the projection of the support
    pub proj: [(usize, usize); 3],
    /// per ky index (n1 entries): the kx range of the support's columns (kx, ky); (0, 0) = no column in this row
    pub rows: Vec<(usize, usize)>,
    /// per kx index (n0 entries): the ky range of the support's columns at that kx (the y-pass entries written by the
    /// forward / read by the inverse); (0, 0) = none
    pub ycols: Vec<(usize, usize)>,
    /// per column `ky · n0 + kx`: the kz range of the support on that column (the z-pass entries written / read);
    /// (0, 0) = not in the support
    pub zcols: Vec<(usize, usize)>,
}

impl KSupport {
    /// Everything (no pruning) for a grid `n = [n0, n1, n2]` (x fastest).
    pub fn full(n: [usize; 3]) -> Self {
        Self { proj: n.map(|m| (m, 0)), rows: vec![(n[0], 0); n[1]], ycols: vec![(n[1], 0); n[0]], zcols: vec![(n[2], 0); n[0] * n[1]] }
    }

    /// Smallest wrapped range `[0, h) ∪ [n − t, n)` containing a set of indices given by `has(i)`, split at n / 2
    /// (h ≥ 1: index 0, the k-space centre, is always kept).
    pub fn hull(n: usize, has: impl Fn(usize) -> bool) -> (usize, usize) {
        let s = n / 2;
        let h = (0..s).rev().find(|&i| has(i)).map_or(1, |i| i + 1).max(1);
        let t = (s..n).find(|&i| has(i)).map_or(0, |i| n - i);
        (h, t)
    }

    /// From a column mask `xy(kx, ky)` and one kz mask for all columns (`n = [n0, n1, n2]`).
    pub fn from_masks(n: [usize; 3], xy: impl Fn(usize, usize) -> bool, z: impl Fn(usize) -> bool) -> Self {
        let zr = Self::hull(n[2], z);
        Self::from_columns(n, xy, |_, _| zr)
    }

    /// From a column mask `xy(kx, ky)` and the kz range `zr(kx, ky)` of every column in it (wrapped, as [`hull`](Self::hull)).
    pub fn from_columns(n: [usize; 3], xy: impl Fn(usize, usize) -> bool, zr: impl Fn(usize, usize) -> (usize, usize)) -> Self {
        let rows: Vec<(usize, usize)> = (0..n[1])
            .map(|ky| if (0..n[0]).any(|kx| xy(kx, ky)) { Self::hull(n[0], |kx| xy(kx, ky)) } else { (0, 0) })
            .collect();
        let ycols: Vec<(usize, usize)> = (0..n[0])
            .map(|kx| if (0..n[1]).any(|ky| xy(kx, ky)) { Self::hull(n[1], |ky| xy(kx, ky)) } else { (0, 0) })
            .collect();
        let zcols: Vec<(usize, usize)> = (0..n[0] * n[1]).map(|q| if xy(q % n[0], q / n[0]) { zr(q % n[0], q / n[0]) } else { (0, 0) }).collect();
        let mx = |v: &[(usize, usize)]| (v.iter().map(|r| r.0).max().unwrap_or(1).max(1), v.iter().map(|r| r.1).max().unwrap_or(0));
        let px = mx(&rows);
        let py = Self::hull(n[1], |ky| rows[ky] != (0, 0));
        let pz = mx(&zcols);
        Self { proj: [px, py, pz], rows, ycols, zcols }
    }

    /// fraction of the full z-pass columns, y-pass lines (kx) and per-axis output entries that the support keeps
    pub fn fractions(&self, n: [usize; 3]) -> (f64, [f64; 3]) {
        let cols: usize = self.rows.iter().map(|r| r.0 + r.1).sum();
        (cols as f64 / (n[0] * n[1]) as f64, [0, 1, 2].map(|a| (self.proj[a].0 + self.proj[a].1) as f64 / n[a] as f64))
    }

    /// fraction of the grid's cells in the support (Σ of the column kz ranges)
    pub fn cell_fraction(&self, n: [usize; 3]) -> f64 {
        self.zcols.iter().map(|r| r.0 + r.1).sum::<usize>() as f64 / (n[0] * n[1] * n[2]) as f64
    }

    fn check(&self, n: [usize; 3]) -> Result<(), FftError> {
        let within = |r: (usize, usize), p: (usize, usize)| r.0 <= p.0 && r.1 <= p.1;
        let ok = self.rows.len() == n[1]
            && self.ycols.len() == n[0]
            && self.zcols.len() == n[0] * n[1]
            && (0..3).all(|a| self.proj[a].0 >= 1 && self.proj[a].0 + self.proj[a].1 <= n[a])
            && self.rows.iter().all(|&r| within(r, self.proj[0]))
            && self.ycols.iter().all(|&r| r.0 + r.1 <= n[1])
            && self.zcols.iter().all(|&r| within(r, self.proj[2]));
        if ok { Ok(()) } else { Err(FftError::Invalid(format!("k support does not fit the grid {n:?} (rows {}, proj {:?})", self.rows.len(), self.proj))) }
    }
}

pub struct PrunedChanFft3<R: Real> {
    /// `[n2, n1, n0]`
    pub n: [usize; 3],
    pub nch: usize,
    /// per axis `[2, 1, 0]`: (head, tail) of the wrapped block
    pub keep: [(usize, usize); 3],
    /// k-support pruning ([`new_ksupport`](Self::new_ksupport)); None: every k-space line
    pub ksup: Option<KSupport>,
    fwd: [AxisPlan<R>; 3],
    inv: [AxisPlan<R>; 3],
}

impl<R: Real> PrunedChanFft3<R> {
    /// `n = [n2, n1, n0]`, `keep[a] = (hi, lo)` for the same axes: the block is `[0, hi) ∪ [n − lo, n)`. `scale_inv`
    /// multiplies the inverse (1 = unnormalised, as the forward).
    pub fn new(dev: &Device, n: [usize; 3], nch: usize, keep: [(usize, usize); 3], scale_inv: f64) -> Result<Self, FftError> {
        Self::new_ksupport(dev, n, nch, keep, scale_inv, None)
    }

    /// As [`new`](Self::new), additionally pruned to the k-space support `ks` (module docs): the forward is exact on the
    /// support only, the inverse expects input that is zero outside it. `ks` is in `[x, y, z]` axis order (`n` is
    /// `[n2, n1, n0]`).
    pub fn new_ksupport(dev: &Device, n: [usize; 3], nch: usize, keep: [(usize, usize); 3], scale_inv: f64, ks: Option<&KSupport>) -> Result<Self, FftError> {
        if let Some(k) = ks {
            k.check([n[2], n[1], n[0]])?;
        }
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
        // ranged: 0 = all lines, 1 = per-group line ranges, 2 / 3 = also per-line entry ranges read / written
        let mk = |sp: AxisSpec, split: Option<LineSplit>, ranged: u8| -> Result<AxisPlan<R>, FftError> {
            let mut p = match ranged {
                0 => AxisPlan::from_spec(dev, &sp)?,
                1 => AxisPlan::from_spec_ranged(dev, &sp)?,
                _ => AxisPlan::from_spec_ranged_lines(dev, &sp, ranged == 2, ranged == 3)?,
            };
            if let Some(s) = split {
                p.split = s;
            }
            Ok(p)
        };
        let st = dev.default_stream();
        let (fwd, inv) = match ks {
            None => (
                [
                    mk(spec(n0, Forward, 1.0, addr(nch, cnt(1) * cnt(0), sx, sy), keep[2], full(2)), Some(split_x), 0)?,
                    mk(spec(n1, Forward, 1.0, addr(n0 * nch, cnt(0), sy, sz), keep[1], full(1)), Some(split_y), 0)?,
                    mk(spec(n2, Forward, 1.0, addr(n1 * n0 * nch, 1, sz, 0), keep[0], full(0)), None, 0)?,
                ],
                [
                    mk(spec(n2, Inverse, 1.0, addr(n1 * n0 * nch, 1, sz, 0), full(0), keep[0]), None, 0)?,
                    mk(spec(n1, Inverse, 1.0, addr(n0 * nch, cnt(0), sy, sz), full(1), keep[1]), Some(split_y), 0)?,
                    mk(spec(n0, Inverse, scale_inv, addr(nch, cnt(1) * cnt(0), sx, sy), full(2), keep[2]), Some(split_x), 0)?,
                ],
            ),
            Some(k) => {
                // x pass: lines (y, z) in the block, writing / reading only the support's kx range; y pass: lines
                // (channel, kx) of the kx range × z in the block, writing / reading the ky range of each kx
                // (`ycols`); z pass: per ky row, the (channel, kx) lines of the row's columns, writing / reading the kz
                // range of each column (`zcols`; rows of no column launch idle blocks). Lines (channel, kx) are
                // la = kx · nch + channel in both passes, so the line tables are indexed q = lb · stride + la / nch.
                let [px, py, pz] = k.proj;
                let chr = |(h, t): (usize, usize)| (h * nch, t * nch);
                let mut f = [
                    mk(spec(n0, Forward, 1.0, addr(nch, cnt(1) * cnt(0), sx, sy), keep[2], px), Some(split_x), 0)?,
                    mk(spec(n1, Forward, 1.0, addr(n0 * nch, cnt(0), sy, sz), keep[1], py), Some(split_y), 3)?,
                    mk(spec(n2, Forward, 1.0, addr(n0 * nch, n1, sz, sy), keep[0], pz), None, 3)?,
                ];
                let mut i = [
                    mk(spec(n2, Inverse, 1.0, addr(n0 * nch, n1, sz, sy), pz, keep[0]), None, 2)?,
                    mk(spec(n1, Inverse, 1.0, addr(n0 * nch, cnt(0), sy, sz), py, keep[1]), Some(split_y), 2)?,
                    mk(spec(n0, Inverse, scale_inv, addr(nch, cnt(1) * cnt(0), sx, sy), px, keep[2]), Some(split_x), 0)?,
                ];
                let ylines = vec![chr(px); cnt(0)];
                let zrows: Vec<(usize, usize)> = k.rows.iter().map(|&r| chr(r)).collect();
                f[1].set_arange(&st, &ylines)?;
                f[2].set_arange(&st, &zrows)?;
                i[0].set_arange(&st, &zrows)?;
                i[1].set_arange(&st, &ylines)?;
                f[1].set_lrange(&st, &k.ycols, nch, 0)?;
                i[1].set_lrange(&st, &k.ycols, nch, 0)?;
                f[2].set_lrange(&st, &k.zcols, nch, n0)?;
                i[0].set_lrange(&st, &k.zcols, nch, n0)?;
                (f, i)
            }
        };
        Ok(Self { n, nch, keep, ksup: ks.cloned(), fwd, inv })
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
