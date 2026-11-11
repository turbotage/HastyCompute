//! Operator abstractions for the solvers: one linear operator (`LinOp`, CG family) and a stack of sample sets on one
//! image grid (`SetOp`, P²nP), with implementations for the plain SENSE operator and the segmented physics NUFFT.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use hasty_cuda::C32;
use hasty_nufft::{SegFatMaps, SegNufft, SegPoints, SegSegments};

use crate::gpu::Kernels;
use crate::op::SenseOp;
use crate::Error;

pub trait LinOp {
    /// image / data lengths (complex elements)
    fn len_x(&self) -> usize;
    fn len_y(&self) -> usize;
    fn kernels(&self) -> &Kernels;
    fn forward(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error>;
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error>;
    /// out = A^H A x
    fn normal(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, out: &mut CudaSlice<C32>) -> Result<(), Error> {
        let mut y = st.alloc_zeros::<C32>(self.len_y())?;
        self.forward(st, x, &mut y)?;
        self.adjoint(st, &y, out)
    }
}

/// Encoding operators of a stack of sample sets sharing one image grid: set t maps an image of `len_x(t)` elements
/// (one frame, or all frames of a time-resolved set) to `len_y(t)` data (coil-major).
pub trait SetOp {
    fn nsets(&self) -> usize;
    fn len_x(&self, t: usize) -> usize;
    fn len_y(&self, t: usize) -> usize;
    fn kernels(&self) -> &Kernels;
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error>;
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error>;

    /// r = E_t x − y (the residual of set t, `len_y(t)` values). Default: `forward` then subtract; operators that can
    /// accumulate onto −y directly override it (`SegSets`: fused, no extra pass over the data).
    fn forward_residual(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &CudaSlice<C32>, r: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.forward(st, t, x, r)?;
        self.kernels().axpby(st, num_complex::Complex64::new(-1.0, 0.0), y, num_complex::Complex64::new(1.0, 0.0), r)
    }

    /// Several images on the same samples of set t: y = Σ_k w_k ⊙ E_t x_k, `ws` per-sample weights (one value per
    /// sample, the same for every coil; None = 1). Default: one `forward` per image plus the weights; operators that
    /// can share the gridding over the images override it (`SegSets`).
    fn forward_multi(&mut self, st: &Arc<CudaStream>, t: usize, xs: &[&CudaSlice<C32>], ws: &[Option<&CudaSlice<C32>>], y: &mut CudaSlice<C32>) -> Result<(), Error> {
        assert!(!xs.is_empty() && ws.len() == xs.len());
        let one = num_complex::Complex64::new(1.0, 0.0);
        let mut tmp: Option<CudaSlice<C32>> = None;
        for (k, (x, w)) in xs.iter().zip(ws).enumerate() {
            if k == 0 && w.is_none() {
                self.forward(st, t, x, y)?;
                continue;
            }
            if tmp.is_none() {
                tmp = Some(st.alloc_zeros::<C32>(y.len())?);
            }
            let z = tmp.as_mut().unwrap();
            self.forward(st, t, x, z)?;
            match w {
                Some(w) => self.kernels().sample_weight(st, z, w, y, w.len(), y.len() / w.len(), false, k > 0)?,
                None => self.kernels().axpby(st, one, z, one, y)?,
            }
        }
        Ok(())
    }

    /// Per-set sample tables of the operator that the memory planner can place (`plan::choose_tables`): (bytes of all
    /// sets' tables, bytes of the largest set). Default: none (SENSE coordinates and maps are not placeable).
    fn table_bytes(&self) -> (usize, usize) {
        (0, 0)
    }

    /// whether the tables currently live in pinned host memory (streamed into device slots per set)
    fn tables_on_host(&self) -> bool {
        false
    }

    /// Move the per-set sample tables to pinned host memory (`on_host`; each set's tables are uploaded into one of two
    /// device slots when used) or back to the device. Returns device bytes freed (negative: allocated). Results are
    /// identical either way. Default: nothing to move.
    fn place_tables(&mut self, _ctx: &Arc<CudaContext>, _st: &Arc<CudaStream>, _on_host: bool) -> Result<i64, Error> {
        Ok(0)
    }

    /// Restrict every set to the coils `idx` (stochastic coil sampling, `p2np::CoilSampling`): forward / adjoint then
    /// map to / from data with idx.len() coils (coil-major in idx order; `len_y` follows); None restores all coils.
    /// Returns false when the operator does not support coil subsets (default).
    fn select_coils(&mut self, _st: &Arc<CudaStream>, _idx: Option<&[usize]>) -> Result<bool, Error> {
        Ok(false)
    }

    /// Switch forward / adjoint to the operator's cheap surrogate Ẽ_t ≈ E_t on the same samples and coils (e.g. fewer
    /// time segments; used by the lagged gradient, `p2np::P2np::with_lagged`) or back (`on = false`). Returns false
    /// when the operator has no surrogate (default; nothing changes).
    fn use_surrogate(&mut self, _on: bool) -> bool {
        false
    }

    /// coils of the data (all of them, also while a subset is selected); 0 = unknown (default)
    fn n_coils(&self) -> usize {
        0
    }

    /// x_k = E_tᴴ(w̄_k ⊙ y) for every image k (conjugated per-sample weights; None = 1). Default: one `adjoint` per
    /// image.
    fn adjoint_multi(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, ws: &[Option<&CudaSlice<C32>>], xs: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        assert!(!xs.is_empty() && ws.len() == xs.len());
        let mut tmp: Option<CudaSlice<C32>> = None;
        for (x, w) in xs.iter_mut().zip(ws) {
            match w {
                None => self.adjoint(st, t, y, x)?,
                Some(w) => {
                    if tmp.is_none() {
                        tmp = Some(st.alloc_zeros::<C32>(y.len())?);
                    }
                    let z = tmp.as_mut().unwrap();
                    self.kernels().sample_weight(st, y, w, z, w.len(), y.len() / w.len(), true, false)?;
                    self.adjoint(st, t, z, x)?;
                }
            }
        }
        Ok(())
    }

    /// Sets sharing one image x (temporal pooling: all frames of an encode, or the frames of one group of a coarse
    /// temporal level): y_m = E_{t_m} x for every member t_m of `members` (x of the members' common `len_x`, y_m of
    /// `len_y(t_m)`). Default: one `forward` per member; operators that share work over the members override it
    /// (`SegSets`: one grid FFT per segment and coil batch for the whole group instead of one per member).
    fn forward_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, ys: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        assert_eq!(members.len(), ys.len(), "forward_group: one output per member");
        for (&t, y) in members.iter().zip(ys.iter_mut()) {
            self.forward(st, t, x, y)?;
        }
        Ok(())
    }

    /// r_m = E_{t_m} x − y_m for every member (`ys` the members' data). Default: `forward_residual` per member.
    fn forward_residual_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, ys: &[&CudaSlice<C32>], rs: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        assert!(members.len() == ys.len() && members.len() == rs.len(), "forward_residual_group: one data and one output per member");
        for ((&t, y), r) in members.iter().zip(ys).zip(rs.iter_mut()) {
            self.forward_residual(st, t, x, y, r)?;
        }
        Ok(())
    }

    /// x = Σ_m E_{t_m}ᴴ y_m (the adjoint of [`forward_group`](Self::forward_group)). Default: one `adjoint` per
    /// member, summed.
    fn adjoint_group(&mut self, st: &Arc<CudaStream>, members: &[usize], ys: &[&CudaSlice<C32>], x: &mut CudaSlice<C32>) -> Result<(), Error> {
        assert!(!members.is_empty() && members.len() == ys.len(), "adjoint_group: one input per member");
        self.adjoint(st, members[0], ys[0], x)?;
        if members.len() > 1 {
            let one = num_complex::Complex64::new(1.0, 0.0);
            let mut tmp = st.alloc_zeros::<C32>(x.len())?;
            for (&t, y) in members.iter().zip(ys).skip(1) {
                self.adjoint(st, t, y, &mut tmp)?;
                self.kernels().axpby(st, one, &tmp, one, x)?;
            }
        }
        Ok(())
    }
}

/// r_m −= y_m for every member (group residual from a group forward)
fn subtract_data(k: &Kernels, st: &Arc<CudaStream>, ys: &[&CudaSlice<C32>], rs: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
    assert_eq!(ys.len(), rs.len(), "one data per output");
    for (y, r) in ys.iter().zip(rs.iter_mut()) {
        k.axpby(st, num_complex::Complex64::new(-1.0, 0.0), y, num_complex::Complex64::new(1.0, 0.0), r)?;
    }
    Ok(())
}

/// the entries `idx` (ascending positions) of `ys`
fn pick_mut<'a>(ys: &'a mut [&mut CudaSlice<C32>], idx: &[usize]) -> Vec<&'a mut CudaSlice<C32>> {
    ys.iter_mut().enumerate().filter(|(j, _)| idx.binary_search(j).is_ok()).map(|(_, y)| &mut **y).collect()
}

/// One set of a stack as a [`LinOp`].
pub struct SetView<'a> {
    pub op: &'a mut dyn SetOp,
    pub t: usize,
}

impl LinOp for SetView<'_> {
    fn len_x(&self) -> usize {
        self.op.len_x(self.t)
    }
    fn len_y(&self) -> usize {
        self.op.len_y(self.t)
    }
    fn kernels(&self) -> &Kernels {
        self.op.kernels()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.op.forward(st, self.t, x, y)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.op.adjoint(st, self.t, y, x)
    }
}

impl LinOp for SenseOp {
    fn len_x(&self) -> usize {
        self.nv
    }
    fn len_y(&self) -> usize {
        self.nc * self.m
    }
    fn kernels(&self) -> &Kernels {
        SenseOp::kernels(self)
    }
    fn forward(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        SenseOp::forward(self, st, x, y)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        SenseOp::adjoint(self, st, y, x)
    }
    fn normal(&mut self, st: &Arc<CudaStream>, x: &CudaSlice<C32>, out: &mut CudaSlice<C32>) -> Result<(), Error> {
        SenseOp::normal(self, st, x, out)
    }
}

/// Plain SENSE sets: one `SenseOp` re-pointed to each set's coordinates, optional per-set coil maps (e.g. maps carrying
/// a per-encode phase).
pub struct SenseSets<'a> {
    pub op: &'a mut SenseOp,
    pub coords: Vec<[CudaSlice<f32>; 3]>,
    pub ms: Vec<usize>,
    pub maps: Vec<CudaSlice<C32>>,
    pub map_of_set: Vec<usize>,
}

impl SenseSets<'_> {
    fn with<R>(&mut self, st: &Arc<CudaStream>, t: usize, f: impl FnOnce(&mut SenseOp) -> Result<R, Error>) -> Result<R, Error> {
        let c = &self.coords[t];
        self.op.set_points(st, [&c[0], &c[1], &c[2]], self.ms[t])?;
        let mi = self.map_of_set.get(t).copied();
        if let Some(i) = mi {
            self.op.swap_smaps(&mut self.maps[i]);
        }
        let r = f(self.op);
        if let Some(i) = mi {
            self.op.swap_smaps(&mut self.maps[i]);
        }
        r
    }
}

impl SetOp for SenseSets<'_> {
    fn nsets(&self) -> usize {
        self.ms.len()
    }
    fn len_x(&self, _t: usize) -> usize {
        self.op.nv
    }
    fn len_y(&self, t: usize) -> usize {
        self.op.nc * self.ms[t]
    }
    fn kernels(&self) -> &Kernels {
        self.op.kernels()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.with(st, t, |op| op.forward(st, x, y))
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        self.with(st, t, |op| op.adjoint(st, y, x))
    }
}

/// Segmented physics NUFFT sets: one engine (grid slots, FFT plans, kernels, coil maps), per-set points, segment tables
/// shared by the sets of one encode.
pub struct SegSets {
    pub eng: SegNufft,
    pub pts: Vec<SegPoints>,
    pub segs: Vec<SegSegments>,
    pub seg_of_set: Vec<usize>,
    pub kern: Arc<Kernels>,
    /// optional cheap surrogate on the same sample tables ([`SegSets::with_surrogate`]); None = none
    pub surrogate: Option<SegSurrogate>,
}

/// A second engine on the same samples (e.g. fewer time segments) with its own segment tables, one per encode as
/// [`SegSets::segs`]. While `on`, the [`SegSets`] forward / adjoint run through it ([`SetOp::use_surrogate`]).
pub struct SegSurrogate {
    pub eng: SegNufft,
    pub segs: Vec<SegSegments>,
    pub on: bool,
}

impl SetOp for SegSets {
    fn nsets(&self) -> usize {
        self.pts.len()
    }
    fn len_x(&self, _t: usize) -> usize {
        self.eng.nv() * self.eng.frames
    }
    fn len_y(&self, t: usize) -> usize {
        self.eng.nc * self.pts[t].m
    }
    fn kernels(&self) -> &Kernels {
        &self.kern
    }
    fn select_coils(&mut self, st: &Arc<CudaStream>, idx: Option<&[usize]>) -> Result<bool, Error> {
        self.eng.select_coils(st, idx)?;
        if let Some(s) = &mut self.surrogate {
            s.eng.select_coils(st, idx)?;
        }
        Ok(true)
    }
    fn use_surrogate(&mut self, on: bool) -> bool {
        match &mut self.surrogate {
            Some(s) => {
                s.on = on;
                true
            }
            None => false,
        }
    }
    fn n_coils(&self) -> usize {
        self.eng.nc_all()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let i = self.seg_of_set[t];
        match &mut self.surrogate {
            Some(s) if s.on => s.eng.forward(st, &s.segs[i], &self.pts[t], x, y)?,
            _ => self.eng.forward(st, &self.segs[i], &self.pts[t], x, y)?,
        }
        // streamed tables: upload set t + 1 while this set's adjoint runs (no-op when resident / no prefetch)
        self.prefetch_next(st, t)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let i = self.seg_of_set[t];
        match &mut self.surrogate {
            Some(s) if s.on => Ok(s.eng.adjoint(st, &s.segs[i], &self.pts[t], y, x)?),
            _ => Ok(self.eng.adjoint(st, &self.segs[i], &self.pts[t], y, x)?),
        }
    }
    /// fused residual (`SegNufft::forward_residual`): the gather accumulates onto −y
    fn forward_residual(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &CudaSlice<C32>, r: &mut CudaSlice<C32>) -> Result<(), Error> {
        let i = self.seg_of_set[t];
        match &mut self.surrogate {
            Some(s) if s.on => s.eng.forward_residual(st, &s.segs[i], &self.pts[t], x, y, r)?,
            _ => self.eng.forward_residual(st, &self.segs[i], &self.pts[t], x, y, r)?,
        }
        self.prefetch_next(st, t)
    }
    /// shared gridding over the images (`SegNufft::forward_multi`)
    fn forward_multi(&mut self, st: &Arc<CudaStream>, t: usize, xs: &[&CudaSlice<C32>], ws: &[Option<&CudaSlice<C32>>], y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let i = self.seg_of_set[t];
        match &mut self.surrogate {
            Some(s) if s.on => s.eng.forward_multi(st, &s.segs[i], &self.pts[t], xs, ws, y)?,
            _ => self.eng.forward_multi(st, &self.segs[i], &self.pts[t], xs, ws, y)?,
        }
        self.prefetch_next(st, t)
    }
    fn adjoint_multi(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, ws: &[Option<&CudaSlice<C32>>], xs: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        let i = self.seg_of_set[t];
        match &mut self.surrogate {
            Some(s) if s.on => Ok(s.eng.adjoint_multi(st, &s.segs[i], &self.pts[t], y, ws, xs)?),
            _ => Ok(self.eng.adjoint_multi(st, &self.segs[i], &self.pts[t], y, ws, xs)?),
        }
    }
    /// one grid FFT per (segment, coil batch) for all members of a segment table (`SegNufft::forward_group`);
    /// members with different tables (encodes) form separate sub-groups
    fn forward_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, ys: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        assert_eq!(members.len(), ys.len(), "forward_group: one output per member");
        for (si, idx) in self.by_table(members) {
            let pts: Vec<&SegPoints> = idx.iter().map(|&j| &self.pts[members[j]]).collect();
            let mut out = pick_mut(ys, &idx);
            match &mut self.surrogate {
                Some(s) if s.on => s.eng.forward_group(st, &s.segs[si], &pts, x, &mut out)?,
                _ => self.eng.forward_group(st, &self.segs[si], &pts, x, &mut out)?,
            }
        }
        Ok(())
    }
    /// fused group residual (`SegNufft::forward_residual_group`)
    fn forward_residual_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, ys: &[&CudaSlice<C32>], rs: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        assert!(members.len() == ys.len() && members.len() == rs.len(), "forward_residual_group: one data and one output per member");
        for (si, idx) in self.by_table(members) {
            let pts: Vec<&SegPoints> = idx.iter().map(|&j| &self.pts[members[j]]).collect();
            let d: Vec<&CudaSlice<C32>> = idx.iter().map(|&j| ys[j]).collect();
            let mut out = pick_mut(rs, &idx);
            match &mut self.surrogate {
                Some(s) if s.on => s.eng.forward_residual_group(st, &s.segs[si], &pts, x, &d, &mut out)?,
                _ => self.eng.forward_residual_group(st, &self.segs[si], &pts, x, &d, &mut out)?,
            }
        }
        Ok(())
    }
    /// one inverse grid FFT per (segment, coil batch) for all members of a segment table (`SegNufft::adjoint_group`)
    fn adjoint_group(&mut self, st: &Arc<CudaStream>, members: &[usize], ys: &[&CudaSlice<C32>], x: &mut CudaSlice<C32>) -> Result<(), Error> {
        assert!(!members.is_empty() && members.len() == ys.len(), "adjoint_group: one input per member");
        let mut tmp: Option<CudaSlice<C32>> = None;
        for (k, (si, idx)) in self.by_table(members).into_iter().enumerate() {
            let pts: Vec<&SegPoints> = idx.iter().map(|&j| &self.pts[members[j]]).collect();
            let d: Vec<&CudaSlice<C32>> = idx.iter().map(|&j| ys[j]).collect();
            if k > 0 && tmp.is_none() {
                tmp = Some(st.alloc_zeros::<C32>(x.len())?);
            }
            let out: &mut CudaSlice<C32> = if k == 0 { &mut *x } else { tmp.as_mut().unwrap() };
            match &mut self.surrogate {
                Some(s) if s.on => s.eng.adjoint_group(st, &s.segs[si], &pts, &d, out)?,
                _ => self.eng.adjoint_group(st, &self.segs[si], &pts, &d, out)?,
            }
            if k > 0 {
                let one = num_complex::Complex64::new(1.0, 0.0);
                self.kern.axpby(st, one, tmp.as_ref().unwrap(), one, x)?;
            }
        }
        Ok(())
    }
    fn table_bytes(&self) -> (usize, usize) {
        (self.pts.iter().map(|p| p.table_bytes()).sum(), self.pts.iter().map(|p| p.table_bytes()).max().unwrap_or(0))
    }
    fn tables_on_host(&self) -> bool {
        self.pts.iter().any(|p| p.is_host())
    }
    fn place_tables(&mut self, ctx: &Arc<CudaContext>, st: &Arc<CudaStream>, on_host: bool) -> Result<i64, Error> {
        let mut freed = 0i64;
        for p in self.pts.iter_mut() {
            freed += if on_host { p.to_host(ctx, st)? as i64 } else { -(p.to_device(st)? as i64) };
        }
        let mut engs: Vec<&mut SegNufft> = vec![&mut self.eng];
        if let Some(s) = &mut self.surrogate {
            engs.push(&mut s.eng);
        }
        for eng in engs {
            if on_host {
                let refs: Vec<&SegPoints> = self.pts.iter().collect();
                eng.stream_reserve(st, &refs)?;
                freed -= eng.stream_slot_bytes() as i64;
            } else {
                freed += eng.stream_slot_bytes() as i64;
                eng.stream_release(st)?;
            }
        }
        Ok(freed)
    }
}

impl SegSets {
    /// positions in `members` by segment table (tables in first-appearance order, positions ascending): the
    /// sub-groups that can share grid transforms
    fn by_table(&self, members: &[usize]) -> Vec<(usize, Vec<usize>)> {
        let mut v: Vec<(usize, Vec<usize>)> = Vec::new();
        for (j, &t) in members.iter().enumerate() {
            let s = self.seg_of_set[t];
            match v.iter_mut().find(|(k, _)| *k == s) {
                Some((_, p)) => p.push(j),
                None => v.push((s, vec![j])),
            }
        }
        v
    }

    /// with `stream_prefetch` and host-resident tables: upload set t + 1 into the active engine's slots while the
    /// caller goes on
    fn prefetch_next(&mut self, st: &Arc<CudaStream>, t: usize) -> Result<(), Error> {
        let n = self.pts.len();
        let eng = match &mut self.surrogate {
            Some(s) if s.on => &mut s.eng,
            _ => &mut self.eng,
        };
        if eng.stream_prefetch && n > 1 && self.pts[(t + 1) % n].is_host() {
            eng.prefetch(st, &self.pts[(t + 1) % n])?;
        }
        Ok(())
    }

    /// Attach a cheap surrogate engine on the same samples: `eng` must have the same grid, width, correction order,
    /// frames, coils and tile edge / item length as the main engine (only the segment count may differ), so every
    /// set's tables serve both; `segs` holds one segment table per encode, as [`SegSets::segs`] (built for `eng`'s
    /// segment count). Coil maps must already be set on `eng`. The main operator is unchanged until
    /// [`SetOp::use_surrogate`]`(true)`.
    pub fn with_surrogate(mut self, mut eng: SegNufft, segs: Vec<SegSegments>) -> Self {
        let m = &self.eng;
        assert_eq!(eng.plan.nf, m.plan.nf, "surrogate: fine grid differs");
        assert_eq!(eng.plan.ker.w, m.plan.ker.w, "surrogate: kernel width differs");
        assert_eq!((eng.order, eng.frames, eng.nf, eng.nc_all()), (m.order, m.frames, m.nf, m.nc_all()), "surrogate: order / frames / coils differ");
        assert_eq!(eng.res, m.res, "surrogate: voxel size differs");
        assert_eq!((eng.tiles_fwd.tt, eng.tiles_fwd.max_item), (m.tiles_fwd.tt, m.tiles_fwd.max_item), "surrogate: tile lists differ");
        assert_eq!(segs.len(), self.segs.len(), "surrogate: one segment table per encode");
        eng.stream_prefetch = m.stream_prefetch;
        self.surrogate = Some(SegSurrogate { eng, segs, on: false });
        self
    }
}

/// Per-sample weights around any stack of sets: E'_t = diag(w_t) E_t. Used for the voxel basis function: a rect
/// voxel of size Δ contributes Π_a sinc(k_a Δ_a / 2) at sample k (a cubic B-spline basis: the 4th power), which the
/// plain operators (Dirac voxels) omit (research/velocity_model_sr.md §B1).
pub struct WeightedSets<'a> {
    pub inner: &'a mut dyn SetOp,
    /// per set, one complex weight per sample
    pub w: Vec<CudaSlice<C32>>,
}

impl<'a> WeightedSets<'a> {
    pub fn new(st: &Arc<CudaStream>, inner: &'a mut dyn SetOp, w: Vec<Vec<C32>>) -> Result<Self, Error> {
        assert_eq!(w.len(), inner.nsets());
        let w = w.iter().map(|v| st.clone_htod(v)).collect::<Result<_, _>>()?;
        Ok(Self { inner, w })
    }

    /// voxel-basis weights for one set's coordinates (k·Δ, radians per voxel): Π_a sinc(c_a / 2)^power
    pub fn basis_weights(coords: [&[f32]; 3], power: i32) -> Vec<C32> {
        let sinc = |x: f64| if x.abs() < 1e-8 { 1.0 } else { x.sin() / x };
        (0..coords[0].len()).map(|n| C32::new((0..3).map(|a| sinc(coords[a][n] as f64 / 2.0)).product::<f64>().powi(power) as f32, 0.0)).collect()
    }
}

impl SetOp for WeightedSets<'_> {
    fn nsets(&self) -> usize {
        self.inner.nsets()
    }
    fn len_x(&self, t: usize) -> usize {
        self.inner.len_x(t)
    }
    fn len_y(&self, t: usize) -> usize {
        self.inner.len_y(t)
    }
    fn kernels(&self) -> &Kernels {
        self.inner.kernels()
    }
    fn select_coils(&mut self, st: &Arc<CudaStream>, idx: Option<&[usize]>) -> Result<bool, Error> {
        self.inner.select_coils(st, idx)
    }
    fn use_surrogate(&mut self, on: bool) -> bool {
        self.inner.use_surrogate(on)
    }
    fn n_coils(&self) -> usize {
        self.inner.n_coils()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        // per-call scratch (stream-ordered allocation): no second copy of all data
        let mut tmp = st.alloc_zeros::<C32>(y.len())?;
        self.inner.forward(st, t, x, &mut tmp)?;
        let m = self.w[t].len();
        let nc = y.len() / m;
        self.inner.kernels().sample_weight(st, &tmp, &self.w[t], y, m, nc, false, false)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let m = self.w[t].len();
        let nc = y.len() / m;
        let mut tmp = st.alloc_zeros::<C32>(y.len())?;
        self.inner.kernels().sample_weight(st, y, &self.w[t], &mut tmp, m, nc, true, false)?;
        self.inner.adjoint(st, t, &tmp, x)
    }
    /// diag(w) Σ_k w_k ⊙ E x_k through the inner operator's multi path
    fn forward_multi(&mut self, st: &Arc<CudaStream>, t: usize, xs: &[&CudaSlice<C32>], ws: &[Option<&CudaSlice<C32>>], y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let mut tmp = st.alloc_zeros::<C32>(y.len())?;
        self.inner.forward_multi(st, t, xs, ws, &mut tmp)?;
        let m = self.w[t].len();
        let nc = y.len() / m;
        self.inner.kernels().sample_weight(st, &tmp, &self.w[t], y, m, nc, false, false)
    }
    fn adjoint_multi(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, ws: &[Option<&CudaSlice<C32>>], xs: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        let m = self.w[t].len();
        let nc = y.len() / m;
        let mut tmp = st.alloc_zeros::<C32>(y.len())?;
        self.inner.kernels().sample_weight(st, y, &self.w[t], &mut tmp, m, nc, true, false)?;
        self.inner.adjoint_multi(st, t, &tmp, ws, xs)
    }
    /// the inner operator's group forward, then each member's weights
    fn forward_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, ys: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        self.inner.forward_group(st, members, x, ys)?;
        for (&t, y) in members.iter().zip(ys.iter_mut()) {
            let m = self.w[t].len();
            let mut tmp = unsafe { st.alloc::<C32>(y.len())? };
            st.memcpy_dtod(&**y, &mut tmp)?;
            self.inner.kernels().sample_weight(st, &tmp, &self.w[t], y, m, y.len() / m, false, false)?;
        }
        Ok(())
    }
    fn forward_residual_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, ys: &[&CudaSlice<C32>], rs: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        self.forward_group(st, members, x, rs)?;
        subtract_data(self.inner.kernels(), st, ys, rs)
    }
    /// each member's conjugated weights, then the inner operator's group adjoint
    fn adjoint_group(&mut self, st: &Arc<CudaStream>, members: &[usize], ys: &[&CudaSlice<C32>], x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let mut wy = Vec::with_capacity(members.len());
        for (&t, y) in members.iter().zip(ys) {
            let m = self.w[t].len();
            let mut z = unsafe { st.alloc::<C32>(y.len())? };
            self.inner.kernels().sample_weight(st, y, &self.w[t], &mut z, m, y.len() / m, true, false)?;
            wy.push(z);
        }
        let refs: Vec<&CudaSlice<C32>> = wy.iter().collect();
        self.inner.adjoint_group(st, members, &refs, x)
    }
    fn table_bytes(&self) -> (usize, usize) {
        self.inner.table_bytes()
    }
    fn tables_on_host(&self) -> bool {
        self.inner.tables_on_host()
    }
    fn place_tables(&mut self, ctx: &Arc<CudaContext>, st: &Arc<CudaStream>, on_host: bool) -> Result<i64, Error> {
        self.inner.place_tables(ctx, st, on_host)
    }
}

/// Water and fat with a known fat fraction around any stack of sets: one complex image x per set (both species share
/// its phase and velocity encoding — they move together in a voxel), the fat part ff·x additionally carries the
/// per-sample fat spectrum c(τ) = Σ_p α_p e^{−i ω_p τ} (τ = time since excitation, the simulator's convention):
///   E'_t x = E_t((1 − ff) x) + c_t ⊙ E_t(ff x),   E'_tᴴ r = (1 − ff) E_tᴴ r + ff E_tᴴ(c̄_t ⊙ r).
/// The inner operator models B0 (and the concomitant fields) only, so its time segments cover the B0 range, not the
/// B0 + chemical-shift range; mixed water / fat voxels and the multi-peak spectrum are exact. ff (signal fat fraction,
/// e.g. from the B0 / R2* / fat scan, in [0, 1]) per voxel of the image grid; a set with several frames repeats it.
/// Cost: the inner operator's FFTs for two images per forward / adjoint. By default both species go through the inner
/// operator's multi-image path (`SetOp::forward_multi`: one gridding pass with weights 1 and c for `SegSets`, two plain
/// applications otherwise) — identical to two applications in exact arithmetic; `shared_gridding = false` keeps the
/// two separate applications.
pub struct FatSets<'a> {
    pub inner: &'a mut dyn SetOp,
    /// fat-fraction maps as (1 − ff, ff) real parts, each of its sets' image length (one frame, or the frames of a
    /// time-resolved set concatenated), and the map of each set
    maps: Vec<(CudaSlice<C32>, CudaSlice<C32>)>,
    map_of_set: Vec<usize>,
    /// per set, the fat spectrum per sample
    pub c: Vec<CudaSlice<C32>>,
    /// both species in one multi-image call of the inner operator (default true); false: two inner applications
    pub shared_gridding: bool,
    /// group maps of [`SetOp::forward_group`] for members with different maps: the members' map indices (sorted) and
    /// the mean (1 − ff, ff) pair
    group_maps: Vec<(Vec<usize>, MapPair)>,
}

/// a (1 − ff, ff) map pair of [`FatSets`]
type MapPair = (CudaSlice<C32>, CudaSlice<C32>);

impl<'a> FatSets<'a> {
    /// one fat-fraction map per voxel for every set (repeated over the frames of a time-resolved set); `c` per set
    /// and sample (see [`fat_spectrum_weights`])
    pub fn new(st: &Arc<CudaStream>, inner: &'a mut dyn SetOp, ff: &[f32], c: Vec<Vec<C32>>) -> Result<Self, Error> {
        let ns = inner.nsets();
        let lens: Vec<usize> = (0..ns).map(|t| inner.len_x(t)).collect();
        let mut uniq: Vec<usize> = lens.clone();
        uniq.sort_unstable();
        uniq.dedup();
        let maps_h: Vec<Vec<f32>> = uniq.iter().map(|&l| {
            assert_eq!(l % ff.len(), 0, "image of {l} elements is not a whole number of {}-voxel frames", ff.len());
            (0..l / ff.len()).flat_map(|_| ff.iter().cloned()).collect()
        }).collect();
        let map_of_set = lens.iter().map(|l| uniq.iter().position(|u| u == l).unwrap()).collect();
        Self::with_maps(st, inner, &maps_h, map_of_set, c)
    }

    /// per-set fat-fraction maps: `maps[k]` has the image length of the sets with `map_of_set[t] == k` (e.g. one map
    /// per cardiac frame, the fat moving with the heart)
    pub fn with_maps(st: &Arc<CudaStream>, inner: &'a mut dyn SetOp, maps: &[Vec<f32>], map_of_set: Vec<usize>, c: Vec<Vec<C32>>) -> Result<Self, Error> {
        assert_eq!(c.len(), inner.nsets());
        assert_eq!(map_of_set.len(), inner.nsets());
        for (t, &k) in map_of_set.iter().enumerate() {
            assert_eq!(maps[k].len(), inner.len_x(t), "fat map {k} has {} values, set {t} images {}", maps[k].len(), inner.len_x(t));
        }
        let maps = maps
            .iter()
            .map(|ff| -> Result<_, Error> {
                let ww: Vec<C32> = ff.iter().map(|&f| C32::new(1.0 - f.clamp(0.0, 1.0), 0.0)).collect();
                let wf: Vec<C32> = ff.iter().map(|&f| C32::new(f.clamp(0.0, 1.0), 0.0)).collect();
                Ok((st.clone_htod(&ww)?, st.clone_htod(&wf)?))
            })
            .collect::<Result<_, _>>()?;
        let c = c.iter().map(|v| st.clone_htod(v)).collect::<Result<_, _>>()?;
        Ok(Self { inner, maps, map_of_set, c, shared_gridding: true, group_maps: Vec::new() })
    }

    /// the same operator with the two species applied separately (the path before the multi-image operators)
    pub fn separate(mut self) -> Self {
        self.shared_gridding = false;
        self
    }

    /// The (1 − ff, ff) pair of a group: (false, i) = map i shared by all members, (true, i) = entry i of
    /// `group_maps`, the members' mean (built on first use; the approximation of a pooled level whose members' fat
    /// fractions differ, as `SegNufft::forward_fat_group`)
    fn group_pair(&mut self, st: &Arc<CudaStream>, members: &[usize]) -> Result<(bool, usize), Error> {
        let mut key: Vec<usize> = members.iter().map(|&t| self.map_of_set[t]).collect();
        if key.iter().all(|&k| k == key[0]) {
            return Ok((false, key[0]));
        }
        key.sort_unstable();
        if let Some(i) = self.group_maps.iter().position(|(k, _)| *k == key) {
            return Ok((true, i));
        }
        let n = self.maps[key[0]].0.len();
        assert!(key.iter().all(|&k| self.maps[k].0.len() == n), "fat group: members' maps of different lengths");
        let (mut ww, mut wf) = (st.alloc_zeros::<C32>(n)?, st.alloc_zeros::<C32>(n)?);
        let (w, one) = (num_complex::Complex64::new(1.0 / key.len() as f64, 0.0), num_complex::Complex64::new(1.0, 0.0));
        for &k in &key {
            self.inner.kernels().axpby(st, w, &self.maps[k].0, one, &mut ww)?;
            self.inner.kernels().axpby(st, w, &self.maps[k].1, one, &mut wf)?;
        }
        if self.group_maps.len() >= 64 {
            self.group_maps.remove(0);
        }
        self.group_maps.push((key, (ww, wf)));
        Ok((true, self.group_maps.len() - 1))
    }
}

/// Water and fat with a known fat fraction as ONE segmented image (`recon_p2np --fat-model joint` with per-frame maps):
/// the segment diagonals carry each voxel's water / fat mix, P_l (1 + ff (c_l − 1)) with c_l the fat spectrum at the
/// segment node (`hasty_nufft::build_segments_fat` → `SegNufft::segments_fat`), so a set costs one plain operator with
/// L segments instead of FatSets' two images (FFT channels L instead of 2L; L about 1.5× the B0-only count, see
/// hasty-nufft tests/seg_b0_direct.rs `fat_species_error_budget_cpu`). The interpolator table does not depend on ff,
/// so one table per encode serves every map: one map per set (static (frame, encode) sets) or one map per frame of a
/// time-resolved set (`map_of_set[t] = None`). A single map can instead be folded into the diagonals
/// (`hasty_nufft::fold_fat_fraction`, `build_segments_complex`) and run through plain `SegSets`.
pub struct FatJointSets {
    /// the segmented operator; `segs` from `SegNufft::segments_fat`
    pub sets: SegSets,
    pub maps: SegFatMaps,
    /// per set: Some(i) = map i for every frame, None = map f for frame f
    pub map_of_set: Vec<Option<usize>>,
}

impl SetOp for FatJointSets {
    fn nsets(&self) -> usize {
        self.sets.nsets()
    }
    fn len_x(&self, t: usize) -> usize {
        self.sets.len_x(t)
    }
    fn len_y(&self, t: usize) -> usize {
        self.sets.len_y(t)
    }
    fn kernels(&self) -> &Kernels {
        &self.sets.kern
    }
    fn select_coils(&mut self, st: &Arc<CudaStream>, idx: Option<&[usize]>) -> Result<bool, Error> {
        self.sets.select_coils(st, idx)
    }
    fn n_coils(&self) -> usize {
        self.sets.n_coils()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let s = &mut self.sets;
        s.eng.forward_fat(st, &s.segs[s.seg_of_set[t]], &s.pts[t], &self.maps, self.map_of_set[t], x, y)?;
        s.prefetch_next(st, t)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let s = &mut self.sets;
        Ok(s.eng.adjoint_fat(st, &s.segs[s.seg_of_set[t]], &s.pts[t], &self.maps, self.map_of_set[t], y, x)?)
    }
    /// one grid FFT per (segment, coil batch) per segment table and ONE fat map per group: the members' map if they
    /// share one, else their voxel-wise mean (`SegFatMaps::mean`) — exact for a shared map, an approximation of the
    /// member-wise operator where the pooled frames' fat fractions differ
    fn forward_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, ys: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        self.fat_forward_group(st, members, x, None, ys)
    }
    fn forward_residual_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, ys: &[&CudaSlice<C32>], rs: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        self.fat_forward_group(st, members, x, Some(ys), rs)
    }
    /// the adjoint of [`forward_group`](SetOp::forward_group) (same group map)
    fn adjoint_group(&mut self, st: &Arc<CudaStream>, members: &[usize], ys: &[&CudaSlice<C32>], x: &mut CudaSlice<C32>) -> Result<(), Error> {
        assert!(!members.is_empty() && members.len() == ys.len(), "adjoint_group: one input per member");
        let s = &mut self.sets;
        let mut tmp: Option<CudaSlice<C32>> = None;
        for (k, (si, idx)) in s.by_table(members).into_iter().enumerate() {
            let pts: Vec<&SegPoints> = idx.iter().map(|&j| &s.pts[members[j]]).collect();
            let of: Vec<Option<usize>> = idx.iter().map(|&j| self.map_of_set[members[j]]).collect();
            let d: Vec<&CudaSlice<C32>> = idx.iter().map(|&j| ys[j]).collect();
            let (maps, map) = group_fat_map(&mut self.maps, st, &of)?;
            if k > 0 && tmp.is_none() {
                tmp = Some(st.alloc_zeros::<C32>(x.len())?);
            }
            let out: &mut CudaSlice<C32> = if k == 0 { &mut *x } else { tmp.as_mut().unwrap() };
            s.eng.adjoint_fat_group(st, &s.segs[si], &pts, maps, map, &d, out)?;
            if k > 0 {
                let one = num_complex::Complex64::new(1.0, 0.0);
                s.kern.axpby(st, one, tmp.as_ref().unwrap(), one, x)?;
            }
        }
        Ok(())
    }
    fn table_bytes(&self) -> (usize, usize) {
        self.sets.table_bytes()
    }
    fn tables_on_host(&self) -> bool {
        self.sets.tables_on_host()
    }
    fn place_tables(&mut self, ctx: &Arc<CudaContext>, st: &Arc<CudaStream>, on_host: bool) -> Result<i64, Error> {
        self.sets.place_tables(ctx, st, on_host)
    }
}

impl FatJointSets {
    /// group forward (data: residuals) per segment table with one fat map per sub-group
    fn fat_forward_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, data: Option<&[&CudaSlice<C32>]>, ys: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        assert!(members.len() == ys.len() && data.is_none_or(|d| d.len() == members.len()), "forward_group: one output (and data) per member");
        let s = &mut self.sets;
        for (si, idx) in s.by_table(members) {
            let pts: Vec<&SegPoints> = idx.iter().map(|&j| &s.pts[members[j]]).collect();
            let of: Vec<Option<usize>> = idx.iter().map(|&j| self.map_of_set[members[j]]).collect();
            let d: Option<Vec<&CudaSlice<C32>>> = data.map(|d| idx.iter().map(|&j| d[j]).collect());
            let (maps, map) = group_fat_map(&mut self.maps, st, &of)?;
            let mut out = pick_mut(ys, &idx);
            s.eng.forward_fat_group(st, &s.segs[si], &pts, maps, map, x, d.as_deref(), &mut out)?;
        }
        Ok(())
    }
}

/// The fat map of a group from its members' maps (`map_of_set` entries): all per-frame (None) → the per-frame maps;
/// one shared map → that map; different maps → their voxel-wise mean (cached in `maps`, one-map form, index 0).
fn group_fat_map<'m>(maps: &'m mut SegFatMaps, st: &Arc<CudaStream>, of: &[Option<usize>]) -> Result<(&'m SegFatMaps, Option<usize>), Error> {
    if of.iter().all(|m| m.is_none()) {
        return Ok((&*maps, None));
    }
    assert!(of.iter().all(|m| m.is_some()), "fat group: members with per-frame and with fixed maps mixed");
    let ids: Vec<usize> = of.iter().map(|m| m.unwrap()).collect();
    if ids.iter().all(|&i| i == ids[0]) {
        return Ok((&*maps, Some(ids[0])));
    }
    Ok((maps.mean(st, &ids)?, Some(0)))
}

/// c(τ) per sample of a set: the fat spectrum relative to water at field `b0` (T), τ = time since excitation (s)
pub fn fat_spectrum_weights(ppm: &[f64], amp: &[f64], b0: f64, tau: &[f64]) -> Vec<C32> {
    let gbar = 42.577_478_51e6;
    let om: Vec<f64> = ppm.iter().map(|d| 2.0 * std::f64::consts::PI * gbar * b0 * d * 1e-6).collect();
    tau.iter()
        .map(|&t| {
            let (mut re, mut im) = (0.0, 0.0);
            for (w, a) in om.iter().zip(amp) {
                re += a * (w * t).cos();
                im -= a * (w * t).sin();
            }
            C32::new(re as f32, im as f32)
        })
        .collect()
}

impl SetOp for FatSets<'_> {
    fn table_bytes(&self) -> (usize, usize) {
        self.inner.table_bytes()
    }
    fn tables_on_host(&self) -> bool {
        self.inner.tables_on_host()
    }
    fn place_tables(&mut self, ctx: &Arc<CudaContext>, st: &Arc<CudaStream>, on_host: bool) -> Result<i64, Error> {
        self.inner.place_tables(ctx, st, on_host)
    }
    fn nsets(&self) -> usize {
        self.inner.nsets()
    }
    fn len_x(&self, t: usize) -> usize {
        self.inner.len_x(t)
    }
    fn len_y(&self, t: usize) -> usize {
        self.inner.len_y(t)
    }
    fn kernels(&self) -> &Kernels {
        self.inner.kernels()
    }
    fn select_coils(&mut self, st: &Arc<CudaStream>, idx: Option<&[usize]>) -> Result<bool, Error> {
        self.inner.select_coils(st, idx)
    }
    fn use_surrogate(&mut self, on: bool) -> bool {
        self.inner.use_surrogate(on)
    }
    fn n_coils(&self) -> usize {
        self.inner.n_coils()
    }
    fn forward(&mut self, st: &Arc<CudaStream>, t: usize, x: &CudaSlice<C32>, y: &mut CudaSlice<C32>) -> Result<(), Error> {
        let (ww, wf) = (&self.maps[self.map_of_set[t]].0, &self.maps[self.map_of_set[t]].1);
        if self.shared_gridding {
            // y = E((1 − ff) x) + c ⊙ E(ff x), one multi-image call
            let mut xw = st.alloc_zeros::<C32>(x.len())?;
            let mut xf = st.alloc_zeros::<C32>(x.len())?;
            self.inner.kernels().mul_real(st, x, ww, &mut xw)?;
            self.inner.kernels().mul_real(st, x, wf, &mut xf)?;
            return self.inner.forward_multi(st, t, &[&xw, &xf], &[None, Some(&self.c[t])], y);
        }
        let mut xs = st.alloc_zeros::<C32>(x.len())?;
        // water part straight into y
        self.inner.kernels().mul_real(st, x, ww, &mut xs)?;
        self.inner.forward(st, t, &xs, y)?;
        // fat part: c ⊙ E(ff x), accumulated
        self.inner.kernels().mul_real(st, x, wf, &mut xs)?;
        let mut yf = st.alloc_zeros::<C32>(y.len())?;
        self.inner.forward(st, t, &xs, &mut yf)?;
        let m = self.c[t].len();
        self.inner.kernels().sample_weight(st, &yf, &self.c[t], y, m, y.len() / m, false, true)
    }
    fn adjoint(&mut self, st: &Arc<CudaStream>, t: usize, y: &CudaSlice<C32>, x: &mut CudaSlice<C32>) -> Result<(), Error> {
        let (ww, wf) = (&self.maps[self.map_of_set[t]].0, &self.maps[self.map_of_set[t]].1);
        if self.shared_gridding {
            // x = (1 − ff) Eᴴ y + ff Eᴴ(c̄ ⊙ y), one multi-image call
            let mut xw = st.alloc_zeros::<C32>(x.len())?;
            let mut xf = st.alloc_zeros::<C32>(x.len())?;
            self.inner.adjoint_multi(st, t, y, &[None, Some(&self.c[t])], &mut [&mut xw, &mut xf])?;
            self.inner.kernels().mul_real(st, &xw, ww, x)?;
            self.inner.kernels().mul_real(st, &xf, wf, &mut xw)?;
            let one = num_complex::Complex64::new(1.0, 0.0);
            return self.inner.kernels().axpby(st, one, &xw, one, x);
        }
        let mut xs = st.alloc_zeros::<C32>(x.len())?;
        self.inner.adjoint(st, t, y, &mut xs)?;
        self.inner.kernels().mul_real(st, &xs, ww, x)?;
        let m = self.c[t].len();
        let mut yc = st.alloc_zeros::<C32>(y.len())?;
        self.inner.kernels().sample_weight(st, y, &self.c[t], &mut yc, m, y.len() / m, true, false)?;
        self.inner.adjoint(st, t, &yc, &mut xs)?;
        let mut xf = st.alloc_zeros::<C32>(x.len())?;
        self.inner.kernels().mul_real(st, &xs, wf, &mut xf)?;
        self.inner.kernels().axpby(st, num_complex::Complex64::new(1.0, 0.0), &xf, num_complex::Complex64::new(1.0, 0.0), x)
    }
    /// y_m = E_m((1 − ff) x) + c_m ⊙ E_m(ff x): two inner group forwards (one image each) with the group's map (the
    /// members' mean where their maps differ, see `group_pair`)
    fn forward_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, ys: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        assert_eq!(members.len(), ys.len(), "forward_group: one output per member");
        let (mean, i) = self.group_pair(st, members)?;
        let (ww, wf) = if mean { (&self.group_maps[i].1 .0, &self.group_maps[i].1 .1) } else { (&self.maps[i].0, &self.maps[i].1) };
        let mut xw = st.alloc_zeros::<C32>(x.len())?;
        let mut xf = st.alloc_zeros::<C32>(x.len())?;
        self.inner.kernels().mul_real(st, x, ww, &mut xw)?;
        self.inner.kernels().mul_real(st, x, wf, &mut xf)?;
        self.inner.forward_group(st, members, &xw, ys)?;
        drop(xw);
        let mut yf: Vec<CudaSlice<C32>> = ys.iter().map(|y| unsafe { st.alloc::<C32>(y.len()) }).collect::<Result<_, _>>()?;
        {
            let mut refs: Vec<&mut CudaSlice<C32>> = yf.iter_mut().collect();
            self.inner.forward_group(st, members, &xf, &mut refs)?;
        }
        for ((&t, y), z) in members.iter().zip(ys.iter_mut()).zip(&yf) {
            let m = self.c[t].len();
            self.inner.kernels().sample_weight(st, z, &self.c[t], y, m, y.len() / m, false, true)?;
        }
        Ok(())
    }
    fn forward_residual_group(&mut self, st: &Arc<CudaStream>, members: &[usize], x: &CudaSlice<C32>, ys: &[&CudaSlice<C32>], rs: &mut [&mut CudaSlice<C32>]) -> Result<(), Error> {
        self.forward_group(st, members, x, rs)?;
        subtract_data(self.inner.kernels(), st, ys, rs)
    }
    /// x = (1 − ff) Σ_m E_mᴴ y_m + ff Σ_m E_mᴴ(c̄_m ⊙ y_m), the adjoint of `forward_group` (same group map)
    fn adjoint_group(&mut self, st: &Arc<CudaStream>, members: &[usize], ys: &[&CudaSlice<C32>], x: &mut CudaSlice<C32>) -> Result<(), Error> {
        assert!(!members.is_empty() && members.len() == ys.len(), "adjoint_group: one input per member");
        let (mean, i) = self.group_pair(st, members)?;
        let mut aw = st.alloc_zeros::<C32>(x.len())?;
        self.inner.adjoint_group(st, members, ys, &mut aw)?;
        let mut yc = Vec::with_capacity(members.len());
        for (&t, y) in members.iter().zip(ys) {
            let m = self.c[t].len();
            let mut z = unsafe { st.alloc::<C32>(y.len())? };
            self.inner.kernels().sample_weight(st, y, &self.c[t], &mut z, m, y.len() / m, true, false)?;
            yc.push(z);
        }
        let mut af = st.alloc_zeros::<C32>(x.len())?;
        {
            let refs: Vec<&CudaSlice<C32>> = yc.iter().collect();
            self.inner.adjoint_group(st, members, &refs, &mut af)?;
        }
        drop(yc);
        let (ww, wf) = if mean { (&self.group_maps[i].1 .0, &self.group_maps[i].1 .1) } else { (&self.maps[i].0, &self.maps[i].1) };
        self.inner.kernels().mul_real(st, &aw, ww, x)?;
        self.inner.kernels().mul_real(st, &af, wf, &mut aw)?;
        let one = num_complex::Complex64::new(1.0, 0.0);
        self.inner.kernels().axpby(st, one, &aw, one, x)
    }
}
