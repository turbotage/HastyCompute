//! Staging k-space data from a host [`KStore`](crate::kstore::KStore) to the device, one set at a time: two pinned
//! buffers of the largest set, a worker thread that decodes (or copies) the next set into one while the GPU works on
//! the current set from the other. A device copy of exactly the set's length is made per use (stream-ordered
//! allocation), so operators see an ordinary `CudaSlice`.
//!
//! One `Stager` per GPU stream; several stagers (GPUs) can share one store (`Arc<KStore>`, reads are `&self`).
//!
//! Packed staging ([`Stager::with_packing`], research/bfp_storage.md §7 step 5): the PCIe copy carries BFP words
//! instead of f32 — the store's own bytes for a `bfp` store, otherwise the host encodes each set right after the
//! store decode (width from the σ rule) — and the device decodes them into the f32 set copy at upload
//! (`hasty_bfp::xfer`), so the operators still see an ordinary `CudaSlice<C32>`. A decode inside the residual
//! kernels (no f32 copy at all) is the next step; it belongs to the engine. [`Stager::make_resident`] keeps every
//! set packed in device memory instead (no PCIe per gradient; VRAM at the BFP size of the data plus one f32 set).

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream, PinnedHostSlice};
use hasty_cuda::C32;

use crate::kstore::{BfpRule, KStore};
use crate::Error;
use hasty_bfp::Bfp;

#[derive(Clone, Copy, Debug, Default)]
pub struct StageStats {
    /// sets staged
    pub sets: usize,
    /// bytes uploaded
    pub bytes: usize,
    /// host time decoding (worker, overlapped with the GPU)
    pub decode_s: f64,
    /// host time the GPU loop waited for a decode to finish (not overlapped)
    pub wait_s: f64,
    /// f32 bytes the uploaded sets represent (equal to `bytes` without packing)
    pub f32_bytes: usize,
}

/// Packed staging switch `HASTY_KSTAGE` (recon_p2np `--kstage`): unset / `f32`: f32 uploads (as before); `bfp`: the
/// σ rule (step ≤ σ/4); `bfp+qREL`: step ≤ REL σ; `bfpN`: N bits; a `:resident` suffix keeps every set packed in
/// device memory ([`Stager::make_resident`]). Returns (rule, resident).
pub fn stage_from_env() -> Result<Option<(BfpRule, bool)>, Error> {
    let v = std::env::var("HASTY_KSTAGE").unwrap_or_default();
    if v.is_empty() || v == "f32" {
        return Ok(None);
    }
    let (r, resident) = match v.strip_suffix(":resident") {
        Some(r) => (r, true),
        None => (v.as_str(), false),
    };
    let spec = crate::kstore::StoreSpec::parse(r, std::path::Path::new("/unused")).map_err(|e| Error::Format(format!("HASTY_KSTAGE '{v}': {e}")))?;
    match spec.bfp {
        Some(rule) => Ok(Some((rule, resident))),
        None => Err(Error::Format(format!("HASTY_KSTAGE '{v}': expected f32 | bfp[+qREL] | bfpN [:resident]"))),
    }
}

/// bytes per complex value over 8 the planner assumes for a rule (a σ rule: 18 bits, above the 15 – 16 measured on
/// ph0001, research/bfp_storage.md §9)
pub fn planning_ratio(rule: &BfpRule) -> f64 {
    let b = match rule {
        BfpRule::Bits(b) => *b,
        BfpRule::Sigma { .. } => 18,
    };
    Bfp::new(b).bytes_per_complex() / 8.0
}

/// pinned packed slots and the width rule of [`Stager::with_packing`]
struct Packing {
    rule: BfpRule,
    pm: [PinnedHostSlice<u32>; 2],
    pe: [PinnedHostSlice<u8>; 2],
    /// all sets packed on the device ([`Stager::make_resident`]): (width, words, exponent bytes) per set
    resident: Option<Vec<(Bfp, CudaSlice<u32>, CudaSlice<u8>)>>,
}

/// Fill packed slot buffers with set t (coil rows `coils`): the store's own BFP bytes, or decode into `scratch`
/// (f32) and encode with the width from `rule` (peak of the set, the store's σ). Returns (width, words, ex bytes).
fn pack_set(store: &KStore, rule: &BfpRule, t: usize, coils: Option<&[usize]>, scratch: &mut [C32], pm: &mut [u32], pe: &mut [u8]) -> Result<(Bfp, usize, usize), Error> {
    use rayon::prelude::*;
    let nco = coils.map_or(store.nvc(), |c| c.len());
    let m = store.len(t);
    if store.bfp_fmt(t).is_some() && !store.is_view() {
        let fmt = store.read_packed(t, coils, pm, pe)?;
        return Ok((fmt, nco * fmt.mant_words(2 * m), nco * Bfp::blocks(2 * m)));
    }
    let n = nco * m;
    store.read(t, coils, &mut scratch[..n])?;
    let peak = scratch[..n].par_iter().map(|z| z.re.abs().max(z.im.abs())).reduce(|| 0.0f32, f32::max);
    let fmt = rule.bits(peak, store.sigma())?;
    let (rw, rb) = (fmt.mant_words(2 * m), Bfp::blocks(2 * m));
    if m > 0 {
        pm[..nco * rw].par_chunks_mut(rw).zip(pe[..nco * rb].par_chunks_mut(rb)).enumerate().for_each(|(c, (mw, eb))| {
            fmt.encode_serial(hasty_bfp::as_reals(&scratch[c * m..(c + 1) * m]), mw, eb);
        });
    }
    Ok((fmt, nco * rw, nco * rb))
}

pub struct Stager {
    store: Arc<KStore>,
    pin: [PinnedHostSlice<C32>; 2],
    /// recorded after the upload out of pin[i]; waited on before pin[i] is written again
    ev: [CudaEvent; 2],
    pack: Option<Packing>,
    /// `HASTY_KSTAGE=...:resident`: make the sets resident at the first use (the stream is known then)
    want_resident: bool,
    pub stats: StageStats,
}

impl Stager {
    pub fn new(ctx: &Arc<CudaContext>, store: Arc<KStore>) -> Result<Self, Error> {
        let n = store.max_set().max(1);
        let pin = [unsafe { ctx.alloc_pinned_with_flags::<C32>(n, 0)? }, unsafe { ctx.alloc_pinned_with_flags::<C32>(n, 0)? }];
        let ev = [ctx.new_event(None)?, ctx.new_event(None)?];
        let s = Self { store, pin, ev, pack: None, want_resident: false, stats: StageStats::default() };
        match stage_from_env()? {
            None => Ok(s),
            Some((rule, resident)) => {
                let mut s = s.with_packing(ctx, rule)?;
                s.want_resident = resident;
                eprintln!("k-space staging: {}{}", rule.describe(), if resident { ", all sets resident on the device" } else { ", packed uploads decoded on the device" });
                Ok(s)
            }
        }
    }

    /// Packed staging with the width rule `rule` (see the module docs): pinned packed slots sized for 24 bits of the
    /// largest set. A σ rule needs the store's noise level.
    pub fn with_packing(mut self, ctx: &Arc<CudaContext>, rule: BfpRule) -> Result<Self, Error> {
        if matches!(rule, BfpRule::Sigma { .. }) && self.store.sigma().is_none() {
            return Err(Error::Format("packed staging: the σ rule needs the store's noise level".into()));
        }
        let (mut nw, mut ne) = (1usize, 4usize);
        for t in 0..self.store.n_sets() {
            let m = self.store.len(t);
            nw = nw.max(self.store.nvc() * Bfp::new(24).mant_words(2 * m));
            ne = ne.max((self.store.nvc() * Bfp::blocks(2 * m)).div_ceil(4) * 4);
        }
        let pm = unsafe { [ctx.alloc_pinned_with_flags::<u32>(nw, 0)?, ctx.alloc_pinned_with_flags::<u32>(nw, 0)?] };
        let pe = unsafe { [ctx.alloc_pinned_with_flags::<u8>(ne, 0)?, ctx.alloc_pinned_with_flags::<u8>(ne, 0)?] };
        self.pack = Some(Packing { rule, pm, pe, resident: None });
        Ok(self)
    }

    /// whether uploads are packed
    pub fn is_packed(&self) -> bool {
        self.pack.is_some()
    }

    /// Upload every set packed into device memory once (needs [`with_packing`](Self::with_packing)); later uses
    /// decode from there (all coils only). Returns the device bytes held.
    pub fn make_resident(&mut self, st: &Arc<CudaStream>) -> Result<usize, Error> {
        let store = self.store.clone();
        let p = self.pack.as_mut().ok_or_else(|| Error::Format("resident packed k-space: call with_packing first".into()))?;
        let mut scratch = if (0..store.n_sets()).all(|t| store.bfp_fmt(t).is_some()) && !store.is_view() { Vec::new() } else { vec![C32::new(0.0, 0.0); store.max_set()] };
        let mut res = Vec::with_capacity(store.n_sets());
        let mut bytes = 0;
        for t in 0..store.n_sets() {
            self.ev[0].synchronize()?;
            let (fmt, nw, ne) = pack_set(&store, &p.rule, t, None, &mut scratch, p.pm[0].as_mut_slice()?, p.pe[0].as_mut_slice()?)?;
            let mut dm = unsafe { st.alloc::<u32>(nw.max(1))? };
            let mut de = unsafe { st.alloc::<u8>(ne.max(4))? };
            st.memcpy_htod(&p.pm[0].as_slice()?[..nw], &mut dm)?;
            st.memcpy_htod(&p.pe[0].as_slice()?[..ne], &mut de)?;
            self.ev[0].record(st)?;
            bytes += 4 * nw + ne;
            res.push((fmt, dm, de));
        }
        st.synchronize()?;
        p.resident = Some(res);
        Ok(bytes)
    }

    /// pinned + device bytes of the packed path (pinned slots; resident sets)
    pub fn packed_bytes(&self) -> (usize, usize) {
        self.pack.as_ref().map_or((0, 0), |p| (2 * (4 * p.pm[0].len() + p.pe[0].len()), p.resident.as_ref().map_or(0, |r| r.iter().map(|(_, m, e)| 4 * m.len() + e.len()).sum())))
    }

    /// decode packed slot bytes (already on the device) into a new f32 set buffer of `nco` rows × m samples
    fn decode_dev(st: &Arc<CudaStream>, fmt: Bfp, dm: &CudaSlice<u32>, de: &CudaSlice<u8>, m: usize, nco: usize) -> Result<CudaSlice<C32>, Error> {
        let mut d = unsafe { st.alloc::<C32>((nco * m).max(1))? };
        hasty_bfp::xfer::kernels(&st.context().clone(), fmt)?.unpack(st, dm, de, 2 * m, nco, &mut d)?;
        Ok(d)
    }

    /// [`upload`](Self::upload) on the packed path: one set packed into slot 0, uploaded and decoded into the
    /// returned buffer (resident: decoded from device memory)
    fn upload_packed(&mut self, st: &Arc<CudaStream>, t: usize, coils: Option<&[usize]>) -> Result<CudaSlice<C32>, Error> {
        if std::mem::take(&mut self.want_resident) {
            let b = self.make_resident(st)?;
            eprintln!("k-space staging: {:.2} GiB packed resident on the device", b as f64 / (1u64 << 30) as f64);
        }
        let store = self.store.clone();
        let nco = coils.map_or(store.nvc(), |c| c.len());
        let m = store.len(t);
        let p = self.pack.as_mut().unwrap();
        if let Some(res) = &p.resident {
            if coils.is_some() {
                return Err(Error::Format("resident packed k-space: coil subsets are not supported".into()));
            }
            let (fmt, dm, de) = &res[t];
            self.stats.sets += 1;
            self.stats.f32_bytes += nco * m * 8;
            return Self::decode_dev(st, *fmt, dm, de, m, nco);
        }
        self.ev[0].synchronize()?;
        let t0 = Instant::now();
        let (fmt, nw, ne) = pack_set(&store, &p.rule, t, coils, self.pin[0].as_mut_slice()?, p.pm[0].as_mut_slice()?, p.pe[0].as_mut_slice()?)?;
        self.stats.decode_s += t0.elapsed().as_secs_f64();
        let mut dm = unsafe { st.alloc::<u32>(nw.max(1))? };
        let mut de = unsafe { st.alloc::<u8>(ne.max(4))? };
        st.memcpy_htod(&p.pm[0].as_slice()?[..nw], &mut dm)?;
        st.memcpy_htod(&p.pe[0].as_slice()?[..ne], &mut de)?;
        self.ev[0].record(st)?;
        self.stats.sets += 1;
        self.stats.bytes += 4 * nw + ne;
        self.stats.f32_bytes += nco * m * 8;
        Self::decode_dev(st, fmt, &dm, &de, m, nco)
    }

    /// [`for_each`](Self::for_each) on the packed path
    fn for_each_packed(&mut self, st: &Arc<CudaStream>, order: &[usize], coils: Option<&[usize]>, f: &mut dyn FnMut(usize, &CudaSlice<C32>) -> Result<(), Error>) -> Result<(), Error> {
        if std::mem::take(&mut self.want_resident) {
            let b = self.make_resident(st)?;
            eprintln!("k-space staging: {:.2} GiB packed resident on the device", b as f64 / (1u64 << 30) as f64);
        }
        let store = self.store.clone();
        let nco = coils.map_or(store.nvc(), |c| c.len());
        let p = self.pack.as_mut().unwrap();
        if let Some(res) = &p.resident {
            if coils.is_some() {
                return Err(Error::Format("resident packed k-space: coil subsets are not supported".into()));
            }
            for &t in order {
                let (fmt, dm, de) = &res[t];
                let d = Self::decode_dev(st, *fmt, dm, de, store.len(t), nco)?;
                self.stats.sets += 1;
                self.stats.f32_bytes += nco * store.len(t) * 8;
                f(t, &d)?;
            }
            return Ok(());
        }
        let rule = p.rule;
        let ts = Instant::now();
        let mut meta = [(Bfp::new(16), 0usize, 0usize); 2];
        {
            self.ev[0].synchronize()?;
            meta[0] = pack_set(&store, &rule, order[0], coils, self.pin[0].as_mut_slice()?, p.pm[0].as_mut_slice()?, p.pe[0].as_mut_slice()?)?;
        }
        self.stats.decode_s += ts.elapsed().as_secs_f64();
        for (k, &t) in order.iter().enumerate() {
            let (cur, nxt) = (k % 2, (k + 1) % 2);
            let m = store.len(t);
            let (fmt, nw, ne) = meta[cur];
            let mut dm = unsafe { st.alloc::<u32>(nw.max(1))? };
            let mut de = unsafe { st.alloc::<u8>(ne.max(4))? };
            st.memcpy_htod(&p.pm[cur].as_slice()?[..nw], &mut dm)?;
            st.memcpy_htod(&p.pe[cur].as_slice()?[..ne], &mut de)?;
            self.ev[cur].record(st)?;
            let d = Self::decode_dev(st, fmt, &dm, &de, m, nco)?;
            self.stats.sets += 1;
            self.stats.bytes += 4 * nw + ne;
            self.stats.f32_bytes += nco * m * 8;
            let next = order.get(k + 1).copied();
            let (pa, pb) = self.pin.split_at_mut(1);
            let scr: &mut PinnedHostSlice<C32> = if nxt == 0 { &mut pa[0] } else { &mut pb[0] };
            let (ma, mb) = p.pm.split_at_mut(1);
            let pmn: &mut PinnedHostSlice<u32> = if nxt == 0 { &mut ma[0] } else { &mut mb[0] };
            let (ea, eb) = p.pe.split_at_mut(1);
            let pen: &mut PinnedHostSlice<u8> = if nxt == 0 { &mut ea[0] } else { &mut eb[0] };
            let ev_other = &self.ev[nxt];
            let store_r = &store;
            let (res, dec) = std::thread::scope(|s| {
                let h = next.map(|tn| {
                    s.spawn(move || -> Result<((Bfp, usize, usize), f64), Error> {
                        ev_other.synchronize()?;
                        let t0 = Instant::now();
                        let r = pack_set(store_r, &rule, tn, coils, scr.as_mut_slice()?, pmn.as_mut_slice()?, pen.as_mut_slice()?)?;
                        Ok((r, t0.elapsed().as_secs_f64()))
                    })
                });
                let r = f(t, &d);
                let tw = Instant::now();
                let dec = h.map(|h| h.join().expect("staging worker panicked"));
                (r, dec.map(|d| (d, tw.elapsed().as_secs_f64())))
            });
            res?;
            if let Some((d, ws)) = dec {
                let (mt, ds) = d?;
                meta[nxt] = mt;
                self.stats.decode_s += ds;
                self.stats.wait_s += ws;
            }
        }
        Ok(())
    }

    pub fn store(&self) -> &Arc<KStore> {
        &self.store
    }

    /// pinned bytes held
    pub fn pinned_bytes(&self) -> usize {
        2 * self.pin[0].len() * 8
    }

    /// Visit the sets in `order`, each with its data on the device (coil rows `coils`, None = all, coil-major) —
    /// `f(t, y_t)`. The next set is decoded on a worker thread meanwhile. `f` may enqueue asynchronous work that
    /// reads y_t: the device copy lives until the stream is done with it (stream-ordered free).
    pub fn for_each(&mut self, st: &Arc<CudaStream>, order: &[usize], coils: Option<&[usize]>, f: &mut dyn FnMut(usize, &CudaSlice<C32>) -> Result<(), Error>) -> Result<(), Error> {
        if order.is_empty() {
            return Ok(());
        }
        if self.pack.is_some() {
            return self.for_each_packed(st, order, coils, f);
        }
        let nco = coils.map_or(self.store.nvc(), |c| c.len());
        let store = self.store.clone();
        let n_of = |t: usize| nco * store.len(t);
        let ts = Instant::now();
        {
            let n = n_of(order[0]);
            self.ev[0].synchronize()?;
            store.read(order[0], coils, &mut self.pin[0].as_mut_slice()?[..n])?;
        }
        self.stats.decode_s += ts.elapsed().as_secs_f64();
        for (k, &t) in order.iter().enumerate() {
            let (cur, nxt) = (k % 2, (k + 1) % 2);
            let n = n_of(t);
            let mut d = unsafe { st.alloc::<C32>(n.max(1))? };
            {
                // the pinned prefix as a plain slice: the copy is asynchronous (pinned memory), so pin[cur] is not
                // written again before ev[cur] has completed
                let src = &self.pin[cur].as_slice()?[..n];
                st.memcpy_htod(src, &mut d)?;
                self.ev[cur].record(st)?;
            }
            self.stats.sets += 1;
            self.stats.bytes += n * 8;
            self.stats.f32_bytes += n * 8;
            let next = order.get(k + 1).copied();
            let (pa, pb) = self.pin.split_at_mut(1);
            let other: &mut PinnedHostSlice<C32> = if nxt == 0 { &mut pa[0] } else { &mut pb[0] };
            let ev_other = &self.ev[nxt];
            let (res, dec) = std::thread::scope(|s| {
                let h = next.map(|tn| {
                    let nn = n_of(tn);
                    let store = &store;
                    s.spawn(move || -> Result<f64, Error> {
                        ev_other.synchronize()?;
                        let t0 = Instant::now();
                        store.read(tn, coils, &mut other.as_mut_slice()?[..nn])?;
                        Ok(t0.elapsed().as_secs_f64())
                    })
                });
                let r = f(t, &d);
                let tw = Instant::now();
                let dec = h.map(|h| h.join().expect("staging worker panicked"));
                (r, dec.map(|d| (d, tw.elapsed().as_secs_f64())))
            });
            res?;
            if let Some(d) = dec {
                let (ds, ws) = d;
                self.stats.decode_s += ds?;
                self.stats.wait_s += ws;
            }
        }
        Ok(())
    }

    /// one set (all coils or a subset) as a new device buffer, without prefetch
    pub fn upload(&mut self, st: &Arc<CudaStream>, t: usize, coils: Option<&[usize]>) -> Result<CudaSlice<C32>, Error> {
        if self.pack.is_some() {
            return self.upload_packed(st, t, coils);
        }
        let n = coils.map_or(self.store.nvc(), |c| c.len()) * self.store.len(t);
        self.ev[0].synchronize()?;
        let t0 = Instant::now();
        self.store.read(t, coils, &mut self.pin[0].as_mut_slice()?[..n])?;
        self.stats.decode_s += t0.elapsed().as_secs_f64();
        let mut d = unsafe { st.alloc::<C32>(n.max(1))? };
        st.memcpy_htod(&self.pin[0].as_slice()?[..n], &mut d)?;
        self.ev[0].record(st)?;
        self.stats.sets += 1;
        self.stats.bytes += n * 8;
        self.stats.f32_bytes += n * 8;
        Ok(d)
    }
}
