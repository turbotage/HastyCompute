//! Staging k-space data from a host [`KStore`](crate::kstore::KStore) to the device, one set at a time: two pinned
//! buffers of the largest set, a worker thread that decodes (or copies) the next set into one while the GPU works on
//! the current set from the other. A device copy of exactly the set's length is made per use (stream-ordered
//! allocation), so operators see an ordinary `CudaSlice`.
//!
//! One `Stager` per GPU stream; several stagers (GPUs) can share one store (`Arc<KStore>`, reads are `&self`).

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream, PinnedHostSlice};
use hasty_cuda::C32;

use crate::kstore::KStore;
use crate::Error;

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
}

pub struct Stager {
    store: Arc<KStore>,
    pin: [PinnedHostSlice<C32>; 2],
    /// recorded after the upload out of pin[i]; waited on before pin[i] is written again
    ev: [CudaEvent; 2],
    pub stats: StageStats,
}

impl Stager {
    pub fn new(ctx: &Arc<CudaContext>, store: Arc<KStore>) -> Result<Self, Error> {
        let n = store.max_set().max(1);
        let pin = [unsafe { ctx.alloc_pinned_with_flags::<C32>(n, 0)? }, unsafe { ctx.alloc_pinned_with_flags::<C32>(n, 0)? }];
        let ev = [ctx.new_event(None)?, ctx.new_event(None)?];
        Ok(Self { store, pin, ev, stats: StageStats::default() })
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
        Ok(d)
    }
}
