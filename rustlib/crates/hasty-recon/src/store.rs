//! Chunked vector storage on the device or in pinned host memory (swap-in / swap-out), chosen per buffer by the
//! memory planner (`plan.rs`). Algorithms touch one chunk at a time: a host chunk is staged through a device scratch
//! buffer of the same length (`Scratch`), so the device needs only the chunks in flight.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, PinnedHostSlice};
use hasty_cuda::C32;
use hasty_value::Value;

use crate::checkpoint::{dev_value, value_dev};
use crate::Error;

pub enum Chunk {
    Dev(CudaSlice<C32>),
    Host(PinnedHostSlice<C32>),
}

impl Chunk {
    pub fn len(&self) -> usize {
        match self {
            Chunk::Dev(d) => d.len(),
            Chunk::Host(h) => h.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct Store {
    pub chunks: Vec<Chunk>,
    pub on_host: bool,
}

fn pinned(ctx: &Arc<CudaContext>, n: usize) -> Result<PinnedHostSlice<C32>, Error> {
    // plain page-locked memory (not write-combined: checkpoints read it on the CPU)
    Ok(unsafe { ctx.alloc_pinned_with_flags::<C32>(n.max(1), 0)? })
}

impl Store {
    /// zeroed store with the given chunk lengths
    pub fn zeros(ctx: &Arc<CudaContext>, st: &Arc<CudaStream>, lens: &[usize], on_host: bool) -> Result<Self, Error> {
        let chunks = lens
            .iter()
            .map(|&n| -> Result<Chunk, Error> {
                if on_host {
                    let mut h = pinned(ctx, n)?;
                    h.as_mut_slice()?.fill(C32::new(0.0, 0.0));
                    Ok(Chunk::Host(h))
                } else {
                    Ok(Chunk::Dev(st.alloc_zeros(n)?))
                }
            })
            .collect::<Result<_, _>>()?;
        Ok(Self { chunks, on_host })
    }

    pub fn bytes(&self) -> usize {
        self.chunks.iter().map(|c| c.len() * 8).sum()
    }

    /// chunk t on the device: itself, or staged into `scratch` (same length)
    pub fn view<'a>(&'a self, st: &Arc<CudaStream>, t: usize, scratch: &'a mut CudaSlice<C32>) -> Result<&'a CudaSlice<C32>, Error> {
        match &self.chunks[t] {
            Chunk::Dev(d) => Ok(d),
            Chunk::Host(h) => {
                st.memcpy_htod(h, scratch)?;
                Ok(scratch)
            }
        }
    }

    /// chunk t ← src
    pub fn put(&mut self, st: &Arc<CudaStream>, t: usize, src: &CudaSlice<C32>) -> Result<(), Error> {
        match &mut self.chunks[t] {
            Chunk::Dev(d) => st.memcpy_dtod(src, d)?,
            Chunk::Host(h) => st.memcpy_dtoh(src, h)?,
        }
        Ok(())
    }

    /// checkpoint format of `checkpoint::stack_value` (a list of complex64 tensors)
    pub fn to_value(&self, st: &Arc<CudaStream>) -> Result<Value, Error> {
        let mut items = vec![];
        for c in &self.chunks {
            items.push(match c {
                Chunk::Dev(d) => dev_value(st, d)?,
                Chunk::Host(h) => {
                    st.synchronize()?;
                    let s = h.as_slice()?;
                    let b = unsafe { std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * 8) };
                    Value::Tensor(hasty_value::Tensor::from_bytes(hasty_value::DType::C64, vec![s.len()], b)?)
                }
            });
        }
        Ok(Value::List(items))
    }

    pub fn load_value(&mut self, st: &Arc<CudaStream>, v: &Value) -> Result<(), Error> {
        let Value::List(items) = v else { return Err(Error::Format("checkpoint: expected a list".into())) };
        if items.len() != self.chunks.len() {
            return Err(Error::Format(format!("checkpoint: {} chunks, expected {}", items.len(), self.chunks.len())));
        }
        for (c, x) in self.chunks.iter_mut().zip(items) {
            let n = c.len();
            let d = value_dev(st, x, n)?;
            match c {
                Chunk::Dev(dd) => *dd = d,
                Chunk::Host(h) => st.memcpy_dtoh(&d, h)?,
            }
        }
        Ok(())
    }
}

/// device scratch buffers keyed by length (several per length)
#[derive(Default)]
pub struct Scratch {
    pool: HashMap<usize, Vec<CudaSlice<C32>>>,
}

impl Scratch {
    /// `k` distinct buffers of length n (allocated on first use)
    pub fn get(&mut self, st: &Arc<CudaStream>, n: usize, k: usize) -> Result<&mut Vec<CudaSlice<C32>>, Error> {
        let v = self.pool.entry(n).or_default();
        while v.len() < k {
            v.push(st.alloc_zeros(n)?);
        }
        Ok(v)
    }

    pub fn bytes(&self) -> usize {
        self.pool.iter().map(|(n, v)| n * 8 * v.len()).sum()
    }
}
