//! Interruptible, resumable runs. A [`RunCtl`] owns the run directory: iterative loops save their state there
//! (`state_<name>.hv`, hasty value format, LZ4, written atomically) every `every` iterations and when a stop is
//! requested — SIGINT / SIGTERM, or a file named `STOP` in the run directory — and then return
//! [`Outcome::Stopped`]; rerunning the same command loads the state and continues from that iteration.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Once};

use cudarc::driver::{CudaSlice, CudaStream};
use hasty_cuda::C32;
use hasty_value::{DType, Options, Tensor, Value};

use crate::p2np::Stack;
use crate::Error;

static STOP: AtomicBool = AtomicBool::new(false);
static INSTALL: Once = Once::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Finished,
    /// stopped before this iteration (the state file continues there)
    Stopped(usize),
}

pub struct RunCtl {
    dir: PathBuf,
    /// checkpoint interval in iterations (0: only on stop)
    pub every: usize,
}

impl RunCtl {
    pub fn new(dir: &Path, every: usize) -> Self {
        INSTALL.call_once(|| {
            let _ = ctrlc::set_handler(|| {
                if STOP.swap(true, Ordering::SeqCst) {
                    eprintln!("second interrupt: exiting without checkpoint");
                    std::process::exit(130);
                }
                eprintln!("stop requested: finishing the current iteration and saving a checkpoint (interrupt again to abort)");
            });
        });
        let _ = std::fs::remove_file(dir.join("STOP"));
        Self { dir: dir.to_path_buf(), every }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn stop_requested(&self) -> bool {
        STOP.load(Ordering::SeqCst) || self.dir.join("STOP").exists()
    }

    /// save a checkpoint after `next − 1` iterations?
    pub fn due(&self, next: usize) -> bool {
        self.every > 0 && next % self.every == 0
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.join(format!("state_{name}.hv"))
    }

    pub fn save(&self, name: &str, v: &Value) -> Result<(), Error> {
        let tmp = self.dir.join(format!("state_{name}.hv.tmp"));
        {
            let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
            hasty_value::write(v, &Options::default(), &mut w)?;
            std::io::Write::flush(&mut w)?;
            w.get_ref().sync_all()?;
        }
        std::fs::rename(&tmp, self.path(name))?;
        Ok(())
    }

    pub fn load(&self, name: &str) -> Result<Option<Value>, Error> {
        let p = self.path(name);
        if !p.exists() {
            return Ok(None);
        }
        Ok(Some(hasty_value::read(&mut std::io::BufReader::new(std::fs::File::open(p)?))?))
    }

    /// the loop finished: its state is no longer needed
    pub fn clear(&self, name: &str) {
        let _ = std::fs::remove_file(self.path(name));
    }
}

/// a device vector as a complex64 tensor (C32 is #[repr(C)] {re, im}: the complex64 layout)
pub fn dev_value(st: &Arc<CudaStream>, v: &CudaSlice<C32>) -> Result<Value, Error> {
    let h = st.clone_dtoh(v)?;
    // SAFETY: C32 is two f32, no padding
    let b = unsafe { std::slice::from_raw_parts(h.as_ptr() as *const u8, h.len() * 8) };
    Ok(Value::Tensor(Tensor::from_bytes(DType::C64, vec![h.len()], b)?))
}

pub fn value_dev(st: &Arc<CudaStream>, v: &Value, len: usize) -> Result<CudaSlice<C32>, Error> {
    let t = v.as_tensor().ok_or_else(|| Error::Format("checkpoint: expected a tensor".into()))?;
    if t.dtype() != DType::C64 || t.numel() != len {
        return Err(Error::Format(format!("checkpoint: {} {} values, expected {len} complex64", t.numel(), t.dtype().name())));
    }
    // SAFETY: the tensor buffer is 8-byte aligned and holds `len` complex64 = C32 values
    let h = unsafe { std::slice::from_raw_parts(t.bytes().as_ptr() as *const C32, len) };
    Ok(st.clone_htod(h)?)
}

pub fn stack_value(st: &Arc<CudaStream>, s: &Stack) -> Result<Value, Error> {
    Ok(Value::List(s.v.iter().map(|v| dev_value(st, v)).collect::<Result<_, _>>()?))
}

pub fn value_stack(st: &Arc<CudaStream>, v: &Value, lens: &[usize]) -> Result<Stack, Error> {
    let Value::List(items) = v else { return Err(Error::Format("checkpoint: expected a list".into())) };
    if items.len() != lens.len() {
        return Err(Error::Format(format!("checkpoint: {} sets, expected {}", items.len(), lens.len())));
    }
    Ok(Stack { v: items.iter().zip(lens).map(|(x, &n)| value_dev(st, x, n)).collect::<Result<_, _>>()? })
}

pub fn get<'a>(v: &'a Value, k: &str) -> Result<&'a Value, Error> {
    v.get(k).ok_or_else(|| Error::Format(format!("checkpoint: missing {k}")))
}

pub fn get_usize(v: &Value, k: &str) -> Result<usize, Error> {
    match get(v, k)? {
        Value::Int(i) => Ok(*i as usize),
        _ => Err(Error::Format(format!("checkpoint: {k} is not an integer"))),
    }
}

pub fn get_f64(v: &Value, k: &str) -> Result<f64, Error> {
    match get(v, k)? {
        Value::Float(f) => Ok(*f),
        _ => Err(Error::Format(format!("checkpoint: {k} is not a float"))),
    }
}
