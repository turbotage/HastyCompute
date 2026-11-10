//! GPU pacer: throttle long GPU jobs to a duty cycle or a board-power target, so they can run without the fans at full
//! speed. Call [`pace`] at natural boundaries of a GPU loop (after an LBM step, a simulator bin, a set of a
//! reconstruction). Configuration from the environment, read once:
//! - unset (default): full speed. `pace` is one atomic load and a branch: no synchronisation, no overhead.
//! - `HASTY_GPU_DUTY=f` (0 < f < 1): the GPU may be busy a fraction f of the time. At each pace point the stream is
//!   synchronised and the thread sleeps busy · (1 − f) / f, busy = wall time since the end of the last pace sleep.
//! - `HASTY_GPU_WATTS=W`: a target board power. A background thread reads the power (NVML, libnvidia-ml.so.1, every
//!   250 ms), averages it over `HASTY_GPU_WATTS_WINDOW` s (default 3.5, boxcar) and once per second steers the duty,
//!   duty ← duty · (W / P̄)^0.3, within [0.05, 1].
//! - `HASTY_GPU_PACE_LOG=1`: print the duty / power every few seconds.
//! - `HASTY_GPU_SLICE=s` (default 0.05): loops with long kernels split them into slices of about s seconds of GPU time
//!   while paced ([`slice_seconds`]), so the GPU alternates fast and its temperature stays smooth.
//! The driver-level alternative is `sudo nvidia-smi -pl W` (exact, zero cost, needs root).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use cudarc::driver::CudaStream;

struct Pacer {
    /// duty as f64 bits (changed by the power controller)
    duty: AtomicU64,
    last: Mutex<Instant>,
    log: bool,
    watts: Option<f64>,
    power: AtomicU64,
}

static PACER: OnceLock<Option<Pacer>> = OnceLock::new();

fn init() -> Option<Pacer> {
    let duty = std::env::var("HASTY_GPU_DUTY").ok().and_then(|v| v.parse::<f64>().ok()).filter(|d| *d > 0.0 && *d < 1.0);
    let watts = std::env::var("HASTY_GPU_WATTS").ok().and_then(|v| v.parse::<f64>().ok()).filter(|w| *w > 0.0);
    if duty.is_none() && watts.is_none() {
        return None;
    }
    let log = std::env::var("HASTY_GPU_PACE_LOG").map_or(false, |v| v != "0");
    let d0 = duty.unwrap_or(0.7);
    eprintln!("GPU pacer: {}", match watts { Some(w) => format!("target {w:.0} W (initial duty {d0:.2})"), None => format!("duty {d0:.2}") });
    Some(Pacer { duty: AtomicU64::new(d0.to_bits()), last: Mutex::new(Instant::now()), log, watts, power: AtomicU64::new(0f64.to_bits()) })
}

/// the pacer, if configured (the power thread starts on first use)
fn pacer() -> Option<&'static Pacer> {
    let p = PACER.get_or_init(init).as_ref()?;
    if p.watts.is_some() {
        static THREAD: std::sync::Once = std::sync::Once::new();
        THREAD.call_once(|| {
            std::thread::spawn(|| power_loop(PACER.get().unwrap().as_ref().unwrap()));
        });
    }
    Some(p)
}

/// pace the calling GPU loop (see the module docs); a no-op unless configured
#[inline]
pub fn pace(st: &Arc<CudaStream>) {
    if let Some(Some(_)) = PACER.get() {
        pace_slow(st);
    } else if PACER.get().is_none() {
        // first call: read the configuration
        if pacer().is_some() {
            pace_slow(st);
        }
    }
}

#[cold]
fn pace_slow(st: &Arc<CudaStream>) {
    let Some(p) = pacer() else { return };
    let _ = st.synchronize();
    pace_host(p);
}

/// pace without a CUDA stream (the caller has synchronised, or for host-driven loops)
pub fn pace_synced() {
    if let Some(p) = pacer() {
        pace_host(p);
    }
}

fn pace_host(p: &Pacer) {
    let duty = f64::from_bits(p.duty.load(Ordering::Relaxed));
    let mut last = p.last.lock().unwrap();
    let busy = last.elapsed().as_secs_f64();
    if duty < 1.0 {
        let s = (busy * (1.0 - duty) / duty).min(5.0);
        if s > 1e-4 {
            std::thread::sleep(Duration::from_secs_f64(s));
        }
    }
    *last = Instant::now();
}

/// whether a pacer is configured
pub fn active() -> bool {
    pacer().is_some()
}

/// With a pacer configured: the GPU time (s) a long kernel should run between pace points
/// (`HASTY_GPU_SLICE`, default 0.05 s), so the duty cycle switches at ~10–20 Hz and the temperature stays smooth
/// instead of swinging with second-long launches. None (unpaced): launch whole, nothing changes.
pub fn slice_seconds() -> Option<f64> {
    pacer()?;
    Some(std::env::var("HASTY_GPU_SLICE").ok().and_then(|v| v.parse::<f64>().ok()).filter(|s| *s > 1e-3).unwrap_or(0.05))
}

fn power_loop(p: &'static Pacer) {
    let Some(nv) = Nvml::open() else {
        eprintln!("GPU pacer: NVML (libnvidia-ml.so.1) unavailable — keeping the initial duty");
        return;
    };
    let target = p.watts.unwrap();
    // boxcar mean over the window (HASTY_GPU_WATTS_WINDOW s, default 3.5), sampled every 250 ms; the duty is adjusted
    // once per second with a gentle gain (a lagging average and a fast controller would oscillate)
    let window = std::env::var("HASTY_GPU_WATTS_WINDOW").ok().and_then(|v| v.parse::<f64>().ok()).filter(|w| *w > 0.25).unwrap_or(3.5);
    let n = ((window / 0.25).round() as usize).max(1);
    let mut ring: std::collections::VecDeque<f64> = std::collections::VecDeque::with_capacity(n);
    let (mut t_ctl, mut t_log) = (Instant::now(), Instant::now());
    loop {
        std::thread::sleep(Duration::from_millis(250));
        let Some(w) = nv.power_w() else { continue };
        if ring.len() == n {
            ring.pop_front();
        }
        ring.push_back(w);
        let mean = ring.iter().sum::<f64>() / ring.len() as f64;
        p.power.store(mean.to_bits(), Ordering::Relaxed);
        if t_ctl.elapsed().as_secs_f64() >= 1.0 && ring.len() * 2 >= n {
            let d = f64::from_bits(p.duty.load(Ordering::Relaxed));
            let nd = (d * (target / mean.max(1.0)).powf(0.3)).clamp(0.05, 1.0);
            p.duty.store(nd.to_bits(), Ordering::Relaxed);
            t_ctl = Instant::now();
        }
        if p.log && t_log.elapsed().as_secs_f64() > 5.0 {
            eprintln!("GPU pacer: {mean:.0} W over {window:.1} s (target {target:.0}), duty {:.2}", f64::from_bits(p.duty.load(Ordering::Relaxed)));
            t_log = Instant::now();
        }
    }
}

/// minimal NVML through dlopen: board power of device 0
struct Nvml {
    _lib: libloading::Library,
    dev: *mut std::ffi::c_void,
    power: unsafe extern "C" fn(*mut std::ffi::c_void, *mut u32) -> i32,
}

unsafe impl Send for Nvml {}

impl Nvml {
    fn open() -> Option<Self> {
        unsafe {
            let lib = libloading::Library::new("libnvidia-ml.so.1").ok()?;
            let init: libloading::Symbol<unsafe extern "C" fn() -> i32> = lib.get(b"nvmlInit_v2").ok()?;
            if init() != 0 {
                return None;
            }
            let handle: libloading::Symbol<unsafe extern "C" fn(u32, *mut *mut std::ffi::c_void) -> i32> = lib.get(b"nvmlDeviceGetHandleByIndex_v2").ok()?;
            let mut dev = std::ptr::null_mut();
            let idx = std::env::var("CUDA_VISIBLE_DEVICES").ok().and_then(|v| v.split(',').next().and_then(|s| s.trim().parse::<u32>().ok())).unwrap_or(0);
            if handle(idx, &mut dev) != 0 {
                return None;
            }
            let power: libloading::Symbol<unsafe extern "C" fn(*mut std::ffi::c_void, *mut u32) -> i32> = lib.get(b"nvmlDeviceGetPowerUsage").ok()?;
            let power = *power;
            Some(Self { _lib: lib, dev, power })
        }
    }

    fn power_w(&self) -> Option<f64> {
        let mut mw = 0u32;
        let r = unsafe { (self.power)(self.dev, &mut mw) };
        (r == 0).then_some(mw as f64 * 1e-3)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn unconfigured_pacer_is_a_no_op() {
        // the environment of the test run has neither variable: no pacer, no synchronisation
        if std::env::var("HASTY_GPU_DUTY").is_err() && std::env::var("HASTY_GPU_WATTS").is_err() {
            assert!(!super::active());
            let t = std::time::Instant::now();
            for _ in 0..1_000_000 {
                super::pace_synced();
            }
            assert!(t.elapsed().as_secs_f64() < 0.05, "1e6 calls took {:?}", t.elapsed());
        }
    }
}
