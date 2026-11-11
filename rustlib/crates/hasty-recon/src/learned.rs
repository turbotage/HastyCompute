//! Learned 4D-flow prior (the exported O_h-equivariant 7-point denoiser, `rustlib/tools/denoiser`) for P²nP.
//!
//! The libtorch backend is a plugin: `crates/hasty-denoise-torch` builds `libhasty_denoise_torch.so` (its own cargo
//! workspace; build command in its Cargo.toml) with a C ABI (`hdt_*`), loaded here with dlopen — libtorch never
//! enters the main build. Default plugin path: `rustlib/target/torch/release/libhasty_denoise_torch.so`, overridden
//! by `HASTY_DENOISE_PLUGIN`.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};

use hasty_cuda::C32;

use crate::Error;

extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}
const RTLD_NOW: c_int = 2;

type AbiFn = unsafe extern "C" fn() -> u32;
type OpenFn = unsafe extern "C" fn(*const c_char, c_int, c_int, *mut c_char, c_int) -> *mut c_void;
type OnCudaFn = unsafe extern "C" fn(*mut c_void) -> c_int;
#[allow(clippy::type_complexity)]
type DenoiseFn = unsafe extern "C" fn(*mut c_void, *mut f32, usize, *const usize, usize, *const f32, f32, c_int, usize, usize, *mut c_char, c_int) -> c_int;
type AutocastFn = unsafe extern "C" fn(*mut c_void, c_int) -> c_int;
type CloseFn = unsafe extern "C" fn(*mut c_void);

pub const NE: usize = 7;

pub struct LearnedPrior {
    handle: *mut c_void,
    denoise_fn: DenoiseFn,
    autocast_fn: AutocastFn,
    close_fn: CloseFn,
    pub on_cuda: bool,
    pub dir: PathBuf,
}

pub fn default_plugin() -> PathBuf {
    std::env::var_os("HASTY_DENOISE_PLUGIN").map(PathBuf::from).unwrap_or_else(|| hasty_site::target().join("torch/release/libhasty_denoise_torch.so"))
}

fn dl_err() -> String {
    unsafe {
        let e = dlerror();
        if e.is_null() {
            "unknown dlopen error".into()
        } else {
            CStr::from_ptr(e).to_string_lossy().into_owned()
        }
    }
}

fn err_string(buf: &[u8]) -> String {
    let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

impl LearnedPrior {
    /// Load an export directory (`model.json` + `model_aoti.pt2` or `model.ts`; AOTI preferred when present) through
    /// the plugin at `plugin` ([`default_plugin`]).
    pub fn load(plugin: &Path, dir: &Path, cuda: bool) -> Result<Self, Error> {
        let lib = CString::new(plugin.to_string_lossy().as_bytes()).unwrap();
        let h = unsafe { dlopen(lib.as_ptr(), RTLD_NOW) };
        if h.is_null() {
            return Err(Error::Format(format!("{}: {} (build it: see crates/hasty-denoise-torch/Cargo.toml)", plugin.display(), dl_err())));
        }
        let sym = |name: &str| -> Result<*mut c_void, Error> {
            let n = CString::new(name).unwrap();
            let p = unsafe { dlsym(h, n.as_ptr()) };
            if p.is_null() {
                Err(Error::Format(format!("{}: no symbol {name}: {}", plugin.display(), dl_err())))
            } else {
                Ok(p)
            }
        };
        // SAFETY: the ABI version is checked before any other symbol is used; the symbols are the plugin's
        // `extern "C"` functions with exactly these signatures in ABI version 2
        let abi: AbiFn = unsafe { std::mem::transmute::<*mut c_void, AbiFn>(sym("hdt_abi_version")?) };
        let v = unsafe { abi() };
        if v != 2 {
            return Err(Error::Format(format!("{}: plugin ABI {v}, expected 2 (rebuild the plugin)", plugin.display())));
        }
        let (open, on_cuda, denoise_fn, autocast_fn, close_fn): (OpenFn, OnCudaFn, DenoiseFn, AutocastFn, CloseFn) = unsafe {
            (
                std::mem::transmute::<*mut c_void, OpenFn>(sym("hdt_open")?),
                std::mem::transmute::<*mut c_void, OnCudaFn>(sym("hdt_on_cuda")?),
                std::mem::transmute::<*mut c_void, DenoiseFn>(sym("hdt_denoise")?),
                std::mem::transmute::<*mut c_void, AutocastFn>(sym("hdt_set_autocast")?),
                std::mem::transmute::<*mut c_void, CloseFn>(sym("hdt_close")?),
            )
        };
        let format = if dir.join("model_aoti.pt2").exists() { 1 } else { 0 };
        let d = CString::new(dir.to_string_lossy().as_bytes()).unwrap();
        let mut err = vec![0u8; 4096];
        let handle = unsafe { open(d.as_ptr(), format, cuda as c_int, err.as_mut_ptr() as *mut c_char, err.len() as c_int) };
        if handle.is_null() {
            return Err(Error::Format(format!("{}: {}", dir.display(), err_string(&err))));
        }
        // the library stays loaded for the process lifetime (libtorch does not support unloading)
        Ok(Self { handle, denoise_fn, autocast_fn, close_fn, on_cuda: unsafe { on_cuda(handle) } != 0, dir: dir.to_path_buf() })
    }

    /// Denoise a hasty stack in place: `images[frame * 7 + e]`, x-fastest `dims`, the full cardiac cycle;
    /// `kv` = π/VENC per encode ([ref, x_lo, x_hi, y_lo, y_hi, z_lo, z_hi]); `h` voxel size (m); hasty-mrsim's
    /// exp(−i kv v) convention. Exact overlap tiling bounds the memory: `tile` = spatial tiles of about that many
    /// voxels per axis (None: whole axes; the 28-voxel margins make small tiles expensive), `keep_frames` = cyclic
    /// time windows keeping that many frames (± 3 margin frames; None: the whole cycle per call).
    pub fn denoise(&self, images: &mut [C32], dims: [usize; 3], frames: usize, kv: [f32; NE], h: f32, tile: Option<usize>, keep_frames: Option<usize>) -> Result<(), Error> {
        let mut err = vec![0u8; 4096];
        // SAFETY: C32 is #[repr(C)] {re, im} f32 (the plugin's layout)
        let rc = unsafe { (self.denoise_fn)(self.handle, images.as_mut_ptr() as *mut f32, images.len(), dims.as_ptr(), frames, kv.as_ptr(), h, -1, tile.unwrap_or(0), keep_frames.unwrap_or(0), err.as_mut_ptr() as *mut c_char, err.len() as c_int) };
        if rc != 0 {
            return Err(Error::Format(format!("denoiser: {}", err_string(&err))));
        }
        Ok(())
    }
}

impl LearnedPrior {
    /// mixed precision on CUDA: 0 float32, 1 bf16 autocast (the training regime), 2 fp16 autocast
    pub fn set_autocast(&self, mode: i32) -> Result<(), Error> {
        if unsafe { (self.autocast_fn)(self.handle, mode) } != 0 {
            return Err(Error::Format(format!("autocast {mode}: needs a TorchScript model on CUDA")));
        }
        Ok(())
    }
}

impl Drop for LearnedPrior {
    fn drop(&mut self) {
        unsafe { (self.close_fn)(self.handle) }
    }
}
