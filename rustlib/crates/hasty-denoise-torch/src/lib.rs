//! libtorch backend of the hasty 4D-flow denoiser (layout conversion: `hasty-denoise`). A small C ABI shim
//! (csrc/shim.cpp) over libtorch's C++ API loads either the TorchScript export (`model.ts`) or an AOTInductor package
//! (`model_aoti.pt2`, from `export.py --aoti`: torch.export + inductor ahead-of-time compilation — the supported,
//! fast path; TorchScript is deprecated upstream). It is compiled against the training venv's libtorch, so it does
//! not lag behind PyTorch releases the way generated bindings (tch-rs) do.
//! Host round trip per call for now; running on the P²nP CUDA stream with device pointers is the follow-up.
//!
//! Plugin: the crate also builds as a cdylib (`libhasty_denoise_torch.so`) with a small C ABI (`hdt_*`, below), which
//! the main workspace (`hasty_recon::learned`) loads with dlopen — libtorch never enters the main build.

use std::ffi::{c_char, c_int, c_void, CString};
use std::path::Path;

use hasty_denoise::{from_model, kv_model, tiled, to_model, Error, ModelInfo, NE};
use num_complex::Complex32 as C32;

extern "C" {
    fn hd_load(path: *const c_char, kind: c_int, device: c_int, err: *mut c_char, errlen: c_int) -> *mut c_void;
    fn hd_forward(h: *mut c_void, x: *const f32, xshape: *const i64, kv: *const f32, hh: *const f32, out: *mut f32, err: *mut c_char, errlen: c_int) -> c_int;
    fn hd_free(h: *mut c_void);
    fn hd_cuda_available() -> c_int;
    fn hd_set_autocast(h: *mut c_void, mode: c_int) -> c_int;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// `model.ts`
    TorchScript,
    /// `model_aoti.pt2` (compiled for the device it was exported on)
    Aoti,
}

pub struct TorchDenoiser {
    handle: *mut c_void,
    pub info: ModelInfo,
    pub format: Format,
    pub cuda: bool,
}

// the libtorch module is used from one thread at a time through &mut-free calls; libtorch modules are thread-safe
// for inference, the handle is owned
unsafe impl Send for TorchDenoiser {}

fn err_string(buf: &[u8]) -> String {
    let n = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..n]).into_owned()
}

pub fn cuda_available() -> bool {
    unsafe { hd_cuda_available() != 0 }
}

impl TorchDenoiser {
    /// Load an export directory (`model.json` + `model.ts` or `model_aoti.pt2`); `cuda` selects device 0.
    pub fn load(dir: &Path, format: Format, cuda: bool) -> Result<Self, Error> {
        let info = ModelInfo::load(dir)?;
        let file = dir.join(match format {
            Format::TorchScript => "model.ts",
            Format::Aoti => "model_aoti.pt2",
        });
        let path = CString::new(file.to_string_lossy().as_bytes()).map_err(|e| Error::Backend(e.to_string()))?;
        let mut err = vec![0u8; 4096];
        let cuda = cuda && cuda_available();
        let kind = if format == Format::TorchScript { 0 } else { 1 };
        let handle = unsafe { hd_load(path.as_ptr(), kind, if cuda { 0 } else { -1 }, err.as_mut_ptr() as *mut c_char, err.len() as c_int) };
        if handle.is_null() {
            return Err(Error::Backend(format!("{}: {}", file.display(), err_string(&err))));
        }
        Ok(Self { handle, info, format, cuda })
    }

    /// Denoise a hasty stack in place: `images[frame * 7 + e]` x-fastest `[X, Y, Z]`, full cardiac cycle;
    /// `kv` = π/VENC per hasty slot, `h` isotropic voxel size (m), `phase_sign` = −1 for exp(−i kv v) data.
    pub fn denoise(&self, images: &mut [C32], dims: [usize; 3], frames: usize, kv: [f32; NE], h: f32, phase_sign: i8) -> Result<(), Error> {
        self.denoise_tiled(images, dims, frames, kv, h, phase_sign, None, None)
    }

    /// Mixed precision (TorchScript on CUDA only): 0 float32, 1 bfloat16 autocast (the training regime), 2 float16.
    pub fn set_autocast(&self, mode: i32) -> Result<(), Error> {
        if unsafe { hd_set_autocast(self.handle, mode) } != 0 {
            return Err(Error::Backend(format!("autocast {mode} needs a TorchScript model on CUDA")));
        }
        Ok(())
    }

    /// [`denoise`](Self::denoise) by exact overlap tiles (`hasty_denoise::tiled`): spatial tiles of about `tile`
    /// voxels per axis and cyclic time windows keeping `keep_frames` frames each. Same result, bounded memory.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise_tiled(&self, images: &mut [C32], dims: [usize; 3], frames: usize, kv: [f32; NE], h: f32, phase_sign: i8, tile: Option<usize>, keep_frames: Option<usize>) -> Result<(), Error> {
        let (data, p) = to_model(images, dims, frames, self.info.pad_multiple, phase_sign)?;
        let kvm = kv_model(kv);
        let mut call = |x: &[f32], nt: usize, e: [usize; 3]| -> Result<Vec<f32>, Error> {
            let shape: [i64; 7] = [1, NE as i64, 2, nt as i64, e[0] as i64, e[1] as i64, e[2] as i64];
            let mut out = vec![0f32; x.len()];
            let mut err = vec![0u8; 4096];
            let rc = unsafe { hd_forward(self.handle, x.as_ptr(), shape.as_ptr(), kvm.as_ptr(), &h, out.as_mut_ptr(), err.as_mut_ptr() as *mut c_char, err.len() as c_int) };
            if rc != 0 {
                return Err(Error::Backend(err_string(&err)));
            }
            Ok(out)
        };
        let out = tiled(&data, frames, p, &self.info, tile, keep_frames, &mut call)?;
        from_model(&out, p, images, dims, frames, phase_sign)
    }
}

impl Drop for TorchDenoiser {
    fn drop(&mut self) {
        unsafe { hd_free(self.handle) }
    }
}

// ------------------------------------------------------------------------------------------------ plugin C ABI
// Used by hasty_recon::learned through dlopen. Errors: a NUL-terminated message in `err` (≤ errlen bytes).

fn put_err(err: *mut c_char, errlen: c_int, msg: &str) {
    if err.is_null() || errlen <= 0 {
        return;
    }
    let b = msg.as_bytes();
    let n = b.len().min(errlen as usize - 1);
    unsafe {
        std::ptr::copy_nonoverlapping(b.as_ptr() as *const c_char, err, n);
        *err.add(n) = 0;
    }
}

/// ABI version of the `hdt_*` functions
#[no_mangle]
pub extern "C" fn hdt_abi_version() -> u32 {
    2
}

/// open an export directory; `format` 0 TorchScript, 1 AOTInductor; `cuda` ≠ 0: device 0 when available
///
/// # Safety
/// `dir` is a NUL-terminated path; `err` has room for `errlen` bytes.
#[no_mangle]
pub unsafe extern "C" fn hdt_open(dir: *const c_char, format: c_int, cuda: c_int, err: *mut c_char, errlen: c_int) -> *mut c_void {
    let dir = std::ffi::CStr::from_ptr(dir).to_string_lossy().into_owned();
    let fmt = if format == 1 { Format::Aoti } else { Format::TorchScript };
    match TorchDenoiser::load(Path::new(&dir), fmt, cuda != 0) {
        Ok(d) => Box::into_raw(Box::new(d)) as *mut c_void,
        Err(e) => {
            put_err(err, errlen, &e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// mixed precision: 0 float32, 1 bf16 autocast, 2 fp16 autocast (TorchScript on CUDA); 0 on success
///
/// # Safety
/// `h` comes from [`hdt_open`].
#[no_mangle]
pub unsafe extern "C" fn hdt_set_autocast(h: *mut c_void, mode: c_int) -> c_int {
    (*(h as *mut TorchDenoiser)).set_autocast(mode).is_err() as c_int
}

/// 1 when the opened model runs on CUDA
///
/// # Safety
/// `h` comes from [`hdt_open`].
#[no_mangle]
pub unsafe extern "C" fn hdt_on_cuda(h: *mut c_void) -> c_int {
    (*(h as *mut TorchDenoiser)).cuda as c_int
}

/// Denoise a hasty stack in place (see [`TorchDenoiser::denoise_tiled`]): `images` = 2 × `n_complex` f32 (re, im),
/// `images[frame * 7 + e]` x-fastest `dims[0..3]`; `kv[7]` = π/VENC per hasty slot; `tile` 0 = whole volume per
/// axis; `keep_frames` 0 = the whole cycle per call, else cyclic time windows keeping that many frames. 0 on success.
///
/// # Safety
/// Pointers valid for the given sizes; `h` from [`hdt_open`]; one call per handle at a time.
#[no_mangle]
pub unsafe extern "C" fn hdt_denoise(h: *mut c_void, images: *mut f32, n_complex: usize, dims: *const usize, frames: usize, kv: *const f32, hvox: f32, phase_sign: c_int, tile: usize, keep_frames: usize, err: *mut c_char, errlen: c_int) -> c_int {
    let d = &*(h as *mut TorchDenoiser);
    let imgs = std::slice::from_raw_parts_mut(images as *mut C32, n_complex);
    let dims = [*dims, *dims.add(1), *dims.add(2)];
    let mut k = [0f32; NE];
    k.copy_from_slice(std::slice::from_raw_parts(kv, NE));
    match d.denoise_tiled(imgs, dims, frames, k, hvox, if phase_sign < 0 { -1 } else { 1 }, (tile > 0).then_some(tile), (keep_frames > 0).then_some(keep_frames)) {
        Ok(()) => 0,
        Err(e) => {
            put_err(err, errlen, &e.to_string());
            1
        }
    }
}

/// # Safety
/// `h` from [`hdt_open`], not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn hdt_close(h: *mut c_void) {
    if !h.is_null() {
        drop(Box::from_raw(h as *mut TorchDenoiser));
    }
}
