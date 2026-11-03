use std::{
    ffi::{c_char, CStr, CString},
    fs,
    path::PathBuf,
    ptr,
};

use cudarc::nvrtc::sys;

use crate::{Error, Result};

#[derive(Clone, Debug, Default)]
pub struct CompileSettings {
    /// Extra NVRTC options, e.g. `-DRADIX=4`, `--use_fast_math`.
    pub options: Vec<String>,
    /// Skip the disk cache (still dumps if `HASTY_DUMP_KERNELS` is set).
    pub no_cache: bool,
}

/// FNV-1a, 64-bit. Stable across Rust versions (unlike `DefaultHasher`).
fn fnv64(seed: u64, bytes: &[u8]) -> u64 {
    let mut h = seed;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn nvrtc_version() -> (i32, i32) {
    let (mut a, mut b) = (0, 0);
    unsafe {
        let _ = sys::nvrtcVersion(&mut a, &mut b);
    }
    (a, b)
}

fn cache_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("HASTY_KERNEL_CACHE") {
        return Some(PathBuf::from(d));
    }
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".cache/hasty/kernels"))
}

/// Compile CUDA C++ `src` to a cubin for `arch` (e.g. `sm_89`).
///
/// Cache key = hash(source, arch, options, NVRTC version). Setting `HASTY_DUMP_KERNELS=dir`
/// writes every generated source to `dir/<name>-<hash>.cu`.
pub fn compile_cubin(
    name: &str,
    src: &str,
    arch: &str,
    settings: &CompileSettings,
) -> Result<Vec<u8>> {
    let (maj, min) = nvrtc_version();
    let mut h = fnv64(0xcbf29ce484222325, src.as_bytes());
    h = fnv64(h, arch.as_bytes());
    for o in &settings.options {
        h = fnv64(h, o.as_bytes());
    }
    h = fnv64(h, format!("nvrtc{maj}.{min}").as_bytes());
    let hash = format!("{h:016x}");

    if let Ok(dir) = std::env::var("HASTY_DUMP_KERNELS") {
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(PathBuf::from(dir).join(format!("{name}-{hash}.cu")), src);
    }

    let cache_file = cache_dir().map(|d| d.join(format!("{name}-{hash}.cubin")));
    if !settings.no_cache {
        if let Some(f) = &cache_file {
            if let Ok(bytes) = fs::read(f) {
                return Ok(bytes);
            }
        }
    }

    let err = |log: String| Error::Nvrtc {
        name: name.to_owned(),
        hash: hash.clone(),
        log,
    };
    let csrc = CString::new(src).map_err(|_| err("source contains NUL".into()))?;
    let cname = CString::new(format!("{name}.cu")).unwrap();

    let mut opts: Vec<CString> = vec![
        CString::new(format!("--gpu-architecture={arch}")).unwrap(),
        CString::new("--std=c++17").unwrap(),
    ];
    for o in &settings.options {
        opts.push(CString::new(o.as_str()).map_err(|_| err("option contains NUL".into()))?);
    }
    let opt_ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();

    unsafe {
        let mut prog: sys::nvrtcProgram = ptr::null_mut();
        let r = sys::nvrtcCreateProgram(
            &mut prog,
            csrc.as_ptr(),
            cname.as_ptr(),
            0,
            ptr::null(),
            ptr::null(),
        );
        if r != sys::nvrtcResult::NVRTC_SUCCESS {
            return Err(err(format!("nvrtcCreateProgram: {r:?}")));
        }
        let r = sys::nvrtcCompileProgram(prog, opt_ptrs.len() as i32, opt_ptrs.as_ptr());

        let mut log_size = 0usize;
        let _ = sys::nvrtcGetProgramLogSize(prog, &mut log_size);
        let mut log = vec![0 as c_char; log_size.max(1)];
        let _ = sys::nvrtcGetProgramLog(prog, log.as_mut_ptr());
        let log_str = CStr::from_ptr(log.as_ptr()).to_string_lossy().into_owned();

        if r != sys::nvrtcResult::NVRTC_SUCCESS {
            let _ = sys::nvrtcDestroyProgram(&mut prog);
            return Err(err(format!("{r:?}\n{log_str}")));
        }

        let mut size = 0usize;
        let r = sys::nvrtcGetCUBINSize(prog, &mut size);
        if r != sys::nvrtcResult::NVRTC_SUCCESS || size == 0 {
            let _ = sys::nvrtcDestroyProgram(&mut prog);
            return Err(err(format!(
                "nvrtcGetCUBINSize: {r:?} size={size} (is `{arch}` a real sm_ arch?)"
            )));
        }
        let mut cubin = vec![0u8; size];
        let r = sys::nvrtcGetCUBIN(prog, cubin.as_mut_ptr() as *mut c_char);
        let _ = sys::nvrtcDestroyProgram(&mut prog);
        if r != sys::nvrtcResult::NVRTC_SUCCESS {
            return Err(err(format!("nvrtcGetCUBIN: {r:?}")));
        }

        if let Some(f) = &cache_file {
            if let Some(p) = f.parent() {
                let _ = fs::create_dir_all(p);
            }
            // write-then-rename so concurrent processes never see a partial file
            let tmp = f.with_extension(format!("tmp{}", std::process::id()));
            if fs::write(&tmp, &cubin).is_ok() {
                let _ = fs::rename(&tmp, f);
            }
        }
        Ok(cubin)
    }
}
