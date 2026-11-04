//! C ABI over [`hasty_value::LazyFile`] for fast random-access reads from other languages (Python: ctypes in
//! `hastycompute.value_format`). Each open file owns its thread pool, created by the opening process, so handles
//! opened after a fork (DataLoader workers) never touch a pool inherited from the parent.
//!
//! Calls return 0 on success, −1 on error (`hv_last_error` gives the message, per thread).

use std::cell::RefCell;
use std::ffi::{c_char, CStr};

use hasty_value::slice::{parse, Segment, Sel};
use hasty_value::{Lazy, LazyFile, TensorRef};

pub struct Handle {
    file: LazyFile,
    pool: rayon::ThreadPool,
}

thread_local! {
    static LAST: RefCell<String> = const { RefCell::new(String::new()) };
}

fn fail(m: impl Into<String>) -> i32 {
    LAST.with(|l| *l.borrow_mut() = m.into());
    -1
}

fn tensor<'a>(h: &'a Handle, key: *const c_char) -> Result<&'a TensorRef, String> {
    let key = unsafe { CStr::from_ptr(key) }.to_str().map_err(|e| e.to_string())?;
    let mut cur: &Lazy = &h.file.root;
    for seg in parse(key).map_err(|e| e.to_string())? {
        cur = match (seg, cur) {
            (Segment::Key(k), Lazy::Dict(_)) => cur.get(&k).ok_or_else(|| format!("no key {k:?}"))?,
            (Segment::Items(it), Lazy::List(v) | Lazy::Tuple(v)) => match it.as_slice() {
                [Sel::Index(i)] => v.get(*i as usize).ok_or_else(|| format!("index {i} out of range"))?,
                _ => return Err("lists take one integer index".into()),
            },
            (s, _) => return Err(format!("cannot apply {s:?}")),
        };
    }
    cur.tensor().ok_or_else(|| format!("{key:?} is not a tensor"))
}

/// open `path` with `threads` decoding threads (0: all cores); null on error
#[no_mangle]
pub extern "C" fn hv_open(path: *const c_char, threads: u32) -> *mut Handle {
    let r = (|| -> Result<Handle, String> {
        let p = unsafe { CStr::from_ptr(path) }.to_str().map_err(|e| e.to_string())?;
        let file = LazyFile::open(std::path::Path::new(p)).map_err(|e| e.to_string())?;
        let pool = rayon::ThreadPoolBuilder::new().num_threads(threads as usize).build().map_err(|e| e.to_string())?;
        Ok(Handle { file, pool })
    })();
    match r {
        Ok(h) => Box::into_raw(Box::new(h)),
        Err(e) => {
            fail(e);
            std::ptr::null_mut()
        }
    }
}

#[no_mangle]
pub extern "C" fn hv_close(h: *mut Handle) {
    if !h.is_null() {
        drop(unsafe { Box::from_raw(h) });
    }
}

/// dtype code and shape (up to `cap` dims) of the tensor at slice path `key` (e.g. `["x"]`)
#[no_mangle]
pub extern "C" fn hv_tensor_info(h: *const Handle, key: *const c_char, dtype: *mut u8, ndim: *mut u32, shape: *mut u64, cap: u32) -> i32 {
    let h = unsafe { &*h };
    match tensor(h, key) {
        Ok(t) => {
            if t.shape.len() > cap as usize {
                return fail("too many dimensions");
            }
            unsafe {
                *dtype = t.dtype as u8;
                *ndim = t.shape.len() as u32;
                for (d, &n) in t.shape.iter().enumerate() {
                    *shape.add(d) = n as u64;
                }
            }
            0
        }
        Err(e) => fail(e),
    }
}

/// read the box [lo, hi) of the tensor at `key` into `out` (C-order, `out_len` bytes)
#[no_mangle]
pub extern "C" fn hv_read_box(h: *const Handle, key: *const c_char, lo: *const u64, hi: *const u64, out: *mut u8, out_len: u64) -> i32 {
    let h = unsafe { &*h };
    let t = match tensor(h, key) {
        Ok(t) => t,
        Err(e) => return fail(e),
    };
    let nd = t.shape.len();
    let lo: Vec<usize> = (0..nd).map(|d| unsafe { *lo.add(d) } as usize).collect();
    let hi: Vec<usize> = (0..nd).map(|d| unsafe { *hi.add(d) } as usize).collect();
    match h.pool.install(|| h.file.read_box(t, &lo, &hi)) {
        Ok(b) => {
            if b.nbytes() as u64 != out_len {
                return fail(format!("output buffer has {out_len} bytes, box needs {}", b.nbytes()));
            }
            unsafe { std::ptr::copy_nonoverlapping(b.bytes().as_ptr(), out, b.nbytes()) };
            0
        }
        Err(e) => fail(e.to_string()),
    }
}

/// copy the last error message of this thread into `buf`
#[no_mangle]
pub extern "C" fn hv_last_error(buf: *mut c_char, len: u32) -> i32 {
    LAST.with(|l| {
        let s = l.borrow();
        let n = s.len().min(len.saturating_sub(1) as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(s.as_ptr() as *const c_char, buf, n);
            *buf.add(n) = 0;
        }
        n as i32
    })
}
