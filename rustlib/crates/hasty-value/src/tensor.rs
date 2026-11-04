use crate::{DType, Element, Error, Result};

/// 8-byte aligned byte buffer (so that every element type can be viewed in place).
#[derive(Clone, Default, PartialEq, Eq)]
pub struct Buffer {
    words: Vec<u64>,
    len: usize,
}

impl Buffer {
    pub fn zeroed(len: usize) -> Self {
        Self { words: vec![0u64; len.div_ceil(8)], len }
    }

    pub fn from_bytes(b: &[u8]) -> Self {
        let mut s = Self::zeroed(b.len());
        s.as_bytes_mut().copy_from_slice(b);
        s
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: the words cover at least `len` initialised bytes
        unsafe { std::slice::from_raw_parts(self.words.as_ptr() as *const u8, self.len) }
    }

    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above, uniquely borrowed
        unsafe { std::slice::from_raw_parts_mut(self.words.as_mut_ptr() as *mut u8, self.len) }
    }
}

impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Buffer({} bytes)", self.len)
    }
}

/// Contiguous host tensor, C-order (last index fastest), little-endian.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    dtype: DType,
    shape: Vec<usize>,
    data: Buffer,
}

fn as_bytes<T: Element>(v: &[T]) -> &[u8] {
    // SAFETY: Element types are padding-free plain data
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

impl Tensor {
    pub fn new(dtype: DType, shape: Vec<usize>, data: Buffer) -> Result<Self> {
        let n: usize = shape.iter().product();
        if data.len() != n * dtype.size() {
            return Err(Error::Type(format!("tensor {shape:?} {} needs {} bytes, got {}", dtype.name(), n * dtype.size(), data.len())));
        }
        Ok(Self { dtype, shape, data })
    }

    pub fn from_bytes(dtype: DType, shape: Vec<usize>, bytes: &[u8]) -> Result<Self> {
        Self::new(dtype, shape, Buffer::from_bytes(bytes))
    }

    pub fn zeros(dtype: DType, shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self { dtype, shape, data: Buffer::zeroed(n * dtype.size()) }
    }

    /// C-order tensor of shape `shape` from elements (copied).
    pub fn from_slice<T: Element>(shape: &[usize], v: &[T]) -> Result<Self> {
        Self::from_bytes(T::DTYPE, shape.to_vec(), as_bytes(v))
    }

    /// hasty layout: `dims` fastest first (x, y, z, ...), i.e. C-order shape reversed.
    pub fn from_fortran<T: Element>(dims: &[usize], v: &[T]) -> Result<Self> {
        let shape: Vec<usize> = dims.iter().rev().copied().collect();
        Self::from_slice(&shape, v)
    }

    pub fn scalar<T: Element>(v: T) -> Self {
        Self::from_slice(&[], &[v]).expect("scalar")
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn nbytes(&self) -> usize {
        self.data.len()
    }

    pub fn bytes(&self) -> &[u8] {
        self.data.as_bytes()
    }

    pub fn bytes_mut(&mut self) -> &mut [u8] {
        self.data.as_bytes_mut()
    }

    pub fn into_parts(self) -> (DType, Vec<usize>, Buffer) {
        (self.dtype, self.shape, self.data)
    }

    pub fn reshape(mut self, shape: Vec<usize>) -> Result<Self> {
        if shape.iter().product::<usize>() != self.numel() {
            return Err(Error::Type(format!("cannot reshape {:?} to {shape:?}", self.shape)));
        }
        self.shape = shape;
        Ok(self)
    }

    fn check<T: Element>(&self) -> Result<()> {
        if T::DTYPE != self.dtype {
            return Err(Error::Type(format!("tensor is {}, requested {}", self.dtype.name(), T::DTYPE.name())));
        }
        Ok(())
    }

    /// In-place element view (the buffer is 8-byte aligned).
    pub fn as_slice<T: Element>(&self) -> Result<&[T]> {
        self.check::<T>()?;
        let b = self.data.as_bytes();
        // SAFETY: dtype checked, buffer aligned to 8 ≥ align_of::<T>() for every Element, length a multiple of size
        Ok(unsafe { std::slice::from_raw_parts(b.as_ptr() as *const T, self.numel()) })
    }

    pub fn as_slice_mut<T: Element>(&mut self) -> Result<&mut [T]> {
        self.check::<T>()?;
        let n = self.numel();
        let b = self.data.as_bytes_mut();
        // SAFETY: as above
        Ok(unsafe { std::slice::from_raw_parts_mut(b.as_mut_ptr() as *mut T, n) })
    }

    pub fn to_vec<T: Element>(&self) -> Result<Vec<T>> {
        Ok(self.as_slice::<T>()?.to_vec())
    }

    /// Element `i` (flat, C-order) converted to f64 components (re, im); for display / statistics.
    pub fn get_f64(&self, i: usize) -> (f64, f64) {
        let b = &self.bytes()[i * self.dtype.size()..(i + 1) * self.dtype.size()];
        let f = |n: usize| -> [u8; 8] {
            let mut a = [0u8; 8];
            a[..n].copy_from_slice(&b[..n]);
            a
        };
        match self.dtype {
            DType::Bool | DType::U8 => (b[0] as f64, 0.0),
            DType::I8 => (b[0] as i8 as f64, 0.0),
            DType::U16 => (u16::from_le_bytes([b[0], b[1]]) as f64, 0.0),
            DType::I16 => (i16::from_le_bytes([b[0], b[1]]) as f64, 0.0),
            DType::U32 => (u32::from_le_bytes(b.try_into().unwrap()) as f64, 0.0),
            DType::I32 => (i32::from_le_bytes(b.try_into().unwrap()) as f64, 0.0),
            DType::U64 => (u64::from_le_bytes(f(8)) as f64, 0.0),
            DType::I64 => (i64::from_le_bytes(f(8)) as f64, 0.0),
            DType::F16 => (f16_to_f64(u16::from_le_bytes([b[0], b[1]])), 0.0),
            DType::BF16 => (f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16) as f64, 0.0),
            DType::F32 => (f32::from_le_bytes(b.try_into().unwrap()) as f64, 0.0),
            DType::F64 => (f64::from_le_bytes(b.try_into().unwrap()), 0.0),
            DType::C64 => (f32::from_le_bytes(b[0..4].try_into().unwrap()) as f64, f32::from_le_bytes(b[4..8].try_into().unwrap()) as f64),
            DType::C128 => (f64::from_le_bytes(b[0..8].try_into().unwrap()), f64::from_le_bytes(b[8..16].try_into().unwrap())),
        }
    }
}

/// IEEE half → f64
pub(crate) fn f16_to_f64(h: u16) -> f64 {
    let s = if h >> 15 == 1 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    match e {
        0 => s * m * 2f64.powi(-24),
        31 => {
            if m == 0.0 {
                s * f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => s * (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}
