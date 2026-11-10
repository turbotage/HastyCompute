//! Dense 3-D volumes with a voxel → physical affine, `.npy` I/O, trilinear sampling.

use std::io::Read;
use std::path::Path;

use hasty_mrsim::Vec3;

use crate::Error;

/// x-fastest storage: index (i, j, k) at i + nx (j + ny k). `affine` maps voxel index → physical position (m).
#[derive(Clone, Debug)]
pub struct Volume<T> {
    pub dims: [usize; 3],
    pub affine: [[f64; 4]; 4],
    pub data: Vec<T>,
}

pub trait NpyElem: Copy + Default {
    const DESCR: &'static [&'static str];
    fn from_le(b: &[u8]) -> Self;
}
impl NpyElem for f32 {
    const DESCR: &'static [&'static str] = &["<f4"];
    fn from_le(b: &[u8]) -> Self {
        f32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }
}
impl NpyElem for u8 {
    const DESCR: &'static [&'static str] = &["|u1", "<u1", "|b1"];
    fn from_le(b: &[u8]) -> Self {
        b[0]
    }
}

impl<T: Copy + Default> Volume<T> {
    pub fn new(dims: [usize; 3], affine: [[f64; 4]; 4]) -> Self {
        Self { dims, affine, data: vec![T::default(); dims[0] * dims[1] * dims[2]] }
    }
    #[inline]
    pub fn idx(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dims[0] * (j + self.dims[1] * k)
    }
    #[inline]
    pub fn get(&self, i: usize, j: usize, k: usize) -> T {
        self.data[self.idx(i, j, k)]
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    /// voxel (fractional) → physical
    pub fn to_phys(&self, v: [f64; 3]) -> Vec3 {
        let a = &self.affine;
        [0, 1, 2].map(|r| a[r][0] * v[0] + a[r][1] * v[1] + a[r][2] * v[2] + a[r][3])
    }
    /// physical → voxel (fractional); the affine's 3x3 block is inverted on every call (cheap, general)
    pub fn to_voxel(&self, x: &Vec3) -> [f64; 3] {
        let a = &self.affine;
        let m = [[a[0][0], a[0][1], a[0][2]], [a[1][0], a[1][1], a[1][2]], [a[2][0], a[2][1], a[2][2]]];
        let inv = inv3(&m);
        let d = [x[0] - a[0][3], x[1] - a[1][3], x[2] - a[2][3]];
        [0, 1, 2].map(|r| inv[r][0] * d[0] + inv[r][1] * d[1] + inv[r][2] * d[2])
    }
    pub fn voxel_size(&self) -> Vec3 {
        let a = &self.affine;
        [0, 1, 2].map(|c| (a[0][c] * a[0][c] + a[1][c] * a[1][c] + a[2][c] * a[2][c]).sqrt())
    }
    pub fn map<U: Copy + Default>(&self, f: impl Fn(T) -> U) -> Volume<U> {
        Volume { dims: self.dims, affine: self.affine, data: self.data.iter().map(|&v| f(v)).collect() }
    }
}

pub fn inv3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let c = |r0: usize, c0: usize, r1: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    [
        [c(1, 1, 2, 2) / det, -c(0, 1, 2, 2) / det, c(0, 1, 1, 2) / det],
        [-c(1, 0, 2, 2) / det, c(0, 0, 2, 2) / det, -c(0, 0, 1, 2) / det],
        [c(1, 0, 2, 1) / det, -c(0, 0, 2, 1) / det, c(0, 0, 1, 1) / det],
    ]
}

impl Volume<f32> {
    /// trilinear interpolation at a physical point; `outside` beyond the grid
    pub fn sample(&self, x: &Vec3, outside: f32) -> f32 {
        let v = self.to_voxel(x);
        self.sample_voxel(v, outside)
    }
    pub fn sample_voxel(&self, v: [f64; 3], outside: f32) -> f32 {
        let mut i0 = [0usize; 3];
        let mut f = [0.0f64; 3];
        for a in 0..3 {
            if !(v[a] >= 0.0 && v[a] <= (self.dims[a] - 1) as f64) {
                return outside;
            }
            let fl = v[a].floor().min((self.dims[a] - 2) as f64);
            i0[a] = fl as usize;
            f[a] = v[a] - fl;
        }
        let mut acc = 0.0f64;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let w = (if dx == 1 { f[0] } else { 1.0 - f[0] }) * (if dy == 1 { f[1] } else { 1.0 - f[1] }) * (if dz == 1 { f[2] } else { 1.0 - f[2] });
                    acc += w * self.get(i0[0] + dx, i0[1] + dy, i0[2] + dz) as f64;
                }
            }
        }
        acc as f32
    }
}

/// Read a `.npy` array of shape (nx, ny, nz) into x-fastest storage (C- and Fortran-ordered files are both accepted).
pub fn read_npy<T: NpyElem>(path: &Path, affine: [[f64; 4]; 4]) -> Result<Volume<T>, Error> {
    let mut f = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    if &buf[..6] != b"\x93NUMPY" {
        return Err(Error::Format(format!("{}: not an npy file", path.display())));
    }
    let (hlen, hstart) = if buf[6] == 1 { (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10) } else { (u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize, 12) };
    let header = std::str::from_utf8(&buf[hstart..hstart + hlen]).map_err(|e| Error::Format(e.to_string()))?;
    let descr = header.split("'descr':").nth(1).and_then(|s| s.split('\'').nth(1)).unwrap_or("");
    if !T::DESCR.contains(&descr) {
        return Err(Error::Format(format!("{}: dtype {descr} not one of {:?}", path.display(), T::DESCR)));
    }
    let fortran = header.contains("'fortran_order': True");
    let shape_s = header.split("'shape':").nth(1).and_then(|s| s.split('(').nth(1)).and_then(|s| s.split(')').next()).unwrap_or("");
    let shape: Vec<usize> = shape_s.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    if shape.len() != 3 {
        return Err(Error::Format(format!("{}: expected a 3-D array, shape {shape:?}", path.display())));
    }
    let dims = [shape[0], shape[1], shape[2]];
    let es = std::mem::size_of::<T>();
    let body = &buf[hstart + hlen..];
    let n = dims[0] * dims[1] * dims[2];
    if body.len() < n * es {
        return Err(Error::Format(format!("{}: truncated", path.display())));
    }
    let mut v = Volume::<T>::new(dims, affine);
    if fortran {
        for (q, d) in v.data.iter_mut().enumerate() {
            *d = T::from_le(&body[q * es..]);
        }
    } else {
        for i in 0..dims[0] {
            for j in 0..dims[1] {
                for k in 0..dims[2] {
                    let src = (i * dims[1] + j) * dims[2] + k;
                    let dst = i + dims[0] * (j + dims[1] * k);
                    v.data[dst] = T::from_le(&body[src * es..]);
                }
            }
        }
    }
    Ok(v)
}

/// Write an f32 volume as a Fortran-ordered `.npy` (shape (nx, ny, nz)); for inspection from Python.
pub fn write_npy_f32(path: &Path, v: &Volume<f32>) -> Result<(), Error> {
    let bytes: Vec<u8> = v.data.iter().flat_map(|x| x.to_le_bytes()).collect();
    write_npy(path, "<f4", &v.dims, true, &bytes)
}

/// Write raw little-endian element bytes as `.npy` with the given dtype descriptor (`<f4`, `<c8`, `<f8`, `|u1`, …).
/// `fortran = true`: first index fastest.
pub fn write_npy(path: &Path, descr: &str, shape: &[usize], fortran: bool, bytes: &[u8]) -> Result<(), Error> {
    let mut out = npy_header(descr, shape, fortran);
    out.extend_from_slice(bytes);
    std::fs::write(path, out)?;
    Ok(())
}

/// the `.npy` header (magic, version 1.0, dict padded to a multiple of 64 bytes) for streaming writers: the element
/// bytes follow it, row-major (or first index fastest with `fortran`)
pub fn npy_header(descr: &str, shape: &[usize], fortran: bool) -> Vec<u8> {
    let shp = if shape.len() == 1 { format!("({},)", shape[0]) } else { format!("({})", shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")) };
    let mut header = format!("{{'descr': '{descr}', 'fortran_order': {}, 'shape': {shp}, }}", if fortran { "True" } else { "False" });
    let total = 10 + header.len() + 1;
    let pad = (64 - total % 64) % 64;
    header.push_str(&" ".repeat(pad));
    header.push('\n');
    let mut out = Vec::with_capacity(10 + header.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(header.len() as u16).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out
}
