//! `.npy` read / write (N-D, little-endian; C or Fortran order). A Fortran-ordered array of shape (X, Y, Z) is
//! returned as the C-order tensor (Z, Y, X) with the same bytes (no copy), i.e. hasty's x-fastest layout; `fortran`
//! in the result says it happened.

use std::path::Path;

use crate::{Buffer, DType, Error, Result, Tensor};

fn dtype_of(descr: &str) -> Result<DType> {
    let d = descr.trim_start_matches(['<', '|', '=']);
    if descr.starts_with('>') {
        return Err(Error::Format(format!("big-endian npy {descr} not supported")));
    }
    Ok(match d {
        "b1" => DType::Bool,
        "u1" => DType::U8,
        "i1" => DType::I8,
        "u2" => DType::U16,
        "i2" => DType::I16,
        "u4" => DType::U32,
        "i4" => DType::I32,
        "u8" => DType::U64,
        "i8" => DType::I64,
        "f2" => DType::F16,
        "f4" => DType::F32,
        "f8" => DType::F64,
        "c8" => DType::C64,
        "c16" => DType::C128,
        _ => return Err(Error::Format(format!("unsupported npy dtype {descr}"))),
    })
}

fn descr_of(d: DType) -> Result<&'static str> {
    Ok(match d {
        DType::Bool => "|b1",
        DType::U8 => "|u1",
        DType::I8 => "|i1",
        DType::U16 => "<u2",
        DType::I16 => "<i2",
        DType::U32 => "<u4",
        DType::I32 => "<i4",
        DType::U64 => "<u8",
        DType::I64 => "<i8",
        DType::F16 => "<f2",
        DType::F32 => "<f4",
        DType::F64 => "<f8",
        DType::C64 => "<c8",
        DType::C128 => "<c16",
        DType::BF16 => return Err(Error::Format("bfloat16 has no npy dtype".into())),
    })
}

/// (dtype, numpy shape, fortran) from the bytes of an npy file (at least its header); the header length is
/// `10 + u16` (v1) or `12 + u32` (v2 / v3)
fn parse_header(bytes: &[u8]) -> Result<(DType, Vec<usize>, bool)> {
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(Error::Format("not an npy file".into()));
    }
    let (hlen, off) = header_len(bytes)?;
    if bytes.len() < off + hlen {
        return Err(Error::Format("npy header truncated".into()));
    }
    let header = std::str::from_utf8(&bytes[off..off + hlen]).map_err(|_| Error::Format("npy header".into()))?;
    let field = |k: &str| -> Option<&str> {
        let i = header.find(&format!("'{k}'"))? + k.len() + 2;
        Some(header[i..].trim_start_matches([':', ' ']))
    };
    let descr = field("descr").and_then(|s| s.split('\'').nth(1)).ok_or_else(|| Error::Format("npy descr".into()))?;
    let fortran = field("fortran_order").map(|s| s.starts_with("True")).unwrap_or(false);
    let sh = field("shape").ok_or_else(|| Error::Format("npy shape".into()))?;
    let sh = &sh[sh.find('(').unwrap_or(0) + 1..sh.find(')').unwrap_or(sh.len())];
    let shape: Vec<usize> = sh.split(',').map(str::trim).filter(|s| !s.is_empty()).map(|s| s.parse().map_err(|_| Error::Format(format!("npy shape {sh}")))).collect::<Result<_>>()?;
    Ok((dtype_of(descr)?, shape, fortran))
}

fn header_len(bytes: &[u8]) -> Result<(usize, usize)> {
    Ok(match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12),
        v => return Err(Error::Format(format!("npy version {v}"))),
    })
}

/// (tensor, the file was Fortran-ordered)
pub fn parse(bytes: &[u8]) -> Result<(Tensor, bool)> {
    let (dtype, mut shape, fortran) = parse_header(bytes)?;
    let (hlen, off) = header_len(bytes)?;
    let n: usize = shape.iter().product::<usize>() * dtype.size();
    let data = &bytes[off + hlen..];
    if data.len() < n {
        return Err(Error::Format(format!("npy data: {} of {n} bytes", data.len())));
    }
    if fortran {
        shape.reverse();
    }
    Ok((Tensor::new(dtype, shape, Buffer::from_bytes(&data[..n]))?, fortran))
}

pub fn read(path: &Path) -> Result<(Tensor, bool)> {
    parse(&std::fs::read(path)?)
}

/// write a C-ordered npy
pub fn write(path: &Path, t: &Tensor) -> Result<()> {
    write_raw(path, t.dtype(), t.shape(), false, t.bytes())
}

/// the npy header (v1) for an array of `shape` (as numpy reports it) and memory order
pub fn header_bytes(dtype: DType, shape: &[usize], fortran: bool) -> Result<Vec<u8>> {
    let shape = match shape.len() {
        1 => format!("({},)", shape[0]),
        _ => format!("({})", shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ")),
    };
    let mut h = format!("{{'descr': '{}', 'fortran_order': {}, 'shape': {shape}, }}", descr_of(dtype)?, if fortran { "True" } else { "False" });
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut out = Vec::with_capacity(10 + h.len());
    out.extend_from_slice(b"\x93NUMPY\x01\x00");
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    Ok(out)
}

/// write raw little-endian bytes as an npy of `shape` (numpy's shape; `fortran`: first index fastest), streamed
pub fn write_raw(path: &Path, dtype: DType, shape: &[usize], fortran: bool, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut w = std::io::BufWriter::with_capacity(8 << 20, std::fs::File::create(path)?);
    w.write_all(&header_bytes(dtype, shape, fortran)?)?;
    w.write_all(bytes)?;
    w.flush()?;
    Ok(())
}

/// ((dtype, numpy shape, fortran), byte offset of the data) of an npy file, reading only its header
pub fn header(path: &Path) -> Result<((DType, Vec<usize>, bool), u64)> {
    use std::os::unix::fs::FileExt;
    let f = std::fs::File::open(path)?;
    let mut pre = [0u8; 12];
    f.read_exact_at(&mut pre, 0)?;
    if &pre[..6] != b"\x93NUMPY" {
        return Err(Error::Format(format!("{}: not an npy file", path.display())));
    }
    let (hlen, off) = match pre[6] {
        1 => (u16::from_le_bytes([pre[8], pre[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes(pre[8..12].try_into().unwrap()) as usize, 12),
        v => return Err(Error::Format(format!("npy version {v}"))),
    };
    let mut hb = vec![0u8; hlen];
    f.read_exact_at(&mut hb, off as u64)?;
    let mut fake = Vec::with_capacity(off + hlen);
    fake.extend_from_slice(&pre[..off]);
    fake.extend_from_slice(&hb);
    let (d, s, fo) = parse_header(&fake)?;
    Ok(((d, s, fo), (off + hlen) as u64))
}
