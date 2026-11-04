use num_complex::{Complex32, Complex64};

use crate::{Error, Result};

/// Element type. The discriminant is the wire code (FORMAT.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum DType {
    Bool = 0,
    U8 = 1,
    I8 = 2,
    U16 = 3,
    I16 = 4,
    U32 = 5,
    I32 = 6,
    U64 = 7,
    I64 = 8,
    F16 = 9,
    BF16 = 10,
    F32 = 11,
    F64 = 12,
    C64 = 13,
    C128 = 14,
}

const ALL: [DType; 15] = [
    DType::Bool,
    DType::U8,
    DType::I8,
    DType::U16,
    DType::I16,
    DType::U32,
    DType::I32,
    DType::U64,
    DType::I64,
    DType::F16,
    DType::BF16,
    DType::F32,
    DType::F64,
    DType::C64,
    DType::C128,
];

impl DType {
    pub fn from_code(c: u8) -> Result<Self> {
        ALL.get(c as usize).copied().ok_or_else(|| Error::Format(format!("unknown dtype code {c}")))
    }

    /// bytes per element
    pub fn size(self) -> usize {
        match self {
            DType::Bool | DType::U8 | DType::I8 => 1,
            DType::U16 | DType::I16 | DType::F16 | DType::BF16 => 2,
            DType::U32 | DType::I32 | DType::F32 => 4,
            DType::U64 | DType::I64 | DType::F64 | DType::C64 => 8,
            DType::C128 => 16,
        }
    }

    /// word size for the byte-shuffle filter: complex numbers shuffle as their real/imag components
    pub fn word(self) -> usize {
        match self {
            DType::C64 => 4,
            DType::C128 => 8,
            d => d.size(),
        }
    }

    pub fn is_integer(self) -> bool {
        matches!(self, DType::Bool | DType::U8 | DType::I8 | DType::U16 | DType::I16 | DType::U32 | DType::I32 | DType::U64 | DType::I64)
    }

    pub fn is_complex(self) -> bool {
        matches!(self, DType::C64 | DType::C128)
    }

    /// numpy-style name
    pub fn name(self) -> &'static str {
        match self {
            DType::Bool => "bool",
            DType::U8 => "uint8",
            DType::I8 => "int8",
            DType::U16 => "uint16",
            DType::I16 => "int16",
            DType::U32 => "uint32",
            DType::I32 => "int32",
            DType::U64 => "uint64",
            DType::I64 => "int64",
            DType::F16 => "float16",
            DType::BF16 => "bfloat16",
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::C64 => "complex64",
            DType::C128 => "complex128",
        }
    }

    pub fn from_name(s: &str) -> Result<Self> {
        ALL.iter().copied().find(|d| d.name() == s).ok_or_else(|| Error::Type(format!("unknown dtype {s}")))
    }
}

/// Plain-old-data element types with a [`DType`].
///
/// # Safety
/// Implementors must be `Copy` types without padding or invalid bit patterns, of size `DTYPE.size()`.
pub unsafe trait Element: Copy + Send + Sync + 'static {
    const DTYPE: DType;
}

macro_rules! element {
    ($($t:ty => $d:ident),*) => { $(unsafe impl Element for $t { const DTYPE: DType = DType::$d; })* };
}
element!(u8 => U8, i8 => I8, u16 => U16, i16 => I16, u32 => U32, i32 => I32, u64 => U64, i64 => I64, f32 => F32, f64 => F64, Complex32 => C64, Complex64 => C128);
