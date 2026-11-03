//! Interleaved complex scalar types that can live in cudarc device buffers.

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

macro_rules! cx_type {
    ($name:ident, $t:ty) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, Default, PartialEq)]
        pub struct $name {
            pub re: $t,
            pub im: $t,
        }
        impl $name {
            pub const fn new(re: $t, im: $t) -> Self {
                Self { re, im }
            }
        }
        unsafe impl DeviceRepr for $name {}
        unsafe impl ValidAsZeroBits for $name {}
    };
}

cx_type!(C32, f32);
cx_type!(C64, f64);

/// Floating-point precision of a transform: ties the real type, its interleaved complex type and the CUDA C++ spelling together.
pub trait Real: Copy + Default + DeviceRepr + ValidAsZeroBits + Send + Sync + 'static {
    type Complex: DeviceRepr + ValidAsZeroBits + Copy + Default + Send + Sync + 'static;
    /// CUDA C++ scalar type name.
    const CTYPE: &'static str;
    const IS_F64: bool;
    const BYTES: usize;
    fn complex(re: f64, im: f64) -> Self::Complex;
    fn parts(c: Self::Complex) -> (f64, f64);
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Real for f32 {
    type Complex = C32;
    const CTYPE: &'static str = "float";
    const IS_F64: bool = false;
    const BYTES: usize = 4;
    fn complex(re: f64, im: f64) -> C32 {
        C32::new(re as f32, im as f32)
    }
    fn parts(c: C32) -> (f64, f64) {
        (c.re as f64, c.im as f64)
    }
    fn from_f64(x: f64) -> f32 {
        x as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Real for f64 {
    type Complex = C64;
    const CTYPE: &'static str = "double";
    const IS_F64: bool = true;
    const BYTES: usize = 8;
    fn complex(re: f64, im: f64) -> C64 {
        C64::new(re, im)
    }
    fn parts(c: C64) -> (f64, f64) {
        (c.re, c.im)
    }
    fn from_f64(x: f64) -> f64 {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
}
