//! Host-side packing of k-space multipliers for [`crate::KernelFormat`].

/// IEEE binary16 bits of `x`, round to nearest even; saturates to +-inf above 65504.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let mid = 1 << (shift - 1);
        let r = if rem > mid || (rem == mid && (half & 1) == 1) {
            half + 1
        } else {
            half
        };
        return sign | r as u16;
    }
    let half = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    let r = if rem > 0x1000 || (rem == 0x1000 && (half & 1) == 1) {
        half + 1
    } else {
        half
    };
    sign | r as u16
}

/// bfloat16 bits of `x`, round to nearest even.
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if (b & 0x7fff_ffff) > 0x7f80_0000 {
        return ((b >> 16) | 0x40) as u16;
    }
    let lsb = (b >> 16) & 1;
    ((b.wrapping_add(0x7fff + lsb)) >> 16) as u16
}

pub fn pack_half(re: f32, im: f32) -> u32 {
    f32_to_f16_bits(re) as u32 | ((f32_to_f16_bits(im) as u32) << 16)
}

pub fn pack_bf16(re: f32, im: f32) -> u32 {
    f32_to_bf16_bits(re) as u32 | ((f32_to_bf16_bits(im) as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn f16_to_f32(h: u16) -> f32 {
        let s = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let e = ((h >> 10) & 0x1f) as i32;
        let m = (h & 0x3ff) as f32;
        if e == 0 {
            s * m * 2f32.powi(-24)
        } else if e == 31 {
            s * f32::INFINITY
        } else {
            s * (1.0 + m / 1024.0) * 2f32.powi(e - 15)
        }
    }
    #[test]
    fn half_roundtrip_error() {
        for i in 0..2000 {
            let x = ((i as f32) * 0.37 - 300.0) * 1.01;
            if x.abs() < 6.2e-5 {
                continue;
            }
            let y = f16_to_f32(f32_to_f16_bits(x));
            assert!(((y - x) / x).abs() <= 4.9e-4 + 1e-7, "{x} -> {y}");
        }
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(70000.0), 0x7c00);
    }
    #[test]
    fn bf16_basic() {
        assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
        let x = 3.14159f32;
        let y = f32::from_bits((f32_to_bf16_bits(x) as u32) << 16);
        assert!(((y - x) / x).abs() < 4e-3);
    }
}
