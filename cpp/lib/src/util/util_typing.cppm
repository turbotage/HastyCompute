module;

#include "pch.hpp"

export module util_mod:typing;

import std;
import :meta;
// Uggly additions for the float types until stdfloat is in for clang compiler

namespace hasty {
namespace _convert {
    
std::uint16_t double_to_float16(double d) {
    // Naive: clamp to range, no rounding
    if (d == 0.0) return 0;
    if (std::isnan(d)) return 0x7e00;
    if (std::isinf(d)) return d < 0 ? 0xfc00 : 0x7c00;
    int sign = d < 0 ? 1 : 0;
    d = std::fabs(d);
    int exp;
    double frac = std::frexp(d, &exp);
    exp += 14;
    if (exp <= 0) return sign << 15; // underflow
    if (exp >= 31) return (sign << 15) | (0x1f << 10); // overflow
    std::uint16_t mant = (std::uint16_t)(frac * 1024) & 0x3ff;
    return (sign << 15) | ((exp & 0x1f) << 10) | mant;
}
double float16_to_double(std::uint16_t h) {
    // IEEE 754 half-precision to double
    std::uint16_t sign = (h >> 15) & 0x1;
    std::uint16_t exp = (h >> 10) & 0x1F;
    std::uint16_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0 : 0.0;
        return (sign ? -1 : 1) * std::ldexp((double)mant, -24);
    }
    if (exp == 31) {
        return mant ? std::numeric_limits<double>::quiet_NaN() : (sign ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity());
    }
    return (sign ? -1 : 1) * std::ldexp((double)(mant | 0x400), exp - 25);
}
double bfloat16_to_double(std::uint16_t h) {
    // bfloat16: just upper 16 bits of float
    std::uint32_t bits = ((std::uint32_t)h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return (double)f;
}
std::uint16_t double_to_bfloat16(double d) {
    float f = static_cast<float>(d);
    std::uint32_t bits;
    std::memcpy(&bits, &f, sizeof(f));
    return (std::uint16_t)(bits >> 16);
}

}
}

namespace std {
    struct float16_t {
        std::uint16_t data;

        float16_t() : data(0) {}
        explicit float16_t(std::uint16_t d) : data(d) {}
        explicit float16_t(double d) : data(hasty::_convert::double_to_float16(d)) {}

        double to_double() const { return hasty::_convert::float16_to_double(data); }

        // Arithmetic operators
        float16_t operator+(const float16_t& other) const { return float16_t(to_double() + other.to_double()); }
        float16_t operator-(const float16_t& other) const { return float16_t(to_double() - other.to_double()); }
        float16_t operator*(const float16_t& other) const { return float16_t(to_double() * other.to_double()); }
        float16_t operator/(const float16_t& other) const { return float16_t(to_double() / other.to_double()); }

        // Comparison operators
        bool operator==(const float16_t& other) const { return data == other.data; }
        bool operator!=(const float16_t& other) const { return data != other.data; }

    };

    struct bfloat16_t {
        std::uint16_t data;

        bfloat16_t() : data(0) {}
        explicit bfloat16_t(std::uint16_t d) : data(d) {}
        explicit bfloat16_t(double d) : data(hasty::_convert::double_to_bfloat16(d)) {}

        double to_double() const { return hasty::_convert::bfloat16_to_double(data); }

        // Arithmetic operators
        bfloat16_t operator+(const bfloat16_t& other) const { return bfloat16_t(to_double() + other.to_double()); }
        bfloat16_t operator-(const bfloat16_t& other) const { return bfloat16_t(to_double() - other.to_double()); }
        bfloat16_t operator*(const bfloat16_t& other) const { return bfloat16_t(to_double() * other.to_double()); }
        bfloat16_t operator/(const bfloat16_t& other) const { return bfloat16_t(to_double() / other.to_double()); }

        // Comparison operators
        bool operator==(const bfloat16_t& other) const { return data == other.data; }
        bool operator!=(const bfloat16_t& other) const { return data != other.data; }

        // Conversion helpers
    };

    using float32_t = float;
    using float64_t = double;
}

namespace hasty {

export using b8     =  bool;
export using i8     =  std::int8_t;
export using i16    =  std::int16_t;
export using i32    =  std::int32_t;
export using i64    =  std::int64_t;
export using u8     =  std::uint8_t;
export using u16    =  std::uint16_t;
export using u32    =  std::uint32_t;
export using u64    =  std::uint64_t;
export using f16    =  std::float16_t;
export using bf16   =  std::bfloat16_t;
export using f32    =  std::float32_t;
export using f64    =  std::float64_t;
export using c32    =  std::complex<f16>;
export using cb32   =  std::complex<bf16>;
export using c64    =  std::complex<f32>;
export using c128   =  std::complex<f64>;

export template<typename T>
concept is_tensor_type = 
                        std::is_same_v<T,b8>   ||
                        std::is_same_v<T,i8>   ||
                        std::is_same_v<T,i16>  ||
                        std::is_same_v<T,i32>  ||
                        std::is_same_v<T,i64>  ||
                        std::is_same_v<T,u8>   ||
                        std::is_same_v<T,u16>  ||
                        std::is_same_v<T,u32>  ||
                        std::is_same_v<T,u64>  ||
                        std::is_same_v<T,f16>  ||
                        std::is_same_v<T,bf16> ||
                        std::is_same_v<T,f32>  ||
                        std::is_same_v<T,f64>  ||
                        std::is_same_v<T,c32>  ||
                        std::is_same_v<T,c64>  ||
                        std::is_same_v<T,c128>;

export template<typename T>
concept is_fp_tensor_type = 
                        std::is_same_v<T,f16>  ||
                        std::is_same_v<T,bf16> ||
                        std::is_same_v<T,f32>  ||
                        std::is_same_v<T,f64>  ||
                        std::is_same_v<T,c32>  ||
                        std::is_same_v<T,c64>  ||
                        std::is_same_v<T,c128>;

export template<typename T>
concept is_complex_fp_tensor_type = 
                        std::is_same_v<T,c32>  ||
                        std::is_same_v<T,c64>  ||
                        std::is_same_v<T,c128>;

export template<typename T>
concept is_real_fp_tensor_type = 
                        std::is_same_v<T,f16>  ||
                        std::is_same_v<T,bf16> ||
                        std::is_same_v<T,f32>  ||
                        std::is_same_v<T,f64>;

export template<typename T>
concept is_integral_tensor_type =
                        std::is_same_v<T,i8>   ||
                        std::is_same_v<T,i16>  ||
                        std::is_same_v<T,i32>  ||
                        std::is_same_v<T,i64>  ||
                        std::is_same_v<T,u8>   ||
                        std::is_same_v<T,u16>  ||
                        std::is_same_v<T,u32>  ||
                        std::is_same_v<T,u64>;

export using cuda_t = empty_strong_typedef<struct cuda_>;
export using cpu_t  = empty_strong_typedef<struct cpu_>;

export template<typename T>
concept is_device = std::is_same_v<T, cuda_t> || std::is_same_v<T, cpu_t>;



}
