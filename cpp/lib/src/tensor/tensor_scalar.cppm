module;

export module tensor_mod:scalar;

import std;
import util;

import :background;

namespace hasty {

export class Scalar {
public:

    Scalar() noexcept : m_tag(Tag::HAS_i) { v.i = 0; }

    ~Scalar() = default;

    Scalar(i8 vv) noexcept : m_tag(Tag::HAS_i) { v.i = static_cast<i64>(vv); }
    Scalar(i16 vv) noexcept : m_tag(Tag::HAS_i) { v.i = static_cast<i64>(vv); }
    Scalar(i32 vv) noexcept : m_tag(Tag::HAS_i) { v.i = static_cast<i64>(vv); }
    Scalar(i64 vv) noexcept : m_tag(Tag::HAS_i) { v.i = vv; }
    Scalar(u8 vv) noexcept : m_tag(Tag::HAS_u) { v.u = static_cast<u64>(vv); }
    Scalar(u16 vv) noexcept : m_tag(Tag::HAS_u) { v.u = static_cast<u64>(vv); }
    Scalar(u32 vv) noexcept : m_tag(Tag::HAS_u) { v.u = static_cast<u64>(vv); }
    Scalar(u64 vv) noexcept : m_tag(Tag::HAS_u) { v.u = vv; }
    Scalar(f32 vv) noexcept : m_tag(Tag::HAS_d) { v.d = static_cast<f64>(vv); }
    Scalar(f64 vv) noexcept : m_tag(Tag::HAS_d) { v.d = vv; }
    Scalar(std::complex<f32> vv) noexcept : m_tag(Tag::HAS_z) { 
        v.z = std::complex<f64>(static_cast<f64>(vv.real()), static_cast<f64>(vv.imag())); 
    }
    Scalar(std::complex<f64> vv) noexcept : m_tag(Tag::HAS_z) { v.z = vv; }
    Scalar(bool vv) noexcept : m_tag(Tag::HAS_b) { v.i = vv ? 1 : 0; }

    f64 to_f64() const noexcept {
        switch (m_tag) {
            case Tag::HAS_d: return v.d;
            case Tag::HAS_z: return v.z.real();
            case Tag::HAS_i: return static_cast<f64>(v.i);
            case Tag::HAS_u: return static_cast<f64>(v.u);
            case Tag::HAS_b: return v.i ? 1.0 : 0.0;
            default: return std::numeric_limits<f64>::quiet_NaN(); // Should not reach here
        }
    }

    bool is_fp() const noexcept { return m_tag == Tag::HAS_d; }
    bool is_int(bool include_bool) const noexcept { 
        return m_tag == Tag::HAS_i || m_tag == Tag::HAS_u || (include_bool && is_bool()); 
    }
    bool is_complex() const noexcept { return m_tag == Tag::HAS_z; }
    bool is_bool() const noexcept { return m_tag == Tag::HAS_b; }

    Scalar operator-() const noexcept {
        switch (m_tag) {
            case Tag::HAS_d: return Scalar(-v.d);
            case Tag::HAS_z: return Scalar(-v.z);
            case Tag::HAS_i: return Scalar(-v.i);
            case Tag::HAS_u: return Scalar(-static_cast<i64>(v.u));
            case Tag::HAS_b: return Scalar(!v.i);
            default: return Scalar(std::numeric_limits<f64>::quiet_NaN()); // Should not reach here
        }
    }

    Scalar& operator=(const Scalar& other) & noexcept {
        if (this != &other) {
            m_tag = other.m_tag;
            v = other.v;
        }
        return *this;
    }

    Scalar& operator=(Scalar&& other) && noexcept {
        if (this != &other) {
            m_tag = other.m_tag;
            v = other.v;
        }
        return *this;
    }

    bool equal(f64 num) const noexcept {
        switch (m_tag) {
            case Tag::HAS_d: return v.d == num;
            case Tag::HAS_i: return static_cast<f64>(v.i) == num;
            case Tag::HAS_u: return static_cast<f64>(v.u) == num;
            default: return false;
        }
    }

    /*
    eScalarType dtype() const noexcept {
        switch (m_tag) {
            case Tag::HAS_d: return eScalarType::Float64;
            case Tag::HAS_i: return eScalarType::Int64;
            case Tag::HAS_u: return eScalarType::UInt64;
            case Tag::HAS_z: return eScalarType::ComplexDouble;
            case Tag::HAS_b: return eScalarType::Bool;
            default: return eScalarType::Undefined; // Should not reach here
        }
    }
    */

    hat::Scalar to_torch() const noexcept {
        switch (m_tag) {
            case Tag::HAS_d: return hat::Scalar(v.d);
            case Tag::HAS_i: return hat::Scalar(v.i);
            case Tag::HAS_u: return hat::Scalar(v.u);
            case Tag::HAS_z: return hat::Scalar(hc10::complex<double>(v.z));
            case Tag::HAS_b: return hat::Scalar(v.i != 0);
            default: return hat::Scalar(std::numeric_limits<f64>::quiet_NaN()); // Should not reach here
        }
    }

private:
    enum class Tag { 
        HAS_d, HAS_i, HAS_u, HAS_z, HAS_b
    };

    union v_t {
        f64 d{};
        i64 i;
        u64 u;
        std::complex<f64> z;
        v_t() {}
    } v;

    Tag m_tag;

};

}