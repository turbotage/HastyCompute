module;

#include <cuda_runtime.h>

export module hasty_tensor_mod:background;

import std;
import hasty_util_mod;
import hasty_torch_wrapper;

namespace hasty {

// <================== DEVICE INDEX ==================> //
export using DeviceIndex = i8;

export namespace device_alias {
    inline constexpr DeviceIndex CPU = -1;
    inline constexpr DeviceIndex CUDA0 = 0;
    inline constexpr DeviceIndex CUDA1 = 1;
    inline constexpr DeviceIndex CUDA2 = 2;
    inline constexpr DeviceIndex CUDA3 = 3;
    inline constexpr DeviceIndex CUDA4 = 4;
    inline constexpr DeviceIndex CUDA5 = 5;
    inline constexpr DeviceIndex CUDA6 = 6;
    inline constexpr DeviceIndex CUDA7 = 7;
    inline constexpr DeviceIndex CUDA8 = 8;
    inline constexpr DeviceIndex CUDA9 = 9;
    inline constexpr DeviceIndex CUDA10 = 10;
    inline constexpr DeviceIndex CUDA11 = 11;
    inline constexpr DeviceIndex CUDA12 = 12;
    inline constexpr DeviceIndex CUDA13 = 13;
    inline constexpr DeviceIndex CUDA14 = 14;
    inline constexpr DeviceIndex CUDA15 = 15;
    inline constexpr DeviceIndex MAX_CUDA_DEVICES = 16;
}

static_assert(std::is_same_v<DeviceIndex, hat::DeviceIndex>,
    "Underlying types of DeviceIndex and hat::DeviceIndex must match"
);

namespace deviceidx {
    export inline constexpr DeviceIndex from_torch(hat::DeviceIndex index) {
        return static_cast<DeviceIndex>(index);
    }
    export inline constexpr hat::DeviceIndex to_torch(DeviceIndex index) {
        return static_cast<hat::DeviceIndex>(index);
    }
}



// <================== DEVICE TYPE ==================> //
export enum struct eDeviceType : i8 {
    CPU     = std::to_underlying(hat::DeviceType::CPU),
    CUDA    = std::to_underlying(hat::DeviceType::CUDA)
};
static_assert(std::is_same_v<
    std::underlying_type_t<eDeviceType>,
    std::underlying_type_t<hat::DeviceType>>,
    "Underlying types of device_type and hat::DeviceType must match"
);

namespace devicetype {
    export inline constexpr eDeviceType from_torch(hat::DeviceType dtype) {
        return static_cast<eDeviceType>(dtype);
    }
    export inline constexpr hat::DeviceType to_torch(eDeviceType dtype) {
        return static_cast<hat::DeviceType>(dtype);
    }
}


// <================== SCALAR TYPE ==================> //
export enum struct eScalarType : i8 {
    Byte            = std::to_underlying(hat::ScalarType::Byte),
    Char            = std::to_underlying(hat::ScalarType::Char),
    Short           = std::to_underlying(hat::ScalarType::Short),
    Int             = std::to_underlying(hat::ScalarType::Int),
    Long            = std::to_underlying(hat::ScalarType::Long),
    Half            = std::to_underlying(hat::ScalarType::Half),
    Float           = std::to_underlying(hat::ScalarType::Float),
    Double          = std::to_underlying(hat::ScalarType::Double),
    ComplexHalf     = std::to_underlying(hat::ScalarType::ComplexHalf),
    ComplexFloat    = std::to_underlying(hat::ScalarType::ComplexFloat),
    ComplexDouble   = std::to_underlying(hat::ScalarType::ComplexDouble),
    Bool            = std::to_underlying(hat::ScalarType::Bool),
    BFloat16        = std::to_underlying(hat::ScalarType::BFloat16)
};

export namespace scalar_alias {
    inline constexpr eScalarType u8 = eScalarType::Byte;
    inline constexpr eScalarType i8 = eScalarType::Char;
    inline constexpr eScalarType i16 = eScalarType::Short;
    inline constexpr eScalarType i32 = eScalarType::Int;
    inline constexpr eScalarType i64 = eScalarType::Long;
    inline constexpr eScalarType f16 = eScalarType::Half;
    inline constexpr eScalarType f32 = eScalarType::Float;
    inline constexpr eScalarType f64 = eScalarType::Double;
    inline constexpr eScalarType c16 = eScalarType::ComplexHalf;
    inline constexpr eScalarType c32 = eScalarType::ComplexFloat;
    inline constexpr eScalarType c64 = eScalarType::ComplexDouble;
    inline constexpr eScalarType b8 = eScalarType::Bool;
    inline constexpr eScalarType bf16 = eScalarType::BFloat16;
}

export std::string scalar_type_to_string(eScalarType dtype) {
    switch (dtype) {
        case scalar_alias::u8: return "u8";
        case scalar_alias::i8: return "i8";
        case scalar_alias::i16: return "i16";
        case scalar_alias::i32: return "i32";
        case scalar_alias::i64: return "i64";
        case scalar_alias::f16: return "f16";
        case scalar_alias::f32: return "f32";
        case scalar_alias::f64: return "f64";
        case scalar_alias::c16: return "c16";
        case scalar_alias::c32: return "c32";
        case scalar_alias::c64: return "c64";
        case scalar_alias::b8: return "b8";
        case scalar_alias::bf16: return "bf16";
        default: return "unknown";
    }
}

export eScalarType string_to_scalar_type(const std::string& s)
{
    if (s == "u8")   return eScalarType::Byte;
	if (s == "i8")   return eScalarType::Char;
	if (s == "i16")  return eScalarType::Short;
	if (s == "i32")  return eScalarType::Int;
	if (s == "i64")  return eScalarType::Long;
	if (s == "f16")  return eScalarType::Half;
	if (s == "f32")  return eScalarType::Float;
	if (s == "f64")  return eScalarType::Double;
	if (s == "c16")  return eScalarType::ComplexHalf;
	if (s == "c32")  return eScalarType::ComplexFloat;
	if (s == "c64")  return eScalarType::ComplexDouble;
	if (s == "b8")   return eScalarType::Bool;
	if (s == "bf16") return eScalarType::BFloat16;
	throw std::runtime_error("Unknown scalar dtype '" + s + "' in HDF5 file");
}

export template<is_tensor_type T>
inline constexpr eScalarType scalar_type_of() {
    if constexpr (std::is_same_v<T, u8>) return scalar_alias::u8;
    else if constexpr (std::is_same_v<T, i8>) return scalar_alias::i8;
    else if constexpr (std::is_same_v<T, i16>) return scalar_alias::i16;
    else if constexpr (std::is_same_v<T, i32>) return scalar_alias::i32;
    else if constexpr (std::is_same_v<T, i64>) return scalar_alias::i64;
    else if constexpr (std::is_same_v<T, f16>) return scalar_alias::f16;
    else if constexpr (std::is_same_v<T, f32>) return scalar_alias::f32;
    else if constexpr (std::is_same_v<T, f64>) return scalar_alias::f64;
    else if constexpr (std::is_same_v<T, c32>) return scalar_alias::c32;
    else if constexpr (std::is_same_v<T, c64>) return scalar_alias::c64;
    else if constexpr (std::is_same_v<T, b8>) return scalar_alias::b8;
    else if constexpr (std::is_same_v<T, bf16>) return scalar_alias::bf16;
    else static_assert(always_false<T>, "Unsupported tensor type");
}

export i64 scalar_type_size(eScalarType dtype) {
    switch (dtype) {
        case scalar_alias::u8:
        case scalar_alias::i8:
        case scalar_alias::b8:
            return 1;
        case scalar_alias::i16:
        case scalar_alias::f16:
        case scalar_alias::bf16:
            return 2;
        case scalar_alias::i32:
        case scalar_alias::f32:
            return 4;
        case scalar_alias::i64:
        case scalar_alias::f64:
            return 8;
        case scalar_alias::c16:
            return 4; // complex half is 4 bytes (2 for real, 2 for imag)
        case scalar_alias::c32:
            return 8; // complex float is 8 bytes (4 for real, 4 for imag)
        case scalar_alias::c64:
            return 16; // complex double is 16 bytes (8 for real, 8 for imag)
        default:
            throw std::invalid_argument("Unknown scalar type");
    }
}

static_assert(std::is_same_v<
    std::underlying_type_t<eScalarType>,
    std::underlying_type_t<hat::ScalarType>>,
    "Underlying types of scalar_type and hat::ScalarType must match"
);

namespace scalartype {
    export inline constexpr eScalarType from_torch(hat::ScalarType dtype) {
        return static_cast<eScalarType>(dtype);
    }
    
    export inline constexpr hat::ScalarType to_torch(eScalarType dtype) {
        return static_cast<hat::ScalarType>(dtype);
    }
}

// <================== MEMORY FORMAT ==================> //
export enum struct eMemoryFormat : i8 {
    Contiguous      = std::to_underlying(hat::MemoryFormat::Contiguous),
    Preserve        = std::to_underlying(hat::MemoryFormat::Preserve),
    ChannelsLast    = std::to_underlying(hat::MemoryFormat::ChannelsLast),
    ChannelsLast3d  = std::to_underlying(hat::MemoryFormat::ChannelsLast3d)
};
static_assert(std::is_same_v<
    std::underlying_type_t<eMemoryFormat>,
    std::underlying_type_t<hat::MemoryFormat>>,
    "Underlying types of memory_format and hat::MemoryFormat must match"
);

namespace memformat {
    export inline constexpr eMemoryFormat from_torch(hat::MemoryFormat fmt) {
        return static_cast<eMemoryFormat>(fmt);
    }
    
    export inline constexpr hat::MemoryFormat to_torch(eMemoryFormat fmt) {
        return static_cast<hat::MemoryFormat>(fmt);
    }
}


// <================== LAYOUT ==================> //
export enum struct eLayout : i8 {
    Strided     = std::to_underlying(hat::Layout::Strided),
    Sparse      = std::to_underlying(hat::Layout::Sparse),
    SparseCsr   = std::to_underlying(hat::Layout::SparseCsr),
    SparseCsc   = std::to_underlying(hat::Layout::SparseCsc),
    SparseBsr   = std::to_underlying(hat::Layout::SparseBsr),
    SparseBsc   = std::to_underlying(hat::Layout::SparseBsc),
    Mkldnn      = std::to_underlying(hat::Layout::Mkldnn),
    Jagged      = std::to_underlying(hat::Layout::Jagged)
};
static_assert(std::is_same_v<
    std::underlying_type_t<eLayout>,
    std::underlying_type_t<hat::Layout>>,
    "Underlying types of layout and hat::Layout must match"
);

namespace layout {
    export inline constexpr eLayout from_torch(hat::Layout layout) {
        return static_cast<eLayout>(layout);
    }
    
    export inline constexpr hat::Layout to_torch(eLayout layout) {
        return static_cast<hat::Layout>(layout);
    }
}

// <================== DEVICE ==================> //
export struct Device {
    eDeviceType type;
    DeviceIndex index;

    Device()
        : type(eDeviceType::CPU), index(device_alias::CPU) {}

    Device(eDeviceType t, DeviceIndex i = device_alias::CPU)
        : type(t), index(i) 
    {
        if (type == eDeviceType::CPU && index != device_alias::CPU) {
            throw std::invalid_argument("CPU device must have index CPU");
        }
    }

    Device(hat::Device d)
        : type(devicetype::from_torch(d.type())), index(deviceidx::from_torch(d.index()))
    {}

    inline std::string str() const {
        if (type == eDeviceType::CPU) {
            return "cpu";
        } else {
            return "cuda:" + std::to_string(static_cast<i16>(index));
        }
    }

    inline bool has_index() const {
        return index != device_alias::CPU;
    }

    inline hat::Device torch_device() const {
        if (type == eDeviceType::CPU) {
            return hat::Device(hat::DeviceType::CPU);
        } else {
            return hat::Device(hat::DeviceType::CUDA, hat::DeviceIndex(index));
        }
    }

    inline hat::DeviceType torch_device_type() const {
        if (type == eDeviceType::CPU) {
            return hat::DeviceType::CPU;
        } else {
            return hat::DeviceType::CUDA;
        }
    }

    inline hat::DeviceIndex torch_device_index() const {
        if (type == eDeviceType::CPU) {
            return hat::DeviceIndex(-1);
        } else {
            return hat::DeviceIndex(index);
        }
    }

    static Device from_string(const std::string& s) {
        if (s == "cpu") {
            return Device(eDeviceType::CPU, device_alias::CPU);
        } else if (s.rfind("cuda:", 0) == 0) {
            std::string index_str = s.substr(5);
            try {
                int idx = std::stoi(index_str);
                if (idx < 0 || idx >= device_alias::MAX_CUDA_DEVICES) {
                    throw std::out_of_range("CUDA device index out of range");
                }
                return Device(eDeviceType::CUDA, static_cast<DeviceIndex>(idx));
            } catch (const std::exception& e) {
                throw std::invalid_argument("Invalid CUDA device string: " + s);
            }
        } else {
            throw std::invalid_argument("Unknown device string: " + s);
        }
    }

};

static_assert(sizeof(Device) == 2,
    "Size of Device should be 2 bytes"
);

// <================== TENSOR OPTIONS ==================> //
export struct TensorOptions {
private:
    Device                  m_dev;
    eScalarType             m_dtype;
    eLayout                 m_layout;
    eMemoryFormat           m_memformat;

    bool m_has_device : 1     = false;
    bool m_has_dtype  : 1     = false;
    bool m_has_layout : 1     = false;
    bool m_has_memformat : 1  = false;
    bool m_requires_grad : 1  = false;
    bool m_pinned_memory : 1  = false;

public:

    TensorOptions() = default;

    TensorOptions(Device dev) : m_dev(dev), m_has_device(true) {}
    TensorOptions(eScalarType dtype) : m_dtype(dtype), m_has_dtype(true) {}
    TensorOptions(eLayout layout) : m_layout(layout), m_has_layout(true) {}
    TensorOptions(eMemoryFormat memformat) : m_memformat(memformat), m_has_memformat(true) {}

    TensorOptions(Device dev, eScalarType dtype)
        : m_dev(dev), m_dtype(dtype), m_has_device(true), m_has_dtype(true) {}

    [[nodiscard]] inline TensorOptions device(Device d) const noexcept {
        TensorOptions r = *this;
        r.m_dev = d;
        r.m_has_device = true;
        return r;
    }

    [[nodiscard]] inline TensorOptions device(Opt<Device> d) const noexcept {
        TensorOptions r = *this;
        if (d) {
            r.m_dev = *d;
            r.m_has_device = true;
        } else {
            r.m_has_device = false;
        }
        return r;
    }

    inline TensorOptions& device_(Device d) noexcept {
        m_dev = d;
        m_has_device = true;
        return *this;
    }

    inline TensorOptions& device_(Opt<Device> d) noexcept {
        if (d) {
            m_dev = *d;
            m_has_device = true;
        } else {
            m_has_device = false;
        }
        return *this;
    }

    [[nodiscard]] inline TensorOptions dtype(eScalarType dt) const noexcept {
        TensorOptions r = *this;
        r.m_dtype = dt;
        r.m_has_dtype = true;
        return r;
    }

    [[nodiscard]] inline TensorOptions dtype(Opt<eScalarType> dt) const noexcept {
        TensorOptions r = *this;
        if (dt) {
            r.m_dtype = *dt;
            r.m_has_dtype = true;
        } else {
            r.m_has_dtype = false;
        }
        return r;
    }

    inline TensorOptions& dtype_(eScalarType dt) noexcept {
        m_dtype = dt;
        m_has_dtype = true;
        return *this;
    }

    inline TensorOptions& dtype_(Opt<eScalarType> dt) noexcept {
        if (dt) {
            m_dtype = *dt;
            m_has_dtype = true;
        } else {
            m_has_dtype = false;
        }
        return *this;
    }

    [[nodiscard]] inline TensorOptions layout(eLayout l) const noexcept {
        TensorOptions r = *this;
        r.m_layout = l;
        r.m_has_layout = true;
        return r;
    }

    [[nodiscard]] inline TensorOptions layout(Opt<eLayout> l) const noexcept {
        TensorOptions r = *this;
        if (l) {
            r.m_layout = *l;
            r.m_has_layout = true;
        } else {
            r.m_has_layout = false;
        }
        return r;
    }

    inline TensorOptions& layout_(eLayout l) noexcept {
        m_layout = l;
        m_has_layout = true;
        return *this;
    }

    inline TensorOptions& layout_(Opt<eLayout> l) noexcept {
        if (l) {
            m_layout = *l;
            m_has_layout = true;
        } else {
            m_has_layout = false;
        }
        return *this;
    }

    [[nodiscard]] inline TensorOptions memory_format(eMemoryFormat mf) const noexcept {
        TensorOptions r = *this;
        r.m_memformat = mf;
        r.m_has_memformat = true;
        return r;
    }

    [[nodiscard]] inline TensorOptions memory_format(Opt<eMemoryFormat> mf) const noexcept {
        TensorOptions r = *this;
        if (mf) {
            r.m_memformat = *mf;
            r.m_has_memformat = true;
        } else {
            r.m_has_memformat = false;
        }
        return r;
    }

    inline TensorOptions& memory_format_(eMemoryFormat mf) noexcept {
        m_memformat = mf;
        m_has_memformat = true;
        return *this;
    }

    inline TensorOptions& memory_format_(Opt<eMemoryFormat> mf) noexcept {
        if (mf) {
            m_memformat = *mf;
            m_has_memformat = true;
        } else {
            m_has_memformat = false;
        }
        return *this;
    }

    inline const Opt<Device>& get_device() const noexcept { return m_dev; }
    inline const Opt<eScalarType>& get_dtype() const noexcept { return m_dtype; }
    inline const Opt<eLayout>& get_layout() const noexcept { return m_layout; }
    inline const Opt<eMemoryFormat>& get_memory_format() const noexcept { return m_memformat; }
    inline bool get_requires_grad() const noexcept { return m_requires_grad; }
    inline bool get_pinned_memory() const noexcept { return m_pinned_memory; }

    hat::TensorOptions to_torch() const {
        hat::TensorOptions opts;

        if (m_has_device) opts = opts.device(m_dev.torch_device());
        if (m_has_dtype) opts = opts.dtype(scalartype::to_torch(m_dtype));
        if (m_has_layout) opts = opts.layout(layout::to_torch(m_layout));
        if (m_has_memformat) opts = opts.memory_format(memformat::to_torch(m_memformat));

        opts = opts.requires_grad(m_requires_grad);
        opts = opts.pinned_memory(m_pinned_memory);

        return opts;
    }

    TensorOptions merge_in(const TensorOptions& other) const {
        TensorOptions r = *this;

        if (other.m_has_device) r.m_dev = other.m_dev;
        if (other.m_has_dtype) r.m_dtype = other.m_dtype;
        if (other.m_has_layout) r.m_layout = other.m_layout;
        if (other.m_has_memformat) r.m_memformat = other.m_memformat;

        r.m_requires_grad = other.m_requires_grad;
        r.m_pinned_memory = other.m_pinned_memory;

        return r;
    }

};

// <================== TENSOR INDEXING ==================> //

export enum class eTensorIndexType {
    None = std::to_underlying(hat::indexing::TensorIndexType::None),
    Ellipsis = std::to_underlying(hat::indexing::TensorIndexType::Ellipsis),
    Boolean = std::to_underlying(hat::indexing::TensorIndexType::Boolean),
    Slice = std::to_underlying(hat::indexing::TensorIndexType::Slice),
    Tensor = std::to_underlying(hat::indexing::TensorIndexType::Tensor)
};

export using NoneIndexType = nullopt_t;
export constexpr NoneIndexType None = nullopt;

struct EllipsisIndexType final {
    EllipsisIndexType() = default;
};
export constexpr EllipsisIndexType Ellipsis = EllipsisIndexType();

export struct Slice final {
private:
    hat::indexing::Slice m_torch_slice;
public:

    Slice(const hat::indexing::Slice& torch_slice)
        : m_torch_slice(torch_slice) {}

    Slice(
        Opt<i64> start_index = nullopt,
        Opt<i64> stop_index = nullopt,
        Opt<i64> step_index = nullopt)
    {
        Opt<hc10::SymInt> torch_start = nullopt;
        Opt<hc10::SymInt> torch_stop = nullopt;
        Opt<hc10::SymInt> torch_step = nullopt;

        if (start_index) {
            torch_start = hc10::SymInt(*start_index);
        }
        if (stop_index) {
            torch_stop = hc10::SymInt(*stop_index);
        }
        if (step_index) {
            torch_step = hc10::SymInt(*step_index);
        }

        m_torch_slice = hat::indexing::Slice(
            torch_start,
            torch_stop,
            torch_step
        );
    }

    inline Opt<i64> start() const { return m_torch_slice.start().expect_int(); }
    inline Opt<i64> stop() const { return m_torch_slice.stop().expect_int(); }
    inline Opt<i64> step() const { return m_torch_slice.step().expect_int(); }

    inline hat::indexing::Slice to_torch() const {
        return m_torch_slice;
    }

};

export using SliceIndexType = Slice;

export class Tensor; // forward declaration for TensorIndex constructor

export struct TensorIndex final {
private:
    hat::indexing::TensorIndex m_torch_index;
public:

    TensorIndex(NoneIndexType) : m_torch_index(hat::indexing::None) {}
    TensorIndex(EllipsisIndexType) : m_torch_index(hat::indexing::Ellipsis) {}
    TensorIndex(const char* sv) : m_torch_index(sv) {}
    TensorIndex(const std::string& sv) : m_torch_index(sv.c_str()) {}
    TensorIndex(i64 integer) : m_torch_index(integer) {}
    TensorIndex(i32 integer) : m_torch_index(integer) {}
    TensorIndex(bool boolean) : m_torch_index(boolean) {}

    TensorIndex(const Slice& slice) : m_torch_index(slice.to_torch()) {}
    TensorIndex(const Tensor& tensor);

    inline bool     is_none() const { return m_torch_index.is_none(); }
    inline bool     is_ellipsis() const { return m_torch_index.is_ellipsis(); }
    inline bool     is_integer() const { return m_torch_index.is_integer(); }
    inline i64      integer() const { return m_torch_index.integer().expect_int(); }
    inline bool     is_boolean() const { return m_torch_index.is_boolean(); }
    inline bool     boolean() const { return m_torch_index.boolean(); }
    inline bool     is_slice() const { return m_torch_index.is_slice(); }
    inline Slice    slice() const { return Slice(m_torch_index.slice()); }
    inline bool     is_tensor() const { return m_torch_index.is_tensor(); }
    Tensor          tensor() const;

    inline hat::indexing::TensorIndex to_torch() const {
        return m_torch_index;
    }
};

export using TensorIndexType = TensorIndex;

template<typename T>
concept is_tensor_index_type = std::is_convertible_v<T, TensorIndexType>;




// <================== CUDA GUARD ==================> //
export namespace cuda {

    struct CUDAGuard {
        CUDAGuard(Device device)
            : m_guard(device.torch_device()) {}

        CUDAGuard(const CUDAGuard&)            = delete;
        CUDAGuard& operator=(const CUDAGuard&) = delete;

    private:
        hat::cuda::CUDAGuard m_guard;
    };

    struct CUDAStream {
        CUDAStream(Device device)
            : m_stream(hat::cuda::getDefaultCUDAStream(device.torch_device_index())) {}

        CUDAStream(const CUDAStream&)            = delete;
        CUDAStream& operator=(const CUDAStream&) = delete;

        inline hat::cuda::CUDAStream get_torch() const {
            return m_stream;
        }

        cudaStream_t stream() const {
            return m_stream.stream();
        }
    private:
        hat::cuda::CUDAStream m_stream;
    };

    struct CUDAStreamGuard {
        CUDAStreamGuard(const CUDAStream& stream)
            : m_guard(stream.get_torch()) {}

        CUDAStreamGuard(const CUDAStreamGuard&)            = delete;
        CUDAStreamGuard& operator=(const CUDAStreamGuard&) = delete;

    private:
        hat::cuda::CUDAStreamGuard m_guard;
    };

}


// <================== GRAD / INFERENCE GUARDS ==================> //

export struct NoGradGuard {
    NoGradGuard() = default;

    NoGradGuard(const NoGradGuard&)            = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    htorch::NoGradGuard m_guard;
};

export struct InferenceMode {
    InferenceMode(bool enabled = true)
        : m_guard(enabled) {}

    InferenceMode(const InferenceMode&)            = delete;
    InferenceMode& operator=(const InferenceMode&) = delete;

private:
    hat::InferenceMode m_guard;
};




}