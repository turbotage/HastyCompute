module;

export module tensor:background;

import std;
import util;
import torch_wrapper;

namespace hasty {

// <================== DEVICE INDEX ==================> //
using device_idx = i8;

namespace device_alias {
    inline constexpr device_idx CPU = -1;
    inline constexpr device_idx CUDA0 = 0;
    inline constexpr device_idx CUDA1 = 1;
    inline constexpr device_idx CUDA2 = 2;
    inline constexpr device_idx CUDA3 = 3;
    inline constexpr device_idx CUDA4 = 4;
    inline constexpr device_idx CUDA5 = 5;
    inline constexpr device_idx CUDA6 = 6;
    inline constexpr device_idx CUDA7 = 7;
    inline constexpr device_idx CUDA8 = 8;
    inline constexpr device_idx CUDA9 = 9;
    inline constexpr device_idx CUDA10 = 10;
    inline constexpr device_idx CUDA11 = 11;
    inline constexpr device_idx CUDA12 = 12;
    inline constexpr device_idx CUDA13 = 13;
    inline constexpr device_idx CUDA14 = 14;
    inline constexpr device_idx CUDA15 = 15;
}

static_assert(std::is_same_v<device_idx, hat::DeviceIndex>,
    "Underlying types of device_idx and hat::DeviceIndex must match"
);

export inline constexpr device_idx from_torch(hat::DeviceIndex index) {
    return static_cast<device_idx>(index);
}
export inline constexpr hat::DeviceIndex to_torch(device_idx index) {
    return static_cast<hat::DeviceIndex>(index);
}

// <================== DEVICE TYPE ==================> //
export enum struct device_type : i8 {
    CPU     = std::to_underlying(hat::DeviceType::CPU),
    CUDA    = std::to_underlying(hat::DeviceType::CUDA)
};
static_assert(std::is_same_v<
    std::underlying_type_t<device_type>,
    std::underlying_type_t<hat::DeviceType>>,
    "Underlying types of device_type and hat::DeviceType must match"
);

export inline constexpr device_type from_torch(hat::DeviceType dtype) {
    return static_cast<device_type>(dtype);
}
export inline constexpr hat::DeviceType to_torch(device_type dtype) {
    return static_cast<hat::DeviceType>(dtype);
}

// <================== SCALAR TYPE ==================> //
export enum struct scalar_type : i8 {
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
static_assert(std::is_same_v<
    std::underlying_type_t<scalar_type>,
    std::underlying_type_t<hat::ScalarType>>,
    "Underlying types of scalar_type and hat::ScalarType must match"
);

export inline constexpr scalar_type from_torch(hat::ScalarType dtype) {
    return static_cast<scalar_type>(dtype);
}

export inline constexpr hat::ScalarType to_torch(scalar_type dtype) {
    return static_cast<hat::ScalarType>(dtype);
}

export namespace scalar_alias {
    inline constexpr scalar_type u8 = scalar_type::Byte;
    inline constexpr scalar_type i8 = scalar_type::Char;
    inline constexpr scalar_type i16 = scalar_type::Short;
    inline constexpr scalar_type i32 = scalar_type::Int;
    inline constexpr scalar_type i64 = scalar_type::Long;
    inline constexpr scalar_type f16 = scalar_type::Half;
    inline constexpr scalar_type f32 = scalar_type::Float;
    inline constexpr scalar_type f64 = scalar_type::Double;
    inline constexpr scalar_type c16 = scalar_type::ComplexHalf;
    inline constexpr scalar_type c32 = scalar_type::ComplexFloat;
    inline constexpr scalar_type c64 = scalar_type::ComplexDouble;
    inline constexpr scalar_type b8 = scalar_type::Bool;
    inline constexpr scalar_type bf16 = scalar_type::BFloat16;
}

// <================== MEMORY FORMAT ==================> //
export enum struct memory_format : i8 {
    Contiguous      = std::to_underlying(hat::MemoryFormat::Contiguous),
    Preserve        = std::to_underlying(hat::MemoryFormat::Preserve),
    ChannelsLast    = std::to_underlying(hat::MemoryFormat::ChannelsLast),
    ChannelsLast3d  = std::to_underlying(hat::MemoryFormat::ChannelsLast3d)
};
static_assert(std::is_same_v<
    std::underlying_type_t<memory_format>,
    std::underlying_type_t<hat::MemoryFormat>>,
    "Underlying types of memory_format and hat::MemoryFormat must match"
);

export inline constexpr memory_format from_torch(hat::MemoryFormat fmt) {
    return static_cast<memory_format>(fmt);
}

export inline constexpr hat::MemoryFormat to_torch(memory_format fmt) {
    return static_cast<hat::MemoryFormat>(fmt);
}

// <================== LAYOUT ==================> //
export enum struct layout : i8 {
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
    std::underlying_type_t<layout>,
    std::underlying_type_t<hat::Layout>>,
    "Underlying types of layout and hat::Layout must match"
);

export inline constexpr layout from_torch(hat::Layout layout) {
    return static_cast<layout>(layout);
}

export inline constexpr hat::Layout to_torch(layout layout) {
    return static_cast<hat::Layout>(layout);
}

// <================== DEVICE ==================> //
export struct device {
    device_type type;
    device_idx index;

    device(device_type t, device_idx i = device_alias::CPU)
        : type(t), index(i) 
    {
        if (type == device_type::CPU && index != device_alias::CPU) {
            throw std::invalid_argument("CPU device must have index CPU");
        }
    }

    device(hat::Device d)
        : type(from_torch(d.type())), index(from_torch(d.index()))
    {}

    inline std::string str() const {
        if (type == device_type::CPU) {
            return "cpu";
        } else {
            return "cuda:" + std::to_string(static_cast<i16>(index));
        }
    }

    inline hat::Device torch_device() const {
        if (type == device_type::CPU) {
            return hat::Device(hat::DeviceType::CPU);
        } else {
            return hat::Device(hat::DeviceType::CUDA, hat::DeviceIndex(index));
        }
    }

    inline hat::DeviceType torch_device_type() const {
        if (type == device_type::CPU) {
            return hat::DeviceType::CPU;
        } else {
            return hat::DeviceType::CUDA;
        }
    }

    inline hat::DeviceIndex torch_device_index() const {
        if (type == device_type::CPU) {
            return hat::DeviceIndex(-1);
        } else {
            return hat::DeviceIndex(index);
        }
    }

};

// <================== TENSOR OPTIONS ==================> //
export struct tensor_options {
    opt<device> dev;
    opt<scalar_type> dtype;
    opt<layout> layout;
    opt<memory_format> memformat;
    bool requires_grad = false;
    bool pinned_memory = false;

    tensor_options() = default;

    tensor_options(device dev) : dev(dev) {}
    tensor_options(scalar_type dtype) : dtype(dtype) {}
    tensor_options(layout layout) : layout(layout) {}
    tensor_options(memory_format memformat) : memformat(memformat) {}

    [[nodiscard]] inline tensor_options device(opt<device> d) const noexcept {
        tensor_options r = *this;
        r.dev = d;
        return r;
    }

    inline tensor_options& device_(opt<device> d) noexcept {
        dev = d;
        return *this;
    }

    [[nodiscard]] inline tensor_options dtype(opt<scalar_type> dt) const noexcept {
        tensor_options r = *this;
        r.dtype = dt;
        return r;
    }

    inline tensor_options& dtype_(opt<scalar_type> dt) noexcept {
        dtype = dt;
        return *this;
    }

    [[nodiscard]] inline tensor_options layout(opt<layout> l) const noexcept {
        tensor_options r = *this;
        r.layout = l;
        return r;
    }

    inline tensor_options& layout_(opt<layout> l) noexcept {
        layout = l;
        return *this;
    }

    [[nodiscard]] inline tensor_options memory_format(
        opt<memory_format> mf) const noexcept {
        tensor_options r = *this;
        r.memformat = mf;
        return r;
    }

    inline tensor_options& memory_format_(opt<memory_format> mf) noexcept {
        memformat = mf;
        return *this;
    }

    inline const opt<device>& get_device() const noexcept { return dev; }
    inline const opt<scalar_type>& get_dtype() const noexcept { return dtype; }
    inline const opt<layout>& get_layout() const noexcept { return layout; }
    inline const opt<memory_format>& get_memory_format() const noexcept { return memformat; }
    inline bool get_requires_grad() const noexcept { return requires_grad; }
    inline bool get_pinned_memory() const noexcept { return pinned_memory; }

    hat::TensorOptions to_torch() const {
        hat::TensorOptions opts;

        if (dev) opts = opts.device(dev->torch_device());
        if (dtype) opts = opts.dtype(to_torch(*dtype));
        if (layout) opts = opts.layout(to_torch(*layout));
        if (memformat) opts = opts.memory_format(to_torch(*memformat));

        opts = opts.requires_grad(requires_grad);
        opts = opts.pinned_memory(pinned_memory);

        return opts;
    }

    tensor_options merge_in(const tensor_options& other) const {
        tensor_options r = *this;

        if (other.dev) r.dev = other.dev;
        if (other.dtype) r.dtype = other.dtype;
        if (other.layout) r.layout = other.layout;
        if (other.memformat) r.memformat = other.memformat;

        r.requires_grad = other.requires_grad;
        r.pinned_memory = other.pinned_memory;

        return r;
    }

};