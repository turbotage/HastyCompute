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



}