#pragma once
// Lightweight non-owning view of a Tensor's memory for plotting/adapters.
// Designed to be trivially copyable and header-only.

#include <cstdint>
#include <cstddef>
#include <vector>
#include <type_traits>

#include <nlohmann/json.hpp>
#include <plotlypp/traits.hpp>

namespace hasty {

// A small set of simple dtypes that the plotting serializer understands.
enum struct SimpleDType : std::uint8_t {
	Null = 0,
	F32  = 1,
	F64  = 2,
	I64  = 3,
	I32  = 4,
	I16  = 5,
	B8   = 6
};

struct TensorSpanningView {
	// pointer to element storage (element-size depends on dtype)
	const void* data = nullptr;

	// simple dtype code (see SimpleDType)
	SimpleDType simple_dtype = SimpleDType::Null;

	// rank and shape/stride information (element strides, not bytes)
	int ndim = 0;
	std::vector<std::int64_t> sizes;
	std::vector<std::int64_t> strides;

	template<typename T>
	const T* data_ptr() const noexcept { return reinterpret_cast<const T*>(data); }
};

} // namespace hasty

// Inform Plotly++ that `hasty::TensorSpanningView` is an accepted data array range.
// We specialize the extension point `is_plotly_data_array_extension` and provide a
// `range_element_type` mapping. We choose `double` as the element type since the
// JSON serializer emits numeric values as double for floating/integer types.
namespace plotlypp {
template<>
struct is_plotly_data_array_extension<hasty::TensorSpanningView> : std::true_type {};

template<>
struct range_element_type<hasty::TensorSpanningView> { using type = double; };
}


// Provide a nlohmann::adl_serializer for TensorSpanningView so `json = view` works.
namespace nlohmann {
template<>
struct adl_serializer<hasty::TensorSpanningView> {
    static void to_json(json& j, const hasty::TensorSpanningView& v) {
        if (v.ndim == 0) { j = nullptr; return; }
        if (v.ndim == 1) {
            j = json::array();
            const auto n = v.sizes[0];
            switch (v.simple_dtype) {
                case hasty::SimpleDType::F32: {
                    auto ptr = reinterpret_cast<const float*>(v.data);
                    for (std::int64_t i = 0; i < n; ++i) j.push_back(static_cast<double>(ptr[i * v.strides[0]]));
                    return;
                }
                case hasty::SimpleDType::F64: {
                    auto ptr = reinterpret_cast<const double*>(v.data);
                    for (std::int64_t i = 0; i < n; ++i) j.push_back(ptr[i * v.strides[0]]);
                    return;
                }
                case hasty::SimpleDType::I64: {
                    auto ptr = reinterpret_cast<const std::int64_t*>(v.data);
                    for (std::int64_t i = 0; i < n; ++i) j.push_back(ptr[i * v.strides[0]]);
                    return;
                }
                case hasty::SimpleDType::I32: {
                    auto ptr = reinterpret_cast<const std::int32_t*>(v.data);
                    for (std::int64_t i = 0; i < n; ++i) j.push_back(ptr[i * v.strides[0]]);
                    return;
                }
                case hasty::SimpleDType::I16: {
                    auto ptr = reinterpret_cast<const std::int16_t*>(v.data);
                    for (std::int64_t i = 0; i < n; ++i) j.push_back(ptr[i * v.strides[0]]);
                    return;
                }
                case hasty::SimpleDType::B8: {
                    auto ptr = reinterpret_cast<const uint8_t*>(v.data);
                    for (std::int64_t i = 0; i < n; ++i) j.push_back(static_cast<bool>(ptr[i * v.strides[0]]));
                    return;
                }
                default:
                    throw std::runtime_error("TensorSpanningView JSON: unsupported dtype");
            }
        }
        if (v.ndim == 2) {
            j = json::array();
            const auto rows = v.sizes[0];
            const auto cols = v.sizes[1];
            switch (v.simple_dtype) {
                case hasty::SimpleDType::F32: {
                    auto ptr = reinterpret_cast<const float*>(v.data);
                    for (std::int64_t r = 0; r < rows; ++r) {
                        json jr = json::array();
                        for (std::int64_t c = 0; c < cols; ++c) jr.push_back(static_cast<double>(ptr[r * v.strides[0] + c * v.strides[1]]));
                        j.push_back(std::move(jr));
                    }
                    return;
                }
                case hasty::SimpleDType::F64: {
                    auto ptr = reinterpret_cast<const double*>(v.data);
                    for (std::int64_t r = 0; r < rows; ++r) {
                        json jr = json::array();
                        for (std::int64_t c = 0; c < cols; ++c) jr.push_back(ptr[r * v.strides[0] + c * v.strides[1]]);
                        j.push_back(std::move(jr));
                    }
                    return;
                }
                case hasty::SimpleDType::I64: {
                    auto ptr = reinterpret_cast<const std::int64_t*>(v.data);
                    for (std::int64_t r = 0; r < rows; ++r) {
                        json jr = json::array();
                        for (std::int64_t c = 0; c < cols; ++c) jr.push_back(ptr[r * v.strides[0] + c * v.strides[1]]);
                        j.push_back(std::move(jr));
                    }
                    return;
                }
                case hasty::SimpleDType::I32: {
                    auto ptr = reinterpret_cast<const std::int32_t*>(v.data);
                    for (std::int64_t r = 0; r < rows; ++r) {
                        json jr = json::array();
                        for (std::int64_t c = 0; c < cols; ++c) jr.push_back(ptr[r * v.strides[0] + c * v.strides[1]]);
                        j.push_back(std::move(jr));
                    }
                    return;
                }
                case hasty::SimpleDType::I16: {
                    auto ptr = reinterpret_cast<const std::int16_t*>(v.data);
                    for (std::int64_t r = 0; r < rows; ++r) {
                        json jr = json::array();
                        for (std::int64_t c = 0; c < cols; ++c) jr.push_back(ptr[r * v.strides[0] + c * v.strides[1]]);
                        j.push_back(std::move(jr));
                    }
                    return;
                }
                case hasty::SimpleDType::B8: {
                    auto ptr = reinterpret_cast<const uint8_t*>(v.data);
                    for (std::int64_t r = 0; r < rows; ++r) {
                        json jr = json::array();
                        for (std::int64_t c = 0; c < cols; ++c) jr.push_back(static_cast<bool>(ptr[r * v.strides[0] + c * v.strides[1]]));
                        j.push_back(std::move(jr));
                    }
                    return;
                }
                default:
                    throw std::runtime_error("TensorSpanningView JSON: unsupported dtype");
            }
        }
        throw std::runtime_error("TensorSpanningView JSON: unsupported ndim");
    }
    static void from_json(const json& /*j*/, hasty::TensorSpanningView& /*v*/) {
        throw std::runtime_error("TensorSpanningView JSON: from_json not supported");
    }
};

}



