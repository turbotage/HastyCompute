module;

#include <nlohmann/json.hpp>

export module hasty_tensor_mod:json;

import std;
import :tensor;
// JSON (de)serializer for hasty::Tensor that emits Plotly-friendly arrays
namespace nlohmann {

template<>
struct adl_serializer<hasty::Tensor> {
    static void to_json(json& j, const hasty::Tensor& t) {
        // Produce Plotly-friendly JSON arrays (1D -> array, 2D -> array-of-arrays)
        auto tc = t.cpu();
        int ndim = static_cast<int>(tc.ndimension());

        if (ndim < 0 || ndim > 2) {
            throw std::runtime_error("tensor JSON serializer: unsupported tensor rank for plotting");
        }

        auto sizes = tc.sizes();
        auto strides = tc.strides();

        if (ndim == 0 || ndim == 1) {
            // scalar or 1D
            if (ndim == 0) {
                switch (tc.scalar_type()) {
                    case hasty::scalar_alias::f32: j = tc.item<hasty::f32>(); return;
                    case hasty::scalar_alias::f64: j = tc.item<hasty::f64>(); return;
                    case hasty::scalar_alias::i64: j = tc.item<hasty::i64>(); return;
                    case hasty::scalar_alias::i32: j = tc.item<hasty::i32>(); return;
                    case hasty::scalar_alias::i16: j = tc.item<hasty::i16>(); return;
                    case hasty::scalar_alias::b8: j = tc.item<bool>(); return;
                    default: throw std::runtime_error("tensor JSON serializer: unsupported dtype for plotting");
                }
            }

            hasty::i64 n = sizes[0];
            j = json::array();
            switch (tc.scalar_type()) {
                case hasty::scalar_alias::f32: {
                    const auto* data = tc.cast_const_data_ptr<hasty::f32>();
                    for (hasty::i64 i = 0; i < n; ++i) {
                        hasty::i64 idx = i * strides[0];
                        j.push_back(static_cast<double>(data[idx]));
                    }
                    return;
                }
                case hasty::scalar_alias::f64: {
                    const auto* data = tc.cast_const_data_ptr<hasty::f64>();
                    for (hasty::i64 i = 0; i < n; ++i) {
                        hasty::i64 idx = i * strides[0];
                        j.push_back(data[idx]);
                    }
                    return;
                }
                case hasty::scalar_alias::i64: {
                    const auto* data = tc.cast_const_data_ptr<hasty::i64>();
                    for (hasty::i64 i = 0; i < n; ++i) {
                        hasty::i64 idx = i * strides[0];
                        j.push_back(data[idx]);
                    }
                    return;
                }
                case hasty::scalar_alias::i32: {
                    const auto* data = tc.cast_const_data_ptr<hasty::i32>();
                    for (hasty::i64 i = 0; i < n; ++i) {
                        hasty::i64 idx = i * strides[0];
                        j.push_back(data[idx]);
                    }
                    return;
                }
                case hasty::scalar_alias::i16: {
                    const auto* data = tc.cast_const_data_ptr<hasty::i16>();
                    for (hasty::i64 i = 0; i < n; ++i) {
                        hasty::i64 idx = i * strides[0];
                        j.push_back(data[idx]);
                    }
                    return;
                }
                case hasty::scalar_alias::b8: {
                    const auto* data = tc.cast_const_data_ptr<hasty::b8>();
                    for (hasty::i64 i = 0; i < n; ++i) {
                        hasty::i64 idx = i * strides[0];
                        j.push_back(static_cast<bool>(data[idx]));
                    }
                    return;
                }
                default:
                    throw std::runtime_error("tensor JSON serializer: unsupported dtype for plotting");
            }
        } else {
            // 2D
            hasty::i64 rows = sizes[0];
            hasty::i64 cols = sizes[1];
            j = json::array();
            switch (tc.scalar_type()) {
                case hasty::scalar_alias::f32: {
                    const auto* data = tc.cast_const_data_ptr<hasty::f32>();
                    for (hasty::i64 r = 0; r < rows; ++r) {
                        json jr = json::array();
                        for (hasty::i64 c = 0; c < cols; ++c) {
                            hasty::i64 idx = r * strides[0] + c * strides[1];
                            jr.push_back(static_cast<double>(data[idx]));
                        }
                        j.push_back(std::move(jr));
                    }
                    return;
                }
                case hasty::scalar_alias::f64: {
                    const auto* data = tc.cast_const_data_ptr<hasty::f64>();
                    for (hasty::i64 r = 0; r < rows; ++r) {
                        json jr = json::array();
                        for (hasty::i64 c = 0; c < cols; ++c) {
                            hasty::i64 idx = r * strides[0] + c * strides[1];
                            jr.push_back(data[idx]);
                        }
                        j.push_back(std::move(jr));
                    }
                    return;
                }
                case hasty::scalar_alias::i64: {
                    const auto* data = tc.cast_const_data_ptr<hasty::i64>();
                    for (hasty::i64 r = 0; r < rows; ++r) {
                        json jr = json::array();
                        for (hasty::i64 c = 0; c < cols; ++c) {
                            hasty::i64 idx = r * strides[0] + c * strides[1];
                            jr.push_back(data[idx]);
                        }
                        j.push_back(std::move(jr));
                    }
                    return;
                }
                case hasty::scalar_alias::i32: {
                    const auto* data = tc.cast_const_data_ptr<hasty::i32>();
                    for (hasty::i64 r = 0; r < rows; ++r) {
                        json jr = json::array();
                        for (hasty::i64 c = 0; c < cols; ++c) {
                            hasty::i64 idx = r * strides[0] + c * strides[1];
                            jr.push_back(data[idx]);
                        }
                        j.push_back(std::move(jr));
                    }
                    return;
                }
                case hasty::scalar_alias::i16: {
                    const auto* data = tc.cast_const_data_ptr<hasty::i16>();
                    for (hasty::i64 r = 0; r < rows; ++r) {
                        json jr = json::array();
                        for (hasty::i64 c = 0; c < cols; ++c) {
                            hasty::i64 idx = r * strides[0] + c * strides[1];
                            jr.push_back(data[idx]);
                        }
                        j.push_back(std::move(jr));
                    }
                    return;
                }
                case hasty::scalar_alias::b8: {
                    const auto* data = tc.cast_const_data_ptr<hasty::b8>();
                    for (hasty::i64 r = 0; r < rows; ++r) {
                        json jr = json::array();
                        for (hasty::i64 c = 0; c < cols; ++c) {
                            hasty::i64 idx = r * strides[0] + c * strides[1];
                            jr.push_back(static_cast<bool>(data[idx]));
                        }
                        j.push_back(std::move(jr));
                    }
                    return;
                }
                default:
                    throw std::runtime_error("tensor JSON serializer: unsupported dtype for plotting");
            }
        }
    }

    static void from_json(const json& j, hasty::Tensor& t) {
        // Accept either Plotly-style arrays (1D or 2D) or the raw-bytes object format.
        if (j.is_array()) {
            // 1D or 2D nested arrays
            if (j.empty()) {
                // empty tensor -> default-constructed empty Tensor
                t = hasty::Tensor();
                return;
            }

            // detect if nested (2D)
            if (j.front().is_array()) {
                // 2D
                size_t rows = j.size();
                size_t cols = j.front().size();
                // determine dtype: if any float -> f64, else if any bool -> b8, else i64
                bool any_float = false;
                bool any_bool = false;
                for (const auto &row : j) {
                    for (const auto &el : row) {
                        if (el.is_number_float()) any_float = true;
                        if (el.is_boolean()) any_bool = true;
                    }
                }
                if (any_float) {
                    std::vector<uint8_t> raw; raw.resize(rows * cols * sizeof(hasty::f64));
                    hasty::f64* dst = reinterpret_cast<hasty::f64*>(raw.data());
                    for (size_t r = 0; r < rows; ++r) {
                        const auto &row = j[r];
                        for (size_t c = 0; c < cols; ++c) dst[r*cols + c] = row[c].get<hasty::f64>();
                    }
                    t = hasty::Tensor::from_vector(std::move(raw), hasty::ArrayRef<hasty::i64>(std::vector<hasty::i64>{static_cast<hasty::i64>(rows), static_cast<hasty::i64>(cols)}), hasty::scalar_alias::f64, hasty::Device());
                    return;
                } else if (any_bool) {
                    std::vector<uint8_t> raw; raw.resize(rows * cols * sizeof(hasty::b8));
                    hasty::b8* dst = reinterpret_cast<hasty::b8*>(raw.data());
                    for (size_t r = 0; r < rows; ++r) {
                        const auto &row = j[r];
                        for (size_t c = 0; c < cols; ++c) dst[r*cols + c] = row[c].get<bool>();
                    }
                    t = hasty::Tensor::from_vector(std::move(raw), hasty::ArrayRef<hasty::i64>(std::vector<hasty::i64>{static_cast<hasty::i64>(rows), static_cast<hasty::i64>(cols)}), hasty::scalar_alias::b8, hasty::Device());
                    return;
                } else {
                    // integer
                    std::vector<uint8_t> raw; raw.resize(rows * cols * sizeof(hasty::i64));
                    hasty::i64* dst = reinterpret_cast<hasty::i64*>(raw.data());
                    for (size_t r = 0; r < rows; ++r) {
                        const auto &row = j[r];
                        for (size_t c = 0; c < cols; ++c) dst[r*cols + c] = row[c].get<hasty::i64>();
                    }
                    t = hasty::Tensor::from_vector(std::move(raw), hasty::ArrayRef<hasty::i64>(std::vector<hasty::i64>{static_cast<hasty::i64>(rows), static_cast<hasty::i64>(cols)}), hasty::scalar_alias::i64, hasty::Device());
                    return;
                }
            } else {
                // 1D
                size_t n = j.size();
                bool any_float = false;
                bool any_bool = false;
                for (const auto &el : j) {
                    if (el.is_number_float()) any_float = true;
                    if (el.is_boolean()) any_bool = true;
                }
                if (any_float) {
                    std::vector<uint8_t> raw; raw.resize(n * sizeof(hasty::f64));
                    hasty::f64* dst = reinterpret_cast<hasty::f64*>(raw.data());
                    for (size_t i = 0; i < n; ++i) dst[i] = j[i].get<hasty::f64>();
                    t = hasty::Tensor::from_vector(std::move(raw), hasty::ArrayRef<hasty::i64>(std::vector<hasty::i64>{static_cast<hasty::i64>(n)}), hasty::scalar_alias::f64, hasty::Device());
                    return;
                } else if (any_bool) {
                    std::vector<uint8_t> raw; raw.resize(n * sizeof(hasty::b8));
                    hasty::b8* dst = reinterpret_cast<hasty::b8*>(raw.data());
                    for (size_t i = 0; i < n; ++i) dst[i] = j[i].get<bool>();
                    t = hasty::Tensor::from_vector(std::move(raw), hasty::ArrayRef<hasty::i64>(std::vector<hasty::i64>{static_cast<hasty::i64>(n)}), hasty::scalar_alias::b8, hasty::Device());
                    return;
                } else {
                    std::vector<uint8_t> raw; raw.resize(n * sizeof(hasty::i64));
                    hasty::i64* dst = reinterpret_cast<hasty::i64*>(raw.data());
                    for (size_t i = 0; i < n; ++i) dst[i] = j[i].get<hasty::i64>();
                    t = hasty::Tensor::from_vector(std::move(raw), hasty::ArrayRef<hasty::i64>(std::vector<hasty::i64>{static_cast<hasty::i64>(n)}), hasty::scalar_alias::i64, hasty::Device());
                    return;
                }
            }
        }

        // Fallback: expect object with shape/dtype/data as raw bytes
        auto shape = j.at("shape").get<std::vector<hasty::i64>>();

        // dtype
        auto dtype_str = j.at("dtype").get<std::string>();
        hasty::eScalarType dtype;
        if (dtype_str == "u8") dtype = hasty::scalar_alias::u8;
        else if (dtype_str == "i8") dtype = hasty::scalar_alias::i8;
        else if (dtype_str == "i16") dtype = hasty::scalar_alias::i16;
        else if (dtype_str == "i32") dtype = hasty::scalar_alias::i32;
        else if (dtype_str == "i64") dtype = hasty::scalar_alias::i64;
        else if (dtype_str == "f16") dtype = hasty::scalar_alias::f16;
        else if (dtype_str == "f32") dtype = hasty::scalar_alias::f32;
        else if (dtype_str == "f64") dtype = hasty::scalar_alias::f64;
        else if (dtype_str == "c16") dtype = hasty::scalar_alias::c16;
        else if (dtype_str == "c32") dtype = hasty::scalar_alias::c32;
        else if (dtype_str == "c64") dtype = hasty::scalar_alias::c64;
        else if (dtype_str == "b8") dtype = hasty::scalar_alias::b8;
        else if (dtype_str == "bf16") dtype = hasty::scalar_alias::bf16;
        else throw std::runtime_error(std::string("Unsupported dtype in JSON: ") + dtype_str);

        // device (optional)
        hasty::Device device;
        if (j.contains("device")) {
            auto device_info = j.at("device");
            int device_type = device_info.at("type").get<int>();
            int device_index = device_info.at("index").get<int>();
            device = hasty::Device(static_cast<hasty::eDeviceType>(device_type), static_cast<hasty::DeviceIndex>(device_index));
        }

        // data bytes
        auto data = j.at("data").get<std::vector<uint8_t>>();

        // construct tensor from raw bytes
        t = hasty::Tensor::from_vector(std::move(data), hasty::ArrayRef<hasty::i64>(shape), dtype, device);
    }
};

} // namespace nlohmann
