module;

export module generic_value;

import std;
import util_mod;
import tensor_mod;
import thread_stream;

namespace hasty {

export class GenericValue {
public:

    enum struct eType : u8 {
        NONE = 0,
        TENSOR,
        VECTOR,
        DICT,
        TUPLE,
        STRING,
    };

private:
    eType m_type;
    std::variant<
        std::monostate,
        Tensor,
        std::vector<GenericValue>,
        std::unordered_map<std::string, GenericValue>,
        std::string
    > m_data;

public:

    GenericValue() : m_type(eType::NONE), m_data(std::monostate{}) {}

    GenericValue(Tensor tensor)
        : m_type(eType::TENSOR), m_data(std::in_place_type<Tensor>, std::move(tensor)) {}

    GenericValue(std::vector<GenericValue> vec)
        : m_type(eType::VECTOR), m_data(std::move(vec)) {}

    GenericValue(std::unordered_map<std::string, GenericValue> dict)
        : m_type(eType::DICT), m_data(std::move(dict)) {}

    GenericValue(const std::string& str)
        : m_type(eType::STRING), m_data(str) {}

    static GenericValue make_tuple(std::vector<GenericValue> elements) {
        GenericValue ret;
        ret.m_type = eType::TUPLE;
        ret.m_data = std::move(elements);
        return ret;
    }

    eType type() const { return m_type; }

    bool is_tensor() const {
        return m_type == eType::TENSOR;
    }

    const Tensor& as_tensor() const {
        return std::get<Tensor>(m_data);
    }

    bool is_vector() const {
        return m_type == eType::VECTOR;
    }

    const std::vector<GenericValue>& as_vector() const {
        return std::get<std::vector<GenericValue>>(m_data);
    }

    template<is_tensor_container T>
    requires (is_specialization_v<T, std::vector>)
    T as_vector() const {
        const auto& vector = as_vector();
        T result;
        result.reserve(vector.size());
        for (const auto& gv : vector) {
            result.emplace_back(gv.as<typename T::value_type>());
        }
        return result;
    }

    bool is_dict() const {
        return m_type == eType::DICT;
    }

    const std::unordered_map<std::string, GenericValue>& as_dict() const {
        return std::get<std::unordered_map<std::string, GenericValue>>(m_data);
    }

    template<is_tensor_container T>
    requires (is_specialization_v<T, std::unordered_map>)
    T as_dict() const {
        const auto& dict = as_dict();
        T result;
        for (const auto& [key, gv] : dict) {
            result.emplace(key, gv.as<typename T::mapped_type>());
        }
        return result;
    }

    bool is_string() const {
        return m_type == eType::STRING;
    }

    const std::string& as_string() const {
        return std::get<std::string>(m_data);
    }

    template<std::size_t N>
    auto as_tuple() const {
        const auto& vector = as_vector();
        if (vector.size() != N) {
            throw std::runtime_error("Tuple size mismatch");
        }
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::make_tuple(vector[Is]...);
        }(std::make_index_sequence<N>{});
    }

    bool is_tuple() const {
        return m_type == eType::TUPLE;
    }

    template<is_tensor_container T>
    requires (is_specialization_v<T, std::tuple>)
    auto as_tuple() const {
        using TT = TupleTraits<T>;
        const auto& vector = as_vector();
        if (vector.size() != TT::Size) {
            throw std::runtime_error("Tuple size mismatch");
        }
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::make_tuple(vector[Is].as<typename TT::template Nth<Is>>()...);
        }(std::make_index_sequence<TT::Size>{});
    }

    template<is_tensor_container T>
    T as() const {
        if constexpr (is_specialization_v<T, std::vector>) {
            return as_vector<T>();
        } else if constexpr (is_specialization_v<T, std::unordered_map>) {
            return as_dict<T>();
        } else if constexpr (is_specialization_v<T, std::tuple>) {
            return as_tuple<T>();
        } else {
            static_assert(!std::is_same_v<T, T>, "Unsupported tensor container type");
        }
    }

    std::string metadata_string() const {
        switch (m_type) {
        case eType::NONE:
            return "None";
        case eType::TENSOR:
            return tensor_metadata_string();
        case eType::VECTOR:
            return "Vector(size=" + std::to_string(as_vector().size()) + ")";
        case eType::DICT:
            return "Dict(size=" + std::to_string(as_dict().size()) + ")";
        case eType::TUPLE:
            return "Tuple(size=" + std::to_string(as_vector().size()) + ")";
        case eType::STRING:
            return "String(length=" + std::to_string(as_string().size()) + ")";
        }
    }

    std::string unfolded_metadata_string(i32 tablevel = 0) const
    {
        switch (m_type) {
        case eType::NONE:
            return "None";
        case eType::TENSOR:
            return tensor_metadata_string();
        case eType::VECTOR: {
            TabbedWriter writer;
            writer.write_line("Vector(size=" + std::to_string(as_vector().size()) + "){");
            writer.tab_in();
            const auto& elements = as_vector();
            auto elements_size = elements.size();
            for (std::size_t i = 0; i < elements_size; ++i) {
                const auto& elem = elements[i];
                writer.append_string_tabbed(elem.unfolded_metadata_string());
                if (i != elements_size - 1) {
                    writer.append_string_tabbed(",");
                }
                if (!writer.on_newline()) {
                    writer.write_line();
                }
            }
            writer.write_line("}");
            return writer.str();
        }
        case eType::DICT: {
            TabbedWriter writer;
            writer.write_line("Dict(size=" + std::to_string(as_dict().size()) + "){");
            writer.tab_in();
            int i = 0;
            const auto& dict = as_dict();
            auto dict_size = dict.size();
            for (const auto& [key, value] : dict) {
                writer.append_string_tabbed("{\n\t" + key + "\n\t" + value.unfolded_metadata_string());
                if (!writer.on_newline()) {
                    writer.write_line();
                }
                ++i;
                if (static_cast<std::size_t>(i) < dict_size) {
                    writer.write_line("},");
                } else {
                    writer.write_line("}");
                }
            }
            writer.write_line("}");
            return writer.str();
        }
        case eType::TUPLE: {
            TabbedWriter writer;
            writer.write_line("Tuple(size=" + std::to_string(as_vector().size()) + "){");
            writer.tab_in();
            const auto& elements = as_vector();
            auto elements_size = elements.size();
            for (std::size_t i = 0; i < elements_size; ++i) {
                const auto& elem = elements[i];
                writer.append_string_tabbed(elem.unfolded_metadata_string());
                if (i != elements_size - 1) {
                    writer.append_string_tabbed(",");
                }
                if (!writer.on_newline()) {
                    writer.write_line();
                }
            }
            writer.write_line("}");
            return writer.str();
        }
        case eType::STRING:
            return "String(length=" + std::to_string(as_string().size()) + ") { " + as_string() + "  }";
        }
    }

    std::string tensor_metadata_string() const {
        return as_tensor().metadata_string();
    }

};

export class GenericValueSerializer {
public:

    struct SerializedTensorHeader {
        u8 device_type;
        u8 scalar_type;
        u8 ndim;
        i8 device_index;
        i64 total_elements;
    };

    static void serialize(GenericValue value, threadsafe_stream& stream)
    {
        switch (value.type()) {
        case GenericValue::eType::NONE:
            serialize_none(stream);
            break;
        case GenericValue::eType::TENSOR:
            serialize_tensor(value.as_tensor(), stream);
            break;
        case GenericValue::eType::VECTOR:
            serialize_vector(value.as_vector(), stream);
            break;
        case GenericValue::eType::DICT:
            serialize_dict(value.as_dict(), stream);
            break;
        case GenericValue::eType::TUPLE:
            serialize_tuple(value.as_vector(), stream);
            break;
        case GenericValue::eType::STRING:
            serialize_string(value.as_string(), stream);
            break;
    }

    static void serialize_none(threadsafe_stream& stream) {
        // No data to write for None, we can just send a header with type NONE
        stream.write({static_cast<u8>(GenericValue::eType::NONE)});
    }

    static void serialize_tensor(const Tensor& tensor, threadsafe_stream& stream) {
        // First write a header with the tensor metadata
        SerializedTensorHeader header;
        auto device = tensor.device();
        header.device_type = static_cast<u8>(device.type);
        header.scalar_type = static_cast<u8>(tensor.scalar_type());
        header.ndim = static_cast<u8>(tensor.sizes().size());
        header.device_index = device.has_index() ? device.index : -1;
        header.total_elements = tensor.numel();

        // Write the header
        {
            std::vector<u8> header_bytes(sizeof(SerializedTensorHeader));
            std::memcpy(header_bytes.data(), &header, sizeof(SerializedTensorHeader));
            stream.write(header_bytes);
        }

        // Write the shape

        // Then write the raw tensor data
        const u8* data_ptr = static_cast<const u8*>(tensor.const_data_ptr());
        i64 bytes_per_element = scalar_type_size(tensor.scalar_type());
        i64 total_bytes = header.total_elements * bytes_per_element;

        // We will write the data in chunks of at most 4MB
        const i64 chunk_size = 4 * 1024 * 1024;
        i64 bytes_written = 0;
        while (bytes_written < total_bytes) {
            i64 bytes_to_write = std::min(chunk_size, total_bytes - bytes_written);
            std::vector<u8> chunk(data_ptr + bytes_written, data_ptr + bytes_written + bytes_to_write);
            stream.write(std::move(chunk));
            bytes_written += bytes_to_write;
        }
    }

};

}
