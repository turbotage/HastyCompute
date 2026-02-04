module;

export module generic_value;

import tensor_mod;

namespace hasty {

export class GenericValue {
public:

    enum struct eType {
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
        : m_type(eType::TENSOR), m_data(tensor) {}

    GenericValue(std::vector<GenericValue> vec)
        : m_type(eType::VECTOR), m_data(vec) {}

    GenericValue(std::unordered_map<std::string, GenericValue> dict)
        : m_type(eType::DICT), m_data(dict) {}

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
    requires (std::is_specialization_v<T, std::vector>)
    T as_vector() const {
        std::vector<GenericValue> vector = as_vector();
        T result;
        result.reserve(vector.size());
        for (auto& gv : vector) {
            // Ensure each element can be converted to T's value_type
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
    requires (std::is_specialization_v<T, std::unordered_map>)
    T as_dict() const {
        std::unordered_map<std::string, GenericValue> dict = as_dict();
        T result;
        for (auto& [key, gv] : dict) {
            // Ensure each value can be converted to T's mapped_type
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
        std::vector<GenericValue> vector = as_vector();
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
    requires std::is_specialization_v<T, std::tuple>
    auto as_tuple() const {
        using TT = TupleTraits<T>;
        std::vector<GenericValue> vector = as_vector();
        if (vector.size() != TT::Size) {
            throw std::runtime_error("Tuple size mismatch");
        }
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return std::make_tuple(vector[Is].as<typename TT::template Nth<Is>>()...);
        }(std::make_index_sequence<TT::Size>{});
    }

    template<is_tensor_container T>
    T as() const {
        if constexpr (std::is_specialization_v<T, std::vector>) {
            return as_vector<T>();
        } else if constexpr (std::is_specialization_v<T, std::unordered_map>) {
            return as_dict<T>();
        } else if constexpr (std::is_specialization_v<T, std::tuple>) {
            return as_tuple<T>();
        } else {
            static_assert(always_false<T>::value, "Unsupported tensor container type");
        }
    }

    std::string metadata_string() const  {
        switch (m_type) {
        case eType::NONE:
            return "None";
        case eType::TENSOR:
            return tensor_metadata_string()
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

    std::string unfolded_metadata_string(i32 tablevel = 0) const  {
        switch (m_type) {
        case eType::NONE:
            return "None";
        case eType::TENSOR:
            return tensor_metadata_string();
        case eType::VECTOR: {
            TabbedWriter writer;
            writer.write_line("Vector(size=" + std::to_string(as_vector().size()) + "){");
            writer.tab_in();
            auto elements = as_vector();
			auto elements_size = elements.size();
            for (const auto& [i, elem] : enumerate(elements)) {
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
			auto dict = as_dict();
			auto dict_size = dict.size();
            for (const auto& [key, value] : dict) {
                writer.append_string_tabbed("{\n	" + key + "\n	" + value.unfolded_metadata_string());
				if (!writer.on_newline()) {
					writer.write_line();
				}
                ++i;
				if (i < dict_size) {
					writer.write_line("},");
				} else {
					writer.write_line("}");
				}
            }
            writer.write_line("}");
            return writer.str();
        }
        case eType::TUPLE: {
            std::string result = "Tuple(size=" + std::to_string(as_vector().size()) + ")[\n";
            writer.tab_in();
            auto elements = as_vector();
			auto elements_size = elements.size();
            for (const auto& [i, elem] : enumerate(elements)) {
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



};

}