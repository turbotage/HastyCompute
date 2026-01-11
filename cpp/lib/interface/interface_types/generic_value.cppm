module;

export module generic_value;

import tensor_mod;

namespace hasty {

export class GenericValue {
public:

    enum struct Type {
        NONE = 0,
        TENSOR,
        VECTOR,
        DICT,
        TUPLE,
        STRING,
    };
    
private:
    Type m_type;
    std::variant<
        std::monostate,
        Tensor,
        std::vector<GenericValue>,
        std::unordered_map<std::string, GenericValue>,
        std::string
    > m_data;

public:

    GenericValue() : m_type(Type::NONE), m_data(std::monostate{}) {}

    GenericValue(Tensor tensor)
        : m_type(Type::TENSOR), m_data(tensor) {}

    GenericValue(std::vector<GenericValue> vec)
        : m_type(Type::VECTOR), m_data(vec) {}

    GenericValue(std::unordered_map<std::string, GenericValue> dict)
        : m_type(Type::DICT), m_data(dict) {}

    GenericValue(const std::string& str)
        : m_type(Type::STRING), m_data(str) {}

    static GenericValue make_tuple(std::vector<GenericValue> elements) {
        GenericValue ret;
        ret.m_type = Type::TUPLE;
        ret.m_data = std::move(elements);
        return ret;
    }

    Type type() const { return m_type; }

    const Tensor& as_tensor() const {
        return std::get<Tensor>(m_data);
    }

    const std::vector<GenericValue>& as_vector() const {
        return std::get<std::vector<GenericValue>>(m_data);
    }

    const std::unordered_map<std::string, GenericValue>& as_dict() const {
        return std::get<std::unordered_map<std::string, GenericValue>>(m_data);
    }

    const std::string& as_string() const {
        return std::get<std::string>(m_data);
    }

    

};

}