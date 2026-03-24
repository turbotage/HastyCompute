module;

export module hasty_generic_value_mod;

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_threading_mod;

constexpr hasty::i64 SERIALIZE_CHUNK_SIZE = 2 * 1024 * 1024;
constexpr hasty::i64 DESERIALIZE_CHUNK_SIZE = 32 * 1024 * 1024;

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

    inline const std::unordered_map<std::string, GenericValue>& as_dict() const {
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

    const GenericValue& operator[](const std::string& key) const {
        if (!is_dict()) {
            throw std::runtime_error("Not a dict");
        }
        const auto& dict = as_dict();
        auto it = dict.find(key);
        if (it == dict.end()) {
            throw std::runtime_error("Key not found: " + key);
        }
        return it->second;
    }

    GenericValue& operator[](const std::string& key) {
        if (!is_dict()) {
            throw std::runtime_error("Not a dict");
        }
        auto& dict = std::get<std::unordered_map<std::string, GenericValue>>(m_data);
        return dict[key]; // Will default-construct if key doesn't exist
    }
    
    const GenericValue& operator[](std::size_t index) const {
        if (!is_vector() && !is_tuple()) {
            throw std::runtime_error("Not a vector or tuple");
        }
        const auto& vector = as_vector();
        if (index >= vector.size()) {
            throw std::runtime_error("Index out of bounds");
        }
        return vector[index];
    }

    GenericValue& operator[](std::size_t index) {
        if (!is_vector() && !is_tuple()) {
            throw std::runtime_error("Not a vector or tuple");
        }
        auto& vector = std::get<std::vector<GenericValue>>(m_data);
        if (index >= vector.size()) {
            throw std::runtime_error("Index out of bounds");
        }
        return vector[index];
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
    }

private:

    static void serialize_none(threadsafe_stream& stream) 
    {
        // No data to write for None, we can just send a header with type NONE
        stream.write({static_cast<u8>(GenericValue::eType::NONE)});
    }

    static void serialize_tensor(const Tensor& tensor, threadsafe_stream& stream) 
    {
        // First write a header with the tensor metadata
        SerializedTensorHeader header;
        auto device = tensor.device();
        header.device_type = static_cast<u8>(device.type);
        header.scalar_type = static_cast<u8>(tensor.scalar_type());
        header.ndim = static_cast<u8>(tensor.sizes().size());
        header.device_index = device.has_index() ? device.index : -1;
        header.total_elements = tensor.numel();

        // Write type and the header
        {
            std::vector<u8> header_bytes(sizeof(SerializedTensorHeader) + 1);
            header_bytes[0] = static_cast<u8>(GenericValue::eType::TENSOR);
            std::memcpy(header_bytes.data()+1, &header, sizeof(SerializedTensorHeader));
            stream.write(std::move(header_bytes));
        }

        // Write the shape
        {
            const auto& sizes = tensor.sizes();
            std::vector<u8> shape_bytes(sizeof(i64) * header.ndim);
            std::memcpy(shape_bytes.data(), sizes.data(), sizeof(i64) * header.ndim);
            stream.write(std::move(shape_bytes));
        }

        // Then write the raw tensor data
        const u8* data_ptr = tensor.cast_const_data_ptr<u8>();
        i64 bytes_per_element = scalar_type_size(tensor.scalar_type());
        i64 total_bytes = header.total_elements * bytes_per_element;

        i64 bytes_written = 0;
        while (bytes_written < total_bytes) {
            i64 bytes_to_write = std::min(SERIALIZE_CHUNK_SIZE, total_bytes - bytes_written);
            std::vector<u8> chunk(data_ptr + bytes_written, data_ptr + bytes_written + bytes_to_write);
            stream.write(std::move(chunk));
            bytes_written += bytes_to_write;

            while(stream.pending_chunks() > 20) {
                // If the stream has more than 10 pending chunks, 
                // wait a bit before writing more to avoid excessive memory usage
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    static void serialize_vector(const std::vector<GenericValue>& vec, threadsafe_stream& stream) 
    {
        // Write the type and the size of the vector
        {
            std::vector<u8> header_bytes(sizeof(u64) + 1);
            header_bytes[0] = static_cast<u8>(GenericValue::eType::VECTOR);
            u64 size = vec.size();
            std::memcpy(header_bytes.data() + 1, &size, sizeof(u64));
            stream.write(std::move(header_bytes));
        }

        // Then serialize each element in the vector
        for (const auto& elem : vec) {
            serialize(elem, stream);
        }
    }

    static void serialize_dict(const std::unordered_map<std::string, GenericValue>& dict, threadsafe_stream& stream) 
    {
        // Write the type and the size of the dictionary
        {
            std::vector<u8> header_bytes(sizeof(u64) + 1);
            header_bytes[0] = static_cast<u8>(GenericValue::eType::DICT);
            u64 size = dict.size();
            std::memcpy(header_bytes.data() + 1, &size, sizeof(u64));
            stream.write(std::move(header_bytes));
        }

        // Then serialize each key-value pair in the dictionary
        for (const auto& [key, value] : dict) {
            serialize_string(key, stream);
            serialize(value, stream);
        }
    }

    static void serialize_tuple(const std::vector<GenericValue>& elements, threadsafe_stream& stream) 
    {
        // Write the type and the size of the tuple
        {
            std::vector<u8> header_bytes(sizeof(u64) + 1);
            header_bytes[0] = static_cast<u8>(GenericValue::eType::TUPLE);
            u64 size = elements.size();
            std::memcpy(header_bytes.data() + 1, &size, sizeof(u64));
            stream.write(std::move(header_bytes));
        }

        // Then serialize each element in the tuple
        for (const auto& elem : elements) {
            serialize(elem, stream);
        }
    }

    static void serialize_string(const std::string& str, threadsafe_stream& stream) 
    {
        // Write the type and the length of the string
        std::vector<u8> bytes(sizeof(u64) + 1);
        {
            bytes[0] = static_cast<u8>(GenericValue::eType::STRING);
            u64 length = str.size();
            std::memcpy(bytes.data() + 1, &length, sizeof(u64));
        }

        // Then write the string data
        //std::vector<u8> string_bytes(str.begin(), str.end());
        bytes.reserve(bytes.size() + str.size());
        bytes.insert(bytes.end(), str.begin(), str.end());
        stream.write(std::move(bytes));
    }

public:

    static GenericValue deserialize(threadsafe_stream& stream)
    {
        // First read the type byte
        std::vector<u8> type_byte_vec = stream.read_exact_nbytes_blocking(1).first;
        if (type_byte_vec.empty()) {
            throw std::runtime_error("Failed to read type byte");
        }
        u8 type_byte = type_byte_vec[0];
        eType type = static_cast<eType>(type_byte);

        switch (type) {
        case eType::NONE:
            return deserialize_none(stream);
        case eType::TENSOR:
            return deserialize_tensor(stream);
        case eType::VECTOR:
            return deserialize_vector(stream);
        case eType::DICT:
            return deserialize_dict(stream);
        case eType::TUPLE:
            return deserialize_tuple(stream);
        case eType::STRING:
            return deserialize_string(stream);
        default:
            throw std::runtime_error("Unknown type byte: " + std::to_string(type_byte));
        }
    }

private:

    static GenericValue deserialize_none(threadsafe_stream& stream) 
    {
        // Nothing to read for None, we have already consumed the type byte
        return GenericValue();
    }

    static GenericValue deserialize_tensor(threadsafe_stream& stream) 
    {
        // First read the header
        SerializedTensorHeader header;
        {
            std::vector<u8> header_bytes = stream.read_exact_nbytes_blocking(sizeof(SerializedTensorHeader)).first;
            if (header_bytes.size() != sizeof(SerializedTensorHeader)) {
                throw std::runtime_error("Failed to read tensor header");
            }
            std::memcpy(&header, header_bytes.data(), sizeof(SerializedTensorHeader));
        }

        // Then read the shape
        std::vector<i64> sizes(header.ndim);
        {
            std::vector<u8> shape_bytes = stream.read_exact_nbytes_blocking(sizeof(i64) * header.ndim).first;
            if (shape_bytes.size() != sizeof(i64) * header.ndim) {
                throw std::runtime_error("Failed to read tensor shape");
            }
            std::memcpy(sizes.data(), shape_bytes.data(), sizeof(i64) * header.ndim);
        }

        // Then read the raw tensor data
        i64 total_elements = header.total_elements;
        i64 bytes_per_element = scalar_type_size(static_cast<eScalarType>(header.scalar_type));
        i64 total_bytes = total_elements * bytes_per_element;

        std::vector<u8> data_bytes;
        data_bytes.reserve(total_bytes);
        i64 bytes_read = 0;
        while (bytes_read < total_bytes) {
            std::vector<u8> chunk = stream.read_max_nbytes_blocking(std::min(static_cast<i64>(DESERIALIZE_CHUNK_SIZE), total_bytes - bytes_read)).first;
            if (chunk.empty()) {
                throw std::runtime_error("Failed to read tensor data");
            }
            data_bytes.insert(data_bytes.end(), chunk.begin(), chunk.end());
            bytes_read += chunk.size();
        }

        // Now we have all the data, we can construct the tensor
        Tensor tensor = Tensor::from_vector(
            std::move(data_bytes),
            sizes,
            static_cast<eScalarType>(header.scalar_type),
            Device(static_cast<eDeviceType>(header.device_type), header.device_index)
        );

        return GenericValue(std::move(tensor));
    }

    static GenericValue deserialize_vector(threadsafe_stream& stream)
    {
        // First read the size of the vector
        std::vector<u8> header_bytes = stream.read_exact_nbytes_blocking(sizeof(u64)).first;
        if (header_bytes.size() != sizeof(u64)) {
            throw std::runtime_error("Failed to read vector header");
        }
        u64 size;
        std::memcpy(&size, header_bytes.data(), sizeof(u64));

        std::vector<GenericValue> elements;
        elements.reserve(size);
        for (u64 i = 0; i < size; ++i) {
            elements.push_back(deserialize(stream));
        }

        return GenericValue(std::move(elements));
    }

    static GenericValue deserialize_dict(threadsafe_stream& stream) 
    {
        // First read the size of the dictionary
        std::vector<u8> header_bytes = stream.read_exact_nbytes_blocking(sizeof(u64)).first;
        if (header_bytes.size() != sizeof(u64)) {
            throw std::runtime_error("Failed to read dictionary header");
        }
        u64 size;
        std::memcpy(&size, header_bytes.data(), sizeof(u64));

        std::unordered_map<std::string, GenericValue> dict;
        for (u64 i = 0; i < size; ++i) {
            std::string key = deserialize_string(stream).as_string();
            GenericValue value = deserialize(stream);
            dict.emplace(std::move(key), std::move(value));
        }

        return GenericValue(std::move(dict));
    }

    static GenericValue deserialize_tuple(threadsafe_stream& stream) 
    {
        // First read the size of the tuple
        std::vector<u8> header_bytes = stream.read_exact_nbytes_blocking(sizeof(u64)).first;
        if (header_bytes.size() != sizeof(u64)) {
            throw std::runtime_error("Failed to read tuple header");
        }
        u64 size;
        std::memcpy(&size, header_bytes.data(), sizeof(u64));

        std::vector<GenericValue> elements;
        elements.reserve(size);
        for (u64 i = 0; i < size; ++i) {
            elements.push_back(deserialize(stream));
        }

        return GenericValue::make_tuple(std::move(elements));
    }

    static GenericValue deserialize_string(threadsafe_stream& stream) 
    {
        // First read the length of the string
        std::vector<u8> header_bytes = stream.read_exact_nbytes_blocking(sizeof(u64)).first;
        if (header_bytes.size() != sizeof(u64)) {
            throw std::runtime_error("Failed to read string header");
        }
        u64 length;
        std::memcpy(&length, header_bytes.data(), sizeof(u64));

        // Then read the string data
        std::vector<u8> string_bytes = stream.read_exact_nbytes_blocking(length).first;
        if (string_bytes.size() != length) {
            throw std::runtime_error("Failed to read string data");
        }

        std::string str(string_bytes.begin(), string_bytes.end());
        return GenericValue(std::move(str));
    }

};

}
