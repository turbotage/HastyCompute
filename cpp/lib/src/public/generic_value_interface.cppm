module;

export module generic_value_interface;

import std;
import 

namespace hasty {

export struct StreamedGenericValue {
    std::string gv_typename;
    std::shared_ptr<threadsafe_stream> stream;
};

export struct GenericValueInterface {
public:
    std::type_info m_type;
    std::shared_ptr<void> m_obj;
    std::function<StreamedGenericValue(std::shared_ptr<void> obj)> m_streamer;
    
    template<typename T>
    const T& get() const {
        if (m_type != typeid(T)) {
            throw std::runtime_error("Type mismatch in GenericValueInterface::get");
        }
        return *std::static_pointer_cast<T>(m_obj);
    }

    template<typename T>
    T& get() {
        if (m_type != typeid(T)) {
            throw std::runtime_error("Type mismatch in GenericValueInterface::get");
        }
        return *std::static_pointer_cast<T>(m_obj);
    }

    template<typename T>
    std::shared_ptr<T> get_ptr() const {
        if (m_type != typeid(T)) {
            throw std::runtime_error("Type mismatch in GenericValueInterface::get_ptr");
        }
        return std::static_pointer_cast<T>(m_obj);
    }

    StreamedGenericValue get_streamed() const {
        return m_streamer(m_obj);
    }


    template<typename T>
    GenericValueInterface(std::shared_ptr<T> obj, std::function<StreamedGenericValue(std::shared_ptr<void> obj)> streamer)
        : m_type(typeid(T)), m_obj(std::move(obj)), m_streamer(std::move(streamer))
    {}

private:
};

}