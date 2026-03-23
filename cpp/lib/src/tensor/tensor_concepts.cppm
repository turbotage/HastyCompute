module;

export module hasty_tensor_mod:concepts;

import std;

import :tensor;

namespace hasty {

export template<typename T, int Depth = 0>
struct is_tensor_container_impl {
    static constexpr bool value = std::is_same_v<T,hasty::Tensor>;
};

// Specialization for std::vector
template<typename T, int Depth>
requires (Depth < 10)
struct is_tensor_container_impl<std::vector<T>, Depth> {
	static constexpr bool value = is_tensor_container_impl<T, Depth+1>::value;
};

// Specialization for std::unordered_map
template<typename K, typename V, int Depth>
requires (Depth < 10) && std::is_same_v<K, std::string>
struct is_tensor_container_impl<std::unordered_map<K, V>, Depth> {
	static constexpr bool value = is_tensor_container_impl<V, Depth+1>::value;
};

// Specialization for std::tuple
template<typename... Ts, int Depth>
requires (Depth < 10)
struct is_tensor_container_impl<std::tuple<Ts...>, Depth> {
	static constexpr bool value = (is_tensor_container_impl<Ts, Depth+1>::value && ...);
};

export template<typename T>
concept is_tensor_container = is_tensor_container_impl<T,0>::value;

export template<typename T, int Depth>
concept is_tensor_container_depthlimited = is_tensor_container_impl<T, Depth>::value;

}
