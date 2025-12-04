module;

#include "pch.hpp"

export module util:meta;

import std;

namespace hasty {

export template <typename T, T... S, typename F>
constexpr void for_sequence(std::integer_sequence<T, S...>, F f) {
	(static_cast<void>(f(std::integral_constant<T, S>{})), ...);
}

export template<auto n, typename F>
constexpr void for_sequence(F f) {
	for_sequence(std::make_integer_sequence<decltype(n), n>{}, f);
}

export template<auto n, typename F, typename V>
constexpr V for_sequence(F f, const V& t) {
	V tcopy = t;
	for_sequence<n>([&tcopy, &f](auto i) {
		f(i, tcopy);
	});
	return tcopy;
}

template<typename T>
struct type_tag { using type = T; };

template<typename Tuple, typename F, std::size_t... I>
constexpr void for_each_type_impl(F&& f, std::index_sequence<I...>) {
    (f(type_tag<std::tuple_element_t<I, Tuple>>{}), ...);
}

export template<typename Tuple, typename F>
constexpr void for_each_type(F&& f) {
	for_each_type_impl<Tuple>(std::forward<F>(f), std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

export template<typename... Args>
struct tuple_traits {
	using tuple = std::tuple<Args...>;
	static constexpr size_t Size = sizeof...(Args);

	template <std::size_t N>
	using nth = typename std::tuple_element<N, tuple>::type;
	using first = nth<0>;
	using last = nth<Size - 1>;
};

export template<typename... Args>
struct tuple_traits<std::tuple<Args...>> {
	using tuple = std::tuple<Args...>;
	static constexpr size_t Size = sizeof...(Args);

	template <std::size_t N>
	using nth = typename std::tuple_element<N, tuple>::type;

	using first = nth<0>;
	using last = nth<Size - 1>;
};

export template<>
struct tuple_traits<> {
	using tuple = std::tuple<>;
	static constexpr size_t Size = 0;
};

export template<>
struct tuple_traits<std::tuple<>> {
	using tuple = std::tuple<>;
	static constexpr size_t Size = 0;
};

export template<typename T>
concept is_const = std::is_const_v<T>;

export template<typename T>
concept is_pointer = std::is_pointer_v<T>;

export template<typename T>
concept is_reference = std::is_reference_v<T>;

export template<typename T>
concept is_volatile = std::is_volatile_v<T>;

export template<typename T>
concept is_pure_type = !is_pointer<T> && !is_reference<T> && !is_const<T> && !is_volatile<T>;


}