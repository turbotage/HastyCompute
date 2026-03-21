module;

//#include "pch.hpp"

export module util_mod:meta;

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

template <typename T, typename = void>
struct has_strong_value : std::false_type{};

template <typename T>
struct has_strong_value<T, decltype((void)T::strong_value, void())> : std::true_type {};

struct strong_typedef_base {};

template<typename T>
concept is_strong_type = std::is_base_of_v<strong_typedef_base, T> && has_strong_value<T>::value;

export template<typename T, typename U>
struct strong_typedef : public strong_typedef_base {

	strong_typedef() = default;

	T strong_value;
};

export template<typename T>
struct empty_strong_typedef : public strong_typedef_base {
	empty_strong_typedef() = default;
};

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
struct TupleTraits {
	using Tuple = std::tuple<Args...>;
	static constexpr std::size_t Size = sizeof...(Args);

	template <std::size_t N>
	using Nth = typename std::tuple_element<N, Tuple>::type;
	using first = Nth<0>;
	using last = Nth<Size - 1>;
};

export template<typename... Args>
struct TupleTraits<std::tuple<Args...>> {
	using Tuple = std::tuple<Args...>;
	static constexpr std::size_t Size = sizeof...(Args);

	template <std::size_t N>
	using Nth = typename std::tuple_element<N, Tuple>::type;

	using first = Nth<0>;
	using last = Nth<Size - 1>;
};

export template<>
struct TupleTraits<> {
	using Tuple = std::tuple<>;
	static constexpr std::size_t Size = 0;
};

export template<>
struct TupleTraits<std::tuple<>> {
	using Tuple = std::tuple<>;
	static constexpr std::size_t Size = 0;
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

export template<typename T, typename K>
concept is_type_restrict = std::same_as<std::remove_cvref_t<T>, K>;

export template<std::size_t T>
concept is_dim3 = (T == 1) || (T == 2) || (T == 3);

template <class T, template <class...> class Template>
struct is_specialization : std::false_type {};

template <template <class...> class Template, class... Args>
struct is_specialization<Template<Args...>, Template> : std::true_type {};

export template <class T, template <class...> class Template>
constexpr bool is_specialization_v = is_specialization<T, Template>::value;


}