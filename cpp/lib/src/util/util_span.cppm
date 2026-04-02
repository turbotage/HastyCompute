module;

//#include "pch.hpp"

export module hasty_util_mod:span;

import std;
import :meta;
import :typing;
import hasty_torch_wrapper;


namespace hasty {


export template<typename T>
class ArrayRef {
private:
	const T* m_data;
	std::size_t m_size;
public:

	constexpr ArrayRef() noexcept : m_data(nullptr), m_size(0) {}

	constexpr ArrayRef(const T* data, std::size_t size) noexcept
		: m_data(data), m_size(size) {}

	template<typename Allocator>
	constexpr ArrayRef(const std::vector<T, Allocator>& vec) noexcept
		: m_data(vec.data()), m_size(vec.size()) {}

	template<std::size_t N>
	constexpr ArrayRef(const std::array<T, N>& arr) noexcept
		: m_data(arr.data()), m_size(N) {}

	template<std::size_t N>
	constexpr ArrayRef(const T (&arr)[N]) noexcept
		: m_data(arr), m_size(N) {}

	template<typename It>
	constexpr ArrayRef(It begin, It end) noexcept
		: m_data(std::addressof(*begin)), m_size(static_cast<std::size_t>(std::distance(begin, end))) {}

	constexpr ArrayRef(const hat::ArrayRef<T>& other) noexcept
		: m_data(other.data()), m_size(other.size()) {}

	constexpr ArrayRef(std::initializer_list<T> il) noexcept
    : m_data(il.begin()), m_size(il.size()) {}

	constexpr inline const T* data() const noexcept { return m_data; }
	constexpr inline std::size_t size() const noexcept { return m_size; }
	constexpr inline bool empty() const noexcept { return m_size == 0; }

	constexpr inline const T& operator[](std::size_t index) const noexcept {
		return m_data[index];
	}
	constexpr inline const T& front() const noexcept {
		return m_data[0];
	}
	constexpr inline const T& back() const noexcept {
		return m_data[m_size - 1];
	}

	constexpr inline const T* begin() const noexcept { return m_data; }
	constexpr inline const T* end() const noexcept { return m_data + m_size; }
	constexpr inline const T* cbegin() const noexcept { return m_data; }
	constexpr inline const T* cend() const noexcept { return m_data + m_size; }

	constexpr ArrayRef<T> slice(std::size_t start, std::size_t length) const noexcept {
		return ArrayRef<T>(m_data + start, length);
	}
	constexpr ArrayRef<T> slice(std::size_t start) const noexcept {
		return ArrayRef<T>(m_data + start, m_size - start);
	}

	friend constexpr bool operator==(const ArrayRef<T>& lhs, const ArrayRef<T>& rhs) noexcept {
		return lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin());
	}
	friend constexpr bool operator!=(const ArrayRef<T>& lhs, const ArrayRef<T>& rhs) noexcept {
		return !(lhs == rhs);
	}

	hat::ArrayRef<T> to_torch() const noexcept {
		return hat::ArrayRef<T>(m_data, m_size);
	}

	std::string str() const {
		std::string s;
		for (std::size_t i = 0; i < m_size; ++i) {
			if (i > 0) s += ",";
			s += std::to_string(m_data[i]);
		}
		return s;
	}

	bool equals(const ArrayRef<T>& other) const noexcept {
		if (m_size != other.m_size) {
			return false;
		}
		for (std::size_t i = 0; i < m_size; ++i) {
			if (m_data[i] != other.m_data[i]) {
				return false;
			}
		}
		return true;
	}

	std::vector<T> vec() const {
		return std::vector<T>(m_data, m_data + m_size);
	}

};

export template<typename It>
ArrayRef(It, It) -> ArrayRef<typename std::iterator_traits<It>::value_type>;

export using IntArrayRef = ArrayRef<i64>;


}