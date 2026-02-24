module;

#include "pch.hpp"

export module util_mod:containers;

import std;

namespace hasty {

/* @brief
A simple vector-based set implementation. It isn't a true set, it isn't sorted. But it doesn't allow duplicates
*/
export template <typename T>
class vset {
public:
	// Default constructor
	vset() = default;

	// Initializer-list constructor
	vset(std::initializer_list<T> init) {
		data_.reserve(init.size());
		for (const auto& v : init) {
			if (!contains(v)) {
				data_.push_back(v);
			}
		}
	}

	// Copy constructor
	vset(const vset& other) = default;

	// Move constructor
	vset(vset&& other) noexcept = default;

	// Construct from another container (e.g., vector)
	template <typename Container>
	explicit vset(const Container& cont) {
		for (const auto& v : cont) {
			insert(v);
		}
	}

	// Copy assignment
	vset& operator=(const vset& other) = default;

	// Move assignment
	vset& operator=(vset&& other) noexcept = default;


	std::pair<std::reference_wrapper<T>,bool> insert(const T& value) {
		auto it = std::find(data_.begin(), data_.end(), value);
		if (it == data_.end()) {
			data_.push_back(value);
			return {data_.back(), true};
		}
		return {*it, false};
	}

	std::pair<std::reference_wrapper<T>,bool> insert(T&& value) {
		auto it = std::find(data_.begin(), data_.end(), value);
		if (it == data_.end()) {
			data_.push_back(std::move(value));
			return {data_.back(), true};
		}
		return {*it, false};
	}

	void insert_without_check(const T& value) {
		data_.push_back(value);
	}

	void insert_without_check(T&& value) {
		data_.push_back(std::move(value));
	}

	bool contains(const T& value) const {
		return std::find(data_.begin(), data_.end(), value) != data_.end();
	}

	T& get(const T& value) {
		auto it = std::find(data_.begin(), data_.end(), value);
		if (it != data_.end()) {
			return *it;
		}
		throw std::out_of_range("Value not found in vector_set");
	}

	const T& get(const T& value) const {
		auto it = std::find(data_.begin(), data_.end(), value);
		if (it != data_.end()) {
			return *it;
		}
		throw std::out_of_range("Value not found in vector_set");
	}

	bool erase(const T& value) {
		auto it = std::find(data_.begin(), data_.end(), value);
		if (it != data_.end()) {
			data_.erase(it);
			return true;
		}
		return false;
	}

	void reserve(size_t new_capacity) {
		data_.reserve(new_capacity);
	}

	auto begin() const { return data_.begin(); }
	auto end() const { return data_.end(); }

	std::size_t size() const { return data_.size(); }
	bool empty() const { return data_.empty(); }
	void clear() { data_.clear(); }

	const std::vector<T>& vec() const { return data_; }

	// Merge another vector_set into this one
	void merge(const vset& other) {
		data_.reserve(data_.size() + other.size());
		for (const auto& v : other) {
			insert(v);
		}
	}

	// Merge from an rvalue (can move elements)
	void merge(vset&& other) {
		data_.reserve(data_.size() + other.size());
		for (auto& v : other.data_) {
			insert(std::move(v));
		}
	}

	// Merge from any container type (e.g., std::vector<T>)
	template <typename Container>
	void merge(const Container& cont) {
		data_.reserve(data_.size() + cont.size());
		for (const auto& v : cont) {
			insert(v);
		}
	}

	friend vset merge(const vset& a, const vset& b) {
		vset result;
		result.data_.reserve(a.size() + b.size());

		for (const auto& v : a) {
			result.insert(v);
		}
		for (const auto& v : b) {
			result.insert(v);
		}

		return result;
	}

	// Merge two rvalue vector_set&& (move from both)
	friend vset merge(vset&& a, vset&& b) {
		vset result;
		result.data_.reserve(a.size() + b.size());

		for (auto& v : a.data_) {
			result.insert(std::move(v));
		}
		for (auto& v : b.data_) {
			result.insert(std::move(v));
		}

		return result;
	}

	// Merge one rvalue and one lvalue
	friend vset merge(vset&& a, const vset& b) {
		vset result;
		result.data_.reserve(a.size() + b.size());

		for (auto& v : a.data_) {
			result.insert(std::move(v));
		}
		for (const auto& v : b) {
			result.insert(v);
		}

		return result;
	}

	// Merge one lvalue and one rvalue
	friend vset merge(const vset& a, vset&& b) {
		vset result;
		result.data_.reserve(a.size() + b.size());

		for (const auto& v : a) {
			result.insert(v);
		}
		for (auto& v : b.data_) {
			result.insert(std::move(v));
		}

		return result;
	}

private:
	std::vector<T> data_;
};



export template<typename Key, typename Value>
class vmap {
private:
	std::vector<std::pair<Key, Value>> _data;

	auto lower_bound(const Key& key) {
		return std::lower_bound(_data.begin(), _data.end(), key,
			[](const auto& a, const Key& k){ return a.first < k; });
	}

	auto lower_bound(const Key& key) const {
		return std::lower_bound(_data.begin(), _data.end(), key,
			[](const auto& a, const Key& k){ return a.first < k; });
	}

public:
	using iterator = typename std::vector<std::pair<Key, Value>>::iterator;
	using const_iterator = typename std::vector<std::pair<Key, Value>>::const_iterator;

	vmap() = default;

	template<typename InputIt>
	vmap(InputIt first, InputIt last) {
		for (auto it = first; it != last; ++it)
			insert(*it);
	}

	// Number of elements
	size_t size() const { return _data.size(); }
	bool empty() const { return _data.empty(); }

	// Check if a key exists
	bool contains(const Key& key) const {
		auto it = lower_bound(key);
		return it != _data.end() && it->first == key;
	}

	// Insert key-value, ignore if key exists
	std::pair<iterator, bool> insert(const std::pair<Key, Value>& kv) {
		auto it = lower_bound(kv.first);
		if (it != _data.end() && it->first == kv.first)
			return {it, false}; // already exists
		it = _data.insert(it, kv);
		return {it, true};
	}

	std::pair<iterator, bool> insert(std::pair<Key, Value>&& kv) {
		auto it = lower_bound(kv.first);
		if (it != _data.end() && it->first == kv.first)
			return {it, false};
		it = _data.insert(it, std::move(kv));
		return {it, true};
	}

	// Access element by key, inserts default if not present
	Value& operator[](const Key& key) {
		auto it = lower_bound(key);
		if (it != _data.end() && it->first == key)
			return it->second;
		it = _data.insert(it, {key, Value{}});
		return it->second;
	}

	// Find element
	iterator find(const Key& key) {
		auto it = lower_bound(key);
		if (it != _data.end() && it->first == key)
			return it;
		return _data.end();
	}

	const_iterator find(const Key& key) const {
		auto it = lower_bound(key);
		if (it != _data.end() && it->first == key)
			return it;
		return _data.end();
	}

	// Erase by key
	size_t erase(const Key& key) {
		auto it = lower_bound(key);
		if (it != _data.end() && it->first == key) {
			_data.erase(it);
			return 1;
		}
		return 0;
	}

	vset<Key> keys() const {
		vset<Key> result;
		result.reserve(_data.size());
		for (const auto& kv : _data) {
			result.insert_without_check(kv.first);
		}
		return result;
	}

	// Iterators
	iterator begin() { return _data.begin(); }
	iterator end() { return _data.end(); }
	const_iterator begin() const { return _data.begin(); }
	const_iterator end() const { return _data.end(); }
	const_iterator cbegin() const { return _data.cbegin(); }
	const_iterator cend() const { return _data.cend(); }
};


}