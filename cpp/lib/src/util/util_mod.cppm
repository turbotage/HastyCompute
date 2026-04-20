module;

export module hasty_util_mod;

export import :alias;
export import :containers;
export import :idx;
export import :meta;
export import :span;
export import :str;
export import :typing;

namespace hasty {

export template<typename T>
class move {
public:

	explicit move(T&& obj) : _obj(std::move(obj)) {}

	// Deleted copy constructor and copy assignment operator
	move(const move&) = delete;
	move& operator=(const move&) = delete;

	// Deleted move constructor and move assignment operator
	move(move&&) = delete;
	move& operator=(move&&) = delete;

	// Access the underlying object
	T& get() { return _obj; }
	const T& get() const { return _obj; }

private:
	T&& _obj;
};

export std::array<std::uint8_t, 16> generate_uuid() {
	static std::mutex mtx;
	static std::mt19937_64 rng{std::random_device{}()};
	static std::uniform_int_distribution<std::uint64_t> dist;
	std::lock_guard lock(mtx);
	std::array<std::uint8_t, 16> uuid;
	std::uint64_t a = dist(rng), b = dist(rng);
	std::memcpy(uuid.data(),     &a, 8);
	std::memcpy(uuid.data() + 8, &b, 8);
	return uuid;
}

}