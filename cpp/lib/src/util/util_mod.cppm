module;

export module hasty_util_mod;

export import :alias;
export import :containers;
export import :idx;
export import :meta;
export import :span;
export import :str;
export import :stream;
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

export std::array<u8, 16> generate_uuid() {
	static std::mutex mtx;
	static std::mt19937_64 rng{std::random_device{}()};
	static std::uniform_int_distribution<u64> dist;
	std::lock_guard lock(mtx);
	std::array<u8, 16> uuid;
	std::uint64_t a = dist(rng), b = dist(rng);
	std::memcpy(uuid.data(),     &a, 8);
	std::memcpy(uuid.data() + 8, &b, 8);
	return uuid;
}

export std::string uuid_to_hex(const std::array<u8, 16>& uuid) {
    static constexpr char hex_chars[] = "0123456789abcdef";
    std::string out(32, '0');
    for (int i = 0; i < 16; ++i) {
        out[i * 2]     = hex_chars[(uuid[i] >> 4) & 0xF];
        out[i * 2 + 1] = hex_chars[uuid[i] & 0xF];
    }
    return out;
}

export std::array<u8, 16> hex_to_uuid_array(const std::string& hex) {
    if (hex.size() != 32)
        throw std::runtime_error("[python] UUID hex string must be 32 chars");
    std::array<u8, 16> out{};
    for (int i = 0; i < 16; ++i) {
        auto nibble = [](char c) -> u8 {
            if (c >= '0' && c <= '9') return static_cast<u8>(c - '0');
            if (c >= 'a' && c <= 'f') return static_cast<u8>(c - 'a' + 10);
            if (c >= 'A' && c <= 'F') return static_cast<u8>(c - 'A' + 10);
            throw std::runtime_error("[python] Invalid hex char");
        };
        out[i] = static_cast<u8>((nibble(hex[i*2]) << 4) | nibble(hex[i*2+1]));
    }
    return out;
}



}