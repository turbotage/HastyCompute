module;

export module hasty_util_mod;

export import :alias;
export import :containers;
export import :idx;
export import :io;
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

export u64 nanotime() {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}

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

// For bank keys, which are raw 16-byte strings rather than std::array<u8,16>.
// Raw key bytes must never be embedded verbatim in exception/error messages —
// they're essentially never valid UTF-8, and any such message that reaches a
// protobuf string field (e.g. via e.what() -> error_msg) corrupts the wire format.
export std::string uuid_to_hex(const std::string& key) {
    static constexpr char hex_chars[] = "0123456789abcdef";
    std::string out(key.size() * 2, '0');
    for (std::size_t i = 0; i < key.size(); ++i) {
        auto byte = static_cast<u8>(key[i]);
        out[i * 2]     = hex_chars[(byte >> 4) & 0xF];
        out[i * 2 + 1] = hex_chars[byte & 0xF];
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

export std::array<u8, 24> generate_timed_uuid() {
    std::array<u8, 16> uuid = generate_uuid();
    u64 timestamp = nanotime();
    std::array<u8, 24> out{};
    std::memcpy(out.data(), &timestamp, 8);
    std::memcpy(out.data() + 8, uuid.data(), 16);
    return out;
}


static constexpr u8 hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    throw std::runtime_error("invalid hex character");
}


export std::string timed_uuid_to_hex(const std::array<u8, 24>& data) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out;
    out.resize(48);
    for (std::size_t i = 0; i < 24; ++i) {
        out[i * 2]     = hex[data[i] >> 4];
        out[i * 2 + 1] = hex[data[i] & 0x0F];
    }
    return out;
}

export std::array<u8, 24> hex_to_timed_uuid(std::string_view hexstr) {
    if (hexstr.size() != 48) {
        throw std::runtime_error("invalid hex length");
    }
    std::array<u8, 24> out{};
    for (std::size_t i = 0; i < 24; ++i) {
        u8 hi = hex_value(hexstr[i * 2]);
        u8 lo = hex_value(hexstr[i * 2 + 1]);

        out[i] = static_cast<u8>((hi << 4) | lo);
    }
    return out;
}

export std::array<u8, 48> timed_uuid_to_hex_array(const std::array<u8, 24>& data) {
    static constexpr char hex[] = "0123456789abcdef";
    std::array<u8, 48> out{};
    for (std::size_t i = 0; i < 24; ++i) {
        out[i * 2]     = hex[data[i] >> 4];
        out[i * 2 + 1] = hex[data[i] & 0x0F];
    }
    return out;
}

export std::array<u8, 24> hex_array_to_timed_uuid(const std::array<u8, 48>& hex) {
    std::array<u8, 24> out{};
    for (std::size_t i = 0; i < 24; ++i) {
        u8 hi = hex_value(hex[i * 2]);
        u8 lo = hex_value(hex[i * 2 + 1]);
        out[i] = static_cast<u8>((hi << 4) | lo);
    }
    return out;
}

export std::string timed_uuid_hex_array_to_string(const std::array<u8, 48>& hex) {
    std::string out;
    out.reserve(48);
    for (const auto& c : hex) {
        out.push_back(static_cast<char>(c));
    }
    return out;
}

// std::hash<std::string_view> produces ONE size_t (8 bytes on 64-bit) --
// not nearly enough to fill a 24-byte id. Extend it by re-hashing the
// string concatenated with a round-specific fixed salt, round by round,
// concatenating each round's 8 bytes until n_bytes is filled (or the last
// round is truncated to whatever's left). NOT cryptographic (std::hash
// gives no such guarantee) -- adequate for cache-key collision avoidance
// in this non-adversarial setting (internal content-addressed cache keys),
// not a security boundary.
static void extend_hash_bytes(std::string_view s, u8* out, std::size_t n_bytes) {
    static constexpr std::string_view salts[] = {"", "|hasty-salt-1", "|hasty-salt-2", "|hasty-salt-3"};
    std::size_t filled = 0;
    for (std::size_t round = 0; filled < n_bytes; ++round) {
        if (round >= std::size(salts))
            throw std::runtime_error("extend_hash_bytes: n_bytes exceeds available salt rounds");
        std::string round_input(s);
        round_input += salts[round];
        std::size_t h = std::hash<std::string>{}(round_input);
        std::size_t take = std::min(sizeof(h), n_bytes - filled);
        std::memcpy(out + filled, &h, take);
        filled += take;
    }
}

// Arbitrary string -> 24-byte digest (hash-extended, see extend_hash_bytes)
// -> hex-encoded into the same 48-ASCII-char id format as
// timed_uuid_to_hex_array. Deterministic -- same string always produces the
// same id, for cache-key lookup (CachedTensor::exists/the string
// constructor).
export std::array<u8, 48> string_to_hex_array(std::string_view s) {
    std::array<u8, 24> digest{};
    extend_hash_bytes(s, digest.data(), 24);
    return timed_uuid_to_hex_array(digest);
}

// Same hashing, but the first 8 bytes are the current timestamp (matching
// generate_timed_uuid's own [8 bytes time][16 bytes payload] layout) rather
// than purely hash-derived -- every call produces a DIFFERENT id even for
// the same string, for "tag a freshly generated cache entry with a
// human-readable label" use cases, as opposed to string_to_hex_array's
// deterministic lookup-by-key use case.
export std::array<u8, 48> timed_string_to_hex_array(std::string_view s) {
    std::array<u8, 24> digest{};
    u64 timestamp = nanotime();
    std::memcpy(digest.data(), &timestamp, 8);
    extend_hash_bytes(s, digest.data() + 8, 16);
    return timed_uuid_to_hex_array(digest);
}

}