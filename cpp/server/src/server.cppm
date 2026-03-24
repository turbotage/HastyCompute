export module hasty_server_mod;

import std;
import hasty_util_mod;
import hasty_tensor_mod;
import hasty_generic_value_mod;
import hasty_threading_mod;

// ---------------------------------------------------------------------------
// CommandRegistry
// ---------------------------------------------------------------------------

export using CommandFn = std::function<
    std::vector<hasty::GenericValue>(
        const std::string& options,
        std::vector<hasty::GenericValue> inputs)>;

export class CommandRegistry {
public:
    void register_command(std::int32_t id, std::string name, CommandFn fn) {
        _commands.emplace(id, Entry{ std::move(name), std::move(fn) });
    }

    std::vector<hasty::GenericValue> execute(
        std::int32_t id,
        const std::string& options,
        std::vector<hasty::GenericValue> inputs) const
    {
        auto it = _commands.find(id);
        if (it == _commands.end())
            throw std::runtime_error("Unknown function_id: " + std::to_string(id));
        return it->second.fn(options, std::move(inputs));
    }

    bool has_command(std::int32_t id) const { return _commands.contains(id); }

private:
    struct Entry { std::string name; CommandFn fn; };
    std::unordered_map<std::int32_t, Entry> _commands;
};

// ---------------------------------------------------------------------------
// GenericValueBank
// ---------------------------------------------------------------------------

export class GenericValueBank {
public:
    std::array<std::uint8_t, 16> push(hasty::GenericValue value) {
        auto uuid = generate_uuid();
        std::string key(reinterpret_cast<const char*>(uuid.data()), 16);
        std::unique_lock lock(_mutex);
        _bank.emplace(std::move(key), std::move(value));
        return uuid;
    }

    std::optional<hasty::GenericValue> fetch(const std::string& key) const {
        std::shared_lock lock(_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end()) return std::nullopt;
        return it->second;
    }

    bool store(const std::string& key, hasty::GenericValue value) {
        std::unique_lock lock(_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end()) return false;
        it->second = std::move(value);
        return true;
    }

    bool remove(const std::string& key) {
        std::unique_lock lock(_mutex);
        return _bank.erase(key) > 0;
    }

private:
    static std::array<std::uint8_t, 16> generate_uuid() {
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

    mutable std::shared_mutex _mutex;
    std::unordered_map<std::string, hasty::GenericValue> _bank;
};

// ---------------------------------------------------------------------------
// ServerHandle  (pimpl — Impl defined in server_impl.cpp)
// ---------------------------------------------------------------------------

export class ServerHandle {
public:
    ServerHandle() = default;
    ~ServerHandle();
    ServerHandle(ServerHandle&&) noexcept;
    ServerHandle& operator=(ServerHandle&&) noexcept;

    void wait();
    void shutdown();

    struct Impl;
private:
    std::unique_ptr<Impl> _impl;

    friend ServerHandle start_server(
        GenericValueBank&, CommandRegistry&, const std::string&);
};

// ---------------------------------------------------------------------------
// start_server  (defined in server_impl.cpp)
// ---------------------------------------------------------------------------

export ServerHandle start_server(
    GenericValueBank& bank,
    CommandRegistry& registry,
    const std::string& address = "0.0.0.0:50051");
