export module hasty_server_mod:generic_value_bank;

import std;
import hasty_util_mod;
import hasty_generic_value_mod;

namespace hasty {

export class GenericValueBank {
public:

    bool contains(const std::string& key) const {
        std::shared_lock lock(_mutex);
        return _bank.contains(key);
    }

    std::array<std::uint8_t, 16> push_value(hasty::GenericValue value) {
        auto uuid = hasty::generate_uuid();
        std::string key(reinterpret_cast<const char*>(uuid.data()), 16);
        std::unique_lock lock(_mutex);
        _bank.emplace(std::move(key), std::move(value));
        return uuid;
    }

    void push_value_with_key(const std::array<std::uint8_t, 16>& uuid, hasty::GenericValue value) {
        std::string key(reinterpret_cast<const char*>(uuid.data()), 16);
        std::unique_lock lock(_mutex);
        if (_bank.contains(key))
            throw std::runtime_error("Key already exists in GenericValueBank: " + key);
        _bank.emplace(std::move(key), std::move(value));
    }

    void push_value_with_key(const std::string& key, hasty::GenericValue value) {
        std::unique_lock lock(_mutex);
        if (_bank.contains(key))
            throw std::runtime_error("Key already exists in GenericValueBank: " + key);
        _bank[key] = std::move(value);
    }

    const hasty::GenericValue& fetch_value(const std::string& key) const {
        std::shared_lock lock(_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end())
            throw std::runtime_error("Key not found in GenericValueBank: " + key);
        return it->second;
    }

    hasty::GenericValue fetch_value(const std::string& key, const std::string& slice_info) const {
        std::shared_lock lock(_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end())
            throw std::runtime_error("Key not found in GenericValueBank: " + key);
        if (slice_info.empty())
            throw std::runtime_error("slice_info cannot be empty for fetch_value with slice_info");
        return it->second.fetch_slice(slice_info);
    }

    void write_value(const std::string& key, const hasty::GenericValue& value) {
        std::unique_lock lock(_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end())
            throw std::runtime_error("Key not found in GenericValueBank: " + key);
        it->second = value;
    }

    void write_value(const std::string& key, const std::string& slice_info, const hasty::GenericValue& value) {
        std::unique_lock lock(_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end())
            throw std::runtime_error("Key not found in GenericValueBank: " + key);
        it->second.write_slice(slice_info, value);
    }


    bool delete_value(const std::string& key) {
        std::unique_lock lock(_mutex);
        return _bank.erase(key) > 0;
    }

private:

    mutable std::shared_mutex _mutex;
    std::unordered_map<std::string, hasty::GenericValue> _bank;
};

}

namespace hasty {

    export extern GenericValueBank global_generic_value_bank;

}
