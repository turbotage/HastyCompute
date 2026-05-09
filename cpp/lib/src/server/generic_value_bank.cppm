export module hasty_server_mod:generic_value_bank;

import std;
import hasty_util_mod;
import hasty_generic_value_mod;

namespace hasty {
namespace server {

export class GenericValueBank {
public:

    bool contains(const std::string& key) const {
        std::shared_lock lock(_mutex);
        return _bank.contains(key);
    }

    std::array<u8, 16> push_value(hasty::GenericValue value) {
        auto uuid = hasty::generate_uuid();
        std::string key(reinterpret_cast<const char*>(uuid.data()), 16);
        std::unique_lock lock(_mutex);
        _bank.emplace(std::move(key), std::move(value));
        return uuid;
    }

    void push_value_with_key(const std::array<u8, 16>& uuid, hasty::GenericValue value) {
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

    std::vector<std::string> list_keys() const {
        std::shared_lock lock(_mutex);
        std::vector<std::string> keys;
        keys.reserve(_bank.size());
        for (const auto& [key, _] : _bank) {
            keys.push_back(key);
        }
        return keys;
    }

    std::optional<std::string> read_metadata(const std::string& key) const {
        std::shared_lock lock(_mutex);
        auto it = _metadata.find(key);
        if (it == _metadata.end()) return std::nullopt;
        return it->second;
    }

    void write_metadata(const std::string& key, std::string value) {
        std::unique_lock lock(_mutex);
        if (!_bank.contains(key))
            throw std::runtime_error("Key not found in GenericValueBank: " + key);
        _metadata[key] = std::move(value);
    }

    bool delete_metadata(const std::string& key) {
        std::unique_lock lock(_mutex);
        return _metadata.erase(key) > 0;
    }

private:

    mutable std::shared_mutex _mutex;
    std::unordered_map<std::string, hasty::GenericValue> _bank;
    std::unordered_map<std::string, std::string> _metadata;
};

}
}

namespace hasty {
namespace server {

    export extern GenericValueBank global_generic_value_bank;

}
}
