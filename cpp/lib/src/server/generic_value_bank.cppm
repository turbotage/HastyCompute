export module hasty_server_mod:generic_value_bank;

import std;
import hasty_util_mod;
import hasty_generic_value_mod;

namespace hasty {
namespace server {

// Per-key, optional hooks. "on_*" runs before the action, given the value/metadata
// as it stands before the action (mutable, so it can transform what gets
// committed/returned). "after_*" runs once the action has completed.
// Delete has no "after" hook: once an entry is erased there's nothing left to
// hand the callback that means anything.
export struct GenericValueBankCallbacks {
    using Callback = std::function<void(hasty::GenericValue& value, std::string& metadata)>;

    Opt<Callback> on_write_callback;
    Opt<Callback> after_write_callback;

    Opt<Callback> on_fetch_callback;
    Opt<Callback> after_fetch_callback;

    Opt<Callback> on_delete_callback;

    Opt<Callback> on_write_metadata_callback;
    Opt<Callback> after_write_metadata_callback;

    Opt<Callback> on_read_metadata_callback;
    Opt<Callback> after_read_metadata_callback;

    Opt<Callback> on_delete_metadata_callback;
};

export class GenericValueBank {
public:
    std::array<u8, 16> push_value(hasty::GenericValue value, Opt<GenericValueBankCallbacks> callbacks = std::nullopt) {
        auto uuid = hasty::generate_uuid();
        std::string key(reinterpret_cast<const char*>(uuid.data()), 16);
        std::unique_lock map_lock(_map_mutex);
        _bank.try_emplace(std::move(key), std::move(value), std::string{}, std::move(callbacks));
        return uuid;
    }

    void push_value_with_key(const std::array<u8, 16>& uuid, hasty::GenericValue value, Opt<GenericValueBankCallbacks> callbacks = std::nullopt) {
        std::string key(reinterpret_cast<const char*>(uuid.data()), 16);
        std::unique_lock map_lock(_map_mutex);
        if (_bank.contains(key))
            throw std::runtime_error("Key already exists in GenericValueBank: " + hasty::uuid_to_hex(key));
        _bank.try_emplace(std::move(key), std::move(value), std::string{}, std::move(callbacks));
    }

    void push_value_with_key(const std::string& key, hasty::GenericValue value, Opt<GenericValueBankCallbacks> callbacks = std::nullopt) {
        std::unique_lock map_lock(_map_mutex);
        if (_bank.contains(key))
            throw std::runtime_error("Key already exists in GenericValueBank: " + hasty::uuid_to_hex(key));
        _bank.try_emplace(key, std::move(value), std::string{}, std::move(callbacks));
    }

    void clear_callbacks(const std::string& key) {
        std::shared_lock map_lock(_map_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end()) return;
        std::unique_lock entry_lock(it->second.mutex);
        it->second.callbacks.reset();
    }

    bool contains(const std::string& key) const {
        std::shared_lock map_lock(_map_mutex);
        return _bank.contains(key);
    }

    // The outer map lock only ever guards map structure (key insertion/erasure);
    // it's taken shared here so lookups on different keys never contend with
    // each other. Per-entry work then takes that entry's own mutex, so two
    // threads touching different keys run fully concurrently, and only same-key
    // access serializes — including any value mutation a callback performs.
    const hasty::GenericValue& fetch_value(const std::string& key) const {
        std::shared_lock map_lock(_map_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end())
            throw std::runtime_error("Key not found in GenericValueBank: " + hasty::uuid_to_hex(key));
        BankEntry& entry = it->second;
        std::unique_lock entry_lock(entry.mutex);
        if (entry.callbacks && entry.callbacks->on_fetch_callback)
            (*entry.callbacks->on_fetch_callback)(entry.value, entry.metadata);
        entry_lock.unlock();
        if (entry.callbacks && entry.callbacks->after_fetch_callback)
            (*entry.callbacks->after_fetch_callback)(entry.value, entry.metadata);
        return entry.value;
    }

    hasty::GenericValue fetch_value(const std::string& key, const std::string& slice_info) const {
        if (slice_info.empty())
            throw std::runtime_error("slice_info cannot be empty for fetch_value with slice_info");
        std::shared_lock map_lock(_map_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end())
            throw std::runtime_error("Key not found in GenericValueBank: " + hasty::uuid_to_hex(key));
        BankEntry& entry = it->second;
        std::unique_lock entry_lock(entry.mutex);
        // on_fetch runs on the full stored value before the slice is taken, so
        // it can transform what gets sliced.
        if (entry.callbacks && entry.callbacks->on_fetch_callback)
            (*entry.callbacks->on_fetch_callback)(entry.value, entry.metadata);
        hasty::GenericValue sliced = entry.value.fetch_slice(slice_info);
        entry_lock.unlock();
        // after_fetch runs on the sliced result, not the stored value.
        if (entry.callbacks && entry.callbacks->after_fetch_callback)
            (*entry.callbacks->after_fetch_callback)(sliced, entry.metadata);
        return sliced;
    }

    void write_value(const std::string& key, hasty::GenericValue value) {
        std::shared_lock map_lock(_map_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end())
            throw std::runtime_error("Key not found in GenericValueBank: " + hasty::uuid_to_hex(key));
        BankEntry& entry = it->second;
        std::unique_lock entry_lock(entry.mutex);
        if (entry.callbacks && entry.callbacks->on_write_callback)
            (*entry.callbacks->on_write_callback)(value, entry.metadata);
        entry.value = std::move(value);
        if (entry.callbacks && entry.callbacks->after_write_callback)
            (*entry.callbacks->after_write_callback)(entry.value, entry.metadata);
    }

    void write_value(const std::string& key, const std::string& slice_info, hasty::GenericValue value) {
        std::shared_lock map_lock(_map_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end())
            throw std::runtime_error("Key not found in GenericValueBank: " + hasty::uuid_to_hex(key));
        BankEntry& entry = it->second;
        std::unique_lock entry_lock(entry.mutex);
        if (entry.callbacks && entry.callbacks->on_write_callback)
            (*entry.callbacks->on_write_callback)(value, entry.metadata);
        entry.value.write_slice(slice_info, value);
        if (entry.callbacks && entry.callbacks->after_write_callback)
            (*entry.callbacks->after_write_callback)(entry.value, entry.metadata);
    }

    // Structural removal needs the map lock exclusively (it changes which keys
    // exist), so this naturally serializes against every other operation on
    // this key — no separate entry lock needed.
    bool delete_value(const std::string& key) {
        std::unique_lock map_lock(_map_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end()) return false;
        if (it->second.callbacks && it->second.callbacks->on_delete_callback)
            (*it->second.callbacks->on_delete_callback)(it->second.value, it->second.metadata);
        _bank.erase(it);
        return true;
    }

    std::vector<std::string> list_keys() const {
        std::shared_lock map_lock(_map_mutex);
        std::vector<std::string> keys;
        keys.reserve(_bank.size());
        for (const auto& [key, _] : _bank) {
            keys.push_back(key);
        }
        return keys;
    }

    std::optional<std::string> read_metadata(const std::string& key) const {
        std::shared_lock map_lock(_map_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end()) return std::nullopt;
        BankEntry& entry = it->second;
        std::unique_lock entry_lock(entry.mutex);
        if (entry.callbacks && entry.callbacks->on_read_metadata_callback)
            (*entry.callbacks->on_read_metadata_callback)(entry.value, entry.metadata);
        std::string metadata_copy = entry.metadata;
        if (entry.callbacks && entry.callbacks->after_read_metadata_callback)
            (*entry.callbacks->after_read_metadata_callback)(entry.value, metadata_copy);
        return metadata_copy;
    }

    void write_metadata(const std::string& key, std::string value) {
        std::shared_lock map_lock(_map_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end())
            throw std::runtime_error("Key not found in GenericValueBank: " + hasty::uuid_to_hex(key));
        BankEntry& entry = it->second;
        std::unique_lock entry_lock(entry.mutex);
        if (entry.callbacks && entry.callbacks->on_write_metadata_callback)
            (*entry.callbacks->on_write_metadata_callback)(entry.value, value);
        entry.metadata = std::move(value);
        if (entry.callbacks && entry.callbacks->after_write_metadata_callback)
            (*entry.callbacks->after_write_metadata_callback)(entry.value, entry.metadata);
    }

    bool delete_metadata(const std::string& key) {
        std::shared_lock map_lock(_map_mutex);
        auto it = _bank.find(key);
        if (it == _bank.end()) return false;
        BankEntry& entry = it->second;
        std::unique_lock entry_lock(entry.mutex);
        if (entry.metadata.empty() && !entry.callbacks) return false;
        if (entry.callbacks && entry.callbacks->on_delete_metadata_callback)
            (*entry.callbacks->on_delete_metadata_callback)(entry.value, entry.metadata);
        bool had_metadata = !entry.metadata.empty();
        entry.metadata.clear();
        return had_metadata;
    }

    void clear_all_values() {
        std::unique_lock map_lock(_map_mutex);
        _bank.clear();
    }

    void clear_all_metadata() {
        std::shared_lock map_lock(_map_mutex);
        for (auto& [_, entry] : _bank) {
            std::unique_lock entry_lock(entry.mutex);
            entry.metadata.clear();
        }
    }

    void clear_all() {
        std::unique_lock map_lock(_map_mutex);
        _bank.clear();
    }

private:
    struct BankEntry {
        BankEntry(hasty::GenericValue v, std::string m, Opt<GenericValueBankCallbacks> c)
            : value(std::move(v)), metadata(std::move(m)), callbacks(std::move(c)) {}

        hasty::GenericValue value;
        std::string metadata;
        Opt<GenericValueBankCallbacks> callbacks;
        // Guards this entry's value/metadata/callbacks only — independent of
        // every other entry's mutex, so different keys never contend.
        mutable std::mutex mutex;
    };

    mutable std::shared_mutex _map_mutex;
    // mutable: const methods (fetch_value, read_metadata, ...) still need to
    // mutate an entry's own mutex/value/metadata under that entry's lock —
    // the map structure itself isn't touched by them.
    mutable std::unordered_map<std::string, BankEntry> _bank;
};

}
}

namespace hasty {
namespace server {

    export extern GenericValueBank global_generic_value_bank;

}
}
