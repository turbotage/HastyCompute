module;

module hasty_tensor_mod;

import std;
import hasty_util_mod;
import hasty_io_mod;
import hasty_generic_value_mod;

namespace hasty {

bool CachedTensor::exists(std::string_view id_string, bool persistent)
{
    // id_string is an arbitrary caller-chosen key (e.g. a config-derived
    // cache key), not the on-disk hex id itself -- hash it the same way the
    // constructor/push_to_file do before building the path, or this would
    // check the wrong file entirely.
    auto hex_id = timed_uuid_hex_array_to_string(string_to_hex_array(id_string));
    auto& dir = persistent ? hasty::io::persistent_cache_dir : hasty::io::tensor_cache_dir;
    auto path = dir / hex_id;
    return std::filesystem::exists(path);
}

Tensor CachedTensor::load_from_file() const
{
    auto id_string = timed_uuid_hex_array_to_string(_base->id);
    auto& dir = _base->persistent ? hasty::io::persistent_cache_dir : hasty::io::tensor_cache_dir;
    auto path = dir / id_string;
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Cached tensor file not found: " + path.string());
    }
    return hasty::io::blosc::read_generic_value(path).as_tensor();
}

void CachedTensor::push_to_file() const
{
    auto id_string = timed_uuid_hex_array_to_string(_base->id);
    auto& dir = _base->persistent ? hasty::io::persistent_cache_dir : hasty::io::tensor_cache_dir;
    auto path = dir / id_string;
    hasty::io::blosc::write_generic_value(GenericValue(*_base->cpu_base), path);
}

}