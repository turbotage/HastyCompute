module;

module hasty_tensor_mod;

import std;
import hasty_util_mod;
import hasty_io_mod;
import hasty_generic_value_mod;

namespace hasty {

Tensor CachedTensor::load_from_file() const
{
    auto id_string = timed_uuid_hex_array_to_string(_base->id);
    auto path = hasty::io::tensor_cache_dir / id_string;
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Cached tensor file not found: " + path.string());
    }
    return hasty::io::blosc::read_generic_value(path).as_tensor();
}

void CachedTensor::push_to_file() const
{
    auto id_string = timed_uuid_hex_array_to_string(_base->id);
    auto path = hasty::io::tensor_cache_dir / id_string;
    hasty::io::blosc::write_generic_value(GenericValue(*_base->cpu_base), path);
}

}