module;

export module hasty_tensor_mod:tensor_cached;

import std;
import hasty_util_mod;
import hasty_torch_wrapper;
import hasty_io_mod;

import :tensor;


namespace hasty {

export class CachedTensor {
public:
    CachedTensor() : _base(std::make_shared<CachedTensorBase>()) {}

    // Constructs a handle to a tensor ALREADY on disk (e.g. a previously
    // cache_in_file'd result being looked up again) -- does not load it yet
    // (loading stays lazy, via get_tensor(), same as every other path in
    // this class), but DOES eagerly verify the file exists so a missing
    // cache entry fails fast at construction, not later at first use deep
    // inside some unrelated call site.
    // persistent=false (default): looks in tensor_cache_dir, the
    // within-one-run scratch space setup_default_dirs() purges every
    // startup. persistent=true: looks in persistent_cache_dir instead,
    // which is never purged -- for caches meant to survive across separate
    // process invocations (e.g. an expensive fit keyed by config, checked
    // again on the next run).
    explicit CachedTensor(std::string_view id_string, bool persistent = false)
        : _base(std::make_shared<CachedTensorBase>())
    {
        if (!exists(id_string, persistent))
            throw std::runtime_error("CachedTensor: no cached tensor found for id " + std::string(id_string));
        _base->id = string_to_hex_array(id_string);
        _base->persistent = persistent;
    }

    // Check whether a cached tensor exists on disk for this id, without
    // constructing anything -- e.g. for a "compute once, check first next
    // time" cache: build the id from your own key, call this, and only
    // recompute + cache_in_file() on a miss.
    static bool exists(std::string_view id_string, bool persistent = false);

    CachedTensor(Tensor&& base, std::optional<std::string_view> id_string = std::nullopt, bool persistent = false)
        : _base(std::make_shared<CachedTensorBase>())
    {
        if (id_string.has_value()) {
            _base->id = string_to_hex_array(*id_string);
        } else {
            _base->id = timed_uuid_to_hex_array(generate_timed_uuid());
        }
        _base->persistent = persistent;

        if (base.device().type == eDeviceType::CPU) {
            _base->cpu_base = std::move(base);
        } else if (base.device().type == eDeviceType::CUDA) {
            if (base.device().index > device_alias::MAX_CUDA_DEVICES)
                throw std::invalid_argument("Device index out of range for caching");
            _base->cuda_cache[base.device().index] = std::move(base);
        } else {
            throw std::invalid_argument("Unsupported device type for caching");
        }
    }

    CachedTensor(const CachedTensor& other) = delete;
    CachedTensor(CachedTensor&& other) noexcept = default;

    CachedTensor& operator=(const CachedTensor& other) & = delete;
    CachedTensor& operator=(CachedTensor&& other) & noexcept = default;

    ~CachedTensor() {
        if (!_base) return;
        // Persistent entries are meant to survive past this object's
        // lifetime -- only transient (tensor_cache_dir) entries get cleaned
        // up here.
        if (_base->persistent) return;
        auto id_string = timed_uuid_hex_array_to_string(_base->id);
        remove_file_if_exists(hasty::io::tensor_cache_dir / id_string);
    }

    inline const Tensor& get_tensor(Device device) const {
        return load_onto_device(device);
    }

    inline Tensor& get_mutable_tensor(Device device) {
        return load_onto_device(device);
    }

    void cache_on_cpu() {
        if (_base->cpu_base.has_value()) return;
        for (auto& value : _base->cuda_cache) {
            if (value.has_value()) {
                _base->cpu_base = value->to(Device{eDeviceType::CPU, -1});
                return;
            }
        }
    }

    // Move to CPU and evict all CUDA copies — keeps only one live copy on CPU.
    void move_to_cpu() {
        cache_on_cpu();
        for (auto& entry : _base->cuda_cache) entry = nullopt;
    }

    void cache_in_file(bool evict_after_caching = true) {
        if (!_base->cpu_base.has_value()) {
            cache_on_cpu();
        }
        push_to_file();
        if (evict_after_caching) {
            _base->cpu_base = nullopt;
            for (auto& entry : _base->cuda_cache) entry = nullopt;
        }
    }

    void clear_cache_on_device(Device device) {
        if (device.type == eDeviceType::CPU) {
            _base->cpu_base = nullopt;
        } else if (device.type == eDeviceType::CUDA) {
            if (device.index > device_alias::MAX_CUDA_DEVICES)
                throw std::invalid_argument("Device index out of range for caching");
            _base->cuda_cache[device.index] = nullopt;
        } else {
            throw std::invalid_argument("Unsupported device type for caching");
        }
    }

    void clear_all_cache() {
        _base->cpu_base = nullopt;
        for (auto& entry : _base->cuda_cache) entry = nullopt;
    }

    bool is_cached_on_device(Device device) const {
        if (device.type == eDeviceType::CPU) {
            return _base->cpu_base.has_value();
        } else if (device.type == eDeviceType::CUDA) {
            if (device.index > device_alias::MAX_CUDA_DEVICES)
                throw std::invalid_argument("Device index out of range for caching");
            return _base->cuda_cache[device.index].has_value();
        } else {
            throw std::invalid_argument("Unsupported device type for caching");
        }
    }

private:

    Tensor load_from_file() const;
    void push_to_file() const;

    Tensor& load_onto_device(Device device) const {
        if (device.type == eDeviceType::CPU)        return load_onto_cpu();
        else if (device.type == eDeviceType::CUDA)  return load_onto_cuda(device);
        throw std::invalid_argument("Unsupported device type for caching");
    }

    Tensor& load_onto_cpu() const {
        if (_base->cpu_base.has_value()) return *_base->cpu_base;
        for (auto& value : _base->cuda_cache) {
            if (value.has_value()) {
                _base->cpu_base = value->to(Device{eDeviceType::CPU, -1});
                return *_base->cpu_base;
            }
        }
        _base->cpu_base = load_from_file();
        if (_base->cpu_base->device().type != eDeviceType::CPU)
            throw std::runtime_error(
                "Cached tensor loaded from file has wrong device type: expected CPU but found " +
                _base->cpu_base->device().str());
        return *_base->cpu_base;
    }

    Tensor& load_onto_cuda(Device device) const {
        if (device.type != eDeviceType::CUDA)
            throw std::invalid_argument("load_onto_cuda called with non-CUDA device");
        if (device.index > device_alias::MAX_CUDA_DEVICES)
            throw std::invalid_argument("Device index out of range for caching");
        if (_base->cuda_cache[device.index].has_value())
            return *_base->cuda_cache[device.index];
        for (auto& value : _base->cuda_cache) {
            if (value.has_value()) {
                _base->cuda_cache[device.index] = value->to(device);
                return *_base->cuda_cache[device.index];
            }
        }
        if (_base->cpu_base.has_value()) {
            _base->cuda_cache[device.index] = _base->cpu_base->to(device);
            return *_base->cuda_cache[device.index];
        }
        _base->cpu_base = load_from_file();
        if (_base->cpu_base->device().type != eDeviceType::CPU)
            throw std::runtime_error(
                "Cached tensor loaded from file has wrong device type: expected CPU but found " +
                _base->cpu_base->device().str());
        _base->cuda_cache[device.index] = _base->cpu_base->to(device);
        _base->cpu_base = nullopt;  // if tensor wasnt already on CPU we evict CPU copy after loading to CUDA
        return *_base->cuda_cache[device.index];
    }

    struct CachedTensorBase {
        Opt<Tensor> cpu_base;
        Arr<Opt<Tensor>, device_alias::MAX_CUDA_DEVICES> cuda_cache;
        Arr<u8, 48> id;
        bool persistent = false;
    };
    std::shared_ptr<CachedTensorBase> _base;
};

}
