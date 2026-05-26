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

    CachedTensor(Tensor&& base) : _base(std::make_shared<CachedTensorBase>()) {
        _base->id = timed_uuid_to_hex_array(generate_timed_uuid());

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
    };
    std::shared_ptr<CachedTensorBase> _base;
};

}
