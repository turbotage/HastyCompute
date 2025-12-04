module;

export module tensor;

import std;
import util;
import torch_wrapper;

namespace hasty {


export enum struct device_idx : i16 {
    CPU = -1,
    CUDA0 = 0,
    CUDA1 = 1,
    CUDA2 = 2,
    CUDA3 = 3,
    CUDA4 = 4,
    CUDA5 = 5,
    CUDA6 = 6,
    CUDA7 = 7,
    CUDA8 = 8,
    CUDA9 = 9,
    CUDA10 = 10,
    CUDA11 = 11,
    CUDA12 = 12,
    CUDA13 = 13,
    CUDA14 = 14,
    CUDA15 = 15
};

export enum struct device_type : i16 {
    CPU = 0,
    CUDA = 1
};

export struct device {
    device_type type;
    device_idx index;

    device(device_type t, device_idx i = device_idx::CPU)
        : type(t), index(i) 
    {
        if (type == device_type::CPU && index != device_idx::CPU) {
            throw std::invalid_argument("CPU device must have index CPU");
        }
    }

    inline std::string str() const {
        if (type == device_type::CPU) {
            return "cpu";
        } else {
            return "cuda:" + std::to_string(static_cast<i16>(index));
        }
    }

    inline hat::Device torch_device() const {
        if (type == device_type::CPU) {
            return hat::Device(hat::DeviceType::CPU);
        } else {
            return hat::Device(hat::DeviceType::CUDA, hat::DeviceIndex(index));
        }
    }

    inline hat::DeviceType torch_device_type() const {
        if (type == device_type::CPU) {
            return hat::DeviceType::CPU;
        } else {
            return hat::DeviceType::CUDA;
        }
    }

    inline hat::DeviceIndex torch_device_index() const {
        if (type == device_type::CPU) {
            return hat::DeviceIndex(-1);
        } else {
            return hat::DeviceIndex(index);
        }
    }

};


export class tensor {
public:

    tensor(hat::Tensor base);

    inline tensor contiguous() const {
        return tensor(_base.contiguous());
    }

    i64 numel() const {
        return _base.numel();
    }

    template<typename T>
    T item() const {
        return _base.item<T>();
    }

    void* mutable_data_ptr() {
        return _base.mutable_data_ptr();
    }

    template<is_pure_type T>
    requires (is_tensor_type<T>)
    T* mutable_data_ptr() {
        return _base.mutable_data_ptr<T>();
    }

    template<is_pure_type T>
    requires (is_tensor_type<T>)
    const T* const_data_ptr() const {
        return _base.const_data_ptr<T>();
    }

    const void* const_data_ptr() const {
        return _base.const_data_ptr();
    }


    i64 size(i32 dim) const {
        return _base.size(dim);
    }

    dspan sizes() const {
        auto s = _base.sizes();
        return dspan(s.data(), s.size());
    }

    dspan strides() const {
        auto s = _base.strides();
        return dspan(s.data(), s.size());
    }

    bool is_contiguous() const {
        return _base.is_contiguous();
    }

    bool is_view() const {
        return _base.is_view();
    }

    tensor unsqueeze(i32 dim) const {
        return tensor(_base.unsqueeze(dim));
    }

    tensor& unsqueeze_(i32 dim) {
        _base.unsqueeze_(dim);
        return *this;
    }

    tensor view(dspan sizes) const {
        return _base.view(sizes.to_arr_ref());
    }


private:
    hat::Tensor _base;
};

}