module;

export module tensor;

import std;
import util;
import torch_wrapper;

export import :background;

namespace hasty {

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

    tensor view(scalar_type dtype) const {
        return tensor(_base.view(to_torch(dtype)));
    }

    tensor to(scalar_type dtype, bool non_blocking=false, bool copy = false, opt<memory_format> memformat = nullopt) const {
        return tensor(
            _base.to(
                to_torch(dtype), non_blocking, copy, 
                std::bit_cast<opt<hat::MemoryFormat>>(memformat)
            )
        );
    }

    

private:
    hat::Tensor _base;
};

}