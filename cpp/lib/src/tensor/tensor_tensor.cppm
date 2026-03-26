module;

#include <cuComplex.h>

export module hasty_tensor_mod:tensor;

import std;
import hasty_util_mod;
import hasty_torch_wrapper;

import :background;
import :scalar;


namespace hasty {

export class Tensor {
public:

    Tensor() = default;

    Tensor(hat::Tensor base) : _base(std::move(base)) {}

    Tensor(const Tensor& other) : _base(other._base) {}
    Tensor(Tensor&& other) noexcept : _base(std::move(other._base)) {}

    Tensor& operator=(const Tensor& other) & noexcept { _base = other._base; return *this; }
    Tensor& operator=(Tensor&& other) & noexcept { _base = std::move(other._base); return *this; }
    Tensor& operator=(const Scalar& other) && { return fill_(other); }

    inline Tensor& operator=(const Tensor& rhs) && { return copy_(rhs); }
    inline Tensor& operator=(Tensor&& rhs) && { return copy_(rhs); }

    inline Tensor operator~() const { return Tensor(_base.bitwise_not()); }
    inline Tensor operator-() const { return Tensor(_base.neg()); }

    inline Tensor& operator+=(const Tensor& other) { return add_(other); }
    inline Tensor& operator+=(const Scalar& other) { return add_(other); }
    inline Tensor& operator-=(const Tensor& other) { return sub_(other); }
    inline Tensor& operator-=(const Scalar& other) { return sub_(other); }
    inline Tensor& operator*=(const Tensor& other) { return mul_(other); }
    inline Tensor& operator*=(const Scalar& other) { return mul_(other); }
    inline Tensor& operator/=(const Tensor& other) { return div_(other); }
    inline Tensor& operator/=(const Scalar& other) { return div_(other); }
    inline Tensor& operator&=(const Tensor& other) { return bitwise_and_(other); }
    inline Tensor& operator&=(const Scalar& other) { return bitwise_and_(other); }
    inline Tensor& operator|=(const Tensor& other) { return bitwise_or_(other); }
    inline Tensor& operator|=(const Scalar& other) { return bitwise_or_(other); }
    inline Tensor& operator^=(const Tensor& other) { return bitwise_xor_(other); }
    inline Tensor& operator^=(const Scalar& other) { return bitwise_xor_(other); }

    // LibTorch extensions

    inline static Tensor from_blob(void* data, ArrayRef<i64> sizes, eScalarType dtype, Device device) {
        return Tensor(hat::from_blob(data, sizes.to_torch(), TensorOptions(device).dtype(dtype).to_torch()));
    }

    inline static Tensor from_vector(std::vector<u8>&& data, ArrayRef<i64> sizes, eScalarType dtype, Device device) {
        auto options = TensorOptions(device).dtype(dtype).to_torch();
        void* data_ptr = data.data();

        auto shared_ptr_vec = std::make_shared<std::vector<u8>>(std::move(data));

        auto deleter = [vec = std::move(shared_ptr_vec)](void* ptr) mutable {};

        auto tensor = hat::from_blob(data_ptr, sizes.to_torch(), deleter, options);

        return Tensor(tensor);
    }

    inline std::pair<eDeviceType, i32> get_device_info() const {
        const auto& device = this->device();
        return {device.type, device.has_index() ? device.index : -1};
    }

    std::string toString() const { return _base.toString(); }

    std::string metadata_string() const {
        std::string device_str = device().str();
        std::string dtype_str = scalar_type_to_string(scalar_type());
        std::string shape_str = "shape=(";
        for (int i = 0; i < _base.dim(); ++i) {
            if (i > 0) shape_str += ",";
            shape_str += std::to_string(_base.size(i));
        }
        shape_str += ")";
        return "Tensor[dtype=" + dtype_str + ",device=" + device_str + "," + shape_str + "]";
    }

    std::string statistics_string() const {
        std::string ret = metadata_string();
        ret += "\n\t min=" + std::to_string(_base.min().item<double>());
        ret += "\n\t max=" + std::to_string(_base.max().item<double>());
        ret += "\n\t mean=" + std::to_string(_base.mean().item<double>());
        ret += "\n\t std=" + std::to_string(_base.std().item<double>());
        ret += "\n\t median=" + std::to_string(_base.median().item<double>());
        ret += "\n\t";
        return ret;
    }

    // LibTorch wrappers

    inline Tensor contiguous() const { return Tensor(_base.contiguous()); }

    inline i64 numel() const { return _base.numel(); }

    template<typename T>
    inline T item() const { return _base.item<T>(); }

    inline void* mutable_data_ptr() const { return _base.mutable_data_ptr(); }

    template<is_pure_type T>
    requires (is_tensor_type<T>)
    inline T* mutable_data_ptr() const {
        // ATen only pre-instantiates mutable_data_ptr for c10::complex, not std::complex.
        // For complex types route through void* to avoid a missing symbol at link time.
        if constexpr (std::is_same_v<T, c64> || std::is_same_v<T, c128>)
            return reinterpret_cast<T*>(_base.mutable_data_ptr());
        else
            return _base.mutable_data_ptr<T>();
    }

    template<is_pure_type T>
    requires (is_tensor_type<T>)
    inline const T* const_data_ptr() const {
        if constexpr (std::is_same_v<T, c64> || std::is_same_v<T, c128>)
            return reinterpret_cast<const T*>(_base.const_data_ptr());
        else
            return _base.const_data_ptr<T>();
    }

    inline const void* const_data_ptr() const { return _base.const_data_ptr(); }

    template<is_pure_type T>
    const T* cast_const_data_ptr() const {
        if constexpr(is_tensor_type<T>) {
            return reinterpret_cast<const T*>(_base.const_data_ptr());
        }
        else if constexpr(std::is_same_v<T, cuFloatComplex>) {
            if (scalar_type() != eScalarType::ComplexFloat) {
                throw std::runtime_error("Tensor scalar type is not ComplexFloat");
            }
            return reinterpret_cast<const T*>(_base.const_data_ptr());
        }
        else if constexpr(std::is_same_v<T, cuDoubleComplex>) {
            if (scalar_type() != eScalarType::ComplexDouble) {
                throw std::runtime_error("Tensor scalar type is not ComplexDouble");
            }
            return reinterpret_cast<const T*>(_base.const_data_ptr());
        }
        else {
            static_assert(always_false<T>, "Unsupported type for cast_const_data_ptr");
        }
    }

    template<is_pure_type T>
    T* cast_data_ptr() const {
        if constexpr(is_tensor_type<T>) {
            return reinterpret_cast<T*>(_base.mutable_data_ptr());
        }
        else if constexpr(std::is_same_v<T, cuFloatComplex>) {
            if (scalar_type() != eScalarType::ComplexFloat) {
                throw std::runtime_error("Tensor scalar type is not ComplexFloat");
            }
            return reinterpret_cast<T*>(_base.mutable_data_ptr());
        }
        else if constexpr(std::is_same_v<T, cuDoubleComplex>) {
            if (scalar_type() != eScalarType::ComplexDouble) {
                throw std::runtime_error("Tensor scalar type is not ComplexDouble");
            }
            return reinterpret_cast<T*>(_base.mutable_data_ptr());
        }
        else {
            static_assert(always_false<T>, "Unsupported type for cast_const_data_ptr");
        }
    }

    inline Device device() const { return Device(_base.device()); }

    inline eScalarType scalar_type() const {
        return scalartype::from_torch(_base.scalar_type());
    }

    inline eScalarType dtype() const {
        return scalartype::from_torch(_base.scalar_type());
    }

    inline i64 size(i32 dim) const { return _base.size(dim); }

    inline ArrayRef<i64> sizes() const { return ArrayRef<i64>(_base.sizes()); }

    inline i64 ndimension() const { return _base.ndimension(); }

    inline ArrayRef<i64> strides() const { return ArrayRef<i64>(_base.strides()); }

    inline bool is_contiguous() const { return _base.is_contiguous(); }

    inline bool is_view() const { return _base.is_view(); }

    inline bool is_complex() const noexcept { return _base.is_complex(); }

    inline Tensor unsqueeze(i32 dim) const { return Tensor(_base.unsqueeze(dim)); }

    inline Tensor& unsqueeze_(i32 dim) { _base.unsqueeze_(dim); return *this; }

    inline Tensor view(ArrayRef<i64> sizes) const { return _base.view(sizes.to_torch()); }

    inline Tensor view(eScalarType dtype) const { return Tensor(_base.view(scalartype::to_torch(dtype))); }

    inline Tensor view_as(const Tensor& other) const { return Tensor(_base.view_as(other._base)); }

    inline Tensor flip(ArrayRef<i64> dims) const { return Tensor(_base.flip(dims.to_torch())); }

    inline Tensor narrow(i64 dim, i64 start, i64 length) const { return Tensor(_base.narrow(dim, start, length)); }

    inline Tensor conj() const { return Tensor(_base.conj()); }

    inline Tensor clone() const { return Tensor(_base.clone()); }

    inline std::vector<i64> sizes_vec() const {
        auto s = _base.sizes();
        return std::vector<i64>(s.begin(), s.end());
    }

    inline Tensor remainder(const Scalar& other) const { return Tensor(_base.remainder(other.to_torch())); }
    inline Tensor& remainder_(const Scalar& other) { _base.remainder_(other.to_torch()); return *this; }
    inline Tensor remainder(const Tensor& other) const { return Tensor(_base.remainder(other._base)); }
    inline Tensor& remainder_(const Tensor& other) { _base.remainder_(other._base); return *this; }

    inline Tensor to(eScalarType dtype, bool non_blocking=false, bool copy = false, Opt<eMemoryFormat> memformat = nullopt) const {
        return Tensor(
            _base.to(
                scalartype::to_torch(dtype), non_blocking, copy, 
                std::bit_cast<Opt<hat::MemoryFormat>>(memformat)
            )
        );
    }

    Tensor to(Device dev, bool non_blocking=false, bool copy = false, Opt<eMemoryFormat> memformat = nullopt) const {
        return Tensor(
            _base.to(
                dev.torch_device(), non_blocking, copy, 
                std::bit_cast<Opt<hat::MemoryFormat>>(memformat)
            )
        );
    }

    Tensor to(const TensorOptions& options, bool non_blocking=false, bool copy = false) const {
        return Tensor(
            _base.to(
                options.to_torch(), non_blocking, copy
            )
        );
    }
    
    inline Tensor select(i64 dim, i64 index) const { return Tensor(_base.select(dim, index)); }
    inline Tensor select_scatter(const Tensor& src, i64 dim, i64 index) const { return Tensor(_base.select_scatter(src._base, dim, index)); }
    inline Tensor index_select(i64 dim, const Tensor& index) const { return Tensor(_base.index_select(dim, index._base)); }
    inline Tensor masked_select(const Tensor& mask) const { return Tensor(_base.masked_select(mask._base)); }
    
    inline Tensor& fill_(const Scalar& value) const {
        _base.fill_(value.to_torch());
        return const_cast<Tensor&>(*this);
    }

    inline Tensor& fill_(const Tensor& value) const {
        _base.fill_(value._base);
        return const_cast<Tensor&>(*this);
    }

    inline Tensor& copy_(const Tensor& src, bool non_blocking=false) const {
        _base.copy_(src._base, non_blocking);
        return const_cast<Tensor&>(*this);
    }


    Tensor add(const Tensor& other, const Scalar& alpha=1) const {
        return Tensor(_base.add(other._base, alpha.to_torch()));
    }
    Tensor& add_(const Tensor& other, const Scalar& alpha=1) {
        _base.add_(other._base, alpha.to_torch());
        return *this;
    }
    Tensor add(const Scalar& other, const Scalar& alpha=1) const {
        return Tensor(_base.add(other.to_torch(), alpha.to_torch()));
    }
    Tensor& add_(const Scalar& other, const Scalar& alpha=1) {
        _base.add_(other.to_torch(), alpha.to_torch());
        return *this;
    }

    Tensor sub(const Tensor& other, const Scalar& alpha=1) const {
        return Tensor(_base.sub(other._base, alpha.to_torch()));
    }
    Tensor& sub_(const Tensor& other, const Scalar& alpha=1) {
        _base.sub_(other._base, alpha.to_torch());
        return *this;
    }
    Tensor sub(const Scalar& other, const Scalar& alpha=1) const {
        return Tensor(_base.sub(other.to_torch(), alpha.to_torch()));
    }
    Tensor& sub_(const Scalar& other, const Scalar& alpha=1) {
        _base.sub_(other.to_torch(), alpha.to_torch());
        return *this;
    }

    Tensor mul(const Tensor& other) const {
        return Tensor(_base.mul(other._base));
    }
    Tensor& mul_(const Tensor& other) {
        _base.mul_(other._base);
        return *this;
    }
    Tensor mul(const Scalar& other) const {
        return Tensor(_base.mul(other.to_torch()));
    }
    Tensor& mul_(const Scalar& other) {
        _base.mul_(other.to_torch());
        return *this;
    }

    Tensor div(const Tensor& other) const {
        return Tensor(_base.div(other._base));
    }
    Tensor& div_(const Tensor& other) {
        _base.div_(other._base);
        return *this;
    }
    Tensor div(const Scalar& other) const {
        return Tensor(_base.div(other.to_torch()));
    }
    Tensor& div_(const Scalar& other) {
        _base.div_(other.to_torch());
        return *this;
    }

    Tensor bitwise_and(const Tensor& other) const {
        return Tensor(_base.bitwise_and(other._base));
    }
    Tensor& bitwise_and_(const Tensor& other) {
        _base.bitwise_and_(other._base);
        return *this;
    }
    Tensor bitwise_and(const Scalar& other) const {
        return Tensor(_base.bitwise_and(other.to_torch()));
    }
    Tensor& bitwise_and_(const Scalar& other) {
        _base.bitwise_and_(other.to_torch());
        return *this;
    }

    Tensor bitwise_or(const Tensor& other) const {
        return Tensor(_base.bitwise_or(other._base));
    }
    Tensor& bitwise_or_(const Tensor& other) {
        _base.bitwise_or_(other._base);
        return *this;
    }
    Tensor bitwise_or(const Scalar& other) const {
        return Tensor(_base.bitwise_or(other.to_torch()));
    }
    Tensor& bitwise_or_(const Scalar& other) {
        _base.bitwise_or_(other.to_torch());
        return *this;
    }

    Tensor bitwise_xor(const Tensor& other) const {
        return Tensor(_base.bitwise_xor(other._base));
    }
    Tensor& bitwise_xor_(const Tensor& other) {
        _base.bitwise_xor_(other._base);
        return *this;
    }
    Tensor bitwise_xor(const Scalar& other) const {
        return Tensor(_base.bitwise_xor(other.to_torch()));
    }
    Tensor& bitwise_xor_(const Scalar& other) {
        _base.bitwise_xor_(other.to_torch());
        return *this;
    }






    hat::Tensor to_torch() const { return _base; }

private:
    hat::Tensor _base;


    friend Tensor empty(ArrayRef<i64> sizes, TensorOptions options);
    friend Tensor empty_like(const Tensor& other);
    friend Tensor zeros(ArrayRef<i64> sizes, TensorOptions options);
    friend Tensor zeros_like(const Tensor& other);
    friend Tensor ones(ArrayRef<i64> sizes, TensorOptions options);
    friend Tensor ones_like(const Tensor& other);
    friend Tensor rand(ArrayRef<i64> sizes, TensorOptions options);
    friend Tensor rand_like(const Tensor& other);

};
 

}