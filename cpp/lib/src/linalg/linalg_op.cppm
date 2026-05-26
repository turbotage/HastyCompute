module;

export module hasty_linalg_mod:op;

import std;
import hasty_util_mod;
import hasty_tensor_mod;

namespace hasty {
namespace linalg {

// ─── LinearOperator ──────────────────────────────────────────────────────────

export class LinearOperator {
public:
    using fn_t = std::function<Tensor(const Tensor&)>;

    LinearOperator(i64 m, i64 n, fn_t mv, fn_t rmv,
                   eScalarType dtype, Device device)
        : _m(m), _n(n), _mv(std::move(mv)), _rmv(std::move(rmv)),
          _dtype(dtype), _dev(device)
    {
        if (dtype != eScalarType::ComplexFloat &&
            dtype != eScalarType::ComplexDouble)
            throw std::invalid_argument(
                "LinearOperator: dtype must be ComplexFloat or ComplexDouble");
    }

    i64         m()      const noexcept { return _m; }
    i64         n()      const noexcept { return _n; }
    eScalarType dtype()  const noexcept { return _dtype; }
    Device      device() const noexcept { return _dev; }

    Tensor matvec (const Tensor& x) const { return _mv(x); }
    Tensor rmatvec(const Tensor& y) const { return _rmv(y); }

private:
    i64 _m, _n;
    fn_t _mv, _rmv;
    eScalarType _dtype;
    Device _dev;
};





}
}