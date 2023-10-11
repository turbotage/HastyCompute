#include "opalgs.hpp"


at::Tensor hasty::op::power_iteration(const Operator& A, Vector& v, int iters)
{
	at::Tensor max_eig = v.norm();
	for (int i = 0; i < iters; ++i) {
		if (A.has_inplace_apply())
			A.apply_inplace(v);
		else
			v = std::move(A * v);

		max_eig = v.norm();

		v /= max_eig;
	}

	return max_eig;
}

void hasty::op::conjugate_gradient(const Operator& A, Vector& x, const Vector& b, const std::optional<Operator>& P, int iters)
{
	Vector r = b - A * x;
	
	Vector z = P.has_value() ? (*P) * r : r;

	Vector p = z;
	
	at::Tensor rzold = at::real(vdot(r, z));
	double resid = std::sqrt(rzold.item<double>());

	if (P.has_value()) {

		for (int i = 0; i < iters; ++i) {
			Vector Ap = A * p;
			at::Tensor pAp = at::real(vdot(p, Ap));
			if (pAp.item<double>() <= 0.0) {
				throw std::runtime_error("A was not positive definite");
			}

			at::Tensor alpha = rzold / pAp
		}
	}

}