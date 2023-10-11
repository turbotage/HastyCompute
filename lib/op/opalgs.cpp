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
	if (P.has_value()) {
		Vector r = b - A * x;
		Vector p = r.clone();

		for (int i = 0; i < iters; ++i) {
			Vector Ap = A * p;
			at::Tensor pAp = at::real()
		}
	}

}