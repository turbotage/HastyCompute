module;

#include "../torch_util.hpp"


module opalgs;


at::Tensor& hasty::op::OperatorAlg::access_vectensor(Vector& vec) const
{
	return vec._tensor;
}

const at::Tensor& hasty::op::OperatorAlg::access_vectensor(const Vector& vec) const
{
	return vec._tensor;
}

std::vector<hasty::op::Vector>& hasty::op::OperatorAlg::access_vecchilds(Vector& vec) const
{
	return vec._children;
}

const std::vector<hasty::op::Vector>& hasty::op::OperatorAlg::access_vecchilds(const Vector& vec) const
{
	return vec._children;
}


at::Tensor hasty::op::PowerIteration::run(const op::Operator& A, op::Vector& v, int iters)
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

hasty::op::ConjugateGradient::ConjugateGradient(std::shared_ptr<op::Operator> A, std::shared_ptr<op::Vector> b, std::shared_ptr<op::Operator> P)
	: _A(std::move(A)), _b(std::move(b)), _P(std::move(P))
{}

void hasty::op::ConjugateGradient::run(op::Vector& x, int iter, double tol)
{
	if (iter < 1)
		return;

	Vector r = *_b - (*_A) * x;

	Vector z = _P != nullptr ? (*_P) * r : r;

	Vector p = z;

	at::Tensor rzold = at::real(vdot(r, z));
	double resid = std::sqrt(rzold.item<double>());

	for (int i = 0; i < iter; ++i) {

		if (resid < tol)
			return;

		Vector Ap = (*_A) * p;
		at::Tensor pAp = at::real(vdot(p, Ap));
		if (pAp.item<double>() <= 0.0) {
			throw std::runtime_error("A was not positive definite");
		}

		at::Tensor alpha = rzold / pAp;
		x += p * alpha;

		r -= Ap * alpha;

		if (_P != nullptr)
			z = (*_P) * r;
		else
			z = r;

		at::Tensor rznew = at::real(vdot(r, z));

		// beta = rznew / rzold
		p *= rznew / rzold;
		p += z;

		rzold = rznew;

		resid = std::sqrt(rzold.item<double>());

	}
}



hasty::op::Admm::Admm(const std::shared_ptr<AdmmMinimizer>& xmin, const std::shared_ptr<AdmmMinimizer>& zmin)
	: _xmin(xmin), _zmin(zmin)
{
}

void hasty::op::Admm::run(Admm::Context& ctx)
{
	for (int i = 0; i < ctx.admm_iter; ++i) {
		_xmin->solve(ctx);
		_zmin->solve(ctx);
		*ctx.u = (*ctx.A)(*ctx.x) + (*ctx.B)(*ctx.z) - (*ctx.z);
	}

}




/*
void hasty::op::ADMM(ADMMCtx& ctx, const std::function<void(ADMMCtx&)>& minL_x, const std::function<void(ADMMCtx&)>& minL_z)
{
	for (int i = 0; i < ctx.max_iters; ++i) {
		minL_x(ctx);
		minL_z(ctx);

		ctx.u = ctx.A(ctx.x) + ctx.B(ctx.z) - ctx.c;
	}
}
*/