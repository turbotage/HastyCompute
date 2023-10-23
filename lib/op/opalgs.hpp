#pragma once

#include "op.hpp"
#include <optional>

namespace hasty {
	namespace op {

		at::Tensor power_iteration(const Operator& A, Vector& v, int iters=30);

		void conjugate_gradient(const Operator& A, Vector& x, const Vector& b, const std::optional<Operator>& P, 
			int iters = 30, double tol = 0.0);

		void stacked_conjugate_gradient(const Operator& A, Vector& x, const Vector& b, const std::optional<Operator>& P,
			int iters = 30, double tol = 0.0);

		void gradient_descent(const Operator& gradf, Vector& x);

		struct ADMMCtx {
			at::Tensor rho;
			Vector x;
			Vector z;
			Vector u;
			Vector c;
			Operator A;
			Operator B;

			int max_iters;
		};

		void ADMM(ADMMCtx& ctx, const std::function<void(ADMMCtx&)>& minL_x, const std::function<void(ADMMCtx&)>& minL_z);

	}
}
