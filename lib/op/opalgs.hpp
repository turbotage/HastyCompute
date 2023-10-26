#pragma once

#include "op.hpp"
#include <optional>
#include "../threading/thread_pool.hpp"

namespace hasty {
	namespace op {

		at::Tensor power_iteration(const Operator& A, Vector& v, int iters=30);

		void conjugate_gradient(const Operator& A, Vector& x, const Vector& b, const std::optional<Operator>& P, 
			int iters = 30, double tol = 0.0);

		void stacked_conjugate_gradient(const Operator& A, Vector& x, const Vector& b, const std::optional<Operator>& P,
			int iters = 30, double tol = 0.0);

		void gradient_descent(const Operator& gradf, Vector& x);


		template<typename DeviceContext>
		class ConjugateGradientLoader {
		public:

			struct LoadResult {
				op::Operator A;
				op::Vector b;
				std::optional<Operator> P;
			};

			virtual LoadResult load(DeviceContext& dctx, size_t idx) = 0;
			
		};

		template<typename DeviceContext>
		class DefaultConjugateGradientLoader : public ConjugateGradientLoader<DeviceContext> {
		public:

			DefaultConjugateGradientLoader(const LoadResult& lr)
				: _load_result({lr}) {}

			DefaultConjugateGradientLoader(const std::vector<LoadResult>& lrs)
				: _load_result(lrs) {}

			LoadResult load(DeviceContext& dctxt, size_t idx)
			{

			}

		private:
			std::vector<LoadResult> _load_results;
		};

		template<typename DeviceContext>
		class ConjugateGradient {
		public:

			ConjugateGradient(const std::shared_ptr<ConjugateGradientLoader>& loader,
				const std::shared_ptr<ContextThreadPool<DeviceContext>>& thread_pool,
				int iters = 30, double tol = 0.0)
			{

			}

		private:
			std::shared_ptr<ConjugateGradientLoader> _cg_loader;
			std::shared_ptr<ContextThreadPool<DeviceContext>> _thread_pool;
		};


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
