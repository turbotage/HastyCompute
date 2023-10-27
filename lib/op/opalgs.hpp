#pragma once

#include "op.hpp"
#include <optional>
#include "../threading/thread_pool.hpp"

namespace hasty {
	namespace op {

		class OperatorAlg {
		protected:

			at::Tensor& access_vectensor(Vector& vec) const;
			const at::Tensor& access_vectensor(const Vector& vec) const;

			std::vector<Vector>& access_vecchilds(Vector& vec) const;
			const std::vector<Vector>& access_vecchilds(const Vector& vec) const;
		};

		at::Tensor power_iteration(const Operator& A, Vector& v, int iters=30);

		void conjugate_gradient(const Operator& A, Vector& x, const Vector& b, const std::optional<Operator>& P, 
			int iters = 30, double tol = 0.0);


		void gradient_descent(const Operator& gradf, Vector& x);



		class ConjugateGradient : public OperatorAlg {
		public:

			ConjugateGradient(const op::Operator& A, const op::Vector& b, const std::optional<op::Operator>& P);

			void run(op::Vector& x, int iter = 30, double tol = 0.0);

		private:
			op::Operator _A;
			op::Vector _b;
			std::optional<op::Operator> _P;
		};

		template<typename DeviceContext>
		class BatchConjugateGradientLoader {
		public:

			struct LoadResult {
				op::Operator A;
				op::Vector b;
				std::optional<Operator> P;
			};

			virtual LoadResult load(DeviceContext& dctx, size_t idx) = 0;
			
		};

		template<typename DeviceContext>
		class DefaultBatchConjugateGradientLoader : public ConjugateGradientLoader<DeviceContext> {
		public:

			DefaultConjugateGradientLoader(const LoadResult& lr)
				: _load_result({lr}) {}

			DefaultConjugateGradientLoader(const std::vector<LoadResult>& lrs)
				: _load_result(lrs) {}

			ConjugateGradientLoader::LoadResult load(DeviceContext& dctxt, size_t idx) override
			{
				return _load_results[idx];
			}

		private:
			std::vector<ConjugateGradientLoader::LoadResult> _load_results;
		};

		template<typename DeviceContext>
		class BatchConjugateGradient : public OperatorAlg {
		public:

			BatchConjugateGradient(const std::shared_ptr<BatchConjugateGradientLoader>& loader,
				const std::shared_ptr<ContextThreadPool<DeviceContext>>& thread_pool)
				: _cg_loader(loader), _thread_pool(thread_pool)
			{}

			void run(op::Vector& x, int iter = 30, double tol = 0.0)
			{
				std::deque<std::future<void>> futures;

				std::function<void(DeviceContext&)> batch_applier;

				auto& children = access_vecchilds(x);
				for (int i = 0; i < children.size(); ++i) {
					auto& child = children[i];

					batch_applier = [this, i, child](DeviceContext& context) {
						BatchConjugateGradientLoader::LoadResult result = _cg_loader.load(context, i);
						ConjugateGradient(result.A, result.b, result.P).run(child, iter, tol);
					};

					futures.emplace_back(_thread_pool->enqueue(batch_applier));

					if (futures.size() > 8 * _thread_pool->nthreads()) {
						torch_util::future_catcher(futures.front());
						futures.pop_front();
					}
				}

				// we wait for all promises
				while (futures.size() > 0) {
					torch_util::future_catcher(futures.front());
					futures.pop_front();
				}
			}

		private:
			std::shared_ptr<BatchConjugateGradientLoader> _cg_loader;
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
