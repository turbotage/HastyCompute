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

		class PowerIteration : public OperatorAlg {
		public:

			PowerIteration(const op::Operator& A);

			at::Tensor run(op::Vector& v, int iters = 30);

		private:
			op::Operator _A;
		};

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

		class AdmmMinimizer : public OperatorAlg {
		public:

			virtual void solve(Admm::Context& ctx);

		private:

		};

		class Admm {
		public:

			struct Context {
				// Ax + Bz = c
				op::Operator A;
				op::Operator B;
				op::Vector c;

				op::Vector x; // x
				op::Vector z; // z
				op::Vector u; // scaled dual variable

				int xiter = 30;
				double xtol = 0.0;
				int ziter = 30;
				double ztol = 0.0;

				int admm_iter = 30;
				double admm_tol = 0.0;
			};

			Admm(const std::shared_ptr<AdmmMinimizer>& xmin, const std::shared_ptr<AdmmMinimizer>& zmin);

			void run(Admm::Context& ctx);

		private:
			std::shared_ptr<Context> _ctx;
			std::shared_ptr<AdmmMinimizer> _xmin;
			std::shared_ptr<AdmmMinimizer> _zmin;
		};


	}
}
