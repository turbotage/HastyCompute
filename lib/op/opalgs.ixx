module;

#include <torch/torch.h>

export module opalgs;

import <future>;

import thread_pool;
import torch_util;
import vec;
import op;

namespace hasty {
	namespace op {

		export class OperatorAlg {
		protected:

			at::Tensor& access_vectensor(Vector& vec) const;
			const at::Tensor& access_vectensor(const Vector& vec) const;

			std::vector<Vector>& access_vecchilds(Vector& vec) const;
			const std::vector<Vector>& access_vecchilds(const Vector& vec) const;
		};

		export class PowerIteration : public OperatorAlg {
		public:

			PowerIteration() = default;

			at::Tensor run(const op::Operator& A, op::Vector& v, int iters = 30);

		};

		export class ConjugateGradient : public OperatorAlg {
		public:

			ConjugateGradient(sptr<op::Operator> A, sptr<op::Vector> b, sptr<op::Operator> P);

			void run(op::Vector& x, int iter = 30, double tol = 0.0);

		private:
			sptr<op::Operator> _A;
			sptr<op::Vector> _b;
			sptr<op::Operator> _P;
		};

		export struct ConjugateGradientLoadResult {
			sptr<op::Operator> A;
			op::Vector b;
			sptr<Operator> P;
		};

		export template<typename DeviceContext>
		class BatchedConjugateGradientLoader {
		public:

			virtual ConjugateGradientLoadResult load(DeviceContext& dctx, size_t idx) = 0;

		};

		export template<typename DeviceContext>
		class DefaultBatchedConjugateGradientLoader : public BatchedConjugateGradientLoader<DeviceContext> {
		public:

			DefaultBatchedConjugateGradientLoader(const std::vector<ConjugateGradientLoadResult>& lrs)
				: _load_results(lrs)
			{}

			DefaultBatchedConjugateGradientLoader(const ConjugateGradientLoadResult& lr)
				: _load_results({ lr })
			{}

			ConjugateGradientLoadResult load(DeviceContext& dctxt, size_t idx) override
			{
				return _load_results[idx];
			}

		private:
			std::vector<ConjugateGradientLoadResult> _load_results;
		};

		export template<typename DeviceContext>
		class BatchedConjugateGradient : public OperatorAlg {
		public:

			BatchedConjugateGradient(sptr<BatchedConjugateGradientLoader<DeviceContext>> loader,
				sptr<hasty::ContextThreadPool<DeviceContext>> thread_pool)
				: _cg_loader(std::move(loader)), _thread_pool(std::move(thread_pool))
			{}

			void run(op::Vector& x, int iter = 30, double tol = 0.0)
			{
				std::deque<std::future<void>> futures;

				std::function<void(DeviceContext&)> batch_applier;

				auto& children = access_vecchilds(x);
				for (int i = 0; i < children.size(); ++i) {
					auto& child = children[i];

					batch_applier = [this, i, child](DeviceContext& context) {
						ConjugateGradientLoadResult result = _cg_loader.load(context, i);
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

			const sptr<BatchedConjugateGradientLoader<DeviceContext>>& get_loader() const
			{
				return _cg_loader;
			}

			const sptr<ContextThreadPool<DeviceContext>>& get_threadpool() const
			{
				return _thread_pool;
			}

		private:
			sptr<BatchedConjugateGradientLoader<DeviceContext>> _cg_loader;
			sptr<ContextThreadPool<DeviceContext>> _thread_pool;
		};

		class AdmmMinimizer;

		export class Admm {
		public:

			struct Context {
				// Ax + Bz = c
				sptr<op::AdjointableOp> A;
				std::optional<op::Vector> c; // offset

				op::Vector x; // x
				op::Vector z; // z
				op::Vector u; // scaled dual variable

				sptr<op::AdjointableOp> AHA;

				double rho;

				int xiter = 30;
				double xtol = 0.0;
				int ziter = 30;
				double ztol = 0.0;

				int admm_iter = 30;
				double admm_tol = 0.0;

				std::mutex ctxmut;
			};

			Admm(sptr<AdmmMinimizer> xmin, sptr<AdmmMinimizer> zmin);

			void run(Admm::Context& ctx);

		private:
			sptr<Context> _ctx;
			sptr<AdmmMinimizer> _xmin;
			sptr<AdmmMinimizer> _zmin;
		};

		export class AdmmMinimizer : public OperatorAlg {
		public:

			virtual void solve(Admm::Context& ctx) = 0;

			virtual void set_iters(int64_t iters) = 0;

			virtual void set_tol(double tol) = 0;

		private:

		};



	}
}
