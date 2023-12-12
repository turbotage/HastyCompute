module;

#include <torch/torch.h>

export module flowmri;

import opalgs;
import thread_pool;
import device;
import svt;

namespace hasty {
	namespace mri {

		/*
		std::vector<at::Tensor> get_preconditioners(const std::vector<at::Tensor>& coords, at::Tensor smaps,
			std::optional<std::vector<at::Stream>> streams = std::nullopt)
		{

		}
		*/

		export template<hasty::DeviceContextConcept DContext>
		class FivePointNuclearNormAdmmMinimizer : public hasty::op::AdmmMinimizer {
		public:

			FivePointNuclearNormAdmmMinimizer(
				std::shared_ptr<ContextThreadPool<DContext>> threadpool,
				const std::array<int64_t, 3>& block_strides, 
				const std::array<int64_t, 3>& block_shape,
				double lambda, bool soft
			)
				: _svtop(std::move(threadpool), block_strides, block_shape, soft), _lambda(lambda)
			{}

			void solve(op::Admm::Context& ctx) override {
				if (ctx.c.has_value())
					return _svtop->apply(*ctx.c - ctx.u - ctx.A->apply(ctx.x), lambda);
				else
					_svtop->apply(ctx.u.neg() - ctx.A->apply(ctx.x), lambda);
			}

		private:
			std::unique_ptr<svt::Normal3DBlocksSVTProxOp<DContext>> _svtop;
			double _lambda;
		};


		class FivePointNuclearNormAdmm {
		public:

			FivePointNuclearNormAdmm(
				// Sense part
				std::vector<at::Stream> sense_streams,
				std::vector<at::Tensor> coords, std::vector<int64_t> nmodes,
				std::vector<at::Tensor> kdata, at::Tensor smaps,
				at::optional<std::vector<at::Tensor>> preconds,
				// Prox part
				std::vector<at::Stream> prox_streams,
				const std::array<int64_t, 3>& block_strides,
				const std::array<int64_t, 3>& block_shape,
				double lambda
			)
			{

			}

		private:

		};

		


	}
}

