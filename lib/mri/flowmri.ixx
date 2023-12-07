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
		class FiveEncNuclearNormAdmmMinimizer : public hasty::op::AdmmMinimizer {
		public:

			FiveEncNuclearNormAdmmMinimizer(
				std::shared_ptr<ContextThreadPool<DContext>> threadpool,
				const std::array<int64_t, 3>& block_strides, 
				const std::array<int64_t, 3>& block_shape
			)
				: _svtop(std::move(threadpool), block_strides, block_shape, true)
			{}

			void solve(op::Admm::Context& ctx) override {
				return _svtop->apply(ctx.c - ctx.u - ctx.A->apply(ctx.x));
			}

		private:
			std::unique_ptr<svt::Normal3DBlocksSVTProxOp<DContext>> _svtop;
		};


	}
}

