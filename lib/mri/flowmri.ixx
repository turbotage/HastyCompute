module;

#include <torch/torch.h>

export module flowmri;

import opalgs;
import thread_pool;
import device;

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
				std::shared_ptr<ContextThreadPool<DContext>> threadpool
			)
				: _threadpool(std::move(threadpool))
			{}

			void solve(op::Admm::Context& ctx) override {

			}



		private:
			std::shared_ptr<ContextThreadPool<DContext>> _threadpool;
		};


	}
}

