module;

#include <torch/torch.h>

export module flowmri;

import opalgs;
import thread_pool;

namespace hasty {
	namespace mri {

		/*
		std::vector<at::Tensor> get_preconditioners(const std::vector<at::Tensor>& coords, at::Tensor smaps,
			std::optional<std::vector<at::Stream>> streams = std::nullopt)
		{

		}
		*/

		export class FiveEncNuclearNormAdmmMinimizer : public hasty::op::AdmmMinimizer {
		public:

			FiveEncNuclearNormAdmmMinimizer(std::unique_ptr<ContextThreadPool<at::Stream>> threadpool)
				: _threadpool(std::move(threadpool))
			{

			}

		private:
			std::unique_ptr<ContextThreadPool<at::Stream>> _threadpool;
		};


	}
}

