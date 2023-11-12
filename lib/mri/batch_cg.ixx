module;

#include "../torch_util.hpp"

export module batch_cg;

import opalgs;
import opalgebra;
import mriop;

namespace hasty {
	namespace mri {

		struct SenseDeviceContext {
			at::Tensor smaps;

			at::Stream stream;
		};

		class SenseAdmmLoader : public op::BatchConjugateGradientLoader<SenseDeviceContext> {
		public:

			SenseAdmmLoader(
				const std::vector<at::Tensor>& coords, const std::vector<int64_t>& nmodes,
				const std::vector<at::Tensor>& kdata, const at::Tensor& smaps,
				const std::shared_ptr<op::Admm::Context>& ctx,
				const at::optional<std::vector<at::Tensor>>& preconds = at::nullopt);

			op::ConjugateGradientLoadResult load(SenseDeviceContext& dctx, size_t idx) override;

		private:
			std::vector<at::Tensor> _coords;
			std::vector<int64_t> _nmodes;
			std::vector<at::Tensor> _kdata;
			at::Tensor _smaps;

			std::shared_ptr<op::Admm::Context> _ctx;

			at::optional<std::vector<at::Tensor>> _preconds;
		};


	}
}
