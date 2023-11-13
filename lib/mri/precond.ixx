module;

#include <torch/torch.h>

export module precond;

import torch_util;
import op;

namespace hasty {
	namespace op {

		class CirculantPreconditioner : public op::Operator {
		public:

			at::Tensor build_diagonal(at::Tensor smaps, at::Tensor coord, const at::optional<at::Tensor>& weights = at::nullopt,
				const at::optional<double>& lambda = 0.0);



		private:

		};

	}
}

