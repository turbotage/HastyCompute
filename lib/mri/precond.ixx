module;

#include <torch/torch.h>

export module precond;

import torch_util;
import op;

namespace hasty {
	namespace mri {

		export class CirculantPreconditionerOp : public op::AdjointableOp {
		public:

			static std::unique_ptr<CirculantPreconditionerOp> Create(at::Tensor diag, bool centered, const std::optional<std::string>& norm, bool adjointify);

			static at::Tensor build_diagonal(at::Tensor smaps, at::Tensor coord, const at::optional<at::Tensor>& weights = at::nullopt,
				const at::optional<double>& lambda = 0.0);

		protected:

			CirculantPreconditionerOp(at::Tensor diag, bool centered, const std::optional<std::string>& norm, bool adjointify);

			op::Vector apply(const op::Vector& in) const override;

			std::shared_ptr<Operator> to_device(at::Stream stream) const override;

			std::shared_ptr<AdjointableOp> adjoint() const override;

		private:
			at::Tensor _diag;
			bool _centered;
			std::optional<std::string> _norm;
			bool _adjointify;
		};

	}
}

