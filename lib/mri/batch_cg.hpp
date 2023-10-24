#pragma once

#include "../torch_util.hpp"
#include "../op/op.hpp"

namespace hasty {
	namespace mri {

		template<typename F> concept CGLoader =
			std::invocable<F, at::Stream, size_t>&&
			std::same_as <
			std::tuple<op::Operator, op::Vector, at::optional<op::Operator>>,
			std::invoke_result_t<F, at::Stream, size_t>>;

		class BatchCG {
		public:

			template<CGLoader T>
			BatchCG(T const& loader, const std::vector<at::Stream>& streams) {

			}

		private:
			
		};

	}
}

